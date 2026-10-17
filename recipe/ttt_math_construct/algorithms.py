import math
from collections import defaultdict, deque

import numpy as np
import torch

from verl.trainer.ppo import core_algos


def _ensure_advantage_estimator_symbol() -> None:
    # `ray_trainer.py` checks this attribute directly. Add a recipe-local alias
    # so we do not have to edit verl/trainer/.
    if not hasattr(core_algos.AdvantageEstimator, "TTT_ENTROPIC_ADAPTIVE"):
        setattr(core_algos.AdvantageEstimator, "TTT_ENTROPIC_ADAPTIVE", "ttt_entropic_adaptive")


# ---------------------------------------------------------------------------
# Adaptive δ controller
#
# State lives as function attributes on _compute_ttt_entropic_adaptive_advantage
# (prefixed _adap_) so it persists across steps within the driver process.
# δ for step t is computed from step t-1 metrics (natural one-step lag).
#
# Priority ladder (first rule that fires):
#   P1  KL brake      — ref-KL EMA > threshold → reset δ to init
#   P2  Winner-take-all — β-max-hit-rate EMA high AND eff-n EMA low
#                         → decrease δ to spread advantage weights
#   P3  Stagnation    — no reward improvement for N steps AND active groups
#                       exist → decrease δ to diversify gradient
#   P4  Over-exploring — eff-n EMA high AND low constant-group fraction
#                        → increase δ to sharpen advantage focus
#   P0  Hold          — healthy regime, keep current δ
# ---------------------------------------------------------------------------


def _adaptive_init(config) -> None:
    """Initialise adaptive state on the advantage function (first call only)."""
    fn = _compute_ttt_entropic_adaptive_advantage
    if hasattr(fn, "_adap_initialized"):
        return
    fn._adap_initialized = True
    fn._adap_delta = float(config.get("ttt_entropic_kl_budget", math.log(2.0)))
    fn._adap_ema_const_frac = None
    fn._adap_ema_eff_n = None
    fn._adap_ema_kl_ref = None
    fn._adap_ema_beta_max_rate = None
    fn._adap_ema_reward = None  # smoothed batch-mean reward (for logging)
    fn._adap_best_seen = -float("inf")  # running max of batch-max reward (for P3)
    fn._adap_reward_peak = -float("inf")  # alias kept for backward-compat logs
    fn._adap_steps_no_improve = 0
    fn._adap_last_reason = "init"
    # Rolling window of recent batch-max rewards. Stagnation is detected by
    # comparing the current step's max against the prior window — this avoids
    # the failure mode where one lucky early batch makes ``_adap_best_seen``
    # so high that subsequent normal batches never reset the no-improve
    # counter, causing δ to collapse to MIN even while the policy is still
    # learning.
    stag_steps_init = int(config.get("ttt_adaptive_stagnation_steps", 15))
    fn._adap_recent_window = deque(maxlen=max(stag_steps_init, 1))


def _resolve_delta_adaptive(config) -> float:
    """Return the pre-computed adaptive δ (set at the end of the previous step)."""
    _adaptive_init(config)
    return _compute_ttt_entropic_adaptive_advantage._adap_delta


def _update_adaptive_delta(
    config,
    const_frac: float,
    eff_n: float,
    kl_ref: float,
    beta_max_rate: float,
    reward_mean: float,
    reward_max: float,
    global_steps: int = 0,
) -> None:
    """Update EMA state and recompute δ for the next step (called at step end)."""
    fn = _compute_ttt_entropic_adaptive_advantage

    delta_init = float(config.get("ttt_entropic_kl_budget", math.log(2.0)))
    # MIN floor raised from 0.25·ln2 (=0.173) to 0.5·ln2 (=0.347): below ~0.3
    # the entropic solver lands β so small that advantages collapse toward a
    # uniform group baseline (eff_n→k, |adv|→0) and the policy gradient
    # essentially vanishes. Empirical c910/lza_simd_half runs showed adv.max
    # falling from ~10 (fixed δ=ln2) to ~0.5 once δ hit 0.173, with
    # corresponding loss of learning signal. 0.347 still allows meaningful
    # spreading vs the init value while keeping advantages sharp enough to
    # drive updates.
    delta_min = float(config.get("ttt_entropic_kl_budget_adaptive_min", 0.5 * math.log(2.0)))
    delta_max = float(config.get("ttt_entropic_kl_budget_adaptive_max", 4.0 * math.log(2.0)))
    alpha = float(config.get("ttt_adaptive_ema_alpha", 0.3))
    reward_alpha = float(config.get("ttt_adaptive_reward_ema_alpha", 0.1))

    def ema(old, new, a=alpha):
        return new if old is None else a * new + (1.0 - a) * old

    fn._adap_ema_const_frac = ema(fn._adap_ema_const_frac, const_frac)
    fn._adap_ema_eff_n = ema(fn._adap_ema_eff_n, eff_n)
    fn._adap_ema_kl_ref = ema(fn._adap_ema_kl_ref, kl_ref)
    fn._adap_ema_beta_max_rate = ema(fn._adap_ema_beta_max_rate, beta_max_rate)
    fn._adap_ema_reward = ema(fn._adap_ema_reward, reward_mean, a=reward_alpha)

    stag_min_improve = float(config.get("ttt_adaptive_stagnation_min_improve", 0.02))
    stag_steps = int(config.get("ttt_adaptive_stagnation_steps", 15))
    # Resize the rolling window if the configured stagnation horizon changed
    # at runtime (rare, but cheap to support).
    if fn._adap_recent_window.maxlen != max(stag_steps, 1):
        fn._adap_recent_window = deque(fn._adap_recent_window, maxlen=max(stag_steps, 1))

    # Stagnation now uses a rolling window of the most recent ``stag_steps``
    # batch-maxes instead of an all-time best. Reason: a single noisy lucky
    # batch (e.g. step 5 with reward.max=15.18) used to permanently raise
    # ``_adap_best_seen``, after which subsequent batches almost never beat
    # the (best + min_improve·|best|) threshold, so the no-improve counter
    # ramped to >stag_steps within ~10 steps and δ chained down to MIN even
    # though the policy was still actively learning. Comparing against the
    # window's prior best (excluding the current step) detects only true
    # plateaus relative to recent behaviour.
    prior_best = max(fn._adap_recent_window) if len(fn._adap_recent_window) > 0 else -float("inf")
    fn._adap_recent_window.append(reward_max)
    # ``_adap_best_seen`` is still tracked for logging compatibility.
    if reward_max > fn._adap_best_seen:
        fn._adap_best_seen = reward_max
        fn._adap_reward_peak = reward_max  # keep alias in sync for logs
    improve_ref = max(abs(prior_best), 1.0) if prior_best > -float("inf") else 1.0
    if prior_best == -float("inf") or reward_max > prior_best + stag_min_improve * improve_ref:
        fn._adap_steps_no_improve = 0
    else:
        fn._adap_steps_no_improve += 1

    # Thresholds (all configurable).
    kl_brake = float(config.get("ttt_adaptive_kl_brake_threshold", 0.4))
    winner_beta_hi = float(config.get("ttt_adaptive_winner_beta_rate_hi", 0.6))
    # P2 eff_n joint gate (re-enabled). In the lza_simd_half v8/v9 runs we
    # observed β saturation EMA crossing 0.6 even though eff_n EMA was ~5
    # (close to k=8) — i.e. rewards were nearly uniform within groups, β
    # saturated trying to reach δ=ln 2 against a flat reward landscape, but
    # this is NOT winner-take-all. Lowering δ in this regime collapses
    # advantages (adv.max ~3 → ~0.5) and stalls learning. Real winner-take-all
    # has eff_n close to 1-2; we only fire P2 when both signals agree.
    winner_eff_n_lo = float(config.get("ttt_adaptive_winner_eff_n_lo", 3.5))
    explore_eff_n_hi = float(config.get("ttt_adaptive_explore_eff_n_hi", 5.5))
    explore_const_lo = float(config.get("ttt_adaptive_explore_const_lo", 0.3))
    stag_const_guard = float(config.get("ttt_adaptive_stagnation_const_guard", 0.7))
    # P3 only fires when advantages are still SHARP (low eff_n). When eff_n is
    # already high, the group's reward spread has been packed into nearly
    # uniform weights, and lowering δ further only flattens gradients more —
    # the opposite of what the "spread the winner" P3 rule intends.
    stag_eff_n_lo = float(config.get("ttt_adaptive_stagnation_eff_n_lo", 4.0))
    plateau_guard = int(config.get("ttt_adaptive_plateau_guard", 5))
    up = float(config.get("ttt_adaptive_up_factor", 1.2))
    dn = float(config.get("ttt_adaptive_down_factor", 0.85))
    # Warm-up: hold δ at init while EMAs spin up and the policy gets a chance
    # to explore. Without this, the controller can fire P3 within the first
    # ~25 steps based on noisy early signals and drag δ to MIN before the
    # policy has produced any meaningful gradient.
    warmup = int(config.get("ttt_adaptive_warmup_steps", 20))

    delta = fn._adap_delta
    reason = "hold"

    if int(global_steps or 0) < warmup:
        # During warm-up: only KL-brake (true policy collapse) is allowed to
        # alter δ; all other rules are suppressed.
        if fn._adap_ema_kl_ref is not None and fn._adap_ema_kl_ref > kl_brake:
            delta = delta_init
            fn._adap_steps_no_improve = 0
            reason = "kl_brake"
        else:
            reason = "warmup"
        fn._adap_delta = delta
        fn._adap_last_reason = reason
        return

    # P1 — collapse brake
    if fn._adap_ema_kl_ref is not None and fn._adap_ema_kl_ref > kl_brake:
        delta = delta_init
        fn._adap_steps_no_improve = 0
        reason = "kl_brake"

    # P2 — winner-take-all: β saturated in majority of groups AND group
    # weights are genuinely concentrated (low eff_n EMA). The eff_n joint gate
    # is critical: β saturating with HIGH eff_n means rewards are too flat to
    # reach δ=ln 2 (the entropic solver wants a sharper distribution but the
    # raw reward spread won't let it) — that's a sign to HOLD δ, not lower it.
    # Lowering δ in the flat-reward regime crushes advantages (adv.max → ~0.5)
    # and the policy stops learning, which is exactly the lza_simd_half v8/v9
    # failure mode we are guarding against.
    elif (
        fn._adap_ema_beta_max_rate is not None
        and fn._adap_ema_beta_max_rate > winner_beta_hi
        and fn._adap_ema_eff_n is not None
        and fn._adap_ema_eff_n < winner_eff_n_lo
    ):
        delta = max(delta * dn, delta_min)
        reason = "winner_take_all"

    # P3 — stagnation: no new batch-max within the rolling window AND signal
    # is not dominated by constant-reward groups AND advantages are still
    # sharp enough that spreading them is the right remedy. The ``eff_n_lo``
    # gate is the critical addition: when eff_n EMA is already large the
    # group weights are near-uniform; lowering δ further would flatten them
    # to zero advantage and stall learning.
    elif (
        fn._adap_steps_no_improve >= stag_steps
        and fn._adap_ema_const_frac is not None
        and fn._adap_ema_const_frac < stag_const_guard
        and fn._adap_ema_eff_n is not None
        and fn._adap_ema_eff_n < stag_eff_n_lo
    ):
        delta = max(delta * dn, delta_min)
        reason = "stagnation"

    # P4 — over-exploring: advantages too diffuse AND learning is still making
    # progress. The ``plateau_guard`` prevents P4 from firing during late-stage
    # micro-refinement plateaus where const_frac→0 naturally but raising δ only
    # flattens gradient signal and blocks breakthrough.
    elif (
        fn._adap_ema_eff_n is not None
        and fn._adap_ema_eff_n > explore_eff_n_hi
        and fn._adap_ema_const_frac is not None
        and fn._adap_ema_const_frac < explore_const_lo
        and fn._adap_steps_no_improve < plateau_guard
    ):
        delta = min(delta * up, delta_max)
        reason = "over_explore"

    fn._adap_delta = delta
    fn._adap_last_reason = reason


def _resolve_delta_schedule(config, global_steps, total_training_steps) -> float:
    """Return δ for the current training step under the configured schedule.

    Modes:
      - ``fixed``:    constant ``ttt_entropic_kl_budget`` (default ln 2).
      - ``schedule``: interpolate from ``_start`` to ``_end`` by progress in
        ``[0, 1]``, shape controlled by ``ttt_entropic_kl_budget_schedule``
        (``linear`` or ``cosine``).
      - ``adaptive``: metric-driven controller; δ is updated each step based
        on EMA signals (constant-group fraction, effective group size, KL from
        ref, β-max-hit rate, reward trend). See ``_update_adaptive_delta``.
    """
    mode = str(config.get("ttt_entropic_kl_budget_mode", "fixed")).lower()
    default_delta = float(config.get("ttt_entropic_kl_budget", math.log(2.0)))

    if mode == "fixed":
        return default_delta

    if mode == "adaptive":
        return _resolve_delta_adaptive(config)

    if mode == "schedule":
        start = float(config.get("ttt_entropic_kl_budget_start", default_delta))
        end = float(config.get("ttt_entropic_kl_budget_end", default_delta))
        shape = str(config.get("ttt_entropic_kl_budget_schedule", "cosine")).lower()
        total = max(int(total_training_steps or 0), 1)
        step = max(int(global_steps or 0), 0)
        t = min(step / total, 1.0)
        if shape == "linear":
            frac = t
        else:  # cosine (default): smooth start/end, faster middle
            frac = 0.5 * (1.0 - math.cos(math.pi * t))
        return start + (end - start) * frac

    raise ValueError(f"Unknown ttt_entropic_kl_budget_mode: {mode}")


def _compute_ttt_entropic_adaptive_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config=None,
    **kwargs,
):
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)

    global_steps = kwargs.get("global_steps", 0)
    total_training_steps = kwargs.get("total_training_steps", 0)
    mode = str(config.get("ttt_entropic_kl_budget_mode", "fixed") if config else "fixed").lower()
    delta = _resolve_delta_schedule(config, global_steps, total_training_steps)

    beta_max = float(config.get("ttt_entropic_beta_max", 1e6))
    iters = int(config.get("ttt_entropic_beta_iters", 60))
    eps = float(config.get("ttt_entropic_epsilon", 1e-12))
    remove_constant_groups = bool(config.get("ttt_remove_constant_reward_groups", True))
    constant_reward_tol = float(config.get("ttt_constant_reward_tol", 1e-12))

    id2indices = defaultdict(list)
    for i in range(scores.shape[0]):
        id2indices[index[i]].append(i)

    # Per-group diagnostic accumulators — consumed by ray_trainer via
    # ``_compute_ttt_entropic_adaptive_advantage.last_adv_metrics``.
    per_group_beta: list[float] = []
    per_group_kl: list[float] = []
    per_group_eff_n: list[float] = []
    per_group_reward_std: list[float] = []
    per_group_reward_range: list[float] = []
    per_group_hit_max: list[int] = []
    num_constant_skipped = 0
    num_degenerate_skipped = 0
    num_groups = 0

    with torch.no_grad():
        for idxs in id2indices.values():
            rewards = scores[idxs].float()
            k = rewards.numel()
            num_groups += 1
            if k <= 1:
                advantages[idxs] = 0.0
                num_degenerate_skipped += 1
                continue
            if remove_constant_groups and torch.max(rewards) - torch.min(rewards) <= constant_reward_tol:
                advantages[idxs] = 0.0
                num_constant_skipped += 1
                continue

            per_group_reward_std.append(float(torch.std(rewards, unbiased=False).item()))
            per_group_reward_range.append(float((torch.max(rewards) - torch.min(rewards)).item()))

            centered = rewards - rewards.max()

            def kl_hat(beta_scalar: float) -> float:
                beta = rewards.new_tensor(beta_scalar)
                logits = beta * centered
                logq = logits - torch.logsumexp(logits, dim=0, keepdim=True)
                q = torch.exp(logq)
                log_k = math.log(k)
                kl = torch.sum(q * (logq + log_k))
                return float(kl.item())

            lo, hi = 0.0, 1.0
            solved_beta = None
            if kl_hat(hi) < delta:
                while hi < beta_max and kl_hat(hi) < delta:
                    hi *= 2.0
                if kl_hat(hi) < delta:
                    solved_beta = rewards.new_tensor(hi)

            if solved_beta is None:
                hi = min(hi, beta_max)
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    if kl_hat(mid) < delta:
                        lo = mid
                    else:
                        hi = mid
                solved_beta = rewards.new_tensor(hi)

            beta_val = float(solved_beta.item())
            kl_val = kl_hat(beta_val)
            # β is considered clamped when the solver returned the upper bound
            # *and* KL was still below δ (i.e. it never bracketed).
            hit_max = int(beta_val >= beta_max * 0.999 and kl_val < delta - 1e-6)
            per_group_beta.append(beta_val)
            per_group_kl.append(kl_val)
            # effective sample size: exp(H(q)) = k · exp(-KL)
            per_group_eff_n.append(k * math.exp(-kl_val))
            per_group_hit_max.append(hit_max)

            exp_scores = torch.exp(solved_beta * centered)
            loo_norm = (exp_scores.sum() - exp_scores) / max(k - 1, 1)
            weights = exp_scores / (loo_norm + eps)
            advantages[idxs] = weights - 1.0

    advantages = advantages.unsqueeze(-1) * response_mask

    # Discover-style KL penalty: adjust advantages toward frozen base model
    kl_penalty_coef = float(config.get("ttt_kl_penalty_coef", 0.0))
    kl_penalty_active = 0
    kl_penalty_avg = 0.0
    if kl_penalty_coef > 0.0 and "old_log_probs" in kwargs and "ref_log_prob" in kwargs:
        old_lp = kwargs["old_log_probs"]
        ref_lp = kwargs["ref_log_prob"]
        per_token_kl = (old_lp - ref_lp) * response_mask
        mask_sum = response_mask.sum().clamp(min=1.0)
        avg_kl = per_token_kl.sum() / mask_sum
        advantages = advantages + kl_penalty_coef * response_mask * (avg_kl - per_token_kl)
        kl_penalty_active = 1
        kl_penalty_avg = float(avg_kl.item())

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _max(xs):
        return float(max(xs)) if xs else 0.0

    const_frac = (num_constant_skipped / num_groups) if num_groups else 0.0
    eff_n_mean = _mean(per_group_eff_n)
    beta_max_rate = _mean(per_group_hit_max)

    metrics = {
        "adv/ttt/delta_used": float(delta),
        "adv/ttt/global_steps": int(global_steps or 0),
        "adv/ttt/num_groups": int(num_groups),
        "adv/ttt/num_groups_active": int(len(per_group_beta)),
        "adv/ttt/constant_groups_frac": const_frac,
        "adv/ttt/beta_solved_mean": _mean(per_group_beta),
        "adv/ttt/beta_solved_max": _max(per_group_beta),
        "adv/ttt/beta_max_hit_rate": beta_max_rate,
        "adv/ttt/kl_achieved_mean": _mean(per_group_kl),
        "adv/ttt/effective_group_size_mean": eff_n_mean,
        "adv/ttt/group_reward_std_mean": _mean(per_group_reward_std),
        "adv/ttt/group_reward_range_mean": _mean(per_group_reward_range),
        "adv/ttt/kl_penalty_coef": float(kl_penalty_coef),
        "adv/ttt/kl_penalty_active": int(kl_penalty_active),
        "adv/ttt/kl_penalty_avg_kl": kl_penalty_avg,
    }

    # Adaptive δ: use this step's metrics to compute δ for the next step,
    # and publish the full adaptive state so SwanLab can visualise it.
    if mode == "adaptive":
        reward_mean = float(scores.mean().item())
        reward_max = float(scores.max().item()) if scores.numel() > 0 else 0.0
        _update_adaptive_delta(
            config=config,
            const_frac=const_frac,
            eff_n=eff_n_mean,
            kl_ref=kl_penalty_avg,
            beta_max_rate=beta_max_rate,
            reward_mean=reward_mean,
            reward_max=reward_max,
            global_steps=int(global_steps or 0),
        )
        fn = _compute_ttt_entropic_adaptive_advantage
        # Encode reason as integer for numeric logging (0=init/hold/warmup,
        # 1=kl_brake, 2=winner_take_all, 3=stagnation, 4=over_explore).
        _reason_code = {
            "init": 0,
            "hold": 0,
            "warmup": 0,
            "kl_brake": 1,
            "winner_take_all": 2,
            "stagnation": 3,
            "over_explore": 4,
        }
        metrics["adv/ttt/adaptive_delta_next"] = fn._adap_delta
        metrics["adv/ttt/adaptive_ema_const_frac"] = fn._adap_ema_const_frac or 0.0
        metrics["adv/ttt/adaptive_ema_eff_n"] = fn._adap_ema_eff_n or 0.0
        metrics["adv/ttt/adaptive_ema_kl_ref"] = fn._adap_ema_kl_ref or 0.0
        metrics["adv/ttt/adaptive_ema_beta_max_rate"] = fn._adap_ema_beta_max_rate or 0.0
        metrics["adv/ttt/adaptive_ema_reward"] = fn._adap_ema_reward or 0.0
        metrics["adv/ttt/adaptive_reward_peak"] = float(fn._adap_reward_peak)
        metrics["adv/ttt/adaptive_best_seen"] = float(fn._adap_best_seen) if fn._adap_best_seen > -float("inf") else 0.0
        metrics["adv/ttt/adaptive_steps_no_improve"] = int(fn._adap_steps_no_improve)
        metrics["adv/ttt/adaptive_reason_code"] = float(_reason_code.get(fn._adap_last_reason, 0))

    _compute_ttt_entropic_adaptive_advantage.last_adv_metrics = metrics

    returns = advantages
    return advantages, returns


# Initialize the metrics attribute so the ray_trainer drain is always safe.
_compute_ttt_entropic_adaptive_advantage.last_adv_metrics = {}


def _compute_importance_sampling_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
):
    pg_losses = -log_prob * advantages
    pg_loss = core_algos.agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    zero = torch.tensor(0.0, device=log_prob.device)
    return pg_loss, zero, zero, zero


def ensure_ttt_discover_algorithms_registered() -> None:
    _ensure_advantage_estimator_symbol()

    if "ttt_entropic_adaptive" not in core_algos.ADV_ESTIMATOR_REGISTRY:
        core_algos.register_adv_est("ttt_entropic_adaptive")(_compute_ttt_entropic_adaptive_advantage)

    if "importance_sampling" not in core_algos.POLICY_LOSS_REGISTRY:
        core_algos.register_policy_loss("importance_sampling")(_compute_importance_sampling_policy_loss)


ensure_ttt_discover_algorithms_registered()

"""RTL state and PUCT state pool for TTT-Discover Verilog generation."""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .rtl_utils import RTLLMProblem, verilog_dedup_key


@dataclass
class RTLState:
    """Represents a state in the TTT-Discover search tree for RTL generation."""

    state_id: str
    problem_id: str
    timestep: int
    value: float
    raw_score: float
    verilog_code: str
    syntax_ok: bool = False
    func_ok: bool = False
    ppa_score: float = 0.0  # ADP (Area × Delay Product) when tech-mapped, else cell count
    ppa_area: float = 0.0   # physical area in μm² (tech-mapped only)
    ppa_delay: float = 0.0  # critical path delay in ps (tech-mapped only)
    code: str = ""
    parent_ids: list[str] = field(default_factory=list)
    parent_values: list[float] = field(default_factory=list)
    observation: str = ""


class RTLStatePool:
    """State archive with PUCT sampling and lineage blocking for RTL generation.

    Adapted from ttt_math_construct/state_pool.py with string-based deduplication.
    """

    def __init__(
        self,
        problems: dict[str, RTLLMProblem],
        max_buffer_size: int = 500,
        topk_children: int = 2,
        puct_c: float = 1.0,
        seed: int = 42,
        storage_dir: str | None = None,
        history_every: int = 0,
    ):
        self.problems = problems
        self.max_buffer_size = max(int(max_buffer_size), len(problems))
        self.topk_children = int(topk_children)
        self.puct_c = float(puct_c)
        self._rng = np.random.default_rng(int(seed))
        self.storage_dir = str(storage_dir) if storage_dir else ""
        self.history_every = max(int(history_every), 0)

        self._states: list[RTLState] = []
        self._state_by_id: dict[str, RTLState] = {}
        self._initial_state_ids: set[str] = set()
        self._children: dict[str, set[str]] = defaultdict(set)
        self._construction_keys: set[str] = set()

        # PUCT statistics
        self._n: dict[str, int] = {}
        self._m: dict[str, float] = {}
        self._T: int = 0

        self._last_sampled_ids: list[str] = []
        self._last_scores: dict[str, float] = {}
        self._best_reward_ever: float = 0.0
        self._new_states_added_last: int = 0

        restored = self._try_restore()
        if not restored:
            self._bootstrap_initial_states()
            self.save(step=0)

    def __len__(self) -> int:
        return len(self._states)

    def get_state(self, state_id: str) -> RTLState | None:
        return self._state_by_id.get(state_id)

    def sample_parent_state_ids(self, num_states: int, block_lineage: bool = True) -> list[str]:
        if not self._states:
            self._bootstrap_initial_states()
        if not self._states:
            return []

        candidates = list(self._states)
        values = np.array([float(s.value) for s in candidates], dtype=np.float64)
        scale = float(max(np.max(values) - np.min(values), 1e-6))
        prior = self._rank_prior(values)
        sqrt_t = float(np.sqrt(1.0 + self._T))

        scored = []
        for idx, state in enumerate(candidates):
            n_visits = self._n.get(state.state_id, 0)
            q_value = self._m.get(state.state_id, state.value) if n_visits > 0 else state.value
            bonus = self.puct_c * scale * float(prior[idx]) * sqrt_t / float(1 + n_visits)
            total = float(q_value + bonus)
            scored.append((total, state.value, state.state_id))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        picked_ids: list[str] = []
        blocked_ids: set[str] = set()

        for _, _, sid in scored:
            if len(picked_ids) >= num_states:
                break
            if block_lineage and sid in blocked_ids:
                continue
            picked_ids.append(sid)
            if block_lineage and num_states > 1:
                blocked_ids.update(self._get_full_lineage_ids(sid))

        if len(picked_ids) < num_states:
            for _, _, sid in scored:
                if len(picked_ids) >= num_states:
                    break
                if sid not in picked_ids:
                    picked_ids.append(sid)

        self._last_sampled_ids = picked_ids
        self._last_scores = {sid: score for score, _, sid in scored}
        return picked_ids

    def sample_parent_state_ids_epsilon_greedy(
        self, num_states: int, eps: float = 0.1
    ) -> list[str]:
        """ε-greedy reuse baseline: w.p. ε pick uniform random; else argmax by state value."""
        if not self._states:
            self._bootstrap_initial_states()
        if not self._states:
            return []

        candidates = list(self._states)
        ranked = sorted(
            candidates, key=lambda s: (float(s.value), s.state_id), reverse=True
        )

        picked_ids: list[str] = []
        exploit_idx = 0
        for _ in range(num_states):
            if float(self._rng.random()) < float(eps):
                pool = [s for s in candidates if s.state_id not in picked_ids] or candidates
                picked_ids.append(str(self._rng.choice([s.state_id for s in pool])))
            else:
                while exploit_idx < len(ranked) and ranked[exploit_idx].state_id in picked_ids:
                    exploit_idx += 1
                if exploit_idx < len(ranked):
                    picked_ids.append(ranked[exploit_idx].state_id)
                    exploit_idx += 1
                else:
                    picked_ids.append(ranked[0].state_id)

        self._last_sampled_ids = picked_ids
        return picked_ids

    def sample_parent_state_ids_no_reuse(self, num_states: int) -> list[str]:
        """No-reuse baseline: always return initial (seed) states, tiled to batch size."""
        if not self._initial_state_ids:
            self._bootstrap_initial_states()
        initial_ids = [
            sid for sid in (s.state_id for s in self._states)
            if sid in self._initial_state_ids
        ]
        if not initial_ids:
            return []
        repeats = (num_states + len(initial_ids) - 1) // len(initial_ids)
        tiled = (initial_ids * repeats)[:num_states]
        self._last_sampled_ids = tiled
        return tiled

    def update_from_rollouts(self, rollouts: list[dict[str, Any]], step: int) -> None:
        if not rollouts:
            self._new_states_added_last = 0
            return

        parent_to_rollouts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in rollouts:
            parent_id = str(item.get("parent_state_id", "")).strip()
            if parent_id:
                parent_to_rollouts[parent_id].append(item)

        added = 0
        for parent_id, grouped in parent_to_rollouts.items():
            parent_state = self._state_by_id.get(parent_id)
            if parent_state is None:
                continue

            valid_children = []
            for item in grouped:
                if not bool(item.get("success", False)):
                    continue
                verilog_code = item.get("construction", "")
                if not verilog_code or not isinstance(verilog_code, str):
                    continue

                reward = item.get("reward")
                if reward is None:
                    continue

                valid_children.append(
                    {
                        "verilog_code": str(verilog_code),
                        "value": float(reward),
                        "raw_score": float(item.get("raw_score", reward)),
                        "syntax_ok": bool(item.get("syntax_ok", False)),
                        "func_ok": bool(item.get("func_ok", False)),
                        "ppa_score": float(item.get("ppa_score", 0.0)),
                        "ppa_area": float(item.get("ppa_area", 0.0)),
                        "ppa_delay": float(item.get("ppa_delay", 0.0)),
                        "code": str(item.get("code", "")),
                        "observation": str(item.get("observation", "")),
                    }
                )

            best_value = max([c["value"] for c in valid_children], default=None)
            self._update_puct_stats(parent_state, best_value=best_value, num_rollouts=len(grouped))

            if self.topk_children > 0:
                valid_children = sorted(valid_children, key=lambda x: x["value"], reverse=True)[: self.topk_children]

            for child in valid_children:
                child_state = RTLState(
                    state_id=str(uuid.uuid4()),
                    problem_id=parent_state.problem_id,
                    timestep=int(step),
                    value=float(child["value"]),
                    raw_score=float(child["raw_score"]),
                    verilog_code=child["verilog_code"],
                    syntax_ok=child["syntax_ok"],
                    func_ok=child["func_ok"],
                    ppa_score=child["ppa_score"],
                    ppa_area=child["ppa_area"],
                    ppa_delay=child["ppa_delay"],
                    code=child["code"],
                    parent_ids=[parent_state.state_id] + list(parent_state.parent_ids),
                    parent_values=[parent_state.value] + list(parent_state.parent_values),
                    observation=child["observation"],
                )
                if self._append_state(child_state):
                    added += 1
                    if child_state.value > self._best_reward_ever:
                        self._best_reward_ever = child_state.value

        self._new_states_added_last = added
        self._prune()

    def sample_stats(self) -> dict[str, float]:
        values = [s.value for s in self._states]
        syntax_ok = [1.0 if s.syntax_ok else 0.0 for s in self._states]
        func_ok = [1.0 if s.func_ok else 0.0 for s in self._states]

        # Tree depth: length of parent_ids chain (initial states have depth 0)
        depths = [len(s.parent_ids) for s in self._states]

        # Per-problem best reward
        problem_best: dict[str, float] = {}
        for s in self._states:
            if s.problem_id not in problem_best or s.value > problem_best[s.problem_id]:
                problem_best[s.problem_id] = s.value

        stats = {
            "puct/buffer_size": float(len(self._states)),
            "puct/T": float(self._T),
            "puct/value_max": float(max(values)) if values else 0.0,
            "puct/value_mean": float(np.mean(values)) if values else 0.0,
            "puct/syntax_ok_rate": float(np.mean(syntax_ok)) if syntax_ok else 0.0,
            "puct/func_ok_rate": float(np.mean(func_ok)) if func_ok else 0.0,
            "ttt/best_reward_ever": self._best_reward_ever,
            "ttt/best_reward_per_problem_mean": float(np.mean(list(problem_best.values()))) if problem_best else 0.0,
            "ttt/tree_depth_max": float(max(depths)) if depths else 0.0,
            "ttt/tree_depth_mean": float(np.mean(depths)) if depths else 0.0,
            "ttt/new_states_added": float(self._new_states_added_last),
        }

        # Per-problem best reward breakdown
        for pid, best_val in sorted(problem_best.items()):
            stats[f"ttt_problem/{pid}/best_reward"] = best_val

        return stats

    # ---- Internal Methods ----

    def _bootstrap_initial_states(self) -> None:
        for problem_id, problem in self.problems.items():
            state = self._create_initial_state(problem)
            self._append_state(state, is_initial=True)

    def _create_initial_state(self, problem: RTLLMProblem) -> RTLState:
        return RTLState(
            state_id=str(uuid.uuid4()),
            problem_id=problem.problem_id,
            timestep=-1,
            value=0.0,
            raw_score=0.0,
            verilog_code="",
        )

    def _rank_prior(self, values: np.ndarray) -> np.ndarray:
        num = len(values)
        ranks = np.argsort(np.argsort(-values))
        weights = (num - ranks).astype(np.float64)
        weights /= max(float(weights.sum()), 1e-12)
        return weights

    def _construction_key(self, verilog_code: str) -> str:
        if not verilog_code:
            return ""
        return verilog_dedup_key(verilog_code)

    def _append_state(self, state: RTLState, is_initial: bool = False) -> bool:
        if state.verilog_code:
            key = self._construction_key(state.verilog_code)
            if key in self._construction_keys:
                return False
            self._construction_keys.add(key)
        if state.state_id in self._state_by_id:
            return False
        self._states.append(state)
        self._state_by_id[state.state_id] = state
        if is_initial:
            self._initial_state_ids.add(state.state_id)
        if state.parent_ids:
            self._children[state.parent_ids[0]].add(state.state_id)
        return True

    def _update_puct_stats(self, parent_state: RTLState, best_value: float | None, num_rollouts: int) -> None:
        lineage = [parent_state.state_id] + list(parent_state.parent_ids)
        num_rollouts = max(int(num_rollouts), 1)
        for _ in range(num_rollouts):
            for sid in lineage:
                self._n[sid] = self._n.get(sid, 0) + 1
            self._T += 1
        if best_value is not None:
            old = self._m.get(parent_state.state_id, parent_state.value)
            self._m[parent_state.state_id] = max(old, float(best_value))

    def _get_full_lineage_ids(self, state_id: str) -> set[str]:
        state = self._state_by_id.get(state_id)
        if state is None:
            return {state_id}
        lineage = {state_id}
        lineage.update(state.parent_ids)

        queue = [state_id]
        seen = {state_id}
        while queue:
            sid = queue.pop(0)
            for cid in self._children.get(sid, set()):
                if cid in seen:
                    continue
                seen.add(cid)
                lineage.add(cid)
                child_state = self._state_by_id.get(cid)
                if child_state is not None:
                    lineage.update(child_state.parent_ids)
                queue.append(cid)
        return lineage

    def _prune(self) -> None:
        if self.topk_children > 0:
            keep_ids = set(self._initial_state_ids)
            grouped_children: dict[str, list[RTLState]] = defaultdict(list)
            for state in self._states:
                if state.state_id in keep_ids:
                    continue
                parent_id = state.parent_ids[0] if state.parent_ids else ""
                grouped_children[parent_id].append(state)

            for siblings in grouped_children.values():
                siblings.sort(key=lambda s: s.value, reverse=True)
                keep_ids.update(state.state_id for state in siblings[: self.topk_children])

            self._states = [s for s in self._states if s.state_id in keep_ids]
            self._state_by_id = {s.state_id: s for s in self._states}
            self._construction_keys = {self._construction_key(s.verilog_code) for s in self._states if s.verilog_code}

        if len(self._states) <= self.max_buffer_size:
            self._children = defaultdict(set)
            for state in self._states:
                if state.parent_ids and state.parent_ids[0] in self._state_by_id:
                    self._children[state.parent_ids[0]].add(state.state_id)
            return

        keep_ids = set(self._initial_state_ids)
        non_initial = [s for s in self._states if s.state_id not in keep_ids]
        non_initial.sort(key=lambda s: s.value, reverse=True)
        for state in non_initial:
            if len(keep_ids) >= self.max_buffer_size:
                break
            keep_ids.add(state.state_id)

        self._states = [s for s in self._states if s.state_id in keep_ids]
        self._state_by_id = {s.state_id: s for s in self._states}
        self._construction_keys = {self._construction_key(s.verilog_code) for s in self._states if s.verilog_code}
        self._children = defaultdict(set)
        for state in self._states:
            if state.parent_ids and state.parent_ids[0] in self._state_by_id:
                self._children[state.parent_ids[0]].add(state.state_id)

    # ---- Serialization ----

    def _state_to_dict(self, state: RTLState) -> dict[str, Any]:
        return {
            "state_id": state.state_id,
            "problem_id": state.problem_id,
            "timestep": int(state.timestep),
            "value": float(state.value),
            "raw_score": float(state.raw_score),
            "verilog_code": state.verilog_code,
            "syntax_ok": state.syntax_ok,
            "func_ok": state.func_ok,
            "ppa_score": state.ppa_score,
            "ppa_area": state.ppa_area,
            "ppa_delay": state.ppa_delay,
            "code": state.code,
            "parent_ids": list(state.parent_ids),
            "parent_values": [float(v) for v in state.parent_values],
            "observation": state.observation,
        }

    def _state_from_dict(self, payload: dict[str, Any]) -> RTLState:
        return RTLState(
            state_id=str(payload["state_id"]),
            problem_id=str(payload["problem_id"]),
            timestep=int(payload["timestep"]),
            value=float(payload["value"]),
            raw_score=float(payload["raw_score"]),
            verilog_code=str(payload.get("verilog_code", "")),
            syntax_ok=bool(payload.get("syntax_ok", False)),
            func_ok=bool(payload.get("func_ok", False)),
            ppa_score=float(payload.get("ppa_score", 0.0)),
            ppa_area=float(payload.get("ppa_area", 0.0)),
            ppa_delay=float(payload.get("ppa_delay", 0.0)),
            code=str(payload.get("code", "")),
            parent_ids=[str(v) for v in payload.get("parent_ids", [])],
            parent_values=[float(v) for v in payload.get("parent_values", [])],
            observation=str(payload.get("observation", "")),
        )

    def _latest_state_path(self) -> str:
        return os.path.join(self.storage_dir, "latest.json")

    def _step_state_path(self, step: int) -> str:
        return os.path.join(self.storage_dir, f"state_pool_step_{int(step):06d}.json")

    def _atomic_write_json(self, path: str, payload: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)
        os.replace(tmp_path, path)

    def _serialize(self) -> dict[str, Any]:
        return {
            "states": [self._state_to_dict(state) for state in self._states],
            "initial_state_ids": sorted(self._initial_state_ids),
            "puct_n": {str(k): int(v) for k, v in self._n.items()},
            "puct_m": {str(k): float(v) for k, v in self._m.items()},
            "puct_T": int(self._T),
            "last_sampled_ids": list(self._last_sampled_ids),
            "last_scores": {str(k): float(v) for k, v in self._last_scores.items()},
            "best_reward_ever": float(self._best_reward_ever),
        }

    def _restore_from_payload(self, payload: dict[str, Any]) -> bool:
        states_payload = payload.get("states", [])
        if not states_payload:
            return False

        valid_problem_ids = set(self.problems)
        invalid_problem_ids = sorted(
            {str(item.get("problem_id", "")) for item in states_payload if str(item.get("problem_id", "")) not in valid_problem_ids}
        )
        if invalid_problem_ids:
            print(
                "[RTLStatePool] Ignoring restored states with unknown problem IDs: "
                + ", ".join(invalid_problem_ids)
            )

        filtered_payload = [
            item for item in states_payload if str(item.get("problem_id", "")) in valid_problem_ids
        ]
        if not filtered_payload:
            return False

        self._states = [self._state_from_dict(item) for item in filtered_payload]
        self._state_by_id = {state.state_id: state for state in self._states}
        self._initial_state_ids = {
            str(v) for v in payload.get("initial_state_ids", []) if str(v) in self._state_by_id
        }
        self._construction_keys = {
            self._construction_key(state.verilog_code) for state in self._states if state.verilog_code
        }
        self._children = defaultdict(set)
        for state in self._states:
            if state.parent_ids and state.parent_ids[0] in self._state_by_id:
                self._children[state.parent_ids[0]].add(state.state_id)

        self._n = {str(k): int(v) for k, v in (payload.get("puct_n", {}) or {}).items()}
        self._m = {str(k): float(v) for k, v in (payload.get("puct_m", {}) or {}).items()}
        self._T = int(payload.get("puct_T", 0))
        self._last_sampled_ids = [str(v) for v in payload.get("last_sampled_ids", []) if str(v) in self._state_by_id]
        self._last_scores = {
            str(k): float(v) for k, v in (payload.get("last_scores", {}) or {}).items() if str(k) in self._state_by_id
        }
        self._best_reward_ever = float(payload.get("best_reward_ever", 0.0))
        # Recompute best_reward_ever from states if not saved (backward compat)
        if self._best_reward_ever == 0.0 and self._states:
            self._best_reward_ever = max(s.value for s in self._states)
        return True

    def _try_restore(self) -> bool:
        if not self.storage_dir:
            return False
        latest_path = self._latest_state_path()
        if not os.path.exists(latest_path):
            return False
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False
        return self._restore_from_payload(payload)

    def inject_verified_solutions(self, solutions: list[dict]) -> int:
        """Inject pre-verified solutions as permanent initial states.

        Each entry in ``solutions`` must have:
          - ``problem_id``  (str)
          - ``verilog_code`` (str)
          - ``value``       (float) — RL reward (higher is better)
        Optional fields: ``raw_score``, ``ppa_score``, ``ppa_area``, ``ppa_delay``, ``observation``.

        Skips duplicates via code-based deduplication so it is safe to call on
        restored pools — states already present are silently ignored.  All
        injected states are marked as initial (never pruned).

        Returns the number of states actually added.
        """
        added = 0
        for sol in solutions:
            verilog_code = str(sol.get("verilog_code", "")).strip()
            if not verilog_code:
                continue
            problem_id = str(sol.get("problem_id", ""))
            if problem_id not in self.problems:
                continue
            key = self._construction_key(verilog_code)
            if key and key in self._construction_keys:
                continue  # already in pool
            state = RTLState(
                state_id=str(uuid.uuid4()),
                problem_id=problem_id,
                timestep=0,
                value=float(sol.get("value", 0.0)),
                raw_score=float(sol.get("raw_score", sol.get("value", 0.0))),
                verilog_code=verilog_code,
                syntax_ok=True,
                func_ok=True,
                ppa_score=float(sol.get("ppa_score", 0.0)),
                ppa_area=float(sol.get("ppa_area", 0.0)),
                ppa_delay=float(sol.get("ppa_delay", 0.0)),
                observation=str(sol.get("observation", "")),
            )
            if self._append_state(state, is_initial=True):
                if state.value > self._best_reward_ever:
                    self._best_reward_ever = state.value
                added += 1
        return added

    def save(self, step: int | None = None) -> None:
        if not self.storage_dir:
            return
        payload = self._serialize()
        self._atomic_write_json(self._latest_state_path(), payload)
        if step is not None and self.history_every > 0 and int(step) % self.history_every == 0:
            self._atomic_write_json(self._step_state_path(step), payload)

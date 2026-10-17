"""TTT-Discover curriculum sampler for RTL generation."""

from __future__ import annotations

import importlib
from collections import defaultdict
from typing import Any

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


def _object_array_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return [value]


class TTTRTLSampler(AbstractCurriculumSampler):
    """Curriculum sampler backed by a PUCT state pool for RTL generation."""

    def __init__(self, data_source: Dataset, data_config: DictConfig):
        self.data_source = data_source
        self.data_config = data_config
        self.batch_size = len(data_source)
        self._step = 0

        if not hasattr(data_source, "state_pool"):
            raise TypeError("TTTRTLSampler requires dataset.state_pool.")

        ttt_cfg = data_config.get("ttt_rtl", {})
        self.lineage_blocking = bool(ttt_cfg.get("lineage_blocking", True))

    def __iter__(self):
        sampled_ids = self.data_source.state_pool.sample_parent_state_ids(
            num_states=self.batch_size,
            block_lineage=self.lineage_blocking,
        )
        self.data_source.set_active_state_ids(sampled_ids)
        return iter(range(self.batch_size))

    def __len__(self) -> int:
        return self.batch_size

    def update(self, batch: DataProto) -> None:
        parent_ids = _object_array_to_list(batch.non_tensor_batch.get("ttt_parent_state_id"))
        if not parent_ids:
            parent_ids = self._parent_ids_from_extra_info(batch.non_tensor_batch.get("extra_info"))
        if not parent_ids:
            return

        rewards = _object_array_to_list(batch.non_tensor_batch.get("ttt_reward"))
        if not rewards:
            rewards = _object_array_to_list(batch.non_tensor_batch.get("score"))
        raw_scores = _object_array_to_list(batch.non_tensor_batch.get("ttt_raw_score"))
        constructions = _object_array_to_list(batch.non_tensor_batch.get("ttt_result_construction"))
        codes = _object_array_to_list(batch.non_tensor_batch.get("ttt_code"))
        observations = _object_array_to_list(batch.non_tensor_batch.get("ttt_msg"))
        success_flags = _object_array_to_list(batch.non_tensor_batch.get("ttt_success"))
        syntax_flags = _object_array_to_list(batch.non_tensor_batch.get("ttt_syntax_ok"))
        func_flags = _object_array_to_list(batch.non_tensor_batch.get("ttt_func_ok"))
        ppa_scores = _object_array_to_list(batch.non_tensor_batch.get("ttt_ppa_score"))
        ppa_areas = _object_array_to_list(batch.non_tensor_batch.get("ttt_ppa_area"))
        ppa_delays = _object_array_to_list(batch.non_tensor_batch.get("ttt_ppa_delay"))

        rollouts: list[dict[str, Any]] = []
        n = len(parent_ids)
        for i in range(n):
            rollouts.append(
                {
                    "parent_state_id": str(parent_ids[i]),
                    "reward": self._safe_scalar(rewards, i),
                    "raw_score": self._safe_scalar(raw_scores, i),
                    "construction": str(self._safe_object(constructions, i, default="")),
                    "code": str(self._safe_object(codes, i, default="")),
                    "observation": str(self._safe_object(observations, i, default="")),
                    "success": bool(self._safe_object(success_flags, i, default=False)),
                    "syntax_ok": bool(self._safe_object(syntax_flags, i, default=False)),
                    "func_ok": bool(self._safe_object(func_flags, i, default=False)),
                    "ppa_score": float(self._safe_scalar(ppa_scores, i) or 0.0),
                    "ppa_area": float(self._safe_scalar(ppa_areas, i) or 0.0),
                    "ppa_delay": float(self._safe_scalar(ppa_delays, i) or 0.0),
                }
            )

        step = int(batch.meta_info.get("global_steps", self._step))
        success_count = sum(1 for item in rollouts if item.get("success", False))
        syntax_count = sum(1 for item in rollouts if item.get("syntax_ok", False))
        func_count = sum(1 for item in rollouts if item.get("func_ok", False))
        valid_rewards = [item["reward"] for item in rollouts if item.get("reward", None) is not None]
        mean_reward = float(np.mean(valid_rewards)) if valid_rewards else 0.0
        print(
            f"[ttt_rtl_sampler] step={step} success={success_count}/{n} "
            f"syntax={syntax_count}/{n} func={func_count}/{n} reward_mean={mean_reward:.4f}"
        )
        self.data_source.state_pool.update_from_rollouts(rollouts=rollouts, step=step)
        self._log_ttt_metrics(step=step, rollouts=rollouts)
        self._step = max(self._step + 1, step + 1)

    @staticmethod
    def _safe_scalar(values: list[Any], idx: int) -> float | None:
        if idx >= len(values):
            return None
        try:
            return float(values[idx])
        except Exception:
            return None

    @staticmethod
    def _safe_object(values: list[Any], idx: int, default: Any = None) -> Any:
        if idx >= len(values):
            return default
        return values[idx]

    @staticmethod
    def _parent_ids_from_extra_info(extra_info: Any) -> list[str]:
        items = _object_array_to_list(extra_info)
        parent_ids: list[str] = []
        for item in items:
            if isinstance(item, dict):
                parent_ids.append(str(item.get("ttt_parent_state_id", "")))
            else:
                parent_ids.append("")
        return parent_ids

    def _log_ttt_metrics(self, step: int, rollouts: list[dict[str, Any]]) -> None:
        metrics = self._build_ttt_metrics(rollouts=rollouts)
        if not metrics:
            return

        # Log to SwanLab if available
        swanlab = self._get_swanlab()
        if swanlab is not None:
            try:
                swanlab.log(metrics, step=step)
            except Exception as e:
                print(f"[ttt_rtl_sampler] Warning: SwanLab logging failed at step {step}: {e}")

        # Also log sample outputs for debugging (first 3 rollouts)
        if swanlab is not None and step % 10 == 0:
            try:
                self._log_sample_outputs(swanlab, rollouts[:3], step)
            except Exception as e:
                print(f"[ttt_rtl_sampler] Warning: Sample output logging failed at step {step}: {e}")

    def _build_ttt_metrics(self, rollouts: list[dict[str, Any]]) -> dict[str, float]:
        if not rollouts:
            return {}

        success_flags = np.array([1.0 if item.get("success", False) else 0.0 for item in rollouts], dtype=np.float64)
        syntax_flags = np.array([1.0 if item.get("syntax_ok", False) else 0.0 for item in rollouts], dtype=np.float64)
        func_flags = np.array([1.0 if item.get("func_ok", False) else 0.0 for item in rollouts], dtype=np.float64)
        rewards = [float(item["reward"]) for item in rollouts if item.get("reward", None) is not None]

        grouped_rewards: dict[str, list[float]] = defaultdict(list)
        for item in rollouts:
            parent_id = str(item.get("parent_state_id", ""))
            reward = item.get("reward", None)
            if parent_id and reward is not None:
                grouped_rewards[parent_id].append(float(reward))

        group_stds = []
        non_constant_groups = 0
        for values in grouped_rewards.values():
            if not values:
                continue
            std = float(np.std(np.asarray(values, dtype=np.float64)))
            group_stds.append(std)
            if max(values) - min(values) > 1e-12:
                non_constant_groups += 1

        # Effective batch ratio: groups with non-constant rewards / total groups
        total_groups = len(grouped_rewards)
        effective_batch_ratio = float(non_constant_groups) / float(total_groups) if total_groups > 0 else 0.0

        # Unique code ratio: deduplicated constructions / total rollouts
        constructions = [item.get("construction", "") for item in rollouts]
        non_empty = [c for c in constructions if c]
        unique_code_ratio = float(len(set(non_empty))) / float(len(non_empty)) if non_empty else 0.0

        # Syntax-only rate: passed syntax but failed functional
        syntax_only = sum(
            1 for item in rollouts if item.get("syntax_ok", False) and not item.get("func_ok", False)
        )
        syntax_only_rate = float(syntax_only) / float(len(rollouts)) if rollouts else 0.0

        # Functional pass with PPA: passed functional and has ppa_score > 0
        func_with_ppa = sum(
            1 for item in rollouts if item.get("func_ok", False) and (item.get("ppa_score", 0.0) or 0.0) > 0
        )
        func_with_ppa_rate = float(func_with_ppa) / float(len(rollouts)) if rollouts else 0.0

        metrics = {
            "ttt/success_rate": float(success_flags.mean()) if success_flags.size > 0 else 0.0,
            "ttt/syntax_pass_rate": float(syntax_flags.mean()) if syntax_flags.size > 0 else 0.0,
            "ttt/func_pass_rate": float(func_flags.mean()) if func_flags.size > 0 else 0.0,
            "ttt/reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "ttt/reward_max": float(np.max(rewards)) if rewards else 0.0,
            "ttt/reward_min": float(np.min(rewards)) if rewards else 0.0,
            "ttt/reward_std": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
            "ttt/non_constant_group_ratio": (
                float(non_constant_groups) / float(len(grouped_rewards)) if grouped_rewards else 0.0
            ),
            "ttt/group_reward_std_mean": float(np.mean(group_stds)) if group_stds else 0.0,
            "ttt/num_groups": float(len(grouped_rewards)),
            "ttt/rollouts_per_step": float(len(rollouts)),
            "ttt/effective_batch_ratio": effective_batch_ratio,
            "ttt/unique_code_ratio": unique_code_ratio,
            "ttt/syntax_only_rate": syntax_only_rate,
            "ttt/func_pass_with_ppa_rate": func_with_ppa_rate,
        }

        if hasattr(self.data_source, "state_pool") and hasattr(self.data_source.state_pool, "sample_stats"):
            for key, value in self.data_source.state_pool.sample_stats().items():
                metrics[str(key)] = float(value)

        return metrics

    def _log_sample_outputs(self, swanlab, rollouts: list[dict[str, Any]], step: int) -> None:
        """Log sample Verilog code outputs to SwanLab for inspection."""
        if not rollouts:
            return

        table_data = []
        for i, rollout in enumerate(rollouts[:3]):  # Log first 3 samples
            code = rollout.get("code", "")
            construction = rollout.get("construction", "")
            observation = rollout.get("observation", "")
            reward = rollout.get("reward", 0.0)
            syntax_ok = rollout.get("syntax_ok", False)
            func_ok = rollout.get("func_ok", False)

            # Truncate long outputs
            if len(code) > 500:
                code = code[:500] + "\n...(truncated)"
            if len(construction) > 500:
                construction = str(construction)[:500] + "...(truncated)"
            if len(observation) > 300:
                observation = observation[:300] + "\n...(truncated)"

            table_data.append({
                "sample": i + 1,
                "reward": f"{reward:.4f}",
                "syntax": "✓" if syntax_ok else "✗",
                "functional": "✓" if func_ok else "✗",
                "code_preview": code,
                "feedback": observation,
            })

        # Log as text for now (SwanLab table support may vary)
        text_output = f"## Sample Outputs - Step {step}\n\n"
        for item in table_data:
            text_output += f"### Sample {item['sample']}\n"
            text_output += f"- Reward: {item['reward']}\n"
            text_output += f"- Syntax: {item['syntax']} | Functional: {item['functional']}\n"
            text_output += f"- Code:\n```verilog\n{item['code_preview']}\n```\n"
            text_output += f"- Feedback: {item['feedback']}\n\n"

        swanlab.log({"ttt/sample_outputs": swanlab.Text(text_output, caption=f"Step {step}")}, step=step)

    @staticmethod
    def _get_swanlab():
        try:
            return importlib.import_module("swanlab")
        except Exception:
            return None

"""TTT-Discover curriculum sampler for C910 FPU optimization."""

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


class TTTC910Sampler(AbstractCurriculumSampler):
    """Curriculum sampler backed by a PUCT state pool for C910 FPU optimization."""

    def __init__(self, data_source: Dataset, data_config: DictConfig):
        self.data_source = data_source
        self.data_config = data_config
        self.batch_size = len(data_source)
        self._step = 0

        if not hasattr(data_source, "state_pool"):
            raise TypeError("TTTC910Sampler requires dataset.state_pool.")

        ttt_cfg = data_config.get("ttt_c910", {})
        self.lineage_blocking = bool(ttt_cfg.get("lineage_blocking", True))
        reuse_mode = str(ttt_cfg.get("reuse_mode", "puct")).lower().strip()
        if reuse_mode not in {"puct", "epsilon_greedy", "none"}:
            raise ValueError(
                f"ttt_c910.reuse_mode must be one of puct|epsilon_greedy|none, got {reuse_mode!r}"
            )
        self.reuse_mode = reuse_mode
        self.epsilon_greedy_eps = float(ttt_cfg.get("epsilon_greedy_eps", 0.1))

    def __iter__(self):
        pool = self.data_source.state_pool
        if self.reuse_mode == "puct":
            sampled_ids = pool.sample_parent_state_ids(
                num_states=self.batch_size,
                block_lineage=self.lineage_blocking,
            )
        elif self.reuse_mode == "epsilon_greedy":
            sampled_ids = pool.sample_parent_state_ids_epsilon_greedy(
                num_states=self.batch_size,
                eps=self.epsilon_greedy_eps,
            )
        else:  # none
            sampled_ids = pool.sample_parent_state_ids_no_reuse(num_states=self.batch_size)
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
            f"[ttt_c910_sampler] step={step} success={success_count}/{n} "
            f"syntax={syntax_count}/{n} func={func_count}/{n} reward_mean={mean_reward:.4f}"
        )
        self.data_source.state_pool.update_from_rollouts(rollouts=rollouts, step=step)
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

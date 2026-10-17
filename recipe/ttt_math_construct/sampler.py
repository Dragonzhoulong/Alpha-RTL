from __future__ import annotations

import importlib
from collections import defaultdict
from typing import Any

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler

from .erdos_utils import try_deserialize_construction


def _object_array_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return [value]


class TTTErdosSampler(AbstractCurriculumSampler):
    """Curriculum sampler backed by a PUCT state pool."""

    def __init__(self, data_source: Dataset, data_config: DictConfig):
        self.data_source = data_source
        self.data_config = data_config
        self.batch_size = len(data_source)
        self._step = 0

        if not hasattr(data_source, "state_pool"):
            raise TypeError("TTTErdosSampler requires dataset.state_pool.")

        ttt_cfg = data_config.get("ttt_erdos", {})
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

        rollouts: list[dict[str, Any]] = []
        n = len(parent_ids)
        for i in range(n):
            rollouts.append(
                {
                    "parent_state_id": str(parent_ids[i]),
                    "reward": self._safe_scalar(rewards, i),
                    "raw_score": self._safe_scalar(raw_scores, i),
                    "construction": try_deserialize_construction(self._safe_object(constructions, i)),
                    "code": str(self._safe_object(codes, i, default="")),
                    "observation": str(self._safe_object(observations, i, default="")),
                    "success": bool(self._safe_object(success_flags, i, default=False)),
                }
            )

        step = int(batch.meta_info.get("global_steps", self._step))
        success_count = sum(1 for item in rollouts if item.get("success", False))
        valid_rewards = [item["reward"] for item in rollouts if item.get("reward", None) is not None]
        mean_reward = float(np.mean(valid_rewards)) if valid_rewards else 0.0
        print(f"[ttt_sampler] step={step} success={success_count}/{n} reward_mean={mean_reward:.6f}")
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

        swanlab = self._get_swanlab()
        if swanlab is not None:
            try:
                swanlab.log(metrics, step=step)
            except Exception:
                pass

    def _build_ttt_metrics(self, rollouts: list[dict[str, Any]]) -> dict[str, float]:
        if not rollouts:
            return {}

        success_flags = np.array([1.0 if item.get("success", False) else 0.0 for item in rollouts], dtype=np.float64)
        rewards = [float(item["reward"]) for item in rollouts if item.get("reward", None) is not None]
        raw_scores = [float(item["raw_score"]) for item in rollouts if item.get("raw_score", None) is not None]

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

        metrics = {
            "ttt/success_rate": float(success_flags.mean()) if success_flags.size > 0 else 0.0,
            "ttt/reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "ttt/reward_max": float(np.max(rewards)) if rewards else 0.0,
            "ttt/raw_c5_min": float(np.min(raw_scores)) if raw_scores else 0.0,
            "ttt/raw_c5_mean": float(np.mean(raw_scores)) if raw_scores else 0.0,
            "ttt/non_constant_group_ratio": (
                float(non_constant_groups) / float(len(grouped_rewards)) if grouped_rewards else 0.0
            ),
            "ttt/group_reward_std_mean": float(np.mean(group_stds)) if group_stds else 0.0,
        }

        if hasattr(self.data_source, "state_pool") and hasattr(self.data_source.state_pool, "sample_stats"):
            for key, value in self.data_source.state_pool.sample_stats().items():
                metrics[str(key)] = float(value)

        return metrics

    @staticmethod
    def _get_swanlab():
        try:
            return importlib.import_module("swanlab")
        except Exception:
            return None

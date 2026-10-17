import uuid
from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
from typing import Any

import numpy as np

from .erdos_utils import compute_erdos_c5_bound, normalize_erdos_construction, reward_from_c5


@dataclass
class TTTState:
    state_id: str
    timestep: int
    value: float
    raw_score: float
    construction: list[float]
    code: str = ""
    parent_ids: list[str] = field(default_factory=list)
    parent_values: list[float] = field(default_factory=list)
    observation: str = ""

    def to_prompt(self, target_c5: float) -> str:
        lines = ["You are iteratively optimizing C5 bound."]
        has_code = bool(self.code and self.code.strip())
        if has_code:
            lines.append("Here is the last code we ran:")
            lines.append(self.code.strip())
        else:
            lines.append("No previous code available.")
        if self.parent_values and self.raw_score is not None and self.construction:
            before_c5 = -float(self.parent_values[0])
            after_c5 = float(self.raw_score)
            current_gap = after_c5 - float(target_c5)
            lines.append(f"Here is the C5 bound before and after running the code above (lower is better): {before_c5:.6f} -> {after_c5:.6f}")
            lines.append(
                f"Target: {target_c5:.6f}. Current gap: {current_gap:.6f}. "
                "Further improvements will also be generously rewarded."
            )
        elif self.raw_score is not None:
            current_gap = float(self.raw_score) - float(target_c5)
            lines.append(f"Current C5 bound (lower is better): {self.raw_score:.6f}")
            lines.append(
                f"Target: {target_c5:.6f}. Current gap: {current_gap:.6f}. "
                "Further improvements will also be generously rewarded."
            )
        else:
            lines.append(f"Target C5 bound: {target_c5:.6f}")
        lines.append(f"Current construction length: n={len(self.construction)}.")
        if self.code.strip():
            lines.append("You may revise or replace the previous program.")
        if self.observation and self.observation.strip():
            obs = self.observation.strip()
            if len(obs) > 500:
                obs = "\n\n\t\t ...(TRUNCATED)...\n" + obs[-500:]
            lines.append("")
            lines.append("--- Previous Program Output ---")
            lines.append(obs)
            lines.append("--- End Output ---")
        return "\n".join(lines)


class PUCTStatePool:
    """State archive with PUCT sampling and lineage blocking."""

    def __init__(
        self,
        initial_state_count: int = 64,
        min_points: int = 40,
        max_points: int = 100,
        max_buffer_size: int = 1000,
        topk_children: int = 2,
        puct_c: float = 1.0,
        key_precision: int = 8,
        seed: int = 42,
        storage_dir: str | None = None,
        history_every: int = 0,
    ):
        self.initial_state_count = max(int(initial_state_count), 1)
        self.min_points = int(min_points)
        self.max_points = int(max_points)
        self.max_buffer_size = max(int(max_buffer_size), self.initial_state_count)
        self.topk_children = int(topk_children)
        self.puct_c = float(puct_c)
        self.key_precision = int(key_precision)
        self._rng = np.random.default_rng(int(seed))
        self.storage_dir = str(storage_dir) if storage_dir else ""
        self.history_every = max(int(history_every), 0)

        self._states: list[TTTState] = []
        self._state_by_id: dict[str, TTTState] = {}
        self._initial_state_ids: set[str] = set()
        self._children: dict[str, set[str]] = defaultdict(set)
        self._construction_keys: set[tuple] = set()

        self._n: dict[str, int] = {}
        self._m: dict[str, float] = {}
        self._T: int = 0

        self._last_sampled_ids: list[str] = []
        self._last_scores: dict[str, float] = {}

        restored = self._try_restore()
        if not restored:
            self._bootstrap_initial_states()
            self.save(step=0)

    def __len__(self) -> int:
        return len(self._states)

    def get_state(self, state_id: str) -> TTTState | None:
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

    def update_from_rollouts(self, rollouts: list[dict[str, Any]], step: int) -> None:
        if not rollouts:
            return

        parent_to_rollouts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in rollouts:
            parent_id = str(item.get("parent_state_id", "")).strip()
            if parent_id:
                parent_to_rollouts[parent_id].append(item)

        for parent_id, grouped in parent_to_rollouts.items():
            parent_state = self._state_by_id.get(parent_id)
            if parent_state is None:
                continue

            valid_children = []
            for item in grouped:
                if not bool(item.get("success", False)):
                    continue
                construction = item.get("construction")
                if construction is None:
                    continue
                try:
                    normalized = normalize_erdos_construction(construction)
                    raw_score = item.get("raw_score")
                    if raw_score is None:
                        raw_score = compute_erdos_c5_bound(normalized)
                    reward = item.get("reward")
                    if reward is None:
                        reward = reward_from_c5(float(raw_score))
                    valid_children.append(
                        {
                            "construction": normalized.tolist(),
                            "raw_score": float(raw_score),
                            "reward": float(reward),
                            "value": -float(raw_score),
                            "code": str(item.get("code", "")),
                            "observation": str(item.get("observation", "")),
                        }
                    )
                except Exception:
                    continue

            best_value = max([c["value"] for c in valid_children], default=None)
            self._update_puct_stats(parent_state, best_value=best_value, num_rollouts=len(grouped))

            if self.topk_children > 0:
                valid_children = sorted(valid_children, key=lambda x: x["value"], reverse=True)[: self.topk_children]

            for child in valid_children:
                child_state = TTTState(
                    state_id=str(uuid.uuid4()),
                    timestep=int(step),
                    value=float(child["value"]),
                    raw_score=float(child["raw_score"]),
                    construction=child["construction"],
                    code=child["code"],
                    parent_ids=[parent_state.state_id] + list(parent_state.parent_ids),
                    parent_values=[parent_state.value] + list(parent_state.parent_values),
                    observation=child["observation"],
                )
                self._append_state(child_state)

        self._prune()

    def sample_stats(self) -> dict[str, float]:
        values = [s.value for s in self._states]
        raw_scores = [s.raw_score for s in self._states]
        return {
            "puct/buffer_size": float(len(self._states)),
            "puct/T": float(self._T),
            "puct/value_max": float(max(values)) if values else 0.0,
            "puct/value_mean": float(np.mean(values)) if values else 0.0,
            "puct/raw_c5_min": float(min(raw_scores)) if raw_scores else 0.0,
        }

    def _bootstrap_initial_states(self) -> None:
        for _ in range(self.initial_state_count):
            state = self._create_initial_state()
            self._append_state(state, is_initial=True)

    def _state_to_dict(self, state: TTTState) -> dict[str, Any]:
        return {
            "state_id": state.state_id,
            "timestep": int(state.timestep),
            "value": float(state.value),
            "raw_score": float(state.raw_score),
            "construction": [float(v) for v in state.construction],
            "code": state.code,
            "parent_ids": list(state.parent_ids),
            "parent_values": [float(v) for v in state.parent_values],
            "observation": state.observation,
        }

    def _state_from_dict(self, payload: dict[str, Any]) -> TTTState:
        return TTTState(
            state_id=str(payload["state_id"]),
            timestep=int(payload["timestep"]),
            value=float(payload["value"]),
            raw_score=float(payload["raw_score"]),
            construction=[float(v) for v in payload["construction"]],
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
        }

    def _restore_from_payload(self, payload: dict[str, Any]) -> bool:
        states_payload = payload.get("states", [])
        if not states_payload:
            return False

        self._states = [self._state_from_dict(item) for item in states_payload]
        self._state_by_id = {state.state_id: state for state in self._states}
        self._initial_state_ids = {str(v) for v in payload.get("initial_state_ids", [])}
        self._construction_keys = {self._construction_key(state.construction) for state in self._states}
        self._children = defaultdict(set)
        for state in self._states:
            if state.parent_ids and state.parent_ids[0] in self._state_by_id:
                self._children[state.parent_ids[0]].add(state.state_id)

        self._n = {str(k): int(v) for k, v in (payload.get("puct_n", {}) or {}).items()}
        self._m = {str(k): float(v) for k, v in (payload.get("puct_m", {}) or {}).items()}
        self._T = int(payload.get("puct_T", 0))
        self._last_sampled_ids = [str(v) for v in payload.get("last_sampled_ids", [])]
        self._last_scores = {str(k): float(v) for k, v in (payload.get("last_scores", {}) or {}).items()}
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

    def save(self, step: int | None = None) -> None:
        if not self.storage_dir:
            return
        payload = self._serialize()
        self._atomic_write_json(self._latest_state_path(), payload)
        if step is not None and self.history_every > 0 and int(step) % self.history_every == 0:
            self._atomic_write_json(self._step_state_path(step), payload)

    def _create_initial_state(self) -> TTTState:
        n_points = int(self._rng.integers(self.min_points, self.max_points))
        construction = np.ones(n_points, dtype=np.float64) * 0.5
        perturbation = self._rng.uniform(-0.4, 0.4, n_points)
        perturbation = perturbation - float(np.mean(perturbation))
        construction = normalize_erdos_construction(construction + perturbation)
        raw_score = compute_erdos_c5_bound(construction)
        return TTTState(
            state_id=str(uuid.uuid4()),
            timestep=-1,
            value=-raw_score,
            raw_score=raw_score,
            construction=[float(v) for v in construction.tolist()],
        )

    def _rank_prior(self, values: np.ndarray) -> np.ndarray:
        num = len(values)
        ranks = np.argsort(np.argsort(-values))
        weights = (num - ranks).astype(np.float64)
        weights /= max(float(weights.sum()), 1e-12)
        return weights

    def _construction_key(self, construction: list[float]) -> tuple:
        arr = np.asarray(construction, dtype=np.float64).reshape(-1)
        rounded = np.round(arr, self.key_precision)
        return tuple(float(x) for x in rounded.tolist())

    def _append_state(self, state: TTTState, is_initial: bool = False) -> bool:
        key = self._construction_key(state.construction)
        if key in self._construction_keys:
            return False
        if state.state_id in self._state_by_id:
            return False
        self._construction_keys.add(key)
        self._states.append(state)
        self._state_by_id[state.state_id] = state
        if is_initial:
            self._initial_state_ids.add(state.state_id)
        if state.parent_ids:
            self._children[state.parent_ids[0]].add(state.state_id)
        return True

    def _update_puct_stats(self, parent_state: TTTState, best_value: float | None, num_rollouts: int) -> None:
        lineage = [parent_state.state_id] + list(parent_state.parent_ids)
        num_rollouts = max(int(num_rollouts), 1)
        for _ in range(num_rollouts):
            for sid in lineage:
                self._n[sid] = self._n.get(sid, 0) + 1
            self._T += 1
        if best_value is not None:
            old = self._m.get(parent_state.state_id, best_value)
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
            grouped_children: dict[str, list[TTTState]] = defaultdict(list)
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
            self._construction_keys = {self._construction_key(s.construction) for s in self._states}

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
        self._construction_keys = {self._construction_key(s.construction) for s in self._states}
        self._children = defaultdict(set)
        for state in self._states:
            if state.parent_ids and state.parent_ids[0] in self._state_by_id:
                self._children[state.parent_ids[0]].add(state.state_id)

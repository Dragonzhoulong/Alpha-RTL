import json
import re
from typing import Any

import numpy as np

DEFAULT_REWARD_EPS = 1e-8


def extract_python_code(solution_str: str) -> str:
    """Extract Python code from model output with robust fallbacks."""
    text = str(solution_str or "").strip()
    if not text:
        return ""

    fenced_blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_blocks:
        candidate = fenced_blocks[-1].strip()
        if candidate:
            return candidate

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for marker in ("<|assistant|>", "<assistant>", "</assistant>", "<analysis>", "</analysis>"):
        normalized = normalized.replace(marker, "")

    run_pos = normalized.find("def run(")
    if run_pos != -1:
        candidate = normalized[run_pos:]
        if "```" in candidate:
            candidate = candidate.split("```", 1)[0]
        return candidate.strip()

    if "```" in normalized:
        content = normalized.split("```", 1)[-1]
        if "```" in content:
            content = content.split("```", 1)[0]
        content = content.strip()
        if content:
            return content

    return normalized.strip()


def normalize_erdos_construction(values: Any) -> np.ndarray:
    """Project an array into the feasible Erdos step-function constraints."""
    h_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if h_values.ndim != 1 or h_values.size < 2:
        raise ValueError("h_values must be a 1D array with at least 2 elements.")
    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contains NaN or inf.")

    h_values = np.clip(h_values, 0.0, 1.0)
    n_points = h_values.shape[0]
    target_sum = n_points / 2.0
    cur_sum = float(h_values.sum())
    if cur_sum <= 1e-12:
        h_values.fill(0.5)
        cur_sum = float(h_values.sum())
    h_values *= target_sum / cur_sum

    # Iteratively re-normalize when clipping pushes values out-of-range.
    for _ in range(8):
        if np.all((h_values >= 0.0) & (h_values <= 1.0)):
            break
        h_values = np.clip(h_values, 0.0, 1.0)
        fixed = (h_values <= 0.0) | (h_values >= 1.0)
        free = ~fixed
        if not np.any(free):
            break
        remain = target_sum - float(h_values[fixed].sum())
        free_sum = float(h_values[free].sum())
        if free_sum <= 1e-12:
            h_values[free] = remain / max(int(np.sum(free)), 1)
        else:
            h_values[free] *= remain / free_sum

    h_values = np.clip(h_values, 0.0, 1.0)
    final_sum = float(h_values.sum())
    if abs(final_sum - target_sum) > 1e-3:
        raise ValueError(f"Invalid construction sum: {final_sum} (target={target_sum}).")
    return h_values


def compute_erdos_c5_bound(h_values: Any) -> float:
    h_values = normalize_erdos_construction(h_values)
    n_points = h_values.shape[0]
    dx = 2.0 / n_points
    correlation = np.correlate(h_values, 1.0 - h_values, mode="full") * dx
    c5_bound = float(np.max(correlation))
    if not np.isfinite(c5_bound) or c5_bound <= 0.0:
        raise ValueError(f"Invalid C5 bound: {c5_bound}")
    return c5_bound


def evaluate_erdos_solution(h_values: Any, c5_bound: Any, n_points: Any) -> float:
    """Validate solution tuple and return computed c5."""
    n_points = int(n_points)
    arr = normalize_erdos_construction(h_values)
    if arr.shape[0] != n_points:
        raise ValueError(f"Expected n_points={n_points}, got {arr.shape[0]}")
    reported = float(c5_bound)
    computed = compute_erdos_c5_bound(arr)
    if not np.isclose(reported, computed, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported={reported:.6f}, computed={computed:.6f}")
    return computed


def reward_from_c5(c5_bound: float, eps: float = DEFAULT_REWARD_EPS) -> float:
    return float(1.0 / (eps + float(c5_bound)))


def to_serializable_list(values: Any) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(v) for v in arr.tolist()]


def try_deserialize_construction(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            value = json.loads(stripped)
        except Exception:
            return None
    try:
        return to_serializable_list(value)
    except Exception:
        return None

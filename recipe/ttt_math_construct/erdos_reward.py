from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
from typing import Any

import numpy as np

from recipe.ttt_math_construct.erdos_utils import (
    DEFAULT_REWARD_EPS,
    evaluate_erdos_solution,
    extract_python_code,
    normalize_erdos_construction,
    reward_from_c5,
    to_serializable_list,
)


def _compute_c5_from_h_values(h_values: Any) -> tuple[np.ndarray, float]:
    normalized = normalize_erdos_construction(h_values)
    n_points = int(normalized.shape[0])
    dx = 2.0 / n_points
    c5_bound = float(np.max(np.correlate(normalized, 1.0 - normalized, mode="full") * dx))
    return normalized, c5_bound


def _normalize_run_result(result: Any) -> tuple[np.ndarray, float]:
    h_values = None
    c5_bound = None
    n_points = None

    if isinstance(result, dict):
        h_values = result.get("h_values", result.get("h", result.get("construction", None)))
        c5_bound = result.get("c5_bound", result.get("c5", None))
        n_points = result.get("n_points", result.get("n", None))
    elif isinstance(result, (tuple, list)):
        if len(result) >= 1:
            h_values = result[0]
        if len(result) >= 2:
            c5_bound = result[1]
        if len(result) >= 3:
            n_points = result[2]
    else:
        h_values = result

    if h_values is None:
        raise ValueError("run() output has no construction values.")

    if c5_bound is None or n_points is None:
        return _compute_c5_from_h_values(h_values)

    checked_c5 = evaluate_erdos_solution(h_values=h_values, c5_bound=c5_bound, n_points=n_points)
    normalized = normalize_erdos_construction(h_values)
    return normalized, float(checked_c5)


def _evaluate_program_worker(
    queue: mp.Queue,
    code: str,
    initial_h_values: list[float],
    seed: int,
    budget_s: float,
) -> None:
    stdout_buffer = io.StringIO()
    try:
        initial_h = normalize_erdos_construction(initial_h_values)
        global_scope = {
            "np": np,
            "evaluate_erdos_solution": evaluate_erdos_solution,
            "initial_h_values": initial_h.copy(),
        }
        local_scope: dict[str, Any] = {}

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
            exec(code, global_scope, local_scope)  # noqa: S102
            run_fn = local_scope.get("run", None) or global_scope.get("run", None)
            if callable(run_fn):
                result = run_fn(seed=int(seed), budget_s=float(budget_s))
            else:
                result = None
                for key in ("h_values", "h", "construction", "best_h"):
                    value = local_scope.get(key, None)
                    if value is not None:
                        result = value
                        break
                if result is None:
                    for key in ("h_values", "h", "construction", "best_h"):
                        value = global_scope.get(key, None)
                        if value is not None:
                            result = value
                            break
                if result is None:
                    raise ValueError(
                        "Program must define callable `run(seed=..., budget_s=..., **kwargs)` "
                        "or set h_values/h/construction."
                    )

        normalized, c5_bound = _normalize_run_result(result)

        queue.put(
            {
                "ok": True,
                "raw_score": float(c5_bound),
                "construction": to_serializable_list(normalized),
                "stdout": stdout_buffer.getvalue().strip(),
            }
        )
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "error": str(exc),
                "stdout": stdout_buffer.getvalue().strip(),
            }
        )


def _run_with_timeout(
    code: str,
    initial_h_values: list[float],
    seed: int,
    budget_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    # Prefer fork on Linux/Ray workers to avoid pickling issues when this module
    # is dynamically imported as `custom_module` by VERL's reward loader.
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_evaluate_program_worker,
        args=(queue, code, initial_h_values, int(seed), float(budget_s)),
    )
    process.start()
    process.join(timeout=float(timeout_s))

    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
        queue.close()
        queue.join_thread()
        return {"ok": False, "error": f"Execution timeout after {timeout_s}s.", "stdout": ""}

    try:
        result = queue.get(timeout=0.1)
    except Exception:
        result = {"ok": False, "error": f"Process exited with code {process.exitcode}.", "stdout": ""}
    finally:
        queue.close()
        queue.join_thread()
    return result


def _failure_payload(parent_state_id: str, code: str, msg: str, failure_reward: float) -> dict[str, Any]:
    return {
        "score": float(failure_reward),
        "ttt_reward": float(failure_reward),
        "ttt_raw_score": None,
        "ttt_parent_raw_score": None,
        "ttt_is_improvement": False,
        "ttt_is_valid_solution": False,
        "ttt_result_construction": "",
        "ttt_parent_state_id": parent_state_id,
        "ttt_code": code,
        "ttt_msg": msg,
        "ttt_success": False,
    }


def compute_score(
    data_source: str,  # noqa: ARG001 - kept for reward manager signature
    solution_str: str,
    ground_truth: str,  # noqa: ARG001 - kept for reward manager signature
    extra_info: dict | None = None,
    timeout_s: float = 20.0,
    budget_s: float = 12.0,
    seed: int = 42,
    reward_eps: float = DEFAULT_REWARD_EPS,
    failure_reward: float = 0.0,
) -> dict[str, Any]:
    extra_info = extra_info or {}
    parent_state_id = str(extra_info.get("ttt_parent_state_id", ""))
    parent_raw_score = extra_info.get("ttt_parent_raw_score", None)
    initial_h_values = extra_info.get("ttt_parent_construction", None)
    if initial_h_values is None:
        return _failure_payload(
            parent_state_id=parent_state_id,
            code="",
            msg="Missing `ttt_parent_construction` in extra_info.",
            failure_reward=failure_reward,
        )

    code = extract_python_code(solution_str=solution_str)
    if not code.strip():
        return _failure_payload(
            parent_state_id=parent_state_id,
            code=code,
            msg="Empty python program.",
            failure_reward=failure_reward,
        )

    result = _run_with_timeout(
        code=code,
        initial_h_values=initial_h_values,
        seed=seed,
        budget_s=budget_s,
        timeout_s=timeout_s,
    )
    if not bool(result.get("ok", False)):
        err_msg = str(result.get("error", "Unknown execution error."))
        stdout = str(result.get("stdout", "")).strip()
        full_msg = err_msg if not stdout else f"{err_msg}\n{stdout}"
        return _failure_payload(
            parent_state_id=parent_state_id,
            code=code,
            msg=full_msg,
            failure_reward=failure_reward,
        )
    raw_score = float(result["raw_score"])
    reward = reward_from_c5(raw_score, eps=float(reward_eps))
    stdout = str(result.get("stdout", "")).strip()
    msg = f"C5 bound: {raw_score:.6f}"
    if stdout:
        msg = f"{msg}\n{stdout}"

    return {
        "score": reward,
        "ttt_reward": reward,
        "ttt_raw_score": raw_score,
        "ttt_parent_raw_score": (float(parent_raw_score) if parent_raw_score is not None else None),
        "ttt_is_improvement": (
            bool(raw_score < float(parent_raw_score)) if parent_raw_score is not None else False
        ),
        "ttt_is_valid_solution": True,
        # Keep this as string to avoid ragged np.array conversion in trainer.
        "ttt_result_construction": json.dumps(result["construction"], separators=(",", ":")),
        "ttt_parent_state_id": parent_state_id,
        "ttt_code": code,
        "ttt_msg": msg,
        "ttt_success": True,
    }

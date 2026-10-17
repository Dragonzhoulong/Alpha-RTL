"""TTT-Discover dataset for RTL generation with multi-problem support."""

from __future__ import annotations

import difflib
import os
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from .rtl_state import RTLState, RTLStatePool
from .rtl_utils import (
    RTLLMProblem,
    extract_tb_param_overrides,
    load_rtllm_problems,
    run_opensta_timing,
    run_yosys_synth,
)

# Import to trigger algorithm registration
from recipe.ttt_math_construct.algorithms import ensure_ttt_discover_algorithms_registered


class TTTRTLDataset(Dataset):
    """Dataset that renders prompts from a mutable TTT state pool for RTL generation."""

    def __init__(
        self,
        data_files: str | list[str],  # noqa: ARG002 - kept for compatible signature
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if processor is not None:
            raise ValueError("TTTRTLDataset currently supports text-only models.")

        ensure_ttt_discover_algorithms_registered()

        self.tokenizer = tokenizer
        self.config = config
        self.max_prompt_length = int(config.get("max_prompt_length", 4096))
        self.truncation = str(config.get("truncation", "left"))
        self.data_source = str(config.get("ttt_data_source", "ttt/rtl"))

        ttt_cfg = config.get("ttt_rtl", {})
        self.dataset_size = int(config.get("train_batch_size", 16))

        # Liberty file for technology-mapped synthesis (read from data.ttt_rtl config)
        self.liberty_file = str(ttt_cfg.get("liberty_file", ""))
        # OpenSTA config — must match reward_kwargs so baseline uses the same tool chain
        _opensta = ttt_cfg.get("opensta_binary", None)
        self.opensta_binary: str | None = str(_opensta) if _opensta else None
        self.clk_period_ns: float = float(ttt_cfg.get("clk_period_ns", 10.0))
        # When include_power is True, baseline PPA = area * delay * power_w (ADPP);
        # must match the reward_kwargs flag so r_ppa ratio is unit-consistent.
        self.include_power: bool = bool(ttt_cfg.get("include_power", False))
        # Paper-style power reporting knobs (see rtl_utils.run_opensta_timing).
        # Must match reward_kwargs for baseline/online unit consistency.
        _act = ttt_cfg.get("power_activity", None)
        self.power_activity: float | None = float(_act) if _act is not None and float(_act) >= 0 else None
        self.power_duty: float = float(ttt_cfg.get("power_duty", 0.5))
        self.power_clock_from_delay: bool = bool(ttt_cfg.get("power_clock_from_delay", False))
        self.power_clock_margin: float = float(ttt_cfg.get("power_clock_margin", 1.25))

        # Load RTLLM problems
        rtllm_dir = str(ttt_cfg.get("rtllm_dir", "RTLLM"))
        problems_str = str(ttt_cfg.get("problems", ""))
        problem_list = [p.strip() for p in problems_str.split(",") if p.strip()] if problems_str else None

        self.problems = load_rtllm_problems(rtllm_dir=rtllm_dir, problems=problem_list)
        if problem_list:
            missing = [problem_id for problem_id in problem_list if problem_id not in self.problems]
            if missing:
                available_problem_ids = sorted(load_rtllm_problems(rtllm_dir=rtllm_dir).keys())
                details: list[str] = []
                for problem_id in missing:
                    matches = difflib.get_close_matches(problem_id, available_problem_ids, n=3, cutoff=0.6)
                    if matches:
                        details.append(f"{problem_id} (did you mean: {', '.join(matches)})")
                    else:
                        details.append(problem_id)
                raise RuntimeError(
                    "Requested RTLLM problems were not found: "
                    + ", ".join(details)
                )
        if not self.problems:
            raise RuntimeError(f"No RTLLM problems found in {rtllm_dir}")

        # Auto-compute PPA baselines by synthesizing each problem's reference
        self.ppa_baselines = self._compute_ppa_baselines()

        # Initialize state pool
        self.state_pool = RTLStatePool(
            problems=self.problems,
            max_buffer_size=int(ttt_cfg.get("max_buffer_size", 500)),
            topk_children=int(ttt_cfg.get("topk_children", 2)),
            puct_c=float(ttt_cfg.get("puct_c", 1.0)),
            seed=int(ttt_cfg.get("seed", 42)),
            storage_dir=ttt_cfg.get("state_pool_dir", None),
            history_every=int(ttt_cfg.get("state_pool_history_every", 0)),
        )

        # Optional: pre-seed the pool with verified solutions so PUCT has a
        # func_ok=True parent to branch from on step 1. Safe to set on restored
        # pools — injection is idempotent via code-based deduplication.
        seed_file = ttt_cfg.get("seed_solutions_file", None)
        if seed_file:
            seed_path = os.path.expanduser(str(seed_file))
            if not os.path.isfile(seed_path):
                raise RuntimeError(f"seed_solutions_file not found: {seed_path}")
            import json as _json
            with open(seed_path, "r", encoding="utf-8") as f:
                seed_payload = _json.load(f)
            if isinstance(seed_payload, dict):
                seed_payload = seed_payload.get("solutions", [])
            added = self.state_pool.inject_verified_solutions(list(seed_payload))
            print(f"[TTTRTLDataset] Injected {added} verified seed solutions from {seed_path}")

        self._active_state_ids: list[str] = []
        self.set_active_state_ids(
            self.state_pool.sample_parent_state_ids(
                num_states=self.dataset_size,
                block_lineage=bool(ttt_cfg.get("lineage_blocking", True)),
            )
        )

    def __len__(self) -> int:
        return self.dataset_size

    def set_active_state_ids(self, state_ids: list[str]) -> None:
        if not state_ids:
            state_ids = self.state_pool.sample_parent_state_ids(num_states=max(self.dataset_size, 1), block_lineage=False)
        if not state_ids:
            raise RuntimeError("State pool is empty; cannot sample active states.")
        repeats = (self.dataset_size + len(state_ids) - 1) // len(state_ids)
        tiled = (state_ids * repeats)[: self.dataset_size]
        self._active_state_ids = tiled

    def _get_state(self, item: int) -> RTLState:
        state_id = self._active_state_ids[item]
        state = self.state_pool.get_state(state_id)
        if state is not None:
            return state
        fallback_ids = self.state_pool.sample_parent_state_ids(num_states=1, block_lineage=False)
        if not fallback_ids:
            raise RuntimeError("State pool became empty unexpectedly.")
        fallback = self.state_pool.get_state(fallback_ids[0])
        if fallback is None:
            raise RuntimeError("Failed to fetch fallback state.")
        return fallback

    def _get_problem(self, problem_id: str) -> RTLLMProblem:
        problem = self.problems.get(problem_id)
        if problem is None:
            raise RuntimeError(f"Problem {problem_id} not found in loaded problems.")
        return problem

    def _compute_ppa_baselines(self) -> dict[str, dict[str, float]]:
        """Synthesize each problem's reference to get a baseline ADPP or ADP.

        Requires ``liberty_file`` and ``opensta_binary``. The baseline's
        ``primary`` field uses the same unit as the online reward — ADPP when
        ``include_power`` is on, ADP otherwise — so ``r_ppa = primary / online``
        is unit-consistent. If the baseline can't be computed, ``primary`` stays
        absent and the online reward falls back to a scale-matched normalizer.
        """
        import os
        import shutil
        import tempfile

        if not self.liberty_file or not os.path.isfile(self.liberty_file):
            raise RuntimeError(
                f"TTTRTLDataset requires a valid liberty_file; got {self.liberty_file!r}."
            )
        if not self.opensta_binary:
            raise RuntimeError(
                "TTTRTLDataset requires opensta_binary — the baseline metric must "
                "match the online reward tool chain."
            )

        baselines: dict[str, dict[str, float]] = {}
        for pid, problem in self.problems.items():
            ref_code = self._read_reference(problem)
            if not ref_code:
                print(
                    f"[TTTRTLDataset] WARNING: no reference implementation for {pid}; "
                    f"ppa_baseline=0 → online reward will use scale-matched fallback normalizers."
                )
                continue
            work_dir = tempfile.mkdtemp(prefix=f"ttt_rtl_baseline_{pid}_")
            try:
                tb_param_overrides = extract_tb_param_overrides(
                    problem.testbench_path, problem.design_name
                )
                result = run_yosys_synth(
                    verilog_code=ref_code,
                    design_name=problem.design_name,
                    work_dir=work_dir,
                    timeout_s=60.0,
                    liberty_file=self.liberty_file,
                    param_overrides=tb_param_overrides or None,
                )
                if not (result["success"] and result["area"] > 0 and result.get("netlist_path")):
                    log_tail = (result.get("log") or "").splitlines()[-5:]
                    print(
                        f"[TTTRTLDataset] WARNING: reference synthesis failed for {pid} "
                        f"(success={result['success']}, area={result.get('area', 0)}); "
                        f"ppa_baseline=0 \u2192 online reward will use the scale-matched fallback normalizer."
                    )
                    if log_tail:
                        print(f"[TTTRTLDataset]   yosys log tail: {' | '.join(log_tail)}")
                    continue

                area = float(result["area"])
                sta = run_opensta_timing(
                    netlist_path=result["netlist_path"],
                    top_module=result.get("top_module", problem.design_name),
                    liberty_file=self.liberty_file,
                    timeout_s=60.0,
                    opensta_binary=self.opensta_binary,
                    clk_period_ns=self.clk_period_ns,
                    power_activity=self.power_activity,
                    power_duty=self.power_duty,
                    power_clock_from_delay=self.power_clock_from_delay,
                    power_clock_margin=self.power_clock_margin,
                )
                if not (sta["success"] and sta["delay_ps"] > 0):
                    print(
                        f"[TTTRTLDataset] WARNING: OpenSTA failed on reference for {pid}; "
                        f"ppa_baseline=0 \u2192 online reward will use the scale-matched fallback normalizer."
                    )
                    continue

                delay = float(sta["delay_ps"])
                adp = area * delay
                power_w = float(sta.get("power_w", 0.0))
                adpp = adp * power_w if power_w > 0 else 0.0

                if self.include_power:
                    if adpp <= 0:
                        print(
                            f"[TTTRTLDataset] WARNING: include_power=True but power report "
                            f"failed for {pid}; ppa_baseline=0 \u2192 online reward will use "
                            f"the scale-matched fallback normalizer."
                        )
                        continue
                    primary = adpp
                    print(
                        f"[TTTRTLDataset] PPA baseline for {pid} (ADPP): "
                        f"area={area:.1f}\u03bcm\u00b2, delay={delay:.1f}ps, "
                        f"power={power_w * 1e6:.3f}\u03bcW, ADPP={adpp:.3e}"
                    )
                else:
                    primary = adp
                    print(
                        f"[TTTRTLDataset] PPA baseline for {pid} (ADP): "
                        f"area={area:.1f}\u03bcm\u00b2, delay={delay:.1f}ps, ADP={adp:.1f}"
                    )

                baselines[pid] = {
                    "area": area,
                    "delay": delay,
                    "power": power_w,
                    "adp": adp,
                    "adpp": adpp,
                    "primary": primary,
                }
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
        return baselines

    def _read_reference(self, problem: RTLLMProblem) -> str:
        """Read the verified reference implementation and fix the module name."""
        import os
        import re
        if not problem.reference_path or not os.path.isfile(problem.reference_path):
            return ""
        try:
            with open(problem.reference_path, "r", encoding="utf-8") as f:
                code = f.read().strip()
            # Extract the expected module name from the testbench
            expected_name = self._extract_tb_module_name(problem)
            if expected_name:
                # Replace verified module name with the name testbench expects
                code = re.sub(
                    r"\bmodule\s+verified_\w+",
                    f"module {expected_name}",
                    code,
                    count=1,
                )
            return code
        except Exception:
            return ""

    @staticmethod
    def _extract_tb_module_name(problem: RTLLMProblem) -> str:
        """Extract the DUT module name from the testbench file."""
        import os
        import re
        tb_path = problem.testbench_path
        if not tb_path or not os.path.isfile(tb_path):
            return ""
        try:
            with open(tb_path, "r", encoding="utf-8") as f:
                tb_code = f.read()
            # Match DUT instantiation: module_name #(...) inst_name (
            # or: module_name inst_name (
            # Process line by line to skip comments and module declaration
            skip = {"module", "reg", "wire", "input", "output", "assign",
                    "always", "initial", "begin", "end", "integer", "parameter",
                    "localparam", "genvar", "generate", "endgenerate", "if",
                    "else", "for", "case", "endcase", "endmodule", "function",
                    "task", "timescale"}
            for line in tb_code.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("//") or stripped.startswith("`"):
                    continue
                # Match: identifier #(...) identifier ( or identifier identifier (
                m = re.match(r"\s*(\w+)\s+(?:#\s*\(.*?\)\s+)?(\w+)\s*\(", stripped)
                if m:
                    mod_name = m.group(1)
                    if mod_name not in skip and not mod_name.startswith("$"):
                        return mod_name
        except Exception:
            pass
        return ""

    def _build_user_prompt(self, state: RTLState, problem: RTLLMProblem) -> str:
        parts: list[str] = []
        baseline_info = self.ppa_baselines.get(problem.problem_id, {})
        metric_label = "A*D*P" if self.include_power else "ADP"

        def _fmt_metric(v: float) -> str:
            return f"{v:.3e}" if self.include_power else f"{v:.1f}"

        baseline_primary = float(baseline_info.get("primary", 0.0)) if isinstance(baseline_info, dict) else 0.0
        baseline_area = float(baseline_info.get("area", 0.0)) if isinstance(baseline_info, dict) else 0.0
        baseline_delay = float(baseline_info.get("delay", 0.0)) if isinstance(baseline_info, dict) else 0.0
        has_baseline = baseline_primary > 0

        # For initial states, provide the verified reference as a starting point
        if not state.verilog_code and problem.reference_path:
            ref_code = self._read_reference(problem)
            if ref_code:
                parts.append("## Reference Implementation\n")
                if has_baseline:
                    parts.append(
                        f"Below is a known working implementation that synthesizes "
                        f"to area={baseline_area:.1f}\u03bcm\u00b2, delay={baseline_delay:.1f}ps "
                        f"({metric_label}={_fmt_metric(baseline_primary)}). Your task is to produce "
                        f"a functionally correct implementation with lower {metric_label}.\n"
                    )
                else:
                    parts.append(
                        f"Below is a known working implementation. Your task is to produce a "
                        f"functionally correct implementation that minimizes {metric_label} in "
                        f"tech-mapped synthesis (lower is better).\n"
                    )
                parts.append("```verilog")
                _ref_char_limit = max(6000, (self.max_prompt_length - 3000) * 4)
                if len(ref_code) > _ref_char_limit:
                    ref_code = "// ...(beginning truncated)\n" + ref_code[-_ref_char_limit:]
                parts.append(ref_code)
                parts.append("```\n")

        parts.append("## Design Specification\n")
        parts.append(problem.design_description)
        parts.append("")

        if state.verilog_code:
            parts.append("## Previous Attempt\n")
            parts.append("```verilog")
            code_text = state.verilog_code
            _code_char_budget = max(3000, (self.max_prompt_length - 2000) * 4)
            if len(code_text) > _code_char_budget:
                code_text = code_text[:_code_char_budget] + "\n// ...(truncated)"
            parts.append(code_text)
            parts.append("```\n")

            parts.append("## Feedback from Previous Attempt\n")
            parts.append(f"- Syntax: {'PASS' if state.syntax_ok else 'FAIL'}")
            parts.append(f"- Functional Test: {'PASS' if state.func_ok else ('FAIL' if state.syntax_ok else 'N/A')}")

            if state.ppa_score > 0 and state.ppa_area > 0 and state.ppa_delay > 0:
                ppa_line = (
                    f"- Synthesis: area={state.ppa_area:.1f}\u03bcm\u00b2, "
                    f"delay={state.ppa_delay:.1f}ps, "
                    f"{metric_label}={_fmt_metric(state.ppa_score)}"
                )
                if has_baseline:
                    ppa_line += f" (reference {metric_label}: {_fmt_metric(baseline_primary)})"
                parts.append(ppa_line)

                if state.parent_ids:
                    parent = self.state_pool.get_state(state.parent_ids[0])
                    if parent and parent.ppa_score > 0:
                        delta = state.ppa_score - parent.ppa_score
                        if delta < 0:
                            parts.append(f"  Previous {metric_label}: {_fmt_metric(parent.ppa_score)} -> current: {_fmt_metric(state.ppa_score)} (improved by {_fmt_metric(-delta)})")
                        elif delta > 0:
                            parts.append(f"  Previous {metric_label}: {_fmt_metric(parent.ppa_score)} -> current: {_fmt_metric(state.ppa_score)} (regressed by {_fmt_metric(delta)})")

            if state.observation and state.observation.strip():
                obs = state.observation.strip()
                if len(obs) > 800:
                    obs = obs[:800] + "\n...(truncated)"
                parts.append(f"\n{obs}")

            if state.func_ok and state.ppa_score > 0 and state.ppa_area > 0 and state.ppa_delay > 0:
                parts.append(
                    f"\nYou are iteratively optimizing {metric_label} (lower is better). "
                    f"Current: area={state.ppa_area:.1f}\u03bcm\u00b2, delay={state.ppa_delay:.1f}ps, {metric_label}={_fmt_metric(state.ppa_score)}. "
                    f"Produce a functionally correct design with {metric_label} lower than {_fmt_metric(state.ppa_score)}."
                )
            elif state.func_ok:
                parts.append("\nThe design is functionally correct. Try to optimize for better PPA in synthesis.")
            else:
                parts.append("\nImprove the design based on this feedback.")
        else:
            if has_baseline:
                parts.append(
                    f"Write a complete Verilog module that satisfies the specification above "
                    f"with {metric_label} lower than {_fmt_metric(baseline_primary)} when synthesized. "
                    f"Lower {metric_label} is better."
                )
            else:
                parts.append(
                    f"Write a complete Verilog module that satisfies the specification above. "
                    f"Minimize {metric_label} in tech-mapped synthesis (lower is better)."
                )

        parts.append("\nOutput only Verilog code within ```verilog fences.")

        return "\n".join(parts)

    def __getitem__(self, item: int) -> dict:
        state = self._get_state(item)
        problem = self._get_problem(state.problem_id)

        messages = [
            {"role": "system", "content": "You are an expert Verilog RTL designer."},
            {"role": "user", "content": self._build_user_prompt(state, problem)},
        ]
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} exceeds max_prompt_length.")

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "data_source": self.data_source,
            "reward_model": {"ground_truth": ""},
            "extra_info": {
                "ttt_parent_state_id": state.state_id,
                "ttt_problem_id": problem.problem_id,
                "ttt_testbench_path": problem.testbench_path,
                "ttt_design_name": problem.design_name,
                "ttt_data_files": ",".join(problem.data_files),
                "ttt_parent_raw_score": float(state.raw_score),
                "ttt_ppa_baseline": float(
                    self.ppa_baselines.get(problem.problem_id, {}).get("primary", 0.0)
                    if isinstance(self.ppa_baselines.get(problem.problem_id), dict)
                    else self.ppa_baselines.get(problem.problem_id, 0)
                ),
            },
            "index": item,
        }

    def on_batch_end(self, batch) -> None:
        step = int(batch.meta_info.get("global_steps", -1)) + 1
        self.state_pool.save(step=step)

"""TTT-Discover dataset for C910 FPU component optimization."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from recipe.ttt_rtl.rtl_state import RTLState, RTLStatePool
from recipe.ttt_math_construct.algorithms import ensure_ttt_discover_algorithms_registered

from .c910_utils import (
    C910Problem,
    load_c910_problems,
    run_opensta_timing,
    run_yosys_synth,
)


class _C910ProblemAdapter:
    """Wraps C910Problem to satisfy RTLStatePool's problems dict interface.

    RTLStatePool expects dict values with .problem_id attribute — C910Problem
    already has this, so this is a thin passthrough.
    """

    def __init__(self, problem: C910Problem):
        self.problem_id = problem.problem_id


class TTTC910Dataset(Dataset):
    """Dataset that renders prompts from a mutable TTT state pool for C910 FPU optimization."""

    def __init__(
        self,
        data_files: str | list[str],  # noqa: ARG002
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if processor is not None:
            raise ValueError("TTTC910Dataset currently supports text-only models.")

        ensure_ttt_discover_algorithms_registered()

        self.tokenizer = tokenizer
        self.config = config
        self.max_prompt_length = int(config.get("max_prompt_length", 4096))
        self.truncation = str(config.get("truncation", "left"))
        self.data_source = str(config.get("ttt_data_source", "ttt/c910"))

        ttt_cfg = config.get("ttt_c910", {})
        self.dataset_size = int(config.get("train_batch_size", 16))

        # Liberty & OpenSTA config
        self.liberty_file = str(ttt_cfg.get("liberty_file", ""))
        _opensta = ttt_cfg.get("opensta_binary", None)
        self.opensta_binary: str | None = str(_opensta) if _opensta else None
        self.clk_period_ns: float = float(ttt_cfg.get("clk_period_ns", 10.0))
        self.enable_thinking: bool = bool(ttt_cfg.get("enable_thinking", True))

        # Load C910 problems
        problems_dir = str(ttt_cfg.get("problems_dir", os.path.join(os.path.dirname(__file__), "problems")))
        problems_str = str(ttt_cfg.get("problems", ""))
        problem_list = [p.strip() for p in problems_str.split(",") if p.strip()] if problems_str else None

        self.problems = load_c910_problems(problems_dir=problems_dir, problems=problem_list)
        if not self.problems:
            raise RuntimeError(f"No C910 problems found in {problems_dir}")

        # Auto-compute PPA baselines
        self.ppa_baselines = self._compute_ppa_baselines()

        # RTLStatePool expects dict of objects with .problem_id
        pool_problems = {pid: _C910ProblemAdapter(p) for pid, p in self.problems.items()}

        self.state_pool = RTLStatePool(
            problems=pool_problems,
            max_buffer_size=int(ttt_cfg.get("max_buffer_size", 500)),
            topk_children=int(ttt_cfg.get("topk_children", 2)),
            puct_c=float(ttt_cfg.get("puct_c", 1.0)),
            seed=int(ttt_cfg.get("seed", 42)),
            storage_dir=ttt_cfg.get("state_pool_dir", None),
            history_every=int(ttt_cfg.get("state_pool_history_every", 0)),
        )

        # Seed the pool with verified solutions from disk (safe to call on
        # restored pools — dedup skips already-present states).
        if bool(ttt_cfg.get("seed_from_solutions", True)):
            n_seeded = self._seed_pool_from_problems(problems_dir, ttt_cfg)
            if n_seeded > 0:
                self.state_pool.save(step=0)
                print(f"[TTTC910Dataset] Seeded state pool with {n_seeded} verified solutions")

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

    def _get_problem(self, problem_id: str) -> C910Problem:
        problem = self.problems.get(problem_id)
        if problem is None:
            raise RuntimeError(f"Problem {problem_id} not found.")
        return problem

    def _compute_ppa_baselines(self) -> dict[str, dict[str, float]]:
        """Synthesize each problem's reference implementation to get baseline PPA."""
        baselines: dict[str, dict[str, float]] = {}
        for pid, problem in self.problems.items():
            ref_code = self._read_reference(problem)
            if not ref_code:
                continue
            work_dir = tempfile.mkdtemp(prefix=f"ttt_c910_baseline_{pid}_")
            try:
                result = run_yosys_synth(
                    verilog_code=ref_code,
                    design_name=problem.design_name,
                    work_dir=work_dir,
                    timeout_s=120.0,
                    liberty_file=self.liberty_file if self.liberty_file else None,
                    dep_files=problem.dep_files,
                )
                if result["success"] and result["area"] > 0:
                    area = float(result["area"])
                    delay = float(result.get("delay", 0.0))
                    adp = float(result.get("adp", 0.0))

                    if (
                        self.opensta_binary
                        and result.get("netlist_path")
                        and os.path.isfile(result.get("netlist_path", ""))
                        and self.liberty_file
                    ):
                        sta = run_opensta_timing(
                            netlist_path=result["netlist_path"],
                            top_module=result.get("top_module", problem.design_name),
                            liberty_file=self.liberty_file,
                            timeout_s=60.0,
                            opensta_binary=self.opensta_binary,
                            clk_period_ns=self.clk_period_ns,
                        )
                        if sta["success"] and sta["delay_ps"] > 0:
                            delay = sta["delay_ps"]
                            adp = area * delay

                    print(
                        f"[TTTC910Dataset] PPA baseline for {pid}: "
                        f"area={area:.1f}\u03bcm\u00b2, delay={delay:.1f}ps, ADP={adp:.1f}"
                    )
                    baselines[pid] = {"area": area, "delay": delay, "adp": adp}
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
        return baselines

    def _read_reference(self, problem: C910Problem) -> str:
        """Read the reference implementation."""
        if not problem.reference_path or not os.path.isfile(problem.reference_path):
            return ""
        try:
            with open(problem.reference_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    def _build_user_prompt(self, state: RTLState, problem: C910Problem) -> str:
        parts: list[str] = []
        baseline_info = self.ppa_baselines.get(problem.problem_id, {})
        baseline_adp = baseline_info.get("adp", 0.0)
        baseline_area = baseline_info.get("area", 0.0)
        baseline_delay = baseline_info.get("delay", 0.0)
        use_adp = baseline_adp > 0

        # For initial states, put reference code BEFORE spec so that
        # left-truncation preserves the critical spec/interface at the end.
        if not state.verilog_code:
            ref_code = self._read_reference(problem)
            if ref_code:
                parts.append("## Reference Implementation\n")
                if use_adp:
                    parts.append(
                        f"The reference synthesizes to area={baseline_area:.1f}\u03bcm\u00b2, "
                        f"delay={baseline_delay:.1f}ps (ADP={baseline_adp:.1f}). "
                        f"Produce a functionally correct implementation with lower ADP.\n"
                    )
                parts.append("```verilog")
                parts.append(ref_code)
                parts.append("```\n")

            if problem.prompt_template:
                parts.append(problem.prompt_template)
            else:
                parts.append("## Design Specification\n")
                parts.append(problem.description)

            if use_adp:
                parts.append(
                    f"\nOptimize for ADP (Area \u00d7 Delay Product) lower than {baseline_adp:.1f}."
                )
        else:
            # Iterative improvement: show full spec + previous attempt + feedback.
            # Inspired by Discover's prompt: keep the complete specification (including
            # building blocks, constraints, truth tables) so the model has full context
            # for making informed improvements.
            if problem.prompt_template:
                # Show full template — the template contains critical info like encoding
                # direction tables and building block interfaces needed for correctness.
                # For truncation safety, we strip the "Baseline Implementation" section
                # (the model already has its own code) and the "Optimization Goals" section
                # (replaced by specific feedback below).
                template_lines = problem.prompt_template.split("\n")
                filtered: list[str] = []
                skip_section = False
                for line in template_lines:
                    if line.startswith("## Baseline Implementation"):
                        skip_section = True
                        continue
                    elif line.startswith("## Optimization Goals"):
                        skip_section = True
                        continue
                    elif skip_section and line.startswith("## "):
                        skip_section = False
                    if not skip_section:
                        filtered.append(line)
                parts.append("\n".join(filtered))
            else:
                parts.append("## Design Specification\n")
                parts.append(problem.description)

            parts.append("\n## Previous Attempt\n")
            parts.append("```verilog")
            code_text = state.verilog_code
            parts.append(code_text)
            parts.append("```\n")

            parts.append("## Feedback from Previous Attempt\n")
            parts.append(f"- Syntax: {'PASS' if state.syntax_ok else 'FAIL'}")
            parts.append(f"- Functional Test: {'PASS' if state.func_ok else ('FAIL' if state.syntax_ok else 'N/A')}")

            if state.ppa_score > 0:
                if state.ppa_area > 0 and state.ppa_delay > 0:
                    ppa_line = (
                        f"- Synthesis: area={state.ppa_area:.1f}\u03bcm\u00b2, "
                        f"delay={state.ppa_delay:.1f}ps, ADP={state.ppa_score:.1f}"
                    )
                    if use_adp:
                        ppa_line += f" (reference ADP: {baseline_adp:.1f})"
                    parts.append(ppa_line)

            if state.observation and state.observation.strip():
                obs = state.observation.strip()
                # Keep the tail (error details are at the end of simulation output)
                if len(obs) > 800:
                    obs = obs[-800:]
                    obs = "...(truncated)\n" + obs
                parts.append(f"\n{obs}")

            if state.func_ok and state.ppa_score > 0:
                if state.ppa_area > 0 and state.ppa_delay > 0:
                    parts.append(
                        f"\nYou are iteratively optimizing ADP (lower is better). "
                        f"Current: area={state.ppa_area:.1f}\u03bcm\u00b2, "
                        f"delay={state.ppa_delay:.1f}ps, ADP={state.ppa_score:.1f}. "
                        f"Target: ADP lower than {state.ppa_score:.1f}. "
                        f"Try a different encoding strategy — e.g., a different tree topology, "
                        f"direct boolean equations, or alternative grouping of the 24 bits."
                    )
            elif state.func_ok:
                parts.append(
                    "\nThe design is functionally correct. "
                    "Now optimize for lower ADP by trying a different priority encoding strategy."
                )
            elif state.syntax_ok:
                parts.append(
                    "\nThe design compiles but fails functional tests. "
                    "Check that the pre-encoding formulas match the specification exactly, "
                    "and verify the priority encoding direction (MSB-first: bit 23 set → result 0)."
                )
            else:
                parts.append(
                    "\nThe design has syntax errors. "
                    "Use valid Verilog-2001/2005 syntax. Avoid SystemVerilog-only constructs. "
                    "Declare all wires and regs explicitly."
                )

        parts.append("\nOutput only Verilog code within ```verilog fences.")
        return "\n".join(parts)

    def __getitem__(self, item: int) -> dict:
        state = self._get_state(item)
        problem = self._get_problem(state.problem_id)

        messages = [
            {"role": "system", "content": "You are an expert Verilog RTL designer specializing in high-performance arithmetic units."},
            {"role": "user", "content": self._build_user_prompt(state, problem)},
        ]
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking)
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
                "ttt_parent_raw_score": float(state.raw_score),
                "ttt_ppa_baseline": float(
                    self.ppa_baselines.get(problem.problem_id, {}).get("adp", 0.0)
                ),
            },
            "index": item,
        }

    def _seed_pool_from_problems(self, problems_dir: str, ttt_cfg) -> int:
        """Load verified solutions from each problem's solutions/ directory and inject
        them as permanent initial states in the PUCT pool.

        Expects the following layout (created by format_sft_data.py / measure_solutions.py):
            {problems_dir}/{problem_id}/solutions/reward_measurements.json
            {problems_dir}/{problem_id}/solutions/{sol_name}/{design_name}.v

        Returns the total number of states injected (0 on restore where all are duplicates).
        """
        solutions: list[dict] = []
        for problem_id, problem in self.problems.items():
            sol_root = os.path.join(problems_dir, problem_id, "solutions")
            measurements_path = os.path.join(sol_root, "reward_measurements.json")
            if not os.path.isfile(measurements_path):
                continue
            try:
                with open(measurements_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"[TTTC910Dataset] Warning: could not read {measurements_path}: {exc}")
                continue

            for sol_name, metrics in data.get("results", {}).items():
                if not metrics.get("testbench_pass", False):
                    continue
                adp = float(metrics.get("adp", 0.0))
                if adp <= 0:
                    continue
                verilog_path = os.path.join(sol_root, sol_name, f"{problem.design_name}.v")
                if not os.path.isfile(verilog_path):
                    continue
                try:
                    with open(verilog_path, "r", encoding="utf-8") as f:
                        code = f.read().strip()
                except Exception:
                    continue
                if not code:
                    continue
                solutions.append({
                    "problem_id": problem_id,
                    "verilog_code": code,
                    "value": float(metrics.get("reward", 0.0)),
                    "raw_score": float(metrics.get("reward", 0.0)),
                    "ppa_score": adp,
                    "ppa_area": float(metrics.get("area", 0.0)),
                    "ppa_delay": float(metrics.get("delay_ps", 0.0)),
                })

        if not solutions:
            return 0
        n = self.state_pool.inject_verified_solutions(solutions)
        print(f"[TTTC910Dataset] inject_verified_solutions: {n}/{len(solutions)} added "
              f"({len(solutions) - n} skipped as duplicates)")
        return n

    def on_batch_end(self, batch) -> None:
        step = int(batch.meta_info.get("global_steps", -1)) + 1
        self.state_pool.save(step=step)

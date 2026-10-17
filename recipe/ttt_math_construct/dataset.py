from __future__ import annotations

from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from .algorithms import ensure_ttt_discover_algorithms_registered
from .state_pool import PUCTStatePool, TTTState


class TTTErdosDataset(Dataset):
    """Dataset that renders prompts from a mutable TTT state pool."""

    def __init__(
        self,
        data_files: str | list[str],  # noqa: ARG002 - kept for compatible signature
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if processor is not None:
            raise ValueError("TTTErdosDataset currently supports text-only models.")

        ensure_ttt_discover_algorithms_registered()

        self.tokenizer = tokenizer
        self.config = config
        self.max_prompt_length = int(config.get("max_prompt_length", 2048))
        self.truncation = str(config.get("truncation", "left"))
        self.data_source = str(config.get("ttt_data_source", "ttt/erdos_min_overlap"))

        ttt_cfg = config.get("ttt_erdos", {})
        self.target_c5 = float(ttt_cfg.get("target_c5", 0.3808))
        self.dataset_size = int(config.get("train_batch_size", 64))

        self.state_pool = PUCTStatePool(
            initial_state_count=int(ttt_cfg.get("initial_state_count", self.dataset_size)),
            min_points=int(ttt_cfg.get("min_points", 40)),
            max_points=int(ttt_cfg.get("max_points", 100)),
            max_buffer_size=int(ttt_cfg.get("max_buffer_size", 1000)),
            topk_children=int(ttt_cfg.get("topk_children", 2)),
            puct_c=float(ttt_cfg.get("puct_c", 1.0)),
            key_precision=int(ttt_cfg.get("key_precision", 8)),
            seed=int(ttt_cfg.get("seed", 42)),
            storage_dir=ttt_cfg.get("state_pool_dir", None),
            history_every=int(ttt_cfg.get("state_pool_history_every", 0)),
        )

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

    def _get_state(self, item: int) -> TTTState:
        state_id = self._active_state_ids[item]
        state = self.state_pool.get_state(state_id)
        if state is not None:
            return state
        # Fallback for rare stale ids after pruning.
        fallback_ids = self.state_pool.sample_parent_state_ids(num_states=1, block_lineage=False)
        if not fallback_ids:
            raise RuntimeError("State pool became empty unexpectedly.")
        fallback = self.state_pool.get_state(fallback_ids[0])
        if fallback is None:
            raise RuntimeError("Failed to fetch fallback state.")
        return fallback

    def _build_user_prompt(self, state: TTTState) -> str:
        state_ctx = state.to_prompt(target_c5=self.target_c5)
        construction_section = ""
        if state.construction:
            construction_section = (
                "You may want to start your search from the current construction, "
                f"which you can access through the `initial_h_values` global variable (n={len(state.construction)} samples).\n"
                "You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.\n"
            )
        if state.code and state.code.strip():
            code_section = (
                "Reason about how you could further improve this construction.\n"
                "Ideally, try to do something different than the above algorithm. "
                "This could mean using different algorithmic ideas, adjusting heuristics, or sweeping different hyperparameters.\n"
                "Unless you make a meaningful improvement, you will not be rewarded."
            )
        else:
            code_section = "Write code to optimize this construction."
        return (
            "You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.\n"
            "Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C5.\n\n"
            "## Problem\n\n"
            "Find a step function h: [0, 2] -> [0, 1] that minimizes the overlap integral:\n\n"
            "C5 = max_k integral h(x) * (1 - h(x + k)) dx\n\n"
            "**Constraints**:\n"
            "1. h(x) in [0, 1] for all x\n"
            "2. integral_0^2 h(x) dx = 1\n\n"
            "**Discretization**: represent h as n_points samples over [0, 2].\n"
            "With dx = 2.0 / n_points:\n"
            "- 0 <= h[i] <= 1 for all i\n"
            "- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)\n\n"
            "The evaluation computes: C5 = max(np.correlate(h, 1 - h, mode='full') * dx)\n\n"
            "Smaller sequences with less than 1k samples are preferred because they are faster to optimize and evaluate.\n\n"
            "Lower C5 values are better because they provide tighter upper bounds on the Erdős constant.\n\n"
            "## Budget & Resources\n"
            "- Time budget: 1000s for your code to run\n"
            "- CPUs: 2 available\n\n"
            "## Rules\n"
            "- Define `run(seed=42, budget_s=1000, **kwargs)` that returns `(h_values, c5_bound, n_points)`.\n"
            "- Use scipy, numpy, cvxpy, or math.\n"
            "- Make all helper functions top level; do not use closures or lambdas.\n"
            "- No filesystem or network IO.\n"
            "- `evaluate_erdos_solution()` and `initial_h_values` are pre-imported.\n"
            "- Your function must complete within `budget_s` seconds and return the best solution found.\n"
            "- Output only Python code. Do not add markdown fences or prose.\n\n"
            "Lower is better. Current record: C5 <= 0.38092. "
            f"Our goal is to find a construction that shows C5 <= {self.target_c5:.5f}.\n\n"
            f"{state_ctx}\n\n"
            f"{construction_section}"
            f"{code_section}\n"
        )

    def __getitem__(self, item: int) -> dict:
        state = self._get_state(item)
        messages = [
            {"role": "system", "content": "You are a careful scientific programming assistant."},
            {"role": "user", "content": self._build_user_prompt(state)},
        ]
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
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
                "ttt_parent_construction": list(state.construction),
                "ttt_parent_raw_score": float(state.raw_score),
            },
            "index": item,
        }

    def on_batch_end(self, batch) -> None:
        step = int(batch.meta_info.get("global_steps", -1)) + 1
        self.state_pool.save(step=step)

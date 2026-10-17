# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collect diverse Verilog solutions from strong models via OpenRouter."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import aiohttp
import yaml

from recipe.ttt_rtl.rtl_utils import RTLLMProblem, load_rtllm_problems, run_yosys_synth


def _read_reference(problem: RTLLMProblem) -> str:
    """Read the verified reference implementation and fix the module name.

    Replicates TTTRTLDataset._read_reference logic.
    """
    if not problem.reference_path or not os.path.isfile(problem.reference_path):
        return ""
    try:
        with open(problem.reference_path, "r", encoding="utf-8") as f:
            code = f.read().strip()
        expected_name = _extract_tb_module_name(problem)
        if expected_name:
            code = re.sub(r"\bmodule\s+verified_\w+", f"module {expected_name}", code, count=1)
        return code
    except Exception:
        return ""


def _extract_tb_module_name(problem: RTLLMProblem) -> str:
    """Extract the DUT module name from the testbench file.

    Replicates TTTRTLDataset._extract_tb_module_name logic.
    """
    tb_path = problem.testbench_path
    if not tb_path or not os.path.isfile(tb_path):
        return ""
    try:
        with open(tb_path, "r", encoding="utf-8") as f:
            tb_code = f.read()
        skip = {
            "module", "reg", "wire", "input", "output", "assign",
            "always", "initial", "begin", "end", "integer", "parameter",
            "localparam", "genvar", "generate", "endgenerate", "if",
            "else", "for", "case", "endcase", "endmodule", "function",
            "task", "timescale",
        }
        for line in tb_code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("`"):
                continue
            m = re.match(r"\s*(\w+)\s+(?:#\s*\(.*?\)\s+)?(\w+)\s*\(", stripped)
            if m:
                mod_name = m.group(1)
                if mod_name not in skip and not mod_name.startswith("$"):
                    return mod_name
    except Exception:
        pass
    return ""


def compute_ppa_baselines(problems: dict[str, RTLLMProblem]) -> dict[str, int]:
    """Synthesize each problem's reference to get baseline cell counts.

    Replicates TTTRTLDataset._compute_ppa_baselines logic.
    """
    baselines: dict[str, int] = {}
    for pid, problem in problems.items():
        ref_code = _read_reference(problem)
        if not ref_code:
            continue
        work_dir = tempfile.mkdtemp(prefix=f"ttt_rtl_baseline_{pid}_")
        try:
            result = run_yosys_synth(
                verilog_code=ref_code,
                design_name=problem.design_name,
                work_dir=work_dir,
                timeout_s=60.0,
            )
            if result["success"] and result["area"] > 0:
                baselines[pid] = int(result["area"])
                print(f"[baseline] {pid}: {baselines[pid]} cells")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    return baselines


def build_initial_prompt(problem: RTLLMProblem, baseline: int) -> str:
    """Build the initial-state user prompt for a problem.

    Replicates the initial-state branch of TTTRTLDataset._build_user_prompt.
    """
    parts: list[str] = []

    ref_code = _read_reference(problem)
    if ref_code:
        parts.append("## Reference Implementation\n")
        if baseline > 0:
            parts.append(
                f"Below is a known working implementation that synthesizes "
                f"to {baseline} cells. Your task is to produce a functionally "
                f"correct implementation with fewer cells.\n"
            )
        else:
            parts.append(
                "Below is a known working implementation. "
                "Your task is to produce a correct implementation "
                "that may improve upon it (e.g., fewer cells, simpler logic).\n"
            )
        parts.append("```verilog")
        parts.append(ref_code)
        parts.append("```\n")

    parts.append("## Design Specification\n")
    parts.append(problem.design_description)
    parts.append("")

    if baseline > 0:
        parts.append(
            f"Write a complete Verilog module that satisfies the specification above "
            f"with fewer than {baseline} cells when synthesized. Fewer cells is better."
        )
    else:
        parts.append("Write a complete Verilog module that satisfies the specification above.")

    parts.append("\nOutput only Verilog code within ```verilog fences.")

    return "\n".join(parts)


def build_messages(problem: RTLLMProblem, baseline: int) -> list[dict[str, str]]:
    """Build the full message list for a problem."""
    return [
        {"role": "system", "content": "You are an expert Verilog RTL designer."},
        {"role": "user", "content": build_initial_prompt(problem, baseline)},
    ]


def _load_existing_keys(raw_dir: str) -> set[tuple[str, str, int]]:
    """Load existing (problem_id, model_name, sample_idx) tuples for resumption."""
    keys: set[tuple[str, str, int]] = set()
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return keys
    for jsonl_file in raw_path.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    keys.add((record["problem_id"], record["model_name"], record["sample_idx"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return keys


async def _query_openrouter(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    proxy: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict | None:
    """Send a single request to OpenRouter with retry logic."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        async with semaphore:
            try:
                request_kwargs = {
                    "json": payload,
                    "headers": headers,
                    "timeout": aiohttp.ClientTimeout(total=300),
                }
                if proxy:
                    request_kwargs["proxy"] = proxy

                async with session.post(url, **request_kwargs) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data
                    elif resp.status == 429:
                        # Rate limited — back off
                        retry_after = float(resp.headers.get("Retry-After", retry_delay * (attempt + 1)))
                        print(f"  Rate limited, waiting {retry_after:.0f}s...")
                        await asyncio.sleep(retry_after)
                    else:
                        body = await resp.text()
                        print(f"  HTTP {resp.status}: {body[:200]}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                print(f"  Request error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

    return None


def _extract_response_content(response: dict) -> tuple[str, dict]:
    """Extract response content and normalize thinking tokens.

    Returns (full_content, usage_dict).
    """
    choices = response.get("choices", [])
    if not choices:
        return "", {}

    message = choices[0].get("message", {})
    content = message.get("content", "")

    # Some OpenRouter models return reasoning in a separate field
    reasoning = message.get("reasoning", "") or ""
    if reasoning:
        content = f"<thinking>\n{reasoning}\n</thinking>\n\n{content}"

    usage = response.get("usage", {})
    return content, usage


async def collect_for_problem_model(
    session: aiohttp.ClientSession,
    problem: RTLLMProblem,
    baseline: int,
    model_cfg: dict,
    config: dict,
    api_key: str,
    existing_keys: set[tuple[str, str, int]],
    semaphore: asyncio.Semaphore,
    output_file: Path,
    proxy: str | None = None,
) -> int:
    """Collect all samples for one (problem, model) pair."""
    messages = build_messages(problem, baseline)
    collection = config["collection"]
    collected = 0

    for sample_idx in range(collection["num_samples"]):
        key = (problem.problem_id, model_cfg["name"], sample_idx)
        if key in existing_keys:
            continue

        response = await _query_openrouter(
            session=session,
            base_url=collection["openrouter_base_url"],
            api_key=api_key,
            model_id=model_cfg["id"],
            messages=messages,
            max_tokens=model_cfg["max_tokens"],
            temperature=collection["temperature"],
            semaphore=semaphore,
            proxy=proxy,
            max_retries=collection["max_retries"],
            retry_delay=collection["retry_delay"],
        )

        if response is None:
            print(f"  FAILED: {problem.problem_id}/{model_cfg['name']}/{sample_idx}")
            continue

        content, usage = _extract_response_content(response)
        if not content:
            print(f"  EMPTY: {problem.problem_id}/{model_cfg['name']}/{sample_idx}")
            continue

        record = {
            "problem_id": problem.problem_id,
            "model_id": model_cfg["id"],
            "model_name": model_cfg["name"],
            "sample_idx": sample_idx,
            "messages_sent": messages,
            "response_content": content,
            "usage": usage,
            "timestamp": time.time(),
        }

        # Append to file atomically
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        collected += 1

    return collected


async def collect_all(
    config: dict,
    problems: dict[str, RTLLMProblem],
    baselines: dict[str, int],
    api_key: str,
    problem_filter: list[str] | None = None,
    model_filter: list[str] | None = None,
    proxy: str | None = None,
) -> None:
    """Collect responses from all models for all problems."""
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_keys = _load_existing_keys(str(raw_dir))
    if existing_keys:
        print(f"Resuming: {len(existing_keys)} existing records found")

    # Filter problems/models if specified
    target_problems = problems
    if problem_filter:
        target_problems = {k: v for k, v in problems.items() if k in problem_filter}

    target_models = config["models"]
    if model_filter:
        target_models = [m for m in target_models if m["name"] in model_filter]

    total_tasks = len(target_problems) * len(target_models)
    print(f"Collecting from {len(target_models)} models x {len(target_problems)} problems = {total_tasks} pairs")

    async with aiohttp.ClientSession(trust_env=True) as session:
        tasks = []
        for model_cfg in target_models:
            semaphore = asyncio.Semaphore(config["collection"]["concurrency"])
            output_file = raw_dir / f"{model_cfg['name']}.jsonl"

            for pid, problem in target_problems.items():
                baseline = baselines.get(pid, 0)
                tasks.append(
                    collect_for_problem_model(
                        session=session,
                        problem=problem,
                        baseline=baseline,
                        model_cfg=model_cfg,
                        config=config,
                        api_key=api_key,
                        existing_keys=existing_keys,
                        semaphore=semaphore,
                        output_file=output_file,
                        proxy=proxy,
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_collected = 0
        errors = 0
        for r in results:
            if isinstance(r, Exception):
                print(f"  Task error: {r}")
                errors += 1
            else:
                total_collected += r

    print(f"\nCollection complete: {total_collected} new responses, {errors} errors")


def main():
    parser = argparse.ArgumentParser(description="Collect Verilog solutions via OpenRouter")
    parser.add_argument("--config", default="recipe/ttt_rtl/sft/config.yaml", help="Config file path")
    parser.add_argument("--problems", type=str, default=None, help="Comma-separated problem IDs to collect")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to use")
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="HTTP/HTTPS proxy URL for OpenRouter requests. Falls back to OPENROUTER_PROXY, then collection.proxy, and also honors HTTP(S)_PROXY via trust_env.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    proxy = args.proxy or os.environ.get("OPENROUTER_PROXY") or config.get("collection", {}).get("proxy")

    # Load problems
    problem_filter = [p.strip() for p in args.problems.split(",")] if args.problems else None
    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    problems = load_rtllm_problems(
        rtllm_dir=config["rtllm_dir"],
        problems=problem_filter,
    )
    if not problems:
        raise RuntimeError(f"No problems found in {config['rtllm_dir']}")
    print(f"Loaded {len(problems)} problems: {list(problems.keys())}")

    # Compute PPA baselines
    print("Computing PPA baselines...")
    baselines = compute_ppa_baselines(problems)

    # Run collection
    asyncio.run(collect_all(
        config=config,
        problems=problems,
        baselines=baselines,
        api_key=api_key,
        problem_filter=problem_filter,
        model_filter=model_filter,
        proxy=proxy,
    ))


if __name__ == "__main__":
    main()

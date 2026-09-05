#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Run a graph-enabled Qwen generation, sleep, wake, and parity cycle."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="Stable virtual addresses are")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _generation(llm: Any, prompt: str, max_tokens: int) -> dict[str, Any]:
    from vllm import SamplingParams

    started = time.perf_counter()
    result = llm.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=max_tokens),
        use_tqdm=False,
    )[0].outputs[0]
    return {
        "seconds": time.perf_counter() - started,
        "token_ids": list(result.token_ids),
        "text": result.text,
    }


def main() -> int:
    args = _parser().parse_args()
    if not 0 < args.gpu_memory_utilization <= 0.2:
        raise ValueError("this validation caps --gpu-memory-utilization at 0.2")
    os.environ["COLDSNAP_HIBERNATE_STATE_PATH"] = str(args.state_json)

    from vllm import LLM

    construction_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        enable_sleep_mode=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_model_len=args.max_model_len,
        max_num_seqs=2,
        language_model_only=True,
        enforce_eager=False,
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 2],
        },
    )
    construction_s = time.perf_counter() - construction_started
    before = _generation(llm, args.prompt, args.max_tokens)

    sleep_started = time.perf_counter()
    llm.sleep(level=1)
    sleep_s = time.perf_counter() - sleep_started
    sleeping_state = json.loads(args.state_json.read_text())

    wake_started = time.perf_counter()
    llm.wake_up()
    wake_s = time.perf_counter() - wake_started
    running_state = json.loads(args.state_json.read_text())
    after = _generation(llm, args.prompt, args.max_tokens)

    exact_parity = (
        before["token_ids"] == after["token_ids"]
        and before["text"] == after["text"]
    )
    sleeping_graph = sleeping_state.get("cuda_graph", {})
    running_graph = running_state.get("cuda_graph", {})
    assertions = {
        "exact_generation_parity": exact_parity,
        "graph_enabled": sleeping_graph.get("enabled") is True,
        "graph_allocation_present": int(
            sleeping_graph.get("allocation_count", 0)
        )
        > 0,
        "graph_paused_while_sleeping": int(
            sleeping_graph.get("paused_count", 0)
        )
        > 0,
        "graph_resumed": int(running_graph.get("paused_count", -1)) == 0,
        "native_hydration_used": running_state.get("hydration_backend")
        in {"buffered", "direct", "gds"},
    }
    output = {
        "model": args.model,
        "construction_s": construction_s,
        "sleep_s": sleep_s,
        "wake_s": wake_s,
        "before": before,
        "after": after,
        "sleeping_state": sleeping_state,
        "running_state": running_state,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    _atomic_json(args.output_json, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

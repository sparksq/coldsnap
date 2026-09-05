#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Run a warmed vLLM engine behind a CRIU-safe file control channel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VLLM_PLUGIN_ROOT = ROOT / "integrations" / "vllm"
if str(VLLM_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_PLUGIN_ROOT))

from coldsnap_vllm_checkpoint import (  # noqa: E402
    checkpoint_prepare,
    checkpoint_restore,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-revision")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--multiprocess-engine", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.01)
    return parser


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"control record is not a JSON object: {path}")
    return value


def _shutdown(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if callable(shutdown):
        shutdown()


def _generate(llm: Any, prompt: str, output_tokens: int) -> dict[str, Any]:
    from vllm import SamplingParams

    started = time.perf_counter()
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=output_tokens),
        use_tqdm=False,
    )
    output = outputs[0].outputs[0]
    return {
        "seconds": time.perf_counter() - started,
        "token_ids": list(output.token_ids),
        "text": output.text,
    }


def _response_path(control_dir: Path, sequence: int) -> Path:
    return control_dir / "responses" / f"{sequence:08d}.json"


def _serve(args: argparse.Namespace, llm: Any, baseline: dict[str, Any]) -> None:
    command_path = args.control_dir / "command.json"
    fatal_path = args.control_dir / "fatal.json"
    last_sequence = 0
    while True:
        try:
            command = _load_json(command_path)
        except FileNotFoundError:
            time.sleep(args.poll_seconds)
            continue
        sequence = int(command.get("sequence", -1))
        if sequence <= last_sequence:
            time.sleep(args.poll_seconds)
            continue
        if command.get("generation") != args.generation:
            raise RuntimeError("control command has the wrong snapshot generation")
        operation = command.get("operation")
        started = time.perf_counter()
        response: dict[str, Any] = {
            "format": 1,
            "generation": args.generation,
            "sequence": sequence,
            "operation": operation,
            "pid": os.getpid(),
        }
        try:
            if operation == "prepare":
                call = checkpoint_prepare(llm)
                response.update(
                    {
                        "status": "prepared",
                        "dispatch": call.dispatch,
                        "response": repr(call.response),
                    }
                )
            elif operation == "restore":
                call = checkpoint_restore(llm)
                response.update(
                    {
                        "status": "restored",
                        "dispatch": call.dispatch,
                        "response": repr(call.response),
                    }
                )
            elif operation == "generate":
                generated = _generate(llm, args.prompt, args.output_tokens)
                response.update(
                    {
                        "status": "generated",
                        "generation_result": generated,
                        "parity": (
                            generated["token_ids"] == baseline["token_ids"]
                            and generated["text"] == baseline["text"]
                        ),
                    }
                )
            elif operation == "shutdown":
                response["status"] = "shutting-down"
            else:
                raise ValueError(f"unsupported control operation: {operation!r}")
            response["seconds"] = time.perf_counter() - started
            _atomic_json(_response_path(args.control_dir, sequence), response)
            last_sequence = sequence
            if operation == "shutdown":
                return
        except BaseException as error:
            response.update(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "seconds": time.perf_counter() - started,
                }
            )
            _atomic_json(_response_path(args.control_dir, sequence), response)
            _atomic_json(fatal_path, response)
            raise


def main() -> int:
    args = _parser().parse_args()
    if not 0 < args.gpu_memory_utilization <= 0.2:
        raise ValueError("gpu-memory-utilization must be in (0, 0.2]")
    if args.poll_seconds <= 0 or args.poll_seconds > 1:
        raise ValueError("poll-seconds must be in (0, 1]")
    args.control_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = (
        "1" if args.multiprocess_engine else "0"
    )
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = None
    try:
        import torch
        from vllm import LLM

        compilation_config = None
        if args.graphs:
            compilation_config = {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": [1, 2],
            }
        construction_started = time.perf_counter()
        llm = LLM(
            model=args.model,
            revision=args.model_revision,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enforce_eager=not args.graphs,
            compilation_config=compilation_config,
            language_model_only=True,
        )
        construction_seconds = time.perf_counter() - construction_started
        baseline = _generate(llm, args.prompt, args.output_tokens)
        torch.cuda.synchronize()
        ready = {
            "format": 1,
            "kind": "vllm-cuda-snapshot-ready",
            "generation": args.generation,
            "pid": os.getpid(),
            "engine_construction_seconds": construction_seconds,
            "baseline": baseline,
            "graphs": args.graphs,
            "multiprocess_engine": args.multiprocess_engine,
            "cuda": {
                "device": torch.cuda.current_device(),
                "name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
            },
        }
        _atomic_json(args.control_dir / "ready.json", ready)
        _serve(args, llm, baseline)
        return 0
    except BaseException as error:
        _atomic_json(
            args.control_dir / "fatal.json",
            {
                "format": 1,
                "generation": args.generation,
                "pid": os.getpid(),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        if llm is not None:
            _shutdown(llm)


if __name__ == "__main__":
    raise SystemExit(main())

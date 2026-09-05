#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Exercise vLLM checkpoint hooks around a CUDA process round-trip."""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing
import os
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


RUNTIME_SHARED = Path(__file__).resolve().parents[2] / "runtime" / "shared"
if str(RUNTIME_SHARED) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SHARED))

from checkpointctl import CUDA  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--multiprocess-engine", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--skip-native-roundtrip", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _generation(llm: Any, prompt: str, output_tokens: int) -> dict[str, Any]:
    from vllm import SamplingParams

    started = time.perf_counter()
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=output_tokens),
        use_tqdm=False,
    )
    return {
        "seconds": time.perf_counter() - started,
        "token_ids": list(outputs[0].outputs[0].token_ids),
        "text": outputs[0].outputs[0].text,
    }


def _target(connection: Connection, config: dict[str, Any]) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = (
        "1" if config["multiprocess_engine"] else "0"
    )
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = None
    try:
        from coldsnap_vllm_checkpoint import checkpoint_prepare, checkpoint_restore
        from vllm import LLM

        construction_started = time.perf_counter()
        llm = LLM(
            model=config["model"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            kv_cache_memory_bytes=config["kv_cache_memory_bytes"],
            max_model_len=config["max_model_len"],
            max_num_seqs=2,
            enforce_eager=not config["graphs"],
            language_model_only=True,
        )
        construction_s = time.perf_counter() - construction_started
        baseline = _generation(llm, config["prompt"], config["output_tokens"])
        connection.send(
            {
                "status": "ready",
                "pid": os.getpid(),
                "engine_construction_s": construction_s,
                "baseline": baseline,
            }
        )
        while True:
            command = connection.recv()
            if command == "prepare":
                started = time.perf_counter()
                call = checkpoint_prepare(llm)
                connection.send(
                    {
                        "status": "prepared",
                        "seconds": time.perf_counter() - started,
                        "dispatch": call.dispatch,
                    }
                )
            elif command == "restore":
                started = time.perf_counter()
                call = checkpoint_restore(llm)
                connection.send(
                    {
                        "status": "restored",
                        "seconds": time.perf_counter() - started,
                        "dispatch": call.dispatch,
                    }
                )
            elif command == "generate":
                connection.send(
                    {
                        "status": "generated",
                        **_generation(
                            llm, config["prompt"], config["output_tokens"]
                        ),
                    }
                )
            elif command == "shutdown":
                break
            else:
                raise ValueError(f"unknown target command: {command}")
    except BaseException as error:
        connection.send(
            {"status": "error", "error": f"{type(error).__name__}: {error}"}
        )
    finally:
        if llm is not None:
            engine = getattr(llm, "llm_engine", None)
            core = getattr(engine, "engine_core", None)
            shutdown = getattr(core, "shutdown", None)
            if callable(shutdown):
                shutdown()
        connection.close()


def _message(connection: Connection, timeout: float) -> dict[str, Any]:
    if not connection.poll(timeout):
        raise TimeoutError("vLLM checkpoint target did not respond")
    value = connection.recv()
    if value.get("status") == "error":
        raise RuntimeError(value["error"])
    return value


def _native_roundtrip(cuda: CUDA, pid: int) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    checkpointed = False
    locked = False
    try:
        state = cuda.state(pid)
        operations.append(state)
        if state.get("state") != "running":
            raise RuntimeError(f"target is not running: {state}")

        lock = cuda.operation("cuCheckpointProcessLock", pid, None)
        operations.append(lock)
        if not lock["success"]:
            return {"supported": False, "operations": operations}
        locked = True
        operations.append(cuda.state(pid))

        checkpoint = cuda.operation("cuCheckpointProcessCheckpoint", pid, None)
        operations.append(checkpoint)
        if not checkpoint["success"]:
            return {"supported": False, "operations": operations}
        checkpointed = True
        locked = False
        operations.append(cuda.state(pid))

        restore = cuda.operation("cuCheckpointProcessRestore", pid, None)
        operations.append(restore)
        if not restore["success"]:
            return {"supported": False, "operations": operations}
        checkpointed = False
        locked = True
        operations.append(cuda.state(pid))

        unlock = cuda.operation("cuCheckpointProcessUnlock", pid, None)
        operations.append(unlock)
        if not unlock["success"]:
            return {"supported": False, "operations": operations}
        locked = False
        operations.append(cuda.state(pid))
        return {
            "supported": all(value["success"] for value in operations),
            "operations": operations,
        }
    finally:
        if checkpointed:
            operations.append(
                cuda.operation("cuCheckpointProcessRestore", pid, None)
            )
            checkpointed = False
            locked = True
        if locked:
            operations.append(cuda.operation("cuCheckpointProcessUnlock", pid, None))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    if not 0 < args.gpu_memory_utilization <= 0.2:
        raise ValueError("gpu-memory-utilization must be in (0, 0.2]")

    config = {
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "max_model_len": args.max_model_len,
        "prompt": args.prompt,
        "output_tokens": args.output_tokens,
        "multiprocess_engine": args.multiprocess_engine,
        "graphs": args.graphs,
    }
    if args.multiprocess_engine and not args.skip_native_roundtrip:
        raise ValueError(
            "native roundtrip currently targets the direct CUDA owner; use "
            "--skip-native-roundtrip for the multiprocess hook prototype"
        )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_target, args=(child, config))
    process.start()
    child.close()
    report: dict[str, Any] = {
        "format": 1,
        "kind": "vllm-cuda-process-checkpoint-feasibility",
        "configuration": config,
        "target_pid": process.pid,
    }
    prepared = False
    try:
        report["target"] = _message(parent, args.timeout)
        parent.send("prepare")
        report["prepare"] = _message(parent, args.timeout)
        prepared = True

        if args.skip_native_roundtrip:
            report["native"] = {"skipped": True, "supported": None}
        else:
            cuda = CUDA()
            init = cuda.operation("cuInit", 0)
            version_value = ctypes.c_int()
            version = cuda.operation(
                "cuDriverGetVersion", ctypes.byref(version_value)
            )
            if version["success"]:
                version["driver_api_version"] = version_value.value
            report["cuda"] = {"initialization": init, "driver_api": version}
            report["native"] = _native_roundtrip(cuda, process.pid)

        parent.send("restore")
        report["restore"] = _message(parent, args.timeout)
        prepared = False
        parent.send("generate")
        report["after_restore"] = _message(parent, args.timeout)
        baseline = report["target"]["baseline"]
        after = report["after_restore"]
        report["generation_parity"] = (
            baseline["token_ids"] == after["token_ids"]
            and baseline["text"] == after["text"]
        )
        report["hook_cycle_supported"] = bool(report["generation_parity"])
        report["full_process_checkpoint_feasible_on_host"] = bool(
            report["hook_cycle_supported"] and report["native"].get("supported")
        )
    finally:
        if process.is_alive():
            if prepared:
                try:
                    parent.send("restore")
                    _message(parent, min(args.timeout, 30))
                except (BrokenPipeError, EOFError, TimeoutError):
                    pass
            try:
                parent.send("shutdown")
            except (BrokenPipeError, EOFError):
                pass
            process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(15)
        report["target_exit_status"] = process.exitcode

    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("hook_cycle_supported") else 1


if __name__ == "__main__":
    raise SystemExit(main())

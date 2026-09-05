#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Measure fresh-process vLLM startup variants and first-response latency."""

# Kept as a standalone harness so subprocess cases can execute this file directly.

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROCESS_STARTED = time.perf_counter()
CASES = {
    "inproc-full",
    "inproc-text",
    "multiproc-full",
    "multiproc-text",
    "serve-full",
    "serve-text",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "inproc-full",
            "inproc-text",
            "multiproc-full",
            "multiproc-text",
        ],
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/vllm-startup"))
    parser.add_argument("--worker-case", choices=sorted(CASES), help=argparse.SUPPRESS)
    parser.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--server-log", type=Path, help=argparse.SUPPRESS)
    return parser


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _offline_worker(args: argparse.Namespace) -> dict[str, Any]:
    case = args.worker_case
    assert case is not None
    in_process = case.startswith("inproc-")
    text_only = case.endswith("-text")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" if in_process else "1"

    import_started = time.perf_counter()
    from vllm import LLM, SamplingParams

    import_s = time.perf_counter() - import_started
    construction_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        language_model_only=text_only,
        load_format=args.load_format,
    )
    construction_s = time.perf_counter() - construction_started
    request_started = time.perf_counter()
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(
            temperature=0,
            max_tokens=args.output_tokens,
        ),
        use_tqdm=False,
    )
    first_response_s = time.perf_counter() - request_started
    token_ids = list(outputs[0].outputs[0].token_ids)
    text = outputs[0].outputs[0].text
    return {
        "case": case,
        "topology": "in_process" if in_process else "multiprocess_end_to_end",
        "language_model_only": text_only,
        "process_to_import_start_s": import_started - PROCESS_STARTED,
        "import_s": import_s,
        "engine_construction_s": construction_s,
        "first_complete_response_s": first_response_s,
        "process_to_first_complete_response_s": time.perf_counter()
        - PROCESS_STARTED,
        "output_tokens": len(token_ids),
        "token_ids": token_ids,
        "text": text,
        "ttft_available": False,
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _health_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=1
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _stream_completion(
    port: int, model: str, prompt: str, output_tokens: int
) -> tuple[float, float, str]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_s: float | None = None
    pieces: list[str] = []
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            text = str(event["choices"][0].get("text", ""))
            if text and first_token_s is None:
                first_token_s = time.perf_counter() - started
            pieces.append(text)
    total_s = time.perf_counter() - started
    if first_token_s is None:
        raise RuntimeError("stream completed without a token-bearing event")
    return first_token_s, total_s, "".join(pieces)


def _serve_worker(args: argparse.Namespace) -> dict[str, Any]:
    case = args.worker_case
    assert case is not None and case.startswith("serve-")
    if args.server_log is None:
        raise ValueError("serve worker requires --server-log")
    text_only = case.endswith("-text")
    port = _free_port()
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--load-format",
        args.load_format,
    ]
    if text_only:
        command.append("--language-model-only")
    if args.enforce_eager:
        command.append("--enforce-eager")
    environment = os.environ.copy()
    environment["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    launched = time.perf_counter()
    with args.server_log.open("wb") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = launched + args.timeout
            while not _health_ready(port):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server exited with status {process.returncode}"
                    )
                if time.perf_counter() >= deadline:
                    raise TimeoutError("vLLM server did not become healthy")
                time.sleep(0.1)
            health_s = time.perf_counter() - launched
            ttft_s, response_s, text = _stream_completion(
                port, args.model, args.prompt, args.output_tokens
            )
            launch_to_first_token_s = time.perf_counter() - launched - (
                response_s - ttft_s
            )
            return {
                "case": case,
                "topology": "serve_end_to_end",
                "language_model_only": text_only,
                "launch_to_health_s": health_s,
                "request_ttft_s": ttft_s,
                "first_complete_response_s": response_s,
                "launch_to_first_token_s": launch_to_first_token_s,
                "text": text,
                "ttft_available": True,
                "server_log": str(args.server_log),
            }
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)


def _worker(args: argparse.Namespace) -> int:
    if args.result_json is None:
        raise ValueError("worker requires --result-json")
    try:
        result = (
            _serve_worker(args)
            if args.worker_case.startswith("serve-")
            else _offline_worker(args)
        )
    except Exception as error:
        result = {
            "case": args.worker_case,
            "error": f"{type(error).__name__}: {error}",
        }
        _write_json(args.result_json, result)
        raise
    _write_json(args.result_json, result)
    return 0


def _case_command(
    args: argparse.Namespace,
    case: str,
    result_path: Path,
    server_log_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        case,
        "--result-json",
        str(result_path),
        "--server-log",
        str(server_log_path),
        "--model",
        args.model,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--output-tokens",
        str(args.output_tokens),
        "--prompt",
        args.prompt,
        "--load-format",
        args.load_format,
        "--timeout",
        str(args.timeout),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    return command


def _matrix(args: argparse.Namespace) -> int:
    unknown = set(args.cases) - CASES
    if unknown:
        raise ValueError(f"unknown startup cases: {sorted(unknown)}")
    if not 0 < args.gpu_memory_utilization <= 0.2:
        raise ValueError("this calibration caps --gpu-memory-utilization at 0.2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in args.cases:
        result_path = args.output_dir / f"{case}.json"
        log_path = args.output_dir / f"{case}.log"
        server_log_path = args.output_dir / f"{case}.server.log"
        started = time.perf_counter()
        with log_path.open("wb") as log:
            completed = subprocess.run(
                _case_command(args, case, result_path, server_log_path),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout + 30,
                check=False,
            )
        if result_path.is_file():
            result = json.loads(result_path.read_text())
        else:
            result = {"case": case, "error": "worker produced no result"}
        result["fresh_process_wall_s"] = time.perf_counter() - started
        result["worker_exit_status"] = completed.returncode
        result["worker_log"] = str(log_path)
        results.append(result)
    matrix = {
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "max_model_len": args.max_model_len,
        "load_format": args.load_format,
        "enforce_eager": args.enforce_eager,
        "cache_environment": {
            name: os.environ.get(name)
            for name in (
                "VLLM_CACHE_ROOT",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "CUDA_CACHE_PATH",
                "CUDA_CACHE_DISABLE",
                "CUDA_CACHE_MAXSIZE",
            )
        },
        "results": results,
    }
    output = args.output_dir / "startup-matrix.json"
    _write_json(output, matrix)
    print(json.dumps(matrix, indent=2, sort_keys=True))
    return 0 if all("error" not in result for result in results) else 1


def main() -> int:
    args = _parser().parse_args()
    if args.worker_case is not None:
        return _worker(args)
    return _matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())

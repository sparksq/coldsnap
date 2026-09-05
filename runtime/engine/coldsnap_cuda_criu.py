#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared CUDA/CRIU primitives and the standalone vLLM qualification runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


PROCESS_STARTED_NS = time.monotonic_ns()
DEFAULT_TARGET = Path(__file__).with_name("vllm_cuda_snapshot_target.py")
DEFAULT_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CONTROLLER_ABI = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "restore", "roundtrip"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--criu", type=Path, default=Path("/opt/criu/criu/criu"))
    parser.add_argument(
        "--criu-rpc",
        type=Path,
        default=Path("/usr/local/bin/coldsnap-criu-rpc"),
    )
    parser.add_argument(
        "--cuda-checkpoint", type=Path, default=Path("/usr/local/bin/cuda-checkpoint")
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--multiprocess-engine", action="store_true")
    parser.add_argument(
        "--cuda-job-mode",
        action="store_true",
        help="bind all spawned CUDA processes to one 610+ checkpoint job",
    )
    parser.add_argument("--minimum-target-pid", type=int, default=512)
    # CRIU must inspect the target's POSIX shared-memory namespace here; this
    # is not general-purpose temporary-file storage.
    parser.add_argument(
        "--shared-memory-dir",
        type=Path,
        default=Path("/dev/shm"),  # nosec B108
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--compact-output", action="store_true")
    return parser


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"record is not a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        while chunk := source.read(16 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _gpu_facts() -> dict[str, str]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,name,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    first = output.splitlines()[0]
    fields = [field.strip() for field in first.split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"unexpected nvidia-smi identity: {first!r}")
    return {"driver": fields[0], "name": fields[1], "compute_capability": fields[2]}


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    hook = (
        Path(__file__).resolve().parents[2]
        / "integrations"
        / "vllm"
        / "coldsnap_vllm_checkpoint.py"
    )
    paths = {
        "target": args.target.resolve(),
        "checkpoint_hook": hook.resolve(),
        "criu": args.criu.resolve(),
        "criu_rpc": args.criu_rpc.resolve(),
        "cuda_checkpoint": args.cuda_checkpoint.resolve(),
    }
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("snapshot runtime is incomplete: " + ", ".join(missing))
    return {
        "format": 1,
        "kind": "vllm-cuda-criu-identity",
        "controller_abi": CONTROLLER_ABI,
        "image_id": args.image_id,
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "gpu": _gpu_facts(),
        "runtime_sha256": {name: _sha256(path) for name, path in paths.items()},
        "configuration": {
            "model": args.model,
            "model_revision": args.model_revision,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "prompt": args.prompt,
            "output_tokens": args.output_tokens,
            "graphs": args.graphs,
            "multiprocess_engine": args.multiprocess_engine,
            "cuda_job_mode": args.cuda_job_mode,
        },
    }


def _process_tree(root_pid: int) -> list[int]:
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                values = {}
                for line in status.read_text(encoding="utf-8").splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        values[key] = value.strip()
                pid = int(status.parent.name)
                parent = int(values["PPid"])
            except (FileNotFoundError, KeyError, PermissionError, ValueError):
                continue
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sorted(descendants)


def _rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0


def _advance_pid_allocator(minimum_pid: int) -> int:
    if minimum_pid < 2:
        raise ValueError("minimum-target-pid must be at least 2")
    allocated = 1
    while allocated < minimum_pid:
        process = subprocess.Popen(
            ["/bin/true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        allocated = process.pid
        if process.wait() != 0:
            raise RuntimeError("failed to advance the PID allocator")
    return allocated


def _criu_command(args: argparse.Namespace, action: str) -> list[str]:
    """Build the direct CRIU command shared by the NCCL qualification harness."""
    return [
        str(args.criu),
        action,
        "--images-dir",
        str(args.artifact_root / "images"),
        "--work-dir",
        str(args.artifact_root / "work"),
        "--libdir",
        str(args.plugin_dir),
        "--shell-job",
        "--file-locks",
        "--link-remap",
        "--manage-cgroups=ignore",
        "-v4",
    ]


def _cgroup_io_stat(path: Path = Path("/sys/fs/cgroup/io.stat")) -> dict[str, int] | None:
    """Sum cgroup-v2 block-I/O counters across devices when available."""
    try:
        totals: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            for field in line.split()[1:]:
                name, separator, value = field.partition("=")
                if separator and value.isdecimal():
                    totals[name] = totals.get(name, 0) + int(value)
        return totals
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _criu_rpc_command(
    args: argparse.Namespace,
    action: str,
    log_file: str,
) -> list[str]:
    if action not in {"dump", "restore"}:
        raise ValueError(f"unsupported CRIU RPC action: {action}")
    command = [
        str(args.criu_rpc),
        "--action",
        action,
        "--criu",
        str(args.criu),
        "--cuda-checkpoint",
        str(args.cuda_checkpoint),
        "--images-dir",
        str(args.artifact_root / "images"),
        "--work-dir",
        str(args.artifact_root / "work"),
        "--log-file",
        log_file,
        "--external-files",
        str(args.artifact_root / "external-files.json"),
        "--result",
        str(args.artifact_root / f"{action}-rpc.json"),
        "--timeout",
        str(max(1, int(args.timeout))),
    ]
    if args.multiprocess_engine:
        command.extend(
            [
                "--cuda-process-tree",
                "--tcp-established",
                "--ghost-limit",
                str(64 * 1024**2),
            ]
        )
        if action == "dump":
            command.extend(["--network-lock", "nftables"])
    return command


def _criu_rpc_result(args: argparse.Namespace, action: str) -> dict[str, Any]:
    result = _load_json(args.artifact_root / f"{action}-rpc.json")
    if (
        result.get("format") != 1
        or result.get("kind") != "coldsnap-criu-rpc-result"
        or result.get("action") != action
    ):
        raise RuntimeError(f"invalid CRIU RPC result: {result!r}")
    count = result.get("external_file_count")
    if not isinstance(count, int) or count < 0:
        raise RuntimeError(f"invalid CRIU RPC external-file count: {result!r}")
    return result


def _run_criu_profiled(command: list[str], environment: dict[str, str]) -> dict[str, Any]:
    """Run CRIU and report wall, child CPU/fault, and cgroup I/O deltas."""
    io_before = _cgroup_io_stat()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    subprocess.run(command, check=True, env=environment)
    wall_seconds = time.perf_counter() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    io_after = _cgroup_io_stat()
    io_delta = None
    if io_before is not None and io_after is not None:
        io_delta = {
            name: io_after.get(name, 0) - io_before.get(name, 0)
            for name in sorted(io_before.keys() | io_after.keys())
        }
    return {
        "wall_seconds": wall_seconds,
        "children_user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime,
        "children_system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime,
        "children_major_faults": usage_after.ru_majflt - usage_before.ru_majflt,
        "children_minor_faults": usage_after.ru_minflt - usage_before.ru_minflt,
        "children_block_inputs": usage_after.ru_inblock - usage_before.ru_inblock,
        "children_block_outputs": usage_after.ru_oublock - usage_before.ru_oublock,
        "cgroup_v2_io_before": io_before,
        "cgroup_v2_io_after": io_after,
        "cgroup_v2_io_delta": io_delta,
        "cgroup_io_scope": "entire controller container during CRIU subprocess",
    }


def _run_criu(command: list[str], environment: dict[str, str]) -> float:
    return float(_run_criu_profiled(command, environment)["wall_seconds"])


def _target_command(args: argparse.Namespace, generation: str) -> list[str]:
    command = [
        sys.executable,
        str(args.target),
        "--control-dir",
        str(args.artifact_root / "control"),
        "--generation",
        generation,
        "--model",
        args.model,
        "--model-revision",
        args.model_revision,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--prompt",
        args.prompt,
        "--output-tokens",
        str(args.output_tokens),
    ]
    if args.graphs:
        command.append("--graphs")
    if args.multiprocess_engine:
        command.append("--multiprocess-engine")
    return command


def _wait_json(
    path: Path,
    timeout: float,
    *,
    fatal_path: Path | None = None,
    process: subprocess.Popen[bytes] | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        if path.is_file():
            return _load_json(path)
        if fatal_path is not None and fatal_path.is_file():
            raise RuntimeError(f"snapshot target failed: {_load_json(fatal_path)}")
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"snapshot target exited early: {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.01)


def _send_command(
    args: argparse.Namespace,
    generation: str,
    sequence: int,
    operation: str,
) -> dict[str, Any]:
    control = args.artifact_root / "control"
    response_path = control / "responses" / f"{sequence:08d}.json"
    response_path.unlink(missing_ok=True)
    _atomic_json(
        control / "command.json",
        {
            "format": 1,
            "generation": generation,
            "sequence": sequence,
            "operation": operation,
        },
    )
    response = _wait_json(
        response_path,
        args.timeout,
        fatal_path=control / "fatal.json",
    )
    if response.get("status") == "error":
        raise RuntimeError(f"target command failed: {response}")
    return response


def _create_cuda_job(args: argparse.Namespace) -> Path:
    job_file = args.artifact_root / "cuda-checkpoint.job"
    if job_file.exists():
        raise FileExistsError(f"refusing to reuse a CUDA checkpoint job: {job_file}")
    script = 'set -eu; cp "$CUDA_CHECKPOINT_JOB_FILE" "$1"'
    subprocess.run(
        [
            str(args.cuda_checkpoint),
            "--launch-job",
            "sh",
            "-c",
            script,
            "coldsnap-cuda-job",
            str(job_file),
        ],
        check=True,
    )
    if not job_file.is_file() or job_file.stat().st_size == 0:
        raise RuntimeError("cuda-checkpoint did not create a reusable job record")
    return job_file


def _preserve_cuda_job(args: argparse.Namespace, job_file: Path) -> dict[str, Any]:
    template = args.artifact_root / "cuda-checkpoint.job.template"
    if template.exists():
        raise FileExistsError(f"CUDA checkpoint job template already exists: {template}")
    shutil.copy2(job_file, template)
    return {
        "working_sha256": _sha256(job_file),
        "template_sha256": _sha256(template),
        "working_inode": job_file.stat().st_ino,
    }


def _reset_cuda_job(args: argparse.Namespace, record: dict[str, Any]) -> Path:
    job_file = args.artifact_root / "cuda-checkpoint.job"
    template = args.artifact_root / "cuda-checkpoint.job.template"
    if _sha256(template) != record.get("template_sha256"):
        raise RuntimeError("immutable CUDA checkpoint job template changed")
    with template.open("rb", buffering=0) as source, job_file.open(
        "r+b", buffering=0
    ) as destination:
        destination.seek(0)
        shutil.copyfileobj(source, destination)
        destination.truncate()
        destination.flush()
        os.fsync(destination.fileno())
    if _sha256(job_file) != record.get("working_sha256"):
        raise RuntimeError("CUDA checkpoint job reset did not reproduce capture state")
    return job_file


def _snapshot_link_remaps(args: argparse.Namespace) -> tuple[int, int]:
    snapshot = args.artifact_root / "shm-image"
    snapshot.mkdir(parents=True, exist_ok=False)
    count = 0
    size = 0
    for source in sorted(args.shared_memory_dir.glob("link_remap.*")):
        if source.is_file():
            destination = snapshot / source.name
            shutil.copy2(source, destination)
            count += 1
            size += destination.stat().st_size
    return count, size


def _prepare_link_remaps(args: argparse.Namespace) -> int:
    snapshot = args.artifact_root / "shm-image"
    if not snapshot.is_dir():
        raise FileNotFoundError(f"shared-memory image is missing: {snapshot}")
    count = 0
    for source in sorted(snapshot.glob("link_remap.*")):
        if source.is_file():
            destination = args.shared_memory_dir / source.name
            destination.unlink(missing_ok=True)
            shutil.copy2(source, destination)
            count += 1
    return count


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def capture(args: argparse.Namespace) -> dict[str, Any]:
    identity = _identity(args)
    images = args.artifact_root / "images"
    work = args.artifact_root / "work"
    control = args.artifact_root / "control"
    if any(
        path.exists()
        for path in (
            images / "inventory.img",
            args.artifact_root / "capture.json",
            args.artifact_root / "external-files.json",
            args.artifact_root / "dump-rpc.json",
        )
    ):
        raise FileExistsError("refusing to overwrite a CUDA process artifact")
    for path in (images, work, control / "responses"):
        path.mkdir(parents=True, exist_ok=True)
    generation = f"qwen-cuda-criu-{uuid.uuid4().hex}"
    job_file = _create_cuda_job(args) if args.cuda_job_mode else None
    environment = os.environ.copy()
    if job_file is not None:
        environment["CUDA_CHECKPOINT_JOB_FILE"] = str(job_file)
    else:
        environment.pop("CUDA_CHECKPOINT_JOB_FILE", None)
    pid_floor = _advance_pid_allocator(args.minimum_target_pid)
    log_path = args.artifact_root / "target.log"
    started = time.perf_counter()
    with log_path.open("ab", buffering=0) as log:
        target = subprocess.Popen(
            _target_command(args, generation),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            ready = _wait_json(
                control / "ready.json",
                args.timeout,
                fatal_path=control / "fatal.json",
                process=target,
            )
            prepared = _send_command(args, generation, 1, "prepare")
            tree = _process_tree(target.pid)
            rss_bytes = sum(_rss_bytes(pid) for pid in tree)
            command = _criu_rpc_command(args, "dump", "dump.log")
            command.extend(["--pid", str(target.pid)])
            dump_seconds = _run_criu(command, environment)
            criu_rpc = _criu_rpc_result(args, "dump")
            target.wait(timeout=30)
            shm_count, shm_bytes = _snapshot_link_remaps(args)
            cuda_job = (
                _preserve_cuda_job(args, job_file) if job_file is not None else None
            )
        except BaseException:
            if target.poll() is None:
                _kill_process_group(target.pid)
                target.wait(timeout=30)
            raise
    image_bytes = sum(path.stat().st_size for path in images.rglob("*") if path.is_file())
    report = {
        "format": 1,
        "kind": "vllm-cuda-criu-capture",
        "generation": generation,
        "identity": identity,
        "target_pid": target.pid,
        "pid_floor": pid_floor,
        "process_tree": tree,
        "process_tree_rss_bytes": rss_bytes,
        "ready": ready,
        "prepare": prepared,
        "criu_rpc": criu_rpc,
        "cuda_job": cuda_job,
        "dump_seconds": dump_seconds,
        "capture_controller_seconds": time.perf_counter() - started,
        "image_bytes": image_bytes,
        "shared_memory_link_count": shm_count,
        "shared_memory_link_bytes": shm_bytes,
    }
    _atomic_json(args.artifact_root / "capture.json", report)
    return report


def _wait_process_exit(pid: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
            if waited == pid:
                return
        except ChildProcessError:
            if not Path(f"/proc/{pid}").exists():
                return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"restored process did not exit: {pid}")
        time.sleep(0.01)


def restore(args: argparse.Namespace) -> dict[str, Any]:
    capture_report = _load_json(args.artifact_root / "capture.json")
    current_identity = _identity(args)
    saved_identity = capture_report.get("identity")
    if saved_identity != current_identity:
        raise RuntimeError(
            "CUDA process artifact identity mismatch: "
            f"saved={capture_report.get('identity')!r} current={current_identity!r}"
        )
    generation = str(capture_report["generation"])
    cuda_job = capture_report.get("cuda_job")
    if args.cuda_job_mode:
        if not isinstance(cuda_job, dict):
            raise RuntimeError("CUDA checkpoint job template is missing")
        job_file = _reset_cuda_job(args, cuda_job)
    else:
        if cuda_job is not None:
            raise RuntimeError("artifact unexpectedly requires CUDA checkpoint job mode")
        job_file = None
    control = args.artifact_root / "control"
    (control / "command.json").unlink(missing_ok=True)
    (control / "fatal.json").unlink(missing_ok=True)
    for sequence in (2, 3, 4):
        (control / "responses" / f"{sequence:08d}.json").unlink(missing_ok=True)
    (args.artifact_root / "restore-rpc.json").unlink(missing_ok=True)
    shm_count = _prepare_link_remaps(args)
    environment = os.environ.copy()
    if job_file is not None:
        environment["CUDA_CHECKPOINT_JOB_FILE"] = str(job_file)
    else:
        environment.pop("CUDA_CHECKPOINT_JOB_FILE", None)
    command = _criu_rpc_command(args, "restore", "restore.log")
    restore_started_ns = time.monotonic_ns()
    restored_pid: int | None = None
    try:
        criu_seconds = _run_criu(command, environment)
        criu_rpc = _criu_rpc_result(args, "restore")
        restored_pid_value = criu_rpc.get("restored_pid")
        if not isinstance(restored_pid_value, int) or restored_pid_value <= 0:
            raise RuntimeError(f"invalid restored PID in CRIU RPC result: {criu_rpc!r}")
        restored_pid = restored_pid_value
        hook = _send_command(args, generation, 2, "restore")
        generated = _send_command(args, generation, 3, "generate")
        response_ns = time.monotonic_ns()
        if generated.get("parity") is not True:
            raise RuntimeError(f"restored generation differs from capture: {generated}")
        shutdown = _send_command(args, generation, 4, "shutdown")
        forced_shutdown = False
        try:
            _wait_process_exit(restored_pid, 2)
        except TimeoutError:
            # Engine teardown is outside the activation measurement and some
            # CUDA/PyTorch combinations block there. The production process
            # would remain serving; bound benchmark cleanup instead.
            forced_shutdown = True
            _kill_process_group(restored_pid)
            _wait_process_exit(restored_pid, 5)
    except BaseException:
        if restored_pid is not None:
            _kill_process_group(restored_pid)
        raise
    report = {
        "format": 1,
        "kind": "vllm-cuda-criu-restore",
        "generation": generation,
        "identity": current_identity,
        "restored_pid": restored_pid,
        "criu_rpc": criu_rpc,
        "criu_restore_seconds": criu_seconds,
        "checkpoint_restore_hook": hook,
        "generation_result": generated,
        "shutdown": shutdown,
        "forced_shutdown": forced_shutdown,
        "shared_memory_link_count": shm_count,
        "restore_start_to_response_seconds": (response_ns - restore_started_ns) / 1e9,
        "controller_start_to_response_seconds": (response_ns - PROCESS_STARTED_NS) / 1e9,
    }
    _atomic_json(args.artifact_root / "restore-result.json", report)
    return report


def main() -> int:
    args = _parser().parse_args()
    if not 0 < args.gpu_memory_utilization <= 0.2:
        raise ValueError("gpu-memory-utilization must be in (0, 0.2]")
    if args.timeout <= 0 or args.timeout > 3600:
        raise ValueError("timeout must be in (0, 3600]")
    if args.multiprocess_engine and not args.cuda_job_mode:
        raise ValueError("multiprocess-engine requires CUDA checkpoint job mode")
    try:
        if args.mode == "capture":
            report = capture(args)
        elif args.mode == "restore":
            report = restore(args)
        else:
            report = {"capture": capture(args), "restore": restore(args)}
    except BaseException as error:
        report = {
            "format": 1,
            "kind": "vllm-cuda-criu-error",
            "error": f"{type(error).__name__}: {error}",
        }
        args.artifact_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.artifact_root / "controller-error.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1
    console = report
    if args.compact_output and report.get("kind") == "vllm-cuda-criu-restore":
        console = {
            "kind": report["kind"],
            "criu_restore_seconds": report["criu_restore_seconds"],
            "restore_hook_seconds": report["checkpoint_restore_hook"]["seconds"],
            "generation_seconds": report["generation_result"]["seconds"],
            "restore_start_to_response_seconds": report[
                "restore_start_to_response_seconds"
            ],
            "token_ids": report["generation_result"]["generation_result"][
                "token_ids"
            ],
            "text": report["generation_result"]["generation_result"]["text"],
            "parity": report["generation_result"]["parity"],
        }
    print(json.dumps(console, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

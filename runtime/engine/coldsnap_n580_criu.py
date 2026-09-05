#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Capture or activate a context-free inference process tree with CRIU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


PROCESS_STARTED_NS = time.monotonic_ns()
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import coldsnap_cuda_criu as base  # noqa: E402


CONTROLLER_ABI = 2
DEFAULT_TARGET = ROOT / "vllm_precuda_criu_target.py"
DEFAULT_PLUGIN_ROOT = ROOT.parent / "vllm_disk_sleep"
_PSM_PATH = re.compile(r"^/dev/shm/psm_[0-9a-f]+$")
_ACCELERATOR_PREFIXES = ("/dev/nvidia", "/dev/dri/", "/dev/infiniband/")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "restore"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--criu", type=Path, default=Path("/opt/criu/criu/criu"))
    parser.add_argument(
        "--criu-rpc",
        type=Path,
        default=Path("/usr/local/bin/coldsnap-criu-rpc"),
    )
    parser.add_argument("--runtime-lib-dir", type=Path)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--plugin-root", type=Path, default=DEFAULT_PLUGIN_ROOT)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--model-revision",
        default="2fc06364715b967f1860aea9cf38778875588b17",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--expected-text", default=" Paris")
    parser.add_argument("--minimum-target-pid", type=int, default=512)
    # CRIU must inspect the target's POSIX shared-memory namespace here; this
    # is not general-purpose temporary-file storage.
    parser.add_argument(
        "--shared-memory-dir",
        type=Path,
        default=Path("/dev/shm"),  # nosec B108
    )
    parser.add_argument("--ghost-limit", type=int, default=64 * 1024**2)
    parser.add_argument("--compress-block-bytes", type=int, default=256 * 1024)
    parser.add_argument("--compress-acceleration", type=int, default=1)
    parser.add_argument("--decompress-threads", type=int, default=1)
    parser.add_argument(
        "--image-io-mode", choices=("writeback", "direct"), default="direct"
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--compact-output", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        while chunk := source.read(16 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "target": args.target.resolve(),
        "criu": args.criu.resolve(),
        "criu_rpc": args.criu_rpc.resolve(),
    }
    entrypoints = sorted(
        args.plugin_root.glob("coldsnap_vllm-*.dist-info/entry_points.txt")
    )
    if len(entrypoints) != 1:
        raise FileNotFoundError(
            "plugin root must contain exactly one ColdSnap vLLM entrypoint"
        )
    paths["plugin_entrypoint"] = entrypoints[0].resolve()
    for path in sorted(args.plugin_root.glob("*.py")):
        paths[f"plugin/{path.name}"] = path.resolve()
    if args.runtime_lib_dir is not None:
        lz4 = args.runtime_lib_dir / "liblz4.so.1"
        if lz4.is_file():
            paths["criu_lz4"] = lz4.resolve()
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("pre-CUDA runtime is incomplete: " + ", ".join(missing))
    return {
        "format": 1,
        "kind": "coldsnap-vllm-precuda-criu-identity",
        "controller_abi": CONTROLLER_ABI,
        "image_id": args.image_id,
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "gpu": base._gpu_facts(),
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
            "expected_text": args.expected_text,
        },
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _wait_json(
    path: Path,
    timeout: float,
    *,
    fatal_paths: tuple[Path, ...] = (),
    process: subprocess.Popen[bytes] | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        if path.is_file():
            return base._load_json(path)
        for fatal in fatal_paths:
            if fatal.is_file():
                raise RuntimeError(f"pre-CUDA target failed: {base._load_json(fatal)}")
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"pre-CUDA target exited early: {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.01)


def _target_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(args.target),
        "--control-dir",
        str(args.artifact_root / "control"),
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


def _child_environment(args: argparse.Namespace, generation: str) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("COLDSNAP_")
    }
    python_path = environment.get("PYTHONPATH")
    paths = [str(args.plugin_root), str(args.target.resolve().parents[1])]
    if python_path:
        paths.append(python_path)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(paths),
            "VLLM_PLUGINS": "coldsnap",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "COLDSNAP_PROCESS_TEMPLATE_PHASE": "pre_worker_import",
            "COLDSNAP_MODEL_LOAD_RELEASE_FILE": str(
                args.artifact_root / "release"
            ),
            "COLDSNAP_MODEL_LOAD_READY_DIR": str(args.artifact_root / "ready"),
            "COLDSNAP_MODEL_LOAD_GENERATION": generation,
            "COLDSNAP_MODEL_LOAD_TIMEOUT_SECONDS": str(args.timeout),
            "COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE": str(
                args.artifact_root / "restore-generation"
            ),
            "COLDSNAP_PROCESS_TEMPLATE_RESTORE_WATCHER_SCOPE": "parent",
        }
    )
    return environment


def _criu_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    if args.runtime_lib_dir is not None:
        current = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            f"{args.runtime_lib_dir}:{current}"
            if current
            else str(args.runtime_lib_dir)
        )
    return environment


def _criu_command(
    args: argparse.Namespace, action: str, attempt_root: Path | None = None
) -> list[str]:
    if action not in {"dump", "restore"}:
        raise ValueError(f"unsupported CRIU action: {action}")
    runtime_root = args.artifact_root if attempt_root is None else attempt_root
    command = [
        str(args.criu_rpc),
        "--action",
        action,
        "--criu",
        str(args.criu),
        "--cpu-only",
        "--images-dir",
        str(runtime_root / "images"),
        "--work-dir",
        str(runtime_root / "work"),
        "--log-file",
        f"{action}.log",
        "--external-files",
        str(runtime_root / "external-files.json"),
        "--result",
        str(runtime_root / f"{action}-rpc.json"),
        "--timeout",
        str(max(1, int(args.timeout))),
        "--tcp-established",
        "--ghost-limit",
        str(args.ghost_limit),
        "--decompress-threads",
        str(args.decompress_threads),
        "--image-io-mode",
        args.image_io_mode,
    ]
    if action == "dump":
        command.extend(["--network-lock", "nftables"])
        if args.compress_block_bytes:
            command.extend(
                [
                    "--compress-block-size",
                    str(args.compress_block_bytes),
                    "--compress-acceleration",
                    str(args.compress_acceleration),
                ]
            )
    else:
        command.append("--nvidia-placeholder-fds")
    return command


def _criu_rpc_result(path: Path, action: str) -> dict[str, Any]:
    result = base._load_json(path / f"{action}-rpc.json")
    if (
        result.get("format") != 1
        or result.get("kind") != "coldsnap-criu-rpc-result"
        or result.get("action") != action
        or not isinstance(result.get("external_file_count"), int)
        or int(result["external_file_count"]) < 0
    ):
        raise RuntimeError(f"invalid CRIU RPC result: {result!r}")
    return result


def _seal_criu_template(args: argparse.Namespace) -> Path:
    """Move the completed CRIU dump into its immutable capsule namespace."""
    template = args.artifact_root / "template"
    template.mkdir()
    os.replace(args.artifact_root / "images", template / "images")
    os.replace(
        args.artifact_root / "external-files.json",
        template / "external-files.json",
    )
    return template


def _materialize_restore_attempt(args: argparse.Namespace) -> Path:
    """Clone one private CRIU image view from the sealed template."""
    template = args.artifact_root / "template"
    if not (template / "images/inventory.img").is_file():
        raise FileNotFoundError("sealed pre-CUDA CRIU image is missing")
    if not (template / "external-files.json").is_file():
        raise FileNotFoundError("sealed pre-CUDA external-file manifest is missing")
    attempt = args.artifact_root / "restore-attempts" / uuid.uuid4().hex
    attempt.parent.mkdir(exist_ok=True)
    attempt.mkdir()
    shutil.copytree(template / "images", attempt / "images", copy_function=shutil.copy2)
    shutil.copy2(template / "external-files.json", attempt / "external-files.json")
    (attempt / "work").mkdir()
    return attempt


def _validate_readiness(
    args: argparse.Namespace,
    generation: str,
    payloads: tuple[dict[str, Any], ...],
) -> None:
    for payload in payloads:
        if payload.get("generation") != generation:
            raise RuntimeError(f"pre-CUDA readiness generation mismatch: {payload}")
        if payload.get("phase") != "pre_worker_import":
            raise RuntimeError(f"pre-CUDA readiness phase mismatch: {payload}")
        if payload.get("role") == "frontend":
            admitted = (
                isinstance(payload.get("cuda_initialized"), bool)
                and payload.get("cuda_driver_context_present") is False
                and payload.get("cuda_primary_context_active") is False
                and payload.get(
                    "criu_frontend_driver_context_free_candidate"
                )
                is True
            )
        else:
            admitted = (
                payload.get("cuda_initialized") is False
                and payload.get("cuda_driver_context_present") is False
                and payload.get("cuda_primary_context_active") is False
                and payload.get("criu_cpu_only_candidate") is True
            )
        if not admitted:
            raise RuntimeError(f"process tree is not context-free: {payload}")
        descriptors = payload.get("accelerator_fds")
        if not isinstance(descriptors, list) or not all(
            isinstance(path, str) for path in descriptors
        ):
            raise RuntimeError(f"invalid accelerator descriptor audit: {payload}")
        unsupported = [
            path for path in descriptors if not path.startswith("/dev/nvidia")
        ]
        if unsupported:
            raise RuntimeError(
                "pre-CUDA controller cannot externalize accelerator descriptors: "
                f"{unsupported}"
            )


def _audit_process_tree(tree: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pid in tree:
        device_maps = []
        for line in Path(f"/proc/{pid}/maps").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            fields = line.split()
            if fields and fields[-1].startswith(_ACCELERATOR_PREFIXES):
                device_maps.append(line)
        if device_maps:
            raise RuntimeError(
                f"PID {pid} has accelerator device mappings at the pre-CUDA "
                f"boundary: {device_maps[:8]}"
            )
        descriptors: list[str] = []
        for descriptor in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                continue
            if target.startswith(_ACCELERATOR_PREFIXES):
                descriptors.append(target)
        unsupported = [
            path for path in descriptors if not path.startswith("/dev/nvidia")
        ]
        if unsupported:
            raise RuntimeError(
                f"PID {pid} has unsupported accelerator descriptors: {unsupported}"
            )
        records.append(
            {"pid": pid, "accelerator_fds": sorted(set(descriptors)), "device_maps": []}
        )
    return records


def _capture_regular_backings(
    args: argparse.Namespace, tree: list[int]
) -> list[dict[str, Any]]:
    paths: set[str] = set()
    for pid in tree:
        try:
            maps = Path(f"/proc/{pid}/maps").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except FileNotFoundError:
            # Helpers may exit after the process-tree audit but before this
            # best-effort backing inventory.  A vanished process owns no
            # state that CRIU will dump, so it must not invalidate capture.
            continue
        for line in maps:
            fields = line.split()
            if fields and _PSM_PATH.fullmatch(fields[-1]):
                paths.add(fields[-1])
        try:
            for descriptor in Path(f"/proc/{pid}/fd").iterdir():
                try:
                    target = os.readlink(descriptor)
                except FileNotFoundError:
                    continue
                if _PSM_PATH.fullmatch(target):
                    paths.add(target)
        except FileNotFoundError:
            # The process can also disappear between reading maps and
            # enumerating descriptors.  Keep any backing already observed.
            continue
    records = []
    for value in sorted(paths):
        try:
            metadata = Path(value).stat()
        except FileNotFoundError:
            # A short-lived process may unlink its own shared-memory object
            # while the inventory is being finalized.
            continue
        records.append(
            {
                "path": value,
                "bytes": metadata.st_size,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
        )
    base._atomic_json(
        args.artifact_root / "restore-regular-backings.json",
        {"format": 1, "files": records},
    )
    return records


def _restore_regular_backings(args: argparse.Namespace) -> list[dict[str, Any]]:
    record = base._load_json(args.artifact_root / "restore-regular-backings.json")
    files = record.get("files")
    if not isinstance(files, list):
        raise RuntimeError("invalid pre-CUDA regular-backing manifest")
    restored = []
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("invalid pre-CUDA regular-backing record")
        path = Path(str(item.get("path", "")))
        if not _PSM_PATH.fullmatch(str(path)):
            raise RuntimeError(f"unsafe pre-CUDA restore path: {path}")
        size = int(item["bytes"])
        mode = int(item["mode"])
        if size < 0 or mode < 0 or mode > 0o777:
            raise RuntimeError(f"invalid pre-CUDA restore metadata: {item}")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, mode)
        try:
            os.ftruncate(descriptor, size)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)
        restored.append({"path": str(path), "bytes": size, "mode": mode})
    return restored


def capture(args: argparse.Namespace) -> dict[str, Any]:
    identity = _identity(args)
    if any(args.artifact_root.iterdir()):
        raise FileExistsError("refusing to reuse a nonempty pre-CUDA artifact root")
    images = args.artifact_root / "images"
    work = args.artifact_root / "work"
    ready_dir = args.artifact_root / "ready"
    control_dir = args.artifact_root / "control"
    protected = (
        images / "inventory.img",
        args.artifact_root / "capture.json",
        args.artifact_root / "external-files.json",
        args.artifact_root / "dump-rpc.json",
    )
    if any(path.exists() for path in protected):
        raise FileExistsError("refusing to overwrite a pre-CUDA process artifact")
    for path in (images, work, ready_dir, control_dir):
        path.mkdir(parents=True, exist_ok=True)
    for path in (
        args.artifact_root / "release",
        args.artifact_root / "restore-generation",
        control_dir / "fatal.json",
        control_dir / "frontend-audit-error.json",
        control_dir / "result.json",
        control_dir / "terminate",
    ):
        path.unlink(missing_ok=True)
    generation = f"vllm-precuda-{uuid.uuid4().hex}"
    environment = _child_environment(args, generation)
    pid_floor = base._advance_pid_allocator(args.minimum_target_pid)
    log_path = args.artifact_root / "target.log"
    started = time.perf_counter()
    with log_path.open("ab", buffering=0) as log:
        target = subprocess.Popen(
            _target_command(args),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            fatals = (
                control_dir / "fatal.json",
                control_dir / "frontend-audit-error.json",
            )
            rank = _wait_json(
                ready_dir / "rank-0.json",
                args.timeout,
                fatal_paths=fatals,
                process=target,
            )
            frontend = _wait_json(
                ready_dir / "frontend.json",
                args.timeout,
                fatal_paths=fatals,
                process=target,
            )
            readiness = (frontend, rank)
            _validate_readiness(args, generation, readiness)
            tree = base._process_tree(target.pid)
            tree_audit = _audit_process_tree(tree)
            rss_bytes = sum(base._rss_bytes(pid) for pid in tree)
            regular_backings = _capture_regular_backings(args, tree)
            command = _criu_command(args, "dump")
            command.extend(["--pid", str(target.pid)])
            dump_profile = base._run_criu_profiled(
                command, _criu_environment(args)
            )
            criu_rpc = _criu_rpc_result(args.artifact_root, "dump")
            target.wait(timeout=60)
            shm_count, shm_bytes = base._snapshot_link_remaps(args)
            image_bytes = sum(
                path.stat().st_size for path in images.rglob("*") if path.is_file()
            )
            _seal_criu_template(args)
        except BaseException:
            if target.poll() is None:
                base._kill_process_group(target.pid)
                target.wait(timeout=30)
            raise
    report = {
        "format": 1,
        "kind": "coldsnap-vllm-precuda-criu-capture",
        "generation": generation,
        "identity": identity,
        "target_pid": target.pid,
        "pid_floor": pid_floor,
        "process_tree": tree,
        "process_tree_rss_bytes": rss_bytes,
        "process_tree_audit": tree_audit,
        "readiness": readiness,
        "regular_backings": regular_backings,
        "criu_rpc": criu_rpc,
        "dump_profile": dump_profile,
        "capture_controller_seconds": time.perf_counter() - started,
        "image_bytes": image_bytes,
        "shared_memory_link_count": shm_count,
        "shared_memory_link_bytes": shm_bytes,
    }
    base._atomic_json(args.artifact_root / "capture.json", report)
    return report


def _wait_process_exit(pid: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    process = Path(f"/proc/{pid}")
    while process.exists():
        try:
            state = next(
                line for line in (process / "status").read_text().splitlines()
                if line.startswith("State:")
            )
        except (FileNotFoundError, StopIteration):
            return
        if "Z" in state.split()[1:2]:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"restored process did not exit: {pid}")
        time.sleep(0.01)


def restore(args: argparse.Namespace) -> dict[str, Any]:
    capture_report = base._load_json(args.artifact_root / "capture.json")
    identity = _identity(args)
    if capture_report.get("identity") != identity:
        raise RuntimeError("pre-CUDA process artifact identity mismatch")
    generation = str(capture_report["generation"])
    control_dir = args.artifact_root / "control"
    for path in (
        args.artifact_root / "release",
        args.artifact_root / "restore-generation",
        args.artifact_root / "restore-rpc.json",
        control_dir / "fatal.json",
        control_dir / "frontend-audit-error.json",
        control_dir / "result.json",
        control_dir / "terminate",
    ):
        path.unlink(missing_ok=True)
    regular_backings = _restore_regular_backings(args)
    shm_count = base._prepare_link_remaps(args)
    attempt = _materialize_restore_attempt(args)
    command = _criu_command(args, "restore", attempt)
    restore_started_ns = time.monotonic_ns()
    restored_pid: int | None = None
    forced_shutdown = False
    cleanup_pending = False
    try:
        criu_profile = base._run_criu_profiled(command, _criu_environment(args))
        criu_rpc = _criu_rpc_result(attempt, "restore")
        restored_pid_value = criu_rpc.get("restored_pid")
        if not isinstance(restored_pid_value, int) or restored_pid_value <= 0:
            raise RuntimeError(f"invalid restored PID: {criu_rpc}")
        restored_pid = restored_pid_value
        release_ns = time.monotonic_ns()
        _atomic_text(args.artifact_root / "restore-generation", generation + "\n")
        _atomic_text(args.artifact_root / "release", generation + "\n")
        target = _wait_json(
            control_dir / "result.json",
            args.timeout,
            fatal_paths=(
                control_dir / "fatal.json",
                control_dir / "frontend-audit-error.json",
            ),
        )
        response_ns = time.monotonic_ns()
        if target.get("text") != args.expected_text:
            raise RuntimeError(
                f"restored inference mismatch: expected={args.expected_text!r} "
                f"actual={target.get('text')!r}"
            )
        (control_dir / "terminate").touch()
        try:
            _wait_process_exit(restored_pid, 5)
        except TimeoutError:
            forced_shutdown = True
            base._kill_process_group(restored_pid)
            try:
                _wait_process_exit(restored_pid, 5)
            except TimeoutError:
                # Correct inference is the activation boundary. A detached
                # CRIU root can remain visible as a zombie until container
                # init exits; do not turn benchmark-only teardown into a
                # failed resurrection after the response is safely persisted.
                cleanup_pending = True
    except BaseException:
        if restored_pid is not None:
            base._kill_process_group(restored_pid)
        raise
    restored_records = []
    for path in sorted((args.artifact_root / "ready").glob("restored-*.json")):
        restored_records.append(base._load_json(path))
    report = {
        "format": 1,
        "kind": "coldsnap-vllm-precuda-criu-restore",
        "generation": generation,
        "identity": identity,
        "restored_pid": restored_pid,
        "criu_rpc": criu_rpc,
        "restore_attempt": str(attempt.relative_to(args.artifact_root)),
        "criu_restore_profile": criu_profile,
        "criu_restore_seconds": criu_profile["wall_seconds"],
        "release_to_first_token_seconds": (response_ns - release_ns) / 1e9,
        "restore_start_to_first_token_seconds": (
            response_ns - restore_started_ns
        )
        / 1e9,
        "controller_start_to_first_token_seconds": (
            response_ns - PROCESS_STARTED_NS
        )
        / 1e9,
        "target": target,
        "restored_driver_clients": restored_records,
        "regular_backings": regular_backings,
        "shared_memory_link_count": shm_count,
        "forced_shutdown": forced_shutdown,
        "cleanup_pending_at_controller_exit": cleanup_pending,
    }
    base._atomic_json(args.artifact_root / "restore-result.json", report)
    return report


def main() -> int:
    args = _parser().parse_args()
    if not 0 < args.gpu_memory_utilization <= 0.2:
        raise ValueError("gpu-memory-utilization must be in (0, 0.2]")
    if args.timeout <= 0 or args.timeout > 3600:
        raise ValueError("timeout must be in (0, 3600]")
    if (
        args.compress_block_bytes < 0
        or args.compress_block_bytes > 4 * 1024**2
        or (
            args.compress_block_bytes
            and args.compress_block_bytes % os.sysconf("SC_PAGE_SIZE") != 0
        )
    ):
        raise ValueError("compression block must be zero or a page multiple up to 4 MiB")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        report = capture(args) if args.mode == "capture" else restore(args)
    except BaseException as error:
        report = {
            "format": 1,
            "kind": "coldsnap-vllm-precuda-criu-error",
            "error": f"{type(error).__name__}: {error}",
        }
        base._atomic_json(args.artifact_root / "controller-error.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1
    console = report
    if args.compact_output and report.get("kind", "").endswith("restore"):
        console = {
            "kind": report["kind"],
            "criu_restore_seconds": report["criu_restore_seconds"],
            "release_to_first_token_seconds": report[
                "release_to_first_token_seconds"
            ],
            "restore_start_to_first_token_seconds": report[
                "restore_start_to_first_token_seconds"
            ],
            "token_ids": report["target"]["token_ids"],
            "text": report["target"]["text"],
        }
    print(json.dumps(console, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

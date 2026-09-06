#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Capture or restore one n610 rank of a distributed inference service."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import os
import platform
import re
import signal
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import coldsnap_cuda_criu as base  # noqa: E402
import coldsnap_activation_logs as activation_logs  # noqa: E402
import coldsnap_service_runtime as service_runtime  # noqa: E402
from coldsnap_coord import Client as CoordinatorClient  # noqa: E402


CONTROLLER_ABI = 8
_COORDINATOR_MAX_WAIT_SECONDS = 3600.0
_FLASHINFER_LOG = re.compile(r"\[(/root/\.cache/flashinfer/[A-Za-z0-9._/-]+/flashinfer_jit\.log)\]")
_PSM_PATH = re.compile(r"^/dev/shm/psm_[0-9a-f]+$")
_SGLANG_LOADS_PATH = re.compile(r"^/dev/shm/sglang_loads_[A-Za-z0-9._-]{1,200}_[0-9a-f]{8}\.shm$")
_CRIU_LOG_TIME = re.compile(r"^\(([0-9]+(?:\.[0-9]+)?)\)")

_bounded_json = service_runtime._bounded_json
_capture_shape_calibration = service_runtime._capture_shape_calibration
_collective_rpc = service_runtime._collective_rpc
_coordinator_wait = service_runtime._coordinator_wait
_hibernate_states = service_runtime._hibernate_states
_hydration_manifests = service_runtime._hydration_manifests
_identity_http_port = service_runtime._identity_http_port
_identity_mismatch_paths = service_runtime._identity_mismatch_paths
_infer = service_runtime._infer
_load_active_nccl_runtime = service_runtime._load_active_nccl_runtime
_normalize_placement_command = service_runtime._normalize_placement_command
_prepare_sglang_hydration = service_runtime._prepare_sglang_hydration
_request = service_runtime._request
_select_weight_provider = service_runtime._select_weight_provider
_sglang_memory_payload = service_runtime._sglang_memory_payload
_stage_restore_runtime_environment = service_runtime._stage_restore_runtime_environment
_stage_restore_transport_environment = service_runtime._stage_restore_transport_environment
_torch_cuda_userspace_version = service_runtime._torch_cuda_userspace_version
_visible_gpu_facts = service_runtime._visible_gpu_facts
_wait_http = service_runtime._wait_http


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "restore"))
    parser.add_argument("--engine", choices=("vllm", "sglang"), default="vllm")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--weight-provider", choices=("native", "recovery"), default="recovery")
    parser.add_argument("--target-launcher", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--nccl-active-runtime", type=Path, required=True)
    parser.add_argument("--criu", type=Path, required=True)
    parser.add_argument("--criu-rpc", type=Path, required=True)
    parser.add_argument("--runtime-lib-dir", type=Path, required=True)
    parser.add_argument("--cuda-checkpoint", type=Path, required=True)
    parser.add_argument("--native-hydration-library", type=Path, required=True)
    parser.add_argument("--coordinator-path", type=Path, required=True)
    parser.add_argument("--activation-namespace", required=True)
    parser.add_argument("--minimum-target-pid", type=int, default=512)
    # CRIU must inspect the target's POSIX shared-memory namespace here; this
    # is not general-purpose temporary-file storage.
    parser.add_argument(
        "--shared-memory-dir",
        type=Path,
        default=Path("/dev/shm"),  # nosec B108
    )
    parser.add_argument("--ghost-limit", type=int, default=64 * 1024**2)
    parser.add_argument("--criu-compress-block-bytes", type=int, default=256 * 1024)
    parser.add_argument("--criu-compress-acceleration", type=int, default=1)
    parser.add_argument("--criu-decompress-threads", type=int, default=1)
    parser.add_argument("--criu-image-io-mode", choices=("writeback", "direct"), default="direct")
    parser.add_argument(
        "--activation-state",
        choices=("running", "warm"),
        default="running",
        help="return after full activation or hold after NCCL restore before weight hydration",
    )
    parser.add_argument(
        "--kernel-compatibility", choices=("capability", "exact"), default="capability"
    )
    parser.add_argument("--allow-compatible-criu-runtime", action="store_true")
    parser.add_argument(
        "--leave-stopped",
        action="store_true",
        help="resume the restored tree only after every distributed rank exists",
    )
    parser.add_argument(
        "--defer-network-unlock",
        action="store_true",
        help="hold CRIU's restored TCP firewall until every launch unit reaches network-unlock",
    )
    parser.add_argument(
        "--tcp-address-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="rebind captured TCP endpoints to their destination placement",
    )
    parser.add_argument(
        "--tcp-port-shift",
        type=int,
        default=0,
        help="rotate restored placement TCP ports to a fresh activation identity",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--compute-sanitizer", action="store_true")
    parser.add_argument("serve_command", nargs=argparse.REMAINDER)
    return parser


def _criu_log_profile(path: Path) -> dict[str, Any] | None:
    """Extract coarse CRIU phases without changing CRIU itself."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    milestones = {
        "reading_image_tree": "Reading image tree",
        "pre_restore_scripts": "Running pre-restore scripts",
        "late_external_device_hook": "Run late stage hook from criu master",
        "restore_finished": "Restore finished successfully",
    }
    milestone_seconds: dict[str, float] = {}
    largest_gaps: list[dict[str, Any]] = []
    previous_seconds: float | None = None
    previous_line = 0
    first_seconds: float | None = None
    last_seconds: float | None = None
    for line_number, line in enumerate(lines, start=1):
        match = _CRIU_LOG_TIME.match(line)
        if match is None:
            continue
        seconds = float(match.group(1))
        first_seconds = seconds if first_seconds is None else first_seconds
        last_seconds = seconds
        if previous_seconds is not None and seconds > previous_seconds:
            largest_gaps.append(
                {
                    "seconds": seconds - previous_seconds,
                    "from_line": previous_line,
                    "to_line": line_number,
                    "next_event": line[:240],
                }
            )
        previous_seconds = seconds
        previous_line = line_number
        for name, marker in milestones.items():
            if name not in milestone_seconds and marker in line:
                milestone_seconds[name] = seconds
    largest_gaps.sort(key=lambda gap: float(gap["seconds"]), reverse=True)
    return {
        "first_log_seconds": first_seconds,
        "last_log_seconds": last_seconds,
        "milestone_seconds": milestone_seconds,
        "largest_timed_gaps": largest_gaps[:10],
    }




def _daemon_request(
    args: argparse.Namespace,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> concurrent.futures.Future[Any]:
    """Start an intentionally capture-blocked request without owning shutdown."""
    future: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(_request(args, method, path, payload))
        except BaseException as error:
            future.set_exception(error)

    threading.Thread(
        target=run,
        name="coldsnap-sglang-capture-hold",
        daemon=True,
    ).start()
    return future




def _checkpoint(args: argparse.Namespace, operation: str) -> Any:
    deadline = time.monotonic() + args.timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        remaining = max(0.001, deadline - time.monotonic())
        wait = min(_COORDINATOR_MAX_WAIT_SECONDS, remaining)
        try:
            return _request(
                args,
                "POST",
                "/collective_rpc",
                {
                    "method": f"checkpoint_{operation}",
                    "timeout": max(1, int(wait)),
                },
                timeout=wait,
            )
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.25)
    raise TimeoutError(f"checkpoint_{operation} failed: {last_error}")


def _checkpoint_max_seconds(response: Any, key: str) -> float | None:
    """Return the distributed critical-path value from worker RPC results."""

    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        return None
    values = [
        float(result[key])
        for result in response["results"]
        if isinstance(result, dict)
        and isinstance(result.get(key), (int, float))
        and not isinstance(result.get(key), bool)
        and float(result[key]) >= 0
        and math.isfinite(float(result[key]))
    ]
    return max(values) if values else None




def _retained_nccl_graphs() -> bool:
    return os.environ.get("COLDSNAP_GRAPH_POLICY", "").strip() == "preserve-nccl-exec"


def _stage_in_place_nccl_activation(args: argparse.Namespace) -> Path | None:
    if not _retained_nccl_graphs():
        return None
    if os.environ.get("COLDSNAP_NCCL_IN_PLACE_MODE", "").strip() != "net-reconnect-v1":
        raise RuntimeError("retained NCCL graphs require the exact net-reconnect-v1 provider")
    raw = os.environ.get("COLDSNAP_NCCL_IN_PLACE_ACTIVATION_PATH", "").strip()
    expected = args.artifact_root / "nccl-in-place-activation"
    if not raw or Path(raw) != expected:
        raise RuntimeError("retained NCCL graph activation path is outside the rank capsule")
    if re.fullmatch(r"[A-Za-z0-9._-]{1,160}", args.activation_namespace) is None:
        raise RuntimeError("NCCL in-place activation namespace is invalid")
    temporary = expected.with_name(f".{expected.name}.{os.getpid()}.tmp")
    temporary.write_text(args.activation_namespace + "\n", encoding="utf-8")
    os.replace(temporary, expected)
    return expected








def _http_error(error: BaseException) -> dict[str, Any]:
    result = {"type": type(error).__name__, "message": str(error)}
    if isinstance(error, urllib.error.HTTPError):
        result["status"] = error.code
        try:
            result["body"] = error.read().decode("utf-8", errors="replace")[-16384:]
        except OSError:
            pass
    return result




def _arm_sglang_async_graphs(
    args: argparse.Namespace,
    coordinator: CoordinatorClient,
    prefix: str,
) -> bool:
    if (
        args.engine != "sglang"
        or args.activation_state != "running"
        or os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS", "0") != "1"
    ):
        return False
    arm_value = os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE", "").strip()
    if not arm_value:
        raise RuntimeError("SGLang async graph activation has no arm file")
    Path(arm_value).touch()
    key = f"{prefix}:sglang-async-graphs-armed"
    coordinator.set(f"{key}:{args.rank}", "1")
    for rank in range(args.world_size):
        _coordinator_wait(coordinator, f"{key}:{rank}", args.timeout)
    return True


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    active = _load_active_nccl_runtime(args.nccl_active_runtime)
    resolved = active["_resolved"]
    paths: dict[str, Path] = {
        "target_launcher": args.target_launcher,
        "criu": args.criu,
        "criu_rpc": args.criu_rpc,
        "criu_lz4": args.runtime_lib_dir / "liblz4.so.1",
        "cuda_checkpoint": args.cuda_checkpoint,
        "native_hydration": args.native_hydration_library,
        "nccl_library": resolved["nccl-runtime"],
        "nccl_checkpoint_shim": resolved["checkpoint-shim"],
        "nccl_dlsym_bridge": resolved["bridge"],
    }
    if args.engine == "vllm":
        paths.update(
            {
                "engine_plugin": args.plugin_root / "coldsnap_plugin.py",
                "engine_nccl_adapter": args.plugin_root / "coldsnap_vllm_nccl_checkpoint.py",
                "engine_kv_capacity_adapter": args.plugin_root / "coldsnap_vllm_kv_capacity.py",
                "engine_kv_payload_adapter": args.plugin_root / "coldsnap_vllm_kv_payload.py",
            }
        )
    else:
        paths.update(
            {
                "engine_plugin": args.plugin_root / "coldsnap_sglang" / "plugin.py",
                "engine_nccl_adapter": args.plugin_root / "coldsnap_nccl_checkpoint.py",
            }
        )
    for path in sorted(args.plugin_root.glob("*.py")):
        paths[f"engine_plugin_bundle/{path.name}"] = path
    for path in sorted((args.plugin_root / "coldsnap_sglang").glob("*.py")):
        paths[f"engine_plugin_bundle/coldsnap_sglang/{path.name}"] = path
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("snapshot runtime is incomplete: " + ", ".join(missing))
    command = args.serve_command[1:] if args.serve_command[:1] == ["--"] else args.serve_command
    if not command:
        raise ValueError(f"the {args.engine} serve command is required")
    identity = {
        "format": 1,
        "kind": f"coldsnap-{args.engine}-cuda-criu-identity",
        "controller_abi": CONTROLLER_ABI,
        "engine": args.engine,
        "unit": os.environ.get("COLDSNAP_EXPECTED_UNIT", ""),
        "unit_index": args.rank,
        "unit_count": args.world_size,
        "image_id": args.image_id,
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "gpus": _visible_gpu_facts(),
        "cuda_userspace": _torch_cuda_userspace_version(),
        "model": args.model,
        "model_revision": args.model_revision,
        "validation_prompt": args.prompt,
        "validation_expected": args.expected,
        "serve_command": command,
        "runtime_sha256": {name: base._sha256(path) for name, path in paths.items()},
        "nccl_provider": {
            "format": 1,
            "provider_id": active["provider_id"],
            "provider_revision": active["provider_revision"],
            "platform_key": active["platform_key"],
            "manifest_sha256": active["manifest_sha256"],
            "provider_abi": active["provider_abi"],
            "checkpoint_abi": active["checkpoint_abi"],
            "provider_nccl_runtime": active["provider_nccl_runtime"],
            "provider_selection": active["provider_selection"],
            "supported_nccl_runtimes": active["supported_nccl_runtimes"],
            "capabilities": active["capabilities"],
            "limitations": active["limitations"],
            "qualification": active["qualification"],
            "files": {
                role: {
                    "sha256": item["sha256"],
                    "build_id": item["build_id"],
                    "soname": item["soname"],
                }
                for role, item in sorted(active["files"].items())
            },
            "bridge": {
                "abi": active["bridge"]["abi"],
                "sha256": active["bridge"]["sha256"],
            },
            "preload_order": [
                "coldsnap-dlsym-bridge",
                "checkpoint-shim",
                "nccl-runtime",
            ],
        },
    }
    if args.criu_compress_block_bytes:
        identity["criu_compression"] = {
            "mode": "lz4-block",
            "block_bytes": args.criu_compress_block_bytes,
            "acceleration": args.criu_compress_acceleration,
        }
    return identity






def _numeric_version(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value):
        raise RuntimeError(f"process artifact has invalid {field} version")
    return tuple(int(component) for component in value.split("."))














def _identity_mismatch_error(captured: object, current: object, *, scope: str = "") -> RuntimeError:
    paths = _identity_mismatch_paths(captured, current)
    detail = ", ".join(paths[:8]) if paths else "unknown fields"
    return RuntimeError(f"process artifact identity mismatch{scope}: {detail}")


def _identity_compatibility(
    captured: object,
    current: dict[str, Any],
    allow_criu_runtime_upgrade: bool,
    kernel_compatibility: str = "capability",
) -> str:
    if captured == current:
        return "exact-v8-placement-independent"
    if not isinstance(captured, dict):
        raise _identity_mismatch_error(captured, current)
    captured_compatible = copy.deepcopy(captured)
    current_compatible = copy.deepcopy(current)

    if captured_compatible.get("controller_abi") != CONTROLLER_ABI:
        raise RuntimeError("process artifact predates the portable identity policy")
    for identity in (captured_compatible, current_compatible):
        identity["serve_command"] = _normalize_placement_command(identity.get("serve_command"))
    kernel_differs = captured_compatible.get("kernel") != current_compatible.get("kernel")
    if kernel_compatibility == "capability":
        current_compatible["kernel"] = captured_compatible.get("kernel")
    elif kernel_compatibility != "exact":
        raise RuntimeError(f"unsupported kernel compatibility policy: {kernel_compatibility!r}")
    # Device slots are stable artifact identity; physical device ordinals are not.
    captured_gpus = captured_compatible.get("gpus")
    current_gpus = current_compatible.get("gpus")
    if (
        not isinstance(captured_gpus, list)
        or not isinstance(current_gpus, list)
        or len(captured_gpus) != len(current_gpus)
        or not captured_gpus
        or not all(isinstance(gpu, dict) for gpu in captured_gpus + current_gpus)
    ):
        raise RuntimeError("process artifact has invalid GPU identity")
    for slot, (captured_gpu, current_gpu) in enumerate(
        zip(captured_gpus, current_gpus, strict=True)
    ):
        captured_driver = _numeric_version(
            captured_gpu.get("driver"), f"captured NVIDIA driver for slot {slot}"
        )
        current_driver = _numeric_version(
            current_gpu.get("driver"), f"current NVIDIA driver for slot {slot}"
        )
        if current_driver < captured_driver:
            raise RuntimeError(
                f"current NVIDIA driver for slot {slot} is older than the captured minimum"
            )
        # The capsule pins CUDA userspace; driver upgrades remain placement.
        current_gpu["driver"] = captured_gpu["driver"]

    compatibility = (
        "portable-v8-capability-kernel"
        if kernel_differs and kernel_compatibility == "capability"
        else "portable-v8-placement-independent"
    )
    if allow_criu_runtime_upgrade:
        compatibility = "portable-v8-explicit-criu-runtime-upgrade"
    for identity in (captured_compatible, current_compatible):
        runtime = identity.get("runtime_sha256")
        if not isinstance(runtime, dict):
            raise RuntimeError("process artifact has invalid runtime identity")
        if allow_criu_runtime_upgrade:
            for name in ("criu", "criu_rpc", "criu_lz4"):
                runtime.pop(name, None)
    if captured_compatible != current_compatible:
        scope = " outside CRIU runtime" if allow_criu_runtime_upgrade else ""
        raise _identity_mismatch_error(captured_compatible, current_compatible, scope=scope)
    return compatibility


def _child_command(args: argparse.Namespace) -> list[str]:
    command = args.serve_command[1:] if args.serve_command[:1] == ["--"] else args.serve_command
    return [sys.executable, str(args.target_launcher), "--", *command]


def _without_expandable_segments(value: str | None) -> str:
    settings: list[str] = []
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        name = item.split(":", 1)[0].strip()
        if name == "expandable_segments":
            continue
        settings.append(item)
    settings.append("expandable_segments:False")
    return ",".join(settings)


def _child_environment(args: argparse.Namespace, job_file: Path) -> dict[str, str]:
    environment = os.environ.copy()
    engine = getattr(args, "engine", "vllm")
    environment["COLDSNAP_ENGINE"] = engine
    if engine == "sglang":
        # SGLang's TorchMemorySaver owns stable VMM regions during capture and
        # explicitly cannot coexist with PyTorch expandable segments. Preserve
        # every other allocator tuning while making that one ownership choice
        # deterministic inside ColdSnap rather than in a recipe environment.
        environment["PYTORCH_CUDA_ALLOC_CONF"] = _without_expandable_segments(
            environment.get("PYTORCH_CUDA_ALLOC_CONF")
        )
        if "PYTORCH_ALLOC_CONF" in environment:
            environment["PYTORCH_ALLOC_CONF"] = _without_expandable_segments(
                environment.get("PYTORCH_ALLOC_CONF")
            )
    environment["CUDA_CHECKPOINT_JOB_FILE"] = str(job_file)
    active = _load_active_nccl_runtime(args.nccl_active_runtime)
    environment["LD_PRELOAD"] = ":".join(active["preload_order"])
    environment["COLDSNAP_NCCL_CHECKPOINT_SHIM_PATH"] = str(active["_resolved"]["checkpoint-shim"])
    environment["COLDSNAP_NCCL_PROVIDER_ID"] = active["provider_id"]
    environment["COLDSNAP_NCCL_PROVIDER_REVISION"] = str(active["provider_revision"])
    environment["COLDSNAP_NCCL_DLSYM_BRIDGE_ABI"] = str(active["bridge"]["abi"])
    return environment


def _criu_environment(args: argparse.Namespace, job_file: Path) -> dict[str, str]:
    environment = os.environ.copy()
    current = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{args.runtime_lib_dir}:{current}" if current else str(args.runtime_lib_dir)
    )
    environment["CUDA_CHECKPOINT_JOB_FILE"] = str(job_file)
    environment.pop("LD_PRELOAD", None)
    return environment
















def _sglang_cuda_restore_hold_root(artifact_root: Path) -> Path:
    return artifact_root / "cuda-restore-hold"


def _wait_sglang_cuda_restore_hold(
    args: argparse.Namespace,
    process: subprocess.Popen[bytes],
    request: concurrent.futures.Future[Any] | None,
) -> list[dict[str, Any]]:
    root = _sglang_cuda_restore_hold_root(args.artifact_root)
    deadline = time.monotonic() + args.timeout
    while True:
        records: list[dict[str, Any]] = []
        for path in sorted(root.glob("*.ready.json")):
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"SGLang CUDA restore hold is not a regular file: {path}")
            record = _bounded_json(path)
            if (
                record.get("format") != 1
                or record.get("kind") != "coldsnap-sglang-cuda-restore-hold"
                or not isinstance(record.get("worker_id"), str)
                or not isinstance(record.get("pid"), int)
            ):
                raise RuntimeError(f"invalid SGLang CUDA restore hold record: {path}")
            records.append(record)
        workers = {record["worker_id"] for record in records}
        if len(records) == args.worker_count and len(workers) == args.worker_count:
            return records
        if len(records) > args.worker_count:
            raise RuntimeError(
                "SGLang CUDA restore hold contains more workers than the launch unit: "
                f"expected={args.worker_count} actual={len(records)}"
            )
        if request is not None and request.done():
            request.result()
            raise RuntimeError("SGLang sleep completed before reaching the CRIU hold")
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang target exited before reaching the CRIU hold: {process.returncode}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for SGLang workers to reach the CRIU hold: "
                f"expected={args.worker_count} actual={len(records)}"
            )
        time.sleep(0.01)


def _release_sglang_cuda_restore_hold(args: argparse.Namespace) -> bool:
    root = _sglang_cuda_restore_hold_root(args.artifact_root)
    ready = sorted(root.glob("*.ready.json"))
    if not ready:
        return False
    if len(ready) != args.worker_count:
        raise RuntimeError(
            "captured SGLang CUDA restore hold differs from the launch unit: "
            f"expected={args.worker_count} actual={len(ready)}"
        )
    release = root / "release"
    release.unlink(missing_ok=True)
    _publish_barrier_release(release)
    return True


def _sglang_nccl_observations(states: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for worker, state in sorted(states.items()):
        observation = state.get("nccl") if isinstance(state, dict) else None
        if not isinstance(observation, dict):
            raise RuntimeError(f"SGLang worker {worker} lacks NCCL checkpoint evidence")
        if observation.get("worker_id") != worker or not isinstance(
            observation.get("version"), dict
        ):
            raise RuntimeError(f"SGLang worker {worker} has invalid NCCL evidence")
        results.append(observation)
    return {"results": results}






def _command_load_format(command: list[str]) -> str:
    for index, argument in enumerate(command):
        if argument == "--load-format":
            if index + 1 >= len(command):
                raise RuntimeError("--load-format requires a value")
            return command[index + 1]
        if argument.startswith("--load-format="):
            return argument.split("=", 1)[1]
    return "auto"


def _criu_command(args: argparse.Namespace, action: str, log_file: str) -> list[str]:
    command = [
        str(args.criu_rpc),
        "--action",
        action,
        "--criu",
        str(args.criu),
        "--cuda-checkpoint",
        str(args.cuda_checkpoint),
        "--cuda-process-tree",
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
        str(min(3600, max(1, int(args.timeout)))),
        "--tcp-established",
        "--ghost-limit",
        str(args.ghost_limit),
        "--decompress-threads",
        str(args.criu_decompress_threads),
        "--image-io-mode",
        args.criu_image_io_mode,
    ]
    if action == "dump":
        command.extend(["--network-lock", "nftables"])
        if args.criu_compress_block_bytes:
            command.extend(
                [
                    "--compress-block-size",
                    str(args.criu_compress_block_bytes),
                    "--compress-acceleration",
                    str(args.criu_compress_acceleration),
                ]
            )
    elif getattr(args, "tcp_address_map", []):
        for mapping in args.tcp_address_map:
            command.extend(["--tcp-address-map", mapping])
        captured_http_port = getattr(args, "captured_http_port", args.http_port)
        if captured_http_port != args.http_port:
            command.extend(["--tcp-port-map", f"{captured_http_port}={args.http_port}"])
        if getattr(args, "tcp_port_shift", 0):
            command.extend(["--tcp-port-shift", str(args.tcp_port_shift)])
            if captured_http_port == args.http_port:
                command.extend(["--tcp-preserve-port", str(args.http_port)])
    if action == "restore" and getattr(args, "leave_stopped", False):
        command.append("--leave-stopped")
    if action == "restore" and getattr(args, "defer_network_unlock", False):
        command.append("--defer-network-unlock")
    return command


def _restore_command(args: argparse.Namespace) -> list[str]:
    command = _criu_command(args, "restore", "restore.log")
    if not args.compute_sanitizer:
        return command
    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        raise FileNotFoundError("compute-sanitizer is unavailable in the image")
    return [
        sanitizer,
        "--tool",
        "memcheck",
        "--target-processes",
        "all",
        "--force-blocking-launches",
        "--report-api-errors",
        "all",
        "--print-limit",
        "20",
        "--require-cuda-init",
        "no",
        "--error-exitcode",
        "86",
        "--log-file",
        str(args.artifact_root / "work/compute-sanitizer.log"),
        *command,
    ]


def _deferred_cuda_restore_command(args: argparse.Namespace, restored_pid: int) -> list[str]:
    return [
        str(args.criu_rpc),
        "--action",
        "cuda-restore",
        "--cuda-checkpoint",
        str(args.cuda_checkpoint),
        "--cuda-process-tree",
        "--cuda-processes-result",
        str(args.artifact_root / "dump-rpc.json"),
        "--pid",
        str(restored_pid),
        "--result",
        str(args.artifact_root / "cuda-restore-rpc.json"),
        "--timeout",
        str(min(3600, max(1, int(args.timeout)))),
    ]


def _restorable_regular_backing(path: str, engine: str) -> bool:
    return _PSM_PATH.fullmatch(path) is not None or (
        engine == "sglang" and _SGLANG_LOADS_PATH.fullmatch(path) is not None
    )


def _capture_regular_backings(args: argparse.Namespace, tree: list[int]) -> list[dict[str, Any]]:
    paths: set[str] = set()
    for pid in tree:
        maps = Path(f"/proc/{pid}/maps")
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if fields and _restorable_regular_backing(fields[-1], args.engine):
                paths.add(fields[-1])
        for descriptor in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                continue
            if _restorable_regular_backing(target, args.engine) or _FLASHINFER_LOG.fullmatch(
                f"[{target}]"
            ):
                paths.add(target)

    records: list[dict[str, Any]] = []
    for value in sorted(paths):
        target = Path(f"/proc/{tree[0]}/root") / value.removeprefix("/")
        metadata = target.stat()
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


def _restore_regular_backings(
    args: argparse.Namespace, capture_report: dict[str, Any]
) -> list[dict[str, Any]]:
    """Recreate bounded volatile files whose mapped pages live in CRIU images."""
    dump_log = args.artifact_root / "work/dump.log"
    records: dict[Path, dict[str, int]] = {}
    captured = capture_report.get("regular_backings")
    persisted_path = args.artifact_root / "restore-regular-backings.json"
    if not isinstance(captured, list) and persisted_path.is_file():
        persisted = base._load_json(persisted_path)
        captured = persisted.get("files")
    if isinstance(captured, list):
        for item in captured:
            if not isinstance(item, dict):
                raise RuntimeError("invalid captured regular-backing record")
            path = Path(str(item.get("path", "")))
            if not (
                _restorable_regular_backing(str(path), args.engine)
                or _FLASHINFER_LOG.fullmatch(f"[{path}]")
            ):
                raise RuntimeError(f"unsafe captured restore path: {path}")
            records[path] = {
                "bytes": int(item["bytes"]),
                "mode": int(item["mode"]),
            }
    for line in dump_log.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "Handling VMA with the following smaps entry: "
        if marker in line:
            fields = line.split(marker, 1)[1].split()
            if len(fields) >= 6 and _restorable_regular_backing(fields[-1], args.engine):
                start_text, separator, end_text = fields[0].partition("-")
                if not separator:
                    raise RuntimeError(f"malformed CRIU VMA range: {fields[0]!r}")
                start = int(start_text, 16)
                end = int(end_text, 16)
                offset = int(fields[2], 16)
                if start >= end or offset < 0:
                    raise RuntimeError(f"invalid CRIU VMA record: {fields!r}")
                path = Path(fields[-1])
                candidate = offset + end - start
                record = records.setdefault(path, {"bytes": 0, "mode": 0o600})
                if not isinstance(captured, list):
                    record["bytes"] = max(record["bytes"], candidate)
        for match in _FLASHINFER_LOG.finditer(line):
            path = Path(match.group(1))
            if ".." in path.parts:
                raise RuntimeError(f"unsafe FlashInfer restore path: {path}")
            records.setdefault(path, {"bytes": 0, "mode": 0o644})

    previous_restore = args.artifact_root / "work/restore.log"
    if previous_restore.is_file():
        for line in previous_restore.read_text(encoding="utf-8", errors="replace").splitlines():
            size_match = re.search(
                r"File (dev/shm/psm_[0-9a-f]+) has bad size \d+ "
                r"\(expect (\d+)\)",
                line,
            )
            if size_match:
                path = Path("/") / size_match.group(1)
                record = records.setdefault(path, {"bytes": 0, "mode": 0o600})
                record["bytes"] = int(size_match.group(2))
            mode_match = re.search(
                r"File (root/\.cache/flashinfer/[^ ]+/flashinfer_jit\.log) "
                r"has bad mode [0-7]+ \(expect 0100([0-7]{3})\)",
                line,
            )
            if mode_match:
                path = Path("/") / mode_match.group(1)
                record = records.setdefault(path, {"bytes": 0, "mode": 0o644})
                record["mode"] = int(mode_match.group(2), 8)

    restored: list[dict[str, Any]] = []
    for path, record in sorted(records.items(), key=lambda item: str(item[0])):
        size = record["bytes"]
        mode = record["mode"]
        if size < 0 or mode < 0 or mode > 0o777:
            raise RuntimeError(f"invalid restore metadata for {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, mode)
        try:
            os.ftruncate(descriptor, size)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)
        restored.append({"path": str(path), "bytes": size, "mode": mode})
    base._atomic_json(persisted_path, {"format": 1, "files": restored})
    return restored


def capture(args: argparse.Namespace) -> dict[str, Any]:
    identity = _identity(args)
    images = args.artifact_root / "images"
    work = args.artifact_root / "work"
    if any(
        path.exists()
        for path in (
            images / "inventory.img",
            args.artifact_root / "capture.json",
            args.artifact_root / activation_logs.CAPTURE_LOG_FILENAME,
            args.artifact_root / activation_logs.TARGET_LOG_FILENAME,
            args.artifact_root / "external-files.json",
            args.artifact_root / "dump-rpc.json",
        )
    ):
        raise FileExistsError("refusing to overwrite a process artifact")
    for path in (images, work):
        path.mkdir(parents=True, exist_ok=True)
    (args.artifact_root / "async-graphs-arm").unlink(missing_ok=True)
    (args.artifact_root / "async-graphs-ready.json").unlink(missing_ok=True)
    hold_root = _sglang_cuda_restore_hold_root(args.artifact_root)
    if hold_root.exists():
        for path in hold_root.glob("*.ready.json"):
            path.unlink()
        (hold_root / "release").unlink(missing_ok=True)
    generation = f"{args.engine}-{uuid.uuid4().hex}"
    job_file = base._create_cuda_job(args)
    pid_floor = base._advance_pid_allocator(args.minimum_target_pid)
    coordinator = CoordinatorClient.from_path(args.coordinator_path)
    prepare_only = os.environ.get("COLDSNAP_CAPTURE_PREPARE_ONLY", "0") == "1"
    log_path = args.artifact_root / "target.log"
    started = time.perf_counter()
    sleep_future: concurrent.futures.Future[Any] | None = None
    shape_calibration = None
    with log_path.open("ab", buffering=0) as log:
        target = subprocess.Popen(
            _child_command(args),
            env=_child_environment(args, job_file),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_http(args, "/health", process=target)
            before = None
            sleep = None
            sleep_seconds = None
            prepare = None
            ib_before_sleep = None
            ib_after_sleep = None
            prepare_error = None
            key = f"{args.activation_namespace}:capture-prepared"
            sleep_key = f"{args.activation_namespace}:capture-slept"
            diagnostic_key = f"{args.activation_namespace}:capture-prepare-only"
            if args.engine == "sglang" and not prepare_only:
                if args.rank == 0:
                    before = _infer(args)
                    phase = time.perf_counter()
                    sleep_future = _daemon_request(
                        args,
                        "POST",
                        "/release_memory_occupation",
                        _sglang_memory_payload(),
                    )
                _wait_sglang_cuda_restore_hold(args, target, sleep_future)
                if args.rank == 0:
                    sleep_seconds = time.perf_counter() - phase
                    sleep = {"deferred_at": "cuda-restore-hold"}
                coordinator.set(f"{sleep_key}:{args.rank}", "1")
                for rank in range(args.world_size):
                    _coordinator_wait(coordinator, f"{sleep_key}:{rank}", args.timeout)
            elif args.rank == 0:
                before = _infer(args)
                if args.engine == "vllm":
                    ib_before_sleep = _collective_rpc(args, "coldsnap_nccl_ib_status")
                phase = time.perf_counter()
                if args.engine == "vllm":
                    sleep = _request(args, "POST", "/sleep?level=1")
                else:
                    sleep = _request(
                        args,
                        "POST",
                        "/release_memory_occupation",
                        _sglang_memory_payload(),
                    )
                sleep_seconds = time.perf_counter() - phase
                coordinator.set(sleep_key, "1")
                if prepare_only and args.engine == "vllm":
                    ib_after_sleep = _collective_rpc(args, "coldsnap_nccl_ib_status")
                    try:
                        prepare = _collective_rpc(args, "checkpoint_prepare")
                    except BaseException as error:
                        prepare_error = _http_error(error)
                    diagnostic = {
                        "format": 1,
                        "kind": "coldsnap-nccl-capture-prepare-only",
                        "rank": args.rank,
                        "identity": identity,
                        "pre_checkpoint_response": before,
                        "disk_sleep": sleep,
                        "disk_sleep_seconds": sleep_seconds,
                        "ib_before_sleep": ib_before_sleep,
                        "ib_after_sleep": ib_after_sleep,
                        "checkpoint_prepare": prepare,
                        "checkpoint_prepare_error": prepare_error,
                    }
                    coordinator.set(diagnostic_key, json.dumps(diagnostic))
                elif args.engine == "vllm":
                    prepare = _checkpoint(args, "prepare")
                    coordinator.set(key, "1")
                else:
                    coordinator.set(key, "1")
            else:
                _coordinator_wait(coordinator, sleep_key, args.timeout)
                if prepare_only and args.engine == "vllm":
                    diagnostic = json.loads(
                        _coordinator_wait(coordinator, diagnostic_key, args.timeout)
                    )
                else:
                    _coordinator_wait(coordinator, key, args.timeout)
            if args.engine == "sglang":
                _prepare_sglang_hydration(args)
            hibernate_states = _hibernate_states(args.artifact_root)
            shape_calibration = _capture_shape_calibration(args, coordinator)
            if args.engine == "sglang":
                ib_before_sleep = _sglang_nccl_observations(hibernate_states)
                prepare = ib_before_sleep
                if prepare_only:
                    diagnostic = {
                        "format": 1,
                        "kind": "coldsnap-nccl-capture-prepare-only",
                        "rank": args.rank,
                        "identity": identity,
                        "pre_checkpoint_response": before,
                        "disk_sleep": sleep,
                        "disk_sleep_seconds": sleep_seconds,
                        "checkpoint_prepare": prepare,
                    }
            if prepare_only:
                diagnostic["rank"] = args.rank
                diagnostic["hibernate_states"] = hibernate_states
                diagnostic["shape_calibration"] = shape_calibration
                diagnostic["controller_seconds"] = time.perf_counter() - started
                base._atomic_json(args.artifact_root / "capture.json", diagnostic)
                base._kill_process_group(target.pid)
                target.wait(timeout=30)
                return diagnostic
            barrier = f"{args.activation_namespace}:capture-ready"
            coordinator.set(f"{barrier}:{args.rank}", "1")
            for rank in range(args.world_size):
                _coordinator_wait(coordinator, f"{barrier}:{rank}", args.timeout)
            tree = base._process_tree(target.pid)
            rss_bytes = sum(base._rss_bytes(pid) for pid in tree)
            regular_backings = _capture_regular_backings(args, tree)
            command = _criu_command(args, "dump", "dump.log")
            command.extend(["--pid", str(target.pid)])
            dump_seconds = base._run_criu(command, _criu_environment(args, job_file))
            criu_rpc = base._criu_rpc_result(args, "dump")
            target.wait(timeout=60)
            shm_count, shm_bytes = base._snapshot_link_remaps(args)
            cuda_job = base._preserve_cuda_job(args, job_file)
        except BaseException:
            if target.poll() is None:
                base._kill_process_group(target.pid)
                target.wait(timeout=30)
            raise
    capture_log = activation_logs.archive_capture_log(args.artifact_root)
    image_bytes = sum(path.stat().st_size for path in images.rglob("*") if path.is_file())
    report = {
        "format": 1,
        "kind": f"coldsnap-{args.engine}-cuda-criu-capture",
        "generation": generation,
        "identity": identity,
        "target_pid": target.pid,
        "pid_floor": pid_floor,
        "process_tree": tree,
        "process_tree_rss_bytes": rss_bytes,
        "regular_backings": regular_backings,
        "pre_checkpoint_response": before,
        "disk_sleep": sleep,
        "disk_sleep_seconds": sleep_seconds,
        "hibernate_states": hibernate_states,
        "shape_calibration": shape_calibration,
        "capture_log": capture_log,
        "checkpoint_prepare": prepare,
        "nccl_prepare_seconds": _checkpoint_max_seconds(prepare, "nccl_prepare_seconds"),
        "nccl_network_reset_seconds": _checkpoint_max_seconds(
            prepare, "nccl_network_reset_seconds"
        ),
        "nccl_ib_quiesce_seconds": _checkpoint_max_seconds(prepare, "ib_quiesce_s"),
        "engine_checkpoint_prepare_seconds": _checkpoint_max_seconds(
            prepare, "engine_checkpoint_prepare_seconds"
        ),
        "nccl_workers_before_sleep": ib_before_sleep,
        "criu_rpc": criu_rpc,
        "cuda_job": cuda_job,
        "dump_seconds": dump_seconds,
        "capture_controller_seconds": time.perf_counter() - started,
        "image_bytes": image_bytes,
        "shared_memory_link_count": shm_count,
        "shared_memory_link_bytes": shm_bytes,
    }
    base._atomic_json(args.artifact_root / "capture.json", report)
    return report


def _publish_barrier_release(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(descriptor, b"1\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_restore_with_network_unlock_barrier(
    args: argparse.Namespace,
    command: list[str],
    environment: dict[str, str],
    coordinator: CoordinatorClient,
) -> tuple[dict[str, Any], float]:
    work = args.artifact_root / "work"
    ready = work / "network-unlock-ready"
    release = work / "network-unlock-release"
    ready.unlink(missing_ok=True)
    release.unlink(missing_ok=True)
    barrier_started = time.perf_counter()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(base._run_criu_profiled, command, environment)
    try:
        deadline = time.monotonic() + args.timeout
        while True:
            try:
                metadata = ready.lstat()
            except FileNotFoundError:
                if future.done():
                    future.result()
                    raise RuntimeError(
                        "CRIU restore completed without a network-unlock notification"
                    ) from None
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out waiting for CRIU network-unlock readiness"
                    ) from None
                time.sleep(0.01)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("CRIU network-unlock readiness is not a regular file")
            break

        prefix = f"{args.activation_namespace}:network-unlock-ready"
        coordinator.set(f"{prefix}:{args.rank}", "1")
        for unit_index in range(args.world_size):
            _coordinator_wait(coordinator, f"{prefix}:{unit_index}", args.timeout)
        barrier_seconds = time.perf_counter() - barrier_started
        _publish_barrier_release(release)
        return future.result(), barrier_seconds
    finally:
        if ready.exists() and not release.exists():
            _publish_barrier_release(release)
        executor.shutdown(wait=True)


def restore(args: argparse.Namespace) -> dict[str, Any]:
    transport_environment = _stage_restore_transport_environment(args)
    runtime_environment = _stage_restore_runtime_environment(args)
    manifests = _hydration_manifests(args.artifact_root)
    for manifest_path in manifests.values():
        _select_weight_provider(manifest_path, args.weight_provider)
    capture_report = base._load_json(args.artifact_root / "capture.json")
    identity = _identity(args)
    captured_identity = capture_report.get("identity")
    identity_compatibility = _identity_compatibility(
        captured_identity,
        identity,
        args.allow_compatible_criu_runtime,
        args.kernel_compatibility,
    )
    args.captured_http_port = _identity_http_port(captured_identity)
    cuda_job = capture_report.get("cuda_job")
    if not isinstance(cuda_job, dict):
        raise RuntimeError("CUDA checkpoint job template is missing")
    job_file = base._reset_cuda_job(args, cuda_job)
    regular_backings = _restore_regular_backings(args, capture_report)
    shm_count = base._prepare_link_remaps(args)
    (args.artifact_root / "restore-rpc.json").unlink(missing_ok=True)
    (args.artifact_root / "cuda-restore-rpc.json").unlink(missing_ok=True)
    ready_path = args.artifact_root / "restore-ready.json"
    ready_path.unlink(missing_ok=True)
    # A deferred-graph marker belongs to one activation, while the captured
    # process carries the reusable generation and eager-step counter.
    (args.artifact_root / "async-graphs-ready.json").unlink(missing_ok=True)
    (args.artifact_root / "async-graphs-arm").unlink(missing_ok=True)
    command = _restore_command(args)
    coordinator = CoordinatorClient.from_path(args.coordinator_path)
    # Every interval below is observed by this fresh controller process. CRIU
    # restores the target's monotonic clock, so a timestamp taken inside the
    # restored process cannot be subtracted against one taken out here.
    started = time.perf_counter()
    timeline: list[dict[str, Any]] = []

    def mark(name: str) -> None:
        timeline.append(
            {
                "name": name,
                "unix_ns": time.time_ns(),
                "since_restore_begin_s": time.perf_counter() - started,
            }
        )

    mark("restore_begin")
    log_boundary = activation_logs.restore_log_boundary(
        artifact_root=args.artifact_root,
        activation_namespace=args.activation_namespace,
        rank=args.rank,
        capture_report=capture_report,
    )
    mark("restore_log_boundary")
    environment = _criu_environment(args, job_file)
    network_unlock_barrier_seconds = None
    if args.defer_network_unlock:
        criu_profile, network_unlock_barrier_seconds = _run_restore_with_network_unlock_barrier(
            args, command, environment, coordinator
        )
        mark("network_unlock_barrier_end")
    else:
        criu_profile = base._run_criu_profiled(command, environment)
    criu_seconds = float(criu_profile["wall_seconds"])
    criu_rpc = base._criu_rpc_result(args, "restore")
    criu_profile["log_profile"] = _criu_log_profile(args.artifact_root / "work/restore.log")
    mark("criu_restore_end")
    restored_pid_value = criu_rpc.get("restored_pid")
    if not isinstance(restored_pid_value, int) or restored_pid_value <= 0:
        raise RuntimeError(f"invalid restored PID in CRIU RPC result: {criu_rpc!r}")
    restored_pid = restored_pid_value
    prefix = args.activation_namespace
    coordinator.set(f"{prefix}:criu-restored:{args.rank}", str(restored_pid))
    coordinator.set(f"{prefix}:criu-seconds:{args.rank}", repr(criu_seconds))
    for rank in range(args.world_size):
        _coordinator_wait(coordinator, f"{prefix}:criu-restored:{rank}", args.timeout)
    mark("rank_barrier_end")
    # Publishing each rank's CRIU duration lets a single report separate this
    # rank's own restore from time spent waiting on a slower peer.
    peer_criu_seconds = {
        str(rank): float(
            _coordinator_wait(coordinator, f"{prefix}:criu-seconds:{rank}", args.timeout)
        )
        for rank in range(args.world_size)
    }
    barrier_wait_s = max(peer_criu_seconds.values()) - criu_seconds
    cuda_restore_rpc = None
    cuda_restore_seconds = None
    peer_cuda_restore_seconds = None
    cuda_barrier_wait_s = None
    if args.leave_stopped:
        mark("deferred_cuda_restore_begin")
        command = _deferred_cuda_restore_command(args, restored_pid)
        cuda_restore_seconds = base._run_criu(command, _criu_environment(args, job_file))
        cuda_restore_rpc = base._criu_rpc_result(args, "cuda-restore")
        mark("deferred_cuda_restore_end")
        cuda_barrier = f"{prefix}:cuda-restored"
        coordinator.set(f"{cuda_barrier}:{args.rank}", repr(cuda_restore_seconds))
        peer_cuda_restore_seconds = {
            str(rank): float(_coordinator_wait(coordinator, f"{cuda_barrier}:{rank}", args.timeout))
            for rank in range(args.world_size)
        }
        cuda_barrier_wait_s = max(peer_cuda_restore_seconds.values()) - cuda_restore_seconds
        mark("distributed_cuda_restore_barrier_end")
        if _release_sglang_cuda_restore_hold(args):
            hold_barrier = f"{prefix}:sglang-cuda-hold-released"
            coordinator.set(f"{hold_barrier}:{args.rank}", "1")
            for rank in range(args.world_size):
                _coordinator_wait(coordinator, f"{hold_barrier}:{rank}", args.timeout)
            mark("distributed_sglang_cuda_hold_release_end")
        os.killpg(restored_pid, signal.SIGCONT)
        resumed = f"{prefix}:process-resumed"
        coordinator.set(f"{resumed}:{args.rank}", "1")
        for rank in range(args.world_size):
            _coordinator_wait(coordinator, f"{resumed}:{rank}", args.timeout)
        mark("distributed_process_resume_end")

    checkpoint_restore = None
    checkpoint_restore_seconds = None
    wake_up = None
    wake_up_seconds = None
    health_seconds = None
    post_restore_response = None
    retained_graph_fallback = None
    if (
        args.engine == "sglang"
        and args.activation_state == "running"
        and os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS", "0") == "1"
    ):
        mark("async_graph_arm_begin")
    if _arm_sglang_async_graphs(args, coordinator, prefix):
        mark("async_graph_arm_end")
        mark("async_graphs_armed")
    in_place_activation = _stage_in_place_nccl_activation(args)
    if in_place_activation is not None:
        mark("nccl_in_place_activation_ready")
    if args.rank == 0:
        if args.engine == "vllm":
            mark("checkpoint_restore_begin")
            phase = time.perf_counter()
            checkpoint_restore = _checkpoint(args, "restore")
            checkpoint_restore_seconds = time.perf_counter() - phase
            mark("checkpoint_restore_end")
        if args.activation_state == "running":
            mark("wake_up_begin")
            phase = time.perf_counter()
            if args.engine == "vllm":
                wake_up = _request(args, "POST", "/wake_up")
            else:
                if args.weight_provider == "recovery":
                    weights_wake = _request(
                        args,
                        "POST",
                        "/resume_memory_occupation",
                        _sglang_memory_payload("recovery", "weights"),
                    )
                    runtime_wake = _request(
                        args,
                        "POST",
                        "/resume_memory_occupation",
                        _sglang_memory_payload("recovery", "runtime"),
                    )
                    wake_up = {
                        "weights": weights_wake,
                        "runtime": runtime_wake,
                    }
                else:
                    wake_up = _request(
                        args,
                        "POST",
                        "/resume_memory_occupation",
                        _sglang_memory_payload(args.weight_provider),
                    )
                checkpoint_restore = {
                    "integrated_with": "resume_memory_occupation",
                    "response": wake_up,
                }
            wake_up_seconds = time.perf_counter() - phase
            mark("wake_up_end")
            mark("health_wait_begin")
            phase = time.perf_counter()
            _wait_http(args, "/health")
            health_seconds = time.perf_counter() - phase
            mark("health_ready")
            try:
                post_restore_response = _infer(args)
            except BaseException as retained_error:
                if not _retained_nccl_graphs():
                    raise
                mark("retained_graph_validation_failed")
                reason = (
                    "retained graph acceptance failed: "
                    f"{type(retained_error).__name__}: {retained_error}"
                )
                if args.engine == "vllm":
                    fallback_response = _collective_rpc(
                        args, "coldsnap_prepare_cuda_graph_retry"
                    )
                else:
                    graph_payload = {"tags": ["cuda_graph"]}
                    fallback_response = {
                        "release": _request(
                            args,
                            "POST",
                            "/release_memory_occupation",
                            graph_payload,
                        ),
                        "resume": _request(
                            args,
                            "POST",
                            "/resume_memory_occupation",
                            graph_payload,
                        ),
                    }
                arm_path = os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE")
                if not arm_path:
                    raise RuntimeError(
                        reason + "; async graph fallback has no arm file"
                    ) from retained_error
                Path(arm_path).touch()
                mark("retained_graph_fallback_armed")
                post_restore_response = _infer(args)
                retained_graph_fallback = {
                    "reason": reason,
                    "worker_status": fallback_response,
                    "result": "eager-request-succeeded",
                }
                mark("retained_graph_fallback_succeeded")
            mark("acceptance_ready")
        else:
            if args.engine == "vllm":
                sleeping = _request(args, "GET", "/is_sleeping")
                if not isinstance(sleeping, dict) or sleeping.get("is_sleeping") is not True:
                    raise RuntimeError(f"warm restore reached a non-sleeping backend: {sleeping!r}")
            mark("warm_hold_ready")

    if args.rank == 0:
        coordinator.set(f"{prefix}:service-ready", "1")
        coordinator.set(f"{prefix}:service-state", args.activation_state)
    else:
        _coordinator_wait(coordinator, f"{prefix}:service-ready", args.timeout)
        mark("service_ready_observed")
    if (
        args.engine == "vllm"
        and args.activation_state == "running"
        and os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS", "0") == "1"
        and not _retained_nccl_graphs()
    ):
        mark("async_graph_arm_begin")
        graph_key = f"{prefix}:arm-async-graphs"
        if args.rank == 0:
            coordinator.set(graph_key, "1")
        else:
            _coordinator_wait(coordinator, graph_key, args.timeout)
        arm_path = os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE")
        if not arm_path:
            raise RuntimeError("async graph activation has no arm file")
        Path(arm_path).touch()
        mark("async_graph_arm_end")
        mark("async_graphs_armed")
    # Each restored worker writes its own artifact-local state. Read it only
    # after rank 0 has observed the distributed service as healthy, so rank 1's
    # resume/native/discard critical path is present in the activation report.
    hibernate_states = _hibernate_states(args.artifact_root)
    mark("hibernate_state_loaded")

    report = {
        "format": 2,
        "kind": f"coldsnap-{args.engine}-cuda-criu-restore-ready",
        "activation_namespace": args.activation_namespace,
        "activation_state": args.activation_state,
        "identity_compatibility": identity_compatibility,
        "rank": args.rank,
        "world_size": args.world_size,
        "restored_pid": restored_pid,
        "criu_restore_seconds": criu_seconds,
        "criu_restore_profile": criu_profile,
        "criu_rpc": criu_rpc,
        "network_unlock_barrier_seconds": network_unlock_barrier_seconds,
        "peer_criu_restore_seconds": peer_criu_seconds,
        "rank_barrier_wait_seconds": barrier_wait_s,
        "deferred_cuda_restore": cuda_restore_rpc,
        "deferred_cuda_restore_seconds": cuda_restore_seconds,
        "peer_cuda_restore_seconds": peer_cuda_restore_seconds,
        "cuda_restore_barrier_wait_seconds": cuda_barrier_wait_s,
        "checkpoint_restore": checkpoint_restore,
        "checkpoint_restore_seconds": checkpoint_restore_seconds,
        "nccl_restore_seconds": _checkpoint_max_seconds(checkpoint_restore, "nccl_restore_seconds"),
        "engine_checkpoint_restore_seconds": _checkpoint_max_seconds(
            checkpoint_restore, "engine_checkpoint_restore_seconds"
        ),
        "wake_up": wake_up,
        "wake_up_seconds": wake_up_seconds,
        "health_wait_seconds": health_seconds,
        "post_restore_response": post_restore_response,
        "retained_graph_fallback": retained_graph_fallback,
        "hibernate_states": hibernate_states,
        "restore_controller_seconds": time.perf_counter() - started,
        "shared_memory_link_count": shm_count,
        "regular_backings": regular_backings,
        "log_boundary": log_boundary,
        "transport_environment": (
            str(transport_environment) if transport_environment is not None else None
        ),
        "runtime_environment": (
            str(runtime_environment) if runtime_environment is not None else None
        ),
        "timeline": timeline,
    }
    # Every rank publishes its own report so no phase is attributable only to
    # rank 0. Rank 0 additionally writes the aggregate name the controller polls.
    unit = os.environ.get("COLDSNAP_EXPECTED_UNIT", "")
    if not unit:
        raise RuntimeError("COLDSNAP_EXPECTED_UNIT is required")
    base._atomic_json(args.artifact_root / f"restore-ready-unit-{unit}.json", report)
    if args.rank == 0:
        base._atomic_json(ready_path, report)
    while True:
        time.sleep(60)


def main() -> int:
    args = _parser().parse_args()
    if args.world_size <= 0 or args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("rank must be within a positive world size")
    if args.worker_count <= 0:
        raise ValueError("worker count must be positive")
    if args.timeout <= 0 or args.timeout > 7200:
        raise ValueError("timeout must be in (0, 7200]")
    if (
        args.criu_compress_block_bytes < 0
        or args.criu_compress_block_bytes > 4 * 1024**2
        or (
            args.criu_compress_block_bytes
            and args.criu_compress_block_bytes % os.sysconf("SC_PAGE_SIZE") != 0
        )
    ):
        raise ValueError("CRIU compression block must be zero or a page multiple up to 4 MiB")
    if not 1 <= args.criu_compress_acceleration <= 65537:
        raise ValueError("CRIU compression acceleration must be in [1, 65537]")
    if not 0 <= args.criu_decompress_threads <= 1024:
        raise ValueError("CRIU decompression threads must be in [0, 1024]")
    if not 0 <= args.tcp_port_shift <= 64511:
        raise ValueError("TCP port shift must be in [0, 64511]")
    if args.tcp_port_shift and not args.tcp_address_map:
        raise ValueError("TCP port shift requires at least one TCP address mapping")
    if args.defer_network_unlock and (not args.leave_stopped or not args.tcp_address_map):
        raise ValueError("deferred network unlock requires leave-stopped and TCP address mapping")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        with service_runtime.StartupReadiness(args):
            report = capture(args) if args.mode == "capture" else restore(args)
    except BaseException as error:
        report = {
            "format": 1,
            "kind": f"coldsnap-{args.engine}-cuda-criu-error",
            "rank": args.rank,
            "error": f"{type(error).__name__}: {error}",
        }
        base._atomic_json(args.artifact_root / "controller-error.json", report)
        try:
            CoordinatorClient.from_path(args.coordinator_path).set(
                f"{args.activation_namespace}:service-state",
                json.dumps(report, sort_keys=True),
            )
        except BaseException:
            # Preserve the original controller failure. The local artifact is
            # still authoritative if the coordinator itself is unavailable.
            pass
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

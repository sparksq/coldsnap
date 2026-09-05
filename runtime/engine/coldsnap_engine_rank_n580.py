#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Capture or restore one rank of a context-free distributed inference template."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import coldsnap_cuda_criu as base  # noqa: E402
import coldsnap_activation_logs as activation_logs  # noqa: E402
import coldsnap_n580_criu as precuda  # noqa: E402
import coldsnap_service_runtime as service_runtime  # noqa: E402
from coldsnap_coord import Client as CoordinatorClient  # noqa: E402


CONTROLLER_ABI = 2
PROCESS_TEMPLATE_PLACEMENT_ABI = 2
MODEL_PAYLOAD_STAGE_ROOT = Path("/var/cache/coldsnap/model-payloads")
ACTIVATION_DIRECTORY_PREFIX = "activation-directory-"
PRE_EXEC_PHASE = "pre_exec"
PRE_WORKER_IMPORT_PHASE = "pre_worker_import"
QUALIFICATION_PHASE_ENV = "COLDSNAP_N580_QUALIFICATION_PROCESS_TEMPLATE_PHASE"
QUALIFICATION_DRIVER_LIBRARY_ENV = "COLDSNAP_N580_QUALIFICATION_ALLOW_CAPTURED_DRIVER_LIBRARIES"
PORTABLE_WORKER_REEXEC_ENV = "COLDSNAP_N580_PORTABLE_WORKER_REEXEC"
QUALIFICATION_DRIVER_LIBRARY_ROOT = "qualification-driver-libraries"
QUALIFICATION_DRIVER_LIBRARY_MAX_FILES = 32
QUALIFICATION_DRIVER_LIBRARY_MAX_BYTES = 1024**3


class _RestoredProcess:
    """Minimal Popen-compatible liveness probe for a CRIU-restored root."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        try:
            stat_line = Path(f"/proc/{self.pid}/stat").read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, ProcessLookupError):
            self.returncode = -1
            return self.returncode
        closing = stat_line.rfind(")")
        if closing < 0 or closing + 2 >= len(stat_line):
            self.returncode = -1
            return self.returncode
        if stat_line[closing + 2] == "Z":
            self.returncode = -1
            return self.returncode
        return None


def _require_process_template_placement_abi(capture_report: dict[str, Any]) -> None:
    if capture_report.get("process_template_placement_abi") != PROCESS_TEMPLATE_PLACEMENT_ABI:
        raise RuntimeError(
            "n580 process-template placement ABI is unsupported; recapture it "
            "with a current ColdSnap release"
        )


def _qualification_enabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise RuntimeError(f"{name} must be a boolean value")


def _process_template_phase(args: argparse.Namespace) -> str:
    phase = os.environ.get(QUALIFICATION_PHASE_ENV, "").strip() or PRE_EXEC_PHASE
    if phase == PRE_EXEC_PHASE:
        return phase
    if phase == PRE_WORKER_IMPORT_PHASE and args.engine == "vllm":
        return phase
    raise RuntimeError(
        f"{QUALIFICATION_PHASE_ENV} only permits {PRE_EXEC_PHASE!r}, or "
        f"{PRE_WORKER_IMPORT_PHASE!r} for vLLM qualification"
    )


def _capture_process_template_phase(capture_report: dict[str, Any]) -> str:
    phase = capture_report.get("process_template_phase", PRE_EXEC_PHASE)
    if phase not in {PRE_EXEC_PHASE, PRE_WORKER_IMPORT_PHASE}:
        raise RuntimeError(f"n580 process-template phase is invalid: {phase!r}")
    if (
        phase == PRE_WORKER_IMPORT_PHASE
        and capture_report.get("captured_driver_libraries_qualified") is not True
    ):
        raise RuntimeError(
            "n580 pre-worker-import artifact lacks the required capture-time "
            "driver-library qualification evidence"
        )
    return str(phase)


def _arm_sglang_async_graphs(
    args: argparse.Namespace,
    coordinator: CoordinatorClient,
    prefix: str,
) -> bool:
    """Arm post-restore SGLang graphs without coupling driver modules.

    n580 restores a deliberately old, context-free process template and then
    executes the current engine runtime.  Keep this driver-owned activation
    step local instead of requiring the capsule's n610 controller module to
    expose the same private helper.
    """
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
        service_runtime._coordinator_wait(
            coordinator,
            f"{key}:{rank}",
            args.timeout,
        )
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "restore"))
    parser.add_argument("--engine", choices=("vllm", "sglang"), default="vllm")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--generation", required=True)
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
    parser.add_argument("--cuda-checkpoint", type=Path, required=True)
    parser.add_argument("--native-hydration-library", type=Path, required=True)
    parser.add_argument("--criu-plugin-dir", type=Path, required=True)
    parser.add_argument("--nvml-dlopen-shim", type=Path, required=True)
    parser.add_argument("--runtime-lib-dir", type=Path, required=True)
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
    parser.add_argument(
        "--criu-compress-block-bytes",
        dest="compress_block_bytes",
        type=int,
        default=256 * 1024,
    )
    parser.add_argument(
        "--criu-compress-acceleration",
        dest="compress_acceleration",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--criu-decompress-threads",
        dest="decompress_threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--criu-image-io-mode",
        dest="image_io_mode",
        choices=("writeback", "direct"),
        default="direct",
    )
    parser.add_argument("--activation-state", choices=("running", "warm"), default="running")
    parser.add_argument(
        "--kernel-compatibility", choices=("capability", "exact"), default="capability"
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--allow-compatible-criu-runtime", action="store_true")
    parser.add_argument("--tcp-address-map", action="append", default=[])
    parser.add_argument("--tcp-port-shift", type=int, default=0)
    parser.add_argument("--defer-network-unlock", action="store_true")
    parser.add_argument("serve_command", nargs=argparse.REMAINDER)
    return parser


def _sha256(path: Path) -> str:
    with path.open("rb", buffering=0) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _serve_command(args: argparse.Namespace) -> list[str]:
    command = args.serve_command[1:] if args.serve_command[:1] == ["--"] else args.serve_command
    if not command:
        raise ValueError(f"an {args.engine} serve command is required")
    return command


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    if not args.target_launcher.resolve().is_file():
        raise FileNotFoundError(
            f"pre-CUDA TP2 runtime is incomplete: target_launcher={args.target_launcher.resolve()}"
        )
    paths = {
        "criu": args.criu.resolve(),
        "criu_rpc": args.criu_rpc.resolve(),
        "criu_nvidia_reset_plugin": (
            args.criu_plugin_dir / "libcoldsnap_criu_nvidia_reset.so"
        ).resolve(),
        "nvml_dlopen_shim": args.nvml_dlopen_shim.resolve(),
        "criu_lz4": (args.runtime_lib_dir / "liblz4.so.1").resolve(),
    }
    distribution = "coldsnap_vllm" if args.engine == "vllm" else "coldsnap_sglang"
    entrypoints = sorted(args.plugin_root.glob(f"{distribution}-*.dist-info/entry_points.txt"))
    if len(entrypoints) != 1:
        raise FileNotFoundError(f"plugin root must contain one {args.engine} entrypoint")
    paths["plugin_entrypoint"] = entrypoints[0].resolve()
    for path in sorted(args.plugin_root.glob("*.py")):
        paths[f"plugin/{path.name}"] = path.resolve()
    for path in sorted((args.plugin_root / "coldsnap_sglang").glob("*.py")):
        paths[f"plugin/coldsnap_sglang/{path.name}"] = path.resolve()
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("pre-CUDA TP2 runtime is incomplete: " + ", ".join(missing))
    return {
        "format": 1,
        "kind": f"coldsnap-{args.engine}-n580-identity",
        "controller_abi": CONTROLLER_ABI,
        "engine": args.engine,
        "rank": args.rank,
        "world_size": args.world_size,
        "worker_count": args.worker_count,
        "image_id": args.image_id,
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "unit": os.environ.get("COLDSNAP_EXPECTED_UNIT", ""),
        "cuda_userspace": service_runtime._torch_cuda_userspace_version(),
        "gpus": service_runtime._visible_gpu_facts(),
        "model": args.model,
        "model_revision": args.model_revision,
        "serve_command": _serve_command(args),
        "runtime_sha256": {name: _sha256(path) for name, path in paths.items()},
    }


def _changed_tcp_placement(mappings: list[str]) -> bool:
    for mapping in mappings:
        before, separator, after = mapping.partition("=")
        if not separator or not before or not after:
            raise RuntimeError(f"invalid TCP address mapping: {mapping!r}")
        if before != after:
            return True
    return False


def _write_process_template_placement(
    args: argparse.Namespace,
) -> bool:
    _changed_tcp_placement(args.tcp_address_map)
    address_map: dict[str, str] = {}
    seen_addresses: set[str] = set()
    for mapping in args.tcp_address_map:
        before, _, after = mapping.partition("=")
        if before in seen_addresses:
            raise RuntimeError(f"duplicate TCP address mapping: {before!r}")
        seen_addresses.add(before)
        address_map[before] = after
    capture_report = base._load_json(args.artifact_root / "capture.json")
    _require_process_template_placement_abi(capture_report)
    process_template_phase = _capture_process_template_phase(capture_report)
    path = args.artifact_root / "restore-placement.json"
    path.unlink(missing_ok=True)
    master_port = service_runtime._command_master_port(_serve_command(args))
    placement: dict[str, object] = {
        "format": 1,
        "kind": "coldsnap-process-template-placement",
        "generation": args.generation,
        "master_address": args.master_address,
        "master_port": service_runtime._shift_tcp_port(master_port, args.tcp_port_shift),
        "http_port": args.http_port,
    }
    if process_template_phase == PRE_WORKER_IMPORT_PHASE:
        # The vLLM pre-worker hook must also retarget its serialized, lazy
        # scheduler queue before the worker connects. A pre-exec launcher has
        # no engine-owned queue yet and deliberately consumes the smaller
        # placement contract.
        placement.update(
            {
                "tcp_address_map": address_map,
                "tcp_port_shift": args.tcp_port_shift,
            }
        )
    base._atomic_json(
        path,
        placement,
    )
    return True


def _validate_nvidia_driver_library_portability(
    captured_identity: object,
    current_identity: object,
    process_template_phase: str = PRE_EXEC_PHASE,
    allow_captured_driver_libraries: bool = False,
    portable_worker_reexec: bool = False,
) -> str:
    if not isinstance(captured_identity, dict) or not isinstance(current_identity, dict):
        raise RuntimeError("n580 process artifact identity is invalid")
    captured_gpus = captured_identity.get("gpus")
    current_gpus = current_identity.get("gpus")
    if not isinstance(captured_gpus, list) or not isinstance(current_gpus, list):
        raise RuntimeError("n580 process artifact GPU identity is invalid")
    captured_drivers = {gpu.get("driver") for gpu in captured_gpus if isinstance(gpu, dict)}
    current_drivers = {gpu.get("driver") for gpu in current_gpus if isinstance(gpu, dict)}
    if len(captured_drivers) != 1 or len(current_drivers) != 1:
        raise RuntimeError("n580 requires one NVIDIA driver version per restore unit")
    captured_driver = captured_drivers.pop()
    current_driver = current_drivers.pop()
    if not isinstance(captured_driver, str) or not isinstance(current_driver, str):
        raise RuntimeError("n580 NVIDIA driver identity is invalid")
    if process_template_phase == PRE_EXEC_PHASE:
        return "pre-exec-current-driver"
    if process_template_phase != PRE_WORKER_IMPORT_PHASE:
        raise RuntimeError(f"n580 process-template phase is invalid: {process_template_phase!r}")
    if captured_driver == current_driver:
        return (
            "pre-worker-import-portable-worker-reexec"
            if portable_worker_reexec
            else "pre-worker-import-exact-driver"
        )
    if portable_worker_reexec:
        return "pre-worker-import-portable-worker-reexec"
    if not allow_captured_driver_libraries:
        raise RuntimeError(
            "n580 pre-worker-import artifact contains capture-driver CUDA/NVML "
            f"libraries ({captured_driver}); restoring with {current_driver} requires "
            f"the explicit {QUALIFICATION_DRIVER_LIBRARY_ENV}=1 qualification override"
        )
    return "pre-worker-import-forced-cross-driver"


def _driver_library_paths(tree_audit: list[dict[str, Any]]) -> list[Path]:
    paths: set[Path] = set()
    for process in tree_audit:
        mapped = process.get("nvidia_driver_library_maps")
        if not isinstance(mapped, list):
            raise RuntimeError("n580 driver-library audit inventory is invalid")
        for value in mapped:
            if not isinstance(value, str):
                raise RuntimeError("n580 driver-library audit path is invalid")
            path = Path(value)
            if not path.is_absolute() or not any(
                path.name == name or path.name.startswith(f"{name}.")
                for name in ("libcuda.so", "libnvidia-ml.so")
            ):
                raise RuntimeError(f"n580 driver-library audit path is unsafe: {value!r}")
            paths.add(path)
    if len(paths) > QUALIFICATION_DRIVER_LIBRARY_MAX_FILES:
        raise RuntimeError("n580 qualification driver-library inventory is too large")
    return sorted(paths)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_qualification_driver_libraries(
    artifact_root: Path,
    tree_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for path in _driver_library_paths(tree_audit):
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"n580 qualification driver library is not regular: {path}")
        total_bytes += metadata.st_size
        if total_bytes > QUALIFICATION_DRIVER_LIBRARY_MAX_BYTES:
            raise RuntimeError("n580 qualification driver libraries exceed the size limit")
        relative = Path(QUALIFICATION_DRIVER_LIBRARY_ROOT) / path.relative_to("/")
        destination = artifact_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        os.chmod(destination, stat.S_IMODE(metadata.st_mode))
        records.append(
            {
                "path": str(path),
                "artifact_path": relative.as_posix(),
                "bytes": metadata.st_size,
                "mode": stat.S_IMODE(metadata.st_mode),
                "sha256": _file_sha256(destination),
            }
        )
    if not records:
        raise RuntimeError("n580 qualification found no capture-driver library mappings")
    return records


def _restore_qualification_driver_libraries(
    artifact_root: Path,
    records: object,
    *,
    destination_root: Path = Path("/"),
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise RuntimeError("n580 qualification artifact lacks captured driver libraries")
    restored: list[dict[str, Any]] = []
    total_bytes = 0
    if len(records) > QUALIFICATION_DRIVER_LIBRARY_MAX_FILES:
        raise RuntimeError("n580 qualification driver-library inventory is too large")
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("n580 qualification driver-library record is invalid")
        path_value = record.get("path")
        artifact_value = record.get("artifact_path")
        byte_count = record.get("bytes")
        mode = record.get("mode")
        sha256 = record.get("sha256")
        if (
            not isinstance(path_value, str)
            or not isinstance(artifact_value, str)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(mode, int)
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise RuntimeError("n580 qualification driver-library record is invalid")
        original = Path(path_value)
        allowed_paths = _driver_library_paths(
            [{"nvidia_driver_library_maps": [path_value]}]
        )
        expected_artifact = Path(QUALIFICATION_DRIVER_LIBRARY_ROOT) / original.relative_to("/")
        if (
            original not in allowed_paths
            or Path(artifact_value) != expected_artifact
        ):
            raise RuntimeError("n580 qualification driver-library path is invalid")
        total_bytes += byte_count
        if total_bytes > QUALIFICATION_DRIVER_LIBRARY_MAX_BYTES:
            raise RuntimeError("n580 qualification driver libraries exceed the size limit")
        source = artifact_root / expected_artifact
        source_metadata = source.stat()
        if (
            not stat.S_ISREG(source_metadata.st_mode)
            or source_metadata.st_size != byte_count
            or _file_sha256(source) != sha256
        ):
            raise RuntimeError(f"n580 qualification driver library is corrupt: {source}")
        destination = destination_root / original.relative_to("/")
        action = "resident"
        if destination.exists():
            destination_metadata = destination.stat()
            if (
                not stat.S_ISREG(destination_metadata.st_mode)
                or destination_metadata.st_size != byte_count
                or _file_sha256(destination) != sha256
            ):
                raise RuntimeError(
                    f"n580 qualification refuses to replace a different driver library: {destination}"
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.coldsnap-{uuid.uuid4().hex}")
            try:
                shutil.copyfile(source, temporary)
                os.chmod(temporary, mode)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            action = "materialized"
        restored.append({**record, "action": action})
    return restored


def _numeric_version(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{field} is invalid")
    try:
        return tuple(int(item) for item in value.split("."))
    except ValueError as error:
        raise RuntimeError(f"{field} is invalid") from error


def _identity_compatibility(
    captured: object,
    current: object,
    kernel_compatibility: str = "capability",
    allow_compatible_criu_runtime: bool = False,
) -> str:
    if not isinstance(captured, dict) or not isinstance(current, dict):
        raise RuntimeError("n580 process artifact identity is invalid")
    captured_value = json.loads(json.dumps(captured))
    current_value = json.loads(json.dumps(current))
    captured_gpus = captured_value.get("gpus")
    current_gpus = current_value.get("gpus")
    if (
        not isinstance(captured_gpus, list)
        or not isinstance(current_gpus, list)
        or len(captured_gpus) != len(current_gpus)
    ):
        raise RuntimeError("n580 process artifact GPU identity is invalid")
    for slot, (captured_gpu, current_gpu) in enumerate(
        zip(captured_gpus, current_gpus, strict=True)
    ):
        if not isinstance(captured_gpu, dict) or not isinstance(current_gpu, dict):
            raise RuntimeError("n580 process artifact GPU identity is invalid")
        if captured_gpu.get("name") != current_gpu.get("name") or captured_gpu.get(
            "compute_capability"
        ) != current_gpu.get("compute_capability"):
            raise RuntimeError(f"n580 GPU identity differs for slot {slot}")
        if _numeric_version(
            current_gpu.get("driver"), f"current NVIDIA driver for slot {slot}"
        ) < _numeric_version(captured_gpu.get("driver"), f"captured NVIDIA driver for slot {slot}"):
            raise RuntimeError("current NVIDIA driver is older than the n580 capture")
        current_gpu["driver"] = captured_gpu["driver"]
    captured_value["serve_command"] = service_runtime._normalize_placement_command(
        captured_value.get("serve_command")
    )
    current_value["serve_command"] = service_runtime._normalize_placement_command(
        current_value.get("serve_command")
    )
    # The target launcher is supplied by the content-addressed activation
    # runtime, whose format and snapshot-driver ABI are checked before a rank
    # is started. It is therefore patchable controller state, not immutable
    # capsule state. Normalize the field for format-9 captures which recorded
    # it before that boundary was made explicit.
    for identity in (captured_value, current_value):
        runtime = identity.get("runtime_sha256")
        if not isinstance(runtime, dict):
            raise RuntimeError("n580 process artifact runtime identity is invalid")
        runtime.pop("target_launcher", None)
    if allow_compatible_criu_runtime:
        for identity in (captured_value, current_value):
            runtime = identity.get("runtime_sha256")
            assert isinstance(runtime, dict)
            runtime.pop("criu_rpc", None)
    kernel_differs = captured_value.get("kernel") != current_value.get("kernel")
    if kernel_compatibility == "capability":
        current_value["kernel"] = captured_value.get("kernel")
    elif kernel_compatibility != "exact":
        raise RuntimeError(f"unsupported kernel compatibility policy: {kernel_compatibility!r}")
    if captured_value != current_value:
        paths = service_runtime._identity_mismatch_paths(captured_value, current_value)
        detail = ", ".join(paths[:8]) if paths else "unknown fields"
        raise RuntimeError(f"n580 process artifact identity mismatch: {detail}")
    if kernel_differs and kernel_compatibility == "capability":
        compatibility = "n580-capability-kernel-v2"
    else:
        compatibility = "n580-placement-independent-v2"
    if allow_compatible_criu_runtime:
        compatibility += "+criu-rpc-overlay"
    return compatibility


def _child_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    active = service_runtime._load_active_nccl_runtime(args.nccl_active_runtime)
    preload_order = [*active["preload_order"], str(args.nvml_dlopen_shim)]
    environment["LD_PRELOAD"] = ":".join(dict.fromkeys(preload_order))
    environment["COLDSNAP_NCCL_CHECKPOINT_SHIM_PATH"] = str(active["_resolved"]["checkpoint-shim"])
    environment["COLDSNAP_NCCL_PROVIDER_ID"] = active["provider_id"]
    environment["COLDSNAP_NCCL_PROVIDER_REVISION"] = str(active["provider_revision"])
    environment["COLDSNAP_NCCL_DLSYM_BRIDGE_ABI"] = str(active["bridge"]["abi"])
    plugin = str(args.plugin_root)
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{plugin}:{current_pythonpath}" if current_pythonpath else plugin
    environment.update(
        {
            "COLDSNAP_ENGINE": args.engine,
            "COLDSNAP_MODEL_LOAD_GENERATION": args.generation,
            "COLDSNAP_MODEL_LOAD_READY_DIR": str(args.artifact_root / "ready"),
            "COLDSNAP_MODEL_LOAD_RELEASE_FILE": str(args.artifact_root / "release"),
            "COLDSNAP_MODEL_LOAD_TIMEOUT_SECONDS": str(args.timeout),
            "COLDSNAP_PROCESS_TEMPLATE_PHASE": _process_template_phase(args),
            "COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE": str(
                args.artifact_root / "restore-generation"
            ),
            "COLDSNAP_PROCESS_TEMPLATE_RESTORE_WATCHER_SCOPE": "parent",
            "COLDSNAP_PROCESS_TEMPLATE_RESTORE_LOAD_FORMAT": "coldsnap",
        }
    )
    if args.engine == "vllm":
        environment["VLLM_PLUGINS"] = "coldsnap"
    else:
        environment["SGLANG_PLUGINS"] = "coldsnap"
    return environment


def _child_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(args.target_launcher),
        "--",
        *_serve_command(args),
    ]


def _criu_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    current = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{args.runtime_lib_dir}:{current}" if current else str(args.runtime_lib_dir)
    )
    return environment


def _criu_command(
    args: argparse.Namespace, action: str, runtime_root: Path | None = None
) -> list[str]:
    root = args.artifact_root if runtime_root is None else runtime_root
    command = [
        str(args.criu_rpc),
        "--action",
        action,
        "--criu",
        str(args.criu),
        "--cpu-only",
        "--criu-plugin-dir",
        str(args.criu_plugin_dir),
        "--images-dir",
        str(root / "images"),
        "--work-dir",
        str(root / "work"),
        "--log-file",
        f"{action}.log",
        "--external-files",
        str(root / "external-files.json"),
        "--result",
        str(root / f"{action}-rpc.json"),
        "--timeout",
        str(min(3600, max(1, int(args.timeout)))),
        "--tcp-established",
        "--ghost-limit",
        str(args.ghost_limit),
        "--decompress-threads",
        str(args.decompress_threads),
        "--image-io-mode",
        args.image_io_mode,
    ]
    if action == "dump":
        command.extend(
            [
                "--network-lock",
                "nftables",
                "--leave-running",
                "--link-remap-source-dir",
                str(args.shared_memory_dir),
                "--link-remap-snapshot-dir",
                str(args.artifact_root / "shm-image"),
            ]
        )
        if args.compress_block_bytes:
            command.extend(
                [
                    "--compress-block-size",
                    str(args.compress_block_bytes),
                    "--compress-acceleration",
                    str(args.compress_acceleration),
                ]
            )
    elif action == "restore":
        command.append("--nvidia-placeholder-fds")
        for mapping in args.tcp_address_map:
            command.extend(["--tcp-address-map", mapping])
        if args.tcp_address_map:
            command.append("--tcp-allow-empty-map")
        captured_http_port = getattr(args, "captured_http_port", args.http_port)
        if captured_http_port != args.http_port:
            command.extend(["--tcp-port-map", f"{captured_http_port}={args.http_port}"])
        if args.tcp_port_shift:
            command.extend(["--tcp-port-shift", str(args.tcp_port_shift)])
            if captured_http_port == args.http_port:
                command.extend(["--tcp-preserve-port", str(args.http_port)])
        if args.defer_network_unlock:
            command.extend(["--leave-stopped", "--defer-network-unlock"])
    else:
        raise ValueError(f"unsupported CRIU action: {action}")
    return command


def _primary_context_active() -> bool | None:
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    state = cuda.cuDevicePrimaryCtxGetState
    state.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
    state.restype = ctypes.c_int
    flags = ctypes.c_uint()
    active = ctypes.c_int()
    code = int(state(0, ctypes.byref(flags), ctypes.byref(active)))
    if code == 0:
        return bool(active.value)
    if code == 3:  # CUDA_ERROR_NOT_INITIALIZED
        return False
    return None


def _validate_rank_readiness(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    required = {
        "generation": args.generation,
        "phase": _process_template_phase(args),
        "cuda_initialized": False,
        "cuda_driver_context_present": False,
        "cuda_primary_context_active": False,
        "criu_cpu_only_candidate": True,
    }
    failures = {
        name: {"actual": payload.get(name), "expected": expected}
        for name, expected in required.items()
        if payload.get(name) != expected
    }
    if failures:
        raise RuntimeError(f"worker is not context-free: {failures}")
    rank = payload.get("rank")
    if not isinstance(rank, int) or rank < 0:
        raise RuntimeError(f"worker has invalid global rank: {rank!r}")
    if not _model_identity_matches(
        payload.get("model"), payload.get("revision"), args.model, args.model_revision
    ):
        raise RuntimeError("worker model identity differs from the capture request")


def _model_identity_matches(
    resolved_model: object,
    resolved_revision: object,
    requested_model: str,
    requested_revision: str,
) -> bool:
    """Accept an exact model ID or its pinned Hugging Face snapshot path."""
    if resolved_revision != requested_revision or not isinstance(resolved_model, str):
        return False
    if resolved_model == requested_model:
        return True
    if requested_model.count("/") != 1:
        return False
    snapshot = Path(resolved_model)
    expected_repository = "models--" + requested_model.replace("/", "--")
    return (
        snapshot.name == requested_revision
        and snapshot.parent.name == "snapshots"
        and snapshot.parent.parent.name == expected_repository
    )


def _wait_unit_readiness(
    args: argparse.Namespace, process: subprocess.Popen[bytes]
) -> list[dict[str, Any]]:
    ready_dir = args.artifact_root / "ready"
    expected = args.worker_count if _process_template_phase(args) == PRE_WORKER_IMPORT_PHASE else 1
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{args.engine} exited with {process.returncode} before n580 template readiness"
            )
        paths = sorted(ready_dir.glob("rank-*.json"))
        if len(paths) > expected:
            raise RuntimeError("n580 process tree published too many worker readiness records")
        if len(paths) == expected:
            records = [base._load_json(path) for path in paths]
            for record in records:
                _validate_rank_readiness(args, record)
            ranks = [int(record["rank"]) for record in records]
            if len(set(ranks)) != len(ranks):
                raise RuntimeError("n580 process tree published duplicate global ranks")
            return records
        time.sleep(0.025)
    raise TimeoutError(f"timed out waiting for all n580 {args.engine} templates")


def _audit_tree(
    tree: list[int], *, allow_captured_driver_libraries: bool = False
) -> list[dict[str, Any]]:
    audit = precuda._audit_process_tree(tree)
    failures = []
    for record in audit:
        pid = int(record["pid"])
        driver_library_maps = sorted(
            {
                fields[-1].removesuffix(" (deleted)")
                for line in Path(f"/proc/{pid}/maps")
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
                if (fields := line.split())
                and any(name in fields[-1] for name in ("libcuda.so", "libnvidia-ml.so"))
            }
        )
        record["nvidia_driver_library_maps"] = driver_library_maps
        device_maps = record.get("device_maps") or []
        unsupported_fds = [
            target
            for target in record.get("accelerator_fds") or []
            if str(target).startswith(("/dev/dri/", "/dev/infiniband/"))
        ]
        rejected_driver_library_maps = (
            [] if allow_captured_driver_libraries else driver_library_maps
        )
        if device_maps or unsupported_fds or rejected_driver_library_maps:
            failures.append(
                {
                    "pid": pid,
                    "device_maps": device_maps,
                    "unsupported_accelerator_fds": unsupported_fds,
                    "nvidia_driver_library_maps": driver_library_maps,
                }
            )
    if failures:
        raise RuntimeError(f"pre-CUDA process tree owns accelerator state: {failures}")
    primary = _primary_context_active()
    if primary is not False:
        raise RuntimeError(f"CUDA primary-context state is not provably inactive: {primary}")
    return audit


def _unlink_owned_shm_name_collisions(tree: list[int]) -> list[dict[str, Any]]:
    """Free names reused after an older POSIX semaphore was unlinked.

    CRIU materializes every deleted mapping through a unique ``link_remap.*``
    backing. It refuses that safe remap when the mapping's former pathname has
    since been reused. Only unlink a colliding current pathname when its inode
    is demonstrably mapped by this exact process tree.
    """
    deleted_names: set[Path] = set()
    mapped_inodes: set[int] = set()
    for pid in tree:
        for line in (
            Path(f"/proc/{pid}/maps").read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            fields = line.split()
            if len(fields) < 6:
                continue
            inode = int(fields[4])
            path = fields[5]
            mapped_inodes.add(inode)
            # POSIX named semaphores are represented by sem.* objects in the
            # shared-memory namespace; this is not a temporary-file path.
            if (
                path.startswith("/dev/shm/sem.")  # nosec B108
                and fields[-1] == "(deleted)"
            ):
                deleted_names.add(Path(path))
    removed = []
    for path in sorted(deleted_names):
        try:
            metadata = path.stat()
        except FileNotFoundError:
            continue
        if metadata.st_ino not in mapped_inodes:
            raise RuntimeError(f"shared-memory collision is not owned by the target tree: {path}")
        path.unlink()
        removed.append({"path": str(path), "inode": metadata.st_ino})
    return removed


def _zombie_processes(tree: list[int]) -> list[dict[str, Any]]:
    zombies = []
    for pid in tree:
        fields = {}
        for line in (
            Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            name, separator, value = line.partition(":")
            if separator:
                fields[name] = value.strip()
        if fields.get("State", "").startswith("Z"):
            zombies.append(
                {
                    "pid": pid,
                    "name": fields.get("Name"),
                    "parent_pid": int(fields.get("PPid", "0")),
                    "threads": int(fields.get("Threads", "0")),
                }
            )
    return zombies


def _observe_capture_tree(
    root_pid: int,
    timeout: float = 5.0,
    *,
    allow_captured_driver_libraries: bool = False,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Audit one internally consistent process-tree observation.

    CPU-only frontend helpers may exit while the tree is being enumerated.
    Resample that ordinary race, but never hide the serving root's exit.
    """
    deadline = time.monotonic() + timeout
    last_error: FileNotFoundError | None = None
    while time.monotonic() < deadline:
        tree = base._process_tree(root_pid)
        try:
            return tree, _audit_tree(
                tree,
                allow_captured_driver_libraries=allow_captured_driver_libraries,
            )
        except FileNotFoundError as error:
            if not Path(f"/proc/{root_pid}").is_dir():
                raise RuntimeError(
                    "n580 serving root exited while observing its process tree"
                ) from error
            last_error = error
            time.sleep(0.025)
    raise RuntimeError(f"n580 process tree did not remain observable: {last_error}")


def _stabilize_capture_tree(
    root_pid: int,
    timeout: float = 15.0,
    *,
    allow_captured_driver_libraries: bool = False,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    """Require a quiet CPU process tree before CRIU freezes it."""
    deadline = time.monotonic() + timeout
    stable_observations = 0
    removed: list[dict[str, Any]] = []
    last_zombies: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        tree = base._process_tree(root_pid)
        try:
            last_zombies = _zombie_processes(tree)
            collisions = _unlink_owned_shm_name_collisions(tree)
        except FileNotFoundError as error:
            if not Path(f"/proc/{root_pid}").is_dir():
                raise RuntimeError(
                    "n580 serving root exited while stabilizing its process tree"
                ) from error
            stable_observations = 0
            time.sleep(0.025)
            continue
        removed.extend(collisions)
        if last_zombies or collisions:
            stable_observations = 0
        else:
            stable_observations += 1
            if stable_observations >= 10:
                try:
                    return (
                        tree,
                        _audit_tree(
                            tree,
                            allow_captured_driver_libraries=(allow_captured_driver_libraries),
                        ),
                        removed,
                    )
                except FileNotFoundError as error:
                    if not Path(f"/proc/{root_pid}").is_dir():
                        raise RuntimeError(
                            "n580 serving root exited while auditing its stable tree"
                        ) from error
                    stable_observations = 0
        time.sleep(0.1)
    raise RuntimeError(
        "pre-CUDA process tree did not quiesce before capture: "
        f"zombies={last_zombies} removed_collisions={removed}"
    )


def _wait_signal(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.025)
    raise TimeoutError(f"timed out waiting for {path}")


def _wait_signal_or_process_exit(path: Path, pid: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        status = Path(f"/proc/{pid}/status")
        try:
            fields = status.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError as error:
            raise RuntimeError(f"restored process {pid} exited before {path.name}") from error
        state = next(
            (
                line.partition(":")[2].strip()
                for line in fields.splitlines()
                if line.startswith("State:")
            ),
            "",
        )
        if state.startswith("Z"):
            raise RuntimeError(f"restored process {pid} became a zombie before {path.name}")
        time.sleep(0.025)
    raise TimeoutError(f"timed out waiting for {path}")


def _seal_template(args: argparse.Namespace) -> Path:
    template = args.artifact_root / "template"
    template.mkdir()
    os.replace(args.artifact_root / "images", template / "images")
    os.replace(args.artifact_root / "external-files.json", template / "external-files.json")
    return template


def _materialize_attempt(args: argparse.Namespace) -> Path:
    template = args.artifact_root / "template"
    if not (template / "images/inventory.img").is_file():
        raise FileNotFoundError("sealed TP2 pre-CUDA image is missing")
    attempt = args.artifact_root / "restore-attempts" / uuid.uuid4().hex
    attempt.parent.mkdir(exist_ok=True)
    attempt.mkdir()
    shutil.copytree(template / "images", attempt / "images", copy_function=shutil.copy2)
    shutil.copy2(template / "external-files.json", attempt / "external-files.json")
    (attempt / "work").mkdir()
    return attempt


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
    attempt: Path,
    command: list[str],
    environment: dict[str, str],
    coordinator: CoordinatorClient,
) -> tuple[dict[str, Any], float]:
    ready = attempt / "work/network-unlock-ready"
    release = attempt / "work/network-unlock-release"
    started = time.perf_counter()
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
                        "n580 CRIU restore completed without network-unlock readiness"
                    ) from None
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for n580 network unlock") from None
                time.sleep(0.01)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("n580 network-unlock readiness is not a regular file")
            break
        barrier = f"{args.activation_namespace}:n580-network-unlock-ready"
        coordinator.set(f"{barrier}:{args.rank}", "1")
        for unit_rank in range(args.world_size):
            service_runtime._coordinator_wait(coordinator, f"{barrier}:{unit_rank}", args.timeout)
        seconds = time.perf_counter() - started
        _publish_barrier_release(release)
        return future.result(), seconds
    finally:
        if ready.exists() and not release.exists():
            _publish_barrier_release(release)
        executor.shutdown(wait=True)


def capture(args: argparse.Namespace) -> dict[str, Any]:
    identity = _identity(args)
    process_template_phase = _process_template_phase(args)
    allow_captured_driver_libraries = _qualification_enabled(QUALIFICATION_DRIVER_LIBRARY_ENV)
    portable_worker_reexec = _qualification_enabled(PORTABLE_WORKER_REEXEC_ENV)
    if allow_captured_driver_libraries and process_template_phase != PRE_WORKER_IMPORT_PHASE:
        raise RuntimeError(
            f"{QUALIFICATION_DRIVER_LIBRARY_ENV} is only valid with "
            f"{QUALIFICATION_PHASE_ENV}={PRE_WORKER_IMPORT_PHASE}"
        )
    if portable_worker_reexec and (
        args.engine != "vllm"
        or process_template_phase != PRE_WORKER_IMPORT_PHASE
        or not allow_captured_driver_libraries
    ):
        raise RuntimeError(
            f"{PORTABLE_WORKER_REEXEC_ENV} requires vLLM with "
            f"{QUALIFICATION_PHASE_ENV}={PRE_WORKER_IMPORT_PHASE} and "
            f"{QUALIFICATION_DRIVER_LIBRARY_ENV}=1"
        )
    if any(args.artifact_root.iterdir()):
        raise FileExistsError("refusing to reuse a nonempty TP2 pre-CUDA artifact")
    for path in (
        args.artifact_root / "images",
        args.artifact_root / "work",
        args.artifact_root / "ready",
    ):
        path.mkdir(parents=True)
    pid_floor = base._advance_pid_allocator(args.minimum_target_pid)
    log_path = args.artifact_root / "target.log"
    started = time.perf_counter()
    coordinator = CoordinatorClient.from_path(args.coordinator_path)
    with log_path.open("ab", buffering=0) as log:
        target = subprocess.Popen(
            _child_command(args),
            env=_child_environment(args),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            readiness = _wait_unit_readiness(args, target)
            tree, tree_audit = _observe_capture_tree(
                target.pid,
                allow_captured_driver_libraries=allow_captured_driver_libraries,
            )
            base._atomic_json(
                args.artifact_root / "capture-ready.json",
                {
                    "format": 1,
                    "generation": args.generation,
                    "rank": args.rank,
                    "process_tree": tree,
                    "readiness": readiness,
                    "process_tree_audit": tree_audit,
                },
            )
            barrier = f"{args.activation_namespace}:n580-capture-ready"
            coordinator.set(f"{barrier}:{args.rank}", "1")
            for unit_rank in range(args.world_size):
                service_runtime._coordinator_wait(
                    coordinator, f"{barrier}:{unit_rank}", args.timeout
                )
            # Rank readiness is a worker boundary. The API frontend can still
            # be finishing its CPU-only processor pool, so require a full
            # second with no zombies or newly reused semaphore names.
            tree, tree_audit, unlinked_shm_collisions = _stabilize_capture_tree(
                target.pid,
                allow_captured_driver_libraries=allow_captured_driver_libraries,
            )
            qualification_driver_libraries = (
                _capture_qualification_driver_libraries(args.artifact_root, tree_audit)
                if allow_captured_driver_libraries
                else []
            )
            regular_backings = precuda._capture_regular_backings(args, tree)
            rss_bytes = sum(base._rss_bytes(pid) for pid in tree)
            command = _criu_command(args, "dump")
            command.extend(["--pid", str(target.pid)])
            profile = base._run_criu_profiled(command, _criu_environment(args))
            criu_rpc = precuda._criu_rpc_result(args.artifact_root, "dump")
            remaps = [
                path
                for path in (args.artifact_root / "shm-image").glob("link_remap.*")
                if path.is_file()
            ]
            shm_count = len(remaps)
            shm_bytes = sum(path.stat().st_size for path in remaps)
            image_bytes = sum(
                path.stat().st_size
                for path in (args.artifact_root / "images").rglob("*")
                if path.is_file()
            )
            _seal_template(args)
            dumped = f"{args.activation_namespace}:n580-dumped"
            coordinator.set(f"{dumped}:{args.rank}", "1")
            for unit_rank in range(args.world_size):
                service_runtime._coordinator_wait(
                    coordinator, f"{dumped}:{unit_rank}", args.timeout
                )
            precuda._atomic_text(args.artifact_root / "release", args.generation + "\n")
            accepted_response = None
            sleep_response = None
            sleep_seconds = None
            if args.rank == 0:
                service_runtime._wait_http(args, "/health", process=target)
                accepted_response = service_runtime._infer(args)
                sleep_started = time.perf_counter()
                if args.engine == "vllm":
                    sleep_response = service_runtime._request(args, "POST", "/sleep?level=1")
                else:
                    sleep_response = service_runtime._request(
                        args,
                        "POST",
                        "/release_memory_occupation",
                        service_runtime._sglang_memory_payload(),
                    )
                sleep_seconds = time.perf_counter() - sleep_started
                coordinator.set(f"{args.activation_namespace}:n580-slept", "1")
            else:
                service_runtime._coordinator_wait(
                    coordinator,
                    f"{args.activation_namespace}:n580-slept",
                    args.timeout,
                )
            if args.engine == "sglang":
                service_runtime._prepare_sglang_hydration(args)
            hibernate_states = service_runtime._hibernate_states(args.artifact_root)
            shape_calibration = service_runtime._capture_shape_calibration(
                args,
                coordinator,
            )
            base._kill_process_group(target.pid)
            target.wait(timeout=60)
        except BaseException:
            if target.poll() is None:
                base._kill_process_group(target.pid)
                target.wait(timeout=30)
            raise
    capture_log = activation_logs.archive_capture_log(args.artifact_root)
    report = {
        "format": 1,
        "kind": f"coldsnap-{args.engine}-n580-capture",
        "engine": args.engine,
        "generation": args.generation,
        "process_template_placement_abi": PROCESS_TEMPLATE_PLACEMENT_ABI,
        "process_template_phase": process_template_phase,
        "captured_driver_libraries_qualified": allow_captured_driver_libraries,
        "portable_worker_reexec": portable_worker_reexec,
        "rank": args.rank,
        "identity": identity,
        "target_pid": target.pid,
        "pid_floor": pid_floor,
        "process_tree": tree,
        "process_tree_rss_bytes": rss_bytes,
        "process_tree_audit": tree_audit,
        "qualification_driver_libraries": qualification_driver_libraries,
        "unlinked_shm_name_collisions": unlinked_shm_collisions,
        "readiness": readiness,
        "regular_backings": regular_backings,
        "criu_rpc": criu_rpc,
        "criu_profile": profile,
        "dump_seconds": profile["wall_seconds"],
        "acceptance_response": accepted_response,
        "disk_sleep": sleep_response,
        "disk_sleep_seconds": sleep_seconds,
        "hibernate_states": hibernate_states,
        "shape_calibration": shape_calibration,
        "capture_log": capture_log,
        "capture_controller_seconds": time.perf_counter() - started,
        "image_bytes": image_bytes,
        "shared_memory_link_count": shm_count,
        "shared_memory_link_bytes": shm_bytes,
    }
    base._atomic_json(args.artifact_root / "capture.json", report)
    return report


def restore(args: argparse.Namespace) -> dict[str, Any]:
    if args.activation_state != "running":
        raise RuntimeError("n580 does not support a pre-hydration warm activation")
    transport_environment = service_runtime._stage_restore_transport_environment(args)
    runtime_environment = service_runtime._stage_restore_runtime_environment(args)
    manifests = service_runtime._hydration_manifests(args.artifact_root)
    for worker, manifest_path in manifests.items():
        precuda._atomic_text(
            manifest_path.parent.parent / f"{ACTIVATION_DIRECTORY_PREFIX}{worker}",
            manifest_path.parent.name + "\n",
        )
        if args.weight_provider == "native":
            source = MODEL_PAYLOAD_STAGE_ROOT / f"{worker}.pack"
            if not source.is_file():
                raise RuntimeError(f"n580 native model payload is not staged for {worker}")
            target = manifest_path.with_name("model-weights.pack")
            target.unlink(missing_ok=True)
            target.symlink_to(source)
            service_runtime._select_weight_provider(manifest_path, "native")
        else:
            # Recovery is already the process's configured provider. Keep the
            # activation-local native selector absent so later lifecycle sleeps
            # continue to use the captured residual plus safetensors replay.
            if args.engine == "sglang":
                service_runtime._select_weight_provider(manifest_path, "recovery")
            else:
                manifest_path.with_name("activation-provider").unlink(missing_ok=True)
    capture_report = base._load_json(args.artifact_root / "capture.json")
    if capture_report.get("generation") != args.generation:
        raise RuntimeError("n580 process-template generation mismatch")
    _require_process_template_placement_abi(capture_report)
    identity = _identity(args)
    captured_identity = capture_report.get("identity")
    process_template_phase = _capture_process_template_phase(capture_report)
    identity_compatibility = _identity_compatibility(
        captured_identity,
        identity,
        args.kernel_compatibility,
        args.allow_compatible_criu_runtime,
    )
    nvidia_driver_library_portability = _validate_nvidia_driver_library_portability(
        captured_identity,
        identity,
        process_template_phase,
        _qualification_enabled(QUALIFICATION_DRIVER_LIBRARY_ENV),
        capture_report.get("portable_worker_reexec") is True,
    )
    qualification_driver_libraries = (
        _restore_qualification_driver_libraries(
            args.artifact_root,
            capture_report.get("qualification_driver_libraries"),
        )
        if process_template_phase == PRE_WORKER_IMPORT_PHASE
        else []
    )
    args.captured_http_port = service_runtime._identity_http_port(captured_identity)
    for path in (
        args.artifact_root / "release",
        args.artifact_root / "restore-generation",
        args.artifact_root / "restore-controller-ready.json",
        args.artifact_root / "restore-result.json",
        args.artifact_root / "restore-ready.json",
        args.artifact_root / "async-graphs-arm",
        args.artifact_root / "async-graphs-ready.json",
        args.artifact_root / "restore-placement.json",
    ):
        path.unlink(missing_ok=True)
    for path in (args.artifact_root / "ready").glob("restored-*.json"):
        path.unlink(missing_ok=True)
    regular_backings = precuda._restore_regular_backings(args)
    shm_count = base._prepare_link_remaps(args)
    attempt = _materialize_attempt(args)
    process_template_placement = _write_process_template_placement(args)
    precuda._atomic_text(args.artifact_root / "restore-generation", args.generation + "\n")
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
    coordinator = CoordinatorClient.from_path(args.coordinator_path)
    command = _criu_command(args, "restore", attempt)
    environment = _criu_environment(args)
    network_unlock_barrier_seconds = None
    if args.defer_network_unlock:
        profile, network_unlock_barrier_seconds = _run_restore_with_network_unlock_barrier(
            args, attempt, command, environment, coordinator
        )
    else:
        profile = base._run_criu_profiled(command, environment)
    mark("criu_restore_end")
    criu_rpc = precuda._criu_rpc_result(attempt, "restore")
    restored_pid = criu_rpc.get("restored_pid")
    if not isinstance(restored_pid, int) or restored_pid <= 0:
        raise RuntimeError(f"invalid restored PID: {criu_rpc}")
    prefix = args.activation_namespace
    coordinator.set(f"{prefix}:n580-criu-restored:{args.rank}", str(restored_pid))
    for unit_rank in range(args.world_size):
        service_runtime._coordinator_wait(
            coordinator, f"{prefix}:n580-criu-restored:{unit_rank}", args.timeout
        )
    mark("rank_barrier_end")
    if args.defer_network_unlock:
        os.killpg(restored_pid, signal.SIGCONT)
    release_seconds = time.perf_counter() - started
    precuda._atomic_text(args.artifact_root / "release", args.generation + "\n")
    checkpoint_ready = f"{prefix}:n580-released"
    coordinator.set(f"{checkpoint_ready}:{args.rank}", "1")
    for unit_rank in range(args.world_size):
        service_runtime._coordinator_wait(
            coordinator, f"{checkpoint_ready}:{unit_rank}", args.timeout
        )
    mark("distributed_process_release_end")

    if (
        args.engine == "sglang"
        and args.activation_state == "running"
        and os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS", "0") == "1"
    ):
        mark("async_graph_arm_begin")
    if _arm_sglang_async_graphs(args, coordinator, prefix):
        mark("async_graph_arm_end")
        mark("async_graphs_armed")

    health_seconds = None
    post_restore_response = None
    if args.rank == 0:
        restored_process = _RestoredProcess(restored_pid)
        mark("health_wait_begin")
        health_started = time.perf_counter()
        service_runtime._wait_http(args, "/health", process=restored_process)
        health_seconds = time.perf_counter() - health_started
        mark("health_ready")
        post_restore_response = service_runtime._infer(args)
        mark("acceptance_ready")
        coordinator.set(f"{prefix}:service-ready", "1")
        coordinator.set(f"{prefix}:service-state", "ready")
    else:
        service_runtime._coordinator_wait(coordinator, f"{prefix}:service-ready", args.timeout)
        mark("service_ready_observed")

    if (
        args.engine == "vllm"
        and os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS", "0") == "1"
    ):
        mark("async_graph_arm_begin")
        graph_key = f"{prefix}:arm-async-graphs"
        if args.rank == 0:
            coordinator.set(graph_key, "1")
        else:
            service_runtime._coordinator_wait(coordinator, graph_key, args.timeout)
        arm_path = os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE")
        if not arm_path:
            raise RuntimeError("async graph activation has no arm file")
        Path(arm_path).touch()
        mark("async_graph_arm_end")
        mark("async_graphs_armed")

    restored_clients = [
        base._load_json(path)
        for path in sorted((args.artifact_root / "ready").glob("restored-*.json"))
    ]
    hibernate_states = service_runtime._hibernate_states(args.artifact_root)
    mark("hibernate_state_loaded")
    report = {
        "format": 1,
        "kind": f"coldsnap-{args.engine}-n580-restore",
        "engine": args.engine,
        "generation": args.generation,
        "rank": args.rank,
        "identity": identity,
        "identity_compatibility": identity_compatibility,
        "nvidia_driver_library_portability": nvidia_driver_library_portability,
        "qualification_driver_libraries": qualification_driver_libraries,
        "process_template_phase": process_template_phase,
        "portable_worker_reexec": capture_report.get("portable_worker_reexec") is True,
        "restored_pid": restored_pid,
        "restore_attempt": str(attempt.relative_to(args.artifact_root)),
        "criu_rpc": criu_rpc,
        "criu_restore_profile": profile,
        "criu_restore_seconds": profile["wall_seconds"],
        "controller_to_release_seconds": release_seconds,
        "network_unlock_barrier_seconds": network_unlock_barrier_seconds,
        "process_template_placement": process_template_placement,
        "health_wait_seconds": health_seconds,
        "weight_provider": args.weight_provider,
        "restore_controller_seconds": time.perf_counter() - started,
        "post_restore_response": post_restore_response,
        "restored_driver_clients": restored_clients,
        "hibernate_states": hibernate_states,
        "regular_backings": regular_backings,
        "transport_environment": (
            str(transport_environment) if transport_environment is not None else None
        ),
        "runtime_environment": (
            str(runtime_environment) if runtime_environment is not None else None
        ),
        "log_boundary": log_boundary,
        "shared_memory_link_count": shm_count,
        "timeline": timeline,
    }
    base._atomic_json(args.artifact_root / "restore-result.json", report)
    unit = os.environ.get("COLDSNAP_EXPECTED_UNIT", "")
    if not unit:
        raise RuntimeError("COLDSNAP_EXPECTED_UNIT is required")
    base._atomic_json(args.artifact_root / f"restore-ready-unit-{unit}.json", report)
    if args.rank == 0:
        base._atomic_json(args.artifact_root / "restore-ready.json", report)
    while True:
        time.sleep(60)


def main() -> int:
    args = _parser().parse_args()
    if args.world_size <= 0 or args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("rank must be within a positive world size")
    if args.worker_count <= 0:
        raise ValueError("worker-count must be positive")
    if args.timeout <= 0 or args.timeout > 7200:
        raise ValueError("timeout must be in (0, 7200]")
    if not 0 <= args.tcp_port_shift <= 64511:
        raise ValueError("TCP port shift must be in [0, 64511]")
    if args.tcp_port_shift and not args.tcp_address_map:
        raise ValueError("TCP port shift requires at least one address mapping")
    if args.defer_network_unlock and not args.tcp_address_map:
        raise ValueError("deferred network unlock requires an address mapping")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        report = capture(args) if args.mode == "capture" else restore(args)
    except BaseException as error:
        report = {
            "format": 1,
            "kind": f"coldsnap-{args.engine}-n580-error",
            "engine": args.engine,
            "rank": args.rank,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        base._atomic_json(args.artifact_root / "controller-error.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

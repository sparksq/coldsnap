#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Install the CRIU-safe syscall policy, then exec an inference-engine CLI."""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import importlib.util
import ipaddress
import json
import os
import re
import shlex
import stat
import sys
import time
from pathlib import Path


CAPTURE_LOAD_FORMAT_ENV = "COLDSNAP_CAPTURE_LOAD_FORMAT"
_LOAD_FORMAT_OPTION = "--load-format"
_RESTORE_PLACEMENT_FILENAME = "restore-placement.json"
_RESTORE_RUNTIME_ENVIRONMENT_FILENAME = "restore-runtime-environment.json"
_NVIDIA_DRIVER_LIBRARY_NAMES = ("libcuda.so", "libnvidia-ml.so")
_TRANSPORT_ENVIRONMENT_NAMES = {
    "GLOO_SOCKET_IFNAME",
    "MN_IF_NAME",
    "NODE_IP",
    "TP_SOCKET_IFNAME",
    "UCX_NET_DEVICES",
    "VLLM_HOST_IP",
}
_TRANSPORT_ENVIRONMENT_PREFIXES = ("NCCL_", "OMPI_MCA_", "UCX_")
_RESTORE_RUNTIME_ENVIRONMENT_NAMES = {
    "COLDSNAP_API_MM_WARM_FILE",
    "COLDSNAP_API_MM_WARMUP_DELAY_SECONDS",
    "COLDSNAP_ASYNC_CUDA_GRAPHS",
    "COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE",
    "COLDSNAP_ASYNC_CUDA_GRAPHS_GENERATION",
    "COLDSNAP_ASYNC_CUDA_GRAPHS_READY_FILE",
    "COLDSNAP_DEFERRED_API_MM_WARMUP",
    "COLDSNAP_DEFERRED_WARMUP",
    "COLDSNAP_DEFERRED_WARMUP_ARM_FILE",
    "COLDSNAP_FREE_MEMORY_RESERVE_BYTES",
    "COLDSNAP_FULLY_WARM_FILE",
    "COLDSNAP_GRAPH_POLICY",
    "COLDSNAP_GRAPH_POLICY_REQUESTED",
    "COLDSNAP_KV_CAPACITY_GUARD",
    "COLDSNAP_REQUIRED_FREE_MEMORY_BYTES",
    "COLDSNAP_SHAPE_CALIBRATION",
    "COLDSNAP_STARTUP_PLAN_MAX_SHORTFALL_BYTES",
    "COLDSNAP_VLLM_OVERRIDE_CUMEM",
    "COLDSNAP_WARMUP_GENERATION",
    "VLLM_ENABLE_STARTUP_PLAN",
}
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROCESS_TEMPLATE_BARRIER_ENVIRONMENT = (
    "COLDSNAP_MODEL_LOAD_GENERATION",
    "COLDSNAP_MODEL_LOAD_READY_DIR",
    "COLDSNAP_MODEL_LOAD_RELEASE_FILE",
    "COLDSNAP_MODEL_LOAD_TIMEOUT_SECONDS",
    "COLDSNAP_MODEL_LOAD_POLL_SECONDS",
    "COLDSNAP_PROCESS_TEMPLATE_PHASE",
    "COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE",
    "COLDSNAP_PROCESS_TEMPLATE_RESTORE_WATCHER_SCOPE",
    "COLDSNAP_PROCESS_TEMPLATE_RESTORE_LOAD_FORMAT",
)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _accelerator_state() -> tuple[list[str], list[str]]:
    mappings: set[str] = set()
    descriptors: set[str] = set()
    try:
        for line in (
            Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            fields = line.split()
            if not fields:
                continue
            path = fields[5] if len(fields) >= 6 else fields[-1]
            if path.startswith(("/dev/nvidia", "/dev/dri/", "/dev/infiniband/")) or any(
                name in path for name in _NVIDIA_DRIVER_LIBRARY_NAMES
            ):
                mappings.add(path)
        for descriptor in Path("/proc/self/fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith(("/dev/nvidia", "/dev/dri/", "/dev/infiniband/")):
                descriptors.add(target)
    except OSError as error:
        raise RuntimeError("cannot audit the pre-exec accelerator state") from error
    return sorted(mappings), sorted(descriptors)


def _process_template_pre_exec() -> None:
    """Publish a driver-portable n580 boundary before the serving CLI execs.

    This boundary deliberately precedes both inference engines. A later vLLM
    worker-import boundary retained the capture host's NVML and CUDA-driver
    link maps in the API, engine-core, and worker processes. Those ELF mappings
    are not portable to a newer NVIDIA userspace driver even when the CUDA ABI
    and hardware are compatible. The launcher boundary retains no engine or
    driver state; engine caches and ColdSnap's optimized weight providers still
    accelerate the post-restore startup path.
    """
    if os.environ.get("COLDSNAP_PROCESS_TEMPLATE_PHASE") != "pre_exec":
        return
    generation = os.environ.get("COLDSNAP_MODEL_LOAD_GENERATION", "").strip()
    ready_value = os.environ.get("COLDSNAP_MODEL_LOAD_READY_DIR", "").strip()
    release_value = os.environ.get("COLDSNAP_MODEL_LOAD_RELEASE_FILE", "").strip()
    try:
        rank = int(os.environ["COLDSNAP_UNIT_INDEX"])
        timeout = float(os.environ.get("COLDSNAP_MODEL_LOAD_TIMEOUT_SECONDS", "0"))
    except (KeyError, ValueError) as error:
        raise RuntimeError("invalid pre-exec process-template configuration") from error
    if not generation or not ready_value or not release_value or rank < 0 or timeout <= 0:
        raise RuntimeError("incomplete pre-exec process-template configuration")
    mappings, descriptors = _accelerator_state()
    if mappings or descriptors:
        raise RuntimeError(
            "pre-exec process template owns accelerator or NVIDIA driver state: "
            f"mappings={mappings!r} descriptors={descriptors!r}"
        )
    ready_dir = Path(ready_value)
    release = Path(release_value)
    _atomic_json(
        ready_dir / f"rank-{rank}.json",
        {
            "schema": 1,
            "generation": generation,
            "pid": os.getpid(),
            "rank": rank,
            "local_rank": 0,
            "model": os.environ.get("COLDSNAP_MODEL_ID", ""),
            "revision": os.environ.get("COLDSNAP_MODEL_REVISION", ""),
            "tensor_parallel_size": int(os.environ.get("COLDSNAP_TP_SIZE", "1")),
            "phase": "pre_exec",
            "cuda_initialized": False,
            "cuda_driver_context_present": False,
            "cuda_primary_context_active": False,
            "accelerator_fds": descriptors,
            "criu_cpu_only_candidate": True,
            "ready_monotonic": time.monotonic(),
        },
    )
    deadline = time.monotonic() + timeout
    while True:
        try:
            released = release.read_text(encoding="utf-8").strip() == generation
        except FileNotFoundError:
            released = False
        if released:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for pre-exec process-template release")
        time.sleep(0.025)
    marker_value = os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE", "").strip()
    if marker_value:
        try:
            restored = Path(marker_value).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            restored = ""
        if restored == generation:
            # Do not re-export the already committed native payload while a
            # restored service performs its activation load.
            os.environ["COLDSNAP_EXPORT_MODEL_PAYLOAD"] = "0"
            os.environ["COLDSNAP_PROCESS_TEMPLATE_RESTORED"] = "1"


def _clear_process_template_barrier_environment() -> None:
    """Consume the one-shot launcher barrier before the engine exec.

    The restored/provider marker is intentionally not part of this set. It is
    durable activation state consumed by the vLLM/SGLang plugins after exec;
    the generation, release files, and phase describe only the CRIU launcher
    that has now completed. A later engine-owned boundary must retain these
    values until its plugin publishes readiness and receives release.
    """
    if os.environ.get("COLDSNAP_PROCESS_TEMPLATE_PHASE") != "pre_exec":
        return
    for name in _PROCESS_TEMPLATE_BARRIER_ENVIRONMENT:
        os.environ.pop(name, None)


def _is_transport_environment(name: str) -> bool:
    return name in _TRANSPORT_ENVIRONMENT_NAMES or name.startswith(_TRANSPORT_ENVIRONMENT_PREFIXES)


def _apply_restore_transport_environment() -> list[str] | None:
    """Apply the manager-staged destination fabric identity before exec."""
    if os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORED") != "1":
        return None
    raw_path = os.environ.get("COLDSNAP_RESTORE_TRANSPORT_ENVIRONMENT_PATH")
    if raw_path is None:
        raise RuntimeError("restored process template has no transport environment")
    path = Path(raw_path)
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            raise RuntimeError("restore transport environment is not a bounded regular file")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read restore transport environment: {error}") from error
    expected_unit = os.environ.get("COLDSNAP_EXPECTED_UNIT")
    if (
        not isinstance(payload, dict)
        or payload.get("format") != 1
        or payload.get("kind") != "coldsnap-restore-transport-environment"
        or not expected_unit
        or payload.get("unit") != expected_unit
        or not isinstance(payload.get("variables"), dict)
    ):
        raise RuntimeError("restore transport environment is invalid")
    variables = payload["variables"]
    for name, value in variables.items():
        if (
            not isinstance(name, str)
            or _ENVIRONMENT_NAME.fullmatch(name) is None
            or not _is_transport_environment(name)
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise RuntimeError("restore transport environment contains an invalid variable")
    for name in tuple(os.environ):
        if _is_transport_environment(name) and name not in variables:
            del os.environ[name]
    os.environ.update(variables)
    return sorted(variables)


def _apply_restore_runtime_environment() -> list[str] | None:
    """Apply target-selected lifecycle policy before a pre-exec engine start."""
    if os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORED") != "1":
        return None
    marker_value = os.environ.get(
        "COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE", ""
    ).strip()
    raw_path = os.environ.get("COLDSNAP_RESTORE_RUNTIME_ENVIRONMENT_PATH", "").strip()
    if not marker_value or not raw_path:
        raise RuntimeError("restored process template has no runtime environment")
    path = Path(raw_path)
    expected = Path(marker_value).parent / _RESTORE_RUNTIME_ENVIRONMENT_FILENAME
    if path != expected:
        raise RuntimeError("restore runtime environment is outside the capsule")
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            raise RuntimeError("restore runtime environment is not a bounded regular file")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read restore runtime environment: {error}") from error
    expected_unit = os.environ.get("COLDSNAP_EXPECTED_UNIT")
    if (
        not isinstance(payload, dict)
        or set(payload) != {"format", "kind", "unit", "variables"}
        or payload.get("format") != 1
        or payload.get("kind") != "coldsnap-restore-runtime-environment"
        or not expected_unit
        or payload.get("unit") != expected_unit
        or not isinstance(payload.get("variables"), dict)
    ):
        raise RuntimeError("restore runtime environment is invalid")
    variables = payload["variables"]
    if any(
        not isinstance(name, str)
        or name not in _RESTORE_RUNTIME_ENVIRONMENT_NAMES
        or not isinstance(value, str)
        or "\x00" in value
        for name, value in variables.items()
    ):
        raise RuntimeError("restore runtime environment contains an invalid variable")
    for name in _RESTORE_RUNTIME_ENVIRONMENT_NAMES:
        os.environ.pop(name, None)
    os.environ.update(variables)
    return sorted(variables)


def _sglang_startup_provider() -> str:
    """Resolve one controller-selected provider for every worker in this unit."""
    if os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORED") != "1":
        return ""
    root_value = os.environ.get("COLDSNAP_PROCESS_ARTIFACT_ROOT", "").strip()
    graph_value = os.environ.get("COLDSNAP_EXECUTION_GRAPH", "").strip()
    if not root_value or not graph_value:
        raise RuntimeError("restored SGLang startup lacks its artifact or execution graph")
    try:
        graph = json.loads(graph_value)
    except json.JSONDecodeError as error:
        raise RuntimeError("restored SGLang execution graph is invalid") from error
    slots = graph.get("by_process_slot") if isinstance(graph, dict) else None
    if not isinstance(slots, dict) or not slots:
        raise RuntimeError("restored SGLang execution graph has no workers")
    workers = set(slots.values())
    if not all(isinstance(worker, str) and worker for worker in workers):
        raise RuntimeError("restored SGLang execution graph has invalid workers")
    providers: set[str] = set()
    root = Path(root_value)
    for worker in workers:
        selector = root / "hydration" / worker / "activation-provider"
        try:
            provider = selector.read_text(encoding="utf-8").strip()
        except FileNotFoundError as error:
            raise RuntimeError(
                f"restored SGLang worker {worker} has no activation provider"
            ) from error
        if provider not in {"native", "recovery"}:
            raise RuntimeError(
                f"restored SGLang worker {worker} has invalid activation provider {provider!r}"
            )
        providers.add(provider)
    if len(providers) != 1:
        raise RuntimeError("restored SGLang unit selected mixed weight providers")
    return providers.pop()


def _configured_load_format(command: list[str]) -> str | None:
    shell_index = _shell_command_index(command)
    if shell_index is not None:
        try:
            return _configured_load_format(_split_shell_command(command[shell_index]))
        except ValueError as error:
            raise ValueError("inference-engine shell command is invalid") from error
    for index, argument in enumerate(command):
        if argument == _LOAD_FORMAT_OPTION:
            if index + 1 >= len(command):
                raise ValueError("--load-format requires a value")
            return command[index + 1]
        if argument.startswith(_LOAD_FORMAT_OPTION + "="):
            return argument.split("=", 1)[1]
    return None


def _shell_command_index(command: list[str]) -> int | None:
    if not command or Path(command[0]).name not in {"bash", "sh"}:
        return None
    try:
        index = command.index("-c")
    except ValueError:
        return None
    return index + 1 if index + 1 < len(command) else None


def _split_shell_command(payload: str) -> list[str]:
    """Tokenize a shell command after applying POSIX line continuations.

    ``shlex.split`` retains the newline from a backslash-newline pair as a
    literal argument, while the shell removes the pair before tokenization.
    Sparkrun recipes conventionally use those continuations for readability,
    so mirror shell preprocessing before inspecting or rewriting options.
    """
    return shlex.split(payload.replace("\\\r\n", "").replace("\\\n", ""))


def _restored_placement() -> dict[str, object] | None:
    if os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORED") != "1":
        return None
    marker_value = os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE", "").strip()
    generation = os.environ.get("COLDSNAP_MODEL_LOAD_GENERATION", "").strip()
    if not marker_value or not generation:
        raise RuntimeError("restored process-template placement lacks generation identity")
    path = Path(marker_value).parent / _RESTORE_PLACEMENT_FILENAME
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
        raise RuntimeError("restored process-template placement is not a small regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("restored process-template placement is invalid") from error
    if not isinstance(value, dict) or set(value) != {
        "format",
        "kind",
        "generation",
        "master_address",
        "master_port",
        "http_port",
    }:
        raise RuntimeError("restored process-template placement has an invalid shape")
    if (
        value.get("format") != 1
        or value.get("kind") != "coldsnap-process-template-placement"
        or value.get("generation") != generation
    ):
        raise RuntimeError("restored process-template placement has an invalid identity")
    try:
        address = ipaddress.ip_address(value.get("master_address"))
    except ValueError as error:
        raise RuntimeError("restored process-template master address is invalid") from error
    if (
        address.is_unspecified
        or address.is_loopback
        or address.is_multicast
        or address.is_link_local
    ):
        raise RuntimeError("restored process-template master address is not portable")
    for name in ("master_port", "http_port"):
        port = value.get(name)
        if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port < 65536:
            raise RuntimeError(f"restored process-template {name} is invalid")
    return value


def _replace_restore_placement(command: list[str], placement: dict[str, object]) -> list[str]:
    result = list(command)
    shell_index = _shell_command_index(result)
    if shell_index is not None:
        try:
            tokens = _split_shell_command(result[shell_index])
        except ValueError as error:
            raise ValueError("inference-engine shell command is invalid") from error
        result[shell_index] = shlex.join(_replace_restore_placement(tokens, placement))
        return result
    address = str(placement["master_address"])
    master_port = int(placement["master_port"])
    http_port = int(placement["http_port"])
    dist_address = f"[{address}]:{master_port}" if ":" in address else f"{address}:{master_port}"
    values = {
        "--master-addr": address,
        "--master-address": address,
        "--master-port": str(master_port),
        "--dist-init-addr": dist_address,
        "--port": str(http_port),
    }
    for index, argument in enumerate(result):
        if argument in values:
            if index + 1 >= len(result):
                raise ValueError(f"{argument} requires a value")
            result[index + 1] = values[argument]
            continue
        for option, value in values.items():
            if argument.startswith(option + "="):
                result[index] = f"{option}={value}"
                break
    return result


def _replace_load_format(command: list[str], load_format: str) -> list[str]:
    result = list(command)
    shell_index = _shell_command_index(result)
    if shell_index is not None:
        payload = result[shell_index]
        try:
            tokens = _split_shell_command(payload)
        except ValueError as error:
            raise ValueError("inference-engine shell command is invalid") from error
        if _configured_load_format(tokens) is None:
            # Preserve the recipe's original quoting and shell structure when
            # adding the common no-explicit-format case.
            result[shell_index] = (
                payload.rstrip() + f" {_LOAD_FORMAT_OPTION} {shlex.quote(load_format)}"
            )
        else:
            result[shell_index] = shlex.join(_replace_load_format(tokens, load_format))
        return result
    for index, argument in enumerate(result):
        if argument == _LOAD_FORMAT_OPTION:
            if index + 1 >= len(result):
                raise ValueError("--load-format requires a value")
            result[index + 1] = load_format
            return result
        if argument.startswith(_LOAD_FORMAT_OPTION + "="):
            result[index] = f"{_LOAD_FORMAT_OPTION}={load_format}"
            return result
    result.extend((_LOAD_FORMAT_OPTION, load_format))
    return result


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _capture_load_format(command: list[str]) -> tuple[list[str], str | None]:
    """Resolve the fastest installed safetensors loader for capture only.

    The adapter sets ``COLDSNAP_CAPTURE_LOAD_FORMAT`` to opt in.  An explicit
    recipe format is retained as the final fallback, while InstantTensor and
    FastSafeTensors are preferred when their runtime packages are installed.
    Recovery does not execute this policy: the captured vLLM load config is
    switched to ColdSnap immediately before the process snapshot.
    """
    requested = os.environ.get(CAPTURE_LOAD_FORMAT_ENV)
    if requested is None:
        return command, None
    requested = requested.strip().lower() or (_configured_load_format(command) or "auto")
    if requested == "coldsnap":
        requested = "safetensors"

    candidates = (
        ("instanttensor", "instanttensor"),
        ("fastsafetensors", "fastsafetensors"),
    )
    selected = next(
        (load_format for load_format, module in candidates if _module_available(module)),
        requested if requested in {"auto", "safetensors"} else "safetensors",
    )
    return _replace_load_format(command, selected), selected


def _block_io_uring() -> None:
    library_name = ctypes.util.find_library("seccomp")
    if library_name is None:
        raise RuntimeError("libseccomp is required to block io_uring")
    seccomp = ctypes.CDLL(library_name, use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]

    allow = 0x7FFF0000
    deny = 0x00050000 | errno.EPERM
    context = seccomp.seccomp_init(allow)
    if not context:
        raise OSError(ctypes.get_errno(), "seccomp_init failed")
    try:
        for name in (b"io_uring_setup", b"io_uring_enter", b"io_uring_register"):
            number = seccomp.seccomp_syscall_resolve_name(name)
            if number < 0:
                raise RuntimeError(f"libseccomp cannot resolve {name.decode()}")
            result = seccomp.seccomp_rule_add(context, deny, number, 0)
            if result != 0:
                raise OSError(-result, f"seccomp_rule_add failed for {name.decode()}")
        result = seccomp.seccomp_load(context)
        if result != 0:
            raise OSError(-result, "seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)


def main() -> int:
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("an inference-engine command is required")
    engine = os.environ.get("COLDSNAP_ENGINE", "vllm").strip().lower()
    if engine not in {"vllm", "sglang"}:
        raise ValueError(f"unsupported ColdSnap engine {engine!r}")
    _process_template_pre_exec()
    placement = _restored_placement()
    if placement is not None:
        command = _replace_restore_placement(command, placement)
        os.environ["MASTER_ADDR"] = str(placement["master_address"])
        os.environ["MASTER_PORT"] = str(placement["master_port"])
        print(
            "ColdSnap applied restored process placement "
            f"{placement['master_address']}:{placement['master_port']}",
            file=sys.stderr,
            flush=True,
        )
    transport_environment = _apply_restore_transport_environment()
    if transport_environment is not None:
        print(
            "ColdSnap applied restored transport environment: " + ", ".join(transport_environment),
            file=sys.stderr,
            flush=True,
        )
    runtime_environment = _apply_restore_runtime_environment()
    if runtime_environment is not None:
        print(
            "ColdSnap applied restored runtime environment: "
            + ", ".join(runtime_environment),
            file=sys.stderr,
            flush=True,
        )
    _clear_process_template_barrier_environment()
    selected = None
    if engine == "vllm":
        if os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORED") == "1":
            # Both activation providers enter through the recovery-aware
            # loader. Its per-worker selector chooses the staged native model
            # payload or safetensors recovery after vLLM constructs the stable
            # VA layout.
            command = _replace_load_format(command, "coldsnap")
            print(
                "ColdSnap restored vLLM startup selected coldsnap weights",
                file=sys.stderr,
                flush=True,
            )
        else:
            command, selected = _capture_load_format(command)
    elif engine == "sglang":
        provider = _sglang_startup_provider()
        if provider:
            os.environ["COLDSNAP_SGLANG_STARTUP_PROVIDER"] = provider
        if provider == "native":
            # Dummy performs architecture construction and quantization
            # post-processing without reading checkpoint tensors. The SGLang
            # plugin hydrates the native semantic pack before this call
            # returns, so graph warmup never observes dummy bytes.
            command = _replace_load_format(command, "dummy")
            print(
                "ColdSnap restored SGLang startup selected native weights",
                file=sys.stderr,
                flush=True,
            )
    if selected is not None:
        os.environ[CAPTURE_LOAD_FORMAT_ENV] = selected
        print(
            f"ColdSnap capture selected {engine} load format {selected}",
            file=sys.stderr,
            flush=True,
        )
    _block_io_uring()
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Hold vLLM workers at generation-safe activation barriers.

``pre_worker_import`` waits before vLLM resolves/imports the concrete worker
class. This is the preferred CPU-only CRIU/fork boundary because model and
kernel module imports are allowed to initialize CUDA. ``pre_device`` is a
later diagnostic boundary immediately before ``Worker.init_device``; its
readiness record fails closed on a CUDA context and separately inventories
accelerator descriptors for controller validation. ``pre_load`` and
``pre_hydration`` are progressively later live-CUDA activation boundaries.
"""

from __future__ import annotations

import ctypes
import functools
import importlib
import inspect
import ipaddress
import json
import logging
import os
import pickle
import shutil
import stat
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.parse import urlsplit

from coldsnap_import_hook import after_module_import
from coldsnap_vllm import VllmContractError


RELEASE_FILE_ENV = "COLDSNAP_MODEL_LOAD_RELEASE_FILE"
READY_DIR_ENV = "COLDSNAP_MODEL_LOAD_READY_DIR"
GENERATION_ENV = "COLDSNAP_MODEL_LOAD_GENERATION"
TIMEOUT_ENV = "COLDSNAP_MODEL_LOAD_TIMEOUT_SECONDS"
POLL_SECONDS_ENV = "COLDSNAP_MODEL_LOAD_POLL_SECONDS"
PHASE_ENV = "COLDSNAP_PROCESS_TEMPLATE_PHASE"
RESTORE_MARKER_FILE_ENV = "COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE"
RESTORE_WATCHER_SCOPE_ENV = "COLDSNAP_PROCESS_TEMPLATE_RESTORE_WATCHER_SCOPE"
RESTORE_LOAD_FORMAT_ENV = "COLDSNAP_PROCESS_TEMPLATE_RESTORE_LOAD_FORMAT"
RESTORE_RUNTIME_ENVIRONMENT_PATH_ENV = "COLDSNAP_RESTORE_RUNTIME_ENVIRONMENT_PATH"
RESTORE_PLACEMENT_FILENAME = "restore-placement.json"
RESTORE_RUNTIME_ENVIRONMENT_FILENAME = "restore-runtime-environment.json"
PORTABLE_WORKER_REEXEC_ENV = "COLDSNAP_N580_PORTABLE_WORKER_REEXEC"
PORTABLE_WORKER_REEXEC_ACTIVE_ENV = "COLDSNAP_N580_PORTABLE_WORKER_REEXEC_ACTIVE"

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

PRE_DEVICE_PHASE = "pre_device"
PRE_WORKER_IMPORT_PHASE = "pre_worker_import"
PRE_LOAD_PHASE = "pre_load"
PRE_HYDRATION_PHASE = "pre_hydration"
_PHASES = {
    PRE_WORKER_IMPORT_PHASE,
    PRE_DEVICE_PHASE,
    PRE_LOAD_PHASE,
    PRE_HYDRATION_PHASE,
}

_WORKER_IMPORT_MARKER = "_coldsnap_process_template_worker_import_barrier"
_WORKER_IMPORT_WAITED = "_coldsnap_process_template_worker_import_waited"
_DEVICE_MARKER = "_coldsnap_process_template_device_barrier"
_LOAD_MARKER = "_coldsnap_process_template_load_barrier"
_WORKER_PROC_HANDLE_MARKER = "_coldsnap_process_template_input_queue_handle"
_WORKER_MODULE = "vllm.v1.worker.gpu_worker"
_WORKER_BASE_MODULE = "vllm.v1.worker.worker_base"
_WORKER_PROC_MODULE = "vllm.v1.executor.multiproc_executor"
_FLA_UTILS_MODULE = "vllm.third_party.flash_linear_attention.ops.utils"
_NVIDIA_PLACEHOLDER_TOKEN = "/coldsnap-nvidia-placeholder-"

logger = logging.getLogger(__name__)
_installed = False
_restore_reinitialization_lock = threading.Lock()
_restore_reinitialized_generation = ""
_restore_reinitialization_result: dict[str, Any] | None = None
_captured_accelerator_fds: dict[int, str] = {}
_INPUT_QUEUE_HANDLE_UNSET = object()
_active_input_queue_handle: Any = _INPUT_QUEUE_HANDLE_UNSET


def _boolean_environment(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise VllmContractError(f"{name} must be a boolean")


@dataclass(frozen=True)
class ProcessTemplateSettings:
    release_file: Path | None = None
    ready_dir: Path | None = None
    generation: str = ""
    timeout_seconds: float = 0.0
    poll_seconds: float = 0.05
    phase: str = PRE_LOAD_PHASE
    restore_marker_file: Path | None = None

    @property
    def enabled(self) -> bool:
        return self.release_file is not None


def process_template_settings_from_env(
    environment: Mapping[str, str] | None = None,
) -> ProcessTemplateSettings:
    values = os.environ if environment is None else environment
    raw_release = values.get(RELEASE_FILE_ENV)
    raw_ready = values.get(READY_DIR_ENV)
    raw_restore_marker = values.get(RESTORE_MARKER_FILE_ENV)
    generation = values.get(GENERATION_ENV, "").strip()
    phase = values.get(PHASE_ENV, PRE_LOAD_PHASE).strip()
    if phase not in _PHASES:
        raise VllmContractError(f"{PHASE_ENV} must be one of {sorted(_PHASES)}, got {phase!r}")
    configured = (raw_release is not None, raw_ready is not None, bool(generation))
    if any(configured) and not all(configured):
        raise VllmContractError(
            f"{RELEASE_FILE_ENV}, {READY_DIR_ENV}, and {GENERATION_ENV} must be supplied together"
        )
    try:
        timeout_seconds = float(values.get(TIMEOUT_ENV, "0"))
        poll_seconds = float(values.get(POLL_SECONDS_ENV, "0.05"))
    except ValueError as error:
        raise VllmContractError("process-template timing values must be numbers") from error
    if timeout_seconds < 0:
        raise VllmContractError(f"{TIMEOUT_ENV} must not be negative")
    if not 0.001 <= poll_seconds <= 5.0:
        raise VllmContractError(f"{POLL_SECONDS_ENV} must be between 0.001 and 5 seconds")
    if not all(configured):
        return ProcessTemplateSettings(
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            phase=phase,
            restore_marker_file=(Path(str(raw_restore_marker)) if raw_restore_marker else None),
        )
    return ProcessTemplateSettings(
        release_file=Path(str(raw_release)),
        ready_dir=Path(str(raw_ready)),
        generation=generation,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        phase=phase,
        restore_marker_file=(Path(str(raw_restore_marker)) if raw_restore_marker else None),
    )


def _restored_generation(settings: ProcessTemplateSettings) -> bool:
    marker = settings.restore_marker_file
    if marker is None:
        return False
    try:
        return marker.read_text(encoding="utf-8").strip() == settings.generation
    except FileNotFoundError:
        return False


def _restored_runtime_environment(
    settings: ProcessTemplateSettings,
) -> dict[str, str] | None:
    """Read target-selected runtime policy for a restored worker."""
    if not _restored_generation(settings) or settings.restore_marker_file is None:
        return None
    raw_path = os.environ.get(RESTORE_RUNTIME_ENVIRONMENT_PATH_ENV)
    if not raw_path:
        raise VllmContractError("portable restore has no runtime environment handoff")
    path = Path(raw_path)
    expected = settings.restore_marker_file.parent / RESTORE_RUNTIME_ENVIRONMENT_FILENAME
    if path != expected:
        raise VllmContractError("restore runtime environment path is outside the capsule")
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            raise VllmContractError(
                "restore runtime environment is not a bounded regular file"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VllmContractError(
            f"cannot read restore runtime environment: {error}"
        ) from error
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
        raise VllmContractError("restore runtime environment is invalid")
    variables = payload["variables"]
    if any(
        not isinstance(name, str)
        or name not in _RESTORE_RUNTIME_ENVIRONMENT_NAMES
        or not isinstance(value, str)
        or "\x00" in value
        for name, value in variables.items()
    ):
        raise VllmContractError(
            "restore runtime environment contains an invalid variable"
        )
    return dict(variables)


def _apply_restored_runtime_environment(
    settings: ProcessTemplateSettings,
) -> dict[str, str] | None:
    """Replace capture policy with the target-selected restore policy in place.

    Portable cross-driver workers consume the same handoff while constructing
    their exec environment.  An exact-target process template deliberately
    avoids that exec, but it must still stop carrying capture-only controls
    such as shape calibration into the restored engine initialization.
    """
    variables = _restored_runtime_environment(settings)
    if variables is None:
        return None
    for name in _RESTORE_RUNTIME_ENVIRONMENT_NAMES:
        os.environ.pop(name, None)
    os.environ.update(variables)
    logger.info(
        "Applied %d target-selected restore runtime settings in place",
        len(variables),
    )
    return variables


def _restored_placement(settings: ProcessTemplateSettings) -> dict[str, Any] | None:
    if not _restored_generation(settings) or settings.restore_marker_file is None:
        return None
    path = settings.restore_marker_file.parent / RESTORE_PLACEMENT_FILENAME
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
        raise VllmContractError("restored process-template placement is not a small regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VllmContractError("restored process-template placement is invalid") from error
    if not isinstance(value, dict) or set(value) != {
        "format",
        "kind",
        "generation",
        "master_address",
        "master_port",
        "http_port",
        "tcp_address_map",
        "tcp_port_shift",
    }:
        raise VllmContractError("restored process-template placement has an invalid shape")
    if (
        value.get("format") != 1
        or value.get("kind") != "coldsnap-process-template-placement"
        or value.get("generation") != settings.generation
    ):
        raise VllmContractError("restored process-template placement has an invalid identity")
    try:
        address = ipaddress.ip_address(value.get("master_address"))
    except ValueError as error:
        raise VllmContractError("restored process-template master address is invalid") from error
    if (
        address.is_unspecified
        or address.is_loopback
        or address.is_multicast
        or address.is_link_local
    ):
        raise VllmContractError("restored process-template master address is not portable")
    for name in ("master_port", "http_port"):
        port = value.get(name)
        if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port < 65536:
            raise VllmContractError(f"restored process-template {name} is invalid")
    address_map = value.get("tcp_address_map")
    if not isinstance(address_map, dict):
        raise VllmContractError("restored process-template TCP address map is invalid")
    for before, after in address_map.items():
        try:
            ipaddress.ip_address(before)
            ipaddress.ip_address(after)
        except ValueError as error:
            raise VllmContractError(
                "restored process-template TCP address map is invalid"
            ) from error
    port_shift = value.get("tcp_port_shift")
    if (
        isinstance(port_shift, bool)
        or not isinstance(port_shift, int)
        or not 0 <= port_shift < 65536 - 1024
    ):
        raise VllmContractError("restored process-template TCP port shift is invalid")
    return value


def _shift_tcp_port(port: int, shift: int) -> int:
    if not 1024 <= port < 65536:
        raise VllmContractError("vLLM lazy TCP endpoint uses a privileged port")
    return 1024 + ((port - 1024 + shift) % (65536 - 1024))


def _remap_tcp_endpoint(endpoint: str, placement: Mapping[str, Any]) -> str:
    try:
        parsed = urlsplit(endpoint)
        address = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise VllmContractError("vLLM lazy TCP endpoint is invalid") from error
    if parsed.scheme != "tcp" or address is None or port is None or parsed.path:
        raise VllmContractError("vLLM lazy TCP endpoint is invalid")
    address = placement["tcp_address_map"].get(address, address)
    shifted_port = _shift_tcp_port(port, placement["tcp_port_shift"])
    try:
        parsed_address = ipaddress.ip_address(address)
    except ValueError as error:
        raise VllmContractError("vLLM lazy TCP endpoint address is invalid") from error
    host = f"[{parsed_address}]" if parsed_address.version == 6 else str(parsed_address)
    return f"tcp://{host}:{shifted_port}"


def _remap_lazy_input_queue(placement: Mapping[str, Any]) -> dict[str, str | None]:
    """Retarget vLLM's not-yet-connected cross-node scheduler queue.

    The n580 pre-worker boundary is reached from ``WorkerProc.__init__``. Its
    input queue writer already exists in the restored EngineCore, while the
    worker creates the reader only after model initialization. CRIU remaps the
    writer socket, so the serialized reader handle on this active stack must
    follow the same address and port transformation before initialization
    resumes.
    """
    handle = _active_input_queue_handle
    if handle is _INPUT_QUEUE_HANDLE_UNSET:
        raise VllmContractError("vLLM pre-worker restore cannot locate its lazy input queue handle")
    if handle is None:
        return {"captured": None, "restored": None}
    captured = handle.remote_subscribe_addr
    if captured is None:
        return {"captured": None, "restored": None}
    if not isinstance(captured, str):
        raise VllmContractError("vLLM lazy input queue endpoint is invalid")
    restored = _remap_tcp_endpoint(captured, placement)
    handle.remote_subscribe_addr = restored
    logger.info("Remapped restored vLLM lazy input queue %s to %s", captured, restored)
    return {"captured": captured, "restored": restored}


def _remap_restored_local_addresses(
    placement: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Retarget address-bearing environment inherited from the capsule.

    The portable worker re-exec starts a fresh interpreter, but its environment
    is copied from the restored capture. vLLM's ``get_ip()`` reads
    ``VLLM_HOST_IP`` when it creates secondary ZMQ transports, so leaving the
    capture-host value there makes the fresh worker bind an address that does
    not exist on the restore host. Apply the controller's per-host address map
    before exec, just as we do for the primary rendezvous and lazy input queue.
    """
    address_map = placement["tcp_address_map"]
    remapped: dict[str, dict[str, str]] = {}
    for name in ("VLLM_HOST_IP", "NODE_IP"):
        captured = os.environ.get(name, "").strip()
        if not captured:
            continue
        restored = address_map.get(captured, captured)
        os.environ[name] = restored
        remapped[name] = {"captured": captured, "restored": restored}
    return remapped


def _install_worker_proc_handle_hook(
    settings: ProcessTemplateSettings, module: Any | None = None
) -> None:
    if settings.phase != PRE_WORKER_IMPORT_PHASE:
        return
    module = module or importlib.import_module(_WORKER_PROC_MODULE)
    worker_proc = getattr(module, "WorkerProc", None)
    if not isinstance(worker_proc, type):
        raise VllmContractError("vLLM multiproc executor lacks WorkerProc")
    original = getattr(worker_proc, "__init__", None)
    if not callable(original):
        raise VllmContractError("vLLM WorkerProc constructor is unavailable")
    if getattr(original, _WORKER_PROC_HANDLE_MARKER, False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def init_with_input_queue_handle(proc: Any, *args: Any, **kwargs: Any) -> Any:
        global _active_input_queue_handle
        try:
            bound = signature.bind(proc, *args, **kwargs)
            handle = bound.arguments["input_shm_handle"]
        except (KeyError, TypeError) as error:
            raise VllmContractError(
                "vLLM WorkerProc constructor lacks the input queue handle"
            ) from error
        if handle is not None and not hasattr(handle, "remote_subscribe_addr"):
            raise VllmContractError("vLLM input queue handle contract changed")
        previous = _active_input_queue_handle
        _active_input_queue_handle = handle
        try:
            return original(proc, *args, **kwargs)
        finally:
            _active_input_queue_handle = previous

    setattr(init_with_input_queue_handle, _WORKER_PROC_HANDLE_MARKER, True)
    worker_proc.__init__ = init_with_input_queue_handle


def _apply_restored_placement(
    all_kwargs: list[dict[str, Any]], settings: ProcessTemplateSettings
) -> dict[str, Any] | None:
    placement = _restored_placement(settings)
    if placement is None:
        return None
    configs: dict[int, Any] = {}
    for kwargs in all_kwargs:
        if not isinstance(kwargs, dict) or "vllm_config" not in kwargs:
            continue
        try:
            parallel_config = kwargs["vllm_config"].parallel_config
        except AttributeError as error:
            raise VllmContractError(
                "vLLM worker arguments lack a mutable restore parallel config"
            ) from error
        configs[id(parallel_config)] = parallel_config
    if not configs:
        raise VllmContractError("vLLM restore has no worker parallel configs")
    for parallel_config in configs.values():
        if not hasattr(parallel_config, "master_addr") or not hasattr(
            parallel_config, "master_port"
        ):
            raise VllmContractError("vLLM restore parallel config lacks rendezvous placement")
        parallel_config.master_addr = placement["master_address"]
        parallel_config.master_port = placement["master_port"]
    os.environ["MASTER_ADDR"] = placement["master_address"]
    os.environ["MASTER_PORT"] = str(placement["master_port"])
    placement["local_addresses"] = _remap_restored_local_addresses(placement)
    logger.info(
        "Applied restored vLLM rendezvous placement %s:%d to %d worker config(s)",
        placement["master_address"],
        placement["master_port"],
        len(configs),
    )
    placement["lazy_input_queue"] = _remap_lazy_input_queue(placement)
    return placement


def _reset_restored_nvml_client() -> bool:
    """Unload inherited NVML state before reconnecting to the GPU driver.

    A context-free CRIU image may still contain a loaded, cleanly shut down
    NVML library.  Its user-space bookkeeping refers to the capture
    container's driver connection and cannot be made valid by reopening the
    character-device descriptors.  Drop the library and function-pointer
    cache so the next vLLM NVML call establishes a new connection.
    """
    pynvml = sys.modules.get("vllm.third_party.pynvml")
    if pynvml is None:
        return False
    library = getattr(pynvml, "nvmlLib", None)
    cache = getattr(pynvml, "_nvmlGetFunctionPointer_cache", None)
    lock = getattr(pynvml, "libLoadLock", None)
    if library is None or not isinstance(cache, dict) or lock is None:
        return False

    mapped_path = None
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and "/libnvidia-ml.so" in fields[-1]:
            mapped_path = Path(fields[-1].removesuffix(" (deleted)"))
            break
    if mapped_path is None or not mapped_path.is_file():
        raise VllmContractError("cannot locate the mapped NVML shared object")
    handle = int(getattr(library, "_handle", 0))
    with lock:
        cache.clear()
        pynvml.nvmlLib = None
        pynvml._nvmlLib_refcount = 0
    libdl = ctypes.CDLL("libdl.so.2", use_errno=True)
    if handle:
        dlclose = libdl.dlclose
        dlclose.argtypes = [ctypes.c_void_p]
        dlclose.restype = ctypes.c_int
        if dlclose(ctypes.c_void_p(handle)) != 0:
            errno = ctypes.get_errno()
            raise VllmContractError(f"failed to unload inherited NVML client state: errno {errno}")
    descriptor, fresh_path = tempfile.mkstemp(prefix="coldsnap-libnvidia-ml-", suffix=".so")
    os.close(descriptor)
    try:
        shutil.copyfile(mapped_path, fresh_path)
        pynvml.nvmlLib = ctypes.CDLL(
            fresh_path,
            mode=os.RTLD_LAZY | os.RTLD_LOCAL,
        )
    finally:
        Path(fresh_path).unlink(missing_ok=True)
    pynvml._coldsnap_fresh_nvml_library = pynvml.nvmlLib
    logger.info("Reset inherited NVML client state after process restore")
    return True


def _close_restored_accelerator_fds() -> list[str]:
    """Discard context-free driver descriptors recreated by the CRIU plugin."""
    closed: list[str] = []
    try:
        descriptors = list(Path("/proc/self/fd").iterdir())
    except OSError as error:
        raise VllmContractError("cannot enumerate restored accelerator descriptors") from error
    for descriptor in descriptors:
        try:
            fd = int(descriptor.name)
            target = os.readlink(descriptor)
        except (OSError, ValueError):
            continue
        captured_target = _captured_accelerator_fds.get(fd)
        is_accelerator = target.startswith(("/dev/nvidia", "/dev/dri/", "/dev/infiniband/"))
        is_placeholder = _NVIDIA_PLACEHOLDER_TOKEN in target and target.endswith(" (deleted)")
        if captured_target is None and not is_accelerator and not is_placeholder:
            continue
        if (
            captured_target is not None
            and not is_accelerator
            and target != "/dev/null"
            and not is_placeholder
        ):
            raise VllmContractError(
                "captured accelerator descriptor was repurposed before restore "
                f"reinitialization: fd={fd} captured={captured_target} current={target}"
            )
        try:
            os.close(fd)
        except OSError as error:
            raise VllmContractError(
                f"failed to close restored accelerator descriptor {fd}: {target}"
            ) from error
        closed.append(
            target
            if captured_target is None or captured_target == target
            else f"{captured_target} (placeholder {target})"
        )
    return sorted(closed)


def _reinitialize_driver_clients_after_restore(
    settings: ProcessTemplateSettings,
) -> dict[str, Any] | None:
    global _restore_reinitialized_generation, _restore_reinitialization_result
    if not _restored_generation(settings):
        return None
    # The pre-exec launcher sets this before its engine exec. An engine-owned
    # boundary is already inside the restored Python process, so promote the
    # same generation proof here before loader/backend construction resumes.
    os.environ["COLDSNAP_EXPORT_MODEL_PAYLOAD"] = "0"
    os.environ["COLDSNAP_PROCESS_TEMPLATE_RESTORED"] = "1"
    with _restore_reinitialization_lock:
        if _restore_reinitialized_generation == settings.generation:
            return _restore_reinitialization_result
        closed = _close_restored_accelerator_fds()
        nvml_reset = _reset_restored_nvml_client()
        result = {
            "closed_accelerator_fds": closed,
            "nvml_library_reset": nvml_reset,
            "module_file": __file__,
            "pid": os.getpid(),
        }
        _restore_reinitialized_generation = settings.generation
        _restore_reinitialization_result = result
        logger.info("Reinitialized restored driver clients: %s", result)
        return result


def _watch_for_process_restore(settings: ProcessTemplateSettings) -> None:
    while not _released(settings):
        time.sleep(settings.poll_seconds)
    reinitialized = _reinitialize_driver_clients_after_restore(settings)
    if reinitialized is not None and settings.ready_dir is not None:
        _atomic_write(
            settings.ready_dir / f"restored-process-{os.getpid()}.json",
            json.dumps(reinitialized, sort_keys=True) + "\n",
        )


def _start_process_restore_watcher(settings: ProcessTemplateSettings) -> None:
    if settings.restore_marker_file is None:
        return
    threading.Thread(
        target=_watch_for_process_restore,
        args=(settings,),
        name="coldsnap-process-restore-reinitializer",
        daemon=True,
    ).start()


def _rank(worker: Any) -> int:
    for owner in (worker, getattr(worker, "parallel_config", None)):
        value = getattr(owner, "rank", None)
        if value is not None:
            return int(value)
    raise VllmContractError("vLLM GPU worker exposes no global rank")


def _cuda_initialized() -> bool:
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    return bool(callable(is_initialized) and is_initialized())


def _cuda_driver_context_present() -> bool | None:
    """Check the current driver context without calling ``cuInit``.

    ``False`` means the driver explicitly reported no current context. ``None``
    is an unknown result and therefore never qualifies for a CPU-only dump.
    """
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        get_current = driver.cuCtxGetCurrent
        get_current.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        get_current.restype = ctypes.c_int
        context = ctypes.c_void_p()
        code = int(get_current(ctypes.byref(context)))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if code == 0:
        return context.value is not None
    if code == 3:  # CUDA_ERROR_NOT_INITIALIZED
        return False
    return None


def _cuda_primary_context_active(device: int = 0) -> bool | None:
    """Check primary-context activity without initializing the CUDA driver."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        get_state = driver.cuDevicePrimaryCtxGetState
        get_state.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
        ]
        get_state.restype = ctypes.c_int
        flags = ctypes.c_uint()
        active = ctypes.c_int()
        code = int(get_state(device, ctypes.byref(flags), ctypes.byref(active)))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if code == 0:
        return bool(active.value)
    if code == 3:  # CUDA_ERROR_NOT_INITIALIZED
        return False
    return None


def _open_accelerator_fds() -> list[str]:
    """Return device descriptors that make a plain CPU CRIU dump suspect."""
    targets: set[str] = set()
    try:
        descriptors = Path("/proc/self/fd").iterdir()
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith(("/dev/nvidia", "/dev/dri/", "/dev/infiniband/")):
                targets.add(target)
    except OSError:
        # Absence of procfs cannot prove eligibility, so publish a sentinel
        # that makes the controller fail closed.
        return ["<procfs-unavailable>"]
    return sorted(targets)


def _open_accelerator_fd_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        descriptors = Path("/proc/self/fd").iterdir()
        for descriptor in descriptors:
            try:
                fd = int(descriptor.name)
                target = os.readlink(descriptor)
            except (OSError, ValueError):
                continue
            if target.startswith(("/dev/nvidia", "/dev/dri/", "/dev/infiniband/")):
                records.append({"fd": fd, "target": target})
    except OSError:
        return [{"fd": -1, "target": "<procfs-unavailable>"}]
    return sorted(records, key=lambda record: int(record["fd"]))


def _cpu_only_process_audit() -> dict[str, Any]:
    """Return the local evidence used to admit a pre-CUDA process."""
    accelerator_fds = _open_accelerator_fds()
    accelerator_fd_records = _open_accelerator_fd_records()
    if any(int(record["fd"]) < 0 for record in accelerator_fd_records):
        accelerator_fds = ["<procfs-unavailable>"]
    else:
        _captured_accelerator_fds.clear()
        _captured_accelerator_fds.update(
            {int(record["fd"]): str(record["target"]) for record in accelerator_fd_records}
        )
    cuda_initialized = _cuda_initialized()
    driver_context_present = _cuda_driver_context_present()
    primary_context_active = _cuda_primary_context_active()
    return {
        "cuda_initialized": cuda_initialized,
        "cuda_driver_context_present": driver_context_present,
        "cuda_primary_context_active": primary_context_active,
        "accelerator_fds": accelerator_fds,
        "accelerator_fd_records": accelerator_fd_records,
        "criu_cpu_only_candidate": not cuda_initialized
        and driver_context_present is False
        and primary_context_active is False,
        "criu_device_fds_require_validation": bool(accelerator_fds),
    }


def preload_context_free_fla_platform_probe(
    settings: ProcessTemplateSettings | None = None,
) -> bool:
    """Preload FLA metadata without Triton's context-creating device query.

    Qwen3.5 imports the vendored Flash Linear Attention utilities while the
    frontend resolves its model class. The vendored module asks Triton for its
    active target and then asks ``torch.cuda`` for device metadata; both calls
    create a CUDA context before vLLM constructs the EngineCore worker. vLLM's
    platform object obtains the same immutable metadata through NVML without a
    CUDA context. Temporarily route only that import through those safe values,
    then restore every function before normal execution continues.
    """
    resolved = settings or process_template_settings_from_env()
    if not resolved.enabled or resolved.phase != PRE_WORKER_IMPORT_PHASE:
        return False
    if _FLA_UTILS_MODULE in sys.modules:
        audit = _cpu_only_process_audit()
        if not audit["criu_cpu_only_candidate"]:
            raise VllmContractError(
                "Flash Linear Attention was imported after CUDA initialization: "
                f"{json.dumps(audit, sort_keys=True)}"
            )
        return False

    torch = importlib.import_module("torch")
    platforms = importlib.import_module("vllm.platforms")
    triton = importlib.import_module("triton")
    current_platform = getattr(platforms, "current_platform", None)
    cuda = getattr(torch, "cuda", None)
    active_driver = getattr(getattr(triton, "runtime", None), "driver", None)
    active_driver = getattr(active_driver, "active", None)
    if (
        current_platform is None
        or not callable(getattr(current_platform, "is_cuda", None))
        or not current_platform.is_cuda()
        or cuda is None
        or active_driver is None
    ):
        raise VllmContractError(
            "pre-worker FLA context suppression requires the CUDA vLLM platform"
        )
    if _cuda_initialized():
        raise VllmContractError(
            "cannot suppress the FLA platform probe after PyTorch CUDA initialization"
        )

    get_current_target = getattr(active_driver, "get_current_target", None)
    get_device_name = getattr(cuda, "get_device_name", None)
    get_device_capability = getattr(cuda, "get_device_capability", None)
    platform_device_name = getattr(current_platform, "get_device_name", None)
    platform_has_capability = getattr(current_platform, "has_device_capability", None)
    if not all(
        callable(value)
        for value in (
            get_current_target,
            get_device_name,
            get_device_capability,
            platform_device_name,
            platform_has_capability,
        )
    ):
        raise VllmContractError("vLLM FLA platform metadata contract changed")

    def context_free_device_name(device: Any = None) -> str:
        return str(platform_device_name(0 if device is None else device))

    def context_free_device_capability(device: Any = None) -> tuple[int, int]:
        del device
        return (9, 0) if platform_has_capability(90) else (0, 0)

    active_driver.get_current_target = lambda: SimpleNamespace(backend="cuda")
    cuda.get_device_name = context_free_device_name
    cuda.get_device_capability = context_free_device_capability
    try:
        importlib.import_module(_FLA_UTILS_MODULE)
    finally:
        active_driver.get_current_target = get_current_target
        cuda.get_device_name = get_device_name
        cuda.get_device_capability = get_device_capability

    audit = _cpu_only_process_audit()
    if not audit["criu_cpu_only_candidate"]:
        raise VllmContractError(
            "context-free FLA preload initialized CUDA unexpectedly: "
            f"{json.dumps(audit, sort_keys=True)}"
        )
    logger.info("Preloaded Flash Linear Attention without a CUDA context")
    return True


def _ready_payload(worker: Any, settings: ProcessTemplateSettings) -> dict[str, Any]:
    model_config = getattr(worker, "model_config", None)
    parallel_config = getattr(worker, "parallel_config", None)
    payload = {
        "schema": 1,
        "generation": settings.generation,
        "pid": os.getpid(),
        "rank": _rank(worker),
        "local_rank": int(getattr(worker, "local_rank", -1)),
        "model": str(getattr(model_config, "model", "")),
        "revision": str(getattr(model_config, "revision", "")),
        "tensor_parallel_size": int(getattr(parallel_config, "tensor_parallel_size", 1)),
        "phase": settings.phase,
        "ready_monotonic": time.monotonic(),
    }
    if settings.phase in {PRE_WORKER_IMPORT_PHASE, PRE_DEVICE_PHASE}:
        payload.update(_cpu_only_process_audit())
    return payload


def _require_cpu_only_template(payload: Mapping[str, Any]) -> None:
    """Fail before publishing a contaminated pre-CUDA template.

    Controller-side validation remains necessary for accelerator descriptors,
    because a context-free worker may have opened character devices for NVML.
    The worker itself can, however, prove that PyTorch has not initialized CUDA,
    that the CUDA driver reports no current context, and that the primary
    context is inactive. Unknown audit results are rejected rather than being
    advertised as a usable CRIU boundary.
    """
    phase = payload.get("phase")
    if phase not in {PRE_WORKER_IMPORT_PHASE, PRE_DEVICE_PHASE}:
        return
    if (
        payload.get("cuda_initialized") is False
        and payload.get("cuda_driver_context_present") is False
        and payload.get("cuda_primary_context_active") is False
        and payload.get("criu_cpu_only_candidate") is True
    ):
        return
    audit = {
        "cuda_initialized": payload.get("cuda_initialized", "<missing>"),
        "cuda_driver_context_present": payload.get("cuda_driver_context_present", "<missing>"),
        "cuda_primary_context_active": payload.get("cuda_primary_context_active", "<missing>"),
        "criu_cpu_only_candidate": payload.get("criu_cpu_only_candidate", "<missing>"),
    }
    raise VllmContractError(
        "refusing to publish a CUDA-contaminated process template at "
        f"phase {phase!r}: {json.dumps(audit, sort_keys=True)}"
    )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def _released(settings: ProcessTemplateSettings) -> bool:
    assert settings.release_file is not None
    try:
        value = settings.release_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False
    return value == settings.generation


def _wait_for_rank_release(
    rank: int,
    settings: ProcessTemplateSettings,
    payload: dict[str, Any],
    *,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> float:
    if not settings.enabled or settings.ready_dir is None:
        return 0.0
    ready_path = settings.ready_dir / f"rank-{rank}.json"
    _require_cpu_only_template(payload)
    _atomic_write(
        ready_path,
        json.dumps(payload, sort_keys=True) + "\n",
    )
    started = monotonic()
    logger.info(
        "COLDSNAP process template rank %d ready for generation %s; waiting at phase %s",
        rank,
        settings.generation,
        settings.phase,
    )
    while not _released(settings):
        elapsed = monotonic() - started
        if settings.timeout_seconds and elapsed >= settings.timeout_seconds:
            raise TimeoutError(
                "timed out waiting for COLDSNAP model-load release generation "
                f"{settings.generation!r} at {settings.release_file}"
            )
        sleep(settings.poll_seconds)
    elapsed = monotonic() - started
    logger.info(
        "COLDSNAP process template rank %d released after %.3f s; loading model",
        rank,
        elapsed,
    )
    return elapsed


def _wait_for_release(
    worker: Any,
    settings: ProcessTemplateSettings,
    *,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> float:
    return _wait_for_rank_release(
        _rank(worker),
        settings,
        _ready_payload(worker, settings),
        sleep=sleep,
        monotonic=monotonic,
    )


def wait_for_hydration_release(
    *,
    rank: int,
    capture_id: str,
    allocation_count: int,
    allocation_bytes: int,
    replay_allocation_count: int,
) -> float | None:
    """Publish a fully constructed CUDA layout and wait before hydration."""
    settings = process_template_settings_from_env()
    if not settings.enabled or settings.phase != PRE_HYDRATION_PHASE:
        return None
    payload = {
        "schema": 1,
        "generation": settings.generation,
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "model": os.environ.get("COLDSNAP_MODEL_ID", ""),
        "revision": os.environ.get("COLDSNAP_MODEL_REVISION", ""),
        "tensor_parallel_size": int(os.environ.get("COLDSNAP_TP_SIZE", "1")),
        "phase": settings.phase,
        "capture_id": capture_id,
        "allocation_count": allocation_count,
        "allocation_bytes": allocation_bytes,
        "replay_allocation_count": replay_allocation_count,
        "ready_monotonic": time.monotonic(),
    }
    return _wait_for_rank_release(rank, settings, payload)


def _worker_class(module: Any) -> type[Any]:
    worker_classes: list[type[Any]] = []
    for name in ("Worker", "GPUWorker"):
        value = getattr(module, name, None)
        if isinstance(value, type) and value not in worker_classes:
            worker_classes.append(value)
    if len(worker_classes) != 1:
        raise VllmContractError("vllm.v1.worker.gpu_worker must expose one supported worker class")
    return worker_classes[0]


def _worker_import_view(wrapper: Any, all_kwargs: list[dict[str, Any]]) -> Any:
    try:
        kwargs = all_kwargs[int(wrapper.rpc_rank)]
        vllm_config = kwargs["vllm_config"]
        rank = int(kwargs.get("rank", wrapper.global_rank))
        local_rank = int(kwargs.get("local_rank", wrapper.rpc_rank))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise VllmContractError(
            "vLLM WorkerWrapperBase.init_worker arguments are incompatible"
        ) from error
    return SimpleNamespace(
        rank=rank,
        local_rank=local_rank,
        model_config=vllm_config.model_config,
        parallel_config=vllm_config.parallel_config,
    )


def _select_restored_load_format(
    all_kwargs: list[dict[str, Any]], settings: ProcessTemplateSettings
) -> str | None:
    """Apply a restore-only loader without changing the captured CLI identity.

    A pre-worker n580 template is captured after vLLM parses its normal fast
    loader but before it imports the concrete GPU worker.  Capture must retain
    that loader for artifact production, while a restored recovery activation
    must use ColdSnap's replay-aware loader.  Mutate only the worker configs
    after the restore marker proves this is a restored generation.
    """
    selected = os.environ.get(RESTORE_LOAD_FORMAT_ENV, "").strip().lower()
    if not selected or not _restored_generation(settings):
        return None
    configs: dict[int, Any] = {}
    for kwargs in all_kwargs:
        if not isinstance(kwargs, dict) or "vllm_config" not in kwargs:
            continue
        try:
            config = kwargs["vllm_config"]
            load_config = config.load_config
        except AttributeError as error:
            raise VllmContractError(
                "vLLM worker arguments lack a mutable restore load config"
            ) from error
        configs[id(load_config)] = load_config
    if not configs:
        raise VllmContractError("vLLM restore has no worker load configs")
    for load_config in configs.values():
        load_config.load_format = selected
    logger.info(
        "Selected restored vLLM load format %s for %d worker config(s)",
        selected,
        len(configs),
    )
    return selected


def _worker_reexec_connections() -> tuple[Any, Any]:
    """Find the parent-facing pipes on the active WorkerProc.worker_main stack."""
    try:
        frame = sys._getframe(1)
    except (AttributeError, ValueError) as error:
        raise VllmContractError("cannot inspect the vLLM worker bootstrap stack") from error
    while frame is not None:
        if (
            frame.f_code.co_name == "worker_main"
            and frame.f_globals.get("__name__") == _WORKER_PROC_MODULE
        ):
            ready_writer = frame.f_locals.get("ready_writer")
            death_pipe = frame.f_locals.get("death_pipe")
            if ready_writer is None or death_pipe is None:
                raise VllmContractError(
                    "vLLM worker bootstrap lacks its ready or death pipe"
                )
            return ready_writer, death_pipe
        frame = frame.f_back
    raise VllmContractError("cannot locate the active vLLM WorkerProc.worker_main frame")


def _portable_worker_lock_record(settings: ProcessTemplateSettings) -> dict[str, Any]:
    if settings.ready_dir is None:
        raise VllmContractError("portable worker re-exec requires a ready directory")
    path = settings.ready_dir / "portable-worker.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.close(descriptor)
    return {
        "format": 1,
        "kind": "coldsnap-portable-file-lock",
        "path": str(path),
    }


def _connection_record(connection: Any) -> dict[str, Any]:
    try:
        descriptor = int(connection.fileno())
        readable = bool(connection.readable)
        writable = bool(connection.writable)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise VllmContractError("vLLM worker bootstrap pipe is invalid") from error
    os.set_inheritable(descriptor, True)
    return {"fd": descriptor, "readable": readable, "writable": writable}


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as output:
            pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reexec_restored_worker(
    wrapper: Any,
    all_kwargs: list[dict[str, Any]],
    settings: ProcessTemplateSettings,
) -> None:
    """Replace a restored pre-CUDA worker with a target-driver process.

    The restored PID is retained, as are its parent-facing readiness and death
    pipes. Re-exec discards capture-host CUDA/NVML ELF mappings before the
    concrete GPU worker is imported, so the new interpreter resolves the
    target host's driver clients. The EngineCore and API processes remain the
    captured process template and retain their already-prepared CPU state.
    """
    if settings.ready_dir is None:
        raise VllmContractError("portable worker re-exec requires a ready directory")
    rank = _rank(_worker_import_view(wrapper, all_kwargs))
    try:
        source_kwargs = all_kwargs[int(wrapper.rpc_rank)]
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise VllmContractError("portable worker re-exec cannot select worker arguments") from error
    if not isinstance(source_kwargs, dict):
        raise VllmContractError("portable worker re-exec arguments are invalid")
    worker_kwargs = dict(source_kwargs)
    lock = worker_kwargs.pop("shared_worker_lock", None)
    if lock is None:
        raise VllmContractError("portable worker re-exec lacks the shared worker lock")
    if _active_input_queue_handle is _INPUT_QUEUE_HANDLE_UNSET:
        raise VllmContractError("portable worker re-exec lacks the input queue handle")
    ready_writer, death_pipe = _worker_reexec_connections()
    worker_kwargs.update(
        {
            "input_shm_handle": _active_input_queue_handle,
            "ready_pipe": None,
            "death_pipe": None,
            "shared_worker_lock": None,
            "inherited_fds": [],
        }
    )
    payload = {
        "format": 1,
        "kind": "coldsnap-vllm-portable-worker-reexec",
        "generation": settings.generation,
        "rank": rank,
        "worker_kwargs": worker_kwargs,
        # The captured multiprocessing SemLock is host-kernel state and its
        # POSIX semaphore name cannot be reopened on a foreign node. All
        # workers in this unit instead reopen the same target-local file lock.
        "shared_worker_lock": _portable_worker_lock_record(settings),
        "ready_pipe": _connection_record(ready_writer),
        "death_pipe": _connection_record(death_pipe),
    }
    payload_path = settings.ready_dir / f"worker-reexec-rank-{rank}.pickle"
    _atomic_pickle(payload_path, payload)
    environment = os.environ.copy()
    runtime_environment = _restored_runtime_environment(settings)
    if runtime_environment is None:
        raise VllmContractError("portable worker re-exec lacks restore runtime policy")
    for name in _RESTORE_RUNTIME_ENVIRONMENT_NAMES:
        environment.pop(name, None)
    environment.update(runtime_environment)
    for name in (
        RELEASE_FILE_ENV,
        READY_DIR_ENV,
        GENERATION_ENV,
        TIMEOUT_ENV,
        POLL_SECONDS_ENV,
        PHASE_ENV,
        RESTORE_WATCHER_SCOPE_ENV,
    ):
        environment.pop(name, None)
    environment[PORTABLE_WORKER_REEXEC_ACTIVE_ENV] = "1"
    environment["COLDSNAP_PROCESS_TEMPLATE_RESTORED"] = "1"
    environment["COLDSNAP_EXPORT_MODEL_PAYLOAD"] = "0"
    logger.info(
        "Re-executing restored vLLM worker rank %d against target-host driver "
        "clients with %d target runtime settings",
        rank,
        len(runtime_environment),
    )
    os.execve(
        sys.executable,
        [sys.executable, "-m", "coldsnap_vllm_worker_reexec", str(payload_path)],
        environment,
    )
    raise AssertionError("os.execve unexpectedly returned")


def _wait_before_worker_import(
    wrapper: Any,
    all_kwargs: list[dict[str, Any]],
    settings: ProcessTemplateSettings,
) -> None:
    if vars(wrapper).get(_WORKER_IMPORT_WAITED, False):
        return
    view = _worker_import_view(wrapper, all_kwargs)
    waited = _wait_for_release(view, settings)
    worker_reexec = (
        _restored_generation(settings)
        and _boolean_environment(PORTABLE_WORKER_REEXEC_ENV)
        and not _boolean_environment(PORTABLE_WORKER_REEXEC_ACTIVE_ENV)
    )
    runtime_environment = (
        None
        if worker_reexec
        else _apply_restored_runtime_environment(settings)
    )
    reinitialized = (
        None if worker_reexec else _reinitialize_driver_clients_after_restore(settings)
    )
    placement = _apply_restored_placement(all_kwargs, settings)
    selected_load_format = _select_restored_load_format(all_kwargs, settings)
    if (reinitialized is not None or worker_reexec) and settings.ready_dir is not None:
        evidence = dict(reinitialized or {})
        evidence["selected_load_format"] = selected_load_format
        evidence["placement"] = placement
        evidence["portable_worker_reexec"] = worker_reexec
        evidence["restore_runtime_environment"] = sorted(runtime_environment or {})
        _atomic_write(
            settings.ready_dir / f"restored-rank-{_rank(view)}.json",
            json.dumps(evidence, sort_keys=True) + "\n",
        )
    if worker_reexec:
        _reexec_restored_worker(wrapper, all_kwargs, settings)
    setattr(wrapper, _WORKER_IMPORT_WAITED, True)
    wrapper._coldsnap_process_template_wait_seconds = waited


def _install_worker_import_barrier(
    settings: ProcessTemplateSettings, module: Any | None = None
) -> None:
    module = module or importlib.import_module(_WORKER_BASE_MODULE)
    wrapper_class = getattr(module, "WorkerWrapperBase", None)
    if not isinstance(wrapper_class, type):
        raise VllmContractError("vLLM worker_base lacks WorkerWrapperBase")
    original = getattr(wrapper_class, "init_worker", None)
    if not callable(original):
        raise VllmContractError("WorkerWrapperBase.init_worker is unavailable")
    if getattr(original, _WORKER_IMPORT_MARKER, False):
        return

    @functools.wraps(original)
    def init_worker_after_release(
        wrapper: Any, all_kwargs: list[dict[str, Any]], *args: Any, **kwargs: Any
    ) -> Any:
        _wait_before_worker_import(wrapper, all_kwargs, settings)
        return original(wrapper, all_kwargs, *args, **kwargs)

    setattr(init_worker_after_release, _WORKER_IMPORT_MARKER, True)
    wrapper_class.init_worker = init_worker_after_release


def _wait_current_worker_import(settings: ProcessTemplateSettings) -> bool:
    """Handle plugin loading from inside an already-running init_worker call."""
    try:
        frame = sys._getframe(1)
    except (AttributeError, ValueError):
        return False
    while frame is not None:
        wrapper = frame.f_locals.get("self")
        all_kwargs = frame.f_locals.get("all_kwargs")
        owner = type(wrapper)
        if (
            frame.f_code.co_name == "init_worker"
            and owner.__name__ == "WorkerWrapperBase"
            and owner.__module__ == _WORKER_BASE_MODULE
            and isinstance(all_kwargs, list)
        ):
            _wait_before_worker_import(wrapper, all_kwargs, settings)
            return True
        frame = frame.f_back
    return False


def _install_worker_device_barrier(
    settings: ProcessTemplateSettings, module: Any | None = None
) -> None:
    module = module or importlib.import_module(_WORKER_MODULE)
    worker_class = _worker_class(module)
    original = getattr(worker_class, "init_device", None)
    if not callable(original):
        raise VllmContractError("GPU worker init_device is unavailable")
    if getattr(original, _DEVICE_MARKER, False):
        return

    @functools.wraps(original)
    def init_device_after_release(worker: Any, *args: Any, **kwargs: Any) -> Any:
        waited = _wait_for_release(worker, settings)
        worker._coldsnap_process_template_wait_seconds = waited
        return original(worker, *args, **kwargs)

    setattr(init_device_after_release, _DEVICE_MARKER, True)
    worker_class.init_device = init_device_after_release


def _install_worker_load_barrier(
    settings: ProcessTemplateSettings, module: Any | None = None
) -> None:
    module = module or importlib.import_module(_WORKER_MODULE)
    worker_class = _worker_class(module)
    original = getattr(worker_class, "load_model", None)
    if not callable(original):
        raise VllmContractError("GPU worker load_model is unavailable")
    if getattr(original, _LOAD_MARKER, False):
        return

    @functools.wraps(original)
    def load_model_after_release(worker: Any, *args: Any, **kwargs: Any) -> Any:
        waited = _wait_for_release(worker, settings)
        worker._coldsnap_process_template_wait_seconds = waited
        return original(worker, *args, **kwargs)

    setattr(load_model_after_release, _LOAD_MARKER, True)
    worker_class.load_model = load_model_after_release


def install_process_template_hook(
    settings: ProcessTemplateSettings | None = None,
) -> ProcessTemplateSettings:
    """Install the opt-in pre-model-load worker activation barrier."""
    global _installed
    resolved = settings or process_template_settings_from_env()
    if not resolved.enabled or _installed:
        return resolved
    watcher_scope = os.environ.get(RESTORE_WATCHER_SCOPE_ENV, "all").strip()
    if watcher_scope not in {"all", "parent", "none"}:
        raise VllmContractError(f"{RESTORE_WATCHER_SCOPE_ENV} must be 'all', 'parent', or 'none'")
    if watcher_scope == "parent":
        # Registration precedes EngineCore, worker, and frontend-helper
        # creation. Do not let short-lived helpers inherit a daemon thread:
        # CRIU cannot serialize a process that becomes a threaded zombie.
        os.environ[RESTORE_WATCHER_SCOPE_ENV] = "none"
    if watcher_scope != "none":
        _start_process_restore_watcher(resolved)
    if resolved.phase == PRE_WORKER_IMPORT_PHASE:
        after_module_import(
            _WORKER_PROC_MODULE,
            "process-template-input-queue-handle",
            lambda module: _install_worker_proc_handle_hook(resolved, module),
        )
        after_module_import(
            _WORKER_BASE_MODULE,
            "process-template-pre-worker-import",
            lambda module: _install_worker_import_barrier(resolved, module),
        )
        # A spawned worker can first load this plugin from inside the original
        # init_worker call. The new wrapper cannot affect that active frame.
        _wait_current_worker_import(resolved)
    elif resolved.phase == PRE_DEVICE_PHASE:
        after_module_import(
            _WORKER_MODULE,
            "process-template-pre-device",
            lambda module: _install_worker_device_barrier(resolved, module),
        )
    elif resolved.phase == PRE_LOAD_PHASE:
        after_module_import(
            _WORKER_MODULE,
            "process-template-pre-load",
            lambda module: _install_worker_load_barrier(resolved, module),
        )
    _installed = True
    logger.info("Installed COLDSNAP process-template barrier: %s", asdict(resolved))
    return resolved

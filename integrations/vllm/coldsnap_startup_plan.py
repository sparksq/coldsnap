# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Make vLLM's startup-plan device identity work on GB10 unified memory."""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Any

from coldsnap_import_hook import after_module_import


MAX_SHORTFALL_ENV = "COLDSNAP_STARTUP_PLAN_MAX_SHORTFALL_BYTES"
REQUIRED_FREE_MEMORY_ENV = "COLDSNAP_REQUIRED_FREE_MEMORY_BYTES"
FREE_MEMORY_RESERVE_ENV = "COLDSNAP_FREE_MEMORY_RESERVE_BYTES"
_ADMISSION_MARKER = "_coldsnap_free_memory_admission"
_PHASE_FINGERPRINT_ATTR = "_coldsnap_startup_plan_initial_fingerprint"
_PHASE_FINGERPRINT_MARKER = "_coldsnap_phase_stable_fingerprint"
_CONFIGURED_MODEL_LEN_ATTR = "_coldsnap_startup_plan_configured_model_len"
_COMPILE_IDENTITY_MARKER = "_coldsnap_startup_plan_compile_identity"
_GPU_WORKER_MODULE = "vllm.v1.worker.gpu_worker"
_CAPTURE_LOAD_FORMAT_ENV = "COLDSNAP_CAPTURE_LOAD_FORMAT"
_PROCESS_TEMPLATE_RESTORE_LOAD_FORMAT_ENV = (
    "COLDSNAP_PROCESS_TEMPLATE_RESTORE_LOAD_FORMAT"
)
_PROCESS_TEMPLATE_RESTORED_ENV = "COLDSNAP_PROCESS_TEMPLATE_RESTORED"
_VLLM_CACHE_ROOT_ENV = "VLLM_CACHE_ROOT"
_DEFAULT_VLLM_CACHE_ROOT = "/var/cache/coldsnap/runtime/vllm"
_COMPILE_CACHE_MARKER = "coldsnap-compile-cache.json"
_COMPILE_CACHE_NAMESPACE = re.compile(r"^[0-9a-f]{10}$")


def _all_workers_admit_startup_plan(worker: Any, locally_admitted: bool) -> bool:
    """Require every distributed worker to make the same profiling decision.

    vLLM's memory-profile path contains collectives.  If one rank applies its
    cached startup plan while another rank rejects its plan, the former skips
    those collectives and the latter waits forever.  Reduce the local admission
    decisions over the world process group before either path can continue.
    """
    world_size = int(getattr(worker.parallel_config, "world_size", 1))
    if world_size <= 1:
        return locally_admitted

    import torch

    distributed = getattr(torch, "distributed", None)
    if (
        distributed is None
        or not distributed.is_available()
        or not distributed.is_initialized()
    ):
        raise RuntimeError(
            "ColdSnap cannot coordinate startup-plan admission before the "
            "distributed process group is initialized"
        )
    decision = torch.tensor(
        1 if locally_admitted else 0,
        dtype=torch.int32,
        device=worker.device,
    )
    distributed.all_reduce(decision, op=distributed.ReduceOp.MIN)
    return bool(decision.item())


def _vllm_cache_root() -> Path:
    value = os.environ.get(_VLLM_CACHE_ROOT_ENV, _DEFAULT_VLLM_CACHE_ROOT)
    root = Path(value)
    if not root.is_absolute():
        raise RuntimeError(f"{_VLLM_CACHE_ROOT_ENV} must be an absolute path")
    return root


def _compile_cache_marker_path() -> Path:
    return _vllm_cache_root() / _COMPILE_CACHE_MARKER


def _compile_cache_directory(namespace: str, *, require_directory: bool) -> Path:
    if not _COMPILE_CACHE_NAMESPACE.fullmatch(namespace):
        raise RuntimeError("ColdSnap compiler-cache marker has an invalid namespace")
    directory = _vllm_cache_root() / "torch_compile_cache" / namespace
    if require_directory and not directory.is_dir():
        raise RuntimeError(
            "ColdSnap compiler-cache marker references an unavailable directory"
        )
    return directory


def _read_compile_cache_marker() -> Path | None:
    marker = _compile_cache_marker_path()
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("ColdSnap compiler-cache marker is unreadable") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != 1
        or payload.get("kind") != "coldsnap-vllm-compile-cache"
        or not isinstance(payload.get("namespace"), str)
    ):
        raise RuntimeError("ColdSnap compiler-cache marker is invalid")
    return _compile_cache_directory(
        payload["namespace"], require_directory=True
    )


def _write_compile_cache_marker(cache_dir: Any) -> None:
    if not isinstance(cache_dir, str) or not cache_dir:
        raise RuntimeError("vLLM did not expose its compiler-cache directory")
    directory = Path(cache_dir)
    compile_root = _vllm_cache_root() / "torch_compile_cache"
    try:
        relative = directory.relative_to(compile_root)
    except ValueError as error:
        raise RuntimeError(
            "vLLM compiler-cache directory is outside the ColdSnap cache root"
        ) from error
    if len(relative.parts) != 1:
        raise RuntimeError("vLLM compiler-cache directory has an invalid shape")
    _compile_cache_directory(relative.name, require_directory=True)

    marker = _compile_cache_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    payload = {
        "schema": 1,
        "kind": "coldsnap-vllm-compile-cache",
        "namespace": relative.name,
    }
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)


def adjusted_plan_bytes(
    plan: dict[str, Any], current_free_memory: int, maximum_shortfall: int
) -> int | None:
    """Keep the recorded memory envelope when free memory drifts slightly."""
    kv_bytes = plan.get("kv_cache_memory_bytes")
    baseline = plan.get("free_memory_baseline")
    if (
        not isinstance(kv_bytes, int)
        or not isinstance(baseline, int)
        or not isinstance(current_free_memory, int)
        or kv_bytes <= 0
        or maximum_shortfall <= 0
    ):
        return None
    shortfall = baseline - current_free_memory
    if shortfall <= 0 or shortfall > maximum_shortfall or shortfall >= kv_bytes:
        return None
    return kv_bytes - shortfall


def _maximum_shortfall_bytes() -> int:
    raw = os.environ.get(MAX_SHORTFALL_ENV, "0")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{MAX_SHORTFALL_ENV} must be an integer") from error
    if value < 0:
        raise RuntimeError(f"{MAX_SHORTFALL_ENV} must not be negative")
    return value


def minimum_free_memory(baseline: int, reserve: int) -> int:
    if baseline <= 0 or reserve < 0:
        raise RuntimeError("captured free-memory admission values are invalid")
    return baseline + reserve


def _validate_free_memory_admission() -> None:
    raw = os.environ.get(REQUIRED_FREE_MEMORY_ENV)
    if raw is None:
        return
    try:
        baseline = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{REQUIRED_FREE_MEMORY_ENV} must be an integer") from error
    try:
        reserve = int(os.environ.get(FREE_MEMORY_RESERVE_ENV, "0"))
    except ValueError as error:
        raise RuntimeError(f"{FREE_MEMORY_RESERVE_ENV} must be an integer") from error
    required = minimum_free_memory(baseline, reserve)

    import torch

    current_free, _ = torch.cuda.mem_get_info()
    if int(current_free) < required:
        raise RuntimeError(
            "GPU unified-memory admission failed: "
            f"free={int(current_free)} required={required} baseline={baseline} "
            f"reserve={reserve}"
        )


def _install_free_memory_admission_hook(module: Any) -> None:
    worker_classes = []
    for name in ("Worker", "GPUWorker"):
        value = getattr(module, name, None)
        if isinstance(value, type) and value not in worker_classes:
            worker_classes.append(value)
    if len(worker_classes) != 1:
        raise RuntimeError("vLLM GPU worker exposes no unique worker class")
    worker_class = worker_classes[0]
    original = getattr(worker_class, "init_device", None)
    if not callable(original):
        raise RuntimeError("vLLM GPU worker init_device is unavailable")
    if getattr(original, _ADMISSION_MARKER, False):
        return

    @functools.wraps(original)
    def init_device_then_validate(worker: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(worker, *args, **kwargs)
        _validate_free_memory_admission()
        return result

    setattr(init_device_then_validate, _ADMISSION_MARKER, True)
    worker_class.init_device = init_device_then_validate


def _install_compile_cache_identity_hook(module: Any) -> None:
    """Keep capture and startup-plan restores in one compiler cache namespace.

    With automatic model length, an ordinary capture compiles before vLLM
    profiles memory and reduces ``max_model_len``.  A restore that applies a
    persisted startup plan skips profiling, auto-fits the model length first,
    and otherwise computes its compiler-cache key from that derived value.
    The derived scheduling limit does not change the graph captured for the
    original configured model limit.  Temporarily expose the configured value
    only while vLLM compiles or looks up its cache, then restore the fitted
    runtime value.
    """
    worker_classes = []
    for name in ("Worker", "GPUWorker"):
        value = getattr(module, name, None)
        if isinstance(value, type) and value not in worker_classes:
            worker_classes.append(value)
    if len(worker_classes) != 1:
        raise RuntimeError("vLLM GPU worker exposes no unique worker class")
    worker_class = worker_classes[0]
    original = getattr(worker_class, "compile_or_warm_up_model", None)
    if not callable(original):
        raise RuntimeError("vLLM GPU worker compile_or_warm_up_model is unavailable")
    if getattr(original, _COMPILE_IDENTITY_MARKER, False):
        return

    @functools.wraps(original)
    def compile_with_configured_model_len(worker: Any, *args: Any, **kwargs: Any) -> Any:
        vllm_config = getattr(worker, "vllm_config", None)
        compilation_config = getattr(vllm_config, "compilation_config", None)
        if compilation_config is None:
            raise RuntimeError("vLLM compilation configuration is unavailable")
        restoring = (
            os.environ.get(_PROCESS_TEMPLATE_RESTORE_LOAD_FORMAT_ENV, "")
            .strip()
            .lower()
            == "coldsnap"
            or os.environ.get(_PROCESS_TEMPLATE_RESTORED_ENV, "").strip() == "1"
        )
        if restoring:
            captured_cache_dir = _read_compile_cache_marker()
            if captured_cache_dir is not None:
                compilation_config.cache_dir = str(captured_cache_dir)

        configured = getattr(worker, _CONFIGURED_MODEL_LEN_ATTR, None)
        model_config = getattr(vllm_config, "model_config", None)
        current = getattr(model_config, "max_model_len", None)
        use_configured_model_len = (
            not isinstance(configured, int)
            or configured <= 0
            or not isinstance(current, int)
            or current <= 0
            or configured == current
        ) is False
        if use_configured_model_len:
            model_config.max_model_len = configured
        try:
            result = original(worker, *args, **kwargs)
        finally:
            if use_configured_model_len:
                model_config.max_model_len = current

        if os.environ.get(_CAPTURE_LOAD_FORMAT_ENV, "").strip():
            _write_compile_cache_marker(compilation_config.cache_dir)
        return result

    setattr(compile_with_configured_model_len, _COMPILE_IDENTITY_MARKER, True)
    worker_class.compile_or_warm_up_model = compile_with_configured_model_len


def _install_deferred_free_memory_admission() -> None:
    if os.environ.get(REQUIRED_FREE_MEMORY_ENV) is None:
        return
    after_module_import(
        _GPU_WORKER_MODULE,
        "startup-plan-free-memory-admission",
        _install_free_memory_admission_hook,
    )


def _install_deferred_compile_cache_identity() -> None:
    after_module_import(
        _GPU_WORKER_MODULE,
        "startup-plan-compile-cache-identity",
        _install_compile_cache_identity_hook,
    )


def _gb10_unified_memory_bytes(platform: Any) -> int:
    name = str(platform.get_device_name()).lower()
    if "gb10" not in name:
        raise RuntimeError("NVML did not expose device memory for a non-GB10 accelerator")
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    total = pages * page_size
    if pages <= 0 or page_size <= 0 or total <= 0:
        raise RuntimeError("host physical-memory discovery returned invalid values")
    return total


def _install_startup_plan_shortfall_adjustment(
    startup_plan: Any | None = None,
) -> None:
    if startup_plan is None:
        from vllm.v1.worker import startup_plan

    original = startup_plan._applicable_kv_cache_memory_bytes
    if getattr(original, "_coldsnap_shortfall_adjustment", False):
        return

    def applicable_kv_cache_memory_bytes(
        plan: dict[str, Any], current_free_memory: int
    ) -> int | None:
        # The hook is installed while the capture process is assembled, but
        # an exact-target pre-worker-import restore applies its bounded policy
        # later, at the activation barrier. Read the effective value at use
        # time rather than freezing capture-time policy in this closure.
        maximum_shortfall = _maximum_shortfall_bytes()
        adjusted = adjusted_plan_bytes(plan, current_free_memory, maximum_shortfall)
        if adjusted is None:
            return original(plan, current_free_memory)
        shortfall = plan["free_memory_baseline"] - current_free_memory
        startup_plan.logger.info(
            "Applying startup plan after reducing KV memory by %.2f GiB "
            "for a bounded free-memory shortfall (maximum %.2f GiB).",
            shortfall / (1 << 30),
            maximum_shortfall / (1 << 30),
        )
        return adjusted

    applicable_kv_cache_memory_bytes._coldsnap_shortfall_adjustment = True
    startup_plan._applicable_kv_cache_memory_bytes = applicable_kv_cache_memory_bytes


def _install_phase_stable_startup_plan_fingerprint(
    startup_plan_module: Any | None = None,
) -> None:
    """Key a saved plan with the same pre-profile state used at lookup.

    vLLM computes the lookup fingerprint before memory profiling, but computes
    the save fingerprint after profiling and cache initialization have mutated
    derived config state.  On the pinned GB10 build those fingerprints differ,
    making an otherwise valid plan impossible to find.  Preserve vLLM's exact
    fingerprint function and free-memory admission; only save an additional
    alias under the fingerprint observed by the same worker at lookup time.
    """

    if startup_plan_module is None:
        from vllm.v1.worker import startup_plan as startup_plan_module

    original_apply = startup_plan_module.maybe_apply_startup_plan
    original_save = startup_plan_module.maybe_save_startup_plan
    if getattr(original_apply, _PHASE_FINGERPRINT_MARKER, False):
        return

    def fingerprint(worker: Any) -> str:
        return startup_plan_module.compute_plan_fingerprint(
            worker.vllm_config,
            worker.rank,
            worker.parallel_config.world_size,
        )

    @functools.wraps(original_apply)
    def maybe_apply_startup_plan(worker: Any) -> Any:
        model_config = getattr(getattr(worker, "vllm_config", None), "model_config", None)
        configured_model_len = getattr(model_config, "max_model_len", None)
        if isinstance(configured_model_len, int) and configured_model_len > 0:
            setattr(worker, _CONFIGURED_MODEL_LEN_ATTR, configured_model_len)
        initial = fingerprint(worker)
        setattr(worker, _PHASE_FINGERPRINT_ATTR, initial)
        startup_plan_module.logger.info(
            "ColdSnap startup-plan lookup fingerprint %s", initial
        )
        cache_config = getattr(worker, "cache_config", None)
        configured_kv_bytes = getattr(cache_config, "kv_cache_memory_bytes", None)
        result = original_apply(worker)
        if configured_kv_bytes is not None:
            return result

        locally_admitted = (
            getattr(cache_config, "kv_cache_memory_bytes", None) is not None
        )
        if _all_workers_admit_startup_plan(worker, locally_admitted):
            return result

        if locally_admitted:
            cache_config.kv_cache_memory_bytes = None
        startup_plan_module.logger.info(
            "ColdSnap startup plan not applied because at least one distributed "
            "worker requires full memory profiling."
        )
        return result

    @functools.wraps(original_save)
    def maybe_save_startup_plan(worker: Any, kv_cache_memory_bytes: int) -> Any:
        result = original_save(worker, kv_cache_memory_bytes)
        initial = getattr(worker, _PHASE_FINGERPRINT_ATTR, "")
        final = fingerprint(worker)
        if not initial or initial == final:
            return result
        path = startup_plan_module._plan_path(initial)
        payload = {
            "schema": startup_plan_module.PLAN_SCHEMA_VERSION,
            "fingerprint": initial,
            "kv_cache_memory_bytes": int(kv_cache_memory_bytes),
            "free_memory_baseline": int(worker.init_snapshot.free_memory),
            "coldsnap_alias_of": final,
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = f"{path}.tmp.{os.getpid()}"
            with open(temporary, "w", encoding="utf-8") as output:
                json.dump(payload, output)
            os.replace(temporary, path)
        except OSError as error:
            startup_plan_module.logger.warning(
                "Failed to save phase-stable ColdSnap startup plan %s: %s",
                path,
                error,
            )
        else:
            startup_plan_module.logger.info(
                "Saved phase-stable ColdSnap startup plan %s "
                "(post-profile fingerprint %s)",
                path,
                final,
            )
        return result

    setattr(maybe_apply_startup_plan, _PHASE_FINGERPRINT_MARKER, True)
    setattr(maybe_save_startup_plan, _PHASE_FINGERPRINT_MARKER, True)
    startup_plan_module.maybe_apply_startup_plan = maybe_apply_startup_plan
    startup_plan_module.maybe_save_startup_plan = maybe_save_startup_plan


def install_startup_plan_memory_fallback() -> None:
    from vllm.platforms import current_platform
    from vllm.third_party.pynvml import NVMLError_NotSupported

    if hasattr(current_platform, "get_device_total_memory"):
        original = current_platform.get_device_total_memory
        if not getattr(original, "_coldsnap_gb10_fallback", False):

            def get_device_total_memory(*args: Any, **kwargs: Any) -> int:
                try:
                    return int(original(*args, **kwargs))
                except NVMLError_NotSupported:
                    return _gb10_unified_memory_bytes(current_platform)

            get_device_total_memory._coldsnap_gb10_fallback = True
            current_platform.get_device_total_memory = get_device_total_memory

    _install_phase_stable_startup_plan_fingerprint()
    _install_startup_plan_shortfall_adjustment()
    _install_deferred_free_memory_admission()
    _install_deferred_compile_cache_identity()

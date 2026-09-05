# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Deferred CUDA-graph capture for eager-first vLLM startup.

CUDA work remains serialized on vLLM's engine thread.  "Async" means that
server readiness and at least one complete eager request precede capture; it
does not mean that capture races inference on another Python thread.
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from coldsnap_import_hook import after_module_import
from coldsnap_vllm import VllmContractError
from coldsnap_vllm_shape_calibration import (
    calibrate_capture_shapes,
    shape_calibration_enabled,
)


ASYNC_GRAPH_ENV = "COLDSNAP_ASYNC_CUDA_GRAPHS"
READY_FILE_ENV = "COLDSNAP_ASYNC_CUDA_GRAPHS_READY_FILE"
GENERATION_ENV = "COLDSNAP_ASYNC_CUDA_GRAPHS_GENERATION"
ARM_FILE_ENV = "COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE"
_GRAPH_CAPTURE_RETRY_LIMIT = 1
_STALE_GRAPH_POOL_ERROR = "it->second->use_count > 0"

_FORCE_EAGER_ATTRIBUTE = "_coldsnap_force_eager"
_DISPATCH_MARKER = "_coldsnap_async_dispatch_gate"
_WORKER_MARKER = "_coldsnap_async_worker_hook"
_ENGINE_MARKER = "_coldsnap_async_engine_hook"
_WORKER_MODULE = "vllm.v1.worker.gpu_worker"
_ENGINE_MODULE = "vllm.v1.engine.core"

logger = logging.getLogger(__name__)
_installed = False


@dataclass(frozen=True)
class AsyncGraphSettings:
    enabled: bool = False
    ready_file: str = ""
    generation: str = ""
    arm_file: str = ""
    policy: str = "recreate-from-plan"


@dataclass
class WorkerGraphState:
    phase: str
    runner_mode: str
    capture_seconds: float = 0.0
    graph_memory_bytes: int = 0
    graph_memory_delta_bytes: int = 0
    error: str = ""
    capture_attempts: int = 0


def _boolean(value: str | None, *, name: str) -> bool:
    normalized = "0" if value is None else value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise VllmContractError(f"{name} must be a boolean, got {value!r}")


def async_graph_settings_from_env(
    environment: Mapping[str, str] | None = None,
) -> AsyncGraphSettings:
    values = os.environ if environment is None else environment
    enabled = _boolean(values.get(ASYNC_GRAPH_ENV), name=ASYNC_GRAPH_ENV)
    ready_file = values.get(READY_FILE_ENV, "").strip()
    generation = values.get(GENERATION_ENV, "").strip()
    arm_file = values.get(ARM_FILE_ENV, "").strip()
    policy = values.get("COLDSNAP_GRAPH_POLICY", "recreate-from-plan").strip()
    if policy not in {"recreate-from-plan", "preserve-nccl-exec", "disabled"}:
        raise VllmContractError(f"COLDSNAP_GRAPH_POLICY has unsupported value {policy!r}")
    if enabled and policy == "disabled":
        raise VllmContractError(f"{ASYNC_GRAPH_ENV}=1 is inconsistent with disabled graph policy")
    if enabled and bool(ready_file) != bool(generation):
        raise VllmContractError(f"{READY_FILE_ENV} and {GENERATION_ENV} must be set together")
    return AsyncGraphSettings(enabled, ready_file, generation, arm_file, policy)


def _publish_engine_status(
    engine: Any,
    settings: AsyncGraphSettings,
    statuses: list[dict[str, Any]] | None = None,
) -> None:
    if not settings.ready_file:
        return
    path = Path(settings.ready_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    default_phase = (
        "retained_ready" if settings.policy == "preserve-nccl-exec" else "eager"
    )
    payload = {
        "schema": 1,
        "kind": "coldsnap-async-cuda-graphs",
        "generation": settings.generation,
        "pid": os.getpid(),
        "phase": getattr(engine, "_coldsnap_async_graph_phase", default_phase),
        "eager_steps": getattr(engine, "_coldsnap_async_graph_eager_steps", 0),
        "error": getattr(engine, "_coldsnap_async_graph_error", ""),
        "statuses": statuses
        if statuses is not None
        else getattr(engine, "_coldsnap_async_graph_statuses", []),
        "eager_step_timings": getattr(engine, "_coldsnap_async_graph_step_timings", []),
        "published_monotonic_ns": time.monotonic_ns(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _wrap_dispatch_gate(owner: type[Any], state_attribute: str) -> None:
    original = getattr(owner, "dispatch", None)
    if not callable(original):
        raise VllmContractError(f"{owner.__name__}.dispatch is unavailable")
    if getattr(original, _DISPATCH_MARKER, False):
        return

    @functools.wraps(original)
    def dispatch_eager_when_gated(instance: Any, *args: Any, **kwargs: Any) -> Any:
        if not getattr(instance, _FORCE_EAGER_ATTRIBUTE, False):
            return original(instance, *args, **kwargs)
        if not hasattr(instance, state_attribute):
            raise VllmContractError(f"{owner.__name__} lacks runtime state {state_attribute}")
        previous = getattr(instance, state_attribute)
        setattr(instance, state_attribute, False)
        try:
            return original(instance, *args, **kwargs)
        finally:
            setattr(instance, state_attribute, previous)

    setattr(dispatch_eager_when_gated, _DISPATCH_MARKER, True)
    owner.dispatch = dispatch_eager_when_gated


_DISPATCH_TARGETS = (
    (
        "vllm.v1.cudagraph_dispatcher",
        "CudagraphDispatcher",
        "keys_initialized",
    ),
    (
        "vllm.v1.worker.gpu.cudagraph_utils",
        "CudaGraphManager",
        "_graphs_captured",
    ),
)


def _install_dispatch_gate(
    module: Any,
    *,
    module_name: str,
    class_name: str,
    state_attribute: str,
) -> None:
    owner = getattr(module, class_name, None)
    if not isinstance(owner, type):
        raise VllmContractError(f"{module_name} lacks {class_name}")
    _wrap_dispatch_gate(owner, state_attribute)


def _schedule_dispatch_gates() -> None:
    for module_name, class_name, state_attribute in _DISPATCH_TARGETS:
        after_module_import(
            module_name,
            f"async-graph-dispatch-{class_name}",
            lambda module, module_name=module_name, class_name=class_name, state_attribute=state_attribute: (
                _install_dispatch_gate(
                    module,
                    module_name=module_name,
                    class_name=class_name,
                    state_attribute=state_attribute,
                )
            ),
        )


def _runner_gate_targets(runner: Any) -> tuple[tuple[str, Any], ...]:
    candidates = (
        ("dispatcher", getattr(runner, "cudagraph_dispatcher", None)),
        ("manager", getattr(runner, "cudagraph_manager", None)),
    )
    targets: list[tuple[str, Any]] = []
    for name, target in candidates:
        if target is None:
            continue
        dispatch = getattr(type(target), "dispatch", None)
        if not callable(dispatch) or not getattr(dispatch, _DISPATCH_MARKER, False):
            raise VllmContractError(f"vLLM CUDA graph {name} lacks the coldsnap eager gate")
        state_attribute = "keys_initialized" if name == "dispatcher" else "_graphs_captured"
        if not hasattr(target, state_attribute):
            raise VllmContractError(f"vLLM CUDA graph {name} lacks {state_attribute}")
        targets.append((name, target))
    if not targets:
        raise VllmContractError(
            "vLLM model runner exposes neither cudagraph_dispatcher nor cudagraph_manager"
        )
    return tuple(targets)


def _force_runner_eager(runner: Any, enabled: bool) -> str:
    targets = _runner_gate_targets(runner)
    for _, target in targets:
        setattr(target, _FORCE_EAGER_ATTRIBUTE, enabled)
    return "+".join(name for name, _ in targets)


def _worker_status(worker: Any) -> dict[str, Any]:
    world_size = int(os.environ.get("COLDSNAP_WORLD_SIZE", "1"))
    requested = os.environ.get("COLDSNAP_GRAPH_POLICY_REQUESTED", "recreate-from-plan")
    effective = os.environ.get("COLDSNAP_GRAPH_POLICY", "recreate-from-plan")
    decision = "requested_policy_is_supported"
    if requested == "preserve-nccl-exec":
        decision = "experimental_exact_nccl_in_place"
    elif requested == "preserve-exec":
        decision = (
            "provider_reconstructs_communicator"
            if world_size > 1
            else "graph_exec_preservation_is_unqualified"
        )
    audit = {
        "format": 1,
        "policy_requested": requested,
        "policy_effective": effective,
        "dependencies": {
            "nccl_communicators": 1 if world_size > 1 else 0,
            "registered_windows": 0,
            "device_api": False,
            "external_semaphores": 0,
            "provider_owned_allocations": 0,
            "unknown": effective not in {"disabled", "preserve-nccl-exec"},
        },
        "decision": decision,
    }
    state = getattr(worker, "_coldsnap_async_graph_state", None)
    if state is None:
        return {
            "phase": "uninitialized",
            "runner_mode": "",
            "capture_seconds": 0.0,
            "graph_memory_bytes": 0,
            "graph_memory_delta_bytes": 0,
            "error": "",
            "graph_resource_audit": audit,
        }
    return {
        "phase": state.phase,
        "runner_mode": state.runner_mode,
        "capture_seconds": state.capture_seconds,
        "graph_memory_bytes": state.graph_memory_bytes,
        "graph_memory_delta_bytes": state.graph_memory_delta_bytes,
        "error": state.error,
        "capture_attempts": state.capture_attempts,
        "shape_calibration": getattr(worker, "_coldsnap_shape_calibration", None),
        "graph_resource_audit": audit,
    }


def _validate_worker(worker: Any) -> Any:
    model_config = getattr(worker, "model_config", None)
    if model_config is None:
        raise VllmContractError("GPU worker lacks model_config")
    if bool(getattr(model_config, "enforce_eager", False)):
        raise VllmContractError(f"{ASYNC_GRAPH_ENV}=1 requires vLLM enforce_eager=False")
    parallel_config = getattr(worker, "parallel_config", None)
    if parallel_config is None:
        raise VllmContractError("GPU worker lacks parallel_config")
    if int(getattr(parallel_config, "data_parallel_size", 1)) != 1:
        raise VllmContractError("deferred CUDA graph capture does not yet support data parallelism")
    compilation_config = getattr(worker, "compilation_config", None)
    mode = getattr(compilation_config, "cudagraph_mode", None)
    if mode is None or getattr(mode, "name", str(mode)) == "NONE":
        raise VllmContractError(f"{ASYNC_GRAPH_ENV}=1 requires a non-NONE cudagraph_mode")
    runner = getattr(worker, "model_runner", None)
    if runner is None or not callable(getattr(runner, "capture_model", None)):
        raise VllmContractError("GPU worker model runner cannot capture CUDA graphs")
    return runner


def _compile_eager_first(
    worker: Any,
    original: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    runner = _validate_worker(worker)
    runner_mode = _force_runner_eager(runner, True)
    state = WorkerGraphState("initializing", runner_mode)
    worker._coldsnap_async_graph_state = state

    instance_attributes = vars(runner)
    had_instance_capture = "capture_model" in instance_attributes
    previous_instance_capture = instance_attributes.get("capture_model")
    deferred_calls = 0

    def defer_capture(*capture_args: Any, **capture_kwargs: Any) -> int:
        nonlocal deferred_calls
        del capture_args, capture_kwargs
        deferred_calls += 1
        logger.info("Deferring CUDA graph capture; serving will start eagerly")
        return 0

    runner.capture_model = defer_capture
    try:
        result = original(worker, *args, **kwargs)
    except BaseException as error:
        state.phase = "failed"
        state.error = f"startup warmup failed: {error}"
        raise
    finally:
        if had_instance_capture:
            runner.capture_model = previous_instance_capture
        else:
            del runner.capture_model

    if deferred_calls != 1:
        state.phase = "failed"
        state.error = "compile_or_warm_up_model did not invoke capture_model exactly once"
        raise VllmContractError(state.error)

    # Compile the graph-capture shapes eagerly while this process is still the
    # one that will be snapshotted. Entries written to the compile cache after a
    # snapshot are not usable by the restored process, so deferring capture
    # without this leaves those kernels to be recompiled on every activation.
    if shape_calibration_enabled():
        try:
            worker._coldsnap_shape_calibration = calibrate_capture_shapes(runner)
        except BaseException as error:
            state.phase = "failed"
            state.error = f"shape calibration failed: {error}"
            raise

    state.phase = "eager"
    return result


def _compile_retained_first(
    worker: Any,
    original: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Capture the engine's normal graph set before the process snapshot."""

    runner = _validate_worker(worker)
    runner_mode = "+".join(name for name, _ in _runner_gate_targets(runner))
    state = WorkerGraphState("capturing-retained", runner_mode, capture_attempts=1)
    worker._coldsnap_async_graph_state = state
    started = time.perf_counter()
    try:
        result = original(worker, *args, **kwargs)
        if shape_calibration_enabled():
            worker._coldsnap_shape_calibration = calibrate_capture_shapes(runner)
    except BaseException as error:
        state.phase = "failed"
        state.capture_seconds = time.perf_counter() - started
        state.error = f"retained startup graph capture failed: {error}"
        raise
    state.phase = "retained_ready"
    state.capture_seconds = time.perf_counter() - started
    logger.info(
        "Retained CUDA graph capture ready in %.3f s for exact NCCL reattachment",
        state.capture_seconds,
    )
    return result


def _capture_worker_graphs(worker: Any) -> dict[str, Any]:
    state = getattr(worker, "_coldsnap_async_graph_state", None)
    if not isinstance(state, WorkerGraphState):
        raise VllmContractError("deferred CUDA graph worker state is unavailable")
    if state.phase != "eager":
        return _worker_status(worker)

    runner = _validate_worker(worker)
    state.phase = "capturing"
    state.error = ""
    state.capture_attempts += 1
    _force_runner_eager(runner, False)
    started = time.perf_counter()
    try:
        _refresh_worker_graph_pools(runner)
        graph_memory_delta_bytes = int(runner.capture_model())
    except BaseException as error:
        _force_runner_eager(runner, True)
        state.phase = "failed"
        state.capture_seconds = time.perf_counter() - started
        state.error = f"{type(error).__name__}: {error}"
        logger.exception("Deferred CUDA graph capture failed; retaining eager mode")
        return _worker_status(worker)

    state.phase = "ready"
    state.capture_seconds = time.perf_counter() - started
    state.graph_memory_delta_bytes = graph_memory_delta_bytes
    state.graph_memory_bytes = max(graph_memory_delta_bytes, 0)
    logger.info(
        "Deferred CUDA graph capture ready in %.3f s (memory delta %d bytes)",
        state.capture_seconds,
        state.graph_memory_delta_bytes,
    )
    return _worker_status(worker)


def _refresh_worker_graph_pools(runner: Any) -> None:
    """Replace graph-pool tokens from the pre-snapshot CUDA allocator epoch."""

    import torch
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.distributed.device_communicators.pynccl_allocator import (
        set_graph_pool_id,
    )
    from vllm.platforms import current_platform

    torch.accelerator.synchronize()
    managers = []
    manager = getattr(runner, "cudagraph_manager", None)
    if manager is not None:
        managers.append(manager)
    speculator = getattr(runner, "speculator", None)
    get_managers = getattr(speculator, "get_cudagraph_managers", None)
    if callable(get_managers):
        for value in get_managers():
            if value is not None and value not in managers:
                managers.append(value)

    for value in managers:
        clear = getattr(value, "clear", None)
        if not callable(clear):
            raise VllmContractError("vLLM CUDA graph manager lacks clear()")
        clear()
    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()

    platform_type = type(current_platform)
    if not hasattr(platform_type, "_global_graph_pool"):
        raise VllmContractError("vLLM platform lacks a global CUDA graph pool")
    platform_type._global_graph_pool = None
    pool = current_platform.get_global_graph_pool()
    if pool is None:
        raise VllmContractError("vLLM returned no CUDA graph pool handle")

    for wrapper_type in (CUDAGraphWrapper, BreakableCUDAGraphWrapper):
        instances = getattr(wrapper_type, "_all_instances", None)
        if instances is None:
            raise VllmContractError(f"{wrapper_type.__name__} lacks graph-wrapper inventory")
        for wrapper in list(instances):
            wrapper.graph_pool = pool
    for value in managers:
        value.pool = pool
    set_graph_pool_id(pool)


def _prepare_worker_graph_retry(worker: Any, reason: str = "") -> dict[str, Any]:
    state = getattr(worker, "_coldsnap_async_graph_state", None)
    if not isinstance(state, WorkerGraphState):
        raise VllmContractError("deferred CUDA graph worker state is unavailable")
    runner = _validate_worker(worker)
    _force_runner_eager(runner, True)
    state.phase = "eager"
    state.error = reason
    return _worker_status(worker)


def _fallback_worker_to_eager(worker: Any, reason: str = "") -> dict[str, Any]:
    state = getattr(worker, "_coldsnap_async_graph_state", None)
    if not isinstance(state, WorkerGraphState):
        raise VllmContractError("deferred CUDA graph worker state is unavailable")
    runner = _validate_worker(worker)
    _force_runner_eager(runner, True)
    state.phase = "fallback_eager"
    if reason:
        state.error = reason
    return _worker_status(worker)


def _install_worker_hook(settings: AsyncGraphSettings, module: Any | None = None) -> None:
    module = module or importlib.import_module(_WORKER_MODULE)
    worker_classes: list[type[Any]] = []
    for name in ("Worker", "GPUWorker"):
        value = getattr(module, name, None)
        if isinstance(value, type) and value not in worker_classes:
            worker_classes.append(value)
    if len(worker_classes) != 1:
        raise VllmContractError("vllm.v1.worker.gpu_worker must expose one supported worker class")
    worker_class = worker_classes[0]
    original = getattr(worker_class, "compile_or_warm_up_model", None)
    if not callable(original):
        raise VllmContractError("GPU worker compile_or_warm_up_model is unavailable")
    if getattr(original, _WORKER_MARKER, False):
        return
    for name in (
        "coldsnap_capture_cuda_graphs",
        "coldsnap_prepare_cuda_graph_retry",
        "coldsnap_fallback_to_eager_graphs",
        "coldsnap_async_graph_status",
    ):
        if hasattr(worker_class, name):
            raise VllmContractError(f"GPU worker already defines {name}")

    @functools.wraps(original)
    def compile_or_warm_up_eager_first(worker: Any, *args: Any, **kwargs: Any) -> Any:
        if not settings.enabled:
            return original(worker, *args, **kwargs)
        if settings.policy == "preserve-nccl-exec":
            return _compile_retained_first(worker, original, *args, **kwargs)
        return _compile_eager_first(worker, original, *args, **kwargs)

    setattr(compile_or_warm_up_eager_first, _WORKER_MARKER, True)
    worker_class.compile_or_warm_up_model = compile_or_warm_up_eager_first
    worker_class.coldsnap_capture_cuda_graphs = _capture_worker_graphs
    worker_class.coldsnap_prepare_cuda_graph_retry = _prepare_worker_graph_retry
    worker_class.coldsnap_fallback_to_eager_graphs = _fallback_worker_to_eager
    worker_class.coldsnap_async_graph_status = _worker_status


def _engine_is_idle(engine: Any) -> bool:
    if engine.scheduler.has_requests():
        return False
    batch_queue = getattr(engine, "batch_queue", None)
    if batch_queue:
        return False
    input_queue = getattr(engine, "input_queue", None)
    return input_queue is None or input_queue.empty()


def _fallback_engine(engine: Any, reason: str) -> None:
    engine.model_executor.collective_rpc("coldsnap_fallback_to_eager_graphs", args=(reason,))
    engine._coldsnap_async_graph_phase = "fallback_eager"
    engine._coldsnap_async_graph_error = reason


def _retryable_graph_capture_failure(statuses: list[dict[str, Any]]) -> bool:
    failures = [status for status in statuses if status.get("phase") != "ready"]
    return bool(failures) and all(
        _STALE_GRAPH_POOL_ERROR in str(status.get("error", "")) for status in failures
    )


def _maybe_capture_after_step(
    engine: Any,
    model_executed: bool,
    settings: AsyncGraphSettings,
) -> None:
    default_phase = "retained_ready" if settings.policy == "preserve-nccl-exec" else "eager"
    phase = getattr(engine, "_coldsnap_async_graph_phase", default_phase)
    if phase == "retained_ready" and settings.arm_file and Path(settings.arm_file).is_file():
        # The controller uses the existing arm marker only when retained graph
        # admission failed but the engine survived. The worker RPC has already
        # gated graph dispatch to eager for this just-completed step; re-enter
        # the qualified scheduler-idle recapture path now.
        phase = "eager"
        engine._coldsnap_async_graph_phase = phase
        engine._coldsnap_async_graph_error = (
            "retained graph validation failed; rebuilding from the engine plan"
        )
    if phase != "eager":
        return
    warmup_phase = getattr(engine, "_coldsnap_deferred_warmup_phase", "ready")
    if warmup_phase in {"eager", "warming"}:
        return
    if warmup_phase == "failed":
        reason = "deferred warmup failed; CUDA graph capture was not attempted"
        _fallback_engine(engine, reason)
        _publish_engine_status(engine, settings)
        logger.error(reason)
        return
    if model_executed:
        engine._coldsnap_async_graph_eager_steps = (
            getattr(engine, "_coldsnap_async_graph_eager_steps", 0) + 1
        )
    if settings.arm_file and not Path(settings.arm_file).is_file():
        return
    eager_steps = getattr(engine, "_coldsnap_async_graph_eager_steps", 0)
    if eager_steps < 1 or not _engine_is_idle(engine):
        return

    engine._coldsnap_async_graph_phase = "capturing"
    logger.info(
        "Starting scheduler-idle CUDA graph capture after %d eager steps",
        eager_steps,
    )
    try:
        statuses = engine.model_executor.collective_rpc("coldsnap_capture_cuda_graphs")
    except BaseException as error:
        reason = f"collective CUDA graph capture failed: {type(error).__name__}: {error}"
        try:
            _fallback_engine(engine, reason)
        except BaseException as rollback_error:
            raise RuntimeError(
                "CUDA graph capture and eager rollback both failed"
            ) from rollback_error
        _publish_engine_status(engine, settings)
        logger.exception("%s; all workers returned to eager dispatch", reason)
        return

    failed = [status for status in statuses if status.get("phase") != "ready"]
    if failed:
        reason = f"one or more CUDA graph workers failed capture: {failed}"
        retries = int(getattr(engine, "_coldsnap_async_graph_retries", 0))
        if retries < _GRAPH_CAPTURE_RETRY_LIMIT and _retryable_graph_capture_failure(statuses):
            try:
                retry_statuses = engine.model_executor.collective_rpc(
                    "coldsnap_prepare_cuda_graph_retry", args=(reason,)
                )
            except BaseException as error:
                reason = (
                    f"{reason}; CUDA graph retry preparation failed: "
                    f"{type(error).__name__}: {error}"
                )
            else:
                engine._coldsnap_async_graph_retries = retries + 1
                engine._coldsnap_async_graph_phase = "eager"
                engine._coldsnap_async_graph_error = reason
                engine._coldsnap_async_graph_statuses = retry_statuses
                _publish_engine_status(engine, settings, retry_statuses)
                logger.warning(
                    "Deferred CUDA graph capture used a stale allocator pool; "
                    "refreshed every worker and will retry once while idle"
                )
                return
        _fallback_engine(engine, reason)
        engine._coldsnap_async_graph_statuses = statuses
        _publish_engine_status(engine, settings, statuses)
        logger.error("%s; all workers returned to eager dispatch", reason)
        return
    engine._coldsnap_async_graph_phase = "ready"
    engine._coldsnap_async_graph_statuses = statuses
    _publish_engine_status(engine, settings, statuses)
    logger.info("All CUDA graph workers transitioned from eager to replay mode")


def _install_engine_hook(settings: AsyncGraphSettings, module: Any | None = None) -> None:
    module = module or importlib.import_module(_ENGINE_MODULE)
    engine_class = getattr(module, "EngineCoreProc", None)
    if not isinstance(engine_class, type):
        raise VllmContractError("vllm.v1.engine.core lacks EngineCoreProc")
    original = getattr(engine_class, "_process_engine_step", None)
    if not callable(original):
        raise VllmContractError("EngineCoreProc._process_engine_step is unavailable")
    if getattr(original, _ENGINE_MARKER, False):
        return

    @functools.wraps(original)
    def process_step_then_capture(engine: Any, *args: Any, **kwargs: Any) -> bool:
        if not settings.enabled:
            return bool(original(engine, *args, **kwargs))
        started_ns = time.time_ns()
        started = time.perf_counter()
        model_executed = bool(original(engine, *args, **kwargs))
        wall_seconds = time.perf_counter() - started
        completed_ns = time.time_ns()
        if model_executed:
            timings = getattr(engine, "_coldsnap_async_graph_step_timings", None)
            if not isinstance(timings, list):
                timings = []
                engine._coldsnap_async_graph_step_timings = timings
            if len(timings) < 256:
                timings.append(
                    {
                        "index": len(timings),
                        "started_unix_ns": started_ns,
                        "completed_unix_ns": completed_ns,
                        "wall_seconds": wall_seconds,
                    }
                )
        _maybe_capture_after_step(engine, model_executed, settings)
        return model_executed

    setattr(process_step_then_capture, _ENGINE_MARKER, True)
    engine_class._process_engine_step = process_step_then_capture


def install_async_graph_capture_hooks(
    settings: AsyncGraphSettings | None = None,
) -> AsyncGraphSettings:
    """Install opt-in eager-first, scheduler-idle graph capture hooks."""
    global _installed
    resolved = settings or async_graph_settings_from_env()
    if not resolved.enabled:
        return resolved
    if _installed:
        return resolved
    _schedule_dispatch_gates()
    after_module_import(
        _WORKER_MODULE,
        "async-graph-worker",
        lambda module: _install_worker_hook(resolved, module),
    )
    after_module_import(
        _ENGINE_MODULE,
        "async-graph-engine",
        lambda module: _install_engine_hook(resolved, module),
    )
    _installed = True
    return resolved

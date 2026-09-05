# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Defer artifact-redundant vLLM warmup until after one eager request.

Memory sizing is never guessed: ``profile_run`` is deferred only when vLLM
already has an explicit ``kv_cache_memory_bytes`` value (normally supplied by
a validated startup-plan artifact).  The deferred work runs collectively on
the engine thread while the scheduler is idle, before deferred CUDA-graph
capture, and leaves multimodal profiling enabled.
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


DEFERRED_WARMUP_ENV = "COLDSNAP_DEFERRED_WARMUP"
FULLY_WARM_FILE_ENV = "COLDSNAP_FULLY_WARM_FILE"
WARMUP_GENERATION_ENV = "COLDSNAP_WARMUP_GENERATION"
WARMUP_ARM_FILE_ENV = "COLDSNAP_DEFERRED_WARMUP_ARM_FILE"

_DETERMINE_MARKER = "_coldsnap_deferred_profile_hook"
_KERNEL_MARKER = "_coldsnap_deferred_kernel_hook"
_ENGINE_MARKER = "_coldsnap_deferred_warmup_engine_hook"
_WORKER_MODULE = "vllm.v1.worker.gpu_worker"
_ENGINE_MODULE = "vllm.v1.engine.core"

logger = logging.getLogger(__name__)
_installed = False


@dataclass(frozen=True)
class DeferredWarmupSettings:
    enabled: bool = False
    fully_warm_file: Path | None = None
    generation: str = ""
    arm_file: Path | None = None


@dataclass
class WorkerWarmupState:
    phase: str = "startup"
    profile_deferred: bool = False
    kernel_deferred: bool = False
    runtime_kernel_deferred: bool = False
    warmup_seconds: float = 0.0
    error: str = ""


def _boolean(value: str | None, *, name: str) -> bool:
    normalized = "0" if value is None else value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise VllmContractError(f"{name} must be a boolean, got {value!r}")


def deferred_warmup_settings_from_env(
    environment: Mapping[str, str] | None = None,
) -> DeferredWarmupSettings:
    values = os.environ if environment is None else environment
    enabled = _boolean(values.get(DEFERRED_WARMUP_ENV), name=DEFERRED_WARMUP_ENV)
    raw_file = values.get(FULLY_WARM_FILE_ENV)
    raw_arm = values.get(WARMUP_ARM_FILE_ENV)
    return DeferredWarmupSettings(
        enabled=enabled,
        fully_warm_file=Path(raw_file) if raw_file else None,
        generation=values.get(WARMUP_GENERATION_ENV, ""),
        arm_file=Path(raw_arm) if raw_arm else None,
    )


def _effective_settings(captured: DeferredWarmupSettings) -> DeferredWarmupSettings:
    """Resolve target policy for processes restored from a capture-time hook."""
    from coldsnap_vllm_process_template import (
        _restored_runtime_environment,
        process_template_settings_from_env,
    )

    restored = _restored_runtime_environment(process_template_settings_from_env())
    if restored is None:
        return captured
    environment = dict(os.environ)
    environment.update(restored)
    return deferred_warmup_settings_from_env(environment)


def _effective_engine_settings(
    engine: Any, captured: DeferredWarmupSettings
) -> DeferredWarmupSettings:
    cached = getattr(engine, "_coldsnap_deferred_warmup_settings", None)
    if isinstance(cached, DeferredWarmupSettings):
        return cached
    effective = _effective_settings(captured)
    # The restored generation is immutable. Cache only target-selected
    # settings; capture-time disabled settings must not mask a later CRIU
    # restore handoff in this same process image.
    if effective is not captured:
        engine._coldsnap_deferred_warmup_settings = effective
    return effective


def _has_explicit_kv_artifact(worker: Any) -> bool:
    kv_bytes = getattr(
        getattr(worker, "cache_config", None), "kv_cache_memory_bytes", None
    )
    return isinstance(kv_bytes, int) and not isinstance(kv_bytes, bool) and kv_bytes > 0


def _state(worker: Any) -> WorkerWarmupState:
    state = getattr(worker, "_coldsnap_deferred_warmup_state", None)
    if isinstance(state, WorkerWarmupState):
        return state
    state = WorkerWarmupState()
    worker._coldsnap_deferred_warmup_state = state
    return state


def _worker_status(worker: Any) -> dict[str, Any]:
    state = _state(worker)
    return {
        "phase": state.phase,
        "profile_deferred": state.profile_deferred,
        "kernel_deferred": state.kernel_deferred,
        "runtime_kernel_deferred": state.runtime_kernel_deferred,
        "warmup_seconds": state.warmup_seconds,
        "error": state.error,
    }


def _determine_with_deferred_profile(
    worker: Any,
    original: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    runner = getattr(worker, "model_runner", None)
    profile_run = getattr(runner, "profile_run", None)
    if not callable(profile_run):
        raise VllmContractError("GPU model runner profile_run is unavailable")
    instance_attributes = vars(runner)
    had_instance_profile = "profile_run" in instance_attributes
    previous_instance_profile = instance_attributes.get("profile_run")

    def profile_or_defer(*profile_args: Any, **profile_kwargs: Any) -> Any:
        kv_bytes = getattr(
            getattr(worker, "cache_config", None), "kv_cache_memory_bytes", None
        )
        if not _has_explicit_kv_artifact(worker):
            logger.info(
                "COLDSNAP retained synchronous profile_run because vLLM has no "
                "explicit KV-cache memory artifact"
            )
            return profile_run(*profile_args, **profile_kwargs)
        state = _state(worker)
        state.profile_deferred = True
        state.phase = "eager"
        logger.info(
            "Deferring vLLM profile_run with explicit KV cache size %d bytes; "
            "multimodal and language profiling will run after the first request",
            kv_bytes,
        )
        return None

    runner.profile_run = profile_or_defer
    try:
        return original(worker, *args, **kwargs)
    finally:
        if had_instance_profile:
            runner.profile_run = previous_instance_profile
        else:
            del runner.profile_run


def _run_deferred_warmup(
    worker: Any,
    original_kernel_warmup: Callable[..., Any],
    original_runtime_kernel_warmup: Callable[..., Any],
) -> dict[str, Any]:
    state = _state(worker)
    if state.phase in {"ready", "failed"}:
        return _worker_status(worker)
    if not (
        state.profile_deferred
        or state.kernel_deferred
        or state.runtime_kernel_deferred
    ):
        state.phase = "ready"
        return _worker_status(worker)

    state.phase = "warming"
    started = time.perf_counter()
    try:
        if state.profile_deferred:
            worker.model_runner.profile_run()
        runtime_complete = False
        if state.kernel_deferred:
            runtime_complete = bool(original_kernel_warmup(worker))
        if state.runtime_kernel_deferred and not runtime_complete:
            original_runtime_kernel_warmup(worker)
    except Exception as error:
        state.phase = "failed"
        state.error = f"{type(error).__name__}: {error}"
        logger.exception("Deferred vLLM warmup failed")
    else:
        state.phase = "ready"
    state.warmup_seconds = time.perf_counter() - started
    return _worker_status(worker)


def _worker_class(module: Any) -> type[Any]:
    classes: list[type[Any]] = []
    for name in ("Worker", "GPUWorker"):
        value = getattr(module, name, None)
        if isinstance(value, type) and value not in classes:
            classes.append(value)
    if len(classes) != 1:
        raise VllmContractError(
            "vllm.v1.worker.gpu_worker must expose one supported worker class"
        )
    return classes[0]


def _install_worker_hooks(
    settings: DeferredWarmupSettings, module: Any | None = None
) -> None:
    module = module or importlib.import_module(_WORKER_MODULE)
    worker_class = _worker_class(module)
    original_determine = getattr(worker_class, "determine_available_memory", None)
    original_kernel = getattr(module, "kernel_warmup", None)
    original_runtime_kernel = getattr(module, "runtime_kernel_warmup", None)
    if not (
        callable(original_determine)
        and callable(original_kernel)
        and callable(original_runtime_kernel)
    ):
        raise VllmContractError(
            "GPU worker lacks determine_available_memory or kernel warmup hooks"
        )
    if not getattr(original_determine, _DETERMINE_MARKER, False):

        @functools.wraps(original_determine)
        def determine_available_memory(worker: Any, *args: Any, **kwargs: Any) -> Any:
            effective = _effective_settings(settings)
            if not effective.enabled:
                return original_determine(worker, *args, **kwargs)
            return _determine_with_deferred_profile(
                worker, original_determine, *args, **kwargs
            )

        setattr(determine_available_memory, _DETERMINE_MARKER, True)
        worker_class.determine_available_memory = determine_available_memory

    if not getattr(original_kernel, _KERNEL_MARKER, False):

        @functools.wraps(original_kernel)
        def kernel_warmup(worker: Any, *args: Any, **kwargs: Any) -> Any:
            effective = _effective_settings(settings)
            if not effective.enabled or not _has_explicit_kv_artifact(worker):
                return original_kernel(worker, *args, **kwargs)
            state = _state(worker)
            state.kernel_deferred = True
            state.phase = "eager"
            logger.info(
                "Deferring vLLM kernel warmup until after the first eager request"
            )
            # Older vLLM branches interpret False as "runtime warmup remains".
            return False

        setattr(kernel_warmup, _KERNEL_MARKER, True)
        module.kernel_warmup = kernel_warmup

    original_runtime = getattr(module, "runtime_kernel_warmup", None)
    if not getattr(original_runtime, _KERNEL_MARKER, False):

        @functools.wraps(original_runtime_kernel)
        def runtime_kernel_warmup(worker: Any, *args: Any, **kwargs: Any) -> Any:
            effective = _effective_settings(settings)
            if not effective.enabled or not _has_explicit_kv_artifact(worker):
                return original_runtime_kernel(worker, *args, **kwargs)
            state = _state(worker)
            state.runtime_kernel_deferred = True
            state.phase = "eager"
            logger.info(
                "Deferring vLLM runtime-dependent kernel warmup until after "
                "the first accepted request"
            )
            return None

        setattr(runtime_kernel_warmup, _KERNEL_MARKER, True)
        module.runtime_kernel_warmup = runtime_kernel_warmup

    for name in ("coldsnap_run_deferred_warmup", "coldsnap_deferred_warmup_status"):
        if hasattr(worker_class, name):
            raise VllmContractError(f"GPU worker already defines {name}")
    worker_class.coldsnap_run_deferred_warmup = functools.partialmethod(
        _run_deferred_warmup,
        original_kernel_warmup=original_kernel,
        original_runtime_kernel_warmup=original_runtime_kernel,
    )
    worker_class.coldsnap_deferred_warmup_status = _worker_status


def _engine_is_idle(engine: Any) -> bool:
    if engine.scheduler.has_requests():
        return False
    batch_queue = getattr(engine, "batch_queue", None)
    if batch_queue:
        return False
    input_queue = getattr(engine, "input_queue", None)
    return input_queue is None or input_queue.empty()


def _reset_prefix_cache(engine: Any) -> None:
    reset = getattr(engine.scheduler, "reset_prefix_cache", None)
    if not callable(reset):
        raise VllmContractError(
            "scheduler reset_prefix_cache is required after deferred profile_run"
        )
    if reset() is False:
        raise RuntimeError("failed to reset prefix cache after deferred warmup")


def _write_fully_warm_file(
    settings: DeferredWarmupSettings, statuses: list[dict[str, Any]]
) -> None:
    path = settings.fully_warm_file
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "generation": settings.generation,
        "ready_monotonic": time.monotonic(),
        "workers": statuses,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _maybe_warm_after_step(
    engine: Any,
    model_executed: bool,
    settings: DeferredWarmupSettings,
) -> None:
    phase = getattr(engine, "_coldsnap_deferred_warmup_phase", "eager")
    if phase != "eager":
        return
    if model_executed:
        engine._coldsnap_deferred_warmup_eager_steps = (
            getattr(engine, "_coldsnap_deferred_warmup_eager_steps", 0) + 1
        )
    eager_steps = getattr(engine, "_coldsnap_deferred_warmup_eager_steps", 0)
    if eager_steps < 1:
        return
    if settings.arm_file is not None and not settings.arm_file.is_file():
        return
    if not _engine_is_idle(engine):
        return

    engine._coldsnap_deferred_warmup_phase = "warming"
    logger.info(
        "Starting scheduler-idle deferred warmup after %d eager steps", eager_steps
    )
    try:
        statuses = engine.model_executor.collective_rpc(
            "coldsnap_run_deferred_warmup"
        )
        _reset_prefix_cache(engine)
    except Exception as error:
        engine._coldsnap_deferred_warmup_phase = "failed"
        engine._coldsnap_deferred_warmup_error = (
            f"{type(error).__name__}: {error}"
        )
        _write_fully_warm_file(
            settings,
            [
                {
                    "phase": "failed",
                    "profile_deferred": True,
                    "kernel_deferred": True,
                    "warmup_seconds": 0.0,
                    "error": engine._coldsnap_deferred_warmup_error,
                }
            ],
        )
        logger.exception("Collective deferred warmup failed")
        return

    failed = [status for status in statuses if status.get("phase") != "ready"]
    if failed:
        engine._coldsnap_deferred_warmup_phase = "failed"
        engine._coldsnap_deferred_warmup_error = str(failed)
        _write_fully_warm_file(settings, statuses)
        logger.error("One or more deferred-warmup workers failed: %s", failed)
        return
    engine._coldsnap_deferred_warmup_phase = "ready"
    engine._coldsnap_deferred_warmup_statuses = statuses
    _write_fully_warm_file(settings, statuses)
    logger.info("All vLLM workers completed deferred warmup")


def _install_engine_hook(
    settings: DeferredWarmupSettings, module: Any | None = None
) -> None:
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
    def process_step_then_warm(engine: Any, *args: Any, **kwargs: Any) -> bool:
        model_executed = bool(original(engine, *args, **kwargs))
        effective = _effective_engine_settings(engine, settings)
        if effective.enabled:
            _maybe_warm_after_step(engine, model_executed, effective)
        return model_executed

    setattr(process_step_then_warm, _ENGINE_MARKER, True)
    engine_class._process_engine_step = process_step_then_warm


def install_deferred_warmup_hooks(
    settings: DeferredWarmupSettings | None = None,
) -> DeferredWarmupSettings:
    """Install opt-in eager-first profile and kernel warmup hooks."""
    global _installed
    resolved = settings or deferred_warmup_settings_from_env()
    if _installed:
        return resolved
    after_module_import(
        _WORKER_MODULE,
        "deferred-warmup-worker",
        lambda module: _install_worker_hooks(resolved, module),
    )
    after_module_import(
        _ENGINE_MODULE,
        "deferred-warmup-engine",
        lambda module: _install_engine_hook(resolved, module),
    )
    _installed = True
    return resolved

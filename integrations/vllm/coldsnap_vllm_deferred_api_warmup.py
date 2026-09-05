# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Move vLLM's API-side multimodal warmup off the readiness path.

The work is submitted to the renderer's existing single-worker multimodal
executor.  Text-only traffic can start immediately. A real multimodal request
interrupts the speculative start delay and uses vLLM's normal request path;
if warmup has already started, the same queue safely orders the request after
it. Chat-template warmup remains synchronous because it is cheap and is needed
by the first text request.
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from coldsnap_vllm import VllmContractError


DEFERRED_API_MM_WARMUP_ENV = "COLDSNAP_DEFERRED_API_MM_WARMUP"
API_MM_WARM_FILE_ENV = "COLDSNAP_API_MM_WARM_FILE"
WARMUP_GENERATION_ENV = "COLDSNAP_WARMUP_GENERATION"
START_DELAY_SECONDS_ENV = "COLDSNAP_API_MM_WARMUP_DELAY_SECONDS"

_WARMUP_MARKER = "_coldsnap_deferred_api_mm_warmup_hook"
_DEMAND_MARKER = "_coldsnap_multimodal_demand_priority_hook"

logger = logging.getLogger(__name__)
_installed = False


@dataclass(frozen=True)
class DeferredApiWarmupSettings:
    enabled: bool = False
    fully_warm_file: Path | None = None
    generation: str = ""
    start_delay_seconds: float = 0.0


def _boolean(value: str | None, *, name: str) -> bool:
    normalized = "0" if value is None else value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise VllmContractError(f"{name} must be a boolean, got {value!r}")


def deferred_api_warmup_settings_from_env(
    environment: Mapping[str, str] | None = None,
) -> DeferredApiWarmupSettings:
    values = os.environ if environment is None else environment
    raw_file = values.get(API_MM_WARM_FILE_ENV)
    try:
        start_delay_seconds = float(values.get(START_DELAY_SECONDS_ENV, "0"))
    except ValueError as error:
        raise VllmContractError(
            f"{START_DELAY_SECONDS_ENV} must be a number"
        ) from error
    if not 0 <= start_delay_seconds <= 3600:
        raise VllmContractError(
            f"{START_DELAY_SECONDS_ENV} must be between 0 and 3600 seconds"
        )
    return DeferredApiWarmupSettings(
        enabled=_boolean(
            values.get(DEFERRED_API_MM_WARMUP_ENV),
            name=DEFERRED_API_MM_WARMUP_ENV,
        ),
        fully_warm_file=Path(raw_file) if raw_file else None,
        generation=values.get(WARMUP_GENERATION_ENV, ""),
        start_delay_seconds=start_delay_seconds,
    )


def _effective_settings(
    captured: DeferredApiWarmupSettings,
) -> DeferredApiWarmupSettings:
    """Resolve target policy from the restore handoff when one is active.

    The API process is restored rather than re-executed, so its Python hook
    must be present in the capture even though capture itself performs the
    normal synchronous warmup.  The bounded, unit-bound handoff is produced by
    the fresh restore controller before CRIU releases the process.
    """

    from coldsnap_vllm_process_template import (
        _restored_runtime_environment,
        process_template_settings_from_env,
    )

    restored = _restored_runtime_environment(process_template_settings_from_env())
    if restored is None:
        return captured
    environment = dict(os.environ)
    environment.update(restored)
    return deferred_api_warmup_settings_from_env(environment)


def _write_status(
    settings: DeferredApiWarmupSettings,
    *,
    phase: str,
    elapsed_s: float,
    error: str = "",
) -> None:
    path = settings.fully_warm_file
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "generation": settings.generation,
        "phase": phase,
        "ready_monotonic": time.monotonic(),
        "warmup_seconds": elapsed_s,
        "error": error,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_multimodal_warmup(
    renderer: Any,
    mm_processor: Any,
    readonly_mm_processor: Any,
    settings: DeferredApiWarmupSettings,
) -> dict[str, Any]:
    started = time.perf_counter()
    errors: list[str] = []
    if settings.start_delay_seconds:
        time.sleep(settings.start_delay_seconds)

    from vllm.utils.torch_utils import set_default_torch_num_threads

    with set_default_torch_num_threads(1):
        if mm_processor is not None:
            try:
                renderer._warmup_mm_processor(
                    mm_processor,
                    log_prefix="Deferred multi-modal",
                )
            except Exception as error:
                errors.append(f"multi-modal: {type(error).__name__}: {error}")
                logger.warning("Deferred multi-modal warmup failed", exc_info=True)
            finally:
                renderer.clear_mm_cache()

        if readonly_mm_processor is not None:
            try:
                renderer._warmup_mm_processor(
                    readonly_mm_processor,
                    log_prefix="Deferred readonly multi-modal",
                )
            except Exception as error:
                errors.append(f"readonly: {type(error).__name__}: {error}")
                logger.warning(
                    "Deferred readonly multi-modal warmup failed", exc_info=True
                )
            finally:
                renderer._clear_processor_cache(readonly_mm_processor)

    elapsed = time.perf_counter() - started
    phase = "failed" if errors else "ready"
    error_text = "; ".join(errors)
    status = {
        "phase": phase,
        "warmup_seconds": elapsed,
        "error": error_text,
    }
    renderer._coldsnap_deferred_api_mm_warmup_status = status
    _write_status(settings, phase=phase, elapsed_s=elapsed, error=error_text)
    return status


def _install_multimodal_demand_priority(
    renderer: Any,
    settings: DeferredApiWarmupSettings,
    signal_demand: Callable[[], None],
) -> None:
    """Let a real multimodal request supersede the speculative delay."""

    original = getattr(renderer, "_process_multimodal_async", None)
    if not callable(original):
        raise VllmContractError(
            "vLLM renderer lacks the asynchronous multimodal request path"
        )
    if getattr(original, _DEMAND_MARKER, False):
        return

    @functools.wraps(original)
    async def process_multimodal_with_demand_priority(
        *args: Any, **kwargs: Any
    ) -> Any:
        started = time.perf_counter()
        signal_demand()
        result = await original(*args, **kwargs)
        status = getattr(
            renderer,
            "_coldsnap_deferred_api_mm_warmup_status",
            {},
        )
        if status.get("phase") == "warming":
            elapsed = time.perf_counter() - started
            ready = {
                "phase": "ready",
                "warmup_seconds": elapsed,
                "error": "",
                "source": "request",
            }
            renderer._coldsnap_deferred_api_mm_warmup_status = ready
            _write_status(settings, phase="ready", elapsed_s=elapsed)
        return result

    setattr(process_multimodal_with_demand_priority, _DEMAND_MARKER, True)
    renderer._process_multimodal_async = process_multimodal_with_demand_priority


def _warmup_with_deferred_multimodal(
    renderer: Any,
    original: Any,
    settings: DeferredApiWarmupSettings,
    *args: Any,
    **kwargs: Any,
) -> None:
    mm_processor = getattr(renderer, "mm_processor", None)
    readonly_mm_processor = getattr(renderer, "_readonly_mm_processor", None)

    # Let vLLM own chat-template warmup and its error handling while hiding
    # only the two processors that the original method warms synchronously.
    renderer.mm_processor = None
    renderer._readonly_mm_processor = None
    try:
        original(renderer, *args, **kwargs)
    finally:
        renderer.mm_processor = mm_processor
        renderer._readonly_mm_processor = readonly_mm_processor

    if mm_processor is None and readonly_mm_processor is None:
        return
    executor = getattr(renderer, "_mm_executor", None)
    submit = getattr(executor, "submit", None)
    if not callable(submit):
        raise VllmContractError(
            "vLLM renderer lacks the single-worker multimodal executor"
        )
    renderer._coldsnap_deferred_api_mm_warmup_status = {
        "phase": "warming",
        "warmup_seconds": 0.0,
        "error": "",
    }
    demand = threading.Event()
    schedule_lock = threading.Lock()
    timer: threading.Timer | None = None

    def signal_demand() -> None:
        demand.set()
        with schedule_lock:
            if timer is not None:
                timer.cancel()

    _install_multimodal_demand_priority(renderer, settings, signal_demand)

    def submit_warmup() -> None:
        with schedule_lock:
            if demand.is_set():
                return
            renderer._coldsnap_deferred_api_mm_warmup_future = submit(
                _run_multimodal_warmup,
                renderer,
                mm_processor,
                readonly_mm_processor,
                replace(settings, start_delay_seconds=0),
            )

    if settings.start_delay_seconds:
        timer = threading.Timer(settings.start_delay_seconds, submit_warmup)
        timer.daemon = True
        renderer._coldsnap_deferred_api_mm_warmup_timer = timer
        timer.start()
    else:
        submit_warmup()


def install_deferred_api_warmup_hook(
    settings: DeferredApiWarmupSettings | None = None,
) -> DeferredApiWarmupSettings:
    """Install the opt-in API renderer warmup hook."""
    global _installed
    resolved = settings or deferred_api_warmup_settings_from_env()
    if _installed:
        return resolved

    module = importlib.import_module("vllm.renderers.base")
    renderer_class = getattr(module, "BaseRenderer", None)
    if not isinstance(renderer_class, type):
        raise VllmContractError("vllm.renderers.base lacks BaseRenderer")
    original = getattr(renderer_class, "warmup", None)
    if not callable(original):
        raise VllmContractError("vLLM BaseRenderer.warmup is unavailable")
    if getattr(original, _WARMUP_MARKER, False):
        _installed = True
        return resolved

    @functools.wraps(original)
    def warmup(renderer: Any, *args: Any, **kwargs: Any) -> None:
        effective = _effective_settings(resolved)
        if effective.enabled:
            _warmup_with_deferred_multimodal(
                renderer, original, effective, *args, **kwargs
            )
        else:
            original(renderer, *args, **kwargs)

    setattr(warmup, _WARMUP_MARKER, True)
    renderer_class.warmup = warmup
    _installed = True
    return resolved

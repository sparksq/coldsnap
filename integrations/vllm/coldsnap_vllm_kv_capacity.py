# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Keep CuMem KV sizing inside vLLM's requested-memory envelope.

Some vLLM builds reconstitute ``non_kv_cache_memory`` after profiling with
allocator counters that do not describe PyTorch pluggable memory pools.  The
result can be an oversized KV cache even though the same worker also publishes
allocator-independent ``total_consumed`` and transient-peak measurements.

This adapter applies a one-way safety invariant after vLLM's own calculation:
an automatic KV estimate may not exceed ``requested - consumed - peak``.  It
never increases capacity, does not affect explicit ``kv_cache_memory_bytes``,
and is a no-op on engines whose estimate already respects the invariant.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, Mapping

from coldsnap_import_hook import after_module_import
from coldsnap_vllm import VllmContractError


KV_CAPACITY_GUARD_ENV = "COLDSNAP_KV_CAPACITY_GUARD"
_WORKER_MODULE = "vllm.v1.worker.gpu_worker"
_MARKER = "_coldsnap_kv_capacity_guard"

logger = logging.getLogger(__name__)
_installed = False


def _boolean(value: str | None, *, name: str) -> bool:
    normalized = "0" if value is None else value.strip().lower()
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise VllmContractError(f"{name} must be a boolean, got {value!r}")


def kv_capacity_guard_enabled(
    environment: Mapping[str, str] | None = None,
) -> bool:
    values = os.environ if environment is None else environment
    return _boolean(values.get(KV_CAPACITY_GUARD_ENV), name=KV_CAPACITY_GUARD_ENV)


def _worker_class(module: Any) -> type[Any]:
    classes: list[type[Any]] = []
    for name in ("Worker", "GPUWorker"):
        value = getattr(module, name, None)
        if isinstance(value, type) and value not in classes:
            classes.append(value)
    if len(classes) != 1:
        raise VllmContractError("vllm.v1.worker.gpu_worker must expose one supported worker class")
    return classes[0]


def _guard_automatic_capacity(
    worker: Any,
    reported_bytes: Any,
    reserve_mm_ipc_gpu_memory: Callable[[int, Any, int], int],
) -> int:
    if type(reported_bytes) is not int or reported_bytes <= 0:
        raise VllmContractError(
            "vLLM determine_available_memory must return positive integer bytes"
        )

    cache_config = getattr(worker, "cache_config", None)
    if cache_config is None:
        raise VllmContractError("GPU worker lacks cache_config")
    if getattr(cache_config, "kv_cache_memory_bytes", None) is not None:
        return reported_bytes

    fields: dict[str, int] = {}
    for name in ("requested_memory", "total_consumed", "peak_activation_memory"):
        value = getattr(worker, name, None)
        if type(value) not in {int, float} or value < 0:
            raise VllmContractError(f"GPU worker lacks a nonnegative numeric {name} capacity input")
        fields[name] = int(value)

    unreserved_limit = (
        fields["requested_memory"] - fields["total_consumed"] - fields["peak_activation_memory"]
    )
    if unreserved_limit <= 0:
        raise VllmContractError(
            "vLLM's measured non-KV memory exceeds the requested-memory envelope"
        )

    model_config = getattr(worker, "model_config", None)
    parallel_config = getattr(worker, "parallel_config", None)
    if model_config is None or parallel_config is None:
        raise VllmContractError("GPU worker lacks model or parallel config for KV reservation")
    guarded_bytes = int(
        reserve_mm_ipc_gpu_memory(
            unreserved_limit,
            getattr(model_config, "multimodal_config", None),
            getattr(parallel_config, "_api_process_count", 1),
        )
    )
    if guarded_bytes <= 0:
        raise VllmContractError("KV capacity guard left no memory for the cache")

    selected_bytes = min(reported_bytes, guarded_bytes)
    status = {
        "format": 1,
        "kind": "coldsnap-kv-capacity-guard",
        "reported_bytes": reported_bytes,
        "requested_memory_bytes": fields["requested_memory"],
        "total_consumed_bytes": fields["total_consumed"],
        "peak_activation_memory_bytes": fields["peak_activation_memory"],
        "unreserved_limit_bytes": unreserved_limit,
        "guarded_bytes": guarded_bytes,
        "selected_bytes": selected_bytes,
        "clamped": selected_bytes < reported_bytes,
    }
    worker._coldsnap_kv_capacity_guard_status = status
    if status["clamped"]:
        # Keep vLLM's later audit and startup-plan persistence consistent with
        # the capacity that the EngineCore actually receives.
        worker.available_kv_cache_memory_bytes = unreserved_limit
        logger.warning(
            "Clamped vLLM automatic KV capacity from %.2f GiB to %.2f GiB: "
            "the CuMem estimate exceeded the requested-memory envelope by "
            "%.2f GiB",
            reported_bytes / (1 << 30),
            selected_bytes / (1 << 30),
            (reported_bytes - selected_bytes) / (1 << 30),
        )
    return selected_bytes


def _install_worker_hook(module: Any) -> None:
    worker_class = _worker_class(module)
    original = getattr(worker_class, "determine_available_memory", None)
    reserve = getattr(module, "reserve_mm_ipc_gpu_memory", None)
    if not callable(original) or not callable(reserve):
        raise VllmContractError(
            "GPU worker lacks determine_available_memory or multimodal reservation"
        )
    if getattr(original, _MARKER, False):
        return

    @functools.wraps(original)
    def determine_available_memory(worker: Any, *args: Any, **kwargs: Any) -> int:
        reported = original(worker, *args, **kwargs)
        return _guard_automatic_capacity(worker, reported, reserve)

    setattr(determine_available_memory, _MARKER, True)
    worker_class.determine_available_memory = determine_available_memory


def install_kv_capacity_guard() -> bool:
    """Install the guard when explicitly enabled or using ColdSnap CuMem."""
    global _installed
    enabled = kv_capacity_guard_enabled()
    if not enabled or _installed:
        return enabled
    after_module_import(
        _WORKER_MODULE,
        "kv-capacity-guard",
        _install_worker_hook,
    )
    _installed = True
    return enabled

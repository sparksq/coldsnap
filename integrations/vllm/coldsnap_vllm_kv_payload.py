# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Identify vLLM KV payload allocations without model-specific hooks.

vLLM's CuMem ``kv_cache`` scope can also contain attention metadata, block
tables, verifier state, and graph-manager workspaces. Those objects are not
request KV contents and proved unsafe to discard across a CUDA process
checkpoint. Both current runner families expose the actual cache tensors as
``model_runner.kv_caches``. This adapter records their storage extents after
vLLM initializes them, allowing the memory provider to select payload backing
semantically instead of relying on allocator order or model-specific sizes.
"""

from __future__ import annotations

import functools
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from coldsnap_import_hook import after_module_import
from coldsnap_vllm import VllmContractError


_RUNNER_MODULES = (
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu_model_runner",
)
_INITIALIZE_MARKER = "_coldsnap_kv_payload_tracking_hook"

_lock = threading.RLock()
_payload_extents: tuple["KvPayloadExtent", ...] = ()
_installed = False


@dataclass(frozen=True, order=True)
class KvPayloadExtent:
    pointer: int
    size: int


def _tensor_values(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _tensor_values(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensor_values(child)
        return
    if callable(getattr(value, "untyped_storage", None)):
        yield value


def _storage_extent(tensor: Any) -> KvPayloadExtent:
    try:
        storage = tensor.untyped_storage()
        pointer = int(storage.data_ptr())
        size = int(storage.nbytes())
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise VllmContractError(
            "vLLM KV tensor storage must expose data_ptr() and nbytes()"
        ) from error
    if pointer <= 0 or size <= 0:
        raise VllmContractError(
            f"vLLM KV tensor storage has invalid extent {pointer:#x}+{size}"
        )
    return KvPayloadExtent(pointer, size)


def record_kv_payload(runner: Any) -> tuple[KvPayloadExtent, ...]:
    """Replace the process-local payload inventory from a runner instance."""
    caches = getattr(runner, "kv_caches", None)
    if caches is None:
        raise VllmContractError(
            "vLLM model runner did not publish kv_caches after initialization"
        )
    extents = tuple(sorted({_storage_extent(value) for value in _tensor_values(caches)}))
    if not extents:
        raise VllmContractError(
            "vLLM model runner published no tensor storage in kv_caches"
        )
    global _payload_extents
    with _lock:
        _payload_extents = extents
    return extents


def kv_payload_extents() -> tuple[KvPayloadExtent, ...]:
    with _lock:
        return _payload_extents


def _install_runner_hook(module: Any) -> None:
    runner_class = getattr(module, "GPUModelRunner", None)
    if not isinstance(runner_class, type):
        raise VllmContractError(
            f"{module.__name__} does not expose GPUModelRunner"
        )
    original = getattr(runner_class, "initialize_kv_cache", None)
    if not callable(original):
        raise VllmContractError(
            f"{module.__name__}.GPUModelRunner lacks initialize_kv_cache"
        )
    if getattr(original, _INITIALIZE_MARKER, False):
        return

    @functools.wraps(original)
    def initialize_then_record(runner: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(runner, *args, **kwargs)
        record_kv_payload(runner)
        return result

    setattr(initialize_then_record, _INITIALIZE_MARKER, True)
    runner_class.initialize_kv_cache = initialize_then_record


def install_kv_payload_tracking_hooks() -> None:
    """Track semantic cache storage in whichever vLLM runner is imported."""
    global _installed
    if _installed:
        return
    regions = {
        value.strip()
        for value in os.environ.get("COLDSNAP_DISCARD_REGIONS", "").split(",")
        if value.strip()
    }
    if "kv_cache" not in regions:
        return
    for module_name in _RUNNER_MODULES:
        after_module_import(
            module_name,
            "kv-payload-tracking",
            _install_runner_hook,
        )
    _installed = True

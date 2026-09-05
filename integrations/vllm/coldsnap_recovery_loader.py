# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Recovery-aware vLLM loader backed by original safetensors files.

The loader deliberately leaves model semantics to vLLM.  It only materializes
checkpoint tensors on CUDA, then yields them to the model's ordinary
``load_weights`` implementation.  During live-process recovery vLLM's
``reload_weights`` path performs the same model-specific sharding, packing, and
finalization and copies the results back into the captured kernel tensor
storages at their stable virtual addresses.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import logging
import math
import os
import struct
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from coldsnap_core.hydration import HydrationExtent, NativeHydrator
from coldsnap_vllm import prepare_synthetic_weight_source, register_model_loader


LOAD_FORMAT = "coldsnap"
RECOVERY_SOURCE_ENV = "COLDSNAP_RECOVERY_WEIGHT_SOURCE"
RECOVERY_SOURCE = "safetensors"
DEFAULT_SOURCE = "blob"
LOADER_BACKEND_ENV = "COLDSNAP_RECOVERY_LOADER_BACKEND"
CAPTURE_LOAD_FORMAT_ENV = "COLDSNAP_CAPTURE_LOAD_FORMAT"
DEFAULT_LOADER_BACKEND = "direct"
LOADER_DISTRIBUTED_ENV = "COLDSNAP_RECOVERY_LOADER_DISTRIBUTED"
LOADER_STATUS_GROUP_ENV = "COLDSNAP_RECOVERY_LOADER_STATUS_GROUP"
LOADER_COLLECTIVE_BYTES_ENV = "COLDSNAP_RECOVERY_LOADER_COLLECTIVE_BYTES"
LOADER_STAGING_BYTES_ENV = "COLDSNAP_RECOVERY_LOADER_STAGING_BYTES"
LOADER_VERIFY_BYTES_ENV = "COLDSNAP_RECOVERY_LOADER_VERIFY_BYTES"
LOADER_VERIFY_MODEL_BYTES_ENV = "COLDSNAP_RECOVERY_LOADER_VERIFY_MODEL_BYTES"
LOADER_DERIVED_BUFFER_MAX_BYTES_ENV = "COLDSNAP_RECOVERY_DERIVED_BUFFER_MAX_BYTES"
LOADER_LOCAL_PACKED_MAX_EXTENTS_ENV = "COLDSNAP_RECOVERY_LOADER_LOCAL_PACKED_MAX_EXTENTS"
LOADER_LOCAL_PACKED_MIN_EXTENT_BYTES_ENV = "COLDSNAP_RECOVERY_LOADER_LOCAL_PACKED_MIN_EXTENT_BYTES"
LOADER_IO_PROBE_BYTES_ENV = "COLDSNAP_RECOVERY_LOADER_IO_PROBE_BYTES"
LOADER_IO_WEIGHT_EXPONENT_ENV = "COLDSNAP_RECOVERY_LOADER_IO_WEIGHT_EXPONENT"
LOADER_TRANSPORT_PIPELINE_DEPTH_ENV = "COLDSNAP_RECOVERY_LOADER_TRANSPORT_PIPELINE_DEPTH"
PROCESS_TEMPLATE_RESTORED_ENV = "COLDSNAP_PROCESS_TEMPLATE_RESTORED"
MODEL_PAYLOAD_MATERIALIZATION_CONTROL_ENV = (
    "COLDSNAP_MODEL_PAYLOAD_MATERIALIZATION_CONTROL"
)
DEFAULT_MODEL_PAYLOAD_MATERIALIZATION_CONTROL = (
    "/run/coldsnap/model-payload-materialization.json"
)
DEFAULT_DERIVED_BUFFER_MAX_BYTES = 64 * 1024**2
CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR = "_coldsnap_checkpoint_destination_tensor_names"
CHECKPOINT_COPY_PLAN_ATTR = "_coldsnap_checkpoint_copy_plan"
INITIAL_LOAD_METRICS_ATTR = "_coldsnap_initial_recovery_load_metrics"
RECOVERY_REPLAY_PLANS_ATTR = "_coldsnap_recovery_replay_plans"
RECOVERY_IO_WEIGHTS_ATTR = "_coldsnap_recovery_io_weights"
DEFAULT_STAGING_BYTES = 256 * 1024**2
DEFAULT_COLLECTIVE_BYTES = 64 * 1024**2
DEFAULT_LOCAL_PACKED_MAX_EXTENTS = 256 * 1024
DEFAULT_LOCAL_PACKED_MIN_EXTENT_BYTES = 64 * 1024
DEFAULT_IO_PROBE_BYTES = 256 * 1024**2
DEFAULT_IO_WEIGHT_EXPONENT = 0.75
DEFAULT_TRANSPORT_PIPELINE_DEPTH = 1
DERIVED_SCALE_BUFFER_NAMES = frozenset(
    {
        "_q_scale",
        "_k_scale",
        "_v_scale",
        "_prob_scale",
    }
)
DERIVED_ROPE_BUFFER_NAMES = frozenset({"cos_sin_cache", "cos_sin_cache_bf16"})
MAX_EXACT_ROPE_BUFFER_BYTES = 1024**2
MAX_HEADER_BYTES = 256 * 1024**2
MAX_INT64 = (1 << 63) - 1
B12X_PREPARED_WEIGHT_FIELDS = (
    "a1_gscale",
    "w1_fp4",
    "w1_blockscale",
    "w1_alphas",
    "a2_gscale",
    "w2_fp4",
    "w2_blockscale",
    "w2_alphas",
)
B12X_SOURCE_STORAGE_FIELDS = (
    ("w13_weight", "w1_fp4"),
    ("w13_weight_scale", "w1_blockscale"),
    ("w2_weight", "w2_fp4"),
    ("w2_weight_scale", "w2_blockscale"),
)
B12X_RELOAD_WEIGHT_FIELDS = tuple(
    prepared_name for _source_name, prepared_name in B12X_SOURCE_STORAGE_FIELDS
)
B12X_DERIVED_RUNTIME_FIELDS = tuple(
    field_name
    for field_name in B12X_PREPARED_WEIGHT_FIELDS
    if field_name not in B12X_RELOAD_WEIGHT_FIELDS
)
B12X_MHC_RELOAD_DEPENDENCY_FIELDS = ("hc_attn_fn", "hc_ffn_fn")

_RECOVERY_CONSUMER_STORAGE_ISOLATION: ContextVar[bool] = ContextVar(
    "coldsnap_recovery_consumer_storage_isolation",
    default=False,
)
_ACTIVE_RECOVERY_SOURCE: ContextVar[Any | None] = ContextVar(
    "coldsnap_active_recovery_source",
    default=None,
)
_RECOVERY_SKIP_SOURCE_NAMES: ContextVar[frozenset[str]] = ContextVar(
    "coldsnap_recovery_skip_source_names",
    default=frozenset(),
)
_RECOVERY_PRELOADED_DESTINATION_NAMES: ContextVar[frozenset[str]] = ContextVar(
    "coldsnap_recovery_preloaded_destination_names",
    default=frozenset(),
)
_ACTIVE_RECOVERY_ADAPTERS: ContextVar[tuple[Any, ...]] = ContextVar(
    "coldsnap_active_recovery_adapters",
    default=(),
)

_DTYPE_ATTRIBUTES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
}


def _view_span_bytes(tensor: Any) -> int:
    """Return the storage span touched by a non-negative-stride tensor view."""
    if int(tensor.numel()) == 0:
        return 0
    shape = tuple(int(value) for value in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    if len(shape) != len(stride) or any(value < 0 for value in stride):
        raise ValueError("recovery replay requires non-negative tensor strides")
    last_element = sum(
        (dimension - 1) * step for dimension, step in zip(shape, stride, strict=True) if dimension
    )
    return (last_element + 1) * int(tensor.element_size())


def _observed_recovery_copy(
    source: Any,
    destination_name: str,
    destination: Any,
    copied_to: Any,
    copied_from: Any,
) -> tuple[RecoveryCopy | None, str | None]:
    """Build a replay record for one source-to-model ``copy_`` operation.

    The source and destination dtypes may differ.  That is not an unsupported
    transform: replay invokes the same typed ``copy_`` and therefore preserves
    PyTorch's conversion semantics.  This matters for serialized packed
    formats such as signed-byte checkpoint storage copied into unsigned-byte
    kernel storage, while remaining general for other vLLM loaders.
    """
    try:
        if tuple(copied_to.shape) != tuple(copied_from.shape):
            return None, "shape-mismatch"
        if int(copied_to.numel()) == 0:
            return None, "empty-copy"
        copy_bytes = int(copied_to.numel()) * int(copied_to.element_size())
        source_offset = int(copied_from.data_ptr()) - int(source.pointer)
        destination_offset = int(copied_to.data_ptr()) - int(destination.data_ptr())
        destination_bytes = int(destination.numel()) * int(destination.element_size())
        if source_offset < 0:
            return None, "source-not-aliased"
        if destination_offset < 0:
            return None, "destination-not-aliased"
        if source_offset + _view_span_bytes(copied_from) > int(source.descriptor.length):
            return None, "source-span"
        if destination_offset + _view_span_bytes(copied_to) > destination_bytes:
            return None, "destination-span"
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None, "metadata-error"

    descriptor = source.descriptor
    return RecoveryCopy(
        source_name=source.name,
        source_path=source.path,
        source_dtype_name=descriptor.dtype_name,
        source_shape=descriptor.shape,
        source_file_offset=descriptor.file_offset,
        source_length=descriptor.length,
        source_view_dtype=str(copied_from.dtype),
        source_view_offset_bytes=source_offset,
        source_view_shape=tuple(int(value) for value in copied_from.shape),
        source_view_stride=tuple(int(value) for value in copied_from.stride()),
        destination_name=destination_name,
        destination_dtype=str(destination.dtype),
        destination_shape=tuple(int(value) for value in destination.shape),
        destination_view_offset_bytes=destination_offset,
        destination_view_shape=tuple(int(value) for value in copied_to.shape),
        destination_view_stride=tuple(int(value) for value in copied_to.stride()),
        copy_bytes=copy_bytes,
    ), None


def _observe_weight_loader_copy(
    source: RecoverySourceTensor,
    destination_name: str,
    destination: Any,
    loader: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, tuple[RecoveryCopy, ...], bool, tuple[str, ...]]:
    """Run one vLLM loader while observing its model-destination mutations."""
    from torch.utils._python_dispatch import TorchDispatchMode

    destination_start = int(destination.data_ptr())
    destination_end = destination_start + int(destination.numel() * destination.element_size())

    class CopyObserver(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.copies: list[RecoveryCopy] = []
            self.unsupported = False
            self.rejection_reasons: list[str] = []

        def touches_destination(self, value: Any) -> bool:
            try:
                if int(value.numel()) == 0:
                    return False
                start = int(value.data_ptr())
                end = start + _view_span_bytes(value)
                return start < destination_end and destination_start < end
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False

        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            result = func(*args, **(kwargs or {}))
            schema = getattr(func, "_schema", None)
            if not bool(getattr(schema, "is_mutable", False)) or not args:
                return result
            target = args[0]
            if not self.touches_destination(target):
                return result
            operation = str(getattr(schema, "name", ""))
            if operation == "aten::zero_":
                # Replay adapters zero their complete claimed destinations
                # before applying observed checkpoint copies.
                return result
            if operation != "aten::copy_" or len(args) < 2:
                self.unsupported = True
                self.rejection_reasons.append(f"mutation:{operation or '<unknown>'}")
                return result
            observed, reason = _observed_recovery_copy(
                source,
                destination_name,
                destination,
                target,
                args[1],
            )
            if observed is None:
                self.unsupported = True
                self.rejection_reasons.append(reason or "copy-rejected")
            else:
                self.copies.append(observed)
            return result

    observer = CopyObserver()
    with observer:
        result = loader(*args, **kwargs)
    return (
        result,
        tuple(observer.copies),
        observer.unsupported,
        tuple(observer.rejection_reasons),
    )


@dataclass(frozen=True)
class SafetensorDescriptor:
    name: str
    dtype_name: str
    shape: tuple[int, ...]
    file_offset: int
    length: int


@dataclass(frozen=True)
class RecoverySourceTensor:
    """One live iterator tensor and its immutable checkpoint identity."""

    name: str
    path: str
    descriptor: SafetensorDescriptor
    pointer: int


@dataclass(frozen=True)
class RecoveryCopy:
    """Observed checkpoint view copied into a model tensor view.

    The plan describes tensor metadata rather than loader names or model
    classes. Backends may claim destinations they know how to finalize after
    the copies; everything else continues through vLLM's ordinary loader. The
    source and destination view dtypes are kept independently so replay can
    reproduce an observed typed ``copy_`` conversion without interpreting the
    checkpoint format itself.
    """

    source_name: str
    source_path: str
    source_dtype_name: str
    source_shape: tuple[int, ...]
    source_file_offset: int
    source_length: int
    source_view_dtype: str
    source_view_offset_bytes: int
    source_view_shape: tuple[int, ...]
    source_view_stride: tuple[int, ...]
    destination_name: str
    destination_dtype: str
    destination_shape: tuple[int, ...]
    destination_view_offset_bytes: int
    destination_view_shape: tuple[int, ...]
    destination_view_stride: tuple[int, ...]
    copy_bytes: int

    @property
    def source_key(self) -> tuple[Any, ...]:
        return (
            self.source_name,
            self.source_path,
            self.source_dtype_name,
            self.source_shape,
            self.source_file_offset,
            self.source_length,
        )


@dataclass(frozen=True)
class RecoveryCopyPlan:
    copies: tuple[RecoveryCopy, ...]
    unsupported_destinations: frozenset[str]


@dataclass(frozen=True)
class _ReplaySourceRange:
    """One rank's bounded file range backing all of its views of a source."""

    source_key: tuple[Any, ...]
    descriptor: SafetensorDescriptor
    source_base_offset_bytes: int


@dataclass(frozen=True)
class _ReplayPackedExtent:
    """One contiguous part of a strided source view in a packed slab."""

    file_offset: int
    packed_offset_bytes: int
    length: int


def _qualify_tensor_name(module_name: str, tensor_name: str) -> str:
    if not module_name:
        return tensor_name
    if not tensor_name:
        return module_name
    return f"{module_name}.{tensor_name}"


def _named_module_items(model: Any) -> Iterator[tuple[str, Any]]:
    yield "", model
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return
    seen = {id(model)}
    for module_name, module in named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        yield module_name, module


def _model_checkpoint_destination_names(model: Any) -> frozenset[str] | None:
    """Collect loader metadata through wrappers and rebase it to ``model``.

    vLLM may pass the unwrapped architecture to ``ModelLoader.load_weights``
    and later expose a compiled or execution wrapper from ``get_model()``.
    Loader metadata therefore belongs to the module on which it was observed,
    not necessarily the object used by sleep/recovery hooks.
    """
    discovered = False
    names: set[str] = set()
    for module_name, module in _named_module_items(model):
        local = getattr(module, CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR, None)
        if local is None:
            continue
        if not isinstance(local, (set, frozenset)) or not all(
            isinstance(name, str) for name in local
        ):
            raise RuntimeError("recovery checkpoint destination tensor names are invalid")
        discovered = True
        names.update(_qualify_tensor_name(module_name, name) for name in local)
    return frozenset(names) if discovered else None


def _model_checkpoint_copy_plan(model: Any) -> RecoveryCopyPlan | None:
    """Collect exact-copy plans through wrappers in the root namespace."""
    discovered = False
    copies: list[RecoveryCopy] = []
    unsupported: set[str] = set()
    for module_name, module in _named_module_items(model):
        local = getattr(module, CHECKPOINT_COPY_PLAN_ATTR, None)
        if local is None:
            continue
        if not isinstance(local, RecoveryCopyPlan):
            raise RuntimeError("recovery checkpoint copy plan is invalid")
        discovered = True
        copies.extend(
            replace(
                copy,
                destination_name=_qualify_tensor_name(module_name, copy.destination_name),
            )
            for copy in local.copies
        )
        unsupported.update(
            _qualify_tensor_name(module_name, name) for name in local.unsupported_destinations
        )
    if not discovered:
        return None
    return RecoveryCopyPlan(tuple(copies), frozenset(unsupported))


@dataclass(frozen=True)
class RecoveryLoadMetrics:
    files: int
    tensors: int
    logical_bytes: int
    local_read_bytes: int
    local_read_extents: int
    io_seconds: float
    collective_seconds: float
    collective_calls: int
    verification_seconds: float
    verified_sample_bytes: int
    total_seconds: float
    backend: str
    distributed_world_size: int
    staging_layout: str
    staging_buffer_bytes: int
    collective_buffer_bytes: int
    io_policy: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "tensors": self.tensors,
            "logical_bytes": self.logical_bytes,
            "local_read_bytes": self.local_read_bytes,
            "local_read_extents": self.local_read_extents,
            "io_seconds": self.io_seconds,
            "collective_seconds": self.collective_seconds,
            "collective_calls": self.collective_calls,
            "verification_seconds": self.verification_seconds,
            "verified_sample_bytes": self.verified_sample_bytes,
            "total_seconds": self.total_seconds,
            "backend": self.backend,
            "distributed_world_size": self.distributed_world_size,
            "staging_layout": self.staging_layout,
            "staging_buffer_bytes": self.staging_buffer_bytes,
            "collective_buffer_bytes": self.collective_buffer_bytes,
            "io_policy": [dict(value) for value in self.io_policy],
        }


_last_metrics: RecoveryLoadMetrics | None = None
_last_direct_replay_metrics: dict[str, Any] | None = None
_process_template_load_metrics: list[RecoveryLoadMetrics] = []
_io_policy_observations: dict[str, dict[str, Any]] = {}
_io_policy_lock = threading.Lock()


def recovery_weights_enabled() -> bool:
    value = os.environ.get(RECOVERY_SOURCE_ENV, "")
    if value not in {"", DEFAULT_SOURCE, RECOVERY_SOURCE}:
        raise ValueError(
            f"{RECOVERY_SOURCE_ENV} must be empty, {DEFAULT_SOURCE!r}, or {RECOVERY_SOURCE!r}"
        )
    return value == RECOVERY_SOURCE


def last_recovery_load_metrics() -> dict[str, Any] | None:
    return _reported_recovery_load_metrics(
        _last_metrics,
        _last_direct_replay_metrics,
    )


def _reported_recovery_load_metrics(
    loader: RecoveryLoadMetrics | None,
    adapter_replay: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Report one aggregate recovery interval with non-overlapping children.

    Direct adapter replay runs immediately before vLLM's remaining model-loader
    work.  Historically ``total_seconds`` described only that latter loader
    interval while ``adapter_replay`` was presented as its child, producing an
    impossible timing tree.  Preserve the component duration as
    ``loader_seconds`` and make ``total_seconds`` cover both sequential phases.
    """

    result = loader.as_dict() if loader is not None else None
    if adapter_replay is None:
        return result
    if result is None:
        result = {}
    loader_seconds = float(result.get("total_seconds", 0.0))
    replay_seconds = float(adapter_replay.get("total_seconds", 0.0))
    result["loader_seconds"] = loader_seconds
    result["total_seconds"] = loader_seconds + replay_seconds
    result["adapter_replay"] = dict(adapter_replay)
    return result


def _publish_process_template_load_metrics(metrics: RecoveryLoadMetrics) -> None:
    """Expose n580 startup hydration through the normal worker state channel."""

    global _process_template_load_metrics

    if os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORED") != "1":
        return
    state_root = os.environ.get("COLDSNAP_HIBERNATE_STATE_DIR", "").strip()
    if not state_root:
        return
    from coldsnap_core.topology import worker_id

    worker = worker_id()
    _process_template_load_metrics.append(metrics)
    combined = _combine_capture_metrics(_process_template_load_metrics)
    if combined is None:
        return
    recovery = _reported_recovery_load_metrics(
        combined,
        _last_direct_replay_metrics,
    )
    if recovery is None:
        return
    resume_seconds = float(recovery["total_seconds"])
    value = {
        "format": 1,
        "kind": "coldsnap-vllm-process-template-recovery-load",
        "state": "running",
        "worker_id": worker,
        "updated_unix": time.time(),
        "resume_seconds": resume_seconds,
        "hydration_backend": "recovery-safetensors+" + combined.backend,
        "read_io_mode": combined.backend,
        "weight_recovery_source": RECOVERY_SOURCE,
        "phase_seconds": {
            "model_reload_seconds": resume_seconds,
            "recovery_loader": recovery,
        },
    }
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{worker}.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _model_initial_recovery_load_metrics(model: Any) -> RecoveryLoadMetrics | None:
    """Find the primary checkpoint profile through execution-model wrappers."""
    candidates: list[RecoveryLoadMetrics] = []
    seen: set[int] = set()
    for _module_name, module in _named_module_items(model):
        metrics = getattr(module, INITIAL_LOAD_METRICS_ATTR, None)
        if not isinstance(metrics, RecoveryLoadMetrics) or id(metrics) in seen:
            continue
        seen.add(id(metrics))
        candidates.append(metrics)
    if not candidates:
        return None
    # A speculative draft can use the same loader after the target. Select the
    # model with the largest checkpoint footprint instead of relying on load
    # order or architecture-specific wrapper names.
    return max(
        candidates,
        key=lambda metrics: (
            metrics.logical_bytes,
            metrics.local_read_bytes,
            metrics.tensors,
        ),
    )


def _model_initial_recovery_io_rate(model: Any) -> float | None:
    metrics = _model_initial_recovery_load_metrics(model)
    if metrics is None or metrics.local_read_bytes <= 0 or metrics.io_seconds <= 0:
        return None
    return metrics.local_read_bytes / metrics.io_seconds


def _model_verify_bytes() -> int:
    raw = os.environ.get(LOADER_VERIFY_MODEL_BYTES_ENV, "0")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{LOADER_VERIFY_MODEL_BYTES_ENV} must be an integer") from error
    if value < 0 or value > 4096:
        raise ValueError(f"{LOADER_VERIFY_MODEL_BYTES_ENV} must be in [0, 4096]")
    return value


def _derived_buffer_max_bytes() -> int:
    raw = os.environ.get(
        LOADER_DERIVED_BUFFER_MAX_BYTES_ENV,
        str(DEFAULT_DERIVED_BUFFER_MAX_BYTES),
    )
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{LOADER_DERIVED_BUFFER_MAX_BYTES_ENV} must be an integer") from error
    if value <= 0 or value > 1024**3:
        raise ValueError(f"{LOADER_DERIVED_BUFFER_MAX_BYTES_ENV} must be in [1, {1024**3}]")
    return value


def _capture_derived_buffers(model: Any) -> tuple[dict[str, Any], int]:
    selected: dict[str, Any] = {}
    total_bytes = 0
    limit = _derived_buffer_max_bytes()
    for name, buffer in model.named_buffers():
        leaf = name.rsplit(".", 1)[-1]
        buffer_bytes = int(buffer.numel() * buffer.element_size())
        preserve = leaf in DERIVED_SCALE_BUFFER_NAMES or (
            leaf in DERIVED_ROPE_BUFFER_NAMES and buffer_bytes <= MAX_EXACT_ROPE_BUFFER_BYTES
        )
        if not preserve:
            continue
        copied = buffer.detach().cpu().clone()
        total_bytes += buffer_bytes
        if total_bytes > limit:
            raise RuntimeError(
                "recovery derived buffers exceed the configured CRIU capsule "
                f"limit: bytes={total_bytes} limit={limit}"
            )
        selected[name] = copied
    return selected, total_bytes


def _b12x_prepared_owners(
    model: Any,
) -> list[tuple[str, Any, Any, Any]]:
    """Discover vLLM layers whose B12x runtime owns prepared expert storage."""
    result: list[tuple[str, Any, Any, Any]] = []
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return result
    seen: set[int] = set()
    for module_name, module in named_modules():
        quant_method = getattr(module, "quant_method", None)
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        fused_experts = getattr(moe_kernel, "fused_experts", None)
        lookup = getattr(fused_experts, "_lookup_prepared_experts", None)
        if not callable(lookup) or id(fused_experts) in seen:
            continue
        prepared = lookup()
        if prepared is None:
            continue
        seen.add(id(fused_experts))
        result.append((module_name, module, fused_experts, prepared))
    return result


def _model_weight_tensors(model: Any) -> list[tuple[str, Any]]:
    """Return observed checkpoint destinations plus backend-owned weights."""
    destination_names = _model_checkpoint_destination_names(model)
    if destination_names is None:
        # Retaining parameters without an observed vLLM weight-loader write in
        # the residual capsule is conservative and portable: it costs space,
        # but avoids classifying runtime-created Parameters as checkpoint
        # weights based on model-specific naming or over-reported load results.
        tensors: list[tuple[str, Any]] = []
    else:
        if not isinstance(destination_names, (set, frozenset)) or not all(
            isinstance(name, str) for name in destination_names
        ):
            raise RuntimeError("recovery checkpoint destination tensor names are invalid")
        tensors = []
        named_buffers = getattr(model, "named_buffers", lambda: ())
        for named_tensors in (model.named_parameters, named_buffers):
            tensors.extend(
                (name, tensor) for name, tensor in named_tensors() if name in destination_names
            )
    for module_name, _module, _owner, prepared in _b12x_prepared_owners(model):
        prefix = module_name or "<root>"
        # Only source-checkpoint-backed weight/scale allocations belong here.
        # B12x activation scales, runtime alphas, and other derived handles are
        # deliberately left in the small residual blob so recovery does not
        # zero a cache that its normal preparation path expects to reuse.
        for field_name in B12X_RELOAD_WEIGHT_FIELDS:
            tensor = getattr(prepared, field_name, None)
            if tensor is None:
                raise RuntimeError(
                    "B12x prepared expert owner is missing canonical weight "
                    f"field {field_name!r} at {prefix!r}"
                )
            tensors.append((f"{prefix}.prepared_experts.{field_name}", tensor))
    return tensors


def _model_semantic_tensors(model: Any) -> list[tuple[str, Any]]:
    """Return all registered parameters plus backend-owned runtime state.

    Derived backend tensors may be legitimately reallocated by the ordinary
    vLLM preparation path. Their captured allocations remain in the residual
    blob for any restored graph references, while these live-owner samples
    verify that rebuilt values are semantically identical.
    """
    tensors = list(model.named_parameters())
    for module_name, _module, _owner, prepared in _b12x_prepared_owners(model):
        prefix = module_name or "<root>"
        for field_name in B12X_PREPARED_WEIGHT_FIELDS:
            tensor = getattr(prepared, field_name, None)
            if tensor is None:
                raise RuntimeError(
                    "B12x prepared expert owner is missing runtime "
                    f"field {field_name!r} at {prefix!r}"
                )
            tensors.append((f"{prefix}.prepared_experts.{field_name}", tensor))
    return tensors


def _alias_meta_tensor_from_storage(
    torch_module: Any,
    meta_tensor: Any,
    stable_tensor: Any,
    *,
    label: str,
) -> Any:
    """Materialize a meta tensor as a typed view of captured stable storage."""
    if not bool(getattr(meta_tensor, "is_meta", False)):
        raise RuntimeError(f"recovery source {label!r} is not a meta tensor")
    if not bool(stable_tensor.is_cuda) or not stable_tensor.is_contiguous():
        raise RuntimeError(f"recovery stable owner {label!r} must be contiguous CUDA storage")
    meta_bytes = int(meta_tensor.numel() * meta_tensor.element_size())
    stable_bytes = int(stable_tensor.numel() * stable_tensor.element_size())
    if meta_bytes != stable_bytes:
        raise RuntimeError(
            f"recovery stable owner {label!r} byte size changed: "
            f"source={meta_bytes} prepared={stable_bytes}"
        )
    storage = stable_tensor.untyped_storage()
    byte_offset = int(stable_tensor.data_ptr()) - int(storage.data_ptr())
    element_size = int(meta_tensor.element_size())
    if byte_offset < 0 or byte_offset % element_size:
        raise RuntimeError(f"recovery stable owner {label!r} has an incompatible storage offset")
    if byte_offset + meta_bytes > int(storage.nbytes()):
        raise RuntimeError(f"recovery stable owner {label!r} exceeds its backing storage")

    tensor = torch_module.empty(
        0,
        dtype=meta_tensor.dtype,
        device=stable_tensor.device,
        requires_grad=False,
    )
    tensor.set_(
        storage,
        byte_offset // element_size,
        tuple(meta_tensor.size()),
        tuple(meta_tensor.stride()),
    )
    tensor.__class__ = meta_tensor.__class__
    tensor.__dict__ = meta_tensor.__dict__.copy()
    return tensor


@dataclass
class _B12xRecoveryStorageAdapter:
    """Bind B12x's transferred source format to its captured stable owner.

    vLLM's layerwise reloader reconstructs ordinary Parameters on demand. B12x
    deliberately releases its source Parameter handles after transferring or
    repacking them into ``_prepared_experts``. During recovery, expose those
    same allocations through source-format views while each layer is loaded;
    B12x then performs its normal in-place preparation directly at the virtual
    addresses referenced by captured graphs.
    """

    module_name: str
    layer: Any
    owner: Any
    prepared: Any
    source_parameters_released: bool
    materialized: bool = False
    captured_source_parameters: dict[str, Any] | None = None
    direct_replay_complete: bool = False
    direct_replay_finalized: bool = False
    direct_replay_finalization_deferred: bool = False
    direct_replay_prepared: Any | None = None

    requires_consumer_storage_isolation = True
    allows_partial_direct_replay = True

    @property
    def name(self) -> str:
        return f"b12x:{self.module_name or '<root>'}"

    def begin_reload(self) -> None:
        if not self.source_parameters_released:
            raise RuntimeError(f"{self.name} does not have transferred source parameters")
        self.captured_source_parameters = {
            source_name: getattr(self.layer, source_name)
            for source_name, _prepared_name in B12X_SOURCE_STORAGE_FIELDS
        }
        self.materialized = False
        self.direct_replay_complete = False
        self.direct_replay_finalized = False
        self.direct_replay_finalization_deferred = False
        self.direct_replay_prepared = None
        self.owner._prepared_experts = None
        self.owner._source_parameters_released = False

    def owns_layer(self, layer: Any) -> bool:
        return layer is self.layer

    @property
    def direct_replay_destination_names(self) -> frozenset[str]:
        prefix = f"{self.module_name}." if self.module_name else ""
        return frozenset(
            prefix + source_name for source_name, _prepared_name in B12X_SOURCE_STORAGE_FIELDS
        )

    def _live_owner(self) -> Any | None:
        """Resolve the owner currently installed by the model's quant method.

        Some vLLM quant methods rebuild their modular MoE kernel in
        ``process_weights_after_loading``.  In that case the owner discovered
        before reload is intentionally replaced, even though the new B12x
        owner consumes the stable source aliases installed by this adapter.
        """
        quant_method = getattr(self.layer, "quant_method", None)
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        owner = getattr(moe_kernel, "fused_experts", None)
        lookup = getattr(owner, "_lookup_prepared_experts", None)
        return owner if callable(lookup) else None

    def materialize_layer(self, torch_module: Any, info: Any) -> None:
        restore_metadata = getattr(info, "restore_metadata", None)
        if (
            not isinstance(restore_metadata, tuple)
            or len(restore_metadata) != 2
            or not isinstance(restore_metadata[0], dict)
        ):
            raise RuntimeError(f"{self.name} has invalid vLLM reload metadata")
        restore_parameters = restore_metadata[0]
        for source_name, prepared_name in B12X_SOURCE_STORAGE_FIELDS:
            meta_tensor = restore_parameters.get(source_name)
            stable_tensor = getattr(self.prepared, prepared_name, None)
            if meta_tensor is None or stable_tensor is None:
                raise RuntimeError(f"{self.name} cannot bind {source_name!r} to {prepared_name!r}")
            alias = _alias_meta_tensor_from_storage(
                torch_module,
                meta_tensor,
                stable_tensor,
                label=f"{self.name}.{source_name}",
            )
            setattr(self.layer, source_name, alias)
        # A direct replay may bind these aliases before vLLM initializes its
        # layerwise reload. That initialization temporarily restores the layer
        # to meta, so bind the same stable owner again when vLLM materializes
        # the layer for fallback tensors. Rebinding is idempotent; no bytes are
        # cleared here.
        self.materialized = True

    def begin_direct_replay(self, torch_module: Any) -> None:
        """Expose and clear source-format views before exact copy replay."""
        layerwise = importlib.import_module("vllm.model_executor.model_loader.reload.layerwise")
        get_info = getattr(layerwise, "get_layerwise_info", None)
        if not callable(get_info):
            raise RuntimeError(f"{self.name} cannot access vLLM reload metadata")
        self.materialize_layer(torch_module, get_info(self.layer))
        with torch_module.no_grad():
            for source_name, _prepared_name in B12X_SOURCE_STORAGE_FIELDS:
                getattr(self.layer, source_name).zero_()

    def configure_direct_replay(self, destination_names: frozenset[str]) -> None:
        """Record whether replay replaces every source consumed by this owner."""
        requested = self.direct_replay_destination_names
        if not destination_names or not destination_names <= requested:
            raise RuntimeError(f"{self.name} received invalid direct replay destinations")
        self.direct_replay_complete = destination_names == requested

    def configure_deferred_direct_replay_finalization(
        self,
        normal_reload_destination_names: frozenset[str],
    ) -> None:
        """Let vLLM finalize once when another tensor activates this layer."""
        prefix = f"{self.module_name}." if self.module_name else ""
        self.direct_replay_finalization_deferred = any(
            name.startswith(prefix) for name in normal_reload_destination_names
        )

    def _validate_stable_prepared(self, prepared: Any) -> None:
        for field_name in B12X_RELOAD_WEIGHT_FIELDS:
            before = getattr(self.prepared, field_name, None)
            after = getattr(prepared, field_name, None)
            if before is None or after is None:
                raise RuntimeError(f"{self.name} prepared owner is missing {field_name!r}")
            before_bytes = int(before.numel() * before.element_size())
            after_bytes = int(after.numel() * after.element_size())
            if int(before.data_ptr()) != int(after.data_ptr()) or before_bytes != after_bytes:
                raise RuntimeError(
                    f"{self.name} changed stable storage for {field_name!r}: "
                    f"before={int(before.data_ptr())}/{before_bytes} "
                    f"after={int(after.data_ptr())}/{after_bytes}"
                )

    def finalize_direct_replay(self, torch_module: Any) -> None:
        """Run the installed vLLM quantizer after a complete source replay.

        The observed checkpoint copies reconstruct B12x's source-format views;
        they do not replace its supported packing and ownership-transfer step.
        Calling the layer's existing quant method keeps that transformation in
        the pinned vLLM/B12x implementation and makes the adapter independent
        of a particular packed layout.
        """
        if not self.direct_replay_complete:
            return
        if self.direct_replay_finalization_deferred:
            return
        quant_method = getattr(self.layer, "quant_method", None)
        process = getattr(quant_method, "process_weights_after_loading", None)
        if not callable(process):
            raise RuntimeError(f"{self.name} has no vLLM quantization finalizer")
        if hasattr(self.layer, "_already_called_process_weights_after_loading"):
            delattr(self.layer, "_already_called_process_weights_after_loading")
        with torch_module.no_grad():
            process(self.layer)
        update_tp = getattr(self.layer, "update_param_tp_status", None)
        if callable(update_tp):
            update_tp()

        owner = self._live_owner()
        lookup = getattr(owner, "_lookup_prepared_experts", None)
        prepared = lookup() if callable(lookup) else None
        if prepared is None:
            raise RuntimeError(f"{self.name} finalizer did not publish prepared weights")
        self._validate_stable_prepared(prepared)
        if not bool(getattr(owner, "_source_parameters_released", False)):
            raise RuntimeError(f"{self.name} finalizer did not release source parameters")
        self.direct_replay_prepared = prepared
        self.direct_replay_finalized = True

    def end_direct_replay(self) -> None:
        """Restore captured released handles before vLLM records kernel state.

        Replayed bytes remain in ``self.prepared``. The temporary full-shape
        aliases must not become vLLM's kernel-tensor baseline because B12x's
        normal finalizer intentionally replaces its source parameters with
        empty tensors after transferring them back into prepared storage.
        """
        if self.captured_source_parameters is None:
            raise RuntimeError(f"{self.name} has no captured source parameters")
        for source_name, parameter in self.captured_source_parameters.items():
            setattr(self.layer, source_name, parameter)
        if (
            self.direct_replay_complete
            and not self.direct_replay_finalized
            and not self.direct_replay_finalization_deferred
        ):
            # vLLM deliberately does not invoke the layer finalizer when every
            # loadable source was preloaded. Retaining the captured owner is a
            # safe fallback only for adapters whose replay already represents
            # runtime format. B12x's normal orchestrated path finalizes first.
            owner = self._live_owner()
            if owner is None:
                raise RuntimeError(f"{self.name} lost its B12x prepared owner")
            owner._prepared_experts = self.prepared
            owner._source_parameters_released = True

    def finish_reload(self) -> None:
        if not self.materialized:
            raise RuntimeError(f"{self.name} was not materialized during reload")
        owner = self._live_owner()
        if owner is None:
            raise RuntimeError(f"{self.name} lost its B12x prepared owner")
        lookup = getattr(owner, "_lookup_prepared_experts", None)
        prepared = lookup() if callable(lookup) else None
        if prepared is None and self.direct_replay_complete:
            # A model-level finalizer may replace the modular owner even when
            # the layerwise quantizer had no work. Retain the finalized owner,
            # including its newly derived state, across that replacement.
            retained = (
                self.direct_replay_prepared
                if self.direct_replay_prepared is not None
                else self.prepared
            )
            owner._prepared_experts = retained
            owner._source_parameters_released = True
            prepared = retained
        if prepared is None:
            raise RuntimeError(f"{self.name} did not rebuild its prepared owner")
        self._validate_stable_prepared(prepared)
        if not bool(getattr(owner, "_source_parameters_released", False)):
            raise RuntimeError(f"{self.name} did not release its source parameters")
        self.captured_source_parameters = None
        self.direct_replay_complete = False
        self.direct_replay_finalized = False
        self.direct_replay_finalization_deferred = False
        self.direct_replay_prepared = None

    def abort_reload(self) -> None:
        # The quant method may have installed a replacement modular kernel
        # before a later adapter failed.  Restore both the captured owner and
        # the currently reachable one so either lifecycle is left usable.
        owners = [self.owner, self._live_owner()]
        restored: set[int] = set()
        for owner in owners:
            if owner is None or id(owner) in restored:
                continue
            restored.add(id(owner))
            owner._prepared_experts = self.prepared
            owner._source_parameters_released = self.source_parameters_released
        if self.captured_source_parameters is not None:
            for source_name, parameter in self.captured_source_parameters.items():
                setattr(self.layer, source_name, parameter)
            self.captured_source_parameters = None
        self.direct_replay_complete = False
        self.direct_replay_finalized = False
        self.direct_replay_finalization_deferred = False
        self.direct_replay_prepared = None


def _discover_b12x_recovery_storage(model: Any) -> list[Any]:
    """Return B12x adapters only for its transferred-storage lifecycle."""
    adapters: list[Any] = []
    for module_name, module, owner, prepared in _b12x_prepared_owners(model):
        plan = getattr(prepared, "plan", None)
        if not bool(getattr(plan, "discards_source_parameters", False)):
            continue
        adapters.append(
            _B12xRecoveryStorageAdapter(
                module_name=module_name,
                layer=module,
                owner=owner,
                prepared=prepared,
                source_parameters_released=bool(
                    getattr(owner, "_source_parameters_released", False)
                ),
            )
        )
    return adapters


_RECOVERY_STORAGE_ADAPTER_DISCOVERERS: list[Any] = [_discover_b12x_recovery_storage]


def register_recovery_storage_adapter(discoverer: Any) -> None:
    """Register a backend-neutral stable-storage adapter discoverer.

    A discoverer accepts the live model and returns adapter objects implementing
    ``begin_reload``, ``owns_layer``, ``materialize_layer``, ``finish_reload``,
    and ``abort_reload``. Direct-replay adapters may additionally implement
    ``configure_direct_replay``, ``begin_direct_replay``,
    ``finalize_direct_replay``, and ``end_direct_replay``. This exposes
    temporary destination aliases and lets an adapter invoke its framework's
    supported source-to-runtime finalizer when replay alone is not the runtime
    representation, without teaching the loader a model-specific layout.
    This keeps the vLLM reload bridge independent of any one quantization
    backend and leaves an equivalent integration point for an SGLang bridge.
    Adapters that need the framework's layer finalizer may expose
    ``required_normal_reload_destination_names`` and optionally
    ``configure_deferred_direct_replay_finalization``.
    """
    if not callable(discoverer):
        raise TypeError("recovery storage adapter discoverer must be callable")
    if discoverer not in _RECOVERY_STORAGE_ADAPTER_DISCOVERERS:
        _RECOVERY_STORAGE_ADAPTER_DISCOVERERS.append(discoverer)


def _recovery_storage_adapters(model: Any) -> list[Any]:
    """Discover backend adapters; ordinary registered weights need none."""
    adapters: list[Any] = []
    for discoverer in _RECOVERY_STORAGE_ADAPTER_DISCOVERERS:
        discovered = discoverer(model)
        if discovered:
            adapters.extend(discovered)
    return adapters


def _discover_b12x_mhc_reload_dependencies(model: Any) -> frozenset[str]:
    """Keep B12x MHC inputs live for its eager model-level refresh.

    Discovery is structural and intentionally does not depend on a concrete
    DeepSeek model class. The generic dependency contract below is also usable
    by other vLLM architectures and by a future SGLang reload bridge.
    """
    names: set[str] = set()
    for module_name, module in _named_module_items(model):
        if not bool(getattr(module, "_use_b12x_mhc", False)):
            continue
        if not callable(getattr(module, "refresh_b12x_mhc_bf16_weights", None)):
            continue
        for tensor_name in B12X_MHC_RELOAD_DEPENDENCY_FIELDS:
            if getattr(module, tensor_name, None) is not None:
                names.add(_qualify_tensor_name(module_name, tensor_name))
    return frozenset(names)


_RECOVERY_RELOAD_DEPENDENCY_DISCOVERERS: list[Any] = [_discover_b12x_mhc_reload_dependencies]


def register_recovery_reload_dependencies(discoverer: Any) -> None:
    """Register destinations that model code consumes during weight reload.

    A discoverer accepts the live model and returns fully qualified registered
    tensor names. Those destinations remain on the framework's normal source
    path even when their byte-copy plan is otherwise replayable. This small
    contract avoids embedding architecture-specific reload ordering in the
    safetensors transport.
    """
    if not callable(discoverer):
        raise TypeError("recovery reload dependency discoverer must be callable")
    if discoverer not in _RECOVERY_RELOAD_DEPENDENCY_DISCOVERERS:
        _RECOVERY_RELOAD_DEPENDENCY_DISCOVERERS.append(discoverer)


def _recovery_reload_dependency_names(model: Any) -> frozenset[str]:
    names: set[str] = set()
    for discoverer in _RECOVERY_RELOAD_DEPENDENCY_DISCOVERERS:
        discovered = discoverer(model)
        if discovered is None:
            continue
        if not isinstance(discovered, (set, frozenset, tuple, list)) or not all(
            isinstance(name, str) and name for name in discovered
        ):
            raise RuntimeError("recovery reload dependency discoverer returned invalid names")
        names.update(discovered)
    return frozenset(names)


def _adapter_replay_plan(
    model: Any,
    adapters: Sequence[Any],
    *,
    excluded_destinations: frozenset[str] = frozenset(),
) -> RecoveryCopyPlan:
    """Keep only complete, byte-exact plans claimed by a backend adapter."""
    captured = _model_checkpoint_copy_plan(model)
    if captured is None:
        return RecoveryCopyPlan((), frozenset())

    by_destination: dict[str, list[RecoveryCopy]] = {}
    for copy in captured.copies:
        by_destination.setdefault(copy.destination_name, []).append(copy)

    candidates: list[tuple[frozenset[str], list[RecoveryCopy]]] = []
    rejected: set[str] = set()
    for adapter in adapters:
        raw_names = getattr(adapter, "direct_replay_destination_names", ())
        requested_names = frozenset(raw_names)
        if not requested_names:
            continue
        partial = bool(getattr(adapter, "allows_partial_direct_replay", False))
        if requested_names & excluded_destinations and not partial:
            # A destination consumed eagerly by model code must remain on the
            # framework's ordinary reload path. Do not partially select an
            # otherwise complete adapter behind its back.
            rejected.update(requested_names)
            continue
        available_names = frozenset(
            name
            for name in requested_names
            if name not in excluded_destinations
            and name not in captured.unsupported_destinations
            and by_destination.get(name)
        )
        if partial:
            names = available_names
            rejected.update(requested_names - available_names)
            if not names:
                continue
        else:
            names = requested_names
            if names != available_names:
                rejected.update(names)
                continue
        adapter_copies = [copy for name in sorted(names) for copy in by_destination[name]]
        candidates.append((names, adapter_copies))

    # A checkpoint tensor may legitimately feed multiple independently
    # finalized destinations. Select complete adapters as a group, then remove
    # any candidate whose source is also consumed by an unselected destination.
    # This makes skip-source eligibility a property of the observed graph, not
    # of a model or backend naming convention.
    while candidates:
        selected_names = frozenset().union(*(names for names, _copies in candidates))
        unsafe_sources = {
            copy.source_key
            for copy in captured.copies
            if copy.destination_name not in selected_names
        }
        unsafe = [
            (names, copies)
            for names, copies in candidates
            if any(copy.source_key in unsafe_sources for copy in copies)
        ]
        if not unsafe:
            break
        for names, copies in unsafe:
            rejected.update(names)
            candidates.remove((names, copies))

    selected = [copy for _names, copies in candidates for copy in copies]
    selected.sort(
        key=lambda copy: (
            copy.source_path,
            copy.source_file_offset,
            copy.source_name,
            copy.destination_name,
            copy.destination_view_offset_bytes,
        )
    )
    return RecoveryCopyPlan(tuple(selected), frozenset(rejected))


def _fully_covered_replay_destinations(
    model: Any,
    captured: RecoveryCopyPlan,
) -> frozenset[str]:
    """Select ordinary CUDA tensors reconstructed by exact contiguous views."""
    destinations = dict(model.named_parameters())
    destinations.update(dict(model.named_buffers()))
    by_destination: dict[str, list[RecoveryCopy]] = {}
    for copy in captured.copies:
        by_destination.setdefault(copy.destination_name, []).append(copy)

    selected: set[str] = set()
    for name, copies in by_destination.items():
        tensor = destinations.get(name)
        if (
            name in captured.unsupported_destinations
            or tensor is None
            or not bool(getattr(tensor, "is_cuda", False))
            or not bool(tensor.is_contiguous())
        ):
            continue
        tensor_bytes = int(tensor.numel() * tensor.element_size())
        if tensor_bytes <= 0:
            continue
        ranges: list[tuple[int, int]] = []
        compatible = True
        for copy in copies:
            start = copy.destination_view_offset_bytes
            end = start + copy.copy_bytes
            if (
                copy.destination_shape != tuple(tensor.shape)
                or copy.destination_dtype != str(tensor.dtype)
                or copy.destination_view_stride != _contiguous_stride(copy.destination_view_shape)
                or start < 0
                or end > tensor_bytes
            ):
                compatible = False
                break
            ranges.append((start, end))
        if not compatible:
            continue
        covered_end = 0
        for start, end in sorted(ranges):
            if start > covered_end:
                break
            covered_end = max(covered_end, end)
        if covered_end == tensor_bytes:
            selected.add(name)
    return frozenset(selected)


def _recovery_replay_plan(model: Any, adapters: Sequence[Any]) -> RecoveryCopyPlan:
    """Combine adapter-owned replay with conservative ordinary tensor replay."""
    captured = _model_checkpoint_copy_plan(model)
    if captured is None:
        return RecoveryCopyPlan((), frozenset())
    adapter_dependencies = frozenset().union(
        *(
            frozenset(getattr(adapter, "required_normal_reload_destination_names", ()))
            for adapter in adapters
        )
    )
    reload_dependencies = _recovery_reload_dependency_names(model) | adapter_dependencies
    adapter_plan = _adapter_replay_plan(
        model,
        adapters,
        excluded_destinations=reload_dependencies,
    )
    adapter_names = frozenset(copy.destination_name for copy in adapter_plan.copies)
    generic_names = (
        _fully_covered_replay_destinations(model, captured) - adapter_names - reload_dependencies
    )

    # A skipped source must not feed any destination left to vLLM. Remove
    # generic candidates to a fixed point; adapter candidates already enforce
    # this invariant as complete backend-owned groups.
    selected_names = set(adapter_names | generic_names)
    while True:
        unsafe_sources = {
            copy.source_key
            for copy in captured.copies
            if copy.destination_name not in selected_names
        }
        rejected_names = {
            copy.destination_name
            for copy in captured.copies
            if copy.destination_name in generic_names and copy.source_key in unsafe_sources
        }
        if not rejected_names:
            break
        generic_names -= rejected_names
        selected_names = set(adapter_names | generic_names)

    selected = [copy for copy in captured.copies if copy.destination_name in selected_names]
    selected.sort(
        key=lambda copy: (
            copy.source_path,
            copy.source_file_offset,
            copy.source_name,
            copy.destination_name,
            copy.destination_view_offset_bytes,
        )
    )
    rejected = set(adapter_plan.unsupported_destinations)
    rejected.update(
        _fully_covered_replay_destinations(model, captured) - generic_names - adapter_names
    )
    return RecoveryCopyPlan(tuple(selected), frozenset(rejected))


def _gather_recovery_copy_plans(
    torch_module: Any, local_plan: RecoveryCopyPlan
) -> tuple[RecoveryCopyPlan, ...]:
    """Publish rank-local copy views on the snapshot-portable TP group.

    A vLLM CPU/Gloo group retains TCP state which is not portable across an
    n610 process restore.  The TP device group is explicitly primed before the
    capture boundary and restored by ColdSnap's NCCL path, so every recovery
    collective which may run again after restore must use that group.
    """
    try:
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
    except (AssertionError, ImportError):
        return (local_plan,)
    world_size = int(tp.world_size)
    if world_size <= 1:
        return (local_plan,)
    group = tp.device_group
    gathered: list[Any] = [None] * world_size
    torch_module.distributed.all_gather_object(
        gathered,
        local_plan,
        group=group,
    )
    if not all(isinstance(plan, RecoveryCopyPlan) for plan in gathered):
        raise RuntimeError("recovery TP peers published an invalid copy plan")
    return tuple(gathered)


def _bounded_rank_io_weights(rates: Sequence[float]) -> tuple[float, ...]:
    if not rates or not all(
        isinstance(value, (float, int)) and not isinstance(value, bool) and value > 0
        for value in rates
    ):
        raise ValueError("recovery rank I/O rates must be positive numbers")
    normalized = [float(value) for value in rates]
    ordered = sorted(normalized)
    median = ordered[len(ordered) // 2]
    lower = median / 4.0
    upper = median * 4.0
    return tuple(min(max(rate, lower), upper) for rate in normalized)


def _damped_rank_io_weights(
    rates: Sequence[float],
    exponent: float,
) -> tuple[float, ...]:
    """Retain measured asymmetry without overfitting one initial load."""
    bounded = _bounded_rank_io_weights(rates)
    if not 0.0 <= exponent <= 1.0:
        raise ValueError("recovery rank I/O weight exponent must be in [0, 1]")
    center = math.exp(sum(math.log(rate) for rate in bounded) / len(bounded))
    return tuple(center * (rate / center) ** exponent for rate in bounded)


def _gather_recovery_io_weights(
    torch_module: Any,
    local_rate: float | None,
) -> tuple[float, ...]:
    """Persist initial-load device rates while capture latency is off-path."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
    except (AssertionError, ImportError):
        gathered = (local_rate,) if local_rate is not None else ()
        return _damped_rank_io_weights(gathered, _io_weight_exponent()) if gathered else ()
    world_size = int(tp.world_size)
    if world_size <= 1:
        gathered = (local_rate,) if local_rate is not None else ()
        return _damped_rank_io_weights(gathered, _io_weight_exponent()) if gathered else ()
    # See _gather_recovery_copy_plans: captured CPU/Gloo transports are not a
    # portable post-restore control plane.
    group = tp.device_group
    gathered: list[Any] = [None] * world_size
    torch_module.distributed.all_gather_object(
        gathered,
        local_rate,
        group=group,
    )
    if not all(
        isinstance(value, (float, int)) and not isinstance(value, bool) and value > 0
        for value in gathered
    ):
        return ()
    return _damped_rank_io_weights(gathered, _io_weight_exponent())


def _io_weight_exponent() -> float:
    raw = os.environ.get(
        LOADER_IO_WEIGHT_EXPONENT_ENV,
        str(DEFAULT_IO_WEIGHT_EXPONENT),
    )
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{LOADER_IO_WEIGHT_EXPONENT_ENV} must be a number") from error
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{LOADER_IO_WEIGHT_EXPONENT_ENV} must be in [0, 1]")
    return value


def _residual_tensor_names_by_layer(model: Any) -> dict[int, frozenset[str]]:
    """Return direct registered tensors preserved by the residual capsule."""
    destination_names = _model_checkpoint_destination_names(model)
    if destination_names is None:
        return {}
    if not isinstance(destination_names, (set, frozenset)) or not all(
        isinstance(name, str) for name in destination_names
    ):
        raise RuntimeError("recovery checkpoint destination tensor names are invalid")

    residual: dict[int, frozenset[str]] = {}
    for module_name, module in model.named_modules():
        registered: dict[str, Any] = {}
        for attribute in ("_parameters", "_buffers"):
            values = getattr(module, attribute, None)
            if isinstance(values, dict):
                registered.update(values)
        names = frozenset(
            leaf_name
            for leaf_name, tensor in registered.items()
            if tensor is not None
            and (f"{module_name}.{leaf_name}" if module_name else leaf_name)
            not in destination_names
        )
        if names:
            residual[id(module)] = names
    return residual


@contextmanager
def _vllm_residual_tensor_skip(meta: Any, names: frozenset[str]) -> Iterator[None]:
    """Temporarily keep one layer's residual tensors out of meta reload."""
    if not names:
        yield
        return
    load_added = names - meta.SKIP_LOAD_TENSORS
    tensor_added = names - meta.SKIP_TENSORS
    meta.SKIP_LOAD_TENSORS.update(load_added)
    meta.SKIP_TENSORS.update(tensor_added)
    try:
        yield
    finally:
        meta.SKIP_TENSORS.difference_update(tensor_added)
        meta.SKIP_LOAD_TENSORS.difference_update(load_added)


@contextmanager
def _vllm_preloaded_tensor_skip(meta: Any, names: frozenset[str]) -> Iterator[None]:
    """Exclude replayed tensors from layer completion accounting only.

    Unlike residual runtime state, these tensors must still participate in
    meta restoration and materialization so an adapter can expose its stable
    storage. They simply do not need a second weight-loader invocation.
    """
    load_added = names - meta.SKIP_LOAD_TENSORS
    meta.SKIP_LOAD_TENSORS.update(load_added)
    try:
        yield
    finally:
        meta.SKIP_LOAD_TENSORS.difference_update(load_added)


@contextmanager
def _recovery_reload_storage(
    model: Any,
    *,
    prepare_preloaded: bool = False,
) -> Iterator[tuple[str, ...]]:
    """Install per-backend source materializers around vLLM's normal reload."""
    adapters = _recovery_storage_adapters(model)
    residual_by_layer = _residual_tensor_names_by_layer(model)
    if not adapters and not residual_by_layer and not prepare_preloaded:
        yield ()
        return

    layerwise = importlib.import_module("vllm.model_executor.model_loader.reload.layerwise")
    meta = importlib.import_module("vllm.model_executor.model_loader.reload.meta")
    original_materialize = getattr(layerwise, "materialize_layer", None)
    original_restore = getattr(layerwise, "restore_layer_on_meta", None)
    original_get_size = getattr(layerwise, "get_layer_size", None)
    original_wrap = getattr(layerwise, "_wrap_parameters_weight_loader", None)
    if not all(
        callable(value)
        for value in (
            original_materialize,
            original_restore,
            original_get_size,
            original_wrap,
        )
    ):
        raise RuntimeError("recovery storage requires vLLM layerwise reload hooks")
    import torch

    tensor_owners: dict[str, tuple[int, str, Any]] = {}
    for module_name, module in model.named_modules():
        for attribute in ("_parameters", "_buffers"):
            values = getattr(module, attribute, None)
            if not isinstance(values, dict):
                continue
            for tensor_name, tensor in values.items():
                if tensor is not None:
                    tensor_owners[_qualify_tensor_name(module_name, tensor_name)] = (
                        id(module),
                        tensor_name,
                        tensor,
                    )

    adapter_destination_names = frozenset().union(
        *(
            frozenset(getattr(adapter, "direct_replay_destination_names", ()))
            for adapter in adapters
        )
    )

    started: list[Any] = []
    succeeded = False
    isolation_token = _RECOVERY_CONSUMER_STORAGE_ISOLATION.set(
        any(
            bool(getattr(adapter, "requires_consumer_storage_isolation", False))
            for adapter in adapters
        )
    )
    adapters_token = _ACTIVE_RECOVERY_ADAPTERS.set(tuple(adapters))
    try:
        for adapter in adapters:
            adapter.begin_reload()
            started.append(adapter)

        def residual_names(layer: Any) -> frozenset[str]:
            return residual_by_layer.get(id(layer), frozenset())

        def preloaded_names(layer: Any) -> frozenset[str]:
            layer_id = id(layer)
            return frozenset(
                tensor_name
                for full_name in _RECOVERY_PRELOADED_DESTINATION_NAMES.get()
                if (owner := tensor_owners.get(full_name)) is not None and owner[0] == layer_id
                for tensor_name in (owner[1],)
            )

        def generic_preloaded_tensors(layer: Any) -> dict[str, Any]:
            layer_id = id(layer)
            return {
                owner[1]: owner[2]
                for full_name in _RECOVERY_PRELOADED_DESTINATION_NAMES.get()
                if full_name not in adapter_destination_names
                and (owner := tensor_owners.get(full_name)) is not None
                and owner[0] == layer_id
            }

        @functools.wraps(original_restore)
        def restore_recovery_layer(layer: Any, info: Any) -> Any:
            with _vllm_residual_tensor_skip(meta, residual_names(layer)):
                return original_restore(layer, info)

        @functools.wraps(original_get_size)
        def recovery_layer_size(layer: Any) -> Any:
            with _vllm_residual_tensor_skip(meta, residual_names(layer)):
                with _vllm_preloaded_tensor_skip(meta, preloaded_names(layer)):
                    return original_get_size(layer)

        @functools.wraps(original_wrap)
        def wrap_recovery_parameters(layer: Any) -> Any:
            with _vllm_residual_tensor_skip(meta, residual_names(layer)):
                with _vllm_preloaded_tensor_skip(meta, preloaded_names(layer)):
                    return original_wrap(layer)

        @functools.wraps(original_materialize)
        def materialize_recovery_layer(layer: Any, info: Any) -> Any:
            matches = [adapter for adapter in adapters if adapter.owns_layer(layer)]
            if len(matches) > 1:
                raise RuntimeError("multiple recovery storage adapters claimed one vLLM layer")
            with _vllm_residual_tensor_skip(meta, residual_names(layer)):
                if matches:
                    matches[0].materialize_layer(torch, info)
                for tensor_name, stable_tensor in generic_preloaded_tensors(layer).items():
                    current = getattr(layer, tensor_name, None)
                    if current is None:
                        raise RuntimeError(
                            f"recovery preloaded destination disappeared: {tensor_name}"
                        )
                    if bool(getattr(current, "is_meta", False)):
                        alias = _alias_meta_tensor_from_storage(
                            torch,
                            current,
                            stable_tensor,
                            label=f"preloaded:{tensor_name}",
                        )
                        setattr(layer, tensor_name, alias)
                        continue
                    try:
                        shares_stable_storage = int(current.data_ptr()) == int(
                            stable_tensor.data_ptr()
                        )
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        shares_stable_storage = current is stable_tensor
                    if not shares_stable_storage:
                        raise RuntimeError(
                            "recovery preloaded destination materialized outside "
                            f"captured stable storage: {tensor_name}"
                        )
                return original_materialize(layer, info)

        layerwise.restore_layer_on_meta = restore_recovery_layer
        layerwise.get_layer_size = recovery_layer_size
        layerwise._wrap_parameters_weight_loader = wrap_recovery_parameters
        layerwise.materialize_layer = materialize_recovery_layer
        yield tuple(adapter.name for adapter in adapters)
        for adapter in adapters:
            adapter.finish_reload()
        succeeded = True
    finally:
        _ACTIVE_RECOVERY_ADAPTERS.reset(adapters_token)
        _RECOVERY_CONSUMER_STORAGE_ISOLATION.reset(isolation_token)
        layerwise.materialize_layer = original_materialize
        layerwise.restore_layer_on_meta = original_restore
        layerwise.get_layer_size = original_get_size
        layerwise._wrap_parameters_weight_loader = original_wrap
        if not succeeded:
            for adapter in reversed(started):
                adapter.abort_reload()


def _tensor_weight_layout(
    tensors: Iterable[tuple[str, Any]],
    backend: Any,
    *,
    label: str,
    allow_outside_weight_pool: bool = False,
) -> list[tuple[int, int, str]]:
    """Resolve tensor bytes and placement-independent merged-range identities."""
    provider = getattr(backend, "memory_provider", None)
    allocations_method = getattr(provider, "allocations", None)
    if not callable(allocations_method):
        raise RuntimeError(f"{label} backend cannot enumerate weight allocations")
    allocations = sorted(allocations_method("weights"), key=lambda allocation: allocation.pointer)
    if not allocations:
        raise RuntimeError(f"{label} backend has no live weight allocations")

    intervals: list[tuple[int, int, int, str]] = []
    seen: set[tuple[str, int, int]] = set()
    for name, tensor in tensors:
        elements = int(tensor.numel())
        if elements == 0:
            continue
        if not tensor.is_cuda:
            if allow_outside_weight_pool:
                continue
            raise RuntimeError(f"{label} tensor {name!r} must use CUDA storage")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{label} tensor {name!r} must be contiguous")
        pointer = int(tensor.data_ptr())
        size = elements * int(tensor.element_size())
        identity = (name, pointer, size)
        if identity in seen:
            continue
        seen.add(identity)
        matches = [
            allocation
            for allocation in allocations
            if allocation.pointer <= pointer
            and pointer + size <= allocation.pointer + allocation.size
        ]
        if not matches and allow_outside_weight_pool:
            continue
        if len(matches) != 1:
            raise RuntimeError(
                f"{label} tensor {name!r} does not resolve to exactly one "
                f"weight allocation (matches={len(matches)})"
            )
        intervals.append((int(matches[0].pointer), pointer, pointer + size, name))

    merged: list[dict[str, Any]] = []
    for allocation_pointer, start, end, name in sorted(intervals):
        if (
            merged
            and allocation_pointer == merged[-1]["allocation_pointer"]
            and start <= merged[-1]["end"]
        ):
            merged[-1]["end"] = max(int(merged[-1]["end"]), end)
            merged[-1]["members"].append((name, start, end))
        else:
            merged.append(
                {
                    "allocation_pointer": allocation_pointer,
                    "start": start,
                    "end": end,
                    "members": [(name, start, end)],
                }
            )
    if not merged:
        raise RuntimeError(f"{label} exposes no CUDA weight-pool bytes")
    layout: list[tuple[int, int, str]] = []
    for item in merged:
        start = int(item["start"])
        end = int(item["end"])
        members = sorted(
            (name, member_start - start, member_end - member_start)
            for name, member_start, member_end in item["members"]
        )
        semantic_id = hashlib.sha256(
            json.dumps(members, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        layout.append((start, end - start, semantic_id))
    return layout


def _model_weight_layout(model: Any, backend: Any) -> list[tuple[int, int, str]]:
    """Resolve reload-owned bytes and their placement-independent identities."""
    return _tensor_weight_layout(
        _model_weight_tensors(model),
        backend,
        label="recovery model weight",
    )


def _native_model_payload_layout(model: Any, backend: Any) -> list[tuple[int, int, str]]:
    """Resolve all model-owned state that a native wake must reproduce.

    Recovery reconstructs only checkpoint destinations and separately restores
    a bounded derived-buffer capsule. A native payload has no reload callback,
    so it must also carry registered parameters, buffers, and backend-owned
    prepared tensors. Non-weight-pool tensors remain owned by their normal
    runtime lifecycle and are intentionally skipped.
    """
    tensors = list(model.named_parameters())
    tensors.extend(list(getattr(model, "named_buffers", lambda: ())()))
    for module_name, _module, _owner, prepared in _b12x_prepared_owners(model):
        prefix = module_name or "<root>"
        for field_name in B12X_PREPARED_WEIGHT_FIELDS:
            tensor = getattr(prepared, field_name, None)
            if tensor is None:
                raise RuntimeError(
                    "B12x prepared expert owner is missing native payload "
                    f"field {field_name!r} at {prefix!r}"
                )
            tensors.append((f"{prefix}.prepared_experts.{field_name}", tensor))
    return _tensor_weight_layout(
        tensors,
        backend,
        label="native model payload",
        allow_outside_weight_pool=True,
    )


def _model_weight_ranges(model: Any, backend: Any) -> list[tuple[int, int]]:
    """Compatibility view of the reload-owned model weight layout."""
    return [(pointer, size) for pointer, size, _semantic_id in _model_weight_layout(model, backend)]


def _bind_model_weight_layout(model: Any, backend: Any) -> list[tuple[int, int]]:
    layout = _model_weight_layout(model, backend)
    set_ranges = getattr(backend, "set_model_weight_ranges", None)
    set_semantics = getattr(backend, "set_model_weight_semantics", None)
    if not callable(set_ranges) or not callable(set_semantics):
        raise RuntimeError("recovery backend cannot bind semantic model weight ranges")
    ranges = [(pointer, size) for pointer, size, _semantic_id in layout]
    set_ranges(ranges)
    set_semantics(layout)
    return ranges


def _bind_native_model_payload_layout(model: Any, backend: Any) -> list[tuple[int, int]]:
    layout = _native_model_payload_layout(model, backend)
    setter = getattr(backend, "set_native_model_payload_semantics", None)
    if not callable(setter):
        raise RuntimeError("recovery backend cannot bind native model payload semantics")
    setter(layout)
    return [(pointer, size) for pointer, size, _semantic_id in layout]


def _sleep_layout_binding_policy(backend: Any) -> tuple[bool, bool]:
    """Choose reload and native semantic layouts for a live sleep.

    Native activation must bind the full portable payload layout even when a
    capture-time export has already completed.  The reload layout remains
    useful for recovery metadata, while the native layout is the identity used
    to relocate the immutable model pack.
    """
    exports_native = bool(getattr(backend, "exports_model_payload", False))
    uses_native = bool(getattr(backend, "uses_native_model_payload", False))
    bind_reload = (
        bool(getattr(backend, "uses_model_weight_recovery", False)) or exports_native or uses_native
    )
    return bind_reload, exports_native or uses_native


def _restore_derived_buffers(torch_module: Any, model: Any, captured: dict[str, Any]) -> int:
    current = dict(model.named_buffers())
    restored_bytes = 0
    with torch_module.no_grad():
        for name, source in captured.items():
            target = current.get(name)
            if target is None:
                raise RuntimeError(f"recovery derived buffer disappeared: {name}")
            if tuple(target.shape) != tuple(source.shape) or target.dtype != source.dtype:
                raise RuntimeError(
                    "recovery derived buffer metadata changed for "
                    f"{name}: captured={tuple(source.shape)}/{source.dtype} "
                    f"restored={tuple(target.shape)}/{target.dtype}"
                )
            target.copy_(source.to(device=target.device))
            restored_bytes += int(target.numel() * target.element_size())
    return restored_bytes


def _tensor_sample(torch_module: Any, tensor: Any, sample_bytes: int) -> Any:
    # A dtype-changing view rejects zero-dimensional tensors. Flatten first so
    # scalar scales and biases use the same byte-sampling path as other tensors.
    raw = tensor.detach().contiguous().reshape(-1).view(torch_module.uint8)
    if raw.numel() > sample_bytes * 2:
        raw = torch_module.cat((raw[:sample_bytes], raw[-sample_bytes:]))
    return raw.cpu().clone()


def _capture_model_samples(
    torch_module: Any, model: Any, sample_bytes: int
) -> dict[str, tuple[tuple[int, ...], str, Any]]:
    tensors = {f"weight:{name}": tensor for name, tensor in _model_semantic_tensors(model)}
    tensors.update({f"buffer:{name}": tensor for name, tensor in model.named_buffers()})
    return {
        name: (
            tuple(tensor.shape),
            str(tensor.dtype),
            _tensor_sample(torch_module, tensor, sample_bytes),
        )
        for name, tensor in tensors.items()
        if not tensor.is_meta
    }


def _compare_model_samples(
    torch_module: Any,
    model: Any,
    expected: dict[str, tuple[tuple[int, ...], str, Any]],
    sample_bytes: int,
) -> list[dict[str, Any]]:
    tensors = {f"weight:{name}": tensor for name, tensor in _model_semantic_tensors(model)}
    tensors.update({f"buffer:{name}": tensor for name, tensor in model.named_buffers()})
    mismatches: list[dict[str, Any]] = []
    for name, (shape, dtype, sample) in expected.items():
        tensor = tensors.get(name)
        reason = None
        if tensor is None:
            reason = "missing"
        elif tuple(tensor.shape) != shape:
            reason = f"shape:{tuple(tensor.shape)}"
        elif str(tensor.dtype) != dtype:
            reason = f"dtype:{tensor.dtype}"
        elif not torch_module.equal(_tensor_sample(torch_module, tensor, sample_bytes), sample):
            reason = "sample"
        if reason is not None:
            mismatches.append(
                {
                    "name": name,
                    "reason": reason,
                    "expected_shape": shape,
                    "expected_dtype": dtype,
                }
            )
    for name in sorted(tensors.keys() - expected.keys()):
        mismatches.append({"name": name, "reason": "unexpected"})
    return mismatches


def _is_int(value: Any) -> bool:
    return type(value) is int


def _checked_elements(shape: tuple[int, ...], name: str) -> int:
    elements = 1
    for dimension in shape:
        if dimension < 0:
            raise RuntimeError(f"negative safetensors dimension for {name!r}")
        if dimension and elements > MAX_INT64 // dimension:
            raise RuntimeError(f"safetensors shape overflows for {name!r}")
        elements *= dimension
    return elements


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb", buffering=0) as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise RuntimeError(f"invalid safetensors header in {path}")
        size = struct.unpack("<Q", raw_size)[0]
        if size <= 0 or size > MAX_HEADER_BYTES:
            raise RuntimeError(f"invalid safetensors header size {size} in {path}")
        payload = stream.read(size)
        if len(payload) != size:
            raise RuntimeError(f"short safetensors header in {path}")
    try:
        header = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid safetensors JSON header in {path}") from error
    if not isinstance(header, dict):
        raise RuntimeError(f"safetensors header is not an object in {path}")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise RuntimeError(f"invalid safetensors metadata in {path}")
    return 8 + size, header


def _matches_prefix(name: str, prefixes: Sequence[str] | None) -> bool:
    return prefixes is None or any(
        name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes
    )


def _descriptors(
    path: Path,
    *,
    indexed_tensor_files: dict[str, str] | None,
    weight_name_prefixes: Sequence[str] | None,
) -> list[SafetensorDescriptor]:
    data_offset, header = _read_safetensors_header(path)
    file_bytes = path.stat().st_size
    absolute_path = os.path.abspath(path)
    result: list[SafetensorDescriptor] = []
    occupied: list[tuple[int, int, str]] = []
    for name, raw in header.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise RuntimeError(f"invalid tensor entry in {path}")
        dtype_name = raw.get("dtype")
        shape_value = raw.get("shape")
        offsets = raw.get("data_offsets")
        if (
            not isinstance(dtype_name, str)
            or dtype_name not in _DTYPE_ATTRIBUTES
            or not isinstance(shape_value, list)
            or not all(_is_int(value) for value in shape_value)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_is_int(value) for value in offsets)
        ):
            raise RuntimeError(f"invalid safetensors metadata for {name!r} in {path}")
        shape = tuple(shape_value)
        start, end = offsets
        if start < 0 or end < start or data_offset + end > file_bytes:
            raise RuntimeError(f"invalid safetensors range for {name!r} in {path}")
        occupied.append((start, end, name))
        selected_path = indexed_tensor_files.get(name) if indexed_tensor_files is not None else None
        if indexed_tensor_files is not None and selected_path != absolute_path:
            continue
        if not _matches_prefix(name, weight_name_prefixes):
            continue
        result.append(
            SafetensorDescriptor(
                name=name,
                dtype_name=dtype_name,
                shape=shape,
                file_offset=data_offset + start,
                length=end - start,
            )
        )

    previous_end = 0
    for start, end, name in sorted(occupied):
        if start < previous_end:
            raise RuntimeError(f"overlapping safetensors range for {name!r} in {path}")
        previous_end = end
    if indexed_tensor_files is None:
        return sorted(result, key=lambda item: item.file_offset)

    # The safetensors header/physical layout is commonly ordered by tensor
    # size or name.  That order interleaves vLLM parent modules and forces the
    # layerwise reload path to retain several modules' source tensors at once.
    # Hugging Face's index retains the state-dict insertion order used by the
    # model publisher, which groups a module's checkpoint inputs together.
    # Preserve it here.  Besides bounding reload memory, this lets vLLM finish
    # and copy one module back into its captured stable storage before moving
    # to the next module.
    index_order = {
        name: position
        for position, (name, selected_path) in enumerate(indexed_tensor_files.items())
        if selected_path == absolute_path
    }
    return sorted(result, key=lambda item: index_order[item.name])


def _index_map(folder: str) -> dict[str, str] | None:
    path = Path(folder) / "model.safetensors.index.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    weight_map = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(filename, str) for name, filename in weight_map.items()
    ):
        raise RuntimeError(f"invalid safetensors weight map in {path}")
    return {
        name: os.path.abspath(os.path.join(folder, filename))
        for name, filename in weight_map.items()
    }


def _dtype(torch_module: Any, dtype_name: str) -> Any:
    attribute = _DTYPE_ATTRIBUTES[dtype_name]
    value = getattr(torch_module, attribute, None)
    if value is None:
        raise RuntimeError(f"installed torch lacks safetensors dtype {dtype_name} ({attribute})")
    return value


def _validate_descriptor_size(torch_module: Any, descriptor: SafetensorDescriptor) -> Any:
    dtype = _dtype(torch_module, descriptor.dtype_name)
    probe = torch_module.empty((), dtype=dtype, device="cpu")
    expected = _checked_elements(descriptor.shape, descriptor.name) * int(probe.element_size())
    if expected != descriptor.length:
        raise RuntimeError(
            f"safetensors byte length mismatch for {descriptor.name!r}: "
            f"header={descriptor.length} shape={expected}"
        )
    return dtype


def _distributed_context(torch_module: Any) -> tuple[Any | None, int, int]:
    enabled = os.environ.get(LOADER_DISTRIBUTED_ENV, "1")
    if enabled not in {"0", "1"}:
        raise ValueError(f"{LOADER_DISTRIBUTED_ENV} must be 0 or 1")
    if enabled == "0":
        return None, 0, 1
    try:
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
    except (AssertionError, ImportError):
        return None, 0, 1
    if int(tp.world_size) <= 1:
        return None, 0, 1
    group = tp.device_group
    distributed = torch_module.distributed
    return group, int(distributed.get_rank(group)), int(distributed.get_world_size(group))


def _distributed_status_group() -> Any | None:
    """Return an optional CPU TP group for replay error fencing.

    The CUDA TP group is the default because ColdSnap explicitly restores its
    NCCL transport.  A vLLM CPU/Gloo group can retain a TCPStore socket whose
    peer connection was reset across CRIU restore, even while NCCL is healthy.
    """
    mode = os.environ.get(LOADER_STATUS_GROUP_ENV, "device")
    if mode not in {"device", "cpu"}:
        raise ValueError(f"{LOADER_STATUS_GROUP_ENV} must be device or cpu")
    if mode == "device":
        return None
    try:
        from vllm.distributed.parallel_state import get_tp_group

        return getattr(get_tp_group(), "cpu_group", None)
    except (AssertionError, ImportError):
        return None


def _prime_recovery_device_group(torch_module: Any) -> float:
    """Materialize the TP NCCL communicator while the capture store is live.

    Fast capture loaders do not necessarily issue a collective through
    vLLM's torch ``device_group``. Recovery replay does, and a communicator
    first created after CRIU restore would try to obtain its NCCL unique ID
    through a non-portable captured TCPStore connection. One checked scalar
    reduction at capture makes the communicator part of the NCCL checkpoint
    boundary, matching captures whose initial ColdSnap load already exercised
    the distributed replay path.
    """
    group, rank, world_size = _distributed_context(torch_module)
    if group is None or world_size <= 1:
        return 0.0
    device = torch_module.cuda.current_device()
    probe = torch_module.tensor(
        [rank + 1],
        dtype=torch_module.int32,
        device=device,
    )
    started = time.perf_counter()
    torch_module.distributed.all_reduce(probe, group=group)
    torch_module.cuda.synchronize(device)
    expected = world_size * (world_size + 1) // 2
    if int(probe.item()) != expected:
        raise RuntimeError("recovery TP device-group prime returned an invalid reduction")
    return time.perf_counter() - started


def _global_source_rank(distributed: Any, group: Any, group_rank: int) -> int:
    resolver = getattr(distributed, "get_global_rank", None)
    if callable(resolver):
        return int(resolver(group, group_rank))
    # Older torch releases without get_global_rank are safe only when the TP
    # group is also the complete world group, as in the qualified TP2 profile.
    if int(distributed.get_world_size()) != int(distributed.get_world_size(group)):
        raise RuntimeError("torch cannot resolve a subgroup rank for recovery loading")
    return group_rank


def _owners(
    descriptors: Sequence[SafetensorDescriptor],
    world_size: int,
    *,
    rotation: int = 0,
    physical_order: bool = False,
) -> list[int]:
    if world_size <= 0:
        raise ValueError("recovery loader world size must be positive")
    if not descriptors:
        return []
    order = (
        sorted(
            range(len(descriptors)),
            key=lambda index: descriptors[index].file_offset,
        )
        if physical_order
        else list(range(len(descriptors)))
    )
    ordered = [descriptors[index] for index in order]
    total = sum(descriptor.length for descriptor in ordered)
    prefix = [0]
    for descriptor in ordered:
        prefix.append(prefix[-1] + descriptor.length)
    boundaries = [0]
    lower = 0
    for partition in range(1, world_size):
        target = total * partition / world_size
        boundary = min(
            range(lower, len(prefix)),
            key=lambda index: (abs(prefix[index] - target), index),
        )
        boundaries.append(boundary)
        lower = boundary
    boundaries.append(len(descriptors))
    ordered_result = [0] * len(ordered)
    for partition, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
        owner = (partition + rotation) % world_size
        ordered_result[start:end] = [owner] * (end - start)
    result = [0] * len(descriptors)
    for ordered_index, descriptor_index in enumerate(order):
        result[descriptor_index] = ordered_result[ordered_index]
    return result


def _weighted_owners(
    descriptors: Sequence[SafetensorDescriptor],
    rank_weights: Sequence[float],
    *,
    rotation: int = 0,
    physical_order: bool = False,
) -> list[int]:
    """Partition physical bytes in proportion to measured rank throughput."""
    world_size = len(rank_weights)
    if world_size <= 0 or any(weight <= 0 for weight in rank_weights):
        raise ValueError("recovery rank I/O weights must be positive")
    if not descriptors:
        return []
    order = (
        sorted(
            range(len(descriptors)),
            key=lambda index: descriptors[index].file_offset,
        )
        if physical_order
        else list(range(len(descriptors)))
    )
    ordered = [descriptors[index] for index in order]
    total = sum(descriptor.length for descriptor in ordered)
    prefix = [0]
    for descriptor in ordered:
        prefix.append(prefix[-1] + descriptor.length)

    rank_order = [(index + rotation) % world_size for index in range(world_size)]
    total_weight = sum(rank_weights)
    boundaries = [0]
    cumulative_weight = 0.0
    lower = 0
    for rank in rank_order[:-1]:
        cumulative_weight += rank_weights[rank]
        target = total * cumulative_weight / total_weight
        boundary = min(
            range(lower, len(prefix)),
            key=lambda index: (abs(prefix[index] - target), index),
        )
        boundaries.append(boundary)
        lower = boundary
    boundaries.append(len(ordered))

    ordered_result = [0] * len(ordered)
    for rank, (start, end) in zip(
        rank_order,
        zip(boundaries, boundaries[1:], strict=False),
        strict=True,
    ):
        ordered_result[start:end] = [rank] * (end - start)
    result = [0] * len(descriptors)
    for ordered_index, descriptor_index in enumerate(order):
        result[descriptor_index] = ordered_result[ordered_index]
    return result


def _capacity_aware_additions(
    rank_weights: Sequence[float],
    assigned_bytes: Sequence[int],
    additional_bytes: int,
) -> tuple[float, ...]:
    """Allocate new shared reads to equalize predicted rank completion time."""
    if (
        not rank_weights
        or len(rank_weights) != len(assigned_bytes)
        or any(weight <= 0 for weight in rank_weights)
        or any(value < 0 for value in assigned_bytes)
        or additional_bytes < 0
    ):
        raise ValueError("recovery capacity inputs are invalid")
    if additional_bytes == 0:
        return tuple(0.0 for _weight in rank_weights)
    lower = min(
        assigned / weight for assigned, weight in zip(assigned_bytes, rank_weights, strict=True)
    )
    upper = max(
        (sum(assigned_bytes) + additional_bytes) / min(rank_weights),
        max(
            assigned / weight for assigned, weight in zip(assigned_bytes, rank_weights, strict=True)
        ),
    )
    for _iteration in range(80):
        middle = (lower + upper) / 2.0
        capacity = sum(
            max(0.0, middle * weight - assigned)
            for assigned, weight in zip(assigned_bytes, rank_weights, strict=True)
        )
        if capacity < additional_bytes:
            lower = middle
        else:
            upper = middle
    additions = [
        max(0.0, upper * weight - assigned)
        for assigned, weight in zip(assigned_bytes, rank_weights, strict=True)
    ]
    scale = additional_bytes / sum(additions)
    return tuple(value * scale for value in additions)


def _targeted_owners(
    descriptors: Sequence[SafetensorDescriptor],
    targets: Sequence[float],
    *,
    rotation: int = 0,
) -> list[int]:
    """Partition physical descriptors contiguously against per-rank byte targets."""
    if not targets or any(value < 0 for value in targets) or not any(targets):
        raise ValueError("recovery owner targets must include positive capacity")
    if not descriptors:
        return []
    world_size = len(targets)
    active = [rank for rank in range(world_size) if targets[rank] > 0]
    active.sort(key=lambda rank: (rank - rotation) % world_size)
    total = sum(descriptor.length for descriptor in descriptors)
    target_total = sum(targets[rank] for rank in active)
    prefix = [0]
    for descriptor in descriptors:
        prefix.append(prefix[-1] + descriptor.length)
    boundaries = [0]
    cumulative = 0.0
    lower = 0
    for rank in active[:-1]:
        cumulative += targets[rank]
        target = total * cumulative / target_total
        boundary = min(
            range(lower, len(prefix)),
            key=lambda index: (abs(prefix[index] - target), index),
        )
        boundaries.append(boundary)
        lower = boundary
    boundaries.append(len(descriptors))
    result = [active[-1]] * len(descriptors)
    for rank, (start, end) in zip(
        active,
        zip(boundaries, boundaries[1:], strict=False),
        strict=True,
    ):
        result[start:end] = [rank] * (end - start)
    return result


def _owner_layout(
    descriptors: Sequence[SafetensorDescriptor],
    owners: Sequence[int],
    world_size: int,
) -> tuple[list[int], list[int]]:
    if len(descriptors) != len(owners):
        raise ValueError("recovery owner layout length mismatch")
    sizes = [0] * world_size
    offsets: list[int] = []
    for descriptor, owner in zip(descriptors, owners, strict=True):
        if owner < 0 or owner >= world_size:
            raise ValueError("recovery tensor owner is outside the TP group")
        offsets.append(sizes[owner])
        sizes[owner] += descriptor.length
    return sizes, offsets


def _descriptor_batches(
    descriptors: Sequence[SafetensorDescriptor], max_bytes: int
) -> Iterator[tuple[int, int]]:
    """Partition descriptors without splitting a checkpoint tensor.

    A tensor larger than the target occupies one batch by itself. This keeps
    peak staging bounded by ``max(max_bytes, largest_tensor_bytes)`` while
    preserving the checkpoint tensor boundaries expected by vLLM.
    """
    if max_bytes <= 0:
        raise ValueError("recovery staging bytes must be positive")
    start = 0
    batch_bytes = 0
    for index, descriptor in enumerate(descriptors):
        if index > start and batch_bytes + descriptor.length > max_bytes:
            yield start, index
            start = index
            batch_bytes = 0
        batch_bytes += descriptor.length
    if start < len(descriptors):
        yield start, len(descriptors)


def _storage_views(
    storage: Any,
    descriptors: Sequence[SafetensorDescriptor],
    *,
    physical_order: bool = False,
) -> list[Any]:
    expected = sum(descriptor.length for descriptor in descriptors)
    if int(storage.numel()) != expected:
        raise ValueError(f"recovery staging slab has {storage.numel()} bytes; expected {expected}")
    order = (
        sorted(
            range(len(descriptors)),
            key=lambda index: descriptors[index].file_offset,
        )
        if physical_order
        else list(range(len(descriptors)))
    )
    offset = 0
    views: list[Any | None] = [None] * len(descriptors)
    for index in order:
        descriptor = descriptors[index]
        views[index] = storage[offset : offset + descriptor.length]
        offset += descriptor.length
    return list(views)


def _coalesced_extents(
    descriptors: Sequence[SafetensorDescriptor],
    tensors: Sequence[Any],
    owners: Sequence[int],
    rank: int,
    *,
    coalesce_adjacent: bool = False,
) -> list[HydrationExtent]:
    selected: list[tuple[SafetensorDescriptor, Any]] = []
    for descriptor, tensor, owner in zip(descriptors, tensors, owners, strict=True):
        if owner == rank and descriptor.length > 0:
            selected.append((descriptor, tensor))
    if coalesce_adjacent:
        selected.sort(key=lambda item: item[0].file_offset)

    extents: list[HydrationExtent] = []
    for descriptor, tensor in selected:
        destination = int(tensor.data_ptr())
        if (
            coalesce_adjacent
            and extents
            and extents[-1].file_offset + extents[-1].length == descriptor.file_offset
            and extents[-1].destination + extents[-1].length == destination
        ):
            previous = extents[-1]
            extents[-1] = HydrationExtent(
                file_offset=previous.file_offset,
                destination=previous.destination,
                length=previous.length + descriptor.length,
            )
        else:
            extents.append(
                HydrationExtent(
                    file_offset=descriptor.file_offset,
                    destination=destination,
                    length=descriptor.length,
                )
            )
    return extents


def _split_direct_extents(
    extents: Sequence[HydrationExtent], alignment: int = 4096
) -> tuple[list[HydrationExtent], list[HydrationExtent]]:
    """Split semantic ranges into an O_DIRECT interior and tiny buffered edges."""
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("recovery direct alignment must be a power of two")
    direct: list[HydrationExtent] = []
    buffered: list[HydrationExtent] = []
    for extent in extents:
        start = extent.file_offset
        end = start + extent.length
        aligned_start = (start + alignment - 1) & -alignment
        aligned_end = end & -alignment
        if aligned_start >= aligned_end:
            buffered.append(extent)
            continue
        if start < aligned_start:
            buffered.append(
                HydrationExtent(
                    file_offset=start,
                    destination=extent.destination,
                    length=aligned_start - start,
                )
            )
        direct.append(
            HydrationExtent(
                file_offset=aligned_start,
                destination=extent.destination + aligned_start - start,
                length=aligned_end - aligned_start,
            )
        )
        if aligned_end < end:
            buffered.append(
                HydrationExtent(
                    file_offset=aligned_end,
                    destination=extent.destination + aligned_end - start,
                    length=end - aligned_end,
                )
            )
    return direct, buffered


def _broadcast_slab_regions(
    torch_module: Any,
    group: Any,
    staging_slab: Any,
    storages: Sequence[Any],
    owners: Sequence[int],
    max_chunk_bytes: int,
) -> int:
    """Broadcast physically contiguous owner ranges without packing copies."""
    if len(storages) != len(owners):
        raise ValueError("recovery broadcast layout length mismatch")
    if max_chunk_bytes <= 0:
        raise ValueError("recovery broadcast chunk size must be positive")
    base = int(staging_slab.data_ptr())
    limit = base + int(staging_slab.numel())
    rows = sorted(
        (
            int(storage.data_ptr()),
            int(storage.numel()),
            owner,
        )
        for storage, owner in zip(storages, owners, strict=True)
        if int(storage.numel()) > 0
    )
    regions: list[tuple[int, int, int]] = []
    for pointer, length, owner in rows:
        if pointer < base or pointer + length > limit:
            raise ValueError("recovery tensor storage lies outside staging slab")
        if regions and regions[-1][2] == owner and regions[-1][0] + regions[-1][1] == pointer:
            start, previous_length, _ = regions[-1]
            regions[-1] = (start, previous_length + length, owner)
        else:
            regions.append((pointer, length, owner))

    distributed = torch_module.distributed
    calls = 0
    for pointer, length, owner in regions:
        source_rank = _global_source_rank(distributed, group, owner)
        offset = pointer - base
        region_end = offset + length
        while offset < region_end:
            chunk_end = min(offset + max_chunk_bytes, region_end)
            distributed.broadcast(
                staging_slab[offset:chunk_end],
                src=source_rank,
                group=group,
            )
            calls += 1
            offset = chunk_end
    return calls


def _broadcast_tensors(
    torch_module: Any,
    group: Any,
    storages: Sequence[Any],
    owners: Sequence[int],
    world_size: int,
    buffer_bytes: int,
) -> int:
    """Broadcast independent tensor storages with bounded temporary buffers.

    vLLM's checkpoint reload can retain a yielded tensor until its parent layer
    is complete. Each yielded tensor therefore owns its storage; otherwise one
    retained view pins a whole multi-gigabyte safetensors-file slab. PyTorch's
    coalesced collective preserves those independent lifetimes while amortizing
    NCCL launch overhead through a temporary buffer that is released before the
    tensors are yielded to vLLM.
    """
    if len(storages) != len(owners):
        raise ValueError("recovery broadcast layout length mismatch")
    distributed = torch_module.distributed
    broadcast_coalesced = getattr(distributed, "_broadcast_coalesced", None)
    calls = 0
    for owner in range(world_size):
        selected = [
            storage
            for storage, selected_owner in zip(storages, owners, strict=True)
            if selected_owner == owner and storage.numel() > 0
        ]
        if not selected:
            continue
        source_rank = _global_source_rank(distributed, group, owner)
        if callable(broadcast_coalesced):
            broadcast_coalesced(group, selected, buffer_bytes, source_rank)
            calls += 1
        else:
            for storage in selected:
                distributed.broadcast(storage, src=source_rank, group=group)
                calls += 1
    return calls


def _verify_tensor_samples(
    torch_module: Any,
    path: Path,
    descriptors: Sequence[SafetensorDescriptor],
    storages: Sequence[Any],
    sample_bytes: int,
) -> int:
    """Compare bounded head/tail samples after distributed hydration."""
    if sample_bytes <= 0:
        return 0
    if len(descriptors) != len(storages):
        raise ValueError("recovery verification layout length mismatch")

    parts: list[Any] = []
    segments: list[tuple[str, str, int]] = []
    expected = bytearray()
    descriptor = os.open(path, os.O_RDONLY)
    try:
        for tensor, storage in zip(descriptors, storages, strict=True):
            head = min(sample_bytes, tensor.length)
            if head:
                parts.append(storage[:head])
                segments.append((tensor.name, "head", head))
                expected.extend(os.pread(descriptor, head, tensor.file_offset))
            tail = min(sample_bytes, tensor.length - head)
            if tail:
                parts.append(storage[tensor.length - tail :])
                segments.append((tensor.name, "tail", tail))
                expected.extend(
                    os.pread(
                        descriptor,
                        tail,
                        tensor.file_offset + tensor.length - tail,
                    )
                )
    finally:
        os.close(descriptor)

    if not parts:
        return 0
    actual = bytes(torch_module.cat(parts).cpu().tolist())
    if actual == expected:
        return len(expected)

    mismatch = next(
        index
        for index, (actual_byte, expected_byte) in enumerate(zip(actual, expected, strict=True))
        if actual_byte != expected_byte
    )
    offset = 0
    for name, location, length in segments:
        if mismatch < offset + length:
            raise RuntimeError(
                "recovery tensor sample mismatch for "
                f"{name!r} ({location}+{mismatch - offset}) in {path}"
            )
        offset += length
    raise AssertionError("recovery tensor sample mismatch was not localized")


def _native_hydrator() -> tuple[NativeHydrator | None, str]:
    requested = os.environ.get(LOADER_BACKEND_ENV, DEFAULT_LOADER_BACKEND)
    if requested not in {"auto", "buffered", "direct", "mmap", "torch"}:
        raise ValueError(f"{LOADER_BACKEND_ENV} must be auto, buffered, direct, mmap, or torch")
    if requested == "mmap":
        return None, "mmap"
    library = os.environ.get("COLDSNAP_HYDRATION_NATIVE_LIBRARY")
    if requested == "torch" or (requested == "auto" and not library):
        return None, "torch"
    if not library:
        raise RuntimeError(f"{LOADER_BACKEND_ENV}={requested} requires the native hydrator")
    hydrator = NativeHydrator(library)
    expected = os.environ.get("COLDSNAP_HYDRATION_NATIVE_SHA256")
    if expected and hydrator.sha256 != expected:
        raise RuntimeError("recovery loader native hydration library SHA-256 mismatch")
    return hydrator, requested


_REMOTE_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "fuse.sshfs",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "smb2",
        "smb3",
    }
)
_LOCAL_DIRECT_FILESYSTEMS = frozenset({"btrfs", "ext2", "ext3", "ext4", "xfs"})


def _unescape_mountinfo(value: str) -> str:
    for escaped, decoded in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, decoded)
    return value


def _filesystem_for_open_file(path: Path) -> dict[str, str]:
    """Resolve filesystem identity from the opened file's device number."""

    # Hugging Face snapshots normally expose model shards as symlinks into the
    # repository's content-addressed ``blobs`` directory.  Resolve that chain
    # before applying O_NOFOLLOW to the final object: rejecting the snapshot
    # entry itself makes otherwise valid, pinned HF layouts unusable, while
    # opening the resolved leaf still prevents silently accepting a final
    # symlink substituted between resolution and open.
    resolved_path = path.resolve(strict=True)
    descriptor = os.open(
        resolved_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    device = f"{os.major(value.st_dev)}:{os.minor(value.st_dev)}"
    candidates: list[tuple[int, dict[str, str]]] = []
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    resolved = str(resolved_path)
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if len(fields) <= separator + 2 or fields[2] != device:
            continue
        mountpoint = _unescape_mountinfo(fields[4])
        if resolved != mountpoint and not resolved.startswith(mountpoint.rstrip("/") + "/"):
            continue
        candidates.append(
            (
                len(mountpoint),
                {
                    "filesystem_type": fields[separator + 1],
                    "mount_source": _unescape_mountinfo(fields[separator + 2]),
                    "mountpoint": mountpoint,
                    "device": device,
                },
            )
        )
    if not candidates:
        return {
            "filesystem_type": "unknown",
            "mount_source": "unknown",
            "mountpoint": "unknown",
            "device": device,
        }
    return max(candidates, key=lambda item: item[0])[1]


def _select_recovery_io(path: Path, requested: str) -> tuple[str, dict[str, Any]]:
    filesystem = _filesystem_for_open_file(path)
    fs_type = filesystem["filesystem_type"].lower()
    if requested != "auto":
        effective = requested
        reason = "explicit-policy"
    elif fs_type in _REMOTE_FILESYSTEMS or fs_type.startswith("fuse."):
        effective = "buffered"
        reason = "remote-filesystem-prefers-page-cache"
    elif fs_type in _LOCAL_DIRECT_FILESYSTEMS:
        effective = "direct"
        reason = "qualified-local-filesystem"
    else:
        effective = "buffered"
        reason = "unknown-filesystem-safe-default"
    observation: dict[str, Any] = {
        "path": str(path),
        "requested": requested,
        "effective": effective,
        "reason": reason,
        **filesystem,
    }
    with _io_policy_lock:
        _io_policy_observations[str(path)] = observation
    return effective, observation


def _native_hydrate_owned(
    hydrator: NativeHydrator,
    path: Path,
    extents: Sequence[HydrationExtent],
    *,
    backend: str,
    chunk_bytes: int,
    queue_depth: int,
    cuda_device: int,
) -> tuple[int, str]:
    """Hydrate coalesced ranges through the fastest compatible native path.

    Original safetensors payloads generally begin and end off a 4096-byte
    boundary.  Large aligned interiors still use O_DIRECT; only the sub-page
    edges use the native buffered pipeline.
    """
    selected_backend, observation = _select_recovery_io(path, backend)
    direct_available = selected_backend == "direct" and hydrator.available("direct")
    if selected_backend == "direct" and not direct_available:
        raise RuntimeError("recovery loader direct hydration is unavailable")
    if direct_available:
        direct, buffered = _split_direct_extents(extents)
    else:
        direct, buffered = [], list(extents)

    if direct:
        try:
            hydrator.hydrate(
                path,
                direct,
                backend="direct",
                chunk_bytes=chunk_bytes,
                queue_depth=queue_depth,
                cuda_device=cuda_device,
                preverified=False,
                register_device_buffers=False,
            )
        except (OSError, RuntimeError) as error:
            if backend != "auto":
                raise
            observation["effective"] = "buffered"
            observation["reason"] = "direct-open-fallback"
            observation["direct_fallback_errno"] = int(getattr(error, "errno", 0) or 0)
            observation["direct_fallback_error"] = type(error).__name__
            with _io_policy_lock:
                _io_policy_observations[str(path)] = dict(observation)
            direct, buffered = [], list(extents)
    if buffered:
        hydrator.hydrate(
            path,
            buffered,
            backend="buffered",
            chunk_bytes=chunk_bytes,
            queue_depth=queue_depth,
            cuda_device=cuda_device,
            preverified=False,
            register_device_buffers=False,
        )
    if direct and buffered:
        effective = "direct+buffered-edges"
    elif direct:
        effective = "direct"
    else:
        effective = "buffered"
    return len(direct) + len(buffered), effective


def _timed_native_hydrate_owned(
    hydrator: NativeHydrator,
    path: Path,
    extents: Sequence[HydrationExtent],
    *,
    backend: str,
    chunk_bytes: int,
    queue_depth: int,
    cuda_device: int,
) -> tuple[int, str, float]:
    """Hydrate one transport batch and retain service time across threads."""
    started = time.perf_counter()
    extent_count, effective = _native_hydrate_owned(
        hydrator,
        path,
        extents,
        backend=backend,
        chunk_bytes=chunk_bytes,
        queue_depth=queue_depth,
        cuda_device=cuda_device,
    )
    return extent_count, effective, time.perf_counter() - started


def _probe_recovery_rank_io(
    torch_module: Any,
    hydrator: Any,
    backend: str,
    group: Any,
    status_group: Any,
    device: Any,
    sources: dict[tuple[Any, ...], SafetensorDescriptor],
    *,
    chunk_bytes: int,
    queue_depth: int,
    probe_bytes: int,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Measure each rank's local checkpoint throughput for weighted ownership."""
    if group is None:
        return (1.0,), {
            "bytes": 0,
            "local_seconds": 0.0,
            "collective_seconds": 0.0,
        }
    by_path: dict[str, list[SafetensorDescriptor]] = {}
    for key, descriptor in sources.items():
        by_path.setdefault(str(key[1]), []).append(descriptor)
    path_text, descriptors = min(by_path.items(), key=lambda item: item[0])
    descriptors.sort(key=lambda item: item.file_offset)

    available = sum(descriptor.length for descriptor in descriptors)
    selected_bytes = min(probe_bytes, available)
    if selected_bytes <= 0:
        raise RuntimeError("recovery I/O probe found no checkpoint bytes")
    probe_slab = torch_module.empty(
        selected_bytes,
        dtype=torch_module.uint8,
        device=device,
    )
    base = int(probe_slab.data_ptr())
    extents: list[HydrationExtent] = []
    remaining = selected_bytes
    destination_offset = 0
    for descriptor in descriptors:
        if remaining <= 0:
            break
        length = min(descriptor.length, remaining)
        extent = HydrationExtent(
            file_offset=descriptor.file_offset,
            destination=base + destination_offset,
            length=length,
        )
        if (
            extents
            and extents[-1].file_offset + extents[-1].length == extent.file_offset
            and extents[-1].destination + extents[-1].length == extent.destination
        ):
            previous = extents[-1]
            extents[-1] = HydrationExtent(
                file_offset=previous.file_offset,
                destination=previous.destination,
                length=previous.length + extent.length,
            )
        else:
            extents.append(extent)
        destination_offset += length
        remaining -= length

    local_error: BaseException | None = None
    phase = time.perf_counter()
    try:
        _native_hydrate_owned(
            hydrator,
            Path(path_text),
            extents,
            backend=backend,
            chunk_bytes=chunk_bytes,
            queue_depth=queue_depth,
            cuda_device=device,
        )
        torch_module.cuda.synchronize(device)
    except BaseException as error:
        local_error = error
    local_seconds = time.perf_counter() - phase

    phase = time.perf_counter()
    status_device = "cpu" if status_group is not None else device
    status = torch_module.tensor(
        [1 if local_error is not None else 0],
        dtype=torch_module.int32,
        device=status_device,
    )
    control_group = status_group if status_group is not None else group
    torch_module.distributed.all_reduce(status, group=control_group)
    if int(status.item()) != 0:
        if local_error is not None:
            raise local_error
        raise RuntimeError("a peer failed during the recovery I/O probe")

    local_rate = selected_bytes / max(local_seconds, 1e-6)
    world_size = int(torch_module.distributed.get_world_size(group))
    gathered: list[Any] = [None] * world_size
    torch_module.distributed.all_gather_object(
        gathered,
        local_rate,
        group=control_group,
    )
    collective_seconds = time.perf_counter() - phase
    try:
        weights = _damped_rank_io_weights(gathered, _io_weight_exponent())
    except ValueError as error:
        raise RuntimeError("recovery I/O probe peers published invalid throughput") from error
    rates = [float(value) for value in gathered]
    del probe_slab
    return weights, {
        "bytes": selected_bytes,
        "local_seconds": local_seconds,
        "collective_seconds": collective_seconds,
        "rank_bytes_per_second": rates,
        "rank_weights": list(weights),
        "source": "restore-probe",
    }


def _torch_load_owned(
    source: Any,
    descriptors: Sequence[SafetensorDescriptor],
    tensors: Sequence[Any],
    owners: Sequence[int],
    rank: int,
) -> None:
    selected = {
        descriptor.name: tensor
        for descriptor, tensor, owner in zip(descriptors, tensors, owners, strict=True)
        if owner == rank
    }
    if not selected:
        return
    for name, destination in selected.items():
        destination.copy_(source.get_tensor(name), non_blocking=False)


def _dtype_from_string(torch_module: Any, value: str) -> Any:
    prefix = "torch."
    if not value.startswith(prefix):
        raise RuntimeError(f"invalid recovery replay dtype {value!r}")
    dtype = getattr(torch_module, value[len(prefix) :], None)
    if dtype is None:
        raise RuntimeError(f"installed torch lacks recovery replay dtype {value!r}")
    return dtype


def _typed_storage_view(
    torch_module: Any,
    tensor: Any,
    *,
    relative_byte_offset: int,
    dtype_name: str,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> Any:
    """Rebuild a captured typed view relative to a live tensor's first byte."""
    dtype = _dtype_from_string(torch_module, dtype_name)
    element_size = int(torch_module.empty((), dtype=dtype).element_size())
    if len(shape) != len(stride) or any(dimension < 0 for dimension in shape):
        raise RuntimeError("recovery replay view has an invalid shape")
    if any(step < 0 for step in stride):
        raise RuntimeError("recovery replay view has a negative stride")
    if any(dimension == 0 for dimension in shape):
        span_bytes = 0
    else:
        last_element = sum(
            (dimension - 1) * step for dimension, step in zip(shape, stride, strict=True)
        )
        span_bytes = (last_element + 1) * element_size
    tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
    relative_byte_offset = int(relative_byte_offset)
    if relative_byte_offset < 0 or relative_byte_offset + span_bytes > tensor_bytes:
        raise RuntimeError("recovery replay view exceeds its live tensor")
    storage = tensor.untyped_storage()
    tensor_byte_offset = int(tensor.data_ptr()) - int(storage.data_ptr())
    byte_offset = tensor_byte_offset + relative_byte_offset
    if byte_offset < 0 or byte_offset % element_size:
        raise RuntimeError("recovery replay view has an invalid storage offset")
    if byte_offset + span_bytes > int(storage.nbytes()):
        raise RuntimeError("recovery replay view exceeds its backing storage")
    result = torch_module.empty(
        0,
        dtype=dtype,
        device=tensor.device,
        requires_grad=False,
    )
    result.set_(storage, byte_offset // element_size, shape, stride)
    return result


def _copy_order(copy: RecoveryCopy) -> tuple[Any, ...]:
    return (
        copy.source_path,
        copy.source_file_offset,
        copy.source_name,
        copy.destination_name,
        copy.destination_view_offset_bytes,
        copy.source_view_offset_bytes,
    )


def _replay_transport_owners(
    rows: Sequence[tuple[tuple[Any, ...], SafetensorDescriptor]],
    rank_copies: Sequence[dict[tuple[Any, ...], list[RecoveryCopy]]],
    world_size: int,
    *,
    rotation: int,
    rank_io_weights: Sequence[float] | None = None,
    rank_assigned_bytes: Sequence[int] | None = None,
) -> list[tuple[int, tuple[int, ...]]]:
    """Assign replay reads to ranks which actually consume each source.

    A source used by one rank is read by that rank and never enters a
    collective. Shared sources retain deterministic balanced ownership and are
    broadcast once. This preserves the generic TP transport contract while
    avoiding all-gather-like traffic for expert-parallel or otherwise
    rank-exclusive checkpoint tensors.
    """
    descriptors = [descriptor for _key, descriptor in rows]
    balanced = (
        _weighted_owners(
            descriptors,
            rank_io_weights,
            rotation=rotation,
            physical_order=True,
        )
        if rank_io_weights is not None
        else _owners(
            descriptors,
            world_size,
            rotation=rotation,
            physical_order=True,
        )
    )
    consumers_by_row = [
        tuple(rank for rank, copies in enumerate(rank_copies) if key in copies)
        for key, _descriptor in rows
    ]
    if rank_assigned_bytes is not None:
        if len(rank_assigned_bytes) != world_size or any(
            value < 0 for value in rank_assigned_bytes
        ):
            raise ValueError("recovery assigned rank bytes are invalid")
        shared_indices = [
            index
            for index, consumers in enumerate(consumers_by_row)
            if consumers == tuple(range(world_size))
        ]
        if shared_indices:
            weights = rank_io_weights or (1.0,) * world_size
            shared_descriptors = [descriptors[index] for index in shared_indices]
            targets = _capacity_aware_additions(
                weights,
                rank_assigned_bytes,
                sum(descriptor.length for descriptor in shared_descriptors),
            )
            shared_owners = _targeted_owners(
                shared_descriptors,
                targets,
                rotation=rotation,
            )
            for index, owner in zip(shared_indices, shared_owners, strict=True):
                balanced[index] = owner

    assignments: list[tuple[int, tuple[int, ...]]] = []
    for consumers, balanced_owner in zip(consumers_by_row, balanced, strict=True):
        if not consumers:
            raise RuntimeError("recovery replay source has no consuming TP rank")
        if len(consumers) == 1:
            owner = consumers[0]
        elif consumers == tuple(range(world_size)):
            owner = balanced_owner
        else:
            owner = consumers[balanced_owner % len(consumers)]
        assignments.append((owner, consumers))
    return assignments


def _replay_view_span_bytes(torch_module: Any, copy: RecoveryCopy) -> int:
    dtype = _dtype_from_string(torch_module, copy.source_view_dtype)
    element_size = int(torch_module.empty((), dtype=dtype).element_size())
    shape = copy.source_view_shape
    stride = copy.source_view_stride
    if len(shape) != len(stride) or any(dimension < 0 for dimension in shape):
        raise RuntimeError("recovery replay source view has an invalid shape")
    if any(step < 0 for step in stride):
        raise RuntimeError("recovery replay source view has a negative stride")
    if any(dimension == 0 for dimension in shape):
        return 0
    last_element = sum(
        (dimension - 1) * step for dimension, step in zip(shape, stride, strict=True)
    )
    return (last_element + 1) * element_size


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = [0] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        running *= shape[index]
    return tuple(stride)


def _replay_pack_layout(
    torch_module: Any,
    copies: Sequence[RecoveryCopy],
) -> tuple[list[int], int]:
    """Lay out logical source views contiguously with dtype alignment."""
    offsets: list[int] = []
    end = 0
    for copy in copies:
        dtype = _dtype_from_string(torch_module, copy.source_view_dtype)
        element_size = int(torch_module.empty((), dtype=dtype).element_size())
        end = ((end + element_size - 1) // element_size) * element_size
        offsets.append(end)
        elements = 1
        for dimension in copy.source_view_shape:
            elements *= dimension
        end += elements * element_size
    return offsets, end


def _packed_copy_extent_geometry(
    torch_module: Any,
    copy: RecoveryCopy,
) -> tuple[int, int, int]:
    """Return (extent count, elements per extent, element size)."""
    dtype = _dtype_from_string(torch_module, copy.source_view_dtype)
    element_size = int(torch_module.empty((), dtype=dtype).element_size())
    shape = copy.source_view_shape
    stride = copy.source_view_stride
    if len(shape) != len(stride) or any(dimension <= 0 for dimension in shape):
        raise RuntimeError("recovery packed source view has an invalid shape")
    if any(step < 0 for step in stride):
        raise RuntimeError("recovery packed source view has a negative stride")

    suffix_start = len(shape)
    contiguous_elements = 1
    for index in range(len(shape) - 1, -1, -1):
        dimension = shape[index]
        if dimension == 1:
            suffix_start = index
            continue
        if stride[index] != contiguous_elements:
            break
        suffix_start = index
        contiguous_elements *= dimension

    extent_count = 1
    for dimension in shape[:suffix_start]:
        extent_count *= dimension
    return extent_count, contiguous_elements, element_size


def _packed_copy_extents(
    torch_module: Any,
    copy: RecoveryCopy,
    packed_offset_bytes: int,
) -> list[_ReplayPackedExtent]:
    """Decompose one logical strided view into contiguous file extents."""
    extent_count, contiguous_elements, element_size = _packed_copy_extent_geometry(
        torch_module,
        copy,
    )
    shape = copy.source_view_shape
    stride = copy.source_view_stride
    suffix_start = len(shape)
    expected = 1
    for index in range(len(shape) - 1, -1, -1):
        dimension = shape[index]
        if dimension == 1:
            suffix_start = index
            continue
        if stride[index] != expected:
            break
        suffix_start = index
        expected *= dimension

    prefix_shape = shape[:suffix_start]
    prefix_stride = stride[:suffix_start]
    extent_bytes = contiguous_elements * element_size
    extents: list[_ReplayPackedExtent] = []
    for linear_index in range(extent_count):
        remaining = linear_index
        source_elements = 0
        for dimension, step in zip(
            reversed(prefix_shape),
            reversed(prefix_stride),
            strict=True,
        ):
            coordinate = remaining % dimension
            remaining //= dimension
            source_elements += coordinate * step
        item = _ReplayPackedExtent(
            file_offset=(
                copy.source_file_offset
                + copy.source_view_offset_bytes
                + source_elements * element_size
            ),
            packed_offset_bytes=packed_offset_bytes + linear_index * extent_bytes,
            length=extent_bytes,
        )
        if (
            extents
            and extents[-1].file_offset + extents[-1].length == item.file_offset
            and extents[-1].packed_offset_bytes + extents[-1].length == item.packed_offset_bytes
        ):
            previous = extents[-1]
            extents[-1] = _ReplayPackedExtent(
                file_offset=previous.file_offset,
                packed_offset_bytes=previous.packed_offset_bytes,
                length=previous.length + item.length,
            )
        else:
            extents.append(item)

    packed_bytes = extent_count * extent_bytes
    expected_bytes = 1
    for dimension in shape:
        expected_bytes *= dimension
    expected_bytes *= element_size
    if packed_bytes != expected_bytes:
        raise RuntimeError("recovery packed source geometry changed")
    return extents


def _rank_local_packed_replay_qualified(
    torch_module: Any,
    copies: Sequence[RecoveryCopy],
    *,
    max_extents: int,
    min_extent_bytes: int,
) -> bool:
    """Bound scatter-read fragmentation before selecting local packed I/O."""
    extent_count = 0
    packed_bytes = 0
    for copy in copies:
        count, contiguous_elements, element_size = _packed_copy_extent_geometry(
            torch_module,
            copy,
        )
        extent_count += count
        packed_bytes += count * contiguous_elements * element_size
        if extent_count > max_extents:
            return False
    return bool(extent_count and packed_bytes // extent_count >= min_extent_bytes)


def _rank_local_packed_replay_plans_qualified(
    torch_module: Any,
    plans: Sequence[RecoveryCopyPlan],
    *,
    max_extents: int,
    min_extent_bytes: int,
) -> bool:
    """Select local scatter reads only when aggregate I/O is non-duplicative."""
    aggregate_bytes: Counter[tuple[Any, ...]] = Counter()
    source_lengths: dict[tuple[Any, ...], int] = {}
    for plan in plans:
        if not _rank_local_packed_replay_qualified(
            torch_module,
            plan.copies,
            max_extents=max_extents,
            min_extent_bytes=min_extent_bytes,
        ):
            return False
        for copy in plan.copies:
            count, contiguous_elements, element_size = _packed_copy_extent_geometry(
                torch_module,
                copy,
            )
            aggregate_bytes[copy.source_key] += count * contiguous_elements * element_size
            source_lengths[copy.source_key] = copy.source_length
    return all(
        logical_bytes <= source_lengths[key] for key, logical_bytes in aggregate_bytes.items()
    )


def _pack_recovery_copies(
    torch_module: Any,
    copies: Sequence[RecoveryCopy],
    source_tensors: dict[tuple[Any, ...], Any],
    packed_slab: Any,
) -> int:
    offsets, packed_bytes = _replay_pack_layout(torch_module, copies)
    if int(packed_slab.numel() * packed_slab.element_size()) < packed_bytes:
        raise RuntimeError("recovery replay peer-pack slab is too small")
    for copy, offset in zip(copies, offsets, strict=True):
        source_view = _typed_storage_view(
            torch_module,
            source_tensors[copy.source_key],
            relative_byte_offset=copy.source_view_offset_bytes,
            dtype_name=copy.source_view_dtype,
            shape=copy.source_view_shape,
            stride=copy.source_view_stride,
        )
        packed_view = _typed_storage_view(
            torch_module,
            packed_slab,
            relative_byte_offset=offset,
            dtype_name=copy.source_view_dtype,
            shape=copy.source_view_shape,
            stride=_contiguous_stride(copy.source_view_shape),
        )
        packed_view.copy_(source_view)
    return packed_bytes


def _apply_packed_recovery_copies(
    torch_module: Any,
    copies: Sequence[RecoveryCopy],
    packed_slab: Any,
    destinations: dict[str, Any],
) -> int:
    offsets, packed_bytes = _replay_pack_layout(torch_module, copies)
    if int(packed_slab.numel() * packed_slab.element_size()) < packed_bytes:
        raise RuntimeError("recovery replay peer receive slab is too small")
    copied_bytes = 0
    for copy, offset in zip(copies, offsets, strict=True):
        source_view = _typed_storage_view(
            torch_module,
            packed_slab,
            relative_byte_offset=offset,
            dtype_name=copy.source_view_dtype,
            shape=copy.source_view_shape,
            stride=_contiguous_stride(copy.source_view_shape),
        )
        destination = destinations[copy.destination_name]
        destination_view = _typed_storage_view(
            torch_module,
            destination,
            relative_byte_offset=copy.destination_view_offset_bytes,
            dtype_name=copy.destination_dtype,
            shape=copy.destination_view_shape,
            stride=copy.destination_view_stride,
        )
        if (
            tuple(source_view.shape) != tuple(destination_view.shape)
            or int(destination_view.numel() * destination_view.element_size()) != copy.copy_bytes
        ):
            raise RuntimeError("recovery packed replay copy metadata changed")
        destination_view.copy_(source_view)
        copied_bytes += copy.copy_bytes
    return copied_bytes


def _rank_local_replay_ranges(
    torch_module: Any,
    sources: dict[tuple[Any, ...], SafetensorDescriptor],
    rank_copies: Sequence[dict[tuple[Any, ...], list[RecoveryCopy]]],
    rank: int,
) -> list[_ReplaySourceRange] | None:
    """Plan local file reads when TP source views do not duplicate I/O.

    Reading each rank's observed source spans avoids transmitting checkpoint
    bytes that are already present in every rank's safetensors cache. If the
    aggregate rank spans exceed the serialized source (for example, a fully
    replicated destination), return ``None`` so the balanced read+broadcast
    path retains its lower aggregate I/O contract.
    """
    local: list[_ReplaySourceRange] = []
    for key, descriptor in sources.items():
        aggregate_bytes = 0
        rank_ranges: list[tuple[int, int] | None] = []
        for grouped in rank_copies:
            copies = grouped.get(key, ())
            if not copies:
                rank_ranges.append(None)
                continue
            start = min(copy.source_view_offset_bytes for copy in copies)
            end = max(
                copy.source_view_offset_bytes + _replay_view_span_bytes(torch_module, copy)
                for copy in copies
            )
            if start < 0 or end <= start or end > descriptor.length:
                raise RuntimeError(f"recovery replay source range changed for {descriptor.name!r}")
            aggregate_bytes += end - start
            rank_ranges.append((start, end))
        if aggregate_bytes > descriptor.length:
            return None
        selected = rank_ranges[rank]
        if selected is None:
            continue
        start, end = selected
        local.append(
            _ReplaySourceRange(
                source_key=key,
                descriptor=SafetensorDescriptor(
                    name=descriptor.name,
                    dtype_name="U8",
                    shape=(end - start,),
                    file_offset=descriptor.file_offset + start,
                    length=end - start,
                ),
                source_base_offset_bytes=start,
            )
        )
    local.sort(
        key=lambda item: (
            item.source_key[1],
            item.descriptor.file_offset,
            item.descriptor.name,
        )
    )
    return local


def _execute_rank_local_replay(
    torch_module: Any,
    hydrator: Any,
    backend: str,
    group: Any,
    status_group: Any,
    rank: int,
    device: Any,
    ranges: Sequence[_ReplaySourceRange],
    copies: Sequence[RecoveryCopy],
    destinations: dict[str, Any],
    *,
    source_bytes: int,
    chunk_bytes: int,
    queue_depth: int,
    staging_bytes: int,
) -> dict[str, Any]:
    """Hydrate rank-specific source views through one reusable CUDA slab."""
    by_path: dict[str, list[_ReplaySourceRange]] = {}
    for item in ranges:
        by_path.setdefault(str(item.source_key[1]), []).append(item)
    work: list[tuple[Path, list[_ReplaySourceRange]]] = []
    for path_text, rows in sorted(by_path.items()):
        rows.sort(key=lambda item: item.descriptor.file_offset)
        descriptors = [item.descriptor for item in rows]
        for batch_start, batch_end in _descriptor_batches(descriptors, staging_bytes):
            work.append((Path(path_text), rows[batch_start:batch_end]))
    if not work:
        raise RuntimeError("recovery rank-local replay has no source ranges")

    copies_by_source: dict[tuple[Any, ...], list[RecoveryCopy]] = {}
    for copy in copies:
        copies_by_source.setdefault(copy.source_key, []).append(copy)
    for grouped in copies_by_source.values():
        grouped.sort(key=_copy_order)

    started = time.perf_counter()
    setup_phase = time.perf_counter()
    largest_batch = max(sum(item.descriptor.length for item in rows) for _path, rows in work)
    staging_slab = torch_module.empty(
        largest_batch,
        dtype=torch_module.uint8,
        device=device,
    )
    torch_module.cuda.synchronize(device)
    staging_seconds = time.perf_counter() - setup_phase
    io_seconds = 0.0
    copy_seconds = 0.0
    local_read_bytes = 0
    local_read_extents = 0
    replayed_bytes = 0
    load_error: BaseException | None = None
    try:
        for path, rows in work:
            batch_descriptors = [item.descriptor for item in rows]
            batch_bytes = sum(item.length for item in batch_descriptors)
            batch_slab = staging_slab[:batch_bytes]
            source_views = _storage_views(
                batch_slab,
                batch_descriptors,
                physical_order=True,
            )
            source_tensors = {
                item.source_key: tensor for item, tensor in zip(rows, source_views, strict=True)
            }
            source_base_offsets = {item.source_key: item.source_base_offset_bytes for item in rows}
            phase = time.perf_counter()
            extents = _coalesced_extents(
                batch_descriptors,
                source_views,
                [rank] * len(batch_descriptors),
                rank,
                coalesce_adjacent=True,
            )
            extent_count, _effective = _native_hydrate_owned(
                hydrator,
                path,
                extents,
                backend=backend,
                chunk_bytes=chunk_bytes,
                queue_depth=queue_depth,
                cuda_device=device,
            )
            io_seconds += time.perf_counter() - phase
            local_read_extents += extent_count
            local_read_bytes += batch_bytes

            batch_copies: list[RecoveryCopy] = []
            for item in rows:
                batch_copies.extend(copies_by_source.get(item.source_key, ()))
            batch_copies.sort(key=_copy_order)
            phase = time.perf_counter()
            replayed_bytes += _apply_recovery_copies(
                torch_module,
                batch_copies,
                source_tensors,
                destinations,
                source_base_offsets=source_base_offsets,
            )
            torch_module.cuda.synchronize(device)
            copy_seconds += time.perf_counter() - phase
    except BaseException as error:
        load_error = error

    status_seconds = 0.0
    status_calls = 0
    if group is not None:
        phase = time.perf_counter()
        status_device = "cpu" if status_group is not None else device
        status = torch_module.tensor(
            [1 if load_error is not None else 0],
            dtype=torch_module.int32,
            device=status_device,
        )
        torch_module.distributed.all_reduce(
            status,
            group=status_group if status_group is not None else group,
        )
        status_seconds += time.perf_counter() - phase
        status_calls += 1
        if int(status.item()) != 0:
            if load_error is not None:
                raise load_error
            raise RuntimeError("a peer failed while reading rank-local replay ranges")
    elif load_error is not None:
        raise load_error

    return {
        "backend": f"adapter-replay+{backend}+tp-rank-local-views",
        "files": len(by_path),
        "sources": len(ranges),
        "copies": len(copies),
        "source_bytes": source_bytes,
        "local_read_bytes": local_read_bytes,
        "local_read_extents": local_read_extents,
        "broadcast_bytes": 0,
        "rank_local_bytes": local_read_bytes,
        "replayed_bytes": replayed_bytes,
        "collective_calls": 0,
        "status_calls": status_calls,
        "status_seconds": status_seconds,
        "batches": len(work),
        "staging_seconds": staging_seconds,
        "io_seconds": io_seconds,
        "io_wait_seconds": io_seconds,
        "io_overlap_seconds": 0.0,
        "transport_pipeline_depth": 1,
        "broadcast_seconds": 0.0,
        "copy_seconds": copy_seconds,
        "transport_seconds": time.perf_counter() - started,
    }


def _rank_local_packed_work(
    torch_module: Any,
    copies: Sequence[RecoveryCopy],
    staging_bytes: int,
) -> list[tuple[Path, list[RecoveryCopy]]]:
    by_path: dict[str, list[RecoveryCopy]] = {}
    for copy in copies:
        by_path.setdefault(copy.source_path, []).append(copy)
    work: list[tuple[Path, list[RecoveryCopy]]] = []
    for path_text, rows in sorted(by_path.items()):
        rows.sort(key=_copy_order)
        batch: list[RecoveryCopy] = []
        batch_bytes = 0
        for copy in rows:
            dtype = _dtype_from_string(torch_module, copy.source_view_dtype)
            element_size = int(torch_module.empty((), dtype=dtype).element_size())
            logical_bytes = element_size
            for dimension in copy.source_view_shape:
                logical_bytes *= dimension
            aligned_start = ((batch_bytes + element_size - 1) // element_size) * element_size
            if batch and aligned_start + logical_bytes > staging_bytes:
                work.append((Path(path_text), batch))
                batch = []
                batch_bytes = 0
                aligned_start = 0
            if logical_bytes > staging_bytes:
                raise RuntimeError(
                    f"recovery packed source view exceeds staging budget: {copy.source_name!r}"
                )
            batch.append(copy)
            batch_bytes = aligned_start + logical_bytes
        if batch:
            work.append((Path(path_text), batch))
    return work


def _execute_rank_local_packed_replay(
    torch_module: Any,
    hydrator: Any,
    backend: str,
    group: Any,
    status_group: Any,
    device: Any,
    copies: Sequence[RecoveryCopy],
    destinations: dict[str, Any],
    *,
    source_bytes: int,
    chunk_bytes: int,
    queue_depth: int,
    staging_bytes: int,
) -> dict[str, Any]:
    """Hydrate logical strided source views from each rank's local cache."""
    work = _rank_local_packed_work(torch_module, copies, staging_bytes)
    if not work:
        raise RuntimeError("recovery rank-local packed replay has no source views")

    started = time.perf_counter()
    phase = time.perf_counter()
    largest_batch = max(_replay_pack_layout(torch_module, batch)[1] for _path, batch in work)
    staging_slab = torch_module.empty(
        largest_batch,
        dtype=torch_module.uint8,
        device=device,
    )
    torch_module.cuda.synchronize(device)
    staging_seconds = time.perf_counter() - phase
    io_seconds = 0.0
    copy_seconds = 0.0
    local_read_bytes = 0
    local_read_extents = 0
    replayed_bytes = 0
    replay_error: BaseException | None = None

    try:
        for path, batch_copies in work:
            offsets, packed_bytes = _replay_pack_layout(torch_module, batch_copies)
            batch_slab = staging_slab[:packed_bytes]
            native_extents: list[HydrationExtent] = []
            base = int(batch_slab.data_ptr())
            for copy, offset in zip(batch_copies, offsets, strict=True):
                packed_extents = _packed_copy_extents(torch_module, copy, offset)
                source_limit = copy.source_file_offset + copy.source_length
                for item in packed_extents:
                    if (
                        item.file_offset < copy.source_file_offset
                        or item.file_offset + item.length > source_limit
                    ):
                        raise RuntimeError(
                            f"recovery packed source extent changed for {copy.source_name!r}"
                        )
                    extent = HydrationExtent(
                        file_offset=item.file_offset,
                        destination=base + item.packed_offset_bytes,
                        length=item.length,
                    )
                    if (
                        native_extents
                        and native_extents[-1].file_offset + native_extents[-1].length
                        == extent.file_offset
                        and native_extents[-1].destination + native_extents[-1].length
                        == extent.destination
                    ):
                        previous = native_extents[-1]
                        native_extents[-1] = HydrationExtent(
                            file_offset=previous.file_offset,
                            destination=previous.destination,
                            length=previous.length + extent.length,
                        )
                    else:
                        native_extents.append(extent)

            phase = time.perf_counter()
            extent_count, _effective = _native_hydrate_owned(
                hydrator,
                path,
                native_extents,
                backend=backend,
                chunk_bytes=chunk_bytes,
                queue_depth=queue_depth,
                cuda_device=device,
            )
            io_seconds += time.perf_counter() - phase
            local_read_bytes += packed_bytes
            local_read_extents += extent_count

            phase = time.perf_counter()
            replayed_bytes += _apply_packed_recovery_copies(
                torch_module,
                batch_copies,
                batch_slab,
                destinations,
            )
            torch_module.cuda.synchronize(device)
            copy_seconds += time.perf_counter() - phase
    except BaseException as error:
        replay_error = error

    status_seconds = 0.0
    status_calls = 0
    if group is not None:
        phase = time.perf_counter()
        status_device = "cpu" if status_group is not None else device
        status = torch_module.tensor(
            [1 if replay_error is not None else 0],
            dtype=torch_module.int32,
            device=status_device,
        )
        torch_module.distributed.all_reduce(
            status,
            group=status_group if status_group is not None else group,
        )
        status_seconds += time.perf_counter() - phase
        status_calls += 1
        if int(status.item()) != 0:
            if replay_error is not None:
                raise replay_error
            raise RuntimeError("a peer failed while reading packed recovery views")
    elif replay_error is not None:
        raise replay_error

    return {
        "backend": f"adapter-replay+{backend}+tp-rank-local-packed-views",
        "files": len({copy.source_path for copy in copies}),
        "sources": len({copy.source_key for copy in copies}),
        "copies": len(copies),
        "source_bytes": source_bytes,
        "local_read_bytes": local_read_bytes,
        "local_read_extents": local_read_extents,
        "broadcast_bytes": 0,
        "packed_peer_bytes": 0,
        "packed_transport_batches": 0,
        "rank_local_bytes": local_read_bytes,
        "replayed_bytes": replayed_bytes,
        "collective_calls": 0,
        "status_calls": status_calls,
        "status_seconds": status_seconds,
        "batches": len(work),
        "staging_seconds": staging_seconds,
        "io_seconds": io_seconds,
        "io_wait_seconds": io_seconds,
        "io_overlap_seconds": 0.0,
        "transport_pipeline_depth": 1,
        "broadcast_seconds": 0.0,
        "copy_seconds": copy_seconds,
        "transport_seconds": time.perf_counter() - started,
    }


def _finish_direct_replay(
    torch_module: Any,
    direct_adapters: Sequence[tuple[Any, frozenset[str]]],
) -> tuple[float, float]:
    phase = time.perf_counter()
    torch_module.accelerator.synchronize()
    replay_sync_seconds = time.perf_counter() - phase
    phase = time.perf_counter()
    for adapter, _planned_names in direct_adapters:
        finalize_direct_replay = getattr(adapter, "finalize_direct_replay", None)
        if callable(finalize_direct_replay):
            finalize_direct_replay(torch_module)
    torch_module.accelerator.synchronize()
    finalizer_seconds = time.perf_counter() - phase
    for adapter, _planned_names in direct_adapters:
        end_direct_replay = getattr(adapter, "end_direct_replay", None)
        if callable(end_direct_replay):
            end_direct_replay()
    return replay_sync_seconds, finalizer_seconds


def _synchronize_replay_stream(torch_module: Any, device: Any) -> None:
    """Wait for replay work without draining an overlapping native I/O stream."""
    current_stream = getattr(torch_module.cuda, "current_stream", None)
    if callable(current_stream):
        current_stream(device).synchronize()
    else:
        torch_module.cuda.synchronize(device)


def _apply_recovery_copies(
    torch_module: Any,
    copies: Sequence[RecoveryCopy],
    source_tensors: dict[tuple[Any, ...], Any],
    destinations: dict[str, Any],
    *,
    source_base_offsets: dict[tuple[Any, ...], int] | None = None,
) -> int:
    copied_bytes = 0
    for copy in copies:
        source_tensor = source_tensors[copy.source_key]
        source_view = _typed_storage_view(
            torch_module,
            source_tensor,
            relative_byte_offset=(
                copy.source_view_offset_bytes - (source_base_offsets or {}).get(copy.source_key, 0)
            ),
            dtype_name=copy.source_view_dtype,
            shape=copy.source_view_shape,
            stride=copy.source_view_stride,
        )
        destination_tensor = destinations[copy.destination_name]
        destination_view = _typed_storage_view(
            torch_module,
            destination_tensor,
            relative_byte_offset=copy.destination_view_offset_bytes,
            dtype_name=copy.destination_dtype,
            shape=copy.destination_view_shape,
            stride=copy.destination_view_stride,
        )
        if (
            tuple(source_view.shape) != tuple(destination_view.shape)
            or int(source_view.numel()) != int(destination_view.numel())
            or int(destination_view.numel() * destination_view.element_size()) != copy.copy_bytes
        ):
            raise RuntimeError("recovery replay copy metadata changed")
        destination_view.copy_(source_view)
        copied_bytes += copy.copy_bytes
    return copied_bytes


def _validate_replay_destinations(plan: RecoveryCopyPlan, model: Any) -> dict[str, Any]:
    destinations = dict(model.named_parameters())
    destinations.update(dict(model.named_buffers()))
    selected: dict[str, Any] = {}
    for copy in plan.copies:
        tensor = destinations.get(copy.destination_name)
        if tensor is None:
            raise RuntimeError(f"recovery replay destination disappeared: {copy.destination_name}")
        if (
            tuple(tensor.shape) != copy.destination_shape
            or str(tensor.dtype) != copy.destination_dtype
            or not bool(tensor.is_cuda)
            or not bool(tensor.is_contiguous())
        ):
            raise RuntimeError(
                f"recovery replay destination metadata changed for {copy.destination_name}"
            )
        selected[copy.destination_name] = tensor
    return selected


def _execute_adapter_replay(
    torch_module: Any,
    model: Any,
    plans: Sequence[RecoveryCopyPlan],
    adapters: Sequence[Any],
    *,
    captured_rank_io_weights: Sequence[float] | None = None,
) -> tuple[frozenset[str], frozenset[str], dict[str, Any] | None]:
    """Replay exact checkpoint copies into stable registered/adapter storage."""
    if not plans:
        return frozenset(), frozenset(), None
    with _io_policy_lock:
        _io_policy_observations.clear()
    hydrator, backend = _native_hydrator()
    if hydrator is None or backend in {"mmap", "torch"}:
        return frozenset(), frozenset(), None

    group, rank, world_size = _distributed_context(torch_module)
    status_group = _distributed_status_group() if group is not None else None
    if len(plans) != world_size:
        raise RuntimeError(
            f"recovery copy-plan world size changed: captured={len(plans)} live={world_size}"
        )
    local_plan = plans[rank]
    local_destinations = {copy.destination_name for copy in local_plan.copies}
    checkpoint_destinations = _model_checkpoint_destination_names(model) or frozenset()
    normal_reload_destinations = checkpoint_destinations - local_destinations
    direct_adapters: list[tuple[Any, frozenset[str]]] = []
    for adapter in adapters:
        names = frozenset(getattr(adapter, "direct_replay_destination_names", ()))
        planned_names = names & local_destinations
        complete = names <= local_destinations
        partial = bool(getattr(adapter, "allows_partial_direct_replay", False))
        if (
            planned_names
            and (complete or partial)
            and callable(getattr(adapter, "begin_direct_replay", None))
        ):
            direct_adapters.append((adapter, planned_names))
    if not local_plan.copies:
        return frozenset(), frozenset(), None

    for adapter, planned_names in direct_adapters:
        configure = getattr(adapter, "configure_direct_replay", None)
        if callable(configure):
            configure(planned_names)
        configure_deferred = getattr(
            adapter,
            "configure_deferred_direct_replay_finalization",
            None,
        )
        if callable(configure_deferred):
            configure_deferred(normal_reload_destinations)
        adapter.begin_direct_replay(torch_module)
    destinations = _validate_replay_destinations(local_plan, model)

    rank_copies: list[dict[tuple[Any, ...], list[RecoveryCopy]]] = []
    sources: dict[tuple[Any, ...], SafetensorDescriptor] = {}
    for plan in plans:
        grouped: dict[tuple[Any, ...], list[RecoveryCopy]] = {}
        for copy in plan.copies:
            grouped.setdefault(copy.source_key, []).append(copy)
            descriptor = SafetensorDescriptor(
                name=copy.source_name,
                dtype_name=copy.source_dtype_name,
                shape=copy.source_shape,
                file_offset=copy.source_file_offset,
                length=copy.source_length,
            )
            previous = sources.setdefault(copy.source_key, descriptor)
            if previous != descriptor:
                raise RuntimeError(f"recovery peers disagree about source {copy.source_name!r}")
        for copies in grouped.values():
            copies.sort(key=_copy_order)
        rank_copies.append(grouped)

    chunk_bytes = int(os.environ.get("COLDSNAP_RECOVERY_LOADER_CHUNK_BYTES", str(64 * 1024**2)))
    queue_depth = int(os.environ.get("COLDSNAP_RECOVERY_LOADER_QUEUE_DEPTH", "4"))
    staging_bytes = int(os.environ.get(LOADER_STAGING_BYTES_ENV, str(DEFAULT_STAGING_BYTES)))
    collective_bytes = int(
        os.environ.get(LOADER_COLLECTIVE_BYTES_ENV, str(DEFAULT_COLLECTIVE_BYTES))
    )
    local_packed_max_extents = int(
        os.environ.get(
            LOADER_LOCAL_PACKED_MAX_EXTENTS_ENV,
            str(DEFAULT_LOCAL_PACKED_MAX_EXTENTS),
        )
    )
    local_packed_min_extent_bytes = int(
        os.environ.get(
            LOADER_LOCAL_PACKED_MIN_EXTENT_BYTES_ENV,
            str(DEFAULT_LOCAL_PACKED_MIN_EXTENT_BYTES),
        )
    )
    io_probe_bytes = int(
        os.environ.get(
            LOADER_IO_PROBE_BYTES_ENV,
            str(DEFAULT_IO_PROBE_BYTES),
        )
    )
    transport_pipeline_depth = int(
        os.environ.get(
            LOADER_TRANSPORT_PIPELINE_DEPTH_ENV,
            str(DEFAULT_TRANSPORT_PIPELINE_DEPTH),
        )
    )
    if collective_bytes <= 0:
        raise ValueError("recovery collective bytes must be positive")
    if staging_bytes <= 0:
        raise ValueError("recovery staging bytes must be positive")
    if local_packed_max_extents <= 0:
        raise ValueError("recovery local packed maximum extents must be positive")
    if local_packed_min_extent_bytes <= 0:
        raise ValueError("recovery local packed minimum extent bytes must be positive")
    if io_probe_bytes < 0:
        raise ValueError("recovery I/O probe bytes cannot be negative")
    if transport_pipeline_depth not in {1, 2}:
        raise ValueError("recovery transport pipeline depth must be 1 or 2")
    device = torch_module.cuda.current_device()

    local_ranges = _rank_local_replay_ranges(
        torch_module,
        sources,
        rank_copies,
        rank,
    )
    if local_ranges is not None:
        metrics = _execute_rank_local_replay(
            torch_module,
            hydrator,
            backend,
            group,
            status_group,
            rank,
            device,
            local_ranges,
            local_plan.copies,
            destinations,
            source_bytes=sum(item.length for item in sources.values()),
            chunk_bytes=chunk_bytes,
            queue_depth=queue_depth,
            staging_bytes=staging_bytes,
        )
        replay_sync_seconds, finalizer_seconds = _finish_direct_replay(
            torch_module,
            direct_adapters,
        )
        metrics["replay_sync_seconds"] = replay_sync_seconds
        metrics["finalizer_seconds"] = finalizer_seconds
        metrics["total_seconds"] = (
            metrics.pop("transport_seconds") + replay_sync_seconds + finalizer_seconds
        )
        return (
            frozenset(copy.source_name for copy in local_plan.copies),
            frozenset(copy.destination_name for copy in local_plan.copies),
            metrics,
        )

    if _rank_local_packed_replay_plans_qualified(
        torch_module,
        plans,
        max_extents=local_packed_max_extents,
        min_extent_bytes=local_packed_min_extent_bytes,
    ):
        metrics = _execute_rank_local_packed_replay(
            torch_module,
            hydrator,
            backend,
            group,
            status_group,
            device,
            local_plan.copies,
            destinations,
            source_bytes=sum(item.length for item in sources.values()),
            chunk_bytes=chunk_bytes,
            queue_depth=queue_depth,
            staging_bytes=staging_bytes,
        )
        replay_sync_seconds, finalizer_seconds = _finish_direct_replay(
            torch_module,
            direct_adapters,
        )
        metrics["replay_sync_seconds"] = replay_sync_seconds
        metrics["finalizer_seconds"] = finalizer_seconds
        metrics["total_seconds"] = (
            metrics.pop("transport_seconds") + replay_sync_seconds + finalizer_seconds
        )
        return (
            frozenset(copy.source_name for copy in local_plan.copies),
            frozenset(copy.destination_name for copy in local_plan.copies),
            metrics,
        )

    started = time.perf_counter()
    captured_io_weights: tuple[float, ...] | None = None
    if captured_rank_io_weights:
        if len(captured_rank_io_weights) == world_size:
            try:
                captured_io_weights = _bounded_rank_io_weights(captured_rank_io_weights)
            except ValueError:
                captured_io_weights = None
        io_probe_metrics = {
            "bytes": 0,
            "local_seconds": 0.0,
            "collective_seconds": 0.0,
            "rank_weights": list(captured_io_weights or ()),
            "source": "initial-load" if captured_io_weights is not None else "invalid-captured",
        }
    else:
        io_probe_metrics = {
            "bytes": 0,
            "local_seconds": 0.0,
            "collective_seconds": 0.0,
            "source": "disabled",
        }
    rank_io_weights = captured_io_weights
    if group is not None and io_probe_bytes:
        rank_io_weights, io_probe_metrics = _probe_recovery_rank_io(
            torch_module,
            hydrator,
            backend,
            group,
            status_group,
            device,
            sources,
            chunk_bytes=chunk_bytes,
            queue_depth=queue_depth,
            probe_bytes=io_probe_bytes,
        )
        io_probe_metrics["captured_rank_weights"] = list(captured_io_weights or ())

    by_path: dict[str, list[tuple[tuple[Any, ...], SafetensorDescriptor]]] = {}
    for key, descriptor in sources.items():
        by_path.setdefault(key[1], []).append((key, descriptor))
    io_seconds = 0.0
    io_wait_seconds = 0.0
    broadcast_seconds = 0.0
    copy_seconds = 0.0
    local_read_bytes = 0
    local_read_extents = 0
    broadcast_bytes = 0
    rank_local_bytes = 0
    replayed_bytes = 0
    collective_calls = 0
    status_calls = 0
    status_seconds = 0.0
    batches = 0
    work_by_owner: list[
        list[
            tuple[
                Path,
                list[tuple[tuple[Any, ...], SafetensorDescriptor]],
                tuple[int, ...],
            ]
        ]
    ] = [[] for _rank in range(world_size)]
    planned_rank_bytes = [0] * world_size
    for key, descriptor in sources.items():
        consumers = tuple(rank for rank, copies in enumerate(rank_copies) if key in copies)
        if len(consumers) == 1:
            planned_rank_bytes[consumers[0]] += descriptor.length
    for file_number, (path_text, rows) in enumerate(sorted(by_path.items())):
        rows.sort(key=lambda item: item[1].file_offset)
        assignments = _replay_transport_owners(
            rows,
            rank_copies,
            world_size,
            rotation=file_number % world_size,
            rank_io_weights=rank_io_weights,
            rank_assigned_bytes=planned_rank_bytes,
        )
        grouped: dict[
            tuple[int, tuple[int, ...]],
            list[tuple[tuple[Any, ...], SafetensorDescriptor]],
        ] = {}
        for row, assignment in zip(rows, assignments, strict=True):
            grouped.setdefault(assignment, []).append(row)
            owner, consumers = assignment
            if len(consumers) > 1:
                planned_rank_bytes[owner] += row[1].length
        for (owner, consumers), owned_rows in sorted(grouped.items()):
            owned_descriptors = [descriptor for _key, descriptor in owned_rows]
            for batch_start, batch_end in _descriptor_batches(owned_descriptors, staging_bytes):
                work_by_owner[owner].append(
                    (Path(path_text), owned_rows[batch_start:batch_end], consumers)
                )

    planned_owner_bytes = [
        sum(
            descriptor.length
            for _path, batch_rows, _consumers in owner_work
            for _key, descriptor in batch_rows
        )
        for owner_work in work_by_owner
    ]

    largest_slab_bytes = max(
        (
            sum(descriptor.length for _key, descriptor in batch_rows)
            for owner_work in work_by_owner
            for _path, batch_rows, _consumers in owner_work
        ),
        default=0,
    )
    if largest_slab_bytes <= 0:
        raise RuntimeError("recovery replay transport has no source batches")
    largest_peer_pack_bytes = 0
    if world_size == 2:
        for owner, owner_work in enumerate(work_by_owner):
            peer = 1 - owner
            for _path, batch_rows, consumers in owner_work:
                if len(consumers) <= 1:
                    continue
                peer_copies = [
                    copy
                    for key, _descriptor in batch_rows
                    for copy in rank_copies[peer].get(key, ())
                ]
                peer_copies.sort(key=_copy_order)
                _offsets, packed_bytes = _replay_pack_layout(torch_module, peer_copies)
                largest_peer_pack_bytes = max(largest_peer_pack_bytes, packed_bytes)
    phase = time.perf_counter()
    source_staging_storage = torch_module.empty(
        largest_slab_bytes * transport_pipeline_depth,
        dtype=torch_module.uint8,
        device=device,
    )
    source_staging_slabs = [
        source_staging_storage[index * largest_slab_bytes : (index + 1) * largest_slab_bytes]
        for index in range(transport_pipeline_depth)
    ]
    receive_staging_slab = (
        torch_module.empty(
            max(largest_slab_bytes, largest_peer_pack_bytes),
            dtype=torch_module.uint8,
            device=device,
        )
        if world_size > 1
        else source_staging_slabs[0]
    )
    torch_module.cuda.synchronize(device)
    staging_seconds = time.perf_counter() - phase
    packed_peer_bytes = 0
    packed_transport_batches = 0
    replay_error: BaseException | None = None

    rounds = max((len(work) for work in work_by_owner), default=0)

    def schedule_local_read(
        round_index: int,
        executor: ThreadPoolExecutor | None,
    ) -> tuple[
        Any | None,
        dict[tuple[Any, ...], Any],
        Future[tuple[int, str, float]] | None,
        tuple[int, str, float] | None,
        int,
        BaseException | None,
    ]:
        local_work = (
            work_by_owner[rank][round_index] if round_index < len(work_by_owner[rank]) else None
        )
        if local_work is None:
            return None, {}, None, None, 0, None
        path, batch_rows, _batch_consumers = local_work
        batch_keys = [key for key, _descriptor in batch_rows]
        batch_descriptors = [descriptor for _key, descriptor in batch_rows]
        batch_bytes = sum(item.length for item in batch_descriptors)
        source_slab = source_staging_slabs[round_index % transport_pipeline_depth][:batch_bytes]
        try:
            source_tensors = dict(
                zip(
                    batch_keys,
                    _storage_views(
                        source_slab,
                        batch_descriptors,
                        physical_order=True,
                    ),
                    strict=True,
                )
            )
            extents = _coalesced_extents(
                batch_descriptors,
                list(source_tensors.values()),
                [rank] * len(batch_descriptors),
                rank,
                coalesce_adjacent=True,
            )
        except BaseException as error:
            source_slab.zero_()
            _synchronize_replay_stream(torch_module, device)
            return source_slab, {}, None, None, batch_bytes, error
        if replay_error is not None:
            source_slab.zero_()
            _synchronize_replay_stream(torch_module, device)
            return source_slab, source_tensors, None, None, batch_bytes, None
        arguments = (
            hydrator,
            path,
            extents,
        )
        options = {
            "backend": backend,
            "chunk_bytes": chunk_bytes,
            "queue_depth": queue_depth,
            "cuda_device": device,
        }
        if executor is not None:
            return (
                source_slab,
                source_tensors,
                executor.submit(_timed_native_hydrate_owned, *arguments, **options),
                None,
                batch_bytes,
                None,
            )
        try:
            result = _timed_native_hydrate_owned(*arguments, **options)
        except BaseException as error:
            source_slab.zero_()
            _synchronize_replay_stream(torch_module, device)
            return source_slab, source_tensors, None, None, batch_bytes, error
        return source_slab, source_tensors, None, result, batch_bytes, None

    executor_context = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="coldsnap-replay-io")
        if transport_pipeline_depth == 2
        else nullcontext(None)
    )
    with executor_context as read_executor:
        pending_read = schedule_local_read(0, read_executor) if rounds else None
        for round_index in range(rounds):
            assert pending_read is not None
            (
                source_slab,
                source_tensors,
                read_future,
                read_result,
                local_batch_bytes,
                read_error,
            ) = pending_read
            if read_future is not None:
                phase = time.perf_counter()
                try:
                    read_result = read_future.result()
                except BaseException as error:
                    read_error = error
                io_wait_seconds += time.perf_counter() - phase
            elif read_result is not None:
                io_wait_seconds += read_result[2]
            if read_result is not None:
                extent_count, _effective, service_seconds = read_result
                io_seconds += service_seconds
                local_read_extents += extent_count
                local_read_bytes += local_batch_bytes
            if read_error is not None:
                if replay_error is None:
                    replay_error = read_error
                if source_slab is not None:
                    source_slab.zero_()
                    _synchronize_replay_stream(torch_module, device)

            next_pending = None
            if transport_pipeline_depth == 2 and round_index + 1 < rounds:
                next_pending = schedule_local_read(round_index + 1, read_executor)

            # Keep a deterministic collective order even after a local read or
            # copy failure. The failed rank supplies a zero payload and every rank
            # reports failure in one final status reduction, avoiding both peer
            # deadlock and a blocking control collective before every data round.
            # TP2 packs only the peer's logical views into the reusable receive
            # slab; larger groups retain the raw-source broadcast contract below.
            for owner in range(world_size):
                if round_index >= len(work_by_owner[owner]):
                    continue
                _path, batch_rows, consumers = work_by_owner[owner][round_index]
                batch_keys = [key for key, _descriptor in batch_rows]
                batch_descriptors = [descriptor for _key, descriptor in batch_rows]
                slab_bytes = sum(descriptor.length for descriptor in batch_descriptors)

                # Rank-exclusive sources already reside on their only consumer.
                # No peer allocates a receive slab and no collective is issued.
                if consumers == (owner,) and rank != owner:
                    continue

                if group is not None and world_size == 2 and len(consumers) > 1:
                    peer = 1 - owner
                    owner_copies = [
                        copy for key in batch_keys for copy in rank_copies[owner].get(key, ())
                    ]
                    peer_copies = [
                        copy for key in batch_keys for copy in rank_copies[peer].get(key, ())
                    ]
                    owner_copies.sort(key=_copy_order)
                    peer_copies.sort(key=_copy_order)
                    _offsets, peer_bytes = _replay_pack_layout(torch_module, peer_copies)
                    transfer_slab = receive_staging_slab[:peer_bytes]

                    phase = time.perf_counter()
                    copied = 0
                    if rank == owner:
                        assert source_slab is not None
                        try:
                            if replay_error is None:
                                copied = _apply_recovery_copies(
                                    torch_module,
                                    owner_copies,
                                    source_tensors,
                                    destinations,
                                )
                                packed = _pack_recovery_copies(
                                    torch_module,
                                    peer_copies,
                                    source_tensors,
                                    transfer_slab,
                                )
                                if packed != peer_bytes:
                                    raise RuntimeError("recovery replay peer-pack size changed")
                            else:
                                transfer_slab.zero_()
                        except BaseException as error:
                            if replay_error is None:
                                replay_error = error
                            copied = 0
                            transfer_slab.zero_()
                    _synchronize_replay_stream(torch_module, device)
                    copy_seconds += time.perf_counter() - phase

                    phase = time.perf_counter()
                    source_rank = _global_source_rank(torch_module.distributed, group, owner)
                    offset = 0
                    while offset < peer_bytes:
                        end = min(offset + collective_bytes, peer_bytes)
                        torch_module.distributed.broadcast(
                            transfer_slab[offset:end],
                            src=source_rank,
                            group=group,
                        )
                        collective_calls += 1
                        offset = end
                    _synchronize_replay_stream(torch_module, device)
                    broadcast_seconds += time.perf_counter() - phase

                    if rank == peer:
                        phase = time.perf_counter()
                        try:
                            if replay_error is None:
                                copied = _apply_packed_recovery_copies(
                                    torch_module,
                                    peer_copies,
                                    transfer_slab,
                                    destinations,
                                )
                        except BaseException as error:
                            if replay_error is None:
                                replay_error = error
                            copied = 0
                        _synchronize_replay_stream(torch_module, device)
                        copy_seconds += time.perf_counter() - phase
                    replayed_bytes += copied
                    broadcast_bytes += peer_bytes
                    packed_peer_bytes += peer_bytes
                    packed_transport_batches += 1
                    batches += 1
                    continue

                phase = time.perf_counter()
                if rank == owner:
                    assert source_slab is not None
                    broadcast_slab = source_slab
                    broadcast_sources = source_tensors
                else:
                    broadcast_slab = receive_staging_slab[:slab_bytes]
                    broadcast_sources = {}
                    try:
                        broadcast_sources = dict(
                            zip(
                                batch_keys,
                                _storage_views(
                                    broadcast_slab,
                                    batch_descriptors,
                                    physical_order=True,
                                ),
                                strict=True,
                            )
                        )
                    except BaseException as error:
                        if replay_error is None:
                            replay_error = error
                if group is not None and len(consumers) > 1:
                    source_rank = _global_source_rank(torch_module.distributed, group, owner)
                    offset = 0
                    while offset < slab_bytes:
                        end = min(offset + collective_bytes, slab_bytes)
                        torch_module.distributed.broadcast(
                            broadcast_slab[offset:end],
                            src=source_rank,
                            group=group,
                        )
                        collective_calls += 1
                        offset = end
                _synchronize_replay_stream(torch_module, device)
                broadcast_seconds += time.perf_counter() - phase
                if len(consumers) > 1:
                    broadcast_bytes += slab_bytes
                else:
                    rank_local_bytes += slab_bytes

                phase = time.perf_counter()
                local_copies: list[RecoveryCopy] = []
                for key in batch_keys:
                    local_copies.extend(rank_copies[rank].get(key, ()))
                local_copies.sort(key=_copy_order)
                copied = 0
                try:
                    if replay_error is None:
                        copied = _apply_recovery_copies(
                            torch_module,
                            local_copies,
                            broadcast_sources,
                            destinations,
                        )
                except BaseException as error:
                    if replay_error is None:
                        replay_error = error
                _synchronize_replay_stream(torch_module, device)
                copy_seconds += time.perf_counter() - phase
                replayed_bytes += copied
                batches += 1
                del broadcast_sources

            if transport_pipeline_depth == 1 and round_index + 1 < rounds:
                next_pending = schedule_local_read(round_index + 1, read_executor)
            pending_read = next_pending

    if group is not None:
        phase = time.perf_counter()
        status_device = "cpu" if status_group is not None else device
        status = torch_module.tensor(
            [1 if replay_error is not None else 0],
            dtype=torch_module.int32,
            device=status_device,
        )
        torch_module.distributed.all_reduce(
            status,
            group=status_group if status_group is not None else group,
        )
        status_seconds += time.perf_counter() - phase
        status_calls += 1
        if int(status.item()) != 0:
            if replay_error is not None:
                raise replay_error
            raise RuntimeError("a peer failed while replaying recovery safetensors")
    elif replay_error is not None:
        raise replay_error

    replay_sync_seconds, finalizer_seconds = _finish_direct_replay(
        torch_module,
        direct_adapters,
    )
    source_names = frozenset(copy.source_name for copy in local_plan.copies)
    destination_names = frozenset(copy.destination_name for copy in local_plan.copies)
    return (
        source_names,
        destination_names,
        {
            "backend": (
                f"adapter-replay+{backend}+tp2-peer-packed"
                if packed_transport_batches
                else f"adapter-replay+{backend}+tp-consumer-aware"
            ),
            "files": len(by_path),
            "sources": len(sources),
            "copies": sum(len(value) for value in rank_copies[rank].values()),
            "source_bytes": sum(item.length for item in sources.values()),
            "io_probe": io_probe_metrics,
            "io_policy": list(_io_policy_observations.values()),
            "planned_owner_bytes": planned_owner_bytes,
            "local_read_bytes": local_read_bytes,
            "local_read_extents": local_read_extents,
            "broadcast_bytes": broadcast_bytes,
            "packed_peer_bytes": packed_peer_bytes,
            "packed_transport_batches": packed_transport_batches,
            "rank_local_bytes": rank_local_bytes,
            "replayed_bytes": replayed_bytes,
            "collective_calls": collective_calls,
            "status_calls": status_calls,
            "status_seconds": status_seconds,
            "batches": batches,
            "staging_seconds": staging_seconds,
            "io_seconds": io_seconds,
            "io_wait_seconds": io_wait_seconds,
            "io_overlap_seconds": max(0.0, io_seconds - io_wait_seconds),
            "transport_pipeline_depth": transport_pipeline_depth,
            "broadcast_seconds": broadcast_seconds,
            "copy_seconds": copy_seconds,
            "replay_sync_seconds": replay_sync_seconds,
            "finalizer_seconds": finalizer_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    )


def _mmap_weights_iterator(
    files: Sequence[str],
    *,
    folder: str,
    prefix: str,
    weight_name_prefixes: Sequence[str] | None,
    local_expert_ids: set[int] | None,
    verify_bytes: int,
    logger: Any,
    started: float,
) -> Iterator[tuple[str, Any]]:
    """Expose lazy CPU mmap tensors in publisher/model order.

    This is the production recovery transport.  ``safe_open`` maps the
    original HF shards without eagerly copying them.  vLLM's ordinary model
    loaders then touch only the TP/EP slices they consume and perform the
    model-specific packing before the layerwise reload copies results back to
    the captured stable CUDA storages.
    """
    global _last_metrics

    import torch
    from safetensors import safe_open
    from vllm.model_executor.model_loader.weight_utils import should_skip_weight

    index = _index_map(folder)
    logical_bytes = 0
    tensor_count = 0
    verification_seconds = 0.0
    verified_sample_bytes = 0
    seen: set[str] = set()

    for file_number, filename in enumerate(sorted(files)):
        path = Path(filename)
        descriptors = _descriptors(
            path,
            indexed_tensor_files=index,
            weight_name_prefixes=weight_name_prefixes,
        )
        duplicates = sorted({item.name for item in descriptors} & seen)
        if duplicates:
            raise RuntimeError(f"duplicate selected safetensors tensor {duplicates[0]!r}")
        seen.update(item.name for item in descriptors)
        logger.info(
            "ColdSnap recovery mmap opening safetensors file %d/%d: %s",
            file_number + 1,
            len(files),
            path.name,
        )
        with safe_open(path, framework="pt", device="cpu") as source:
            for descriptor in descriptors:
                if should_skip_weight(descriptor.name, local_expert_ids):
                    continue
                full_name = prefix + descriptor.name
                if full_name in _RECOVERY_SKIP_SOURCE_NAMES.get():
                    continue
                tensor = source.get_tensor(descriptor.name)
                actual_bytes = int(tensor.numel() * tensor.element_size())
                if tuple(tensor.shape) != descriptor.shape or actual_bytes != descriptor.length:
                    raise RuntimeError(
                        f"safetensors tensor metadata changed for {descriptor.name!r} in {path}"
                    )
                if verify_bytes:
                    phase = time.perf_counter()
                    raw = tensor.reshape(-1).view(torch.uint8)
                    verified_sample_bytes += _verify_tensor_samples(
                        torch,
                        path,
                        [descriptor],
                        [raw],
                        verify_bytes,
                    )
                    verification_seconds += time.perf_counter() - phase
                logical_bytes += descriptor.length
                tensor_count += 1
                active = RecoverySourceTensor(
                    name=full_name,
                    path=os.path.abspath(path),
                    descriptor=descriptor,
                    pointer=int(tensor.data_ptr()),
                )
                token = _ACTIVE_RECOVERY_SOURCE.set(active)
                try:
                    yield full_name, tensor
                finally:
                    _ACTIVE_RECOVERY_SOURCE.reset(token)

        logger.info(
            "ColdSnap recovery mmap supplied %d/%d safetensors files (%.2f GiB logical)",
            file_number + 1,
            len(files),
            logical_bytes / 1024**3,
        )

    total_seconds = time.perf_counter() - started
    _last_metrics = RecoveryLoadMetrics(
        files=len(files),
        tensors=tensor_count,
        logical_bytes=logical_bytes,
        # mmap page faults and the model loader's TP slices are intentionally
        # demand-driven; there is no truthful explicit-read byte counter here.
        local_read_bytes=0,
        local_read_extents=0,
        io_seconds=0.0,
        collective_seconds=0.0,
        collective_calls=0,
        verification_seconds=verification_seconds,
        verified_sample_bytes=verified_sample_bytes,
        total_seconds=total_seconds,
        backend="mmap",
        distributed_world_size=1,
        staging_layout="vllm-lazy-mmap",
        staging_buffer_bytes=0,
        collective_buffer_bytes=0,
    )
    logger.info(
        "ColdSnap recovery mmap supplied %d tensors (%.2f GiB logical) in "
        "%.3f s; vLLM performed demand-paged TP/EP slicing",
        tensor_count,
        logical_bytes / 1024**3,
        total_seconds,
    )


def recovery_weights_iterator(
    files: Sequence[str],
    *,
    folder: str,
    prefix: str,
    weight_name_prefixes: Sequence[str] | None,
    local_expert_ids: set[int] | None = None,
) -> Iterator[tuple[str, Any]]:
    """Yield checkpoint tensors through the selected recovery transport."""
    global _last_metrics

    _last_metrics = None
    with _io_policy_lock:
        _io_policy_observations.clear()
    import torch
    from vllm.logger import init_logger
    from vllm.model_executor.model_loader.weight_utils import should_skip_weight

    logger = init_logger(__name__)
    started = time.perf_counter()
    logger.info("ColdSnap recovery iterator entered for %d safetensors files", len(files))
    hydrator, backend = _native_hydrator()
    logger.info("ColdSnap recovery iterator selected %s source transport", backend)
    verify_bytes = int(os.environ.get(LOADER_VERIFY_BYTES_ENV, "0"))
    if verify_bytes < 0 or verify_bytes > 4096:
        raise ValueError(f"{LOADER_VERIFY_BYTES_ENV} must be between 0 and 4096")
    if backend == "mmap":
        yield from _mmap_weights_iterator(
            files,
            folder=folder,
            prefix=prefix,
            weight_name_prefixes=weight_name_prefixes,
            local_expert_ids=local_expert_ids,
            verify_bytes=verify_bytes,
            logger=logger,
            started=started,
        )
        return
    group, rank, world_size = _distributed_context(torch)
    logger.info("ColdSnap recovery iterator resolved TP rank %d/%d", rank, world_size)
    index = _index_map(folder)
    logger.info(
        "ColdSnap recovery iterator parsed checkpoint index (%d entries)",
        len(index) if index is not None else 0,
    )
    chunk_bytes = int(os.environ.get("COLDSNAP_RECOVERY_LOADER_CHUNK_BYTES", str(64 * 1024**2)))
    queue_depth = int(os.environ.get("COLDSNAP_RECOVERY_LOADER_QUEUE_DEPTH", "4"))
    collective_buffer_bytes = int(
        os.environ.get(LOADER_COLLECTIVE_BYTES_ENV, str(DEFAULT_COLLECTIVE_BYTES))
    )
    staging_buffer_bytes = int(os.environ.get(LOADER_STAGING_BYTES_ENV, str(DEFAULT_STAGING_BYTES)))
    if chunk_bytes <= 0 or chunk_bytes % 4096:
        raise ValueError("COLDSNAP_RECOVERY_LOADER_CHUNK_BYTES must be 4096-byte aligned")
    if not 1 <= queue_depth <= 64:
        raise ValueError("COLDSNAP_RECOVERY_LOADER_QUEUE_DEPTH must be between 1 and 64")
    if collective_buffer_bytes <= 0 or collective_buffer_bytes % 4096:
        raise ValueError(f"{LOADER_COLLECTIVE_BYTES_ENV} must be 4096-byte aligned")
    if staging_buffer_bytes <= 0:
        raise ValueError(f"{LOADER_STAGING_BYTES_ENV} must be positive")

    logical_bytes = 0
    local_read_bytes = 0
    local_read_extents = 0
    tensor_count = 0
    io_seconds = 0.0
    collective_seconds = 0.0
    collective_calls = 0
    verification_seconds = 0.0
    verified_sample_bytes = 0
    effective_backends: set[str] = set()
    seen: set[str] = set()
    device = torch.cuda.current_device()
    isolate_consumer_storage = _RECOVERY_CONSUMER_STORAGE_ISOLATION.get()

    for file_number, filename in enumerate(sorted(files)):
        path = Path(filename)
        logger.info(
            "ColdSnap recovery loader opening safetensors file %d/%d: %s",
            file_number + 1,
            len(files),
            path.name,
        )
        descriptors = _descriptors(
            path,
            indexed_tensor_files=index,
            weight_name_prefixes=weight_name_prefixes,
        )
        descriptors = [
            descriptor
            for descriptor in descriptors
            if not should_skip_weight(descriptor.name, local_expert_ids)
            and prefix + descriptor.name not in _RECOVERY_SKIP_SOURCE_NAMES.get()
        ]
        # Native hydration operates on file extents, so batch in physical
        # safetensors order. Keeping publisher order here fragments a TP
        # rank's otherwise contiguous share of a shard into tens of thousands
        # of reads. Tensor names remain attached to their descriptors, and
        # vLLM model loaders do not require state-dict iteration order for
        # correctness. The layerwise bridge independently bounds incomplete
        # module state when a checkpoint layout interleaves parent layers.
        descriptors.sort(key=lambda descriptor: descriptor.file_offset)
        duplicates = sorted({item.name for item in descriptors} & seen)
        if duplicates:
            raise RuntimeError(f"duplicate selected safetensors tensor {duplicates[0]!r}")
        seen.update(item.name for item in descriptors)
        owners = _owners(
            descriptors,
            world_size,
            rotation=file_number % world_size,
            physical_order=True,
        )
        if hydrator is None:
            from safetensors import safe_open

            source_context = safe_open(path, framework="pt", device="cpu")
        else:
            source_context = nullcontext(None)
        with source_context as source:
            for batch_start, batch_end in _descriptor_batches(descriptors, staging_buffer_bytes):
                batch_descriptors = descriptors[batch_start:batch_end]
                batch_owners = owners[batch_start:batch_end]
                # Use one bounded slab rather than thousands of CUDA
                # allocations per DS4 shard. Tensor views retain at most this
                # batch (or one unavoidable oversized tensor) while vLLM
                # finishes the parent layer.
                staging_slab = torch.empty(
                    sum(descriptor.length for descriptor in batch_descriptors),
                    dtype=torch.uint8,
                    device=device,
                )
                tensor_storages = _storage_views(
                    staging_slab,
                    batch_descriptors,
                    physical_order=True,
                )
                tensors: list[Any] = []
                for descriptor, storage in zip(batch_descriptors, tensor_storages, strict=True):
                    dtype = _validate_descriptor_size(torch, descriptor)
                    tensors.append(storage.view(dtype).view(descriptor.shape))

                load_error: BaseException | None = None
                phase = time.perf_counter()
                try:
                    owned = [
                        (descriptor, tensor)
                        for descriptor, tensor, owner in zip(
                            batch_descriptors,
                            tensors,
                            batch_owners,
                            strict=True,
                        )
                        if owner == rank and descriptor.length > 0
                    ]
                    extents = _coalesced_extents(
                        batch_descriptors,
                        tensors,
                        batch_owners,
                        rank,
                        coalesce_adjacent=True,
                    )
                    if owned and hydrator is not None:
                        extent_count, effective_backend = _native_hydrate_owned(
                            hydrator,
                            path,
                            extents,
                            backend=backend,
                            chunk_bytes=chunk_bytes,
                            queue_depth=queue_depth,
                            cuda_device=device,
                        )
                        local_read_extents += extent_count
                        effective_backends.add(effective_backend)
                    elif owned:
                        _torch_load_owned(
                            source,
                            batch_descriptors,
                            tensors,
                            batch_owners,
                            rank,
                        )
                except BaseException as error:
                    load_error = error
                io_seconds += time.perf_counter() - phase
                local_read_bytes += sum(
                    descriptor.length
                    for descriptor, owner in zip(batch_descriptors, batch_owners, strict=True)
                    if owner == rank
                )

                if group is not None:
                    status = torch.tensor(
                        [1 if load_error is not None else 0],
                        dtype=torch.int32,
                        device=device,
                    )
                    torch.distributed.all_reduce(status, group=group)
                    if int(status.item()) != 0:
                        if load_error is not None:
                            raise load_error
                        raise RuntimeError("a peer failed while reading recovery safetensors")
                elif load_error is not None:
                    raise load_error

                phase = time.perf_counter()
                if group is not None:
                    collective_calls += _broadcast_slab_regions(
                        torch,
                        group,
                        staging_slab,
                        tensor_storages,
                        batch_owners,
                        collective_buffer_bytes,
                    )
                collective_seconds += time.perf_counter() - phase

                phase = time.perf_counter()
                verified_sample_bytes += _verify_tensor_samples(
                    torch,
                    path,
                    batch_descriptors,
                    tensor_storages,
                    verify_bytes,
                )
                verification_seconds += time.perf_counter() - phase

                if isolate_consumer_storage:
                    # Isolate backend recovery from the native-I/O/NCCL
                    # allocation. Some quantized loaders launch asynchronous
                    # in-place work and cannot consume transport aliases. One
                    # clone per bounded batch preserves stream ordering without
                    # the allocation overhead of cloning every expert tensor.
                    consumer_slab = staging_slab.clone()
                    consumer_storages = _storage_views(
                        consumer_slab,
                        batch_descriptors,
                        physical_order=True,
                    )
                    consumer_tensors: list[Any] = []
                    for descriptor, storage in zip(
                        batch_descriptors, consumer_storages, strict=True
                    ):
                        dtype = _validate_descriptor_size(torch, descriptor)
                        consumer_tensors.append(storage.view(dtype).view(descriptor.shape))
                else:
                    consumer_slab = staging_slab
                    consumer_storages = tensor_storages
                    consumer_tensors = tensors

                for descriptor, tensor in zip(batch_descriptors, consumer_tensors, strict=True):
                    logical_bytes += descriptor.length
                    tensor_count += 1
                    full_name = prefix + descriptor.name
                    active = RecoverySourceTensor(
                        name=full_name,
                        path=os.path.abspath(path),
                        descriptor=descriptor,
                        pointer=int(tensor.data_ptr()),
                    )
                    token = _ACTIVE_RECOVERY_SOURCE.set(active)
                    try:
                        yield full_name, tensor
                    finally:
                        _ACTIVE_RECOVERY_SOURCE.reset(token)
                # A model loader may use auxiliary CUDA streams without
                # recording the source allocation on PyTorch's caching
                # allocator. Do not recycle the physical slab until all such
                # consumers have completed.
                torch.cuda.synchronize(device)
                del (
                    consumer_tensors,
                    consumer_storages,
                    consumer_slab,
                    tensors,
                    tensor_storages,
                    staging_slab,
                )

        logger.info(
            "ColdSnap recovery loader supplied %d/%d safetensors files (%.2f GiB logical)",
            file_number + 1,
            len(files),
            logical_bytes / 1024**3,
        )

    total_seconds = time.perf_counter() - started
    _last_metrics = RecoveryLoadMetrics(
        files=len(files),
        tensors=tensor_count,
        logical_bytes=logical_bytes,
        local_read_bytes=local_read_bytes,
        local_read_extents=local_read_extents,
        io_seconds=io_seconds,
        collective_seconds=collective_seconds,
        collective_calls=collective_calls,
        verification_seconds=verification_seconds,
        verified_sample_bytes=verified_sample_bytes,
        total_seconds=total_seconds,
        backend=("+".join(sorted(effective_backends)) if effective_backends else backend),
        distributed_world_size=world_size,
        staging_layout=(
            "bounded-transport+consumer-slabs"
            if isolate_consumer_storage
            else "bounded-transport-slab"
        ),
        staging_buffer_bytes=staging_buffer_bytes,
        collective_buffer_bytes=collective_buffer_bytes,
        io_policy=tuple(_io_policy_observations.values()),
    )
    logger.info(
        "ColdSnap recovery loader supplied %d tensors (%.2f GiB logical, "
        "%.2f GiB local reads/%d ranges) through %s in %.3f s (I/O %.3f s, "
        "collectives %.3f s/%d calls, verified %d sample bytes in %.3f s, "
        "world size %d)",
        tensor_count,
        logical_bytes / 1024**3,
        local_read_bytes / 1024**3,
        local_read_extents,
        "+".join(sorted(effective_backends)) if effective_backends else backend,
        total_seconds,
        io_seconds,
        collective_seconds,
        collective_calls,
        verified_sample_bytes,
        verification_seconds,
        world_size,
    )


def _install_worker_wake_hook() -> None:
    from vllm.v1.worker import gpu_worker

    GPUWorker = getattr(gpu_worker, "GPUWorker", None)
    if GPUWorker is None:
        GPUWorker = getattr(gpu_worker, "Worker", None)
    if GPUWorker is None or not callable(getattr(GPUWorker, "wake_up", None)):
        raise RuntimeError("installed vLLM has no supported GPU worker wake contract")

    if getattr(GPUWorker, "_coldsnap_recovery_loader_installed", False):
        return
    original = GPUWorker.wake_up
    original_sleep = getattr(GPUWorker, "sleep", None)
    original_compile = getattr(GPUWorker, "compile_or_warm_up_model", None)

    def initial_materialization_requested() -> bool:
        if os.environ.get(PROCESS_TEMPLATE_RESTORED_ENV) != "1":
            return False
        control = Path(
            os.environ.get(
                MODEL_PAYLOAD_MATERIALIZATION_CONTROL_ENV,
                DEFAULT_MODEL_PAYLOAD_MATERIALIZATION_CONTROL,
            )
        )
        return control.is_file()

    if initial_materialization_requested() and not callable(original_compile):
        raise RuntimeError(
            "n580 model payload materialization requires vLLM's "
            "compile_or_warm_up_model worker contract"
        )

    if callable(original_compile):

        @functools.wraps(original_compile)
        def compile_with_initial_materialization(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_compile(self, *args, **kwargs)
            if not initial_materialization_requested():
                return result
            backend = self._get_sleep_mode_backend()
            materialize = getattr(
                backend, "materialize_initial_recovery_payload", None
            )
            if not callable(materialize):
                raise RuntimeError(
                    "n580 model payload materialization requires a compatible "
                    "ColdSnap disk backend"
                )
            get_model = getattr(self.model_runner, "get_model", None)
            if not callable(get_model):
                raise RuntimeError(
                    "n580 model payload materialization cannot access the live vLLM model"
                )
            model = get_model()
            ranges = _bind_model_weight_layout(model, backend)
            native_ranges = _bind_native_model_payload_layout(model, backend)
            runtime_logger = logging.getLogger(__name__)
            runtime_logger.info(
                "ColdSnap process-template recovery bound %d reload ranges "
                "(%.2f GiB) and %d native semantic ranges (%.2f GiB) for "
                "model payload materialization",
                len(ranges),
                sum(size for _, size in ranges) / 1024**3,
                len(native_ranges),
                sum(size for _, size in native_ranges) / 1024**3,
            )
            receipt = materialize()
            if receipt is not None:
                runtime_logger.info(
                    "ColdSnap process-template model payload materialization %s: %s",
                    receipt.get("state", "completed"),
                    receipt.get("path", "unknown path"),
                )
            return result

        GPUWorker.compile_or_warm_up_model = compile_with_initial_materialization

    def reload_and_restore_derived_buffers(worker: Any) -> None:
        """Reload checkpoint weights while preserving residual runtime state."""
        global _last_direct_replay_metrics

        _last_direct_replay_metrics = None
        model_runner = worker.model_runner
        get_model = getattr(model_runner, "get_model", None)
        if not callable(get_model):
            raise RuntimeError("recovery reload cannot access the live vLLM model")
        model = get_model()
        from vllm.logger import init_logger

        runtime_logger = init_logger("vllm.v1.worker.gpu_worker")
        replay_plans = getattr(worker, RECOVERY_REPLAY_PLANS_ATTR, ())
        with _recovery_reload_storage(
            model,
            prepare_preloaded=bool(replay_plans),
        ) as storage_adapters:
            if storage_adapters:
                runtime_logger.info(
                    "ColdSnap recovery armed %d stable-storage adapters: %s",
                    len(storage_adapters),
                    ", ".join(storage_adapters),
                )
            skip_sources: frozenset[str] = frozenset()
            preloaded_destinations: frozenset[str] = frozenset()
            if replay_plans:
                import torch

                (
                    skip_sources,
                    preloaded_destinations,
                    _last_direct_replay_metrics,
                ) = _execute_adapter_replay(
                    torch,
                    model,
                    replay_plans,
                    _ACTIVE_RECOVERY_ADAPTERS.get(),
                    captured_rank_io_weights=getattr(
                        worker,
                        RECOVERY_IO_WEIGHTS_ATTR,
                        (),
                    ),
                )
                if _last_direct_replay_metrics is not None:
                    runtime_logger.info(
                        "ColdSnap recovery replayed %d checkpoint sources into %d destinations "
                        "(%0.2f GiB rank payload) in %.3f s",
                        len(skip_sources),
                        len(preloaded_destinations),
                        _last_direct_replay_metrics["replayed_bytes"] / 1024**3,
                        _last_direct_replay_metrics["total_seconds"],
                    )
                    runtime_logger.info(
                        "ColdSnap recovery replay transport read %.2f GiB locally "
                        "(%.2f GiB rank-exclusive), broadcast %.2f GiB in %d calls; "
                        "staging %.3f s, status %.3f s in %d calls, I/O %.3f s "
                        "(wait %.3f s, overlap %.3f s, pipeline %d), transport %.3f s, "
                        "copy %.3f s, sync %.3f s, finalize %.3f s",
                        _last_direct_replay_metrics["local_read_bytes"] / 1024**3,
                        _last_direct_replay_metrics["rank_local_bytes"] / 1024**3,
                        _last_direct_replay_metrics["broadcast_bytes"] / 1024**3,
                        _last_direct_replay_metrics["collective_calls"],
                        _last_direct_replay_metrics["staging_seconds"],
                        _last_direct_replay_metrics["status_seconds"],
                        _last_direct_replay_metrics["status_calls"],
                        _last_direct_replay_metrics["io_seconds"],
                        _last_direct_replay_metrics["io_wait_seconds"],
                        _last_direct_replay_metrics["io_overlap_seconds"],
                        _last_direct_replay_metrics["transport_pipeline_depth"],
                        _last_direct_replay_metrics["broadcast_seconds"],
                        _last_direct_replay_metrics["copy_seconds"],
                        _last_direct_replay_metrics["replay_sync_seconds"],
                        _last_direct_replay_metrics["finalizer_seconds"],
                    )
            skip_token = _RECOVERY_SKIP_SOURCE_NAMES.set(skip_sources)
            preloaded_token = _RECOVERY_PRELOADED_DESTINATION_NAMES.set(preloaded_destinations)
            try:
                model_runner.reload_weights()
            finally:
                _RECOVERY_PRELOADED_DESTINATION_NAMES.reset(preloaded_token)
                _RECOVERY_SKIP_SOURCE_NAMES.reset(skip_token)
        if storage_adapters:
            runtime_logger.info(
                "ColdSnap recovery validated %d stable-storage adapters",
                len(storage_adapters),
            )
        captured_buffers = getattr(worker, "_coldsnap_recovery_derived_buffers", None)
        if captured_buffers is None:
            raise RuntimeError("recovery wake has no captured derived-buffer capsule")
        restored_bytes = 0
        torch = None
        if captured_buffers:
            import torch as torch_module

            torch = torch_module
            restored_bytes = _restore_derived_buffers(torch, model, captured_buffers)
            torch.accelerator.synchronize()
        logging.getLogger(__name__).info(
            "ColdSnap recovery restored %d exact derived buffers (%.2f MiB)",
            len(captured_buffers),
            restored_bytes / 1024**2,
        )
        sample_bytes = _model_verify_bytes()
        expected = getattr(worker, "_coldsnap_recovery_model_samples", None)
        if sample_bytes and expected is None:
            raise RuntimeError("recovery model verification has no captured samples")
        if expected is not None:
            if torch is None:
                import torch as torch_module

                torch = torch_module
            mismatches = _compare_model_samples(torch, model, expected, sample_bytes)
            logging.getLogger(__name__).info(
                "ColdSnap recovery model verification compared %d tensors; %d mismatched",
                len(expected),
                len(mismatches),
            )
            if mismatches:
                raise RuntimeError(
                    "recovery model tensor samples changed: "
                    + json.dumps(mismatches, separators=(",", ":"))
                )

    if callable(original_sleep):

        @functools.wraps(original_sleep)
        def sleep_with_recovery_samples(self: Any, level: int = 1) -> Any:
            get_model = getattr(self.model_runner, "get_model", None)
            if not callable(get_model):
                raise RuntimeError("recovery capture cannot access the live vLLM model")
            model = get_model()
            backend = self._get_sleep_mode_backend()
            bind_reload_layout, bind_native_layout = _sleep_layout_binding_policy(backend)
            if bind_reload_layout:
                model_weight_ranges = _bind_model_weight_layout(model, backend)
                logging.getLogger(__name__).info(
                    "ColdSnap recovery bound %d reload-owned weight ranges "
                    "(%.2f GiB) for residual-blob exclusion",
                    len(model_weight_ranges),
                    sum(size for _, size in model_weight_ranges) / 1024**3,
                )
            if bind_native_layout:
                native_ranges = _bind_native_model_payload_layout(model, backend)
                logging.getLogger(__name__).info(
                    "ColdSnap bound %d semantic native model ranges (%.2f GiB)",
                    len(native_ranges),
                    sum(size for _, size in native_ranges) / 1024**3,
                )
            started = time.perf_counter()
            buffers, buffer_bytes = _capture_derived_buffers(model)
            self._coldsnap_recovery_derived_buffers = buffers
            logging.getLogger(__name__).info(
                "ColdSnap recovery captured %d exact derived buffers (%.2f MiB) in %.3f s",
                len(buffers),
                buffer_bytes / 1024**2,
                time.perf_counter() - started,
            )
            sample_bytes = _model_verify_bytes()
            if sample_bytes:
                import torch

                started = time.perf_counter()
                samples = _capture_model_samples(torch, model, sample_bytes)
                self._coldsnap_recovery_model_samples = samples
                logging.getLogger(__name__).info(
                    "ColdSnap recovery captured %d model tensor samples in %.3f s",
                    len(samples),
                    time.perf_counter() - started,
                )
            adapters = _recovery_storage_adapters(model)
            local_plan = _recovery_replay_plan(model, adapters)
            captured_plan = _model_checkpoint_copy_plan(model)
            reload_dependencies = _recovery_reload_dependency_names(model) | frozenset().union(
                *(
                    frozenset(getattr(adapter, "required_normal_reload_destination_names", ()))
                    for adapter in adapters
                )
            )
            adapter_destinations = frozenset().union(
                *(
                    frozenset(getattr(adapter, "direct_replay_destination_names", ()))
                    for adapter in adapters
                )
            )
            selected_destinations = frozenset(copy.destination_name for copy in local_plan.copies)
            from vllm.logger import init_logger

            init_logger("vllm.v1.worker.gpu_worker").info(
                "ColdSnap replay selection found %d adapters requesting %d destinations "
                "with %d normal-reload dependencies; "
                "captured=%d copies/%d unsupported; eligible=%d copies across %d "
                "destinations/%d rejected",
                len(adapters),
                len(adapter_destinations),
                len(reload_dependencies),
                len(captured_plan.copies) if captured_plan is not None else 0,
                (len(captured_plan.unsupported_destinations) if captured_plan is not None else 0),
                len(local_plan.copies),
                len(selected_destinations),
                len(local_plan.unsupported_destinations),
            )
            replay_plans = getattr(self, RECOVERY_REPLAY_PLANS_ATTR, ())
            recovery_io_weights = getattr(self, RECOVERY_IO_WEIGHTS_ATTR, ())
            local_io_rate = _model_initial_recovery_io_rate(model)
            if replay_plans:
                init_logger("vllm.v1.worker.gpu_worker").info(
                    "ColdSnap reused %d immutable capture-time recovery copy plans",
                    len(replay_plans),
                )
            else:
                if local_plan.copies:
                    import torch

                    replay_plans = _gather_recovery_copy_plans(torch, local_plan)
                else:
                    replay_plans = ()
                setattr(self, RECOVERY_REPLAY_PLANS_ATTR, replay_plans)
                recovery_io_weights = (
                    _gather_recovery_io_weights(torch, local_io_rate) if replay_plans else ()
                )
                setattr(self, RECOVERY_IO_WEIGHTS_ATTR, recovery_io_weights)
            if replay_plans:
                import torch

                capture_format = os.environ.get(CAPTURE_LOAD_FORMAT_ENV, "").strip().lower()
                if capture_format:
                    prime_seconds = _prime_recovery_device_group(torch)
                    init_logger("vllm.v1.worker.gpu_worker").info(
                        "ColdSnap primed the recovery TP device group for portable restore in %.3f s",
                        prime_seconds,
                    )
                unique_sources = {copy.source_key for plan in replay_plans for copy in plan.copies}
                logging.getLogger(__name__).info(
                    "ColdSnap recovery captured %d rank copy plans for %d "
                    "checkpoint sources (%.2f GiB source)",
                    len(replay_plans),
                    len(unique_sources),
                    sum(int(key[-1]) for key in unique_sources) / 1024**3,
                )
                logging.getLogger(__name__).info(
                    "ColdSnap recovery captured initial-load I/O weights %s (local rate %s GiB/s)",
                    [round(weight / 1024**3, 3) for weight in recovery_io_weights],
                    (round(local_io_rate / 1024**3, 3) if local_io_rate is not None else None),
                )
                packed_profiles: list[tuple[int, int]] = []
                for plan in replay_plans:
                    extent_count = 0
                    packed_bytes = 0
                    for copy in plan.copies:
                        count, contiguous_elements, element_size = _packed_copy_extent_geometry(
                            torch, copy
                        )
                        extent_count += count
                        packed_bytes += count * contiguous_elements * element_size
                    packed_profiles.append((extent_count, packed_bytes))
                max_extents = int(
                    os.environ.get(
                        LOADER_LOCAL_PACKED_MAX_EXTENTS_ENV,
                        str(DEFAULT_LOCAL_PACKED_MAX_EXTENTS),
                    )
                )
                min_extent_bytes = int(
                    os.environ.get(
                        LOADER_LOCAL_PACKED_MIN_EXTENT_BYTES_ENV,
                        str(DEFAULT_LOCAL_PACKED_MIN_EXTENT_BYTES),
                    )
                )
                logging.getLogger(__name__).info(
                    "ColdSnap local packed replay profile extents=%s average-bytes=%s; "
                    "qualified=%s (limits max-extents=%d min-average-bytes=%d)",
                    [count for count, _packed_bytes in packed_profiles],
                    [
                        packed_bytes // count if count else 0
                        for count, packed_bytes in packed_profiles
                    ],
                    _rank_local_packed_replay_plans_qualified(
                        torch,
                        replay_plans,
                        max_extents=max_extents,
                        min_extent_bytes=min_extent_bytes,
                    ),
                    max_extents,
                    min_extent_bytes,
                )
            capture_format = os.environ.get(CAPTURE_LOAD_FORMAT_ENV, "").strip().lower()
            live_value = getattr(self.model_runner.load_config, "load_format", "")
            live_format = str(getattr(live_value, "value", live_value)).lower()
            if capture_format and live_format != LOAD_FORMAT:
                if live_format != capture_format:
                    raise RuntimeError(
                        "ColdSnap capture load format changed before snapshot: "
                        f"selected={capture_format!r} live={live_format!r}"
                    )
                self.model_runner.load_config.load_format = LOAD_FORMAT
                runner_config = getattr(self.model_runner, "vllm_config", None)
                if runner_config is not None:
                    runner_config.load_config = self.model_runner.load_config
                worker_config = getattr(self, "vllm_config", None)
                if worker_config is not None:
                    worker_config.load_config = self.model_runner.load_config
                init_logger("vllm.v1.worker.gpu_worker").info(
                    "ColdSnap capture loaded weights with %s and armed %s for restore",
                    capture_format,
                    LOAD_FORMAT,
                )
            return original_sleep(self, level)

        GPUWorker.sleep = sleep_with_recovery_samples

    @functools.wraps(original)
    def wake_up_with_recovery(self: Any, tags: list[str] | None = None) -> Any:
        backend = self._get_sleep_mode_backend()
        uses_recovery = bool(getattr(backend, "uses_model_weight_recovery", False))
        wake_weights = tags is None or "weights" in tags
        if not uses_recovery or not wake_weights:
            return original(self, tags)
        load_format = str(getattr(self.model_runner.load_config, "load_format", ""))
        if load_format != LOAD_FORMAT:
            raise RuntimeError(
                f"safetensors weight recovery requires --load-format {LOAD_FORMAT}; "
                f"restored model uses {load_format!r}"
            )
        backend.set_weight_recovery_callback(
            functools.partial(reload_and_restore_derived_buffers, self)
        )
        try:
            return original(self, tags)
        finally:
            backend.set_weight_recovery_callback(None)

    GPUWorker.wake_up = wake_up_with_recovery
    GPUWorker._coldsnap_recovery_loader_installed = True


def install_model_payload_capture_hook() -> None:
    """Bind model-owned ranges when capture exports a payload without replay.

    The n580 process-template path loads through vLLM's selected normal loader,
    so it must not install or arm the n610 recovery loader. It still needs the
    semantic model ranges in order to emit the same model-only content object.
    """
    if os.environ.get("COLDSNAP_EXPORT_MODEL_PAYLOAD") != "1" or recovery_weights_enabled():
        return
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    # Preserve the configured fast loader while observing its actual
    # checkpoint-to-destination writes. Without this metadata, normal-loader
    # captures would conservatively classify ordinary parameters as residual
    # state and could not converge on the n610 model-payload identity.
    _install_capture_loader_observer(DefaultModelLoader)
    from vllm.v1.worker import gpu_worker

    GPUWorker = getattr(gpu_worker, "GPUWorker", None)
    if GPUWorker is None:
        GPUWorker = getattr(gpu_worker, "Worker", None)
    if GPUWorker is None or not callable(getattr(GPUWorker, "sleep", None)):
        raise RuntimeError("installed vLLM has no supported GPU worker sleep contract")
    if getattr(GPUWorker, "_coldsnap_model_payload_capture_installed", False):
        return
    original_sleep = GPUWorker.sleep

    @functools.wraps(original_sleep)
    def sleep_with_model_payload_ranges(self: Any, level: int = 1) -> Any:
        get_model = getattr(self.model_runner, "get_model", None)
        if not callable(get_model):
            raise RuntimeError("model payload capture cannot access the live vLLM model")
        backend = self._get_sleep_mode_backend()
        if bool(getattr(backend, "exports_model_payload", False)):
            model = get_model()
            ranges = _bind_model_weight_layout(model, backend)
            native_ranges = _bind_native_model_payload_layout(model, backend)
            logging.getLogger(__name__).info(
                "ColdSnap bound %d reload ranges (%.2f GiB) and %d native semantic "
                "ranges (%.2f GiB) for shared payload export",
                len(ranges),
                sum(size for _, size in ranges) / 1024**3,
                len(native_ranges),
                sum(size for _, size in native_ranges) / 1024**3,
            )
        return original_sleep(self, level)

    GPUWorker.sleep = sleep_with_model_payload_ranges
    GPUWorker._coldsnap_model_payload_capture_installed = True


def _install_recovery_hybrid_draft_bridge() -> None:
    """Keep speculative drafts outside target-only ColdSnap hydration.

    The qualified DS4 image has a resolver that selects lazy safetensors for an
    MTP draft when the target uses InstantTensor. ColdSnap is an equivalent
    target-only format at model-construction time, so delegate through that
    resolver with an InstantTensor alias and translate only an unchanged target
    result back to ColdSnap. Any draft-specific result remains untouched.

    Other images need not carry that historical resolver. For those, wrap
    vLLM's public ``get_model`` helper and select lazy safetensors whenever it
    constructs the configured speculative draft. This is intentionally keyed
    to vLLM's model identity, not a B12x or model-name special case.
    """
    import importlib

    model_loader = importlib.import_module("vllm.model_executor.model_loader")
    original = getattr(model_loader, "_instanttensor_draft_load_config", None)
    if not callable(original):
        original_get_model = getattr(model_loader, "get_model", None)
        if not callable(original_get_model) or getattr(
            original_get_model, "_coldsnap_recovery_bridge_installed", False
        ):
            return

        @functools.wraps(original_get_model)
        def get_model_with_recovery_draft_split(
            *,
            vllm_config: Any,
            model_config: Any | None = None,
            prefix: str = "",
            load_config: Any | None = None,
        ) -> Any:
            if model_config is None:
                model_config = vllm_config.model_config
            effective = load_config or vllm_config.load_config
            load_format = getattr(effective.load_format, "value", effective.load_format)
            speculative_config = getattr(vllm_config, "speculative_config", None)
            draft_model_config = getattr(speculative_config, "draft_model_config", None)
            if (
                str(load_format).lower() == LOAD_FORMAT
                and draft_model_config is not None
                and model_config is draft_model_config
            ):
                from vllm.config import replace

                effective = replace(
                    effective,
                    load_format="safetensors",
                    safetensors_load_strategy="lazy",
                )
            return original_get_model(
                vllm_config=vllm_config,
                model_config=model_config,
                prefix=prefix,
                load_config=effective,
            )

        get_model_with_recovery_draft_split._coldsnap_recovery_bridge_installed = True
        model_loader.get_model = get_model_with_recovery_draft_split
        return
    if getattr(original, "_coldsnap_recovery_bridge_installed", False):
        return

    @functools.wraps(original)
    def resolve_recovery_draft_load_config(
        vllm_config: Any,
        model_config: Any,
        load_config: Any | None,
    ) -> Any:
        effective = load_config or vllm_config.load_config
        load_format = getattr(effective.load_format, "value", effective.load_format)
        if str(load_format).lower() != LOAD_FORMAT:
            return original(vllm_config, model_config, load_config)

        from vllm.config import replace

        aliased = replace(effective, load_format="instanttensor")
        resolved = original(vllm_config, model_config, aliased)
        resolved_format = getattr(resolved.load_format, "value", resolved.load_format)
        if str(resolved_format).lower() == "instanttensor":
            return effective
        return resolved

    resolve_recovery_draft_load_config._coldsnap_recovery_bridge_installed = True
    model_loader._instanttensor_draft_load_config = resolve_recovery_draft_load_config


def _loader_format(loader: Any) -> str:
    value = getattr(getattr(loader, "load_config", None), "load_format", "")
    return str(getattr(value, "value", value)).lower()


def _capture_observer_enabled(loader: Any) -> bool:
    configured = os.environ.get(CAPTURE_LOAD_FORMAT_ENV, "").strip().lower()
    return bool(configured) and configured != LOAD_FORMAT and _loader_format(loader) == configured


def _capture_source_descriptors(
    loader: Any, source: Any
) -> tuple[Any, dict[str, tuple[str, SafetensorDescriptor]]]:
    """Resolve immutable file ranges for a fast capture iterator.

    InstantTensor and FastSafeTensors still yield ordinary CUDA tensors to
    vLLM.  Parsing the already-local safetensors headers lets the common copy
    observer attach the same immutable source identity used by ColdSnap's
    restore iterator without replacing the fast I/O path.
    """
    prepared = prepare_synthetic_weight_source(loader, source)
    indexed = _index_map(prepared.folder)
    descriptors: dict[str, tuple[str, SafetensorDescriptor]] = {}
    for filename in prepared.files:
        path = Path(filename)
        for descriptor in _descriptors(
            path,
            indexed_tensor_files=indexed,
            weight_name_prefixes=prepared.weight_name_prefixes,
        ):
            name = prepared.prefix + descriptor.name
            if name in descriptors:
                raise RuntimeError(f"duplicate capture safetensors source {name!r}")
            descriptors[name] = (os.path.abspath(path), descriptor)
    if not descriptors:
        raise RuntimeError("capture safetensors observer found no source tensors")
    return prepared, descriptors


def _observed_capture_iterator(
    loader: Any, source: Any, iterator: Any
) -> Iterator[tuple[str, Any]]:
    """Decorate a fast vLLM source iterator with recovery-copy identities."""
    prepared, descriptors = _capture_source_descriptors(loader, source)
    started = time.perf_counter()
    logical_bytes = 0
    tensors = 0
    bootstrap_tensors: list[dict[str, Any]] = []
    completed = False
    try:
        for name, tensor in iterator:
            selected = descriptors.get(name)
            if selected is None:
                raise RuntimeError(
                    f"capture loader yielded safetensors source {name!r} without immutable metadata"
                )
            path, descriptor = selected
            actual_bytes = int(tensor.numel()) * int(tensor.element_size())
            if tuple(tensor.shape) != descriptor.shape or actual_bytes != descriptor.length:
                raise RuntimeError(f"capture loader tensor metadata changed for {name!r} in {path}")
            active = RecoverySourceTensor(
                name=name,
                path=path,
                descriptor=descriptor,
                pointer=int(tensor.data_ptr()),
            )
            token = _ACTIVE_RECOVERY_SOURCE.set(active)
            try:
                yield name, tensor
            finally:
                _ACTIVE_RECOVERY_SOURCE.reset(token)
            logical_bytes += descriptor.length
            tensors += 1
            bootstrap_tensors.append(
                {
                    "name": name,
                    "dtype": descriptor.dtype_name,
                    "shape": list(descriptor.shape),
                    "length": descriptor.length,
                }
            )
        completed = True
    finally:
        if completed:
            from coldsnap_synthetic_loader import record_native_bootstrap_source

            record_native_bootstrap_source(source, prepared.prefix, bootstrap_tensors)
        seconds = time.perf_counter() - started
        try:
            import torch

            _group, _rank, world_size = _distributed_context(torch)
        except (AssertionError, ImportError, RuntimeError):
            world_size = 1
        local_bytes = logical_bytes // max(world_size, 1)
        metrics = RecoveryLoadMetrics(
            files=len(prepared.files),
            tensors=tensors,
            logical_bytes=logical_bytes,
            local_read_bytes=local_bytes,
            local_read_extents=0,
            io_seconds=seconds,
            collective_seconds=0.0,
            collective_calls=0,
            verification_seconds=0.0,
            verified_sample_bytes=0,
            total_seconds=seconds,
            backend="capture-observer+" + _loader_format(loader),
            distributed_world_size=world_size,
            staging_layout="vllm-fast-loader",
            staging_buffer_bytes=0,
            collective_buffer_bytes=0,
        )
        collected = getattr(loader, "_coldsnap_capture_source_metrics", None)
        if isinstance(collected, list):
            collected.append(metrics)


def _combine_capture_metrics(metrics: Sequence[RecoveryLoadMetrics]) -> RecoveryLoadMetrics | None:
    if not metrics:
        return None
    return RecoveryLoadMetrics(
        files=sum(item.files for item in metrics),
        tensors=sum(item.tensors for item in metrics),
        logical_bytes=sum(item.logical_bytes for item in metrics),
        local_read_bytes=sum(item.local_read_bytes for item in metrics),
        local_read_extents=sum(item.local_read_extents for item in metrics),
        io_seconds=sum(item.io_seconds for item in metrics),
        collective_seconds=sum(item.collective_seconds for item in metrics),
        collective_calls=sum(item.collective_calls for item in metrics),
        verification_seconds=sum(item.verification_seconds for item in metrics),
        verified_sample_bytes=sum(item.verified_sample_bytes for item in metrics),
        total_seconds=sum(item.total_seconds for item in metrics),
        backend="+".join(dict.fromkeys(item.backend for item in metrics)),
        distributed_world_size=max(item.distributed_world_size for item in metrics),
        staging_layout="vllm-fast-loader",
        staging_buffer_bytes=max(item.staging_buffer_bytes for item in metrics),
        collective_buffer_bytes=max(item.collective_buffer_bytes for item in metrics),
    )


def _load_weights_with_recovery_observer(
    loader: Any,
    model: Any,
    load_weights: Any,
) -> Any:
    """Record the model loader's exact source-to-destination copy operations."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    destination_names: set[str] = set()
    replay_copies: list[RecoveryCopy] = []
    unsupported_destinations: set[str] = set()
    rejection_reasons: Counter[str] = Counter()
    wrapped: list[tuple[Any, bool, Any]] = []
    destinations = list(model.named_parameters()) + list(model.named_buffers())
    for name, destination in destinations:
        had_loader = hasattr(destination, "weight_loader")
        destination_loader = getattr(destination, "weight_loader", default_weight_loader)

        def recording_loader(
            *args: Any,
            _name: str = name,
            _loader: Any = destination_loader,
            _destination: Any = destination,
            **kwargs: Any,
        ) -> Any:
            destination_names.add(_name)
            source = _ACTIVE_RECOVERY_SOURCE.get()
            if source is None:
                return _loader(*args, **kwargs)
            result, copies, unsupported, reasons = _observe_weight_loader_copy(
                source,
                _name,
                _destination,
                _loader,
                args,
                kwargs,
            )
            replay_copies.extend(copies)
            rejection_reasons.update(reasons)
            if unsupported:
                unsupported_destinations.add(_name)
            return result

        recording_loader.__name__ = getattr(
            destination_loader, "__name__", "coldsnap_recording_weight_loader"
        )
        recording_loader.__wrapped__ = destination_loader
        destination.weight_loader = recording_loader
        wrapped.append((destination, had_loader, destination_loader))
    try:
        loader._coldsnap_capture_source_metrics = []
        result = load_weights()
        metrics = _combine_capture_metrics(loader._coldsnap_capture_source_metrics)
        if metrics is None and _last_metrics is not None:
            metrics = _last_metrics
        if metrics is not None:
            setattr(model, INITIAL_LOAD_METRICS_ATTR, metrics)
            _publish_process_template_load_metrics(metrics)
        return result
    finally:
        setattr(
            model,
            CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR,
            frozenset(destination_names),
        )
        setattr(
            model,
            CHECKPOINT_COPY_PLAN_ATTR,
            RecoveryCopyPlan(
                copies=tuple(replay_copies),
                unsupported_destinations=frozenset(unsupported_destinations),
            ),
        )
        from vllm.logger import init_logger

        init_logger("vllm.model_executor.model_loader.default_loader").info(
            "ColdSnap observed %d exact copies across %d checkpoint "
            "destinations; %d destinations unsupported; rejections=%s",
            len(replay_copies),
            len(destination_names),
            len(unsupported_destinations),
            dict(rejection_reasons.most_common(6)),
        )
        for destination, had_loader, destination_loader in reversed(wrapped):
            if had_loader:
                destination.weight_loader = destination_loader
            else:
                delattr(destination, "weight_loader")
        if hasattr(loader, "_coldsnap_capture_source_metrics"):
            del loader._coldsnap_capture_source_metrics


def _install_capture_loader_observer(default_loader: type[Any]) -> None:
    """Observe fast capture loaders without changing their I/O implementation."""
    if getattr(default_loader, "_coldsnap_capture_observer_installed", False):
        return
    original_iterator = default_loader._get_weights_iterator
    original_load_weights = default_loader.load_weights

    @functools.wraps(original_iterator)
    def observed_iterator(self: Any, source: Any) -> Any:
        iterator = original_iterator(self, source)
        if not _capture_observer_enabled(self):
            return iterator
        return _observed_capture_iterator(self, source, iterator)

    @functools.wraps(original_load_weights)
    def observed_load_weights(self: Any, model: Any, model_config: Any) -> Any:
        if not _capture_observer_enabled(self):
            return original_load_weights(self, model, model_config)
        return _load_weights_with_recovery_observer(
            self,
            model,
            lambda: original_load_weights(self, model, model_config),
        )

    default_loader._get_weights_iterator = observed_iterator
    default_loader.load_weights = observed_load_weights
    default_loader._coldsnap_capture_observer_installed = True


def install_recovery_aware_loader() -> None:
    """Register the custom format and live-worker bridge when explicitly enabled."""
    if not recovery_weights_enabled():
        return
    configured_format = os.environ.get("COLDSNAP_LOAD_FORMAT")
    if configured_format not in {None, "", LOAD_FORMAT}:
        raise RuntimeError(f"safetensors recovery requires COLDSNAP_LOAD_FORMAT={LOAD_FORMAT}")
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    if getattr(DefaultModelLoader, "_coldsnap_recovery_loader_installed", False):
        return

    class ColdSnapRecoveryModelLoader(DefaultModelLoader):
        """Default vLLM loader with a recovery-aware safetensors source."""

        def load_weights(self: Any, model: Any, model_config: Any) -> Any:
            return _load_weights_with_recovery_observer(
                self,
                model,
                lambda: super(ColdSnapRecoveryModelLoader, self).load_weights(model, model_config),
            )

        def _get_weights_iterator(self: Any, source: Any) -> Any:
            from vllm.logger import init_logger

            loader_logger = init_logger(__name__)
            loader_logger.info(
                "ColdSnap recovery loader preparing source %s",
                source.model_or_path,
            )
            # Qualification-only control: retain ColdSnap's zero-blob recovery
            # and stable-VA remap while substituting vLLM's known-good source
            # iterator. This isolates loader tensor semantics from recovery.
            if os.environ.get(LOADER_BACKEND_ENV) == "instanttensor":
                original_format = self.load_config.load_format
                self.load_config.load_format = "instanttensor"
                try:
                    return super()._get_weights_iterator(source)
                finally:
                    self.load_config.load_format = original_format
            original_format = self.load_config.load_format
            self.load_config.load_format = "safetensors"
            try:
                prepared = prepare_synthetic_weight_source(self, source)
            finally:
                self.load_config.load_format = original_format
            loader_logger.info(
                "ColdSnap recovery loader prepared %d safetensors files",
                len(prepared.files),
            )
            if getattr(self, "counter_before_loading_weights", 0.0) == 0.0:
                self.counter_before_loading_weights = time.perf_counter()
            return recovery_weights_iterator(
                prepared.files,
                folder=prepared.folder,
                prefix=prepared.prefix,
                weight_name_prefixes=prepared.weight_name_prefixes,
                local_expert_ids=getattr(self, "local_expert_ids", None),
            )

    ColdSnapRecoveryModelLoader.__module__ = __name__
    _install_capture_loader_observer(DefaultModelLoader)
    if not register_model_loader(LOAD_FORMAT, ColdSnapRecoveryModelLoader):
        raise RuntimeError("recovery-aware loading requires vLLM's public model-loader registry")
    DefaultModelLoader._coldsnap_recovery_loader_installed = True
    _install_recovery_hybrid_draft_bridge()
    _install_worker_wake_hook()

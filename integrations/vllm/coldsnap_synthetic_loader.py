# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Capture-time loader bridge for native bootstrap metadata."""

from __future__ import annotations

import base64
import binascii
import json
import os
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from coldsnap_materializer import NeutralTensorMaterializer
from coldsnap_vllm import register_model_loader


_TORCH_DTYPE_ATTRIBUTES = {
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

NATIVE_BOOTSTRAP_INDEX_NAME = "native-bootstrap-index.json"
NATIVE_BOOTSTRAP_INDEX_FORMAT = 3
NATIVE_BOOTSTRAP_INDEX_KIND = "coldsnap-native-bootstrap-index"
_NATIVE_BOOTSTRAP_INDEX_LOCK = threading.Lock()
_BOOTSTRAP_EXACT_DTYPES = frozenset({"BOOL", "I16", "I32", "I64", "U16", "U32", "U64"})
_BOOTSTRAP_EXACT_MAX_BYTES = 64 * 1024


def native_bootstrap_tensor_metadata(name: str, descriptor: Any, tensor: Any) -> dict[str, Any]:
    """Keep bounded integer geometry values that model loaders validate.

    Packed byte weights and floating-point model weights remain payload-free.
    These small values are captured before the consumer can mutate its source.
    """
    entry = {
        "name": name, "dtype": descriptor.dtype_name,
        "shape": list(descriptor.shape), "length": descriptor.length,
    }
    if (
        descriptor.dtype_name in _BOOTSTRAP_EXACT_DTYPES
        and 0 < descriptor.length <= _BOOTSTRAP_EXACT_MAX_BYTES
        and not bool(getattr(tensor, "is_meta", False))
    ):
        import torch

        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        if len(raw) != descriptor.length:
            raise RuntimeError("native bootstrap metadata byte length changed")
        entry["bootstrap_data"] = base64.b64encode(raw).decode("ascii")
    return entry


def _exact_bootstrap_tensor(torch_module: Any, item: dict[str, Any], dtype: Any) -> Any:
    encoded = item["bootstrap_data"]
    if (
        item["dtype"] not in _BOOTSTRAP_EXACT_DTYPES
        or not 0 < item["length"] <= _BOOTSTRAP_EXACT_MAX_BYTES
        or not isinstance(encoded, str)
        or len(encoded) > ((_BOOTSTRAP_EXACT_MAX_BYTES + 2) // 3) * 4
    ):
        raise RuntimeError("native bootstrap exact metadata is invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RuntimeError("native bootstrap exact metadata encoding is invalid") from error
    if len(raw) != item["length"]:
        raise RuntimeError("native bootstrap exact metadata byte length changed")
    return torch_module.frombuffer(bytearray(raw), dtype=dtype).reshape(tuple(item["shape"]))


def _source_identity(source: Any, prefix: str | None = None) -> dict[str, Any]:
    raw_prefixes = getattr(source, "weight_name_prefixes", None)
    return {
        "model_or_path": str(getattr(source, "model_or_path", "")),
        "revision": str(getattr(source, "revision", "") or ""),
        "subfolder": str(getattr(source, "subfolder", "") or ""),
        "prefix": str(getattr(source, "prefix", "") if prefix is None else prefix),
        "weight_name_prefixes": (
            [str(value) for value in raw_prefixes]
            if raw_prefixes is not None
            else []
        ),
    }


def _native_bootstrap_root() -> Path | None:
    root = os.environ.get("COLDSNAP_DISK_SLEEP_DIR")
    if not root:
        return None
    from coldsnap_disk_backend import _snapshot_directory

    return _snapshot_directory(Path(root))


def record_native_bootstrap_source(
    source: Any,
    prefix: str,
    tensors: list[dict[str, Any]],
) -> Path | None:
    """Persist compact iterator metadata beside an n580 native manifest."""
    if (
        os.environ.get("COLDSNAP_EXPORT_MODEL_PAYLOAD") != "1"
        or os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORED") == "1"
    ):
        return None
    root = _native_bootstrap_root()
    if root is None or not tensors:
        return None
    from coldsnap_disk_backend import _atomic_json, _worker_id

    root.mkdir(parents=True, exist_ok=True)
    path = root / NATIVE_BOOTSTRAP_INDEX_NAME
    identity = _source_identity(source, prefix)
    value: dict[str, Any] = {
        "format": NATIVE_BOOTSTRAP_INDEX_FORMAT,
        "kind": NATIVE_BOOTSTRAP_INDEX_KIND,
        "capture_id": os.environ.get("COLDSNAP_CAPTURE_ID", ""),
        "worker_id": _worker_id(),
        "sources": [],
    }
    with _NATIVE_BOOTSTRAP_INDEX_LOCK:
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(existing, dict)
                and existing.get("format") == NATIVE_BOOTSTRAP_INDEX_FORMAT
                and existing.get("kind") == NATIVE_BOOTSTRAP_INDEX_KIND
                and existing.get("capture_id") == value["capture_id"]
                and existing.get("worker_id") == value["worker_id"]
                and isinstance(existing.get("sources"), list)
            ):
                value = existing
            else:
                raise RuntimeError(
                    "native bootstrap index identity changed during capture"
                )
        record = {"identity": identity, "tensors": tensors}
        retained = [
            item
            for item in value["sources"]
            if isinstance(item, dict) and item.get("identity") != identity
        ]
        retained.append(record)
        retained.sort(key=lambda item: json.dumps(item["identity"], sort_keys=True))
        value["sources"] = retained
        _atomic_json(path, value)
    return path


def _native_bootstrap_requested() -> bool:
    root = _native_bootstrap_root()
    if root is None:
        return False
    try:
        provider = (root / "activation-provider").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False
    if provider != "native":
        return False
    for name in (
        "native-manifest.json",
        "model-weights.pack",
        NATIVE_BOOTSTRAP_INDEX_NAME,
    ):
        if not (root / name).is_file():
            raise RuntimeError(f"n580 native bootstrap is missing {name}; recapture with native bootstrap support")
    return True


def _native_bootstrap_weights(source: Any) -> Generator[tuple[str, Any], None, None]:
    root = _native_bootstrap_root()
    if root is None:
        raise RuntimeError("n580 native bootstrap has no hydration root")
    value = json.loads((root / NATIVE_BOOTSTRAP_INDEX_NAME).read_text(encoding="utf-8"))
    from coldsnap_disk_backend import _worker_id

    if (
        not isinstance(value, dict)
        or value.get("format") not in {1, 2, NATIVE_BOOTSTRAP_INDEX_FORMAT}
        or value.get("kind") != NATIVE_BOOTSTRAP_INDEX_KIND
        or value.get("capture_id") != os.environ.get("COLDSNAP_CAPTURE_ID", "")
        or value.get("worker_id") != _worker_id()
        or not isinstance(value.get("sources"), list)
    ):
        raise RuntimeError("n580 native bootstrap index identity is invalid")
    identity = _source_identity(source)
    matches = [
        item
        for item in value["sources"]
        if isinstance(item, dict) and item.get("identity") == identity
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("tensors"), list):
        raise RuntimeError("n580 native bootstrap source is absent or ambiguous")
    materializer = NeutralTensorMaterializer()
    import torch
    from vllm.logger import init_logger

    started = time.monotonic()
    logical_bytes = 0
    exact_source_bytes = 0
    for item in matches[0]["tensors"]:
        if not isinstance(item, dict):
            raise RuntimeError("n580 native bootstrap tensor metadata is invalid")
        name = item.get("name")
        dtype_name = item.get("dtype")
        shape = item.get("shape")
        length = item.get("length")
        dtype_attribute = _TORCH_DTYPE_ATTRIBUTES.get(str(dtype_name))
        dtype = getattr(torch, dtype_attribute, None) if dtype_attribute else None
        if (
            not isinstance(name, str)
            or not name
            or dtype is None
            or not isinstance(shape, list)
            or not all(type(value) is int and value >= 0 for value in shape)
            or type(length) is not int
            or length <= 0
        ):
            raise RuntimeError("n580 native bootstrap tensor metadata is invalid")
        from coldsnap_recovery_loader import (
            SafetensorDescriptor, _file_source_tensor, _validate_descriptor_size,
        )
        file_source = item.get("file_source")
        prefix = str(getattr(source, "prefix", ""))
        if not name.startswith(prefix):
            raise RuntimeError("n580 native bootstrap tensor prefix changed")
        file_filter = getattr(source, "file_weight_filter", None)
        expects_file = file_filter is not None and file_filter(name[len(prefix):])
        if expects_file != (file_source is not None):
            raise RuntimeError("native bootstrap file-backed tensor contract changed; recapture required")
        descriptor = SafetensorDescriptor(name, dtype_name, tuple(shape), 0, length)
        _validate_descriptor_size(torch, descriptor)
        logical_bytes += length
        if file_source is not None:
            if "bootstrap_data" in item:
                raise RuntimeError("file-backed native bootstrap cannot embed tensor data")
            if (
                not isinstance(file_source, dict)
                or not isinstance(file_source.get("path"), str)
                or not Path(file_source["path"]).is_absolute()
                or type(file_source.get("offset")) is not int
                or file_source["offset"] < 8
            ):
                raise RuntimeError("native bootstrap file range is invalid")
            path = Path(file_source["path"])
            if file_source["offset"] + length > path.stat().st_size:
                raise RuntimeError("native bootstrap file range exceeds checkpoint file")
            descriptor = SafetensorDescriptor(name, dtype_name, tuple(shape), file_source["offset"], length)
            yield name, _file_source_tensor(path, descriptor)
        elif "bootstrap_data" in item:
            exact_source_bytes += length
            yield name, _exact_bootstrap_tensor(torch, item, dtype)
        else:
            yield name, materializer.tensor(torch, name, tuple(shape), dtype)
    init_logger("vllm.model_executor.model_loader.default_loader").info(
        "Native bootstrap supplied %d tensors (%.2f GiB logical, %.2f MiB physical) in %.3f s",
        len(matches[0]["tensors"]),
        logical_bytes / 1024**3,
        (materializer.stats.physical_source_bytes + exact_source_bytes) / 1024**2,
        time.monotonic() - started,
    )


def install_synthetic_weight_loader() -> None:
    process_template = bool(os.environ.get("COLDSNAP_PROCESS_TEMPLATE_PHASE"))
    if not process_template:
        return

    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    if getattr(DefaultModelLoader, "_coldsnap_synthetic_installed", False):
        return

    class ColdSnapSyntheticModelLoader(DefaultModelLoader):
        """Registered loader inheriting the activation-aware iterator bridge."""

        def _get_weights_iterator(self, source: Any) -> Any:
            if _native_bootstrap_requested():
                return _native_bootstrap_weights(source)
            return super()._get_weights_iterator(source)

    ColdSnapSyntheticModelLoader.__module__ = __name__
    formats = {
        os.environ.get("COLDSNAP_CAPTURE_LOAD_FORMAT", "").strip().lower()
    }
    formats.discard("")
    if not formats:
        raise RuntimeError("process-template capture requires COLDSNAP_CAPTURE_LOAD_FORMAT")
    for load_format in sorted(formats):
        register_model_loader(load_format, ColdSnapSyntheticModelLoader)
    DefaultModelLoader._coldsnap_synthetic_installed = True

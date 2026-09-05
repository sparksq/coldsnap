# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Capture-time loader bridge for native bootstrap metadata."""

from __future__ import annotations

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
}

NATIVE_BOOTSTRAP_INDEX_NAME = "native-bootstrap-index.json"
NATIVE_BOOTSTRAP_INDEX_FORMAT = 1
NATIVE_BOOTSTRAP_INDEX_KIND = "coldsnap-native-bootstrap-index"
_NATIVE_BOOTSTRAP_INDEX_LOCK = threading.Lock()


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
        not os.environ.get("COLDSNAP_PROCESS_TEMPLATE_PHASE")
        or os.environ.get("COLDSNAP_EXPORT_MODEL_PAYLOAD") != "1"
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
            raise RuntimeError(f"n580 native bootstrap is missing {name}")
    return True


def _native_bootstrap_weights(source: Any) -> Generator[tuple[str, Any], None, None]:
    root = _native_bootstrap_root()
    if root is None:
        raise RuntimeError("n580 native bootstrap has no hydration root")
    value = json.loads((root / NATIVE_BOOTSTRAP_INDEX_NAME).read_text(encoding="utf-8"))
    from coldsnap_disk_backend import _worker_id

    if (
        not isinstance(value, dict)
        or value.get("format") != NATIVE_BOOTSTRAP_INDEX_FORMAT
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
        logical_bytes += length
        yield name, materializer.tensor(torch, name, tuple(shape), dtype)
    init_logger(__name__).info(
        "Native bootstrap supplied %d tensors (%.2f GiB logical, %.2f MiB physical) in %.3f s",
        materializer.stats.tensors,
        logical_bytes / 1024**3,
        materializer.stats.physical_source_bytes / 1024**2,
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

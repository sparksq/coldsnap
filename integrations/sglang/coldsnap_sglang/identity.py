# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Compatibility identity for controller-managed SGLang process snapshots."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

from coldsnap_core.artifact import ArtifactError, SemanticTensorArtifact
from coldsnap_core.topology import (
    execution_graph,
    global_rank as topology_global_rank,
    worker_id as topology_worker_id,
)


PLUGIN_ABI = 2
GIB = 1024**3


def _jsonable(value: Any, *, _stack: frozenset[int] = frozenset()) -> Any:
    """Return a deterministic JSON representation without invoking ``repr``.

    Hugging Face configurations can contain other ``PretrainedConfig`` objects.
    Some releases return those objects from ``to_dict`` instead of recursively
    normalizing them, and their ``repr`` delegates to the same failing JSON
    encoder.  Walk the value by capability so new config classes remain
    supported without an architecture allowlist.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value, _stack=_stack)
    if isinstance(value, os.PathLike):
        return os.fspath(value)

    identity = id(value)
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if identity in _stack:
        return {"__coldsnap_cycle_type__": value_type}
    nested_stack = _stack | {identity}

    if isinstance(value, dict):
        return {
            str(key): _jsonable(item, _stack=nested_stack)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, _stack=nested_stack) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item, _stack=nested_stack) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            ),
        )

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict(), _stack=nested_stack)
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        attributes = vars(value)
    except TypeError:
        attributes = None
    if attributes is not None:
        return {
            "__coldsnap_object_type__": value_type,
            "attributes": _jsonable(attributes, _stack=nested_stack),
        }
    return {"__coldsnap_opaque_type__": value_type}


def _config_digest(model_config: Any) -> str:
    config = model_config.hf_config
    value = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plugin_digest() -> str:
    import coldsnap_core
    import coldsnap_sglang

    roots = {
        "coldsnap_core": Path(inspect.getfile(coldsnap_core)).resolve().parent,
        "coldsnap_sglang": Path(inspect.getfile(coldsnap_sglang)).resolve().parent,
    }
    digest = hashlib.sha256()
    for package, root in sorted(roots.items()):
        for path in sorted(root.glob("*.py")):
            digest.update(f"{package}/{path.name}".encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _cuda_driver_version() -> int | None:
    import ctypes

    try:
        library = ctypes.CDLL("libcuda.so.1")
        value = ctypes.c_int()
        if library.cuDriverGetVersion(ctypes.byref(value)) == 0:
            return int(value.value)
    except (AttributeError, OSError):
        pass
    return None


def _parallel_identity() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        for name in (
            "tp_size", "tp_rank", "pp_size", "pp_rank", "dp_size", "dp_rank",
            "ep_size", "ep_rank", "cp_size", "cp_rank",
        ):
            if hasattr(parallel, name):
                result[name] = int(getattr(parallel, name))
    except Exception:
        pass
    if not all(name in result for name in ("tp_size", "tp_rank", "pp_size", "pp_rank")):
        try:
            from sglang.srt.distributed.parallel_state import (
                get_pipeline_model_parallel_rank,
                get_pipeline_model_parallel_world_size,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            result.update({
                "tp_size": int(get_tensor_model_parallel_world_size()),
                "tp_rank": int(get_tensor_model_parallel_rank()),
                "pp_size": int(get_pipeline_model_parallel_world_size()),
                "pp_rank": int(get_pipeline_model_parallel_rank()),
            })
        except Exception:
            result.update({
                "tp_size": int(os.environ.get("WORLD_SIZE", "1")),
                "tp_rank": int(os.environ.get("LOCAL_RANK", "0")),
                "pp_size": 1,
                "pp_rank": 0,
            })
    try:
        import torch

        distributed = torch.distributed
        if distributed.is_available() and distributed.is_initialized():
            result["global_rank"] = int(distributed.get_rank())
            result["world_size"] = int(distributed.get_world_size())
    except Exception:
        pass
    graph = execution_graph()
    assert graph is not None
    resolved_rank = topology_global_rank(graph)
    assert resolved_rank is not None
    if "global_rank" in result and result["global_rank"] != resolved_rank:
        raise RuntimeError(
            "SGLang distributed rank disagrees with ColdSnap execution topology"
        )
    result["global_rank"] = resolved_rank
    result["unit"] = graph["unit"]
    result["worker_id"] = topology_worker_id(graph)
    result["execution"] = graph
    result.setdefault("local_rank", int(os.environ.get("LOCAL_RANK", result["tp_rank"])))
    result.setdefault("world_size", int(os.environ.get("WORLD_SIZE", result["tp_size"])))
    return result


def _capacity_class(total_memory: int) -> int:
    """Round to a stable GiB class instead of binding allocator page noise."""
    if total_memory <= 0:
        raise ArtifactError("CUDA device reported invalid total memory")
    return max(GIB, ((total_memory + GIB // 2) // GIB) * GIB)


def _driver_abi_major(version: int | None) -> int | None:
    # CUDA encodes driver API compatibility as 1000 * major + 10 * minor.
    return version // 1000 if version is not None else None


def validate_model_config(model_config: Any) -> None:
    if not bool(getattr(model_config, "is_generation", True)):
        raise ArtifactError(
            "ColdSnap SGLang requires a generation model"
        )


def build_identity(model_config: Any) -> dict[str, Any]:
    import torch
    from sglang.version import __version__ as sglang_version

    from .compat import contract_digest

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    parallel = _parallel_identity()
    return {
        "plugin_abi": PLUGIN_ABI,
        "plugin_code_sha256": _plugin_digest(),
        "engine": "sglang",
        "engine_version": str(sglang_version),
        "engine_contract_sha256": contract_digest(),
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_driver_abi_major": _driver_abi_major(_cuda_driver_version()),
        "model_path": str(model_config.model_path),
        "model_revision": getattr(model_config, "revision", None),
        "model_commit_hash": getattr(model_config.hf_config, "_commit_hash", None),
        "architectures": list(model_config.hf_config.architectures or ()),
        "hf_config_sha256": _config_digest(model_config),
        "dtype": str(model_config.dtype),
        "quantization": getattr(model_config, "quantization", None),
        "parallel": parallel,
        "device": {
            "name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "capacity_class_bytes": _capacity_class(int(properties.total_memory)),
        },
    }


def artifact_for(
    root: Path,
    identity: dict[str, Any],
    *,
    lock_root: Path | None = None,
) -> SemanticTensorArtifact:
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    worker = str(identity["parallel"]["worker_id"])
    if not worker or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in worker):
        raise ArtifactError("ColdSnap SGLang worker identity is invalid")
    return SemanticTensorArtifact(
        root / "workers" / worker / digest,
        lock_root=lock_root,
    )

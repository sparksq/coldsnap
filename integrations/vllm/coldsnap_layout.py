# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Semantic matching between captured and fresh vLLM weight allocations."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from coldsnap_disk_backend import GIB


def model_tensors(model: Any) -> list[tuple[str, str, Any]]:
    """Return registered and backend-owned tensors with stable owner paths."""
    import torch

    tensors: list[tuple[str, str, Any]] = []
    tensors.extend(
        (name, "parameter", tensor)
        for name, tensor in model.named_parameters(remove_duplicate=False)
    )
    tensors.extend(
        (name, "buffer", tensor)
        for name, tensor in model.named_buffers(remove_duplicate=False)
    )

    seen_tensors = {id(tensor) for _, _, tensor in tensors}
    seen_objects: set[int] = set()
    standard_module_attrs = {"_parameters", "_buffers", "_modules"}

    def visit(value: Any, path: str, depth: int) -> None:
        if isinstance(value, torch.Tensor):
            identity = id(value)
            if identity not in seen_tensors:
                seen_tensors.add(identity)
                tensors.append((path, "object", value))
            return
        if value is None or depth <= 0 or isinstance(
            value, (str, bytes, int, float, bool, torch.dtype, torch.device)
        ):
            return

        identity = id(value)
        if identity in seen_objects:
            return
        seen_objects.add(identity)

        if isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                visit(value[key], f"{path}[{key!r}]", depth - 1)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", depth - 1)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}", depth - 1)
            return

        if type(value).__module__.startswith(("sparkinfer.", "vllm.", "types")):
            try:
                attributes = vars(value)
            except TypeError:
                attributes = {}
            for name in sorted(attributes):
                visit(attributes[name], f"{path}.{name}", depth - 1)

    modules = list(model.named_modules(remove_duplicate=False))
    for module_name, module in modules:
        quant_method = getattr(module, "quant_method", None)
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        fused_experts = getattr(moe_kernel, "fused_experts", None)
        prepared_experts = getattr(fused_experts, "_prepared_experts", None)
        if prepared_experts is not None:
            prefix = module_name or "<root>"
            visit(
                prepared_experts,
                f"{prefix}.quant_method.moe_kernel.fused_experts._prepared_experts",
                16,
            )

    for module_name, module in modules:
        prefix = module_name or "<root>"
        for attr_name, value in sorted(vars(module).items()):
            if attr_name not in standard_module_attrs:
                visit(value, f"{prefix}.{attr_name}", 16)
    return tensors


def tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def find_allocation(
    ptr: int, nbytes: int, allocations: list[tuple[int, int]]
) -> int | None:
    for index, (base, size) in enumerate(allocations):
        if base <= ptr and ptr + nbytes <= base + size:
            return index
    return None


def reference_fingerprint(refs: list[dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                ref["name"],
                ref["kind"],
                tuple(ref["shape"]),
                tuple(ref["stride"]),
                ref["dtype"],
                int(ref["nbytes"]),
                int(ref["allocation_offset"]),
            )
            for ref in refs
        )
    )


def semantic_replay_map(
    saved: list[dict[str, Any]],
    current: list[tuple[int, int]],
    model: Any,
) -> tuple[list[tuple[int, int, dict[str, Any]]], list[int], dict[str, int]]:
    """Map saved allocations to a fresh process without relying on addresses."""
    import torch

    current_refs: list[list[dict[str, Any]]] = [[] for _ in current]
    for name, kind, tensor in model_tensors(model):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cuda"
            or tensor.numel() == 0
        ):
            continue
        nbytes = tensor_nbytes(tensor)
        ptr = int(tensor.data_ptr())
        current_index = find_allocation(ptr, nbytes, current)
        if current_index is None:
            storage = tensor.untyped_storage()
            current_index = find_allocation(
                int(storage.data_ptr()), int(storage.nbytes()), current
            )
        if current_index is None:
            continue
        current_refs[current_index].append(
            {
                "name": name,
                "kind": kind,
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "nbytes": nbytes,
                "allocation_offset": ptr - current[current_index][0],
            }
        )

    source_by_fingerprint: dict[tuple[Any, ...], list[int]] = {}
    current_by_fingerprint: dict[tuple[Any, ...], list[int]] = {}
    for index, entry in enumerate(saved):
        fingerprint = reference_fingerprint(entry.get("refs", []))
        if fingerprint:
            source_by_fingerprint.setdefault(fingerprint, []).append(index)
    for index, refs in enumerate(current_refs):
        fingerprint = reference_fingerprint(refs)
        if fingerprint:
            current_by_fingerprint.setdefault(fingerprint, []).append(index)

    duplicate_source = [
        indices for indices in source_by_fingerprint.values() if len(indices) != 1
    ]
    duplicate_current = [
        indices for indices in current_by_fingerprint.values() if len(indices) != 1
    ]
    if duplicate_source or duplicate_current:
        raise RuntimeError(
            "semantic replay contains ambiguous allocation fingerprints: "
            f"source={duplicate_source} current={duplicate_current}"
        )

    mapping: dict[int, int] = {}
    used_current: set[int] = set()
    semantic_matches = 0
    for fingerprint, source_indices in source_by_fingerprint.items():
        current_indices = current_by_fingerprint.get(fingerprint, [])
        if len(current_indices) != 1:
            raise RuntimeError(
                f"semantic replay cannot place source allocation {source_indices[0]}"
            )
        source_index = source_indices[0]
        current_index = current_indices[0]
        if int(saved[source_index]["size"]) != current[current_index][1]:
            raise RuntimeError(
                f"semantic allocation size mismatch at source {source_index}"
            )
        mapping[source_index] = current_index
        used_current.add(current_index)
        semantic_matches += 1

    unmatched_current_refs = [
        index
        for index, refs in enumerate(current_refs)
        if refs and index not in used_current
    ]
    if unmatched_current_refs:
        raise RuntimeError(
            "fresh model has unmatched referenced allocations: "
            f"{unmatched_current_refs}"
        )

    replay = [
        (mapping[index], current[mapping[index]][0], entry)
        for index, entry in enumerate(saved)
    ]
    extras = [index for index in range(len(current)) if index not in used_current]
    extra_bytes = sum(current[index][1] for index in extras)
    if len(extras) > 16 or extra_bytes > GIB:
        raise RuntimeError(
            f"semantic replay extras exceed safety bound: {len(extras)}, {extra_bytes}"
        )
    return replay, extras, {
        "semantic_matches": semantic_matches,
        "pointer_matches": 0,
        "order_matches": 0,
    }

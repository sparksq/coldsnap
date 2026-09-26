# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Portable provenance for B12x's in-place gate/up weight normalization."""

from __future__ import annotations

import importlib
from typing import Any


_FORMAT = 1
_KIND = "coldsnap-b12x-native-execution-state"


def _bindings(model: Any) -> list[tuple[dict[str, Any], Any, dict[Any, Any]]]:
    from coldsnap_recovery_loader import _b12x_prepared_owners, _b12x_weight_storage

    result = []
    for name, _module, _owner, prepared in _b12x_prepared_owners(model):
        storage = _b12x_weight_storage(prepared)
        if not any(
            cls.__name__ == "B12XFP4ExpertWeights"
            and cls.__module__ in {"b12x.moe.fused_moe", "b12x.moe.fused_moe._impl"}
            for cls in type(storage).__mro__
        ):
            continue
        plan = storage.plan
        # Other prepared formats normalize while packing and do not use this
        # lazy, pointer-keyed first-use registry. Non-gated weights cannot flip.
        if plan.w13_layout != "w31" or storage.w1_fp4.shape[1] != 2 * plan.intermediate_size:
            continue
        implementation = importlib.import_module("b12x.moe.fused_moe._impl")
        registry = getattr(implementation, "_W13_NORMALIZED_STORAGES", None)
        normalizer = getattr(implementation, "_ensure_w13_kernel_order_inplace", None)
        if registry is None and normalizer is None:
            continue
        if not isinstance(registry, dict) or not callable(normalizer):
            raise RuntimeError("native B12x gate/up normalization contract changed")
        signature = {
            "owner": name,
            "quant_modes": sorted(plan.quant_modes),
            "activation": plan.activation,
            "w13_layout": plan.w13_layout,
            "num_experts": plan.num_experts,
            "intermediate_size": plan.intermediate_size,
            "hidden_size": plan.hidden_size,
            "tensors": {
                field: {
                    "shape": list(getattr(storage, field).shape),
                    "stride": list(getattr(storage, field).stride()),
                    "dtype": str(getattr(storage, field).dtype),
                }
                for field in ("w1_fp4", "w1_blockscale")
            },
        }
        result.append((signature, storage, registry))
    return result


def _key(storage: Any) -> tuple[int, int]:
    return storage.w1_fp4.data_ptr(), storage.w1_blockscale.data_ptr()


def capture_native_state(model: Any) -> dict[str, Any]:
    """Record the order of the exact bytes being exported, without reading them."""
    owners = []
    for signature, storage, registry in _bindings(model):
        key = _key(storage)
        normalized = key in registry
        if normalized:
            retained = registry[key]
            if (
                not isinstance(retained, tuple) or len(retained) != 2
                or tuple(tensor.data_ptr() for tensor in retained) != key
            ):
                raise RuntimeError("native B12x normalization registry has inconsistent storage")
        owners.append({**signature, "normalized": normalized})
    return {"format": _FORMAT, "kind": _KIND, "owners": owners}


def restore_native_state(model: Any, state: Any) -> int:
    """Rebind captured order proofs after hydration, before warmup or graphs.

    The pack may already contain kernel-order weights. Re-running the first-use
    normalizer would flip those bytes back to checkpoint order. Restore only
    the proven registry entries; never transform the admitted payload bytes.
    """
    bindings = _bindings(model)
    if state is None and not bindings:
        return 0
    if (
        not isinstance(state, dict) or state.get("format") != _FORMAT
        or state.get("kind") != _KIND or not isinstance(state.get("owners"), list)
    ):
        raise RuntimeError(
            "native B12x payload lacks captured gate/up normalization state; "
            "recapture or use safetensors recovery"
        )
    records = state["owners"]
    by_owner = {}
    for record in records:
        if (
            not isinstance(record, dict) or not isinstance(record.get("owner"), str)
            or type(record.get("normalized")) is not bool or record["owner"] in by_owner
        ):
            raise RuntimeError("native B12x normalization state is invalid")
        by_owner[record["owner"]] = record
    if set(by_owner) != {signature["owner"] for signature, _storage, _registry in bindings}:
        raise RuntimeError("native B12x normalization owners differ from the captured model")
    # Validate every owner before changing any registry entry.
    for signature, _storage, _registry in bindings:
        record = by_owner[signature["owner"]]
        if {key: value for key, value in record.items() if key != "normalized"} != signature:
            raise RuntimeError(
                f"native B12x normalization geometry changed at {signature['owner']!r}"
            )
    for signature, storage, registry in bindings:
        if by_owner[signature["owner"]]["normalized"]:
            registry[_key(storage)] = (storage.w1_fp4, storage.w1_blockscale)
        else:
            registry.pop(_key(storage), None)
    return len(bindings)

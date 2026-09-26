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
    bindings = _validated_bindings(model, state)
    by_owner = {record["owner"]: record for record in state["owners"]} if state else {}
    for signature, storage, registry in bindings:
        if by_owner[signature["owner"]]["normalized"]:
            registry[_key(storage)] = (storage.w1_fp4, storage.w1_blockscale)
        else:
            registry.pop(_key(storage), None)
    return len(bindings)


def _validated_bindings(model: Any, state: Any) -> list[tuple[dict[str, Any], Any, dict[Any, Any]]]:
    bindings = _bindings(model)
    if state is None and not bindings:
        return bindings
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
    return bindings


def _normalization_mode(signature: dict[str, Any]) -> str:
    modes = set(signature["quant_modes"])
    if modes and modes <= {"nvfp4", "w4a8_nvfp4"}:
        return "nvfp4"
    if modes == {"w4a8_mx"}:
        return "w4a8_mx"
    raise RuntimeError(f"unsupported B12x gate/up recovery quantization modes: {sorted(modes)!r}")


def begin_recovery_reload(model: Any) -> dict[str, Any]:
    """Invalidate first-use proofs before checkpoint-order bytes overwrite them."""
    metadata = capture_native_state(model)
    bindings = _validated_bindings(model, metadata)
    # Resolve every contract before mutating a live registry. The returned
    # pointers also prevent recovery from replacing captured graph storage.
    for signature, _storage, _registry in bindings:
        _normalization_mode(signature)
    pointers = {signature["owner"]: _key(storage) for signature, storage, _registry in bindings}
    for _signature, storage, registry in bindings:
        registry.pop(_key(storage), None)
    return {"metadata": metadata, "pointers": pointers}


def finish_recovery_reload(model: Any, saved: dict[str, Any]) -> int:
    """Reproduce captured order before sample validation or retained graph use.

    A loader may have normalized its new bytes during finalization. Respect
    that fresh registry entry; otherwise call B12x's own bounded in-place
    normalizer exactly once. Native hydration never takes this path.
    """
    metadata = saved["metadata"]
    bindings = _validated_bindings(model, metadata)
    records = {record["owner"]: record for record in metadata["owners"]}
    for signature, storage, registry in bindings:
        name = signature["owner"]
        if _key(storage) != saved["pointers"][name]:
            raise RuntimeError(f"B12x recovery changed captured gate/up storage at {name!r}")
        if not records[name]["normalized"] and _key(storage) in registry:
            raise RuntimeError(f"B12x recovery changed captured gate/up order at {name!r}")
        _normalization_mode(signature)
    if bindings:
        normalizer = importlib.import_module("b12x.moe.fused_moe._impl")._ensure_w13_kernel_order_inplace
        for signature, storage, _registry in bindings:
            if records[signature["owner"]]["normalized"]:
                normalizer(
                    storage.w1_fp4, storage.w1_blockscale,
                    n=signature["intermediate_size"], k=signature["hidden_size"],
                    quant_mode=_normalization_mode(signature),
                )
    return len(bindings)


_PLAN_FIELDS = ("_plan", "_plan_key", "_plan_activation", "_plan_route_on_input")
_OWNER_CONFIG_FIELDS = ("_quant_mode", "_source_format", "_w13_layout", "_apply_router_weight_on_input")


def _execution_storage(prepared: Any) -> dict[str, Any]:
    from coldsnap_recovery_loader import B12X_PREPARED_WEIGHT_FIELDS, _b12x_weight_storage

    storage = _b12x_weight_storage(prepared)
    return {
        name: (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
        for name in B12X_PREPARED_WEIGHT_FIELDS
        for tensor in (getattr(storage, name),)
    }


def capture_recovery_plans(model: Any) -> list[dict[str, Any]]:
    """Retain warmed plans while vLLM rebuilds layer-owned execution owners."""
    from coldsnap_recovery_loader import _b12x_prepared_owners, _is_b12x_layer_cached_expert

    records = []
    for name, layer, owner, prepared in _b12x_prepared_owners(model):
        # Earlier B12xExperts uses a callable _plan and a lazy _plans cache.
        # Only the newer explicitly prepared execution contract needs rebinding.
        if (
            not _is_b12x_layer_cached_expert(owner)
            or not callable(getattr(owner, "_plan_for_tokens", None))
        ):
            continue
        fields = {field: getattr(owner, field) for field in _PLAN_FIELDS}
        key = fields["_plan_key"]
        if (
            fields["_plan"] is None or not isinstance(key, tuple) or len(key) != 4
            or key[-1] != id(prepared)
            or fields["_plan_activation"] != layer.activation
            or fields["_plan_route_on_input"] != bool(layer.apply_router_weight_on_input)
        ):
            raise RuntimeError(f"B12x recovery lacks a captured execution plan at {name!r}")
        records.append({
            "name": name, "layer": layer, "prepared": prepared, "fields": fields,
            "storage": _execution_storage(prepared),
            "config": tuple(getattr(owner, field) for field in _OWNER_CONFIG_FIELDS),
        })
    return records


def restore_recovery_plans(model: Any, records: list[dict[str, Any]]) -> int:
    """Reattach plans only to the exact captured prepared tensor owner.

    The pinned vLLM finalizer reuses packed storage and the prepared object,
    but creates a new B12xExperts with empty execution-plan fields. Its old
    plan still owns the warmed bindings. Identity and storage checks ensure
    that restoring those bindings cannot silently admit replacement weights.
    """
    from coldsnap_recovery_loader import _b12x_prepared_owners, _is_b12x_layer_cached_expert

    current = {name: (layer, owner, prepared) for name, layer, owner, prepared in _b12x_prepared_owners(model)}
    validated = []
    for record in records:
        name = record["name"]
        if name not in current:
            raise RuntimeError(f"B12x recovery lost its execution owner at {name!r}")
        layer, owner, prepared = current[name]
        fields = record["fields"]
        if (
            layer is not record["layer"] or not _is_b12x_layer_cached_expert(owner)
            or prepared is not record["prepared"]
            or _execution_storage(prepared) != record["storage"]
            or tuple(getattr(owner, field) for field in _OWNER_CONFIG_FIELDS) != record["config"]
            or layer.activation != fields["_plan_activation"]
            or bool(layer.apply_router_weight_on_input) != fields["_plan_route_on_input"]
        ):
            raise RuntimeError(f"B12x recovery changed captured execution bindings at {name!r}")
        validated.append((owner, fields))
    for owner, fields in validated:
        for field, value in fields.items():
            setattr(owner, field, value)
    return len(validated)

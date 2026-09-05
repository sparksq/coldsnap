# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Narrow compatibility boundary for the vLLM internals used by ColdSnap.

Everything outside this module should operate on ColdSnap concepts rather than
depending on a particular vLLM source layout.  Structural checks intentionally
fail closed: an unsupported upstream change should be reported during startup,
not after weight allocations have been unmapped or partially restored.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import inspect
from dataclasses import dataclass
from typing import Any


SLEEP_BACKEND_NAME = "coldsnap_disk"


class VllmContractError(RuntimeError):
    """The installed vLLM does not expose a contract required by ColdSnap."""


def _version() -> str:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _contract_error(message: str) -> VllmContractError:
    return VllmContractError(f"unsupported vLLM {_version()} contract: {message}")


@dataclass(frozen=True)
class Allocation:
    """Stable view of one entry in vLLM's tagged CUDA allocation pool."""

    pointer: int
    size: int
    tag: str
    handle: Any
    data: Any

    @property
    def is_released(self) -> bool:
        return bool(self.data.is_asleep)

    def set_released(self, value: bool) -> None:
        self.data.is_asleep = value

    # Compatibility names for callers that still speak vLLM's sleep contract.
    @property
    def is_asleep(self) -> bool:
        return self.is_released

    def set_asleep(self, value: bool) -> None:
        self.set_released(value)


@dataclass(frozen=True)
class SyntheticWeightSource:
    """Prepared safetensor metadata needed by the payload-free iterator."""

    folder: str
    files: tuple[str, ...]
    prefix: str
    weight_name_prefixes: tuple[str, ...] | None


def get_allocator() -> Any:
    try:
        module = importlib.import_module("vllm.device_allocator")
        factory = module.get_mem_allocator_instance
        allocator = factory()
    except (ImportError, AttributeError) as error:
        raise _contract_error(
            "vllm.device_allocator.get_mem_allocator_instance is unavailable"
        ) from error
    # Validate the collection shape before a caller can mutate allocator state.
    allocations(allocator)
    return allocator


def _allocation_size(data: Any) -> int:
    handle = getattr(data, "handle", None)
    try:
        size = int(handle[1])
    except (IndexError, KeyError, TypeError, ValueError):
        for name in ("size", "nbytes"):
            value = getattr(data, name, None)
            if value is not None:
                try:
                    size = int(value)
                    break
                except (TypeError, ValueError):
                    pass
        else:
            raise _contract_error(
                "allocation size is absent from handle[1], size, and nbytes"
            )
    if size <= 0:
        raise _contract_error(f"allocation has invalid size {size}")
    return size


def allocations(allocator: Any, tag: str | None = None) -> tuple[Allocation, ...]:
    mapping = getattr(allocator, "pointer_to_data", None)
    if mapping is None or not callable(getattr(mapping, "items", None)):
        raise _contract_error("CuMem allocator has no pointer_to_data mapping")
    result: list[Allocation] = []
    for pointer, data in mapping.items():
        allocation_tag = getattr(data, "tag", None)
        handle = getattr(data, "handle", None)
        if allocation_tag is None or handle is None or not hasattr(data, "is_asleep"):
            raise _contract_error("allocation data must expose tag, handle, and is_asleep")
        if tag is not None and allocation_tag != tag:
            continue
        try:
            resolved_pointer = int(pointer)
        except (TypeError, ValueError) as error:
            raise _contract_error(f"allocation pointer is invalid: {pointer!r}") from error
        result.append(
            Allocation(
                pointer=resolved_pointer,
                size=_allocation_size(data),
                tag=str(allocation_tag),
                handle=handle,
                data=data,
            )
        )
    return tuple(result)


def pin_memory_enabled() -> bool:
    for module_name in ("vllm.utils.torch_utils", "vllm.utils"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "PIN_MEMORY"):
            return bool(module.PIN_MEMORY)
    raise _contract_error("PIN_MEMORY is unavailable from vLLM utilities")


def _cumem_attribute(name: str) -> Any:
    try:
        module = importlib.import_module("vllm.device_allocator.cumem")
    except ImportError as error:
        raise _contract_error("vLLM CUDA CuMem support is unavailable") from error
    value = getattr(module, name, None)
    if value is None:
        raise _contract_error(f"CuMem module lacks {name}")
    return value


def cuda_runtime() -> Any:
    return _cumem_attribute("libcudart")


def map_allocation(handle: Any) -> None:
    operation = _cumem_attribute("create_and_map")
    if not callable(operation):
        raise _contract_error("CuMem create_and_map is not callable")
    operation(handle)


def unmap_allocation(handle: Any) -> None:
    operation = _cumem_attribute("unmap_and_release")
    if not callable(operation):
        raise _contract_error("CuMem unmap_and_release is not callable")
    operation(handle)


def force_cross_rank_rpc_over_tcp(queue_class: type[Any]) -> None:
    """Disable cross-rank local readers without fixing Queue's full signature."""
    if getattr(queue_class, "_coldsnap_force_rpc_tcp", False):
        return
    original = getattr(queue_class, "__init__", None)
    if not callable(original):
        raise _contract_error("MessageQueue.__init__ is unavailable")
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError) as error:
        raise _contract_error("cannot inspect MessageQueue.__init__") from error
    required = {"n_reader", "n_local_reader"}
    missing = required - set(signature.parameters)
    if missing:
        raise _contract_error(
            "MessageQueue.__init__ lacks " + ", ".join(sorted(missing))
        )

    @functools.wraps(original)
    def init_without_cross_rank_shm(self: Any, *args: Any, **kwargs: Any) -> None:
        try:
            bound = signature.bind(self, *args, **kwargs)
        except TypeError as error:
            raise _contract_error("MessageQueue.__init__ call shape changed") from error
        n_reader = int(bound.arguments["n_reader"])
        n_local_reader = int(bound.arguments["n_local_reader"])
        if n_reader > 1 and n_local_reader > 0:
            bound.arguments["n_local_reader"] = 0
            if "local_reader_ranks" in signature.parameters:
                bound.arguments["local_reader_ranks"] = []
        original(*bound.args, **bound.kwargs)

    queue_class.__init__ = init_without_cross_rank_shm
    queue_class._coldsnap_force_rpc_tcp = True


def register_sleep_backend(factory: Any, backend_class: type[Any]) -> str:
    """Register the named backend through vLLM's public API when available."""
    register = getattr(factory, "register_backend", None)
    if callable(register):
        try:
            register(
                SLEEP_BACKEND_NAME,
                backend_class.__module__,
                backend_class.__name__,
            )
        except ValueError as error:
            resolve = getattr(factory, "get_backend_class", None)
            if not callable(resolve):
                raise _contract_error(
                    f"backend {SLEEP_BACKEND_NAME!r} is already registered"
                ) from error
            try:
                existing = resolve(SLEEP_BACKEND_NAME)
            except Exception as resolve_error:
                raise _contract_error(
                    f"cannot resolve existing backend {SLEEP_BACKEND_NAME!r}"
                ) from resolve_error
            if existing is not backend_class:
                raise _contract_error(
                    f"backend name {SLEEP_BACKEND_NAME!r} belongs to "
                    f"{existing.__module__}.{existing.__name__}"
                ) from error
        return "public"

    raise _contract_error("SleepModeBackendFactory has no registration API")


def override_default_sleep_backend(factory: Any, backend_class: type[Any]) -> None:
    """Bridge vLLM releases that register backends but expose no selector."""
    registry = getattr(factory, "_registry", None)
    if not isinstance(registry, dict) or "cumem" not in registry:
        raise _contract_error(
            "cannot select ColdSnap: no CLI selector or cumem registry entry"
        )
    registry["cumem"] = lambda: backend_class


_PREPARE_SOURCE_ATTRIBUTES = {
    "model_name_or_path": "model_or_path",
    "subfolder": "subfolder",
    "revision": "revision",
    "fall_back_to_pt": "fall_back_to_pt",
    "allow_patterns_overrides": "allow_patterns_overrides",
    "weight_name_prefixes": "weight_name_prefixes",
}


def supported_prepare_parameters() -> frozenset[str]:
    """Names understood by the structural model-loader adapter."""
    return frozenset(_PREPARE_SOURCE_ATTRIBUTES)


def prepare_synthetic_weight_source(loader: Any, source: Any) -> SyntheticWeightSource:
    """Call the installed vLLM's weight preparation contract structurally."""
    prepare = getattr(loader, "_prepare_weights", None)
    if not callable(prepare):
        raise _contract_error("DefaultModelLoader._prepare_weights is unavailable")
    try:
        parameters = inspect.signature(prepare).parameters
    except (TypeError, ValueError) as error:
        raise _contract_error("cannot inspect DefaultModelLoader._prepare_weights") from error

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, parameter in parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        attribute = _PREPARE_SOURCE_ATTRIBUTES.get(name)
        if attribute is None:
            if parameter.default is inspect.Parameter.empty:
                raise _contract_error(
                    f"_prepare_weights added required parameter {name!r}"
                )
            continue
        if not hasattr(source, attribute):
            if name == "weight_name_prefixes":
                value = None
            elif parameter.default is not inspect.Parameter.empty:
                continue
            else:
                raise _contract_error(
                    f"weight Source has no required attribute {attribute!r}"
                )
        else:
            value = getattr(source, attribute)
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value

    try:
        prepared = prepare(*args, **kwargs)
    except TypeError as error:
        raise _contract_error(
            "DefaultModelLoader._prepare_weights rejected the adapted call"
        ) from error
    if not isinstance(prepared, tuple) or len(prepared) < 3:
        raise _contract_error("_prepare_weights must return folder, files, and format")
    folder = prepared[0]
    if not isinstance(folder, str):
        raise _contract_error("_prepare_weights returned an invalid checkpoint folder")
    files = prepared[1]
    if not isinstance(files, (list, tuple)) or not all(
        isinstance(path, str) for path in files
    ):
        raise _contract_error("_prepare_weights returned an invalid file collection")
    if prepared[2] is not True:
        raise _contract_error("synthetic restore requires safetensors checkpoint files")

    prefix = getattr(source, "prefix", None)
    if not isinstance(prefix, str):
        raise _contract_error("weight Source has no string prefix")
    raw_prefixes = getattr(source, "weight_name_prefixes", None)
    if raw_prefixes is not None:
        try:
            prefixes = tuple(str(value) for value in raw_prefixes)
        except TypeError as error:
            raise _contract_error("weight_name_prefixes is not iterable") from error
    else:
        prefixes = None
    return SyntheticWeightSource(folder, tuple(files), prefix, prefixes)


def register_model_loader(load_format: str, loader_class: type[Any]) -> bool:
    """Register a loader through vLLM's required public loader API."""
    try:
        module = importlib.import_module("vllm.model_executor.model_loader")
    except ImportError as error:
        raise _contract_error("vLLM model-loader registry is unavailable") from error
    register = getattr(module, "register_model_loader", None)
    if not callable(register):
        raise _contract_error("vLLM model-loader registry has no registration API")
    register(load_format)(loader_class)
    return True

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Cooperative vLLM CUDA-context epoch prototype.

This is intentionally narrower than CUDA API interposition. vLLM's CuMem
allocator already gives ColdSnap the three pieces needed for a semantic reset:
stable virtual addresses, enumerated allocation recipes, and host-side mutable
boxes that hold ``CUmemGenericAllocationHandle`` values. After a normal sleep
externalizes the bytes, this module can destroy the primary context and later
map fresh physical backing into the VMM reservations retained at the exact same
addresses.

The first prototype is one-device-per-process/eager only. It does not claim
that CUDA graphs, custom streams, library handles, or NCCL communicators
survive a context epoch without their own cooperative recipes.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from .coldsnap_vllm import (
        allocations,
        cuda_runtime,
        get_allocator,
        unmap_allocation,
    )
except ImportError:  # Plugin directory is also supported directly on PYTHONPATH.
    from coldsnap_vllm import (  # type: ignore[no-redef]
        allocations,
        cuda_runtime,
        get_allocator,
        unmap_allocation,
    )


CUDA_EPOCH_LIBRARY_ENV = "COLDSNAP_CUDA_EPOCH_NATIVE_LIBRARY"


class CudaEpochError(RuntimeError):
    """The cooperative CUDA epoch could not complete safely."""


class _NativeAllocation(ctypes.Structure):
    _fields_ = [
        ("address", ctypes.c_size_t),
        ("mapped_bytes", ctypes.c_uint64),
        ("handle_box_address", ctypes.c_size_t),
        ("device", ctypes.c_int),
    ]


class _NativeContextProbe(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("device", ctypes.c_int),
        ("cu_init", ctypes.c_int),
        ("cu_device_get", ctypes.c_int),
        ("cu_ctx_get_current", ctypes.c_int),
        ("current_context_present", ctypes.c_int),
        ("cu_ctx_clear_current", ctypes.c_int),
        ("cu_primary_get_state", ctypes.c_int),
        ("primary_flags", ctypes.c_uint),
        ("primary_active", ctypes.c_int),
        ("cu_primary_retain", ctypes.c_int),
        ("cu_primary_release", ctypes.c_int),
        ("cu_private_create", ctypes.c_int),
        ("cu_private_get_current", ctypes.c_int),
        ("private_context_matched", ctypes.c_int),
        ("cu_private_alloc", ctypes.c_int),
        ("cu_private_memset", ctypes.c_int),
        ("cu_private_copy_to_host", ctypes.c_int),
        ("cu_private_synchronize", ctypes.c_int),
        ("cu_private_free", ctypes.c_int),
        ("driver_payload_verified", ctypes.c_int),
        ("runtime_get_device", ctypes.c_int),
        ("runtime_device", ctypes.c_int),
        ("runtime_malloc", ctypes.c_int),
        ("runtime_memset", ctypes.c_int),
        ("runtime_copy_to_host", ctypes.c_int),
        ("runtime_synchronize", ctypes.c_int),
        ("runtime_free", ctypes.c_int),
        ("runtime_payload_verified", ctypes.c_int),
        ("cu_private_destroy", ctypes.c_int),
        ("cu_ctx_clear_after_probe", ctypes.c_int),
    ]


_CONTEXT_PROBE_ABI = 1
_PROBE_NOT_RUN = -(2**31)


@dataclass(frozen=True)
class CudaEpochAllocation:
    """CPU-resident recipe for one sleeping vLLM VMM allocation."""

    address: int
    mapped_bytes: int
    handle_box_address: int
    device: int
    tag: str
    data: Any = field(compare=False, repr=False)
    externalized_by_epoch: bool = False

    @property
    def handle_key(self) -> tuple[int, int, int, int]:
        return (
            self.device,
            self.mapped_bytes,
            self.address,
            self.handle_box_address,
        )


def _version_tuple(value: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise CudaEpochError(f"cannot determine PyTorch version from {value!r}")
    return int(match.group(1)), int(match.group(2))


def _require_pytorch_2_10() -> str:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise CudaEpochError("the CUDA epoch prototype requires PyTorch 2.10+") from error
    if _version_tuple(version) < (2, 10):
        raise CudaEpochError(
            f"the CUDA epoch prototype requires PyTorch 2.10+, got {version}"
        )
    return version


def _allocation_recipe(
    allocation: Any, externalized_addresses: frozenset[int]
) -> CudaEpochAllocation:
    handle = allocation.handle
    if not isinstance(handle, tuple) or len(handle) != 4:
        raise CudaEpochError(
            "vLLM CUDA epoch requires a four-field CUDA CuMem handle tuple"
        )
    try:
        device, mapped_bytes, address, handle_box_address = map(int, handle)
    except (TypeError, ValueError) as error:
        raise CudaEpochError(
            "vLLM CUDA epoch requires integer CUDA CuMem handle fields"
        ) from error
    if (
        device < 0
        or mapped_bytes <= 0
        or address <= 0
        or handle_box_address <= 0
    ):
        raise CudaEpochError("vLLM CUDA CuMem handle fields are invalid")
    if address != allocation.pointer or mapped_bytes != allocation.size:
        raise CudaEpochError(
            "vLLM CUDA CuMem handle disagrees with allocator pointer/size metadata"
        )
    if not allocation.is_released:
        raise CudaEpochError(
            f"CUDA epoch allocation {address:#x} is still mapped; call sleep first"
        )
    return CudaEpochAllocation(
        address=address,
        mapped_bytes=mapped_bytes,
        handle_box_address=handle_box_address,
        device=device,
        tag=allocation.tag,
        data=allocation.data,
        externalized_by_epoch=address in externalized_addresses,
    )


def _externalize_active_allocation(allocation: Any) -> None:
    import torch

    data = allocation.data
    if getattr(data, "cpu_backup_tensor", None) is not None:
        raise CudaEpochError(
            f"active CUDA epoch allocation {allocation.pointer:#x} already has a backup"
        )
    backup = torch.empty(
        allocation.size,
        dtype=torch.uint8,
        device="cpu",
        # CUDA host registrations and their allocator events belong to the
        # context being destroyed. Epoch-spanning bytes must remain ordinary
        # CPU mappings that a fresh driver context can read.
        pin_memory=False,
    )
    cuda_runtime().cudaMemcpy(
        backup.data_ptr(), allocation.pointer, allocation.size
    )
    try:
        unmap_allocation(allocation.handle)
    except BaseException:
        # Keep the bytes reachable for diagnosis if the driver partially
        # completed the unmap. The epoch itself will refuse to reset.
        data.cpu_backup_tensor = backup
        raise
    data.cpu_backup_tensor = backup
    allocation.set_released(True)


def _validate_recipes(
    recipes: tuple[CudaEpochAllocation, ...],
) -> tuple[CudaEpochAllocation, ...]:
    if not recipes:
        raise CudaEpochError("vLLM CUDA epoch found no sleeping allocations")
    ordered = tuple(sorted(recipes, key=lambda item: (item.device, item.address)))
    devices = {item.device for item in ordered}
    if len(devices) != 1:
        raise CudaEpochError(
            "the CUDA epoch prototype requires exactly one device per process"
        )
    keys = {item.handle_key for item in ordered}
    if len(keys) != len(ordered):
        raise CudaEpochError("vLLM CUDA epoch allocation recipes contain duplicates")
    previous_device = -1
    previous_end = 0
    for item in ordered:
        if item.device == previous_device and item.address < previous_end:
            raise CudaEpochError("vLLM CUDA epoch allocation recipes overlap")
        previous_device = item.device
        previous_end = item.address + item.mapped_bytes
    return ordered


class NativeCudaEpoch:
    """Validated ctypes binding for the native reset/rebind primitive."""

    def __init__(self, library_path: str | os.PathLike[str]) -> None:
        path = Path(library_path).resolve()
        if not path.is_file():
            raise CudaEpochError(f"native CUDA epoch library does not exist: {path}")
        self.path = path
        self.library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self._configure_abi()

    def _configure_abi(self) -> None:
        library = self.library
        library.coldsnap_cuda_epoch_reset_device.argtypes = [ctypes.c_int]
        library.coldsnap_cuda_epoch_reset_device.restype = ctypes.c_int
        library.coldsnap_cuda_epoch_probe_context.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_NativeContextProbe),
        ]
        library.coldsnap_cuda_epoch_probe_context.restype = ctypes.c_int
        library.coldsnap_cuda_epoch_driver_result_name.argtypes = [ctypes.c_int]
        library.coldsnap_cuda_epoch_driver_result_name.restype = ctypes.c_char_p
        library.coldsnap_cuda_epoch_runtime_result_name.argtypes = [ctypes.c_int]
        library.coldsnap_cuda_epoch_runtime_result_name.restype = ctypes.c_char_p
        library.coldsnap_cuda_epoch_rebind.argtypes = [
            ctypes.POINTER(_NativeAllocation),
            ctypes.c_size_t,
        ]
        library.coldsnap_cuda_epoch_rebind.restype = ctypes.c_int
        library.coldsnap_cuda_epoch_activate_device.argtypes = [ctypes.c_int]
        library.coldsnap_cuda_epoch_activate_device.restype = ctypes.c_int
        library.coldsnap_cuda_epoch_copy_from_host.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        library.coldsnap_cuda_epoch_copy_from_host.restype = ctypes.c_int
        library.coldsnap_cuda_epoch_fill_zero.argtypes = [
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        library.coldsnap_cuda_epoch_fill_zero.restype = ctypes.c_int
        library.coldsnap_cuda_epoch_synchronize.argtypes = []
        library.coldsnap_cuda_epoch_synchronize.restype = ctypes.c_int
        library.coldsnap_cuda_epoch_last_error.argtypes = []
        library.coldsnap_cuda_epoch_last_error.restype = ctypes.c_char_p

    def _error(self, operation: str) -> CudaEpochError:
        value = self.library.coldsnap_cuda_epoch_last_error()
        detail = value.decode("utf-8", errors="replace") if value else "unknown error"
        return CudaEpochError(f"native CUDA epoch {operation} failed: {detail}")

    def reset(self, device: int) -> None:
        if self.library.coldsnap_cuda_epoch_reset_device(device) != 0:
            raise self._error("reset")

    @staticmethod
    def _decode_result(value: bytes | None) -> str:
        return value.decode("utf-8", errors="replace") if value else "UNKNOWN"

    def probe_context(self, device: int) -> dict[str, Any]:
        """Probe primary/private context creation without changing policy."""
        probe = _NativeContextProbe()
        if self.library.coldsnap_cuda_epoch_probe_context(
            device, ctypes.byref(probe)
        ) != 0:
            raise self._error("context probe")
        if (
            probe.abi_version != _CONTEXT_PROBE_ABI
            or probe.struct_size != ctypes.sizeof(_NativeContextProbe)
            or probe.device != device
        ):
            raise CudaEpochError("native CUDA epoch context probe ABI mismatch")

        def driver_result(code: int) -> dict[str, Any]:
            return {
                "code": code,
                "name": self._decode_result(
                    self.library.coldsnap_cuda_epoch_driver_result_name(code)
                ),
                "ran": code != _PROBE_NOT_RUN,
                "success": code == 0,
            }

        def runtime_result(code: int) -> dict[str, Any]:
            return {
                "code": code,
                "name": self._decode_result(
                    self.library.coldsnap_cuda_epoch_runtime_result_name(code)
                ),
                "ran": code != _PROBE_NOT_RUN,
                "success": code == 0,
            }

        primary = {
            "get_state": driver_result(probe.cu_primary_get_state),
            "flags": probe.primary_flags,
            "active": (
                None if probe.primary_active < 0 else bool(probe.primary_active)
            ),
            "retain": driver_result(probe.cu_primary_retain),
            "release": driver_result(probe.cu_primary_release),
        }
        primary["usable"] = bool(
            primary["retain"]["success"] and primary["release"]["success"]
        )
        private_driver = {
            "create": driver_result(probe.cu_private_create),
            "get_current": driver_result(probe.cu_private_get_current),
            "context_matched": (
                None
                if probe.private_context_matched < 0
                else bool(probe.private_context_matched)
            ),
            "alloc_4k": driver_result(probe.cu_private_alloc),
            "memset": driver_result(probe.cu_private_memset),
            "copy_to_host": driver_result(probe.cu_private_copy_to_host),
            "synchronize": driver_result(probe.cu_private_synchronize),
            "free": driver_result(probe.cu_private_free),
            "payload_verified": (
                None
                if probe.driver_payload_verified < 0
                else bool(probe.driver_payload_verified)
            ),
            "destroy": driver_result(probe.cu_private_destroy),
        }
        private_driver["usable"] = bool(
            private_driver["create"]["success"]
            and private_driver["get_current"]["success"]
            and private_driver["context_matched"]
            and private_driver["alloc_4k"]["success"]
            and private_driver["memset"]["success"]
            and private_driver["copy_to_host"]["success"]
            and private_driver["synchronize"]["success"]
            and private_driver["free"]["success"]
            and private_driver["payload_verified"]
            and private_driver["destroy"]["success"]
        )
        private_runtime = {
            "get_device": runtime_result(probe.runtime_get_device),
            "device": probe.runtime_device if probe.runtime_device >= 0 else None,
            "malloc_4k": runtime_result(probe.runtime_malloc),
            "memset": runtime_result(probe.runtime_memset),
            "copy_to_host": runtime_result(probe.runtime_copy_to_host),
            "synchronize": runtime_result(probe.runtime_synchronize),
            "free": runtime_result(probe.runtime_free),
            "payload_verified": (
                None
                if probe.runtime_payload_verified < 0
                else bool(probe.runtime_payload_verified)
            ),
        }
        private_runtime["usable"] = bool(
            private_runtime["get_device"]["success"]
            and private_runtime["device"] == device
            and private_runtime["malloc_4k"]["success"]
            and private_runtime["memset"]["success"]
            and private_runtime["copy_to_host"]["success"]
            and private_runtime["synchronize"]["success"]
            and private_runtime["free"]["success"]
            and private_runtime["payload_verified"]
        )
        return {
            "schema": 1,
            "device": device,
            "initialization": {
                "cu_init": driver_result(probe.cu_init),
                "cu_device_get": driver_result(probe.cu_device_get),
                "get_current": driver_result(probe.cu_ctx_get_current),
                "current_context_present": (
                    None
                    if probe.current_context_present < 0
                    else bool(probe.current_context_present)
                ),
                "clear_current": driver_result(probe.cu_ctx_clear_current),
            },
            "primary": primary,
            "private": {
                "attempted": private_driver["create"]["ran"],
                "driver": private_driver,
                "runtime": private_runtime,
            },
            "clear_after_probe": driver_result(probe.cu_ctx_clear_after_probe),
            "policy_changed": False,
        }

    def rebind(self, recipes: tuple[CudaEpochAllocation, ...]) -> None:
        native = (_NativeAllocation * len(recipes))(
            *(
                _NativeAllocation(
                    address=item.address,
                    mapped_bytes=item.mapped_bytes,
                    handle_box_address=item.handle_box_address,
                    device=item.device,
                )
                for item in recipes
            )
        )
        if self.library.coldsnap_cuda_epoch_rebind(native, len(recipes)) != 0:
            raise self._error("rebind")

    def activate(self, device: int) -> None:
        if self.library.coldsnap_cuda_epoch_activate_device(device) != 0:
            raise self._error("activate device")

    def copy_from_host(self, destination: int, source: int, size: int) -> None:
        if self.library.coldsnap_cuda_epoch_copy_from_host(
            destination, source, size
        ) != 0:
            raise self._error("copy from host")

    def fill_zero(self, destination: int, size: int) -> None:
        if self.library.coldsnap_cuda_epoch_fill_zero(destination, size) != 0:
            raise self._error("fill zero")

    def synchronize(self) -> None:
        if self.library.coldsnap_cuda_epoch_synchronize() != 0:
            raise self._error("synchronize")


class VllmCudaEpoch:
    """One reset/rebind lifecycle for vLLM's singleton CUDA allocator."""

    def __init__(
        self,
        recipes: tuple[CudaEpochAllocation, ...],
        native: Any,
        *,
        torch_version: str,
        runtime: Any | None = None,
    ) -> None:
        self.recipes = _validate_recipes(recipes)
        self.native = native
        self.torch_version = torch_version
        self.runtime = runtime
        self._state = "CAPTURED"
        self._lock = threading.Lock()

    @classmethod
    def capture(
        cls,
        allocator: Any | None = None,
        *,
        library_path: str | os.PathLike[str] | None = None,
        native: Any | None = None,
        externalize_active: bool = False,
    ) -> "VllmCudaEpoch":
        torch_version = _require_pytorch_2_10()
        resolved_allocator = get_allocator() if allocator is None else allocator
        current = allocations(resolved_allocator)
        runtime = None
        try:
            runtime_module = importlib.import_module("coldsnap_vllm_cuda_runtime")
        except ImportError:
            runtime_module = None
        if runtime_module is not None:
            runtime = runtime_module.capture_cuda_epoch_runtime(
                tuple((item.pointer, item.size) for item in current)
            )
        if runtime is not None:
            # Closing a MemPool after its backing is externally unmapped makes
            # its normal cache cleanup double-unmap released blocks.  Seal it
            # first, then refresh because the pool exit can reclaim unused
            # cached allocations from the singleton inventory.
            runtime.seal()
            current = allocations(resolved_allocator)
        externalized: set[int] = set()
        if externalize_active:
            for allocation in current:
                if not allocation.is_released:
                    _externalize_active_allocation(allocation)
                    externalized.add(allocation.pointer)
        recipes = tuple(
            _allocation_recipe(allocation, frozenset(externalized))
            for allocation in current
        )
        if native is None:
            configured = (
                os.environ.get(CUDA_EPOCH_LIBRARY_ENV)
                if library_path is None
                else os.fspath(library_path)
            )
            if not configured:
                raise CudaEpochError(
                    f"{CUDA_EPOCH_LIBRARY_ENV} must name libcoldsnap_cuda_epoch.so"
                )
            native = NativeCudaEpoch(configured)
        return cls(
            recipes,
            native,
            torch_version=torch_version,
            runtime=runtime,
        )

    @property
    def state(self) -> str:
        return self._state

    @property
    def device(self) -> int:
        return self.recipes[0].device

    def reset(self) -> dict[str, Any]:
        with self._lock:
            if self._state != "CAPTURED":
                raise CudaEpochError(
                    f"CUDA epoch reset requires CAPTURED state, got {self._state}"
                )
            if self.runtime is not None:
                self.runtime.prepare_reset()
            self.native.reset(self.device)
            self._state = "RESET"
        return self.inventory()

    def rebind(self) -> dict[str, Any]:
        """Recreate VMM backing and the CUDA resources needed before wake.

        Keeping this boundary separate from :meth:`resume` lets distributed
        runtimes rebuild their communicators against the fresh CUDA context
        before vLLM hydrates its sleeping allocation pools.
        """
        with self._lock:
            if self._state != "RESET":
                raise CudaEpochError(
                    f"CUDA epoch rebind requires RESET state, got {self._state}"
                )
            self._state = "REBINDING"
        try:
            self.native.rebind(self.recipes)
            if self.runtime is not None:
                self.runtime.rebuild(self.native)
        except BaseException:
            with self._lock:
                self._state = "ERROR"
            raise
        with self._lock:
            self._state = "REBOUND"
        return self.inventory()

    def probe_context(self) -> dict[str, Any]:
        """Diagnose post-reset context creation without adopting a fallback."""
        with self._lock:
            if self._state != "RESET":
                raise CudaEpochError(
                    "CUDA epoch context probe requires RESET state, "
                    f"got {self._state}"
                )
            result = self.native.probe_context(self.device)
        if not isinstance(result, dict):
            raise CudaEpochError("native CUDA epoch context probe returned invalid data")
        return result

    @staticmethod
    def _normalize_handle(handle: Any) -> tuple[int, int, int, int]:
        if not isinstance(handle, tuple) or len(handle) != 4:
            raise CudaEpochError("vLLM wake supplied an invalid CuMem handle")
        try:
            return tuple(map(int, handle))  # type: ignore[return-value]
        except (TypeError, ValueError) as error:
            raise CudaEpochError("vLLM wake supplied non-integer CuMem fields") from error

    def resume(self, callback: Callable[[], Any]) -> Any:
        """Run normal vLLM hydration after an optional explicit rebind."""
        if not callable(callback):
            raise TypeError("CUDA epoch resume callback must be callable")
        if self._state == "RESET":
            self.rebind()
        elif self._state != "REBOUND":
            raise CudaEpochError(
                f"CUDA epoch resume requires RESET or REBOUND state, got {self._state}"
            )
        module = importlib.import_module("vllm.device_allocator.cumem")
        original = getattr(module, "create_and_map", None)
        if not callable(original):
            raise CudaEpochError("vLLM CuMem create_and_map is unavailable")
        pending = {item.handle_key for item in self.recipes}
        required = {
            item.handle_key
            for item in self.recipes
            if not item.externalized_by_epoch
        }
        published: list[tuple[CudaEpochAllocation, str]] = []
        for item in self.recipes:
            if not item.externalized_by_epoch:
                continue
            data = item.data
            backup = getattr(data, "cpu_backup_tensor", None)
            if backup is None:
                raise CudaEpochError(
                    f"CUDA epoch residual {item.address:#x} lost its host backup"
                )
            size = int(backup.numel()) * int(backup.element_size())
            if size != item.mapped_bytes:
                raise CudaEpochError(
                    f"CUDA epoch residual {item.address:#x} backup size changed"
                )
            copy_from_host = getattr(self.native, "copy_from_host", None)
            if not callable(copy_from_host):
                raise CudaEpochError(
                    "native CUDA epoch hydration copy is unavailable"
                )
            copy_from_host(item.address, backup.data_ptr(), item.mapped_bytes)
            original_tag = str(data.tag)
            data.tag = "coldsnap_cuda_epoch_residual"
            data.cpu_backup_tensor = None
            data.is_asleep = False
            published.append((item, original_tag))

        provider_class = None
        provider_originals: dict[str, Any] = {}
        native_copy = getattr(self.native, "copy_from_host", None)
        native_zero = getattr(self.native, "fill_zero", None)
        native_synchronize = getattr(self.native, "synchronize", None)
        if all(callable(item) for item in (native_copy, native_zero, native_synchronize)):
            try:
                provider_module = importlib.import_module("coldsnap_vllm_memory")
                provider_class = provider_module.VllmMemoryProvider
            except (ImportError, AttributeError):
                provider_class = None
        if provider_class is not None:
            for name in ("copy_from_host", "fill_zero", "synchronize"):
                provider_originals[name] = getattr(provider_class, name)

            def epoch_copy_from_host(
                provider: Any, destination: int, source: Any, size: int
            ) -> None:
                provider._validate_copy(source, size)
                native_copy(destination, source.pointer, size)

            def epoch_fill_zero(
                _provider: Any, destination: int, size: int
            ) -> None:
                native_zero(destination, size)

            def epoch_synchronize(_provider: Any) -> None:
                native_synchronize()

            provider_class.copy_from_host = epoch_copy_from_host
            provider_class.fill_zero = epoch_fill_zero
            provider_class.synchronize = epoch_synchronize

        def accept_prebound(handle: Any) -> None:
            key = self._normalize_handle(handle)
            if key in pending:
                pending.remove(key)
                return
            original(handle)

        module.create_and_map = accept_prebound
        try:
            result = callback()
        finally:
            if getattr(module, "create_and_map", None) is accept_prebound:
                module.create_and_map = original
            for item, original_tag in published:
                item.data.tag = original_tag
            if provider_class is not None:
                for name, original_method in provider_originals.items():
                    setattr(provider_class, name, original_method)
        unclaimed_required = pending & required
        if unclaimed_required:
            raise CudaEpochError(
                "vLLM wake did not claim "
                f"{len(unclaimed_required)} required prebound CUDA allocations"
            )
        with self._lock:
            if self._state != "REBOUND":
                raise CudaEpochError(
                    f"CUDA epoch wake handoff expected REBOUND, got {self._state}"
                )
            self._state = "ACTIVE"
        return result

    def inventory(self) -> dict[str, Any]:
        inventory = {
            "state": self._state,
            "torch_version": self.torch_version,
            "device": self.device,
            "allocation_count": len(self.recipes),
            "mapped_bytes": sum(item.mapped_bytes for item in self.recipes),
            "epoch_externalized_count": sum(
                item.externalized_by_epoch for item in self.recipes
            ),
            "epoch_externalized_bytes": sum(
                item.mapped_bytes
                for item in self.recipes
                if item.externalized_by_epoch
            ),
            "tags": sorted({item.tag for item in self.recipes}),
            "addresses": [item.address for item in self.recipes],
        }
        if self.runtime is not None:
            inventory["runtime"] = self.runtime.inventory()
        return inventory

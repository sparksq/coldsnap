# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Native stable-address CUDA-graph allocation controller."""

from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from coldsnap_core.memory import (
    CUDA_GRAPH,
    MemoryCapabilityError,
    MemoryLayoutDescriptor,
    MemoryLayoutKind,
    MemoryRegion,
    MemoryResidencyStats,
    ProviderCapabilities,
    resolve_regions,
)


GRAPH_LIBRARY_ENV = "COLDSNAP_GRAPH_NATIVE_LIBRARY"


class GraphMemoryError(MemoryCapabilityError):
    """The native graph-memory provider could not complete an operation."""


@dataclass(frozen=True)
class GraphRegionStats:
    allocation_count: int
    raw_bytes: int
    mapped_bytes: int
    paused_count: int

    @property
    def is_released(self) -> bool:
        return self.paused_count > 0


@dataclass(frozen=True)
class GraphAllocation:
    address: int
    raw_bytes: int
    mapped_bytes: int
    device: int
    pool_id: int
    state: int


class DisabledCudaGraphController:
    """No-op graph controller used when native interception is disabled."""

    enabled = False
    capabilities = ProviderCapabilities(
        provider="cuda_graph_disabled",
        tagged_regions=False,
        stable_virtual_addresses=False,
        scoped_interception=False,
        allocation_enumeration=False,
        external_snapshot_store=False,
        graph_regions=False,
        per_device_synchronization=False,
    )

    @contextmanager
    def capture_region(
        self,
        *,
        pool_id: int = 0,
        device: int | None = None,
    ) -> Iterator[None]:
        del pool_id, device
        yield

    def release(self, regions: list[str | MemoryRegion] | None = None) -> None:
        del regions

    def restore(self, regions: list[str | MemoryRegion] | None = None) -> None:
        del regions

    def stats(self) -> GraphRegionStats:
        return GraphRegionStats(0, 0, 0, 0)

    def allocations(self) -> tuple[GraphAllocation, ...]:
        return ()


class NativeCudaGraphController:
    """Own graph-private CUDA backing without owning graph executables."""

    enabled = True
    capabilities = ProviderCapabilities(
        provider="coldsnap_cuda_vmm_graph",
        tagged_regions=True,
        stable_virtual_addresses=True,
        scoped_interception=True,
        allocation_enumeration=True,
        external_snapshot_store=False,
        graph_regions=True,
        per_device_synchronization=True,
        native_backing_stores=frozenset({"discard"}),
        exact_address_reservation=True,
        page_granular_mapping=True,
        release_backing_retain_va=True,
        mapping_recipe_replay=True,
    )

    def __init__(self, library_path: str | os.PathLike[str]) -> None:
        path = Path(library_path).resolve()
        if not path.is_file():
            raise GraphMemoryError(f"native CUDA graph library does not exist: {path}")
        self.library_path = path
        self._library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self._lifecycle_revision = 0
        self._configure_abi()
        if self._library.coldsnap_graph_interposition_active() != 1:
            raise GraphMemoryError(
                f"{path} is loaded but does not own cudaMalloc interposition; "
                "start the process with this exact path in LD_PRELOAD"
            )

    def _configure_abi(self) -> None:
        library = self._library
        library.coldsnap_graph_region_push.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        library.coldsnap_graph_region_push.restype = ctypes.c_int
        library.coldsnap_graph_region_pop.argtypes = []
        library.coldsnap_graph_region_pop.restype = ctypes.c_int
        library.coldsnap_graph_pause.argtypes = [ctypes.c_char_p, ctypes.c_int]
        library.coldsnap_graph_pause.restype = ctypes.c_int
        library.coldsnap_graph_resume.argtypes = [ctypes.c_char_p, ctypes.c_int]
        library.coldsnap_graph_resume.restype = ctypes.c_int
        library.coldsnap_graph_region_stats.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        library.coldsnap_graph_region_stats.restype = ctypes.c_int
        library.coldsnap_graph_allocation_at.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
        ]
        library.coldsnap_graph_allocation_at.restype = ctypes.c_int
        library.coldsnap_graph_interposition_active.argtypes = []
        library.coldsnap_graph_interposition_active.restype = ctypes.c_int
        library.coldsnap_graph_last_error.argtypes = []
        library.coldsnap_graph_last_error.restype = ctypes.c_char_p

    def _error(self, operation: str) -> GraphMemoryError:
        value = self._library.coldsnap_graph_last_error()
        detail = value.decode("utf-8", errors="replace") if value else "unknown error"
        return GraphMemoryError(f"native CUDA graph {operation} failed: {detail}")

    @staticmethod
    def _resolve_graph_regions(
        regions: list[str | MemoryRegion] | None,
    ) -> tuple[MemoryRegion, ...]:
        if regions is None:
            return (CUDA_GRAPH,)
        resolved = resolve_regions(regions)
        if any(region.name != CUDA_GRAPH.name for region in resolved):
            names = ", ".join(region.name for region in resolved)
            raise GraphMemoryError(
                f"native CUDA graph controller only manages cuda_graph, got {names}"
            )
        return resolved

    @contextmanager
    def capture_region(
        self,
        *,
        pool_id: int = 0,
        device: int | None = None,
    ) -> Iterator[None]:
        resolved_device = -1 if device is None else int(device)
        result = self._library.coldsnap_graph_region_push(
            CUDA_GRAPH.name.encode(), resolved_device, int(pool_id)
        )
        if result != 0:
            raise self._error("region push")
        try:
            yield
        finally:
            if self._library.coldsnap_graph_region_pop() != 0:
                raise self._error("region pop")

    def release(self, regions: list[str | MemoryRegion] | None = None) -> None:
        self._resolve_graph_regions(regions)
        if self._library.coldsnap_graph_pause(CUDA_GRAPH.name.encode(), -1) != 0:
            raise self._error("pause")
        self._lifecycle_revision += 1

    def restore(self, regions: list[str | MemoryRegion] | None = None) -> None:
        self._resolve_graph_regions(regions)
        if self._library.coldsnap_graph_resume(CUDA_GRAPH.name.encode(), -1) != 0:
            raise self._error("resume")
        self._lifecycle_revision += 1

    def stats(self) -> GraphRegionStats:
        count = ctypes.c_uint64()
        raw = ctypes.c_uint64()
        mapped = ctypes.c_uint64()
        paused = ctypes.c_uint64()
        result = self._library.coldsnap_graph_region_stats(
            CUDA_GRAPH.name.encode(),
            -1,
            ctypes.byref(count),
            ctypes.byref(raw),
            ctypes.byref(mapped),
            ctypes.byref(paused),
        )
        if result != 0:
            raise self._error("stats")
        return GraphRegionStats(count.value, raw.value, mapped.value, paused.value)

    def allocations(self) -> tuple[GraphAllocation, ...]:
        stats = self.stats()
        result: list[GraphAllocation] = []
        for index in range(stats.allocation_count):
            address = ctypes.c_size_t()
            raw = ctypes.c_uint64()
            mapped = ctypes.c_uint64()
            device = ctypes.c_int()
            pool_id = ctypes.c_uint64()
            state = ctypes.c_int()
            status = self._library.coldsnap_graph_allocation_at(
                CUDA_GRAPH.name.encode(),
                -1,
                index,
                ctypes.byref(address),
                ctypes.byref(raw),
                ctypes.byref(mapped),
                ctypes.byref(device),
                ctypes.byref(pool_id),
                ctypes.byref(state),
            )
            if status != 0:
                raise self._error("allocation enumeration")
            result.append(
                GraphAllocation(
                    address=address.value,
                    raw_bytes=raw.value,
                    mapped_bytes=mapped.value,
                    device=device.value,
                    pool_id=pool_id.value,
                    state=state.value,
                )
            )
        return tuple(result)

    def describe_layout(
        self, region: str | MemoryRegion = CUDA_GRAPH
    ) -> MemoryLayoutDescriptor:
        self._resolve_graph_regions([region])
        values = tuple(sorted(self.allocations(), key=lambda item: item.address))
        stats = self.stats()
        return MemoryLayoutDescriptor.create(
            provider=self.capabilities.provider,
            provider_abi=1,
            region=CUDA_GRAPH,
            device="cuda:all",
            base_address=min((item.address for item in values), default=None),
            logical_bytes=stats.raw_bytes,
            allocation_granularity=2 * 1024 * 1024,
            mapping_granularity=2 * 1024 * 1024,
            layout_kind=MemoryLayoutKind.OPAQUE,
            groups=tuple(sorted({
                f"device-{item.device}:pool-{item.pool_id}" for item in values
            })),
            extent_count=len(values),
            extension={
                "allocations": [
                    {
                        "address": item.address,
                        "raw_bytes": item.raw_bytes,
                        "mapped_bytes": item.mapped_bytes,
                        "device": item.device,
                        "pool_id": item.pool_id,
                    }
                    for item in values
                ]
            },
        )

    def residency_stats(
        self, region: str | MemoryRegion = CUDA_GRAPH
    ) -> MemoryResidencyStats:
        self._resolve_graph_regions([region])
        stats = self.stats()
        page_bytes = 2 * 1024 * 1024
        result = MemoryResidencyStats(
            format=1,
            provider=self.capabilities.provider,
            region=CUDA_GRAPH.name,
            logical_bytes=stats.raw_bytes,
            mapped_physical_bytes=stats.mapped_bytes,
            reserved_physical_bytes=0,
            preserved_bytes=0,
            discardable_bytes=0,
            externally_hydrated_bytes=0,
            allocation_count=stats.allocation_count,
            page_count=(stats.mapped_bytes + page_bytes - 1) // page_bytes,
            released_mapping_count=stats.paused_count,
            lifecycle_revision=self._lifecycle_revision,
            lifecycle_epoch=f"cuda-graph-vmm-{self._lifecycle_revision}",
        )
        result.validate()
        return result


def graph_controller_from_env(
    *, required: bool = False
) -> NativeCudaGraphController | DisabledCudaGraphController:
    value = os.environ.get(GRAPH_LIBRARY_ENV)
    if not value:
        if required:
            raise GraphMemoryError(f"{GRAPH_LIBRARY_ENV} is required")
        return DisabledCudaGraphController()
    return NativeCudaGraphController(value)

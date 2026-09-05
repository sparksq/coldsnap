# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""vLLM implementation of ColdSnap's engine-neutral allocation-provider API."""

from __future__ import annotations

from typing import Any

from coldsnap_core.memory import (
    HostStage,
    MemoryCapabilityError,
    MemoryLayoutDescriptor,
    MemoryLayoutKind,
    MemoryRegion,
    MemoryResidencyStats,
    PreservationPolicy,
    ProviderCapabilities,
    resolve_regions,
)
from coldsnap_vllm import (
    Allocation,
    allocations,
    cuda_runtime,
    get_allocator,
    map_allocation,
    pin_memory_enabled,
    unmap_allocation,
)


class VllmMemoryProvider:
    """Expose vLLM's tagged CuMem pool without leaking it into snapshot code."""

    capabilities = ProviderCapabilities(
        provider="vllm_cumem",
        tagged_regions=True,
        stable_virtual_addresses=True,
        scoped_interception=False,
        allocation_enumeration=True,
        external_snapshot_store=True,
        graph_regions=False,
        per_device_synchronization=True,
        native_backing_stores=frozenset({"discard"}),
        exact_address_reservation=True,
        page_granular_mapping=True,
        release_backing_retain_va=True,
        mapping_recipe_replay=True,
    )

    def __init__(self, allocator: Any | None = None) -> None:
        self._allocator = allocator
        self._lifecycle_revision = 0

    def _resolve_allocator(self) -> Any:
        if self._allocator is None:
            self._allocator = get_allocator()
        return self._allocator

    def allocations(self, tag: str | None = None) -> tuple[Allocation, ...]:
        return allocations(self._resolve_allocator(), tag)

    def select_allocations(
        self, region: str, purpose: str
    ) -> tuple[Allocation, ...]:
        """Resolve an engine-semantic subset of a tagged allocation region."""
        if region != "kv_cache" or purpose != "payload":
            raise ValueError(
                f"vLLM provider cannot select purpose {purpose!r} from "
                f"region {region!r}"
            )
        from coldsnap_vllm_kv_payload import kv_payload_extents

        extents = kv_payload_extents()
        if not extents:
            raise RuntimeError(
                "vLLM KV payload inventory is empty; refusing heuristic discard"
            )
        allocated = self.allocations(region)
        selected: dict[int, Allocation] = {}
        for extent in extents:
            matches = tuple(
                allocation
                for allocation in allocated
                if allocation.pointer <= extent.pointer
                and extent.pointer + extent.size
                <= allocation.pointer + allocation.size
            )
            if len(matches) != 1:
                raise RuntimeError(
                    "vLLM KV payload extent must resolve to exactly one tagged "
                    f"allocation: pointer={extent.pointer:#x} size={extent.size} "
                    f"matches={len(matches)}"
                )
            selected[matches[0].pointer] = matches[0]
        return tuple(selected[pointer] for pointer in sorted(selected))

    def allocate_host_stage(self, size: int) -> HostStage:
        if size <= 0:
            raise ValueError("host stage size must be positive")
        import torch

        tensor = torch.empty(
            size,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory_enabled(),
        )
        return HostStage(
            owner=tensor,
            view=memoryview(tensor.numpy()).cast("B"),
            pointer=tensor.data_ptr(),
        )

    def copy_to_host(
        self, destination: HostStage, source_pointer: int, size: int
    ) -> None:
        self._validate_copy(destination, size)
        cuda_runtime().cudaMemcpy(destination.pointer, source_pointer, size)

    def copy_from_host(
        self, destination_pointer: int, source: HostStage, size: int
    ) -> None:
        self._validate_copy(source, size)
        cuda_runtime().cudaMemcpy(destination_pointer, source.pointer, size)

    @staticmethod
    def _validate_copy(stage: HostStage, size: int) -> None:
        if size <= 0 or size > len(stage.view):
            raise ValueError(
                f"copy size {size} exceeds host stage capacity {len(stage.view)}"
            )

    def fill_zero(self, pointer: int, size: int) -> None:
        """Define the contents of freshly mapped device memory.

        Remapping a released region hands back pages with undefined content.
        Engines differ on whether they reinitialize it: vLLM's v1 runner zeroes
        the KV cache in post_kv_cache_wake_up, its v2 runner only rebuilds block
        table layouts. Whoever discarded the bytes owns leaving them defined, so
        ColdSnap does it rather than depending on the engine.
        """
        if size <= 0:
            raise ValueError("fill size must be positive")
        cuda_runtime().cudaMemset(pointer, 0, size)

    def release(self, allocation: Allocation) -> None:
        try:
            unmap_allocation(allocation.handle)
        finally:
            allocation.set_released(True)
            self._lifecycle_revision += 1

    def remap(self, allocation: Allocation) -> None:
        map_allocation(allocation.handle)
        self._lifecycle_revision += 1

    @staticmethod
    def _region(value: str | MemoryRegion) -> MemoryRegion:
        return resolve_regions([value])[0]

    def describe_layout(self, region: str | MemoryRegion) -> MemoryLayoutDescriptor:
        resolved = self._region(region)
        values = tuple(sorted(self.allocations(resolved.name), key=lambda item: item.pointer))
        if not values:
            raise MemoryCapabilityError(
                f"vLLM CuMem has no allocations for region {resolved.name!r}"
            )
        granularity = 2 * 1024 * 1024
        return MemoryLayoutDescriptor.create(
            provider=self.capabilities.provider,
            provider_abi=1,
            region=resolved,
            device="cuda:current",
            base_address=values[0].pointer,
            logical_bytes=sum(item.size for item in values),
            allocation_granularity=granularity,
            mapping_granularity=granularity,
            layout_kind=MemoryLayoutKind.CONTIGUOUS,
            extent_count=len(values),
            extension={
                "extents": [
                    {"offset": item.pointer - values[0].pointer, "bytes": item.size}
                    for item in values
                ]
            },
        )

    def residency_stats(self, region: str | MemoryRegion) -> MemoryResidencyStats:
        resolved = self._region(region)
        values = self.allocations(resolved.name)
        logical = sum(item.size for item in values)
        mapped = sum(item.size for item in values if not item.is_released)
        page_bytes = 2 * 1024 * 1024
        result = MemoryResidencyStats(
            format=1,
            provider=self.capabilities.provider,
            region=resolved.name,
            logical_bytes=logical,
            mapped_physical_bytes=mapped,
            reserved_physical_bytes=0,
            preserved_bytes=mapped if resolved.policy is PreservationPolicy.PRESERVE else 0,
            discardable_bytes=mapped if resolved.policy is PreservationPolicy.DISCARD else 0,
            externally_hydrated_bytes=0,
            allocation_count=len(values),
            page_count=sum((item.size + page_bytes - 1) // page_bytes for item in values),
            released_mapping_count=sum(item.is_released for item in values),
            lifecycle_revision=self._lifecycle_revision,
            lifecycle_epoch=f"vllm-cumem-{self._lifecycle_revision}",
        )
        result.validate()
        return result

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()

    def empty_cache(self) -> None:
        import torch

        torch.cuda.empty_cache()

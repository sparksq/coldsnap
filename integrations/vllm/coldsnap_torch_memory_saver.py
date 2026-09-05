# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Optional torch_memory_saver region controller for PyTorch engines."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any, Iterator

from coldsnap_core.memory import (
    CUDA_GRAPH,
    MemoryCapabilityError,
    MemoryRegion,
    NoopRegionController,
    PreservationPolicy,
    ProviderCapabilities,
    resolve_regions,
)


class TorchMemorySaverRegionController:
    """Use TMS for semantic regions without claiming external snapshot support."""

    enabled = True

    @classmethod
    def create(
        cls,
        enabled: bool,
        *,
        saver: Any | None = None,
        graph_regions: bool = False,
    ) -> TorchMemorySaverRegionController | NoopRegionController:
        if not enabled:
            return NoopRegionController()
        return cls(saver=saver, graph_regions=graph_regions)

    def __init__(
        self, *, saver: Any | None = None, graph_regions: bool = False
    ) -> None:
        if saver is None:
            try:
                import torch_memory_saver
            except ImportError as error:
                raise MemoryCapabilityError(
                    "torch_memory_saver is required for the TMS region controller"
                ) from error
            saver = torch_memory_saver.torch_memory_saver
        self._saver = saver
        try:
            region_parameters = inspect.signature(saver.region).parameters
        except (TypeError, ValueError) as error:
            raise MemoryCapabilityError(
                "cannot inspect torch_memory_saver.region"
            ) from error
        native_stores = {"discard", "cpu"}
        if "enable_disk_backup" in region_parameters:
            native_stores.add("disk")
        self.capabilities = ProviderCapabilities(
            provider="torch_memory_saver",
            tagged_regions=True,
            stable_virtual_addresses=True,
            scoped_interception=True,
            allocation_enumeration=False,
            external_snapshot_store=False,
            graph_regions=graph_regions,
            per_device_synchronization=True,
            native_backing_stores=frozenset(native_stores),
        )
        self._region_backing: dict[str, str] = {}

    @staticmethod
    def _resolve_region(value: str | MemoryRegion) -> MemoryRegion:
        return resolve_regions([value])[0]

    def _validate_backing(self, region: MemoryRegion, backing: str) -> None:
        if region.policy is PreservationPolicy.EXTERNAL:
            raise MemoryCapabilityError(
                f"region {region.name!r} is externally owned and must not enter "
                "torch_memory_saver"
            )
        if backing not in self.capabilities.native_backing_stores:
            raise MemoryCapabilityError(
                f"torch_memory_saver backing store {backing!r} is unavailable; "
                f"supported stores: {', '.join(sorted(self.capabilities.native_backing_stores))}"
            )
        if region.policy is PreservationPolicy.PRESERVE and backing == "discard":
            raise MemoryCapabilityError(
                f"region {region.name!r} requires preserved contents; select the "
                "cpu or disk native backing store, or an external snapshot provider"
            )

    @contextmanager
    def region(
        self,
        region: str | MemoryRegion,
        *,
        native_backing_store: str = "discard",
    ) -> Iterator[None]:
        resolved = self._resolve_region(region)
        self._validate_backing(resolved, native_backing_store)
        kwargs = {
            "tag": resolved.name,
            "enable_cpu_backup": native_backing_store == "cpu",
        }
        if "disk" in self.capabilities.native_backing_stores:
            kwargs["enable_disk_backup"] = native_backing_store == "disk"
        previous = self._region_backing.get(resolved.name)
        if previous is not None and previous != native_backing_store:
            raise MemoryCapabilityError(
                f"region {resolved.name!r} already uses backing store {previous!r}"
            )
        self._region_backing[resolved.name] = native_backing_store
        with self._saver.region(**kwargs):
            yield

    @contextmanager
    def cuda_graph(
        self,
        cuda_graph: Any,
        *,
        pool: Any = None,
        stream: Any = None,
        capture_error_mode: str = "global",
        region: str | MemoryRegion = CUDA_GRAPH,
    ) -> Iterator[None]:
        self.capabilities.require("graph_regions", "manage CUDA graph allocations")
        resolved = self._resolve_region(region)
        if resolved.policy is PreservationPolicy.EXTERNAL:
            raise MemoryCapabilityError("external CUDA graph memory cannot be managed")
        with self._saver.cuda_graph(
            cuda_graph,
            pool=pool,
            stream=stream,
            capture_error_mode=capture_error_mode,
            tag=resolved.name,
            enable_cpu_backup=False,
        ):
            yield

    def release(self, regions: list[str | MemoryRegion] | None = None) -> None:
        if regions is None:
            self._saver.pause()
            return
        for region in resolve_regions(regions):
            if region.policy is PreservationPolicy.EXTERNAL:
                raise MemoryCapabilityError(
                    f"cannot release externally owned region {region.name!r}"
                )
            self._saver.pause(region.name)

    def restore(self, regions: list[str | MemoryRegion] | None = None) -> None:
        if regions is None:
            self._saver.resume()
            return
        for region in resolve_regions(regions):
            if region.policy is PreservationPolicy.EXTERNAL:
                raise MemoryCapabilityError(
                    f"cannot restore externally owned region {region.name!r}"
                )
            self._saver.resume(region.name)

    def snapshot_allocations(self, region: str | MemoryRegion) -> None:
        resolved = self._resolve_region(region)
        raise MemoryCapabilityError(
            f"torch_memory_saver does not expose allocation enumeration for "
            f"{resolved.name!r}; use its native cpu/disk backing or extend its "
            "provider ABI before attaching the ColdSnap snapshot store"
        )

    def describe_layout(self, region: str | MemoryRegion) -> None:
        resolved = self._resolve_region(region)
        raise MemoryCapabilityError(
            "torch_memory_saver does not expose a deterministic allocation "
            f"layout for {resolved.name!r}; descriptive layout is unavailable"
        )

    def residency_stats(self, region: str | MemoryRegion) -> None:
        resolved = self._resolve_region(region)
        raise MemoryCapabilityError(
            "torch_memory_saver does not expose mapped/reserved byte accounting "
            f"for {resolved.name!r}; residency statistics are unavailable"
        )

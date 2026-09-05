# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Engine-neutral contracts for live GPU-memory hibernation.

Inference engines decide what allocations mean and when they are safe to
release. Memory providers own allocation discovery and virtual-memory
operations. Snapshot stores preserve bytes. Keeping those responsibilities
separate lets ColdSnap support multiple engines without pretending their
allocator and lifecycle internals are interchangeable.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterator, Protocol, runtime_checkable


class MemoryCapabilityError(RuntimeError):
    """A provider cannot safely perform the requested memory operation."""


class PreservationPolicy(str, Enum):
    """What must happen to a region's contents across release and restore."""

    PRESERVE = "preserve"
    DISCARD = "discard"
    RECREATE = "recreate"
    EXTERNAL = "external"


class MemoryLayoutKind(str, Enum):
    """Bounded provider layout shapes understood by the core contract."""

    CONTIGUOUS = "contiguous"
    PER_LAYER = "per-layer"
    UNIFIED = "unified"
    OPAQUE = "opaque"


class MemoryCheckpointAction(str, Enum):
    """Bounded physical-lifecycle actions in an immutable checkpoint plan."""

    RETAIN = "retain"
    RELEASE_BACKING = "release-backing"
    UNMAP_PAGES = "unmap-pages"
    REMAP_PAGES = "remap-pages"
    REPLAY_MAPPING = "replay-mapping"


class MemoryTransportAction(str, Enum):
    """Bounded registration actions owned by a memory provider."""

    RETAIN = "retain"
    DETACH_REGISTRATIONS = "detach-registrations"
    REATTACH_REGISTRATIONS = "reattach-registrations"


@dataclass(frozen=True)
class MemoryRegion:
    """Stable semantic identity and lifecycle policy for engine allocations."""

    name: str
    policy: PreservationPolicy
    stable_address: bool = True

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name or any(
            character.isspace() for character in self.name
        ):
            raise ValueError(f"invalid memory region name {self.name!r}")
        if self.policy is PreservationPolicy.EXTERNAL and self.stable_address:
            raise ValueError("external memory cannot promise provider-owned addresses")


WEIGHTS = MemoryRegion("weights", PreservationPolicy.PRESERVE)
KV_CACHE = MemoryRegion("kv_cache", PreservationPolicy.DISCARD)
CUDA_GRAPH = MemoryRegion("cuda_graph", PreservationPolicy.RECREATE)
WORKSPACE = MemoryRegion("workspace", PreservationPolicy.RECREATE)
EXTERNAL = MemoryRegion(
    "external", PreservationPolicy.EXTERNAL, stable_address=False
)

STANDARD_REGIONS = MappingProxyType(
    {
        region.name: region
        for region in (WEIGHTS, KV_CACHE, CUDA_GRAPH, WORKSPACE, EXTERNAL)
    }
)


def resolve_regions(values: list[str | MemoryRegion] | None) -> tuple[MemoryRegion, ...]:
    """Resolve names without changing caller order or accepting duplicates."""
    if values is None:
        return tuple(STANDARD_REGIONS.values())
    result: list[MemoryRegion] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, MemoryRegion):
            region = value
        else:
            try:
                region = STANDARD_REGIONS[value]
            except KeyError as error:
                raise MemoryCapabilityError(
                    f"unknown memory region {value!r}; expected one of "
                    + ", ".join(STANDARD_REGIONS)
                ) from error
        if region.name in seen:
            raise ValueError(f"duplicate memory region {region.name!r}")
        seen.add(region.name)
        result.append(region)
    return tuple(result)


@dataclass(frozen=True)
class ProviderCapabilities:
    """Features an allocation or region provider can actually guarantee."""

    provider: str
    tagged_regions: bool
    stable_virtual_addresses: bool
    scoped_interception: bool
    allocation_enumeration: bool
    external_snapshot_store: bool
    graph_regions: bool
    per_device_synchronization: bool
    provider_abi: int = 1
    native_backing_stores: frozenset[str] = frozenset()
    exact_address_reservation: bool = False
    page_granular_mapping: bool = False
    batched_mapping: bool = False
    release_backing_retain_va: bool = False
    sentinel_backing: bool = False
    distributed_atomic_mapping: bool = False
    revisioned_limits: bool = False
    mapping_recipe_replay: bool = False
    transport_registration_ownership: bool = False
    background_reserve: bool = False
    checkpoint_quiescence: bool = False

    def require(self, attribute: str, operation: str) -> None:
        if not bool(getattr(self, attribute, False)):
            raise MemoryCapabilityError(
                f"provider {self.provider!r} cannot {operation}: "
                f"capability {attribute!r} is unavailable"
            )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_digest(value: dict[str, Any]) -> str:
    payload = _canonical_json(value).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MemoryLayoutDescriptor:
    """Deterministic description of one provider-owned semantic region."""

    format: int
    provider: str
    provider_abi: int
    region: MemoryRegion
    device: str
    base_address: int | None
    logical_bytes: int
    allocation_granularity: int
    mapping_granularity: int
    layout_kind: MemoryLayoutKind
    groups: tuple[str, ...]
    extent_count: int
    transport_registration_mode: str
    extension_json: str
    layout_digest: str

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        provider_abi: int,
        region: MemoryRegion,
        device: str,
        base_address: int | None,
        logical_bytes: int,
        allocation_granularity: int,
        mapping_granularity: int,
        layout_kind: MemoryLayoutKind,
        groups: tuple[str, ...] = (),
        extent_count: int,
        transport_registration_mode: str = "none",
        extension: dict[str, Any] | None = None,
    ) -> MemoryLayoutDescriptor:
        extension_value = dict(extension or {})
        extension_json = _canonical_json(extension_value)
        if len(extension_json.encode("ascii")) > 1024 * 1024:
            raise ValueError("memory layout extension exceeds 1 MiB")
        fields = {
            "format": 1,
            "provider": provider,
            "provider_abi": provider_abi,
            "region": {
                "name": region.name,
                "policy": region.policy.value,
                "stable_address": region.stable_address,
            },
            "device": device,
            "base_address": base_address,
            "logical_bytes": logical_bytes,
            "allocation_granularity": allocation_granularity,
            "mapping_granularity": mapping_granularity,
            "layout_kind": layout_kind.value,
            "groups": list(groups),
            "extent_count": extent_count,
            "transport_registration_mode": transport_registration_mode,
            "extension": extension_value,
        }
        result = cls(
            format=1,
            provider=provider,
            provider_abi=provider_abi,
            region=region,
            device=device,
            base_address=base_address,
            logical_bytes=logical_bytes,
            allocation_granularity=allocation_granularity,
            mapping_granularity=mapping_granularity,
            layout_kind=layout_kind,
            groups=groups,
            extent_count=extent_count,
            transport_registration_mode=transport_registration_mode,
            extension_json=extension_json,
            layout_digest=_canonical_digest(fields),
        )
        result.validate()
        return result

    def validate(self) -> None:
        try:
            extension = json.loads(self.extension_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("memory layout extension is invalid JSON") from error
        if (
            self.format != 1
            or not self.provider
            or self.provider_abi <= 0
            or not self.device
            or (self.base_address is not None and self.base_address < 0)
            or (
                self.region.stable_address
                and self.logical_bytes > 0
                and self.base_address is None
            )
            or self.logical_bytes < 0
            or self.allocation_granularity <= 0
            or self.mapping_granularity <= 0
            or self.extent_count < 0
            or (self.logical_bytes == 0) != (self.extent_count == 0)
            or not self.transport_registration_mode
            or len(set(self.groups)) != len(self.groups)
            or any(not group for group in self.groups)
            or not isinstance(extension, dict)
            or self.extension_json != _canonical_json(extension)
            or len(self.extension_json.encode("ascii")) > 1024 * 1024
        ):
            raise ValueError("memory layout descriptor is invalid")
        fields = {
            "format": self.format,
            "provider": self.provider,
            "provider_abi": self.provider_abi,
            "region": {
                "name": self.region.name,
                "policy": self.region.policy.value,
                "stable_address": self.region.stable_address,
            },
            "device": self.device,
            "base_address": self.base_address,
            "logical_bytes": self.logical_bytes,
            "allocation_granularity": self.allocation_granularity,
            "mapping_granularity": self.mapping_granularity,
            "layout_kind": self.layout_kind.value,
            "groups": list(self.groups),
            "extent_count": self.extent_count,
            "transport_registration_mode": self.transport_registration_mode,
            "extension": extension,
        }
        if self.layout_digest != _canonical_digest(fields):
            raise ValueError("memory layout digest does not match descriptor")

    @property
    def extension(self) -> dict[str, Any]:
        """Return a defensive copy of the bounded engine-specific payload."""

        value = json.loads(self.extension_json)
        if not isinstance(value, dict):
            raise ValueError("memory layout extension is not an object")
        return value


@dataclass(frozen=True)
class MemoryResidencyStats:
    format: int
    provider: str
    region: str
    logical_bytes: int
    mapped_physical_bytes: int
    reserved_physical_bytes: int
    preserved_bytes: int
    discardable_bytes: int
    externally_hydrated_bytes: int
    allocation_count: int
    page_count: int
    released_mapping_count: int
    lifecycle_revision: int
    lifecycle_epoch: str

    def validate(self) -> None:
        numeric = (
            self.logical_bytes,
            self.mapped_physical_bytes,
            self.reserved_physical_bytes,
            self.preserved_bytes,
            self.discardable_bytes,
            self.externally_hydrated_bytes,
            self.allocation_count,
            self.page_count,
            self.released_mapping_count,
            self.lifecycle_revision,
        )
        if (
            self.format != 1
            or not self.provider
            or not self.region
            or not self.lifecycle_epoch
            or any(value < 0 for value in numeric)
            or self.released_mapping_count > self.allocation_count
            or (
                self.preserved_bytes
                + self.discardable_bytes
                + self.externally_hydrated_bytes
                > self.mapped_physical_bytes
            )
        ):
            raise ValueError("memory residency statistics are invalid")


@dataclass(frozen=True)
class MemoryCheckpointPlan:
    format: int
    generation: str
    owner: str
    provider: str
    provider_abi: int
    layout_digests: tuple[tuple[str, str], ...]
    policies: tuple[tuple[str, PreservationPolicy], ...]
    actions: tuple[MemoryCheckpointAction, ...]
    mapped_bytes_before: int
    mapped_bytes_after: int
    reserved_bytes_before: int
    reserved_bytes_after: int
    transport_actions: tuple[MemoryTransportAction, ...]
    participants: tuple[str, ...]
    quiescence_evidence: str
    terminal_on_failure: bool
    abort_safe: bool
    plan_digest: str

    @classmethod
    def create(
        cls,
        *,
        generation: str,
        owner: str,
        provider: str,
        provider_abi: int,
        layouts: tuple[MemoryLayoutDescriptor, ...],
        actions: tuple[MemoryCheckpointAction, ...],
        mapped_bytes_before: int,
        mapped_bytes_after: int,
        reserved_bytes_before: int,
        reserved_bytes_after: int,
        transport_actions: tuple[MemoryTransportAction, ...] = (),
        participants: tuple[str, ...],
        quiescence_evidence: str,
        terminal_on_failure: bool,
        abort_safe: bool,
    ) -> MemoryCheckpointPlan:
        for layout in layouts:
            layout.validate()
            if layout.provider != provider or layout.provider_abi != provider_abi:
                raise ValueError(
                    "memory checkpoint layout provider differs from plan provider"
                )
        layout_digests = tuple(sorted((item.region.name, item.layout_digest) for item in layouts))
        policies = tuple(sorted((item.region.name, item.region.policy) for item in layouts))
        fields = {
            "format": 1,
            "generation": generation,
            "owner": owner,
            "provider": provider,
            "provider_abi": provider_abi,
            "layout_digests": layout_digests,
            "policies": tuple((name, policy.value) for name, policy in policies),
            "actions": tuple(action.value for action in actions),
            "mapped_bytes_before": mapped_bytes_before,
            "mapped_bytes_after": mapped_bytes_after,
            "reserved_bytes_before": reserved_bytes_before,
            "reserved_bytes_after": reserved_bytes_after,
            "transport_actions": tuple(action.value for action in transport_actions),
            "participants": tuple(sorted(participants)),
            "quiescence_evidence": quiescence_evidence,
            "terminal_on_failure": terminal_on_failure,
            "abort_safe": abort_safe,
        }
        plan = cls(
            format=1,
            generation=generation,
            owner=owner,
            provider=provider,
            provider_abi=provider_abi,
            layout_digests=layout_digests,
            policies=policies,
            actions=actions,
            mapped_bytes_before=mapped_bytes_before,
            mapped_bytes_after=mapped_bytes_after,
            reserved_bytes_before=reserved_bytes_before,
            reserved_bytes_after=reserved_bytes_after,
            transport_actions=transport_actions,
            participants=tuple(sorted(participants)),
            quiescence_evidence=quiescence_evidence,
            terminal_on_failure=terminal_on_failure,
            abort_safe=abort_safe,
            plan_digest=_canonical_digest(fields),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if (
            self.format != 1
            or not self.generation
            or not self.owner
            or not self.provider
            or self.provider_abi <= 0
            or not self.layout_digests
            or not self.participants
            or tuple(sorted(set(self.participants))) != self.participants
            or self.owner not in self.participants
            or not self.quiescence_evidence
            or not self.actions
            or tuple(sorted(set(self.layout_digests))) != self.layout_digests
            or tuple(sorted(set(self.policies))) != self.policies
            or len({name for name, _ in self.layout_digests})
            != len(self.layout_digests)
            or len({name for name, _ in self.policies}) != len(self.policies)
            or {name for name, _ in self.layout_digests}
            != {name for name, _ in self.policies}
            or any(
                not isinstance(action, MemoryCheckpointAction)
                for action in self.actions
            )
            or any(
                not isinstance(action, MemoryTransportAction)
                for action in self.transport_actions
            )
            or len(set(self.actions)) != len(self.actions)
            or len(set(self.transport_actions)) != len(self.transport_actions)
            or any(
                value < 0
                for value in (
                    self.mapped_bytes_before,
                    self.mapped_bytes_after,
                    self.reserved_bytes_before,
                    self.reserved_bytes_after,
                )
            )
        ):
            raise ValueError("memory checkpoint plan is invalid")
        fields = {
            "format": self.format,
            "generation": self.generation,
            "owner": self.owner,
            "provider": self.provider,
            "provider_abi": self.provider_abi,
            "layout_digests": self.layout_digests,
            "policies": tuple((name, policy.value) for name, policy in self.policies),
            "actions": tuple(action.value for action in self.actions),
            "mapped_bytes_before": self.mapped_bytes_before,
            "mapped_bytes_after": self.mapped_bytes_after,
            "reserved_bytes_before": self.reserved_bytes_before,
            "reserved_bytes_after": self.reserved_bytes_after,
            "transport_actions": tuple(
                action.value for action in self.transport_actions
            ),
            "participants": self.participants,
            "quiescence_evidence": self.quiescence_evidence,
            "terminal_on_failure": self.terminal_on_failure,
            "abort_safe": self.abort_safe,
        }
        if self.plan_digest != _canonical_digest(fields):
            raise ValueError("memory checkpoint plan digest does not match plan")


def require_checkpoint_plan(
    provider: Any,
    regions: list[str | MemoryRegion] | None,
    generation: str,
) -> MemoryCheckpointPlan:
    """Prepare a plan only after proving the optional lifecycle ABI exists."""

    capabilities = provider.capabilities
    capabilities.require("checkpoint_quiescence", "prepare a memory checkpoint")
    operation = getattr(provider, "prepare_checkpoint", None)
    if not callable(operation):
        raise MemoryCapabilityError(
            f"provider {capabilities.provider!r} advertises checkpoint_quiescence "
            "but has no prepare_checkpoint operation"
        )
    plan = operation(resolve_regions(regions), generation)
    if not isinstance(plan, MemoryCheckpointPlan):
        raise MemoryCapabilityError("memory provider returned an invalid checkpoint plan")
    plan.validate()
    if (
        plan.provider != capabilities.provider
        or plan.provider_abi != capabilities.provider_abi
    ):
        raise MemoryCapabilityError(
            "memory checkpoint plan provider identity differs from active provider"
        )
    if plan.generation != generation:
        raise MemoryCapabilityError(
            "memory checkpoint plan generation differs from request"
        )
    resolved_regions = resolve_regions(regions)
    expected_regions = {region.name for region in resolved_regions}
    planned_regions = {name for name, _ in plan.layout_digests}
    if planned_regions != expected_regions:
        raise MemoryCapabilityError(
            "memory checkpoint plan region inventory differs from request"
        )
    if dict(plan.policies) != {
        region.name: region.policy for region in resolved_regions
    }:
        raise MemoryCapabilityError(
            "memory checkpoint plan policies differ from requested regions"
        )
    action_requirements = {
        MemoryCheckpointAction.RELEASE_BACKING: (
            "release_backing_retain_va",
        ),
        MemoryCheckpointAction.UNMAP_PAGES: ("page_granular_mapping",),
        MemoryCheckpointAction.REMAP_PAGES: (
            "page_granular_mapping",
            "exact_address_reservation",
        ),
        MemoryCheckpointAction.REPLAY_MAPPING: (
            "mapping_recipe_replay",
            "exact_address_reservation",
        ),
    }
    for action in plan.actions:
        for capability in action_requirements.get(action, ()):
            capabilities.require(capability, f"execute checkpoint action {action.value}")
    if any(
        action is not MemoryTransportAction.RETAIN
        for action in plan.transport_actions
    ):
        capabilities.require(
            "transport_registration_ownership",
            "execute checkpoint transport-registration actions",
        )
    if len(plan.participants) > 1 and any(
        action is not MemoryCheckpointAction.RETAIN for action in plan.actions
    ):
        capabilities.require(
            "distributed_atomic_mapping",
            "coordinate a distributed memory checkpoint",
        )
    return plan


def agree_checkpoint_plans(plans: tuple[MemoryCheckpointPlan, ...]) -> str:
    """Fail before release unless all ranks accepted one immutable plan."""

    if not plans:
        raise MemoryCapabilityError("distributed checkpoint has no participant plans")
    for plan in plans:
        plan.validate()
    first = plans[0]
    common = {
        (
            plan.generation,
            plan.provider,
            plan.provider_abi,
            plan.policies,
            plan.actions,
            plan.transport_actions,
            plan.participants,
            plan.terminal_on_failure,
            plan.abort_safe,
        )
        for plan in plans
    }
    owners = {plan.owner for plan in plans}
    if (
        len(common) != 1
        or len(plans) != len(first.participants)
        or owners != set(first.participants)
    ):
        raise MemoryCapabilityError(
            "distributed memory checkpoint plans disagree on policy, generation, or participants"
        )
    # Per-rank layouts may legitimately differ. Bind every accepted immutable
    # plan into one deterministic group receipt instead of requiring identical
    # per-rank layout digests.
    return _canonical_digest(
        {
            "format": 1,
            "kind": "coldsnap-memory-checkpoint-plan-agreement",
            "plans": sorted((plan.owner, plan.plan_digest) for plan in plans),
        }
    )


@runtime_checkable
class ManagedAllocation(Protocol):
    pointer: int
    size: int
    tag: str

    @property
    def is_released(self) -> bool: ...

    def set_released(self, value: bool) -> None: ...


@dataclass(frozen=True)
class HostStage:
    """Provider-owned host buffer used to stream snapshot chunks."""

    owner: Any
    view: memoryview
    pointer: Any


@runtime_checkable
class SnapshotMemoryProvider(Protocol):
    """Allocator operations required by an external transactional store."""

    capabilities: ProviderCapabilities

    def allocations(self, tag: str | None = None) -> tuple[ManagedAllocation, ...]: ...

    def allocate_host_stage(self, size: int) -> HostStage: ...

    def copy_to_host(
        self, destination: HostStage, source_pointer: int, size: int
    ) -> None: ...

    def copy_from_host(
        self, destination_pointer: int, source: HostStage, size: int
    ) -> None: ...

    def release(self, allocation: ManagedAllocation) -> None: ...

    def remap(self, allocation: ManagedAllocation) -> None: ...

    def synchronize(self) -> None: ...

    def empty_cache(self) -> None: ...


@runtime_checkable
class RegionController(Protocol):
    """Scoped allocation and release API used by engine integrations."""

    capabilities: ProviderCapabilities

    def region(
        self, region: str | MemoryRegion, *, native_backing_store: str = "discard"
    ) -> Any: ...

    def release(self, regions: list[str | MemoryRegion] | None = None) -> None: ...

    def restore(self, regions: list[str | MemoryRegion] | None = None) -> None: ...


@runtime_checkable
class CheckpointMemoryProvider(Protocol):
    """Optional first-party descriptive and transactional memory extension."""

    capabilities: ProviderCapabilities

    def describe_layout(self, region: str | MemoryRegion) -> MemoryLayoutDescriptor: ...

    def residency_stats(self, region: str | MemoryRegion) -> MemoryResidencyStats: ...

    def prepare_checkpoint(
        self, regions: tuple[MemoryRegion, ...], generation: str
    ) -> MemoryCheckpointPlan: ...

    def commit_checkpoint(self, plan: MemoryCheckpointPlan) -> dict[str, Any]: ...

    def restore_checkpoint(self, plan: MemoryCheckpointPlan) -> dict[str, Any]: ...

    def abort_checkpoint(self, plan: MemoryCheckpointPlan) -> dict[str, Any]: ...


class NoopRegionController:
    """Disabled controller that preserves the engine integration call shape."""

    capabilities = ProviderCapabilities(
        provider="noop",
        tagged_regions=False,
        stable_virtual_addresses=False,
        scoped_interception=False,
        allocation_enumeration=False,
        external_snapshot_store=False,
        graph_regions=False,
        per_device_synchronization=False,
    )
    enabled = False

    @contextmanager
    def region(
        self, region: str | MemoryRegion, *, native_backing_store: str = "discard"
    ) -> Iterator[None]:
        del region, native_backing_store
        yield

    def release(self, regions: list[str | MemoryRegion] | None = None) -> None:
        del regions

    def restore(self, regions: list[str | MemoryRegion] | None = None) -> None:
        del regions

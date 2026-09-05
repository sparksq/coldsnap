# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

import coldsnap_vllm_memory as vllm_memory  # noqa: E402
from coldsnap_disk_backend import DiskCuMemBackend  # noqa: E402
from coldsnap_core.memory import (  # noqa: E402
    CUDA_GRAPH,
    EXTERNAL,
    KV_CACHE,
    STANDARD_REGIONS,
    WEIGHTS,
    WORKSPACE,
    HostStage,
    MemoryCheckpointAction,
    MemoryCheckpointPlan,
    MemoryCapabilityError,
    MemoryLayoutDescriptor,
    MemoryLayoutKind,
    MemoryRegion,
    MemoryTransportAction,
    NoopRegionController,
    PreservationPolicy,
    agree_checkpoint_plans,
    require_checkpoint_plan,
    resolve_regions,
)
from coldsnap_torch_memory_saver import (  # noqa: E402
    TorchMemorySaverRegionController,
)
from coldsnap_vllm_memory import VllmMemoryProvider  # noqa: E402


class FakeTorchMemorySaver:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    @contextmanager
    def region(
        self,
        *,
        tag: str,
        enable_cpu_backup: bool = False,
        enable_disk_backup: bool = False,
    ):
        self.events.append(
            ("region-enter", tag, enable_cpu_backup, enable_disk_backup)
        )
        yield
        self.events.append(("region-exit", tag))

    @contextmanager
    def cuda_graph(
        self,
        cuda_graph,
        *,
        pool=None,
        stream=None,
        capture_error_mode: str,
        tag: str,
        enable_cpu_backup: bool,
    ):
        self.events.append(
            (
                "graph-enter",
                cuda_graph,
                pool,
                stream,
                capture_error_mode,
                tag,
                enable_cpu_backup,
            )
        )
        yield
        self.events.append(("graph-exit", tag))

    def pause(self, tag: str | None = None) -> None:
        self.events.append(("pause", tag))

    def resume(self, tag: str | None = None) -> None:
        self.events.append(("resume", tag))


class MemoryContractTest(unittest.TestCase):
    def test_standard_regions_have_explicit_lifecycle_policies(self) -> None:
        self.assertEqual(
            list(STANDARD_REGIONS),
            ["weights", "kv_cache", "cuda_graph", "workspace", "external"],
        )
        self.assertEqual(WEIGHTS.policy, PreservationPolicy.PRESERVE)
        self.assertEqual(KV_CACHE.policy, PreservationPolicy.DISCARD)
        self.assertEqual(CUDA_GRAPH.policy, PreservationPolicy.RECREATE)
        self.assertEqual(WORKSPACE.policy, PreservationPolicy.RECREATE)
        self.assertEqual(EXTERNAL.policy, PreservationPolicy.EXTERNAL)
        self.assertFalse(EXTERNAL.stable_address)

    def test_region_resolution_preserves_order_and_rejects_ambiguity(self) -> None:
        self.assertEqual(
            resolve_regions(["workspace", WEIGHTS]),
            (WORKSPACE, WEIGHTS),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            resolve_regions(["weights", "weights"])
        with self.assertRaisesRegex(MemoryCapabilityError, "unknown memory region"):
            resolve_regions(["not-an-engine-region"])

    def test_external_regions_cannot_claim_provider_owned_stable_addresses(self) -> None:
        with self.assertRaisesRegex(ValueError, "external memory"):
            MemoryRegion("shared", PreservationPolicy.EXTERNAL)

    def test_noop_controller_preserves_integration_call_shape(self) -> None:
        controller = NoopRegionController()
        with controller.region("future-engine-region"):
            pass
        controller.release(["weights"])
        controller.restore(["weights"])
        self.assertFalse(controller.enabled)

    def test_capabilities_fail_closed(self) -> None:
        capabilities = NoopRegionController.capabilities
        with self.assertRaisesRegex(
            MemoryCapabilityError, "external_snapshot_store"
        ):
            capabilities.require(
                "external_snapshot_store", "attach a transactional store"
            )
        with self.assertRaisesRegex(
            MemoryCapabilityError, "external_snapshot_store"
        ):
            DiskCuMemBackend(SimpleNamespace(capabilities=capabilities))

    def test_layout_digest_covers_page_shape_and_extension(self) -> None:
        common = dict(
            provider="test", provider_abi=1, region=WEIGHTS, device="cuda:0",
            base_address=0x100000, logical_bytes=4096,
            allocation_granularity=4096, mapping_granularity=4096,
            layout_kind=MemoryLayoutKind.CONTIGUOUS, extent_count=1,
            extension={"tensor_generation": 2},
        )
        first = MemoryLayoutDescriptor.create(**common)
        second = MemoryLayoutDescriptor.create(**common)
        changed = MemoryLayoutDescriptor.create(
            **{**common, "mapping_granularity": 8192}
        )
        self.assertEqual(first.layout_digest, second.layout_digest)
        self.assertNotEqual(first.layout_digest, changed.layout_digest)
        extension = first.extension
        extension["tensor_generation"] = 99
        self.assertEqual(first.extension, {"tensor_generation": 2})
        first.validate()

    def test_checkpoint_plan_agreement_is_fail_closed(self) -> None:
        layout = MemoryLayoutDescriptor.create(
            provider="test", provider_abi=1, region=KV_CACHE, device="cuda:0",
            base_address=0x200000, logical_bytes=8192,
            allocation_granularity=4096, mapping_granularity=4096,
            layout_kind=MemoryLayoutKind.CONTIGUOUS, extent_count=1,
        )
        arguments = dict(
            generation="capture-1", owner="worker-0",
            provider="test", provider_abi=1,
            layouts=(layout,), actions=(MemoryCheckpointAction.RELEASE_BACKING,),
            mapped_bytes_before=8192,
            mapped_bytes_after=0, reserved_bytes_before=0,
            reserved_bytes_after=0, participants=("worker-0", "worker-1"),
            quiescence_evidence="scheduler-idle", terminal_on_failure=True,
            abort_safe=False,
        )
        first = MemoryCheckpointPlan.create(**arguments)
        second = MemoryCheckpointPlan.create(
            **{**arguments, "owner": "worker-1"}
        )
        agreement = agree_checkpoint_plans((first, second))
        self.assertRegex(agreement, r"^sha256:[0-9a-f]{64}$")
        different = MemoryCheckpointPlan.create(
            **{**arguments, "owner": "worker-1", "generation": "capture-2"}
        )
        with self.assertRaisesRegex(MemoryCapabilityError, "disagree"):
            agree_checkpoint_plans((first, different))

    def test_checkpoint_plan_agreement_accepts_rank_local_layouts(self) -> None:
        participants = ("worker-0", "worker-1")
        plans = []
        for index, owner in enumerate(participants):
            layout = MemoryLayoutDescriptor.create(
                provider="test", provider_abi=1, region=KV_CACHE,
                device=f"cuda:{index}", base_address=0x200000 + index * 0x10000,
                logical_bytes=8192, allocation_granularity=4096,
                mapping_granularity=4096,
                layout_kind=MemoryLayoutKind.CONTIGUOUS, extent_count=1,
            )
            plans.append(MemoryCheckpointPlan.create(
                generation="capture-1", owner=owner, provider="test",
                provider_abi=1, layouts=(layout,),
                actions=(MemoryCheckpointAction.RELEASE_BACKING,),
                mapped_bytes_before=8192, mapped_bytes_after=0,
                reserved_bytes_before=0, reserved_bytes_after=0,
                participants=participants, quiescence_evidence="scheduler-idle",
                terminal_on_failure=True, abort_safe=False,
            ))
        self.assertRegex(
            agree_checkpoint_plans(tuple(plans)), r"^sha256:[0-9a-f]{64}$"
        )

    def test_optional_checkpoint_fails_before_provider_operation(self) -> None:
        provider = SimpleNamespace(
            capabilities=NoopRegionController.capabilities,
            prepare_checkpoint=lambda *_: self.fail("prepare must not run"),
        )
        with self.assertRaisesRegex(MemoryCapabilityError, "checkpoint_quiescence"):
            require_checkpoint_plan(provider, ["kv_cache"], "capture-1")

    def test_checkpoint_plan_action_capability_fails_before_release(self) -> None:
        layout = MemoryLayoutDescriptor.create(
            provider="test", provider_abi=1, region=KV_CACHE, device="cuda:0",
            base_address=0x200000, logical_bytes=8192,
            allocation_granularity=4096, mapping_granularity=4096,
            layout_kind=MemoryLayoutKind.CONTIGUOUS, extent_count=1,
        )
        plan = MemoryCheckpointPlan.create(
            generation="capture-1", owner="worker-0", provider="test",
            provider_abi=1, layouts=(layout,),
            actions=(MemoryCheckpointAction.RELEASE_BACKING,),
            mapped_bytes_before=8192, mapped_bytes_after=0,
            reserved_bytes_before=0, reserved_bytes_after=0,
            transport_actions=(MemoryTransportAction.DETACH_REGISTRATIONS,),
            participants=("worker-0",), quiescence_evidence="scheduler-idle",
            terminal_on_failure=True, abort_safe=False,
        )
        provider = SimpleNamespace(
            capabilities=type(NoopRegionController.capabilities)(
                provider="test", tagged_regions=True,
                stable_virtual_addresses=True, scoped_interception=True,
                allocation_enumeration=True, external_snapshot_store=True,
                graph_regions=False, per_device_synchronization=True,
                checkpoint_quiescence=True,
            ),
            prepare_checkpoint=lambda *_: plan,
        )
        with self.assertRaisesRegex(
            MemoryCapabilityError, "release_backing_retain_va"
        ):
            require_checkpoint_plan(provider, ["kv_cache"], "capture-1")

    def test_checkpoint_plan_rejects_duplicate_region_identity(self) -> None:
        first = MemoryLayoutDescriptor.create(
            provider="test", provider_abi=1, region=KV_CACHE, device="cuda:0",
            base_address=0x200000, logical_bytes=4096,
            allocation_granularity=4096, mapping_granularity=4096,
            layout_kind=MemoryLayoutKind.CONTIGUOUS, extent_count=1,
        )
        second = MemoryLayoutDescriptor.create(
            provider="test", provider_abi=1, region=KV_CACHE, device="cuda:1",
            base_address=0x300000, logical_bytes=4096,
            allocation_granularity=4096, mapping_granularity=4096,
            layout_kind=MemoryLayoutKind.CONTIGUOUS, extent_count=1,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint plan is invalid"):
            MemoryCheckpointPlan.create(
                generation="capture-1", owner="worker-0", provider="test",
                provider_abi=1, layouts=(first, second),
                actions=(MemoryCheckpointAction.RETAIN,),
                mapped_bytes_before=8192, mapped_bytes_after=8192,
                reserved_bytes_before=0, reserved_bytes_after=0,
                participants=("worker-0",),
                quiescence_evidence="scheduler-idle",
                terminal_on_failure=True, abort_safe=False,
            )


class VllmMemoryProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.weight_data = SimpleNamespace(
            tag="weights", is_asleep=False, handle=(101, 64)
        )
        self.kv_data = SimpleNamespace(
            tag="kv_cache", is_asleep=False, handle=(202, 32)
        )
        self.allocator = SimpleNamespace(
            pointer_to_data={1000: self.weight_data, 2000: self.kv_data}
        )
        self.provider = VllmMemoryProvider(self.allocator)

    def test_provider_enumerates_semantic_allocations(self) -> None:
        weights = self.provider.allocations("weights")
        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0].pointer, 1000)
        self.assertEqual(weights[0].size, 64)
        self.assertTrue(self.provider.capabilities.stable_virtual_addresses)
        self.assertTrue(self.provider.capabilities.external_snapshot_store)
        self.assertTrue(self.provider.capabilities.per_device_synchronization)

    def test_provider_describes_deterministic_layout_and_residency(self) -> None:
        layout = self.provider.describe_layout("weights")
        stats = self.provider.residency_stats("weights")
        self.assertEqual(layout.region, WEIGHTS)
        self.assertEqual(layout.logical_bytes, 64)
        self.assertEqual(layout.extent_count, 1)
        self.assertEqual(stats.mapped_physical_bytes, 64)
        self.assertEqual(stats.preserved_bytes, 64)

    def test_release_and_remap_preserve_address_and_defer_publication(self) -> None:
        allocation = self.provider.allocations("weights")[0]
        with (
            patch.object(vllm_memory, "unmap_allocation") as unmap,
            patch.object(vllm_memory, "map_allocation") as remap,
        ):
            self.provider.release(allocation)
            self.assertTrue(allocation.is_released)
            self.provider.remap(allocation)
            self.assertTrue(allocation.is_released)
        unmap.assert_called_once_with((101, 64))
        remap.assert_called_once_with((101, 64))
        self.assertEqual(allocation.pointer, 1000)

    def test_failed_unmap_is_published_as_released_for_retry(self) -> None:
        allocation = self.provider.allocations("weights")[0]
        with patch.object(
            vllm_memory,
            "unmap_allocation",
            side_effect=RuntimeError("injected unmap failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected unmap failure"):
                self.provider.release(allocation)
        self.assertTrue(allocation.is_released)

    def test_host_copy_is_provider_owned_and_capacity_checked(self) -> None:
        calls: list[tuple[object, object, int]] = []
        runtime = SimpleNamespace(
            cudaMemcpy=lambda destination, source, size: calls.append(
                (destination, source, size)
            )
        )
        stage = HostStage(
            owner=bytearray(8), view=memoryview(bytearray(8)), pointer=123
        )
        with patch.object(vllm_memory, "cuda_runtime", return_value=runtime):
            self.provider.copy_to_host(stage, 456, 8)
            self.provider.copy_from_host(789, stage, 4)
        self.assertEqual(calls, [(123, 456, 8), (789, 123, 4)])
        with self.assertRaisesRegex(ValueError, "exceeds host stage capacity"):
            self.provider.copy_to_host(stage, 456, 9)


class TorchMemorySaverControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saver = FakeTorchMemorySaver()
        self.controller = TorchMemorySaverRegionController(
            saver=self.saver, graph_regions=False
        )

    def test_disabled_controller_has_no_dependency_or_behavior(self) -> None:
        controller = TorchMemorySaverRegionController.create(
            False, saver=self.saver
        )
        self.assertIsInstance(controller, NoopRegionController)

    def test_scoped_regions_map_policy_to_native_backing(self) -> None:
        with self.controller.region("weights", native_backing_store="cpu"):
            self.saver.events.append(("inside", "weights"))
        with self.controller.region("kv_cache"):
            pass
        self.assertEqual(
            self.saver.events,
            [
                ("region-enter", "weights", True, False),
                ("inside", "weights"),
                ("region-exit", "weights"),
                ("region-enter", "kv_cache", False, False),
                ("region-exit", "kv_cache"),
            ],
        )
        self.assertTrue(self.controller.capabilities.scoped_interception)
        self.assertTrue(self.controller.capabilities.stable_virtual_addresses)
        self.assertEqual(
            self.controller.capabilities.native_backing_stores,
            frozenset({"discard", "cpu", "disk"}),
        )

    def test_preserved_region_cannot_silently_discard_contents(self) -> None:
        with self.assertRaisesRegex(
            MemoryCapabilityError, "requires preserved contents"
        ):
            with self.controller.region("weights"):
                pass

    def test_one_region_cannot_change_backing_policy_mid_process(self) -> None:
        with self.controller.region("workspace", native_backing_store="cpu"):
            pass
        with self.assertRaisesRegex(MemoryCapabilityError, "already uses"):
            with self.controller.region(
                "workspace", native_backing_store="disk"
            ):
                pass

    def test_external_regions_fail_closed(self) -> None:
        with self.assertRaisesRegex(MemoryCapabilityError, "externally owned"):
            with self.controller.region("external"):
                pass
        with self.assertRaisesRegex(MemoryCapabilityError, "externally owned"):
            self.controller.release(["external"])

    def test_release_and_restore_keep_engine_selected_order(self) -> None:
        self.controller.release(["workspace", "kv_cache"])
        self.controller.restore(["kv_cache", "workspace"])
        self.assertEqual(
            self.saver.events,
            [
                ("pause", "workspace"),
                ("pause", "kv_cache"),
                ("resume", "kv_cache"),
                ("resume", "workspace"),
            ],
        )

    def test_graph_regions_are_explicitly_capability_gated(self) -> None:
        with self.assertRaisesRegex(MemoryCapabilityError, "graph_regions"):
            with self.controller.cuda_graph("graph"):
                pass
        controller = TorchMemorySaverRegionController(
            saver=self.saver, graph_regions=True
        )
        with controller.cuda_graph("graph", pool="pool", stream="stream"):
            pass
        self.assertEqual(self.saver.events[0][0], "graph-enter")
        self.assertEqual(self.saver.events[0][5], "cuda_graph")
        self.assertEqual(self.saver.events[1], ("graph-exit", "cuda_graph"))

    def test_external_snapshot_store_is_not_overclaimed(self) -> None:
        self.assertFalse(self.controller.capabilities.allocation_enumeration)
        self.assertFalse(self.controller.capabilities.external_snapshot_store)
        with self.assertRaisesRegex(
            MemoryCapabilityError, "does not expose allocation enumeration"
        ):
            self.controller.snapshot_allocations("weights")


if __name__ == "__main__":
    unittest.main()

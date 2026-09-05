# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib.metadata
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
import coldsnap_cuda_epoch as epoch  # noqa: E402


class FakeNativeEpoch:
    def __init__(self) -> None:
        self.events: list[object] = []

    def reset(self, device: int) -> None:
        self.events.append(("reset", device))

    def probe_context(self, device: int) -> dict[str, object]:
        self.events.append(("probe_context", device))
        return {
            "schema": 1,
            "device": device,
            "primary": {"usable": False},
            "private": {"driver": {"usable": True}},
            "policy_changed": False,
        }

    def rebind(self, recipes: tuple[epoch.CudaEpochAllocation, ...]) -> None:
        self.events.append(
            ("rebind", tuple((item.address, item.mapped_bytes) for item in recipes))
        )

    def activate(self, device: int) -> None:
        self.events.append(("activate", device))

    def copy_from_host(self, destination: int, source: int, size: int) -> None:
        self.events.append(("copy", destination, source, size))

    def fill_zero(self, destination: int, size: int) -> None:
        self.events.append(("zero", destination, size))

    def synchronize(self) -> None:
        self.events.append("synchronize")


class FakeRuntime:
    def __init__(self) -> None:
        self.events: list[str] = []

    def prepare_reset(self) -> None:
        self.events.append("prepare_reset")

    def rebuild(self, native: object) -> None:
        self.native = native
        self.events.append("rebuild")

    def inventory(self) -> dict[str, object]:
        return {"resource_count": 3}


def allocator_fixture(
    *handles: tuple[int, int, int, int], asleep: bool = True
) -> SimpleNamespace:
    return SimpleNamespace(
        pointer_to_data={
            handle[2]: SimpleNamespace(
                tag="weights",
                handle=handle,
                is_asleep=asleep,
            )
            for handle in handles
        }
    )


class VllmCudaEpochTest(unittest.TestCase):
    def _capture(
        self,
        allocator: SimpleNamespace,
        native: FakeNativeEpoch | None = None,
    ) -> epoch.VllmCudaEpoch:
        with patch.object(importlib.metadata, "version", return_value="2.10.0"):
            return epoch.VllmCudaEpoch.capture(
                allocator,
                native=native or FakeNativeEpoch(),
            )

    def test_reset_rebind_and_normal_wake_handoff(self) -> None:
        handles = (
            (0, 65536, 0x100000, 0xA00000),
            (0, 131072, 0x200000, 0xB00000),
        )
        native = FakeNativeEpoch()
        lifecycle = self._capture(allocator_fixture(*handles), native)
        original_calls: list[object] = []
        original = original_calls.append
        cumem = ModuleType("vllm.device_allocator.cumem")
        cumem.create_and_map = original
        wake_events: list[str] = []

        def wake() -> str:
            for handle in handles:
                cumem.create_and_map(handle)
            wake_events.append("hydrated")
            return "awake"

        with patch.dict(sys.modules, {"vllm.device_allocator.cumem": cumem}):
            lifecycle.reset()
            result = lifecycle.resume(wake)

        self.assertEqual(result, "awake")
        self.assertEqual(lifecycle.state, "ACTIVE")
        self.assertEqual(wake_events, ["hydrated"])
        self.assertEqual(original_calls, [])
        self.assertIs(cumem.create_and_map, original)
        self.assertEqual(
            native.events,
            [
                ("reset", 0),
                ("rebind", ((0x100000, 65536), (0x200000, 131072))),
            ],
        )

    def test_runtime_resources_wrap_reset_and_rebind(self) -> None:
        handle = (0, 65536, 0x100000, 0xA00000)
        native = FakeNativeEpoch()
        resources = FakeRuntime()
        data = SimpleNamespace(
            tag="weights",
            handle=handle,
            is_asleep=True,
        )
        recipe = epoch.CudaEpochAllocation(
            address=handle[2],
            mapped_bytes=handle[1],
            handle_box_address=handle[3],
            device=handle[0],
            tag=data.tag,
            data=data,
        )
        lifecycle = epoch.VllmCudaEpoch(
            (recipe,),
            native,
            torch_version="2.10.0",
            runtime=resources,
        )
        cumem = ModuleType("vllm.device_allocator.cumem")
        cumem.create_and_map = lambda _handle: None

        with patch.dict(sys.modules, {"vllm.device_allocator.cumem": cumem}):
            lifecycle.reset()
            lifecycle.resume(lambda: cumem.create_and_map(handle))

        self.assertEqual(resources.events, ["prepare_reset", "rebuild"])
        self.assertIs(resources.native, native)
        self.assertEqual(lifecycle.inventory()["runtime"], {"resource_count": 3})

    def test_explicit_rebind_allows_distributed_restore_before_wake(self) -> None:
        handle = (0, 65536, 0x100000, 0xA00000)
        native = FakeNativeEpoch()
        resources = FakeRuntime()
        data = SimpleNamespace(tag="weights", handle=handle, is_asleep=True)
        recipe = epoch.CudaEpochAllocation(
            address=handle[2],
            mapped_bytes=handle[1],
            handle_box_address=handle[3],
            device=handle[0],
            tag=data.tag,
            data=data,
        )
        lifecycle = epoch.VllmCudaEpoch(
            (recipe,), native, torch_version="2.10.0", runtime=resources
        )
        cumem = ModuleType("vllm.device_allocator.cumem")
        cumem.create_and_map = lambda _handle: None

        with patch.dict(sys.modules, {"vllm.device_allocator.cumem": cumem}):
            lifecycle.reset()
            self.assertEqual(lifecycle.rebind()["state"], "REBOUND")
            self.assertEqual(resources.events, ["prepare_reset", "rebuild"])
            lifecycle.resume(lambda: cumem.create_and_map(handle))

        self.assertEqual(resources.events, ["prepare_reset", "rebuild"])
        self.assertEqual(lifecycle.state, "ACTIVE")

    def test_context_probe_is_diagnostic_only_at_reset_boundary(self) -> None:
        handle = (0, 65536, 0x100000, 0xA00000)
        native = FakeNativeEpoch()
        lifecycle = self._capture(allocator_fixture(handle), native)
        with self.assertRaisesRegex(epoch.CudaEpochError, "requires RESET state"):
            lifecycle.probe_context()

        lifecycle.reset()
        result = lifecycle.probe_context()

        self.assertEqual(result["device"], 0)
        self.assertFalse(result["policy_changed"])
        self.assertEqual(lifecycle.state, "RESET")
        self.assertEqual(native.events, [("reset", 0), ("probe_context", 0)])

    def test_wake_delegates_allocations_outside_captured_epoch(self) -> None:
        handle = (0, 65536, 0x100000, 0xA00000)
        unknown = (0, 65536, 0x300000, 0xC00000)
        lifecycle = self._capture(allocator_fixture(handle))
        original_calls: list[object] = []
        cumem = ModuleType("vllm.device_allocator.cumem")
        cumem.create_and_map = original_calls.append

        def wake() -> None:
            cumem.create_and_map(unknown)
            cumem.create_and_map(handle)

        with patch.dict(sys.modules, {"vllm.device_allocator.cumem": cumem}):
            lifecycle.reset()
            lifecycle.resume(wake)

        self.assertEqual(original_calls, [unknown])

    def test_wake_fails_if_vllm_does_not_claim_every_prebound_allocation(self) -> None:
        handles = (
            (0, 65536, 0x100000, 0xA00000),
            (0, 65536, 0x200000, 0xB00000),
        )
        lifecycle = self._capture(allocator_fixture(*handles))
        cumem = ModuleType("vllm.device_allocator.cumem")
        cumem.create_and_map = lambda _handle: None
        with patch.dict(sys.modules, {"vllm.device_allocator.cumem": cumem}):
            lifecycle.reset()
            with self.assertRaisesRegex(epoch.CudaEpochError, "did not claim 1"):
                lifecycle.resume(lambda: cumem.create_and_map(handles[0]))
        self.assertEqual(lifecycle.state, "REBOUND")

    def test_epoch_externalized_residual_is_published_around_backend_wake(self) -> None:
        handle = (0, 65536, 0x100000, 0xA00000)
        backup = SimpleNamespace(
            numel=lambda: 65536,
            element_size=lambda: 1,
            data_ptr=lambda: 0xC00000,
        )
        data = SimpleNamespace(
            tag="kv_cache",
            handle=handle,
            is_asleep=True,
            cpu_backup_tensor=backup,
        )
        recipe = epoch.CudaEpochAllocation(
            address=handle[2],
            mapped_bytes=handle[1],
            handle_box_address=handle[3],
            device=handle[0],
            tag=data.tag,
            data=data,
            externalized_by_epoch=True,
        )
        native = FakeNativeEpoch()
        lifecycle = epoch.VllmCudaEpoch(
            (recipe,), native, torch_version="2.10.0"
        )
        cumem = ModuleType("vllm.device_allocator.cumem")

        def original(_handle: object) -> None:
            pass

        cumem.create_and_map = original

        def wake() -> str:
            self.assertEqual(data.tag, "coldsnap_cuda_epoch_residual")
            self.assertFalse(data.is_asleep)
            self.assertIsNone(data.cpu_backup_tensor)
            return "awake"

        with (
            patch.dict(sys.modules, {"vllm.device_allocator.cumem": cumem}),
        ):
            lifecycle.reset()
            self.assertEqual(lifecycle.resume(wake), "awake")

        self.assertIn(("copy", handle[2], 0xC00000, handle[1]), native.events)
        self.assertEqual(data.tag, "kv_cache")
        self.assertEqual(lifecycle.state, "ACTIVE")
        self.assertIs(cumem.create_and_map, original)

    def test_capture_requires_all_allocations_to_be_asleep(self) -> None:
        allocator = allocator_fixture(
            (0, 65536, 0x100000, 0xA00000), asleep=False
        )
        with self.assertRaisesRegex(epoch.CudaEpochError, "call sleep first"):
            self._capture(allocator)

    def test_capture_rejects_multiple_devices_in_prototype(self) -> None:
        allocator = allocator_fixture(
            (0, 65536, 0x100000, 0xA00000),
            (1, 65536, 0x200000, 0xB00000),
        )
        with self.assertRaisesRegex(epoch.CudaEpochError, "one device per process"):
            self._capture(allocator)

    def test_capture_requires_pytorch_2_10(self) -> None:
        allocator = allocator_fixture((0, 65536, 0x100000, 0xA00000))
        with (
            patch.object(importlib.metadata, "version", return_value="2.9.1"),
            self.assertRaisesRegex(epoch.CudaEpochError, r"PyTorch 2.10\+"),
        ):
            epoch.VllmCudaEpoch.capture(allocator, native=FakeNativeEpoch())


if __name__ == "__main__":
    unittest.main()

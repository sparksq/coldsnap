# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

import coldsnap_vllm_graphs as vllm_graphs  # noqa: E402
from coldsnap_disk_backend import DiskCuMemBackend  # noqa: E402
from coldsnap_graph_memory import (  # noqa: E402
    DisabledCudaGraphController,
    GraphRegionStats,
    graph_controller_from_env,
)


class FakeGraphController:
    enabled = True

    def __init__(self, events: list[object], *, paused: int = 0) -> None:
        self.events = events
        self.paused = paused

    @contextmanager
    def capture_region(self, *, pool_id: int = 0, device: int | None = None):
        self.events.append(("region-enter", pool_id, device))
        try:
            yield
        finally:
            self.events.append(("region-exit", pool_id, device))

    def release(self) -> None:
        self.events.append("graph-release")
        self.paused = 2

    def restore(self) -> None:
        self.events.append("graph-restore")
        self.paused = 0

    def stats(self) -> GraphRegionStats:
        return GraphRegionStats(2, 20, 32, self.paused)


class GraphControllerTest(unittest.TestCase):
    def test_disabled_controller_is_dependency_free_and_noop(self) -> None:
        controller = DisabledCudaGraphController()
        with controller.capture_region(pool_id=1, device=0):
            pass
        controller.release()
        controller.restore()
        self.assertEqual(controller.stats(), GraphRegionStats(0, 0, 0, 0))
        self.assertEqual(controller.allocations(), ())

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(
                graph_controller_from_env(), DisabledCudaGraphController
            )

    def test_torch_graph_hook_scopes_capture_allocations(self) -> None:
        events: list[object] = []
        pool = object()

        class OriginalGraphContext:
            def __enter__(self):
                events.append("graph-enter")
                return "captured"

            def __exit__(self, exc_type, exc, traceback):
                events.append(("graph-exit", exc_type))

        def original_graph(*args, **kwargs):
            events.append(("graph-create", args, kwargs))
            return OriginalGraphContext()

        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(graph=original_graph)
        controller = FakeGraphController(events)
        with patch.dict(sys.modules, {"torch": torch}):
            vllm_graphs._install_torch_graph_hook(controller)
            with torch.cuda.graph("graph", pool=pool) as value:
                events.append(("body", value))

        self.assertEqual(
            events,
            [
                ("graph-create", ("graph",), {"pool": pool}),
                "graph-enter",
                ("region-enter", id(pool), None),
                ("body", "captured"),
                ("region-exit", id(pool), None),
                ("graph-exit", None),
            ],
        )

    def test_breakable_graph_hook_scopes_manual_capture(self) -> None:
        events: list[object] = []
        pool = object()

        class BreakableCUDAGraphCapture:
            def __init__(self) -> None:
                self.pool = pool

            def _begin_segment(self, value):
                events.append(("begin", value))
                return "begun"

            def _end_segment(self, value):
                events.append(("end", value))
                return "ended"

        module = ModuleType("vllm.compilation.breakable_cudagraph")
        module.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
        controller = FakeGraphController(events)
        with patch.dict(
            sys.modules,
            {"vllm.compilation.breakable_cudagraph": module},
        ):
            vllm_graphs._install_breakable_graph_hook(controller)

        instance = BreakableCUDAGraphCapture()
        self.assertEqual(instance._begin_segment(1), "begun")
        events.append("capturing")
        self.assertEqual(instance._end_segment(2), "ended")
        self.assertEqual(
            events,
            [
                ("begin", 1),
                ("region-enter", id(pool), None),
                "capturing",
                ("region-exit", id(pool), None),
                ("end", 2),
            ],
        )


class BackendGraphLifecycleTest(unittest.TestCase):
    @staticmethod
    def logger_modules():
        logger = ModuleType("vllm.logger")
        logger.init_logger = lambda name: SimpleNamespace(info=lambda *args: None)
        return patch.dict(
            sys.modules,
            {
                "vllm": ModuleType("vllm"),
                "vllm.logger": logger,
            },
        )

    def test_suspend_discards_graph_backing_before_weights(self) -> None:
        events: list[object] = []
        allocation = SimpleNamespace(
            pointer=1000, size=64, tag="weights", is_released=False
        )

        class Memory:
            def allocations(self, tag):
                self_tag = tag
                events.append(("allocations", self_tag))
                return [allocation]

            def synchronize(self):
                events.append("memory-sync")

            def release(self, value):
                self_value = value
                events.append(("weight-release", self_value.size))

            def empty_cache(self):
                events.append("empty-cache")

        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        backend.memory_provider = Memory()
        backend.graph_controller = FakeGraphController(events)
        backend._state = "RUNNING"
        backend.manifest_path = Path("manifest.json")
        backend.blob_path = Path("weights.blob")
        backend._write_state = lambda state, **values: events.append(
            ("state", state, values)
        )
        backend._write_snapshot = lambda memory: {
            "generation": "generation",
            "blob_bytes": 64,
            "blob_stat": {},
            "entries": [{}],
            "direct_io": False,
            "write_io_mode": "buffered",
            "blob_reused": False,
            "write_seconds": 0.1,
            "verification_seconds": 0.0,
            "phase_seconds": {},
        }

        with self.logger_modules():
            backend.suspend()

        self.assertLess(events.index("graph-release"), events.index(("weight-release", 64)))
        sleeping = [event for event in events if event[:2] == ("state", "sleeping")]
        self.assertEqual(sleeping[0][2]["cuda_graph"]["paused_count"], 2)

    def test_resume_restores_graph_before_publishing_weights(self) -> None:
        events: list[object] = []
        allocation = SimpleNamespace(
            pointer=1000, size=64, tag="weights", is_released=True
        )

        class Memory:
            def allocations(self, tag):
                return [allocation]

            def synchronize(self):
                events.append("memory-sync")

        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        backend.memory_provider = Memory()
        backend.graph_controller = FakeGraphController(events, paused=2)
        backend._restore_mapped = set()
        backend._state = "SUSPENDED"
        backend.read_direct = False
        backend.verify_mode = "inline"
        backend.reuse_blob = True
        backend._reusable_generation = None
        backend._reusable_blob_stat = None
        states: list[tuple[str, dict[str, object]]] = []
        backend._write_state = lambda state, **values: states.append((state, values))
        manifest = {
            "generation": "generation",
            "blob_bytes": 64,
            "blob_stat": {},
            "entries": [{}],
        }
        backend._load_manifest = lambda memory: manifest
        backend._restore_pipeline = lambda memory, value, fd: (
            events.append("weight-restore") or {"restored_bytes": 64}
        )

        def commit(memory):
            events.append("weight-commit")
            allocation.is_released = False

        backend._commit_restore = commit
        with tempfile.TemporaryDirectory() as directory:
            backend.blob_path = Path(directory) / "weights.blob"
            backend.manifest_path = Path(directory) / "manifest.json"
            backend.blob_path.write_bytes(b"snapshot")
            with self.logger_modules():
                backend.resume()

        self.assertLess(events.index("graph-restore"), events.index("weight-commit"))
        self.assertLess(events.index("graph-restore"), events.index("memory-sync"))

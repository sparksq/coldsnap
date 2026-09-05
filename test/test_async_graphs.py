# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

import coldsnap_vllm_async_graphs as async_graphs  # noqa: E402


class AsyncGraphCaptureTest(unittest.TestCase):
    class Dispatcher:
        def __init__(self) -> None:
            self.keys_initialized = True

        def dispatch(self):
            return "graph" if self.keys_initialized else "eager"

    @classmethod
    def setUpClass(cls) -> None:
        async_graphs._wrap_dispatch_gate(cls.Dispatcher, "keys_initialized")

    def test_settings_are_opt_in_and_fail_closed(self) -> None:
        self.assertEqual(
            async_graphs.async_graph_settings_from_env({}),
            async_graphs.AsyncGraphSettings(),
        )
        self.assertEqual(
            async_graphs.async_graph_settings_from_env(
                {
                    async_graphs.ASYNC_GRAPH_ENV: "1",
                    async_graphs.ARM_FILE_ENV: "/tmp/graphs.arm",
                }
            ),
            async_graphs.AsyncGraphSettings(True, arm_file="/tmp/graphs.arm"),
        )
        with self.assertRaisesRegex(async_graphs.VllmContractError, "set together"):
            async_graphs.async_graph_settings_from_env(
                {
                    async_graphs.ASYNC_GRAPH_ENV: "1",
                    async_graphs.READY_FILE_ENV: "/tmp/ready.json",
                }
            )
        retained = async_graphs.async_graph_settings_from_env(
            {
                async_graphs.ASYNC_GRAPH_ENV: "1",
                "COLDSNAP_GRAPH_POLICY": "preserve-nccl-exec",
            }
        )
        self.assertEqual(retained.policy, "preserve-nccl-exec")

    def test_dispatch_gate_preserves_graph_state_while_forcing_eager(self) -> None:
        dispatcher = self.Dispatcher()
        dispatcher._coldsnap_force_eager = True
        self.assertEqual(dispatcher.dispatch(), "eager")
        self.assertTrue(dispatcher.keys_initialized)
        dispatcher._coldsnap_force_eager = False
        self.assertEqual(dispatcher.dispatch(), "graph")

    def test_worker_defers_capture_then_atomically_enables_graphs(self) -> None:
        events: list[object] = []

        class Runner:
            def __init__(self, dispatcher) -> None:
                self.cudagraph_dispatcher = dispatcher
                self.capture_calls = 0

            def capture_model(self):
                self.capture_calls += 1
                events.append("capture")
                return 256

        runner = Runner(self.Dispatcher())
        worker = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL")),
            model_runner=runner,
        )

        def original_compile(instance):
            events.append("warmup")
            instance.model_runner.capture_model()
            events.append("startup-ready")
            return "compiled"

        result = async_graphs._compile_eager_first(worker, original_compile)
        self.assertEqual(result, "compiled")
        self.assertEqual(runner.capture_calls, 0)
        self.assertTrue(runner.cudagraph_dispatcher._coldsnap_force_eager)
        self.assertEqual(events, ["warmup", "startup-ready"])

        with patch.object(async_graphs, "_refresh_worker_graph_pools"):
            status = async_graphs._capture_worker_graphs(worker)
        self.assertEqual(status["phase"], "ready")
        self.assertEqual(status["graph_memory_bytes"], 256)
        self.assertEqual(status["graph_memory_delta_bytes"], 256)
        self.assertEqual(runner.capture_calls, 1)
        self.assertFalse(runner.cudagraph_dispatcher._coldsnap_force_eager)

    def test_retained_policy_captures_normal_graphs_before_snapshot(self) -> None:
        class Runner:
            def __init__(self, dispatcher) -> None:
                self.cudagraph_dispatcher = dispatcher
                self.capture_calls = 0

            def capture_model(self):
                self.capture_calls += 1
                return 128

        runner = Runner(self.Dispatcher())
        worker = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL")),
            model_runner=runner,
        )

        def original_compile(instance):
            instance.model_runner.capture_model()
            return "compiled"

        with patch.object(async_graphs, "shape_calibration_enabled", return_value=False):
            result = async_graphs._compile_retained_first(worker, original_compile)
        self.assertEqual(result, "compiled")
        self.assertEqual(runner.capture_calls, 1)
        self.assertEqual(worker._coldsnap_async_graph_state.phase, "retained_ready")
        self.assertFalse(getattr(runner.cudagraph_dispatcher, "_coldsnap_force_eager", False))

    def test_engine_waits_for_idle_after_eager_execution(self) -> None:
        events: list[object] = ["response-queued"]
        busy = True

        class Executor:
            def collective_rpc(self, method, args=()):
                events.append((method, args))
                return [{"phase": "ready"}]

        engine = SimpleNamespace(
            scheduler=SimpleNamespace(has_requests=lambda: busy),
            batch_queue=None,
            input_queue=SimpleNamespace(empty=lambda: True),
            model_executor=Executor(),
        )
        settings = async_graphs.AsyncGraphSettings(True)
        async_graphs._maybe_capture_after_step(engine, True, settings)
        self.assertEqual(events, ["response-queued"])
        self.assertEqual(engine._coldsnap_async_graph_eager_steps, 1)

        busy = False
        async_graphs._maybe_capture_after_step(engine, True, settings)
        self.assertEqual(engine._coldsnap_async_graph_phase, "ready")
        self.assertEqual(
            events,
            ["response-queued", ("coldsnap_capture_cuda_graphs", ())],
        )

    def test_external_arm_file_blocks_capture_without_losing_eager_steps(self) -> None:
        events: list[object] = []

        class Executor:
            def collective_rpc(self, method, args=()):
                events.append((method, args))
                return [{"phase": "ready"}]

        engine = SimpleNamespace(
            scheduler=SimpleNamespace(has_requests=lambda: False),
            batch_queue=None,
            input_queue=SimpleNamespace(empty=lambda: True),
            model_executor=Executor(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            arm = Path(temporary) / "capture.arm"
            settings = async_graphs.AsyncGraphSettings(True, arm_file=str(arm))
            async_graphs._maybe_capture_after_step(engine, True, settings)
            self.assertEqual(events, [])
            self.assertEqual(engine._coldsnap_async_graph_eager_steps, 1)
            arm.touch()
            async_graphs._maybe_capture_after_step(engine, False, settings)

        self.assertEqual(engine._coldsnap_async_graph_phase, "ready")
        self.assertEqual(events, [("coldsnap_capture_cuda_graphs", ())])

    def test_retained_policy_only_enters_async_recapture_when_fallback_is_armed(self) -> None:
        events: list[object] = []

        class Executor:
            def collective_rpc(self, method, args=()):
                events.append((method, args))
                return [{"phase": "ready"}]

        engine = SimpleNamespace(
            scheduler=SimpleNamespace(has_requests=lambda: False),
            batch_queue=None,
            input_queue=SimpleNamespace(empty=lambda: True),
            model_executor=Executor(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            arm = Path(temporary) / "fallback.arm"
            settings = async_graphs.AsyncGraphSettings(
                True,
                arm_file=str(arm),
                policy="preserve-nccl-exec",
            )
            async_graphs._maybe_capture_after_step(engine, True, settings)
            self.assertEqual(events, [])
            self.assertFalse(hasattr(engine, "_coldsnap_async_graph_phase"))
            arm.touch()
            async_graphs._maybe_capture_after_step(engine, True, settings)

        self.assertEqual(engine._coldsnap_async_graph_phase, "ready")
        self.assertEqual(events, [("coldsnap_capture_cuda_graphs", ())])

    def test_engine_publishes_atomic_graph_readiness(self) -> None:
        class Executor:
            def collective_rpc(self, method, args=()):
                return [{"phase": "ready", "capture_seconds": 0.25}]

        engine = SimpleNamespace(
            scheduler=SimpleNamespace(has_requests=lambda: False),
            batch_queue=None,
            input_queue=SimpleNamespace(empty=lambda: True),
            model_executor=Executor(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "graph.json"
            settings = async_graphs.AsyncGraphSettings(True, str(ready), "generation-1")
            async_graphs._maybe_capture_after_step(engine, True, settings)
            payload = json.loads(ready.read_text())
        self.assertEqual(payload["phase"], "ready")
        self.assertEqual(payload["generation"], "generation-1")
        self.assertEqual(payload["statuses"][0]["capture_seconds"], 0.25)

    def test_engine_hook_records_model_execution_wall_time(self) -> None:
        class EngineCoreProc:
            def __init__(self) -> None:
                self.scheduler = SimpleNamespace(has_requests=lambda: False)
                self.batch_queue = None
                self.input_queue = SimpleNamespace(empty=lambda: True)
                self.model_executor = SimpleNamespace(collective_rpc=lambda *args: [])

            def _process_engine_step(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as temporary:
            settings = async_graphs.AsyncGraphSettings(
                True,
                arm_file=str(Path(temporary) / "not-armed"),
            )
            async_graphs._install_engine_hook(
                settings,
                module=SimpleNamespace(EngineCoreProc=EngineCoreProc),
            )
            engine = EngineCoreProc()
            with (
                patch.object(async_graphs.time, "time_ns", side_effect=[10, 20]),
                patch.object(async_graphs.time, "perf_counter", side_effect=[1.0, 1.25]),
            ):
                self.assertTrue(engine._process_engine_step())

        self.assertEqual(
            engine._coldsnap_async_graph_step_timings,
            [
                {
                    "index": 0,
                    "started_unix_ns": 10,
                    "completed_unix_ns": 20,
                    "wall_seconds": 0.25,
                }
            ],
        )

    def test_worker_failure_rolls_every_rank_back_to_eager(self) -> None:
        events: list[object] = []

        class Executor:
            def collective_rpc(self, method, args=()):
                events.append((method, args))
                if method == "coldsnap_capture_cuda_graphs":
                    return [{"phase": "failed", "error": "capture failed"}]
                return [{"phase": "fallback_eager"}]

        engine = SimpleNamespace(
            scheduler=SimpleNamespace(has_requests=lambda: False),
            batch_queue=None,
            input_queue=SimpleNamespace(empty=lambda: True),
            model_executor=Executor(),
        )
        async_graphs._maybe_capture_after_step(engine, True, async_graphs.AsyncGraphSettings(True))
        self.assertEqual(engine._coldsnap_async_graph_phase, "fallback_eager")
        self.assertEqual(events[0][0], "coldsnap_capture_cuda_graphs")
        self.assertEqual(events[1][0], "coldsnap_fallback_to_eager_graphs")

    def test_stale_allocator_pool_failure_is_refreshed_and_retried_once(self) -> None:
        events: list[object] = []
        captures = 0

        class Executor:
            def collective_rpc(self, method, args=()):
                nonlocal captures
                events.append((method, args))
                if method == "coldsnap_capture_cuda_graphs":
                    captures += 1
                    if captures == 1:
                        return [
                            {
                                "phase": "failed",
                                "error": "RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED",
                            }
                        ]
                    return [{"phase": "ready"}]
                if method == "coldsnap_prepare_cuda_graph_retry":
                    return [{"phase": "eager"}]
                return [{"phase": "fallback_eager"}]

        engine = SimpleNamespace(
            scheduler=SimpleNamespace(has_requests=lambda: False),
            batch_queue=None,
            input_queue=SimpleNamespace(empty=lambda: True),
            model_executor=Executor(),
        )
        settings = async_graphs.AsyncGraphSettings(True)

        async_graphs._maybe_capture_after_step(engine, True, settings)
        self.assertEqual(engine._coldsnap_async_graph_phase, "eager")
        self.assertEqual(engine._coldsnap_async_graph_retries, 1)
        async_graphs._maybe_capture_after_step(engine, False, settings)

        self.assertEqual(engine._coldsnap_async_graph_phase, "ready")
        self.assertEqual(
            [event[0] for event in events],
            [
                "coldsnap_capture_cuda_graphs",
                "coldsnap_prepare_cuda_graph_retry",
                "coldsnap_capture_cuda_graphs",
            ],
        )

    def test_graph_capture_waits_for_deferred_warmup(self) -> None:
        events: list[object] = []

        class Executor:
            def collective_rpc(self, method, args=()):
                events.append((method, args))
                return [{"phase": "ready"}]

        engine = SimpleNamespace(
            scheduler=SimpleNamespace(has_requests=lambda: False),
            batch_queue=None,
            input_queue=SimpleNamespace(empty=lambda: True),
            model_executor=Executor(),
            _coldsnap_deferred_warmup_phase="warming",
        )
        async_graphs._maybe_capture_after_step(engine, True, async_graphs.AsyncGraphSettings(True))
        self.assertEqual(events, [])
        engine._coldsnap_deferred_warmup_phase = "ready"
        async_graphs._maybe_capture_after_step(engine, True, async_graphs.AsyncGraphSettings(True))
        self.assertEqual(events[0][0], "coldsnap_capture_cuda_graphs")


if __name__ == "__main__":
    unittest.main()

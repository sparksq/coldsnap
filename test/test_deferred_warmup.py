# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

import coldsnap_vllm_deferred_warmup as deferred  # noqa: E402


class DeferredWarmupTest(unittest.TestCase):
    def test_settings_are_opt_in(self) -> None:
        self.assertEqual(
            deferred.deferred_warmup_settings_from_env({}),
            deferred.DeferredWarmupSettings(),
        )
        settings = deferred.deferred_warmup_settings_from_env(
            {
                deferred.DEFERRED_WARMUP_ENV: "1",
                deferred.FULLY_WARM_FILE_ENV: "/tmp/warm",
                deferred.WARMUP_GENERATION_ENV: "g1",
                deferred.WARMUP_ARM_FILE_ENV: "/tmp/arm",
            }
        )
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.generation, "g1")
        self.assertEqual(settings.arm_file, Path("/tmp/arm"))

    def test_profile_defers_only_with_explicit_kv_artifact(self) -> None:
        events: list[str] = []

        class Runner:
            def profile_run(self) -> None:
                events.append("profile")

        runner = Runner()
        worker = SimpleNamespace(
            model_runner=runner,
            cache_config=SimpleNamespace(kv_cache_memory_bytes=4096),
        )

        def determine(instance) -> int:
            events.append("determine")
            instance.model_runner.profile_run()
            return 4096

        self.assertEqual(
            deferred._determine_with_deferred_profile(worker, determine), 4096
        )
        self.assertEqual(events, ["determine"])
        state = deferred._state(worker)
        self.assertTrue(state.profile_deferred)
        state.kernel_deferred = True

        def kernel(_worker) -> bool:
            events.append("kernel")
            return False

        def runtime_kernel(_worker) -> None:
            events.append("runtime-kernel")

        state.runtime_kernel_deferred = True
        status = deferred._run_deferred_warmup(worker, kernel, runtime_kernel)
        self.assertEqual(status["phase"], "ready")
        self.assertEqual(
            events, ["determine", "profile", "kernel", "runtime-kernel"]
        )

        events.clear()
        fresh = SimpleNamespace(
            model_runner=Runner(),
            cache_config=SimpleNamespace(kv_cache_memory_bytes=None),
        )
        deferred._determine_with_deferred_profile(fresh, determine)
        self.assertEqual(events, ["determine", "profile"])
        self.assertFalse(deferred._state(fresh).profile_deferred)

    def test_engine_waits_for_idle_then_records_full_warm_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "fully-warm.json"
            settings = deferred.DeferredWarmupSettings(
                enabled=True,
                fully_warm_file=ready,
                generation="g9",
            )
            busy = True
            calls: list[str] = []

            class Scheduler:
                def has_requests(self) -> bool:
                    return busy

                def reset_prefix_cache(self) -> bool:
                    calls.append("reset")
                    return True

            class Executor:
                def collective_rpc(self, method: str):
                    calls.append(method)
                    return [{"phase": "ready", "profile_deferred": True}]

            engine = SimpleNamespace(
                scheduler=Scheduler(),
                batch_queue=None,
                input_queue=SimpleNamespace(empty=lambda: True),
                model_executor=Executor(),
            )
            deferred._maybe_warm_after_step(engine, True, settings)
            self.assertFalse(ready.exists())
            busy = False
            deferred._maybe_warm_after_step(engine, False, settings)
            self.assertEqual(engine._coldsnap_deferred_warmup_phase, "ready")
            self.assertEqual(calls, ["coldsnap_run_deferred_warmup", "reset"])
            self.assertEqual(
                json.loads(ready.read_text(encoding="utf-8"))["generation"], "g9"
            )

    def test_worker_hooks_defer_both_kernel_warmup_phases(self) -> None:
        events: list[str] = []

        class Worker:
            def __init__(self) -> None:
                self.cache_config = SimpleNamespace(kv_cache_memory_bytes=4096)
                self.model_runner = SimpleNamespace(
                    profile_run=lambda: events.append("profile")
                )

            def determine_available_memory(self) -> int:
                self.model_runner.profile_run()
                return 4096


        def kernel(_worker) -> bool:
            events.append("kernel")
            return False

        def runtime_kernel(_worker) -> None:
            events.append("runtime-kernel")

        module = SimpleNamespace(
            Worker=Worker,
            kernel_warmup=kernel,
            runtime_kernel_warmup=runtime_kernel,
        )
        settings = deferred.DeferredWarmupSettings(enabled=True)
        with mock.patch.object(deferred, "_effective_settings", return_value=settings):
            deferred._install_worker_hooks(settings, module)
            worker = Worker()
            self.assertEqual(worker.determine_available_memory(), 4096)
            self.assertFalse(module.kernel_warmup(worker))
            self.assertIsNone(module.runtime_kernel_warmup(worker))
            self.assertEqual(events, [])
            status = worker.coldsnap_run_deferred_warmup()
        self.assertEqual(status["phase"], "ready")
        self.assertTrue(status["runtime_kernel_deferred"])
        self.assertEqual(events, ["profile", "kernel", "runtime-kernel"])

    def test_worker_never_swallows_shutdown(self) -> None:
        class Runner:
            @staticmethod
            def profile_run() -> None:
                raise SystemExit()

        worker = SimpleNamespace(model_runner=Runner())
        state = deferred._state(worker)
        state.profile_deferred = True
        state.phase = "eager"
        with self.assertRaises(SystemExit):
            deferred._run_deferred_warmup(
                worker,
                lambda _worker: True,
                lambda _worker: None,
            )

    def test_engine_waits_for_post_acceptance_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory) / "arm"
            settings = deferred.DeferredWarmupSettings(enabled=True, arm_file=arm)
            calls: list[str] = []

            class Scheduler:
                @staticmethod
                def has_requests() -> bool:
                    return False

                @staticmethod
                def reset_prefix_cache() -> bool:
                    calls.append("reset")
                    return True

            class Executor:
                @staticmethod
                def collective_rpc(method: str):
                    calls.append(method)
                    return [{"phase": "ready"}]

            engine = SimpleNamespace(
                scheduler=Scheduler(),
                batch_queue=None,
                input_queue=SimpleNamespace(empty=lambda: True),
                model_executor=Executor(),
            )
            deferred._maybe_warm_after_step(engine, True, settings)
            self.assertEqual(calls, [])
            arm.touch()
            deferred._maybe_warm_after_step(engine, False, settings)
            self.assertEqual(calls, ["coldsnap_run_deferred_warmup", "reset"])

    def test_engine_never_swallows_shutdown(self) -> None:
        class Scheduler:
            @staticmethod
            def has_requests() -> bool:
                return False

        class Executor:
            @staticmethod
            def collective_rpc(_method: str):
                raise SystemExit()

        engine = SimpleNamespace(
            scheduler=Scheduler(),
            batch_queue=None,
            input_queue=SimpleNamespace(empty=lambda: True),
            model_executor=Executor(),
            _coldsnap_deferred_warmup_eager_steps=1,
        )
        settings = deferred.DeferredWarmupSettings(enabled=True)
        with self.assertRaises(SystemExit):
            deferred._maybe_warm_after_step(engine, False, settings)

    def test_collective_failure_publishes_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "fully-warm.json"

            class Scheduler:
                @staticmethod
                def has_requests() -> bool:
                    return False

            class Executor:
                @staticmethod
                def collective_rpc(_method: str):
                    raise RuntimeError("worker exited")

            engine = SimpleNamespace(
                scheduler=Scheduler(),
                batch_queue=None,
                input_queue=SimpleNamespace(empty=lambda: True),
                model_executor=Executor(),
                _coldsnap_deferred_warmup_eager_steps=1,
            )
            settings = deferred.DeferredWarmupSettings(
                enabled=True,
                fully_warm_file=ready,
                generation="failed-generation",
            )
            deferred._maybe_warm_after_step(engine, False, settings)
            payload = json.loads(ready.read_text(encoding="utf-8"))
            self.assertEqual(payload["generation"], "failed-generation")
            self.assertEqual(payload["workers"][0]["phase"], "failed")
            self.assertIn("worker exited", payload["workers"][0]["error"])


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

import coldsnap_vllm_deferred_api_warmup as deferred  # noqa: E402


async def _async_result(*_args, **_kwargs):
    return None


class DeferredApiWarmupTest(unittest.TestCase):
    def test_settings_are_opt_in(self) -> None:
        self.assertEqual(
            deferred.deferred_api_warmup_settings_from_env({}),
            deferred.DeferredApiWarmupSettings(),
        )
        settings = deferred.deferred_api_warmup_settings_from_env(
            {
                deferred.DEFERRED_API_MM_WARMUP_ENV: "1",
                deferred.API_MM_WARM_FILE_ENV: "/tmp/api-warm",
                deferred.WARMUP_GENERATION_ENV: "g1",
                deferred.START_DELAY_SECONDS_ENV: "12.5",
            }
        )
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.fully_warm_file, Path("/tmp/api-warm"))
        self.assertEqual(settings.generation, "g1")
        self.assertEqual(settings.start_delay_seconds, 12.5)

    def test_restore_handoff_overrides_captured_disabled_policy(self) -> None:
        process_template = ModuleType("coldsnap_vllm_process_template")
        process_template.process_template_settings_from_env = lambda: object()
        process_template._restored_runtime_environment = lambda _settings: {
            deferred.DEFERRED_API_MM_WARMUP_ENV: "1",
            deferred.API_MM_WARM_FILE_ENV: "/capsule/api-warm.json",
            deferred.WARMUP_GENERATION_ENV: "restore-1",
            deferred.START_DELAY_SECONDS_ENV: "30",
        }
        with patch.dict(
            sys.modules,
            {"coldsnap_vllm_process_template": process_template},
        ):
            settings = deferred._effective_settings(
                deferred.DeferredApiWarmupSettings()
            )
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.fully_warm_file, Path("/capsule/api-warm.json"))
        self.assertEqual(settings.generation, "restore-1")
        self.assertEqual(settings.start_delay_seconds, 30)

    def test_disabled_capture_installs_restore_capable_hook(self) -> None:
        events: list[str] = []

        class Renderer:
            def warmup(self, value) -> None:
                events.append(f"original:{value}")

        module = SimpleNamespace(BaseRenderer=Renderer)
        previous = deferred._installed
        deferred._installed = False
        try:
            with (
                patch.object(deferred.importlib, "import_module", return_value=module),
                patch.object(
                    deferred,
                    "_effective_settings",
                    return_value=deferred.DeferredApiWarmupSettings(enabled=True),
                ),
                patch.object(deferred, "_warmup_with_deferred_multimodal") as warm,
            ):
                deferred.install_deferred_api_warmup_hook(
                    deferred.DeferredApiWarmupSettings(enabled=False)
                )
                Renderer().warmup("restore")
            warm.assert_called_once()
            self.assertEqual(events, [])
        finally:
            deferred._installed = previous

    def test_chat_warmup_stays_sync_and_mm_warmup_uses_existing_queue(self) -> None:
        events: list[str] = []
        blocker = ThreadPoolExecutor(max_workers=1)
        release = __import__("threading").Event()
        blocker.submit(release.wait)
        renderer = SimpleNamespace(
            mm_processor="mm",
            _readonly_mm_processor="readonly",
            _mm_executor=blocker,
            _process_multimodal_async=_async_result,
        )

        def original(instance, _params) -> None:
            self.assertIsNone(instance.mm_processor)
            self.assertIsNone(instance._readonly_mm_processor)
            events.append("chat")

        with patch.object(deferred, "_run_multimodal_warmup") as background:
            deferred._warmup_with_deferred_multimodal(
                renderer,
                original,
                deferred.DeferredApiWarmupSettings(enabled=True),
                object(),
            )
            self.assertEqual(events, ["chat"])
            self.assertEqual(renderer.mm_processor, "mm")
            self.assertEqual(renderer._readonly_mm_processor, "readonly")
            background.assert_not_called()
            release.set()
            renderer._coldsnap_deferred_api_mm_warmup_future.result(timeout=2)
            background.assert_called_once()
        blocker.shutdown()

    def test_real_multimodal_request_supersedes_speculative_delay(self) -> None:
        events: list[str] = []
        executor = ThreadPoolExecutor(max_workers=1)

        async def process_multimodal(value):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                executor,
                lambda: events.append(f"request:{value}") or "processed",
            )

        renderer = SimpleNamespace(
            mm_processor="mm",
            _readonly_mm_processor=None,
            _mm_executor=executor,
            _process_multimodal_async=process_multimodal,
        )

        def original(_instance, _params) -> None:
            events.append("chat")

        deferred._warmup_with_deferred_multimodal(
            renderer,
            original,
            deferred.DeferredApiWarmupSettings(
                enabled=True,
                start_delay_seconds=30,
            ),
            object(),
        )

        result = asyncio.run(renderer._process_multimodal_async("image"))
        timer = renderer._coldsnap_deferred_api_mm_warmup_timer
        timer.join(timeout=2)
        executor.shutdown()

        self.assertEqual(result, "processed")
        self.assertEqual(events, ["chat", "request:image"])
        self.assertFalse(timer.is_alive())
        self.assertFalse(
            hasattr(renderer, "_coldsnap_deferred_api_mm_warmup_future")
        )
        self.assertEqual(
            renderer._coldsnap_deferred_api_mm_warmup_status["phase"],
            "ready",
        )
        self.assertEqual(
            renderer._coldsnap_deferred_api_mm_warmup_status["source"],
            "request",
        )

    def test_idle_delay_submits_warmup_without_blocking_executor(self) -> None:
        events: list[str] = []
        executor = ThreadPoolExecutor(max_workers=1)
        renderer = SimpleNamespace(
            mm_processor="mm",
            _readonly_mm_processor=None,
            _mm_executor=executor,
            _process_multimodal_async=_async_result,
        )

        def original(_instance, _params) -> None:
            events.append("chat")

        def background(*_args):
            events.append("warmup")
            return {"phase": "ready"}

        with patch.object(deferred, "_run_multimodal_warmup", background):
            deferred._warmup_with_deferred_multimodal(
                renderer,
                original,
                deferred.DeferredApiWarmupSettings(
                    enabled=True,
                    start_delay_seconds=0.01,
                ),
                object(),
            )
            renderer._coldsnap_deferred_api_mm_warmup_timer.join(timeout=2)
            status = renderer._coldsnap_deferred_api_mm_warmup_future.result(
                timeout=2
            )
        executor.shutdown()

        self.assertEqual(status["phase"], "ready")
        self.assertEqual(events, ["chat", "warmup"])

    def test_background_warmup_records_readiness(self) -> None:
        events: list[str] = []

        class Renderer:
            def _warmup_mm_processor(self, processor, *, log_prefix) -> None:
                events.append(f"warm:{processor}:{log_prefix}")

            def clear_mm_cache(self) -> None:
                events.append("clear")

            @staticmethod
            def _clear_processor_cache(processor) -> None:
                events.append(f"clear:{processor}")

        torch_utils = ModuleType("vllm.utils.torch_utils")
        torch_utils.set_default_torch_num_threads = lambda _count: nullcontext()
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "api-warm.json"
            settings = deferred.DeferredApiWarmupSettings(
                enabled=True,
                fully_warm_file=status_path,
                generation="g2",
            )
            with patch.dict(
                sys.modules, {"vllm.utils.torch_utils": torch_utils}
            ):
                status = deferred._run_multimodal_warmup(
                    Renderer(), "mm", "readonly", settings
                )
            self.assertEqual(status["phase"], "ready")
            self.assertEqual(
                events,
                [
                    "warm:mm:Deferred multi-modal",
                    "clear",
                    "warm:readonly:Deferred readonly multi-modal",
                    "clear:readonly",
                ],
            )
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["generation"], "g2")
            self.assertEqual(payload["phase"], "ready")


if __name__ == "__main__":
    unittest.main()

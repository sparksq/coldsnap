# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
import coldsnap_vllm_kv_capacity as capacity  # noqa: E402


class Worker:
    def __init__(self, *, explicit: int | None = None) -> None:
        self.cache_config = types.SimpleNamespace(kv_cache_memory_bytes=explicit)
        self.requested_memory = 24 * (1 << 30)
        self.total_consumed = 19 * (1 << 30)
        self.peak_activation_memory = 2 * (1 << 30)
        self.available_kv_cache_memory_bytes = 11 * (1 << 30)
        self.model_config = types.SimpleNamespace(multimodal_config=object())
        self.parallel_config = types.SimpleNamespace(_api_process_count=2)


class KvCapacityGuardTest(unittest.TestCase):
    def test_guard_requires_its_explicit_policy_environment(self) -> None:
        self.assertFalse(capacity.kv_capacity_guard_enabled({}))
        self.assertFalse(
            capacity.kv_capacity_guard_enabled({"COLDSNAP_VLLM_OVERRIDE_CUMEM": "1"})
        )
        self.assertTrue(
            capacity.kv_capacity_guard_enabled(
                {capacity.KV_CAPACITY_GUARD_ENV: "1"}
            )
        )
        self.assertFalse(
            capacity.kv_capacity_guard_enabled(
                {
                    "COLDSNAP_VLLM_OVERRIDE_CUMEM": "1",
                    capacity.KV_CAPACITY_GUARD_ENV: "0",
                }
            )
        )
        with self.assertRaisesRegex(capacity.VllmContractError, "must be a boolean"):
            capacity.kv_capacity_guard_enabled({capacity.KV_CAPACITY_GUARD_ENV: "perhaps"})

    def test_oversized_automatic_capacity_is_clamped(self) -> None:
        worker = Worker()
        reservations: list[tuple[int, object, int]] = []

        def reserve(value: int, mm_config: object, api_count: int) -> int:
            reservations.append((value, mm_config, api_count))
            return value - 256 * (1 << 20)

        result = capacity._guard_automatic_capacity(worker, 11 * (1 << 30), reserve)

        self.assertEqual(result, 3 * (1 << 30) - 256 * (1 << 20))
        self.assertEqual(reservations[0][0], 3 * (1 << 30))
        self.assertEqual(reservations[0][2], 2)
        self.assertTrue(worker._coldsnap_kv_capacity_guard_status["clamped"])
        self.assertEqual(worker.available_kv_cache_memory_bytes, 3 * (1 << 30))

    def test_guard_never_increases_an_upstream_safe_estimate(self) -> None:
        worker = Worker()
        result = capacity._guard_automatic_capacity(
            worker,
            2 * (1 << 30),
            lambda value, _mm, _apis: value,
        )
        self.assertEqual(result, 2 * (1 << 30))
        self.assertFalse(worker._coldsnap_kv_capacity_guard_status["clamped"])
        self.assertEqual(worker.available_kv_cache_memory_bytes, 11 * (1 << 30))

    def test_explicit_capacity_bypasses_the_guard(self) -> None:
        worker = Worker(explicit=1024)
        result = capacity._guard_automatic_capacity(
            worker,
            1024,
            lambda *_args: self.fail("reservation must not run"),
        )
        self.assertEqual(result, 1024)
        self.assertFalse(hasattr(worker, "_coldsnap_kv_capacity_guard_status"))

    def test_missing_profile_inputs_fail_closed(self) -> None:
        worker = Worker()
        del worker.total_consumed
        with self.assertRaisesRegex(capacity.VllmContractError, "total_consumed capacity input"):
            capacity._guard_automatic_capacity(worker, 1024, lambda value, _mm, _apis: value)

    def test_worker_wrapper_applies_after_vllm_profile(self) -> None:
        events: list[str] = []

        class FakeWorker(Worker):
            def determine_available_memory(self) -> int:
                events.append("vllm")
                return 11 * (1 << 30)

        module = types.SimpleNamespace(
            Worker=FakeWorker,
            reserve_mm_ipc_gpu_memory=lambda value, _mm, _apis: value,
        )
        capacity._install_worker_hook(module)
        worker = FakeWorker()
        self.assertEqual(worker.determine_available_memory(), 3 * (1 << 30))
        self.assertEqual(events, ["vllm"])


if __name__ == "__main__":
    unittest.main()

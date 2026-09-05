# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
import coldsnap_vllm_kv_payload as payload  # noqa: E402
from coldsnap_vllm import Allocation, VllmContractError  # noqa: E402
from coldsnap_vllm_memory import VllmMemoryProvider  # noqa: E402


class Storage:
    def __init__(self, pointer: int, size: int) -> None:
        self.pointer = pointer
        self.size = size

    def data_ptr(self) -> int:
        return self.pointer

    def nbytes(self) -> int:
        return self.size


class Tensor:
    def __init__(self, pointer: int, size: int) -> None:
        self.storage = Storage(pointer, size)

    def untyped_storage(self) -> Storage:
        return self.storage


class KvPayloadTest(unittest.TestCase):
    def test_runner_hook_records_nested_deduplicated_storage(self) -> None:
        class GPUModelRunner:
            def initialize_kv_cache(self, config):
                self.kv_caches = [
                    Tensor(1000, 400),
                    {"shared": Tensor(1000, 400)},
                    [Tensor(2000, 800)],
                ]
                return config

        module = types.SimpleNamespace(
            __name__="synthetic.model_runner", GPUModelRunner=GPUModelRunner
        )
        payload._install_runner_hook(module)
        runner = GPUModelRunner()
        self.assertEqual(runner.initialize_kv_cache("config"), "config")
        self.assertEqual(
            payload.kv_payload_extents(),
            (
                payload.KvPayloadExtent(1000, 400),
                payload.KvPayloadExtent(2000, 800),
            ),
        )

    def test_missing_or_empty_runner_cache_fails_closed(self) -> None:
        with self.assertRaisesRegex(VllmContractError, "did not publish kv_caches"):
            payload.record_kv_payload(types.SimpleNamespace())
        with self.assertRaisesRegex(VllmContractError, "no tensor storage"):
            payload.record_kv_payload(types.SimpleNamespace(kv_caches=[]))

    def test_provider_maps_semantic_storage_to_tagged_allocations(self) -> None:
        values = (
            Allocation(900, 600, "kv_cache", (900, 600), object()),
            Allocation(2000, 100, "kv_cache", (2000, 100), object()),
            Allocation(3000, 50, "kv_cache", (3000, 50), object()),
        )
        provider = VllmMemoryProvider()
        provider.allocations = lambda tag=None: values  # type: ignore[method-assign]
        with patch.object(
            payload,
            "_payload_extents",
            (
                payload.KvPayloadExtent(1000, 400),
                payload.KvPayloadExtent(2000, 100),
            ),
        ):
            selected = provider.select_allocations("kv_cache", "payload")
        self.assertEqual([item.pointer for item in selected], [900, 2000])

    def test_provider_rejects_unmatched_payload_extent(self) -> None:
        provider = VllmMemoryProvider()
        provider.allocations = lambda tag=None: ()  # type: ignore[method-assign]
        with patch.object(
            payload,
            "_payload_extents",
            (payload.KvPayloadExtent(1000, 400),),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                provider.select_allocations("kv_cache", "payload")

    def test_hook_installation_is_noop_unless_payload_discard_is_enabled(self) -> None:
        with (
            patch.object(payload, "_installed", False),
            patch.object(payload, "after_module_import") as after_import,
            patch.dict("os.environ", {}, clear=True),
        ):
            payload.install_kv_payload_tracking_hooks()
            after_import.assert_not_called()

        with (
            patch.object(payload, "_installed", False),
            patch.object(payload, "after_module_import") as after_import,
            patch.dict(
                "os.environ",
                {
                    "COLDSNAP_DISCARD_REGIONS": "kv_cache",
                },
                clear=True,
            ),
        ):
            payload.install_kv_payload_tracking_hooks()
            self.assertEqual(after_import.call_count, 2)


if __name__ == "__main__":
    unittest.main()

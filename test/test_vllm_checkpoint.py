# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

from coldsnap_vllm_checkpoint import (  # noqa: E402
    CheckpointHookError,
    call_checkpoint_hook,
    call_checkpoint_hook_async,
    checkpoint_prepare,
    checkpoint_restore,
)


class VllmCheckpointHookTest(unittest.TestCase):
    def test_sync_llm_dispatches_both_hooks_collectively(self) -> None:
        target = mock.Mock(spec=["collective_rpc"])
        target.collective_rpc.side_effect = [[None], [None]]

        prepared = checkpoint_prepare(target)
        restored = checkpoint_restore(target)

        self.assertEqual(prepared.dispatch, "collective_rpc")
        self.assertEqual(restored.dispatch, "collective_rpc")
        self.assertEqual(
            target.collective_rpc.call_args_list,
            [
                mock.call("checkpoint_prepare"),
                mock.call("checkpoint_restore"),
            ],
        )

    def test_direct_sync_hook_is_preferred(self) -> None:
        target = mock.Mock(spec=["checkpoint_prepare", "collective_rpc"])
        target.checkpoint_prepare.return_value = "prepared"
        result = checkpoint_prepare(target)
        self.assertEqual(result.dispatch, "direct")
        self.assertEqual(result.response, "prepared")
        target.collective_rpc.assert_not_called()

    def test_nested_engine_collective_rpc_is_supported(self) -> None:
        target = mock.Mock(spec=["llm_engine"])
        target.llm_engine = mock.Mock(spec=["collective_rpc"])
        call_checkpoint_hook(target, "checkpoint_prepare")
        target.llm_engine.collective_rpc.assert_called_once_with(
            "checkpoint_prepare"
        )

    def test_missing_boundary_fails_closed(self) -> None:
        with self.assertRaises(CheckpointHookError):
            checkpoint_prepare(object())

    def test_unknown_operation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            call_checkpoint_hook(mock.Mock(), "checkpoint")

    def test_async_direct_hook_is_awaited(self) -> None:
        class Target:
            async def checkpoint_prepare(self) -> str:
                return "prepared"

        result = asyncio.run(
            call_checkpoint_hook_async(Target(), "checkpoint_prepare")
        )
        self.assertEqual(result.dispatch, "direct")
        self.assertEqual(result.response, "prepared")


if __name__ == "__main__":
    unittest.main()

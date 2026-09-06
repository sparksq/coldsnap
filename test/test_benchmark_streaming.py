# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks" / "harnesses"
RUNTIME_SHARED = Path(__file__).resolve().parents[1] / "runtime" / "shared"
RUNTIME_ENGINE = Path(__file__).resolve().parents[1] / "runtime" / "engine"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
for runtime_path in (RUNTIME_SHARED, RUNTIME_ENGINE):
    if str(runtime_path) not in sys.path:
        sys.path.insert(0, str(runtime_path))

import coldsnap_engine_rank_n610 as cuda_criu_node  # noqa: E402
import sparkrun_restore_ttft as restore_ttft  # noqa: E402


class BenchmarkStreamingTest(unittest.TestCase):
    def test_rank0_acceptance_reuses_receipt_without_any_inference(self) -> None:
        args = SimpleNamespace(rank0_source="acceptance", timeout=1, docker_host="head", prompt="Reply OK", expected="OK")
        start = restore_ttft._rfc3339_ns("2026-09-06T12:00:00Z")
        acceptance = {
            "format": 1, "observer": "rank0", "measurement": "rank0-acceptance-v1", "response_validated": True,
            "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(), "max_tokens": 64, "temperature": 0,
            "first_token_unix_ns": start + 3_000_000_000, "first_token_field": "content",
            "readiness": {"port_open_unix_ns": start + 1_000_000_000, "http_ready_unix_ns": start + 2_000_000_000},
        }
        report = {"rank": 0, "post_restore_response": {
            "choices": [{"message": {"content": "OK"}}], "coldsnap_acceptance": acceptance,
        }}
        with patch.object(restore_ttft, "_remote", side_effect=[json.dumps(report), "2026-09-06T12:00:00Z"]) as remote, \
             patch.object(restore_ttft, "_stream") as stream:
            result = restore_ttft._rank0(args, "current", start)
        self.assertEqual(result["container_to_first_token_seconds"], 3)
        self.assertEqual(result["container_to_health_seconds"], 2)
        self.assertEqual(result["container_to_port_open_seconds"], 1)
        self.assertEqual(remote.call_count, 2)
        stream.assert_not_called()
        for field, invalid in (("prompt_sha256", "wrong"), ("max_tokens", 128), ("response_validated", False)):
            saved = acceptance[field]
            acceptance[field] = invalid
            with patch.object(restore_ttft, "_remote", return_value=json.dumps(report)), self.assertRaises(RuntimeError):
                restore_ttft._rank0(args, "current", start)
            acceptance[field] = saved
        with patch.object(restore_ttft, "_remote", side_effect=[json.dumps(report), "2026-09-06T12:00:01Z"]), \
             self.assertRaisesRegex(RuntimeError, "restarted"):
            restore_ttft._rank0(args, "current", start)

    def test_rank0_baseline_uses_upstream_full_validation_probe(self) -> None:
        args = SimpleNamespace(rank0_source="inference", timeout=1, docker_host="head", prompt="Reply OK", expected="OK", api_base="http://head:8000")
        module = ModuleType("sparkrun.orchestration.startup")
        module.run_probe = Mock(return_value={
            "measurement": "sparkrun-rank0-v1", "container_started_unix_ns": 1_000_000_000,
            "first_token_unix_ns": 4_000_000_000, "first_token_field": "reasoning",
            "response_validated": True,
        })
        with patch.dict(sys.modules, {"sparkrun.orchestration.startup": module}):
            result = restore_ttft._rank0(args, "current", 1_000_000_000)
        self.assertEqual(module.run_probe.call_count, 1)
        self.assertEqual(module.run_probe.call_args.args[1]["expected"], "OK")
        self.assertEqual(result["container_to_first_token_seconds"], 3)
        self.assertEqual(result["measurement"], "sparkrun-rank0-v1")

    def test_post_launch_ttft_observer_selects_newest_existing_container(self) -> None:
        args = SimpleNamespace(timeout=1.0, docker_host="host")
        rows = [("old", "rank-0-old"), ("new", "rank-0-new")]
        timestamps = {
            "old": "2026-08-27T12:00:00.000000000Z",
            "new": "2026-08-27T12:00:01.000000000Z",
        }

        with (
            patch.object(restore_ttft, "_container_rows", return_value=rows),
            patch.object(
                restore_ttft,
                "_remote",
                side_effect=lambda _host, *_args: timestamps[_args[-1]],
            ),
        ):
            container_id, name, _ = restore_ttft._wait_container(args, set())

        self.assertEqual(container_id, "new")
        self.assertEqual(name, "rank-0-new")

    def test_criu_log_profile_finds_late_hook_and_largest_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restore.log"
            path.write_text(
                "(00.010000) Reading image tree\n"
                "(00.020000) Running pre-restore scripts\n"
                "(02.500000) Run late stage hook from criu master for external devices\n"
                "(02.750000) Restore finished successfully. Tasks resumed.\n",
                encoding="utf-8",
            )

            profile = cuda_criu_node._criu_log_profile(path)

        assert profile is not None
        self.assertEqual(profile["milestone_seconds"]["late_external_device_hook"], 2.5)
        self.assertAlmostEqual(profile["largest_timed_gaps"][0]["seconds"], 2.48)

    def test_tp2_criu_command_uses_process_tree_rpc_without_plugins(self) -> None:
        args = SimpleNamespace(
            criu_rpc=Path("/runtime/coldsnap-criu-rpc"),
            criu=Path("/opt/criu-runtime/bin/criu"),
            cuda_checkpoint=Path("/usr/local/bin/cuda-checkpoint"),
            artifact_root=Path("/snapshot"),
            timeout=1800.0,
            ghost_limit=64 * 1024**2,
            criu_compress_block_bytes=256 * 1024,
            criu_compress_acceleration=1,
            criu_decompress_threads=2,
            criu_image_io_mode="writeback",
        )

        dump = cuda_criu_node._criu_command(args, "dump", "dump.log")
        restore = cuda_criu_node._criu_command(args, "restore", "restore.log")

        self.assertEqual(dump[0], "/runtime/coldsnap-criu-rpc")
        self.assertIn("--cuda-process-tree", dump)
        self.assertIn("--tcp-established", dump)
        self.assertIn("--ghost-limit", dump)
        self.assertEqual(dump[dump.index("--compress-block-size") + 1], "262144")
        self.assertEqual(dump[dump.index("--compress-acceleration") + 1], "1")
        self.assertNotIn("--compress-block-size", restore)
        self.assertEqual(restore[restore.index("--decompress-threads") + 1], "2")
        self.assertEqual(restore[restore.index("--image-io-mode") + 1], "writeback")
        self.assertEqual(dump[dump.index("--network-lock") + 1], "nftables")
        self.assertNotIn("--network-lock", restore)
        self.assertNotIn("--libdir", dump)
        self.assertNotIn("cuda_plugin.so", dump)


if __name__ == "__main__":
    unittest.main()

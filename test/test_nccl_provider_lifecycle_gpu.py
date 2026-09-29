# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Opt-in real NCCL library lifecycle regressions on one local CUDA GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("COLDSNAP_NCCL_TEST_SOURCE")


@unittest.skipUnless(SOURCE, "set COLDSNAP_NCCL_TEST_SOURCE to a built patched NCCL source tree")
class NcclProviderLifecycleGpuTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("COLDSNAP_NCCL_TEST_TLS") == "1", "set COLDSNAP_NCCL_TEST_TLS=1 for a TLS-enabled runtime")
    def test_encrypted_communicator_lifecycle(self) -> None:
        for mode in ("tls-recreate", "tls-inplace"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                result = subprocess.run([
                    sys.executable,
                    str(ROOT / "benchmarks/harnesses/nccl_provider_lifecycle_smoke.py"),
                    "--nccl-source", str(SOURCE), "--mode", mode, "--output", directory,
                ], capture_output=True, text=True, timeout=180)
                logs = "\n".join(p.read_text() for p in sorted(output.glob("rank-*.log")))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr + logs)
                self.assertTrue(json.loads((output / "result.json").read_text())["passed"])
                for rank in range(2):
                    rows = [json.loads(line) for line in (output / f"rank-{rank}.log").read_text().splitlines()
                            if line.startswith("{")]
                    self.assertTrue(rows[-1]["passed"])
                    self.assertEqual(sum(row.get("phase") == "tls-detached" for row in rows), 2)
                    self.assertEqual(sum(row.get("phase") == "tls-restored" for row in rows), 2)
                    for row in rows:
                        if row.get("phase") == "tls-detached":
                            self.assertEqual(row["tls_active_connections"], 0)
                            self.assertEqual(row["tls_reseed_pending"], 1)
                        elif row.get("phase") == "tls-restored":
                            self.assertGreater(row["tls_active_connections"], 0)
                            self.assertEqual(row["tls_reseed_pending"], 0)

    def test_real_local_window_reconstruction(self) -> None:
        for flags in (0, 1, 2, 4):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                result = subprocess.run([
                    sys.executable,
                    str(ROOT / "benchmarks/harnesses/nccl_provider_lifecycle_smoke.py"),
                    "--nccl-source", str(SOURCE), "--mode", f"window-{flags}", "--output", directory,
                ], capture_output=True, text=True, timeout=180)
                log = (output / "rank-0.log").read_text() if (output / "rank-0.log").exists() else ""
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr + log)
                summary = json.loads((output / "result.json").read_text())
                self.assertTrue(summary["passed"])
                self.assertEqual(summary["ranks"], 1)
                self.assertFalse(summary["criu"])
                rows = [json.loads(line) for line in log.splitlines() if line.startswith("{")]
                self.assertTrue(rows[-1]["original_window_deregistered"])
                self.assertEqual(rows[-1]["window_flags"], flags)
                self.assertEqual(sum(row.get("phase") == "restored" for row in rows), 2)
                self.assertTrue(any(row.get("phase") == "after-rejection" for row in rows))
                inventories = [row for row in rows if row.get("kind") == "coldsnap-nccl-resource-inventory"]
                self.assertTrue(any(not row["admitted"] for row in inventories))
                for row in inventories:
                    comm = row["communicators"][0]
                    self.assertFalse(comm["gin_active"])
                    self.assertFalse(comm["rma_active"])
                    self.assertFalse(comm["cft_active"])

    def test_repeated_lifecycle_and_preflight_preserve_collectives(self) -> None:
        for mode in ("recreate", "inplace", "registration", "progress-recreate", "progress-inplace"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                result = subprocess.run([
                    sys.executable,
                    str(ROOT / "benchmarks/harnesses/nccl_provider_lifecycle_smoke.py"),
                    "--nccl-source", str(SOURCE), "--mode", mode, "--output", directory,
                ], capture_output=True, text=True, timeout=180)
                logs = "\n".join(path.read_text() for path in sorted(output.glob("rank-*.log")))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr + logs)
                self.assertTrue(json.loads((output / "result.json").read_text())["passed"])
                for rank in range(2):
                    records = [json.loads(line) for line in (output / f"rank-{rank}.log").read_text().splitlines()
                               if line.startswith("{")]
                    self.assertTrue(records[-1]["passed"])
                    self.assertEqual(sum(row.get("phase") == "restored" for row in records), 2)
                    inventories = [row for row in records if row.get("kind") == "coldsnap-nccl-resource-inventory"]
                    self.assertGreaterEqual(len(inventories), 4)
                    for inventory in inventories:
                        self.assertTrue(inventory["complete"])
                        self.assertEqual(len(inventory["communicators"]), 2)
                        for comm in inventory["communicators"]:
                            self.assertEqual(comm["nvls_host_mode"], 2147483647)
                            enabled = mode.startswith("progress-")
                            self.assertEqual(comm["progress_counters"], enabled)
                            self.assertEqual(comm["progress_monitors"], int(enabled))
                            self.assertEqual(comm["progress_monitor_registered"], int(enabled))
                    if mode == "registration":
                        self.assertTrue(any(not row["admitted"] for row in inventories))
                        self.assertTrue(any(row.get("phase") == "after-rejection" for row in records))
                    if mode in ("inplace", "progress-inplace"):
                        self.assertTrue(any(row.get("phase") == "graph-after-rejection" for row in records))


if __name__ == "__main__":
    unittest.main()

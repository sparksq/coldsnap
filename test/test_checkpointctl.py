# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "shared"))
from checkpointctl import (  # noqa: E402
    FORMAT,
    KIND,
    _pattern,
    feature_profile,
    probe,
)


class CheckpointProbeTest(unittest.TestCase):
    def test_pattern_is_deterministic_across_block_boundary(self) -> None:
        first = _pattern(8193)
        second = _pattern(8193)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8193)
        self.assertEqual(first[0], first[4096])

    def test_missing_cuda_library_is_structured_unsupported_result(self) -> None:
        with (
            patch("checkpointctl.CUDA", side_effect=OSError("missing")),
            patch("checkpointctl._driver_version", return_value="580.0"),
            patch("checkpointctl._criu_facts", return_value={"installed": False}),
        ):
            result = probe(False, 1 << 20, 1.0)
        self.assertEqual(result["format"], FORMAT)
        self.assertEqual(result["kind"], KIND)
        self.assertFalse(result["library_loaded"])
        self.assertFalse(result["api_available"])
        self.assertIn("missing", result["error"])

    def test_profile_keeps_unsupported_unqualified_and_failed_distinct(self) -> None:
        checkpoint = {
            "library_loaded": True,
            "api_available": False,
            "symbols": {"cuCheckpointProcessGetState": False},
            "roundtrip_supported": None,
            "completed_unix": 2.0,
            "driver_api": {},
        }
        with (
            patch("checkpointctl.probe", return_value=checkpoint),
            patch("checkpointctl.probe_vmm_exact_address", return_value={
                "id": "cuda-vmm-exact-address", "status": "failed",
                "probe": "cuda-vmm-exact-v1", "seconds": 0.1,
                "reason": "failed", "failure_phase": "map",
            }),
            patch("checkpointctl.probe_classic_ipc", return_value={
                "id": "cuda-classic-ipc", "status": "passed",
                "probe": "cuda-classic-ipc-v1", "seconds": 0.1,
            }),
            patch("checkpointctl._driver_version", return_value="610.43.02"),
            patch("checkpointctl._criu_facts", return_value={"installed": True}),
        ):
            result = feature_profile(1 << 20, 1.0)
        statuses = {item["id"]: item["status"] for item in result["features"]}
        self.assertEqual(statuses["cuda-checkpoint-api"], "unsupported")
        self.assertEqual(statuses["cuda-vmm-exact-address"], "failed")
        self.assertEqual(statuses["cuda-vmm-exported-handle-checkpoint"], "unsupported")
        self.assertEqual(statuses["cuda-classic-ipc"], "passed")


if __name__ == "__main__":
    unittest.main()

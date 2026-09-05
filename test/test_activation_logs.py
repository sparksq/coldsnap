# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT / "shared"))
sys.path.insert(0, str(RUNTIME_ROOT / "engine"))
MODULE_PATH = RUNTIME_ROOT / "engine" / "coldsnap_activation_logs.py"
SPEC = importlib.util.spec_from_file_location("coldsnap_activation_logs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
activation_logs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(activation_logs)


class ActivationLogsTest(unittest.TestCase):
    def test_capture_log_is_archived_before_target_is_truncated_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.log"
            captured = b"capture line one\ncapture line two\n"
            target.write_bytes(captured)
            target.chmod(0o640)
            target_inode = target.stat().st_ino

            record = activation_logs.archive_capture_log(root)

            self.assertEqual((root / "capture.log").read_bytes(), captured)
            self.assertEqual(target.read_bytes(), b"")
            self.assertEqual(target.stat().st_ino, target_inode)
            self.assertEqual((root / "capture.log").stat().st_mode & 0o777, 0o640)
            self.assertEqual(
                record,
                {
                    "format": 1,
                    "kind": "coldsnap-capture-log",
                    "path": "capture.log",
                    "bytes": len(captured),
                    "sha256": hashlib.sha256(captured).hexdigest(),
                },
            )

    def test_restore_preserves_legacy_capture_log_and_starts_a_new_activation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.log"
            captured = b"capture-only history\n"
            target.write_bytes(captured)
            target_inode = target.stat().st_ino
            restored_descriptor = os.open(target, os.O_WRONLY | os.O_APPEND)
            # CRIU restores the target's capture-time file offset. O_APPEND
            # must still put the first process write immediately after the
            # shorter activation marker rather than leaving a sparse gap.
            os.lseek(restored_descriptor, 64 * 1024, os.SEEK_SET)
            environment = {
                "COLDSNAP_CAPTURE_ID": "capture-3",
                "COLDSNAP_EXPECTED_UNIT": "unit-1",
            }
            try:
                with mock.patch.dict(os.environ, environment, clear=True):
                    boundary = activation_logs.restore_log_boundary(
                        artifact_root=root,
                        activation_namespace="activation-7",
                        rank=1,
                        capture_report={"generation": "process-generation-2"},
                    )
                os.write(restored_descriptor, b"restored process output\n")
            finally:
                os.close(restored_descriptor)

            self.assertEqual((root / "capture.log").read_bytes(), captured)
            self.assertEqual(target.stat().st_ino, target_inode)
            target_contents = target.read_bytes()
            marker, restored = target_contents.split(b"\n", 1)
            self.assertEqual(restored, b"restored process output\n")
            prefix = b"--- ColdSnap restore boundary "
            suffix = b" ---"
            self.assertTrue(marker.startswith(prefix))
            self.assertTrue(marker.endswith(suffix))
            marker_boundary = json.loads(marker[len(prefix) : -len(suffix)])
            self.assertEqual(marker_boundary, boundary)
            self.assertEqual(boundary["activation_namespace"], "activation-7")
            self.assertEqual(boundary["capture_id"], "capture-3")
            self.assertEqual(boundary["process_generation"], "process-generation-2")
            self.assertEqual(boundary["rank"], 1)
            self.assertEqual(boundary["unit"], "unit-1")
            self.assertTrue(boundary["started_at"].endswith("Z"))
            self.assertNotIn(captured, target_contents)

    def test_restore_keeps_the_capsule_capture_log_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_log = root / "capture.log"
            target = root / "target.log"
            capture_log.write_bytes(b"canonical capture history\n")
            target.write_bytes(b"stale activation history\n")

            activation_logs.restore_log_boundary(
                artifact_root=root,
                activation_namespace="activation-8",
                rank=0,
                capture_report={"generation": "generation"},
            )

            self.assertEqual(capture_log.read_bytes(), b"canonical capture history\n")
            self.assertNotIn(b"stale activation history", target.read_bytes())
            self.assertTrue(target.read_bytes().startswith(b"--- ColdSnap restore boundary "))

    def test_restore_rejects_a_capture_log_symlink_without_resetting_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.log"
            target.write_bytes(b"unmodified target\n")
            outside = root / "outside.log"
            outside.write_bytes(b"outside\n")
            (root / "capture.log").symlink_to(outside)

            with self.assertRaisesRegex(RuntimeError, "capture log is not a regular file"):
                activation_logs.restore_log_boundary(
                    artifact_root=root,
                    activation_namespace="activation-9",
                    rank=0,
                    capture_report={"generation": "generation"},
                )

            self.assertEqual(target.read_bytes(), b"unmodified target\n")
            self.assertEqual(outside.read_bytes(), b"outside\n")


if __name__ == "__main__":
    unittest.main()

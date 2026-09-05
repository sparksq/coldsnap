# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "runtime" / "engine" / "coldsnap_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("coldsnap_lifecycle_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)


class LifecycleHelperTests(unittest.TestCase):
    def test_synchronize_rebinds_capture_identity_then_evidence_admits_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original = LIFECYCLE.HIBERNATE_ROOT
            LIFECYCLE.HIBERNATE_ROOT = Path(temporary)
            try:
                path = LIFECYCLE.HIBERNATE_ROOT / "worker-0.json"
                path.write_text(
                    json.dumps(
                        {
                            "worker_id": "worker-0",
                            "capture_id": "capture-host-specific",
                            "state": "sleeping",
                            "pid": 42,
                        }
                    ),
                    encoding="utf-8",
                )
                LIFECYCLE.synchronize_evidence(
                    Namespace(
                        state="running",
                        capture="capture-portable",
                        workers='["worker-0"]',
                    )
                )
                value = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(value["capture_id"], "capture-portable")
                self.assertEqual(value["state"], "running")
                output = io.StringIO()
                with redirect_stdout(output):
                    LIFECYCLE.evidence(
                        Namespace(
                            capture="capture-portable",
                            workers='["worker-0"]',
                        )
                    )
                reported = json.loads(output.getvalue())
                self.assertEqual(reported[0]["worker_id"], "worker-0")
                self.assertEqual(reported[0]["capture_id"], "capture-portable")
            finally:
                LIFECYCLE.HIBERNATE_ROOT = original

    def test_worker_inventory_is_strict(self) -> None:
        for invalid in ("[]", '["worker-0", "worker-0"]', '[""]', "{}"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RuntimeError):
                    LIFECYCLE._expected_workers(invalid)


if __name__ == "__main__":
    unittest.main()

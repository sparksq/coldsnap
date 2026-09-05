# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import ModuleType


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

from coldsnap_import_hook import after_module_import  # noqa: E402


class PostImportHookTest(unittest.TestCase):
    def test_callback_runs_after_module_body_and_only_once(self) -> None:
        module_name = f"coldsnap_test_{uuid.uuid4().hex}"
        events: list[object] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"{module_name}.py").write_text(
                "value = 'loaded'\n", encoding="utf-8"
            )
            sys.path.insert(0, directory)
            try:
                self.assertTrue(
                    after_module_import(
                        module_name,
                        "test",
                        lambda module: events.append(module.value),
                    )
                )
                self.assertFalse(
                    after_module_import(module_name, "test", lambda module: None)
                )
                module = importlib.import_module(module_name)
                self.assertEqual(module.value, "loaded")
                self.assertEqual(events, ["loaded"])
                self.assertFalse(
                    after_module_import(module_name, "test", lambda module: None)
                )
            finally:
                sys.path.remove(directory)
                sys.modules.pop(module_name, None)

    def test_already_loaded_module_runs_immediately(self) -> None:
        module_name = f"coldsnap_test_{uuid.uuid4().hex}"
        module = ModuleType(module_name)
        module.value = 17
        sys.modules[module_name] = module
        observed: list[int] = []
        try:
            self.assertTrue(
                after_module_import(
                    module_name,
                    "immediate",
                    lambda loaded: observed.append(loaded.value),
                )
            )
        finally:
            sys.modules.pop(module_name, None)
        self.assertEqual(observed, [17])

    def test_callback_failure_aborts_target_import(self) -> None:
        module_name = f"coldsnap_test_{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"{module_name}.py").write_text("value = 1\n", encoding="utf-8")
            sys.path.insert(0, directory)
            try:
                after_module_import(
                    module_name,
                    "failure",
                    lambda module: (_ for _ in ()).throw(RuntimeError("incompatible")),
                )
                with self.assertRaisesRegex(RuntimeError, "incompatible"):
                    importlib.import_module(module_name)
                self.assertNotIn(module_name, sys.modules)
            finally:
                sys.path.remove(directory)
                sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()

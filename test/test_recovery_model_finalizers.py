# SPDX-License-Identifier: AGPL-3.0-only
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
from coldsnap_recovery_loader import _defer_recovery_model_finalizers


class RecoveryModelFinalizerTests(unittest.TestCase):
    def model_type(self, module="vllm.models.deepseek_v4.nvidia.model"):
        def finalize(self, value=None):
            self.events.append(("finalize", self.live, value))
            if not self.live:
                raise RuntimeError("meta source")
        return type("DeepseekV4ForCausalLM", (), {
            "__module__": module, "process_weights_after_loading": finalize,
        })

    def test_finalizes_after_reload_and_restores_method_binding(self):
        model = self.model_type()()
        model.events, model.live = [], False
        with _defer_recovery_model_finalizers(model):
            model.process_weights_after_loading(value=7)
            self.assertEqual(model.events, [])
            model.live = True
        self.assertEqual(model.events, [("finalize", True, 7)])
        self.assertNotIn("process_weights_after_loading", vars(model))

    def test_failed_reload_does_not_finalize_and_restores_instance_override(self):
        model = self.model_type()()
        calls = []
        def original():
            calls.append(True)
        model.process_weights_after_loading = original
        with self.assertRaisesRegex(RuntimeError, "reload failed"):
            with _defer_recovery_model_finalizers(model):
                model.process_weights_after_loading()
                raise RuntimeError("reload failed")
        self.assertEqual(calls, [])
        self.assertIs(model.process_weights_after_loading, original)

    def test_unqualified_model_is_not_intercepted(self):
        model = self.model_type("third_party.models")()
        model.events, model.live = [], True
        with _defer_recovery_model_finalizers(model):
            model.process_weights_after_loading()
            self.assertEqual(model.events, [("finalize", True, None)])

    def test_finalizer_failure_restores_hook(self):
        model = self.model_type()()
        model.events, model.live = [], False
        with self.assertRaisesRegex(RuntimeError, "meta source"):
            with _defer_recovery_model_finalizers(model):
                model.process_weights_after_loading()
        self.assertNotIn("process_weights_after_loading", vars(model))

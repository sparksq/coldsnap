# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Real B12x/Torch regression for native gate/up order provenance (CPU only)."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
import coldsnap_b12x_native_state as state  # noqa: E402
import coldsnap_recovery_loader as recovery  # noqa: E402


class B12xNativeStateTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from b12x.moe import fused_moe
            from b12x.moe.fused_moe import _impl
        except ImportError:
            self.skipTest("requires the pinned vLLM image's Torch and B12x")
        self.torch, self.moe, self.impl = torch, fused_moe, _impl
        self.model = torch.nn.Module()
        self.owners = []
        self.enterContext(patch.object(recovery, "_b12x_prepared_owners", side_effect=lambda model: self.owners))

    def weights(self):
        torch, moe = self.torch, self.moe
        if hasattr(moe, "PackedSource"):
            plan = moe.plan_weights(
                source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
                activation=moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
                geometry=moe.MoEGeometry(num_experts=2, hidden_size=128, intermediate_size=128),
            )
        else:
            plan = moe.plan_weights(
                quant_modes="nvfp4", source_format="modelopt_nvfp4", activation="silu",
                params_dtype=torch.bfloat16, num_experts=2, hidden_size=128,
                intermediate_size=128, w13_layout="w31",
            )
        storage = self.impl.B12XFP4ExpertWeights(
            plan=getattr(plan, "_impl", plan), a1_gscale=torch.ones(2), a2_gscale=torch.ones(2),
            w1_fp4=torch.zeros((2, 256, 64), dtype=torch.uint8),
            w1_blockscale=torch.arange(4096, dtype=torch.int32).remainder(251).to(torch.uint8).reshape(2, 256, 8),
            w1_alphas=torch.ones(2), w2_alphas=torch.ones(2),
            w2_fp4=torch.zeros((2, 128, 64), dtype=torch.uint8),
            w2_blockscale=torch.zeros((2, 128, 8), dtype=torch.uint8),
        )
        storage.w1_fp4[:, :128].fill_(11)
        storage.w1_fp4[:, 128:].fill_(37)
        key = (storage.w1_fp4.data_ptr(), storage.w1_blockscale.data_ptr())
        self.addCleanup(self.impl._W13_NORMALIZED_STORAGES.pop, key, None)
        prepared = (moe.PreparedExperts(plan=plan, _impl=storage)
                    if hasattr(moe, "PreparedExperts") else storage)
        self.owners = [("experts", self.model, object(), prepared)]
        return storage

    def normalize(self, storage):
        self.impl._ensure_w13_kernel_order_inplace(
            storage.w1_fp4, storage.w1_blockscale, n=128, k=128, quant_mode="nvfp4",
        )

    def test_native_kernel_order_is_not_normalized_twice(self):
        captured = self.weights()
        self.normalize(captured)
        metadata = state.capture_native_state(self.model)
        self.assertIs(metadata['owners'][0]['normalized'], True)
        fresh = self.weights()
        fresh.w1_fp4.copy_(captured.w1_fp4)
        fresh.w1_blockscale.copy_(captured.w1_blockscale)
        pointers = (fresh.w1_fp4.data_ptr(), fresh.w1_blockscale.data_ptr())
        # Without the process-local proof, first use reverses the admitted data.
        self.normalize(fresh)
        self.assertFalse(self.torch.equal(fresh.w1_fp4, captured.w1_fp4))
        fresh.w1_fp4.copy_(captured.w1_fp4)
        fresh.w1_blockscale.copy_(captured.w1_blockscale)
        self.impl._W13_NORMALIZED_STORAGES.pop(pointers)
        self.assertEqual(state.restore_native_state(self.model, metadata), 1)
        self.normalize(fresh)
        self.assertTrue(self.torch.equal(fresh.w1_fp4, captured.w1_fp4))
        self.assertTrue(self.torch.equal(fresh.w1_blockscale, captured.w1_blockscale))
        self.assertEqual(pointers, (fresh.w1_fp4.data_ptr(), fresh.w1_blockscale.data_ptr()))

    def test_source_order_clears_stale_proof_and_normalizes_once(self):
        captured = self.weights()
        metadata = state.capture_native_state(self.model)
        source_weight, source_scale = captured.w1_fp4.clone(), captured.w1_blockscale.clone()
        self.assertIs(metadata['owners'][0]['normalized'], False)
        self.normalize(captured)
        expected_weight, expected_scale = captured.w1_fp4.clone(), captured.w1_blockscale.clone()
        captured.w1_fp4.copy_(source_weight)
        captured.w1_blockscale.copy_(source_scale)
        state.restore_native_state(self.model, metadata)
        self.normalize(captured)
        self.assertTrue(self.torch.equal(captured.w1_fp4, expected_weight))
        self.assertTrue(self.torch.equal(captured.w1_blockscale, expected_scale))

    def test_missing_state_requires_recovery_or_recapture(self):
        self.weights()
        with self.assertRaisesRegex(RuntimeError, "recapture or use safetensors recovery"):
            state.restore_native_state(self.model, None)

    def test_invalid_state_cannot_change_registry(self):
        storage = self.weights()
        metadata = state.capture_native_state(self.model)
        self.normalize(storage)
        key = (storage.w1_fp4.data_ptr(), storage.w1_blockscale.data_ptr())
        for mutate in (
            lambda value: value['owners'][0].update(hidden_size=256),
            lambda value: value['owners'][0].update(normalized=1),
            lambda value: value['owners'].append(copy.deepcopy(value['owners'][0])),
            lambda value: value['owners'].clear(),
        ):
            bad = copy.deepcopy(metadata)
            mutate(bad)
            with self.assertRaises(RuntimeError):
                state.restore_native_state(self.model, bad)
            self.assertIn(key, self.impl._W13_NORMALIZED_STORAGES)

    def test_other_prepared_formats_need_no_registry_metadata(self):
        self.assertEqual(state.restore_native_state(self.model, None), 0)
        metadata = state.capture_native_state(self.model)
        self.assertEqual(metadata['owners'], [])
        self.assertEqual(state.restore_native_state(self.model, metadata), 0)


if __name__ == '__main__':
    unittest.main()

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Real B12x/Torch regression for native gate/up order provenance (CPU only)."""

import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
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

    def test_recovery_reloads_source_order_into_captured_storage(self):
        storage = self.weights()
        source = (storage.w1_fp4.clone(), storage.w1_blockscale.clone())
        self.normalize(storage)
        expected = (storage.w1_fp4.clone(), storage.w1_blockscale.clone())
        pointers = (storage.w1_fp4.data_ptr(), storage.w1_blockscale.data_ptr())
        saved = state.begin_recovery_reload(self.model)
        self.assertNotIn(pointers, self.impl._W13_NORMALIZED_STORAGES)
        storage.w1_fp4.copy_(source[0])
        storage.w1_blockscale.copy_(source[1])
        self.assertEqual(state.finish_recovery_reload(self.model, saved), 1)
        self.normalize(storage)
        self.assertTrue(self.torch.equal(storage.w1_fp4, expected[0]))
        self.assertTrue(self.torch.equal(storage.w1_blockscale, expected[1]))
        self.assertEqual(pointers, (storage.w1_fp4.data_ptr(), storage.w1_blockscale.data_ptr()))

    def test_recovery_respects_normalization_during_finalization(self):
        storage = self.weights()
        source = (storage.w1_fp4.clone(), storage.w1_blockscale.clone())
        self.normalize(storage)
        expected = (storage.w1_fp4.clone(), storage.w1_blockscale.clone())
        saved = state.begin_recovery_reload(self.model)
        storage.w1_fp4.copy_(source[0])
        storage.w1_blockscale.copy_(source[1])
        self.normalize(storage)
        state.finish_recovery_reload(self.model, saved)
        self.assertTrue(self.torch.equal(storage.w1_fp4, expected[0]))
        self.assertTrue(self.torch.equal(storage.w1_blockscale, expected[1]))

    def test_recovery_preserves_uncaptured_first_use_order(self):
        storage = self.weights()
        expected = storage.w1_fp4.clone()
        saved = state.begin_recovery_reload(self.model)
        state.finish_recovery_reload(self.model, saved)
        self.assertTrue(self.torch.equal(storage.w1_fp4, expected))
        self.assertNotIn((storage.w1_fp4.data_ptr(), storage.w1_blockscale.data_ptr()), self.impl._W13_NORMALIZED_STORAGES)

    def test_recovery_rejects_replacement_graph_storage(self):
        storage = self.weights()
        self.normalize(storage)
        saved = state.begin_recovery_reload(self.model)
        fresh = self.weights()
        expected = fresh.w1_fp4.clone()
        with self.assertRaisesRegex(RuntimeError, "changed captured gate/up storage"):
            state.finish_recovery_reload(self.model, saved)
        self.assertTrue(self.torch.equal(fresh.w1_fp4, expected))

    def test_recovery_without_b12x_owners_is_unchanged(self):
        saved = state.begin_recovery_reload(self.model)
        self.assertEqual(state.finish_recovery_reload(self.model, saved), 0)

    def execution_owner(self, prepared, plan):
        try:
            from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
        except ImportError:
            self.skipTest("image predates layer-owned B12x experts")
        if not hasattr(B12xExperts, "_plan_for_tokens"):
            self.skipTest("image uses the earlier lazy B12x plan cache")

        owner = B12xExperts.__new__(B12xExperts)
        owner._prepared_experts = prepared
        owner._plan = plan
        owner._plan_key = ((1, 16, 2048), "silu", False, id(prepared)) if plan else None
        owner._plan_activation = "silu" if plan else None
        owner._plan_route_on_input = False if plan else None
        owner._quant_mode = "nvfp4"
        owner._source_format = "modelopt_nvfp4"
        owner._w13_layout = "w31"
        owner._apply_router_weight_on_input = False
        return owner

    def test_recovery_reuses_captured_plan_on_rebuilt_owner(self):
        self.weights()
        prepared = self.owners[0][3]
        layer = self.model
        layer.activation, layer.apply_router_weight_on_input = "silu", False
        plan = SimpleNamespace(warmed=True)
        captured = self.execution_owner(prepared, plan)
        self.owners = [("experts", layer, captured, prepared)]
        records = state.capture_recovery_plans(self.model)
        fresh = self.execution_owner(prepared, None)
        self.owners = [("experts", layer, fresh, prepared)]
        with self.assertRaisesRegex(RuntimeError, "no prepared plan"):
            fresh._plan_for_tokens(63, activation="silu", apply_router_weight_on_input=False)
        self.assertEqual(state.restore_recovery_plans(self.model, records), 1)
        self.assertIs(fresh._plan_for_tokens(63, activation="silu", apply_router_weight_on_input=False), plan)
        self.assertIs(fresh._prepared(), prepared)
        self.assertEqual(fresh._plan_key, captured._plan_key)

    def test_recovery_plan_rejects_changed_storage_and_routing(self):
        storage = self.weights()
        prepared = self.owners[0][3]
        layer = self.model
        layer.activation, layer.apply_router_weight_on_input = "silu", False
        captured = self.execution_owner(prepared, object())
        self.owners = [("experts", layer, captured, prepared)]
        records = state.capture_recovery_plans(self.model)
        fresh = self.execution_owner(prepared, None)
        self.owners = [("experts", layer, fresh, prepared)]
        layer.apply_router_weight_on_input = True
        with self.assertRaisesRegex(RuntimeError, "changed captured execution bindings"):
            state.restore_recovery_plans(self.model, records)
        self.assertIsNone(fresh._plan)
        layer.apply_router_weight_on_input = False
        storage.w1_fp4.set_(storage.w1_fp4.clone())
        with self.assertRaisesRegex(RuntimeError, "changed captured execution bindings"):
            state.restore_recovery_plans(self.model, records)
        self.assertIsNone(fresh._plan)

    def test_recovery_plan_skips_legacy_and_unrelated_owners(self):
        self.weights()
        self.assertEqual(state.capture_recovery_plans(self.model), [])
        self.assertEqual(state.restore_recovery_plans(self.model, []), 0)
        try:
            from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
        except ImportError:
            return
        if hasattr(B12xExperts, "_plan_for_tokens"):
            return
        prepared = self.owners[0][3]
        legacy = B12xExperts.__new__(B12xExperts)
        self.assertTrue(callable(legacy._plan))
        self.owners = [("experts", self.model, legacy, prepared)]
        self.assertEqual(state.capture_recovery_plans(self.model), [])

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

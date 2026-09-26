# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Real-Torch coverage for native bootstrap scale and geometry contracts."""

import base64
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
import coldsnap_disk_backend as disk  # noqa: E402
import coldsnap_recovery_loader as recovery  # noqa: E402
import coldsnap_synthetic_loader as synthetic  # noqa: E402
from coldsnap_materializer import NeutralTensorMaterializer  # noqa: E402


class NativeBootstrapValuesTest(unittest.TestCase):
    def test_native_raw_hydration_refreshes_b12x_input_scale_proof(self):
        try:
            from b12x.moe import fused_moe
            from b12x.moe.fused_moe._impl import B12XFP4ExpertWeights
        except ImportError:
            self.skipTest("requires the current B12x prepared-weight API")
        if "_uniform_a1_scale" not in getattr(B12XFP4ExpertWeights, "__dataclass_fields__", {}):
            self.skipTest("older B12x has no cached input-scale proof")
        torch = self.torch
        if hasattr(fused_moe, "PackedSource"):
            plan = fused_moe.plan_weights(
                source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
                activation=fused_moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
                geometry=fused_moe.MoEGeometry(num_experts=2, hidden_size=128, intermediate_size=128),
            )
        else:
            plan = fused_moe.plan_weights(
                quant_modes="nvfp4", source_format="modelopt_nvfp4", activation="silu",
                params_dtype=torch.bfloat16, num_experts=2, hidden_size=128,
                intermediate_size=128, w13_layout="w31",
            )
        storage = B12XFP4ExpertWeights(
            plan=getattr(plan, "_impl", plan), a1_gscale=torch.ones(2), a2_gscale=torch.ones(2),
            w1_fp4=torch.zeros((2, 256, 64), dtype=torch.uint8),
            w1_blockscale=torch.zeros((2, 256, 8), dtype=torch.uint8),
            w1_alphas=torch.ones(2), w2_alphas=torch.ones(2),
            w2_fp4=torch.zeros((2, 128, 64), dtype=torch.uint8),
            w2_blockscale=torch.zeros((2, 128, 8), dtype=torch.uint8),
            immutable_input_scales=True,
        )
        prepared = (fused_moe.PreparedExperts(plan=plan, _impl=storage)
                    if hasattr(fused_moe, "PreparedExperts") else storage)
        model = torch.nn.Module()
        pointers = {name: getattr(storage, name).data_ptr() for name in recovery.B12X_PREPARED_WEIGHT_FIELDS}
        self.assertTrue(storage.can_share_input(input_scales_static=True))
        version = storage.a1_gscale._version
        # Native I/O mutates storage without a Torch version-counter update.
        storage.a1_gscale.data.copy_(torch.tensor([281., 726.]))
        self.assertEqual(storage.a1_gscale._version, version)
        self.assertTrue(storage.can_share_input(input_scales_static=True))
        with patch.object(recovery, "_b12x_prepared_owners", return_value=[("experts", model, object(), prepared)]):
            counts = recovery._refresh_initial_native_model(model)
        self.assertEqual(counts["expert_scale_proofs"], 1)
        self.assertFalse(storage.can_share_input(input_scales_static=True))
        self.assertEqual(pointers, {name: getattr(storage, name).data_ptr() for name in pointers})
        # Keep the valid uniform optimization when the real values prove it.
        storage.a1_gscale.data.fill_(281.)
        with patch.object(recovery, "_b12x_prepared_owners", return_value=[("experts", model, object(), prepared)]):
            recovery._refresh_initial_native_model(model)
        self.assertTrue(storage.can_share_input(input_scales_static=True))

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("requires real Torch; run in the pinned vLLM image")
        self.torch = torch
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.source = SimpleNamespace(model_or_path="model", prefix="", revision="pinned")
        self.enterContext(patch.object(synthetic, "_native_bootstrap_root", return_value=self.root))
        self.enterContext(patch.object(disk, "_worker_id", return_value="worker-0"))
        self.enterContext(patch.dict(os.environ, {
            "COLDSNAP_CAPTURE_ID": "capture", "COLDSNAP_EXPORT_MODEL_PAYLOAD": "1",
        }, clear=True))
        logger = ModuleType("vllm.logger")
        logger.init_logger = logging.getLogger
        self.enterContext(patch.dict(sys.modules, {logger.__name__: logger}))

    def metadata(self, name, tensor, dtype_name="I64"):
        descriptor = recovery.SafetensorDescriptor(
            name, dtype_name, tuple(tensor.shape), 64, tensor.numel() * tensor.element_size(),
        )
        return synthetic.native_bootstrap_tensor_metadata(name, descriptor, tensor)

    def bootstrap(self, entries):
        synthetic.record_native_bootstrap_source(self.source, "", entries)
        return dict(synthetic._native_bootstrap_weights(self.source))

    def test_quantization_scales_are_positive_with_bounded_storage(self):
        torch = self.torch
        for mode in ("expanded_zero", "dense_zero"):
            for dtype in (torch.float32, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu):
                for leaf in ("weight_scale", "weight_scale_2", "weight_scale_inv", "input_scale"):
                    with self.subTest(mode=mode, dtype=dtype, leaf=leaf):
                        m = NeutralTensorMaterializer(mode=mode)
                        value = m.tensor(torch, "model.embedding.shard_0." + leaf, (4, 8), dtype)
                        self.assertTrue(bool(torch.isfinite(value.float()).all()))
                        self.assertTrue(bool((value.float() == 1).all()))
                        self.assertEqual(value.untyped_storage().nbytes(),
                                         (32 if mode == "dense_zero" else 1) * value.element_size())
        m = NeutralTensorMaterializer()
        weights = m.tensor(torch, "model.embedding.weight", (100_000_000, 128), torch.uint8)
        self.assertEqual(weights.untyped_storage().nbytes(), 1)
        self.assertEqual(weights[0, 0].item(), 0)
        packed = m.tensor(torch, "model.weight_scale", (4,), torch.uint8)
        self.assertEqual(packed[0].item(), 0)

    def test_geometry_round_trip_requires_no_checkpoint_files(self):
        torch = self.torch
        expected = {
            "model.ple.layer_multipliers": torch.tensor([31, 127, 8191], dtype=torch.int64),
            "model.ple.ngram_heads_offsets": torch.tensor([0, 128, 256], dtype=torch.int64),
            "model.ple.ngram_heads_vocab_sizes": torch.tensor([128, 128, 128], dtype=torch.int64),
        }
        entries = [self.metadata(name, value) for name, value in expected.items()]
        entries.append({"name": "model.ple.shard_0.weight_scale", "dtype": "F8_E4M3",
                        "shape": [8, 2], "length": 16})
        entries.append({"name": "model.ple.weight_scale_2", "dtype": "F32",
                        "shape": [1], "length": 4})
        restored = self.bootstrap(entries)
        for name, value in expected.items():
            self.assertTrue(torch.equal(restored[name], value), name)
        self.assertTrue(bool((restored[entries[-2]["name"]].float() > 0).all()))
        self.assertEqual(restored[entries[-1]["name"]].item(), 1)

    def test_large_and_packed_payloads_are_not_embedded(self):
        torch = self.torch
        for dtype_name, dtype, shape in (
            ("I64", torch.int64, (8193,)),
            ("I8", torch.int8, (4,)),
            ("U8", torch.uint8, (4,)),
            ("F32", torch.float32, (4,)),
        ):
            with self.subTest(dtype=dtype_name, shape=shape):
                self.assertNotIn("bootstrap_data", self.metadata("weight", torch.zeros(shape, dtype=dtype), dtype_name))
        value = torch.tensor([True, False], dtype=torch.bool)
        restored = self.bootstrap([self.metadata("flags", value, "BOOL")])
        self.assertTrue(torch.equal(restored["flags"], value))

    def test_capture_records_geometry_before_consumer_mutation(self):
        torch = self.torch
        value = torch.tensor([31, 127, 8191], dtype=torch.int64)
        original = value.clone()
        descriptor = recovery.SafetensorDescriptor("geometry", "I64", (3,), 64, 24)
        prepared = SimpleNamespace(prefix="", files=["model.safetensors"])
        with patch.object(recovery, "_capture_source_descriptors", return_value=(
                prepared, {"geometry": ("model.safetensors", descriptor)})), \
             patch.object(recovery, "_distributed_context", return_value=(None, 0, 1)):
            iterator = recovery._observed_capture_iterator(SimpleNamespace(), self.source, [("geometry", value)])
            self.assertIs(next(iterator)[1], value)
            value.zero_()
            with self.assertRaises(StopIteration):
                next(iterator)
        restored = dict(synthetic._native_bootstrap_weights(self.source))
        self.assertTrue(torch.equal(restored["geometry"], original))

    def test_invalid_exact_metadata_fails_closed(self):
        valid = self.metadata("geometry", self.torch.tensor([3], dtype=self.torch.int64))
        for change in (
            {"bootstrap_data": "invalid!"},
            {"bootstrap_data": base64.b64encode(b"short").decode()},
            {"dtype": "F64"},
            {"length": 65544, "shape": [8193]},
        ):
            with self.subTest(change=change):
                with self.assertRaisesRegex(RuntimeError, "native bootstrap exact metadata"):
                    self.bootstrap([{**valid, **change}])

    def test_v1_and_v2_indices_remain_readable(self):
        entry = {"name": "model.weight", "dtype": "F16", "shape": [4, 8], "length": 64}
        path = synthetic.record_native_bootstrap_source(self.source, "", [entry])
        index = json.loads(path.read_text())
        for version in (1, 2):
            with self.subTest(version=version):
                index["format"] = version
                path.write_text(json.dumps(index))
                restored = dict(synthetic._native_bootstrap_weights(self.source))
                self.assertEqual(tuple(restored["model.weight"].shape), (4, 8))
                self.assertEqual(restored["model.weight"].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()

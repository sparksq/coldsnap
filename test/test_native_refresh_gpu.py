# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Opt-in tests in the qualified V4.1 image: COLDSNAP_TEST_V41_NATIVE_REFRESH=1."""

import os
import unittest
from types import SimpleNamespace


@unittest.skipUnless(os.environ.get("COLDSNAP_TEST_V41_NATIVE_REFRESH"), "V4.1 CUDA image required")
class NativeRefreshGpuTests(unittest.TestCase):
    def test_refresh_matches_real_packer_and_preserves_execution_storage(self):
        import torch
        from torch import nn
        from b12x.gemm import block_fp8_linear
        from vllm.models.deepseek_v4_1.b12x_layers import B12xFP8LinearMethod
        from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4DecoderLayer
        from coldsnap_recovery_loader import _refresh_initial_native_model

        torch.manual_seed(7)
        for width in (256, 4096):
            with self.subTest(width=width), torch.no_grad():
                model = nn.Module()
                layer = nn.Module()
                model.add_module("linear", layer)
                layer.quant_method = B12xFP8LinearMethod(SimpleNamespace(weight_block_size=[32, 32]))
                layer.weight = nn.Parameter(torch.zeros((width, width), device="cuda").to(torch.float8_e4m3fn), requires_grad=False)
                layer.weight_scale_inv = nn.Parameter(torch.full((width // 32, width // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu), requires_grad=False)
                layer.b12x_weight = block_fp8_linear.pack_weight(layer.weight, layer.weight_scale_inv, block_size=(32, 32))
                prepared = layer.b12x_weight
                plan = block_fp8_linear.plan(block_fp8_linear.Caps(
                    device=layer.weight.device, max_tokens=8, in_features=width,
                    out_features=width, block_size=(32, 32)))
                layer.b12x_plans = plans = (plan,)
                fields = ("values", "scale_rows", "scale_mma", "values_tiled")
                pointers = {field: getattr(prepared.weight, field).data_ptr() for field in fields if getattr(prepared.weight, field) is not None}
                mhc = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
                nn.Module.__init__(mhc)
                model.add_module("mhc", mhc)
                mhc.hc_mult, mhc.hidden_size = 4, 16
                mhc.hc_attn_fn = nn.Parameter(torch.zeros((8, 64), device="cuda"), requires_grad=False)
                mhc.hc_attn_fn_broadcast = torch.zeros((8, 16), device="cuda")
                broadcast_pointer = mhc.hc_attn_fn_broadcast.data_ptr()
                # Native hydration restores the registered parameters only.
                layer.weight.copy_(torch.randn((width, width), device="cuda").to(torch.float8_e4m3fn))
                layer.weight_scale_inv.view(torch.uint8).fill_(129)
                mhc.hc_attn_fn.copy_(torch.randn_like(mhc.hc_attn_fn))
                expected = block_fp8_linear.pack_weight(layer.weight, layer.weight_scale_inv, block_size=(32, 32))
                source = torch.randn((1, width), dtype=torch.bfloat16, device="cuda")
                scratch = [torch.empty(shape, dtype=dtype, device="cuda")
                           for shape, dtype in plan.shapes_and_dtypes()]
                output = torch.empty((1, width, 1), dtype=torch.bfloat16, device="cuda")

                def run(weight, plan=plan, scratch=scratch, source=source, output=output):
                    binding = block_fp8_linear.bind(plan, scratch=scratch, source=source,
                                                    packed_weight=weight, output=output)
                    return block_fp8_linear.run(binding=binding).clone()

                torch.cuda.synchronize()
                correct = run(expected)
                torch.cuda.synchronize()
                stale = run(prepared)
                torch.cuda.synchronize()
                self.assertFalse(torch.equal(stale, correct))
                counts = _refresh_initial_native_model(model)
                self.assertEqual(counts, {"block32_linears": 1, "block128_linears": 0, "wo_projections": 0, "mhc_broadcasts": 1})
                self.assertIs(layer.b12x_weight, prepared)
                self.assertIs(layer.b12x_plans, plans)
                for field, pointer in pointers.items():
                    actual = getattr(prepared.weight, field)
                    reference = getattr(expected.weight, field)
                    self.assertEqual(actual.data_ptr(), pointer)
                    self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)), field)
                self.assertEqual(mhc.hc_attn_fn_broadcast.data_ptr(), broadcast_pointer)
                self.assertTrue(torch.equal(mhc.hc_attn_fn_broadcast, mhc.hc_attn_fn.view(-1, 4, 16).sum(dim=1)))
                actual_output = run(prepared)
                torch.cuda.synchronize()
                repeat = run(expected)
                torch.cuda.synchronize()
                print({"width": width, "finite": bool(torch.isfinite(correct).all()),
                       "max_difference": float((actual_output.float() - correct.float()).abs().max()),
                       "repeat_difference": float((repeat.float() - correct.float()).abs().max())}, flush=True)
                self.assertTrue(torch.equal(actual_output, correct))
                torch.cuda.synchronize()

    def test_mxfp8_lm_head_native_layout_restores_real_logits(self):
        import torch
        from torch import nn
        from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel
        from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        from coldsnap_recovery_loader import _b12x_mxfp8_native_tensors, _native_model_payload_layout

        torch.manual_seed(11)
        kernel = B12xMxfp8LinearKernel(Mxfp8LinearLayerConfig())

        def make_model(weight):
            model = nn.Module()
            head = nn.Module()
            model.add_module("lm_head", head)
            head.quant_method = SimpleNamespace(kernel=kernel)
            values, scales = mxfp8_e4m3_quantize(weight)
            head.weight = nn.Parameter(values, requires_grad=False)
            head.weight_scale = nn.Parameter(scales, requires_grad=False)
            kernel.process_weights_after_loading(head)
            head.b12x_activation_mode = "a16"
            return model, head

        with torch.no_grad():
            checkpoint = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda")
            reference, correct_head = make_model(checkpoint)
            bootstrap, empty_head = make_model(torch.zeros_like(checkpoint))
            self.assertEqual(empty_head.weight.numel(), 0)
            self.assertEqual(empty_head.weight_scale.numel(), 0)
            source = torch.randn((2, 512), dtype=torch.bfloat16, device="cuda")
            expected = kernel.apply_weights(correct_head, source)
            stale = kernel.apply_weights(empty_head, source)
            self.assertTrue(bool(torch.isfinite(expected).all()))
            self.assertFalse(torch.equal(stale, expected))
            live = dict(_b12x_mxfp8_native_tensors(bootstrap))
            captured = dict(_b12x_mxfp8_native_tensors(reference))
            self.assertEqual(len(live), 3)
            allocations = {}
            pointers = {}
            for name, tensor in live.items():
                storage = tensor.untyped_storage()
                allocations[storage.data_ptr()] = SimpleNamespace(
                    pointer=storage.data_ptr(), size=storage.nbytes(), tag="weights")
                pointers[name] = tensor.data_ptr()
            backend = SimpleNamespace(memory_provider=SimpleNamespace(
                allocations=lambda tag: list(allocations.values())))
            layout = _native_model_payload_layout(bootstrap, backend)
            self.assertEqual(sum(size for _, size, _ in layout),
                             sum(t.numel() * t.element_size() for t in live.values()))
            for name, tensor in live.items():
                tensor.view(torch.uint8).copy_(captured[name].view(torch.uint8))
            actual = kernel.apply_weights(empty_head, source)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(pointers, {name: tensor.data_ptr() for name, tensor in _b12x_mxfp8_native_tensors(bootstrap)})
            print({"mxfp8_head_extents": len(layout), "logits_exact_after_hydration": True}, flush=True)

    def test_wo_a_prefill_storage_is_in_native_model_layout(self):
        import torch
        from torch import nn
        from unittest.mock import patch
        from vllm.models.deepseek_v4_1 import attention
        from vllm.models.deepseek_v4_1.b12x_layers import B12xFP8LinearMethod
        from coldsnap_recovery_loader import _v41_wo_a_native_tensors, _native_model_payload_layout

        layer = nn.Module()
        model = nn.Module()
        model.add_module("wo_a", layer)
        original = B12xFP8LinearMethod(SimpleNamespace(weight_block_size=[32, 32]))
        layer.quant_method = attention._GroupedLinearMethod(original, 2)
        layer.weight = nn.Parameter(torch.randn((512, 256), device="cuda").to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale_inv = nn.Parameter(torch.full((16, 8), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu), requires_grad=False)
        config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=8))
        with patch.object(attention, "get_current_vllm_config", return_value=config):
            layer.quant_method.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        tensors = _v41_wo_a_native_tensors(model)
        self.assertEqual(len(tensors), 6)
        self.assertTrue(all(tensor.is_contiguous() for _, tensor in tensors))
        for group in range(2):
            owner = layer.quant_method.prefill_weights[group].weight
            for field in ("values", "scale_rows", "scale_mma"):
                view = dict(tensors)[f"wo_a.wo_a_prefill.{group}.{field}"]
                self.assertEqual(view.data_ptr(), getattr(owner, field).data_ptr())
                self.assertEqual(view.numel(), getattr(owner, field).numel())
        # Actual storage envelopes stand in for the CuMem allocation inventory.
        allocations = {}
        for tensor in [layer.weight, *(tensor for _, tensor in tensors)]:
            storage = tensor.untyped_storage()
            allocations[storage.data_ptr()] = SimpleNamespace(
                pointer=storage.data_ptr(), size=storage.nbytes(), tag="weights")
        backend = SimpleNamespace(memory_provider=SimpleNamespace(
            allocations=lambda tag: list(allocations.values())))
        layout = _native_model_payload_layout(model, backend)
        self.assertEqual(sum(size for _, size, _ in layout),
                         layer.weight.numel() * 2 + sum(t.numel() * t.element_size() for _, t in tensors))
        self.assertEqual(len(layout), 7)
        torch.cuda.synchronize()

@unittest.skipUnless(os.environ.get("COLDSNAP_TEST_DSV4_NATIVE_REFRESH"), "DSv4 CUDA image required")
class Dsv4NativeRefreshGpuTests(unittest.TestCase):
    def test_modern_moe_native_layout_and_recovery_keep_prepared_storage(self):
        import torch
        from torch import nn
        from b12x.moe import fused_moe
        try:
            from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
        except ImportError:
            self.skipTest("layer-cached B12x expert API unavailable")
        from coldsnap_recovery_loader import (
            B12X_PREPARED_WEIGHT_FIELDS, B12X_SOURCE_STORAGE_FIELDS,
            _discover_b12x_recovery_storage, _model_weight_tensors,
            _native_model_payload_layout,
        )

        experts, hidden, intermediate = 8, 4096, 1024
        plan = fused_moe.plan_weights(
            quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
            params_dtype=torch.bfloat16, num_experts=experts, hidden_size=hidden,
            intermediate_size=intermediate, w13_layout="w31",
        )

        def prepare(value):
            return fused_moe.prepare_weights(
                plan=plan, params_dtype=torch.bfloat16,
                w1_fp4=torch.full((experts, 2 * intermediate, hidden // 2), value, device="cuda", dtype=torch.uint8),
                w2_fp4=torch.full((experts, hidden, intermediate // 2), value, device="cuda", dtype=torch.uint8),
                w1_blockscale=torch.full((experts, 2 * intermediate, hidden // 32), 117, device="cuda", dtype=torch.uint8),
                w2_blockscale=torch.full((experts, hidden, intermediate // 32), 117, device="cuda", dtype=torch.uint8),
                immutable_input_scales=True, a1_gscale=torch.ones(experts, device="cuda"),
                a2_gscale=torch.ones(experts, device="cuda"), w1_global_scale=torch.ones(experts, device="cuda"),
                w2_global_scale=torch.ones(experts, device="cuda"),
            )

        def run(prepared):
            plan = fused_moe.plan(fused_moe.Caps(
                max_tokens=8, num_topk=2, device="cuda:0", weight_plan=prepared.plan,
                quant_mode="w4a8_mx", swiglu_limit=10.0, core_token_counts=(8,),
            ))
            spec = plan.scratch_specs()[0]
            inputs = torch.full((8, hidden), 0.01, device="cuda", dtype=torch.bfloat16)
            output = torch.empty_like(inputs)
            binding = fused_moe.bind(
                plan, scratch=torch.empty(spec.shape, device="cuda", dtype=spec.dtype),
                a=inputs, experts=prepared,
                topk_ids=torch.arange(2, device="cuda", dtype=torch.int32).expand(8, -1).contiguous(),
                topk_weights=torch.full((8, 2), 0.5, device="cuda"), output=output,
                input_scales_static=True, unit_scale_contract=False,
            )
            fused_moe.run(binding=binding)
            torch.cuda.synchronize()
            return output.clone()

        with torch.no_grad():
            captured, bootstrap = prepare(0x22), prepare(0)
            expected = run(captured)
            self.assertTrue(bool(torch.isfinite(expected).all()))
            self.assertFalse(torch.equal(expected, run(bootstrap)))
            model, layer = nn.Module(), nn.Module()
            model.add_module("experts", layer)
            owner = B12xExperts.__new__(B12xExperts)
            owner._prepared_experts = bootstrap
            owner._source_parameters_released = True
            owner._plans = {}
            layer.quant_method = SimpleNamespace(moe_kernel=SimpleNamespace(fused_experts=owner))
            layer._b12x_prepared_experts = bootstrap
            layer.b12x_warmup_provider = owner
            for name, _ in B12X_SOURCE_STORAGE_FIELDS:
                layer.register_parameter(name, nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False))
            tensors = {name: getattr(bootstrap, name) for name in B12X_PREPARED_WEIGHT_FIELDS}
            allocations = {}
            for tensor in tensors.values():
                storage = tensor.untyped_storage()
                allocations[storage.data_ptr()] = SimpleNamespace(pointer=storage.data_ptr(), size=storage.nbytes(), tag="weights")
            backend = SimpleNamespace(memory_provider=SimpleNamespace(allocations=lambda tag: list(allocations.values())))
            layout = _native_model_payload_layout(model, backend)
            self.assertEqual(len(_model_weight_tensors(model)), 4)
            for tensor in tensors.values():
                start, end = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
                self.assertTrue(any(pointer <= start and end <= pointer + size for pointer, size, _ in layout))
            pointers = {name: tensor.data_ptr() for name, tensor in tensors.items()}
            for name, tensor in tensors.items():
                tensor.copy_(getattr(captured, name))
            self.assertTrue(torch.equal(run(bootstrap), expected))
            adapter, = _discover_b12x_recovery_storage(model)
            adapter.begin_reload()
            self.assertIs(layer._b12x_prepared_experts, bootstrap)
            self.assertIsNone(owner._prepared_experts)
            # Exercise the image's actual packed-storage reuse method with a
            # newly prepared result, as its quantization finalizer does.
            rebuilt = owner._reuse_prepared_storage(layer, prepare(0x33))
            owner._source_parameters_released = True
            adapter.materialized = True
            adapter.finish_reload()
            self.assertIs(rebuilt, bootstrap)
            self.assertEqual(pointers, {name: getattr(rebuilt, name).data_ptr() for name in pointers})
            self.assertTrue(torch.equal(run(rebuilt), run(prepare(0x33))))
            print({"moe_native_layout_bytes": sum(size for _, size, _ in layout),
                   "native_moe_output_exact": True, "recovery_storage_preserved": True}, flush=True)

    def test_legacy_fp8_and_wo_refresh_restore_outputs_without_replacing_storage(self):
        import torch
        from torch import nn
        from b12x.gemm import block_fp8_linear, wo_projection
        from vllm.model_executor.kernels.linear.scaled_mm.b12x import B12xFp8BlockScaledMMKernel
        from vllm.models.deepseek_v4.nvidia import b12x as dsv4_b12x
        DeepseekV4Attention = getattr(dsv4_b12x, "DeepseekV4B12xAttention", None)
        if DeepseekV4Attention is None:
            DeepseekV4Attention = dsv4_b12x.DeepseekV4B12xMLAAttention
        from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer
        from coldsnap_recovery_loader import _refresh_initial_native_model

        torch.manual_seed(17)
        with torch.no_grad():
            model = nn.Module()
            layer = nn.Module()
            model.add_module("linear", layer)
            kernel = B12xFp8BlockScaledMMKernel.__new__(B12xFp8BlockScaledMMKernel)
            layer.quant_method = SimpleNamespace(fp8_linear=kernel)
            layer.weight_block_size = [128, 128]
            layer.weight = nn.Parameter(torch.zeros((256, 256), device="cuda").to(torch.float8_e4m3fn), requires_grad=False)
            layer.weight_scale_inv = nn.Parameter(torch.ones((2, 2), device="cuda"), requires_grad=False)
            kernel.process_weights_after_loading(layer)
            packed = getattr(layer, "b12x_packed_weight", None)
            attention = DeepseekV4Attention.__new__(DeepseekV4Attention)
            nn.Module.__init__(attention)
            model.add_module("attention", attention)
            attention.n_local_groups, attention.n_local_heads = 2, 2
            attention.head_dim, attention.o_lora_rank, attention.hidden_size = 256, 128, 256
            attention._use_b12x_wo = True
            attention._b12x_wo_projection_weights = None
            for name in ("wo_a", "wo_b"):
                projection = nn.Module()
                projection.weight = nn.Parameter(torch.zeros((256, 256), device="cuda").to(torch.float8_e4m3fn), requires_grad=False)
                projection.weight_scale_inv = nn.Parameter(torch.ones((2, 2), device="cuda"), requires_grad=False)
                projection.quant_method = SimpleNamespace(fp8_linear=kernel)
                projection.weight_block_size = [128, 128]
                if packed is not None:
                    # The legacy attention constructor marks its fused children.
                    projection.b12x_skip_generic_block_fp8_linear = True
                kernel.process_weights_after_loading(projection)
                attention.add_module(name, projection)
            attention.setup_b12x_wo_projection()
            if packed is None:
                # The new attention owner disables standalone warmup providers
                # after quantization without the legacy skip marker.
                self.assertIsNone(attention.wo_a.b12x_warmup_provider)
                self.assertIsNone(attention.wo_b.b12x_warmup_provider)
            wo = attention._b12x_wo_projection_weights
            mhc = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
            nn.Module.__init__(mhc)
            model.add_module("mhc", mhc)
            mhc.hc_mult, mhc.hidden_size = 4, 16
            mhc.hc_attn_fn = nn.Parameter(torch.zeros((8, 64), device="cuda"), requires_grad=False)
            mhc.hc_attn_fn_broadcast = torch.zeros((8, 16), device="cuda")
            mhc_pointer = mhc.hc_attn_fn_broadcast.data_ptr()
            fields = ("values", "scale_rows", "scale_mma", "values_tiled")
            owners = {"wo_a": wo.wo_a, "wo_b": wo.wo_b}
            if packed is not None:
                owners["linear"] = packed.weight
            linear_pointers = (layer.weight.data_ptr(), layer.weight_scale_inv.data_ptr())
            pointers = {(name, field): tensor.data_ptr() for name, owner in owners.items()
                        for field in fields if (tensor := getattr(owner, field)) is not None}
            for projection in (layer, attention.wo_a, attention.wo_b):
                projection.weight.copy_(torch.randn((256, 256), device="cuda").to(torch.float8_e4m3fn))
                projection.weight_scale_inv.fill_(0.25)
            mhc.hc_attn_fn.copy_(torch.randn_like(mhc.hc_attn_fn))
            reference = block_fp8_linear.pack_weight(layer.weight, layer.weight_scale_inv, block_size=(128, 128))
            wo_reference = wo_projection.pack_weights(
                attention.wo_a.weight, attention.wo_a.weight_scale_inv,
                attention.wo_b.weight, attention.wo_b.weight_scale_inv,
                groups=2, group_width=256, rank=128, hidden=256)
            source = torch.randn((2, 256), device="cuda", dtype=torch.bfloat16)
            wo_source = torch.randn((2, 2, 256), device="cuda", dtype=torch.bfloat16)

            def linear_run(value):
                return block_fp8_linear.run(source=source, packed_weight=value, expected_m=2).clone()

            wo_plan = wo_projection.plan(wo_projection.Caps(
                device=wo_source.device, max_tokens=2, groups=2,
                group_width=256, rank=128, hidden=256))
            spec = wo_plan.scratch_specs()[0]
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

            def wo_run(value):
                binding = wo_projection.bind(wo_plan, scratch=scratch,
                                             source_tgd=wo_source, weights=value, expected_m=2)
                return wo_projection.run(binding=binding).clone()

            expected_wo, stale_wo = wo_run(wo_reference), wo_run(wo)
            if packed is not None:
                expected, stale = linear_run(reference), linear_run(packed)
                self.assertFalse(torch.equal(stale, expected))
            else:
                # The newer image uses registered FP8 tensors directly.
                self.assertIs(layer.b12x_warmup_provider, kernel)
                kernel.config = SimpleNamespace(out_dtype=torch.bfloat16)
                source_fp8 = source.to(torch.float8_e4m3fn)
                source_scale = torch.ones((2, 2), device="cuda")
                def direct_run(weight, scale):
                    return kernel.apply_block_scaled_mm(source_fp8, weight, source_scale, scale).clone()
                expected = direct_run(layer.weight.clone(), layer.weight_scale_inv.clone())
                self.assertTrue(torch.equal(direct_run(layer.weight, layer.weight_scale_inv), expected))
            torch.cuda.synchronize()
            self.assertTrue(bool(torch.isfinite(expected).all()))
            self.assertTrue(bool(torch.isfinite(expected_wo).all()))
            self.assertFalse(torch.equal(stale_wo, expected_wo))
            counts = _refresh_initial_native_model(model)
            self.assertEqual(counts, {"block32_linears": 0, "block128_linears": int(packed is not None),
                                      "wo_projections": 1, "mhc_broadcasts": 1})
            self.assertIs(getattr(layer, "b12x_packed_weight", None), packed)
            self.assertEqual(linear_pointers, (layer.weight.data_ptr(), layer.weight_scale_inv.data_ptr()))
            self.assertIs(attention._b12x_wo_projection_weights, wo)
            self.assertEqual(pointers, {(name, field): getattr(owner, field).data_ptr()
                                       for name, owner in owners.items() for field in fields
                                       if getattr(owner, field) is not None})
            self.assertEqual(mhc.hc_attn_fn_broadcast.data_ptr(), mhc_pointer)
            self.assertTrue(torch.equal(mhc.hc_attn_fn_broadcast, mhc.hc_attn_fn.view(-1, 4, 16).sum(dim=1)))
            actual = linear_run(packed) if packed is not None else direct_run(layer.weight, layer.weight_scale_inv)
            actual_wo = wo_run(wo)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual_wo, expected_wo))
            print({"legacy_block_fp8_output_exact": True, "legacy_wo_output_exact": True,
                   "execution_storage_preserved": True}, flush=True)


    def test_real_layerwise_reload_defers_model_finalizer(self):
        import torch
        from torch import nn
        from unittest.mock import patch
        from vllm.models.deepseek_v4.nvidia import model as architecture
        from vllm.model_executor.model_loader.reload.layerwise import (
            record_metadata_for_reloading, initialize_layerwise_reload,
            finalize_layerwise_reload,
        )
        from coldsnap_recovery_loader import _defer_recovery_model_finalizers

        cls = architecture.DeepseekV4ForCausalLM
        if not callable(getattr(cls, "process_weights_after_loading", None)):
            self.skipTest("legacy model has no separate root finalizer")
        with torch.no_grad(), patch.object(
            architecture, "get_pp_group", return_value=SimpleNamespace(is_first_rank=True),
        ):
            layer = architecture.DeepseekV4DecoderLayer.__new__(architecture.DeepseekV4DecoderLayer)
            nn.Module.__init__(layer)
            layer.hc_mult, layer.hidden_size = 4, 16
            layer.hc_attn_fn = nn.Parameter(torch.arange(512, dtype=torch.float32, device="cuda").view(8, 64), requires_grad=False)
            layer.hc_attn_fn_broadcast = torch.zeros((8, 16), device="cuda")
            source, target = layer.hc_attn_fn, layer.hc_attn_fn_broadcast
            expected = source.view(-1, 4, 16).sum(dim=1)
            inner = architecture.DeepseekV4Model.__new__(architecture.DeepseekV4Model)
            nn.Module.__init__(inner)
            inner.layers = nn.ModuleList([layer])
            inner.start_layer, inner.end_layer = 0, 1
            events = []
            inner.finalize_mega_moe_weights = lambda: events.append("moe")
            inner.process_b12x_weights_after_loading = lambda: events.append("b12x")
            model = cls.__new__(cls)
            nn.Module.__init__(model)
            model.model = inner
            config = SimpleNamespace(dtype=torch.float32)

            record_metadata_for_reloading(model)
            initialize_layerwise_reload(model)
            self.assertTrue(layer.hc_attn_fn.is_meta)
            with self.assertRaisesRegex(NotImplementedError, "meta tensor"):
                model.process_weights_after_loading()
            finalize_layerwise_reload(model, config)
            events.clear()

            record_metadata_for_reloading(model)
            with _defer_recovery_model_finalizers(model):
                initialize_layerwise_reload(model)
                self.assertTrue(layer.hc_attn_fn.is_meta)
                model.process_weights_after_loading()
                self.assertEqual(events, [])
                finalize_layerwise_reload(model, config)
            self.assertEqual(events, ["moe", "b12x"])
            self.assertIs(layer.hc_attn_fn, source)
            self.assertIs(layer.hc_attn_fn_broadcast, target)
            self.assertTrue(torch.equal(target, expected))
            self.assertNotIn("process_weights_after_loading", vars(model))
            print({"real_layerwise_meta_failure_reproduced": True,
                   "deferred_model_finalizer_exact": True,
                   "execution_storage_preserved": True}, flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
"""Prepared-owner classification and reload ordering for the V4.1 lifecycle."""

import contextlib
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
import coldsnap_recovery_loader as loader  # noqa: E402


class Tensor:
    is_cuda = True
    is_meta = False

    def __init__(self, pointer, size=100):
        self.pointer, self.size = pointer, size

    def numel(self):
        return self.size

    def element_size(self):
        return 1

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return True


def fixture():
    fields = {
        name: Tensor(1200 + i * 100) for i, name in enumerate(loader.B12X_PREPARED_WEIGHT_FIELDS)
    }
    prepared = SimpleNamespace(plan=SimpleNamespace(discards_source_parameters=True), **fields)

    class B12xV41Experts:
        def finalize_weights(self):
            if self.prepared is not None:
                return
            self.calls += 1
            for source, field in loader.B12X_SOURCE_STORAGE_FIELDS:
                assert getattr(self, source).data_ptr() == fields[field].data_ptr()
                setattr(self, source, Tensor(0, 0))
            self.prepared = SimpleNamespace(plan=prepared.plan, **fields)
            self.plan, self.local_ids = object(), object()

    B12xV41Experts.__module__ = "vllm.models.deepseek_v4_1.nvidia.b12x_moe"
    layer = B12xV41Experts()
    layer.calls = 0
    layer.prepared = prepared
    layer.plan, layer.local_ids = object(), object()
    layer._parameters = {source: Tensor(0, 0) for source, _ in loader.B12X_SOURCE_STORAGE_FIELDS}
    for name, tensor in layer._parameters.items():
        setattr(layer, name, tensor)
    names = frozenset("expert." + source for source, _ in loader.B12X_SOURCE_STORAGE_FIELDS)
    model = SimpleNamespace(
        named_modules=lambda: iter([("expert", layer)]),
        named_parameters=lambda: iter(
            ("expert." + name, getattr(layer, name)) for name in layer._parameters
        ),
        named_buffers=lambda: iter(()),
    )
    setattr(model, loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR, names)
    info = SimpleNamespace(
        restore_metadata=(
            {
                source: SimpleNamespace(is_meta=True)
                for source, _ in loader.B12X_SOURCE_STORAGE_FIELDS
            },
            {},
        )
    )
    layerwise = ModuleType("vllm.model_executor.model_loader.reload.layerwise")
    layerwise.get_layerwise_info = lambda _: info
    layerwise.materialize_layer = lambda *_: None
    layerwise.restore_layer_on_meta = lambda *_: None
    layerwise.get_layer_size = lambda _: 0
    layerwise._wrap_parameters_weight_loader = lambda _: None

    def copyback(value, _info):
        # vLLM copies back into the captured, released source Parameters.
        assert all(getattr(value, name).numel() == 0 for name in value._parameters)

    layerwise._copy_and_restore_kernel_tensors = copyback
    meta = ModuleType("vllm.model_executor.model_loader.reload.meta")
    meta.SKIP_LOAD_TENSORS, meta.SKIP_TENSORS = set(), {"bias"}
    torch = ModuleType("torch")
    torch.no_grad = contextlib.nullcontext
    modules = {layerwise.__name__: layerwise, meta.__name__: meta, "torch": torch}
    backend = SimpleNamespace(
        memory_provider=SimpleNamespace(
            allocations=lambda tag: [SimpleNamespace(pointer=1000, size=2000, tag=tag)]
        )
    )
    return model, layer, prepared, info, layerwise, torch, modules, backend


class B12xV41Test(unittest.TestCase):
    def test_prepared_weights_are_in_both_layouts(self):
        model, layer, prepared, _, _, _, _, backend = fixture()
        self.assertEqual(loader._model_weight_ranges(model, backend), [(1300, 200), (1700, 200)])
        self.assertEqual(
            [x[:2] for x in loader._native_model_payload_layout(model, backend)],
            [(1200 + i * 100, 100) for i in range(8)]
        )
        self.assertEqual(len(loader._model_semantic_tensors(model)), 12)
        self.assertIs(loader._b12x_prepared_owners(model)[0][3], prepared)
        # A lookalike on an unrelated module is deliberately not classified.
        layer.__class__.__module__ = "unrelated"
        self.assertEqual(loader._b12x_prepared_owners(model), [])

    def test_mxfp8_head_retains_native_weights_after_source_release(self):
        kernel_type = type("B12xMxfp8LinearKernel", (), {})
        kernel_type.__module__ = "vllm.model_executor.kernels.linear.mxfp8.b12x"
        weight = SimpleNamespace(values=Tensor(1200), scale_rows=Tensor(1300), scale_mma=Tensor(1400))
        head = SimpleNamespace(
            quant_method=SimpleNamespace(kernel=kernel_type()),
            b12x_mxfp8_packed_weight=SimpleNamespace(weight=weight),
        )
        model = SimpleNamespace(
            named_modules=lambda: iter([("lm_head", head)]),
            named_parameters=lambda: iter([("lm_head.weight", Tensor(0, 0))]),
            named_buffers=lambda: iter(()),
        )
        backend = SimpleNamespace(memory_provider=SimpleNamespace(
            allocations=lambda tag: [SimpleNamespace(pointer=1000, size=2000, tag=tag)]))
        expected = loader._native_model_payload_layout(model, backend)
        self.assertEqual([entry[:2] for entry in expected], [(1200, 100), (1300, 100), (1400, 100)])
        # The semantic identities survive allocator relocation.
        for field in vars(weight):
            getattr(weight, field).pointer += 500
        self.assertEqual([entry[2] for entry in expected],
                         [entry[2] for entry in loader._native_model_payload_layout(model, backend)])
        del weight.scale_mma
        with self.assertRaisesRegex(RuntimeError, "MXFP8 packed tensor is absent"):
            loader._native_model_payload_layout(model, backend)
        kernel_type.__module__ = "unrelated"
        self.assertEqual(loader._b12x_mxfp8_native_tensors(model), [])

    def test_missing_prepared_field_fails_closed(self):
        model, _, prepared, *_ = fixture()
        del prepared.w1_fp4
        with self.assertRaisesRegex(RuntimeError, "canonical weight field"):
            loader._model_weight_tensors(model)

    def test_normal_reload_packs_before_empty_parameter_copyback(self):
        model, layer, _, info, layerwise, _, modules, _ = fixture()
        plan, ids = layer.plan, layer.local_ids
        original_copyback = layerwise._copy_and_restore_kernel_tensors
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                loader,
                "_alias_meta_tensor_from_storage",
                side_effect=lambda _t, _m, stable, **_: stable,
            ),
        ):
            with loader._recovery_reload_storage(model):
                layerwise.materialize_layer(layer, info)
                layerwise._copy_and_restore_kernel_tensors(layer, info)
                layer.finalize_weights()  # root-model hook must not repack twice
                self.assertEqual(layer.calls, 1)
        self.assertIs(layer.plan, plan)
        self.assertIs(layer.local_ids, ids)
        self.assertNotIn("finalize_weights", vars(layer))
        self.assertIs(layerwise._copy_and_restore_kernel_tensors, original_copyback)

    def test_complete_replay_retains_execution_objects(self):
        model, layer, _, info, _, torch, _, _ = fixture()
        (adapter,) = loader._recovery_storage_adapters(model)
        plan, ids = layer.plan, layer.local_ids
        adapter.begin_reload()
        with patch.object(
            loader,
            "_alias_meta_tensor_from_storage",
            side_effect=lambda _t, _m, stable, **_: stable,
        ):
            adapter.materialize_layer(torch, info)
        adapter.configure_direct_replay(adapter.direct_replay_destination_names)
        adapter.finalize_direct_replay(torch)
        adapter.end_direct_replay()
        layer.finalize_weights()
        adapter.finish_reload()
        self.assertEqual(layer.calls, 1)
        self.assertIs(layer.plan, plan)
        self.assertIs(layer.local_ids, ids)
        self.assertNotIn("finalize_weights", vars(layer))

    def test_missing_copyback_rejects_before_mutation(self):
        model, layer, prepared, _, layerwise, _, modules, _ = fixture()
        del layerwise._copy_and_restore_kernel_tensors
        with (
            patch.dict(sys.modules, modules),
            self.assertRaisesRegex(RuntimeError, "copyback hook"),
        ):
            with loader._recovery_reload_storage(model):
                self.fail("must not enter")
        self.assertIs(layer.prepared, prepared)
        self.assertNotIn("finalize_weights", vars(layer))

    def test_abort_restores_owner_handles_and_hooks(self):
        model, layer, prepared, info, layerwise, _, modules, _ = fixture()
        plan, ids = layer.plan, layer.local_ids
        sources = dict(layer._parameters)
        original_copyback = layerwise._copy_and_restore_kernel_tensors
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                loader,
                "_alias_meta_tensor_from_storage",
                side_effect=lambda _t, _m, stable, **_: stable,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "later failure"):
                with loader._recovery_reload_storage(model):
                    layerwise.materialize_layer(layer, info)
                    layerwise._copy_and_restore_kernel_tensors(layer, info)
                    raise RuntimeError("later failure")
        self.assertIs(layer.prepared, prepared)
        self.assertIs(layer.plan, plan)
        self.assertIs(layer.local_ids, ids)
        self.assertTrue(all(getattr(layer, name) is value for name, value in sources.items()))
        self.assertNotIn("finalize_weights", vars(layer))
        self.assertIs(layerwise._copy_and_restore_kernel_tensors, original_copyback)


if __name__ == "__main__":
    unittest.main()

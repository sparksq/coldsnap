# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Exercise direct checkpoint transfers with real Torch copies, without a GPU.

Run in a vLLM image when Torch/vLLM are not installed on the development host.
"""

import contextlib
import importlib
import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
import coldsnap_recovery_loader as loader  # noqa: E402


class DirectWeightTransferTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from vllm.model_executor import weight_transfer
        except ImportError:
            self.skipTest("requires the vLLM image's Torch and weight_transfer API")
        self.torch, self.transfer = torch, weight_transfer

        class CUDAParameter(torch.nn.Parameter):
            # Use CPU storage to exercise real copy/stride/replay semantics;
            # only the classifier's GPU ownership predicate is substituted.
            @property
            def is_cuda(self):
                return True

        self.parameter = lambda shape: CUDAParameter(torch.zeros(shape, dtype=torch.uint8), False)
        utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
        utils.default_weight_loader = lambda dst, src: weight_transfer.copy_weight(dst, src)
        logger = ModuleType("vllm.logger")
        logger.init_logger = logging.getLogger
        self.enterContext(patch.dict(sys.modules, {utils.__name__: utils, logger.__name__: logger}))
        self.enterContext(patch.object(loader, "_last_metrics", None))

    def model(self, **parameters):
        return SimpleNamespace(
            named_parameters=lambda: iter(parameters.items()),
            named_buffers=lambda: iter(()),
        )

    @contextlib.contextmanager
    def source(self, name, tensor):
        descriptor = loader.SafetensorDescriptor(
            name=name, dtype_name="U8", shape=tuple(tensor.shape),
            file_offset=64, length=tensor.numel() * tensor.element_size(),
        )
        token = loader._ACTIVE_RECOVERY_SOURCE.set(loader.RecoverySourceTensor(
            name=name, path="/checkpoint/model.safetensors", descriptor=descriptor,
            pointer=tensor.data_ptr(),
        ))
        try:
            yield tensor
        finally:
            loader._ACTIVE_RECOVERY_SOURCE.reset(token)

    def observe(self, model, action):
        return loader._load_weights_with_recovery_observer(SimpleNamespace(), model, action)

    def names(self, model):
        return getattr(model, loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR)

    def plan(self, model):
        return getattr(model, loader.CHECKPOINT_COPY_PLAN_ATTR)

    def test_sharded_table_and_scales_leave_residual_only_after_full_coverage(self):
        torch = self.torch
        table, scales, runtime = self.parameter((8, 4)), self.parameter((8, 2)), self.parameter((4,))
        model = self.model(table=table, scales=scales, runtime=runtime)
        source_tensors = {}

        def load():
            for name, destination in (("table", table), ("scales", scales)):
                width = destination.shape[1]
                for shard in range(2):
                    data = torch.arange(6 * width, dtype=torch.uint8).reshape(6, width) + shard * 40
                    source_tensors[f"{name}.shard{shard}"] = data.clone()
                    with self.source(f"{name}.shard{shard}", data):
                        # A TP rank copies only its four-row overlap of each
                        # checkpoint shard, through a separate storage view.
                        self.transfer.copy_weight(destination.data.narrow(0, shard * 4, 4), data[1:5])
                    self.assertTrue(torch.equal(destination[shard * 4:shard * 4 + 4], data[1:5]))
            return "loaded"

        self.assertEqual(self.observe(model, load), "loaded")
        self.assertEqual(self.names(model), {"table", "scales"})
        plan = self.plan(model)
        self.assertEqual(len(plan.copies), 4)
        self.assertEqual(loader._fully_covered_replay_destinations(model, plan), {"table", "scales"})
        self.assertEqual(sum(copy.copy_bytes for copy in plan.copies), 48)
        self.assertEqual([name for name, _ in loader._model_weight_tensors(model)], ["table", "scales"])
        self.assertFalse(hasattr(table, "weight_loader"))
        self.assertIsNone(self.transfer._writer.get())
        # Exercise the production replay planner and typed-copy path after
        # destroying the captured values; restoring metadata alone is not enough.
        expected = {name: tensor.clone() for name, tensor in (("table", table), ("scales", scales))}
        table.zero_()
        scales.zero_()
        replay = loader._recovery_replay_plan(model, ())
        destinations = loader._validate_replay_destinations(replay, model)
        copied = loader._apply_recovery_copies(
            torch, replay.copies,
            {copy.source_key: source_tensors[copy.source_name] for copy in replay.copies},
            destinations,
        )
        self.assertEqual(copied, 48)
        for name, reference in expected.items():
            self.assertTrue(torch.equal(destinations[name], reference), name)

    def test_scalar_scale_raw_copy_is_replayable(self):
        torch = self.torch
        # The actual Engram global scale is FP32, with a scalar checkpoint
        # view reshaped to the registered one-element parameter.
        tensor = self.parameter((1,)).to(dtype=torch.float32)
        tensor = type(self.parameter((1,)))(tensor, False)
        model = self.model(global_scale=tensor)
        data = torch.tensor(0.125, dtype=torch.float32)
        descriptor = loader.SafetensorDescriptor(
            name="global_scale", dtype_name="F32", shape=(), file_offset=64, length=4,
        )

        def load():
            source = loader.RecoverySourceTensor(
                name="global_scale", path="/checkpoint/model.safetensors",
                descriptor=descriptor, pointer=data.data_ptr(),
            )
            token = loader._ACTIVE_RECOVERY_SOURCE.set(source)
            try:
                tensor.data.copy_(data.reshape(1).to(device=tensor.device, dtype=tensor.dtype))
            finally:
                loader._ACTIVE_RECOVERY_SOURCE.reset(token)

        self.observe(model, load)
        self.assertEqual(self.names(model), {"global_scale"})
        self.assertEqual(len(self.plan(model).copies), 1)
        tensor.zero_()
        replay = loader._recovery_replay_plan(model, ())
        destinations = loader._validate_replay_destinations(replay, model)
        self.assertEqual(loader._apply_recovery_copies(
            torch, replay.copies, {replay.copies[0].source_key: data}, destinations,
        ), 4)
        self.assertEqual(tensor.item(), 0.125)

    def test_scalar_transfer_is_not_double_counted(self):
        for callback in (False, True):
            with self.subTest(parameter_callback=callback):
                tensor = self.parameter((1,))
                model = self.model(scale=tensor)
                data = self.torch.tensor([7], dtype=self.torch.uint8)

                def load(data=data, tensor=tensor, callback=callback):
                    with self.source("scale", data):
                        if callback:
                            tensor.weight_loader(tensor, data)
                        else:
                            self.transfer.copy_weight(tensor, data)

                self.observe(model, load)
                self.assertEqual(self.names(model), {"scale"})
                self.assertEqual(len(self.plan(model).copies), 1)

    def test_mutated_or_unproven_scalar_stays_residual(self):
        for mutation in ("add", "clone", "no_source"):
            with self.subTest(mutation=mutation):
                tensor = self.parameter((1,))
                model = self.model(scale=tensor)
                data = self.torch.tensor([7], dtype=self.torch.uint8)

                def load(data=data, tensor=tensor, mutation=mutation):
                    with self.source("scale", data):
                        tensor.data.copy_(data.clone() if mutation == "clone" else data)
                        if mutation == "add":
                            tensor.add_(1)
                    if mutation == "no_source":
                        tensor.copy_(data)

                self.observe(model, load)
                self.assertEqual(self.names(model), set())
                self.assertEqual(self.plan(model).unsupported_destinations, {"scale"})

    def test_partial_or_missing_source_stays_residual(self):
        for missing in (False, True):
            with self.subTest(missing_source=missing):
                tensor = self.parameter((8,))
                model = self.model(table=tensor)
                data = self.torch.arange(4, dtype=self.torch.uint8)

                def load(data=data, tensor=tensor, missing=missing):
                    with self.source("first", data):
                        self.transfer.copy_weight(tensor[:4], data)
                    if missing:
                        self.transfer.copy_weight(tensor[4:], data)

                self.observe(model, load)
                self.assertEqual(self.names(model), set())
                self.assertEqual(loader._fully_covered_replay_destinations(model, self.plan(model)), set())
                self.assertEqual(self.plan(model).unsupported_destinations, {"table"} if missing else set())

    def test_nonaliased_source_is_not_mistaken_for_checkpoint_copy(self):
        tensor = self.parameter((4,))
        model = self.model(table=tensor)
        data = self.torch.arange(4, dtype=self.torch.uint8)

        def load():
            with self.source("table", data):
                self.transfer.copy_weight(tensor, data.clone() + 1)

        self.observe(model, load)
        self.assertEqual(self.names(model), set())
        self.assertEqual(self.plan(model).unsupported_destinations, {"table"})
        self.assertTrue(self.torch.equal(tensor, data + 1))

    def test_parameter_callback_is_not_double_counted(self):
        tensor = self.parameter((4,))
        model = self.model(table=tensor)
        data = self.torch.arange(4, dtype=self.torch.uint8)
        def original(dst, src):
            return self.transfer.copy_weight(dst, src)
        tensor.weight_loader = original

        def load():
            with self.source("table", data):
                tensor.weight_loader(tensor, data)

        self.observe(model, load)
        self.assertEqual(self.names(model), {"table"})
        self.assertEqual(len(self.plan(model).copies), 1)
        self.assertIs(tensor.weight_loader, original)

    def test_existing_transport_and_lifecycle_hooks_are_preserved(self):
        calls = []
        transfer = self.transfer
        torch = self.torch

        class Writer:
            def __call__(self, dst, src):
                calls.append("copy")
                dst.copy_(src)
                return True

            def flush(self):
                calls.append("flush")

            def finish(self):
                calls.append("finish")

            def materialize(self, src):
                calls.append("materialize")
                return src.clone()

        tensor = self.parameter((4,))
        model = self.model(table=tensor)
        data = torch.arange(4, dtype=torch.uint8)
        writer = Writer()

        def load():
            with self.source("table", data):
                transfer.copy_weight(tensor, data)
            transfer.flush_weight_transfers()
            transfer.finish_weight_transfers()
            self.assertTrue(torch.equal(transfer.materialize_weight(data), data))

        with transfer.weight_transfer(writer):
            self.observe(model, load)
            self.assertIs(transfer._writer.get(), writer)
        self.assertEqual(calls, ["copy", "flush", "finish", "materialize"])
        self.assertEqual(self.names(model), {"table"})
        self.assertEqual(len(self.plan(model).copies), 1)

    def test_failure_restores_writer_and_parameter_loaders(self):
        tensor = self.parameter((4,))
        model = self.model(table=tensor)
        def delegate(*_):
            return False

        def load():
            raise RuntimeError("model failed")

        with self.transfer.weight_transfer(delegate):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                self.observe(model, load)
            self.assertIs(self.transfer._writer.get(), delegate)
        self.assertFalse(hasattr(tensor, "weight_loader"))

    def test_older_image_keeps_parameter_observation(self):
        tensor = self.parameter((4,))
        model = self.model(table=tensor)
        data = self.torch.arange(4, dtype=self.torch.uint8)
        original_import = importlib.import_module

        def old_image(name, *args, **kwargs):
            if name == "vllm.model_executor.weight_transfer":
                raise ImportError("old image")
            return original_import(name, *args, **kwargs)

        def load():
            with self.source("table", data):
                tensor.weight_loader(tensor, data)

        with patch.object(loader.importlib, "import_module", old_image):
            self.observe(model, load)
        self.assertEqual(self.names(model), {"table"})
        self.assertEqual(len(self.plan(model).copies), 1)

    def test_ambiguous_aliases_and_cpu_only_state_are_not_promoted(self):
        tensor = self.parameter((4,))
        alias = self.parameter((0,))
        alias.data = tensor.data
        cpu = self.torch.zeros(4, dtype=self.torch.uint8)
        model = self.model(table=tensor, alias=alias, cpu=cpu)
        data = self.torch.arange(4, dtype=self.torch.uint8)

        def load():
            with self.source("table", data):
                self.transfer.copy_weight(tensor.data, data)
                self.transfer.copy_weight(cpu, data)

        self.observe(model, load)
        self.assertEqual(self.names(model), set())
        self.assertEqual(len(self.plan(model).copies), 0)
        self.assertTrue(self.torch.equal(cpu, data))


if __name__ == "__main__":
    unittest.main()

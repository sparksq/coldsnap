# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Regression coverage for disk Engram and fresh-process native hydration."""

import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
import coldsnap_disk_backend as disk
import coldsnap_recovery_loader as recovery
import coldsnap_synthetic_loader as synthetic
from coldsnap_core.memory import HostStage


class Tensor:
    def __init__(self, shape, dtype, device="cpu"):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.is_meta = device == "meta"

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.dtype.itemsize

    def expand(self, shape):
        return Tensor(shape, self.dtype, self.device)

    def data_ptr(self):
        if self.is_meta:
            raise AssertionError("file-backed tensor must not become a recovery copy source")
        return 4096


class FileBackedWeightsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.name = "layers.1.engram.embed.weight"
        self.shape = (384006168, 256)
        self.length = math.prod(self.shape)
        self.path = self.root / "shard.safetensors"
        header = json.dumps({self.name: {
            "dtype": "F8_E4M3", "shape": self.shape,
            "data_offsets": [0, self.length],
        }}).encode()
        self.offset = 8 + len(header)
        with self.path.open("wb") as stream:
            stream.write(len(header).to_bytes(8, "little"))
            stream.write(header)
            stream.truncate(self.offset + self.length)
        self.torch = ModuleType("torch")
        self.torch.float8_e4m3fn = SimpleNamespace(itemsize=1)
        self.torch.float8_e8m0fnu = SimpleNamespace(itemsize=1)
        self.torch.float16 = SimpleNamespace(itemsize=2)
        self.torch.uint8 = SimpleNamespace(itemsize=1)

        def small_or_meta(shape, *, dtype, device="cpu"):
            if device != "meta" and math.prod(shape) * dtype.itemsize > 4096:
                raise AssertionError("checkpoint-sized physical tensor allocation")
            return Tensor(shape, dtype, device)

        self.torch.empty = Mock(side_effect=small_or_meta)
        self.torch.zeros = Mock(side_effect=small_or_meta)
        self.torch.cuda = SimpleNamespace(current_device=lambda: 0)
        self.logger = Mock()
        weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
        weight_utils.should_skip_weight = Mock(return_value=False)
        self.weight_utils = weight_utils

        def file_source_tensor(source):
            result = self.torch.empty(source.shape, dtype=source.dtype, device="meta")
            result.file_source = source
            return result

        weight_utils.file_source_tensor = file_source_tensor
        transfer = ModuleType("vllm.model_executor.weight_transfer")
        transfer.FileTensorSource = SimpleNamespace
        transfer.get_file_tensor_source = lambda tensor: getattr(tensor, "file_source", None)
        logger = ModuleType("vllm.logger")
        logger.init_logger = lambda name: self.logger
        self.safe = Mock()
        self.safe.get_tensor.side_effect = AssertionError("file payload must stay unopened")
        safetensors = ModuleType("safetensors")
        safetensors.safe_open = Mock()
        safetensors.safe_open.return_value.__enter__ = Mock(return_value=self.safe)
        safetensors.safe_open.return_value.__exit__ = Mock(return_value=False)
        self.modules = {"torch": self.torch, "safetensors": safetensors,
                        "vllm.logger": logger,
                        "vllm.model_executor.model_loader.weight_utils": weight_utils,
                        "vllm.model_executor.weight_transfer": transfer}
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, self.modules).start()
        patch.dict(os.environ, {}, clear=True).start()
        self.source = SimpleNamespace(model_or_path=str(self.root), revision="pinned",
                                      prefix="model.", weight_name_prefixes=None,
                                      file_weight_filter=lambda name: name == self.name)

    def iterate(self, backend="direct", **kwargs):
        hydrate = Mock()
        with patch.object(recovery, "_native_hydrator", return_value=(None if backend in {"torch", "mmap"} else hydrate, backend)), \
             patch.object(recovery, "_distributed_context", return_value=(None, 0, 4)), \
             patch.object(recovery, "_broadcast_tensors") as broadcast, \
             patch.object(recovery, "_native_hydrate_owned") as read:
            values = list(recovery.recovery_weights_iterator(
                [str(self.path)], folder=str(self.root), prefix="model.",
                weight_name_prefixes=None, file_weight_filter=self.source.file_weight_filter,
                **kwargs,
            ))
            read.assert_not_called()
            broadcast.assert_not_called()
        return values

    def test_91_gib_file_source_uses_no_staging_or_collective(self):
        for backend in ("direct", "auto", "torch", "mmap"):
            with self.subTest(backend=backend):
                [(name, tensor)] = self.iterate(backend)
                self.assertEqual(name, "model." + self.name)
                self.assertTrue(tensor.is_meta)
                self.assertEqual(tensor.file_source.path, str(self.path))
                self.assertEqual(tensor.file_source.offset, self.offset)
                self.assertEqual(tensor.shape, self.shape)
                self.assertEqual(recovery._last_metrics.local_read_bytes, 0)
                self.assertEqual(recovery._last_metrics.verified_sample_bytes, 0)
                self.assertEqual(recovery._last_metrics.logical_bytes, self.length)
        self.safe.get_tensor.assert_not_called()
        self.assertLess(self.path.stat().st_blocks * 512, 1024**2)

    def test_file_sources_still_obey_local_expert_filter(self):
        self.weight_utils.should_skip_weight.return_value = True
        self.assertEqual(self.iterate(local_expert_ids={4}), [])
        self.weight_utils.should_skip_weight.assert_called_with(self.name, {4})

    def test_file_sources_still_obey_recovery_skip_names(self):
        token = recovery._RECOVERY_SKIP_SOURCE_NAMES.set(frozenset({"model." + self.name}))
        try:
            self.assertEqual(self.iterate(), [])
        finally:
            recovery._RECOVERY_SKIP_SOURCE_NAMES.reset(token)

    def test_file_metadata_does_not_hold_consumer_copy_identity(self):
        sentinel = object()
        token = recovery._ACTIVE_RECOVERY_SOURCE.set(sentinel)
        try:
            descriptor = recovery._descriptors(self.path, indexed_tensor_files=None, weight_name_prefixes=None)[0]
            iterator = recovery._yield_file_source(self.path, descriptor, "model.")
            next(iterator)
            self.assertIsNone(recovery._ACTIVE_RECOVERY_SOURCE.get())
            iterator.close()
            self.assertIs(recovery._ACTIVE_RECOVERY_SOURCE.get(), sentinel)
        finally:
            recovery._ACTIVE_RECOVERY_SOURCE.reset(token)

    def test_metadata_isolated_from_bounded_ordinary_batches(self):
        descriptors = [recovery.SafetensorDescriptor(name, "U8", (size,), i, size)
                       for i, (name, size) in enumerate([
                           ("a", 3), ("b", 4), (self.name, self.length), ("c", 4), ("d", 4)])]
        self.assertEqual(list(recovery._descriptor_batches(
            descriptors, 7, frozenset({self.name}))), [(0, 2), (2, 3), (3, 4), (4, 5)])
        self.assertEqual(list(recovery._descriptor_batches(descriptors[:2], 7)), [(0, 2)])

    def record(self, tensors):
        os.environ.update(COLDSNAP_EXPORT_MODEL_PAYLOAD="1", COLDSNAP_CAPTURE_ID="capture")
        with patch.object(synthetic, "_native_bootstrap_root", return_value=self.root), \
             patch.object(disk, "_worker_id", return_value="worker-0"):
            return synthetic.record_native_bootstrap_source(self.source, "model.", tensors)

    def bootstrap(self):
        with patch.object(synthetic, "_native_bootstrap_root", return_value=self.root), \
             patch.object(disk, "_worker_id", return_value="worker-0"):
            return list(synthetic._native_bootstrap_weights(self.source))

    def file_entry(self):
        return {"name": "model." + self.name, "dtype": "F8_E4M3",
                "shape": list(self.shape), "length": self.length,
                "file_source": {"path": str(self.path), "offset": self.offset}}

    def test_bootstrap_records_without_phase_and_keeps_file_contract(self):
        path = self.record([self.file_entry()])
        self.assertEqual(json.loads(path.read_text())["format"], 2)
        [(name, tensor)] = self.bootstrap()
        self.assertEqual(name, "model." + self.name)
        self.assertTrue(tensor.is_meta)
        self.assertEqual(tensor.file_source.offset, self.offset)
        self.torch.zeros.assert_not_called()
        os.environ["COLDSNAP_PROCESS_TEMPLATE_RESTORED"] = "1"
        self.assertIsNone(self.record([self.file_entry()]))

    def test_old_ordinary_bootstrap_index_remains_supported(self):
        path = self.record([{"name": "model.weight", "dtype": "F16",
                            "shape": [4, 8], "length": 64}])
        index = json.loads(path.read_text())
        index["format"] = 1
        path.write_text(json.dumps(index))
        [(name, tensor)] = self.bootstrap()
        self.assertEqual(name, "model.weight")
        self.assertEqual(tensor.shape, (4, 8))
        self.torch.zeros.assert_called_once_with((), dtype=self.torch.float16, device="cpu")

    def test_missing_file_metadata_fails_before_materialization(self):
        entry = self.file_entry()
        entry.pop("file_source")
        self.record([entry])
        with self.assertRaisesRegex(RuntimeError, "file-backed tensor contract changed"):
            self.bootstrap()
        self.torch.zeros.assert_not_called()
        self.torch.empty.assert_not_called()

    def test_out_of_bounds_file_metadata_rejected(self):
        entry = self.file_entry()
        entry["file_source"]["offset"] += 1
        self.record([entry])
        with self.assertRaisesRegex(RuntimeError, "exceeds checkpoint file"):
            self.bootstrap()
        self.torch.zeros.assert_not_called()

    def test_capture_observer_records_metadata_without_pointer_or_payload(self):
        descriptor = recovery._descriptors(self.path, indexed_tensor_files=None, weight_name_prefixes=None)[0]
        tensor = recovery._file_source_tensor(self.path, descriptor)
        prepared = SimpleNamespace(prefix="model.", files=[str(self.path)])
        with patch.object(recovery, "_capture_source_descriptors", return_value=(
                prepared, {"model." + self.name: (str(self.path), descriptor)})), \
             patch.object(synthetic, "record_native_bootstrap_source") as record:
            self.assertEqual(list(recovery._observed_capture_iterator(
                SimpleNamespace(), self.source, [("model." + self.name, tensor)])),
                [("model." + self.name, tensor)])
        self.assertEqual(record.call_args.args[2], [self.file_entry()])


class InitialNativeHydrationTest(unittest.TestCase):
    def test_registered_loader_routes_native_before_checkpoint_preparation(self):
        source = SimpleNamespace(model_or_path="model")
        expected = [("weight", object())]

        class DefaultLoader:
            counter_before_loading_weights = 0.0

            def load_weights(self, model, model_config):
                return list(self._get_weights_iterator(source))

        module = ModuleType("vllm.model_executor.model_loader.default_loader")
        module.DefaultModelLoader = DefaultLoader
        logger = ModuleType("vllm.logger")
        logger.init_logger = Mock(return_value=Mock())
        registered = {}
        def register(name, cls):
            registered[name] = cls
            return True

        with patch.dict(sys.modules, {module.__name__: module, logger.__name__: logger}), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(recovery, "recovery_weights_enabled", return_value=True), \
             patch.object(recovery, "register_model_loader", side_effect=register), \
             patch.object(recovery, "_install_capture_loader_observer"), \
             patch.object(recovery, "_install_recovery_hybrid_draft_bridge"), \
             patch.object(recovery, "_install_worker_wake_hook"), \
             patch.object(recovery, "prepare_synthetic_weight_source",
                          side_effect=AssertionError("native must not prepare HF shards")), \
             patch.object(synthetic, "_native_bootstrap_weights", return_value=iter(expected)):
            recovery.install_recovery_aware_loader()
            loader = registered[recovery.LOAD_FORMAT]()
            model = SimpleNamespace()
            token = recovery._INITIAL_NATIVE_BOOTSTRAP.set(True)
            try:
                self.assertEqual(loader.load_weights(model, None), expected)
            finally:
                recovery._INITIAL_NATIVE_BOOTSTRAP.reset(token)
            self.assertTrue(model._coldsnap_native_bootstrap_loaded)
            self.assertGreater(loader.counter_before_loading_weights, 0)
            logger.init_logger.assert_any_call("vllm.model_executor.model_loader.default_loader")

    def test_native_semantics_do_not_depend_on_adjacent_tensor_placement(self):
        def tensor(pointer):
            return SimpleNamespace(is_cuda=True, numel=lambda: 16,
                                   element_size=lambda: 1, is_contiguous=lambda: True,
                                   data_ptr=lambda: pointer)

        pointers = [1000, 1016]
        model = SimpleNamespace(named_parameters=lambda: iter([
            ("first", tensor(pointers[0])), ("second", tensor(pointers[1]))]),
            named_buffers=lambda: iter(()), named_modules=lambda: iter(()))
        backend = SimpleNamespace(memory_provider=SimpleNamespace(allocations=lambda tag: [
            SimpleNamespace(pointer=1000, size=1000, tag=tag)]))
        captured = recovery._native_model_payload_layout(model, backend)
        pointers[:] = [1200, 1300]
        live = recovery._native_model_payload_layout(model, backend)
        self.assertEqual(len(captured), 2)
        self.assertEqual([item[1:] for item in captured], [item[1:] for item in live])

    def test_initial_hydration_rebinds_model_only_and_preserves_live_residual(self):
        allocation = SimpleNamespace(pointer=1000, size=8192, tag="weights", is_released=False)
        state = bytearray(b"R" * 16384)
        memory = SimpleNamespace(
            allocations=lambda tag=None: [allocation], synchronize=Mock(),
            allocate_host_stage=lambda size: HostStage(
                owner=(owner := bytearray(size)), view=memoryview(owner), pointer=0),
            copy_to_host=lambda stage, pointer, size: stage.view.__setitem__(
                slice(None, size), bytes([pointer % 251]) * size),
            copy_from_host=lambda pointer, stage, size: state.__setitem__(
                slice(pointer - allocation.pointer, pointer - allocation.pointer + size),
                stage.view[:size]),
        )
        environment = {"COLDSNAP_CAPTURE_ID": "capture", "COLDSNAP_EXPECTED_RANK": "0",
                       "COLDSNAP_EXECUTION_GRAPH": json.dumps({"unit": "unit-0",
                           "by_process_slot": {"0": "worker-0"}, "groups": {}}),
                       "COLDSNAP_MODEL_ID": "model", "COLDSNAP_MODEL_REVISION": "pinned",
                       "COLDSNAP_EXPORT_MODEL_PAYLOAD": "1", "VLLM_HOST_IP": "node-a"}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict(os.environ, environment, clear=True):
            backend = disk.DiskCuMemBackend.__new__(disk.DiskCuMemBackend)
            backend.snapshot_dir = Path(temporary)
            backend.blob_path = backend.snapshot_dir / "weights.blob"
            backend.manifest_path = backend.snapshot_dir / "manifest.json"
            backend.weight_recovery_source = disk.WEIGHT_SOURCE_BLOB
            backend._model_weight_ranges = ((2000, 4097),)
            backend._native_model_payload_semantics = ((2000, 4097, "a" * 64),)
            backend.chunk_bytes = 8192
            backend.verify_mode = "preverified"
            backend.read_direct = backend.write_direct = False
            backend._stage_cache = []
            backend.memory_provider = memory
            backend.hydrator = None
            backend._write_snapshot(memory)
            (backend.snapshot_dir / disk.ACTIVATION_PROVIDER_NAME).write_text("native\n")
            os.environ["COLDSNAP_PROCESS_TEMPLATE_RESTORED"] = "1"
            allocation.pointer, allocation.size = 50000, 16384
            backend._native_model_payload_semantics = ((53000, 4097, "a" * 64),)
            backend._write_state = Mock()
            with patch.object(disk, "_relocate_split_native_manifest",
                              side_effect=AssertionError("must not rebind captured runtime")):
                result = backend.hydrate_initial_native_weights()
            self.assertEqual(result["bytes"], 4097)
            self.assertEqual(state[:3000], b"R" * 3000)
            self.assertEqual(state[3000:7097], bytes([2000 % 251]) * 4097)
            self.assertEqual(state[7097:], b"R" * (16384 - 7097))
            backend._native_model_payload_semantics = ((53000, 4097, "b" * 64),)
            with patch.object(backend, "_restore_exact_extents") as restore:
                with self.assertRaisesRegex(RuntimeError, "semantic model layout differs"):
                    backend.hydrate_initial_native_weights()
                restore.assert_not_called()

    def test_worker_native_hydration_precedes_return_and_resets_context(self):
        events = []
        model = SimpleNamespace(_coldsnap_native_bootstrap_loaded=True)
        backend = SimpleNamespace(hydrate_initial_native_weights=lambda: (
            events.append("hydrate") or {"bytes": 1, "seconds": .1, "backend": "direct"}))

        class Worker:
            model_runner = SimpleNamespace(get_model=lambda: model)

            def load_model(self):
                events.append(("load", recovery._INITIAL_NATIVE_BOOTSTRAP.get()))
                return 42

            def wake_up(self, *args, **kwargs):
                pass

            def _get_sleep_mode_backend(self):
                return backend

        gpu_worker = ModuleType("vllm.v1.worker.gpu_worker")
        gpu_worker.Worker = Worker
        worker_module = ModuleType("vllm.v1.worker")
        worker_module.gpu_worker = gpu_worker
        logger = ModuleType("vllm.logger")
        logger.init_logger = lambda name: Mock()
        modules = {"vllm.v1.worker": worker_module, "vllm.v1.worker.gpu_worker": gpu_worker,
                   "vllm.logger": logger}
        with patch.dict(sys.modules, modules), \
             patch.dict(os.environ, {"COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1"}, clear=True), \
             patch.object(synthetic, "_native_bootstrap_requested", return_value=True), \
             patch.object(recovery, "_bind_native_model_payload_layout") as bind, \
             patch.object(recovery, "_refresh_initial_native_model", side_effect=lambda model: (
                 events.append("refresh") or {"block32_linears": 1, "block128_linears": 0, "wo_projections": 0, "mhc_broadcasts": 1})):
            recovery._install_worker_wake_hook()
            self.assertEqual(Worker().load_model(), 42)
            self.assertEqual(events, [("load", True), "hydrate", "refresh"])
            bind.assert_called_once_with(model, backend)
            self.assertFalse(recovery._INITIAL_NATIVE_BOOTSTRAP.get())
            model._coldsnap_native_bootstrap_loaded = False
            with self.assertRaisesRegex(RuntimeError, "did not use the native bootstrap"):
                Worker().load_model()
            self.assertFalse(recovery._INITIAL_NATIVE_BOOTSTRAP.get())
            events.clear()
            os.environ.pop("COLDSNAP_PROCESS_TEMPLATE_RESTORED")
            self.assertEqual(Worker().load_model(), 42)
            self.assertEqual(events, [("load", False)])


if __name__ == "__main__":
    unittest.main()

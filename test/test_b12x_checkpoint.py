# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "integrations/core"), str(ROOT / "integrations/vllm")]
from coldsnap_core.checkpoint import CheckpointResourceError, ResourceRegistry  # noqa: E402
import coldsnap_b12x_checkpoint as adapter  # noqa: E402


class Tensor:
    shape = (2, 4)
    dtype = "uint8"
    device = "cuda:0"
    def __init__(self):
        self.pointer = id(self)
    def data_ptr(self):
        return self.pointer
    def stride(self):
        return (4, 1)
    def numel(self):
        return 8
    def element_size(self):
        return 1


class Reader:
    def __init__(self):
        self.sources = []


class Native:
    ABI_VERSION = 1
    def __init__(self):
        self.created = []
        self.fail_add = False
        self.run_threads = []
    def ple_reader(self, *args):
        reader = Reader()
        self.created.append((args, weakref.ref(reader)))
        return reader
    def ple_reader_run(self, *args):
        self.run_threads.append(threading.get_native_id())
        return args[-1]
    def ple_reader_add(self, reader, *source):
        if self.fail_add:
            raise RuntimeError("source open failed")
        reader.sources.append(source)


def module_fixture():
    native = Native()
    torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda device: nullcontext()))
    class DiskRowCache:
        def __init__(self, *, device, max_lookups, table_rows, shard_start, shard_end,
                     shard_rows, weight_row_bytes, scale_row_bytes=0, queue_depth=64):
            self.device, self.max_lookups, self.table_rows = device, max_lookups, table_rows
            self.shard_start, self.shard_end, self.shard_rows = shard_start, shard_end, shard_rows
            self.weight_row_bytes, self.scale_row_bytes = weight_row_bytes, scale_row_bytes
            self._backend, self._native, self._gds = "io_uring", native, None
            self._reader = native.ple_reader(shard_rows, table_rows, shard_start, shard_end,
                                             weight_row_bytes, scale_row_bytes, max_lookups, queue_depth)
            self._sources, self._frozen, self._closed = set(), False, False
            self._lock, self._transaction_thread, self._transaction_stream = threading.RLock(), None, None
            self._cache_used, self.syncs, self.closes = True, 0, 0
            self._cache_done = SimpleNamespace(synchronize=self.sync)
            for name in adapter._BUFFER_NAMES:
                setattr(self, name, Tensor())
        def sync(self):
            self.syncs += 1
        def _require_open(self):
            if self._closed:
                raise RuntimeError("closed")
        def add_shard(self, index, path, offset, *, scale=False):
            self._require_open()
            with self._lock:
                if self._frozen or (scale, index) in self._sources:
                    raise RuntimeError("frozen or duplicate")
                native.ple_reader_add(self._reader, index, str(path), offset, scale)
                self._sources.add((scale, index))
        def require_complete(self):
            self._require_open()
            if self._sources != {(False, 0), (False, 1), (True, 0), (True, 1)}:
                raise RuntimeError("incomplete")
        def freeze(self):
            self.require_complete()
            self._frozen = True
        @contextmanager
        def transaction(self):
            self._require_open()
            with self._lock:
                self._transaction_thread = threading.get_ident()
                try:
                    yield self
                finally:
                    self._transaction_thread = None
        def stats(self):
            self._require_open()
            return {"sources": len(self._reader.sources)}
        def close(self):
            self._reader = None
            self._closed = True
            self.closes += 1
    return SimpleNamespace(DiskRowCache=DiskRowCache, torch=torch, native=native)


class B12xCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.registry = ResourceRegistry()
        self.patch = patch.object(adapter, "REGISTRY", self.registry)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"table.bin"
        self.path.write_bytes(bytes(range(64)))
        self.module = module_fixture()
        self.assertTrue(adapter.install_disk_row_cache_adapter(self.module))
        self.assertFalse(adapter.install_disk_row_cache_adapter(self.module))
        self.cache = self.module.DiskRowCache(device="cuda:0", max_lookups=2, table_rows=4,
                                             shard_start=0, shard_end=4, shard_rows=2,
                                             weight_row_bytes=4, scale_row_bytes=1, queue_depth=8)
        self.addCleanup(self.cache.close)
        for scale in (False, True):
            for index in (0, 1):
                self.cache.add_shard(index, self.path, index*8 + int(scale)*32, scale=scale)
        self.cache.freeze()

    def test_suspend_closes_only_native_reader_and_resume_preserves_all_aliases(self):
        cache = self.cache
        before = adapter._buffer_identity(cache)
        old_reader = weakref.ref(cache._reader)
        sources = list(cache._reader.sources)
        report = self.registry.prepare()
        self.assertEqual(report["resources"][0]["source_ranges"], 4)
        self.assertIsNone(old_reader())
        self.assertEqual(cache.closes, 0)
        self.assertEqual(cache.syncs, 1)
        self.assertEqual(adapter._buffer_identity(cache), before)
        with self.assertRaisesRegex(CheckpointResourceError, "prepared"):
            with cache.transaction():
                self.fail("read admitted while prepared")
        with self.assertRaisesRegex(CheckpointResourceError, "prepared"):
            cache.stats()
        self.registry.restore()
        self.assertEqual(adapter._buffer_identity(cache), before)
        self.assertEqual(cache._reader.sources, sources)
        self.assertEqual(self.module.native.created[-1][0][-1], 8)
        self.assertFalse(cache._cache_used)
        with cache.transaction():
            self.assertEqual(cache.stats()["sources"], 4)

    def test_read_submission_thread_retires_at_prepare_and_restarts_after_restore(self):
        native = self.cache._native
        self.assertEqual(native.ple_reader_run(self.cache._reader, 4), 4)
        worker = native._executor._threads.copy().pop()
        self.assertNotEqual(self.module.native.run_threads[-1], threading.get_native_id())
        self.registry.prepare()
        self.assertFalse(worker.is_alive())
        self.assertIsNone(native._executor)
        self.registry.restore()
        self.assertEqual(native.ple_reader_run(self.cache._reader, 2), 2)
        self.assertIsNot(native._executor._threads.copy().pop(), worker)
        self.cache.close()
        self.assertIsNone(native._executor)

    def test_missing_or_changed_source_fails_before_reader_recreation(self):
        self.registry.prepare()
        self.path.write_bytes(b"truncated")
        with self.assertRaisesRegex(CheckpointResourceError, "changed size|invalid.*range"):
            self.registry.restore()
        self.assertEqual(len(self.module.native.created), 1)
        self.assertIsNone(self.cache._reader)

    def test_partial_source_open_failure_releases_new_reader(self):
        self.registry.prepare()
        self.module.native.fail_add = True
        with self.assertRaisesRegex(CheckpointResourceError, "source open failed"):
            self.registry.restore()
        self.assertIsNone(self.module.native.created[-1][1]())
        self.assertIsNone(self.cache._reader)

    def test_changed_tensor_address_rejects_resume(self):
        self.registry.prepare()
        self.cache.weight.pointer += 16
        with self.assertRaisesRegex(CheckpointResourceError, "identity/address changed"):
            self.registry.restore()
        self.assertEqual(len(self.module.native.created), 1)

    def test_same_thread_transaction_cannot_prepare(self):
        with self.cache.transaction():
            with self.assertRaisesRegex(CheckpointResourceError, "active.*transaction"):
                self.registry.prepare()
        self.assertIsNotNone(self.cache._reader)
        self.assertEqual(self.cache.stats()["sources"], 4)

    def test_exact_replay_is_only_permitted_inside_recovery_scope(self):
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            self.cache.add_shard(0, self.path, 0)
        with adapter.replay_immutable_sources():
            self.cache.add_shard(0, self.path, 0)
            with self.assertRaisesRegex(CheckpointResourceError, "changed.*registration"):
                self.cache.add_shard(0, self.path, 1)
        self.assertEqual(self.cache.stats()["sources"], 4)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            self.cache.add_shard(0, self.path, 0)

    def test_gds_and_mutable_sources_are_not_silently_checkpointed(self):
        self.cache._backend = "gds"
        with self.assertRaisesRegex(CheckpointResourceError, "GDS"):
            self.registry.prepare()
        self.cache._backend = "io_uring"
        self.cache._frozen = False
        with self.assertRaisesRegex(CheckpointResourceError, "mutable"):
            self.registry.prepare()
        self.assertIsNotNone(self.cache._reader)

    def test_closed_cache_is_not_retained_by_registry(self):
        self.cache.close()
        self.assertEqual(self.registry.prepare()["resources"], [])
        self.registry.restore()

    def test_prepare_drains_other_thread_before_destroying_reader(self):
        entered, release = threading.Event(), threading.Event()
        result, reader_errors = [], []
        def transaction():
            try:
                with self.cache.transaction():
                    entered.set()
                    self.assertTrue(release.wait(5))
                    self.assertIsNotNone(self.cache._reader)
            except BaseException as error:
                reader_errors.append(error)
        def prepare():
            try:
                result.append(self.registry.prepare())
            except BaseException as error:
                result.append(error)
        reader = threading.Thread(target=transaction)
        reader.start()
        self.assertTrue(entered.wait(5))
        preparer = threading.Thread(target=prepare)
        preparer.start()
        self.assertIsNotNone(self.cache._reader)
        release.set()
        reader.join(5)
        preparer.join(5)
        self.assertFalse(reader.is_alive() or preparer.is_alive())
        self.assertEqual(reader_errors, [])
        self.assertEqual(result[0]["phase"], "prepared")
        self.registry.restore()


if __name__ == "__main__":
    unittest.main()

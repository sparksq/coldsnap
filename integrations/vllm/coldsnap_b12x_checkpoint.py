# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Suspend B12x DiskRowCache readers without replacing their tensor storage.

The native ABI-1 PyCapsule destructor synchronously tears down its io_uring,
registered buffers, and source FDs. MappedHostAllocation.close() also destroys
CUDA aliases, so checkpoint preparation must drop only the native reader.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import functools
import inspect
import os
import stat
from typing import Any, Iterator
import weakref

from coldsnap_core.checkpoint import CheckpointResourceError, REGISTRY

_REPLAY_SOURCES: ContextVar[bool] = ContextVar("coldsnap_replay_disk_sources", default=False)
_ATTRIBUTE = "_coldsnap_checkpoint_resource"
_BUFFER_NAMES = ("ids_host", "weight", "scale", "weight_host", "scale_host")
_ARGUMENTS = (
    "device", "max_lookups", "table_rows", "shard_start", "shard_end",
    "shard_rows", "weight_row_bytes", "scale_row_bytes", "queue_depth",
)


@contextmanager
def replay_immutable_sources() -> Iterator[None]:
    """Permit exact re-registration during the ordinary recovery weight loader."""
    token = _REPLAY_SOURCES.set(True)
    try:
        yield
    finally:
        _REPLAY_SOURCES.reset(token)


@dataclass(frozen=True)
class _Source:
    shard: int
    path: str
    offset: int
    scale: bool
    size: int

    @classmethod
    def record(cls, shard: int, path: Any, offset: int, scale: bool) -> _Source:
        path = os.path.abspath(os.fsdecode(path))
        info = os.stat(path)
        if not stat.S_ISREG(info.st_mode) or offset < 0 or offset >= info.st_size:
            raise CheckpointResourceError(f"invalid B12x disk source range: {path}:{offset}")
        return cls(shard, path, offset, scale, info.st_size)

    def validate(self) -> None:
        # Content identity belongs to the pinned model/source staging contract.
        # Do not hash entire model shards again at a process checkpoint boundary.
        if self.record(self.shard, self.path, self.offset, self.scale) != self:
            raise CheckpointResourceError(f"B12x disk source changed size: {self.path}")


def _buffer_identity(cache: Any) -> dict[str, Any]:
    result = {}
    for name in _BUFFER_NAMES:
        tensor = getattr(cache, name)
        result[name] = None if tensor is None else (
            id(tensor), int(tensor.data_ptr()), tuple(tensor.shape), tuple(tensor.stride()),
            str(tensor.dtype), str(tensor.device), int(tensor.numel()) * int(tensor.element_size()),
        )
    return result


class _ThreadedReader:
    """Keep Linux io-wq ownership off the long-lived engine thread.

    Closing a ring does not retire its submitting task's last io-wq worker on
    older kernels. Run only the synchronous, CPU-native read on a dedicated
    thread and join it at prepare. CUDA stream/event work stays on the caller.
    """

    def __init__(self, native: Any) -> None:
        self._native = native
        self._executor: ThreadPoolExecutor | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)

    def ple_reader_run(self, *args: Any) -> Any:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coldsnap-disk")
        return self._executor.submit(self._native.ple_reader_run, *args).result()

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


class B12xDiskRowResource:
    def __init__(self, cache: Any, torch: Any, queue_depth: int) -> None:
        self._cache = weakref.ref(cache)
        self._torch = torch
        self._queue_depth = queue_depth
        self._sources: dict[tuple[bool, int], _Source] = {}
        self._buffers: dict[str, Any] | None = None
        self._phase = "active"
        self._reader_calls = _ThreadedReader(cache._native) if cache._backend == "io_uring" else None
        if self._reader_calls is not None:
            cache._native = self._reader_calls
        self.name = REGISTRY.register(self, kind="b12x-disk-row-cache")

    def cache(self) -> Any:
        cache = self._cache()
        if cache is None or cache._closed or self._phase == "closed":
            raise CheckpointResourceError("B12x disk row cache was closed before checkpoint resume")
        return cache

    def require_active(self) -> None:
        if self._phase != "active":
            raise CheckpointResourceError(f"B12x disk row cache is {self._phase}; reads require checkpoint resume")

    def _report(self, phase: str) -> dict[str, Any]:
        return {
            "kind": "b12x-disk-row-cache", "backend": "io_uring", "phase": phase,
            "source_ranges": len(self._sources),
            "source_files": len({source.path for source in self._sources.values()}),
            "buffer_addresses_preserved": True,
        }

    def prepare(self) -> dict[str, Any]:
        cache = self.cache()
        if not cache._lock.acquire(timeout=30):
            raise CheckpointResourceError("timed out draining a B12x disk row transaction")
        try:
            self.require_active()
            if cache._backend != "io_uring" or cache._gds is not None:
                raise CheckpointResourceError("B12x GDS readers do not yet expose a checkpoint suspension contract")
            if cache._transaction_thread is not None:
                raise CheckpointResourceError("cannot checkpoint an active B12x disk row transaction")
            if cache._native.ABI_VERSION != 1 or cache._reader is None:
                raise CheckpointResourceError("unsupported B12x disk reader ABI or ownership")
            if not cache._frozen or set(self._sources) != cache._sources:
                raise CheckpointResourceError("B12x disk source registrations are incomplete or mutable")
            cache.require_complete()
            for source in self._sources.values():
                source.validate()
            if cache._cache_used:
                with self._torch.cuda.device(cache.device):
                    cache._cache_done.synchronize()
            self._buffers = _buffer_identity(cache)
            # No aliases to this capsule are retained by the supported reader.
            # Its destructor closes the ring before returning to Python.
            cache._reader = None
            assert self._reader_calls is not None
            self._reader_calls.stop()
            self._phase = "prepared"
            return self._report("prepared")
        finally:
            cache._lock.release()

    def restore(self) -> dict[str, Any]:
        cache = self.cache()
        with cache._lock:
            if self._phase != "prepared" or cache._reader is not None:
                raise CheckpointResourceError("B12x disk row cache has no suspended reader")
            if _buffer_identity(cache) != self._buffers:
                raise CheckpointResourceError("B12x disk staging tensor identity/address changed across checkpoint")
            for source in self._sources.values():
                source.validate()
            reader = cache._native.ple_reader(
                cache.shard_rows, cache.table_rows, cache.shard_start, cache.shard_end,
                cache.weight_row_bytes, cache.scale_row_bytes, cache.max_lookups, self._queue_depth,
            )
            try:
                for source in self._sources.values():
                    cache._native.ple_reader_add(reader, source.shard, source.path, source.offset, source.scale)
            except BaseException:
                # Dropping the new capsule releases partially registered FDs too.
                reader = None
                raise
            cache._reader = reader
            cache._cache_used = False  # There is no in-flight consumer at this boundary.
            self._phase = "active"
            return self._report("restored")

    def close(self) -> None:
        if self._reader_calls is not None:
            self._reader_calls.stop()
        self._phase = "closed"
        REGISTRY.unregister(self.name)


def install_disk_row_cache_adapter(module: Any) -> bool:
    """Patch only the naturally imported shared reader; never import CUDA here."""
    cls = getattr(module, "DiskRowCache", None)
    if cls is None or getattr(cls, "__coldsnap_checkpoint_resources__", False):
        return False
    original_init = cls.__init__
    original_add = cls.add_shard
    original_transaction = cls.transaction
    original_close = cls.close
    original_stats = cls.stats
    original_require = cls._require_open
    signature = inspect.signature(original_init)

    @functools.wraps(original_init)
    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if set(bound.arguments) != {"self", *_ARGUMENTS}:
            raise CheckpointResourceError("unsupported B12x DiskRowCache constructor contract")
        original_init(self, *args, **kwargs)
        try:
            resource = B12xDiskRowResource(self, module.torch, int(bound.arguments["queue_depth"]))
            setattr(self, _ATTRIBUTE, resource)
        except BaseException:
            original_close(self)
            raise

    @functools.wraps(original_require)
    def require_open(self: Any) -> Any:
        resource = getattr(self, _ATTRIBUTE, None)
        if resource is not None:
            resource.require_active()
        return original_require(self)

    @functools.wraps(original_add)
    def add_shard(self: Any, shard_index: int, path: Any, offset: int, *, scale: bool = False) -> Any:
        resource = getattr(self, _ATTRIBUTE)
        with self._lock:
            resource.require_active()
            key = (bool(scale), int(shard_index))
            if _REPLAY_SOURCES.get() and key in resource._sources:
                if resource._sources[key] != _Source.record(int(shard_index), path, int(offset), bool(scale)):
                    raise CheckpointResourceError("recovery changed an immutable B12x disk source registration")
                return None
            result = original_add(self, shard_index, path, offset, scale=scale)
            if key in self._sources:
                resource._sources[key] = _Source.record(int(shard_index), path, int(offset), bool(scale))
            return result

    @functools.wraps(original_transaction)
    @contextmanager
    def transaction(self: Any) -> Iterator[Any]:
        # The upstream open check precedes lock acquisition. Hold the same lock
        # across it so checkpoint prepare cannot close the reader in between.
        with self._lock:
            getattr(self, _ATTRIBUTE).require_active()
            with original_transaction(self) as cache:
                yield cache

    @functools.wraps(original_stats)
    def stats(self: Any) -> Any:
        with self._lock:
            getattr(self, _ATTRIBUTE).require_active()
            return original_stats(self)

    @functools.wraps(original_close)
    def close(self: Any) -> Any:
        with self._lock:
            result = original_close(self)
            resource = getattr(self, _ATTRIBUTE, None)
            if resource is not None:
                resource.close()
            return result

    cls.__init__ = initialize
    cls.add_shard = add_shard
    cls.transaction = transaction
    cls._require_open = require_open
    cls.stats = stats
    cls.close = close
    cls.__coldsnap_checkpoint_resources__ = True
    return True

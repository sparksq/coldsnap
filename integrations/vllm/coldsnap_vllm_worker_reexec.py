# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Resume a pre-CUDA vLLM worker in a fresh target-driver interpreter."""

from __future__ import annotations

import fcntl
import multiprocessing.connection
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _connection(record: object) -> multiprocessing.connection.Connection:
    if not isinstance(record, dict):
        raise RuntimeError("portable worker re-exec connection is invalid")
    descriptor = record.get("fd")
    readable = record.get("readable")
    writable = record.get("writable")
    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 0
        or not isinstance(readable, bool)
        or not isinstance(writable, bool)
        or not (readable or writable)
    ):
        raise RuntimeError("portable worker re-exec connection is invalid")
    return multiprocessing.connection.Connection(
        descriptor, readable=readable, writable=writable
    )


class _PortableWorkerLock:
    """Process- and thread-safe lock backed by a target-local inode."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        self._thread_lock = threading.Lock()

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0:
            timeout = None
        if not block:
            acquired = self._thread_lock.acquire(blocking=False)
        elif timeout is None:
            acquired = self._thread_lock.acquire()
        else:
            acquired = self._thread_lock.acquire(timeout=timeout)
        if not acquired:
            return False
        try:
            if block and timeout is None:
                fcntl.flock(self._descriptor, fcntl.LOCK_EX)
                return True
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(
                        self._descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    return True
                except BlockingIOError:
                    if not block or (deadline is not None and time.monotonic() >= deadline):
                        self._thread_lock.release()
                        return False
                    time.sleep(0.001)
        except BaseException:
            self._thread_lock.release()
            raise

    def release(self) -> None:
        fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        self._thread_lock.release()

    def __enter__(self) -> "_PortableWorkerLock":
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def _shared_lock(record: object) -> Any:
    if not isinstance(record, dict):
        raise RuntimeError("portable worker re-exec lock is invalid")
    if (
        record.get("format") != 1
        or record.get("kind") != "coldsnap-portable-file-lock"
        or not isinstance(record.get("path"), str)
        or not record["path"]
    ):
        raise RuntimeError("portable worker re-exec lock is invalid")
    return _PortableWorkerLock(Path(record["path"]))


def main(arguments: list[str] | None = None) -> int:
    values = sys.argv[1:] if arguments is None else arguments
    if len(values) != 1:
        raise RuntimeError("portable worker re-exec requires one payload path")
    path = Path(values[0])
    with path.open("rb") as source:
        # The parent writes this one-use, 0600 handoff with mkstemp immediately
        # before exec. It carries multiprocessing objects that JSON cannot
        # preserve and is never populated from an artifact or remote input.
        payload = pickle.load(source)  # nosec B301  # noqa: S301
    path.unlink(missing_ok=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != 1
        or payload.get("kind") != "coldsnap-vllm-portable-worker-reexec"
    ):
        raise RuntimeError("portable worker re-exec payload is invalid")
    worker_kwargs = payload.get("worker_kwargs")
    if not isinstance(worker_kwargs, dict):
        raise RuntimeError("portable worker re-exec arguments are invalid")
    worker_kwargs = dict(worker_kwargs)
    worker_kwargs["ready_pipe"] = _connection(payload.get("ready_pipe"))
    worker_kwargs["death_pipe"] = _connection(payload.get("death_pipe"))
    worker_kwargs["shared_worker_lock"] = _shared_lock(
        payload.get("shared_worker_lock")
    )

    from vllm.v1.executor.multiproc_executor import WorkerProc

    WorkerProc.worker_main(**worker_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Process-local lifecycle for external resources that cannot enter a snapshot.

Adapters own their handles and retain any graph-visible storage. A successful
prepare closes external resources; restore reconstructs them after CUDA and
process restoration. Registration uses weak references so this registry never
extends a model's normal lifetime. Prepared resources are retained until resume.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Iterable, Protocol
import weakref


class CheckpointResourceError(RuntimeError):
    """An external resource could not safely cross the checkpoint boundary."""


class CheckpointResource(Protocol):
    def prepare(self) -> dict[str, Any]:
        """Suspend atomically, or leave the resource usable when raising."""
        ...

    def restore(self) -> dict[str, Any]:
        """Reconstruct handles without replacing retained storage."""
        ...


def assert_no_io_uring(pids: Iterable[int] | None = None) -> None:
    """Reject unhandled rings before CRIU, including rings outside adapters.

    This inspects descriptor types and kernel worker names only; it does not read or hash source files.
    The controller also applies it to the entire quiescent process tree.
    """
    remaining = []
    for pid in (os.getpid(),) if pids is None else pids:
        root = Path(f"/proc/{int(pid)}/fd")
        try:
            entries = list(root.iterdir())
        except FileNotFoundError:
            continue  # A child can exit after process-tree enumeration.
        except OSError as error:
            raise CheckpointResourceError(f"cannot inspect checkpoint descriptors for PID {pid}: {error}") from error
        for entry in entries:
            try:
                target = os.readlink(entry)
            except FileNotFoundError:
                continue
            except OSError as error:
                raise CheckpointResourceError(f"cannot inspect checkpoint descriptor {entry}: {error}") from error
            if target in {"anon_inode:[io_uring]", "anon_inode:io_uring"}:
                remaining.append(f"PID {pid} fd {entry.name}")
        try:
            for task in Path(f"/proc/{int(pid)}/task").iterdir():
                try:
                    name = (task / "comm").read_text().strip()
                except FileNotFoundError:
                    continue
                if name.startswith(("iou-wrk-", "iou-sqp-")):
                    remaining.append(f"PID {pid} thread {task.name} ({name})")
        except FileNotFoundError:
            continue
        except OSError as error:
            raise CheckpointResourceError(f"cannot inspect checkpoint threads for PID {pid}: {error}") from error
    if remaining:
        raise CheckpointResourceError(
            "io_uring resources remain after checkpoint prepare; a resource adapter must "
            "suspend every reader and retire its submitting thread before capture: " + ", ".join(remaining)
        )


class ResourceRegistry:
    """Order resource suspension/resume and roll back a failed preparation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._resources: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
        self._prepared: list[tuple[str, CheckpointResource]] = []
        self._phase = "active"
        self._sequence = 0
        self._report: dict[str, Any] | None = None

    def register(self, resource: CheckpointResource, *, kind: str) -> str:
        with self._lock:
            if self._phase != "active":
                raise CheckpointResourceError(f"cannot register a resource while {self._phase}")
            if not kind or not callable(getattr(resource, "prepare", None)) or not callable(getattr(resource, "restore", None)):
                raise CheckpointResourceError("invalid checkpoint resource contract")
            self._sequence += 1
            name = f"{kind}-{self._sequence}"
            self._resources[name] = resource
            return name

    def unregister(self, name: str) -> None:
        with self._lock:
            self._resources.pop(name, None)

    @staticmethod
    def _result(phase: str, resources: list[dict[str, Any]]) -> dict[str, Any]:
        return {"format": 1, "kind": "coldsnap-checkpoint-resources", "phase": phase, "resources": resources}

    def prepare(self) -> dict[str, Any]:
        with self._lock:
            if self._phase == "prepared":
                assert self._report is not None
                return self._report
            if self._phase != "active":
                raise CheckpointResourceError(f"cannot prepare resources while {self._phase}")
            resources = list(self._resources.items())
            self._phase = "preparing"
        prepared = []
        reports = []
        try:
            for name, resource in resources:
                report = resource.prepare()
                prepared.append((name, resource))
                reports.append({"name": name, **report})
            assert_no_io_uring()
        except BaseException as error:
            failures = []
            for name, resource in reversed(prepared):
                try:
                    resource.restore()
                except BaseException as rollback_error:
                    failures.append(f"{name}: {rollback_error}")
            with self._lock:
                self._phase = "failed" if failures else "active"
            message = f"checkpoint resource prepare failed: {error}"
            if failures:
                message += "; rollback failed: " + "; ".join(failures)
            raise CheckpointResourceError(message) from error
        with self._lock:
            self._prepared = prepared
            self._phase = "prepared"
            self._report = self._result("prepared", reports)
            return self._report

    def restore(self) -> dict[str, Any]:
        with self._lock:
            if self._phase != "prepared":
                raise CheckpointResourceError(f"cannot restore resources while {self._phase}")
            self._phase = "restoring"
            resources = list(self._prepared)
        reports = []
        try:
            for name, resource in resources:
                reports.append({"name": name, **resource.restore()})
        except BaseException as error:
            with self._lock:
                self._phase = "failed"
            raise CheckpointResourceError(f"checkpoint resource restore failed: {name}: {error}") from error
        with self._lock:
            self._prepared = []
            self._phase = "active"
            self._report = None
        return self._result("restored", reports)


REGISTRY = ResourceRegistry()

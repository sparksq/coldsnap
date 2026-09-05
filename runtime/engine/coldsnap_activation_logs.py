# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Activation-scoped log handling shared by ColdSnap snapshot drivers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CAPTURE_LOG_FILENAME = "capture.log"
TARGET_LOG_FILENAME = "target.log"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _open_regular_log(path: Path) -> tuple[int, os.stat_result]:
    """Open one capsule log without following a replacement symlink."""

    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as error:
        raise RuntimeError(f"cannot open capsule log {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"capsule log is not a regular file: {path}")
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _copy_open_log(
    source_descriptor: int,
    source_metadata: os.stat_result,
    destination: Path,
) -> dict[str, Any]:
    """Durably copy an already-open log to a new capsule-local path."""

    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"refusing to overwrite capsule log {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    destination_descriptor: int | None = None
    copied = 0
    try:
        destination_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            stat.S_IMODE(source_metadata.st_mode),
        )
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        while chunk := os.read(source_descriptor, 1024 * 1024):
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("short write while preserving capture log")
                view = view[written:]
                copied += written
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = None
        os.replace(temporary, destination)
    except BaseException:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return {
        "format": 1,
        "kind": "coldsnap-capture-log",
        "path": destination.name,
        "bytes": copied,
        "sha256": _sha256(destination),
    }


def archive_capture_log(artifact_root: Path) -> dict[str, Any]:
    """Preserve capture history, then leave the restored-process log empty.

    The inference process inherits an append-mode descriptor for ``target.log``.
    Copy and truncate only after that process has exited, and truncate the file
    in place so the path and inode recorded by CRIU remain stable.
    """

    target = artifact_root / TARGET_LOG_FILENAME
    descriptor, metadata = _open_regular_log(target)
    try:
        record = _copy_open_log(
            descriptor,
            metadata,
            artifact_root / CAPTURE_LOG_FILENAME,
        )
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.fsync(descriptor)
        return record
    finally:
        os.close(descriptor)


def restore_log_boundary(
    *,
    artifact_root: Path,
    activation_namespace: str,
    rank: int,
    capture_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Reset the activation log in place and write its first boundary line.

    Legacy capsules stored capture history directly in ``target.log``. Preserve
    that history under the new name before resetting the container's writable
    overlay. New capsules already carry ``capture.log`` and an empty target.
    """

    target = artifact_root / TARGET_LOG_FILENAME
    capture_log = artifact_root / CAPTURE_LOG_FILENAME
    descriptor, metadata = _open_regular_log(target)
    try:
        try:
            capture_metadata = capture_log.lstat()
        except FileNotFoundError:
            _copy_open_log(descriptor, metadata, capture_log)
        else:
            if capture_log.is_symlink() or not stat.S_ISREG(capture_metadata.st_mode):
                raise RuntimeError(f"capture log is not a regular file: {capture_log}")

        generation = capture_report.get("generation")
        boundary = {
            "format": 1,
            "kind": "coldsnap-restore-log-boundary",
            "activation_namespace": activation_namespace,
            "capture_id": os.environ.get("COLDSNAP_CAPTURE_ID", "").strip(),
            "process_generation": generation if isinstance(generation, str) else "",
            "rank": rank,
            "started_at": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "unit": os.environ.get("COLDSNAP_EXPECTED_UNIT", "").strip(),
        }
        marker = (
            "--- ColdSnap restore boundary "
            + json.dumps(boundary, sort_keys=True, separators=(",", ":"))
            + " ---\n"
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        view = memoryview(marker)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing restore log boundary")
            view = view[written:]
        os.fsync(descriptor)
        return boundary
    finally:
        os.close(descriptor)

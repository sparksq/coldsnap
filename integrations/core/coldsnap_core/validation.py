# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Versioned content validation for immutable ColdSnap payloads.

The default provider is an operational integrity cache.  A matching record
avoids rereading a large payload; stale or missing evidence causes the
existing bytes to be hashed before a caller considers regeneration or
recovery.  Metadata-only changes are diagnostic and never establish content
corruption.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shlex
import socket
import stat
import subprocess
import time
from pathlib import Path
from typing import Any


RECORD_FORMAT = 1
RECORD_KIND = "coldsnap-payload-validation"
RESULT_KIND = "coldsnap-payload-validation-result"
PROVIDER = "sha256-cache-v1"
RECORD_SUFFIX = ".coldsnap-validation.json"
DEFAULT_HASH_CHUNK_BYTES = 16 * 1024**2
VALIDATION_TRANSPORT_ENV = "COLDSNAP_PAYLOAD_VALIDATION_TRANSPORT"
VALIDATION_COMMAND_ENV = "COLDSNAP_PAYLOAD_VALIDATION_COMMAND"
VALIDATION_RPC_SOCKET_ENV = "COLDSNAP_PAYLOAD_VALIDATION_RPC_SOCKET"
VALIDATION_RPC_TOKEN_FILE_ENV = "COLDSNAP_PAYLOAD_VALIDATION_RPC_TOKEN_FILE"
VALIDATION_TIMEOUT_ENV = "COLDSNAP_PAYLOAD_VALIDATION_TIMEOUT_SECONDS"
MAX_RPC_MESSAGE_BYTES = 1 << 20
MAX_RPC_TOKEN_BYTES = 4096


class PayloadValidationError(RuntimeError):
    """A payload could not be accepted by the configured provider."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def normalize_sha256(value: str) -> str:
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or digest.lower() != digest:
        raise PayloadValidationError(
            "invalid_expected_digest",
            "expected SHA-256 must contain 64 lowercase hexadecimal characters",
        )
    try:
        bytes.fromhex(digest)
    except ValueError as error:
        raise PayloadValidationError(
            "invalid_expected_digest", "expected SHA-256 is not hexadecimal"
        ) from error
    return "sha256:" + digest


def validation_record_path(blob: Path) -> Path:
    return blob.with_name(blob.name + RECORD_SUFFIX)


def _open_regular_file(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise PayloadValidationError("unusable_path", f"payload is not a regular file: {path}")
    return descriptor


def content_identity(value: os.stat_result) -> dict[str, int]:
    """Cheap identity fields which can indicate byte replacement.

    ctime is deliberately excluded.  chmod/chown update ctime without changing
    the committed bytes and must not turn a usable native object into recovery.
    """

    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }


def diagnostic_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "ctime_ns": int(value.st_ctime_ns),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "mode": int(stat.S_IMODE(value.st_mode)),
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(value, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _hash_descriptor(descriptor: int, chunk_bytes: int) -> tuple[str, int, dict[str, int], dict[str, int]]:
    if chunk_bytes <= 0:
        raise ValueError("hash chunk size must be positive")
    before_stat = os.fstat(descriptor)
    before = content_identity(before_stat)
    digest = hashlib.sha256()
    hashed = 0
    while chunk := os.read(descriptor, chunk_bytes):
        digest.update(chunk)
        hashed += len(chunk)
    after_stat = os.fstat(descriptor)
    after = content_identity(after_stat)
    if before != after:
        raise PayloadValidationError(
            "changed_during_validation", "payload metadata changed while SHA-256 was calculated"
        )
    return "sha256:" + digest.hexdigest(), hashed, after, diagnostic_identity(after_stat)


def file_sha256(path: Path, *, chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES) -> str:
    descriptor = _open_regular_file(path)
    try:
        digest, _, _, _ = _hash_descriptor(descriptor, chunk_bytes)
    finally:
        os.close(descriptor)
    return digest.removeprefix("sha256:")


def _record_matches(
    record: Any,
    *,
    blob: Path,
    expected_sha256: str,
    expected_bytes: int,
    identity: dict[str, int],
) -> bool:
    return (
        isinstance(record, dict)
        and record.get("format") == RECORD_FORMAT
        and record.get("kind") == RECORD_KIND
        and record.get("provider") == PROVIDER
        and record.get("blob") == blob.name
        and record.get("expected")
        == {"bytes": expected_bytes, "sha256": expected_sha256}
        and record.get("content_identity") == identity
    )


def publish_validation_record(
    blob: Path,
    expected_sha256: str,
    expected_bytes: int,
    *,
    record_path: Path | None = None,
    evidence: str = "full-sha256-this-operation",
) -> dict[str, Any]:
    expected_sha256 = normalize_sha256(expected_sha256)
    descriptor = _open_regular_file(blob)
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = content_identity(value)
    if identity["size"] != expected_bytes:
        raise PayloadValidationError(
            "size_mismatch",
            f"payload size mismatch: expected={expected_bytes} actual={identity['size']}",
        )
    record = {
        "format": RECORD_FORMAT,
        "kind": RECORD_KIND,
        "provider": PROVIDER,
        "blob": blob.name,
        "expected": {"bytes": expected_bytes, "sha256": expected_sha256},
        "content_identity": identity,
        "diagnostic_identity": diagnostic_identity(value),
        "content_evidence": evidence,
        "validated_unix_ns": time.time_ns(),
    }
    _write_json_atomic(record_path or validation_record_path(blob), record)
    return record


def _validate_payload_python(
    blob: Path,
    expected_sha256: str,
    expected_bytes: int,
    *,
    record_path: Path | None = None,
    chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
) -> dict[str, Any]:
    """Accept cached evidence or revalidate the existing payload in place."""

    started = time.perf_counter()
    expected_sha256 = normalize_sha256(expected_sha256)
    marker = record_path or validation_record_path(blob)
    descriptor = _open_regular_file(blob)
    try:
        initial_stat = os.fstat(descriptor)
        identity = content_identity(initial_stat)
        if identity["size"] != expected_bytes:
            raise PayloadValidationError(
                "size_mismatch",
                f"payload size mismatch: expected={expected_bytes} actual={identity['size']}",
            )
        record: Any = None
        marker_mode_ok = False
        try:
            marker_stat = marker.stat(follow_symlinks=False)
            marker_mode_ok = stat.S_ISREG(marker_stat.st_mode) and marker_stat.st_mode & 0o077 == 0
            record = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        if marker_mode_ok and _record_matches(
            record,
            blob=blob,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
            identity=identity,
        ):
            return {
                "format": RECORD_FORMAT,
                "kind": RESULT_KIND,
                "provider": PROVIDER,
                "decision": "accept",
                "reason": "cached_validation",
                "content_evidence": "cached-full-sha256",
                "bytes_hashed": 0,
                "seconds": time.perf_counter() - started,
                "record": str(marker),
                "content_identity": identity,
                "diagnostic_identity": diagnostic_identity(initial_stat),
            }
        digest, hashed, identity, diagnostics = _hash_descriptor(descriptor, chunk_bytes)
    finally:
        os.close(descriptor)
    if digest != expected_sha256:
        raise PayloadValidationError(
            "digest_mismatch",
            f"payload SHA-256 mismatch: expected={expected_sha256} actual={digest}",
        )
    publish_validation_record(
        blob,
        expected_sha256,
        expected_bytes,
        record_path=marker,
        evidence="full-sha256-this-operation",
    )
    return {
        "format": RECORD_FORMAT,
        "kind": RESULT_KIND,
        "provider": PROVIDER,
        "decision": "accept",
        "reason": "revalidated_existing",
        "content_evidence": "full-sha256-this-operation",
        "bytes_hashed": hashed,
        "seconds": time.perf_counter() - started,
        "record": str(marker),
        "content_identity": identity,
        "diagnostic_identity": diagnostics,
    }


def _validation_timeout() -> float:
    raw = os.environ.get(VALIDATION_TIMEOUT_ENV, "1800")
    try:
        value = float(raw)
    except ValueError as error:
        raise PayloadValidationError(
            "invalid_external_configuration",
            f"{VALIDATION_TIMEOUT_ENV} must be a positive number",
        ) from error
    if not 0 < value <= 24 * 60 * 60:
        raise PayloadValidationError(
            "invalid_external_configuration",
            f"{VALIDATION_TIMEOUT_ENV} must be between zero and 86400 seconds",
        )
    return value


def _validation_transport() -> str:
    configured = os.environ.get(VALIDATION_TRANSPORT_ENV, "auto").strip().lower()
    if configured not in {"auto", "python", "cli", "rpc"}:
        raise PayloadValidationError(
            "invalid_external_configuration",
            f"{VALIDATION_TRANSPORT_ENV} must be auto, python, cli, or rpc",
        )
    if configured != "auto":
        return configured
    if os.environ.get(VALIDATION_RPC_SOCKET_ENV):
        return "rpc"
    if os.environ.get(VALIDATION_COMMAND_ENV):
        return "cli"
    return "python"


def _normalize_external_admission(
    admission: Any,
    *,
    blob: Path,
    marker: Path,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    try:
        validation = admission["validation"]
        if (
            admission["format"] != RECORD_FORMAT
            or admission["kind"] != RESULT_KIND
            or admission["decision"] != "accept"
            or admission["path"] != str(blob)
            or admission["bytes"] != expected_bytes
            or admission["sha256"] != expected_sha256
            or validation["record"] != str(marker)
            or validation["provider"] != PROVIDER
        ):
            raise ValueError("admission identity differs from the request")
        content = {
            name: int(validation[name])
            for name in ("device", "inode", "size", "mtime_ns")
        }
        diagnostics = {
            name: int(validation[name])
            for name in ("ctime_ns", "uid", "gid", "mode")
        }
        if content["size"] != expected_bytes:
            raise ValueError("admission content size differs from the request")
        return {
            "format": RECORD_FORMAT,
            "kind": RESULT_KIND,
            "provider": PROVIDER,
            "decision": "accept",
            "reason": str(validation["reason"]),
            "content_evidence": str(validation["content_evidence"]),
            "bytes_hashed": int(validation["bytes_hashed"]),
            "seconds": float(validation["seconds"]),
            "record": str(marker),
            "content_identity": content,
            "diagnostic_identity": diagnostics,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise PayloadValidationError(
            "invalid_external_admission",
            f"external payload validator returned an invalid admission: {error}",
        ) from error


def _validate_payload_cli(
    blob: Path, marker: Path, expected_sha256: str, expected_bytes: int
) -> dict[str, Any]:
    configured = os.environ.get(VALIDATION_COMMAND_ENV, "")
    try:
        command = shlex.split(configured)
    except ValueError as error:
        raise PayloadValidationError(
            "invalid_external_configuration",
            f"{VALIDATION_COMMAND_ENV} is not a valid command prefix",
        ) from error
    if not command:
        raise PayloadValidationError(
            "invalid_external_configuration",
            f"{VALIDATION_COMMAND_ENV} is required for CLI validation",
        )
    executable = Path(command[0])
    if (
        not executable.is_absolute()
        or Path(os.path.normpath(executable)) != executable
        or not executable.is_file()
    ):
        raise PayloadValidationError(
            "invalid_external_configuration",
            "payload validation command must start with a clean absolute executable path",
        )
    command.extend(
        [
            "--path",
            str(blob),
            "--record",
            str(marker),
            "--expected-sha256",
            expected_sha256,
            "--expected-bytes",
            str(expected_bytes),
        ]
    )
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_validation_timeout(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PayloadValidationError(
            "external_validation_failed", f"payload validation CLI failed: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise PayloadValidationError(
            "external_validation_failed", f"payload validation CLI rejected the payload: {detail}"
        )
    try:
        admission = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PayloadValidationError(
            "invalid_external_admission",
            "payload validation CLI returned invalid JSON",
        ) from error
    return _normalize_external_admission(
        admission,
        blob=blob,
        marker=marker,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )


def _validate_payload_rpc(
    blob: Path, marker: Path, expected_sha256: str, expected_bytes: int
) -> dict[str, Any]:
    rpc_path = Path(os.environ.get(VALIDATION_RPC_SOCKET_ENV, ""))
    token_path = Path(os.environ.get(VALIDATION_RPC_TOKEN_FILE_ENV, ""))
    for value, label in ((rpc_path, "RPC socket"), (token_path, "RPC token file")):
        if not value.is_absolute() or Path(os.path.normpath(value)) != value:
            raise PayloadValidationError(
                "invalid_external_configuration",
                f"payload validation {label} must be a clean absolute path",
            )
    try:
        socket_stat = rpc_path.lstat()
        if not stat.S_ISSOCK(socket_stat.st_mode) or socket_stat.st_mode & 0o077:
            raise OSError("RPC socket must be a private Unix socket")
        descriptor = _open_private_rpc_token(token_path)
        try:
            token = os.read(descriptor, MAX_RPC_TOKEN_BYTES + 1).decode("utf-8").strip()
        finally:
            os.close(descriptor)
    except (OSError, UnicodeDecodeError) as error:
        raise PayloadValidationError(
            "invalid_external_configuration",
            f"cannot load payload validation RPC configuration: {error}",
        ) from error
    if len(token) < 16 or any(character in " \t\r\n\x00" for character in token):
        raise PayloadValidationError(
            "invalid_external_configuration", "payload validation RPC token is invalid"
        )
    identifier = secrets.token_hex(16)
    request = {
        "format": 1,
        "id": identifier,
        "token": token,
        "options": {
            "path": str(blob),
            "record": str(marker),
            "expected_sha256": expected_sha256,
            "expected_bytes": expected_bytes,
        },
    }
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(_validation_timeout())
    try:
        connection.connect(str(rpc_path))
        connection.sendall(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        connection.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        received = 0
        while chunk := connection.recv(min(64 * 1024, MAX_RPC_MESSAGE_BYTES + 1 - received)):
            chunks.append(chunk)
            received += len(chunk)
            if received > MAX_RPC_MESSAGE_BYTES:
                raise PayloadValidationError(
                    "invalid_external_admission",
                    "payload validation RPC response exceeds its size limit",
                )
    except (OSError, TimeoutError) as error:
        raise PayloadValidationError(
            "external_validation_failed", f"payload validation RPC failed: {error}"
        ) from error
    finally:
        connection.close()
    try:
        response = json.loads(b"".join(chunks))
        if response["format"] != 1 or response["id"] != identifier:
            raise ValueError("response identity differs from the request")
        if response["ok"] is not True:
            raise PayloadValidationError(
                "external_validation_failed",
                f"payload validation RPC rejected the payload: {response.get('error', 'unknown error')}",
            )
        admission = response["admission"]
    except PayloadValidationError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PayloadValidationError(
            "invalid_external_admission",
            f"payload validation RPC returned an invalid response: {error}",
        ) from error
    return _normalize_external_admission(
        admission,
        blob=blob,
        marker=marker,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )


def _open_private_rpc_token(path: Path) -> int:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        value = os.fstat(descriptor)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_mode & 0o077
            or value.st_size <= 0
            or value.st_size > MAX_RPC_TOKEN_BYTES
        ):
            raise OSError("RPC token file must be a small private regular file")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def validate_payload(
    blob: Path,
    expected_sha256: str,
    expected_bytes: int,
    *,
    record_path: Path | None = None,
    chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
) -> dict[str, Any]:
    """Admit a payload through the configured canonical validation transport."""
    expected_sha256 = normalize_sha256(expected_sha256)
    marker = record_path or validation_record_path(blob)
    transport = _validation_transport()
    # External validation owns one canonical adjacent record. Alternate marker
    # paths and diagnostic chunk sizing remain local-only contracts.
    if (
        transport == "python"
        or marker != validation_record_path(blob)
        or chunk_bytes != DEFAULT_HASH_CHUNK_BYTES
    ):
        return _validate_payload_python(
            blob,
            expected_sha256,
            expected_bytes,
            record_path=marker,
            chunk_bytes=chunk_bytes,
        )
    if transport == "cli":
        return _validate_payload_cli(blob, marker, expected_sha256, expected_bytes)
    return _validate_payload_rpc(blob, marker, expected_sha256, expected_bytes)


def verify_validation_record(
    blob: Path, expected_sha256: str, marker: Path, *, expected_bytes: int | None = None
) -> dict[str, Any]:
    size = blob.stat(follow_symlinks=False).st_size if expected_bytes is None else expected_bytes
    result = validate_payload(blob, expected_sha256, size, record_path=marker)
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        return record | {
            "sha256": normalize_sha256(expected_sha256).removeprefix("sha256:"),
            "validation": result,
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PayloadValidationError("record_unreadable", "payload validation record is unreadable") from error

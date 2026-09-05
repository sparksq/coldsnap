# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Dependency-free client for ColdSnap's ephemeral native coordinator."""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"CSKV"
VERSION = 1
_REQUEST = struct.Struct("!4sBBHIIIII")
_RESPONSE = struct.Struct("!4sBBHI")
_OP_SET = 1
_OP_GET = 2
_OP_DELETE = 3
_OP_HEALTH = 4
_OP_ADD = 5
_OP_APPEND = 6
_STATUS_OK = 0
_STATUS_NOT_FOUND = 1
_STATUS_UNAUTHORIZED = 2
_STATUS_INVALID = 3
_STATUS_TIMEOUT = 4
_STATUS_TOO_LARGE = 5
_MAX_VALUE = 16 * 1024**2


class CoordinatorError(RuntimeError):
    """The coordinator rejected a request or returned an invalid response."""


class CoordinatorNotFound(KeyError, CoordinatorError):
    """A non-blocking get did not find the requested key."""


@dataclass(frozen=True)
class Endpoint:
    address: str
    scope: str
    token: str

    @classmethod
    def parse(cls, value: str) -> Endpoint:
        fields = value.split()
        if len(fields) != 4 or fields[0] != "coldsnap-coord-v1":
            raise ValueError(
                "invalid coordinator endpoint; expected "
                "'coldsnap-coord-v1 address scope token'"
            )
        endpoint = cls(address=fields[1], scope=fields[2], token=fields[3])
        if any(not field or any(character.isspace() for character in field) for field in fields[1:]):
            raise ValueError("coordinator endpoint fields must not contain whitespace")
        return endpoint

    @classmethod
    def read(cls, path: Path) -> Endpoint:
        if path.stat().st_size > 16 * 1024:
            raise ValueError(f"coordinator endpoint is too large: {path}")
        return cls.parse(path.read_text(encoding="utf-8"))


class Client:
    def __init__(self, endpoint: Endpoint, *, connect_timeout: float = 5.0) -> None:
        self.endpoint = endpoint
        self.connect_timeout = connect_timeout

    @classmethod
    def from_path(cls, path: Path, *, connect_timeout: float = 5.0) -> Client:
        return cls(Endpoint.read(path), connect_timeout=connect_timeout)

    def health(self) -> None:
        self._request(_OP_HEALTH)

    def set(self, key: str, value: str | bytes) -> None:
        encoded = value.encode() if isinstance(value, str) else value
        self._request(_OP_SET, key, encoded)

    def get(self, key: str, *, wait: float = 0.0) -> bytes:
        return self._request(_OP_GET, key, timeout=wait)

    def wait(self, key: str, timeout: float) -> str:
        return self.get(key, wait=timeout).decode()

    def delete(self, key: str) -> None:
        self._request(_OP_DELETE, key)

    def add(self, key: str, amount: int) -> int:
        return int(self._request(_OP_ADD, key, str(amount).encode()).decode())

    def append(self, key: str, value: bytes) -> None:
        self._request(_OP_APPEND, key, bytes(value))

    def _request(
        self,
        operation: int,
        key: str = "",
        value: bytes = b"",
        *,
        timeout: float = 0.0,
    ) -> bytes:
        request = _encode_request(self.endpoint, operation, key, value, timeout)
        network_timeout = self.connect_timeout + timeout if timeout else self.connect_timeout
        host, port = _split_address(self.endpoint.address)
        with socket.create_connection((host, port), timeout=self.connect_timeout) as connection:
            connection.settimeout(network_timeout)
            connection.sendall(request)
            response = _recv_exact(connection, _RESPONSE.size)
            magic, version, status, _flags, value_length = _RESPONSE.unpack(response)
            if magic != MAGIC or version != VERSION:
                raise CoordinatorError("unsupported coordinator response")
            if value_length > _MAX_VALUE:
                raise CoordinatorError("coordinator response exceeds configured limit")
            body = _recv_exact(connection, value_length)
        if status == _STATUS_OK:
            return body
        detail = body.decode(errors="replace") or f"status {status}"
        if status == _STATUS_NOT_FOUND:
            raise CoordinatorNotFound(key)
        if status == _STATUS_TIMEOUT:
            raise TimeoutError(f"timed out waiting for coordinator key {key!r}")
        if status == _STATUS_UNAUTHORIZED:
            raise PermissionError("coordinator authentication failed")
        labels = {
            _STATUS_INVALID: "invalid request",
            _STATUS_TOO_LARGE: "request too large",
        }
        raise CoordinatorError(f"coordinator {labels.get(status, 'request failed')}: {detail}")


def _encode_request(
    endpoint: Endpoint,
    operation: int,
    key: str = "",
    value: bytes = b"",
    timeout: float = 0.0,
) -> bytes:
    if timeout < 0 or timeout > 4_294_967.295:
        raise ValueError("coordinator timeout is out of range")
    token = endpoint.token.encode()
    scoped_key = f"{endpoint.scope}/{key}".encode() if key else b""
    total = len(token) + len(scoped_key) + len(value)
    if len(scoped_key) > 64 * 1024 or len(value) > _MAX_VALUE or total > 0xFFFFFFFF:
        raise ValueError("coordinator request exceeds configured limit")
    header = _REQUEST.pack(
        MAGIC,
        VERSION,
        operation,
        0,
        round(timeout * 1000),
        len(token),
        len(scoped_key),
        len(value),
        total,
    )
    return header + token + scoped_key + value


def _split_address(value: str) -> tuple[str, int]:
    if value.startswith("["):
        host, separator, port_text = value[1:].rpartition("]:")
    else:
        host, separator, port_text = value.rpartition(":")
    if not separator or not host or not port_text.isdecimal():
        raise ValueError(f"invalid coordinator address: {value!r}")
    port = int(port_text)
    if not 0 < port < 65536:
        raise ValueError(f"invalid coordinator port: {port}")
    return host, port


def _recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("coordinator closed an incomplete response")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

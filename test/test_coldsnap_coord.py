# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import socket
import struct
import sys
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "shared"))

import coldsnap_coord  # noqa: E402


class CoordinatorClientTest(unittest.TestCase):
    def test_endpoint_and_ipv6_address(self) -> None:
        endpoint = coldsnap_coord.Endpoint.parse(
            "coldsnap-coord-v1 [::1]:17877 restore-a 0123456789abcdef"
        )
        self.assertEqual(endpoint.scope, "restore-a")
        self.assertEqual(coldsnap_coord._split_address(endpoint.address), ("::1", 17877))

    def test_set_request_and_response(self) -> None:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        address = f"127.0.0.1:{listener.getsockname()[1]}"
        observed: list[tuple[int, bytes, bytes, bytes]] = []

        def serve() -> None:
            connection, _ = listener.accept()
            with connection:
                header = _recv(connection, coldsnap_coord._REQUEST.size)
                (
                    magic,
                    version,
                    operation,
                    _flags,
                    _timeout,
                    token_length,
                    key_length,
                    value_length,
                    total,
                ) = coldsnap_coord._REQUEST.unpack(header)
                payload = _recv(connection, total)
                self.assertEqual((magic, version), (b"CSKV", 1))
                observed.append(
                    (
                        operation,
                        payload[:token_length],
                        payload[token_length : token_length + key_length],
                        payload[-value_length:],
                    )
                )
                connection.sendall(struct.pack("!4sBBHI", b"CSKV", 1, 0, 0, 0))
            listener.close()

        thread = threading.Thread(target=serve)
        thread.start()
        endpoint = coldsnap_coord.Endpoint(address, "scope-1", "0123456789abcdef")
        coldsnap_coord.Client(endpoint).set("ready/0", "yes")
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(
            observed,
            [(1, b"0123456789abcdef", b"scope-1/ready/0", b"yes")],
        )


def _recv(connection: socket.socket, length: int) -> bytes:
    value = bytearray()
    while len(value) < length:
        value.extend(connection.recv(length - len(value)))
    return bytes(value)


if __name__ == "__main__":
    unittest.main()

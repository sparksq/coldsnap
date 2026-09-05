# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import http.server
import socket
import tempfile
import threading
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
from coldsnap_supervisor import GateTCPServer  # noqa: E402


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b"healthy\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def request(port: int) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
        connection.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        chunks = []
        while chunk := connection.recv(4096):
            chunks.append(chunk)
        return b"".join(chunks)


class GateSupervisorTest(unittest.TestCase):
    def test_gate_forwards_only_after_required_files_exist(self) -> None:
        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        with tempfile.TemporaryDirectory() as directory:
            accepted = Path(directory) / "accepted"
            restored = Path(directory) / "restored"
            gate = GateTCPServer(
                ("127.0.0.1", 0),
                ("127.0.0.1", upstream.server_address[1]),
                (accepted, restored),
            )
            gate_thread = threading.Thread(target=gate.serve_forever, daemon=True)
            gate_thread.start()
            try:
                blocked = request(gate.server_address[1])
                self.assertIn(b"503 Service Unavailable", blocked)
                accepted.touch()
                self.assertIn(b"503 Service Unavailable", request(gate.server_address[1]))
                restored.touch()
                forwarded = request(gate.server_address[1])
                self.assertIn(b"200 OK", forwarded)
                self.assertTrue(forwarded.endswith(b"healthy\n"))
            finally:
                gate.shutdown()
                gate.server_close()
                gate_thread.join(timeout=2)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

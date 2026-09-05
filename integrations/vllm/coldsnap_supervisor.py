# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Supervise vLLM and expose it only after controller acceptance."""

from __future__ import annotations

import argparse
import os
import select
import signal
import socket
import socketserver
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


UNAVAILABLE_BODY = b'{"error":"model restore has not been accepted"}\n'


class GateTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        upstream: tuple[str, int],
        required_files: tuple[Path, ...],
    ) -> None:
        self.upstream = upstream
        self.required_files = required_files
        super().__init__(address, GateRequestHandler)

    def accepted(self) -> bool:
        return all(path.is_file() for path in self.required_files)


class GateRequestHandler(socketserver.BaseRequestHandler):
    server: GateTCPServer

    def handle(self) -> None:
        if not self.server.accepted():
            response = (
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(UNAVAILABLE_BODY)}\r\n".encode()
                + b"Retry-After: 1\r\nConnection: close\r\n\r\n"
                + UNAVAILABLE_BODY
            )
            self.request.sendall(response)
            return

        try:
            upstream = socket.create_connection(self.server.upstream, timeout=10)
        except OSError:
            response = (
                b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n"
                b"Connection: close\r\n\r\n"
            )
            self.request.sendall(response)
            return

        with upstream:
            sockets = (self.request, upstream)
            while True:
                readable, _, exceptional = select.select(sockets, (), sockets)
                if exceptional:
                    return
                for source in readable:
                    try:
                        data = source.recv(1024 * 1024)
                    except BlockingIOError:
                        continue
                    if not data:
                        return
                    target = upstream if source is self.request else self.request
                    target.sendall(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # This proxy is the container's intended ingress; GateTCPServer blocks until acceptance.
    parser.add_argument("--listen-host", default="0.0.0.0")  # nosec B104
    parser.add_argument("--listen-port", type=int)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int)
    parser.add_argument("--require-file", action="append", type=Path, default=[])
    parser.add_argument("--cache-source", type=Path)
    parser.add_argument("--cache-destination", type=Path)
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _seed_cache(source: Path | None, destination: Path | None) -> None:
    if source is None and destination is None:
        return
    if source is None or destination is None:
        raise RuntimeError("cache source and destination must be supplied together")
    from coldsnap_cache_bootstrap import seed_cache

    seconds = seed_cache(source, destination)
    print(f"COLDSNAP derived startup cache copy took {seconds:.6f} seconds", flush=True)


def run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise RuntimeError("a supervised command is required")
    _seed_cache(args.cache_source, args.cache_destination)

    server: GateTCPServer | None = None
    proxy_thread: threading.Thread | None = None
    if not args.no_proxy:
        if args.listen_port is None or args.upstream_port is None:
            raise RuntimeError("proxy listen and upstream ports are required")
        server = GateTCPServer(
            (args.listen_host, args.listen_port),
            (args.upstream_host, args.upstream_port),
            tuple(args.require_file),
        )
        proxy_thread = threading.Thread(target=server.serve_forever, daemon=True)
        proxy_thread.start()

    process = subprocess.Popen(command, start_new_session=True)

    def forward(signum: int, _frame: Any) -> None:
        if process.poll() is None:
            os.killpg(process.pid, signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        return process.wait()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if proxy_thread is not None:
            proxy_thread.join(timeout=5)


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

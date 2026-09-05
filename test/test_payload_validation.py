# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import hashlib
import json
import os
import socket
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))

from coldsnap_core import validation  # noqa: E402


def _admission(path: Path, digest: str, size: int, *, identifier: str = "") -> dict:
    value = path.stat(follow_symlinks=False)
    return {
        "format": 1,
        "kind": "coldsnap-payload-validation-result",
        "decision": "accept",
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "validation": {
            "record": str(validation.validation_record_path(path)),
            "provider": "sha256-cache-v1",
            "content_evidence": "full-sha256-this-operation",
            "reason": "revalidated_existing",
            "device": value.st_dev,
            "inode": value.st_ino,
            "size": value.st_size,
            "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns,
            "uid": value.st_uid,
            "gid": value.st_gid,
            "mode": stat.S_IMODE(value.st_mode),
            "bytes_hashed": size,
            "seconds": 0.01,
        },
        "id": identifier,
    }


class PayloadValidationTransportTest(unittest.TestCase):
    def test_python_transport_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.pack"
            path.write_bytes(b"payload")
            digest = "sha256:" + hashlib.sha256(b"payload").hexdigest()
            with patch.dict(
                os.environ,
                {validation.VALIDATION_TRANSPORT_ENV: "python"},
                clear=True,
            ):
                result = validation.validate_payload(path, digest, 7)

        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["bytes_hashed"], 7)

    def test_cli_transport_normalizes_go_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "payload.pack"
            executable = root / "coldsnap"
            path.write_bytes(b"payload")
            executable.write_bytes(b"executable")
            executable.chmod(0o700)
            digest = "sha256:" + hashlib.sha256(b"payload").hexdigest()
            completed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(_admission(path, digest, 7)),
                stderr="",
            )
            environment = {
                validation.VALIDATION_TRANSPORT_ENV: "cli",
                validation.VALIDATION_COMMAND_ENV: f"{executable} payload verify",
            }
            with patch.dict(os.environ, environment, clear=True), patch.object(
                validation.subprocess, "run", return_value=completed
            ) as run:
                result = validation.validate_payload(path, digest, 7)

        self.assertEqual(result["provider"], "sha256-cache-v1")
        self.assertEqual(result["content_identity"]["size"], 7)
        self.assertIn("payload", run.call_args.args[0])
        self.assertIn("verify", run.call_args.args[0])

    def test_rpc_transport_normalizes_go_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "payload.pack"
            path.write_bytes(b"payload")
            digest = "sha256:" + hashlib.sha256(b"payload").hexdigest()
            token_path = root / "token"
            token_path.write_text("0123456789abcdef\n", encoding="utf-8")
            token_path.chmod(0o600)
            socket_path = root / "validator.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(socket_path))
            except PermissionError:
                listener.close()
                self.skipTest("sandbox does not permit creating Unix sockets")
            socket_path.chmod(0o600)
            listener.listen(1)
            observed: list[dict] = []

            def serve() -> None:
                connection, _ = listener.accept()
                with connection:
                    request = json.loads(connection.makefile("rb").read())
                    observed.append(request)
                    response = {
                        "format": 1,
                        "id": request["id"],
                        "ok": True,
                        "admission": _admission(path, digest, 7),
                    }
                    connection.sendall(json.dumps(response).encode("utf-8") + b"\n")

            thread = threading.Thread(target=serve)
            thread.start()
            environment = {
                validation.VALIDATION_TRANSPORT_ENV: "rpc",
                validation.VALIDATION_RPC_SOCKET_ENV: str(socket_path),
                validation.VALIDATION_RPC_TOKEN_FILE_ENV: str(token_path),
            }
            try:
                with patch.dict(os.environ, environment, clear=True):
                    result = validation.validate_payload(path, digest, 7)
            finally:
                thread.join(timeout=5)
                listener.close()

        self.assertFalse(thread.is_alive())
        self.assertEqual(observed[0]["token"], "0123456789abcdef")
        self.assertEqual(result["bytes_hashed"], 7)

    def test_rpc_token_must_be_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "token"
            token_path.write_text("0123456789abcdef\n", encoding="utf-8")
            token_path.chmod(0o644)
            with self.assertRaisesRegex(OSError, "private"):
                validation._open_private_rpc_token(token_path)


if __name__ == "__main__":
    unittest.main()

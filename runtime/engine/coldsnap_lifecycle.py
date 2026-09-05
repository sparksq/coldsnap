# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Release-matched lifecycle helper used inside restored engine containers."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


CAPSULE_ROOT = Path("/opt/coldsnap/capsule")
HIBERNATE_ROOT = CAPSULE_ROOT / "hibernate-states"
HYDRATION_ROOT = CAPSULE_ROOT / "hydration"
RUNTIME_ROOT = Path("/run/coldsnap")
MAXIMUM_JSON_BYTES = 1024 * 1024
MAXIMUM_EVIDENCE_BYTES = 4 * 1024 * 1024


class BackendClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base = f"http://127.0.0.1:{args.port}"
        self.deadline = time.monotonic() + args.timeout

    def request(self, method: str, path: str, payload: Any = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        timeout = max(1.0, min(600.0, self.deadline - time.monotonic()))
        # BackendClient constructs a fixed loopback HTTP origin; callers supply
        # only the path below that origin, never an arbitrary URL or scheme.
        with urllib.request.urlopen(  # nosec B310
            request, timeout=timeout
        ) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    def sleeping(self) -> bool:
        if self.args.engine == "vllm":
            value = self.request("GET", "/is_sleeping")
            if not isinstance(value, dict) or not isinstance(value.get("is_sleeping"), bool):
                raise RuntimeError("invalid /is_sleeping response")
            return bool(value["is_sleeping"])
        states = []
        for path in HIBERNATE_ROOT.glob("*.json"):
            states.append(json.loads(path.read_text(encoding="utf-8")).get("state"))
        if (
            not states
            or any(value not in {"running", "sleeping"} for value in states)
            or len(set(states)) != 1
        ):
            raise RuntimeError("invalid SGLang lifecycle evidence")
        return states[0] == "sleeping"

    def wait_state(self, expected: bool) -> None:
        while time.monotonic() < self.deadline:
            if self.sleeping() == expected:
                return
            time.sleep(0.1)
        raise TimeoutError("backend sleep state did not converge")

    def health(self) -> None:
        while time.monotonic() < self.deadline:
            try:
                self.request("GET", self.args.health_path)
                return
            except Exception:  # noqa: BLE001 - readiness polling boundary
                time.sleep(0.25)
        raise TimeoutError("backend health did not become ready")

    def infer(self) -> bool:
        value = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": self.args.model,
                "messages": [{"role": "user", "content": self.args.prompt}],
                "max_tokens": 64,
                "temperature": 0,
            },
        )
        actual = str(value["choices"][0]["message"]["content"]).strip()
        if actual != self.args.expected:
            raise RuntimeError("validation response mismatch")
        return True


def control(args: argparse.Namespace) -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = os.open(RUNTIME_ROOT / "lifecycle.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = BackendClient(args)
        started = time.monotonic()
        before = client.sleeping()
        accepted = False
        if args.action == "sleep":
            if not before:
                client.health()
                accepted = client.infer()
                if args.engine == "vllm":
                    client.request("POST", "/sleep?level=1&mode=wait")
                else:
                    client.request(
                        "POST",
                        "/release_memory_occupation",
                        {"tags": ["kv_cache", "weights", "cuda_graph"]},
                    )
                client.wait_state(True)
        elif args.action == "wake":
            if before:
                if args.engine == "vllm":
                    client.request("POST", "/wake_up")
                else:
                    client.request(
                        "POST",
                        "/resume_memory_occupation",
                        {"tags": ["kv_cache", "weights", "cuda_graph"]},
                    )
                    providers = {
                        path.read_text(encoding="utf-8").strip()
                        for path in HYDRATION_ROOT.glob("*/activation-provider")
                    }
                    if len(providers) != 1 or not providers <= {"native", "recovery"}:
                        raise RuntimeError("invalid SGLang activation provider")
                    if "recovery" in providers:
                        client.request(
                            "POST",
                            "/update_weights_from_disk",
                            {
                                "model_path": args.model,
                                "load_format": args.load_format,
                                "recapture_cuda_graph": False,
                            },
                        )
                client.wait_state(False)
            client.health()
            accepted = client.infer()
        result = {
            "action": args.action,
            "is_sleeping": client.sleeping(),
            "accepted": accepted,
            "seconds": time.monotonic() - started,
        }
        print(json.dumps(result, sort_keys=True))
    finally:
        os.close(lock)


def _expected_workers(value: str) -> set[str]:
    decoded = json.loads(value)
    if (
        not isinstance(decoded, list)
        or not decoded
        or any(not isinstance(worker, str) or not worker for worker in decoded)
    ):
        raise RuntimeError("lifecycle worker inventory is invalid")
    result = set(decoded)
    if len(result) != len(decoded):
        raise RuntimeError("lifecycle worker inventory contains duplicates")
    return result


def evidence(args: argparse.Namespace) -> None:
    expected = _expected_workers(args.workers)
    result = []
    for path in sorted(HIBERNATE_ROOT.glob("*.json")):
        information = path.lstat()
        if (
            not stat.S_ISREG(information.st_mode)
            or information.st_size <= 0
            or information.st_size > MAXIMUM_EVIDENCE_BYTES
        ):
            raise RuntimeError("invalid hibernation evidence file")
        value = json.loads(path.read_text(encoding="utf-8"))
        worker = value.get("worker_id")
        if worker not in expected or value.get("capture_id") != args.capture:
            raise RuntimeError("hibernation evidence identity mismatch")
        result.append(
            {
                "worker_id": worker,
                "state": value.get("state"),
                "capture_id": value.get("capture_id"),
                "pid": value.get("pid", 0),
                "generation": value.get("generation", ""),
            }
        )
    if {item["worker_id"] for item in result} != expected:
        raise RuntimeError("hibernation evidence worker set mismatch")
    print(json.dumps(result, sort_keys=True))


def synchronize_evidence(args: argparse.Namespace) -> None:
    expected = _expected_workers(args.workers)
    seen: set[str] = set()
    for path in sorted(HIBERNATE_ROOT.glob("*.json")):
        information = path.lstat()
        if (
            not stat.S_ISREG(information.st_mode)
            or information.st_size <= 0
            or information.st_size > MAXIMUM_EVIDENCE_BYTES
        ):
            raise RuntimeError("invalid hibernation evidence file")
        value = json.loads(path.read_text(encoding="utf-8"))
        worker = value.get("worker_id")
        if worker not in expected or worker in seen or path.stem != worker:
            raise RuntimeError("hibernation evidence worker identity mismatch")
        previous_capture = value.get("capture_id")
        if previous_capture is not None and not isinstance(previous_capture, str):
            raise RuntimeError("invalid hibernation evidence capture identity")
        seen.add(worker)
        value.update(
            {
                "state": args.state,
                "capture_id": args.capture,
                "updated_unix": time.time(),
                "operation": "restore",
            }
        )
        data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if seen != expected:
        raise RuntimeError("hibernation evidence worker set mismatch")


def write_marker(_args: argparse.Namespace) -> None:
    data = sys.stdin.buffer.read(MAXIMUM_JSON_BYTES + 1)
    if not data or len(data) > MAXIMUM_JSON_BYTES:
        raise RuntimeError("invalid lifecycle marker input")
    # Decode before publication so the marker is always a JSON object.
    value = json.loads(data)
    if not isinstance(value, dict):
        raise RuntimeError("lifecycle marker must be a JSON object")
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = RUNTIME_ROOT / f".lifecycle.{os.getpid()}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, RUNTIME_ROOT / "lifecycle.json")
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="coldsnap-lifecycle")
    commands = result.add_subparsers(dest="command", required=True)

    control_parser = commands.add_parser("control")
    control_parser.add_argument("engine", choices=("vllm", "sglang"))
    control_parser.add_argument("action", choices=("sleep", "wake", "status"))
    control_parser.add_argument("port", type=int)
    control_parser.add_argument("model")
    control_parser.add_argument("health_path")
    control_parser.add_argument("prompt")
    control_parser.add_argument("expected")
    control_parser.add_argument("load_format")
    control_parser.add_argument("timeout", type=float)
    control_parser.set_defaults(handler=control)

    evidence_parser = commands.add_parser("evidence")
    evidence_parser.add_argument("capture")
    evidence_parser.add_argument("workers")
    evidence_parser.set_defaults(handler=evidence)

    synchronize_parser = commands.add_parser("synchronize-evidence")
    synchronize_parser.add_argument("state", choices=("running", "sleeping"))
    synchronize_parser.add_argument("capture")
    synchronize_parser.add_argument("workers")
    synchronize_parser.set_defaults(handler=synchronize_evidence)

    marker_parser = commands.add_parser("write-marker")
    marker_parser.set_defaults(handler=write_marker)
    return result


def main() -> int:
    args = parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Service-runtime helpers shared by ColdSnap snapshot drivers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import coldsnap_cuda_criu as base  # noqa: E402
from coldsnap_coord import Client as CoordinatorClient  # noqa: E402


_COORDINATOR_MAX_WAIT_SECONDS = 3600.0
_PLACEMENT_ARGUMENT = re.compile(
    r"(?<!\S)(--(?:master-(?:addr(?:ess)?|port)|dist-init-addr|port))(?:\s+|=)(?:'[^']*'|\"[^\"]*\"|\S+)"
)
_HTTP_PORT_ARGUMENT = re.compile(r"(?<!\S)--port(?:\s+|=)(?:'([0-9]+)'|\"([0-9]+)\"|([0-9]+))")
_CUDA_VERSION = re.compile(r"^cuda\s*(?::[^=]+)?=\s*(['\"])([^'\"]+)\1\s*$", re.MULTILINE)
_TRANSPORT_ENVIRONMENT_NAMES = {
    "GLOO_SOCKET_IFNAME",
    "MN_IF_NAME",
    "NODE_IP",
    "TP_SOCKET_IFNAME",
    "UCX_NET_DEVICES",
    "VLLM_HOST_IP",
}
_TRANSPORT_ENVIRONMENT_PREFIXES = ("NCCL_", "OMPI_MCA_", "UCX_")
_RESTORE_RUNTIME_ENVIRONMENT_NAMES = {
    "COLDSNAP_API_MM_WARM_FILE",
    "COLDSNAP_API_MM_WARMUP_DELAY_SECONDS",
    "COLDSNAP_ASYNC_CUDA_GRAPHS",
    "COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE",
    "COLDSNAP_ASYNC_CUDA_GRAPHS_GENERATION",
    "COLDSNAP_ASYNC_CUDA_GRAPHS_READY_FILE",
    "COLDSNAP_DEFERRED_API_MM_WARMUP",
    "COLDSNAP_DEFERRED_WARMUP",
    "COLDSNAP_DEFERRED_WARMUP_ARM_FILE",
    "COLDSNAP_FREE_MEMORY_RESERVE_BYTES",
    "COLDSNAP_FULLY_WARM_FILE",
    "COLDSNAP_GRAPH_POLICY",
    "COLDSNAP_GRAPH_POLICY_REQUESTED",
    "COLDSNAP_KV_CAPACITY_GUARD",
    "COLDSNAP_REQUIRED_FREE_MEMORY_BYTES",
    "COLDSNAP_SHAPE_CALIBRATION",
    "COLDSNAP_STARTUP_PLAN_MAX_SHORTFALL_BYTES",
    "COLDSNAP_VLLM_OVERRIDE_CUMEM",
    "COLDSNAP_WARMUP_GENERATION",
    "VLLM_ENABLE_STARTUP_PLAN",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_ID = re.compile(r"^nccl-[0-9]+\.[0-9]+\.[0-9]+-[0-9]+\+coldsnap\.[0-9]+$")
_ACTIVE_FIELDS = {
    "format",
    "kind",
    "provider_id",
    "provider_revision",
    "platform_key",
    "provider_root",
    "provider_manifest",
    "manifest_sha256",
    "provider_abi",
    "checkpoint_abi",
    "provider_nccl_runtime",
    "provider_selection",
    "supported_nccl_runtimes",
    "capabilities",
    "limitations",
    "qualification",
    "files",
    "bridge",
    "preload_order",
}
_MANIFEST_FIELDS = {
    "format",
    "kind",
    "provider_id",
    "provider_revision",
    "platform_key",
    "provider_nccl_runtime",
    "provider_selection",
    "supported_nccl_runtimes",
    "provider_abi",
    "checkpoint_abi",
    "capabilities",
    "limitations",
    "requirements",
    "files",
    "provenance",
    "qualification",
    "dlsym_bridge_abi",
    "preload_order",
}


def _bounded_json(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > 4 * 1024**2:
            raise RuntimeError(f"{path} must be a bounded regular file")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read NCCL provider record {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"NCCL provider record {path} must be an object")
    return payload


def _exact_fields(payload: dict[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected.difference(payload))
        unknown = sorted(set(payload).difference(expected))
        raise RuntimeError(f"{name} fields are invalid: missing={missing} unknown={unknown}")


def _clean_absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeError(f"active NCCL runtime has invalid {field}")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.normpath(value)) != path:
        raise RuntimeError(f"active NCCL runtime has non-canonical {field}")
    try:
        if path.resolve(strict=True) != path:
            raise RuntimeError(f"active NCCL runtime {field} traverses a symlink")
        info = path.lstat()
    except OSError as error:
        raise RuntimeError(f"active NCCL runtime cannot resolve {field}: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"active NCCL runtime {field} is not a regular file")
    return path


def _load_active_nccl_runtime(path: Path) -> dict[str, Any]:
    active = _bounded_json(path)
    _exact_fields(active, _ACTIVE_FIELDS, "active NCCL runtime")
    if (
        active.get("format") != 2
        or active.get("kind") != "coldsnap-nccl-active-runtime"
        or not isinstance(active.get("provider_id"), str)
        or _PROVIDER_ID.fullmatch(active["provider_id"]) is None
        or not isinstance(active.get("provider_revision"), int)
        or active["provider_revision"] <= 0
        or not isinstance(active.get("platform_key"), str)
        or not isinstance(active.get("checkpoint_abi"), int)
        or active["checkpoint_abi"] <= 0
    ):
        raise RuntimeError("active NCCL runtime identity is invalid")
    provider_root = Path(str(active.get("provider_root")))
    if (
        not provider_root.is_absolute()
        or Path(os.path.normpath(str(provider_root))) != provider_root
    ):
        raise RuntimeError("active NCCL runtime provider_root is invalid")
    provider_abi = active.get("provider_abi")
    if (
        not isinstance(provider_abi, dict)
        or set(provider_abi) != {"major", "minor"}
        or not isinstance(provider_abi.get("major"), int)
        or provider_abi["major"] <= 0
        or not isinstance(provider_abi.get("minor"), int)
        or provider_abi["minor"] < 0
    ):
        raise RuntimeError("active NCCL runtime provider_abi is invalid")
    for field in ("capabilities", "limitations"):
        values = active.get(field)
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) and value for value in values)
            or values != sorted(set(values))
        ):
            raise RuntimeError(f"active NCCL runtime {field} is invalid")
    qualification = active.get("qualification")
    if (
        not isinstance(qualification, dict)
        or set(qualification) != {
            "state",
            "policy",
            "capabilities",
            "transports",
            "checks",
        }
        or qualification.get("state") != "accepted"
    ):
        raise RuntimeError("active NCCL qualification is invalid")
    for field in ("capabilities", "transports", "checks"):
        values = qualification.get(field)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
            or values != sorted(set(values))
        ):
            raise RuntimeError(f"active NCCL qualification {field} is invalid")
    if qualification["capabilities"] != active["capabilities"]:
        raise RuntimeError("active NCCL qualification capabilities are inconsistent")
    runtimes = active.get("supported_nccl_runtimes")
    if not isinstance(runtimes, list) or not runtimes:
        raise RuntimeError("active NCCL runtime has no supported runtime identities")
    runtime_fields = {"version", "release", "path", "soname", "build_id", "sha256"}
    for runtime_identity in runtimes:
        if not isinstance(runtime_identity, dict) or set(runtime_identity) != runtime_fields:
            raise RuntimeError("active NCCL supported runtime identity is invalid")
    provider_runtime = active.get("provider_nccl_runtime")
    provider_runtime_fields = {"version", "release", "soname", "build_id", "sha256"}
    if (
        not isinstance(provider_runtime, dict)
        or set(provider_runtime) != provider_runtime_fields
        or not isinstance(provider_runtime.get("version"), int)
        or provider_runtime["version"] <= 0
        or not isinstance(provider_runtime.get("release"), str)
        or not isinstance(provider_runtime.get("sha256"), str)
        or _SHA256.fullmatch(provider_runtime["sha256"]) is None
    ):
        raise RuntimeError("active NCCL provider runtime identity is invalid")
    if active.get("provider_selection") not in {"exact", "fallback-upgrade"}:
        raise RuntimeError("active NCCL provider selection is invalid")

    manifest_path = _clean_absolute_path(active.get("provider_manifest"), "provider_manifest")
    if manifest_path != provider_root / "provider-manifest.json":
        raise RuntimeError("active NCCL manifest is outside the provider root")
    if base._sha256(manifest_path) != active.get("manifest_sha256"):
        raise RuntimeError("active NCCL provider manifest digest mismatch")
    manifest = _bounded_json(manifest_path)
    _exact_fields(manifest, _MANIFEST_FIELDS, "NCCL provider manifest")
    if manifest.get("format") != 2:
        raise RuntimeError("NCCL provider manifest format is invalid")
    for field in (
        "provider_id",
        "provider_revision",
        "platform_key",
        "provider_nccl_runtime",
        "provider_selection",
        "provider_abi",
        "checkpoint_abi",
        "supported_nccl_runtimes",
        "capabilities",
        "limitations",
        "qualification",
    ):
        if active[field] != manifest.get(field):
            raise RuntimeError(f"active NCCL {field} disagrees with provider manifest")

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        raise RuntimeError("NCCL provider manifest files are invalid")
    by_role: dict[str, dict[str, Any]] = {}
    for file in manifest_files:
        if not isinstance(file, dict) or not isinstance(file.get("role"), str):
            raise RuntimeError("NCCL provider manifest file is invalid")
        if file["role"] in by_role:
            raise RuntimeError("NCCL provider manifest has a duplicate file role")
        by_role[file["role"]] = file
    active_files = active.get("files")
    if not isinstance(active_files, dict) or set(active_files) != {
        "checkpoint-shim",
        "nccl-runtime",
    }:
        raise RuntimeError("active NCCL runtime files are invalid")
    resolved: dict[str, Path] = {}
    for role in ("checkpoint-shim", "nccl-runtime"):
        item = active_files[role]
        if not isinstance(item, dict) or set(item) != {
            "path",
            "sha256",
            "build_id",
            "soname",
        }:
            raise RuntimeError(f"active NCCL {role} record is invalid")
        declared = by_role.get(role)
        if not isinstance(declared, dict):
            raise RuntimeError(f"NCCL provider manifest lacks {role}")
        provider_payload_path = _clean_absolute_path(
            str(provider_root / str(declared.get("path"))),
            f"provider.files.{role}.path",
        )
        expected_path = provider_payload_path
        if role == "checkpoint-shim":
            try:
                active_root = provider_root.parents[2]
            except IndexError as error:
                raise RuntimeError("active NCCL provider root is too shallow") from error
            expected_path = active_root / "common/libcoldsnap-checkpoint-shim.so"
        resolved[role] = _clean_absolute_path(item.get("path"), f"files.{role}.path")
        if resolved[role] != expected_path:
            raise RuntimeError(f"active NCCL {role} path disagrees with manifest")
        for field in ("sha256", "build_id", "soname"):
            if item.get(field) != declared.get(field):
                raise RuntimeError(f"active NCCL {role} {field} disagrees with manifest")
        if not isinstance(item.get("sha256"), str) or _SHA256.fullmatch(item["sha256"]) is None:
            raise RuntimeError(f"active NCCL {role} digest is invalid")
        if base._sha256(provider_payload_path) != item["sha256"]:
            raise RuntimeError(f"active NCCL provider {role} digest mismatch")
        if base._sha256(resolved[role]) != item["sha256"]:
            raise RuntimeError(f"active NCCL {role} digest mismatch")
    for field in ("sha256", "build_id", "soname"):
        if provider_runtime[field] != active_files["nccl-runtime"][field]:
            raise RuntimeError(f"active NCCL provider runtime {field} disagrees with its payload")

    bridge = active.get("bridge")
    if (
        not isinstance(bridge, dict)
        or set(bridge) != {"path", "abi", "sha256"}
        or not isinstance(bridge.get("abi"), int)
        or bridge["abi"] <= 0
        or not isinstance(bridge.get("sha256"), str)
        or _SHA256.fullmatch(bridge["sha256"]) is None
    ):
        raise RuntimeError("active NCCL dlsym bridge is invalid")
    bridge_path = _clean_absolute_path(bridge.get("path"), "bridge.path")
    if base._sha256(bridge_path) != bridge["sha256"]:
        raise RuntimeError("active NCCL dlsym bridge digest mismatch")
    expected_preload = [
        str(bridge_path),
        str(resolved["checkpoint-shim"]),
        str(resolved["nccl-runtime"]),
    ]
    if active.get("preload_order") != expected_preload:
        raise RuntimeError("active NCCL preload order is invalid")
    active["_resolved"] = {
        "checkpoint-shim": resolved["checkpoint-shim"],
        "nccl-runtime": resolved["nccl-runtime"],
        "bridge": bridge_path,
    }
    return active


def _request(
    args: argparse.Namespace,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float | None = None,
) -> Any:
    url = f"http://{args.master_address}:{args.http_port}{path}"
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    request_timeout = args.timeout if timeout is None else timeout
    with urllib.request.urlopen(  # nosec B310
        request, timeout=max(0.001, float(request_timeout))
    ) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def _wait_http(
    args: argparse.Namespace,
    path: str,
    *,
    process: subprocess.Popen[bytes] | None = None,
) -> Any:
    deadline = time.monotonic() + args.timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{args.engine} exited early with status {process.returncode}")
        try:
            return _request(args, "GET", path)
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.25)
    raise TimeoutError(f"HTTP wait for {path} failed: {last_error}")


def _collective_rpc(args: argparse.Namespace, method: str) -> Any:
    return _request(
        args,
        "POST",
        "/collective_rpc",
        {
            "method": method,
            "timeout": max(1, int(min(_COORDINATOR_MAX_WAIT_SECONDS, args.timeout))),
        },
    )


def _shape_calibration_enabled() -> bool:
    value = os.environ.get("COLDSNAP_SHAPE_CALIBRATION", "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise RuntimeError("COLDSNAP_SHAPE_CALIBRATION must be a boolean")


def _validate_shape_calibration_detail(value: Any, engine: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{engine} shape calibration evidence is not an object")
    planned = value.get("planned_shapes")
    warmed = value.get("warmed_shapes")
    seconds = value.get("seconds")
    if (
        value.get("kind") != "coldsnap-shape-calibration"
        or value.get("engine") != engine
        or not isinstance(planned, int)
        or isinstance(planned, bool)
        or planned <= 0
        or warmed != planned
        or not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or float(seconds) < 0
        or not math.isfinite(float(seconds))
        or not isinstance(value.get("modes"), list)
        or not value["modes"]
        or not isinstance(value.get("toolchain"), dict)
        or not isinstance(value.get("cache_root"), str)
        or not value["cache_root"].startswith("/")
    ):
        raise RuntimeError(f"{engine} shape calibration evidence is incomplete")
    return value


def _capture_shape_calibration(
    args: argparse.Namespace,
    coordinator: CoordinatorClient,
) -> dict[str, Any] | None:
    """Collect engine-owned compile-shape coverage before the target exits."""

    if not _shape_calibration_enabled():
        return None
    details: list[dict[str, Any]] = []
    if args.engine == "vllm":
        key = f"{args.activation_namespace}:shape-calibration"
        if args.rank == 0:
            response = _collective_rpc(args, "coldsnap_async_graph_status")
            coordinator.set(key, json.dumps(response, sort_keys=True))
        else:
            response = json.loads(_coordinator_wait(coordinator, key, args.timeout))
        results = response.get("results") if isinstance(response, dict) else None
        if not isinstance(results, list) or not results:
            raise RuntimeError("vLLM shape calibration returned no worker coverage")
        for status in results:
            detail = status.get("shape_calibration") if isinstance(status, dict) else None
            details.append(_validate_shape_calibration_detail(detail, "vllm"))
    elif args.engine == "sglang":
        status_path = args.artifact_root / "async-graphs-ready.json"
        status = base._load_json(status_path)
        if (
            status.get("kind") != "coldsnap-sglang-async-cuda-graphs"
            or status.get("phase") != "calibrated"
        ):
            raise RuntimeError("SGLang startup did not publish calibrated graph coverage")
        details.append(
            _validate_shape_calibration_detail(
                status.get("shape_calibration"),
                "sglang",
            )
        )
    else:
        raise RuntimeError(f"unsupported shape-calibration engine {args.engine!r}")
    planned = sum(int(value["planned_shapes"]) for value in details)
    warmed = sum(int(value["warmed_shapes"]) for value in details)
    if warmed != planned:
        raise RuntimeError(f"{args.engine} warmed {warmed} of {planned} planned graph shapes")
    return {
        "format": 1,
        "kind": "coldsnap-shape-calibration-coverage",
        "engine": args.engine,
        "planned_shapes": planned,
        "warmed_shapes": warmed,
        "seconds": sum(float(value["seconds"]) for value in details),
        "details": details,
    }


def _coordinator_wait(coordinator: CoordinatorClient, key: str, timeout: float) -> str:
    """Honor an operation deadline using coordinator-sized wait chunks."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for coordinator key {key!r}")
        wait = min(_COORDINATOR_MAX_WAIT_SECONDS, remaining)
        try:
            return coordinator.wait(key, wait)
        except TimeoutError:
            if time.monotonic() >= deadline:
                raise


def _port_is_listening(port: int) -> bool:
    """Use the same local TCP LISTEN boundary as Sparkrun, without connecting."""
    for name in ("tcp", "tcp6"):
        try:
            rows = (Path("/proc/net") / name).read_text().splitlines()[1:]
        except FileNotFoundError:
            continue
        for row in rows:
            fields = row.split()
            if len(fields) >= 4 and fields[3] == "0A" and fields[1].rsplit(":", 1)[-1] == f"{port:04X}":
                return True
    return False


def _service_url(args: argparse.Namespace, path: str) -> str:
    address = args.master_address
    if ":" in address and not address.startswith("["):
        address = f"[{address}]"
    return f"http://{address}:{args.http_port}{path}"


def _local_http_open(request: Any, *, timeout: float):
    # These are same-host observations, never traffic for an environment proxy.
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


class StartupReadiness:
    """Advisory rank-0 TCP/HTTP observers, active before restore work starts.

    These probes never gate restore or issue inference. Successful observations
    use the host wall clock, like Docker StartedAt; absence is not zero latency.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.observed: dict[str, int] = {"observer_started_unix_ns": time.time_ns()}
        self.deadline = time.monotonic() + args.timeout
        self.threads: list[threading.Thread] = []

    def __enter__(self):
        if self.args.mode == "restore" and self.args.rank == 0 and self.args.activation_state == "running":
            self.args.startup_readiness = self
            for kind in ("port_open", "http_ready"):
                thread = threading.Thread(target=self._observe, args=(kind,), daemon=True, name=f"coldsnap-{kind}")
                self.threads.append(thread)
                thread.start()
        return self

    def _observe(self, kind: str) -> None:
        while not self.stop.is_set() and time.monotonic() < self.deadline:
            try:
                if kind == "port_open":
                    if not _port_is_listening(self.args.http_port):
                        raise OSError("port is not listening")
                else:
                    url = _service_url(self.args, "/health")
                    with _local_http_open(url, timeout=0.25) as response:
                        if response.status != 200:
                            raise OSError("health is not ready")
                observed = time.time_ns()
                with self.lock:
                    self.observed[kind + "_unix_ns"] = observed
                return
            except (OSError, urllib.error.URLError):
                self.stop.wait(0.05)

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return dict(self.observed)

    def __exit__(self, *_exc):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=0.5)


def _stream_events(response: Any, deadline: float):
    """Read bounded SSE events, rejecting truncated or oversized responses."""
    data: list[str] = []
    event_bytes = total_bytes = 0
    while True:
        if time.perf_counter() >= deadline:
            raise TimeoutError("acceptance stream exceeded its deadline")
        raw = response.readline(65537)
        if not raw:
            raise RuntimeError("acceptance stream ended before [DONE]")
        total_bytes += len(raw)
        event_bytes += len(raw)
        if len(raw) > 65536 or event_bytes > 65536 or total_bytes > 1024 * 1024:
            raise RuntimeError("acceptance stream exceeds its size limit")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
            data = []
            event_bytes = 0
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))


def _infer(args: argparse.Namespace) -> dict[str, Any]:
    """One streaming acceptance request: timestamp first text, validate all text.

    This runs on rank 0. A first token alone never opens the readiness barrier:
    callers receive timing only after the complete stream passes acceptance.
    Failed attempts (including retained-graph retries) cannot leak a TTFT.
    """
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        _service_url(args, "/v1/chat/completions"),
        data=json.dumps(payload).encode(), method="POST",
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    started_unix_ns = time.time_ns()
    first_token_ns = first_token_elapsed = None
    first_token_field = ""
    pieces: dict[str, list[str]] = {"content": [], "reasoning": [], "reasoning_content": []}
    response: dict[str, Any] = {}
    finish_reason = None
    with _local_http_open(request, timeout=args.timeout) as stream:
        for data in _stream_events(stream, started + args.timeout):
            observed_ns, elapsed = time.time_ns(), time.perf_counter() - started
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not isinstance(chunk, dict) or "error" in chunk:
                raise RuntimeError("acceptance stream returned an error or invalid event")
            for key in ("id", "model", "created", "usage"):
                if key in chunk:
                    response[key] = chunk[key]
            choices = chunk.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise RuntimeError("acceptance stream returned invalid choices")
            if not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise RuntimeError("acceptance stream returned an invalid choice")
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise RuntimeError("acceptance stream returned an invalid delta")
            for field in pieces:
                text = delta.get(field)
                if text is not None and not isinstance(text, str):
                    raise RuntimeError("acceptance stream returned non-text output")
                if text:
                    pieces[field].append(text)
                    if first_token_ns is None:
                        first_token_ns, first_token_elapsed = observed_ns, elapsed
                        first_token_field = field
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    if first_token_ns is None or not isinstance(finish_reason, str) or not finish_reason:
        raise RuntimeError("acceptance stream has no non-empty token or finish reason")
    message = {field: "".join(values) for field, values in pieces.items()}
    actual = message["content"].strip()
    if actual != args.expected:
        raise RuntimeError(f"response mismatch: expected={args.expected!r} actual={actual!r}")
    response["object"] = "chat.completion"
    response["choices"] = [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish_reason}]
    response["coldsnap_acceptance"] = {
        "format": 1,
        "measurement": "rank0-acceptance-v1",
        "observer": "rank0",
        "request_started_unix_ns": started_unix_ns,
        "first_token_unix_ns": first_token_ns,
        "first_token_field": first_token_field,
        "request_ttft_seconds": first_token_elapsed,
        "response_seconds": time.perf_counter() - started,
        "response_validated": True,
        "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
        "max_tokens": payload["max_tokens"],
        "temperature": payload["temperature"],
    }
    readiness = getattr(args, "startup_readiness", None)
    if readiness is not None:
        response["coldsnap_acceptance"]["readiness"] = readiness.snapshot()
        readiness.stop.set()
    return response


def _visible_gpu_facts() -> list[dict[str, str]]:
    output = base._command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,name,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    result: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi identity: {line!r}")
        result.append({"driver": fields[0], "name": fields[1], "compute_capability": fields[2]})
    if not result:
        raise RuntimeError("nvidia-smi returned no visible GPU identities")
    return result


def _torch_cuda_userspace_version() -> str:
    """Read torch's build-time CUDA version without importing torch.

    Importing torch in this controller would add work to the restore critical
    path and could initialize process-global CUDA state.  The generated
    ``torch/version.py`` is immutable inside the digest-pinned capsule and is
    sufficient to bind the CUDA userspace major/minor.
    """

    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate torch to identify CUDA userspace")
    version_path = Path(spec.origin).parent / "version.py"
    try:
        payload = version_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"cannot read torch CUDA userspace identity: {error}") from error
    match = _CUDA_VERSION.search(payload)
    if match is None or not match.group(2).strip():
        raise RuntimeError("torch CUDA userspace identity is unavailable")
    return match.group(2).strip()


def _normalize_placement_command(value: object) -> object:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return value
    result = [
        _PLACEMENT_ARGUMENT.sub(lambda match: f"{match.group(1)}=<placement>", item)
        for item in value
    ]
    for index in range(len(result) - 1):
        if result[index] in {
            "--master-addr",
            "--master-address",
            "--master-port",
            "--dist-init-addr",
            "--port",
        }:
            result[index + 1] = "<placement>"
    return result


def _command_http_port(command: list[str]) -> int:
    match = _HTTP_PORT_ARGUMENT.search(" ".join(command))
    if match is not None:
        for value in match.groups():
            if value is not None:
                port = int(value)
                if 0 < port < 65536:
                    return port
    return 8000


def _identity_http_port(identity: object) -> int:
    if not isinstance(identity, dict):
        raise RuntimeError("process artifact has invalid identity")
    command = identity.get("serve_command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise RuntimeError("process artifact has invalid serve command")
    return _command_http_port(command)


def _command_master_port(command: list[str]) -> int:
    payload = " ".join(command)
    match = re.search(r"(?:^|\s)--master-port(?:\s+|=)(\d+)(?:\s|$)", payload)
    if match is not None:
        port = int(match.group(1))
        if 0 < port < 65536:
            return port
    match = re.search(
        r"(?:^|\s)--dist-init-addr(?:\s+|=)(?:\[[^]]+]|[^\s:]+):(\d+)(?:\s|$)",
        payload,
    )
    if match is not None:
        port = int(match.group(1))
        if 0 < port < 65536:
            return port
    return 25000


def _shift_tcp_port(port: int, shift: int) -> int:
    if not 1024 <= port < 65536:
        raise RuntimeError(f"portable TCP port is outside the unprivileged range: {port}")
    if not 0 <= shift < 65536 - 1024:
        raise RuntimeError(f"portable TCP port shift is invalid: {shift}")
    return 1024 + ((port - 1024 + shift) % (65536 - 1024))


def _identity_mismatch_paths(expected: object, actual: object, path: str = "identity") -> list[str]:
    """Return bounded field paths without leaking command or environment values."""

    if type(expected) is not type(actual):
        return [path]
    if isinstance(expected, dict):
        mismatches: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in expected or key not in actual:
                mismatches.append(child)
            else:
                mismatches.extend(_identity_mismatch_paths(expected[key], actual[key], child))
            if len(mismatches) >= 8:
                return mismatches[:8]
        return mismatches
    if isinstance(expected, list):
        mismatches = []
        if len(expected) != len(actual):
            mismatches.append(f"{path}.length")
        for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
            mismatches.extend(_identity_mismatch_paths(left, right, f"{path}[{index}]"))
            if len(mismatches) >= 8:
                return mismatches[:8]
        return mismatches
    return [] if expected == actual else [path]


def _is_transport_environment(name: str) -> bool:
    return name in _TRANSPORT_ENVIRONMENT_NAMES or name.startswith(_TRANSPORT_ENVIRONMENT_PREFIXES)


def _stage_restore_transport_environment(args: argparse.Namespace) -> Path | None:
    raw_path = os.environ.get("COLDSNAP_RESTORE_TRANSPORT_ENVIRONMENT_PATH")
    if raw_path is None:
        return None
    path = Path(raw_path)
    expected = args.artifact_root / "restore-transport-environment.json"
    if path != expected:
        raise RuntimeError("restore transport environment path is outside the rank capsule")
    variables = {
        name: value for name, value in sorted(os.environ.items()) if _is_transport_environment(name)
    }
    base._atomic_json(
        path,
        {
            "format": 1,
            "kind": "coldsnap-restore-transport-environment",
            "unit": os.environ.get("COLDSNAP_EXPECTED_UNIT", ""),
            "variables": variables,
        },
    )
    return path


def _stage_restore_runtime_environment(args: argparse.Namespace) -> Path | None:
    """Publish target-owned lifecycle settings for a re-executed worker.

    CRIU restores the capture-time process environment. A portable n580
    worker exec must instead receive the current graph, warmup, and startup
    policy selected by the controller. Keep this separate from fabric
    identity: the consumer and bounded allowlist are different, and capture-
    only settings such as shape calibration must be explicitly replaced.
    """
    raw_path = os.environ.get("COLDSNAP_RESTORE_RUNTIME_ENVIRONMENT_PATH")
    if raw_path is None:
        return None
    path = Path(raw_path)
    expected = args.artifact_root / "restore-runtime-environment.json"
    if path != expected:
        raise RuntimeError("restore runtime environment path is outside the rank capsule")
    variables = {
        name: os.environ[name]
        for name in sorted(_RESTORE_RUNTIME_ENVIRONMENT_NAMES)
        if name in os.environ
    }
    base._atomic_json(
        path,
        {
            "format": 1,
            "kind": "coldsnap-restore-runtime-environment",
            "unit": os.environ.get("COLDSNAP_EXPECTED_UNIT", ""),
            "variables": variables,
        },
    )
    return path


def _unit_worker_ids() -> set[str]:
    raw = os.environ.get("COLDSNAP_EXECUTION_GRAPH")
    if raw is None:
        raise RuntimeError("COLDSNAP_EXECUTION_GRAPH is required")
    try:
        graph = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("COLDSNAP_EXECUTION_GRAPH is invalid JSON") from error
    slots = graph.get("by_process_slot") if isinstance(graph, dict) else None
    if not isinstance(slots, dict) or not slots:
        raise RuntimeError("ColdSnap execution graph contains no unit workers")
    workers = set(slots.values())
    if len(workers) != len(slots) or not all(
        isinstance(worker, str) and worker for worker in workers
    ):
        raise RuntimeError("ColdSnap execution graph contains invalid unit workers")
    return workers


def _hydration_manifests(artifact_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((artifact_root / "hydration").glob("*/manifest.json")):
        manifest = base._load_json(path)
        worker = manifest.get("worker_id") if isinstance(manifest, dict) else None
        if not isinstance(worker, str) or not worker or worker in result:
            raise RuntimeError("capsule contains an invalid or duplicate worker hydration manifest")
        result[worker] = path
    expected = _unit_worker_ids()
    if set(result) != expected:
        raise RuntimeError(
            "capsule hydration worker set differs from launch unit: "
            f"expected={sorted(expected)!r} actual={sorted(result)!r}"
        )
    return result


def _select_weight_provider(manifest_path: Path, requested: str) -> None:
    """Select recovery weights without relabeling a complete blob capture."""
    manifest = base._load_json(manifest_path)
    weight_source = manifest.get("weight_source", "blob")
    provider_path = manifest_path.parent / "activation-provider"
    if weight_source == "blob":
        if requested != "native":
            raise RuntimeError("full-blob artifacts require the native weight provider")
        # Full-blob artifacts hydrate manifest.json + weights.blob directly.
        # The "native" selector is reserved for optional split-native replay that
        # may accompany a recovery-aware safetensors artifact.
        provider_path.unlink(missing_ok=True)
        return
    if weight_source != "safetensors":
        raise RuntimeError("capsule hydration manifest has an unsupported weight source")
    if requested == "native" and not manifest_path.with_name("native-manifest.json").is_file():
        raise RuntimeError("recovery-aware capsule does not contain a native replay manifest")
    provider_path.write_text(requested + "\n", encoding="utf-8")


def _hibernate_states(artifact_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted((artifact_root / "hibernate-states").glob("*.json")):
        state = base._load_json(path)
        worker = state.get("worker_id") if isinstance(state, dict) else None
        if not isinstance(worker, str) or not worker or worker in result:
            raise RuntimeError("capsule contains an invalid or duplicate worker hibernation state")
        result[worker] = state
    expected = _unit_worker_ids()
    if set(result) != expected:
        raise RuntimeError(
            "capsule hibernation-state worker set differs from launch unit: "
            f"expected={sorted(expected)!r} actual={sorted(result)!r}"
        )
    return result


def _atomic_sglang_pack(
    destination: Path, components: list[tuple[Path, dict[str, Any]]]
) -> list[dict[str, Any]]:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    index: list[dict[str, Any]] = []
    try:
        with temporary.open("w+b", buffering=0) as output:
            position = 0
            for source, metadata in components:
                aligned = (position + 4095) & ~4095
                if aligned != position:
                    output.seek(position)
                    output.write(bytes(aligned - position))
                output.seek(aligned)
                with source.open("rb", buffering=0) as stream:
                    shutil.copyfileobj(stream, output, length=64 * 1024**2)
                size = source.stat().st_size
                index.append({**metadata, "offset": aligned, "bytes": size})
                position = aligned + size
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        return index
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_sglang_hydration(args: argparse.Namespace) -> dict[str, Path]:
    """Commit one replay descriptor and optional native pack per local worker."""
    expected = _unit_worker_ids()
    semantic_root = args.artifact_root / "semantic"
    hydration_root = args.artifact_root / "hydration"
    export_native = os.environ.get("COLDSNAP_EXPORT_MODEL_PAYLOAD", "0") == "1"
    manifests: dict[str, Path] = {}
    for worker in sorted(expected):
        component_roots = sorted((semantic_root / "workers" / worker).glob("*"))
        components: list[tuple[Path, dict[str, Any]]] = []
        replay_components: list[dict[str, Any]] = []
        for root in component_roots:
            manifest_path = root / "manifest.json"
            blob_path = root / "weights.blob"
            if not manifest_path.is_file():
                continue
            manifest = base._load_json(manifest_path)
            relative = root.relative_to(semantic_root).as_posix()
            component = {
                "artifact": relative,
                "identity_sha256": manifest.get("identity_sha256"),
                "blob_bytes": manifest.get("blob_bytes"),
                "blob_sha256": manifest.get("blob_sha256"),
            }
            if (
                not isinstance(component["identity_sha256"], str)
                or _SHA256.fullmatch(component["identity_sha256"]) is None
                or not isinstance(component["blob_bytes"], int)
                or component["blob_bytes"] <= 0
                or not isinstance(component["blob_sha256"], str)
                or _SHA256.fullmatch(component["blob_sha256"]) is None
            ):
                raise RuntimeError(f"SGLang worker {worker} semantic payload metadata is invalid")
            replay_components.append(component)
            if export_native:
                if not blob_path.is_file() or blob_path.stat().st_size != component["blob_bytes"]:
                    raise RuntimeError(f"SGLang worker {worker} semantic payload is incomplete")
                components.append((blob_path, component))

        if export_native and not components:
            raise RuntimeError(f"SGLang worker {worker} exported no native model components")
        worker_root = hydration_root / worker
        worker_root.mkdir(parents=True, exist_ok=True)
        replay_path = worker_root / "manifest.json"
        base._atomic_json(
            replay_path,
            {
                "format": 1,
                "kind": "coldsnap-sglang-replay-plan",
                "engine": "sglang",
                "worker_id": worker,
                "weight_source": "safetensors",
                "model_id": args.model,
                "model_revision": args.model_revision,
                "components": replay_components,
            },
        )
        if export_native:
            native_components = _atomic_sglang_pack(worker_root / "model-weights.pack", components)
            base._atomic_json(
                worker_root / "native-manifest.json",
                {
                    "format": 1,
                    "kind": "coldsnap-sglang-model-payload",
                    "worker_id": worker,
                    "components": native_components,
                },
            )
            for source, _ in components:
                source.unlink()
        manifests[worker] = replay_path
    return manifests


def _sglang_memory_payload(weight_provider: str = "", recovery_phase: str = "") -> dict[str, Any]:
    if weight_provider == "recovery":
        if recovery_phase == "weights":
            return {"tags": ["weights", "coldsnap_recovery_weights"]}
        if recovery_phase == "runtime":
            return {"tags": ["kv_cache", "cuda_graph", "coldsnap_recovery_runtime"]}
        raise ValueError("SGLang recovery resume requires weights or runtime phase")
    tags = ["kv_cache", "weights", "cuda_graph"]
    return {"tags": tags}

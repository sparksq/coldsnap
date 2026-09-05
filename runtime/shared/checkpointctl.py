#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Probe CUDA process checkpoint support without touching a service process."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import os
import platform
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


FORMAT = 1
KIND = "cuda-process-checkpoint-capability"
PROFILE_KIND = "coldsnap-host-feature-profile"
CHECKPOINT_SYMBOLS = (
    "cuCheckpointProcessGetState",
    "cuCheckpointProcessGetRestoreThreadId",
    "cuCheckpointProcessLock",
    "cuCheckpointProcessCheckpoint",
    "cuCheckpointProcessRestore",
    "cuCheckpointProcessUnlock",
)
STATE_NAMES = {
    0: "running",
    1: "locked",
    2: "checkpointed",
    3: "failed",
}


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compression_type", ctypes.c_ubyte),
        ("gpu_direct_rdma_capable", ctypes.c_ubyte),
        ("usage", ctypes.c_ubyte),
        ("reserved", ctypes.c_ubyte * 5),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requested_handle_types", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32_handle_metadata", ctypes.c_void_p),
        ("alloc_flags", _CUmemAllocationFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_ulonglong)]


class _CUipcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * 64)]


class CUDA:
    def __init__(self) -> None:
        self.library = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_LOCAL)
        self._configure()

    def _configure(self) -> None:
        signatures: dict[str, tuple[list[Any], Any]] = {
            "cuInit": ([ctypes.c_uint], ctypes.c_int),
            "cuDriverGetVersion": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "cuGetErrorName": (
                [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
                ctypes.c_int,
            ),
            "cuGetErrorString": (
                [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
                ctypes.c_int,
            ),
            "cuCheckpointProcessGetState": (
                [ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
                ctypes.c_int,
            ),
            "cuCheckpointProcessLock": (
                [ctypes.c_int, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "cuCheckpointProcessCheckpoint": (
                [ctypes.c_int, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "cuCheckpointProcessRestore": (
                [ctypes.c_int, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "cuCheckpointProcessUnlock": (
                [ctypes.c_int, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "cuDeviceGet": (
                [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
                ctypes.c_int,
            ),
            "cuDevicePrimaryCtxRetain": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
                ctypes.c_int,
            ),
            "cuCtxSetCurrent": ([ctypes.c_void_p], ctypes.c_int),
            "cuCtxSynchronize": ([], ctypes.c_int),
            "cuMemAlloc_v2": (
                [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
                ctypes.c_int,
            ),
            "cuMemcpyHtoD_v2": (
                [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
                ctypes.c_int,
            ),
            "cuMemcpyDtoH_v2": (
                [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
                ctypes.c_int,
            ),
            "cuMemFree_v2": ([ctypes.c_uint64], ctypes.c_int),
            "cuIpcGetMemHandle": (
                [ctypes.POINTER(_CUipcMemHandle), ctypes.c_uint64],
                ctypes.c_int,
            ),
            "cuIpcOpenMemHandle_v2": (
                [ctypes.POINTER(ctypes.c_uint64), _CUipcMemHandle, ctypes.c_uint],
                ctypes.c_int,
            ),
            "cuIpcCloseMemHandle": ([ctypes.c_uint64], ctypes.c_int),
            "cuMemGetAllocationGranularity": (
                [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(_CUmemAllocationProp), ctypes.c_int],
                ctypes.c_int,
            ),
            "cuMemAddressReserve": (
                [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_ulonglong],
                ctypes.c_int,
            ),
            "cuMemAddressFree": ([ctypes.c_uint64, ctypes.c_size_t], ctypes.c_int),
            "cuMemCreate": (
                [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.POINTER(_CUmemAllocationProp), ctypes.c_ulonglong],
                ctypes.c_int,
            ),
            "cuMemRelease": ([ctypes.c_uint64], ctypes.c_int),
            "cuMemMap": (
                [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_ulonglong],
                ctypes.c_int,
            ),
            "cuMemUnmap": ([ctypes.c_uint64, ctypes.c_size_t], ctypes.c_int),
            "cuMemSetAccess": (
                [ctypes.c_uint64, ctypes.c_size_t, ctypes.POINTER(_CUmemAccessDesc), ctypes.c_size_t],
                ctypes.c_int,
            ),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name, None)
            if function is None:
                continue
            function.argtypes = arguments
            function.restype = result

    def error_name(self, code: int) -> str:
        output = ctypes.c_char_p()
        if self.library.cuGetErrorName(code, ctypes.byref(output)) == 0:
            return output.value.decode() if output.value else "unknown"
        return "unknown"

    def error_string(self, code: int) -> str:
        output = ctypes.c_char_p()
        if self.library.cuGetErrorString(code, ctypes.byref(output)) == 0:
            return output.value.decode() if output.value else "unknown"
        return "unknown"

    def operation(self, name: str, *arguments: Any) -> dict[str, Any]:
        started = time.monotonic_ns()
        code = int(getattr(self.library, name)(*arguments))
        return {
            "operation": name,
            "code": code,
            "error_name": self.error_name(code),
            "error_string": self.error_string(code),
            "seconds": (time.monotonic_ns() - started) / 1_000_000_000,
            "success": code == 0,
        }

    def state(self, pid: int) -> dict[str, Any]:
        state = ctypes.c_int(-1)
        result = self.operation(
            "cuCheckpointProcessGetState", pid, ctypes.byref(state)
        )
        if result["success"]:
            result["state"] = STATE_NAMES.get(state.value, f"unknown-{state.value}")
            result["state_code"] = state.value
        return result


def _required(cuda: CUDA, name: str, *arguments: Any) -> None:
    if not hasattr(cuda.library, name):
        raise NotImplementedError(f"CUDA driver symbol {name} is unavailable")
    result = cuda.operation(name, *arguments)
    if not result["success"]:
        raise RuntimeError(
            f"{name} failed: {result['error_name']}: {result['error_string']}"
        )


def _pattern(length: int) -> bytearray:
    block = bytes((index * 31 + 7) & 0xFF for index in range(4096))
    return bytearray((block * ((length + len(block) - 1) // len(block)))[:length])


def _cuda_context(cuda: CUDA) -> tuple[int, ctypes.c_void_p]:
    _required(cuda, "cuInit", 0)
    device = ctypes.c_int()
    _required(cuda, "cuDeviceGet", ctypes.byref(device), 0)
    context = ctypes.c_void_p()
    _required(cuda, "cuDevicePrimaryCtxRetain", ctypes.byref(context), device.value)
    _required(cuda, "cuCtxSetCurrent", context)
    return device.value, context


def target(memory_bytes: int) -> int:
    cuda = CUDA()
    _cuda_context(cuda)
    pointer = ctypes.c_uint64()
    _required(cuda, "cuMemAlloc_v2", ctypes.byref(pointer), memory_bytes)
    expected = _pattern(memory_bytes)
    expected_view = (ctypes.c_ubyte * memory_bytes).from_buffer(expected)
    _required(
        cuda,
        "cuMemcpyHtoD_v2",
        pointer.value,
        ctypes.cast(expected_view, ctypes.c_void_p),
        memory_bytes,
    )
    _required(cuda, "cuCtxSynchronize")
    expected_sha256 = hashlib.sha256(expected).hexdigest()
    print(
        json.dumps(
            {
                "status": "ready",
                "pid": os.getpid(),
                "memory_bytes": memory_bytes,
                "expected_sha256": expected_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for line in sys.stdin:
        command = line.strip()
        if command == "verify":
            actual = bytearray(memory_bytes)
            actual_view = (ctypes.c_ubyte * memory_bytes).from_buffer(actual)
            try:
                _required(
                    cuda,
                    "cuMemcpyDtoH_v2",
                    ctypes.cast(actual_view, ctypes.c_void_p),
                    pointer.value,
                    memory_bytes,
                )
                _required(cuda, "cuCtxSynchronize")
                actual_sha256 = hashlib.sha256(actual).hexdigest()
                print(
                    json.dumps(
                        {
                            "status": "verified",
                            "actual_sha256": actual_sha256,
                            "matches": actual_sha256 == expected_sha256,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "status": "verify-failed",
                            "error": f"{type(error).__name__}: {error}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        elif command == "exit":
            return 0
        else:
            print(json.dumps({"status": "invalid-command"}), flush=True)
    return 0


def _ipc_target(handle_text: str, memory_bytes: int, expected_sha256: str) -> int:
    cuda = CUDA()
    _cuda_context(cuda)
    raw = base64.b64decode(handle_text, validate=True)
    if len(raw) != ctypes.sizeof(_CUipcMemHandle):
        raise ValueError("CUDA IPC handle has an invalid size")
    handle = _CUipcMemHandle.from_buffer_copy(raw)
    pointer = ctypes.c_uint64()
    _required(
        cuda,
        "cuIpcOpenMemHandle_v2",
        ctypes.byref(pointer),
        handle,
        1,
    )
    try:
        actual = bytearray(memory_bytes)
        view = (ctypes.c_ubyte * memory_bytes).from_buffer(actual)
        _required(
            cuda,
            "cuMemcpyDtoH_v2",
            ctypes.cast(view, ctypes.c_void_p),
            pointer.value,
            memory_bytes,
        )
        _required(cuda, "cuCtxSynchronize")
        actual_sha256 = hashlib.sha256(actual).hexdigest()
        print(json.dumps({"matches": actual_sha256 == expected_sha256, "actual_sha256": actual_sha256}))
        return 0 if actual_sha256 == expected_sha256 else 1
    finally:
        _required(cuda, "cuIpcCloseMemHandle", pointer.value)


def _feature_result(
    feature: str,
    status: str,
    probe_name: str,
    started: float,
    *,
    reason: str = "",
    failure_phase: str = "",
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": feature,
        "status": status,
        "probe": probe_name,
        "seconds": time.perf_counter() - started,
    }
    if reason:
        value["reason"] = reason
    if failure_phase:
        value["failure_phase"] = failure_phase
    if evidence is not None:
        value["evidence"] = evidence
    return value


def probe_vmm_exact_address() -> dict[str, Any]:
    started = time.perf_counter()
    phase = "load"
    try:
        cuda = CUDA()
        required = (
            "cuMemGetAllocationGranularity", "cuMemAddressReserve", "cuMemAddressFree",
            "cuMemCreate", "cuMemRelease", "cuMemMap", "cuMemUnmap", "cuMemSetAccess",
        )
        missing = [name for name in required if not hasattr(cuda.library, name)]
        if missing:
            return _feature_result(
                "cuda-vmm-exact-address", "unsupported", "cuda-vmm-exact-v1", started,
                reason="missing CUDA VMM symbols: " + ", ".join(missing), failure_phase="symbols",
            )
        phase = "context"
        device, _ = _cuda_context(cuda)
        prop = _CUmemAllocationProp()
        prop.type = 1
        prop.location = _CUmemLocation(1, device)
        granularity = ctypes.c_size_t()
        phase = "granularity"
        _required(cuda, "cuMemGetAllocationGranularity", ctypes.byref(granularity), ctypes.byref(prop), 0)
        size = granularity.value
        address = ctypes.c_uint64()
        handle = ctypes.c_uint64()
        phase = "reserve-initial"
        _required(cuda, "cuMemAddressReserve", ctypes.byref(address), size, 0, 0, 0)
        initial = address.value
        try:
            phase = "create"
            _required(cuda, "cuMemCreate", ctypes.byref(handle), size, ctypes.byref(prop), 0)
            try:
                phase = "map-initial"
                _required(cuda, "cuMemMap", initial, size, 0, handle.value, 0)
                access = _CUmemAccessDesc(_CUmemLocation(1, device), 3)
                _required(cuda, "cuMemSetAccess", initial, size, ctypes.byref(access), 1)
                expected = _pattern(size)
                expected_view = (ctypes.c_ubyte * size).from_buffer(expected)
                phase = "write-initial"
                _required(
                    cuda,
                    "cuMemcpyHtoD_v2",
                    initial,
                    ctypes.cast(expected_view, ctypes.c_void_p),
                    size,
                )
                _required(cuda, "cuCtxSynchronize")
                _required(cuda, "cuMemUnmap", initial, size)
                phase = "free-initial-va"
                _required(cuda, "cuMemAddressFree", initial, size)
                address.value = 0
                phase = "reserve-exact"
                _required(cuda, "cuMemAddressReserve", ctypes.byref(address), size, 0, initial, 0)
                if address.value != initial:
                    raise RuntimeError(
                        f"exact CUDA VA request returned {address.value:#x}, expected {initial:#x}"
                    )
                phase = "map-exact"
                _required(cuda, "cuMemMap", address.value, size, 0, handle.value, 0)
                _required(cuda, "cuMemSetAccess", address.value, size, ctypes.byref(access), 1)
                actual = bytearray(size)
                actual_view = (ctypes.c_ubyte * size).from_buffer(actual)
                phase = "verify-exact"
                _required(
                    cuda,
                    "cuMemcpyDtoH_v2",
                    ctypes.cast(actual_view, ctypes.c_void_p),
                    address.value,
                    size,
                )
                _required(cuda, "cuCtxSynchronize")
                if actual != expected:
                    raise RuntimeError(
                        "private CUDA VMM contents changed across exact-address remap"
                    )
                _required(cuda, "cuMemUnmap", address.value, size)
            finally:
                if address.value:
                    cuda.operation("cuMemAddressFree", address.value, size)
                    address.value = 0
                cuda.operation("cuMemRelease", handle.value)
        except Exception:
            if address.value == initial:
                cuda.operation("cuMemAddressFree", initial, size)
                address.value = 0
            raise
        return _feature_result(
            "cuda-vmm-exact-address", "passed", "cuda-vmm-exact-v1", started,
            evidence={
                "address": initial,
                "bytes": size,
                "device": device,
                "sha256": hashlib.sha256(expected).hexdigest(),
            },
        )
    except NotImplementedError as error:
        return _feature_result(
            "cuda-vmm-exact-address", "unsupported", "cuda-vmm-exact-v1", started,
            reason=str(error), failure_phase=phase,
        )
    except Exception as error:
        return _feature_result(
            "cuda-vmm-exact-address", "failed", "cuda-vmm-exact-v1", started,
            reason=f"{type(error).__name__}: {error}", failure_phase=phase,
        )


def probe_classic_ipc(memory_bytes: int, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    phase = "load"
    pointer = ctypes.c_uint64()
    try:
        cuda = CUDA()
        missing = [
            name for name in ("cuIpcGetMemHandle", "cuIpcOpenMemHandle_v2", "cuIpcCloseMemHandle")
            if not hasattr(cuda.library, name)
        ]
        if missing:
            return _feature_result(
                "cuda-classic-ipc", "unsupported", "cuda-classic-ipc-v1", started,
                reason="missing CUDA IPC symbols: " + ", ".join(missing), failure_phase="symbols",
            )
        phase = "context"
        _cuda_context(cuda)
        phase = "allocate"
        _required(cuda, "cuMemAlloc_v2", ctypes.byref(pointer), memory_bytes)
        expected = _pattern(memory_bytes)
        view = (ctypes.c_ubyte * memory_bytes).from_buffer(expected)
        _required(cuda, "cuMemcpyHtoD_v2", pointer.value, ctypes.cast(view, ctypes.c_void_p), memory_bytes)
        _required(cuda, "cuCtxSynchronize")
        handle = _CUipcMemHandle()
        phase = "export"
        _required(cuda, "cuIpcGetMemHandle", ctypes.byref(handle), pointer.value)
        handle_text = base64.b64encode(bytes(handle)).decode("ascii")
        phase = "consumer"
        result = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()), "_ipc_target",
                "--handle", handle_text, "--memory-bytes", str(memory_bytes),
                "--expected-sha256", hashlib.sha256(expected).hexdigest(),
            ],
            check=False, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"IPC consumer exited {result.returncode}: {(result.stderr or result.stdout).strip()}"
            )
        evidence = json.loads(result.stdout)
        if evidence.get("matches") is not True:
            raise RuntimeError("IPC consumer returned mismatched device bytes")
        return _feature_result(
            "cuda-classic-ipc", "passed", "cuda-classic-ipc-v1", started,
            evidence={"memory_bytes": memory_bytes, **evidence},
        )
    except NotImplementedError as error:
        return _feature_result(
            "cuda-classic-ipc", "unsupported", "cuda-classic-ipc-v1", started,
            reason=str(error), failure_phase=phase,
        )
    except Exception as error:
        return _feature_result(
            "cuda-classic-ipc", "failed", "cuda-classic-ipc-v1", started,
            reason=f"{type(error).__name__}: {error}", failure_phase=phase,
        )
    finally:
        if pointer.value:
            try:
                cuda.operation("cuMemFree_v2", pointer.value)
            except UnboundLocalError:
                pass


def feature_profile(memory_bytes: int, timeout: float) -> dict[str, Any]:
    started = time.time()
    checkpoint = probe(True, memory_bytes, timeout)
    if checkpoint.get("api_available"):
        api_status, api_reason = "passed", ""
    else:
        symbols = checkpoint.get("symbols")
        missing_symbols = isinstance(symbols, dict) and not all(symbols.values())
        api_status = "unsupported" if missing_symbols else "failed"
        api_reason = checkpoint.get("error") or "CUDA process checkpoint symbols are unavailable"
    if checkpoint.get("roundtrip_supported") is True:
        roundtrip_status, roundtrip_reason = "passed", ""
    elif checkpoint.get("api_available"):
        roundtrip_status, roundtrip_reason = "failed", checkpoint.get("error") or "checkpoint state transition or byte canary failed"
    else:
        roundtrip_status, roundtrip_reason = api_status, "CUDA process checkpoint API is unavailable"
    checkpoint_seconds = max(0.0, float(checkpoint.get("completed_unix", started)) - started)
    features = [
        {
            "id": "cuda-checkpoint-api", "status": api_status,
            "probe": "cuda-process-checkpoint-v1", "seconds": checkpoint_seconds,
            **({"reason": api_reason, "failure_phase": "symbols"} if api_reason else {}),
        },
        {
            "id": "cuda-checkpoint-roundtrip", "status": roundtrip_status,
            "probe": "cuda-process-checkpoint-v1", "seconds": checkpoint_seconds,
            **({"reason": roundtrip_reason, "failure_phase": "roundtrip"} if roundtrip_reason else {}),
        },
        probe_vmm_exact_address(),
        probe_classic_ipc(memory_bytes, timeout),
        {
            "id": "cuda-uvm-checkpoint", "status": "unsupported",
            "probe": "known-negative-v1", "seconds": 0.0,
            "reason": "CUDA process checkpoint does not support UVM allocations",
        },
        {
            "id": "cuda-vmm-exported-handle-checkpoint", "status": "unsupported",
            "probe": "known-negative-v1", "seconds": 0.0,
            "reason": "CUDA process checkpoint does not support exported VMM allocation handles",
        },
        {
            "id": "criu-process-template", "status": "unqualified",
            "probe": "capsule-criu-check-v1", "seconds": 0.0,
            "reason": "the controller has not attached capsule-pinned CRIU evidence",
        },
        {
            "id": "criu-process-tree", "status": "unqualified",
            "probe": "capsule-criu-check-v1", "seconds": 0.0,
            "reason": "the controller has not attached capsule-pinned CRIU evidence",
        },
        {
            "id": "cuda-fresh-state", "status": "passed",
            "probe": "disposable-process-boundary-v1", "seconds": 0.0,
        },
        {
            "id": "cuda-initialized-state",
            "status": "passed" if checkpoint.get("roundtrip_supported") is True else roundtrip_status,
            "probe": "cuda-process-checkpoint-v1", "seconds": checkpoint_seconds,
            **({"reason": roundtrip_reason} if roundtrip_reason else {}),
        },
    ]
    return {
        "format": FORMAT,
        "kind": PROFILE_KIND,
        "identity": {
            "boot_id": _read_text(Path("/proc/sys/kernel/random/boot_id")),
            "hostname": platform.node(),
            "architecture": platform.machine(),
            "kernel": platform.release(),
            "driver_version": _driver_version(),
            "cuda_driver_api": checkpoint.get("driver_api", {}).get("driver_api_version"),
            "criu": _criu_facts(),
        },
        "features": features,
        "started_unix": started,
        "completed_unix": time.time(),
    }


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_json_line(process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError("disposable CUDA process did not respond")
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(
            f"disposable CUDA process exited {process.poll()}: {stderr.strip()}"
        )
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError("disposable CUDA process returned invalid JSON")
    return value


def _criu_facts() -> dict[str, Any]:
    executable = shutil.which("criu")
    value: dict[str, Any] = {"installed": executable is not None}
    if executable is None:
        return value
    value["path"] = executable
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value["version"] = (result.stdout or result.stderr).strip()
        value["version_exit_code"] = result.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        value["version_error"] = f"{type(error).__name__}: {error}"
    return value


def _driver_version() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.partition("\n")[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _cleanup_target(cuda: CUDA, process: subprocess.Popen[str]) -> list[dict[str, Any]]:
    cleanup: list[dict[str, Any]] = []
    if process.poll() is None:
        state = cuda.state(process.pid)
        cleanup.append(state)
        if state.get("state") == "checkpointed":
            cleanup.append(
                cuda.operation("cuCheckpointProcessRestore", process.pid, None)
            )
            state = cuda.state(process.pid)
            cleanup.append(state)
        if state.get("state") == "locked":
            cleanup.append(
                cuda.operation("cuCheckpointProcessUnlock", process.pid, None)
            )
        try:
            assert process.stdin is not None
            process.stdin.write("exit\n")
            process.stdin.flush()
            process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    return cleanup


def probe(roundtrip: bool, memory_bytes: int, timeout: float) -> dict[str, Any]:
    started = time.time()
    report: dict[str, Any] = {
        "format": FORMAT,
        "kind": KIND,
        "started_unix": started,
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "kernel": platform.release(),
        "driver_version": _driver_version(),
        "criu": _criu_facts(),
        "roundtrip_requested": roundtrip,
        "memory_bytes": memory_bytes if roundtrip else 0,
    }
    try:
        cuda = CUDA()
    except OSError as error:
        report.update(
            {
                "library_loaded": False,
                "api_available": False,
                "error": f"{type(error).__name__}: {error}",
                "completed_unix": time.time(),
            }
        )
        return report

    report["library_loaded"] = True
    symbols = {
        name: hasattr(cuda.library, name) for name in CHECKPOINT_SYMBOLS
    }
    report["symbols"] = symbols
    init = cuda.operation("cuInit", 0)
    report["initialization"] = init
    driver_version = ctypes.c_int()
    version = cuda.operation("cuDriverGetVersion", ctypes.byref(driver_version))
    if version["success"]:
        version["driver_api_version"] = driver_version.value
    report["driver_api"] = version
    report["api_available"] = all(symbols.values()) and init["success"]
    if not roundtrip or not report["api_available"]:
        report["roundtrip_supported"] = None
        report["completed_unix"] = time.time()
        return report

    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_target",
            "--memory-bytes",
            str(memory_bytes),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    operations: list[dict[str, Any]] = []
    report["target_pid"] = process.pid
    try:
        ready = _read_json_line(process, timeout)
        if ready.get("status") != "ready" or ready.get("pid") != process.pid:
            raise RuntimeError(f"disposable CUDA process was not ready: {ready}")
        report["target"] = ready
        operations.append(cuda.state(process.pid))
        if operations[-1].get("state") != "running":
            raise RuntimeError("disposable CUDA process is not checkpointable")
        for operation in (
            "cuCheckpointProcessLock",
            "cuCheckpointProcessCheckpoint",
            "cuCheckpointProcessRestore",
            "cuCheckpointProcessUnlock",
        ):
            value = cuda.operation(operation, process.pid, None)
            operations.append(value)
            if not value["success"]:
                break
            operations.append(cuda.state(process.pid))
        report["operations"] = operations
        states = [
            operation.get("state")
            for operation in operations
            if operation["operation"] == "cuCheckpointProcessGetState"
        ]
        sequence_succeeded = (
            all(operation["success"] for operation in operations)
            and states
            == ["running", "locked", "checkpointed", "locked", "running"]
        )
        if sequence_succeeded:
            assert process.stdin is not None
            process.stdin.write("verify\n")
            process.stdin.flush()
            verification = _read_json_line(process, timeout)
            report["verification"] = verification
            report["roundtrip_supported"] = bool(verification.get("matches"))
        else:
            report["roundtrip_supported"] = False
    except Exception as error:
        report["roundtrip_supported"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["cleanup"] = _cleanup_target(cuda, process)
    report["completed_unix"] = time.time()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--roundtrip", action="store_true")
    probe_parser.add_argument("--memory-bytes", type=int, default=1 << 20)
    probe_parser.add_argument("--timeout", type=float, default=30.0)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--memory-bytes", type=int, default=1 << 20)
    profile_parser.add_argument("--timeout", type=float, default=30.0)
    target_parser = subparsers.add_parser("_target")
    target_parser.add_argument("--memory-bytes", type=int, required=True)
    ipc_parser = subparsers.add_parser("_ipc_target")
    ipc_parser.add_argument("--handle", required=True)
    ipc_parser.add_argument("--memory-bytes", type=int, required=True)
    ipc_parser.add_argument("--expected-sha256", required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.memory_bytes < 4096 or arguments.memory_bytes > 1 << 30:
        print("error: memory bytes must be between 4096 and 1 GiB", file=sys.stderr)
        return 2
    if arguments.command == "_target":
        try:
            return target(arguments.memory_bytes)
        except Exception as error:
            print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
    if arguments.command == "_ipc_target":
        try:
            return _ipc_target(
                arguments.handle, arguments.memory_bytes, arguments.expected_sha256
            )
        except Exception as error:
            print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
    if arguments.timeout <= 0 or arguments.timeout > 300:
        print("error: timeout must be greater than zero and at most 300", file=sys.stderr)
        return 2
    value = (
        feature_profile(arguments.memory_bytes, arguments.timeout)
        if arguments.command == "profile"
        else probe(arguments.roundtrip, arguments.memory_bytes, arguments.timeout)
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

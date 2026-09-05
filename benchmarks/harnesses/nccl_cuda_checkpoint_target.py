#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Quiesce and verify a two-node NCCL job around CUDA checkpointing."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

RUNTIME_SHARED = Path(__file__).resolve().parents[2] / "runtime" / "shared"
if str(RUNTIME_SHARED) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SHARED))

from coldsnap_coord import Client as CoordinatorClient  # noqa: E402
from coldsnap_coord import CoordinatorNotFound  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--master-port", type=int, default=29610)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--poll-seconds", type=float, default=0.01)
    parser.add_argument("--block-io-uring", action="store_true")
    parser.add_argument("--nccl-checkpoint", action="store_true")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture one NCCL all-reduce graph and replay the unchanged executable",
    )
    parser.add_argument(
        "--in-place-provider",
        action="store_true",
        help="require the experimental communicator/transport in-place provider ABI",
    )
    parser.add_argument("--require-nccl-ib-reset", action="store_true")
    parser.add_argument("--require-nccl-network-reset", action="store_true")
    parser.add_argument("--control-store", choices=("tcp", "coldsnap"), default="tcp")
    parser.add_argument("--control-store-address-path", type=Path)
    parser.add_argument("--control-store-namespace", default="coldsnap-control-v1")
    return parser


def _block_io_uring() -> None:
    """Install a target-only filter so the unfiltered CRIU parent can trace us."""
    library_name = ctypes.util.find_library("seccomp")
    if library_name is None:
        raise RuntimeError("libseccomp is required to block io_uring")
    seccomp = ctypes.CDLL(library_name, use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    allow = 0x7FFF0000
    deny = 0x00050000 | errno.EPERM
    context = seccomp.seccomp_init(allow)
    if not context:
        raise OSError(ctypes.get_errno(), "seccomp_init failed")
    try:
        for name in (b"io_uring_setup", b"io_uring_enter", b"io_uring_register"):
            number = seccomp.seccomp_syscall_resolve_name(name)
            if number < 0:
                raise RuntimeError(f"libseccomp cannot resolve {name.decode()}")
            result = seccomp.seccomp_rule_add(context, deny, number, 0)
            if result != 0:
                raise OSError(-result, f"seccomp_rule_add failed for {name.decode()}")
        result = seccomp.seccomp_load(context)
        if result != 0:
            raise OSError(-result, "seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)


_InPlaceOperation = ctypes.CFUNCTYPE(ctypes.c_int32)
_InPlaceEvidence = ctypes.CFUNCTYPE(ctypes.c_char_p)


class _InPlaceProvider(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_major", ctypes.c_uint32),
        ("abi_minor", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("communicator_suspend", _InPlaceOperation),
        ("transport_detach", _InPlaceOperation),
        ("transport_reattach", _InPlaceOperation),
        ("communicator_resume", _InPlaceOperation),
        ("evidence_json", _InPlaceEvidence),
    ]


class _NCCLCheckpoint:
    def __init__(
        self,
        *,
        require_ib_reset: bool = False,
        require_network_reset: bool = False,
        in_place_provider: bool = False,
    ) -> None:
        library = ctypes.CDLL(None)
        self._prepare = library.ncclCheckpointPrepare
        self._prepare.argtypes = []
        self._prepare.restype = ctypes.c_int
        self._restore = library.ncclCheckpointRestore
        self._restore.argtypes = []
        self._restore.restype = ctypes.c_int
        self._in_place: _InPlaceProvider | None = None
        if in_place_provider:
            query = getattr(library, "coldsnapNcclInPlaceQuery", None)
            if query is None:
                raise RuntimeError(
                    "the exact NCCL provider has no in-place graph-retention ABI"
                )
            query.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.POINTER(_InPlaceProvider)),
            ]
            query.restype = ctypes.c_int32
            table_pointer = ctypes.POINTER(_InPlaceProvider)()
            self._check(query(1, ctypes.byref(table_pointer)), "coldsnapNcclInPlaceQuery")
            if not table_pointer or table_pointer.contents.struct_size < ctypes.sizeof(_InPlaceProvider):
                raise RuntimeError("the exact NCCL in-place provider ABI is truncated")
            self._in_place = table_pointer.contents
            for name in (
                "communicator_suspend", "transport_detach",
                "transport_reattach", "communicator_resume", "evidence_json",
            ):
                if not getattr(self._in_place, name):
                    raise RuntimeError(f"the exact NCCL in-place provider lacks {name}")
        self._ib_reset = getattr(library, "ncclCheckpointIbReset", None)
        if self._ib_reset is not None:
            self._ib_reset.argtypes = []
            self._ib_reset.restype = ctypes.c_int
        if require_ib_reset and self._ib_reset is None:
            raise RuntimeError("the selected NCCL runtime does not export ncclCheckpointIbReset")
        native_network_reset = getattr(library, "ncclCheckpointNetworkReset", None)
        self._network_reset = native_network_reset or self._ib_reset
        if self._network_reset is not None:
            self._network_reset.argtypes = []
            self._network_reset.restype = ctypes.c_int
        if require_network_reset and native_network_reset is None:
            raise RuntimeError(
                "the selected NCCL runtime does not export "
                "ncclCheckpointNetworkReset"
            )
        get_version = library.ncclCheckpointGetVersion
        get_version.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        get_version.restype = ctypes.c_int
        checkpoint_version = ctypes.c_int()
        nccl_version = ctypes.c_int()
        self._check(
            get_version(ctypes.byref(checkpoint_version), ctypes.byref(nccl_version)),
            "ncclCheckpointGetVersion",
        )
        self.version = {
            "checkpoint": checkpoint_version.value,
            "nccl": nccl_version.value,
            "ib_reset": self._ib_reset is not None,
            "network_reset": native_network_reset is not None,
            "in_place_provider": in_place_provider,
        }

    def _in_place_evidence(self) -> dict[str, Any]:
        if self._in_place is None:
            return {}
        raw = self._in_place.evidence_json()
        if not raw:
            raise RuntimeError("NCCL in-place provider returned no structured evidence")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("NCCL in-place provider evidence is not an object")
        return value

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(f"{operation} failed with ncclResult_t={result}")

    def prepare(self) -> None:
        if self._in_place is not None:
            self._check(self._in_place.communicator_suspend(), "ncclCommSuspend/in-place")
            self._check(self._in_place.transport_detach(), "NCCL in-place transport detach")
            self.version["prepare_evidence"] = self._in_place_evidence()
            return
        self._check(self._prepare(), "ncclCheckpointPrepare")
        if self._network_reset is not None:
            self._check(self._network_reset(), "ncclCheckpointNetworkReset")

    def restore(self) -> None:
        if self._in_place is not None:
            self._check(self._in_place.transport_reattach(), "NCCL in-place transport reattach")
            self._check(self._in_place.communicator_resume(), "ncclCommResume/in-place")
            self.version["restore_evidence"] = self._in_place_evidence()
            return
        self._check(self._restore(), "ncclCheckpointRestore")


def _coordinator_store(dist: Any, endpoint_path: Path, namespace: str, timeout: float) -> Any:
    """Build a c10d Store with no persistent socket to serialize in CRIU."""

    client = CoordinatorClient.from_path(endpoint_path)
    prefix = f"{namespace}:"

    class ColdSnapStore(dist.Store):
        def __init__(self) -> None:
            super().__init__()
            self._timeout_seconds = timeout

        def _key(self, key: str) -> str:
            return prefix + key

        def set(self, key: str, value: bytes) -> None:
            client.set(self._key(key), bytes(value))

        def get(self, key: str) -> bytes:
            return client.get(self._key(key), wait=self._timeout_seconds)

        def add(self, key: str, amount: int) -> int:
            return client.add(self._key(key), amount)

        def check(self, keys: list[str]) -> bool:
            if not keys:
                return True
            try:
                for key in keys:
                    client.get(self._key(key))
            except CoordinatorNotFound:
                return False
            return True

        def wait(self, keys: list[str], timeout_value: timedelta | None = None) -> None:
            seconds = (
                timeout_value.total_seconds()
                if timeout_value is not None
                else self._timeout_seconds
            )
            deadline = time.monotonic() + seconds
            while not self.check(keys):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"ColdSnapStore timed out waiting for {keys!r}")
                time.sleep(0.01)

        def multi_get(self, keys: list[str]) -> list[bytes]:
            return [self.get(key) for key in keys]

        def multi_set(self, keys: list[str], values: list[bytes]) -> None:
            if len(keys) != len(values):
                raise ValueError("ColdSnapStore multi_set keys/values length differs")
            for key, value in zip(keys, values, strict=True):
                self.set(key, value)

        def delete_key(self, key: str) -> bool:
            try:
                client.get(self._key(key))
            except CoordinatorNotFound:
                return False
            client.delete(self._key(key))
            return True

        def append(self, key: str, value: bytes) -> None:
            client.append(self._key(key), bytes(value))

        def set_timeout(self, timeout_value: timedelta) -> None:
            seconds = timeout_value.total_seconds()
            if seconds <= 0:
                raise ValueError("ColdSnapStore timeout must be positive")
            self._timeout_seconds = seconds

    return ColdSnapStore()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"control record is not a JSON object: {path}")
    return value


def _rank_value(rank: int, generation: int) -> float:
    return float(rank + 1 + generation * 10)


def _expected_total(world_size: int, generation: int) -> float:
    return float(world_size * (1 + generation * 10) + world_size * (world_size - 1) // 2)


def _all_reduce(torch: Any, dist: Any, args: argparse.Namespace, generation: int) -> dict[str, Any]:
    started = time.perf_counter()
    tensor = torch.tensor(
        [_rank_value(args.rank, generation)],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    value = float(tensor.cpu().item())
    expected = _expected_total(args.world_size, generation)
    if value != expected:
        raise RuntimeError(f"all-reduce mismatch: actual={value} expected={expected}")
    return {
        "generation": generation,
        "actual": value,
        "expected": expected,
        "seconds": time.perf_counter() - started,
    }


class _CapturedCollective:
    """One minimal unchanged NCCL-backed CUDA graph executable."""

    def __init__(self, torch: Any, dist: Any, args: argparse.Namespace) -> None:
        self._torch = torch
        self._dist = dist
        self._args = args
        self._tensor = torch.empty(1, dtype=torch.float64, device="cuda")
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            self._tensor.fill_(_rank_value(args.rank, 0))
            dist.all_reduce(self._tensor)
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self._tensor.fill_(_rank_value(args.rank, 0))
        with torch.cuda.graph(self.graph):
            dist.all_reduce(self._tensor)
        torch.cuda.synchronize()

    def replay(self, generation: int) -> dict[str, Any]:
        started = time.perf_counter()
        self._tensor.fill_(_rank_value(self._args.rank, generation))
        self.graph.replay()
        self._torch.cuda.synchronize()
        value = float(self._tensor.cpu().item())
        expected = _expected_total(self._args.world_size, generation)
        if value != expected:
            raise RuntimeError(
                f"captured all-reduce mismatch: actual={value} expected={expected}"
            )
        return {
            "generation": generation,
            "actual": value,
            "expected": expected,
            "seconds": time.perf_counter() - started,
            "graph_python_identity": id(self.graph),
            "execution": "unchanged-cuda-graph",
        }


def main() -> int:
    args = _parser().parse_args()
    if args.world_size < 2 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be within a world of at least two ranks")
    if args.timeout <= 0 or args.poll_seconds <= 0:
        raise ValueError("timeouts must be positive")
    if args.require_nccl_ib_reset and not args.nccl_checkpoint:
        raise ValueError("--require-nccl-ib-reset requires --nccl-checkpoint")
    if args.require_nccl_network_reset and not args.nccl_checkpoint:
        raise ValueError("--require-nccl-network-reset requires --nccl-checkpoint")
    if args.cuda_graph and not args.nccl_checkpoint:
        raise ValueError("--cuda-graph requires --nccl-checkpoint")
    if args.in_place_provider and not args.cuda_graph:
        raise ValueError("--in-place-provider requires --cuda-graph")
    if args.control_store == "coldsnap" and args.control_store_address_path is None:
        raise ValueError("ColdSnap control store requires --control-store-address-path")
    if args.control_store == "tcp" and args.control_store_address_path is not None:
        raise ValueError("--control-store-address-path only applies to ColdSnap")
    if args.block_io_uring:
        _block_io_uring()
    checkpoint = (
        _NCCLCheckpoint(
            require_ib_reset=args.require_nccl_ib_reset,
            require_network_reset=args.require_nccl_network_reset,
            in_place_provider=args.in_place_provider,
        )
        if args.nccl_checkpoint
        else None
    )

    args.control_dir.mkdir(parents=True, exist_ok=True)
    command_path = args.control_dir / "command.json"
    fatal_path = args.control_dir / "fatal.json"
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)

    import torch
    import torch.distributed as dist

    try:
        torch.cuda.set_device(0)
        process_group_arguments: dict[str, Any] = {
            "backend": "nccl",
            "rank": args.rank,
            "world_size": args.world_size,
            "timeout": timedelta(seconds=args.timeout),
        }
        if args.control_store == "coldsnap":
            process_group_arguments["store"] = _coordinator_store(
                dist,
                args.control_store_address_path,
                args.control_store_namespace,
                args.timeout,
            )
        dist.init_process_group(**process_group_arguments)
        baseline = _all_reduce(torch, dist, args, 0)
        captured_collective = (
            _CapturedCollective(torch, dist, args) if args.cuda_graph else None
        )
        _atomic_json(
            args.control_dir / "ready.json",
            {
                "format": 1,
                "kind": "nccl-cuda-checkpoint-ready",
                "pid": os.getpid(),
                "rank": args.rank,
                "world_size": args.world_size,
                "io_uring_blocked": args.block_io_uring,
                "nccl_checkpoint": checkpoint.version if checkpoint else None,
                "control_store": {
                    "kind": args.control_store,
                    "namespace": (
                        args.control_store_namespace
                        if args.control_store == "coldsnap"
                        else None
                    ),
                },
                "device": torch.cuda.get_device_name(),
                "baseline": baseline,
                "cuda_graph": {
                    "enabled": captured_collective is not None,
                    "policy": "preserve-nccl-exec" if captured_collective else None,
                    "graph_python_identity": (
                        id(captured_collective.graph) if captured_collective else None
                    ),
                },
                "environment": {
                    name: os.environ.get(name)
                    for name in (
                        "NCCL_IB_DISABLE",
                        "NCCL_IB_HCA",
                        "NCCL_IB_RELEASE_ON_FINALIZE",
                        "NCCL_RAS_ENABLE",
                        "NCCL_SOCKET_IFNAME",
                    )
                },
            },
        )

        last_sequence = 0
        while True:
            try:
                command = _load_json(command_path)
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(args.poll_seconds)
                continue
            sequence = int(command.get("sequence", -1))
            if sequence <= last_sequence:
                time.sleep(args.poll_seconds)
                continue
            operation = command.get("operation")
            started = time.perf_counter()
            response: dict[str, Any] = {
                "format": 1,
                "rank": args.rank,
                "sequence": sequence,
                "operation": operation,
                "pid": os.getpid(),
            }
            if operation == "prepare":
                dist.barrier()
                torch.cuda.synchronize()
                if checkpoint:
                    checkpoint.prepare()
                    response["nccl_checkpoint"] = checkpoint.version
                response["status"] = "prepared"
            elif operation == "restore":
                if checkpoint is None:
                    raise RuntimeError("NCCL checkpoint restore was not configured")
                checkpoint.restore()
                response["nccl_checkpoint"] = checkpoint.version
                response["status"] = "restored"
            elif operation == "run":
                generation = int(command.get("generation", sequence))
                response["collective"] = (
                    captured_collective.replay(generation)
                    if captured_collective is not None
                    else _all_reduce(torch, dist, args, generation)
                )
                response["status"] = "complete"
            elif operation == "shutdown":
                dist.barrier()
                response["status"] = "shutting-down"
            else:
                raise ValueError(f"unsupported operation: {operation!r}")
            response["seconds"] = time.perf_counter() - started
            _atomic_json(
                args.control_dir / "responses" / f"{sequence:08d}.json",
                response,
            )
            last_sequence = sequence
            if operation == "shutdown":
                break
        dist.destroy_process_group()
        return 0
    except BaseException as error:
        _atomic_json(
            fatal_path,
            {
                "format": 1,
                "kind": "nccl-cuda-checkpoint-error",
                "pid": os.getpid(),
                "rank": args.rank,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

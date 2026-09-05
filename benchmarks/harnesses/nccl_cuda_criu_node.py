#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Capture or restore one rank of a coordinated NCCL CUDA+CRIU snapshot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


HARNESS_DIR = Path(__file__).resolve().parent
RUNTIME_ENGINE = HARNESS_DIR.parents[1] / "runtime" / "engine"
RUNTIME_SHARED = HARNESS_DIR.parents[1] / "runtime" / "shared"
for path in (HARNESS_DIR, RUNTIME_ENGINE, RUNTIME_SHARED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import coldsnap_cuda_criu as base  # noqa: E402
from coldsnap_coord import Client as CoordinatorClient  # noqa: E402


DEFAULT_TARGET = HARNESS_DIR / "nccl_cuda_checkpoint_target.py"
TRANSPORT_ENVIRONMENT = (
    "NCCL_NET",
    "NCCL_NET_SHARED_COMMS",
    "NCCL_IB_DISABLE",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_HCA",
    "NCCL_IB_RELEASE_ON_FINALIZE",
    "NCCL_CROSS_NIC",
    "NCCL_RAS_ENABLE",
    "NCCL_SOCKET_IFNAME",
)
SOCKET_INODE = re.compile(r"^socket:\[(\d+)\]$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "restore"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--master-port", type=int, default=29620)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--criu", type=Path, default=Path("/opt/criu-runtime/bin/criu"))
    parser.add_argument(
        "--criu-rpc", type=Path, default=Path("/usr/local/bin/coldsnap-criu-rpc")
    )
    parser.add_argument("--runtime-lib-dir", type=Path, default=Path("/opt/criu-runtime/lib"))
    parser.add_argument(
        "--cuda-checkpoint", type=Path, default=Path("/usr/local/bin/cuda-checkpoint")
    )
    parser.add_argument("--minimum-target-pid", type=int, default=512)
    parser.add_argument("--shared-memory-dir", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--ghost-limit", type=int, default=64 * 1024**2)
    parser.add_argument(
        "--network-lock",
        choices=("nftables", "iptables", "skip"),
        default="nftables",
        help="CRIU network-lock backend; nftables avoids an external iptables binary",
    )
    parser.add_argument(
        "--allow-io-uring",
        action="store_true",
        help="skip the CRIU compatibility preflight (not suitable for current CRIU)",
    )
    parser.add_argument("--nccl-checkpoint-shim", type=Path)
    parser.add_argument("--nccl-library", type=Path)
    parser.add_argument("--nccl-checkpoint-coordinator-path", type=Path)
    parser.add_argument("--require-nccl-ib-reset", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--in-place-provider", action="store_true")
    parser.add_argument(
        "--in-place-activation",
        help="manager-issued activation shared by every rank of one in-place restore",
    )
    parser.add_argument("--control-store", choices=("tcp", "coldsnap"), default="tcp")
    parser.add_argument("--control-store-address-path", type=Path)
    parser.add_argument("--control-store-namespace", default="coldsnap-control-v1")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--compact-output", action="store_true")
    return parser


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "target": args.target.resolve(),
        "criu": args.criu.resolve(),
        "criu_rpc": args.criu_rpc.resolve(),
        "cuda_checkpoint": args.cuda_checkpoint.resolve(),
    }
    if args.nccl_checkpoint_shim is not None:
        paths["nccl_checkpoint_shim"] = args.nccl_checkpoint_shim.resolve()
        paths["nccl_library"] = args.nccl_library.resolve()
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("NCCL snapshot runtime is incomplete: " + ", ".join(missing))
    return {
        "format": 2,
        "kind": "nccl-cuda-criu-identity",
        "image_id": args.image_id,
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "gpu": base._gpu_facts(),
        "runtime_sha256": {name: base._sha256(path) for name, path in paths.items()},
        "configuration": {
            "rank": args.rank,
            "world_size": args.world_size,
            "master_address": args.master_address,
            "master_port": args.master_port,
            "artifact_root": str(args.artifact_root.resolve()),
            "ghost_limit": args.ghost_limit,
            "network_lock": args.network_lock,
            "io_uring_blocked": not args.allow_io_uring,
            "nccl_checkpoint": args.nccl_checkpoint_shim is not None,
            "nccl_checkpoint_coordinator_path": (
                str(args.nccl_checkpoint_coordinator_path.resolve())
                if args.nccl_checkpoint_coordinator_path is not None
                else None
            ),
            "require_nccl_ib_reset": args.require_nccl_ib_reset,
            "cuda_graph": args.cuda_graph,
            "in_place_provider": args.in_place_provider,
            "control_store": args.control_store,
            "control_store_address_path": (
                str(args.control_store_address_path.resolve())
                if args.control_store_address_path is not None
                else None
            ),
            "control_store_namespace": (
                args.control_store_namespace if args.control_store == "coldsnap" else None
            ),
            "transport_environment": {name: os.environ.get(name) for name in TRANSPORT_ENVIRONMENT},
        },
        "runtime_paths": {name: str(path) for name, path in paths.items()},
    }


def _target_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(args.target),
        "--rank",
        str(args.rank),
        "--world-size",
        str(args.world_size),
        "--master-address",
        args.master_address,
        "--master-port",
        str(args.master_port),
        "--control-dir",
        str(args.artifact_root / "control"),
        "--timeout",
        str(args.timeout),
    ]
    if not args.allow_io_uring:
        command.append("--block-io-uring")
    if args.nccl_checkpoint_shim is not None:
        command.append("--nccl-checkpoint")
    if args.require_nccl_ib_reset:
        command.append("--require-nccl-ib-reset")
    if args.cuda_graph:
        command.append("--cuda-graph")
    if args.in_place_provider:
        command.append("--in-place-provider")
    command.extend(["--control-store", args.control_store])
    if args.control_store == "coldsnap":
        command.extend(
            [
                "--control-store-address-path",
                str(args.control_store_address_path),
                "--control-store-namespace",
                args.control_store_namespace,
            ]
        )
    return command


def _target_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    if args.nccl_checkpoint_shim is not None:
        environment["LD_PRELOAD"] = (
            f"{args.nccl_checkpoint_shim.resolve()}:{args.nccl_library.resolve()}"
        )
        environment["NCCL_CHECKPOINT_COORDINATOR_PATH"] = str(
            args.nccl_checkpoint_coordinator_path.resolve()
        )
    if args.in_place_provider:
        environment["COLDSNAP_NCCL_IN_PLACE_MODE"] = "net-reconnect-v1"
        environment["COLDSNAP_NCCL_IN_PLACE_ACTIVATION_PATH"] = str(
            (args.artifact_root / "in-place-activation").resolve()
        )
    return environment


def _criu_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    current = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{args.runtime_lib_dir}:{current}" if current else str(args.runtime_lib_dir)
    )
    environment.pop("CUDA_CHECKPOINT_JOB_FILE", None)
    return environment


def _criu_rpc_command(
    args: argparse.Namespace, action: str, *, pid: int | None = None
) -> list[str]:
    command = [
        str(args.criu_rpc),
        "--action",
        action,
        "--criu",
        str(args.criu),
        "--cuda-checkpoint",
        str(args.cuda_checkpoint),
        "--images-dir",
        str(args.artifact_root / "images"),
        "--work-dir",
        str(args.artifact_root / "work"),
        "--log-file",
        f"{action}.log",
        "--external-files",
        str(args.artifact_root / "external-files.json"),
        "--result",
        str(args.artifact_root / f"{action}-rpc.json"),
        "--timeout",
        str(max(1, int(args.timeout))),
        "--compress-block-size",
        "0",
        "--image-io-mode",
        "writeback",
        "--tcp-established",
        "--ghost-limit",
        str(args.ghost_limit),
    ]
    if action == "dump":
        if pid is None:
            raise ValueError("CRIU RPC dump requires a target PID")
        command.extend(["--pid", str(pid), "--network-lock", args.network_lock])
    elif pid is not None:
        raise ValueError("CRIU RPC restore does not accept a target PID")
    return command


def _criu_rpc_result(args: argparse.Namespace, action: str) -> dict[str, Any]:
    result = base._load_json(args.artifact_root / f"{action}-rpc.json")
    if (
        result.get("format") != 1
        or result.get("kind") != "coldsnap-criu-rpc-result"
        or result.get("action") != action
    ):
        raise RuntimeError(f"invalid CRIU RPC result: {result!r}")
    return result


def _tcp_rows(text: str, family: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 10:
            continue
        rows.append(
            {
                "family": family,
                "local": fields[1].upper(),
                "remote": fields[2].upper(),
                "state": fields[3].upper(),
                "inode": fields[9],
            }
        )
    return rows


def _tcp_table(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, family in (("tcp", "ipv4"), ("tcp6", "ipv6")):
        try:
            rows.extend(_tcp_rows((root / name).read_text(encoding="utf-8"), family))
        except FileNotFoundError:
            continue
    return rows


def _captured_tcp_sockets(pid: int) -> list[dict[str, str]]:
    inodes: set[str] = set()
    for descriptor in Path(f"/proc/{pid}/fd").iterdir():
        try:
            match = SOCKET_INODE.match(os.readlink(descriptor))
        except (FileNotFoundError, PermissionError):
            continue
        if match:
            inodes.add(match.group(1))
    sockets = [row for row in _tcp_table(Path(f"/proc/{pid}/net")) if row["inode"] in inodes]
    return sorted(
        sockets,
        key=lambda row: (row["family"], row["local"], row["remote"], row["state"]),
    )


def _assert_tcp_identities_available(saved: object) -> None:
    if not isinstance(saved, list):
        return
    live = {
        (row["family"], row["local"], row["remote"]): row["state"]
        for row in _tcp_table(Path("/proc/net"))
    }
    conflicts: list[str] = []
    for value in saved:
        if not isinstance(value, dict):
            continue
        identity = (
            str(value.get("family")),
            str(value.get("local")),
            str(value.get("remote")),
        )
        if identity in live:
            conflicts.append(f"{identity[0]} {identity[1]} -> {identity[2]} state={live[identity]}")
    if conflicts:
        raise RuntimeError(
            "captured TCP identities are still present on this host; "
            "refusing an unsafe restore: " + "; ".join(conflicts)
        )


def _manager_barrier(args: argparse.Namespace, phase: str) -> None:
    """Synchronize rank controllers without adding a target-process socket."""

    if args.control_store_address_path is None:
        raise ValueError("the in-place harness barrier requires the ColdSnap store")
    client = CoordinatorClient.from_path(args.control_store_address_path)
    prefix = f"nccl-in-place-harness:{args.control_store_namespace}:{phase}"
    client.set(f"{prefix}:rank-{args.rank}", b"ready")
    for rank in range(args.world_size):
        value = client.get(f"{prefix}:rank-{rank}", wait=args.timeout)
        if value != b"ready":
            raise RuntimeError(f"invalid in-place harness barrier value for rank {rank}")


def capture(args: argparse.Namespace) -> dict[str, Any]:
    identity = _identity(args)
    images = args.artifact_root / "images"
    work = args.artifact_root / "work"
    control = args.artifact_root / "control"
    if (images / "inventory.img").exists() or (args.artifact_root / "capture.json").exists():
        raise FileExistsError("refusing to overwrite an NCCL process artifact")
    for path in (images, work, control / "responses"):
        path.mkdir(parents=True, exist_ok=True)
    generation = f"nccl-cuda-criu-{uuid.uuid4().hex}"
    if args.in_place_provider:
        (args.artifact_root / "in-place-activation").write_text(
            generation + "\n", encoding="utf-8"
        )
    pid_floor = base._advance_pid_allocator(args.minimum_target_pid)
    log_path = args.artifact_root / "target.log"
    started = time.perf_counter()
    environment = _target_environment(args)
    target: subprocess.Popen[bytes]
    with log_path.open("ab", buffering=0) as log:
        target = subprocess.Popen(
            _target_command(args),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            ready = base._wait_json(
                control / "ready.json",
                args.timeout,
                fatal_path=control / "fatal.json",
                process=target,
            )
            if ready.get("io_uring_blocked") != (not args.allow_io_uring):
                raise RuntimeError("target io_uring policy differs from artifact identity")
            checkpoint_version = ready.get("nccl_checkpoint")
            if (checkpoint_version is not None) != (args.nccl_checkpoint_shim is not None):
                raise RuntimeError("target NCCL checkpoint policy differs from identity")
            if args.require_nccl_ib_reset and not (
                isinstance(checkpoint_version, dict) and checkpoint_version.get("ib_reset") is True
            ):
                raise RuntimeError("target NCCL runtime lacks the required IB reset ABI")
            expected_store = {
                "kind": args.control_store,
                "namespace": (
                    args.control_store_namespace if args.control_store == "coldsnap" else None
                ),
            }
            if ready.get("control_store") != expected_store:
                raise RuntimeError("target control-store policy differs from identity")
            prepared = base._send_command(args, 0, 1, "prepare")
            if args.in_place_provider:
                _manager_barrier(args, "capture-detached")
            tree = base._process_tree(target.pid)
            captured_sockets = _captured_tcp_sockets(target.pid)
            rss_bytes = sum(base._rss_bytes(pid) for pid in tree)
            command = _criu_rpc_command(args, "dump", pid=target.pid)
            dump_seconds = base._run_criu(command, _criu_environment(args))
            criu_rpc = _criu_rpc_result(args, "dump")
            target.wait(timeout=30)
            shm_count, shm_bytes = base._snapshot_link_remaps(args)
        except BaseException:
            if target.poll() is None:
                base._kill_process_group(target.pid)
                target.wait(timeout=30)
            raise
    image_bytes = sum(path.stat().st_size for path in images.rglob("*") if path.is_file())
    report = {
        "format": 1,
        "kind": "nccl-cuda-criu-capture",
        "generation": generation,
        "identity": identity,
        "target_pid": target.pid,
        "pid_floor": pid_floor,
        "process_tree": tree,
        "captured_tcp_sockets": captured_sockets,
        "process_tree_rss_bytes": rss_bytes,
        "ready": ready,
        "prepare": prepared,
        "dump_seconds": dump_seconds,
        "criu_rpc": criu_rpc,
        "capture_controller_seconds": time.perf_counter() - started,
        "image_bytes": image_bytes,
        "shared_memory_link_count": shm_count,
        "shared_memory_link_bytes": shm_bytes,
    }
    base._atomic_json(args.artifact_root / "capture.json", report)
    return report


def restore(args: argparse.Namespace) -> dict[str, Any]:
    capture_report = base._load_json(args.artifact_root / "capture.json")
    _assert_tcp_identities_available(capture_report.get("captured_tcp_sockets"))
    identity = _identity(args)
    if capture_report.get("identity") != identity:
        raise RuntimeError(
            "NCCL process artifact identity mismatch: "
            f"saved={capture_report.get('identity')!r} current={identity!r}"
        )
    control = args.artifact_root / "control"
    (control / "command.json").unlink(missing_ok=True)
    (control / "fatal.json").unlink(missing_ok=True)
    for sequence in (2, 3, 4):
        (control / "responses" / f"{sequence:08d}.json").unlink(missing_ok=True)
    shm_count = base._prepare_link_remaps(args)
    if args.in_place_provider:
        if not args.in_place_activation:
            raise ValueError(
                "--in-place-activation is required for an in-place restore and must match on every rank"
            )
        (args.artifact_root / "in-place-activation").write_text(
            args.in_place_activation + "\n", encoding="utf-8"
        )
    command = _criu_rpc_command(args, "restore")
    started_ns = time.monotonic_ns()
    restored_pid: int | None = None
    try:
        criu_seconds = base._run_criu(command, _criu_environment(args))
        criu_rpc = _criu_rpc_result(args, "restore")
        restored_pid = int(criu_rpc["restored_pid"])
        checkpoint_restore = None
        collective_sequence = 2
        shutdown_sequence = 3
        if args.nccl_checkpoint_shim is not None:
            checkpoint_restore = base._send_command(args, 0, 2, "restore")
            collective_sequence = 3
            shutdown_sequence = 4
        collective = base._send_command(args, 1, collective_sequence, "run")
        response_ns = time.monotonic_ns()
        result = collective.get("collective", {})
        if result.get("actual") != result.get("expected"):
            raise RuntimeError(f"restored NCCL collective differs: {collective}")
        shutdown = base._send_command(args, 0, shutdown_sequence, "shutdown")
        forced_shutdown = False
        try:
            base._wait_process_exit(restored_pid, 5)
        except TimeoutError:
            forced_shutdown = True
            base._kill_process_group(restored_pid)
            base._wait_process_exit(restored_pid, 5)
    except BaseException:
        if restored_pid is not None:
            base._kill_process_group(restored_pid)
        raise
    report = {
        "format": 1,
        "kind": "nccl-cuda-criu-restore",
        "identity": identity,
        "restored_pid": restored_pid,
        "criu_restore_seconds": criu_seconds,
        "criu_rpc": criu_rpc,
        "nccl_checkpoint_restore": checkpoint_restore,
        "collective": collective,
        "shutdown": shutdown,
        "forced_shutdown": forced_shutdown,
        "shared_memory_link_count": shm_count,
        "restore_start_to_response_seconds": (response_ns - started_ns) / 1e9,
    }
    base._atomic_json(args.artifact_root / "restore-result.json", report)
    return report


def main() -> int:
    args = _parser().parse_args()
    if args.world_size < 2 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be within a world of at least two ranks")
    if args.ghost_limit < 64 * 1024**2:
        raise ValueError("ghost-limit must accommodate NCCL's shared-memory segment")
    if args.timeout <= 0 or args.timeout > 3600:
        raise ValueError("timeout must be in (0, 3600]")
    checkpoint_arguments = (
        args.nccl_checkpoint_shim,
        args.nccl_library,
        args.nccl_checkpoint_coordinator_path,
    )
    if any(value is None for value in checkpoint_arguments) and any(
        value is not None for value in checkpoint_arguments
    ):
        raise ValueError(
            "--nccl-checkpoint-shim, --nccl-library, and "
            "--nccl-checkpoint-coordinator-path are required together"
        )
    if args.require_nccl_ib_reset and args.nccl_checkpoint_shim is None:
        raise ValueError("--require-nccl-ib-reset requires the NCCL checkpoint runtime")
    if args.in_place_activation and not args.in_place_provider:
        raise ValueError("--in-place-activation requires --in-place-provider")
    if args.control_store == "coldsnap" and args.control_store_address_path is None:
        raise ValueError("ColdSnap control store requires --control-store-address-path")
    if args.control_store == "tcp" and args.control_store_address_path is not None:
        raise ValueError("--control-store-address-path only applies to ColdSnap")
    if args.control_store_address_path is not None:
        if not args.control_store_address_path.is_file():
            raise FileNotFoundError(
                f"control-store address file is missing: {args.control_store_address_path}"
            )
    if args.mode == "restore" and args.nccl_checkpoint_coordinator_path is not None:
        if not args.nccl_checkpoint_coordinator_path.is_file():
            raise FileNotFoundError(
                "NCCL checkpoint coordinator file is missing: "
                f"{args.nccl_checkpoint_coordinator_path}"
            )
    try:
        report = capture(args) if args.mode == "capture" else restore(args)
    except BaseException as error:
        report = {
            "format": 1,
            "kind": "nccl-cuda-criu-error",
            "rank": args.rank,
            "error": f"{type(error).__name__}: {error}",
        }
        args.artifact_root.mkdir(parents=True, exist_ok=True)
        base._atomic_json(args.artifact_root / "controller-error.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1
    console = report
    if args.compact_output and report.get("kind") == "nccl-cuda-criu-restore":
        console = {
            "kind": report["kind"],
            "rank": args.rank,
            "criu_restore_seconds": report["criu_restore_seconds"],
            "restore_start_to_response_seconds": report["restore_start_to_response_seconds"],
            "collective": report["collective"]["collective"],
        }
    print(json.dumps(console, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

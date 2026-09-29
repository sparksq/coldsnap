#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Run bounded local NCCL lifecycle/API regressions without CRIU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nccl-source", type=Path, required=True, help="Patched, built NCCL source tree")
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--coordinator", type=Path, default=ROOT / "bin/coldsnap-coordinator")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("recreate", "inplace", "registration", "progress-recreate", "progress-inplace", "window-0", "window-1", "window-2", "window-4", "tls-recreate", "tls-inplace"), required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    window = args.mode.startswith("window-")
    ranks = 1 if window else 2
    source = args.nccl_source.resolve()
    shim = source / "contrib/nccl_checkpoint/build/lib"
    runtime = source / "build/lib"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps({"passed": False, "mode": args.mode}) + "\n")
    with tempfile.TemporaryDirectory(prefix="coldsnap-nccl-lifecycle-") as directory:
        work = Path(directory)
        target = work / "target"
        subprocess.run([
            "g++", "-std=c++17", str(ROOT / "test/native/nccl_provider_lifecycle.cc"),
            "-I" + str(ROOT / "native/nccl/abi"), "-I" + str(source / "build/include"),
            "-I" + str(args.cuda / "include"), "-o", str(target),
            "-Wl,--no-as-needed", "-L" + str(shim), "-lnccl-checkpoint-shim",
            "-L" + str(runtime), "-lnccl", "-L" + str(args.cuda / "lib64"), "-lcudart",
            "-Wl,-rpath," + ":".join(map(str, (shim, runtime, args.cuda / "lib64"))),
        ], check=True, timeout=args.timeout)
        children: list[subprocess.Popen] = []
        logs = []
        coord = None
        try:
            with (args.output / "coordinator.log").open("wb") as coord_log:
                coord = subprocess.Popen([
                    str(args.coordinator.resolve()), "-listen", "127.0.0.1:0",
                    "-advertise-host", "127.0.0.1", "-endpoint-file", str(work / "endpoint"),
                    "-scope", f"lifecycle-{os.getpid()}",
                ], stdout=coord_log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + args.timeout
                while not (work / "endpoint").exists():
                    if coord.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("coordinator did not become ready")
                    time.sleep(0.05)
                for rank in range(ranks):
                    env = os.environ.copy()
                    env.update(
                        NCCL_IB_DISABLE="1", NCCL_IB_RELEASE_ON_FINALIZE="1", NCCL_NET="Socket",
                        NCCL_SOCKET_IFNAME="=lo", NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1",
                        NCCL_CUMEM_HOST_ENABLE="0", NCCL_NVLS_ENABLE="0", NCCL_COLLNET_ENABLE="0",
                        NCCL_HOSTID=f"local-lifecycle-{rank}", NCCL_DEBUG="WARN",
                        NCCL_CHECKPOINT_TERMINATION="abort-sync",
                        NCCL_CHECKPOINT_COORDINATOR_PATH=str(work / "endpoint"),
                        NCCL_CHECKPOINT_COORDINATOR_TIMEOUT="15",
                        COLDSNAP_NCCL_IN_PLACE_MODE="net-reconnect-v1",
                        NCCL_PROGRESS_COUNTERS="1" if args.mode.startswith("progress-") else "0",
                        NCCL_PROGRESS_COUNTER_MONITOR_POLL_MS="50",
                        NCCL_RAS_ENABLE="1",
                    )
                    if window:
                        env.update(NCCL_CUMEM_ENABLE="1", NCCL_NUM_RMA_CTX="0")
                    # Load only the selected pair; the executable links shim before runtime.
                    env.pop("LD_PRELOAD", None)
                    env["LD_LIBRARY_PATH"] = ":".join(map(str, (shim, runtime, args.cuda / "lib64"))) + ":" + env.get("LD_LIBRARY_PATH", "")
                    log = (args.output / f"rank-{rank}.log").open("wb")
                    logs.append(log)
                    children.append(subprocess.Popen([str(target), str(work), str(rank), args.mode], env=env, stdout=log, stderr=subprocess.STDOUT))
                while True:
                    # Poll every rank: short-circuiting can hide a failed later rank.
                    returncodes = [child.poll() for child in children]
                    if any(code not in (None, 0) for code in returncodes):
                        raise RuntimeError("rank failed; see rank logs")
                    if all(code == 0 for code in returncodes):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("NCCL lifecycle probe timed out")
                    time.sleep(0.05)
                if window:
                    records = [json.loads(line) for line in (args.output / "rank-0.log").read_text().splitlines()
                               if line.startswith("{")]
                    inventories = [row for row in records if row.get("kind") == "coldsnap-nccl-resource-inventory"]
                    # Synthetic handles also represent successful no-op registrations.
                    # Require actual runtime windows before and after reconstruction.
                    if len(inventories) != 8 or any(
                        not row["complete"] or len(row["communicators"]) != 1 for row in inventories
                    ):
                        raise RuntimeError("incomplete window lifecycle evidence")
                    for row in inventories[:-1]:
                        comm = row["communicators"][0]
                        if comm["windows"] != 1 or comm["runtime_windows"] < 1:
                            raise RuntimeError("runtime did not create a real local window")
                    final = inventories[-1]["communicators"][0]
                    before_deregister = inventories[-2]["communicators"][0]
                    # Symmetric kernels can own another internal window until
                    # communicator destruction. Deregistration removes our one.
                    if (final["windows"] != 0 or final["registration_cache_entries"] != 0 or
                            final["runtime_windows"] != before_deregister["runtime_windows"] - 1):
                        raise RuntimeError("user window deregistration left live resources")
                result = {
                    "passed": True, "mode": args.mode, "ranks": ranks, "gpus": 1,
                    "communicators_per_rank": 1 if window else 2, "criu": False, "cross_host": False,
                    "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                               (runtime / "libnccl.so.2", shim / "libnccl-checkpoint-shim.so")},
                }
                (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait()
            if coord is not None:
                coord.terminate()
                try:
                    coord.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    coord.kill()
                    coord.wait()
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()

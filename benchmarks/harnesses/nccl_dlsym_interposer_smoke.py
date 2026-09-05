#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Smoke-test InstantTensor's cached NCCL calls through the checkpoint shim."""

from __future__ import annotations

import ctypes
import json
import os

def main() -> int:
    real_library = ctypes.CDLL(os.environ["COLDSNAP_NCCL_REAL_LIBRARY_PATH"])
    for name in (
        "ncclAllGather",
        "ncclCommCount",
        "ncclCommUserRank",
        "ncclCommInitRank",
        "ncclCommDestroy",
        "ncclAllReduce",
    ):
        getattr(real_library, name)
    path = os.environ["COLDSNAP_NCCL_DLSYM_BRIDGE_PATH"]
    bridge = ctypes.CDLL(path)
    route_mask = bridge.coldsnapNcclDlsymRouteMask
    route_mask.argtypes = []
    route_mask.restype = ctypes.c_uint
    actual = route_mask()
    route_count = bridge.coldsnapNcclDlsymRouteCount
    route_count.argtypes = []
    route_count.restype = ctypes.c_uint
    actual_count = route_count()
    expected = 0b111111
    if actual != expected:
        raise RuntimeError(
            f"InstantTensor NCCL dlsym route mask is {actual:#05b}, "
            f"expected {expected:#05b}"
        )
    if actual_count < 6:
        raise RuntimeError(
            f"NCCL dlsym route count is {actual_count}, expected at least 6"
        )
    print(
        json.dumps(
            {
                "route_count": actual_count,
                "route_mask": actual,
                "expected": expected,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

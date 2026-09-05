#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Smoke-test the synthetic InstantTensor NCCL communicator compatibility ABI."""

from __future__ import annotations

import json

import torch
import torch.distributed as dist

from coldsnap_vllm_nccl_checkpoint import (
    NcclCheckpointRuntime,
    install_instanttensor_nccl_unwrap,
)


def main() -> int:
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29732",
        rank=0,
        world_size=1,
    )
    try:
        value = torch.ones(1, device="cuda")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        synthetic = dist.group.WORLD._get_backend(
            torch.device("cuda:0")
        )._comm_ptr()
        runtime = NcclCheckpointRuntime(require_ib_reset=True)
        real = runtime.unwrap_communicator(synthetic)
        installed = install_instanttensor_nccl_unwrap()
        if synthetic <= 0 or real <= 0 or synthetic == real:
            raise RuntimeError(
                f"invalid communicator translation: synthetic={synthetic} real={real}"
            )
        print(
            json.dumps(
                {
                    "synthetic": synthetic,
                    "real": real,
                    "instanttensor_hook_installed": installed,
                    "version": runtime.version,
                },
                sort_keys=True,
            )
        )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

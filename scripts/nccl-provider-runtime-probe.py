#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Probe a preloaded ColdSnap NCCL provider without creating a communicator."""

from __future__ import annotations

import argparse
import json

from coldsnap_vllm_nccl_checkpoint import NcclCheckpointError, NcclCheckpointRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-ib-reset", action="store_true")
    parser.add_argument("--require-network-reset", action="store_true")
    parser.add_argument("--required-capability", action="append", default=[])
    parser.add_argument("--expect-error-substring")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        runtime = NcclCheckpointRuntime(
            require_ib_reset=args.require_ib_reset,
            require_network_reset=args.require_network_reset,
            required_capabilities=set(args.required_capability),
        )
    except NcclCheckpointError as error:
        if args.expect_error_substring is None:
            raise
        message = str(error)
        if args.expect_error_substring not in message:
            raise RuntimeError(
                f"provider failed with unexpected error: {message}"
            ) from error
        print(json.dumps({"accepted": False, "error": message}, sort_keys=True))
        return 0
    if args.expect_error_substring is not None:
        raise RuntimeError("provider was accepted but rejection was expected")
    print(json.dumps({"accepted": True, "version": runtime.version}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

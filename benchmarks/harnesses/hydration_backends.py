#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Calibrate native buffered, direct, and GDS file-to-GPU hydration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations" / "core"))
sys.path.insert(0, str(ROOT / "integrations" / "vllm"))

from coldsnap_core.hydration import HydrationExtent, NativeHydrator  # noqa: E402


MIB = 1024**2


def _create_blob(path: Path, size: int, value: int) -> None:
    block = bytes([value]) * min(8 * MIB, size)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(fd, 0, size)
        offset = 0
        while offset < size:
            view = block[: min(len(block), size - offset)]
            written = os.pwrite(fd, view, offset)
            if written <= 0:
                raise OSError("pwrite returned no data")
            offset += written
        os.fdatasync(fd)
    finally:
        os.close(fd)


def _drop_cache(path: Path) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--blob", type=Path)
    parser.add_argument("--bytes", type=int, default=512 * MIB)
    parser.add_argument("--chunk-bytes", type=int, default=64 * MIB)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--backends", nargs="+", default=["buffered", "direct", "gds"]
    )
    parser.add_argument("--pattern", type=lambda value: int(value, 0), default=0xA5)
    parser.add_argument("--register-device-buffers", action="store_true")
    parser.add_argument("--max-device-fraction", type=float, default=0.2)
    parser.add_argument("--output-json", type=Path)
    return parser


def _run(args: argparse.Namespace, blob: Path, validate: bool) -> dict[str, Any]:
    import torch

    if args.bytes <= 0 or args.bytes % 4096:
        raise ValueError("--bytes must be positive and 4096-byte aligned")
    if args.chunk_bytes <= 0 or args.chunk_bytes % 4096:
        raise ValueError("--chunk-bytes must be positive and 4096-byte aligned")
    if not 0 < args.max_device_fraction <= 1:
        raise ValueError("--max-device-fraction must be in (0, 1]")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    limit = int(total_bytes * args.max_device_fraction)
    if args.bytes > min(free_bytes, limit):
        raise RuntimeError(
            f"requested {args.bytes} device bytes; free={free_bytes}, "
            f"fraction_limit={limit}"
        )
    destination = torch.empty(args.bytes, dtype=torch.uint8, device="cuda")
    hydrator = NativeHydrator(args.library)
    extent = HydrationExtent(0, destination.data_ptr(), args.bytes)
    records: list[dict[str, Any]] = []
    for backend in args.backends:
        if not hydrator.available(backend):
            records.append({"backend": backend, "available": False})
            continue
        for repetition in range(args.repetitions):
            _drop_cache(blob)
            try:
                metrics = hydrator.hydrate(
                    blob,
                    [extent],
                    backend=backend,
                    chunk_bytes=args.chunk_bytes,
                    queue_depth=args.queue_depth,
                    preverified=True,
                    register_device_buffers=args.register_device_buffers,
                )
                valid = None
                if validate:
                    invalid = torch.count_nonzero(destination != args.pattern).item()
                    valid = invalid == 0
                    if not valid:
                        raise RuntimeError(
                            f"hydrated tensor contains {invalid} invalid bytes"
                        )
                record = metrics.as_dict()
                record.update(
                    {
                        "available": True,
                        "repetition": repetition,
                        "valid": valid,
                        "effective_gib_s": (
                            metrics.bytes / 1024**3 / metrics.total_s
                        ),
                    }
                )
            except Exception as error:
                record = {
                    "backend": backend,
                    "available": True,
                    "repetition": repetition,
                    "error": f"{type(error).__name__}: {error}",
                }
            records.append(record)
    return {
        "blob": str(blob),
        "bytes": args.bytes,
        "chunk_bytes": args.chunk_bytes,
        "queue_depth": args.queue_depth,
        "repetitions": args.repetitions,
        "register_device_buffers": args.register_device_buffers,
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
        "records": records,
    }


def main() -> int:
    args = _parser().parse_args()
    if not 0 <= args.pattern <= 255:
        raise ValueError("--pattern must fit in one byte")
    if args.blob is not None:
        if args.blob.stat().st_size < args.bytes:
            raise ValueError("--blob is smaller than --bytes")
        result = _run(args, args.blob, False)
    else:
        with tempfile.TemporaryDirectory(prefix="coldsnap-hydration-") as directory:
            blob = Path(directory) / "calibration.blob"
            _create_blob(blob, args.bytes, args.pattern)
            result = _run(args, blob, True)
            result["blob"] = "generated-temporary-file"
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

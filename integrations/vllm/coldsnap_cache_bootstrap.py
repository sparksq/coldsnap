# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Seed an immutable derived cache into a fresh worker-local directory."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def seed_cache(source: Path, destination: Path) -> float:
    if not source.is_dir():
        raise RuntimeError(f"derived cache source is not a directory: {source}")
    if destination.exists():
        raise RuntimeError(f"derived cache destination already exists: {destination}")
    started = time.monotonic()
    shutil.copytree(source, destination, symlinks=False)
    return time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    seconds = seed_cache(args.source, args.destination)
    print(f"COLDSNAP derived startup cache copy took {seconds:.6f} seconds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

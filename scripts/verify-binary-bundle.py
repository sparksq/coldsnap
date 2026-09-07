#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Verify a copied OCI binary bundle before publishing its release tag."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


BINARIES = ("coldsnap", "coldsnap-vllm-adapter", "coldsnap-sglang-adapter", "coldsnap-criu-rpc")
ELF_MACHINES = {"amd64": 62, "arm64": 183}
MACHO_CPUS = {"amd64": 0x01000007, "arm64": 0x0100000C}


def verify(root: Path, *, version: str, commit: str, arch: str, os_name: str = "linux", execute: bool = True) -> None:
    root = root.resolve()
    if os_name not in {"linux", "darwin"} or arch not in ELF_MACHINES:
        raise ValueError("unsupported binary-bundle platform")
    binaries = BINARIES if os_name == "linux" else BINARIES[:-1]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("expected a full release commit")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("manifest.json must be a regular file")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "format": 1,
        "kind": "coldsnap-controller-binary-bundle",
        "version": version,
        "commit": commit,
        "platform": f"{os_name}-{arch}",
    }
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("bundle identity does not match the source release")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(binaries):
        raise ValueError("bundle must contain checksums for exactly %s executables" % ("four" if os_name == "linux" else "three"))
    for name in binaries:
        path = root / name
        digest = hashes[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name} must be a regular file")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid checksum for {name}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"checksum mismatch for {name}")
        if os_name == "linux" and (
            data[:6] != b"\x7fELF\x02\x01"
            or len(data) < 20
            or int.from_bytes(data[18:20], "little") != ELF_MACHINES[arch]
        ):
            raise ValueError(f"{name} is not a linux/{arch} ELF executable")
        if os_name == "darwin" and (
            len(data) < 32 or data[:4] != b"\xcf\xfa\xed\xfe"
            or int.from_bytes(data[4:8], "little") != MACHO_CPUS[arch]
            or int.from_bytes(data[12:16], "little") != 2
        ):
            raise ValueError(f"{name} is not a darwin/{arch} Mach-O executable")
    if not execute:
        return
    result = subprocess.run(
        [str(root / "coldsnap"), "version", "--json"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    identity = json.loads(result.stdout)
    if identity != {"version": version, "commit": commit}:
        raise ValueError("controller executable does not match the bundle identity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--arch", choices=sorted(ELF_MACHINES), required=True)
    parser.add_argument("--os", dest="os_name", choices=("linux", "darwin"), default="linux")
    parser.add_argument("--no-execute", action="store_true", help="static verification only; native execution must be tested separately")
    args = parser.parse_args()
    try:
        verify(args.root, version=args.version, commit=args.commit, arch=args.arch, os_name=args.os_name, execute=not args.no_execute)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"binary-bundle: {error}", file=sys.stderr)
        return 1
    print(f"binary-bundle: verified {args.version} at {args.commit} for {args.os_name}/{args.arch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

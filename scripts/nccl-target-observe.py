#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Observe the exact NCCL/CUDA userspace identity in a disposable base image."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import subprocess


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION = re.compile(rb"GLIBCXX_([0-9]+\.[0-9]+(?:\.[0-9]+)?)")


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image-digest", required=True)
    parser.add_argument("--transport", required=True)
    parser.add_argument("--qualification-policy", default="production")
    parser.add_argument(
        "--nccl-policy",
        choices=("exact", "match-or-latest-qualified"),
        default="exact",
        help="provider selection policy; direct ColdSnap use remains exact by default",
    )
    parser.add_argument("--provider-abi-major", type=int, default=1)
    parser.add_argument("--provider-abi-minor", type=int, default=0)
    parser.add_argument("--dlsym-bridge-abi", type=int, default=1)
    parser.add_argument(
        "--capability",
        action="append",
        default=[
            "full-network-reset",
            "ib-roce-device-release",
            "synchronous-termination",
        ],
    )
    return parser


def _check_token(value: str, field: str) -> str:
    if re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", value) is None:
        raise RuntimeError(f"{field} is invalid")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dladdr(function: object) -> Path:
    libdl = ctypes.CDLL(None)
    dladdr = libdl.dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    dladdr.restype = ctypes.c_int
    info = _DlInfo()
    address = ctypes.cast(function, ctypes.c_void_p)
    if dladdr(address, ctypes.byref(info)) == 0 or info.dli_fname is None:
        raise RuntimeError("dladdr could not resolve a loaded library")
    path = Path(os.fsdecode(info.dli_fname)).resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"loaded library is not a regular file: {path}")
    return path


def _readelf(path: Path, option: str) -> str:
    try:
        return subprocess.run(
            ["readelf", option, str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"cannot inspect ELF {path}: {error}") from error


def _elf_identity(path: Path) -> tuple[str, str]:
    dynamic = _readelf(path, "-d")
    notes = _readelf(path, "-n")
    soname = re.findall(r"\(SONAME\).*\[([^]]+)\]", dynamic)
    build_id = re.findall(r"Build ID:\s*([0-9a-f]+)", notes)
    if len(soname) != 1 or len(build_id) != 1:
        raise RuntimeError(f"ELF {path} lacks one SONAME/build ID")
    return soname[0], build_id[0]


def _elf_soname(path: Path) -> str:
    dynamic = _readelf(path, "-d")
    soname = re.findall(r"\(SONAME\).*\[([^]]+)\]", dynamic)
    if len(soname) != 1:
        raise RuntimeError(f"ELF {path} lacks one SONAME")
    return soname[0]


def _nccl() -> tuple[int, Path]:
    library = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_GLOBAL)
    get_version = library.ncclGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    result = get_version(ctypes.byref(version))
    if result != 0 or version.value <= 0:
        raise RuntimeError(f"ncclGetVersion failed with ncclResult_t={result}")
    return version.value, _dladdr(get_version)


def _cuda() -> tuple[str, str]:
    library = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    get_version = library.cudaRuntimeGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    result = get_version(ctypes.byref(version))
    if result != 0 or version.value <= 0:
        raise RuntimeError(f"cudaRuntimeGetVersion failed with cudaError_t={result}")
    path = _dladdr(get_version)
    soname = _elf_soname(path)
    major = version.value // 1000
    minor = (version.value % 1000) // 10
    return f"{major}.{minor}", soname


def _glibc() -> str:
    library = ctypes.CDLL(None)
    get_version = library.gnu_get_libc_version
    get_version.argtypes = []
    get_version.restype = ctypes.c_char_p
    value = get_version()
    if value is None:
        raise RuntimeError("glibc version is unavailable")
    version = value.decode("ascii")
    if re.fullmatch(r"[0-9]+\.[0-9]+", version) is None:
        raise RuntimeError("glibc version is invalid")
    return version


def _libstdcxx_floor() -> str:
    ctypes.CDLL("libstdc++.so.6", mode=ctypes.RTLD_GLOBAL)
    candidates: set[Path] = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "libstdc++.so.6" in fields[5]:
            candidates.add(Path(fields[5]).resolve(strict=True))
    if len(candidates) != 1:
        raise RuntimeError("cannot identify one loaded libstdc++.so.6")
    versions = {
        tuple(int(component) for component in match.decode().split("."))
        for match in _VERSION.findall(next(iter(candidates)).read_bytes())
    }
    if not versions:
        raise RuntimeError("libstdc++ exports no GLIBCXX ABI versions")
    return ".".join(str(component) for component in max(versions))


def main() -> None:
    args = _parser().parse_args()
    if _DIGEST.fullmatch(args.base_image_digest) is None:
        raise RuntimeError("base image must be identified by a sha256 digest")
    if args.provider_abi_major <= 0 or args.provider_abi_minor < 0 or args.dlsym_bridge_abi <= 0:
        raise RuntimeError("provider/bridge ABI requirements are invalid")
    architecture = platform.machine()
    architecture = {"arm64": "aarch64", "amd64": "x86_64"}.get(architecture, architecture)
    if architecture not in {"aarch64", "x86_64"}:
        raise RuntimeError(f"unsupported architecture {architecture!r}")
    nccl_version, nccl_path = _nccl()
    nccl_soname, nccl_build_id = _elf_identity(nccl_path)
    cuda_userspace, cuda_soname = _cuda()
    glibc = _glibc()
    libstdcxx = _libstdcxx_floor()
    release = f"{nccl_version // 10000}.{(nccl_version // 100) % 100}.{nccl_version % 100}"
    cuda_platform = cuda_userspace.removesuffix(".0")
    target = {
        "format": 1,
        "kind": "coldsnap-nccl-provider-target",
        "base_image_digest": args.base_image_digest,
        "platform_key": f"linux-{architecture}-cuda{cuda_platform}-glibc{glibc}",
        "nccl": {
            "version": nccl_version,
            "release": release,
            "path": str(nccl_path),
            "soname": nccl_soname,
            "build_id": nccl_build_id,
            "sha256": _sha256(nccl_path),
        },
        "nccl_policy": args.nccl_policy,
        "requirements": {
            "architecture": architecture,
            "cuda_userspace": cuda_userspace,
            "cuda_runtime_soname": cuda_soname,
            "glibc_min": glibc,
            "libstdcxx_abi_min": libstdcxx,
        },
        "required_provider_abi": {
            "major": args.provider_abi_major,
            "minor": args.provider_abi_minor,
        },
        "required_capabilities": sorted(
            {_check_token(value, "capability") for value in args.capability}
        ),
        "required_dlsym_bridge_abi": args.dlsym_bridge_abi,
        "qualification_policy": _check_token(args.qualification_policy, "qualification policy"),
        "transport": _check_token(args.transport, "transport"),
    }
    print(json.dumps(target, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

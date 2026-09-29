# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Apply release-owned patches to exact upstream sources when available locally."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = os.environ.get("NCCL_SOURCE_ROOT")


@unittest.skipUnless(SOURCE_ROOT, "set NCCL_SOURCE_ROOT to an upstream NCCL Git checkout")
class NcclReleaseSourceTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ is required for the CPU IB cache regression")
    def test_ib_capability_cache_follows_device_lifetime(self) -> None:
        release = ROOT / "native/nccl/releases/2.32.3-1"
        lock = json.loads((release / "source.lock").read_text())
        relative = "src/transport/net_ib/gdr.cc"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            production = source / relative
            production.parent.mkdir(parents=True)
            production.write_bytes(subprocess.check_output(
                ["git", "show", f"{lock['commit']}:{relative}"], cwd=SOURCE_ROOT,
            ))
            subprocess.run([
                "git", "apply", f"--include={relative}",
                str(release / "patches/0010-checkpoint-ib-capability-cache.patch"),
            ], cwd=source, check=True, capture_output=True)
            fixture = ROOT / "test/native/nccl_ib_capability_cache"
            for name in ("common.h", "test.cc"):
                shutil.copy2(fixture / name, production.parent / name)
            binary = source / "test-cache"
            subprocess.run([
                "g++", "-std=c++17", "-pthread", str(production.parent / "test.cc"),
                "-o", str(binary),
            ], check=True, capture_output=True, timeout=30)
            subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)

    @unittest.skipUnless(shutil.which("g++"), "g++ is required for the CPU window regression")
    def test_window_shim_lifetime_and_cft_endpoint_detection(self) -> None:
        cuda_include = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "include"
        if not (cuda_include / "cuda_runtime.h").is_file():
            self.skipTest("CUDA headers are required; no GPU or CUDA runtime is used")
        release = ROOT / "native/nccl/releases/2.32.3-1"
        lock = json.loads((release / "source.lock").read_text())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            archive = source / "source.tar"
            with archive.open("wb") as output:
                subprocess.run([
                    "git", "archive", lock["commit"], "contrib/nccl_checkpoint",
                    "src/include", "src/nccl.h.in", "src/device",
                ], cwd=SOURCE_ROOT, stdout=output, check=True)
            with tarfile.open(archive) as payload:
                payload.extractall(source, filter="data")
            subprocess.run([
                "git", "apply", "--include=contrib/nccl_checkpoint/shim.cc",
                str(release / "patches/0011-checkpoint-window-bookkeeping.patch"),
            ], cwd=source, check=True, capture_output=True)
            include = source / "src/include"
            header = (source / "src/nccl.h.in").read_text()
            for name, value in {"Major": "2", "Minor": "32", "Patch": "3", "Suffix": "", "Version": "23203"}.items():
                header = header.replace("${nccl:" + name + "}", value)
            (include / "nccl.h").write_text(header)
            shim = source / "contrib/nccl_checkpoint"
            (shim / "shim_auto.cc").write_bytes(subprocess.check_output([
                "python3", str(shim / "gen_shim.py"), str(include / "nccl.h"),
                str(include / "nccl_device/host.h"),
            ], stderr=subprocess.PIPE))
            fixture = ROOT / "test/native/nccl_window_bookkeeping"
            flags = [
                "g++", "-std=c++17", "-O1", "-ffunction-sections", "-fdata-sections",
                f"-I{include}", f"-I{cuda_include}", f"-I{shim}",
            ]
            backend = source / "backend.so"
            subprocess.run(flags + [
                "-shared", "-fPIC", str(fixture / "backend.cc"), "-o", str(backend),
            ], check=True, capture_output=True, timeout=60)
            binary = source / "test-window"
            # Compile the real shim and handle map, substituting only its external
            # runtime calls. Unused functions are discarded at link time.
            subprocess.run(flags + [
                str(shim / "shim.cc"), str(shim / "shim_core.cc"), str(fixture / "test.cc"),
                str(backend), "-Wl,--gc-sections", "-ldl", "-o", str(binary),
            ], check=True, capture_output=True, timeout=60)
            subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
            cft_binary = source / "test-cft-endpoints"
            subprocess.run(flags + [
                "-DNCCL_OS_LINUX", f"-I{include / 'plugin'}", f"-I{release}",
                f"-I{ROOT / 'native/nccl/abi'}", str(fixture / "cft-test.cc"),
                "-Wl,--gc-sections", "-o", str(cft_binary),
            ], check=True, capture_output=True, timeout=60)
            subprocess.run([str(cft_binary)], check=True, capture_output=True, timeout=10)

    @unittest.skipUnless(shutil.which("g++"), "g++ is required for the CPU TLS regression")
    def test_tls_connections_and_restore_entropy(self) -> None:
        cuda_include = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "include"
        if not (cuda_include / "cuda_runtime.h").is_file() or not Path("/usr/include/openssl/ssl.h").is_file():
            self.skipTest("CUDA and OpenSSL 3 headers are required; no GPU is used")
        release = ROOT / "native/nccl/releases/2.32.3-1"
        lock = json.loads((release / "source.lock").read_text())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            archive = source / "source.tar"
            with archive.open("wb") as output:
                subprocess.run([
                    "git", "archive", lock["commit"], "src/misc/crypt.cc", "contrib/nccl_checkpoint",
                    "src/include", "src/nccl.h.in", "src/device",
                ], cwd=SOURCE_ROOT, stdout=output, check=True)
            with tarfile.open(archive) as payload:
                payload.extractall(source, filter="data")
            subprocess.run([
                "git", "apply", "--include=src/misc/crypt.cc",
                str(release / "patches/0012-checkpoint-tls-lifecycle.patch"),
            ], cwd=source, check=True, capture_output=True)
            include = source / "src/include"
            header = (source / "src/nccl.h.in").read_text()
            for name, value in {"Major": "2", "Minor": "32", "Patch": "3", "Suffix": "", "Version": "23203"}.items():
                header = header.replace("${nccl:" + name + "}", value)
            (include / "nccl.h").write_text(header)
            for enabled in (False, True):
                with self.subTest(tls_backend=enabled):
                    binary = source / ("test-tls" if enabled else "test-no-tls")
                    flags = ["-DNCCL_TLS_BACKEND_OPENSSL3=1", "-lssl", "-lcrypto"] if enabled else []
                    subprocess.run([
                        "g++", "-std=c++17", "-O1", "-ffunction-sections", "-fdata-sections", "-DNCCL_OS_LINUX",
                        f"-I{source / 'src'}", f"-I{include}", f"-I{include / 'plugin'}", f"-I{cuda_include}",
                        str(ROOT / "test/native/nccl_tls_lifecycle/test.cc"), "-Wl,--gc-sections", "-ldl",
                        *flags, "-o", str(binary),
                    ], check=True, capture_output=True, timeout=60)
                    subprocess.run([str(binary)], check=True, capture_output=True, timeout=15)
            program = """
import ctypes, json
pointer = ctypes.c_void_p()
query = ctypes.CDLL(None).coldsnapNcclProviderQuery
query.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
query.restype = ctypes.c_int
result = query(1, ctypes.byref(pointer))
print(json.dumps({"result": result, "output_present": bool(pointer.value)}))
"""
            for expected in (0, 1):
                provider = source / f"provider-{expected}.so"
                flags = ["-DNCCL_TLS_BACKEND_OPENSSL3=1"] if expected else []
                subprocess.run([
                    "g++", "-std=c++17", "-shared", "-fPIC", *flags,
                    f"-I{include}", f"-I{cuda_include}", f"-I{source / 'contrib/nccl_checkpoint'}",
                    f"-I{ROOT / 'native/nccl/abi'}", str(release / "coldsnap_provider.cc"),
                    "-ldl", "-o", str(provider),
                ], check=True, capture_output=True, timeout=30)
                for compiled in (0, 1):
                    with self.subTest(provider_tls=expected, runtime_tls=compiled):
                        backend = source / f"backend-{compiled}.so"
                        subprocess.run([
                            "g++", "-shared", "-fPIC", f"-DTEST_TLS={compiled}",
                            str(ROOT / "test/native/nccl_tls_lifecycle/provider-backend.cc"),
                            "-o", str(backend),
                        ], check=True, capture_output=True, timeout=30)
                        env = os.environ.copy()
                        env["LD_PRELOAD"] = f"{provider}:{backend}"
                        result = subprocess.run([
                            sys.executable, "-c", program,
                        ], env=env, capture_output=True, text=True, timeout=10)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(json.loads(result.stdout), {
                            "result": 0 if expected == compiled else 5,
                            "output_present": expected == compiled,
                        })

    def test_locked_sources_accept_complete_patch_series_without_fuzz(self) -> None:
        for release in sorted((ROOT / "native/nccl/releases").iterdir()):
            if not (release / "recipe.json").exists():
                continue
            with self.subTest(release=release.name), tempfile.TemporaryDirectory() as directory:
                source = Path(directory)
                lock = json.loads((release / "source.lock").read_text())
                revision = subprocess.check_output(
                    ["git", "rev-parse", f"{lock['signed_tag']}^{{commit}}"],
                    cwd=SOURCE_ROOT,
                    text=True,
                ).strip()
                self.assertEqual(revision, lock["commit"])
                archive = source / "upstream.tar"
                with archive.open("wb") as output:
                    subprocess.run(
                        ["git", "archive", lock["commit"]],
                        cwd=SOURCE_ROOT,
                        stdout=output,
                        check=True,
                    )
                with tarfile.open(archive) as payload:
                    payload.extractall(source, filter="data")
                archive.unlink()
                recipe = json.loads((release / "recipe.json").read_text())
                series = (release / "patches/series").read_text().splitlines()
                self.assertEqual(
                    series, [Path(item["path"]).name for item in recipe["inputs"]["patches"]]
                )
                for patch in series:
                    result = subprocess.run(
                        [
                            "patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch",
                            "-p1", "-i", str(release / "patches" / patch),
                        ],
                        cwd=source,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(list(source.rglob("*.rej")), [])


if __name__ == "__main__":
    unittest.main()

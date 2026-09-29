# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Exercise real private-loader routing with small CPU-only shared libraries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "linux" and shutil.which("cc"), "Linux C compiler required")
class NcclDlsymBridgeTest(unittest.TestCase):
    def test_private_lookup_uses_selected_runtime_for_unwrapped_apis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)

            def compile_library(name: str, source: str | Path) -> Path:
                if isinstance(source, str):
                    path = work / f"{name}.c"
                    path.write_text(source)
                else:
                    path = source
                library = work / f"{name}.so"
                subprocess.run([
                    "cc", "-shared", "-fPIC", str(path), "-ldl", "-o", str(library),
                ], check=True, capture_output=True, timeout=30)
                return library

            bridge = compile_library("bridge", ROOT / "native/coldsnap_nccl_dlsym.c")
            shim = compile_library("shim", "int ncclCommCount(void) { return 101; }\n")
            runtime = compile_library("runtime", """
                int ncclGetVersion(void) { return 23203; }
                int ncclMemAlloc(void) { return 232; }
                int ordinaryLibraryCall(void) { return 232; }
            """)
            private = compile_library("private", """
                int ncclCommCount(void) { return 31; }
                int ncclGetVersion(void) { return 23102; }
                int ncclMemAlloc(void) { return 231; }
                int ordinaryLibraryCall(void) { return 231; }
                int ncclPrivateOnly(void) { return 77; }
            """)
            lookup = compile_library("lookup", """
                #include <dlfcn.h>
                int lookup(void *handle, const char *name, int *error) {
                    dlerror();
                    int (*function)(void) = (int (*)(void))dlsym(handle, name);
                    *error = dlerror() != 0;
                    return function ? function() : -999;
                }
            """)
            program = """
import ctypes, json, sys
private = ctypes.CDLL(sys.argv[1])
lookup = ctypes.CDLL(sys.argv[2]).lookup
lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
lookup.restype = ctypes.c_int
values = {}
for name in ('ncclCommCount', 'ncclGetVersion', 'ncclMemAlloc',
             'ordinaryLibraryCall', 'ncclPrivateOnly'):
    error = ctypes.c_int()
    value = lookup(private._handle, name.encode(), ctypes.byref(error))
    assert error.value == 0, (name, 'successful lookup left a dlerror')
    assert getattr(private, name)() == value
    values[name] = value
error = ctypes.c_int()
assert lookup(private._handle, b'ncclMissing', ctypes.byref(error)) == -999
assert error.value == 1, 'failed lookup did not report dlerror'
print(json.dumps(values))
"""
            for enabled in (False, True):
                with self.subTest(shim_enabled=enabled):
                    environment = os.environ.copy()
                    environment["LD_PRELOAD"] = ":".join(map(str, (bridge, shim, runtime)))
                    environment.pop("COLDSNAP_NCCL_CHECKPOINT_SHIM_PATH", None)
                    if enabled:
                        environment["COLDSNAP_NCCL_CHECKPOINT_SHIM_PATH"] = str(shim)
                    result = subprocess.run([
                        sys.executable, "-c", program, str(private), str(lookup),
                    ], env=environment, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    actual = json.loads(result.stdout)
                    self.assertEqual(actual, {
                        "ncclCommCount": 101 if enabled else 31,
                        "ncclGetVersion": 23203 if enabled else 23102,
                        "ncclMemAlloc": 232 if enabled else 231,
                        "ordinaryLibraryCall": 231,
                        "ncclPrivateOnly": 77,
                    })


if __name__ == "__main__":
    unittest.main()

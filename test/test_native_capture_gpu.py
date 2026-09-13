# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Opt-in CUDA tests: COLDSNAP_TEST_NATIVE_LIBRARY=<built hydration library>."""

import ctypes
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/core"))
from coldsnap_core.hydration import CaptureExtent, HydrationError, HydrationExtent, NativeHydrator  # noqa: E402


@unittest.skipUnless(
    os.environ.get("COLDSNAP_TEST_NATIVE_LIBRARY"), "native CUDA capture test is opt-in"
)
class NativeCaptureGpuTests(unittest.TestCase):
    def setUp(self):
        self.cuda = ctypes.CDLL("libcudart.so.13")
        self.cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cuda.cudaFree.argtypes = [ctypes.c_void_p]
        self.cuda.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.source = ctypes.c_void_p()
        self.data = bytes(index % 251 for index in range(32768))
        self.check(self.cuda.cudaMalloc(ctypes.byref(self.source), len(self.data)))
        self.addCleanup(lambda: self.cuda.cudaFree(self.source))
        self.check(self.cuda.cudaMemcpy(self.source, ctypes.c_char_p(self.data), len(self.data), 1))
        self.transport = NativeHydrator(os.environ["COLDSNAP_TEST_NATIVE_LIBRARY"])

    def check(self, status):
        self.assertEqual(status, 0, f"CUDA status {status}")

    def test_padded_native_capture_preserves_logical_bytes_zeroes_tails_and_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.blob"
            for backend in ("auto", "direct", "buffered"):
                for file_sha in (False, True):
                    with self.subTest(backend=backend, file_sha=file_sha):
                        # Odd lengths, unaligned device pointers, leading/interior
                        # gaps and a partial last chunk exercise the actual CUDA ABI.
                        extents = [
                            CaptureExtent(4096, self.source.value + 3, 4097, "crc32+sha256"),
                            CaptureExtent(16384, self.source.value + 8193, 8191, "crc32+sha256"),
                        ]
                        result = self.transport.capture(
                            path,
                            extents,
                            backend=backend,
                            chunk_bytes=4096,
                            queue_depth=3,
                            file_sha256=file_sha,
                            verify_readback=True,
                            pad_extents=True,
                        )
                        self.assertEqual(
                            result.metrics.backend, "direct" if backend == "auto" else backend
                        )
                        blob = path.read_bytes()
                        self.assertEqual(result.metrics.bytes, 12288)
                        self.assertEqual(len(blob), 24576)
                        expected = bytearray(len(blob))
                        for extent, digest in zip(extents, result.digests, strict=True):
                            start = extent.source - self.source.value
                            data = self.data[start : start + extent.length]
                            expected[extent.file_offset : extent.file_offset + extent.length] = data
                            self.assertEqual(digest.crc32, f"{zlib.crc32(data):08x}")
                            self.assertEqual(digest.sha256, hashlib.sha256(data).hexdigest())
                        self.assertEqual(blob, expected)
                        if file_sha:
                            self.assertEqual(result.file_sha256, hashlib.sha256(blob).hexdigest())
                        self.assertGreater(result.metrics.verification_s, 0)
                        # Corruption remains detectable by ordinary inline hydration.
                        with path.open("r+b") as stream:
                            stream.seek(4096)
                            stream.write(b"!")
                        with self.assertRaisesRegex(HydrationError, "CRC32 mismatch"):
                            self.transport.hydrate(
                                path,
                                [
                                    HydrationExtent(
                                        4096, self.source.value, 4097, result.digests[0].crc32
                                    )
                                ],
                                backend="buffered",
                                chunk_bytes=4096,
                                queue_depth=2,
                                preverified=False,
                            )
                        # Restore source bytes after the deliberately failed hydration.
                        self.check(
                            self.cuda.cudaMemcpy(
                                self.source, ctypes.c_char_p(self.data), len(self.data), 1
                            )
                        )

    def test_legacy_unaligned_auto_stays_buffered_and_explicit_direct_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.blob"
            extents = [CaptureExtent(0, self.source.value, 4097)]
            result = self.transport.capture(
                path, extents, backend="auto", chunk_bytes=4096, queue_depth=2, verify_readback=True
            )
            self.assertEqual(result.metrics.backend, "buffered")
            self.assertEqual(path.read_bytes(), self.data[:4097])
            with self.assertRaisesRegex(HydrationError, "aligned"):
                self.transport.capture(
                    path, extents, backend="direct", chunk_bytes=4096, queue_depth=2
                )

    def test_padding_must_not_overlap_the_next_extent(self):
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            self.transport.capture(
                "/tmp/unused-capture.blob",
                [
                    CaptureExtent(0, self.source.value, 4097),
                    CaptureExtent(4098, self.source.value, 4096),
                ],
                backend="auto",
                chunk_bytes=4096,
                queue_depth=2,
                pad_extents=True,
            )

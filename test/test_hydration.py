# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import ctypes
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

from coldsnap_core import hydration  # noqa: E402
from coldsnap_core.artifact import _hydrate_native_extents  # noqa: E402
from coldsnap_core import hydration as core_hydration  # noqa: E402


class HydrationExtentTests(unittest.TestCase):
    def test_semantic_hydration_preserves_direct_for_aligned_weight_extents(self) -> None:
        calls = []

        class FakeHydrator:
            @staticmethod
            def available(backend):
                return backend in {"direct", "buffered"}

            @staticmethod
            def hydrate(_path, extents, *, backend, **_kwargs):
                calls.append((backend, [extent.length for extent in extents]))
                return SimpleNamespace(
                    backend=backend,
                    bytes=sum(extent.length for extent in extents),
                )

        extents = [
            hydration.HydrationExtent(0, 0x1000, 8192),
            hydration.HydrationExtent(8192, 0x4000, 17),
        ]
        backend, restored = _hydrate_native_extents(
            FakeHydrator(),
            Path("/artifact/model.pack"),
            extents,
            backend="direct",
            chunk_bytes=4096,
            queue_depth=2,
            cuda_device=0,
            preverified=True,
        )

        self.assertEqual(calls, [("direct", [8192]), ("buffered", [17])])
        self.assertEqual(backend, "direct+buffered")
        self.assertEqual(restored, 8209)

    def test_crc_and_sha_contract_is_encoded_for_native_verification(self) -> None:
        extent = hydration.HydrationExtent(
            file_offset=4096,
            destination=0x10000,
            length=8192,
            crc32="1234abcd",
            sha256="01" * 32,
        )

        native = hydration.NativeHydrator._native_extent(
            extent, preverified=False
        )

        self.assertEqual(native.verification, 2)
        self.assertEqual(native.expected_crc32, 0x1234ABCD)
        self.assertEqual(bytes(native.expected_sha256), bytes.fromhex("01" * 32))

    def test_sha_requires_crc(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires CRC32"):
            hydration.HydrationExtent(
                file_offset=0,
                destination=1,
                length=1,
                sha256="01" * 32,
            )

    def test_gds_rejects_unverified_device_direct_bytes(self) -> None:
        hydrator = object.__new__(hydration.NativeHydrator)
        extent = hydration.HydrationExtent(
            file_offset=0,
            destination=0x1000,
            length=4096,
            crc32="00000000",
        )

        with self.assertRaisesRegex(hydration.HydrationError, "immutable"):
            hydrator.hydrate(
                "/tmp/blob",
                [extent],
                backend="gds",
                chunk_bytes=4096,
                queue_depth=1,
            )

    def test_gds_requires_explicit_trust_even_without_extent_digests(self) -> None:
        hydrator = object.__new__(hydration.NativeHydrator)
        extent = hydration.HydrationExtent(
            file_offset=0,
            destination=0x1000,
            length=4096,
        )

        with self.assertRaisesRegex(hydration.HydrationError, "immutable"):
            hydrator.hydrate(
                "/tmp/blob",
                [extent],
                backend="gds",
                chunk_bytes=4096,
                queue_depth=1,
            )

    def test_disabled_environment_is_noop(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(hydration.hydrator_from_env())

    def test_enabled_environment_requires_library(self) -> None:
        with patch.dict(
            os.environ,
            {hydration.HYDRATION_BACKEND_ENV: "buffered"},
            clear=True,
        ):
            with self.assertRaisesRegex(hydration.HydrationError, "is required"):
                hydration.hydrator_from_env()

    def test_expected_native_library_digest_is_verified(self) -> None:
        fake = SimpleNamespace(sha256="01" * 32)
        environment = {
            hydration.HYDRATION_BACKEND_ENV: "buffered",
            hydration.HYDRATION_LIBRARY_ENV: "/runtime/libcoldsnap_hydration.so",
            hydration.HYDRATION_LIBRARY_SHA256_ENV: "01" * 32,
        }
        with patch.dict(os.environ, environment, clear=True), patch.object(
            core_hydration, "NativeHydrator", return_value=fake
        ):
            self.assertEqual(
                hydration.hydrator_from_env(),
                (fake, "buffered"),
            )

    def test_expected_native_library_digest_mismatch_fails_closed(self) -> None:
        fake = SimpleNamespace(sha256="02" * 32)
        environment = {
            hydration.HYDRATION_BACKEND_ENV: "buffered",
            hydration.HYDRATION_LIBRARY_ENV: "/runtime/libcoldsnap_hydration.so",
            hydration.HYDRATION_LIBRARY_SHA256_ENV: "01" * 32,
        }
        with patch.dict(os.environ, environment, clear=True), patch.object(
            core_hydration, "NativeHydrator", return_value=fake
        ):
            with self.assertRaisesRegex(hydration.HydrationError, "mismatch"):
                hydration.hydrator_from_env()

    def test_auto_backend_uses_buffered_for_unaligned_semantic_extent(self) -> None:
        hydrator = object.__new__(hydration.NativeHydrator)
        extent = hydration.HydrationExtent(
            file_offset=4096,
            destination=0x1000,
            length=4097,
        )
        with patch.object(hydrator, "available", return_value=True):
            self.assertEqual(
                hydrator.resolve_backend("auto", [extent]),
                "buffered",
            )

    def test_auto_backend_prefers_direct_for_aligned_semantic_extents(self) -> None:
        hydrator = object.__new__(hydration.NativeHydrator)
        extent = hydration.HydrationExtent(
            file_offset=4096,
            destination=0x1000,
            length=8192,
        )
        with patch.object(hydrator, "available", return_value=True):
            self.assertEqual(hydrator.resolve_backend("auto", [extent]), "direct")

    def test_capture_abi_returns_digests_and_native_metrics(self) -> None:
        hydrator = object.__new__(hydration.NativeHydrator)
        hydrator._capture_backend_available = lambda _backend: 1
        hydrator._library = SimpleNamespace(
            coldsnap_hydration_last_error=lambda: b""
        )

        def capture_file(
            _path, _extents, extent_count, options_pointer, digests, result_pointer
        ):
            options = ctypes.cast(
                options_pointer, ctypes.POINTER(hydration._NativeCaptureOptions)
            ).contents
            result = ctypes.cast(
                result_pointer, ctypes.POINTER(hydration._NativeCaptureResult)
            ).contents
            result.abi_version = hydration.ABI_VERSION
            result.backend = options.backend
            result.bytes = 8192
            result.file_bytes = 12288
            result.chunks = 2
            result.checksummed_extents = extent_count
            result.total_ns = 2_000_000
            result.file_sha256[:] = bytes.fromhex("ab" * 32)
            digests[0].crc32 = 0x1234ABCD
            digests[0].sha256[:] = bytes.fromhex("01" * 32)
            return 0

        hydrator._capture_file = capture_file
        captured = hydrator.capture(
            "/tmp/capture.blob",
            [
                hydration.CaptureExtent(
                    file_offset=4096,
                    source=0x10000,
                    length=8192,
                    checksum="crc32+sha256",
                )
            ],
            backend="direct",
            chunk_bytes=4096,
            queue_depth=2,
            file_sha256=True,
        )

        self.assertEqual(captured.metrics.backend, "direct")
        self.assertEqual(captured.metrics.bytes, 8192)
        self.assertEqual(captured.metrics.file_bytes, 12288)
        self.assertEqual(captured.metrics.total_s, 0.002)
        self.assertEqual(captured.digests[0].crc32, "1234abcd")
        self.assertEqual(captured.digests[0].sha256, "01" * 32)
        self.assertEqual(captured.file_sha256, "ab" * 32)

    def test_capture_auto_uses_buffered_for_unaligned_extent(self) -> None:
        hydrator = object.__new__(hydration.NativeHydrator)
        hydrator._capture_file = object()
        hydrator._capture_backend_available = lambda _backend: 1
        extent = hydration.CaptureExtent(0, 0x1000, 4097)

        self.assertEqual(
            hydrator.resolve_capture_backend("auto", [extent]), "buffered"
        )

    def test_capture_environment_is_independently_selectable(self) -> None:
        fake = SimpleNamespace(
            sha256="01" * 32,
            capture_available=lambda _backend: True,
        )
        environment = {
            hydration.CAPTURE_BACKEND_ENV: "auto",
            hydration.HYDRATION_LIBRARY_ENV: "/runtime/libcoldsnap_hydration.so",
        }
        with patch.dict(os.environ, environment, clear=True), patch.object(
            core_hydration, "NativeHydrator", return_value=fake
        ):
            self.assertEqual(hydration.capture_from_env(), (fake, "auto"))


if __name__ == "__main__":
    unittest.main()

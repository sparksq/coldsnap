# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Engine-neutral native transport between CUDA allocations and durable files."""

from __future__ import annotations

import ctypes
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HYDRATION_LIBRARY_ENV = "COLDSNAP_HYDRATION_NATIVE_LIBRARY"
HYDRATION_BACKEND_ENV = "COLDSNAP_HYDRATION_BACKEND"
HYDRATION_LIBRARY_SHA256_ENV = "COLDSNAP_HYDRATION_NATIVE_SHA256"
CAPTURE_BACKEND_ENV = "COLDSNAP_CAPTURE_BACKEND"
ABI_VERSION = 1

_BACKENDS = {"buffered": 1, "direct": 2, "gds": 3}
_CAPTURE_BACKENDS = {"buffered": 1, "direct": 2}
_VERIFY_NONE = 0
_VERIFY_CRC32 = 1
_VERIFY_CRC32_SHA256 = 2
_REGISTER_DEVICE_BUFFERS = 1
_CAPTURE_FILE_SHA256 = 1
_CAPTURE_VERIFY_READBACK = 2
DIRECT_ALIGNMENT = 4096


class HydrationError(RuntimeError):
    """The native hydration transport rejected or failed a transfer."""


@dataclass(frozen=True)
class HydrationExtent:
    """One contiguous snapshot range and its final CUDA destination."""

    file_offset: int
    destination: int
    length: int
    crc32: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.file_offset < 0 or self.destination <= 0 or self.length <= 0:
            raise ValueError("hydration extent offset, destination, and length are invalid")
        if self.crc32 is not None:
            if len(self.crc32) != 8 or self.crc32.lower() != self.crc32:
                raise ValueError("hydration CRC32 must be eight lowercase hex digits")
            try:
                bytes.fromhex(self.crc32)
            except ValueError as error:
                raise ValueError("hydration CRC32 is not hexadecimal") from error
        if self.sha256 is not None:
            if self.crc32 is None:
                raise ValueError("hydration SHA-256 requires CRC32")
            if len(self.sha256) != 64 or self.sha256.lower() != self.sha256:
                raise ValueError("hydration SHA-256 must be 64 lowercase hex digits")
            try:
                bytes.fromhex(self.sha256)
            except ValueError as error:
                raise ValueError("hydration SHA-256 is not hexadecimal") from error


@dataclass(frozen=True)
class HydrationMetrics:
    """Native transfer timings, including overlapping service-time counters."""

    backend: str
    bytes: int
    chunks: int
    verified_extents: int
    initialization_s: float
    io_service_s: float
    io_wait_s: float
    checksum_s: float
    cuda_enqueue_s: float
    cuda_synchronize_s: float
    total_s: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaptureExtent:
    """One contiguous CUDA source and its destination range in a capture file."""

    file_offset: int
    source: int
    length: int
    checksum: str = "crc32"

    def __post_init__(self) -> None:
        if self.file_offset < 0 or self.source <= 0 or self.length <= 0:
            raise ValueError("capture extent offset, source, and length are invalid")
        if self.checksum not in {"none", "crc32", "crc32+sha256"}:
            raise ValueError(
                "capture checksum must be none, crc32, or crc32+sha256"
            )


@dataclass(frozen=True)
class CaptureDigest:
    """Digests produced from the staged bytes of one capture extent."""

    crc32: str | None
    sha256: str | None


@dataclass(frozen=True)
class CaptureMetrics:
    """Native capture timings, including overlapping service-time counters."""

    backend: str
    bytes: int
    file_bytes: int
    chunks: int
    checksummed_extents: int
    initialization_s: float
    io_service_s: float
    io_wait_s: float
    checksum_s: float
    cuda_enqueue_s: float
    cuda_synchronize_s: float
    durability_s: float
    verification_s: float
    total_s: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaptureResult:
    """Digests and timing evidence returned by one durable native capture."""

    metrics: CaptureMetrics
    digests: tuple[CaptureDigest, ...]
    file_sha256: str | None


class _NativeExtent(ctypes.Structure):
    _fields_ = [
        ("file_offset", ctypes.c_uint64),
        ("destination", ctypes.c_size_t),
        ("length", ctypes.c_uint64),
        ("verification", ctypes.c_uint32),
        ("expected_crc32", ctypes.c_uint32),
        ("expected_sha256", ctypes.c_uint8 * 32),
    ]


class _NativeOptions(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("backend", ctypes.c_uint32),
        ("queue_depth", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("chunk_bytes", ctypes.c_uint64),
        ("cuda_device", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class _NativeResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("backend", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
        ("chunks", ctypes.c_uint64),
        ("verified_extents", ctypes.c_uint64),
        ("initialization_ns", ctypes.c_uint64),
        ("io_service_ns", ctypes.c_uint64),
        ("io_wait_ns", ctypes.c_uint64),
        ("checksum_ns", ctypes.c_uint64),
        ("cuda_enqueue_ns", ctypes.c_uint64),
        ("cuda_synchronize_ns", ctypes.c_uint64),
        ("total_ns", ctypes.c_uint64),
    ]


class _NativeCaptureExtent(ctypes.Structure):
    _fields_ = [
        ("source", ctypes.c_size_t),
        ("file_offset", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("checksum", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _NativeCaptureDigest(ctypes.Structure):
    _fields_ = [
        ("crc32", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("sha256", ctypes.c_uint8 * 32),
    ]


class _NativeCaptureOptions(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("backend", ctypes.c_uint32),
        ("queue_depth", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("chunk_bytes", ctypes.c_uint64),
        ("cuda_device", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class _NativeCaptureResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("backend", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
        ("file_bytes", ctypes.c_uint64),
        ("chunks", ctypes.c_uint64),
        ("checksummed_extents", ctypes.c_uint64),
        ("initialization_ns", ctypes.c_uint64),
        ("io_service_ns", ctypes.c_uint64),
        ("io_wait_ns", ctypes.c_uint64),
        ("checksum_ns", ctypes.c_uint64),
        ("cuda_enqueue_ns", ctypes.c_uint64),
        ("cuda_synchronize_ns", ctypes.c_uint64),
        ("durability_ns", ctypes.c_uint64),
        ("verification_ns", ctypes.c_uint64),
        ("total_ns", ctypes.c_uint64),
        ("file_sha256", ctypes.c_uint8 * 32),
    ]


class NativeHydrator:
    """Bounded asynchronous hydration into engine-owned CUDA allocations."""

    def __init__(self, library_path: str | os.PathLike[str]) -> None:
        path = Path(library_path).resolve()
        if not path.is_file():
            raise HydrationError(f"native hydration library does not exist: {path}")
        self.library_path = path
        self._library = ctypes.CDLL(str(path))
        self._configure_abi()

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.library_path.open("rb", buffering=0) as source:
            while chunk := source.read(16 * 1024**2):
                digest.update(chunk)
        return digest.hexdigest()

    def _configure_abi(self) -> None:
        library = self._library
        library.coldsnap_hydrate_file.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(_NativeExtent),
            ctypes.c_size_t,
            ctypes.POINTER(_NativeOptions),
            ctypes.POINTER(_NativeResult),
        ]
        library.coldsnap_hydrate_file.restype = ctypes.c_int
        library.coldsnap_hydration_backend_available.argtypes = [ctypes.c_uint32]
        library.coldsnap_hydration_backend_available.restype = ctypes.c_int
        library.coldsnap_hydration_last_error.argtypes = []
        library.coldsnap_hydration_last_error.restype = ctypes.c_char_p
        self._capture_file = getattr(library, "coldsnap_capture_file", None)
        self._capture_backend_available = getattr(
            library, "coldsnap_capture_backend_available", None
        )
        if self._capture_file is not None:
            self._capture_file.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(_NativeCaptureExtent),
                ctypes.c_size_t,
                ctypes.POINTER(_NativeCaptureOptions),
                ctypes.POINTER(_NativeCaptureDigest),
                ctypes.POINTER(_NativeCaptureResult),
            ]
            self._capture_file.restype = ctypes.c_int
        if self._capture_backend_available is not None:
            self._capture_backend_available.argtypes = [ctypes.c_uint32]
            self._capture_backend_available.restype = ctypes.c_int

    def _error(self, operation: str) -> HydrationError:
        value = self._library.coldsnap_hydration_last_error()
        detail = value.decode("utf-8", errors="replace") if value else "unknown error"
        return HydrationError(f"native hydration {operation} failed: {detail}")

    def _capture_error(self, operation: str) -> HydrationError:
        value = self._library.coldsnap_hydration_last_error()
        detail = value.decode("utf-8", errors="replace") if value else "unknown error"
        return HydrationError(f"native capture {operation} failed: {detail}")

    @staticmethod
    def _backend_id(backend: str) -> int:
        try:
            return _BACKENDS[backend]
        except KeyError as error:
            raise ValueError(
                "hydration backend must be buffered, direct, or gds"
            ) from error

    def available(self, backend: str) -> bool:
        return bool(
            self._library.coldsnap_hydration_backend_available(
                self._backend_id(backend)
            )
        )

    def capture_available(self, backend: str) -> bool:
        try:
            backend_id = _CAPTURE_BACKENDS[backend]
        except KeyError as error:
            raise ValueError("capture backend must be buffered or direct") from error
        if self._capture_file is None or self._capture_backend_available is None:
            return False
        return bool(self._capture_backend_available(backend_id))

    def resolve_capture_backend(
        self, backend: str, extents: list[CaptureExtent]
    ) -> str:
        if backend != "auto":
            if backend not in _CAPTURE_BACKENDS:
                raise ValueError("capture backend must be auto, buffered, or direct")
            if not self.capture_available(backend):
                raise HydrationError(f"native capture backend is unavailable: {backend}")
            return backend
        direct_compatible = all(
            extent.file_offset % DIRECT_ALIGNMENT == 0
            and extent.length % DIRECT_ALIGNMENT == 0
            for extent in extents
        )
        if direct_compatible and self.capture_available("direct"):
            return "direct"
        if self.capture_available("buffered"):
            return "buffered"
        raise HydrationError(
            "native capture auto policy found no compatible buffered/direct backend"
        )

    def resolve_backend(
        self, backend: str, extents: list[HydrationExtent]
    ) -> str:
        """Resolve policy without sending an invalid layout to O_DIRECT.

        Explicit backends remain fail-closed. ``auto`` prefers direct I/O only
        when every semantic extent satisfies its actual alignment contract;
        irregular layouts such as MiniMax automatically retain the same native
        pipeline through the buffered backend.
        """
        if backend != "auto":
            self._backend_id(backend)
            return backend
        direct_compatible = all(
            extent.file_offset % DIRECT_ALIGNMENT == 0
            and extent.length % DIRECT_ALIGNMENT == 0
            for extent in extents
        )
        if direct_compatible and self.available("direct"):
            return "direct"
        if self.available("buffered"):
            return "buffered"
        raise HydrationError(
            "native hydration auto policy found no compatible buffered/direct backend"
        )

    @staticmethod
    def _native_extent(
        extent: HydrationExtent, *, preverified: bool
    ) -> _NativeExtent:
        if preverified or extent.crc32 is None:
            verification = _VERIFY_NONE
        elif extent.sha256 is None:
            verification = _VERIFY_CRC32
        else:
            verification = _VERIFY_CRC32_SHA256
        digest = bytes.fromhex(extent.sha256) if extent.sha256 else bytes(32)
        return _NativeExtent(
            file_offset=extent.file_offset,
            destination=extent.destination,
            length=extent.length,
            verification=verification,
            expected_crc32=int(extent.crc32, 16) if extent.crc32 else 0,
            expected_sha256=(ctypes.c_uint8 * 32).from_buffer_copy(digest),
        )

    def hydrate(
        self,
        path: str | os.PathLike[str],
        extents: list[HydrationExtent],
        *,
        backend: str,
        chunk_bytes: int,
        queue_depth: int,
        cuda_device: int = -1,
        preverified: bool = False,
        register_device_buffers: bool = False,
    ) -> HydrationMetrics:
        if not extents:
            raise ValueError("hydration requires at least one extent")
        if chunk_bytes <= 0 or chunk_bytes % 4096:
            raise ValueError("hydration chunk size must be 4096-byte aligned")
        if not 1 <= queue_depth <= 64:
            raise ValueError("hydration queue depth must be between 1 and 64")
        backend = self.resolve_backend(backend, extents)
        if backend == "gds" and not preverified:
            raise HydrationError(
                "GDS cannot validate device-direct bytes; use an immutable "
                "preverified artifact"
            )

        backend_id = self._backend_id(backend)
        native_extents = (_NativeExtent * len(extents))(
            *(
                self._native_extent(extent, preverified=preverified)
                for extent in extents
            )
        )
        options = _NativeOptions(
            abi_version=ABI_VERSION,
            backend=backend_id,
            queue_depth=queue_depth,
            flags=(
                _REGISTER_DEVICE_BUFFERS if register_device_buffers else 0
            ),
            chunk_bytes=chunk_bytes,
            cuda_device=cuda_device,
            reserved=0,
        )
        result = _NativeResult()
        status = self._library.coldsnap_hydrate_file(
            os.fsencode(Path(path)),
            native_extents,
            len(extents),
            ctypes.byref(options),
            ctypes.byref(result),
        )
        if status != 0:
            raise self._error(backend)
        if result.abi_version != ABI_VERSION or result.backend != backend_id:
            raise HydrationError("native hydration result has an incompatible ABI")

        seconds = 1e-9
        return HydrationMetrics(
            backend=backend,
            bytes=result.bytes,
            chunks=result.chunks,
            verified_extents=result.verified_extents,
            initialization_s=result.initialization_ns * seconds,
            io_service_s=result.io_service_ns * seconds,
            io_wait_s=result.io_wait_ns * seconds,
            checksum_s=result.checksum_ns * seconds,
            cuda_enqueue_s=result.cuda_enqueue_ns * seconds,
            cuda_synchronize_s=result.cuda_synchronize_ns * seconds,
            total_s=result.total_ns * seconds,
        )

    def capture(
        self,
        path: str | os.PathLike[str],
        extents: list[CaptureExtent],
        *,
        backend: str,
        chunk_bytes: int,
        queue_depth: int,
        cuda_device: int = -1,
        file_sha256: bool = False,
        verify_readback: bool = False,
    ) -> CaptureResult:
        """Capture device extents through the shared staged native pipeline."""
        if not extents:
            raise ValueError("capture requires at least one extent")
        if chunk_bytes <= 0 or chunk_bytes % DIRECT_ALIGNMENT:
            raise ValueError("capture chunk size must be 4096-byte aligned")
        if not 1 <= queue_depth <= 64:
            raise ValueError("capture queue depth must be between 1 and 64")
        previous_end = 0
        for extent in extents:
            if extent.file_offset < previous_end:
                raise ValueError("capture extents must be ordered and non-overlapping")
            previous_end = extent.file_offset + extent.length
        requested_backend = backend
        backend = self.resolve_capture_backend(backend, extents)
        if self._capture_file is None:
            raise HydrationError(
                "native transfer library does not expose the capture ABI"
            )

        checksum_ids = {"none": 0, "crc32": 1, "crc32+sha256": 2}
        # Native auto conveys that O_DIRECT may fall back at open time when the
        # target filesystem rejects it. Explicit direct remains fail-closed.
        backend_id = (
            0
            if requested_backend == "auto" and backend == "direct"
            else _CAPTURE_BACKENDS[backend]
        )
        native_extents = (_NativeCaptureExtent * len(extents))(
            *(
                _NativeCaptureExtent(
                    source=extent.source,
                    file_offset=extent.file_offset,
                    length=extent.length,
                    checksum=checksum_ids[extent.checksum],
                    reserved=0,
                )
                for extent in extents
            )
        )
        digests = (_NativeCaptureDigest * len(extents))()
        flags = 0
        if file_sha256:
            flags |= _CAPTURE_FILE_SHA256
        if verify_readback:
            flags |= _CAPTURE_VERIFY_READBACK
        options = _NativeCaptureOptions(
            abi_version=ABI_VERSION,
            backend=backend_id,
            queue_depth=queue_depth,
            flags=flags,
            chunk_bytes=chunk_bytes,
            cuda_device=cuda_device,
            reserved=0,
        )
        result = _NativeCaptureResult()
        status = self._capture_file(
            os.fsencode(Path(path)),
            native_extents,
            len(extents),
            ctypes.byref(options),
            digests,
            ctypes.byref(result),
        )
        if status != 0:
            raise self._capture_error(backend)
        actual_backends = {value: name for name, value in _CAPTURE_BACKENDS.items()}
        if (
            result.abi_version != ABI_VERSION
            or result.backend not in actual_backends
            or (backend_id != 0 and result.backend != backend_id)
        ):
            raise HydrationError("native capture result has an incompatible ABI")
        backend = actual_backends[result.backend]

        captured_digests: list[CaptureDigest] = []
        for extent, digest in zip(extents, digests, strict=True):
            captured_digests.append(
                CaptureDigest(
                    crc32=(
                        f"{digest.crc32:08x}"
                        if extent.checksum != "none"
                        else None
                    ),
                    sha256=(
                        bytes(digest.sha256).hex()
                        if extent.checksum == "crc32+sha256"
                        else None
                    ),
                )
            )
        seconds = 1e-9
        metrics = CaptureMetrics(
            backend=backend,
            bytes=result.bytes,
            file_bytes=result.file_bytes,
            chunks=result.chunks,
            checksummed_extents=result.checksummed_extents,
            initialization_s=result.initialization_ns * seconds,
            io_service_s=result.io_service_ns * seconds,
            io_wait_s=result.io_wait_ns * seconds,
            checksum_s=result.checksum_ns * seconds,
            cuda_enqueue_s=result.cuda_enqueue_ns * seconds,
            cuda_synchronize_s=result.cuda_synchronize_ns * seconds,
            durability_s=result.durability_ns * seconds,
            verification_s=result.verification_ns * seconds,
            total_s=result.total_ns * seconds,
        )
        return CaptureResult(
            metrics=metrics,
            digests=tuple(captured_digests),
            file_sha256=(bytes(result.file_sha256).hex() if file_sha256 else None),
        )


def _native_transport(backend: str) -> NativeHydrator:
    library = os.environ.get(HYDRATION_LIBRARY_ENV)
    if not library:
        raise HydrationError(
            f"{HYDRATION_LIBRARY_ENV} is required for backend {backend}"
        )
    hydrator = NativeHydrator(library)
    expected_sha256 = os.environ.get(HYDRATION_LIBRARY_SHA256_ENV)
    if expected_sha256:
        if len(expected_sha256) != 64 or expected_sha256.lower() != expected_sha256:
            raise HydrationError(
                f"{HYDRATION_LIBRARY_SHA256_ENV} must be 64 lowercase hexadecimal characters"
            )
        try:
            bytes.fromhex(expected_sha256)
        except ValueError as error:
            raise HydrationError(
                f"{HYDRATION_LIBRARY_SHA256_ENV} is not hexadecimal"
            ) from error
        actual_sha256 = hydrator.sha256
        if actual_sha256 != expected_sha256:
            raise HydrationError(
                "native transfer library SHA-256 mismatch: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
    return hydrator


def hydrator_from_env() -> tuple[NativeHydrator, str] | None:
    """Resolve explicitly enabled native hydration without hidden fallback."""
    backend = os.environ.get(HYDRATION_BACKEND_ENV, "python")
    if backend == "python":
        return None
    if backend not in {*_BACKENDS, "auto"}:
        raise ValueError(
            f"{HYDRATION_BACKEND_ENV} must be python, auto, buffered, direct, or gds"
        )
    return _native_transport(backend), backend


def capture_from_env() -> tuple[NativeHydrator, str] | None:
    """Resolve explicitly enabled native capture without hidden fallback."""
    backend = os.environ.get(CAPTURE_BACKEND_ENV, "python")
    if backend == "python":
        return None
    if backend not in {*_CAPTURE_BACKENDS, "auto"}:
        raise ValueError(
            f"{CAPTURE_BACKEND_ENV} must be python, auto, buffered, or direct"
        )
    transport = _native_transport(backend)
    if not any(transport.capture_available(item) for item in _CAPTURE_BACKENDS):
        raise HydrationError(
            "native transfer library does not expose an available capture backend"
        )
    return transport, backend

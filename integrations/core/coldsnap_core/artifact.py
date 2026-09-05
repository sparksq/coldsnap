# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Reusable semantic GPU tensor artifacts for PyTorch inference engines.

Artifacts bind tensor names and layouts, not process-specific CUDA addresses.
Fresh processes construct their final tensor storages first and hydrate this
blob directly into those storages. CUDA graphs may therefore retain their
captured pointers while the physical weight bytes are replaced.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .hydration import (
    DIRECT_ALIGNMENT,
    CaptureExtent,
    HydrationExtent,
    capture_from_env,
    hydrator_from_env,
)
from .validation import verify_validation_record


ARTIFACT_FORMAT = 1
ARTIFACT_KIND = "coldsnap-pytorch-semantic-weights"
DEFAULT_CHUNK_BYTES = 256 * 1024**2
ALIGNMENT = 4096


class ArtifactError(RuntimeError):
    """A semantic artifact is absent, incompatible, or corrupt."""


@dataclass(frozen=True)
class ArtifactMetrics:
    operation: str
    backend: str
    bytes: int
    storages: int
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _registered_cuda_tensors(model: Any) -> dict[str, Any]:
    nonpersistent_buffers: set[str] = set()
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        try:
            modules = named_modules(remove_duplicate=False)
        except TypeError:
            modules = named_modules()
        for module_name, module in modules:
            for local_name in getattr(module, "_non_persistent_buffers_set", ()):
                nonpersistent_buffers.add(
                    f"{module_name}.{local_name}" if module_name else local_name
                )

    tensors: dict[str, Any] = {}
    for kind, values in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in values:
            # Non-persistent buffers are explicitly runtime-derived state, not
            # checkpoint semantics. Reusing their captured byte extent makes
            # otherwise compatible startups sensitive to harmless cache-size
            # choices (for example an auto-sized rotary cache). Keep the fresh
            # runtime's derived value and hydrate only persistent model state.
            if kind == "buffer" and name in nonpersistent_buffers:
                continue
            if tensor.device.type == "cuda" and tensor.numel() > 0:
                key = f"{kind}:{name}"
                if key in tensors:
                    raise ArtifactError(f"duplicate registered tensor {key!r}")
                tensors[key] = tensor
    if not tensors:
        raise ArtifactError("model has no registered CUDA tensors")
    return tensors


def _tensor_reference(name: str, tensor: Any, storage_pointer: int) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset_bytes": int(tensor.data_ptr()) - storage_pointer,
        "nbytes": int(tensor.numel()) * int(tensor.element_size()),
    }


def _storage_view(tensor: Any) -> Any:
    import torch

    storage = tensor.untyped_storage()
    view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
    return view.set_(storage, 0, (int(storage.nbytes()),), (1,))


def _storage_groups(model: Any) -> list[tuple[Any, list[dict[str, Any]]]]:
    groups: dict[tuple[int, int, int], tuple[Any, list[dict[str, Any]]]] = {}
    for name, tensor in sorted(_registered_cuda_tensors(model).items()):
        storage = tensor.untyped_storage()
        pointer = int(storage.data_ptr())
        nbytes = int(storage.nbytes())
        device_index = int(tensor.device.index or 0)
        key = (pointer, nbytes, device_index)
        if key not in groups:
            groups[key] = (tensor, [])
        groups[key][1].append(_tensor_reference(name, tensor, pointer))
    return sorted(
        groups.values(), key=lambda item: item[1][0]["name"]
    )


def _hydrate_native_extents(
    hydrator: Any,
    path: Path,
    extents: list[HydrationExtent],
    *,
    backend: str,
    chunk_bytes: int,
    queue_depth: int,
    cuda_device: int,
    preverified: bool,
) -> tuple[str, int]:
    """Hydrate semantic extents without forfeiting direct I/O for residuals.

    PyTorch storage lengths are not required to be filesystem-block aligned.
    ColdSnap aligns every storage's file offset, so aligned weight storages can
    use O_DIRECT while the comparatively small irregular storages use the same
    native pipeline in buffered mode.  Extents are never enlarged because that
    would write beyond their CUDA allocations.
    """
    groups: list[tuple[str, list[HydrationExtent]]]
    if backend in {"auto", "direct"}:
        aligned: list[HydrationExtent] = []
        residual: list[HydrationExtent] = []
        for extent in extents:
            target = (
                aligned
                if extent.file_offset % DIRECT_ALIGNMENT == 0
                and extent.length % DIRECT_ALIGNMENT == 0
                else residual
            )
            target.append(extent)
        direct_available = backend == "direct" or hydrator.available("direct")
        if not direct_available:
            residual = extents
            aligned = []
        groups = []
        if aligned:
            groups.append(("direct", aligned))
        if residual:
            groups.append(("buffered", residual))
    else:
        groups = [(backend, extents)]

    used: list[str] = []
    restored_bytes = 0
    for selected_backend, selected_extents in groups:
        metrics = hydrator.hydrate(
            path,
            selected_extents,
            backend=selected_backend,
            chunk_bytes=chunk_bytes,
            queue_depth=queue_depth,
            cuda_device=cuda_device,
            preverified=preverified,
        )
        used.append(metrics.backend)
        restored_bytes += int(metrics.bytes)
    return ("+".join(used), restored_bytes)


def _identity_mismatches(expected: Any, actual: Any, path: str = "identity") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches: list[str] = []
        for key in sorted(expected.keys() | actual.keys()):
            child = f"{path}.{key}"
            if key not in expected:
                mismatches.append(f"{child} is unexpected")
            elif key not in actual:
                mismatches.append(f"{child} is missing")
            else:
                mismatches.extend(_identity_mismatches(expected[key], actual[key], child))
        return mismatches
    if expected != actual:
        return [f"{path}: expected {expected!r}, found {actual!r}"]
    return []


class SemanticTensorArtifact:
    """Transactional capture and address-independent restore of model storages."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        lock_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.blob_path = self.root / "weights.blob"
        if lock_root is None:
            self.lock_path = self.root / ".lock"
        else:
            lock_directory = Path(lock_root)
            key = hashlib.sha256(os.fsencode(self.root.resolve())).hexdigest()
            self.lock_path = lock_directory / f"{key}.lock"

    def _lock(self, *, exclusive: bool):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(
            descriptor,
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
        return descriptor

    @staticmethod
    def _unlock(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def load_manifest(
        self,
        identity: dict[str, Any] | None = None,
        *,
        blob_path: str | os.PathLike[str] | None = None,
        blob_offset: int = 0,
    ) -> dict[str, Any]:
        try:
            with self.manifest_path.open("rb") as stream:
                manifest = json.load(stream)
        except FileNotFoundError as error:
            raise ArtifactError(f"artifact manifest is missing: {self.manifest_path}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(f"cannot read artifact manifest: {error}") from error
        if manifest.get("format") != ARTIFACT_FORMAT or manifest.get("kind") != ARTIFACT_KIND:
            raise ArtifactError("artifact format or kind is incompatible")
        if identity is not None:
            mismatches = _identity_mismatches(identity, manifest.get("identity"))
            if mismatches:
                raise ArtifactError("artifact compatibility mismatch: " + "; ".join(mismatches[:8]))
        selected_blob = self.blob_path if blob_path is None else Path(blob_path)
        if blob_offset < 0:
            raise ArtifactError("artifact blob offset cannot be negative")
        try:
            blob_size = selected_blob.stat().st_size
        except FileNotFoundError as error:
            raise ArtifactError(f"artifact blob is missing: {selected_blob}") from error
        expected_end = blob_offset + int(manifest.get("blob_bytes", -1))
        if (blob_path is None and blob_size != expected_end) or (
            blob_path is not None and blob_size < expected_end
        ):
            raise ArtifactError(
                "artifact blob size mismatch: "
                f"expected {'exactly' if blob_path is None else 'at least'} "
                f"{expected_end}, found {blob_size}"
            )
        return manifest

    def is_compatible(self, identity: dict[str, Any]) -> bool:
        descriptor = self._lock(exclusive=False)
        try:
            self.load_manifest(identity)
            return True
        except ArtifactError:
            return False
        finally:
            self._unlock(descriptor)

    def verify_preverified(
        self,
        manifest: dict[str, Any],
        marker: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        marker_path = (
            Path(marker)
            if marker is not None
            else self.blob_path.with_name(
                self.blob_path.name + ".coldsnap-verified.json"
            )
        )
        try:
            return verify_validation_record(
                self.blob_path,
                str(manifest.get("blob_sha256", "")),
                marker_path,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            raise ArtifactError(
                f"preverified artifact admission failed: {error}"
            ) from error

    def capture(
        self,
        model: Any,
        *,
        identity: dict[str, Any],
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        queue_depth: int = 2,
        replace: bool = False,
    ) -> ArtifactMetrics:
        import torch

        if chunk_bytes <= 0:
            raise ValueError("capture chunk size must be positive")
        if not 1 <= queue_depth <= 64:
            raise ValueError("capture queue depth must be between 1 and 64")
        started = time.perf_counter()
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = self._lock(exclusive=True)
        temporary_blob = self.root / f".weights.blob.{os.getpid()}.tmp"
        temporary_manifest = self.root / f".manifest.json.{os.getpid()}.tmp"
        try:
            if not replace:
                try:
                    self.load_manifest(identity)
                except ArtifactError:
                    pass
                else:
                    raise ArtifactError("a compatible artifact already exists")

            groups = _storage_groups(model)
            position = 0
            layout: list[tuple[Any, list[dict[str, Any]], int, int]] = []
            for tensor, refs in groups:
                position = _align(position)
                storage_bytes = int(tensor.untyped_storage().nbytes())
                layout.append((tensor, refs, position, storage_bytes))
                position += storage_bytes
            total_bytes = position

            entries: list[dict[str, Any]] = []
            capture_backend = "python"
            native = capture_from_env()
            if native is not None:
                if chunk_bytes % ALIGNMENT:
                    raise ValueError(
                        "native capture chunk size must be 4096-byte aligned"
                    )
                transport, capture_backend = native
                capture_result = transport.capture(
                    temporary_blob,
                    [
                        CaptureExtent(
                            file_offset=offset,
                            source=int(tensor.untyped_storage().data_ptr()),
                            length=storage_bytes,
                            checksum="crc32+sha256",
                        )
                        for tensor, _, offset, storage_bytes in layout
                    ],
                    backend=capture_backend,
                    chunk_bytes=chunk_bytes,
                    queue_depth=queue_depth,
                    cuda_device=int(layout[0][0].device.index or 0),
                    file_sha256=True,
                )
                if (
                    capture_result.metrics.bytes
                    != sum(item[3] for item in layout)
                    or capture_result.metrics.file_bytes != total_bytes
                    or capture_result.file_sha256 is None
                ):
                    raise ArtifactError("native capture returned an invalid byte contract")
                capture_backend = capture_result.metrics.backend
                for (_, refs, offset, storage_bytes), digest in zip(
                    layout, capture_result.digests, strict=True
                ):
                    if digest.crc32 is None or digest.sha256 is None:
                        raise ArtifactError("native capture omitted requested digests")
                    entries.append(
                        {
                            "offset": offset,
                            "nbytes": storage_bytes,
                            "crc32": digest.crc32,
                            "sha256": digest.sha256,
                            "refs": refs,
                        }
                    )
                blob_sha256 = capture_result.file_sha256
            else:
                stage_size = min(
                    chunk_bytes,
                    max(int(t.untyped_storage().nbytes()) for t, _ in groups),
                )
                try:
                    stage = torch.empty(
                        stage_size, dtype=torch.uint8, device="cpu", pin_memory=True
                    )
                except RuntimeError:
                    stage = torch.empty(stage_size, dtype=torch.uint8, device="cpu")
                stage_bytes = memoryview(stage.numpy()).cast("B")
                blob_digest = hashlib.sha256()
                written_position = 0
                with temporary_blob.open("w+b", buffering=0) as stream:
                    if hasattr(os, "posix_fallocate"):
                        os.posix_fallocate(stream.fileno(), 0, total_bytes)
                    for tensor, refs, offset, storage_bytes in layout:
                        if offset != written_position:
                            padding = bytes(offset - written_position)
                            stream.seek(written_position)
                            stream.write(padding)
                            blob_digest.update(padding)
                        written_position = offset
                        raw = _storage_view(tensor)
                        storage_digest = hashlib.sha256()
                        crc = 0
                        copied = 0
                        stream.seek(offset)
                        while copied < storage_bytes:
                            count = min(stage_size, storage_bytes - copied)
                            stage[:count].copy_(
                                raw[copied : copied + count], non_blocking=False
                            )
                            view = stage_bytes[:count]
                            stream.write(view)
                            storage_digest.update(view)
                            blob_digest.update(view)
                            crc = zlib.crc32(view, crc)
                            copied += count
                        entries.append(
                            {
                                "offset": offset,
                                "nbytes": storage_bytes,
                                "crc32": f"{crc & 0xFFFFFFFF:08x}",
                                "sha256": storage_digest.hexdigest(),
                                "refs": refs,
                            }
                        )
                        written_position += storage_bytes
                    stream.flush()
                    os.fsync(stream.fileno())
                blob_sha256 = blob_digest.hexdigest()

            manifest = {
                "format": ARTIFACT_FORMAT,
                "kind": ARTIFACT_KIND,
                "identity": identity,
                "identity_sha256": hashlib.sha256(_canonical_json(identity)).hexdigest(),
                "blob": self.blob_path.name,
                "blob_bytes": position,
                "blob_sha256": blob_sha256,
                "storages": entries,
            }
            with temporary_manifest.open("wb") as stream:
                stream.write(_canonical_json(manifest))
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_blob, self.blob_path)
            os.replace(temporary_manifest, self.manifest_path)
            _sync_directory(self.root)
            return ArtifactMetrics(
                operation="capture",
                backend=capture_backend,
                bytes=sum(int(entry["nbytes"]) for entry in entries),
                storages=len(entries),
                seconds=time.perf_counter() - started,
            )
        finally:
            for path in (temporary_blob, temporary_manifest):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            self._unlock(descriptor)

    @staticmethod
    def _validate_ref(saved: dict[str, Any], tensor: Any, storage_pointer: int) -> None:
        current = _tensor_reference(saved["name"], tensor, storage_pointer)
        mismatches = _identity_mismatches(saved, current, f"tensor[{saved['name']!r}]")
        if mismatches:
            raise ArtifactError("tensor layout mismatch: " + "; ".join(mismatches))

    def _resolve_storages(
        self, model: Any, entries: Iterable[dict[str, Any]]
    ) -> list[tuple[Any, dict[str, Any]]]:
        tensors = _registered_cuda_tensors(model)
        saved_names: set[str] = set()
        resolved: list[tuple[Any, dict[str, Any]]] = []
        used_storages: set[tuple[int, int, int]] = set()
        for entry in entries:
            refs = entry.get("refs") or []
            if not refs:
                raise ArtifactError("artifact storage has no semantic references")
            first_name = refs[0].get("name")
            if first_name not in tensors:
                raise ArtifactError(f"fresh model is missing tensor {first_name!r}")
            first_tensor = tensors[first_name]
            storage = first_tensor.untyped_storage()
            pointer = int(storage.data_ptr())
            nbytes = int(storage.nbytes())
            key = (pointer, nbytes, int(first_tensor.device.index or 0))
            if key in used_storages:
                raise ArtifactError(f"multiple artifact storages resolve to {first_name!r}")
            used_storages.add(key)
            if nbytes != int(entry.get("nbytes", -1)):
                raise ArtifactError(
                    f"storage size mismatch for {first_name!r}: "
                    f"expected {entry.get('nbytes')}, found {nbytes}"
                )
            for ref in refs:
                name = ref.get("name")
                if name in saved_names:
                    raise ArtifactError(f"artifact repeats tensor reference {name!r}")
                saved_names.add(name)
                if name not in tensors:
                    raise ArtifactError(f"fresh model is missing tensor {name!r}")
                tensor = tensors[name]
                other_storage = tensor.untyped_storage()
                other_key = (
                    int(other_storage.data_ptr()),
                    int(other_storage.nbytes()),
                    int(tensor.device.index or 0),
                )
                if other_key != key:
                    raise ArtifactError(f"tensor alias layout changed for {name!r}")
                self._validate_ref(ref, tensor, pointer)
            resolved.append((first_tensor, entry))
        extra = sorted(set(tensors) - saved_names)
        if extra:
            raise ArtifactError(f"fresh model has uncaptured CUDA tensors: {extra[:8]}")
        return resolved

    def restore(
        self,
        model: Any,
        *,
        identity: dict[str, Any],
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        queue_depth: int = 4,
        preverified: bool = False,
        preverified_marker: str | os.PathLike[str] | None = None,
        blob_path: str | os.PathLike[str] | None = None,
        blob_offset: int = 0,
    ) -> ArtifactMetrics:
        import torch

        if chunk_bytes <= 0 or chunk_bytes % ALIGNMENT:
            raise ValueError("restore chunk size must be positive and 4096-byte aligned")
        started = time.perf_counter()
        descriptor = self._lock(exclusive=False)
        try:
            selected_blob = self.blob_path if blob_path is None else Path(blob_path)
            manifest = self.load_manifest(
                identity, blob_path=blob_path, blob_offset=blob_offset
            )
            if preverified and blob_path is None:
                self.verify_preverified(manifest, preverified_marker)
            resolved = self._resolve_storages(model, manifest["storages"])
            before = [
                (int(t.untyped_storage().data_ptr()), int(t.untyped_storage().nbytes()))
                for t, _ in resolved
            ]
            native = hydrator_from_env()
            backend = "python"
            if native is not None:
                hydrator, backend = native
                extents = [
                    HydrationExtent(
                        file_offset=blob_offset + int(entry["offset"]),
                        destination=int(tensor.untyped_storage().data_ptr()),
                        length=int(entry["nbytes"]),
                        crc32=entry.get("crc32"),
                        sha256=entry.get("sha256"),
                    )
                    for tensor, entry in resolved
                ]
                backend, restored_bytes = _hydrate_native_extents(
                    hydrator,
                    selected_blob,
                    extents,
                    backend=backend,
                    chunk_bytes=chunk_bytes,
                    queue_depth=queue_depth,
                    cuda_device=int(resolved[0][0].device.index or 0),
                    preverified=preverified,
                )
            else:
                max_storage = max(int(entry["nbytes"]) for _, entry in resolved)
                stage_size = min(chunk_bytes, max_storage)
                try:
                    stage = torch.empty(
                        stage_size, dtype=torch.uint8, device="cpu", pin_memory=True
                    )
                except RuntimeError:
                    stage = torch.empty(stage_size, dtype=torch.uint8, device="cpu")
                stage_bytes = memoryview(stage.numpy()).cast("B")
                restored_bytes = 0
                with selected_blob.open("rb", buffering=0) as stream:
                    for tensor, entry in resolved:
                        raw = _storage_view(tensor)
                        remaining = int(entry["nbytes"])
                        copied = 0
                        digest = hashlib.sha256()
                        crc = 0
                        stream.seek(blob_offset + int(entry["offset"]))
                        while copied < remaining:
                            count = min(stage_size, remaining - copied)
                            view = stage_bytes[:count]
                            read = stream.readinto(view)
                            if read != count:
                                raise ArtifactError(
                                    "artifact blob ended while reading storage at "
                                    f"offset {entry['offset']}"
                                )
                            if not preverified:
                                digest.update(view)
                                crc = zlib.crc32(view, crc)
                            raw[copied : copied + count].copy_(stage[:count], non_blocking=False)
                            copied += count
                        if not preverified:
                            if f"{crc & 0xFFFFFFFF:08x}" != entry.get("crc32"):
                                raise ArtifactError("artifact storage CRC32 mismatch")
                            if digest.hexdigest() != entry.get("sha256"):
                                raise ArtifactError("artifact storage SHA-256 mismatch")
                        restored_bytes += remaining
                torch.cuda.synchronize(resolved[0][0].device)

            after = [
                (int(t.untyped_storage().data_ptr()), int(t.untyped_storage().nbytes()))
                for t, _ in resolved
            ]
            if after != before:
                raise ArtifactError("hydration changed graph-visible tensor storage")
            return ArtifactMetrics(
                operation="restore",
                backend=backend,
                bytes=restored_bytes,
                storages=len(resolved),
                seconds=time.perf_counter() - started,
            )
        finally:
            self._unlock(descriptor)

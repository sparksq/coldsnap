# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Disk-backed vLLM sleep backend for node-local live-process hibernation.

The Python process, CUDA context, compiled code, and NCCL communicators stay
alive. By default, weight allocations are copied to a local disk blob before
their CUDA physical memory is released, then mapped at the same virtual
addresses and copied back during resume. Recovery-aware mode records the stable
allocation map plus its non-weight residual bytes, then repopulates model
parameters from a pinned safetensors revision.

Regions whose declared policy is ``DISCARD`` are unmapped without being copied
anywhere. Their virtual addresses are retained and remapped during resume, so
they cost no blob bytes, no capture I/O, and no restore I/O. This keeps
discardable state such as the KV cache out of any surrounding process or CUDA
snapshot instead of paying to preserve bytes the engine has already declared
worthless.
"""

from __future__ import annotations

import errno
import fcntl
import gc
import hashlib
import json
import os
import socket
import stat
import tempfile
import threading
import time
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from coldsnap_core.hydration import (
    CaptureExtent,
    HydrationExtent,
    capture_from_env,
    hydrator_from_env,
)
from coldsnap_core.validation import (
    PayloadValidationError,
    publish_validation_record,
    validate_payload,
    validation_record_path,
)
from coldsnap_core.topology import (
    execution_graph,
    global_rank as topology_global_rank,
    worker_id as topology_worker_id,
)
from coldsnap_core.memory import (
    STANDARD_REGIONS,
    HostStage,
    PreservationPolicy,
    SnapshotMemoryProvider,
)
from coldsnap_vllm_memory import VllmMemoryProvider


GIB = 1024**3
DEFAULT_CHUNK_BYTES = 256 * 1024**2
DEFAULT_STATE_PATH = "/run/coldsnap/hibernate-state.json"
FLUSH_BYTES = GIB
FORMAT = 3
KIND = "vllm-live-process-cumem"
STATE_FORMAT = 1
STATE_KIND = "coldsnap-live-process-hibernation-state"
# The one region this backend preserves byte-for-byte through a disk blob.
PRESERVED_REGION = "weights"
# Discard/remap is opt-in until each target engine/runtime combination has
# passed activation qualification. The semantic region policy says what may be
# discarded; it does not make doing so safe by default.
DEFAULT_DISCARD_REGIONS = ""
DEFAULT_DISCARD_VERIFY_SAMPLE_BYTES = 64
WEIGHT_SOURCE_ENV = "COLDSNAP_RECOVERY_WEIGHT_SOURCE"
WEIGHT_SOURCE_BLOB = "blob"
WEIGHT_SOURCE_SAFETENSORS = "safetensors"
WEIGHT_SOURCE_SPLIT_NATIVE = "split-native"
MODEL_ID_ENV = "COLDSNAP_MODEL_ID"
MODEL_REVISION_ENV = "COLDSNAP_MODEL_REVISION"
MODEL_METADATA_ENV = "COLDSNAP_MODEL_METADATA_SHA256"
MODEL_PAYLOAD_ENV = "COLDSNAP_EXPORT_MODEL_PAYLOAD"
MODEL_PAYLOAD_NAME = "model-weights.pack"
NATIVE_RESIDUAL_NAME = "native-residual.blob"
NATIVE_MANIFEST_NAME = "native-manifest.json"
ACTIVATION_PROVIDER_NAME = "activation-provider"
ACTIVATION_DIRECTORY_PREFIX = "activation-directory-"
MODEL_PAYLOAD_CACHE_ROOT = "/var/cache/coldsnap/model-payloads"
MODEL_PAYLOAD_MATERIALIZATION_CONTROL = (
    "/run/coldsnap/model-payload-materialization.json"
)
MODEL_PAYLOAD_MATERIALIZATION_FORMAT = 1
MODEL_PAYLOAD_MATERIALIZATION_KIND = "coldsnap-model-payload-materialization"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _write_all_at(fd: int, view: memoryview, offset: int) -> None:
    written = 0
    while written < len(view):
        count = os.pwritev(fd, [view[written:]], offset + written)
        if count <= 0:
            raise OSError("pwritev returned no data")
        written += count


def _read_exact_at(fd: int, view: memoryview, offset: int) -> None:
    read = 0
    while read < len(view):
        count = os.preadv(fd, [view[read:]], offset + read)
        if count == 0:
            raise EOFError(
                f"snapshot ended at offset {offset + read}; wanted {len(view)} bytes"
            )
        read += count


def _timed_read(fd: int, view: memoryview, offset: int) -> float:
    started = time.perf_counter()
    _read_exact_at(fd, view, offset)
    return time.perf_counter() - started


def _align_up(value: int, alignment: int = 4096) -> int:
    return (value + alignment - 1) // alignment * alignment


def _packed_extent_bytes(extents: list[dict[str, Any]]) -> int:
    return sum(_align_up(int(extent["size"])) for extent in extents)


def _split_direct_hydration_extents(
    extents: list[HydrationExtent], alignment: int = 4096
) -> tuple[list[HydrationExtent], list[HydrationExtent]]:
    """Keep aligned interiors on O_DIRECT and route only edges buffered."""
    direct: list[HydrationExtent] = []
    buffered: list[HydrationExtent] = []
    for extent in extents:
        start = extent.file_offset
        end = start + extent.length
        aligned_start = (start + alignment - 1) // alignment * alignment
        aligned_end = end // alignment * alignment
        if aligned_start >= aligned_end:
            buffered.append(extent)
            continue
        if start < aligned_start:
            buffered.append(
                HydrationExtent(
                    file_offset=start,
                    destination=extent.destination,
                    length=aligned_start - start,
                )
            )
        direct.append(
            HydrationExtent(
                file_offset=aligned_start,
                destination=extent.destination + aligned_start - start,
                length=aligned_end - aligned_start,
            )
        )
        if aligned_end < end:
            buffered.append(
                HydrationExtent(
                    file_offset=aligned_end,
                    destination=extent.destination + aligned_end - start,
                    length=end - aligned_end,
                )
            )
    return direct, buffered


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _drop_cache(fd: int, offset: int, length: int) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)


def _open_blob(path: Path, flags: int, direct: bool) -> tuple[int, bool]:
    direct_flag = getattr(os, "O_DIRECT", 0)
    if direct and direct_flag:
        try:
            return os.open(path, flags | direct_flag, 0o600), True
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
                raise
    return os.open(path, flags, 0o600), False


def _io_direct(name: str, default: str) -> bool:
    mode = os.environ.get(name, default)
    if mode not in {"buffered", "direct"}:
        raise ValueError(f"{name} must be buffered or direct")
    return mode == "direct"


def _preallocate(fd: int, size: int) -> None:
    if size <= 0:
        raise ValueError("snapshot preallocation size must be positive")
    if hasattr(os, "posix_fallocate"):
        try:
            os.posix_fallocate(fd, 0, size)
            return
        except OSError as error:
            if error.errno not in {
                errno.EINVAL,
                errno.ENOSYS,
                errno.EOPNOTSUPP,
                errno.ENOTSUP,
            }:
                raise
    # This fallback establishes the final extent and catches basic filesystem
    # limits. Filesystems with fallocate support still fail ENOSPC above before
    # any CUDA allocation is released.
    os.ftruncate(fd, size)


def _rank() -> int:
    rank = topology_global_rank()
    assert rank is not None
    return rank


def _worker_id() -> str:
    """Resolve this accelerator worker without assuming one process per unit."""
    graph = execution_graph()
    assert graph is not None
    return topology_worker_id(graph)


def _identity() -> str:
    host = os.environ.get("VLLM_HOST_IP", socket.gethostname())
    return (
        f"worker-id-{_worker_id()}-host-{host.replace(':', '_')}"
        f"-rank-{_rank()}-pid-{os.getpid()}"
    )


def _directory_identity() -> str:
    if os.environ.get("COLDSNAP_PROCESS_TEMPLATE_PHASE"):
        return f"worker-id-{_worker_id()}-process-template"
    return _identity()


def _captured_native_identity_matches(
    value: Any, worker_id: str, rank: int, pid: int
) -> bool:
    """Validate a capture-owned n580 directory/manifest identity.

    The portable n580 boundary is restored before the inference engine execs,
    so its hydration directory necessarily retains the capture host and worker
    PID.  The activation locator and provider selector are controller-created;
    this check keeps their portability exception scoped to the same logical
    worker and captured rank.
    """
    if not isinstance(value, str):
        return False
    prefix = f"worker-id-{worker_id}-host-"
    if rank < 0:
        return False
    marker = f"-rank-{rank}-pid-"
    if not value.startswith(prefix):
        return False
    host_and_pid = value[len(prefix) :]
    host, separator, identity_pid = host_and_pid.rpartition(marker)
    return bool(
        host
        and separator
        and identity_pid.isdigit()
        and int(identity_pid) == pid
        and pid > 0
    )


def _captured_process_template_identity_matches(
    value: Any, worker_id: str, expected_rank: int
) -> bool:
    """Validate a capture-owned n580 directory across the exec boundary."""
    if not isinstance(value, str):
        return False
    prefix = f"worker-id-{worker_id}-host-"
    if not value.startswith(prefix):
        return False
    host_rank_pid = value[len(prefix) :]
    host_and_rank, pid_separator, pid = host_rank_pid.rpartition("-pid-")
    host, rank_separator, rank_text = host_and_rank.rpartition("-rank-")
    return bool(
        expected_rank >= 0
        and host
        and rank_separator
        and pid_separator
        and rank_text == str(expected_rank)
        and pid.isdigit()
        and int(pid) > 0
    )


def _activated_portable_native_manifest_matches(
    manifest: dict[str, Any], worker_id: str, expected_rank: int
) -> bool:
    """Admit capture ownership only for a selected portable native payload.

    The restored process template is intentionally exec'd into a new vLLM
    worker. PID and host are portable metadata at this boundary; worker, rank,
    capture, blob, and semantic-layout checks remain authoritative.

    The caller must additionally prove that the restore-local
    ``activation-provider`` selected ``native``.  Unlike the temporary
    ``COLDSNAP_PROCESS_TEMPLATE_RESTORED`` environment marker, that selector
    intentionally survives the final engine exec and is the durable controller
    admission record for later live sleep/wake cycles.
    """
    captured_pid = manifest.get("pid")
    captured_rank = manifest.get("rank")
    identity = manifest.get("identity")
    return (
        expected_rank >= 0
        and manifest.get("weight_source") == WEIGHT_SOURCE_SPLIT_NATIVE
        and manifest.get("portable_model_payload") is True
        and manifest.get("worker_id") == worker_id
        and type(captured_rank) is int
        and captured_rank == expected_rank
        and type(captured_pid) is int
        and captured_pid > 0
        and _captured_native_identity_matches(
            identity, worker_id, captured_rank, captured_pid
        )
    )


def _restored_process_template_recovery_manifest_matches(
    manifest: dict[str, Any],
    worker_id: str,
    expected_rank: int,
    activation_provider: str | None,
) -> bool:
    """Admit a recovery manifest across only the n580 worker exec boundary."""
    captured_pid = manifest.get("pid")
    captured_rank = manifest.get("rank")
    return (
        expected_rank >= 0
        # n580 vLLM recovery deliberately leaves this selector absent; SGLang
        # writes an explicit recovery selector because its startup shim needs
        # it.  An explicit native selection is never a recovery-owner proof.
        and activation_provider in {None, "recovery"}
        and manifest.get("weight_source") == WEIGHT_SOURCE_SAFETENSORS
        and manifest.get("worker_id") == worker_id
        and type(captured_rank) is int
        and captured_rank == expected_rank
        and type(captured_pid) is int
        and captured_pid > 0
        and _captured_native_identity_matches(
            manifest.get("identity"), worker_id, captured_rank, captured_pid
        )
    )


def _snapshot_directory(root: Path) -> Path:
    """Resolve one worker directory across an n580 pre-rank transition."""
    worker_id = _worker_id()
    worker_prefix = f"worker-id-{worker_id}-host-"
    pid_suffix = f"-pid-{os.getpid()}"
    process_template_name = f"worker-id-{worker_id}-process-template"
    locator = root / f"{ACTIVATION_DIRECTORY_PREFIX}{worker_id}"
    if locator.is_file():
        name = locator.read_text(encoding="utf-8").strip()
        selected = root / name
        restored_process_template = (
            os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORED") == "1"
            and _captured_process_template_identity_matches(
                name, worker_id, _rank()
            )
        )
        if (
            not name
            or Path(name).name != name
            or (
                name != process_template_name
                and not restored_process_template
                and (
                    not name.startswith(worker_prefix)
                    or not name.endswith(pid_suffix)
                )
            )
            or not selected.is_dir()
        ):
            raise RuntimeError(
                "activation hydration directory does not belong to the live worker"
            )
        return selected
    preferred = root / _directory_identity()
    if preferred.is_dir() or not os.environ.get("COLDSNAP_PROCESS_TEMPLATE_PHASE"):
        return preferred
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith(worker_prefix)
        and path.name.endswith(pid_suffix)
        and "-rank-" in path.name[len(worker_prefix) : -len(pid_suffix)]
    ) if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    activated = [
        path for path in candidates if (path / ACTIVATION_PROVIDER_NAME).is_file()
    ]
    if len(activated) == 1:
        return activated[0]
    if candidates:
        raise RuntimeError(
            "process-template hydration directory is ambiguous for the live worker"
        )
    return preferred


def _portable_identity_matches(value: Any, worker_id: str, rank: int, pid: int) -> bool:
    """Accept a captured host identity on any compatible restore placement."""
    return (
        rank >= 0
        and isinstance(value, str)
        and value.startswith(f"worker-id-{worker_id}-host-")
        and value.endswith(f"-rank-{rank}-pid-{pid}")
    )


def _partition_semantic_weight_extents(
    allocations: list[Any],
    layout: tuple[tuple[int, int, str], ...],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition a weight pool into semantic model state and residual bytes."""
    if not layout:
        raise RuntimeError(f"{label} has no semantic layout")
    by_allocation: dict[int, list[tuple[int, int, str]]] = {
        int(allocation.pointer): [] for allocation in allocations
    }
    previous_end = 0
    semantic_ids: set[str] = set()
    for pointer, size, semantic_id in sorted(layout):
        if (
            pointer <= 0
            or size <= 0
            or pointer < previous_end
            or len(semantic_id) != 64
            or any(character not in "0123456789abcdef" for character in semantic_id)
            or semantic_id in semantic_ids
        ):
            raise RuntimeError(f"{label} semantic ranges are invalid")
        matches = [
            allocation
            for allocation in allocations
            if allocation.pointer <= pointer
            and pointer + size <= allocation.pointer + allocation.size
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{label} range does not resolve to one weight allocation"
            )
        by_allocation[int(matches[0].pointer)].append(
            (pointer, pointer + size, semantic_id)
        )
        previous_end = pointer + size
        semantic_ids.add(semantic_id)

    model_extents: list[dict[str, Any]] = []
    residual_extents: list[dict[str, Any]] = []
    for allocation in allocations:
        allocation_pointer = int(allocation.pointer)
        cursor = allocation_pointer
        allocation_end = allocation_pointer + int(allocation.size)
        for start, end, semantic_id in sorted(by_allocation[allocation_pointer]):
            if start < cursor or end <= start:
                raise RuntimeError(f"{label} semantic ranges overlap")
            if cursor < start:
                residual_extents.append(
                    {
                        "ptr": cursor,
                        "size": start - cursor,
                        "allocation_ptr": allocation_pointer,
                    }
                )
            model_extents.append(
                {
                    "ptr": start,
                    "size": end - start,
                    "allocation_ptr": allocation_pointer,
                    "semantic_id": semantic_id,
                }
            )
            cursor = end
        if cursor < allocation_end:
            residual_extents.append(
                {
                    "ptr": cursor,
                    "size": allocation_end - cursor,
                    "allocation_ptr": allocation_pointer,
                }
            )
    if not model_extents or not residual_extents:
        raise RuntimeError(f"{label} requires model and residual byte ranges")
    return model_extents, residual_extents


def _relocate_split_native_manifest(
    manifest: dict[str, Any],
    allocations: list[Any],
    model_weight_ranges: tuple[tuple[int, int], ...],
    model_weight_semantics: tuple[tuple[int, int, str], ...] = (),
) -> dict[str, Any]:
    """Relocate a portable model payload onto an equivalent fresh VA layout.

    n610 restores the captured CUDA VA map and normally takes the exact-address
    fast path.  n580 deliberately starts a fresh CUDA context, so an otherwise
    identical weight pool may have different base addresses and CuMem block
    boundaries.  Relocation is admitted only when the ordered model-owned and
    residual byte streams exactly cover an all-weights live pool.  Captured
    extents are never split, preserving their packed-file hash/CRC identities;
    a live range may only coalesce consecutive captured extents.
    """
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        return manifest
    captured = sorted(entries, key=lambda entry: int(entry.get("ptr", -1)))
    live = sorted(allocations, key=lambda allocation: int(allocation.pointer))
    captured_layout = [
        (int(entry.get("ptr", -1)), int(entry.get("size", -1)), entry.get("tag"))
        for entry in captured
    ]
    live_layout = [
        (int(allocation.pointer), int(allocation.size), allocation.tag)
        for allocation in live
    ]
    if captured_layout == live_layout:
        return manifest
    if (
        manifest.get("weight_source") != WEIGHT_SOURCE_SPLIT_NATIVE
        or manifest.get("portable_model_payload") is not True
    ):
        return manifest
    model_extents = manifest.get("model_weight_extents")
    residual_extents = manifest.get("residual_extents")
    if (
        not isinstance(model_extents, list)
        or not model_extents
        or not all(isinstance(extent, dict) for extent in model_extents)
        or not isinstance(residual_extents, list)
        or not residual_extents
        or not all(isinstance(extent, dict) for extent in residual_extents)
    ):
        raise RuntimeError("portable native manifest has invalid extents")

    captured_ranges: dict[int, tuple[int, int]] = {}
    try:
        for entry in captured:
            pointer = int(entry["ptr"])
            size = int(entry["size"])
            if (
                pointer < 0
                or size <= 0
                or entry.get("tag") != PRESERVED_REGION
                or pointer in captured_ranges
            ):
                raise ValueError
            captured_ranges[pointer] = (pointer, pointer + size)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("portable native captured allocations are invalid") from error

    live_ranges: list[tuple[int, int, int]] = []
    try:
        for allocation in live:
            pointer = int(allocation.pointer)
            size = int(allocation.size)
            if pointer < 0 or size <= 0 or allocation.tag != PRESERVED_REGION:
                raise ValueError
            live_ranges.append((pointer, pointer + size, pointer))
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("portable native live allocations are invalid") from error
    if not live_ranges:
        raise RuntimeError("portable native live allocation map is empty")
    for previous, current_range in zip(live_ranges, live_ranges[1:], strict=False):
        if current_range[0] < previous[1]:
            raise RuntimeError("portable native live allocations overlap")

    def containing_live_allocation(pointer: int, size: int) -> int:
        matches = [
            base
            for start, end, base in live_ranges
            if start <= pointer and pointer + size <= end
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "portable native relocated extent crosses a live allocation boundary"
            )
        return matches[0]

    def validate_captured_extent(extent: dict[str, Any]) -> None:
        try:
            pointer = int(extent["ptr"])
            size = int(extent["size"])
            allocation_ptr = int(extent["allocation_ptr"])
            start, end = captured_ranges[allocation_ptr]
            if size <= 0 or pointer < start or pointer + size > end:
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "portable native extent does not belong to a captured allocation"
            ) from error

    captured_coverage: dict[int, list[tuple[int, int]]] = {
        pointer: [] for pointer in captured_ranges
    }
    for extent in [*model_extents, *residual_extents]:
        validate_captured_extent(extent)
        pointer = int(extent["ptr"])
        size = int(extent["size"])
        captured_coverage[int(extent["allocation_ptr"])].append(
            (pointer, pointer + size)
        )
    for allocation_pointer, ranges in captured_coverage.items():
        cursor, end = captured_ranges[allocation_pointer]
        for start, range_end in sorted(ranges):
            if start != cursor or range_end <= start:
                raise RuntimeError(
                    "portable native captured extents do not cover allocations"
                )
            cursor = range_end
        if cursor != end:
            raise RuntimeError(
                "portable native captured extents do not cover allocations"
            )

    def relocate_stream(
        sources: list[dict[str, Any]],
        destinations: list[tuple[int, int]],
        label: str,
    ) -> list[dict[str, Any]]:
        ordered = sorted(enumerate(sources), key=lambda item: int(item[1]["ptr"]))
        destination_index = 0
        destination_offset = 0
        relocated: list[dict[str, Any] | None] = [None] * len(sources)
        for source_index, source in ordered:
            size = int(source["size"])
            while (
                destination_index < len(destinations)
                and destination_offset == destinations[destination_index][1]
            ):
                destination_index += 1
                destination_offset = 0
            if destination_index >= len(destinations):
                raise RuntimeError(
                    f"portable native {label} bytes exceed the live layout"
                )
            destination_pointer, destination_size = destinations[destination_index]
            remaining = destination_size - destination_offset
            if size > remaining:
                raise RuntimeError(
                    f"portable native {label} extent would cross a live range boundary"
                )
            pointer = destination_pointer + destination_offset
            extent = dict(source)
            extent["ptr"] = pointer
            extent["allocation_ptr"] = containing_live_allocation(pointer, size)
            relocated[source_index] = extent
            destination_offset += size
        while (
            destination_index < len(destinations)
            and destination_offset == destinations[destination_index][1]
        ):
            destination_index += 1
            destination_offset = 0
        if destination_index != len(destinations):
            raise RuntimeError(
                f"portable native {label} bytes do not cover the live layout"
            )
        return [extent for extent in relocated if extent is not None]

    live_model = sorted((int(pointer), int(size)) for pointer, size in model_weight_ranges)
    if not live_model or any(pointer < 0 or size <= 0 for pointer, size in live_model):
        raise RuntimeError(
            "portable native model extents differ from the live model layout"
        )
    model_by_allocation: dict[int, list[tuple[int, int]]] = {
        base: [] for _start, _end, base in live_ranges
    }
    for pointer, size in live_model:
        allocation_ptr = containing_live_allocation(pointer, size)
        model_by_allocation[allocation_ptr].append((pointer, pointer + size))

    live_residual: list[tuple[int, int]] = []
    for allocation_start, allocation_end, allocation_ptr in live_ranges:
        cursor = allocation_start
        for start, end in sorted(model_by_allocation[allocation_ptr]):
            if start < cursor or end <= start:
                raise RuntimeError("portable native live model ranges overlap")
            if cursor < start:
                live_residual.append((cursor, start - cursor))
            cursor = end
        if cursor < allocation_end:
            live_residual.append((cursor, allocation_end - cursor))
    if not live_residual:
        raise RuntimeError("portable native live layout has no residual ranges")

    try:
        captured_model_by_semantic_id = {
            str(extent["semantic_id"]): extent for extent in model_extents
        }
        live_model_by_semantic_id = {
            str(semantic_id): (int(pointer), int(size))
            for pointer, size, semantic_id in model_weight_semantics
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "portable native semantic model layout is invalid"
        ) from error
    if (
        len(captured_model_by_semantic_id) != len(model_extents)
        or len(live_model_by_semantic_id) != len(model_weight_semantics)
        or set(captured_model_by_semantic_id) != set(live_model_by_semantic_id)
        or set(live_model_by_semantic_id.values()) != set(live_model)
    ):
        raise RuntimeError(
            "portable native semantic model layout differs from the live model"
        )
    relocated_model: list[dict[str, Any]] = []
    for source in model_extents:
        pointer, size = live_model_by_semantic_id[str(source["semantic_id"])]
        if int(source["size"]) != size:
            raise RuntimeError(
                "portable native semantic model extent size differs from the live model"
            )
        extent = dict(source)
        extent["ptr"] = pointer
        extent["allocation_ptr"] = containing_live_allocation(pointer, size)
        relocated_model.append(extent)
    captured_residual_bytes = sum(int(extent["size"]) for extent in residual_extents)
    live_residual_bytes = sum(size for _pointer, size in live_residual)
    residual_layout_changed = captured_residual_bytes != live_residual_bytes
    if residual_layout_changed:
        relocated_residual = [
            {
                "ptr": pointer,
                "size": size,
                "allocation_ptr": containing_live_allocation(pointer, size),
            }
            for pointer, size in live_residual
        ]
    else:
        relocated_residual = relocate_stream(
            residual_extents, live_residual, "residual"
        )

    relocated = dict(manifest)
    relocated["entries"] = [
        {
            "ptr": int(allocation.pointer),
            "size": int(allocation.size),
            "tag": allocation.tag,
        }
        for allocation in live
    ]
    relocated["model_weight_extents"] = relocated_model
    relocated["residual_extents"] = relocated_residual
    relocated["allocation_bytes"] = sum(
        int(allocation.size) for allocation in live
    )
    relocated["residual_bytes"] = live_residual_bytes
    relocated["relocated_from_captured_va"] = True
    if residual_layout_changed:
        # The model object remains content-addressed and reusable, but padding
        # and derived runtime storage belong to this fresh allocation layout.
        # The caller snapshots those live residual ranges before unmapping.
        relocated["requires_live_residual_snapshot"] = True
    return relocated


def _model_source_identity() -> dict[str, str]:
    model_id = os.environ.get(MODEL_ID_ENV, "").strip()
    revision = os.environ.get(MODEL_REVISION_ENV, "").strip()
    if not model_id or not revision:
        raise RuntimeError(
            "safetensors recovery requires pinned COLDSNAP_MODEL_ID and "
            "COLDSNAP_MODEL_REVISION"
        )
    metadata_sha256 = os.environ.get(MODEL_METADATA_ENV, "").strip()
    if metadata_sha256:
        try:
            valid_digest = (
                len(metadata_sha256) == 64
                and metadata_sha256 == metadata_sha256.lower()
                and len(bytes.fromhex(metadata_sha256)) == 32
            )
        except ValueError:
            valid_digest = False
        if not valid_digest:
            raise RuntimeError(
                "COLDSNAP_MODEL_METADATA_SHA256 must be empty or 64 lowercase "
                "hexadecimal characters"
            )
    return {
        "model_id": model_id,
        "revision": revision,
        "metadata_sha256": metadata_sha256,
    }


def _stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _stat_matches(expected: dict[str, Any], actual: dict[str, int]) -> bool:
    return all(int(expected.get(name, -1)) == value for name, value in actual.items())


def _atomic_json(
    path: Path, value: dict[str, Any], *, mode: int | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        if mode is not None:
            os.fchmod(stream.fileno(), mode)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _chunk_descriptors(
    entries: list[dict[str, Any]], chunk_bytes: int
) -> list[tuple[int, int, int, int]]:
    chunks: list[tuple[int, int, int, int]] = []
    for entry in entries:
        ptr = int(entry["ptr"])
        size = int(entry["size"])
        file_offset = int(entry["offset"])
        copied = 0
        while copied < size:
            count = min(chunk_bytes, size - copied)
            chunks.append((ptr + copied, file_offset + copied, count, ptr))
            copied += count
    return chunks


def _as_memory_provider(value: Any) -> SnapshotMemoryProvider:
    if callable(getattr(value, "allocations", None)):
        return value
    return VllmMemoryProvider(value)


def _has_asleep_weights(provider: Any) -> bool:
    memory = _as_memory_provider(provider)
    return any(
        allocation.is_released for allocation in memory.allocations(PRESERVED_REGION)
    )


def _has_asleep_regions(provider: Any, regions: tuple[str, ...]) -> bool:
    memory = _as_memory_provider(provider)
    return any(
        allocation.is_released
        for region in regions
        for allocation in memory.allocations(region)
    )


def _discard_regions(value: str | None = None) -> tuple[str, ...]:
    """Resolve the discardable regions this backend unmaps without preserving.

    Names must resolve to a standard region whose declared policy is DISCARD.
    Requiring the policy rather than a bare name keeps the decision to drop
    bytes owned by the engine-neutral region contract, so a caller cannot
    silently discard something the engine expects to survive.
    """
    raw = os.environ.get("COLDSNAP_DISCARD_REGIONS") if value is None else value
    if raw is None:
        raw = DEFAULT_DISCARD_REGIONS
    names = [item.strip() for item in raw.split(",") if item.strip()]
    resolved: list[str] = []
    for name in names:
        region = STANDARD_REGIONS.get(name)
        if region is None:
            raise ValueError(
                f"COLDSNAP_DISCARD_REGIONS names unknown region {name!r}; expected "
                + ", ".join(STANDARD_REGIONS)
            )
        if region.policy is not PreservationPolicy.DISCARD:
            raise ValueError(
                f"region {name!r} declares policy {region.policy.value!r}; only "
                "regions declared DISCARD may be dropped"
            )
        if not region.stable_address:
            raise ValueError(
                f"region {name!r} does not promise stable addresses and cannot be "
                "remapped after release"
            )
        if name in resolved:
            raise ValueError(f"duplicate discard region {name!r}")
        resolved.append(name)
    return tuple(resolved)


class DiskCuMemBackend:
    """Disk hibernation for allocations in vLLM's tagged weight pool."""

    # Class-level default so instances built with ``__new__`` (tests, and any
    # future rehydration path) never see a missing attribute. A tuple is safe to
    # share; the mutable companion below is created per instance on demand.
    discard_regions: tuple[str, ...] = ()
    discard_verify_sample_bytes = DEFAULT_DISCARD_VERIFY_SAMPLE_BYTES
    weight_recovery_source = WEIGHT_SOURCE_BLOB
    reuse_blob = False
    _weight_recovery_callback: Any | None = None
    _model_weight_ranges: tuple[tuple[int, int], ...] = ()
    _model_weight_semantics: tuple[tuple[int, int, str], ...] = ()
    _native_model_payload_semantics: tuple[tuple[int, int, str], ...] = ()
    _model_payload_exported = False
    _reusable_generation: str | None = None
    _reusable_blob_stat: dict[str, int] | None = None
    _live_native_manifest: dict[str, Any] | None = None
    _live_native_residual_path: Path | None = None
    capture_transport: Any | None = None
    capture_backend = "python"

    @property
    def _discard_mapped(self) -> set[int]:
        """Addresses remapped by an in-flight restore of a discardable region."""
        mapped = self.__dict__.get("_discard_mapped_addresses")
        if mapped is None:
            mapped = set()
            self.__dict__["_discard_mapped_addresses"] = mapped
        return mapped

    def __init__(
        self,
        memory_provider: SnapshotMemoryProvider | None = None,
        graph_controller: Any | None = None,
    ) -> None:
        super().__init__()
        self.memory_provider = memory_provider or VllmMemoryProvider()
        if graph_controller is None:
            from coldsnap_vllm_graphs import get_graph_controller

            graph_controller = get_graph_controller()
        self.graph_controller = graph_controller
        self.memory_provider.capabilities.require(
            "external_snapshot_store", "attach the ColdSnap snapshot store"
        )
        root = os.environ.get("COLDSNAP_DISK_SLEEP_DIR")
        if not root:
            raise RuntimeError("COLDSNAP_DISK_SLEEP_DIR must name a local disk mount")
        self.snapshot_dir = _snapshot_directory(Path(root))
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.blob_path = self.snapshot_dir / "weights.blob"
        self.residual_blob_path = self.blob_path
        self.manifest_path = self.snapshot_dir / "manifest.json"
        state_directory = os.environ.get("COLDSNAP_HIBERNATE_STATE_DIR")
        if state_directory:
            self.state_path = Path(state_directory) / f"{_worker_id()}.json"
        else:
            self.state_path = Path(
                os.environ.get("COLDSNAP_HIBERNATE_STATE_PATH", DEFAULT_STATE_PATH)
            )
        self.chunk_bytes = int(
            os.environ.get("COLDSNAP_DISK_SLEEP_CHUNK_BYTES", DEFAULT_CHUNK_BYTES)
        )
        if self.chunk_bytes <= 0 or self.chunk_bytes % 4096:
            raise ValueError("COLDSNAP_DISK_SLEEP_CHUNK_BYTES must be 4096-byte aligned")
        self.read_direct = _io_direct(
            "COLDSNAP_HIBERNATE_READ_MODE", "direct"
        )
        self.write_direct = _io_direct(
            "COLDSNAP_HIBERNATE_WRITE_MODE", "direct"
        )
        self.reuse_blob = _env_bool("COLDSNAP_HIBERNATE_REUSE_BLOB", True)
        self.weight_recovery_source = os.environ.get(
            WEIGHT_SOURCE_ENV, WEIGHT_SOURCE_BLOB
        )
        if self.weight_recovery_source not in {
            WEIGHT_SOURCE_BLOB,
            WEIGHT_SOURCE_SAFETENSORS,
        }:
            raise ValueError(
                f"{WEIGHT_SOURCE_ENV} must be {WEIGHT_SOURCE_BLOB!r} or "
                f"{WEIGHT_SOURCE_SAFETENSORS!r}"
            )
        if self.weight_recovery_source == WEIGHT_SOURCE_SAFETENSORS:
            self.reuse_blob = False
        self.verify_mode = os.environ.get(
            "COLDSNAP_HIBERNATE_VERIFY_MODE",
            "inline",
        )
        if self.verify_mode not in {"inline", "preverified"}:
            raise ValueError(
                "COLDSNAP_HIBERNATE_VERIFY_MODE must be inline or preverified"
            )
        self.pipeline_depth = int(
            os.environ.get("COLDSNAP_HIBERNATE_PIPELINE_DEPTH", "2")
        )
        if self.pipeline_depth < 1 or self.pipeline_depth > 4:
            raise ValueError("COLDSNAP_HIBERNATE_PIPELINE_DEPTH must be between 1 and 4")
        native_hydration = hydrator_from_env()
        if native_hydration is None:
            self.hydrator = None
            self.hydration_backend = "python"
        else:
            self.hydrator, self.hydration_backend = native_hydration
        native_capture = capture_from_env()
        if native_capture is None:
            self.capture_transport = None
            self.capture_backend = "python"
        else:
            self.capture_transport, self.capture_backend = native_capture
        if self.hydration_backend == "gds" and self.verify_mode != "preverified":
            raise ValueError(
                "GDS hydration requires COLDSNAP_HIBERNATE_VERIFY_MODE=preverified"
            )
        self.discard_regions = _discard_regions()
        if PRESERVED_REGION in self.discard_regions:
            raise ValueError(f"{PRESERVED_REGION!r} cannot be a discard region")
        self._stage_cache: list[HostStage] = []
        # Addresses remain logically asleep until a complete, synchronized restore.
        # This set distinguishes an already remapped allocation from restored data.
        self._restore_mapped: set[int] = set()
        self._reusable_generation: str | None = None
        self._reusable_blob_stat: dict[str, int] | None = None
        self._live_native_manifest = None
        self._live_native_residual_path = None
        self._weight_recovery_callback = None
        self._model_weight_ranges = ()
        self._model_weight_semantics = ()
        self._native_model_payload_semantics = ()
        self._model_payload_exported = False
        self._materialization_thread: threading.Thread | None = None
        self._initial_model_payload_materialization_result: dict[str, Any] | None = None
        self._state = "RUNNING"

    @property
    def uses_model_weight_recovery(self) -> bool:
        return self.weight_recovery_source == WEIGHT_SOURCE_SAFETENSORS

    @property
    def uses_native_model_payload(self) -> bool:
        """Report restore-local native selection before the first manifest load.

        A fresh n580 worker starts with the recovery source default.  The
        activation selector is authoritative earlier than ``_load_manifest``,
        so live sleep must consult it before deciding which semantic model
        layout to bind.
        """
        if self.weight_recovery_source == WEIGHT_SOURCE_SPLIT_NATIVE:
            return True
        activation = self.snapshot_dir / ACTIVATION_PROVIDER_NAME
        try:
            return activation.read_text(encoding="utf-8").strip() == "native"
        except FileNotFoundError:
            return False

    @property
    def exports_model_payload(self) -> bool:
        # Model-payload construction is a capture-time, one-shot operation.
        # An n610 CRIU restore retains the captured Python process environment,
        # including COLDSNAP_EXPORT_MODEL_PAYLOAD=1.  Treating that retained
        # value as a live-sleep policy rebuilt, hashed, and preverified the
        # complete model payload on every post-restore sleep.  The completion
        # bit lives in the captured backend object, so later sleeps preserve
        # only the small runtime residual while retaining the already exported
        # immutable model object.
        return self.model_payload_capture_enabled and not bool(
            getattr(self, "_model_payload_exported", False)
        )

    @property
    def model_payload_capture_enabled(self) -> bool:
        """Whether this process was launched to produce a native model pack."""
        return _env_bool(MODEL_PAYLOAD_ENV, False)

    def set_weight_recovery_callback(self, callback: Any | None) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("weight recovery callback must be callable")
        self._weight_recovery_callback = callback

    def set_model_weight_ranges(
        self, ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...]
    ) -> None:
        """Bind CUDA byte ranges that the model-source reload will reconstruct."""
        normalized = sorted((int(pointer), int(size)) for pointer, size in ranges)
        previous_end = 0
        for pointer, size in normalized:
            if pointer <= 0 or size <= 0 or pointer < previous_end:
                raise ValueError("model weight ranges must be positive and disjoint")
            previous_end = pointer + size
        self._model_weight_ranges = tuple(normalized)
        self._model_weight_semantics = tuple(
            (
                pointer,
                size,
                hashlib.sha256(
                    f"ordinal:{index}:size:{size}".encode("utf-8")
                ).hexdigest(),
            )
            for index, (pointer, size) in enumerate(normalized)
        )

    def set_model_weight_semantics(
        self,
        layout: list[tuple[int, int, str]] | tuple[tuple[int, int, str], ...],
    ) -> None:
        """Bind each live range to a stable model-tensor identity."""
        normalized = sorted(
            (int(pointer), int(size), str(semantic_id))
            for pointer, size, semantic_id in layout
        )
        expected = set(self._model_weight_ranges)
        actual = {(pointer, size) for pointer, size, _semantic_id in normalized}
        semantic_ids = [semantic_id for _pointer, _size, semantic_id in normalized]
        if (
            actual != expected
            or len(normalized) != len(expected)
            or len(set(semantic_ids)) != len(semantic_ids)
            or any(
                len(semantic_id) != 64
                or any(character not in "0123456789abcdef" for character in semantic_id)
                for semantic_id in semantic_ids
            )
        ):
            raise ValueError("model weight semantic layout is invalid")
        self._model_weight_semantics = tuple(normalized)

    def set_native_model_payload_semantics(
        self,
        layout: list[tuple[int, int, str]] | tuple[tuple[int, int, str], ...],
    ) -> None:
        """Bind all semantic model state that a native payload must restore."""
        normalized = tuple(
            sorted(
                (int(pointer), int(size), str(semantic_id))
                for pointer, size, semantic_id in layout
            )
        )
        previous_end = 0
        semantic_ids: set[str] = set()
        for pointer, size, semantic_id in normalized:
            if (
                pointer <= 0
                or size <= 0
                or pointer < previous_end
                or len(semantic_id) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in semantic_id
                )
                or semantic_id in semantic_ids
            ):
                raise ValueError("native model payload semantic layout is invalid")
            previous_end = pointer + size
            semantic_ids.add(semantic_id)
        if not normalized:
            raise ValueError("native model payload semantic layout is empty")
        self._native_model_payload_semantics = normalized

    @classmethod
    def is_supported(cls) -> bool:
        return os.name == "posix" and hasattr(os, "preadv") and hasattr(os, "pwritev")

    @classmethod
    def preserves_communicators(cls) -> bool:
        return True

    @classmethod
    def preserves_compiled_artifacts(cls) -> bool:
        return True

    @classmethod
    def preserves_graphs_with_communicators(cls) -> bool:
        return True

    @classmethod
    def supports_durable_storage(cls) -> bool:
        # The blob is durable, but its CUDA virtual addresses belong to this
        # live process and cannot be imported by a newly started process.
        return False

    def _memory(self, value: Any | None = None) -> SnapshotMemoryProvider:
        if value is not None:
            return _as_memory_provider(value)
        return self.memory_provider

    def _graph_state(self) -> dict[str, int | bool]:
        stats = self.graph_controller.stats()
        return {
            "enabled": bool(self.graph_controller.enabled),
            "allocation_count": stats.allocation_count,
            "raw_bytes": stats.raw_bytes,
            "mapped_bytes": stats.mapped_bytes,
            "paused_count": stats.paused_count,
        }

    def _region_inventory(self, provider: Any) -> dict[str, dict[str, Any]]:
        """Per-region allocation counts and bytes, for activation telemetry."""
        memory = self._memory(provider)
        inventory: dict[str, dict[str, Any]] = {}
        for name in (PRESERVED_REGION,) + self.discard_regions:
            region = STANDARD_REGIONS.get(name)
            allocated = memory.allocations(name)
            inventory[name] = {
                "policy": region.policy.value if region is not None else "unknown",
                "allocation_count": len(allocated),
                "bytes": sum(allocation.size for allocation in allocated),
                "released_count": sum(
                    1 for allocation in allocated if allocation.is_released
                ),
            }
        return inventory

    def _selected_discard_allocations(
        self, memory: SnapshotMemoryProvider, region: str
    ) -> tuple[tuple[int, Any], ...]:
        allocated = tuple(
            enumerate(
                sorted(memory.allocations(region), key=lambda item: item.pointer)
            )
        )
        select_allocations = getattr(memory, "select_allocations", None)
        if not callable(select_allocations):
            raise RuntimeError(
                "memory provider cannot select semantic KV payload allocations"
            )
        payload = tuple(select_allocations(region, "payload"))
        by_pointer = {allocation.pointer: index for index, allocation in allocated}
        if not payload:
            raise RuntimeError(
                f"memory provider selected no payload allocations in {region!r}"
            )
        if len({item.pointer for item in payload}) != len(payload):
            raise RuntimeError(
                f"memory provider returned duplicate payload allocations in {region!r}"
            )
        unknown = sorted(
            item.pointer for item in payload if item.pointer not in by_pointer
        )
        if unknown:
            raise RuntimeError(
                f"memory provider returned payload pointers outside {region!r}: "
                + ", ".join(f"{pointer:#x}" for pointer in unknown)
            )
        return tuple(
            (by_pointer[item.pointer], item)
            for item in sorted(payload, key=lambda item: item.pointer)
        )

    def _verify_discard_extent(
        self,
        memory: SnapshotMemoryProvider,
        pointer: int,
        size: int,
        *,
        synchronize: bool = True,
    ) -> dict[str, Any]:
        """Sample freshly mapped pages without reading the entire region."""
        sample_bytes = min(self.discard_verify_sample_bytes, size)
        stage = memory.allocate_host_stage(sample_bytes)
        offsets = sorted(
            {
                0,
                max(0, (size - sample_bytes) // 2),
                max(0, size - sample_bytes),
            }
        )
        samples: list[dict[str, int]] = []
        # Make every failure attributable to this exact extent rather than a
        # previously queued memset or remap.
        if synchronize:
            memory.synchronize()
        for offset in offsets:
            stage.view[:sample_bytes] = b"\xa5" * sample_bytes
            try:
                memory.copy_to_host(stage, pointer + offset, sample_bytes)
            except Exception as error:
                raise RuntimeError(
                    "discard remap readback failed at "
                    f"pointer={pointer:#x} size={size} offset={offset}"
                ) from error
            nonzero = sum(1 for value in stage.view[:sample_bytes] if value != 0)
            if nonzero:
                raise RuntimeError(
                    "discard remap readback was not zero at "
                    f"pointer={pointer:#x} size={size} offset={offset}; "
                    f"nonzero_bytes={nonzero}/{sample_bytes}"
                )
            samples.append({"offset": offset, "bytes": sample_bytes})
        return {"sample_bytes": sample_bytes, "samples": samples}

    def _release_discard_regions(self, provider: Any) -> dict[str, Any]:
        """Unmap discardable regions without copying their contents anywhere.

        The virtual addresses survive, so the engine's tensors keep their
        identity and a later remap restores addressability. Contents are
        intentionally lost; only regions declared DISCARD reach this path.
        """
        memory = self._memory(provider)
        started = time.perf_counter()
        released: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        for region in self.discard_regions:
            region_bytes = 0
            count = 0
            extents: list[dict[str, int]] = []
            selected = self._selected_discard_allocations(memory, region)
            for index, allocation in selected:
                if allocation.is_released:
                    continue
                memory.release(allocation)
                region_bytes += allocation.size
                count += 1
                extents.append(
                    {
                        "index": index,
                        "pointer": int(allocation.pointer),
                        "size": int(allocation.size),
                    }
                )
            released[region] = {
                "allocation_count": count,
                "bytes": region_bytes,
                "selection_policy": "payload",
                "selected_indices": [index for index, _ in selected],
                # Recorded per allocation so a restore can prove it remapped the
                # same extents, not merely the same number of them.
                "extents": sorted(extents, key=lambda item: item["pointer"]),
            }
            total_bytes += region_bytes
        return {
            "regions": released,
            "bytes": total_bytes,
            "unmap_seconds": time.perf_counter() - started,
        }

    def _remap_discard_regions(
        self, provider: Any, regions: tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        """Remap discardable regions at their original virtual addresses.

        No snapshot bytes are read. ColdSnap zeroes the new physical backing;
        the engine then rebuilds its own layout state in
        ``post_kv_cache_wake_up``.
        """
        memory = self._memory(provider)
        started = time.perf_counter()
        remapped: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        total_map_seconds = 0.0
        total_zero_seconds = 0.0
        total_verification_seconds = 0.0
        for region in self.discard_regions if regions is None else regions:
            region_bytes = 0
            count = 0
            map_seconds = 0.0
            zero_seconds = 0.0
            verification_seconds = 0.0
            extents: list[dict[str, int]] = []
            skipped: list[dict[str, int]] = []
            for index, allocation in enumerate(
                sorted(memory.allocations(region), key=lambda item: item.pointer)
            ):
                if not allocation.is_released:
                    # A mapped allocation in a discard region means something
                    # else restored it; record it so the extent set can be
                    # reconciled against what capture released.
                    skipped.append(
                        {
                            "index": index,
                            "pointer": int(allocation.pointer),
                            "size": int(allocation.size),
                        }
                    )
                    continue
                if allocation.pointer not in self._discard_mapped:
                    phase = time.perf_counter()
                    memory.remap(allocation)
                    extent_map_seconds = time.perf_counter() - phase
                    # Fresh pages carry undefined content. Engines disagree on
                    # who clears it, and a sparse attention backend reading
                    # garbage page indices faults the worker, so the region is
                    # defined here before the engine can observe it.
                    fill_zero = getattr(memory, "fill_zero", None)
                    if not callable(fill_zero):
                        raise RuntimeError(
                            f"memory provider cannot zero discard region "
                            f"{region!r}; refusing to publish undefined memory"
                        )
                    phase = time.perf_counter()
                    fill_zero(allocation.pointer, allocation.size)
                    memory.synchronize()
                    extent_zero_seconds = time.perf_counter() - phase
                    self._discard_mapped.add(allocation.pointer)
                else:
                    extent_map_seconds = 0.0
                    extent_zero_seconds = 0.0
                phase = time.perf_counter()
                verification = self._verify_discard_extent(
                    memory,
                    allocation.pointer,
                    allocation.size,
                    synchronize=False,
                )
                extent_verification_seconds = time.perf_counter() - phase
                map_seconds += extent_map_seconds
                zero_seconds += extent_zero_seconds
                verification_seconds += extent_verification_seconds
                region_bytes += allocation.size
                count += 1
                extents.append(
                    {
                        "index": index,
                        "pointer": int(allocation.pointer),
                        "size": int(allocation.size),
                        "map_seconds": extent_map_seconds,
                        "zero_seconds": extent_zero_seconds,
                        "verification_seconds": extent_verification_seconds,
                        "verification": verification,
                    }
                )
            remapped[region] = {
                "allocation_count": count,
                "bytes": region_bytes,
                "map_seconds": map_seconds,
                "zero_seconds": zero_seconds,
                "verification_seconds": verification_seconds,
                "extents": sorted(extents, key=lambda item: item["pointer"]),
                "already_mapped": sorted(
                    skipped, key=lambda item: item["pointer"]
                ),
            }
            total_bytes += region_bytes
            total_map_seconds += map_seconds
            total_zero_seconds += zero_seconds
            total_verification_seconds += verification_seconds
        return {
            "regions": remapped,
            "bytes": total_bytes,
            "map_seconds": total_map_seconds,
            "zero_seconds": total_zero_seconds,
            "verification_seconds": total_verification_seconds,
            "remap_seconds": time.perf_counter() - started,
        }

    def _commit_discard_regions(
        self, provider: Any, regions: tuple[str, ...] | None = None
    ) -> None:
        memory = self._memory(provider)
        for region in self.discard_regions if regions is None else regions:
            for allocation in memory.allocations(region):
                if (
                    allocation.is_released
                    and allocation.pointer not in self._discard_mapped
                ):
                    raise RuntimeError(
                        f"discard region {region!r} allocation {allocation.pointer} "
                        "was not remapped by the restore"
                    )
                allocation.set_released(False)
        self._discard_mapped.clear()

    def _stages(self, count: int) -> list[HostStage]:
        while len(self._stage_cache) < count:
            self._stage_cache.append(
                self._memory().allocate_host_stage(self.chunk_bytes)
            )
        return self._stage_cache[:count]

    def _stage(self) -> HostStage:
        return self._stages(1)[0]

    def _required_bytes(self, provider: Any) -> int:
        memory = self._memory(provider)
        return sum(
            allocation.size for allocation in memory.allocations("weights")
        )

    def _base_state(self, state: str) -> dict[str, Any]:
        return {
            "format": STATE_FORMAT,
            "kind": STATE_KIND,
            "state": state,
            "updated_unix": time.time(),
            "pid": os.getpid(),
            "identity": _identity(),
            "host": os.environ.get("VLLM_HOST_IP", socket.gethostname()),
            "rank": _rank(),
            "worker_id": _worker_id(),
            "capture_id": os.environ.get("COLDSNAP_CAPTURE_ID", ""),
            "verify_mode": self.verify_mode,
            "pipeline_depth": self.pipeline_depth,
            "hydration_backend": self.hydration_backend,
            "capture_backend": self.capture_backend,
            "weight_recovery_source": self.weight_recovery_source,
        }

    def _write_state(self, state: str, **values: Any) -> dict[str, Any]:
        record = self._base_state(state)
        record.update(values)
        _atomic_json(self.state_path, record)
        return record

    def _ensure_space(self, required: int) -> None:
        filesystem = os.statvfs(self.snapshot_dir)
        available = filesystem.f_bavail * filesystem.f_frsize
        reserve = int(os.environ.get("COLDSNAP_DISK_SLEEP_RESERVE_BYTES", GIB))
        if available < required + reserve:
            raise OSError(
                errno.ENOSPC,
                f"snapshot needs {required / GIB:.2f} GiB plus "
                f"{reserve / GIB:.2f} GiB reserve; only {available / GIB:.2f} GiB free",
            )

    def _write_snapshot(self, provider: Any) -> dict[str, Any]:
        memory = self._memory(provider)
        # Provider selection is activation-local, while an n580 process
        # template necessarily retains its capture-time environment after
        # CRIU restore.  In particular, COLDSNAP_EXPORT_MODEL_PAYLOAD remains
        # true in that restored tree.  Admit a staged native generation before
        # consulting those capture-time policy flags; otherwise the bootstrap
        # sleep would replace the admitted pack with placeholder tensor bytes
        # and remove its activation marker immediately before the first wake.
        reused = self._reusable_snapshot(memory)
        if reused is not None:
            return reused
        if self.uses_model_weight_recovery or self.model_payload_capture_enabled:
            return self._write_recovery_manifest(memory)

        required = self._required_bytes(memory)
        # A completed blob belongs only to an awake generation. It is safe to
        # remove before rewriting because the resident weights remain the
        # recovery source until suspend unmaps them.
        self.manifest_path.unlink(missing_ok=True)
        self.blob_path.unlink(missing_ok=True)
        _sync_directory(self.snapshot_dir)
        self._ensure_space(required)
        tmp_blob = self.blob_path.with_name(f".{self.blob_path.name}.{os.getpid()}.tmp")
        entries: list[dict[str, Any]] = []
        file_offset = 0
        phases = {
            "cuda_copy_s": 0.0,
            "checksum_s": 0.0,
            "disk_write_s": 0.0,
            "preallocate_s": 0.0,
            "sync_s": 0.0,
        }
        started = time.perf_counter()
        verification_s = 0.0
        verified = False
        capture_transport = getattr(self, "capture_transport", None)
        if capture_transport is not None:
            allocations = list(memory.allocations("weights"))
            extents: list[CaptureExtent] = []
            for allocation in allocations:
                extents.append(
                    CaptureExtent(
                        file_offset=file_offset,
                        source=int(allocation.pointer),
                        length=int(allocation.size),
                        checksum="crc32",
                    )
                )
                file_offset += int(allocation.size)
            try:
                captured = capture_transport.capture(
                    tmp_blob,
                    extents,
                    backend=self.capture_backend,
                    chunk_bytes=self.chunk_bytes,
                    queue_depth=self.pipeline_depth,
                    cuda_device=int(os.environ.get("LOCAL_RANK", "0")),
                    verify_readback=self.verify_mode == "preverified",
                )
            except Exception:
                tmp_blob.unlink(missing_ok=True)
                raise
            if captured.metrics.bytes != required or file_offset != required:
                tmp_blob.unlink(missing_ok=True)
                raise RuntimeError("snapshot allocation bytes changed during native capture")
            for allocation, extent, digest in zip(
                allocations, extents, captured.digests, strict=True
            ):
                if digest.crc32 is None:
                    tmp_blob.unlink(missing_ok=True)
                    raise RuntimeError("native capture omitted an allocation CRC32")
                entries.append(
                    {
                        "ptr": int(allocation.pointer),
                        "size": int(allocation.size),
                        "tag": allocation.tag,
                        "offset": extent.file_offset,
                        "crc32": digest.crc32,
                    }
                )
            metrics = captured.metrics
            direct = metrics.backend == "direct"
            phases.update(
                {
                    "cuda_copy_s": metrics.cuda_enqueue_s
                    + metrics.cuda_synchronize_s,
                    "checksum_s": metrics.checksum_s,
                    "disk_write_s": metrics.io_service_s,
                    "preallocate_s": metrics.initialization_s,
                    "sync_s": metrics.durability_s,
                    "io_wait_s": metrics.io_wait_s,
                }
            )
            verification_s = metrics.verification_s
            verified = self.verify_mode == "preverified"
        else:
            stage = self._stage()
            stage_view = stage.view
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fd, direct = _open_blob(tmp_blob, flags, self.write_direct)
            flushed_offset = 0
            try:
                try:
                    phase = time.perf_counter()
                    _preallocate(fd, required)
                    phases["preallocate_s"] += time.perf_counter() - phase
                    for allocation in memory.allocations("weights"):
                        ptr = allocation.pointer
                        size = allocation.size
                        crc = 0
                        allocation_offset = file_offset
                        copied = 0
                        while copied < size:
                            count = min(self.chunk_bytes, size - copied)
                            view = stage_view[:count]
                            phase = time.perf_counter()
                            memory.copy_to_host(stage, ptr + copied, count)
                            phases["cuda_copy_s"] += time.perf_counter() - phase
                            phase = time.perf_counter()
                            crc = zlib.crc32(view, crc)
                            phases["checksum_s"] += time.perf_counter() - phase
                            phase = time.perf_counter()
                            _write_all_at(fd, view, file_offset)
                            phases["disk_write_s"] += time.perf_counter() - phase
                            copied += count
                            file_offset += count
                            if not direct and file_offset - flushed_offset >= FLUSH_BYTES:
                                phase = time.perf_counter()
                                os.fdatasync(fd)
                                _drop_cache(
                                    fd, flushed_offset, file_offset - flushed_offset
                                )
                                phases["sync_s"] += time.perf_counter() - phase
                                flushed_offset = file_offset
                        entries.append(
                            {
                                "ptr": int(ptr),
                                "size": size,
                                "tag": allocation.tag,
                                "offset": allocation_offset,
                                "crc32": f"{crc & 0xFFFFFFFF:08x}",
                            }
                        )
                    phase = time.perf_counter()
                    if file_offset != required:
                        raise RuntimeError(
                            "snapshot allocation bytes changed during disk write"
                        )
                    os.fdatasync(fd)
                    if not direct:
                        _drop_cache(fd, flushed_offset, file_offset - flushed_offset)
                    phases["sync_s"] += time.perf_counter() - phase
                finally:
                    os.close(fd)
            except Exception:
                tmp_blob.unlink(missing_ok=True)
                raise

        os.replace(tmp_blob, self.blob_path)
        os.chmod(self.blob_path, 0o400)
        created_unix = time.time()
        generation = f"{os.getpid()}-{time.time_ns()}"
        blob_stat = _stat_identity(self.blob_path)
        if self.verify_mode == "preverified" and capture_transport is None:
            phase = time.perf_counter()
            self._verify_blob(entries, file_offset)
            verification_s = time.perf_counter() - phase
            verified = True
            blob_stat = _stat_identity(self.blob_path)
        manifest = {
            "format": FORMAT,
            "kind": KIND,
            "identity": _identity(),
            "generation": generation,
            "pid": os.getpid(),
            "host": os.environ.get("VLLM_HOST_IP", socket.gethostname()),
            "rank": _rank(),
            "worker_id": _worker_id(),
            "capture_id": os.environ.get("COLDSNAP_CAPTURE_ID", ""),
            "created_unix": created_unix,
            "blob": self.blob_path.name,
            "blob_bytes": file_offset,
            "blob_stat": blob_stat,
            "direct_io": direct,
            "write_io_mode": "direct" if direct else "buffered",
            "blob_reused": False,
            "verify_mode": self.verify_mode,
            "checksum": "crc32",
            "preverified": verified,
            "verification_seconds": verification_s,
            "entries": entries,
            "write_seconds": time.perf_counter() - started,
            "phase_seconds": phases,
        }
        _atomic_json(self.manifest_path, manifest)
        return manifest

    def _write_recovery_manifest(self, provider: Any) -> dict[str, Any]:
        """Serialize only bytes outside reload-owned weights, then bind HF weights."""
        memory = self._memory(provider)
        started = time.perf_counter()
        export_model_payload = self.exports_model_payload
        self.manifest_path.unlink(missing_ok=True)
        self.blob_path.unlink(missing_ok=True)
        model_payload_path = self.snapshot_dir / MODEL_PAYLOAD_NAME
        native_residual_path = self.snapshot_dir / NATIVE_RESIDUAL_NAME
        native_manifest_path = self.snapshot_dir / NATIVE_MANIFEST_NAME
        if export_model_payload:
            model_payload_path.unlink(missing_ok=True)
            native_residual_path.unlink(missing_ok=True)
            native_manifest_path.unlink(missing_ok=True)
        (self.snapshot_dir / ACTIVATION_PROVIDER_NAME).unlink(missing_ok=True)
        _sync_directory(self.snapshot_dir)
        allocations = sorted(
            memory.allocations(PRESERVED_REGION), key=lambda item: item.pointer
        )
        entries = [
            {
                "ptr": int(allocation.pointer),
                "size": int(allocation.size),
                "tag": allocation.tag,
            }
            for allocation in allocations
        ]
        if not entries or any(entry["size"] <= 0 for entry in entries):
            raise RuntimeError("safetensors recovery requires live weight allocations")

        configured = tuple(getattr(self, "_model_weight_ranges", ()))
        if not configured:
            raise RuntimeError(
                "safetensors recovery has no captured model weight ranges"
            )
        semantic_layout = tuple(getattr(self, "_model_weight_semantics", ()))
        if not semantic_layout:
            semantic_layout = tuple(
                (
                    pointer,
                    size,
                    hashlib.sha256(
                        f"ordinal:{index}:size:{size}".encode("utf-8")
                    ).hexdigest(),
                )
                for index, (pointer, size) in enumerate(configured)
            )
        semantic_ids = {
            (int(pointer), int(size)): str(semantic_id)
            for pointer, size, semantic_id in semantic_layout
        }
        if set(semantic_ids) != set(configured):
            raise RuntimeError(
                "safetensors recovery has no semantic model weight layout"
            )
        model_weight_extents, residual_extents = _partition_semantic_weight_extents(
            allocations,
            semantic_layout,
            label="safetensors recovery",
        )

        residual_bytes = sum(int(extent["size"]) for extent in residual_extents)
        allocation_bytes = sum(entry["size"] for entry in entries)
        model_weight_bytes = sum(
            int(extent["size"]) for extent in model_weight_extents
        )
        native_semantic_layout = tuple(
            getattr(self, "_native_model_payload_semantics", ())
        ) or semantic_layout
        native_model_extents, native_residual_extents = (
            _partition_semantic_weight_extents(
                allocations,
                native_semantic_layout,
                label="native model payload",
            )
            if export_model_payload
            else (model_weight_extents, residual_extents)
        )
        separate_native_residual = native_semantic_layout != semantic_layout
        native_residual_bytes = sum(
            int(extent["size"]) for extent in native_residual_extents
        )
        model_payload_bytes = _packed_extent_bytes(native_model_extents)
        self._ensure_space(
            residual_bytes
            + (
                model_payload_bytes
                + (native_residual_bytes if separate_native_residual else 0)
                if export_model_payload
                else 0
            )
        )
        stage = self._stage()
        tmp_blob = self.blob_path.with_name(
            f".{self.blob_path.name}.{os.getpid()}.tmp"
        )
        fd = os.open(tmp_blob, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        blob_offset = 0
        phases = {
            "cuda_copy_s": 0.0,
            "checksum_s": 0.0,
            "disk_write_s": 0.0,
            "preallocate_s": 0.0,
            "sync_s": 0.0,
        }
        try:
            phase = time.perf_counter()
            _preallocate(fd, residual_bytes)
            phases["preallocate_s"] += time.perf_counter() - phase
            for extent in residual_extents:
                extent["offset"] = blob_offset
                crc = 0
                copied = 0
                while copied < int(extent["size"]):
                    count = min(
                        self.chunk_bytes, int(extent["size"]) - copied
                    )
                    phase = time.perf_counter()
                    memory.copy_to_host(
                        stage, int(extent["ptr"]) + copied, count
                    )
                    phases["cuda_copy_s"] += time.perf_counter() - phase
                    view = stage.view[:count]
                    phase = time.perf_counter()
                    crc = zlib.crc32(view, crc)
                    phases["checksum_s"] += time.perf_counter() - phase
                    phase = time.perf_counter()
                    _write_all_at(fd, view, blob_offset)
                    phases["disk_write_s"] += time.perf_counter() - phase
                    copied += count
                    blob_offset += count
                extent["crc32"] = f"{crc & 0xFFFFFFFF:08x}"
            phase = time.perf_counter()
            os.fdatasync(fd)
            _drop_cache(fd, 0, 0)
            phases["sync_s"] += time.perf_counter() - phase
        except Exception:
            os.close(fd)
            tmp_blob.unlink(missing_ok=True)
            raise
        else:
            os.close(fd)
        if blob_offset != residual_bytes:
            tmp_blob.unlink(missing_ok=True)
            raise RuntimeError("residual snapshot byte count changed during write")
        os.replace(tmp_blob, self.blob_path)
        os.chmod(self.blob_path, 0o400)
        blob_stat = _stat_identity(self.blob_path)
        verification_s = 0.0
        preverified = False
        if self.verify_mode == "preverified":
            phase = time.perf_counter()
            self._verify_blob(residual_extents, residual_bytes, direct=False)
            verification_s = time.perf_counter() - phase
            preverified = True
            blob_stat = _stat_identity(self.blob_path)

        if model_weight_bytes + residual_bytes != allocation_bytes:
            raise RuntimeError(
                "model-weight and residual ranges do not cover allocations"
            )
        created_unix = time.time()
        manifest = {
            "format": FORMAT,
            "kind": KIND,
            "identity": _identity(),
            "generation": f"{os.getpid()}-{time.time_ns()}",
            "pid": os.getpid(),
            "host": os.environ.get("VLLM_HOST_IP", socket.gethostname()),
            "rank": _rank(),
            "worker_id": _worker_id(),
            "capture_id": os.environ.get("COLDSNAP_CAPTURE_ID", ""),
            "created_unix": created_unix,
            "weight_source": WEIGHT_SOURCE_SAFETENSORS,
            "model_source": _model_source_identity(),
            "blob": self.blob_path.name,
            "blob_bytes": residual_bytes,
            "blob_stat": blob_stat,
            # OCI/HF distribution necessarily changes filesystem identity.
            # The capsule digest binds the object, and wake verifies every
            # residual extent against its recorded CRC32 before exposing it.
            "portable_residual_blob": True,
            "allocation_bytes": allocation_bytes,
            "model_weight_bytes": model_weight_bytes,
            "residual_bytes": residual_bytes,
            "model_weight_extents": model_weight_extents,
            "residual_extents": residual_extents,
            "direct_io": False,
            "write_io_mode": "hybrid-residual-buffered",
            "blob_reused": False,
            "verify_mode": f"model-revision+residual-{self.verify_mode}",
            "checksum": "crc32",
            "preverified": preverified,
            "verification_seconds": verification_s,
            "entries": entries,
            "write_seconds": time.perf_counter() - started,
            "phase_seconds": phases,
        }
        _atomic_json(self.manifest_path, manifest)
        if export_model_payload:
            native_residual = {
                "path": self.blob_path,
                "bytes": residual_bytes,
                "stat": blob_stat,
                "extents": residual_extents,
                "preverified": preverified,
            }
            if separate_native_residual:
                native_residual = self._write_buffered_extent_blob(
                    memory,
                    native_residual_extents,
                    native_residual_path,
                )
            manifest["model_payload"] = self._write_model_payload(
                memory,
                native_model_extents,
                model_payload_path,
                native_manifest_path,
                manifest,
                native_residual,
            )
            manifest["model_payload_exported"] = True
            manifest["write_seconds"] = time.perf_counter() - started
            _atomic_json(self.manifest_path, manifest)
            self._model_payload_exported = True
        return manifest

    def _write_buffered_extent_blob(
        self,
        memory: Any,
        source_extents: list[dict[str, Any]],
        path: Path,
    ) -> dict[str, Any]:
        """Write one portable residual object without changing its semantics."""
        extents = [dict(extent) for extent in source_extents]
        required = sum(int(extent["size"]) for extent in extents)
        if required <= 0:
            raise RuntimeError("native residual payload is empty")
        stage = self._stage()
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        offset = 0
        started = time.perf_counter()
        phases = {
            "cuda_copy_s": 0.0,
            "checksum_s": 0.0,
            "disk_write_s": 0.0,
            "preallocate_s": 0.0,
            "sync_s": 0.0,
        }
        try:
            phase = time.perf_counter()
            _preallocate(fd, required)
            phases["preallocate_s"] += time.perf_counter() - phase
            for extent in extents:
                extent["offset"] = offset
                crc = 0
                copied = 0
                while copied < int(extent["size"]):
                    count = min(self.chunk_bytes, int(extent["size"]) - copied)
                    phase = time.perf_counter()
                    memory.copy_to_host(stage, int(extent["ptr"]) + copied, count)
                    phases["cuda_copy_s"] += time.perf_counter() - phase
                    view = stage.view[:count]
                    phase = time.perf_counter()
                    crc = zlib.crc32(view, crc)
                    phases["checksum_s"] += time.perf_counter() - phase
                    phase = time.perf_counter()
                    _write_all_at(fd, view, offset)
                    phases["disk_write_s"] += time.perf_counter() - phase
                    copied += count
                    offset += count
                extent["crc32"] = f"{crc & 0xFFFFFFFF:08x}"
            phase = time.perf_counter()
            os.fdatasync(fd)
            _drop_cache(fd, 0, 0)
            phases["sync_s"] += time.perf_counter() - phase
        except Exception:
            os.close(fd)
            temporary.unlink(missing_ok=True)
            raise
        else:
            os.close(fd)
        if offset != required:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("native residual payload byte count changed during write")
        os.replace(temporary, path)
        os.chmod(path, 0o400)
        verification_s = 0.0
        preverified = False
        if self.verify_mode == "preverified":
            phase = time.perf_counter()
            self._verify_blob(extents, required, direct=False, path=path)
            verification_s = time.perf_counter() - phase
            preverified = True
        return {
            "path": path,
            "bytes": required,
            "stat": _stat_identity(path),
            "extents": extents,
            "preverified": preverified,
            "verification_seconds": verification_s,
            "write_seconds": time.perf_counter() - started,
            "phase_seconds": phases,
        }

    def _write_model_payload(
        self,
        memory: Any,
        model_weight_extents: list[dict[str, Any]],
        payload_path: Path,
        manifest_path: Path,
        recovery_manifest: dict[str, Any],
        native_residual: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Write the model-owned bytes as a shareable external payload.

        Driver/runtime residual bytes stay in ``weights.blob`` and both address
        maps stay in the capsule. The external object's identity consequently
        depends only on the model bytes, allowing compatible driver captures
        with different allocation orders to converge on the same content
        address.
        """
        stage = self._stage()
        fingerprint_started = time.perf_counter()
        canonical_extents: list[dict[str, Any]] = []
        for source_extent in model_weight_extents:
            extent = dict(source_extent)
            digest = hashlib.sha256()
            copied = 0
            while copied < int(extent["size"]):
                count = min(self.chunk_bytes, int(extent["size"]) - copied)
                memory.copy_to_host(stage, int(extent["ptr"]) + copied, count)
                digest.update(stage.view[:count])
                copied += count
            extent["sha256"] = digest.hexdigest()
            canonical_extents.append(extent)
        canonical_extents.sort(key=lambda extent: (extent["sha256"], extent["size"]))
        model_bytes = sum(int(extent["size"]) for extent in canonical_extents)
        required = _packed_extent_bytes(canonical_extents)
        temporary = payload_path.with_name(
            f".{payload_path.name}.{os.getpid()}.tmp"
        )
        fd, direct = _open_blob(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            self.write_direct,
        )
        entries: list[dict[str, Any]] = []
        offset = 0
        payload_digest = hashlib.sha256()
        phases = {
            "cuda_copy_s": 0.0,
            "checksum_s": 0.0,
            "disk_write_s": 0.0,
            "preallocate_s": 0.0,
            "sync_s": 0.0,
            "canonical_fingerprint_s": time.perf_counter() - fingerprint_started,
        }
        try:
            phase = time.perf_counter()
            _preallocate(fd, required)
            phases["preallocate_s"] += time.perf_counter() - phase
            for source_extent in canonical_extents:
                extent = dict(source_extent)
                extent_offset = offset
                copied = 0
                crc = 0
                digest = hashlib.sha256()
                while copied < int(extent["size"]):
                    count = min(self.chunk_bytes, int(extent["size"]) - copied)
                    phase = time.perf_counter()
                    memory.copy_to_host(
                        stage,
                        int(extent["ptr"]) + copied,
                        count,
                    )
                    phases["cuda_copy_s"] += time.perf_counter() - phase
                    view = stage.view[:count]
                    phase = time.perf_counter()
                    crc = zlib.crc32(view, crc)
                    digest.update(view)
                    phases["checksum_s"] += time.perf_counter() - phase
                    phase = time.perf_counter()
                    write_count = count
                    if count % 4096:
                        write_count = _align_up(count)
                        stage.view[count:write_count] = b"\0" * (write_count - count)
                    payload_digest.update(stage.view[:write_count])
                    _write_all_at(fd, stage.view[:write_count], offset)
                    phases["disk_write_s"] += time.perf_counter() - phase
                    copied += count
                    offset += write_count
                if digest.hexdigest() != extent["sha256"]:
                    raise RuntimeError(
                        "model payload extent changed during canonical write"
                    )
                entries.append(
                    {
                        **extent,
                        "offset": extent_offset,
                        "crc32": f"{crc & 0xFFFFFFFF:08x}",
                    }
                )
            phase = time.perf_counter()
            if offset != required:
                raise RuntimeError("model payload byte count changed during write")
            os.fdatasync(fd)
            if not direct:
                _drop_cache(fd, 0, 0)
            phases["sync_s"] += time.perf_counter() - phase
        except Exception:
            os.close(fd)
            temporary.unlink(missing_ok=True)
            raise
        else:
            os.close(fd)
        os.replace(temporary, payload_path)
        os.chmod(payload_path, 0o400)
        payload_stat = _stat_identity(payload_path)
        verification_s = 0.0
        preverified = False
        if self.verify_mode == "preverified":
            phase = time.perf_counter()
            self._verify_blob(
                entries,
                required,
                direct=False,
                path=payload_path,
                expected_verified_bytes=model_bytes,
            )
            verification_s = time.perf_counter() - phase
            preverified = True
            payload_stat = _stat_identity(payload_path)
        payload_seconds = time.perf_counter() - fingerprint_started
        payload_record = {
            "blob": payload_path.name,
            "manifest": manifest_path.name,
            "bytes": required,
            "sha256": "sha256:" + payload_digest.hexdigest(),
            "blob_stat": payload_stat,
            "write_seconds": payload_seconds,
            "verification_seconds": verification_s,
            "phase_seconds": phases,
        }
        if native_residual is None:
            return payload_record
        residual_extents = native_residual.get("extents")
        residual_path = native_residual.get("path")
        if (
            not isinstance(residual_extents, list)
            or not residual_extents
            or not isinstance(residual_path, Path)
            or not residual_path.is_file()
        ):
            raise RuntimeError("native residual payload record is invalid")
        residual_bytes = sum(int(extent["size"]) for extent in residual_extents)
        if model_bytes + residual_bytes != int(recovery_manifest["allocation_bytes"]):
            raise RuntimeError("native model and residual ranges do not cover allocations")
        native_manifest = {
            "format": FORMAT,
            "kind": KIND,
            "identity": _identity(),
            "generation": f"{os.getpid()}-{time.time_ns()}",
            "pid": os.getpid(),
            "host": os.environ.get("VLLM_HOST_IP", socket.gethostname()),
            "rank": _rank(),
            "worker_id": _worker_id(),
            "capture_id": os.environ.get("COLDSNAP_CAPTURE_ID", ""),
            "created_unix": time.time(),
            "weight_source": WEIGHT_SOURCE_SPLIT_NATIVE,
            "blob": payload_path.name,
            "blob_bytes": required,
            "blob_stat": payload_stat,
            "model_blob": payload_path.name,
            "model_blob_bytes": required,
            "model_blob_stat": payload_stat,
            "residual_blob": residual_path.name,
            "residual_blob_bytes": int(native_residual["bytes"]),
            "residual_blob_stat": native_residual["stat"],
            "allocation_bytes": recovery_manifest["allocation_bytes"],
            "model_weight_bytes": model_bytes,
            "residual_bytes": residual_bytes,
            "model_weight_extents": entries,
            "residual_extents": residual_extents,
            "direct_io": direct,
            "write_io_mode": "padded-direct" if direct else "padded-buffered",
            "blob_reused": False,
            "verify_mode": self.verify_mode,
            "checksum": "crc32",
            "preverified": preverified and native_residual.get("preverified") is True,
            # Distribution changes inode/mtime. The adapter verifies the
            # staged payload's artifact SHA-256 before it is bind-mounted.
            "portable_model_payload": True,
            "portable_residual_blob": True,
            "verification_seconds": verification_s,
            "entries": recovery_manifest["entries"],
            "write_seconds": payload_seconds,
            "phase_seconds": phases,
        }
        _atomic_json(manifest_path, native_manifest)
        return payload_record

    def _model_payload_materialization(self) -> dict[str, Any] | None:
        control_path = Path(
            os.environ.get(
                "COLDSNAP_MODEL_PAYLOAD_MATERIALIZATION_CONTROL",
                MODEL_PAYLOAD_MATERIALIZATION_CONTROL,
            )
        )
        if not control_path.is_file():
            return None
        try:
            control = json.loads(control_path.read_text(encoding="utf-8"))
            mode = control["mode"]
            operation_id = control["operation_id"]
            owner_uid = control["owner_uid"]
            owner_gid = control["owner_gid"]
            expected = control["workers"][_worker_id()]
            digest = expected["sha256"]
            expected_bytes = expected["bytes"]
            relative_path = expected["path"]
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "model payload materialization control is invalid"
            ) from error
        if (
            control.get("format") != MODEL_PAYLOAD_MATERIALIZATION_FORMAT
            or control.get("kind") != MODEL_PAYLOAD_MATERIALIZATION_KIND
            or not isinstance(mode, str)
            or mode not in {"async", "required"}
            or not isinstance(operation_id, str)
            or not operation_id
            or len(operation_id) > 128
            or any(
                not (character.isalnum() or character in "._-")
                for character in operation_id
            )
            or not isinstance(owner_uid, int)
            or isinstance(owner_uid, bool)
            or owner_uid < 0
            or not isinstance(owner_gid, int)
            or isinstance(owner_gid, bool)
            or owner_gid < 0
            or not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or len(digest) != 71
            or not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes <= 0
        ):
            raise RuntimeError("model payload materialization control is invalid")
        digest_hex = digest.removeprefix("sha256:")
        try:
            if (
                len(bytes.fromhex(digest_hex)) != 32
                or digest_hex != digest_hex.lower()
            ):
                raise ValueError
        except ValueError as error:
            raise RuntimeError(
                "model payload materialization digest is invalid"
            ) from error
        expected_relative = f"sha256/{digest_hex}.pack"
        if relative_path != expected_relative:
            raise RuntimeError(
                "model payload materialization path is not content-addressed"
            )
        cache_root = Path(
            os.environ.get(
                "COLDSNAP_MODEL_PAYLOAD_CACHE_ROOT", MODEL_PAYLOAD_CACHE_ROOT
            )
        )
        if not cache_root.is_absolute():
            raise RuntimeError("model payload cache root must be absolute")
        target = cache_root / expected_relative
        # Status is diagnostic only. Keep one bounded record per worker rather
        # than accumulating one file for every restore operation forever.
        status = cache_root / ".status" / f"{_worker_id()}.json"
        return {
            "mode": mode,
            "operation_id": operation_id,
            "owner_uid": owner_uid,
            "owner_gid": owner_gid,
            "sha256": digest,
            "bytes": expected_bytes,
            "target": target,
            "status": status,
        }

    @staticmethod
    def _cached_model_payload_validation(
        config: dict[str, Any], *, revalidate: bool
    ) -> dict[str, Any] | None:
        target = Path(config["target"])
        try:
            if not target.is_file():
                return None
            if not revalidate:
                marker = validation_record_path(target)
                if not marker.is_file():
                    return None
            return validate_payload(target, config["sha256"], config["bytes"])
        except PayloadValidationError:
            if revalidate:
                raise
            return None
        except (OSError, RuntimeError, ValueError):
            return None

    @classmethod
    def _cached_model_payload_valid(cls, config: dict[str, Any]) -> bool:
        return cls._cached_model_payload_validation(config, revalidate=False) is not None

    @staticmethod
    def _normalize_materialization_ownership(path: Path, config: dict[str, Any]) -> float:
        started = time.perf_counter()
        value = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(value.st_mode):
            raise RuntimeError("model payload cache entry is not a regular file")
        if value.st_uid != config["owner_uid"] or value.st_gid != config["owner_gid"]:
            os.chown(
                path,
                config["owner_uid"],
                config["owner_gid"],
                follow_symlinks=False,
            )
        return time.perf_counter() - started

    @staticmethod
    def _publish_model_payload_validation(
        target: Path, config: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        record = publish_validation_record(
            target,
            config["sha256"],
            config["bytes"],
            evidence="full-sha256-this-operation",
        )
        marker_path = validation_record_path(target)
        marker_stat = marker_path.stat(follow_symlinks=False)
        if marker_stat.st_uid != config["owner_uid"] or marker_stat.st_gid != config["owner_gid"]:
            os.chown(
                marker_path,
                config["owner_uid"],
                config["owner_gid"],
                follow_symlinks=False,
            )
        return record, time.perf_counter() - started

    @staticmethod
    def _write_materialization_status(
        config: dict[str, Any], result: dict[str, Any]
    ) -> None:
        status = Path(config["status"])
        value = dict(result)
        value.setdefault("worker_id", _worker_id())
        value.setdefault("updated_unix_ns", time.time_ns())
        if config.get("started_unix_ns") is not None:
            value.setdefault("started_unix_ns", int(config["started_unix_ns"]))
        _atomic_json(status, value, mode=0o600)
        status_stat = status.stat(follow_symlinks=False)
        if status_stat.st_uid != config["owner_uid"] or status_stat.st_gid != config["owner_gid"]:
            os.chown(
                status,
                config["owner_uid"],
                config["owner_gid"],
                follow_symlinks=False,
            )

    def _materialize_model_payload(
        self,
        memory: Any,
        recovery_manifest: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        target = Path(config["target"])
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = target.with_name(f".{target.name}.lock")
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        started = time.perf_counter()
        config["started_unix_ns"] = time.time_ns()
        phases: dict[str, float] = {}
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if target.is_file():
                phases["ownership_normalization_s"] = self._normalize_materialization_ownership(
                    target, config
                )
                self._write_materialization_status(
                    config,
                    {
                        "state": "revalidation-needed",
                        "operation_id": config["operation_id"],
                        "path": str(target),
                        "sha256": config["sha256"],
                        "bytes": config["bytes"],
                        "seconds": time.perf_counter() - started,
                    },
                )
                validation_started = time.perf_counter()
                try:
                    validation = self._cached_model_payload_validation(
                        config,
                        revalidate=True,
                    )
                except PayloadValidationError as error:
                    validation = None
                    self._write_materialization_status(
                        config,
                        {
                            "state": "corrupt",
                            "operation_id": config["operation_id"],
                            "path": str(target),
                            "sha256": config["sha256"],
                            "bytes": config["bytes"],
                            "reason": error.reason,
                            "error": str(error),
                            "seconds": time.perf_counter() - started,
                        },
                    )
                phases["existing_validation_s"] = time.perf_counter() - validation_started
            else:
                validation = None
            if validation is not None:
                _record, phases["validation_record_s"] = self._publish_model_payload_validation(
                    target, config
                )
                result = {
                    "state": "ready",
                    "operation_id": config["operation_id"],
                    "reused": True,
                    "path": str(target),
                    "sha256": config["sha256"],
                    "bytes": config["bytes"],
                    "validation": validation,
                    "phase_seconds": phases,
                    "seconds": time.perf_counter() - started,
                }
                self._write_materialization_status(config, result)
                return result
            suffix = f"{os.getpid()}-{threading.get_ident()}"
            staging = target.with_name(
                f".{target.name}.{suffix}.materializing"
            )
            generated_manifest = target.with_name(
                f".{target.name}.{suffix}.native-manifest.json"
            )
            for stale in target.parent.glob(
                f".{target.name}.*.materializing"
            ):
                stale.unlink(missing_ok=True)
            for stale in target.parent.glob(
                f".{target.name}.*.native-manifest.json"
            ):
                stale.unlink(missing_ok=True)
            staging.unlink(missing_ok=True)
            generated_manifest.unlink(missing_ok=True)
            try:
                self._write_materialization_status(
                    config,
                    {
                        "state": "writing",
                        "operation_id": config["operation_id"],
                        "path": str(target),
                        "sha256": config["sha256"],
                        "bytes": config["bytes"],
                        "seconds": time.perf_counter() - started,
                    },
                )
                allocations = sorted(
                    memory.allocations(PRESERVED_REGION),
                    key=lambda allocation: allocation.pointer,
                )
                native_layout = tuple(
                    getattr(self, "_native_model_payload_semantics", ())
                ) or tuple(getattr(self, "_model_weight_semantics", ())) or tuple(
                    (
                        int(extent["ptr"]),
                        int(extent["size"]),
                        str(extent["semantic_id"]),
                    )
                    for extent in recovery_manifest["model_weight_extents"]
                )
                native_model_extents, _native_residual_extents = (
                    _partition_semantic_weight_extents(
                        allocations,
                        native_layout,
                        label="materialized native model payload",
                    )
                )
                generated = self._write_model_payload(
                    memory,
                    native_model_extents,
                    staging,
                    generated_manifest,
                    recovery_manifest,
                    None,
                )
                if (
                    generated["bytes"] != config["bytes"]
                    or generated["sha256"] != config["sha256"]
                ):
                    raise RuntimeError(
                        "materialized model payload differs from the captured identity"
                    )
                self._write_materialization_status(
                    config,
                    {
                        "state": "verifying",
                        "operation_id": config["operation_id"],
                        "path": str(target),
                        "sha256": config["sha256"],
                        "bytes": config["bytes"],
                        "seconds": time.perf_counter() - started,
                    },
                )
                os.replace(staging, target)
                _sync_directory(target.parent)
                phases.update(
                    {str(name): float(value) for name, value in generated.get("phase_seconds", {}).items()}
                )
                phases["payload_write_total_s"] = float(generated.get("write_seconds", 0.0))
                phases["payload_verification_s"] = float(generated.get("verification_seconds", 0.0))
                phases["ownership_normalization_s"] = self._normalize_materialization_ownership(
                    target, config
                )
                self._write_materialization_status(
                    config,
                    {
                        "state": "publishing-validation",
                        "operation_id": config["operation_id"],
                        "path": str(target),
                        "sha256": config["sha256"],
                        "bytes": config["bytes"],
                        "seconds": time.perf_counter() - started,
                        "phase_seconds": phases,
                    },
                )
                _record, phases["validation_record_s"] = self._publish_model_payload_validation(
                    target, config
                )
            finally:
                staging.unlink(missing_ok=True)
                generated_manifest.unlink(missing_ok=True)
            result = {
                "state": "ready",
                "operation_id": config["operation_id"],
                "reused": False,
                "path": str(target),
                "sha256": config["sha256"],
                "bytes": config["bytes"],
                "validation": {
                    "provider": "sha256-cache-v1",
                    "decision": "accept",
                    "reason": "validated_new_payload",
                    "content_evidence": "full-sha256-this-operation",
                    "bytes_hashed": config["bytes"],
                },
                "phase_seconds": phases,
                "seconds": time.perf_counter() - started,
            }
            self._write_materialization_status(config, result)
            return result
        except Exception as error:
            self._write_materialization_status(
                config,
                {
                    "state": "failed",
                    "operation_id": config["operation_id"],
                    "path": str(target),
                    "sha256": config["sha256"],
                    "bytes": config["bytes"],
                    "error": str(error),
                    "reason": getattr(error, "reason", "materialization_failed"),
                    "phase_seconds": phases,
                    "seconds": time.perf_counter() - started,
                },
            )
            raise
        finally:
            os.close(lock_fd)

    def _start_model_payload_materialization(
        self,
        memory: Any,
        recovery_manifest: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        if config["mode"] == "required":
            return self._materialize_model_payload(
                memory, recovery_manifest, config
            )

        def materialize() -> None:
            try:
                self._materialize_model_payload(
                    memory, recovery_manifest, config
                )
            except Exception as error:
                from vllm.logger import init_logger

                init_logger(__name__).warning(
                    "ColdSnap asynchronous model payload materialization failed: %s",
                    error,
                )

        thread = threading.Thread(
            target=materialize,
            name=f"coldsnap-model-payload-{_worker_id()}",
            daemon=True,
        )
        self._materialization_thread = thread
        scheduled = {
            "state": "scheduled",
            "operation_id": config["operation_id"],
            "path": str(config["target"]),
            "sha256": config["sha256"],
            "bytes": config["bytes"],
        }
        self._write_materialization_status(config, scheduled)
        thread.start()
        return scheduled

    def materialize_initial_recovery_payload(self) -> dict[str, Any] | None:
        """Export native weights after an n580 process-template recovery load.

        Unlike an n610 full-process restore, an n580 restore performs its first
        model load after CRIU releases the pre-CUDA process template. There is
        no disk-backend wake in that path, so materialization must be triggered
        once vLLM has finished establishing the complete allocation layout.
        """
        if os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORED") != "1":
            return None
        config = self._model_payload_materialization()
        if config is None:
            return None
        if not self.uses_model_weight_recovery:
            raise RuntimeError(
                "initial model payload materialization requires safetensors recovery"
            )
        previous = getattr(
            self, "_initial_model_payload_materialization_result", None
        )
        if previous is not None:
            return previous
        memory = self._memory()
        manifest = self._load_manifest(
            memory,
            allow_process_template_owner=True,
            allow_materialization_layout=True,
        )
        if manifest.get("weight_source") != WEIGHT_SOURCE_SAFETENSORS:
            raise RuntimeError(
                "initial model payload materialization requires a recovery manifest"
            )
        result = self._start_model_payload_materialization(
            memory, manifest, config
        )
        self._initial_model_payload_materialization_result = result
        return result

    def _wait_model_payload_materialization(self) -> None:
        """Keep weights resident until an asynchronous cache write finishes."""
        thread = getattr(self, "_materialization_thread", None)
        if thread is None or not thread.is_alive():
            return
        thread.join()

    def _reusable_snapshot(self, allocator: Any) -> dict[str, Any] | None:
        # An n580 native activation starts from a pre-CUDA process template.
        # Its metadata-only model bootstrap creates the captured allocation
        # layout in an awake state, then performs one level-1 sleep before the
        # first native wake. Admit the immutable captured pack as that sleep's
        # reusable generation so the bootstrap never writes fake model bytes.
        bootstrap_manifest: dict[str, Any] | None = None
        if (
            self._reusable_generation is None
            and (self.snapshot_dir / ACTIVATION_PROVIDER_NAME).is_file()
        ):
            provider = (self.snapshot_dir / ACTIVATION_PROVIDER_NAME).read_text(
                encoding="utf-8"
            ).strip()
            if provider == "native":
                bootstrap_manifest = self._load_manifest(allocator)
                if (
                    bootstrap_manifest.get("weight_source")
                    != WEIGHT_SOURCE_SPLIT_NATIVE
                ):
                    raise RuntimeError(
                        "native bootstrap did not select the split-native manifest"
                    )
                # An n580 process template retains capture-time environment,
                # where reuse is intentionally disabled.  The admitted native
                # selector is a stronger, restore-local contract: its staged,
                # preverified payload is precisely the generation this sleep
                # must retain.
                self.reuse_blob = True
                self._reusable_generation = str(bootstrap_manifest["generation"])
                self._reusable_blob_stat = _stat_identity(self.blob_path)
        if (
            not self.reuse_blob
            or self._reusable_generation is None
            or self._reusable_blob_stat is None
        ):
            return None
        started = time.perf_counter()
        try:
            # The native bootstrap validation above is complete and is also
            # the admission gate for the captured allocation layout. Reuse
            # that exact result: a generic retry/fallback must never unlink or
            # replace the immutable staged pack with placeholder tensor bytes.
            manifest = (
                bootstrap_manifest
                if bootstrap_manifest is not None
                else self._load_manifest(allocator)
            )
            actual_stat = _stat_identity(self.blob_path)
            if (
                manifest.get("generation") != self._reusable_generation
                or not _stat_matches(self._reusable_blob_stat, actual_stat)
            ):
                raise RuntimeError("restored snapshot generation changed")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
            RuntimeError,
        ):
            if bootstrap_manifest is not None:
                raise
            self._reusable_generation = None
            self._reusable_blob_stat = None
            return None

        elapsed = time.perf_counter() - started
        result = dict(manifest)
        result.update(
            {
                "blob_reused": True,
                "write_io_mode": "reused",
                "write_seconds": elapsed,
                "verification_seconds": 0.0,
                "phase_seconds": {
                    "cuda_copy_s": 0.0,
                    "checksum_s": 0.0,
                    "disk_write_s": 0.0,
                    "preallocate_s": 0.0,
                    "sync_s": 0.0,
                    "reuse_validation_s": elapsed,
                },
            }
        )
        return result

    def _snapshot_live_native_residual(
        self, provider: Any, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        """Bind a portable model object to this process's residual layout.

        A fresh n580 CUDA context can allocate the same semantic model bytes
        with slightly different allocator padding. The content-addressed model
        object is still reusable after strict extent relocation, while derived
        bytes and padding must come from this live process rather than being
        truncated or borrowed from the capture-time layout.
        """
        memory = self._memory(provider)
        source_extents = manifest.get("residual_extents")
        if not isinstance(source_extents, list) or not source_extents:
            raise RuntimeError("portable native live residual layout is invalid")
        extents = [dict(extent) for extent in source_extents]
        residual_bytes = sum(int(extent["size"]) for extent in extents)
        if residual_bytes <= 0:
            raise RuntimeError("portable native live residual layout is empty")

        self._ensure_space(residual_bytes)
        stage = self._stage()
        fd, residual_name = tempfile.mkstemp(
            prefix=f"coldsnap-live-residual-{os.getpid()}-",
            suffix=".blob",
        )
        residual_path = Path(residual_name)
        offset = 0
        started = time.perf_counter()
        try:
            _preallocate(fd, residual_bytes)
            for extent in extents:
                extent["offset"] = offset
                crc = 0
                copied = 0
                while copied < int(extent["size"]):
                    count = min(self.chunk_bytes, int(extent["size"]) - copied)
                    memory.copy_to_host(
                        stage,
                        int(extent["ptr"]) + copied,
                        count,
                    )
                    view = stage.view[:count]
                    crc = zlib.crc32(view, crc)
                    _write_all_at(fd, view, offset)
                    copied += count
                    offset += count
                extent["crc32"] = f"{crc & 0xFFFFFFFF:08x}"
            if offset != residual_bytes:
                raise RuntimeError(
                    "portable native live residual byte count changed during write"
                )
            os.fdatasync(fd)
            _drop_cache(fd, 0, 0)
        except Exception:
            os.close(fd)
            residual_path.unlink(missing_ok=True)
            raise
        else:
            os.close(fd)

        try:
            os.chmod(residual_path, 0o400)
            residual_stat = _stat_identity(residual_path)
            verification_s = 0.0
            preverified = False
            if self.verify_mode == "preverified":
                phase = time.perf_counter()
                self._verify_blob(
                    extents,
                    residual_bytes,
                    direct=False,
                    path=residual_path,
                )
                verification_s = time.perf_counter() - phase
                preverified = True
                residual_stat = _stat_identity(residual_path)
        except Exception:
            residual_path.unlink(missing_ok=True)
            raise

        self.residual_blob_path = residual_path
        rebound = dict(manifest)
        rebound.update(
            {
                "residual_blob": residual_path.name,
                "residual_blob_bytes": residual_bytes,
                "residual_blob_stat": residual_stat,
                "residual_bytes": residual_bytes,
                "residual_extents": extents,
                "preverified": preverified,
                "live_residual_snapshot": True,
                "live_residual_snapshot_seconds": time.perf_counter() - started,
                "live_residual_verification_seconds": verification_s,
            }
        )
        previous_residual = getattr(self, "_live_native_residual_path", None)
        if isinstance(previous_residual, Path) and previous_residual != residual_path:
            previous_residual.unlink(missing_ok=True)
        self._live_native_manifest = rebound
        self._live_native_residual_path = residual_path
        return rebound

    def _verify_blob(
        self,
        entries: list[dict[str, Any]],
        blob_bytes: int,
        *,
        direct: bool | None = None,
        path: Path | None = None,
        expected_verified_bytes: int | None = None,
    ) -> None:
        stage_view = self._stage().view
        blob_path = self.blob_path if path is None else path
        fd, opened_direct = _open_blob(
            blob_path,
            os.O_RDONLY,
            self.read_direct if direct is None else direct,
        )
        verified_bytes = 0
        try:
            for entry in entries:
                crc = 0
                size = int(entry["size"])
                offset = int(entry["offset"])
                copied = 0
                while copied < size:
                    count = min(self.chunk_bytes, size - copied)
                    view = stage_view[:count]
                    _read_exact_at(fd, view, offset + copied)
                    crc = zlib.crc32(view, crc)
                    copied += count
                    verified_bytes += count
                actual = f"{crc & 0xFFFFFFFF:08x}"
                if actual != entry.get("crc32"):
                    raise RuntimeError(
                        f"snapshot post-write checksum mismatch for allocation "
                        f"{entry['ptr']}: expected {entry.get('crc32')}, got {actual}"
                    )
            expected_verified = (
                blob_bytes
                if expected_verified_bytes is None
                else expected_verified_bytes
            )
            if verified_bytes != expected_verified:
                raise RuntimeError("snapshot post-write verification length mismatch")
            if not opened_direct:
                _drop_cache(fd, 0, 0)
        finally:
            os.close(fd)

    def _load_manifest(
        self,
        provider: Any,
        *,
        allow_process_template_owner: bool = False,
        allow_materialization_layout: bool = False,
    ) -> dict[str, Any]:
        memory = self._memory(provider)
        activation = self.snapshot_dir / ACTIVATION_PROVIDER_NAME
        selected: str | None = None
        if activation.is_file():
            selected = activation.read_text(encoding="utf-8").strip()
            # Activation is restore-local proof that any requested native pack
            # was already produced and admitted.  This also protects restores
            # whose process environment still carries the capture-time export
            # request (notably the n610 CRIU path).
            self._model_payload_exported = True
            if selected == "native":
                self.manifest_path = self.snapshot_dir / NATIVE_MANIFEST_NAME
                self.blob_path = self.snapshot_dir / MODEL_PAYLOAD_NAME
                self.weight_recovery_source = WEIGHT_SOURCE_SPLIT_NATIVE
                # Initialization starts in recovery mode because provider
                # selection is activation-local. Once an admitted native pack
                # is selected, permit later live sleep cycles to reuse that
                # immutable payload instead of writing the model again.
                self.reuse_blob = _env_bool(
                    "COLDSNAP_HIBERNATE_REUSE_BLOB", True
                )
            elif selected == "recovery":
                self.manifest_path = self.snapshot_dir / "manifest.json"
                self.blob_path = self.snapshot_dir / "weights.blob"
                self.residual_blob_path = self.blob_path
                self.weight_recovery_source = WEIGHT_SOURCE_SAFETENSORS
            else:
                raise RuntimeError("activation provider must be native or recovery")
        manifest = json.loads(self.manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise RuntimeError("live hibernation manifest must be an object")
        if manifest.get("format") != FORMAT or manifest.get("kind") != KIND:
            raise RuntimeError("live hibernation manifest format is unsupported")
        if selected == "native":
            residual_name = manifest.get("residual_blob")
            if (
                not isinstance(residual_name, str)
                or not residual_name
                or Path(residual_name).name != residual_name
            ):
                raise RuntimeError("native residual blob name is invalid")
            self.residual_blob_path = self.snapshot_dir / residual_name
        allocations = list(memory.allocations("weights"))
        cached_native = getattr(self, "_live_native_manifest", None)
        cached_residual_path = getattr(self, "_live_native_residual_path", None)
        allocations_released = any(
            bool(getattr(allocation, "is_released", False))
            for allocation in allocations
        )
        if (
            selected == "native"
            and allocations_released
            and isinstance(cached_native, dict)
            and isinstance(cached_residual_path, Path)
        ):
            # Suspend captured this process-local residual immediately before
            # unmapping. Wake must reuse that rebound manifest: attempting to
            # capture released CUDA pointers would make cuMemcpy dereference
            # an unmapped VA. A later awake sleep regenerates the residual.
            manifest = cached_native
            self.residual_blob_path = cached_residual_path
        else:
            relocation_semantics = tuple(
                getattr(self, "_model_weight_semantics", ())
            )
            if selected == "native":
                relocation_semantics = tuple(
                    getattr(self, "_native_model_payload_semantics", ())
                ) or relocation_semantics
            relocation_ranges = tuple(
                (int(pointer), int(size))
                for pointer, size, _semantic_id in relocation_semantics
            ) or tuple(getattr(self, "_model_weight_ranges", ()))
            manifest = _relocate_split_native_manifest(
                manifest,
                allocations,
                relocation_ranges,
                relocation_semantics,
            )
            if manifest.get("requires_live_residual_snapshot") is True:
                manifest = self._snapshot_live_native_residual(memory, manifest)
        expected_rank = _rank()
        expected_worker = _worker_id()
        expected_pid = os.getpid()
        weight_source = manifest.get("weight_source", WEIGHT_SOURCE_BLOB)
        if weight_source != self.weight_recovery_source:
            raise RuntimeError(
                "snapshot weight recovery source differs from the live worker: "
                f"manifest={weight_source!r}, worker={self.weight_recovery_source!r}"
            )
        expected_blob = self.blob_path.name
        ownership_mismatches: list[str] = []
        activated_portable_native = (
            selected == "native"
            and _activated_portable_native_manifest_matches(
                manifest, expected_worker, expected_rank
            )
        )
        restored_process_template_recovery = (
            allow_process_template_owner
            and os.environ.get("COLDSNAP_PROCESS_TEMPLATE_RESTORED") == "1"
            and _restored_process_template_recovery_manifest_matches(
                manifest, expected_worker, expected_rank, selected
            )
        )
        materialization_layout = (
            allow_materialization_layout
            and restored_process_template_recovery
        )
        if allow_materialization_layout and not materialization_layout:
            raise RuntimeError(
                "live allocation-layout admission requires a restored "
                "process-template recovery manifest"
            )
        portable_owner = (
            activated_portable_native or restored_process_template_recovery
        )
        if manifest.get("pid") != expected_pid and not portable_owner:
            ownership_mismatches.append("pid")
        if (
            not portable_owner
            and not _portable_identity_matches(
                manifest.get("identity"), expected_worker, expected_rank, expected_pid
            )
        ):
            ownership_mismatches.append("identity")
        captured_rank = manifest.get("rank")
        if captured_rank != expected_rank:
            ownership_mismatches.append("rank")
        if manifest.get("worker_id") != expected_worker:
            ownership_mismatches.append("worker_id")
        if manifest.get("capture_id") != os.environ.get("COLDSNAP_CAPTURE_ID", ""):
            ownership_mismatches.append("capture_id")
        if manifest.get("blob") != expected_blob:
            ownership_mismatches.append("blob")
        if ownership_mismatches:
            raise RuntimeError(
                "snapshot does not belong to this live worker process; mismatched "
                + ", ".join(ownership_mismatches)
            )
        current = {
            (item.pointer, item.size, item.tag)
            for item in allocations
        }
        entries = manifest.get("entries")
        if (
            not isinstance(entries, list)
            or not entries
            or not all(isinstance(entry, dict) for entry in entries)
        ):
            raise RuntimeError("snapshot allocation entries are invalid")
        try:
            saved = {
                (int(entry["ptr"]), int(entry["size"]), entry["tag"])
                for entry in entries
            }
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("snapshot allocation entries are invalid") from error
        if len(saved) != len(entries):
            raise RuntimeError("snapshot allocation entries contain duplicates")
        if current != saved and not materialization_layout:
            raise RuntimeError("snapshot allocation map differs from the live worker")
        if weight_source == WEIGHT_SOURCE_SPLIT_NATIVE:
            model_extents = manifest.get("model_weight_extents")
            residual_extents = manifest.get("residual_extents")
            model_stat = _stat_identity(self.blob_path)
            residual_stat = _stat_identity(self.residual_blob_path)
            if (
                manifest.get("model_blob") != self.blob_path.name
                or manifest.get("residual_blob") != self.residual_blob_path.name
                or not isinstance(model_extents, list)
                or not model_extents
                or not isinstance(residual_extents, list)
                or not residual_extents
                or not all(isinstance(item, dict) for item in model_extents)
                or not all(isinstance(item, dict) for item in residual_extents)
                or model_stat["bytes"] != int(manifest.get("model_blob_bytes", -1))
                or residual_stat["bytes"]
                != int(manifest.get("residual_blob_bytes", -1))
                or manifest.get("portable_model_payload") is not True
                or manifest.get("portable_residual_blob") is not True
                or manifest.get("blob_reused") is not False
                or manifest.get("checksum") != "crc32"
                or manifest.get("verify_mode") != self.verify_mode
                or manifest.get("preverified")
                is not (self.verify_mode == "preverified")
                or int(manifest.get("allocation_bytes", -1))
                != sum(item.size for item in memory.allocations("weights"))
                or int(manifest.get("allocation_bytes", -1))
                != sum(int(entry["size"]) for entry in entries)
                or any(
                    int(entry["size"]) <= 0
                    or entry.get("tag") != PRESERVED_REGION
                    or set(entry) != {"ptr", "size", "tag"}
                    for entry in entries
                )
            ):
                raise RuntimeError("split native manifest is invalid")
            allocation_ranges = {
                int(entry["ptr"]): (
                    int(entry["ptr"]),
                    int(entry["ptr"]) + int(entry["size"]),
                )
                for entry in entries
            }
            covered: dict[int, list[tuple[int, int]]] = {
                pointer: [] for pointer in allocation_ranges
            }
            try:
                for extent in [*model_extents, *residual_extents]:
                    pointer = int(extent["ptr"])
                    size = int(extent["size"])
                    allocation_ptr = int(extent["allocation_ptr"])
                    start, end = allocation_ranges[allocation_ptr]
                    if size <= 0 or pointer < start or pointer + size > end:
                        raise ValueError
                    covered[allocation_ptr].append((pointer, pointer + size))
                for extents, byte_key, padded in (
                    (model_extents, "model_blob_bytes", True),
                    (residual_extents, "residual_blob_bytes", False),
                ):
                    expected_offset = 0
                    for extent in extents:
                        if (
                            int(extent["offset"]) != expected_offset
                            or not isinstance(extent.get("crc32"), str)
                            or len(extent["crc32"]) != 8
                            or (
                                padded
                                and (
                                    not isinstance(extent.get("sha256"), str)
                                    or len(extent["sha256"]) != 64
                                )
                            )
                        ):
                            raise ValueError
                        expected_offset += (
                            _align_up(int(extent["size"]))
                            if padded
                            else int(extent["size"])
                        )
                    if expected_offset != int(manifest[byte_key]):
                        raise ValueError
                if model_extents != sorted(
                    model_extents,
                    key=lambda extent: (extent["sha256"], int(extent["size"])),
                ):
                    raise ValueError
                for allocation_ptr, ranges in covered.items():
                    cursor, end = allocation_ranges[allocation_ptr]
                    for start, range_end in sorted(ranges):
                        if start != cursor or range_end <= start:
                            raise ValueError
                        cursor = range_end
                    if cursor != end:
                        raise ValueError
                model_bytes = sum(int(item["size"]) for item in model_extents)
                residual_bytes = sum(int(item["size"]) for item in residual_extents)
                if (
                    model_bytes != int(manifest["model_weight_bytes"])
                    or residual_bytes != int(manifest["residual_bytes"])
                    or model_bytes + residual_bytes
                    != int(manifest["allocation_bytes"])
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "split native manifest byte ranges are invalid"
                ) from error
            return manifest
        if weight_source == WEIGHT_SOURCE_SAFETENSORS:
            actual_stat = _stat_identity(self.blob_path)
            model_weight_extents = manifest.get("model_weight_extents")
            residual_extents = manifest.get("residual_extents")
            if (
                not isinstance(model_weight_extents, list)
                or not model_weight_extents
                or not isinstance(residual_extents, list)
                or not residual_extents
                or not all(isinstance(item, dict) for item in model_weight_extents)
                or not all(isinstance(item, dict) for item in residual_extents)
                or int(manifest.get("blob_bytes", -1)) <= 0
                or actual_stat["bytes"] != int(manifest.get("blob_bytes", -1))
                or (
                    manifest.get("portable_residual_blob") is not True
                    and not _stat_matches(
                        manifest.get("blob_stat", {}), actual_stat
                    )
                )
                or manifest.get("direct_io") is not False
                or manifest.get("write_io_mode") != "hybrid-residual-buffered"
                or manifest.get("blob_reused") is not False
                or manifest.get("verify_mode")
                != f"model-revision+residual-{self.verify_mode}"
                or manifest.get("checksum") != "crc32"
                or manifest.get("preverified")
                is not (self.verify_mode == "preverified")
                or manifest.get("model_source") != _model_source_identity()
                or (
                    not materialization_layout
                    and int(manifest.get("allocation_bytes", -1))
                    != sum(item.size for item in memory.allocations("weights"))
                )
                or int(manifest.get("allocation_bytes", -1))
                != sum(int(entry["size"]) for entry in entries)
                or any(
                    int(entry["size"]) <= 0
                    or entry.get("tag") != PRESERVED_REGION
                    or set(entry) != {"ptr", "size", "tag"}
                    for entry in entries
                )
            ):
                raise RuntimeError("safetensors recovery manifest is invalid")
            allocation_ranges = {
                int(entry["ptr"]): (int(entry["ptr"]), int(entry["ptr"]) + int(entry["size"]))
                for entry in entries
            }
            covered: dict[int, list[tuple[int, int]]] = {
                pointer: [] for pointer in allocation_ranges
            }
            try:
                for extent in [*model_weight_extents, *residual_extents]:
                    pointer = int(extent["ptr"])
                    size = int(extent["size"])
                    allocation_ptr = int(extent["allocation_ptr"])
                    start, end = allocation_ranges[allocation_ptr]
                    if size <= 0 or pointer < start or pointer + size > end:
                        raise ValueError
                    covered[allocation_ptr].append((pointer, pointer + size))
                expected_offset = 0
                for extent in residual_extents:
                    if (
                        int(extent["offset"]) != expected_offset
                        or not isinstance(extent.get("crc32"), str)
                        or len(extent["crc32"]) != 8
                    ):
                        raise ValueError
                    expected_offset += int(extent["size"])
                if expected_offset != int(manifest["blob_bytes"]):
                    raise ValueError
                for allocation_ptr, ranges in covered.items():
                    cursor, end = allocation_ranges[allocation_ptr]
                    for start, range_end in sorted(ranges):
                        if start != cursor or range_end <= start:
                            raise ValueError
                        cursor = range_end
                    if cursor != end:
                        raise ValueError
                model_weight_bytes = sum(
                    int(extent["size"]) for extent in model_weight_extents
                )
                residual_bytes = sum(
                    int(extent["size"]) for extent in residual_extents
                )
                if (
                    model_weight_bytes != int(manifest["model_weight_bytes"])
                    or residual_bytes != int(manifest["residual_bytes"])
                    or residual_bytes != int(manifest["blob_bytes"])
                    or model_weight_bytes + residual_bytes
                    != int(manifest["allocation_bytes"])
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "safetensors recovery manifest byte ranges are invalid"
                ) from error
            return manifest
        actual_stat = _stat_identity(self.blob_path)
        if actual_stat["bytes"] != int(manifest["blob_bytes"]):
            raise RuntimeError("snapshot blob size does not match its manifest")
        if (
            self.verify_mode == "preverified"
            and not _stat_matches(manifest.get("blob_stat", {}), actual_stat)
        ):
            raise RuntimeError("preverified snapshot blob stat identity changed")
        if self.verify_mode == "preverified" and manifest.get("preverified") is not True:
            raise RuntimeError("snapshot has no completed post-write verification")
        expected_offset = 0
        for entry in entries:
            if int(entry["offset"]) != expected_offset or int(entry["size"]) <= 0:
                raise RuntimeError("snapshot allocation entries are not contiguous")
            expected_offset += int(entry["size"])
        if expected_offset != int(manifest["blob_bytes"]):
            raise RuntimeError("snapshot allocation entries do not cover the blob")
        return manifest

    def suspend(self, level: int = 1) -> None:
        from vllm.logger import init_logger

        if level != 1:
            raise ValueError("disk sleep supports level 1 only")
        # The read-through cache copies live model weights in the background.
        # A later sleep must not unmap those allocations underneath it.
        self._wait_model_payload_materialization()
        memory = self._memory()
        self._state = "SUSPENDING"
        self._write_state("suspending")
        started = time.perf_counter()
        try:
            phase = time.perf_counter()
            memory.synchronize()
            initial_sync_s = time.perf_counter() - phase
            manifest = self._write_snapshot(memory)

            phase = time.perf_counter()
            self.graph_controller.release()
            graph_unmap_s = time.perf_counter() - phase
            graph_state = self._graph_state()

            total_bytes = 0
            phase = time.perf_counter()
            for allocation in memory.allocations(PRESERVED_REGION):
                total_bytes += allocation.size
                memory.release(allocation)
            unmap_s = time.perf_counter() - phase

            # Discardable regions are unmapped last so a failure above leaves
            # them addressable and the process still recoverable.
            discarded = self._release_discard_regions(memory)

            phase = time.perf_counter()
            memory.synchronize()
            memory.empty_cache()
            final_sync_s = time.perf_counter() - phase
            duration = time.perf_counter() - started
            self._state = "SUSPENDED"
            self._write_state(
                "sleeping",
                generation=manifest["generation"],
                manifest=str(self.manifest_path),
                blob=(
                    str(self.blob_path)
                    if manifest.get("blob", self.blob_path.name) is not None
                    else ""
                ),
                blob_bytes=manifest["blob_bytes"],
                blob_stat=manifest["blob_stat"],
                entry_count=len(manifest["entries"]),
                direct_io=manifest["direct_io"],
                write_io_mode=manifest["write_io_mode"],
                blob_reused=manifest["blob_reused"],
                sleep_seconds=duration,
                write_seconds=manifest["write_seconds"],
                verification_seconds=manifest["verification_seconds"],
                unmap_seconds=unmap_s,
                graph_unmap_seconds=graph_unmap_s,
                cuda_graph=graph_state,
                initial_sync_seconds=initial_sync_s,
                final_sync_seconds=final_sync_s,
                phase_seconds=manifest["phase_seconds"],
                discard_regions=list(self.discard_regions),
                discarded_bytes=discarded["bytes"],
                discard_unmap_seconds=discarded["unmap_seconds"],
                discard_detail=discarded["regions"],
                regions=self._region_inventory(memory),
                weight_recovery_source=self.weight_recovery_source,
            )
        except Exception as error:
            self._state = "FAILED"
            self._write_state("failed", operation="sleep", error=str(error))
            raise
        init_logger(__name__).info(
            "Disk sleep %s %.2f GiB in %.3f s, released %.2f GiB, "
            "discarded %.2f GiB from %s",
            (
                "wrote split model payload and residual for"
                if self.exports_model_payload
                else "wrote residual for"
                if self.uses_model_weight_recovery
                else "reused" if manifest["blob_reused"] else "wrote"
            ),
            manifest["blob_bytes"] / GIB,
            duration,
            total_bytes / GIB,
            discarded["bytes"] / GIB,
            ", ".join(self.discard_regions) or "no regions",
        )

    def _restore_pipeline(
        self, provider: Any, manifest: dict[str, Any], fd: int
    ) -> dict[str, float | int | str]:
        memory = self._memory(provider)
        ordered_entries, map_s = self._remap_weight_allocations(memory, manifest)

        if getattr(self, "hydrator", None) is not None:
            extents = [
                HydrationExtent(
                    file_offset=int(entry["offset"]),
                    destination=int(entry["ptr"]),
                    length=int(entry["size"]),
                    crc32=str(entry["crc32"]),
                )
                for entry in ordered_entries
            ]
            native = self.hydrator.hydrate(
                self.blob_path,
                extents,
                backend=self.hydration_backend,
                chunk_bytes=self.chunk_bytes,
                queue_depth=self.pipeline_depth,
                preverified=self.verify_mode == "preverified",
                register_device_buffers=False,
            )
            return {
                "restored_bytes": native.bytes,
                "chunk_count": native.chunks,
                "map_seconds": map_s,
                "disk_read_service_seconds": native.io_service_s,
                "disk_read_wait_seconds": native.io_wait_s,
                "checksum_seconds": native.checksum_s,
                "cuda_enqueue_seconds": native.cuda_enqueue_s,
                "native_initialization_seconds": native.initialization_s,
                "native_synchronize_seconds": native.cuda_synchronize_s,
                "native_total_seconds": native.total_s,
                "native_verified_extents": native.verified_extents,
                "hydration_backend": native.backend,
            }

        chunks = _chunk_descriptors(ordered_entries, self.chunk_bytes)
        if not chunks:
            raise RuntimeError("live hibernation snapshot contains no chunks")
        depth = min(self.pipeline_depth, len(chunks))
        stages = self._stages(depth)
        checksums: dict[int, int] = {int(entry["ptr"]): 0 for entry in ordered_entries}
        read_service_s = 0.0
        read_wait_s = 0.0
        checksum_s = 0.0
        cuda_copy_s = 0.0
        restored_bytes = 0
        pending: list[tuple[tuple[int, int, int, int], int, Future[float]]] = []
        next_chunk = 0

        with ThreadPoolExecutor(max_workers=depth, thread_name_prefix="coldsnap-read") as pool:
            while next_chunk < depth:
                descriptor = chunks[next_chunk]
                buffer_index = next_chunk
                view = stages[buffer_index].view
                pending.append(
                    (
                        descriptor,
                        buffer_index,
                        pool.submit(
                            _timed_read,
                            fd,
                            view[: descriptor[2]],
                            descriptor[1],
                        ),
                    )
                )
                next_chunk += 1

            while pending:
                descriptor, buffer_index, future = pending.pop(0)
                destination, _, count, allocation_ptr = descriptor
                phase = time.perf_counter()
                read_service_s += future.result()
                read_wait_s += time.perf_counter() - phase
                stage = stages[buffer_index]
                view = stage.view
                chunk_view = view[:count]

                if self.verify_mode == "inline":
                    phase = time.perf_counter()
                    checksums[allocation_ptr] = zlib.crc32(
                        chunk_view, checksums[allocation_ptr]
                    )
                    checksum_s += time.perf_counter() - phase

                phase = time.perf_counter()
                memory.copy_from_host(destination, stage, count)
                cuda_copy_s += time.perf_counter() - phase
                restored_bytes += count

                if next_chunk < len(chunks):
                    next_descriptor = chunks[next_chunk]
                    pending.append(
                        (
                            next_descriptor,
                            buffer_index,
                            pool.submit(
                                _timed_read,
                                fd,
                                view[: next_descriptor[2]],
                                next_descriptor[1],
                            ),
                        )
                    )
                    next_chunk += 1

        if self.verify_mode == "inline":
            for entry in ordered_entries:
                expected = entry.get("crc32")
                actual = f"{checksums[int(entry['ptr'])] & 0xFFFFFFFF:08x}"
                if expected != actual:
                    raise RuntimeError(
                        f"snapshot checksum mismatch for allocation {entry['ptr']}: "
                        f"expected {expected}, got {actual}"
                    )

        return {
            "restored_bytes": restored_bytes,
            "chunk_count": len(chunks),
            "map_seconds": map_s,
            "disk_read_service_seconds": read_service_s,
            "disk_read_wait_seconds": read_wait_s,
            "checksum_seconds": checksum_s,
            "cuda_copy_seconds": cuda_copy_s,
        }

    def _remap_weight_allocations(
        self, provider: Any, manifest: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], float]:
        memory = self._memory(provider)
        entries = {int(entry["ptr"]): entry for entry in manifest["entries"]}
        map_started = time.perf_counter()
        ordered_entries: list[dict[str, Any]] = []
        for allocation in memory.allocations(PRESERVED_REGION):
            ptr = allocation.pointer
            entry = entries.get(ptr)
            if entry is None or int(entry["size"]) != allocation.size:
                raise RuntimeError(f"live weight allocation {ptr} is absent from snapshot")
            if allocation.is_released and ptr not in self._restore_mapped:
                memory.remap(allocation)
                self._restore_mapped.add(ptr)
            elif not allocation.is_released and ptr in self._restore_mapped:
                raise RuntimeError(
                    f"live weight allocation {ptr} was published during an incomplete restore"
                )
            ordered_entries.append(entry)
        return ordered_entries, time.perf_counter() - map_started

    def _restore_exact_extents(
        self,
        provider: Any,
        path: Path,
        extents: list[dict[str, Any]],
        label: str,
    ) -> dict[str, Any]:
        memory = self._memory(provider)
        exact = [
            HydrationExtent(
                file_offset=int(entry["offset"]),
                destination=int(entry["ptr"]),
                length=int(entry["size"]),
                crc32=str(entry["crc32"]),
            )
            for entry in extents
        ]
        if getattr(self, "hydrator", None) is not None:
            calls: list[Any] = []
            backend = self.hydration_backend
            if backend in {"auto", "direct"} and self.verify_mode == "preverified":
                direct, buffered = _split_direct_hydration_extents(exact)
                if direct:
                    calls.append(
                        self.hydrator.hydrate(
                            path,
                            direct,
                            backend="direct",
                            chunk_bytes=self.chunk_bytes,
                            queue_depth=self.pipeline_depth,
                            preverified=True,
                            register_device_buffers=False,
                        )
                    )
                if buffered:
                    calls.append(
                        self.hydrator.hydrate(
                            path,
                            buffered,
                            backend="buffered",
                            chunk_bytes=self.chunk_bytes,
                            queue_depth=self.pipeline_depth,
                            preverified=True,
                            register_device_buffers=False,
                        )
                    )
                effective_backend = (
                    "direct+buffered-edges" if direct and buffered
                    else "direct" if direct
                    else "buffered"
                )
            else:
                effective_backend = "buffered" if backend == "direct" else backend
                calls.append(
                    self.hydrator.hydrate(
                        path,
                        exact,
                        backend=effective_backend,
                        chunk_bytes=self.chunk_bytes,
                        queue_depth=self.pipeline_depth,
                        preverified=self.verify_mode == "preverified",
                        register_device_buffers=False,
                    )
                )
            return {
                "bytes": sum(result.bytes for result in calls),
                "chunks": sum(result.chunks for result in calls),
                "backend": effective_backend,
                "seconds": sum(result.total_s for result in calls),
                "io_service_seconds": sum(result.io_service_s for result in calls),
                "io_wait_seconds": sum(result.io_wait_s for result in calls),
                "checksum_seconds": sum(result.checksum_s for result in calls),
                "cuda_enqueue_seconds": sum(result.cuda_enqueue_s for result in calls),
                "initialization_seconds": sum(
                    result.initialization_s for result in calls
                ),
                "cuda_synchronize_seconds": sum(
                    result.cuda_synchronize_s for result in calls
                ),
            }

        stage = self._stage()
        restored_bytes = 0
        chunks = 0
        started = time.perf_counter()
        io_seconds = 0.0
        checksum_seconds = 0.0
        cuda_copy_seconds = 0.0
        fd = os.open(path, os.O_RDONLY)
        try:
            for entry in extents:
                crc = 0
                copied = 0
                while copied < int(entry["size"]):
                    count = min(
                        self.chunk_bytes, int(entry["size"]) - copied
                    )
                    view = stage.view[:count]
                    phase = time.perf_counter()
                    _read_exact_at(fd, view, int(entry["offset"]) + copied)
                    io_seconds += time.perf_counter() - phase
                    phase = time.perf_counter()
                    crc = zlib.crc32(view, crc)
                    checksum_seconds += time.perf_counter() - phase
                    phase = time.perf_counter()
                    memory.copy_from_host(int(entry["ptr"]) + copied, stage, count)
                    cuda_copy_seconds += time.perf_counter() - phase
                    copied += count
                    restored_bytes += count
                    chunks += 1
                if self.verify_mode == "inline":
                    actual = f"{crc & 0xFFFFFFFF:08x}"
                    if actual != entry["crc32"]:
                        raise RuntimeError(
                            f"{label} checksum mismatch for "
                            f"{entry['ptr']}: expected {entry['crc32']}, got {actual}"
                        )
        finally:
            os.close(fd)
        return {
            "bytes": restored_bytes,
            "chunks": chunks,
            "backend": "python-buffered",
            "seconds": time.perf_counter() - started,
            "io_service_seconds": io_seconds,
            "checksum_seconds": checksum_seconds,
            "cuda_copy_seconds": cuda_copy_seconds,
        }

    def _restore_recovery_residuals(
        self, provider: Any, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        return self._restore_exact_extents(
            provider,
            self.blob_path,
            manifest["residual_extents"],
            "recovery residual",
        )

    def _restore_split_native(
        self, provider: Any, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        memory = self._memory(provider)
        entries, map_s = self._remap_weight_allocations(memory, manifest)
        model = self._restore_exact_extents(
            memory,
            self.blob_path,
            manifest["model_weight_extents"],
            "model payload",
        )
        residual = self._restore_exact_extents(
            memory,
            self.residual_blob_path,
            manifest["residual_extents"],
            "runtime residual",
        )
        return {
            "restored_bytes": sum(int(entry["size"]) for entry in entries),
            "chunk_count": int(model["chunks"]) + int(residual["chunks"]),
            "map_seconds": map_s,
            "model_payload_bytes": int(manifest["model_weight_bytes"]),
            "residual_blob_bytes": int(manifest["residual_bytes"]),
            "model_restore": model,
            "residual_restore": residual,
            "hydration_backend": (
                "split-native+"
                + str(model["backend"])
                + "+"
                + str(residual["backend"])
            ),
        }

    def _restore_from_model_source(
        self, provider: Any, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        memory = self._memory(provider)
        callback = self._weight_recovery_callback
        if not callable(callback):
            raise RuntimeError(
                "safetensors recovery reached wake without the vLLM reload callback"
            )
        entries, map_s = self._remap_weight_allocations(memory, manifest)
        # Preserve exact non-weight bytes before vLLM invokes any quantized load
        # kernels. Only reload-owned model-weight spans are reconstructed from
        # the pinned model source and therefore receive a deterministic zero base.
        fill_zero = getattr(memory, "fill_zero", None)
        if not callable(fill_zero):
            raise RuntimeError(
                "memory provider cannot initialize recovery weight allocations"
            )
        initialize_started = time.perf_counter()
        for extent in manifest["model_weight_extents"]:
            fill_zero(int(extent["ptr"]), int(extent["size"]))
        residual_before = self._restore_recovery_residuals(memory, manifest)
        memory.synchronize()
        initialize_s = time.perf_counter() - initialize_started
        load_started = time.perf_counter()
        callback()
        load_s = time.perf_counter() - load_started
        memory.synchronize()
        residual_after = self._restore_recovery_residuals(memory, manifest)
        memory.synchronize()
        from coldsnap_recovery_loader import last_recovery_load_metrics

        loader_metrics = last_recovery_load_metrics()
        return {
            "restored_bytes": sum(int(entry["size"]) for entry in entries),
            "chunk_count": 0,
            "map_seconds": map_s,
            "initialization_seconds": initialize_s,
            "model_reload_seconds": load_s,
            "residual_blob_bytes": int(manifest["residual_bytes"]),
            "model_weight_source_bytes": int(manifest["model_weight_bytes"]),
            "residual_restore_before": residual_before,
            "residual_restore_after": residual_after,
            "hydration_backend": (
                "recovery-safetensors+" + str(residual_before["backend"])
            ),
            "recovery_loader": loader_metrics,
        }

    def _commit_restore(self, provider: Any) -> None:
        memory = self._memory(provider)
        for allocation in memory.allocations("weights"):
            if allocation.is_released and allocation.pointer not in self._restore_mapped:
                raise RuntimeError(
                    f"live weight allocation {allocation.pointer} was not mapped by the restore"
                )
            allocation.set_released(False)
        self._restore_mapped.clear()

    def _resume_selection(
        self, tags: list[str] | None
    ) -> tuple[bool, tuple[str, ...]]:
        """Split vLLM's wake tags into preserved and discardable work."""
        if tags is None:
            return True, self.discard_regions
        known = {PRESERVED_REGION, *self.discard_regions}
        unknown = [tag for tag in tags if tag not in known]
        if unknown:
            raise ValueError(
                "disk wake does not manage region(s) "
                + ", ".join(repr(tag) for tag in unknown)
                + "; managed regions are "
                + ", ".join(sorted(known))
            )
        selected = tuple(
            region for region in self.discard_regions if region in tags
        )
        wake_preserved = PRESERVED_REGION in tags
        if not wake_preserved and not selected:
            raise ValueError("disk wake requires at least one managed region")
        return wake_preserved, selected

    def resume(self, tags: list[str] | None = None) -> None:
        from vllm.logger import init_logger

        wake_preserved, wake_discard = self._resume_selection(tags)
        memory = self._memory()
        self._state = "RESUMING"
        self._write_state("resuming")
        started = time.perf_counter()
        try:
            graph_state = self._graph_state()
            if (
                not (wake_preserved and _has_asleep_weights(memory))
                and not self._restore_mapped
                and not _has_asleep_regions(memory, wake_discard)
                and not self._discard_mapped
                and not graph_state["paused_count"]
            ):
                self._state = "RUNNING"
                self._write_state("running", operation="recover-failed-suspend")
                init_logger(__name__).info(
                    "Disk wake recovered a failed pre-unmap suspend in %.3f s",
                    time.perf_counter() - started,
                )
                return

            manifest: dict[str, Any] | None = None
            direct = False
            metrics: dict[str, Any] = {}
            restored_blob_stat: dict[str, int] | None = None
            if wake_preserved:
                manifest = self._load_manifest(memory)
                if manifest.get("weight_source") == WEIGHT_SOURCE_SAFETENSORS:
                    metrics = self._restore_from_model_source(memory, manifest)
                elif manifest.get("weight_source") == WEIGHT_SOURCE_SPLIT_NATIVE:
                    metrics = self._restore_split_native(memory, manifest)
                    restored_blob_stat = _stat_identity(self.blob_path)
                else:
                    fd, direct = _open_blob(
                        self.blob_path, os.O_RDONLY, self.read_direct
                    )
                    try:
                        metrics = self._restore_pipeline(memory, manifest, fd)
                        if not direct:
                            _drop_cache(fd, 0, 0)
                    finally:
                        os.close(fd)
                    restored_blob_stat = _stat_identity(self.blob_path)

            # Recovery-aware reload may leave its bounded CUDA staging slabs in
            # PyTorch's caching allocator. Release those dead allocations before
            # requesting physical backing for a large discard region. vLLM's
            # native wake path performs the same GC/empty-cache boundary; without
            # it, unified-memory systems can cross into swap reclaim during the
            # final cuMemCreate calls even though the staging tensors are no
            # longer live.
            if wake_discard:
                phase = time.perf_counter()
                gc.collect()
                memory.empty_cache()
                metrics["discard_cache_release_seconds"] = (
                    time.perf_counter() - phase
                )

            # Discardable regions carry no bytes, so this is address work only.
            # It runs before the final synchronize so the engine observes every
            # managed region as addressable at the same boundary.
            remapped = self._remap_discard_regions(memory, wake_discard)
            metrics["discard_remap_seconds"] = remapped["remap_seconds"]
            metrics["discard_remapped_bytes"] = remapped["bytes"]
            metrics["discard_map_seconds"] = remapped["map_seconds"]
            metrics["discard_zero_seconds"] = remapped["zero_seconds"]
            metrics["discard_verification_seconds"] = remapped[
                "verification_seconds"
            ]

            # Graph memory holds captured references into the weight pool, so it
            # is remapped with the weights rather than on a discard-only wake.
            if wake_preserved:
                phase = time.perf_counter()
                self.graph_controller.restore()
                metrics["graph_remap_seconds"] = time.perf_counter() - phase
                graph_state = self._graph_state()
            phase = time.perf_counter()
            memory.synchronize()
            metrics["sync_seconds"] = time.perf_counter() - phase

            if wake_preserved:
                self._commit_restore(memory)
            self._commit_discard_regions(memory, wake_discard)
            if (
                manifest is not None
                and manifest.get("weight_source") == WEIGHT_SOURCE_SAFETENSORS
            ):
                materialization = self._model_payload_materialization()
                if materialization is not None:
                    metrics["model_payload_materialization"] = (
                        self._start_model_payload_materialization(
                            memory, manifest, materialization
                        )
                    )
            if manifest is not None and restored_blob_stat is not None:
                self._reusable_generation = str(manifest["generation"])
                self._reusable_blob_stat = restored_blob_stat
            duration = time.perf_counter() - started
            self._state = "RUNNING"
            self._write_state(
                "running",
                generation=(
                    manifest["generation"] if manifest is not None else ""
                ),
                manifest=str(self.manifest_path),
                blob=(
                    str(self.blob_path)
                    if manifest is not None and manifest.get("blob") is not None
                    else ""
                ),
                blob_bytes=(
                    manifest["blob_bytes"] if manifest is not None else 0
                ),
                blob_stat=(manifest["blob_stat"] if manifest is not None else {}),
                entry_count=(
                    len(manifest["entries"]) if manifest is not None else 0
                ),
                direct_io=direct,
                read_io_mode=(
                    str(metrics.get("hydration_backend"))
                    if metrics.get("hydration_backend")
                    else self.hydration_backend
                    if getattr(self, "hydrator", None) is not None
                    else "direct" if direct else "buffered"
                ),
                blob_reuse_armed=(
                    self.reuse_blob and not self.uses_model_weight_recovery
                ),
                resume_seconds=duration,
                cuda_graph=graph_state,
                phase_seconds=metrics,
                woke_preserved=wake_preserved,
                woke_discard_regions=list(wake_discard),
                discard_remapped_bytes=remapped["bytes"],
                discard_detail=remapped["regions"],
                regions=self._region_inventory(memory),
                weight_recovery_source=self.weight_recovery_source,
            )
        except Exception as error:
            self._state = "FAILED"
            self._write_state("failed", operation="resume", error=str(error))
            raise
        init_logger(__name__).info(
            "Disk wake restored %.2f GiB and remapped %.2f GiB of %s in %.3f s",
            int(metrics.get("restored_bytes", 0)) / GIB,
            remapped["bytes"] / GIB,
            ", ".join(wake_discard) or "no discard regions",
            duration,
        )

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Provider views must restore identical bytes while sharing durable storage."""

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
import coldsnap_disk_backend as disk  # noqa: E402
from coldsnap_core.memory import HostStage  # noqa: E402
from coldsnap_core.validation import PayloadValidationError, validate_payload  # noqa: E402
from test.test_disk_sleep import _single_worker_execution_graph  # noqa: E402


class CaptureStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.environment = patch.dict(
            os.environ,
            {
                "COLDSNAP_EXECUTION_GRAPH": _single_worker_execution_graph(),
                "COLDSNAP_MODEL_ID": "fixture",
                "COLDSNAP_MODEL_REVISION": "pinned",
                "COLDSNAP_EXPORT_MODEL_PAYLOAD": "1",
            },
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.source = bytes(index % 251 for index in range(16384))
        self.restored = bytearray(len(self.source))
        self.allocation = SimpleNamespace(
            pointer=0x10000, size=len(self.source), tag="weights", is_released=False
        )
        self.memory = SimpleNamespace(
            allocations=lambda _tag=None: [self.allocation],
            allocate_host_stage=lambda size: HostStage(
                owner=None, view=memoryview(bytearray(size)), pointer=0
            ),
            copy_to_host=self.copy_to_host,
            copy_from_host=self.copy_from_host,
        )
        self.backend = disk.DiskCuMemBackend.__new__(disk.DiskCuMemBackend)
        self.backend.snapshot_dir = Path(self.temporary.name)
        self.backend.blob_path = self.backend.snapshot_dir / "weights.blob"
        self.backend.manifest_path = self.backend.snapshot_dir / "manifest.json"
        self.backend.weight_recovery_source = disk.WEIGHT_SOURCE_SAFETENSORS
        self.backend._model_weight_ranges = ((0x11000, 4096),)
        # Deliberately non-nested ownership: both providers need some bytes the
        # other classifies as model data. Their residual intersection is stored once.
        self.backend._native_model_payload_semantics = ((0x11800, 4096, "a" * 64),)
        self.backend.chunk_bytes = 4096
        self.backend.pipeline_depth = 2
        self.backend.verify_mode = "preverified"
        self.backend.read_direct = False
        self.backend.write_direct = False
        self.backend._stage_cache = []
        self.backend.memory_provider = self.memory
        self.backend.hydrator = None

    def copy_to_host(self, stage, pointer, size):
        offset = pointer - self.allocation.pointer
        stage.view[:size] = self.source[offset : offset + size]

    def copy_from_host(self, pointer, stage, size):
        offset = pointer - self.allocation.pointer
        self.restored[offset : offset + size] = stage.view[:size]

    def test_crossing_provider_views_restore_same_bytes_and_verify_each_object_once(self):
        with patch.object(
            self.backend, "_verify_blob", wraps=self.backend._verify_blob
        ) as readback:
            recovery = self.backend._write_recovery_manifest(self.memory)
        self.assertEqual(readback.call_count, 1)  # Shared residual only; pack uses SHA admission.
        self.assertEqual(recovery["unique_residual_bytes"], 14336)
        self.assertEqual(self.backend._load_manifest(self.memory), recovery)
        native = json.loads((self.backend.snapshot_dir / disk.NATIVE_MANIFEST_NAME).read_text())
        self.assertEqual(native["residual_blob"], recovery["blob"])
        self.assertFalse((self.backend.snapshot_dir / disk.NATIVE_RESIDUAL_NAME).exists())
        for manifest, is_native in ((recovery, False), (native, True)):
            self.restored[:] = b"\0" * len(self.restored)
            self.backend._restore_exact_extents(
                self.memory, self.backend.blob_path, manifest["residual_extents"], "residual"
            )
            if is_native:
                self.backend._restore_exact_extents(
                    self.memory,
                    self.backend.snapshot_dir / disk.MODEL_PAYLOAD_NAME,
                    manifest["model_weight_extents"],
                    "model",
                )
            else:
                for extent in manifest["model_weight_extents"]:
                    start = extent["ptr"] - self.allocation.pointer
                    self.restored[start : start + extent["size"]] = self.source[
                        start : start + extent["size"]
                    ]
            self.assertEqual(self.restored, self.source)
        payload = recovery["model_payload"]
        pack = self.backend.snapshot_dir / payload["blob"]
        self.assertEqual(
            validate_payload(pack, payload["sha256"], payload["bytes"])["bytes_hashed"], 0
        )
        pack.chmod(0o600)
        with pack.open("r+b") as stream:
            stream.write(b"X")
        with self.assertRaises(PayloadValidationError):
            validate_payload(pack, payload["sha256"], payload["bytes"])

    def test_preverified_fallback_omits_crc_but_inline_still_detects_corruption(self):
        manifest = self.backend._write_recovery_manifest(self.memory)
        with patch.object(disk.zlib, "crc32", side_effect=AssertionError("unexpected CRC")):
            result = self.backend._restore_exact_extents(
                self.memory, self.backend.blob_path, manifest["residual_extents"], "residual"
            )
        self.assertEqual(result["checksum_seconds"], 0)
        self.backend.verify_mode = "inline"
        self.backend.blob_path.chmod(0o600)
        with self.backend.blob_path.open("r+b") as stream:
            stream.write(b"X")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self.backend._restore_exact_extents(
                self.memory, self.backend.blob_path, manifest["residual_extents"], "residual"
            )

    def test_shared_manifest_rejects_overlap_unaligned_and_out_of_bounds_offsets(self):
        manifest = self.backend._write_recovery_manifest(self.memory)
        for offset in (-4096, 1, manifest["blob_bytes"]):
            broken = copy.deepcopy(manifest)
            broken["residual_extents"][0]["offset"] = offset
            self.backend.manifest_path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(RuntimeError, "byte ranges are invalid"):
                self.backend._load_manifest(self.memory)
        broken = copy.deepcopy(manifest)
        broken["residual_extents"][1]["offset"] = broken["residual_extents"][0]["offset"]
        self.backend.manifest_path.write_text(json.dumps(broken))
        with self.assertRaisesRegex(RuntimeError, "byte ranges are invalid"):
            self.backend._load_manifest(self.memory)

    def test_legacy_contiguous_recovery_manifest_still_loads(self):
        manifest = self.backend._write_recovery_manifest(self.memory)
        blob = bytearray()
        for extent in manifest["residual_extents"]:
            extent["offset"] = len(blob)
            start = extent["ptr"] - self.allocation.pointer
            blob.extend(self.source[start : start + extent["size"]])
        self.backend.blob_path.chmod(0o600)
        self.backend.blob_path.write_bytes(blob)
        manifest.pop("residual_layout")
        manifest.update(
            blob_bytes=len(blob),
            blob_stat=disk._stat_identity(self.backend.blob_path),
            write_io_mode="hybrid-residual-buffered",
            direct_io=False,
        )
        self.backend.manifest_path.write_text(json.dumps(manifest))
        self.assertEqual(self.backend._load_manifest(self.memory), manifest)

    def test_shared_union_randomized_partitions_cover_each_view_once(self):
        rng = random.Random(42)
        for _ in range(100):
            views = []
            for _view in range(2):
                bounds = sorted(rng.sample(range(1, 100), 4))
                views.append(
                    [
                        {"ptr": 1000 + start, "size": end - start, "allocation_ptr": 1000}
                        for start, end in zip(bounds[::2], bounds[1::2], strict=True)
                    ]
                )
            stored, references = disk._shared_residual_extents(*views)

            def address_set(extents):
                return {p for e in extents for p in range(e["ptr"], e["ptr"] + e["size"])}

            for view, reference in zip(views, references, strict=True):
                self.assertEqual(address_set(view), address_set(reference))
                self.assertEqual(sum(e["size"] for e in reference), len(address_set(view)))
            union = address_set(views[0]) | address_set(views[1])
            self.assertEqual(address_set(stored), union)
            self.assertEqual(sum(e["size"] for e in stored), len(union))

    def test_residual_and_model_writers_use_native_auto_without_duplicate_readback(self):
        calls = []

        def capture(path, extents, **options):
            calls.append(options)
            blob = bytearray(max(e.file_offset + disk._align_up(e.length) for e in extents))
            digests = []
            for extent in extents:
                start = extent.source - self.allocation.pointer
                data = self.source[start : start + extent.length]
                blob[extent.file_offset : extent.file_offset + extent.length] = data
                digests.append(
                    SimpleNamespace(
                        crc32=f"{zlib.crc32(data):08x}", sha256=hashlib.sha256(data).hexdigest()
                    )
                )
            Path(path).write_bytes(blob)
            return SimpleNamespace(
                metrics=SimpleNamespace(
                    backend="direct",
                    bytes=sum(e.length for e in extents),
                    file_bytes=len(blob),
                    cuda_enqueue_s=0.1,
                    cuda_synchronize_s=0.2,
                    checksum_s=0.3,
                    io_service_s=0.4,
                    initialization_s=0.5,
                    durability_s=0.6,
                    io_wait_s=0.7,
                    verification_s=0.8 if options["verify_readback"] else 0,
                ),
                digests=digests,
                file_sha256=hashlib.sha256(blob).hexdigest(),
            )

        self.backend.capture_transport = SimpleNamespace(capture=capture)
        self.backend.capture_backend = "auto"
        with patch.object(
            self.backend, "_verify_blob", side_effect=AssertionError("extra readback")
        ):
            manifest = self.backend._write_recovery_manifest(self.memory)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["backend"] == "auto" and call["pad_extents"] for call in calls))
        self.assertEqual([call["verify_readback"] for call in calls], [True, False])
        self.assertEqual(manifest["capture_backend"], "native-direct")
        self.assertAlmostEqual(
            manifest["verification_seconds"],
            0.8 + manifest["model_payload"]["verification_seconds"],
        )

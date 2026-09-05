# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import errno
import hashlib
import json
import os
import tempfile
import unittest
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
import coldsnap_disk_backend as disk_backend  # noqa: E402
import coldsnap_recovery_loader as recovery_loader  # noqa: E402
import coldsnap_synthetic_loader as synthetic_loader  # noqa: E402
from coldsnap_core.memory import HostStage  # noqa: E402
from coldsnap_disk_backend import (  # noqa: E402
    DiskCuMemBackend,
    _chunk_descriptors,
    _discard_regions,
    _env_bool,
    _has_asleep_regions,
    _has_asleep_weights,
    _identity,
    _io_direct,
    _preallocate,
    _rank,
    _read_exact_at,
    _stat_matches,
    _write_all_at,
)
from coldsnap_layout import reference_fingerprint, semantic_replay_map  # noqa: E402
from coldsnap_vllm import (  # noqa: E402
    SLEEP_BACKEND_NAME,
    VllmContractError,
    allocations as vllm_allocations,
    force_cross_rank_rpc_over_tcp,
    override_default_sleep_backend,
    prepare_synthetic_weight_source,
    register_sleep_backend,
)
from coldsnap_synthetic_loader import install_synthetic_weight_loader  # noqa: E402
from coldsnap_recovery_loader import (  # noqa: E402
    LOAD_FORMAT as RECOVERY_LOAD_FORMAT,
    SafetensorDescriptor,
    _broadcast_slab_regions as recovery_broadcast_slab_regions,
    _broadcast_tensors as recovery_broadcast_tensors,
    _descriptors as recovery_descriptors,
    _coalesced_extents as recovery_coalesced_extents,
    _descriptor_batches as recovery_descriptor_batches,
    _install_recovery_hybrid_draft_bridge,
    _owner_layout as recovery_owner_layout,
    _owners as recovery_owners,
    _split_direct_extents as recovery_split_direct_extents,
    _storage_views as recovery_storage_views,
    _verify_tensor_samples as recovery_verify_tensor_samples,
    install_recovery_aware_loader,
)
from coldsnap_startup_plan import (  # noqa: E402
    _CONFIGURED_MODEL_LEN_ATTR,
    _all_workers_admit_startup_plan,
    _compile_cache_directory,
    _install_compile_cache_identity_hook,
    _install_phase_stable_startup_plan_fingerprint,
    _install_startup_plan_shortfall_adjustment,
    _validate_free_memory_admission,
    adjusted_plan_bytes,
    install_startup_plan_memory_fallback,
    minimum_free_memory,
)


def _single_worker_execution_graph() -> str:
    return json.dumps(
        {
            "unit": "unit-0",
            "by_process_slot": {"0": "worker-0"},
            "groups": {
                "world": {
                    "kind": "vllm:world",
                    "size": 1,
                    "ranks": {"0": "worker-0"},
                }
            },
        }
    )


class DiskSleepIoTest(unittest.TestCase):
    def test_recovery_filesystem_probe_follows_huggingface_blob_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blob = root / "blobs" / "sha256-test"
            snapshot = root / "snapshots" / "revision"
            blob.parent.mkdir()
            snapshot.mkdir(parents=True)
            blob.write_bytes(b"safetensors")
            shard = snapshot / "layers-0.safetensors"
            shard.symlink_to(blob)

            filesystem = recovery_loader._filesystem_for_open_file(shard)
            value = blob.stat()

        self.assertEqual(filesystem["device"], f"{os.major(value.st_dev)}:{os.minor(value.st_dev)}")

    def test_recovery_auto_selects_io_from_opened_filesystem(self) -> None:
        path = Path("/models/model.safetensors")
        with patch.object(
            recovery_loader,
            "_filesystem_for_open_file",
            return_value={
                "filesystem_type": "nfs4",
                "mount_source": "server:/models",
                "mountpoint": "/models",
                "device": "0:42",
            },
        ):
            effective, remote = recovery_loader._select_recovery_io(path, "auto")
        self.assertEqual(effective, "buffered")
        self.assertEqual(remote["reason"], "remote-filesystem-prefers-page-cache")
        self.assertEqual(remote["mount_source"], "server:/models")

        with patch.object(
            recovery_loader,
            "_filesystem_for_open_file",
            return_value={
                "filesystem_type": "xfs",
                "mount_source": "/dev/nvme0n1p1",
                "mountpoint": "/models",
                "device": "259:1",
            },
        ):
            effective, local = recovery_loader._select_recovery_io(path, "auto")
        self.assertEqual(effective, "direct")
        self.assertEqual(local["reason"], "qualified-local-filesystem")

    def test_explicit_recovery_io_policy_overrides_filesystem_heuristic(self) -> None:
        with patch.object(
            recovery_loader,
            "_filesystem_for_open_file",
            return_value={
                "filesystem_type": "nfs",
                "mount_source": "server:/models",
                "mountpoint": "/models",
                "device": "0:43",
            },
        ):
            effective, observation = recovery_loader._select_recovery_io(
                Path("/models/model.safetensors"),
                "direct",
            )
        self.assertEqual(effective, "direct")
        self.assertEqual(observation["reason"], "explicit-policy")

    def test_recovery_metrics_total_includes_sequential_adapter_replay(self) -> None:
        metrics = recovery_loader.RecoveryLoadMetrics(
            files=1,
            tensors=2,
            logical_bytes=1024,
            local_read_bytes=512,
            local_read_extents=1,
            io_seconds=0.4,
            collective_seconds=0.2,
            collective_calls=1,
            verification_seconds=0.1,
            verified_sample_bytes=64,
            total_seconds=1.5,
            backend="direct",
            distributed_world_size=2,
            staging_layout="test",
            staging_buffer_bytes=4096,
            collective_buffer_bytes=4096,
        )
        previous_metrics = recovery_loader._last_metrics
        previous_replay = recovery_loader._last_direct_replay_metrics
        try:
            recovery_loader._last_metrics = metrics
            recovery_loader._last_direct_replay_metrics = {"total_seconds": 2.25}
            reported = recovery_loader.last_recovery_load_metrics()
        finally:
            recovery_loader._last_metrics = previous_metrics
            recovery_loader._last_direct_replay_metrics = previous_replay

        self.assertIsNotNone(reported)
        self.assertEqual(reported["loader_seconds"], 1.5)
        self.assertEqual(reported["total_seconds"], 3.75)

    def test_n580_recovery_load_publishes_worker_timing_state(self) -> None:
        metrics = recovery_loader.RecoveryLoadMetrics(
            files=2,
            tensors=4,
            logical_bytes=1024,
            local_read_bytes=512,
            local_read_extents=2,
            io_seconds=0.4,
            collective_seconds=0.2,
            collective_calls=1,
            verification_seconds=0.1,
            verified_sample_bytes=64,
            total_seconds=1.5,
            backend="direct",
            distributed_world_size=2,
            staging_layout="test",
            staging_buffer_bytes=4096,
            collective_buffer_bytes=4096,
        )
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                "COLDSNAP_HIBERNATE_STATE_DIR": directory,
                "COLDSNAP_EXECUTION_GRAPH": json.dumps(
                    {
                        "unit": "unit-a",
                        "by_process_slot": {"0": "worker-a"},
                        "groups": {},
                    }
                ),
                "LOCAL_RANK": "0",
            }
            with patch.dict(os.environ, environment, clear=True):
                recovery_loader._process_template_load_metrics = []
                recovery_loader._publish_process_template_load_metrics(metrics)
                recovery_loader._publish_process_template_load_metrics(metrics)

            state = json.loads((Path(directory) / "worker-a.json").read_text(encoding="utf-8"))

        self.assertEqual(state["worker_id"], "worker-a")
        self.assertEqual(state["resume_seconds"], 3.0)
        self.assertEqual(state["phase_seconds"]["recovery_loader"]["io_seconds"], 0.8)
        self.assertEqual(state["hydration_backend"], "recovery-safetensors+direct")
        recovery_loader._process_template_load_metrics = []

    def test_hibernation_worker_uses_runtime_rank_and_unit_mapping(self) -> None:
        with patch.dict(
            os.environ,
            {
                "COLDSNAP_EXPECTED_RANK": "1",
                "COLDSNAP_RANK": "0",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "COLDSNAP_EXECUTION_GRAPH": json.dumps(
                    {"unit": "unit-a", "by_process_slot": {"0": "worker-a"}, "groups": {}}
                ),
                "VLLM_HOST_IP": "node-a",
            },
            clear=True,
        ):
            self.assertEqual(_rank(), 0)
            self.assertIn("worker-id-worker-a-host-node-a-rank-0-pid-", _identity())

    def test_hibernation_identity_requires_a_global_rank(self) -> None:
        environment = {
            "COLDSNAP_EXECUTION_GRAPH": json.dumps(
                {"unit": "unit-a", "by_process_slot": {"0": "worker-a"}, "groups": {}}
            ),
            "VLLM_HOST_IP": "node-a",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            self.assertRaisesRegex(RuntimeError, "cannot resolve ColdSnap global rank"),
        ):
            _identity()

    def test_hibernation_identity_derives_rank_from_engine_world_topology(self) -> None:
        environment = {
            "COLDSNAP_EXECUTION_GRAPH": json.dumps(
                {
                    "unit": "unit-a",
                    "by_process_slot": {"0": "worker-a"},
                    "groups": {
                        "world": {
                            "kind": "vllm:world",
                            "size": 2,
                            "ranks": {"1": "worker-a"},
                        }
                    },
                }
            ),
            "VLLM_HOST_IP": "node-a",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(_rank(), 1)
            self.assertIn("worker-id-worker-a-host-node-a-rank-1-pid-", _identity())

    def test_snapshot_directory_uses_topology_rank_without_rank_environment(
        self,
    ) -> None:
        environment = {
            "COLDSNAP_EXECUTION_GRAPH": json.dumps(
                {
                    "unit": "unit-a",
                    "by_process_slot": {"0": "worker-a"},
                    "groups": {
                        "world": {
                            "kind": "vllm:world",
                            "size": 2,
                            "ranks": {"1": "worker-a"},
                        }
                    },
                }
            ),
            "VLLM_HOST_IP": "node-a",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
        ):
            resolved = disk_backend._snapshot_directory(Path(directory))

        self.assertEqual(
            resolved.name,
            f"worker-id-worker-a-host-node-a-rank-1-pid-{os.getpid()}",
        )

    def test_failed_pre_unmap_suspend_requires_no_snapshot_restore(self) -> None:
        allocator = SimpleNamespace(
            pointer_to_data={
                1: SimpleNamespace(tag="weights", is_asleep=False, handle=(1, 1)),
                2: SimpleNamespace(tag="kv_cache", is_asleep=True, handle=(2, 1)),
            }
        )
        self.assertFalse(_has_asleep_weights(allocator))
        allocator.pointer_to_data[1].is_asleep = True
        self.assertTrue(_has_asleep_weights(allocator))

    def test_failed_read_retry_replays_every_byte_without_double_mapping(self) -> None:
        fixture = self._restore_fixture(b"abcdefgh")
        original = disk_backend._timed_read
        failed = False

        def fail_once(fd: int, view: memoryview, offset: int) -> float:
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("injected read failure")
            return original(fd, view, offset)

        with fixture["modules"], patch.object(disk_backend, "_timed_read", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected read failure"):
                fixture["backend"]._restore_pipeline(
                    fixture["allocator"], fixture["manifest"], fixture["fd"]
                )
        self.assertTrue(fixture["data"].is_asleep)
        self.assertEqual(fixture["map_calls"], [fixture["data"].handle])

        with fixture["modules"]:
            metrics = fixture["backend"]._restore_pipeline(
                fixture["allocator"], fixture["manifest"], fixture["fd"]
            )
        fixture["backend"]._commit_restore(fixture["allocator"])
        self.assertEqual(metrics["restored_bytes"], 8)
        self.assertEqual(fixture["map_calls"], [fixture["data"].handle])
        self.assertFalse(fixture["data"].is_asleep)
        self.assertEqual(fixture["device_bytes"](), b"abcdefgh")
        os.close(fixture["fd"])

    def test_failed_cuda_copy_and_checksum_remain_retryable(self) -> None:
        fixture = self._restore_fixture(b"abcdefgh", fail_cuda_once=True)
        with fixture["modules"]:
            with self.assertRaisesRegex(RuntimeError, "injected CUDA copy failure"):
                fixture["backend"]._restore_pipeline(
                    fixture["allocator"], fixture["manifest"], fixture["fd"]
                )
        self.assertTrue(fixture["data"].is_asleep)
        fixture["manifest"]["entries"][0]["crc32"] = "00000000"
        with fixture["modules"]:
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                fixture["backend"]._restore_pipeline(
                    fixture["allocator"], fixture["manifest"], fixture["fd"]
                )
        self.assertTrue(fixture["data"].is_asleep)
        fixture["manifest"]["entries"][0]["crc32"] = f"{zlib.crc32(b'abcdefgh') & 0xFFFFFFFF:08x}"
        with fixture["modules"]:
            fixture["backend"]._restore_pipeline(
                fixture["allocator"], fixture["manifest"], fixture["fd"]
            )
        fixture["backend"]._commit_restore(fixture["allocator"])
        self.assertEqual(fixture["map_calls"], [fixture["data"].handle])
        self.assertEqual(fixture["device_bytes"](), b"abcdefgh")
        os.close(fixture["fd"])

    def _restore_fixture(self, payload: bytes, fail_cuda_once: bool = False) -> dict[str, object]:
        temporary = tempfile.TemporaryFile()
        temporary.write(payload)
        temporary.flush()
        fd = os.dup(temporary.fileno())
        temporary.close()

        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        backend.chunk_bytes = 4
        backend.pipeline_depth = 1
        backend.verify_mode = "inline"
        backend._restore_mapped = set()
        stages = []

        class Stage:
            def __init__(self) -> None:
                self.buffer = bytearray(4)

            def data_ptr(self) -> object:
                return self

        for _ in range(backend.pipeline_depth):
            stage = Stage()
            stages.append(HostStage(owner=stage, view=memoryview(stage.buffer), pointer=stage))
        backend._stages = lambda count: stages[:count]

        data = SimpleNamespace(tag="weights", is_asleep=True, handle=(99, len(payload)))
        allocator = SimpleNamespace(pointer_to_data={1000: data})
        manifest = {
            "entries": [
                {
                    "ptr": 1000,
                    "size": len(payload),
                    "offset": 0,
                    "tag": "weights",
                    "crc32": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
                }
            ]
        }
        map_calls = []
        device: dict[int, bytes] = {}

        class FakeCudaRuntime:
            failed = False

            @classmethod
            def cudaMemcpy(cls, destination: int, source: Stage, count: int) -> None:
                if fail_cuda_once and not cls.failed:
                    cls.failed = True
                    raise RuntimeError("injected CUDA copy failure")
                device[destination] = bytes(source.buffer[:count])

        cumem = ModuleType("vllm.device_allocator.cumem")
        cumem.create_and_map = map_calls.append
        cumem.unmap_and_release = lambda handle: None
        cumem.libcudart = FakeCudaRuntime
        modules = patch.dict(
            sys.modules,
            {
                "vllm": ModuleType("vllm"),
                "vllm.device_allocator": ModuleType("vllm.device_allocator"),
                "vllm.device_allocator.cumem": cumem,
            },
        )
        return {
            "backend": backend,
            "allocator": allocator,
            "data": data,
            "manifest": manifest,
            "map_calls": map_calls,
            "modules": modules,
            "fd": fd,
            "device_bytes": lambda: b"".join(device[key] for key in sorted(device)),
        }

    def test_hibernation_chunk_descriptors_preserve_allocation_identity(self) -> None:
        entries = [
            {"ptr": 1000, "size": 10, "offset": 0},
            {"ptr": 2000, "size": 5, "offset": 10},
        ]
        self.assertEqual(
            _chunk_descriptors(entries, 4),
            [
                (1000, 0, 4, 1000),
                (1004, 4, 4, 1000),
                (1008, 8, 2, 1000),
                (2000, 10, 4, 2000),
                (2004, 14, 1, 2000),
            ],
        )

    def test_hibernation_io_modes_and_preallocation(self) -> None:
        with patch.dict(os.environ, {"TEST_HIBERNATION_IO_MODE": "buffered"}, clear=False):
            self.assertFalse(_io_direct("TEST_HIBERNATION_IO_MODE", "direct"))
        with patch.dict(os.environ, {"TEST_HIBERNATION_IO_MODE": "invalid"}, clear=False):
            with self.assertRaisesRegex(ValueError, "buffered or direct"):
                _io_direct("TEST_HIBERNATION_IO_MODE", "direct")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preallocated"
            fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
            try:
                _preallocate(fd, 8192)
                self.assertEqual(os.fstat(fd).st_size, 8192)
            finally:
                os.close(fd)
        if hasattr(os, "posix_fallocate"):
            with (
                tempfile.TemporaryFile() as stream,
                patch.object(
                    os,
                    "posix_fallocate",
                    side_effect=OSError(errno.ENOSPC, "full"),
                ),
            ):
                with self.assertRaises(OSError) as raised:
                    _preallocate(stream.fileno(), 8192)
                self.assertEqual(raised.exception.errno, errno.ENOSPC)

    def test_hibernation_reuses_only_the_restored_blob_generation(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "COLDSNAP_EXPECTED_RANK": "0",
                    "COLDSNAP_EXECUTION_GRAPH": _single_worker_execution_graph(),
                    "COLDSNAP_CAPTURE_ID": "capture",
                    "VLLM_HOST_IP": "node-a",
                },
                clear=True,
            ),
        ):
            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = Path(directory)
            backend.blob_path = backend.snapshot_dir / "weights.blob"
            backend.manifest_path = backend.snapshot_dir / "manifest.json"
            backend.verify_mode = "inline"
            backend.reuse_blob = True
            backend.blob_path.write_bytes(b"abcdefgh")
            blob_stat = disk_backend._stat_identity(backend.blob_path)
            generation = "restored-generation"
            manifest = {
                "format": disk_backend.FORMAT,
                "kind": disk_backend.KIND,
                "pid": os.getpid(),
                "identity": _identity(),
                "rank": 0,
                "worker_id": disk_backend._worker_id(),
                "capture_id": "capture",
                "blob": backend.blob_path.name,
                "blob_bytes": 8,
                "blob_stat": blob_stat,
                "generation": generation,
                "direct_io": False,
                "write_io_mode": "buffered",
                "entries": [
                    {
                        "ptr": 1000,
                        "size": 8,
                        "tag": "weights",
                        "offset": 0,
                        "crc32": f"{zlib.crc32(b'abcdefgh') & 0xFFFFFFFF:08x}",
                    }
                ],
            }
            backend.manifest_path.write_text(__import__("json").dumps(manifest))
            backend._reusable_generation = generation
            backend._reusable_blob_stat = blob_stat
            allocator = SimpleNamespace(
                pointer_to_data={
                    1000: SimpleNamespace(tag="weights", handle=(9, 8), is_asleep=False)
                }
            )

            reused = backend._reusable_snapshot(allocator)
            self.assertIsNotNone(reused)
            self.assertTrue(reused["blob_reused"])
            self.assertEqual(reused["write_io_mode"], "reused")
            self.assertEqual(reused["phase_seconds"]["disk_write_s"], 0.0)

            backend.blob_path.write_bytes(b"changed!")
            self.assertIsNone(backend._reusable_snapshot(allocator))
            self.assertIsNone(backend._reusable_generation)

    def test_native_bootstrap_primes_captured_pack_without_writing_fake_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = Path(directory)
            backend.blob_path = backend.snapshot_dir / disk_backend.MODEL_PAYLOAD_NAME
            backend.blob_path.write_bytes(b"captured-model")
            (backend.snapshot_dir / disk_backend.ACTIVATION_PROVIDER_NAME).write_text(
                "native\n", encoding="utf-8"
            )
            # n580 restores the capture-time environment, where reuse was
            # disabled.  Its activation-local native selector must supersede
            # that stale policy.
            backend.reuse_blob = False
            backend._reusable_generation = None
            backend._reusable_blob_stat = None
            manifest = {
                "weight_source": disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
                "generation": "captured-generation",
            }
            loads = []

            def load(_allocator):
                loads.append(True)
                return manifest

            backend._load_manifest = load

            reused = backend._reusable_snapshot(object())

        self.assertIsNotNone(reused)
        self.assertEqual(loads, [True])
        self.assertTrue(backend.reuse_blob)
        self.assertTrue(reused["blob_reused"])
        self.assertEqual(reused["generation"], "captured-generation")
        self.assertEqual(reused["write_seconds"] >= 0, True)

    def test_native_bootstrap_precedes_retained_capture_export_policy(self) -> None:
        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_BLOB
        memory = SimpleNamespace(allocations=lambda _region: ())
        reused = {
            "weight_source": disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
            "generation": "captured-generation",
        }
        with (
            patch.dict(
                os.environ,
                {"COLDSNAP_EXPORT_MODEL_PAYLOAD": "1"},
                clear=True,
            ),
            patch.object(backend, "_reusable_snapshot", return_value=reused) as reuse,
            patch.object(backend, "_write_recovery_manifest") as rewrite,
        ):
            result = backend._write_snapshot(memory)

        self.assertIs(result, reused)
        reuse.assert_called_once_with(memory)
        rewrite.assert_not_called()

    def test_native_capture_writes_primary_snapshot_and_skips_python_readback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = root
            backend.blob_path = root / "weights.blob"
            backend.manifest_path = root / "manifest.json"
            backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_BLOB
            backend.reuse_blob = False
            backend._reusable_generation = None
            backend._reusable_blob_stat = None
            backend.verify_mode = "preverified"
            backend.chunk_bytes = 4096
            backend.pipeline_depth = 2
            backend.capture_backend = "auto"
            backend._ensure_space = lambda _required: None
            allocations = [
                SimpleNamespace(pointer=0x10000, size=8, tag="weights"),
                SimpleNamespace(pointer=0x20000, size=4, tag="weights"),
            ]
            memory = SimpleNamespace(
                allocations=lambda region: allocations if region == "weights" else []
            )
            calls = []

            class CaptureTransport:
                @staticmethod
                def capture(path, extents, **options):
                    calls.append((extents, options))
                    Path(path).write_bytes(b"abcdefghijkl")
                    return SimpleNamespace(
                        metrics=SimpleNamespace(
                            backend="buffered",
                            bytes=12,
                            cuda_enqueue_s=0.1,
                            cuda_synchronize_s=0.2,
                            checksum_s=0.3,
                            io_service_s=0.4,
                            initialization_s=0.5,
                            durability_s=0.6,
                            io_wait_s=0.7,
                            verification_s=0.8,
                        ),
                        digests=(
                            SimpleNamespace(crc32="1234abcd"),
                            SimpleNamespace(crc32="89abcdef"),
                        ),
                    )

            backend.capture_transport = CaptureTransport()
            environment = {
                "COLDSNAP_EXECUTION_GRAPH": _single_worker_execution_graph(),
                "LOCAL_RANK": "0",
                "VLLM_HOST_IP": "node-a",
            }
            with patch.dict(os.environ, environment, clear=True), patch.object(
                backend, "_verify_blob", side_effect=AssertionError("Python readback used")
            ):
                manifest = backend._write_snapshot(memory)
            captured_bytes = backend.blob_path.read_bytes()

        self.assertEqual(captured_bytes, b"abcdefghijkl")
        self.assertEqual([entry["crc32"] for entry in manifest["entries"]], ["1234abcd", "89abcdef"])
        self.assertEqual(manifest["write_io_mode"], "buffered")
        self.assertTrue(manifest["preverified"])
        self.assertEqual(manifest["verification_seconds"], 0.8)
        self.assertEqual(manifest["phase_seconds"]["io_wait_s"], 0.7)
        self.assertEqual([extent.file_offset for extent in calls[0][0]], [0, 8])
        self.assertTrue(calls[0][1]["verify_readback"])

    def test_completed_model_payload_export_is_not_repeated_after_restore(self) -> None:
        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_BLOB
        backend._model_payload_exported = True
        memory = SimpleNamespace(allocations=lambda _region: ())
        residual = {"weight_source": disk_backend.WEIGHT_SOURCE_SAFETENSORS}
        with (
            patch.dict(
                os.environ,
                {"COLDSNAP_EXPORT_MODEL_PAYLOAD": "1"},
                clear=True,
            ),
            patch.object(backend, "_reusable_snapshot", return_value=None),
            patch.object(
                backend, "_write_recovery_manifest", return_value=residual
            ) as write_residual,
        ):
            self.assertTrue(backend.model_payload_capture_enabled)
            self.assertFalse(backend.exports_model_payload)
            result = backend._write_snapshot(memory)

        self.assertIs(result, residual)
        write_residual.assert_called_once_with(memory)

    def test_native_activation_is_visible_before_first_manifest_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = Path(directory)
            backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_SAFETENSORS

            self.assertFalse(backend.uses_native_model_payload)
            (backend.snapshot_dir / disk_backend.ACTIVATION_PROVIDER_NAME).write_text(
                "recovery\n", encoding="utf-8"
            )
            self.assertFalse(backend.uses_native_model_payload)
            (backend.snapshot_dir / disk_backend.ACTIVATION_PROVIDER_NAME).write_text(
                "native\n", encoding="utf-8"
            )
            self.assertTrue(backend.uses_native_model_payload)

            backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE
            (backend.snapshot_dir / disk_backend.ACTIVATION_PROVIDER_NAME).unlink()
            self.assertTrue(backend.uses_native_model_payload)

    def test_native_live_sleep_binds_full_semantic_payload_layout(self) -> None:
        self.assertEqual(
            recovery_loader._sleep_layout_binding_policy(
                SimpleNamespace(
                    uses_model_weight_recovery=False,
                    exports_model_payload=False,
                    uses_native_model_payload=True,
                )
            ),
            (True, True),
        )
        self.assertEqual(
            recovery_loader._sleep_layout_binding_policy(
                SimpleNamespace(
                    uses_model_weight_recovery=True,
                    exports_model_payload=False,
                    uses_native_model_payload=False,
                )
            ),
            (True, False),
        )

    def test_portable_native_manifest_relocates_to_equivalent_live_layout(self) -> None:
        captured = [
            {"ptr": 1000, "size": 100, "tag": "weights"},
            {"ptr": 2000, "size": 200, "tag": "weights"},
        ]
        manifest = {
            "weight_source": disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
            "portable_model_payload": True,
            "entries": captured,
            "model_weight_extents": [
                {
                    "ptr": 2020,
                    "size": 80,
                    "allocation_ptr": 2000,
                    "semantic_id": "b" * 64,
                    "offset": 0,
                    "crc32": "00000000",
                },
                {
                    "ptr": 1010,
                    "size": 40,
                    "allocation_ptr": 1000,
                    "semantic_id": "a" * 64,
                    "offset": 4096,
                    "crc32": "00000000",
                },
            ],
            "residual_extents": [
                {"ptr": 1000, "size": 10, "allocation_ptr": 1000, "offset": 0},
                {"ptr": 1050, "size": 50, "allocation_ptr": 1000, "offset": 10},
                {"ptr": 2000, "size": 20, "allocation_ptr": 2000, "offset": 60},
                {"ptr": 2100, "size": 100, "allocation_ptr": 2000, "offset": 80},
            ],
        }
        live = [
            SimpleNamespace(pointer=5000, size=100, tag="weights"),
            SimpleNamespace(pointer=9000, size=200, tag="weights"),
        ]

        relocated = disk_backend._relocate_split_native_manifest(
            manifest,
            live,
            ((5010, 40), (9020, 80)),
            ((5010, 40, "a" * 64), (9020, 80, "b" * 64)),
        )

        self.assertIsNot(relocated, manifest)
        self.assertTrue(relocated["relocated_from_captured_va"])
        self.assertEqual(
            [(entry["ptr"], entry["size"]) for entry in relocated["entries"]],
            [(5000, 100), (9000, 200)],
        )
        self.assertEqual(
            sorted(
                (extent["ptr"], extent["allocation_ptr"])
                for extent in relocated["model_weight_extents"]
            ),
            [(5010, 5000), (9020, 9000)],
        )

    def test_portable_native_manifest_accepts_coalesced_live_allocations(self) -> None:
        manifest = {
            "weight_source": disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
            "portable_model_payload": True,
            "entries": [
                {"ptr": 1000, "size": 20, "tag": "weights"},
                {"ptr": 2000, "size": 20, "tag": "weights"},
            ],
            # File order is canonical hash order, not address order. Relocation
            # must retain that order and each extent's packed-file identity.
            "model_weight_extents": [
                {
                    "ptr": 2000,
                    "size": 10,
                    "allocation_ptr": 2000,
                    "semantic_id": "b" * 64,
                    "offset": 0,
                    "crc32": "22222222",
                },
                {
                    "ptr": 1000,
                    "size": 10,
                    "allocation_ptr": 1000,
                    "semantic_id": "a" * 64,
                    "offset": 4096,
                    "crc32": "11111111",
                },
            ],
            "residual_extents": [
                {
                    "ptr": 1010,
                    "size": 10,
                    "allocation_ptr": 1000,
                    "offset": 0,
                    "crc32": "33333333",
                },
                {
                    "ptr": 2010,
                    "size": 10,
                    "allocation_ptr": 2000,
                    "offset": 10,
                    "crc32": "44444444",
                },
            ],
        }
        live = [SimpleNamespace(pointer=5000, size=40, tag="weights")]

        relocated = disk_backend._relocate_split_native_manifest(
            manifest,
            live,
            ((5000, 10), (5020, 10)),
            ((5000, 10, "a" * 64), (5020, 10, "b" * 64)),
        )

        self.assertEqual(
            relocated["entries"],
            [{"ptr": 5000, "size": 40, "tag": "weights"}],
        )
        self.assertEqual(
            [
                (extent["ptr"], extent["offset"], extent["crc32"])
                for extent in relocated["model_weight_extents"]
            ],
            [(5020, 0, "22222222"), (5000, 4096, "11111111")],
        )
        self.assertEqual(
            [extent["ptr"] for extent in relocated["residual_extents"]],
            [5010, 5030],
        )

    def test_portable_native_manifest_rebinds_changed_residual_capacity(self) -> None:
        manifest = {
            "weight_source": disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
            "portable_model_payload": True,
            "allocation_bytes": 40,
            "residual_bytes": 20,
            "entries": [
                {"ptr": 1000, "size": 20, "tag": "weights"},
                {"ptr": 2000, "size": 20, "tag": "weights"},
            ],
            "model_weight_extents": [
                {
                    "ptr": 1000,
                    "size": 10,
                    "allocation_ptr": 1000,
                    "semantic_id": "a" * 64,
                    "offset": 0,
                    "crc32": "11111111",
                    "sha256": "1" * 64,
                },
                {
                    "ptr": 2000,
                    "size": 10,
                    "allocation_ptr": 2000,
                    "semantic_id": "b" * 64,
                    "offset": 4096,
                    "crc32": "22222222",
                    "sha256": "2" * 64,
                },
            ],
            "residual_extents": [
                {
                    "ptr": 1010,
                    "size": 10,
                    "allocation_ptr": 1000,
                    "offset": 0,
                    "crc32": "33333333",
                },
                {
                    "ptr": 2010,
                    "size": 10,
                    "allocation_ptr": 2000,
                    "offset": 10,
                    "crc32": "44444444",
                },
            ],
        }
        live = [SimpleNamespace(pointer=5000, size=38, tag="weights")]

        relocated = disk_backend._relocate_split_native_manifest(
            manifest,
            live,
            ((5000, 10), (5020, 10)),
            ((5000, 10, "a" * 64), (5020, 10, "b" * 64)),
        )

        self.assertTrue(relocated["requires_live_residual_snapshot"])
        self.assertEqual(relocated["allocation_bytes"], 38)
        self.assertEqual(relocated["residual_bytes"], 18)
        self.assertEqual(
            relocated["residual_extents"],
            [
                {"ptr": 5010, "size": 10, "allocation_ptr": 5000},
                {"ptr": 5030, "size": 8, "allocation_ptr": 5000},
            ],
        )

    def test_portable_native_manifest_rejects_model_layout_drift(self) -> None:
        manifest = {
            "weight_source": disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
            "portable_model_payload": True,
            "entries": [{"ptr": 1000, "size": 100, "tag": "weights"}],
            "model_weight_extents": [
                {
                    "ptr": 1010,
                    "size": 40,
                    "allocation_ptr": 1000,
                    "semantic_id": "a" * 64,
                }
            ],
            "residual_extents": [
                {"ptr": 1000, "size": 10, "allocation_ptr": 1000},
                {"ptr": 1050, "size": 50, "allocation_ptr": 1000},
            ],
        }
        live = [SimpleNamespace(pointer=5000, size=100, tag="weights")]

        with self.assertRaisesRegex(RuntimeError, "portable native .*model"):
            disk_backend._relocate_split_native_manifest(
                manifest,
                live,
                ((5010, 39),),
                ((5010, 39, "a" * 64),),
            )

    def test_native_bootstrap_never_falls_back_to_rewriting_staged_pack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = Path(directory)
            backend.blob_path = backend.snapshot_dir / disk_backend.MODEL_PAYLOAD_NAME
            backend.blob_path.write_bytes(b"captured-model")
            (backend.snapshot_dir / disk_backend.ACTIVATION_PROVIDER_NAME).write_text(
                "native\n", encoding="utf-8"
            )
            backend.reuse_blob = True
            backend._reusable_generation = None
            backend._reusable_blob_stat = None
            backend._load_manifest = lambda _allocator: {
                "weight_source": disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
                "generation": "captured-generation",
            }

            original = disk_backend._stat_identity
            calls = 0

            def changed_stat(path):
                nonlocal calls
                value = original(path)
                calls += 1
                if calls > 1:
                    value["mtime_ns"] += 1
                return value

            with patch.object(disk_backend, "_stat_identity", changed_stat):
                with self.assertRaisesRegex(RuntimeError, "restored snapshot generation changed"):
                    backend._reusable_snapshot(object())

            self.assertEqual(backend.blob_path.read_bytes(), b"captured-model")

    def test_recovery_manifest_stores_only_residual_and_binds_model_revision(self) -> None:
        allocation = SimpleNamespace(pointer=1000, size=8192, tag="weights")

        class Memory:
            @staticmethod
            def allocations(tag=None):
                return [allocation]

            @staticmethod
            def allocate_host_stage(size):
                owner = bytearray(size)
                return HostStage(owner=owner, view=memoryview(owner), pointer=0)

            @staticmethod
            def copy_to_host(stage, pointer, size):
                stage.view[:size] = bytes([pointer % 251]) * size

        memory = Memory()
        environment = {
            "COLDSNAP_CAPTURE_ID": "capture",
            "COLDSNAP_EXPECTED_RANK": "0",
            "COLDSNAP_EXECUTION_GRAPH": _single_worker_execution_graph(),
            "COLDSNAP_MODEL_ID": "model",
            "COLDSNAP_MODEL_REVISION": "pinned-revision",
            "VLLM_HOST_IP": "node-a",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
        ):
            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = Path(directory)
            backend.blob_path = backend.snapshot_dir / "weights.blob"
            backend.manifest_path = backend.snapshot_dir / "manifest.json"
            backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_SAFETENSORS
            backend._model_weight_ranges = ((2000, 4096),)
            backend.chunk_bytes = 4096
            backend.verify_mode = "inline"
            backend.read_direct = False
            backend._stage_cache = []
            backend.memory_provider = memory
            backend.blob_path.write_bytes(b"obsolete")

            manifest = backend._write_recovery_manifest(memory)

            self.assertTrue(backend.blob_path.is_file())
            self.assertEqual(manifest["blob"], "weights.blob")
            self.assertEqual(manifest["blob_bytes"], 4096)
            self.assertIs(manifest["portable_residual_blob"], True)
            self.assertEqual(manifest["write_io_mode"], "hybrid-residual-buffered")
            self.assertEqual(manifest["allocation_bytes"], 8192)
            self.assertEqual(manifest["model_weight_bytes"], 4096)
            self.assertEqual(manifest["residual_bytes"], 4096)
            self.assertEqual(
                [(item["ptr"], item["size"]) for item in manifest["model_weight_extents"]],
                [(2000, 4096)],
            )
            self.assertEqual(
                manifest["model_source"],
                {
                    "model_id": "model",
                    "revision": "pinned-revision",
                    "metadata_sha256": "",
                },
            )
            self.assertEqual(backend._load_manifest(memory), manifest)

            captured_manifest = dict(manifest)
            captured_manifest.update(
                {
                    "pid": 614,
                    "identity": (
                        "worker-id-worker-0-host-capture-host-rank-0-pid-614"
                    ),
                }
            )
            backend.manifest_path.write_text(json.dumps(captured_manifest))
            allocation.pointer = 1200
            allocation.size = 9216
            with patch.dict(
                os.environ,
                {
                    "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                    "VLLM_HOST_IP": "node-b",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "allocation map differs"):
                    backend._load_manifest(
                        memory, allow_process_template_owner=True
                    )
                with self.assertRaisesRegex(
                    RuntimeError, "layout admission requires"
                ):
                    backend._load_manifest(
                        memory, allow_materialization_layout=True
                    )
                self.assertEqual(
                    backend._load_manifest(
                        memory,
                        allow_process_template_owner=True,
                        allow_materialization_layout=True,
                    ),
                    captured_manifest,
                )
            allocation.pointer = 1000
            allocation.size = 8192
            backend.manifest_path.write_text(json.dumps(manifest))

            blob_stat = backend.blob_path.stat()
            os.utime(
                backend.blob_path,
                ns=(blob_stat.st_atime_ns, blob_stat.st_mtime_ns + 1),
            )
            self.assertEqual(backend._load_manifest(memory), manifest)

            local_only = dict(manifest)
            local_only["portable_residual_blob"] = False
            backend.manifest_path.write_text(json.dumps(local_only))
            with self.assertRaisesRegex(RuntimeError, "manifest is invalid"):
                backend._load_manifest(memory)
            backend.manifest_path.write_text(json.dumps(manifest))

            os.environ["COLDSNAP_MODEL_REVISION"] = "different-revision"
            with self.assertRaisesRegex(RuntimeError, "manifest is invalid"):
                backend._load_manifest(memory)

    def test_recovery_capture_exports_split_native_model_payload(self) -> None:
        allocation = SimpleNamespace(pointer=1000, size=8192, tag="weights", is_released=False)
        restored = bytearray(allocation.size)

        class Memory:
            @staticmethod
            def allocations(tag=None):
                return [allocation]

            @staticmethod
            def allocate_host_stage(size):
                owner = bytearray(size)
                return HostStage(owner=owner, view=memoryview(owner), pointer=0)

            @staticmethod
            def copy_to_host(stage, pointer, size):
                stage.view[:size] = bytes([pointer % 251]) * size

            @staticmethod
            def copy_from_host(pointer, stage, size):
                offset = pointer - allocation.pointer
                restored[offset : offset + size] = stage.view[:size]

            @staticmethod
            def remap(item):
                if item is not allocation:
                    raise AssertionError("unexpected allocation")

        memory = Memory()
        environment = {
            "COLDSNAP_CAPTURE_ID": "capture",
            "COLDSNAP_EXPECTED_RANK": "0",
            "COLDSNAP_EXECUTION_GRAPH": _single_worker_execution_graph(),
            "COLDSNAP_MODEL_ID": "model",
            "COLDSNAP_MODEL_REVISION": "pinned-revision",
            "COLDSNAP_EXPORT_MODEL_PAYLOAD": "1",
            "VLLM_HOST_IP": "node-a",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
        ):
            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = Path(directory)
            backend.blob_path = backend.snapshot_dir / "weights.blob"
            backend.manifest_path = backend.snapshot_dir / "manifest.json"
            # Model-payload export is also used by n580 capture, where vLLM's
            # normal loader remains active and no recovery-loader wake occurs.
            backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_BLOB
            backend._model_weight_ranges = ((2000, 4097),)
            backend._native_model_payload_semantics = (
                (1500, 128, "a" * 64),
                (2000, 4097, "b" * 64),
            )
            backend.chunk_bytes = 8192
            backend.verify_mode = "preverified"
            backend.read_direct = False
            backend.write_direct = False
            backend._stage_cache = []
            backend.memory_provider = memory

            recovery = backend._write_snapshot(memory)

            self.assertEqual(recovery["model_payload"]["bytes"], 12288)
            self.assertTrue(recovery["model_payload_exported"])
            self.assertTrue(backend._model_payload_exported)
            self.assertFalse(backend.exports_model_payload)
            self.assertGreaterEqual(
                recovery["write_seconds"],
                recovery["model_payload"]["write_seconds"],
            )
            self.assertEqual(
                (backend.snapshot_dir / disk_backend.MODEL_PAYLOAD_NAME).stat().st_size,
                12288,
            )
            self.assertEqual(backend.blob_path.stat().st_size, 4095)
            self.assertEqual(
                (backend.snapshot_dir / disk_backend.NATIVE_RESIDUAL_NAME).stat().st_size,
                3967,
            )
            (backend.snapshot_dir / disk_backend.ACTIVATION_PROVIDER_NAME).write_text(
                "native\n", encoding="utf-8"
            )
            # Model payload ownership survives the n580 pre-exec transition:
            # the controller-selected directory retains its capture host/PID
            # and exact execution-topology rank.
            native_manifest_path = backend.snapshot_dir / disk_backend.NATIVE_MANIFEST_NAME
            portable = json.loads(native_manifest_path.read_text(encoding="utf-8"))
            portable.update(
                {
                    "pid": 614,
                    "rank": 0,
                    "identity": ("worker-id-worker-0-host-capture-host-rank-0-pid-614"),
                }
            )
            native_manifest_path.write_text(json.dumps(portable), encoding="utf-8")
            os.environ["COLDSNAP_PROCESS_TEMPLATE_RESTORED"] = "1"
            os.environ["VLLM_HOST_IP"] = "node-b"
            native = backend._load_manifest(memory)
            self.assertEqual(native["weight_source"], disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE)
            self.assertEqual(native["blob"], disk_backend.MODEL_PAYLOAD_NAME)
            self.assertEqual(native["model_weight_bytes"], 4225)
            self.assertEqual(native["residual_bytes"], 3967)
            self.assertEqual(native["residual_blob"], disk_backend.NATIVE_RESIDUAL_NAME)
            self.assertEqual(
                backend.weight_recovery_source,
                disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
            )
            backend.hydrator = None
            backend._restore_mapped = set()
            allocation.is_released = True
            metrics = backend._restore_split_native(memory, native)
            self.assertEqual(metrics["restored_bytes"], 8192)
            self.assertEqual(restored[:500], bytes([1000 % 251]) * 500)
            self.assertEqual(restored[500:628], bytes([1500 % 251]) * 128)
            self.assertEqual(restored[628:1000], bytes([1628 % 251]) * 372)
            self.assertEqual(restored[1000:5097], bytes([2000 % 251]) * 4097)
            self.assertEqual(restored[5097:], bytes([6097 % 251]) * 3095)

    def test_native_manifest_relocation_uses_broader_native_semantics(self) -> None:
        captured = SimpleNamespace(pointer=1000, size=100, tag="weights", is_released=False)
        live = SimpleNamespace(pointer=5000, size=100, tag="weights", is_released=False)

        class Memory:
            @staticmethod
            def allocations(tag=None):
                return [live]

        manifest = {
            "format": disk_backend.FORMAT,
            "kind": disk_backend.KIND,
            "identity": {
                "hostname": "node-a",
                "worker_id": "worker-0",
                "rank": 0,
                "pid": os.getpid(),
            },
            "generation": "captured-generation",
            "pid": os.getpid(),
            "host": "node-a",
            "rank": 0,
            "worker_id": "worker-0",
            "capture_id": "capture",
            "weight_source": disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
            "blob": disk_backend.MODEL_PAYLOAD_NAME,
            "blob_bytes": 8192,
            "model_blob": disk_backend.MODEL_PAYLOAD_NAME,
            "model_blob_bytes": 8192,
            "residual_blob": disk_backend.NATIVE_RESIDUAL_NAME,
            "residual_blob_bytes": 50,
            "allocation_bytes": captured.size,
            "model_weight_bytes": 50,
            "residual_bytes": 50,
            "portable_model_payload": True,
            "portable_residual_blob": True,
            "entries": [{"ptr": captured.pointer, "size": captured.size, "tag": "weights"}],
            "model_weight_extents": [
                {
                    "ptr": 1010,
                    "size": 20,
                    "allocation_ptr": captured.pointer,
                    "semantic_id": "a" * 64,
                },
                {
                    "ptr": 1050,
                    "size": 30,
                    "allocation_ptr": captured.pointer,
                    "semantic_id": "b" * 64,
                },
            ],
            "residual_extents": [
                {"ptr": 1000, "size": 10, "allocation_ptr": captured.pointer},
                {"ptr": 1030, "size": 20, "allocation_ptr": captured.pointer},
                {"ptr": 1080, "size": 20, "allocation_ptr": captured.pointer},
            ],
        }
        environment = {
            "COLDSNAP_CAPTURE_ID": "capture",
            "COLDSNAP_EXPECTED_RANK": "0",
            "COLDSNAP_MODEL_ID": "model",
            "COLDSNAP_MODEL_REVISION": "pinned-revision",
            "COLDSNAP_EXECUTION_GRAPH": json.dumps(
                {
                    "unit": "unit-0",
                    "by_process_slot": {"0": "worker-0"},
                    "groups": {},
                }
            ),
            "VLLM_HOST_IP": "node-a",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
        ):
            manifest["identity"] = disk_backend._identity()
            manifest["pid"] = os.getpid()
            manifest["rank"] = disk_backend._rank()
            manifest["worker_id"] = disk_backend._worker_id()
            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = Path(directory)
            backend.manifest_path = backend.snapshot_dir / disk_backend.NATIVE_MANIFEST_NAME
            backend.blob_path = backend.snapshot_dir / disk_backend.MODEL_PAYLOAD_NAME
            backend.residual_blob_path = backend.snapshot_dir / disk_backend.NATIVE_RESIDUAL_NAME
            backend.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            backend.blob_path.write_bytes(bytes(8192))
            backend.residual_blob_path.write_bytes(bytes(50))
            (backend.snapshot_dir / disk_backend.ACTIVATION_PROVIDER_NAME).write_text(
                "native\n", encoding="utf-8"
            )
            backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_BLOB
            backend._model_weight_ranges = ((5010, 20),)
            backend._model_weight_semantics = ((5010, 20, "a" * 64),)
            backend._native_model_payload_semantics = (
                (5010, 20, "a" * 64),
                (5050, 30, "b" * 64),
            )
            backend._live_native_manifest = None
            backend._live_native_residual_path = None

            with patch.object(
                disk_backend,
                "_relocate_split_native_manifest",
                side_effect=RuntimeError("relocation observed"),
            ) as relocate:
                with self.assertRaisesRegex(RuntimeError, "relocation observed"):
                    backend._load_manifest(Memory())

        self.assertEqual(
            relocate.call_args.args[2],
            ((5010, 20), (5050, 30)),
        )
        self.assertEqual(
            relocate.call_args.args[3],
            ((5010, 20, "a" * 64), (5050, 30, "b" * 64)),
        )

    def test_live_native_residual_uses_secure_unique_file_and_replaces_cache(
        self,
    ) -> None:
        class Memory:
            @staticmethod
            def copy_to_host(stage, pointer, size):
                stage.view[:size] = bytes([pointer % 251]) * size

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous-residual.blob"
            previous.write_bytes(b"old")
            owner = bytearray(16)
            stage = HostStage(owner=owner, view=memoryview(owner), pointer=0)
            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.chunk_bytes = 16
            backend.verify_mode = "inline"
            backend._live_native_residual_path = previous
            backend._memory = lambda _provider: Memory()
            backend._ensure_space = lambda _required: None
            backend._stage = lambda: stage

            real_mkstemp = tempfile.mkstemp

            def secure_create(**kwargs):
                return real_mkstemp(dir=root, **kwargs)

            with patch.object(
                disk_backend.tempfile,
                "mkstemp",
                side_effect=secure_create,
            ) as mkstemp:
                rebound = backend._snapshot_live_native_residual(
                    object(),
                    {"residual_extents": [{"ptr": 7, "size": 8}]},
                )

            residual = backend._live_native_residual_path
            self.assertIsInstance(residual, Path)
            assert isinstance(residual, Path)
            self.assertEqual(residual.parent, root)
            self.assertTrue(residual.name.startswith("coldsnap-live-residual-"))
            self.assertNotEqual(residual.name, f"coldsnap-live-residual-{os.getpid()}.blob")
            self.assertEqual(residual.read_bytes(), bytes([7]) * 8)
            self.assertEqual(residual.stat().st_mode & 0o777, 0o400)
            self.assertFalse(previous.exists())
            self.assertEqual(rebound["residual_blob"], residual.name)
            mkstemp.assert_called_once_with(
                prefix=f"coldsnap-live-residual-{os.getpid()}-",
                suffix=".blob",
            )
            residual.unlink()

    def test_model_payload_identity_excludes_addresses_and_runtime_residuals(self) -> None:
        def capture(root: Path, base: int, residual_byte: int) -> tuple[str, str, dict[str, Any]]:
            allocation = SimpleNamespace(pointer=base, size=8192, tag="weights")
            model_start = base + 1000
            model_size = 4097

            class Memory:
                @staticmethod
                def allocations(tag=None):
                    return [allocation]

                @staticmethod
                def allocate_host_stage(size):
                    owner = bytearray(size)
                    return HostStage(owner=owner, view=memoryview(owner), pointer=0)

                @staticmethod
                def copy_to_host(stage, pointer, size):
                    if model_start <= pointer and pointer + size <= model_start + model_size:
                        offset = pointer - model_start
                        stage.view[:size] = bytes(
                            (index % 251 for index in range(offset, offset + size))
                        )
                    else:
                        stage.view[:size] = bytes([residual_byte]) * size

            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = root
            backend.blob_path = root / "weights.blob"
            backend.manifest_path = root / "manifest.json"
            backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_BLOB
            backend._model_weight_ranges = ((model_start, model_size),)
            backend.chunk_bytes = 8192
            backend.verify_mode = "preverified"
            backend.read_direct = False
            backend.write_direct = False
            backend._stage_cache = []
            backend.memory_provider = Memory()
            backend._write_snapshot(backend.memory_provider)
            payload = (root / disk_backend.MODEL_PAYLOAD_NAME).read_bytes()
            residual = backend.blob_path.read_bytes()
            native = json.loads((root / disk_backend.NATIVE_MANIFEST_NAME).read_text())
            return hashlib.sha256(payload).hexdigest(), hashlib.sha256(residual).hexdigest(), native

        environment = {
            "COLDSNAP_CAPTURE_ID": "capture",
            "COLDSNAP_EXPECTED_RANK": "0",
            "COLDSNAP_EXECUTION_GRAPH": _single_worker_execution_graph(),
            "COLDSNAP_EXPORT_MODEL_PAYLOAD": "1",
            "COLDSNAP_MODEL_ID": "model",
            "COLDSNAP_MODEL_REVISION": "pinned-revision",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
        ):
            first_root = Path(directory) / "n580"
            second_root = Path(directory) / "n610"
            first_root.mkdir()
            second_root.mkdir()
            first_payload, first_residual, first_manifest = capture(first_root, 1000, 0x35)
            second_payload, second_residual, second_manifest = capture(second_root, 100_000, 0x61)

        self.assertEqual(first_payload, second_payload)
        self.assertNotEqual(first_residual, second_residual)
        self.assertNotEqual(first_manifest["entries"], second_manifest["entries"])
        self.assertNotEqual(first_manifest["residual_extents"], second_manifest["residual_extents"])

    def test_model_payload_identity_canonicalizes_extent_order(self) -> None:
        def capture(
            root: Path,
            base: int,
            values_by_pointer: tuple[int, int],
        ) -> tuple[str, dict[str, Any]]:
            allocation = SimpleNamespace(pointer=base, size=24_576, tag="weights")
            model_ranges = ((base + 1000, 4097), (base + 12_000, 4097))

            class Memory:
                @staticmethod
                def allocations(tag=None):
                    return [allocation]

                @staticmethod
                def allocate_host_stage(size):
                    owner = bytearray(size)
                    return HostStage(owner=owner, view=memoryview(owner), pointer=0)

                @staticmethod
                def copy_to_host(stage, pointer, size):
                    for index, (model_pointer, model_size) in enumerate(model_ranges):
                        if (
                            model_pointer <= pointer
                            and pointer + size <= model_pointer + model_size
                        ):
                            stage.view[:size] = bytes([values_by_pointer[index]]) * size
                            return
                    stage.view[:size] = b"\x7f" * size

            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = root
            backend.blob_path = root / "weights.blob"
            backend.manifest_path = root / "manifest.json"
            backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_BLOB
            backend._model_weight_ranges = model_ranges
            backend.chunk_bytes = 8192
            backend.verify_mode = "preverified"
            backend.read_direct = False
            backend.write_direct = False
            backend._stage_cache = []
            backend.memory_provider = Memory()
            backend._write_snapshot(backend.memory_provider)
            payload = (root / disk_backend.MODEL_PAYLOAD_NAME).read_bytes()
            native = json.loads((root / disk_backend.NATIVE_MANIFEST_NAME).read_text())
            return hashlib.sha256(payload).hexdigest(), native

        environment = {
            "COLDSNAP_CAPTURE_ID": "capture",
            "COLDSNAP_EXPECTED_RANK": "0",
            "COLDSNAP_EXECUTION_GRAPH": _single_worker_execution_graph(),
            "COLDSNAP_EXPORT_MODEL_PAYLOAD": "1",
            "COLDSNAP_MODEL_ID": "model",
            "COLDSNAP_MODEL_REVISION": "pinned-revision",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
        ):
            first_root = Path(directory) / "n580"
            second_root = Path(directory) / "n610"
            first_root.mkdir()
            second_root.mkdir()
            first_payload, first_manifest = capture(first_root, 1000, (0x35, 0x61))
            second_payload, second_manifest = capture(second_root, 100_000, (0x61, 0x35))

        self.assertEqual(first_payload, second_payload)
        first_extents = first_manifest["model_weight_extents"]
        second_extents = second_manifest["model_weight_extents"]
        self.assertEqual(
            [extent["sha256"] for extent in first_extents],
            [extent["sha256"] for extent in second_extents],
        )
        self.assertNotEqual(
            [extent["ptr"] for extent in first_extents],
            [extent["ptr"] for extent in second_extents],
        )

    def test_recovery_restore_materializes_and_reuses_local_native_payload(self) -> None:
        allocation = SimpleNamespace(pointer=1000, size=8192, tag="weights")

        class Memory:
            @staticmethod
            def allocations(tag=None):
                return [allocation]

            @staticmethod
            def allocate_host_stage(size):
                owner = bytearray(size)
                return HostStage(owner=owner, view=memoryview(owner), pointer=0)

            @staticmethod
            def copy_to_host(stage, pointer, size):
                stage.view[:size] = bytes([pointer % 251]) * size

        environment = {
            "COLDSNAP_CAPTURE_ID": "capture",
            "COLDSNAP_EXPECTED_RANK": "0",
            "COLDSNAP_EXECUTION_GRAPH": _single_worker_execution_graph(),
            "COLDSNAP_EXPORT_MODEL_PAYLOAD": "1",
            "COLDSNAP_MODEL_ID": "model",
            "COLDSNAP_MODEL_REVISION": "pinned-revision",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
        ):
            root = Path(directory)
            backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
            backend.snapshot_dir = root / "capture"
            backend.snapshot_dir.mkdir()
            backend.blob_path = backend.snapshot_dir / "weights.blob"
            backend.manifest_path = backend.snapshot_dir / "manifest.json"
            backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_BLOB
            backend._model_weight_ranges = ((2000, 4097),)
            backend.chunk_bytes = 8192
            backend.verify_mode = "preverified"
            backend.read_direct = False
            backend.write_direct = False
            backend._stage_cache = []
            backend.memory_provider = Memory()
            recovery = backend._write_snapshot(backend.memory_provider)
            expected = recovery["model_payload"]
            expected_pack = backend.snapshot_dir / disk_backend.MODEL_PAYLOAD_NAME
            self.assertEqual(
                "sha256:" + hashlib.sha256(expected_pack.read_bytes()).hexdigest(),
                expected["sha256"],
            )

            digest = expected["sha256"].removeprefix("sha256:")
            cache_root = root / "cache"
            target = cache_root / "sha256" / f"{digest}.pack"
            config = {
                "mode": "required",
                "operation_id": "materialize",
                "owner_uid": os.getuid(),
                "owner_gid": os.getgid(),
                "sha256": expected["sha256"],
                "bytes": expected["bytes"],
                "target": target,
                "status": cache_root / ".status" / "worker-0.json",
            }
            result = backend._materialize_model_payload(backend.memory_provider, recovery, config)
            self.assertFalse(result["reused"])
            self.assertEqual(target.read_bytes(), expected_pack.read_bytes())
            self.assertTrue(backend._cached_model_payload_valid(config))
            reused = backend._materialize_model_payload(backend.memory_provider, recovery, config)
            self.assertTrue(reused["reused"])

            control = root / "materialize.json"
            async_root = root / "async-cache"
            disk_backend._atomic_json(
                control,
                {
                    "format": disk_backend.MODEL_PAYLOAD_MATERIALIZATION_FORMAT,
                    "kind": disk_backend.MODEL_PAYLOAD_MATERIALIZATION_KIND,
                    "operation_id": "async-materialize",
                    "mode": "async",
                    "owner_uid": os.getuid(),
                    "owner_gid": os.getgid(),
                    "workers": {
                        "worker-0": {
                            "path": f"sha256/{digest}.pack",
                            "bytes": expected["bytes"],
                            "sha256": expected["sha256"],
                        }
                    },
                },
            )
            os.environ["COLDSNAP_MODEL_PAYLOAD_MATERIALIZATION_CONTROL"] = str(control)
            os.environ["COLDSNAP_MODEL_PAYLOAD_CACHE_ROOT"] = str(async_root)
            async_config = backend._model_payload_materialization()
            self.assertIsNotNone(async_config)
            scheduled = backend._start_model_payload_materialization(
                backend.memory_provider, recovery, async_config
            )
            self.assertEqual(scheduled["state"], "scheduled")
            backend._materialization_thread.join(timeout=5)
            self.assertFalse(backend._materialization_thread.is_alive())
            self.assertTrue(backend._cached_model_payload_valid(async_config))

    def test_process_template_initial_load_triggers_model_payload_materialization(self) -> None:
        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        backend.weight_recovery_source = disk_backend.WEIGHT_SOURCE_SAFETENSORS
        backend._initial_model_payload_materialization_result = None
        memory = object()
        config = {"mode": "required"}
        manifest = {"weight_source": disk_backend.WEIGHT_SOURCE_SAFETENSORS}
        events = []
        backend._model_payload_materialization = lambda: config
        backend._memory = lambda: memory
        backend._load_manifest = lambda value, **kwargs: (
            events.append(("manifest", value, kwargs)) or manifest
        )
        backend._start_model_payload_materialization = lambda *values: (
            events.append(("materialize", values))
            or {"state": "ready", "path": "/cache/model.pack"}
        )

        with patch.dict(
            os.environ,
            {"COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1"},
            clear=True,
        ):
            first = backend.materialize_initial_recovery_payload()
            second = backend.materialize_initial_recovery_payload()

        self.assertEqual(first, {"state": "ready", "path": "/cache/model.pack"})
        self.assertIs(second, first)
        self.assertEqual(
            events,
            [
                (
                    "manifest",
                    memory,
                    {
                        "allow_process_template_owner": True,
                        "allow_materialization_layout": True,
                    },
                ),
                ("materialize", (memory, manifest, config)),
            ],
        )

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(backend.materialize_initial_recovery_payload())

    def test_recovery_parameter_ranges_merge_aliases_and_adjacency(self) -> None:
        class Parameter:
            is_cuda = True

            def __init__(self, pointer, elements):
                self.pointer = pointer
                self.elements = elements

            def numel(self):
                return self.elements

            @staticmethod
            def element_size():
                return 2

            @staticmethod
            def is_contiguous():
                return True

            def data_ptr(self):
                return self.pointer

        model = SimpleNamespace(
            named_parameters=lambda: iter(
                [
                    ("left", Parameter(1200, 100)),
                    ("alias", Parameter(1200, 100)),
                    ("right", Parameter(1400, 50)),
                ]
            ),
            **{
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR: frozenset(
                    {"left", "alias", "right"}
                ),
            },
        )
        backend = SimpleNamespace(
            memory_provider=SimpleNamespace(
                allocations=lambda tag: [SimpleNamespace(pointer=1000, size=1000, tag=tag)]
            )
        )

        self.assertEqual(
            recovery_loader._model_weight_ranges(model, backend),
            [(1200, 300)],
        )

    def test_recovery_weight_tensors_keep_unreported_parameters_residual(self) -> None:
        checkpoint = object()
        derived = object()
        checkpoint_buffer = object()
        model = SimpleNamespace(
            named_parameters=lambda: iter([("checkpoint", checkpoint), ("runtime_cache", derived)]),
            named_buffers=lambda: iter([("checkpoint_buffer", checkpoint_buffer)]),
            **{
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR: frozenset(
                    {"checkpoint", "checkpoint_buffer"}
                ),
            },
        )

        self.assertEqual(
            recovery_loader._model_weight_tensors(model),
            [
                ("checkpoint", checkpoint),
                ("checkpoint_buffer", checkpoint_buffer),
            ],
        )
        self.assertEqual(
            recovery_loader._model_semantic_tensors(model),
            [("checkpoint", checkpoint), ("runtime_cache", derived)],
        )

        delattr(
            model,
            recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR,
        )
        self.assertEqual(recovery_loader._model_weight_tensors(model), [])

    def test_native_payload_layout_includes_registered_derived_model_state(self) -> None:
        class Tensor:
            is_cuda = True

            def __init__(self, pointer, elements):
                self.pointer = pointer
                self.elements = elements

            def numel(self):
                return self.elements

            @staticmethod
            def element_size():
                return 2

            @staticmethod
            def is_contiguous():
                return True

            def data_ptr(self):
                return self.pointer

        model = SimpleNamespace(
            named_parameters=lambda: iter(
                [("checkpoint", Tensor(1200, 100)), ("runtime", Tensor(1500, 50))]
            ),
            named_buffers=lambda: iter([("derived_scale", Tensor(1700, 50))]),
            named_modules=lambda: iter(()),
            **{
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR: frozenset({"checkpoint"}),
            },
        )
        backend = SimpleNamespace(
            memory_provider=SimpleNamespace(
                allocations=lambda tag: [SimpleNamespace(pointer=1000, size=1000, tag=tag)]
            )
        )

        self.assertEqual(
            [item[:2] for item in recovery_loader._model_weight_layout(model, backend)],
            [(1200, 200)],
        )
        self.assertEqual(
            [item[:2] for item in recovery_loader._native_model_payload_layout(model, backend)],
            [(1200, 200), (1500, 100), (1700, 100)],
        )

    def test_recovery_weight_ranges_include_b12x_checkpoint_storage_only(self) -> None:
        def tensor(pointer):
            return SimpleNamespace(
                is_cuda=True,
                numel=lambda: 50,
                element_size=lambda: 2,
                is_contiguous=lambda: True,
                data_ptr=lambda: pointer,
            )

        prepared = SimpleNamespace(
            **{
                field_name: tensor(1200 + index * 100)
                for index, field_name in enumerate(recovery_loader.B12X_PREPARED_WEIGHT_FIELDS)
            }
        )
        fused_experts = SimpleNamespace(_lookup_prepared_experts=lambda: prepared)
        module = SimpleNamespace(
            quant_method=SimpleNamespace(moe_kernel=SimpleNamespace(fused_experts=fused_experts))
        )
        model = SimpleNamespace(
            named_parameters=lambda: iter([("dense", tensor(1100))]),
            named_modules=lambda: iter([("model.layers.0.mlp", module)]),
            **{
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR: frozenset({"dense"}),
            },
        )
        backend = SimpleNamespace(
            memory_provider=SimpleNamespace(
                allocations=lambda tag: [SimpleNamespace(pointer=1000, size=2000, tag=tag)]
            )
        )

        self.assertEqual(
            recovery_loader._model_weight_ranges(model, backend),
            [(1100, 100), (1300, 200), (1700, 200)],
        )

    def test_recovery_b12x_adapter_materializes_source_in_stable_storage(self) -> None:
        class Tensor:
            is_cuda = True

            def __init__(self, pointer):
                self.pointer = pointer

            @staticmethod
            def numel():
                return 50

            @staticmethod
            def element_size():
                return 2

            @staticmethod
            def is_contiguous():
                return True

            def data_ptr(self):
                return self.pointer

        stable = {
            field_name: Tensor(1200 + index * 100)
            for index, field_name in enumerate(recovery_loader.B12X_PREPARED_WEIGHT_FIELDS)
        }
        prepared = SimpleNamespace(
            plan=SimpleNamespace(discards_source_parameters=True),
            **stable,
        )

        class Owner:
            _prepared_experts = prepared
            _source_parameters_released = True

            def _lookup_prepared_experts(self):
                return self._prepared_experts

        owner = Owner()
        captured_sources = {
            source_name: object()
            for source_name, _prepared_name in recovery_loader.B12X_SOURCE_STORAGE_FIELDS
        }
        layer = SimpleNamespace(
            quant_method=SimpleNamespace(moe_kernel=SimpleNamespace(fused_experts=owner)),
            **captured_sources,
        )
        model = SimpleNamespace(named_modules=lambda: iter([("model.layers.0.mlp", layer)]))
        restore_parameters = {
            source_name: SimpleNamespace(is_meta=True)
            for source_name, _prepared_name in (recovery_loader.B12X_SOURCE_STORAGE_FIELDS)
        }
        info = SimpleNamespace(restore_metadata=(restore_parameters, {}))
        materialized = []

        layerwise = ModuleType("vllm.model_executor.model_loader.reload.layerwise")

        def materialize_layer(value, _info):
            materialized.append(
                {
                    source_name: getattr(value, source_name)
                    for source_name, _prepared_name in (recovery_loader.B12X_SOURCE_STORAGE_FIELDS)
                }
            )

        layerwise.materialize_layer = materialize_layer
        layerwise.restore_layer_on_meta = lambda _layer, _info: None
        layerwise.get_layer_size = lambda _layer: 0
        layerwise._wrap_parameters_weight_loader = lambda _layer: None
        meta = ModuleType("vllm.model_executor.model_loader.reload.meta")
        meta.SKIP_LOAD_TENSORS = set()
        meta.SKIP_TENSORS = {"bias"}
        modules = {
            "torch": ModuleType("torch"),
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.model_loader": ModuleType("vllm.model_executor.model_loader"),
            "vllm.model_executor.model_loader.reload": ModuleType(
                "vllm.model_executor.model_loader.reload"
            ),
            "vllm.model_executor.model_loader.reload.layerwise": layerwise,
            "vllm.model_executor.model_loader.reload.meta": meta,
        }

        def alias(_torch, _meta, tensor, *, label):
            return (label, tensor)

        with (
            patch.dict(sys.modules, modules),
            patch.object(
                recovery_loader,
                "_alias_meta_tensor_from_storage",
                side_effect=alias,
            ),
            recovery_loader._recovery_reload_storage(model) as adapters,
        ):
            self.assertEqual(adapters, ("b12x:model.layers.0.mlp",))
            self.assertTrue(recovery_loader._RECOVERY_CONSUMER_STORAGE_ISOLATION.get())
            self.assertIsNone(owner._prepared_experts)
            self.assertFalse(owner._source_parameters_released)
            layerwise.materialize_layer(layer, info)
            # vLLM's MXFP4 setup rebuilds the modular MoE kernel during
            # process_weights_after_loading.  The replacement owner must be
            # accepted as long as it prepared the captured stable storage.
            rebuilt_owner = Owner()
            rebuilt_owner._prepared_experts = prepared
            rebuilt_owner._source_parameters_released = True
            layer.quant_method.moe_kernel.fused_experts = rebuilt_owner

        self.assertEqual(len(materialized), 1)
        for source_name, prepared_name in recovery_loader.B12X_SOURCE_STORAGE_FIELDS:
            self.assertIs(materialized[0][source_name][1], stable[prepared_name])
        self.assertIs(layerwise.materialize_layer, materialize_layer)
        self.assertIs(rebuilt_owner._prepared_experts, prepared)
        self.assertTrue(rebuilt_owner._source_parameters_released)
        self.assertFalse(recovery_loader._RECOVERY_CONSUMER_STORAGE_ISOLATION.get())

    def test_recovery_context_keeps_unobserved_parameters_out_of_meta_reload(
        self,
    ) -> None:
        checkpoint = object()
        runtime = object()
        cache = object()
        layer = SimpleNamespace(
            _parameters={"weight": checkpoint, "runtime": runtime},
            _buffers={"cache": cache},
            quant_method=None,
        )
        model = SimpleNamespace(
            named_modules=lambda: iter([("layer", layer)]),
            **{
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR: frozenset(
                    {"layer.weight"}
                )
            },
        )
        observed = []
        meta = ModuleType("vllm.model_executor.model_loader.reload.meta")
        meta.SKIP_LOAD_TENSORS = set()
        meta.SKIP_TENSORS = {"bias"}
        layerwise = ModuleType("vllm.model_executor.model_loader.reload.layerwise")

        def observe(operation):
            def call(_layer, *_args):
                observed.append(
                    (
                        operation,
                        frozenset(meta.SKIP_LOAD_TENSORS),
                        frozenset(meta.SKIP_TENSORS),
                    )
                )
                return 1 if operation == "size" else None

            return call

        original_materialize = observe("materialize")
        original_restore = observe("restore")
        original_size = observe("size")
        original_wrap = observe("wrap")
        layerwise.materialize_layer = original_materialize
        layerwise.restore_layer_on_meta = original_restore
        layerwise.get_layer_size = original_size
        layerwise._wrap_parameters_weight_loader = original_wrap
        modules = {
            "torch": ModuleType("torch"),
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.model_loader": ModuleType("vllm.model_executor.model_loader"),
            "vllm.model_executor.model_loader.reload": ModuleType(
                "vllm.model_executor.model_loader.reload"
            ),
            "vllm.model_executor.model_loader.reload.layerwise": layerwise,
            "vllm.model_executor.model_loader.reload.meta": meta,
        }

        with (
            patch.dict(sys.modules, modules),
            recovery_loader._recovery_reload_storage(model) as adapters,
        ):
            self.assertEqual(adapters, ())
            layerwise.restore_layer_on_meta(layer, object())
            self.assertEqual(layerwise.get_layer_size(layer), 1)
            layerwise._wrap_parameters_weight_loader(layer)
            layerwise.materialize_layer(layer, object())

        self.assertEqual(
            [item[0] for item in observed],
            [
                "restore",
                "size",
                "wrap",
                "materialize",
            ],
        )
        for _operation, skip_load, skip_tensors in observed:
            self.assertIn("runtime", skip_load)
            self.assertIn("runtime", skip_tensors)
            self.assertIn("cache", skip_load)
            self.assertIn("cache", skip_tensors)
            self.assertNotIn("weight", skip_load)
            self.assertNotIn("weight", skip_tensors)
        self.assertEqual(meta.SKIP_LOAD_TENSORS, set())
        self.assertEqual(meta.SKIP_TENSORS, {"bias"})
        self.assertIs(layerwise.materialize_layer, original_materialize)
        self.assertIs(layerwise.restore_layer_on_meta, original_restore)
        self.assertIs(layerwise.get_layer_size, original_size)
        self.assertIs(
            layerwise._wrap_parameters_weight_loader,
            original_wrap,
        )

    def test_recovery_context_excludes_preloaded_weights_from_completion_only(
        self,
    ) -> None:
        weight = object()
        layer = SimpleNamespace(
            weight=weight,
            _parameters={"weight": weight, "scale": object(), "runtime": object()},
            _buffers={},
            quant_method=None,
        )
        model = SimpleNamespace(
            named_modules=lambda: iter([("layer", layer)]),
            **{
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR: frozenset(
                    {"layer.weight", "layer.scale"}
                )
            },
        )
        observed = []
        meta = ModuleType("vllm.model_executor.model_loader.reload.meta")
        meta.SKIP_LOAD_TENSORS = set()
        meta.SKIP_TENSORS = {"bias"}
        layerwise = ModuleType("vllm.model_executor.model_loader.reload.layerwise")

        def observe(operation):
            def call(_layer, *_args):
                observed.append(
                    (
                        operation,
                        frozenset(meta.SKIP_LOAD_TENSORS),
                        frozenset(meta.SKIP_TENSORS),
                    )
                )
                return 1 if operation == "size" else None

            return call

        layerwise.materialize_layer = observe("materialize")
        layerwise.restore_layer_on_meta = observe("restore")
        layerwise.get_layer_size = observe("size")
        layerwise._wrap_parameters_weight_loader = observe("wrap")
        modules = {
            "torch": ModuleType("torch"),
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.model_loader": ModuleType("vllm.model_executor.model_loader"),
            "vllm.model_executor.model_loader.reload": ModuleType(
                "vllm.model_executor.model_loader.reload"
            ),
            "vllm.model_executor.model_loader.reload.layerwise": layerwise,
            "vllm.model_executor.model_loader.reload.meta": meta,
        }

        token = recovery_loader._RECOVERY_PRELOADED_DESTINATION_NAMES.set(
            frozenset({"layer.weight"})
        )
        try:
            with (
                patch.dict(sys.modules, modules),
                recovery_loader._recovery_reload_storage(model),
            ):
                layerwise.restore_layer_on_meta(layer, object())
                layerwise.get_layer_size(layer)
                layerwise._wrap_parameters_weight_loader(layer)
                layerwise.materialize_layer(layer, object())
        finally:
            recovery_loader._RECOVERY_PRELOADED_DESTINATION_NAMES.reset(token)

        by_operation = {
            operation: (skip_load, skip_tensors) for operation, skip_load, skip_tensors in observed
        }
        for operation in ("restore", "materialize"):
            skip_load, skip_tensors = by_operation[operation]
            self.assertNotIn("weight", skip_load)
            self.assertNotIn("weight", skip_tensors)
            self.assertIn("runtime", skip_load)
            self.assertIn("runtime", skip_tensors)
        for operation in ("size", "wrap"):
            skip_load, skip_tensors = by_operation[operation]
            self.assertIn("weight", skip_load)
            self.assertNotIn("weight", skip_tensors)
            self.assertIn("runtime", skip_load)
            self.assertIn("runtime", skip_tensors)
        self.assertEqual(meta.SKIP_LOAD_TENSORS, set())
        self.assertEqual(meta.SKIP_TENSORS, {"bias"})

    def test_recovery_context_aliases_preloaded_weight_to_stable_storage(self) -> None:
        stable = SimpleNamespace(is_meta=False)
        restored_meta = SimpleNamespace(is_meta=True)
        layer = SimpleNamespace(
            weight=stable,
            _parameters={"weight": stable},
            _buffers={},
            quant_method=None,
        )
        model = SimpleNamespace(
            named_modules=lambda: iter([("layer", layer)]),
            **{
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR: frozenset(
                    {"layer.weight"}
                )
            },
        )
        materialized = []
        layerwise = ModuleType("vllm.model_executor.model_loader.reload.layerwise")

        def restore_layer_on_meta(value, _info):
            value.weight = restored_meta
            value._parameters["weight"] = restored_meta

        def materialize_layer(value, _info):
            materialized.append(value.weight)

        layerwise.materialize_layer = materialize_layer
        layerwise.restore_layer_on_meta = restore_layer_on_meta
        layerwise.get_layer_size = lambda _layer: 0
        layerwise._wrap_parameters_weight_loader = lambda _layer: None
        meta = ModuleType("vllm.model_executor.model_loader.reload.meta")
        meta.SKIP_LOAD_TENSORS = set()
        meta.SKIP_TENSORS = set()
        modules = {
            "torch": ModuleType("torch"),
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.model_loader": ModuleType("vllm.model_executor.model_loader"),
            "vllm.model_executor.model_loader.reload": ModuleType(
                "vllm.model_executor.model_loader.reload"
            ),
            "vllm.model_executor.model_loader.reload.layerwise": layerwise,
            "vllm.model_executor.model_loader.reload.meta": meta,
        }

        token = recovery_loader._RECOVERY_PRELOADED_DESTINATION_NAMES.set(
            frozenset({"layer.weight"})
        )
        try:
            with (
                patch.dict(sys.modules, modules),
                patch.object(
                    recovery_loader,
                    "_alias_meta_tensor_from_storage",
                    return_value=("stable-alias", stable),
                ) as alias,
                recovery_loader._recovery_reload_storage(model, prepare_preloaded=True),
            ):
                layerwise.restore_layer_on_meta(layer, object())
                layerwise.materialize_layer(layer, object())
        finally:
            recovery_loader._RECOVERY_PRELOADED_DESTINATION_NAMES.reset(token)

        alias.assert_called_once_with(
            modules["torch"],
            restored_meta,
            stable,
            label="preloaded:weight",
        )
        self.assertEqual(materialized, [("stable-alias", stable)])
        self.assertIs(layerwise.materialize_layer, materialize_layer)

    def test_recovery_b12x_adapter_accepts_relocated_derived_runtime_state(
        self,
    ) -> None:
        class Tensor:
            is_cuda = True

            def __init__(self, pointer, elements):
                self.pointer = pointer
                self.elements = elements

            def numel(self):
                return self.elements

            @staticmethod
            def element_size():
                return 4

            @staticmethod
            def is_contiguous():
                return True

            def data_ptr(self):
                return self.pointer

        fields = {
            field_name: Tensor(1200 + index * 100, 4)
            for index, field_name in enumerate(recovery_loader.B12X_DERIVED_RUNTIME_FIELDS)
        }
        fields.update(
            {
                field_name: Tensor(2000 + index * 100, 50)
                for index, field_name in enumerate(recovery_loader.B12X_RELOAD_WEIGHT_FIELDS)
            }
        )
        plan = SimpleNamespace(
            discards_source_parameters=True,
            source_format="fp4_e8m0_k32",
        )
        prepared = SimpleNamespace(plan=plan, **fields)

        class Owner:
            def __init__(self, value=None, released=False):
                self._prepared_experts = value
                self._source_parameters_released = released

            def _lookup_prepared_experts(self):
                return self._prepared_experts

        captured_owner = Owner(prepared, True)
        quant_method = SimpleNamespace(moe_kernel=SimpleNamespace(fused_experts=captured_owner))
        captured_sources = {
            source_name: object()
            for source_name, _prepared_name in recovery_loader.B12X_SOURCE_STORAGE_FIELDS
        }
        layer = SimpleNamespace(quant_method=quant_method, **captured_sources)
        adapter = recovery_loader._B12xRecoveryStorageAdapter(
            module_name="model.layers.0.mlp",
            layer=layer,
            owner=captured_owner,
            prepared=prepared,
            source_parameters_released=True,
        )

        adapter.begin_reload()
        for source_name, _prepared_name in recovery_loader.B12X_SOURCE_STORAGE_FIELDS:
            setattr(layer, source_name, object())
        adapter.end_direct_replay()
        for source_name, parameter in captured_sources.items():
            self.assertIs(getattr(layer, source_name), parameter)
        adapter.materialized = True
        replacement_fields = dict(fields)
        replacement_fields.update(
            {
                field_name: Tensor(9000 + index * 100, 4)
                for index, field_name in enumerate(recovery_loader.B12X_DERIVED_RUNTIME_FIELDS)
            }
        )
        rebuilt_owner = Owner(
            SimpleNamespace(plan=plan, **replacement_fields),
            True,
        )
        quant_method.moe_kernel.fused_experts = rebuilt_owner
        adapter.finish_reload()

        for field_name in recovery_loader.B12X_RELOAD_WEIGHT_FIELDS:
            self.assertIs(
                getattr(rebuilt_owner._prepared_experts, field_name),
                fields[field_name],
            )
        for field_name in recovery_loader.B12X_DERIVED_RUNTIME_FIELDS:
            self.assertIsNot(
                getattr(rebuilt_owner._prepared_experts, field_name),
                fields[field_name],
            )

    def test_recovery_b12x_complete_replay_retains_prepared_owner(self) -> None:
        class Tensor:
            def __init__(self, pointer):
                self.pointer = pointer

            @staticmethod
            def numel():
                return 4

            @staticmethod
            def element_size():
                return 1

            def data_ptr(self):
                return self.pointer

        fields = {
            field_name: Tensor(2000 + index * 100)
            for index, field_name in enumerate(recovery_loader.B12X_PREPARED_WEIGHT_FIELDS)
        }
        prepared = SimpleNamespace(
            plan=SimpleNamespace(discards_source_parameters=True),
            **fields,
        )

        class Owner:
            def __init__(self):
                self._prepared_experts = prepared
                self._source_parameters_released = True

            def _lookup_prepared_experts(self):
                return self._prepared_experts

        owner = Owner()
        captured_sources = {
            source_name: object()
            for source_name, _prepared_name in recovery_loader.B12X_SOURCE_STORAGE_FIELDS
        }
        layer = SimpleNamespace(
            quant_method=SimpleNamespace(moe_kernel=SimpleNamespace(fused_experts=owner)),
            **captured_sources,
        )
        adapter = recovery_loader._B12xRecoveryStorageAdapter(
            module_name="model.layers.0.mlp",
            layer=layer,
            owner=owner,
            prepared=prepared,
            source_parameters_released=True,
        )

        adapter.begin_reload()
        adapter.configure_direct_replay(adapter.direct_replay_destination_names)
        adapter.materialized = True
        adapter.end_direct_replay()

        self.assertIs(owner._prepared_experts, prepared)
        self.assertTrue(owner._source_parameters_released)
        for source_name, parameter in captured_sources.items():
            self.assertIs(getattr(layer, source_name), parameter)
        adapter.finish_reload()
        self.assertFalse(adapter.direct_replay_complete)

    def test_recovery_b12x_complete_replay_runs_existing_vllm_finalizer(self) -> None:
        class Tensor:
            def __init__(self, pointer):
                self.pointer = pointer

            @staticmethod
            def numel():
                return 4

            @staticmethod
            def element_size():
                return 1

            def data_ptr(self):
                return self.pointer

        fields = {
            field_name: Tensor(2000 + index * 100)
            for index, field_name in enumerate(recovery_loader.B12X_PREPARED_WEIGHT_FIELDS)
        }
        captured_prepared = SimpleNamespace(
            plan=SimpleNamespace(discards_source_parameters=True),
            **fields,
        )

        class Owner:
            def __init__(self, prepared, released):
                self._prepared_experts = prepared
                self._source_parameters_released = released

            def _lookup_prepared_experts(self):
                return self._prepared_experts

        captured_owner = Owner(captured_prepared, True)

        class QuantMethod:
            def __init__(self):
                self.moe_kernel = SimpleNamespace(fused_experts=captured_owner)
                self.calls = 0

            def process_weights_after_loading(self, layer):
                self.calls += 1
                rebuilt_fields = dict(fields)
                for field_name in recovery_loader.B12X_DERIVED_RUNTIME_FIELDS:
                    rebuilt_fields[field_name] = Tensor(9000 + self.calls)
                rebuilt = SimpleNamespace(
                    plan=captured_prepared.plan,
                    **rebuilt_fields,
                )
                self.moe_kernel = SimpleNamespace(
                    fused_experts=Owner(rebuilt, True),
                )

        quant_method = QuantMethod()
        captured_sources = {
            source_name: object()
            for source_name, _prepared_name in recovery_loader.B12X_SOURCE_STORAGE_FIELDS
        }
        tp_updates = []
        layer = SimpleNamespace(
            quant_method=quant_method,
            update_param_tp_status=lambda: tp_updates.append(True),
            _already_called_process_weights_after_loading=True,
            **captured_sources,
        )
        adapter = recovery_loader._B12xRecoveryStorageAdapter(
            module_name="model.layers.0.mlp",
            layer=layer,
            owner=captured_owner,
            prepared=captured_prepared,
            source_parameters_released=True,
        )

        class NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        torch_module = SimpleNamespace(no_grad=NoGrad)
        adapter.begin_reload()
        adapter.configure_direct_replay(adapter.direct_replay_destination_names)
        adapter.materialized = True
        adapter.finalize_direct_replay(torch_module)

        self.assertEqual(quant_method.calls, 1)
        self.assertEqual(tp_updates, [True])
        self.assertFalse(hasattr(layer, "_already_called_process_weights_after_loading"))
        self.assertTrue(adapter.direct_replay_finalized)
        self.assertIsNotNone(adapter.direct_replay_prepared)

        adapter.end_direct_replay()
        for source_name, parameter in captured_sources.items():
            self.assertIs(getattr(layer, source_name), parameter)
        adapter.finish_reload()
        self.assertFalse(adapter.direct_replay_finalized)
        self.assertIsNone(adapter.direct_replay_prepared)

    def test_recovery_parameter_ranges_reject_outside_weight_pool(self) -> None:
        parameter = SimpleNamespace(
            is_cuda=True,
            numel=lambda: 4,
            element_size=lambda: 2,
            is_contiguous=lambda: True,
            data_ptr=lambda: 1996,
        )
        model = SimpleNamespace(
            named_parameters=lambda: iter([("outside", parameter)]),
            **{
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR: frozenset({"outside"}),
            },
        )
        backend = SimpleNamespace(
            memory_provider=SimpleNamespace(
                allocations=lambda tag: [SimpleNamespace(pointer=1000, size=1000, tag=tag)]
            )
        )

        with self.assertRaisesRegex(RuntimeError, "exactly one weight allocation"):
            recovery_loader._model_weight_ranges(model, backend)

    def test_recovery_parameter_ranges_do_not_merge_adjacent_allocations(self) -> None:
        def parameter(pointer):
            return SimpleNamespace(
                is_cuda=True,
                numel=lambda: 50,
                element_size=lambda: 2,
                is_contiguous=lambda: True,
                data_ptr=lambda: pointer,
            )

        model = SimpleNamespace(
            named_parameters=lambda: iter([("left", parameter(1900)), ("right", parameter(2000))]),
            **{
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR: frozenset(
                    {"left", "right"}
                ),
            },
        )
        backend = SimpleNamespace(
            memory_provider=SimpleNamespace(
                allocations=lambda tag: [
                    SimpleNamespace(pointer=1000, size=1000, tag=tag),
                    SimpleNamespace(pointer=2000, size=1000, tag=tag),
                ]
            )
        )

        self.assertEqual(
            recovery_loader._model_weight_ranges(model, backend),
            [(1900, 100), (2000, 100)],
        )

    def test_recovery_backend_rejects_overlapping_parameter_ranges(self) -> None:
        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)

        with self.assertRaisesRegex(ValueError, "positive and disjoint"):
            backend.set_model_weight_ranges([(1000, 100), (1099, 100)])

    def test_model_source_restore_remaps_before_reload_and_requires_callback(self) -> None:
        events = []
        allocation = SimpleNamespace(
            pointer=1000,
            size=8192,
            tag="weights",
            is_released=True,
        )

        class Memory:
            def allocations(self, tag=None):
                return [allocation] if tag in {None, "weights"} else []

            def remap(self, value):
                events.append(("remap", value.pointer))

            def fill_zero(self, pointer, size):
                events.append(("zero", pointer, size))

            def synchronize(self):
                events.append("sync")

        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        backend.memory_provider = Memory()
        backend._restore_mapped = set()
        backend._weight_recovery_callback = None
        manifest = {
            "entries": [{"ptr": 1000, "size": 8192, "tag": "weights"}],
            "model_weight_extents": [{"ptr": 2000, "size": 4096, "allocation_ptr": 1000}],
            "model_weight_bytes": 4096,
            "residual_bytes": 4096,
        }

        with self.assertRaisesRegex(RuntimeError, "reload callback"):
            backend._restore_from_model_source(backend.memory_provider, manifest)
        self.assertEqual(events, [])

        backend.set_weight_recovery_callback(lambda: events.append("reload"))
        with (
            patch.object(
                recovery_loader,
                "last_recovery_load_metrics",
                return_value={"total_seconds": 1.25},
            ),
            patch.object(
                backend,
                "_restore_recovery_residuals",
                side_effect=lambda memory, manifest: (
                    events.append("residual")
                    or {"bytes": 4096, "chunks": 1, "backend": "direct", "seconds": 0.1}
                ),
            ),
        ):
            metrics = backend._restore_from_model_source(backend.memory_provider, manifest)

        self.assertEqual(
            events,
            [
                ("remap", 1000),
                ("zero", 2000, 4096),
                "residual",
                "sync",
                "reload",
                "sync",
                "residual",
                "sync",
            ],
        )
        self.assertEqual(metrics["restored_bytes"], 8192)
        self.assertGreaterEqual(metrics["initialization_seconds"], 0.0)
        self.assertEqual(metrics["hydration_backend"], "recovery-safetensors+direct")
        self.assertEqual(metrics["recovery_loader"], {"total_seconds": 1.25})

    def test_preverified_blob_stat_binds_every_identity_field(self) -> None:
        identity = {
            "device": 1,
            "inode": 2,
            "bytes": 3,
            "mtime_ns": 4,
            "ctime_ns": 5,
        }
        self.assertTrue(_stat_matches(identity, dict(identity)))
        for name in identity:
            changed = dict(identity)
            changed[name] += 1
            self.assertFalse(_stat_matches(identity, changed))

    def test_startup_plan_shortfall_reduces_kv_memory_safely(self) -> None:
        gib = 1 << 30
        plan = {
            "kv_cache_memory_bytes": 10 * gib,
            "free_memory_baseline": 100 * gib,
        }
        self.assertEqual(
            adjusted_plan_bytes(plan, 100 * gib - 256 * (1 << 20), 512 * (1 << 20)),
            10 * gib - 256 * (1 << 20),
        )
        self.assertIsNone(adjusted_plan_bytes(plan, 100 * gib - gib, 512 * (1 << 20)))
        self.assertIsNone(adjusted_plan_bytes(plan, 100 * gib, 512 * (1 << 20)))

    def test_startup_plan_shortfall_uses_restore_time_policy(self) -> None:
        gib = 1 << 30
        plan = {
            "kv_cache_memory_bytes": 10 * gib,
            "free_memory_baseline": 100 * gib,
        }
        startup_plan = SimpleNamespace(
            logger=SimpleNamespace(info=lambda *_args: None),
            _applicable_kv_cache_memory_bytes=lambda *_args: None,
        )
        with patch.dict(os.environ, {"COLDSNAP_STARTUP_PLAN_MAX_SHORTFALL_BYTES": "0"}):
            _install_startup_plan_shortfall_adjustment(startup_plan)
            self.assertIsNone(
                startup_plan._applicable_kv_cache_memory_bytes(
                    plan, 100 * gib - 64 * (1 << 20)
                )
            )
            os.environ["COLDSNAP_STARTUP_PLAN_MAX_SHORTFALL_BYTES"] = str(
                512 * (1 << 20)
            )
            self.assertEqual(
                startup_plan._applicable_kv_cache_memory_bytes(
                    plan, 100 * gib - 64 * (1 << 20)
                ),
                10 * gib - 64 * (1 << 20),
            )

    def test_startup_plan_admission_is_collective(self) -> None:
        class Decision:
            def __init__(self, value: int) -> None:
                self.value = value

            def item(self) -> int:
                return self.value

        decisions: list[Decision] = []
        distributed = SimpleNamespace(
            ReduceOp=SimpleNamespace(MIN="min"),
            is_available=lambda: True,
            is_initialized=lambda: True,
            all_reduce=lambda decision, op: (
                self.assertEqual(op, "min") or setattr(decision, "value", 0)
            ),
        )
        fake_torch = ModuleType("torch")
        fake_torch.int32 = "int32"
        fake_torch.distributed = distributed

        def tensor(value: int, *, dtype: str, device: str) -> Decision:
            self.assertEqual(dtype, "int32")
            self.assertEqual(device, "cuda:0")
            decision = Decision(value)
            decisions.append(decision)
            return decision

        fake_torch.tensor = tensor
        worker = SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=2),
            device="cuda:0",
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertFalse(_all_workers_admit_startup_plan(worker, True))
        self.assertEqual([decision.value for decision in decisions], [0])

    def test_startup_plan_collective_rejection_restores_profile_path(self) -> None:
        events: list[str] = []
        startup_plan = SimpleNamespace(
            PLAN_SCHEMA_VERSION=1,
            logger=SimpleNamespace(info=lambda *args: events.append(str(args[0]))),
            compute_plan_fingerprint=lambda *_args: "before-profile",
            _plan_path=lambda _fingerprint: "/unused",
            maybe_apply_startup_plan=lambda worker: setattr(
                worker.cache_config, "kv_cache_memory_bytes", 750
            ),
            maybe_save_startup_plan=lambda *_args: None,
        )
        worker = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=4096)
            ),
            rank=0,
            parallel_config=SimpleNamespace(world_size=2),
            cache_config=SimpleNamespace(kv_cache_memory_bytes=None),
        )
        with patch(
            "coldsnap_startup_plan._all_workers_admit_startup_plan",
            return_value=False,
        ):
            _install_phase_stable_startup_plan_fingerprint(startup_plan)
            startup_plan.maybe_apply_startup_plan(worker)

        self.assertIsNone(worker.cache_config.kv_cache_memory_bytes)
        self.assertIn(
            "ColdSnap startup plan not applied because at least one distributed "
            "worker requires full memory profiling.",
            events,
        )

    def test_startup_plan_saves_alias_for_pre_profile_fingerprint(self) -> None:
        events: list[str] = []
        state = {"fingerprint": "before-profile"}
        logger = SimpleNamespace(
            info=lambda *args: events.append("info:" + str(args[0])),
            warning=lambda *args: events.append("warning:" + str(args[0])),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            startup_plan = SimpleNamespace(
                PLAN_SCHEMA_VERSION=1,
                logger=logger,
                compute_plan_fingerprint=lambda *_args: state["fingerprint"],
                _plan_path=lambda fingerprint: str(root / f"{fingerprint}.json"),
                maybe_apply_startup_plan=lambda _worker: events.append("apply"),
                maybe_save_startup_plan=lambda _worker, _bytes: events.append("save"),
            )
            worker = SimpleNamespace(
                vllm_config=SimpleNamespace(
                    model_config=SimpleNamespace(max_model_len=4096)
                ),
                rank=0,
                parallel_config=SimpleNamespace(world_size=1),
                init_snapshot=SimpleNamespace(free_memory=1000),
            )
            _install_phase_stable_startup_plan_fingerprint(startup_plan)
            startup_plan.maybe_apply_startup_plan(worker)
            state["fingerprint"] = "after-profile"
            startup_plan.maybe_save_startup_plan(worker, 750)

            alias = json.loads((root / "before-profile.json").read_text())
            self.assertEqual(
                alias,
                {
                    "schema": 1,
                    "fingerprint": "before-profile",
                    "kv_cache_memory_bytes": 750,
                    "free_memory_baseline": 1000,
                    "coldsnap_alias_of": "after-profile",
                },
            )
            self.assertEqual(events[0:2], ["info:ColdSnap startup-plan lookup fingerprint %s", "apply"])
            self.assertEqual(getattr(worker, _CONFIGURED_MODEL_LEN_ATTR), 4096)
            self.assertIn("save", events)

    def test_startup_plan_uses_configured_model_len_for_compile_cache(self) -> None:
        observed: list[int] = []

        class Worker:
            def compile_or_warm_up_model(self) -> str:
                observed.append(self.vllm_config.model_config.max_model_len)
                return "compiled"

        model_config = SimpleNamespace(max_model_len=13_300)
        worker = Worker()
        worker.vllm_config = SimpleNamespace(
            model_config=model_config,
            compilation_config=SimpleNamespace(cache_dir=""),
        )
        setattr(worker, _CONFIGURED_MODEL_LEN_ATTR, 1_048_576)
        module = SimpleNamespace(Worker=Worker)

        _install_compile_cache_identity_hook(module)

        self.assertEqual(worker.compile_or_warm_up_model(), "compiled")
        self.assertEqual(observed, [1_048_576])
        self.assertEqual(model_config.max_model_len, 13_300)

    def test_capture_records_and_restore_reuses_compiler_cache_namespace(self) -> None:
        class Worker:
            def compile_or_warm_up_model(self) -> str:
                observed.append(self.vllm_config.compilation_config.cache_dir)
                if not self.vllm_config.compilation_config.cache_dir:
                    cache_dir = root / "torch_compile_cache" / "84fd94c82e"
                    cache_dir.mkdir(parents=True)
                    self.vllm_config.compilation_config.cache_dir = str(cache_dir)
                return "compiled"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "vllm"
            observed: list[str] = []
            with patch.dict(
                os.environ,
                {
                    "VLLM_CACHE_ROOT": str(root),
                    "COLDSNAP_CAPTURE_LOAD_FORMAT": "instanttensor",
                },
                clear=False,
            ):
                worker = Worker()
                worker.vllm_config = SimpleNamespace(
                    model_config=SimpleNamespace(max_model_len=4096),
                    compilation_config=SimpleNamespace(cache_dir=""),
                )
                _install_compile_cache_identity_hook(SimpleNamespace(Worker=Worker))
                self.assertEqual(worker.compile_or_warm_up_model(), "compiled")

            marker = json.loads(
                (root / "coldsnap-compile-cache.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["namespace"], "84fd94c82e")

            with patch.dict(
                os.environ,
                {
                    "VLLM_CACHE_ROOT": str(root),
                    "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                },
                clear=True,
            ):
                worker = Worker()
                worker.vllm_config = SimpleNamespace(
                    model_config=SimpleNamespace(max_model_len=4096),
                    compilation_config=SimpleNamespace(cache_dir=""),
                )
                self.assertEqual(worker.compile_or_warm_up_model(), "compiled")

            self.assertEqual(
                observed,
                ["", str(root / "torch_compile_cache" / "84fd94c82e")],
            )

    def test_compiler_cache_marker_rejects_unqualified_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ, {"VLLM_CACHE_ROOT": directory}, clear=False
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid namespace"):
                    _compile_cache_directory("../capture", require_directory=False)

    def test_restore_free_memory_admission_keeps_captured_envelope(self) -> None:
        gib = 1 << 30
        self.assertEqual(minimum_free_memory(100 * gib, gib), 101 * gib)
        with self.assertRaisesRegex(RuntimeError, "admission values"):
            minimum_free_memory(gib, -1)
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(mem_get_info=lambda: (98 * gib, 120 * gib))
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.dict(
                os.environ,
                {
                    "COLDSNAP_REQUIRED_FREE_MEMORY_BYTES": str(100 * gib),
                    "COLDSNAP_FREE_MEMORY_RESERVE_BYTES": str(gib),
                    "COLDSNAP_STARTUP_PLAN_MAX_SHORTFALL_BYTES": "0",
                },
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "admission failed"):
                _validate_free_memory_admission()

    def test_positional_round_trip(self) -> None:
        payload = bytearray((index * 17) % 251 for index in range(64 * 1024))
        destination = bytearray(len(payload))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob"
            fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
            try:
                _write_all_at(fd, memoryview(payload), 4096)
                _read_exact_at(fd, memoryview(destination), 4096)
            finally:
                os.close(fd)
        self.assertEqual(destination, payload)

    def test_short_snapshot_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob"
            path.write_bytes(b"short")
            fd = os.open(path, os.O_RDONLY)
            try:
                with self.assertRaises(EOFError):
                    _read_exact_at(fd, memoryview(bytearray(16)), 0)
            finally:
                os.close(fd)

    def test_boolean_environment(self) -> None:
        old = os.environ.get("TEST_DISK_SLEEP_BOOL")
        try:
            os.environ["TEST_DISK_SLEEP_BOOL"] = "off"
            self.assertFalse(_env_bool("TEST_DISK_SLEEP_BOOL", True))
            os.environ["TEST_DISK_SLEEP_BOOL"] = "yes"
            self.assertTrue(_env_bool("TEST_DISK_SLEEP_BOOL", False))
        finally:
            if old is None:
                os.environ.pop("TEST_DISK_SLEEP_BOOL", None)
            else:
                os.environ["TEST_DISK_SLEEP_BOOL"] = old

    def test_reference_fingerprint_is_order_independent(self) -> None:
        refs = [
            {
                "name": "b",
                "kind": "parameter",
                "shape": [2],
                "stride": [1],
                "dtype": "torch.uint8",
                "nbytes": 2,
                "allocation_offset": 4,
            },
            {
                "name": "a",
                "kind": "buffer",
                "shape": [1],
                "stride": [1],
                "dtype": "torch.float32",
                "nbytes": 4,
                "allocation_offset": 0,
            },
        ]
        self.assertEqual(reference_fingerprint(refs), reference_fingerprint(list(reversed(refs))))

    def test_recovery_loader_validates_and_filters_safetensors_extents(self) -> None:
        metadata = {
            "scale": {"dtype": "F8_E8M0", "shape": [4], "data_offsets": [0, 4]},
            "model.layer": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [4, 8],
            },
            "__metadata__": {"format": "pt"},
        }
        encoded = json.dumps(metadata, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"abcdefgh")
            descriptors = recovery_descriptors(
                path,
                indexed_tensor_files={
                    "scale": os.path.abspath(Path(directory) / "other.safetensors"),
                    "model.layer": os.path.abspath(path),
                },
                weight_name_prefixes=("model",),
            )

        self.assertEqual(
            descriptors,
            [
                SafetensorDescriptor(
                    name="model.layer",
                    dtype_name="BF16",
                    shape=(2,),
                    file_offset=8 + len(encoded) + 4,
                    length=4,
                )
            ],
        )

    def test_fast_capture_iterator_attaches_recovery_source_metadata(self) -> None:
        metadata = {
            "weight": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]},
            "__metadata__": {"format": "pt"},
        }
        encoded = json.dumps(metadata, separators=(",", ":")).encode()

        class Tensor:
            shape = (4,)

            @staticmethod
            def numel():
                return 4

            @staticmethod
            def element_size():
                return 1

            @staticmethod
            def data_ptr():
                return 0x1234

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"data")
            loader = SimpleNamespace(
                load_config=SimpleNamespace(load_format="instanttensor"),
                _coldsnap_capture_source_metrics=[],
            )
            source = SimpleNamespace(
                model_or_path=directory,
                subfolder=None,
                revision="revision",
                fall_back_to_pt=False,
                allow_patterns_overrides=None,
                weight_name_prefixes=None,
                prefix="",
            )
            prepared = SimpleNamespace(
                folder=directory,
                files=(str(path),),
                prefix="",
                weight_name_prefixes=None,
            )
            with (
                patch.object(
                    recovery_loader,
                    "prepare_synthetic_weight_source",
                    return_value=prepared,
                ),
                patch.object(
                    recovery_loader,
                    "_distributed_context",
                    return_value=(object(), 0, 2),
                ),
            ):
                iterator = recovery_loader._observed_capture_iterator(
                    loader,
                    source,
                    iter([("weight", Tensor())]),
                )
                name, _tensor = next(iterator)
                active = recovery_loader._ACTIVE_RECOVERY_SOURCE.get()
                self.assertEqual(name, "weight")
                self.assertEqual(active.name, "weight")
                self.assertEqual(active.path, os.path.abspath(path))
                self.assertEqual(active.pointer, 0x1234)
                with self.assertRaises(StopIteration):
                    next(iterator)

        metrics = loader._coldsnap_capture_source_metrics[0]
        self.assertEqual(metrics.backend, "capture-observer+instanttensor")
        self.assertEqual(metrics.logical_bytes, 4)
        self.assertEqual(metrics.local_read_bytes, 4)

    def test_recovery_loader_rejects_overlapping_safetensors_extents(self) -> None:
        metadata = {
            "a": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]},
            "b": {"dtype": "I8", "shape": [3], "data_offsets": [3, 6]},
        }
        encoded = json.dumps(metadata, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"abcdef")
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                recovery_descriptors(
                    path,
                    indexed_tensor_files=None,
                    weight_name_prefixes=None,
                )

    def test_recovery_loader_preserves_hf_index_order_over_physical_order(self) -> None:
        metadata = {
            "layer.gate.weight": {
                "dtype": "I8",
                "shape": [2],
                "data_offsets": [0, 2],
            },
            "layer.direct.weight": {
                "dtype": "I8",
                "shape": [2],
                "data_offsets": [2, 4],
            },
        }
        encoded = json.dumps(metadata, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"abcd")
            descriptors = recovery_descriptors(
                path,
                indexed_tensor_files={
                    "layer.direct.weight": os.path.abspath(path),
                    "layer.gate.weight": os.path.abspath(path),
                },
                weight_name_prefixes=None,
            )

        self.assertEqual(
            [descriptor.name for descriptor in descriptors],
            ["layer.direct.weight", "layer.gate.weight"],
        )

    def test_recovery_loader_defaults_to_native_direct_backend(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"COLDSNAP_HYDRATION_NATIVE_LIBRARY": "/native.so"},
                clear=True,
            ),
            patch.object(recovery_loader, "NativeHydrator") as hydrator_type,
        ):
            hydrator_type.return_value.sha256 = "digest"
            hydrator, backend = recovery_loader._native_hydrator()

        self.assertIs(hydrator, hydrator_type.return_value)
        self.assertEqual(backend, "direct")

    def test_recovery_loader_defaults_status_fencing_to_restored_device_group(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(recovery_loader._distributed_status_group())

    def test_fast_capture_primes_recovery_device_group_before_snapshot(self) -> None:
        events = []

        class Tensor:
            value = 2

            def item(self):
                return self.value

        class Distributed:
            @staticmethod
            def all_reduce(tensor, *, group):
                events.append(("all_reduce", group, tensor.value))
                tensor.value = 3

        class CUDA:
            @staticmethod
            def current_device():
                return 0

            @staticmethod
            def synchronize(device):
                events.append(("synchronize", device))

        torch_module = SimpleNamespace(
            cuda=CUDA(),
            distributed=Distributed(),
            int32="int32",
            tensor=lambda values, **_kwargs: Tensor(),
        )
        with patch.object(
            recovery_loader,
            "_distributed_context",
            return_value=("tp-device", 1, 2),
        ):
            seconds = recovery_loader._prime_recovery_device_group(torch_module)

        self.assertGreaterEqual(seconds, 0.0)
        self.assertEqual(
            events,
            [("all_reduce", "tp-device", 2), ("synchronize", 0)],
        )

    def test_single_rank_capture_skips_recovery_device_group_prime(self) -> None:
        with patch.object(
            recovery_loader,
            "_distributed_context",
            return_value=(None, 0, 1),
        ):
            self.assertEqual(
                recovery_loader._prime_recovery_device_group(SimpleNamespace()),
                0.0,
            )

    def test_recovery_loader_rejects_invalid_status_group(self) -> None:
        with patch.dict(
            os.environ,
            {recovery_loader.LOADER_STATUS_GROUP_ENV: "invalid"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be device or cpu"):
                recovery_loader._distributed_status_group()

    def test_recovery_loader_balances_tensor_bytes_across_tp_ranks(self) -> None:
        descriptors = [
            SafetensorDescriptor(str(index), "I8", (size,), 0, size)
            for index, size in enumerate((7, 6, 5, 4))
        ]
        owners = recovery_owners(descriptors, 2)
        self.assertEqual(owners, [0, 0, 1, 1])
        self.assertEqual(
            recovery_owner_layout(descriptors, owners, 2),
            ([13, 9], [0, 7, 0, 5]),
        )
        self.assertEqual(
            recovery_owners(descriptors, 2, rotation=1),
            [1, 1, 0, 0],
        )

    def test_recovery_loader_assigns_tp_owners_in_physical_order(self) -> None:
        descriptors = [
            SafetensorDescriptor("semantic-first", "I8", (6,), 104, 6),
            SafetensorDescriptor("physical-first", "I8", (4,), 100, 4),
        ]

        self.assertEqual(
            recovery_owners(descriptors, 2, physical_order=True),
            [1, 0],
        )

    def test_recovery_loader_weights_owners_by_measured_io(self) -> None:
        descriptors = [
            SafetensorDescriptor(str(index), "I8", (25,), index * 25, 25) for index in range(4)
        ]

        self.assertEqual(
            recovery_loader._weighted_owners(descriptors, (3.0, 1.0)),
            [0, 0, 0, 1],
        )
        self.assertEqual(
            recovery_loader._weighted_owners(
                descriptors,
                (3.0, 1.0),
                rotation=1,
            ),
            [1, 0, 0, 0],
        )

    def test_recovery_loader_accounts_for_mandatory_rank_reads(self) -> None:
        additions = recovery_loader._capacity_aware_additions(
            (2.0, 1.0),
            (75, 0),
            75,
        )
        self.assertAlmostEqual(sum(additions), 75.0)
        self.assertLess(additions[0], additions[1])
        self.assertAlmostEqual((75 + additions[0]) / 2.0, additions[1], places=5)

        descriptors = [
            SafetensorDescriptor(str(index), "I8", (25,), index * 25, 25) for index in range(3)
        ]
        owners = recovery_loader._targeted_owners(descriptors, additions)
        self.assertEqual(owners.count(1), 2)

    def test_replay_ownership_offsets_shared_reads_from_exclusive_load(self) -> None:
        exclusive = (("exclusive", "/weights", "I8", (75,), 0, 75),)
        rows = [
            (
                (f"shared-{index}", "/weights", "I8", (25,), 75 + index * 25, 25),
                SafetensorDescriptor(f"shared-{index}", "I8", (25,), 75 + index * 25, 25),
            )
            for index in range(3)
        ]
        rank_copies = [
            {exclusive[0]: [object()], **{key: [object()] for key, _descriptor in rows}},
            {key: [object()] for key, _descriptor in rows},
        ]

        assignments = recovery_loader._replay_transport_owners(
            rows,
            rank_copies,
            2,
            rotation=0,
            rank_io_weights=(2.0, 1.0),
            rank_assigned_bytes=(75, 0),
        )

        self.assertEqual([owner for owner, _consumers in assignments].count(1), 2)

    def test_recovery_loader_selects_primary_initial_load_profile_through_wrappers(
        self,
    ) -> None:
        def profile(logical_bytes, local_read_bytes, io_seconds):
            return recovery_loader.RecoveryLoadMetrics(
                files=1,
                tensors=1,
                logical_bytes=logical_bytes,
                local_read_bytes=local_read_bytes,
                local_read_extents=1,
                io_seconds=io_seconds,
                collective_seconds=0.0,
                collective_calls=0,
                verification_seconds=0.0,
                verified_sample_bytes=0,
                total_seconds=io_seconds,
                backend="direct",
                distributed_world_size=2,
                staging_layout="bounded",
                staging_buffer_bytes=1,
                collective_buffer_bytes=1,
            )

        target_profile = profile(1000, 600, 2.0)
        draft_profile = profile(100, 90, 0.1)
        target = SimpleNamespace(**{recovery_loader.INITIAL_LOAD_METRICS_ATTR: target_profile})
        draft = SimpleNamespace(**{recovery_loader.INITIAL_LOAD_METRICS_ATTR: draft_profile})
        wrapper = SimpleNamespace(named_modules=lambda: (("target", target), ("draft", draft)))

        self.assertIs(
            recovery_loader._model_initial_recovery_load_metrics(wrapper),
            target_profile,
        )
        self.assertEqual(
            recovery_loader._model_initial_recovery_io_rate(wrapper),
            300.0,
        )

    def test_recovery_loader_bounds_persisted_rank_io_weights(self) -> None:
        self.assertEqual(
            recovery_loader._bounded_rank_io_weights((16.0, 1.0)),
            (16.0, 4.0),
        )
        damped = recovery_loader._damped_rank_io_weights((16.0, 1.0), 0.5)
        self.assertAlmostEqual(damped[0], 11.313708498984761)
        self.assertAlmostEqual(damped[1], 5.656854249492381)
        with self.assertRaisesRegex(ValueError, "positive numbers"):
            recovery_loader._bounded_rank_io_weights((1.0, 0.0))

    def test_recovery_replay_reads_rank_exclusive_sources_locally(self) -> None:
        first = ("first",)
        second = ("second",)
        shared = ("shared",)
        rows = [
            (first, SafetensorDescriptor("first", "I8", (8,), 100, 8)),
            (second, SafetensorDescriptor("second", "I8", (8,), 108, 8)),
            (shared, SafetensorDescriptor("shared", "I8", (8,), 116, 8)),
        ]
        rank_copies = [
            {first: [object()], shared: [object()]},
            {second: [object()], shared: [object()]},
        ]

        assignments = recovery_loader._replay_transport_owners(
            rows,
            rank_copies,
            2,
            rotation=0,
        )

        self.assertEqual(assignments[0], (0, (0,)))
        self.assertEqual(assignments[1], (1, (1,)))
        self.assertEqual(assignments[2][1], (0, 1))
        self.assertIn(assignments[2][0], (0, 1))

    def test_recovery_replay_reads_disjoint_tp_views_rank_locally(self) -> None:
        class Scalar:
            @staticmethod
            def element_size():
                return 1

        torch_module = SimpleNamespace(
            int8=object(),
            empty=lambda *_args, **_kwargs: Scalar(),
        )
        descriptor = SafetensorDescriptor("source", "I8", (16,), 100, 16)
        key = ("source", "/model/weights.safetensors", "I8", (16,), 100, 16)

        def copy(offset):
            return recovery_loader.RecoveryCopy(
                source_name="source",
                source_path="/model/weights.safetensors",
                source_dtype_name="I8",
                source_shape=(16,),
                source_file_offset=100,
                source_length=16,
                source_view_dtype="torch.int8",
                source_view_offset_bytes=offset,
                source_view_shape=(8,),
                source_view_stride=(1,),
                destination_name="layer.weight",
                destination_dtype="torch.int8",
                destination_shape=(8,),
                destination_view_offset_bytes=0,
                destination_view_shape=(8,),
                destination_view_stride=(1,),
                copy_bytes=8,
            )

        ranges = recovery_loader._rank_local_replay_ranges(
            torch_module,
            {key: descriptor},
            [{key: [copy(0)]}, {key: [copy(8)]}],
            1,
        )

        self.assertIsNotNone(ranges)
        assert ranges is not None
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0].source_base_offset_bytes, 8)
        self.assertEqual(ranges[0].descriptor.file_offset, 108)
        self.assertEqual(ranges[0].descriptor.length, 8)

    def test_recovery_replay_broadcasts_fully_replicated_sources(self) -> None:
        class Scalar:
            @staticmethod
            def element_size():
                return 1

        torch_module = SimpleNamespace(
            int8=object(),
            empty=lambda *_args, **_kwargs: Scalar(),
        )
        descriptor = SafetensorDescriptor("source", "I8", (16,), 100, 16)
        copy = recovery_loader.RecoveryCopy(
            source_name="source",
            source_path="/model/weights.safetensors",
            source_dtype_name="I8",
            source_shape=(16,),
            source_file_offset=100,
            source_length=16,
            source_view_dtype="torch.int8",
            source_view_offset_bytes=0,
            source_view_shape=(16,),
            source_view_stride=(1,),
            destination_name="layer.weight",
            destination_dtype="torch.int8",
            destination_shape=(16,),
            destination_view_offset_bytes=0,
            destination_view_shape=(16,),
            destination_view_stride=(1,),
            copy_bytes=16,
        )
        key = copy.source_key

        ranges = recovery_loader._rank_local_replay_ranges(
            torch_module,
            {key: descriptor},
            [{key: [copy]}, {key: [copy]}],
            0,
        )

        self.assertIsNone(ranges)

    def test_recovery_peer_pack_layout_preserves_dtype_alignment(self) -> None:
        class Scalar:
            def __init__(self, size):
                self.size = size

            def element_size(self):
                return self.size

        int8 = object()
        float32 = object()
        torch_module = SimpleNamespace(
            int8=int8,
            float32=float32,
            empty=lambda *_args, dtype, **_kwargs: Scalar(1 if dtype is int8 else 4),
        )

        def copy(dtype, shape):
            return recovery_loader.RecoveryCopy(
                source_name=str(dtype),
                source_path="/model/weights.safetensors",
                source_dtype_name="I8",
                source_shape=shape,
                source_file_offset=100,
                source_length=64,
                source_view_dtype=dtype,
                source_view_offset_bytes=0,
                source_view_shape=shape,
                source_view_stride=(1,),
                destination_name=str(dtype),
                destination_dtype=dtype,
                destination_shape=shape,
                destination_view_offset_bytes=0,
                destination_view_shape=shape,
                destination_view_stride=(1,),
                copy_bytes=8,
            )

        offsets, total = recovery_loader._replay_pack_layout(
            torch_module,
            [copy("torch.int8", (3,)), copy("torch.float32", (2,))],
        )

        self.assertEqual(offsets, [0, 4])
        self.assertEqual(total, 12)

    def test_recovery_transport_sync_does_not_drain_native_io_stream(self) -> None:
        calls = []

        class Stream:
            @staticmethod
            def synchronize():
                calls.append("stream")

        cuda = SimpleNamespace(
            current_stream=lambda device: calls.append(("device", device)) or Stream(),
            synchronize=lambda _device: calls.append("device-wide"),
        )

        recovery_loader._synchronize_replay_stream(
            SimpleNamespace(cuda=cuda),
            3,
        )

        self.assertEqual(calls, [("device", 3), "stream"])

    def test_recovery_local_pack_decomposes_strided_views(self) -> None:
        class Scalar:
            @staticmethod
            def element_size():
                return 1

        torch_module = SimpleNamespace(
            int8=object(),
            empty=lambda *_args, **_kwargs: Scalar(),
        )
        copy = recovery_loader.RecoveryCopy(
            source_name="source",
            source_path="/model/weights.safetensors",
            source_dtype_name="I8",
            source_shape=(3, 4),
            source_file_offset=100,
            source_length=12,
            source_view_dtype="torch.int8",
            source_view_offset_bytes=0,
            source_view_shape=(3, 2),
            source_view_stride=(4, 1),
            destination_name="layer.weight",
            destination_dtype="torch.int8",
            destination_shape=(3, 2),
            destination_view_offset_bytes=0,
            destination_view_shape=(3, 2),
            destination_view_stride=(2, 1),
            copy_bytes=6,
        )

        extents = recovery_loader._packed_copy_extents(torch_module, copy, 8)

        self.assertEqual(
            [(item.file_offset, item.packed_offset_bytes, item.length) for item in extents],
            [(100, 8, 2), (104, 10, 2), (108, 12, 2)],
        )
        contiguous = recovery_loader.replace(
            copy,
            source_view_stride=(2, 1),
            source_length=6,
        )
        self.assertEqual(
            recovery_loader._packed_copy_extents(torch_module, contiguous, 0),
            [recovery_loader._ReplayPackedExtent(100, 0, 6)],
        )

    def test_recovery_local_pack_rejects_duplicated_replica_io(self) -> None:
        class Scalar:
            @staticmethod
            def element_size():
                return 1

        torch_module = SimpleNamespace(
            int8=object(),
            empty=lambda *_args, **_kwargs: Scalar(),
        )

        def copy(offset, shape=(2, 2), stride=(4, 1)):
            return recovery_loader.RecoveryCopy(
                source_name="source",
                source_path="/model/weights.safetensors",
                source_dtype_name="I8",
                source_shape=(2, 4),
                source_file_offset=100,
                source_length=8,
                source_view_dtype="torch.int8",
                source_view_offset_bytes=offset,
                source_view_shape=shape,
                source_view_stride=stride,
                destination_name="layer.weight",
                destination_dtype="torch.int8",
                destination_shape=shape,
                destination_view_offset_bytes=0,
                destination_view_shape=shape,
                destination_view_stride=(shape[-1], 1),
                copy_bytes=4,
            )

        sharded = (
            recovery_loader.RecoveryCopyPlan((copy(0),), frozenset()),
            recovery_loader.RecoveryCopyPlan((copy(2),), frozenset()),
        )
        self.assertTrue(
            recovery_loader._rank_local_packed_replay_plans_qualified(
                torch_module,
                sharded,
                max_extents=4,
                min_extent_bytes=2,
            )
        )

        replicated = (
            recovery_loader.RecoveryCopyPlan(
                (copy(0, shape=(2, 4), stride=(4, 1)),),
                frozenset(),
            ),
        ) * 2
        self.assertFalse(
            recovery_loader._rank_local_packed_replay_plans_qualified(
                torch_module,
                replicated,
                max_extents=4,
                min_extent_bytes=2,
            )
        )

    def test_recovery_metadata_collectives_use_snapshot_portable_device_group(
        self,
    ) -> None:
        device_group = object()
        cpu_group = object()
        tp_group = SimpleNamespace(
            world_size=2,
            device_group=device_group,
            cpu_group=cpu_group,
        )
        parallel_state = ModuleType("vllm.distributed.parallel_state")
        parallel_state.get_tp_group = lambda: tp_group
        calls = []

        def all_gather_object(output, value, *, group):
            self.assertIs(group, device_group)
            self.assertIsNot(group, cpu_group)
            output[:] = [value, value]
            calls.append(value)

        torch_module = SimpleNamespace(
            distributed=SimpleNamespace(all_gather_object=all_gather_object)
        )
        modules = {
            "vllm": ModuleType("vllm"),
            "vllm.distributed": ModuleType("vllm.distributed"),
            "vllm.distributed.parallel_state": parallel_state,
        }
        plan = recovery_loader.RecoveryCopyPlan((), frozenset())
        with patch.dict(sys.modules, modules):
            plans = recovery_loader._gather_recovery_copy_plans(torch_module, plan)
            rates = recovery_loader._gather_recovery_io_weights(torch_module, 1024.0)

        self.assertEqual(plans, (plan, plan))
        self.assertEqual(rates, (1024.0, 1024.0))
        self.assertEqual(calls, [plan, 1024.0])

    def test_recovery_adapter_plan_requires_complete_exclusive_sources(self) -> None:
        def copy(source, destination, offset=0):
            return recovery_loader.RecoveryCopy(
                source_name=source,
                source_path="/model/weights.safetensors",
                source_dtype_name="U8",
                source_shape=(8,),
                source_file_offset=100,
                source_length=8,
                source_view_dtype="torch.uint8",
                source_view_offset_bytes=0,
                source_view_shape=(4,),
                source_view_stride=(1,),
                destination_name=destination,
                destination_dtype="torch.uint8",
                destination_shape=(8,),
                destination_view_offset_bytes=offset,
                destination_view_shape=(4,),
                destination_view_stride=(1,),
                copy_bytes=4,
            )

        model = SimpleNamespace(
            **{
                recovery_loader.CHECKPOINT_COPY_PLAN_ATTR: recovery_loader.RecoveryCopyPlan(
                    copies=(
                        copy("source.a", "layer.a"),
                        copy("source.b", "layer.b"),
                        copy("source.shared", "layer.c"),
                        copy("source.shared", "outside"),
                    ),
                    unsupported_destinations=frozenset({"layer.bad"}),
                )
            }
        )
        complete = SimpleNamespace(
            direct_replay_destination_names=frozenset({"layer.a", "layer.b"})
        )
        shared = SimpleNamespace(direct_replay_destination_names=frozenset({"layer.c"}))
        unsupported = SimpleNamespace(direct_replay_destination_names=frozenset({"layer.bad"}))

        plan = recovery_loader._adapter_replay_plan(model, [complete, shared, unsupported])

        self.assertEqual(
            {item.destination_name for item in plan.copies},
            {"layer.a", "layer.b"},
        )
        self.assertEqual(
            plan.unsupported_destinations,
            frozenset({"layer.c", "layer.bad"}),
        )

    def test_recovery_adapter_plan_allows_partial_exact_replay(self) -> None:
        copy = recovery_loader.RecoveryCopy(
            source_name="source.weight",
            source_path="/model/weights.safetensors",
            source_dtype_name="U8",
            source_shape=(8,),
            source_file_offset=100,
            source_length=8,
            source_view_dtype="torch.uint8",
            source_view_offset_bytes=0,
            source_view_shape=(8,),
            source_view_stride=(1,),
            destination_name="layer.weight",
            destination_dtype="torch.uint8",
            destination_shape=(8,),
            destination_view_offset_bytes=0,
            destination_view_shape=(8,),
            destination_view_stride=(1,),
            copy_bytes=8,
        )
        model = SimpleNamespace(
            **{
                recovery_loader.CHECKPOINT_COPY_PLAN_ATTR: recovery_loader.RecoveryCopyPlan(
                    copies=(copy,),
                    unsupported_destinations=frozenset({"layer.scale"}),
                )
            }
        )
        adapter = SimpleNamespace(
            direct_replay_destination_names=frozenset({"layer.weight", "layer.scale"}),
            allows_partial_direct_replay=True,
        )

        plan = recovery_loader._adapter_replay_plan(model, [adapter])

        self.assertEqual(plan.copies, (copy,))
        self.assertEqual(
            plan.unsupported_destinations,
            frozenset({"layer.scale"}),
        )

    def test_recovery_plan_leaves_adapter_finalizer_trigger_on_normal_loader(self) -> None:
        def copy(destination, offset):
            return recovery_loader.RecoveryCopy(
                source_name=f"source.{destination}",
                source_path="/model/weights.safetensors",
                source_dtype_name="U8",
                source_shape=(8,),
                source_file_offset=100 + offset,
                source_length=8,
                source_view_dtype="torch.uint8",
                source_view_offset_bytes=0,
                source_view_shape=(8,),
                source_view_stride=(1,),
                destination_name=destination,
                destination_dtype="torch.uint8",
                destination_shape=(8,),
                destination_view_offset_bytes=0,
                destination_view_shape=(8,),
                destination_view_stride=(1,),
                copy_bytes=8,
            )

        tensor = SimpleNamespace(
            is_cuda=True,
            shape=(8,),
            dtype="torch.uint8",
            is_contiguous=lambda: True,
            numel=lambda: 8,
            element_size=lambda: 1,
        )
        destinations = frozenset({"layer.weight", "layer.scale"})
        model = SimpleNamespace(
            named_parameters=lambda: iter((name, tensor) for name in destinations),
            named_buffers=lambda: iter(()),
            named_modules=lambda: iter(()),
            **{
                recovery_loader.CHECKPOINT_COPY_PLAN_ATTR: recovery_loader.RecoveryCopyPlan(
                    (copy("layer.weight", 0), copy("layer.scale", 8)),
                    frozenset(),
                )
            },
        )
        adapter = SimpleNamespace(
            direct_replay_destination_names=destinations,
            required_normal_reload_destination_names=frozenset({"layer.scale"}),
            allows_partial_direct_replay=True,
        )

        plan = recovery_loader._recovery_replay_plan(model, [adapter])

        self.assertEqual(
            frozenset(item.destination_name for item in plan.copies),
            frozenset({"layer.weight"}),
        )
        self.assertEqual(plan.unsupported_destinations, frozenset({"layer.scale"}))

    def test_recovery_plan_includes_only_fully_covered_ordinary_tensors(self) -> None:
        def copy(source, destination, offset, size):
            return recovery_loader.RecoveryCopy(
                source_name=source,
                source_path="/model/weights.safetensors",
                source_dtype_name="U8",
                source_shape=(size,),
                source_file_offset=100 + offset,
                source_length=size,
                source_view_dtype="torch.uint8",
                source_view_offset_bytes=0,
                source_view_shape=(size,),
                source_view_stride=(1,),
                destination_name=destination,
                destination_dtype="torch.uint8",
                destination_shape=(8,),
                destination_view_offset_bytes=offset,
                destination_view_shape=(size,),
                destination_view_stride=(1,),
                copy_bytes=size,
            )

        tensor = SimpleNamespace(
            is_cuda=True,
            shape=(8,),
            dtype="torch.uint8",
            is_contiguous=lambda: True,
            numel=lambda: 8,
            element_size=lambda: 1,
        )
        copies = (
            copy("source.left", "weight", 0, 4),
            copy("source.right", "weight", 4, 4),
            copy("source.partial", "partial", 0, 4),
        )
        model = SimpleNamespace(
            named_parameters=lambda: iter((("weight", tensor), ("partial", tensor))),
            named_buffers=lambda: iter(()),
            **{
                recovery_loader.CHECKPOINT_COPY_PLAN_ATTR: recovery_loader.RecoveryCopyPlan(
                    copies,
                    frozenset(),
                )
            },
        )

        plan = recovery_loader._recovery_replay_plan(model, [])

        self.assertEqual(
            {item.destination_name for item in plan.copies},
            {"weight"},
        )

    def test_recovery_plan_rejects_source_shared_with_fallback_destination(self) -> None:
        def copy(destination, offset):
            return recovery_loader.RecoveryCopy(
                source_name="source.shared",
                source_path="/model/weights.safetensors",
                source_dtype_name="U8",
                source_shape=(8,),
                source_file_offset=100,
                source_length=8,
                source_view_dtype="torch.uint8",
                source_view_offset_bytes=offset,
                source_view_shape=(4,),
                source_view_stride=(1,),
                destination_name=destination,
                destination_dtype="torch.uint8",
                destination_shape=(8,),
                destination_view_offset_bytes=offset,
                destination_view_shape=(4,),
                destination_view_stride=(1,),
                copy_bytes=4,
            )

        tensor = SimpleNamespace(
            is_cuda=True,
            shape=(8,),
            dtype="torch.uint8",
            is_contiguous=lambda: True,
            numel=lambda: 8,
            element_size=lambda: 1,
        )
        model = SimpleNamespace(
            named_parameters=lambda: iter((("weight", tensor), ("partial", tensor))),
            named_buffers=lambda: iter(()),
            **{
                recovery_loader.CHECKPOINT_COPY_PLAN_ATTR: recovery_loader.RecoveryCopyPlan(
                    (copy("weight", 0), copy("weight", 4), copy("partial", 0)),
                    frozenset(),
                )
            },
        )

        plan = recovery_loader._recovery_replay_plan(model, [])

        self.assertEqual(plan.copies, ())

    def test_recovery_plan_keeps_structural_eager_dependencies_on_normal_loader(
        self,
    ) -> None:
        def copy(destination):
            return recovery_loader.RecoveryCopy(
                source_name=f"source.{destination}",
                source_path="/model/weights.safetensors",
                source_dtype_name="U8",
                source_shape=(8,),
                source_file_offset=100,
                source_length=8,
                source_view_dtype="torch.uint8",
                source_view_offset_bytes=0,
                source_view_shape=(8,),
                source_view_stride=(1,),
                destination_name=destination,
                destination_dtype="torch.uint8",
                destination_shape=(8,),
                destination_view_offset_bytes=0,
                destination_view_shape=(8,),
                destination_view_stride=(1,),
                copy_bytes=8,
            )

        tensor = SimpleNamespace(
            is_cuda=True,
            shape=(8,),
            dtype="torch.uint8",
            is_contiguous=lambda: True,
            numel=lambda: 8,
            element_size=lambda: 1,
        )
        mhc = SimpleNamespace(
            _use_b12x_mhc=True,
            refresh_b12x_mhc_bf16_weights=lambda: None,
            hc_attn_fn=tensor,
            hc_ffn_fn=tensor,
        )
        names = ("layer.hc_attn_fn", "layer.hc_ffn_fn", "layer.weight")
        model = SimpleNamespace(
            named_modules=lambda: iter((("layer", mhc),)),
            named_parameters=lambda: iter((name, tensor) for name in names),
            named_buffers=lambda: iter(()),
            **{
                recovery_loader.CHECKPOINT_COPY_PLAN_ATTR: recovery_loader.RecoveryCopyPlan(
                    tuple(copy(name) for name in names),
                    frozenset(),
                )
            },
        )

        self.assertEqual(
            recovery_loader._recovery_reload_dependency_names(model),
            frozenset({"layer.hc_attn_fn", "layer.hc_ffn_fn"}),
        )
        plan = recovery_loader._recovery_replay_plan(model, [])

        self.assertEqual(
            frozenset(item.destination_name for item in plan.copies),
            frozenset({"layer.weight"}),
        )
        self.assertEqual(
            plan.unsupported_destinations,
            frozenset({"layer.hc_attn_fn", "layer.hc_ffn_fn"}),
        )

    def test_recovery_loader_rebases_metadata_through_model_wrappers(self) -> None:
        copy = recovery_loader.RecoveryCopy(
            source_name="source.weight",
            source_path="/model/weights.safetensors",
            source_dtype_name="U8",
            source_shape=(8,),
            source_file_offset=100,
            source_length=8,
            source_view_dtype="torch.uint8",
            source_view_offset_bytes=0,
            source_view_shape=(8,),
            source_view_stride=(1,),
            destination_name="layer.weight",
            destination_dtype="torch.uint8",
            destination_shape=(8,),
            destination_view_offset_bytes=0,
            destination_view_shape=(8,),
            destination_view_stride=(1,),
            copy_bytes=8,
        )
        inner = SimpleNamespace(
            **{
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR: frozenset(
                    {"layer.weight"}
                ),
                recovery_loader.CHECKPOINT_COPY_PLAN_ATTR: recovery_loader.RecoveryCopyPlan(
                    copies=(copy,),
                    unsupported_destinations=frozenset({"layer.bias"}),
                ),
            }
        )
        wrapper = SimpleNamespace(named_modules=lambda: iter((("", wrapper), ("_orig_mod", inner))))

        names = recovery_loader._model_checkpoint_destination_names(wrapper)
        plan = recovery_loader._model_checkpoint_copy_plan(wrapper)

        self.assertEqual(names, frozenset({"_orig_mod.layer.weight"}))
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            [item.destination_name for item in plan.copies],
            ["_orig_mod.layer.weight"],
        )
        self.assertEqual(
            plan.unsupported_destinations,
            frozenset({"_orig_mod.layer.bias"}),
        )

    def test_recovery_view_span_accounts_for_strides(self) -> None:
        tensor = SimpleNamespace(
            numel=lambda: 6,
            shape=(2, 3),
            stride=lambda: (5, 1),
            element_size=lambda: 2,
        )

        self.assertEqual(recovery_loader._view_span_bytes(tensor), 16)

    def test_recovery_copy_plan_preserves_observed_dtype_conversion(self) -> None:
        class Tensor:
            def __init__(self, pointer, dtype):
                self._pointer = pointer
                self.dtype = dtype
                self.shape = (8,)

            def data_ptr(self):
                return self._pointer

            def numel(self):
                return 8

            def element_size(self):
                return 1

            def stride(self):
                return (1,)

        descriptor = SafetensorDescriptor("source", "I8", (8,), 100, 8)
        source = recovery_loader.RecoverySourceTensor(
            name="source",
            path="/model/weights.safetensors",
            descriptor=descriptor,
            pointer=1000,
        )
        destination = Tensor(2000, "torch.uint8")

        copy, reason = recovery_loader._observed_recovery_copy(
            source,
            "layer.weight",
            destination,
            Tensor(2000, "torch.uint8"),
            Tensor(1000, "torch.int8"),
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(copy)
        assert copy is not None
        self.assertEqual(copy.source_view_dtype, "torch.int8")
        self.assertEqual(copy.destination_dtype, "torch.uint8")
        self.assertEqual(copy.copy_bytes, 8)

    def test_recovery_loader_bounds_staging_without_splitting_tensors(self) -> None:
        descriptors = [
            SafetensorDescriptor(str(index), "I8", (size,), 0, size)
            for index, size in enumerate((3, 4, 11, 2, 5))
        ]

        self.assertEqual(
            list(recovery_descriptor_batches(descriptors, 8)),
            [(0, 2), (2, 3), (3, 5)],
        )
        with self.assertRaisesRegex(ValueError, "staging bytes"):
            list(recovery_descriptor_batches(descriptors, 0))

    def test_recovery_loader_uses_views_of_one_bounded_staging_slab(self) -> None:
        class Slab:
            def __init__(self, size):
                self.size = size

            def numel(self):
                return self.size

            def __getitem__(self, item):
                return (item.start, item.stop)

        descriptors = [
            SafetensorDescriptor("a", "I8", (3,), 0, 3),
            SafetensorDescriptor("b", "I8", (5,), 3, 5),
        ]

        self.assertEqual(
            recovery_storage_views(Slab(8), descriptors),
            [(0, 3), (3, 8)],
        )
        with self.assertRaisesRegex(ValueError, "expected 8"):
            recovery_storage_views(Slab(7), descriptors)

    def test_recovery_loader_lays_staging_views_out_in_file_order(self) -> None:
        class Slab:
            @staticmethod
            def numel():
                return 10

            @staticmethod
            def __getitem__(item):
                return (item.start, item.stop)

        descriptors = [
            SafetensorDescriptor("semantic-first", "I8", (6,), 104, 6),
            SafetensorDescriptor("physical-first", "I8", (4,), 100, 4),
        ]

        self.assertEqual(
            recovery_storage_views(Slab(), descriptors, physical_order=True),
            [(4, 10), (0, 4)],
        )

    def test_recovery_loader_coalesces_adjacent_file_and_cuda_ranges(self) -> None:
        descriptors = [
            SafetensorDescriptor("a", "I8", (4,), 100, 4),
            SafetensorDescriptor("b", "I8", (6,), 104, 6),
            SafetensorDescriptor("c", "I8", (2,), 112, 2),
        ]
        tensors = [
            SimpleNamespace(data_ptr=lambda: 1000),
            SimpleNamespace(data_ptr=lambda: 1004),
            SimpleNamespace(data_ptr=lambda: 1010),
        ]
        extents = recovery_coalesced_extents(
            descriptors,
            tensors,
            [0, 0, 0],
            0,
            coalesce_adjacent=True,
        )
        self.assertEqual(
            [(item.file_offset, item.destination, item.length) for item in extents],
            [(100, 1000, 10), (112, 1010, 2)],
        )

    def test_recovery_loader_keeps_independent_storage_extents_separate(self) -> None:
        descriptors = [
            SafetensorDescriptor("a", "I8", (4,), 100, 4),
            SafetensorDescriptor("b", "I8", (6,), 104, 6),
        ]
        tensors = [
            SimpleNamespace(data_ptr=lambda: 1000),
            SimpleNamespace(data_ptr=lambda: 1004),
        ]

        extents = recovery_coalesced_extents(descriptors, tensors, [0, 0], 0)

        self.assertEqual(
            [(item.file_offset, item.destination, item.length) for item in extents],
            [(100, 1000, 4), (104, 1004, 6)],
        )

    def test_recovery_loader_splits_unaligned_edges_from_direct_interior(self) -> None:
        direct, buffered = recovery_split_direct_extents(
            [recovery_loader.HydrationExtent(100, 1000, 10000)]
        )

        self.assertEqual(
            [(item.file_offset, item.destination, item.length) for item in direct],
            [(4096, 4996, 4096)],
        )
        self.assertEqual(
            [(item.file_offset, item.destination, item.length) for item in buffered],
            [(100, 1000, 3996), (8192, 9092, 1908)],
        )

    def test_recovery_loader_broadcasts_contiguous_owner_slab_ranges(self) -> None:
        calls = []

        class View:
            def __init__(self, pointer, size):
                self.pointer = pointer
                self.size = size

            def data_ptr(self):
                return self.pointer

            def numel(self):
                return self.size

        class Slab(View):
            def __getitem__(self, item):
                return (item.start, item.stop)

        class Distributed:
            @staticmethod
            def get_global_rank(group, rank):
                return rank + 10

            @staticmethod
            def broadcast(region, *, src, group):
                calls.append((region, src, group))

        count = recovery_broadcast_slab_regions(
            SimpleNamespace(distributed=Distributed()),
            "tp",
            Slab(1000, 12),
            [View(1000, 4), View(1004, 4), View(1008, 4)],
            [0, 0, 1],
            5,
        )

        self.assertEqual(count, 3)
        self.assertEqual(
            calls,
            [((0, 5), 10, "tp"), ((5, 8), 10, "tp"), ((8, 12), 11, "tp")],
        )

    def test_recovery_loader_coalesces_nccl_without_shared_tensor_storage(self) -> None:
        calls = []

        class Storage:
            def __init__(self, name, size):
                self.name = name
                self.size = size

            def numel(self):
                return self.size

        class Distributed:
            @staticmethod
            def get_global_rank(group, rank):
                return rank + 10

            @staticmethod
            def _broadcast_coalesced(group, tensors, buffer_bytes, source_rank):
                calls.append(
                    (group, [tensor.name for tensor in tensors], buffer_bytes, source_rank)
                )

        torch_module = SimpleNamespace(distributed=Distributed())
        count = recovery_broadcast_tensors(
            torch_module,
            "tp",
            [Storage("a", 4), Storage("b", 0), Storage("c", 8)],
            [0, 0, 1],
            2,
            64 * 1024**2,
        )

        self.assertEqual(count, 2)
        self.assertEqual(
            calls,
            [
                ("tp", ["a"], 64 * 1024**2, 10),
                ("tp", ["c"], 64 * 1024**2, 11),
            ],
        )

    def test_recovery_loader_verifies_bounded_post_collective_samples(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self.values = list(values)

            def __getitem__(self, index):
                return FakeTensor(self.values[index])

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class FakeTorch:
            @staticmethod
            def cat(parts):
                return FakeTensor(value for part in parts for value in part.values)

        source = b"prefixabcdefghsuffix"
        descriptor = SafetensorDescriptor("weight", "I8", (8,), 6, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            path.write_bytes(source)
            verified = recovery_verify_tensor_samples(
                FakeTorch,
                path,
                [descriptor],
                [FakeTensor(b"abcdefgh")],
                2,
            )
            self.assertEqual(verified, 4)
            with self.assertRaisesRegex(RuntimeError, "weight.*head"):
                recovery_verify_tensor_samples(
                    FakeTorch,
                    path,
                    [descriptor],
                    [FakeTensor(b"xbcdefgh")],
                    2,
                )

    def test_latest_vllm_weight_source_contract_is_adapted_structurally(self) -> None:
        calls = []

        class Loader:
            def _prepare_weights(
                self,
                model_name_or_path,
                subfolder,
                revision,
                fall_back_to_pt,
                allow_patterns_overrides,
            ):
                calls.append(
                    (
                        model_name_or_path,
                        subfolder,
                        revision,
                        fall_back_to_pt,
                        allow_patterns_overrides,
                    )
                )
                return "/model", ["/model/weights.safetensors"], True

        source = SimpleNamespace(
            model_or_path="model",
            subfolder=None,
            revision="revision",
            fall_back_to_pt=False,
            allow_patterns_overrides=["*.safetensors"],
            prefix="model.",
        )
        prepared = prepare_synthetic_weight_source(Loader(), source)
        self.assertEqual(
            calls,
            [("model", None, "revision", False, ["*.safetensors"])],
        )
        self.assertEqual(prepared.folder, "/model")
        self.assertEqual(prepared.files, ("/model/weights.safetensors",))
        self.assertEqual(prepared.prefix, "model.")
        self.assertIsNone(prepared.weight_name_prefixes)

    def test_weight_name_prefix_contract_is_supported(self) -> None:
        calls = []

        class Loader:
            def _prepare_weights(
                self,
                model_name_or_path,
                subfolder,
                revision,
                fall_back_to_pt,
                allow_patterns_overrides,
                weight_name_prefixes,
            ):
                calls.append(weight_name_prefixes)
                return "/model", ["/model/weights.safetensors"], True

        source = SimpleNamespace(
            model_or_path="model",
            subfolder=None,
            revision=None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
            weight_name_prefixes=("model.layers",),
            prefix="",
        )
        prepared = prepare_synthetic_weight_source(Loader(), source)
        self.assertEqual(calls, [("model.layers",)])
        self.assertEqual(prepared.weight_name_prefixes, ("model.layers",))

    def test_unknown_required_loader_parameter_fails_before_weight_loading(self) -> None:
        class Loader:
            def _prepare_weights(self, model_name_or_path, new_required_contract):
                raise AssertionError("adapter should reject before calling upstream")

        source = SimpleNamespace(model_or_path="model", prefix="")
        with self.assertRaisesRegex(VllmContractError, "new_required_contract"):
            prepare_synthetic_weight_source(Loader(), source)

    def test_allocator_state_contract_fails_before_mutation(self) -> None:
        allocator = SimpleNamespace(
            pointer_to_data={1: SimpleNamespace(tag="weights", handle=(1, 1))}
        )
        with self.assertRaisesRegex(VllmContractError, "is_asleep"):
            vllm_allocations(allocator, "weights")

    def test_rpc_tcp_override_preserves_unrelated_constructor_parameters(self) -> None:
        class MessageQueue:
            def __init__(
                self,
                n_reader,
                n_local_reader,
                local_reader_ranks=None,
                *,
                future_option="default",
            ):
                self.values = (
                    n_reader,
                    n_local_reader,
                    local_reader_ranks,
                    future_option,
                )

        force_cross_rank_rpc_over_tcp(MessageQueue)
        cross_rank = MessageQueue(2, 1, [0], future_option="preserved")
        self.assertEqual(cross_rank.values, (2, 0, [], "preserved"))
        local = MessageQueue(1, 1, [0], future_option="preserved")
        self.assertEqual(local.values, (1, 1, [0], "preserved"))

    def test_sleep_backend_uses_public_named_registration(self) -> None:
        class Backend:
            pass

        class Factory:
            calls = []

            @classmethod
            def register_backend(cls, name, module_path, class_name):
                cls.calls.append((name, module_path, class_name))

        mode = register_sleep_backend(Factory, Backend)
        self.assertEqual(mode, "public")
        self.assertEqual(
            Factory.calls,
            [(SLEEP_BACKEND_NAME, Backend.__module__, Backend.__name__)],
        )

    def test_explicit_default_backend_override_is_isolated(self) -> None:
        class CuMem:
            pass

        class Backend:
            pass

        class Factory:
            _registry = {"cumem": lambda: CuMem}

        override_default_sleep_backend(Factory, Backend)
        self.assertIs(Factory._registry["cumem"](), Backend)

    def test_synthetic_loader_uses_public_model_loader_registry(self) -> None:
        registered = {}

        class DefaultModelLoader:
            def _get_weights_iterator(self, source):
                return source

        original = DefaultModelLoader._get_weights_iterator

        def register_model_loader(name):
            def register(loader_class):
                registered[name] = loader_class
                return loader_class

            return register

        default_module = ModuleType("vllm.model_executor.model_loader.default_loader")
        default_module.DefaultModelLoader = DefaultModelLoader
        loader_module = ModuleType("vllm.model_executor.model_loader")
        loader_module.register_model_loader = register_model_loader
        modules = {
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.model_loader": loader_module,
            "vllm.model_executor.model_loader.default_loader": default_module,
        }
        environment = {
            "COLDSNAP_PROCESS_TEMPLATE_PHASE": "pre_worker_import",
            "COLDSNAP_CAPTURE_LOAD_FORMAT": "instanttensor",
        }
        with patch.dict(sys.modules, modules), patch.dict(os.environ, environment, clear=True):
            install_synthetic_weight_loader()

        self.assertIn("instanttensor", registered)
        self.assertTrue(issubclass(registered["instanttensor"], DefaultModelLoader))
        self.assertIs(DefaultModelLoader._get_weights_iterator, original)

    def test_n580_bootstrap_index_is_capture_local_and_activation_gated(self) -> None:
        source = SimpleNamespace(
            model_or_path="org/model",
            revision="commit",
            subfolder=None,
            prefix="",
            weight_name_prefixes=None,
        )
        environment = {
            "COLDSNAP_PROCESS_TEMPLATE_PHASE": "pre_worker_import",
            "COLDSNAP_EXPORT_MODEL_PAYLOAD": "1",
            "COLDSNAP_CAPTURE_ID": "capture",
            "COLDSNAP_EXPECTED_RANK": "0",
            "COLDSNAP_DISK_SLEEP_DIR": "unused",
            "COLDSNAP_EXECUTION_GRAPH": json.dumps(
                {
                    "unit": "unit-0",
                    "by_process_slot": {"0": "worker-0"},
                    "groups": {},
                }
            ),
            "VLLM_HOST_IP": "node-a",
        }
        with tempfile.TemporaryDirectory() as directory:
            environment["COLDSNAP_DISK_SLEEP_DIR"] = directory
            with patch.dict(os.environ, environment, clear=True):
                path = synthetic_loader.record_native_bootstrap_source(
                    source,
                    "",
                    [
                        {
                            "name": "model.weight",
                            "dtype": "F16",
                            "shape": [2, 4],
                            "length": 16,
                        }
                    ],
                )
                self.assertIsNotNone(path)
                self.assertFalse(synthetic_loader._native_bootstrap_requested())
                root = path.parent
                (root / "native-manifest.json").write_text("{}\n", encoding="utf-8")
                (root / "model-weights.pack").write_bytes(b"payload")
                (root / "activation-provider").write_text("native\n", encoding="utf-8")
                self.assertTrue(synthetic_loader._native_bootstrap_requested())
                index = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(index["kind"], "coldsnap-native-bootstrap-index")
        self.assertEqual(index["worker_id"], "worker-0")
        self.assertEqual(index["sources"][0]["tensors"][0]["name"], "model.weight")

    def test_process_template_hydration_directory_survives_rank_transition(self) -> None:
        graph = json.dumps(
            {
                "unit": "unit-0",
                "by_process_slot": {"0": "worker-0"},
                "groups": {},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = root / "worker-id-worker-0-process-template"
            captured.mkdir()
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_PHASE": "pre_worker_import",
                "COLDSNAP_EXECUTION_GRAPH": graph,
                "COLDSNAP_EXPECTED_RANK": "0",
                "RANK": "0",
                "VLLM_HOST_IP": "restore-host",
            }
            with patch.dict(os.environ, environment, clear=True):
                resolved = disk_backend._snapshot_directory(root)

        self.assertEqual(resolved, captured)

    def test_activation_locator_overrides_a_rank_qualified_directory(self) -> None:
        graph = json.dumps(
            {
                "unit": "unit-0",
                "by_process_slot": {"0": "worker-0"},
                "groups": {},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = root / "worker-id-worker-0-process-template"
            preferred = root / (f"worker-id-worker-0-host-restore-host-rank-0-pid-{os.getpid()}")
            captured.mkdir()
            preferred.mkdir()
            (root / f"{disk_backend.ACTIVATION_DIRECTORY_PREFIX}worker-0").write_text(
                captured.name + "\n", encoding="utf-8"
            )
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_PHASE": "pre_worker_import",
                "COLDSNAP_EXECUTION_GRAPH": graph,
                "COLDSNAP_EXPECTED_RANK": "0",
                "RANK": "0",
                "VLLM_HOST_IP": "restore-host",
            }
            with patch.dict(os.environ, environment, clear=True):
                resolved = disk_backend._snapshot_directory(root)

        self.assertEqual(resolved, captured)

    def test_restored_process_template_locator_accepts_ranked_capture_identity(self) -> None:
        graph = json.dumps(
            {
                "unit": "unit-0",
                "by_process_slot": {"0": "worker-0"},
                "groups": {},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = root / "worker-id-worker-0-host-capture-host-rank-0-pid-614"
            captured.mkdir()
            (root / f"{disk_backend.ACTIVATION_DIRECTORY_PREFIX}worker-0").write_text(
                captured.name + "\n", encoding="utf-8"
            )
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                "COLDSNAP_EXECUTION_GRAPH": graph,
                "COLDSNAP_EXPECTED_RANK": "0",
                "RANK": "0",
                "VLLM_HOST_IP": "restore-host",
            }
            with patch.dict(os.environ, environment, clear=True):
                resolved = disk_backend._snapshot_directory(root)

        self.assertEqual(resolved, captured)

    def test_restored_process_template_locator_accepts_topology_rank(self) -> None:
        graph = json.dumps(
            {
                "unit": "unit-0",
                "by_process_slot": {"0": "worker-0"},
                "groups": {
                    "world": {
                        "kind": "vllm:world",
                        "size": 2,
                        "ranks": {"0": "worker-0"},
                    }
                },
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = root / "worker-id-worker-0-host-capture-host-rank-0-pid-614"
            captured.mkdir()
            (root / f"{disk_backend.ACTIVATION_DIRECTORY_PREFIX}worker-0").write_text(
                captured.name + "\n", encoding="utf-8"
            )
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                "COLDSNAP_EXECUTION_GRAPH": graph,
                "VLLM_HOST_IP": "restore-host",
            }
            with patch.dict(os.environ, environment, clear=True):
                resolved = disk_backend._snapshot_directory(root)

        self.assertEqual(resolved, captured)

    def test_restored_process_template_locator_rejects_other_topology_rank(
        self,
    ) -> None:
        graph = json.dumps(
            {
                "unit": "unit-0",
                "by_process_slot": {"0": "worker-0"},
                "groups": {
                    "world": {
                        "kind": "vllm:world",
                        "size": 2,
                        "ranks": {"0": "worker-0"},
                    }
                },
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = root / "worker-id-worker-0-host-capture-host-rank-1-pid-614"
            captured.mkdir()
            (root / f"{disk_backend.ACTIVATION_DIRECTORY_PREFIX}worker-0").write_text(
                captured.name + "\n", encoding="utf-8"
            )
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                "COLDSNAP_EXECUTION_GRAPH": graph,
                "VLLM_HOST_IP": "restore-host",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(RuntimeError, "does not belong"),
            ):
                disk_backend._snapshot_directory(root)

    def test_capture_identity_locator_requires_restored_process_template(self) -> None:
        graph = json.dumps(
            {
                "unit": "unit-0",
                "by_process_slot": {"0": "worker-0"},
                "groups": {},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = root / "worker-id-worker-0-host-capture-host-rank-0-pid-614"
            captured.mkdir()
            (root / f"{disk_backend.ACTIVATION_DIRECTORY_PREFIX}worker-0").write_text(
                captured.name + "\n", encoding="utf-8"
            )
            environment = {
                "COLDSNAP_EXECUTION_GRAPH": graph,
                "COLDSNAP_EXPECTED_RANK": "0",
                "RANK": "0",
                "VLLM_HOST_IP": "restore-host",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(RuntimeError, "does not belong"),
            ):
                disk_backend._snapshot_directory(root)

    def test_activated_portable_native_manifest_rejects_unranked_capture_owner(
        self,
    ) -> None:
        manifest = {
            "weight_source": disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
            "portable_model_payload": True,
            "worker_id": "worker-0",
            "rank": -1,
            "pid": 614,
            "identity": ("worker-id-worker-0-host-capture-host-rank-unknown-pid-614"),
        }
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                disk_backend._activated_portable_native_manifest_matches(manifest, "worker-0", 0)
            )
            self.assertFalse(
                disk_backend._activated_portable_native_manifest_matches(manifest, "worker-1", 0)
            )

    def test_activated_portable_native_manifest_accepts_ranked_capture_owner(
        self,
    ) -> None:
        manifest = {
            "weight_source": disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE,
            "portable_model_payload": True,
            "worker_id": "worker-0",
            "rank": 0,
            "pid": 614,
            "identity": "worker-id-worker-0-host-capture-host-rank-0-pid-614",
        }
        # The final vLLM exec intentionally drops the temporary restored-process
        # environment marker.  The durable activation selector is validated by
        # the caller before this ownership exception is considered.
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(
                disk_backend._activated_portable_native_manifest_matches(manifest, "worker-0", 0)
            )
            self.assertFalse(
                disk_backend._activated_portable_native_manifest_matches(manifest, "worker-0", 1)
            )

    def test_process_template_recovery_manifest_admits_only_matching_worker_rank(self) -> None:
        manifest = {
            "weight_source": disk_backend.WEIGHT_SOURCE_SAFETENSORS,
            "worker_id": "worker-0",
            "rank": 0,
            "pid": 614,
            "identity": "worker-id-worker-0-host-capture-host-rank-0-pid-614",
        }
        self.assertTrue(
            disk_backend._restored_process_template_recovery_manifest_matches(
                manifest, "worker-0", 0, None
            )
        )
        self.assertTrue(
            disk_backend._restored_process_template_recovery_manifest_matches(
                manifest, "worker-0", 0, "recovery"
            )
        )
        self.assertFalse(
            disk_backend._restored_process_template_recovery_manifest_matches(
                manifest, "worker-0", 0, "native"
            )
        )
        self.assertFalse(
            disk_backend._restored_process_template_recovery_manifest_matches(
                manifest, "worker-1", 0, None
            )
        )
        self.assertFalse(
            disk_backend._restored_process_template_recovery_manifest_matches(
                manifest, "worker-0", 1, None
            )
        )
        manifest["weight_source"] = disk_backend.WEIGHT_SOURCE_SPLIT_NATIVE
        self.assertFalse(
            disk_backend._restored_process_template_recovery_manifest_matches(
                manifest, "worker-0", 0, None
            )
        )

    def test_process_template_manifest_identity_rejects_unknown_rank(self) -> None:
        environment = {
            "COLDSNAP_PROCESS_TEMPLATE_PHASE": "pre_worker_import",
        }
        identity = f"worker-id-worker-0-host-capture-host-rank-unknown-pid-{os.getpid()}"
        with patch.dict(os.environ, environment, clear=True):
            self.assertFalse(
                disk_backend._portable_identity_matches(identity, "worker-0", 0, os.getpid())
            )
            self.assertFalse(
                disk_backend._portable_identity_matches(identity, "worker-1", 0, os.getpid())
            )

    def test_process_template_compile_materializes_after_layout_is_complete(self) -> None:
        events = []

        class Tensor:
            is_cuda = True

            @staticmethod
            def numel():
                return 128

            @staticmethod
            def element_size():
                return 2

            @staticmethod
            def is_contiguous():
                return True

            @staticmethod
            def data_ptr():
                return 0x1100

        model = SimpleNamespace(
            named_parameters=lambda: iter((("weight", Tensor()),)),
            named_buffers=lambda: iter(()),
            named_modules=lambda: iter(()),
        )
        setattr(
            model,
            recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR,
            frozenset({"weight"}),
        )

        class Backend:
            uses_model_weight_recovery = True
            memory_provider = SimpleNamespace(
                allocations=lambda _tag: [
                    SimpleNamespace(pointer=0x1000, size=0x1000)
                ]
            )

            def set_model_weight_ranges(self, ranges):
                events.append(("ranges", ranges))

            def set_model_weight_semantics(self, layout):
                events.append(("recovery-semantics", layout))

            def set_native_model_payload_semantics(self, layout):
                events.append(("native-semantics", layout))

            def materialize_initial_recovery_payload(self):
                events.append("materialize")
                return {"state": "ready", "path": "/cache/model.pack"}

            @staticmethod
            def set_weight_recovery_callback(_callback):
                return None

        class Worker:
            def __init__(self):
                self.backend = Backend()
                self.model_runner = SimpleNamespace(get_model=lambda: model)

            def _get_sleep_mode_backend(self):
                return self.backend

            def compile_or_warm_up_model(self):
                events.append("compile")
                return "compiled"

            @staticmethod
            def wake_up(tags=None):
                return tags

        gpu_worker_module = ModuleType("vllm.v1.worker.gpu_worker")
        gpu_worker_module.Worker = Worker
        worker_package = ModuleType("vllm.v1.worker")
        worker_package.gpu_worker = gpu_worker_module
        modules = {
            "vllm": ModuleType("vllm"),
            "vllm.v1": ModuleType("vllm.v1"),
            "vllm.v1.worker": worker_package,
            "vllm.v1.worker.gpu_worker": gpu_worker_module,
        }
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "materialize.json"
            control.touch()
            with (
                patch.dict(sys.modules, modules),
                patch.dict(
                    os.environ,
                    {
                        "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                        "COLDSNAP_MODEL_PAYLOAD_MATERIALIZATION_CONTROL": str(control),
                    },
                    clear=True,
                ),
            ):
                recovery_loader._install_worker_wake_hook()
                self.assertEqual(Worker().compile_or_warm_up_model(), "compiled")

        self.assertEqual(events[0], "compile")
        self.assertEqual(events[-1], "materialize")
        self.assertEqual(events[1][0], "ranges")
        self.assertEqual(events[2][0], "recovery-semantics")
        self.assertEqual(events[3][0], "native-semantics")

    def test_recovery_loader_registers_format_and_bridges_worker_wake(self) -> None:
        registered = {}
        events = []

        class DefaultModelLoader:
            def __init__(self):
                self.load_config = SimpleNamespace(load_format=RECOVERY_LOAD_FORMAT)
                self.counter_before_loading_weights = 0.0

            def load_weights(self, model, _model_config):
                events.append("default-load")
                return model.load_weights(self.get_all_weights(_model_config, model))

            def get_all_weights(self, _model_config, _model):
                yield "checkpoint", object()
                yield "source.only", object()

            def _get_weights_iterator(self, _source):
                return iter(())

            def _prepare_weights(
                self,
                model_name_or_path,
                subfolder,
                revision,
                fall_back_to_pt,
                allow_patterns_overrides,
                weight_name_prefixes=None,
            ):
                events.append(("prepare-format", self.load_config.load_format))
                return "/model", ["/model/weights.safetensors"], True

        def register_model_loader(name):
            def register(loader_class):
                registered[name] = loader_class
                return loader_class

            return register

        class Backend:
            uses_model_weight_recovery = True

            def __init__(self):
                self.callback = None

            def set_weight_recovery_callback(self, callback):
                self.callback = callback
                events.append(("callback", callback is not None))

        class Worker:
            def __init__(self):
                self.backend = Backend()
                self._coldsnap_recovery_derived_buffers = {}
                self.model_runner = SimpleNamespace(
                    load_config=SimpleNamespace(load_format=RECOVERY_LOAD_FORMAT),
                    reload_weights=lambda: events.append("reload"),
                    get_model=lambda: SimpleNamespace(modules=lambda: ()),
                )

            def _get_sleep_mode_backend(self):
                return self.backend

            def wake_up(self, tags=None):
                events.append(("wake", tags))
                self.backend.callback()
                return "awake"

        default_module = ModuleType("vllm.model_executor.model_loader.default_loader")
        default_module.DefaultModelLoader = DefaultModelLoader
        loader_module = ModuleType("vllm.model_executor.model_loader")
        loader_module.register_model_loader = register_model_loader
        logger_module = ModuleType("vllm.logger")
        logger_module.init_logger = lambda _name: SimpleNamespace(info=lambda *_args: None)
        weight_utils_module = ModuleType("vllm.model_executor.model_loader.weight_utils")
        weight_utils_module.default_weight_loader = lambda parameter, value: setattr(
            parameter, "value", value
        )
        gpu_worker_module = ModuleType("vllm.v1.worker.gpu_worker")
        gpu_worker_module.Worker = Worker
        worker_package = ModuleType("vllm.v1.worker")
        worker_package.gpu_worker = gpu_worker_module
        modules = {
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.model_loader": loader_module,
            "vllm.model_executor.model_loader.default_loader": default_module,
            "vllm.model_executor.model_loader.weight_utils": weight_utils_module,
            "vllm.logger": logger_module,
            "vllm.v1": ModuleType("vllm.v1"),
            "vllm.v1.worker": worker_package,
            "vllm.v1.worker.gpu_worker": gpu_worker_module,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(
                os.environ,
                {"COLDSNAP_RECOVERY_WEIGHT_SOURCE": "safetensors"},
                clear=True,
            ),
        ):
            install_recovery_aware_loader()
            loader = registered[RECOVERY_LOAD_FORMAT]()
            source = SimpleNamespace(
                model_or_path="model",
                subfolder=None,
                revision="revision",
                fall_back_to_pt=False,
                allow_patterns_overrides=None,
                weight_name_prefixes=None,
                prefix="",
            )
            loader._get_weights_iterator(source)
            worker = Worker()
            self.assertEqual(worker.wake_up(["weights"]), "awake")

        self.assertEqual(events[0], ("prepare-format", "safetensors"))
        self.assertEqual(loader.load_config.load_format, RECOVERY_LOAD_FORMAT)
        self.assertEqual(
            events[1:],
            [("callback", True), ("wake", ["weights"]), "reload", ("callback", False)],
        )

        checkpoint_parameter = SimpleNamespace()
        runtime_parameter = SimpleNamespace()

        def checkpoint_loader(_parameter, value):
            events.append(("parameter-load", value))

        checkpoint_parameter.weight_loader = checkpoint_loader

        def original_model_load(weights):
            received = list(weights)
            self.assertEqual(
                [name for name, _tensor in received],
                ["checkpoint", "source.only"],
            )
            checkpoint_parameter.weight_loader(
                checkpoint_parameter,
                received[0][1],
            )
            return {"checkpoint", "fused.weight"}

        model = SimpleNamespace(
            load_weights=original_model_load,
            named_parameters=lambda: iter(
                [
                    ("checkpoint", checkpoint_parameter),
                    ("runtime", runtime_parameter),
                ]
            ),
            named_buffers=lambda: iter([("checkpoint_alias", checkpoint_parameter)]),
        )
        with patch.dict(sys.modules, modules):
            loader.load_weights(model, object())
        self.assertEqual(
            getattr(
                model,
                recovery_loader.CHECKPOINT_DESTINATION_TENSOR_NAMES_ATTR,
            ),
            frozenset({"checkpoint", "checkpoint_alias"}),
        )
        self.assertIs(checkpoint_parameter.weight_loader, checkpoint_loader)
        self.assertFalse(hasattr(runtime_parameter, "weight_loader"))
        self.assertIn("default-load", events)
        self.assertEqual(events[-1][0], "parameter-load")

    def test_model_payload_capture_observes_normal_loader_without_replacing_it(self) -> None:
        class DefaultModelLoader:
            pass

        class Worker:
            def sleep(self, level=1):
                return level

        default_module = ModuleType("vllm.model_executor.model_loader.default_loader")
        default_module.DefaultModelLoader = DefaultModelLoader
        gpu_worker_module = ModuleType("vllm.v1.worker.gpu_worker")
        gpu_worker_module.Worker = Worker
        worker_package = ModuleType("vllm.v1.worker")
        worker_package.gpu_worker = gpu_worker_module
        modules = {
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.model_loader": ModuleType("vllm.model_executor.model_loader"),
            "vllm.model_executor.model_loader.default_loader": default_module,
            "vllm.v1": ModuleType("vllm.v1"),
            "vllm.v1.worker": worker_package,
            "vllm.v1.worker.gpu_worker": gpu_worker_module,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(
                os.environ,
                {
                    "COLDSNAP_EXPORT_MODEL_PAYLOAD": "1",
                    "COLDSNAP_RECOVERY_WEIGHT_SOURCE": "blob",
                },
                clear=True,
            ),
            patch.object(recovery_loader, "_install_capture_loader_observer") as observer,
        ):
            recovery_loader.install_model_payload_capture_hook()

        observer.assert_called_once_with(DefaultModelLoader)
        self.assertTrue(Worker._coldsnap_model_payload_capture_installed)

    def test_recovery_loader_preserves_existing_hybrid_draft_selection(self) -> None:
        calls = []

        def replace(value, **changes):
            fields = vars(value) | changes
            return SimpleNamespace(**fields)

        def resolve(vllm_config, model_config, load_config):
            calls.append((model_config, load_config.load_format))
            if model_config is vllm_config.speculative_config.draft_model_config:
                return replace(
                    load_config,
                    load_format="safetensors",
                    safetensors_load_strategy="lazy",
                )
            return load_config

        target = object()
        draft = object()
        recovery_config = SimpleNamespace(load_format=RECOVERY_LOAD_FORMAT)
        vllm_config = SimpleNamespace(
            load_config=recovery_config,
            speculative_config=SimpleNamespace(draft_model_config=draft),
        )
        loader_module = ModuleType("vllm.model_executor.model_loader")
        loader_module._instanttensor_draft_load_config = resolve
        config_module = ModuleType("vllm.config")
        config_module.replace = replace
        modules = {
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.model_loader": loader_module,
            "vllm.config": config_module,
        }

        with patch.dict(sys.modules, modules):
            _install_recovery_hybrid_draft_bridge()
            target_config = loader_module._instanttensor_draft_load_config(
                vllm_config, target, recovery_config
            )
            draft_config = loader_module._instanttensor_draft_load_config(
                vllm_config, draft, recovery_config
            )

        self.assertIs(target_config, recovery_config)
        self.assertEqual(draft_config.load_format, "safetensors")
        self.assertEqual(draft_config.safetensors_load_strategy, "lazy")
        self.assertEqual(calls, [(target, "instanttensor"), (draft, "instanttensor")])

    def test_recovery_loader_installs_generic_hybrid_draft_selection(self) -> None:
        calls = []

        def replace(value, **changes):
            return SimpleNamespace(**(vars(value) | changes))

        def get_model(*, vllm_config, model_config, prefix, load_config):
            calls.append((vllm_config, model_config, prefix, load_config))
            return load_config

        target = object()
        draft = object()
        recovery_config = SimpleNamespace(load_format=RECOVERY_LOAD_FORMAT)
        vllm_config = SimpleNamespace(
            model_config=target,
            load_config=recovery_config,
            speculative_config=SimpleNamespace(draft_model_config=draft),
        )
        loader_module = ModuleType("vllm.model_executor.model_loader")
        loader_module.get_model = get_model
        config_module = ModuleType("vllm.config")
        config_module.replace = replace
        modules = {
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.model_loader": loader_module,
            "vllm.config": config_module,
        }

        with patch.dict(sys.modules, modules):
            _install_recovery_hybrid_draft_bridge()
            target_config = loader_module.get_model(
                vllm_config=vllm_config,
                prefix="target",
            )
            draft_config = loader_module.get_model(
                vllm_config=vllm_config,
                model_config=draft,
                prefix="draft",
            )

        self.assertIs(target_config, recovery_config)
        self.assertEqual(draft_config.load_format, "safetensors")
        self.assertEqual(draft_config.safetensors_load_strategy, "lazy")
        self.assertEqual([call[1] for call in calls], [target, draft])

    def test_semantic_replay_has_no_pointer_or_order_fallback(self) -> None:
        class FakeTensor:
            device = SimpleNamespace(type="cuda")
            dtype = "fake.uint8"
            shape = (4,)

            def numel(self) -> int:
                return 4

            def element_size(self) -> int:
                return 1

            def data_ptr(self) -> int:
                return 1000

            def stride(self) -> tuple[int]:
                return (1,)

        tensor = FakeTensor()
        saved = [
            {
                "index": 0,
                "ptr": 1000,
                "size": 4,
                "refs": [
                    {
                        "name": "wrong.owner",
                        "kind": "parameter",
                        "shape": [4],
                        "stride": [1],
                        "dtype": str(getattr(tensor, "dtype", None)),
                        "nbytes": 4,
                        "allocation_offset": 0,
                    }
                ],
            }
        ]
        fake_torch = SimpleNamespace(Tensor=FakeTensor)
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch(
                "coldsnap_layout.model_tensors",
                return_value=[("actual.owner", "parameter", tensor)],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot place"):
                semantic_replay_map(saved, [(1000, 4)], object())

    def test_gb10_startup_plan_falls_back_only_for_nvml_not_supported(self) -> None:
        class NotSupported(Exception):
            pass

        class Platform:
            @staticmethod
            def get_device_total_memory() -> int:
                raise NotSupported()

            @staticmethod
            def get_device_name() -> str:
                return "NVIDIA GB10"

        modules = {
            "vllm.platforms": SimpleNamespace(current_platform=Platform()),
            "vllm.third_party.pynvml": SimpleNamespace(NVMLError_NotSupported=NotSupported),
        }
        with (
            patch.dict(sys.modules, modules),
            patch("coldsnap_startup_plan.os.sysconf", side_effect=[1, 1234]),
            patch(
                "coldsnap_startup_plan._install_phase_stable_startup_plan_fingerprint"
            ),
            patch("coldsnap_startup_plan._install_startup_plan_shortfall_adjustment"),
        ):
            install_startup_plan_memory_fallback()
            platform = modules["vllm.platforms"].current_platform
            self.assertEqual(platform.get_device_total_memory(), 1234)


class DiscardRegionTest(unittest.TestCase):
    """Regions declared DISCARD are unmapped without ever being preserved."""

    @staticmethod
    def logger_modules():
        logger = ModuleType("vllm.logger")
        logger.init_logger = lambda name: SimpleNamespace(info=lambda *a, **k: None)
        return patch.dict(sys.modules, {"vllm": ModuleType("vllm"), "vllm.logger": logger})

    def test_only_regions_declared_discard_may_be_dropped(self) -> None:
        self.assertEqual(_discard_regions("kv_cache"), ("kv_cache",))
        self.assertEqual(_discard_regions(""), ())
        with self.assertRaisesRegex(ValueError, "unknown region"):
            _discard_regions("not_a_region")
        # weights declares PRESERVE and cuda_graph declares RECREATE. Neither
        # may be silently dropped just because a caller named it.
        with self.assertRaisesRegex(ValueError, "declares policy 'preserve'"):
            _discard_regions("weights")
        with self.assertRaisesRegex(ValueError, "declares policy 'recreate'"):
            _discard_regions("cuda_graph")
        with self.assertRaisesRegex(ValueError, "declares policy 'external'"):
            _discard_regions("external")
        with self.assertRaisesRegex(ValueError, "duplicate discard region"):
            _discard_regions("kv_cache,kv_cache")

    def test_discard_regions_read_from_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_discard_regions(), ())
        with patch.dict(os.environ, {"COLDSNAP_DISCARD_REGIONS": " kv_cache "}, clear=False):
            self.assertEqual(_discard_regions(), ("kv_cache",))
        with patch.dict(os.environ, {"COLDSNAP_DISCARD_REGIONS": ""}, clear=False):
            self.assertEqual(_discard_regions(), ())

    def test_discard_delegates_to_provider_payload_semantics(self) -> None:
        allocations = [
            SimpleNamespace(pointer=300, size=100),
            SimpleNamespace(pointer=100, size=10),
            SimpleNamespace(pointer=200, size=20),
        ]

        class Memory:
            def allocations(self, region):
                return allocations

            def select_allocations(self, region, purpose):
                self.selection = (region, purpose)
                return (allocations[0], allocations[2])

        memory = Memory()
        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        selected = backend._selected_discard_allocations(memory, "kv_cache")
        self.assertEqual(memory.selection, ("kv_cache", "payload"))
        self.assertEqual(
            [(index, item.pointer) for index, item in selected],
            [(1, 200), (2, 300)],
        )

    def test_discard_remap_readback_samples_start_middle_and_end(self) -> None:
        copied: list[tuple[int, int]] = []

        class Memory:
            def allocate_host_stage(self, size):
                owner = bytearray(size)
                return HostStage(owner=owner, view=memoryview(owner), pointer=0)

            def synchronize(self):
                copied.append((-1, 0))

            def copy_to_host(self, stage, pointer, size):
                copied.append((pointer, size))
                stage.view[:size] = b"\x00" * size

        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        backend.discard_verify_sample_bytes = 4
        result = backend._verify_discard_extent(Memory(), 1000, 20)
        self.assertEqual(
            copied,
            [(-1, 0), (1000, 4), (1008, 4), (1016, 4)],
        )
        self.assertEqual(
            result,
            {
                "sample_bytes": 4,
                "samples": [
                    {"offset": 0, "bytes": 4},
                    {"offset": 8, "bytes": 4},
                    {"offset": 16, "bytes": 4},
                ],
            },
        )

    def test_discard_remap_readback_rejects_nonzero_pages(self) -> None:
        class Memory:
            def allocate_host_stage(self, size):
                owner = bytearray(size)
                return HostStage(owner=owner, view=memoryview(owner), pointer=0)

            def synchronize(self):
                pass

            def copy_to_host(self, stage, pointer, size):
                stage.view[:size] = b"\x00" * (size - 1) + b"\x01"

        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        backend.discard_verify_sample_bytes = 4
        with self.assertRaisesRegex(RuntimeError, "nonzero_bytes=1/4"):
            backend._verify_discard_extent(Memory(), 1000, 20)

    def _fixture(self) -> dict[str, Any]:
        events: list[object] = []

        class FakeAllocation:
            """Minimal conforming ManagedAllocation."""

            def __init__(self, pointer: int, size: int, tag: str) -> None:
                self.pointer = pointer
                self.size = size
                self.tag = tag
                self.is_released = False

            def set_released(self, value: bool) -> None:
                self.is_released = value

        weights = FakeAllocation(1000, 64, "weights")
        kv = FakeAllocation(2000, 4096, "kv_cache")
        pool = [weights, kv]

        class Memory:
            def allocations(self, tag=None):
                return [item for item in pool if tag is None or item.tag == tag]

            def select_allocations(self, tag, purpose):
                if tag != "kv_cache" or purpose != "payload":
                    raise AssertionError((tag, purpose))
                return self.allocations(tag)

            def allocate_host_stage(self, size):
                owner = bytearray(size)
                return HostStage(owner=owner, view=memoryview(owner), pointer=0)

            def copy_to_host(self, stage, pointer, size):
                events.append(("readback", pointer, size))
                stage.view[:size] = b"\x00" * size

            def synchronize(self):
                events.append("sync")

            def empty_cache(self):
                events.append("empty-cache")

            def release(self, value):
                events.append(("release", value.tag, value.size))
                value.set_released(True)

            def remap(self, value):
                events.append(("remap", value.tag, value.size))

            def fill_zero(self, pointer, size):
                events.append(("zero", pointer, size))

        backend = DiskCuMemBackend.__new__(DiskCuMemBackend)
        backend.memory_provider = Memory()
        backend.discard_regions = ("kv_cache",)
        backend.graph_controller = SimpleNamespace(
            release=lambda: events.append("graph-release"),
            restore=lambda: events.append("graph-restore"),
            enabled=False,
            stats=lambda: SimpleNamespace(
                allocation_count=0, raw_bytes=0, mapped_bytes=0, paused_count=0
            ),
        )
        backend._state = "RUNNING"
        backend._restore_mapped = set()
        backend.read_direct = False
        backend.verify_mode = "inline"
        backend.reuse_blob = True
        backend._reusable_generation = None
        backend._reusable_blob_stat = None
        states: list[tuple[str, dict[str, Any]]] = []
        backend._write_state = lambda state, **values: states.append((state, values))
        return {
            "backend": backend,
            "events": events,
            "states": states,
            "weights": weights,
            "kv": kv,
        }

    def test_suspend_releases_discard_regions_without_writing_bytes(self) -> None:
        fixture = self._fixture()
        backend = fixture["backend"]
        written: list[object] = []
        backend.manifest_path = Path("manifest.json")
        backend.blob_path = Path("weights.blob")

        def snapshot(memory):
            # Only the preserved region may reach the blob writer.
            written.extend(memory.allocations("weights"))
            return {
                "generation": "generation",
                "blob_bytes": 64,
                "blob_stat": {},
                "entries": [{}],
                "direct_io": False,
                "write_io_mode": "buffered",
                "blob_reused": False,
                "write_seconds": 0.1,
                "verification_seconds": 0.0,
                "phase_seconds": {},
            }

        backend._write_snapshot = snapshot
        with self.logger_modules():
            backend.suspend()

        self.assertEqual([item.tag for item in written], ["weights"])
        self.assertTrue(fixture["kv"].is_released)
        self.assertIn(("release", "kv_cache", 4096), fixture["events"])
        # The KV cache is unmapped only after the weights, so a failure while
        # writing the blob leaves it addressable.
        self.assertLess(
            fixture["events"].index(("release", "weights", 64)),
            fixture["events"].index(("release", "kv_cache", 4096)),
        )
        sleeping = [values for state, values in fixture["states"] if state == "sleeping"]
        self.assertEqual(sleeping[0]["discarded_bytes"], 4096)
        self.assertEqual(sleeping[0]["discard_regions"], ["kv_cache"])
        self.assertEqual(
            sleeping[0]["regions"]["kv_cache"],
            {
                "policy": "discard",
                "allocation_count": 1,
                "bytes": 4096,
                "released_count": 1,
            },
        )

    def test_resume_remaps_discard_regions_without_reading_bytes(self) -> None:
        fixture = self._fixture()
        backend = fixture["backend"]
        fixture["weights"].is_released = True
        fixture["kv"].is_released = True
        backend._state = "SUSPENDED"
        manifest = {
            "generation": "generation",
            "blob_bytes": 64,
            "blob_stat": {},
            "entries": [{}],
        }
        backend._load_manifest = lambda memory: manifest
        read_bytes: list[int] = []

        def restore_pipeline(memory, value, fd):
            read_bytes.append(64)
            fixture["events"].append("weight-restore")
            return {"restored_bytes": 64}

        backend._restore_pipeline = restore_pipeline

        def commit(memory):
            fixture["events"].append("weight-commit")
            fixture["weights"].is_released = False

        backend._commit_restore = commit
        with tempfile.TemporaryDirectory() as directory:
            backend.blob_path = Path(directory) / "weights.blob"
            backend.manifest_path = Path(directory) / "manifest.json"
            backend.blob_path.write_bytes(b"snapshot")
            with self.logger_modules():
                backend.resume()

        # The KV cache is remapped but nothing is read for it: the only bytes
        # read belong to the preserved weight blob.
        self.assertEqual(read_bytes, [64])
        self.assertIn(("remap", "kv_cache", 4096), fixture["events"])
        self.assertLess(
            fixture["events"].index("empty-cache"),
            fixture["events"].index(("remap", "kv_cache", 4096)),
        )
        # Remapped pages are undefined, and no engine can be relied on to clear
        # them, so the region is zeroed before it becomes visible.
        self.assertIn(("zero", 2000, 4096), fixture["events"])
        self.assertLess(
            fixture["events"].index(("remap", "kv_cache", 4096)),
            fixture["events"].index(("zero", 2000, 4096)),
        )
        self.assertFalse(fixture["kv"].is_released)
        running = [values for state, values in fixture["states"] if state == "running"]
        self.assertEqual(running[0]["discard_remapped_bytes"], 4096)
        self.assertIn("discard_cache_release_seconds", running[0]["phase_seconds"])
        self.assertEqual(running[0]["woke_discard_regions"], ["kv_cache"])
        self.assertTrue(running[0]["woke_preserved"])

    def test_wake_tags_select_regions_and_reject_unmanaged_names(self) -> None:
        backend = self._fixture()["backend"]
        self.assertEqual(backend._resume_selection(None), (True, ("kv_cache",)))
        self.assertEqual(backend._resume_selection(["weights"]), (True, ()))
        self.assertEqual(backend._resume_selection(["kv_cache"]), (False, ("kv_cache",)))
        self.assertEqual(
            backend._resume_selection(["weights", "kv_cache"]),
            (True, ("kv_cache",)),
        )
        with self.assertRaisesRegex(ValueError, "does not manage region"):
            backend._resume_selection(["cuda_graph"])
        with self.assertRaisesRegex(ValueError, "at least one managed region"):
            backend._resume_selection([])

    def test_discard_only_wake_leaves_weights_and_graphs_alone(self) -> None:
        fixture = self._fixture()
        backend = fixture["backend"]
        fixture["kv"].is_released = True
        backend._state = "SUSPENDED"
        backend.manifest_path = Path("manifest.json")
        backend.blob_path = Path("weights.blob")
        backend._load_manifest = lambda memory: self.fail(
            "a discard-only wake must not read the weight manifest"
        )
        backend._restore_pipeline = lambda *a: self.fail(
            "a discard-only wake must not run the hydration pipeline"
        )
        with self.logger_modules():
            backend.resume(["kv_cache"])

        self.assertIn(("remap", "kv_cache", 4096), fixture["events"])
        self.assertNotIn("graph-restore", fixture["events"])
        self.assertFalse(fixture["kv"].is_released)
        running = [values for state, values in fixture["states"] if state == "running"]
        self.assertFalse(running[0]["woke_preserved"])

    def test_provider_without_zeroing_is_refused(self) -> None:
        """Undefined KV memory faults a sparse attention backend, so a provider
        that cannot define it must fail rather than publish garbage."""
        fixture = self._fixture()
        backend = fixture["backend"]
        fixture["kv"].is_released = True
        backend._state = "SUSPENDED"
        backend.manifest_path = Path("manifest.json")
        backend.blob_path = Path("weights.blob")
        del type(backend.memory_provider).fill_zero
        with self.logger_modules():
            with self.assertRaisesRegex(RuntimeError, "cannot zero discard region"):
                backend.resume(["kv_cache"])

    def test_asleep_region_detection_covers_discard_regions(self) -> None:
        allocator = SimpleNamespace(
            pointer_to_data={
                1: SimpleNamespace(tag="weights", is_asleep=False, handle=(1, 1)),
                2: SimpleNamespace(tag="kv_cache", is_asleep=False, handle=(2, 1)),
            }
        )
        self.assertFalse(_has_asleep_regions(allocator, ("kv_cache",)))
        allocator.pointer_to_data[2].is_asleep = True
        self.assertTrue(_has_asleep_regions(allocator, ("kv_cache",)))
        # A released KV cache must not be mistaken for released weights.
        self.assertFalse(_has_asleep_weights(allocator))


if __name__ == "__main__":
    unittest.main()

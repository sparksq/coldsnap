# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import ast
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations" / "core"))
sys.path.insert(0, str(ROOT / "integrations" / "sglang"))

from coldsnap_core.artifact import (  # noqa: E402
    ARTIFACT_FORMAT,
    ARTIFACT_KIND,
    ArtifactError,
    SemanticTensorArtifact,
)
import coldsnap_core.artifact as core_artifact  # noqa: E402
from coldsnap_core.validation import publish_validation_record  # noqa: E402
from coldsnap_sglang.adapter import ColdSnapMemorySaverAdapter  # noqa: E402
from coldsnap_sglang.compat import (  # noqa: E402
    SGLangCompatibilityError,
    resolve_weight_updater_contract,
)
import coldsnap_sglang.identity as sglang_identity  # noqa: E402
import coldsnap_sglang.plugin as sglang_plugin  # noqa: E402
from coldsnap_sglang.identity import _capacity_class, _driver_abi_major  # noqa: E402
from coldsnap_sglang.settings import Settings, SettingsError  # noqa: E402
from coldsnap_sglang.shape_calibration import (  # noqa: E402
    SGLangShapeCalibrationError,
    calibrate_capture_shapes,
)


class SettingsTest(unittest.TestCase):
    def test_installed_plugin_is_noop_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings.from_env().mode, "off")

    def test_enabled_mode_requires_artifact_root(self) -> None:
        with patch.dict(os.environ, {"COLDSNAP_MODE": "capture"}, clear=True):
            with self.assertRaisesRegex(SettingsError, "ARTIFACT_DIR"):
                Settings.from_env()

    def test_enabled_mode_requires_process_artifact_root(self) -> None:
        with patch.dict(
            os.environ,
            {"COLDSNAP_MODE": "capture", "COLDSNAP_ARTIFACT_DIR": "/artifact"},
            clear=True,
        ):
            with self.assertRaisesRegex(SettingsError, "PROCESS_ARTIFACT_ROOT"):
                Settings.from_env()

    def test_fresh_process_restore_modes_are_not_supported(self) -> None:
        for mode in ("restore", "auto"):
            with (
                self.subTest(mode=mode),
                patch.dict(
                    os.environ,
                    {
                        "COLDSNAP_MODE": mode,
                        "COLDSNAP_ARTIFACT_DIR": "/artifact",
                        "COLDSNAP_PROCESS_ARTIFACT_ROOT": "/process-artifact",
                    },
                    clear=True,
                ),
            ):
                with self.assertRaisesRegex(SettingsError, "off or capture"):
                    Settings.from_env()

    def test_runtime_paths_are_derived_and_discard_backing_is_supported(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "COLDSNAP_MODE": "capture",
                    "COLDSNAP_ARTIFACT_DIR": str(Path(directory) / "artifact"),
                    "COLDSNAP_PROCESS_ARTIFACT_ROOT": str(Path(directory) / "process-artifact"),
                    "COLDSNAP_RUNTIME_DIR": str(Path(directory) / "runtime"),
                },
                clear=True,
            ),
        ):
            settings = Settings.from_env()
            self.assertEqual(settings.live_root, Path(directory) / "runtime/live")
            self.assertEqual(settings.lock_root, Path(directory) / "runtime/locks")

        with patch.dict(
            os.environ,
            {
                "COLDSNAP_MODE": "capture",
                "COLDSNAP_ARTIFACT_DIR": "/artifact",
                "COLDSNAP_PROCESS_ARTIFACT_ROOT": "/process-artifact",
                "COLDSNAP_LIVE_BACKING": "discard",
            },
            clear=True,
        ):
            self.assertEqual(Settings.from_env().live_backing, "discard")


class SGLangShapeCalibrationTest(unittest.TestCase):
    @staticmethod
    def _graph_runner(planned, captured):
        return SimpleNamespace(
            capture_bs=list(planned),
            graphs={value: object() for value in captured},
        )

    def test_engine_owned_decode_and_prefill_plans_are_fully_covered(self) -> None:
        runner = SimpleNamespace(
            decode_cuda_graph_runner=self._graph_runner([1, 2, 4], [1, 2, 4]),
            prefill_cuda_graph_runner=self._graph_runner([8, 16], [8, 16]),
        )
        with patch(
            "coldsnap_sglang.shape_calibration._toolchain",
            return_value={"sglang": "test", "cuda_architecture": "sm_121"},
        ):
            result = calibrate_capture_shapes(runner)

        self.assertEqual(result["engine"], "sglang")
        self.assertEqual(result["strategy"], "verify-engine-startup-capture")
        self.assertEqual(result["planned_shapes"], 5)
        self.assertEqual(result["warmed_shapes"], 5)
        self.assertEqual([mode["mode"] for mode in result["modes"]], ["decode", "prefill"])

    def test_missing_planned_shape_fails_capture_coverage(self) -> None:
        runner = SimpleNamespace(
            decode_cuda_graph_runner=self._graph_runner([1, 2, 4], [1, 2]),
        )
        with self.assertRaisesRegex(SGLangShapeCalibrationError, r"omitted planned shapes \[4\]"):
            calibrate_capture_shapes(runner)

    def test_current_sglang_backend_shape_keys_are_fully_covered(self) -> None:
        class ShapeKey:
            def __init__(self, size, stream_idx=None, variant_label=None):
                self.size = size
                self.stream_idx = stream_idx
                self.variant_label = variant_label

        backend = SimpleNamespace(
            _graphs={
                ShapeKey(1): object(),
                ShapeKey(2): object(),
                ShapeKey(4, variant_label="lora"): object(),
                ShapeKey(4, variant_label="nolora"): object(),
            }
        )
        runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                capture_bs=[1, 2, 4],
                backend=backend,
            )
        )
        with patch(
            "coldsnap_sglang.shape_calibration._toolchain",
            return_value={"sglang": "test", "cuda_architecture": "sm_121"},
        ):
            result = calibrate_capture_shapes(runner)

        self.assertEqual(result["planned_shapes"], 3)
        self.assertEqual(result["warmed_shapes"], 3)
        self.assertEqual(result["modes"][0]["captured_batch_sizes"], [1, 2, 4])

    def test_disabled_phase_eager_runner_is_not_treated_as_a_cuda_graph(self) -> None:
        config = SimpleNamespace(
            decode=SimpleNamespace(backend="full"),
            prefill=SimpleNamespace(backend="disabled"),
        )
        runner = SimpleNamespace(
            server_args=SimpleNamespace(cuda_graph_config=config),
            decode_cuda_graph_runner=self._graph_runner([1, 2, 4], [1, 2, 4]),
            prefill_cuda_graph_runner=SimpleNamespace(),
        )
        with patch(
            "coldsnap_sglang.shape_calibration._toolchain",
            return_value={"sglang": "test", "cuda_architecture": "sm_121"},
        ):
            result = calibrate_capture_shapes(runner)

        self.assertEqual([mode["mode"] for mode in result["modes"]], ["decode"])

    def test_unrecognized_graph_plan_fails_closed(self) -> None:
        runner = SimpleNamespace(decode_cuda_graph_runner=SimpleNamespace(graphs={1: object()}))
        with self.assertRaisesRegex(SGLangShapeCalibrationError, "no recognized capture"):
            calibrate_capture_shapes(runner)


class CriuRestoreHoldTest(unittest.TestCase):
    def test_n610_hold_publishes_worker_identity_and_waits_for_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                mode="capture",
                artifact_root=root / "semantic",
                live_backing="discard",
                live_root=root / "live",
                preverified=True,
                process_artifact_root=root,
            )
            with (
                patch.object(sglang_plugin, "_settings", settings),
                patch.dict(os.environ, {"COLDSNAP_SGLANG_CRIU_HOLD": "1"}, clear=True),
                patch("coldsnap_core.topology.worker_id", return_value="worker-7"),
            ):
                thread = threading.Thread(target=sglang_plugin._wait_at_cuda_restore_hold)
                thread.start()
                ready = root / "cuda-restore-hold/worker-7.ready.json"
                deadline = time.monotonic() + 2
                while not ready.is_file() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.is_file())
                record = json.loads(ready.read_text(encoding="utf-8"))
                self.assertEqual(record["worker_id"], "worker-7")
                self.assertTrue(thread.is_alive())
                (ready.parent / "release").touch()
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())

    def test_async_graph_and_native_startup_settings_are_typed(self) -> None:
        with patch.dict(
            os.environ,
            {
                "COLDSNAP_MODE": "capture",
                "COLDSNAP_ARTIFACT_DIR": "/artifact",
                "COLDSNAP_PROCESS_ARTIFACT_ROOT": "/process-artifact",
                "COLDSNAP_ASYNC_CUDA_GRAPHS": "1",
                "COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE": "/artifact/graphs.arm",
                "COLDSNAP_ASYNC_CUDA_GRAPHS_READY_FILE": "/artifact/graphs.json",
                "COLDSNAP_SGLANG_STARTUP_PROVIDER": "native",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertTrue(settings.async_graphs)
        self.assertEqual(settings.async_graph_arm_file, Path("/artifact/graphs.arm"))
        self.assertEqual(settings.async_graph_ready_file, Path("/artifact/graphs.json"))
        self.assertEqual(settings.startup_provider, "native")


class ArtifactManifestTest(unittest.TestCase):
    def test_semantic_artifact_uses_shared_native_capture_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = SemanticTensorArtifact(directory)

            class Storage:
                def __init__(self, pointer, size):
                    self.pointer = pointer
                    self.size = size

                def data_ptr(self):
                    return self.pointer

                def nbytes(self):
                    return self.size

            class Tensor:
                def __init__(self, pointer, size):
                    self.storage = Storage(pointer, size)
                    self.device = SimpleNamespace(index=0)

                def untyped_storage(self):
                    return self.storage

            groups = [
                (Tensor(0x10000, 3), [{"name": "parameter:a"}]),
                (Tensor(0x20000, 2), [{"name": "parameter:b"}]),
            ]
            calls = []

            class Transport:
                @staticmethod
                def capture(path, extents, **options):
                    calls.append((extents, options))
                    payload = b"abc" + bytes(4093) + b"de"
                    Path(path).write_bytes(payload)
                    return SimpleNamespace(
                        metrics=SimpleNamespace(
                            backend="buffered", bytes=5, file_bytes=len(payload)
                        ),
                        digests=(
                            SimpleNamespace(crc32="352441c2", sha256=hashlib.sha256(b"abc").hexdigest()),
                            SimpleNamespace(crc32="7d90298b", sha256=hashlib.sha256(b"de").hexdigest()),
                        ),
                        file_sha256=hashlib.sha256(payload).hexdigest(),
                    )

            fake_torch = types.ModuleType("torch")
            with (
                patch.dict(sys.modules, {"torch": fake_torch}),
                patch.object(core_artifact, "_storage_groups", return_value=groups),
                patch.object(
                    core_artifact, "capture_from_env", return_value=(Transport(), "auto")
                ),
            ):
                metrics = artifact.capture(object(), identity={"engine": "sglang"})
            manifest = artifact.load_manifest({"engine": "sglang"})

        self.assertEqual(metrics.backend, "buffered")
        self.assertEqual(metrics.bytes, 5)
        self.assertEqual(manifest["blob_bytes"], 4098)
        self.assertEqual([entry["offset"] for entry in manifest["storages"]], [0, 4096])
        self.assertEqual([extent.source for extent in calls[0][0]], [0x10000, 0x20000])
        self.assertTrue(calls[0][1]["file_sha256"])

    def test_nonpersistent_runtime_buffers_are_not_native_weight_semantics(self) -> None:
        tensor = SimpleNamespace(
            device=SimpleNamespace(type="cuda"),
            numel=lambda: 1,
        )
        module = SimpleNamespace(_non_persistent_buffers_set={"derived_cache"})
        model = SimpleNamespace(
            named_modules=lambda **_kwargs: [("", module)],
            named_parameters=lambda **_kwargs: [("weight", tensor)],
            named_buffers=lambda **_kwargs: [
                ("persistent_scale", tensor),
                ("derived_cache", tensor),
            ],
        )

        self.assertEqual(
            set(core_artifact._registered_cuda_tensors(model)),
            {"parameter:weight", "buffer:persistent_scale"},
        )

    def test_manifest_identity_and_blob_size_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = SemanticTensorArtifact(directory)
            artifact.root.mkdir(parents=True, exist_ok=True)
            artifact.blob_path.write_bytes(b"weight-bytes")
            artifact.manifest_path.write_text(
                json.dumps(
                    {
                        "format": ARTIFACT_FORMAT,
                        "kind": ARTIFACT_KIND,
                        "identity": {"engine": "sglang", "tp": 1},
                        "blob_bytes": 12,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                artifact.load_manifest({"engine": "sglang", "tp": 1})["blob_bytes"],
                12,
            )
            with self.assertRaisesRegex(ArtifactError, "identity.tp"):
                artifact.load_manifest({"engine": "sglang", "tp": 2})
            artifact.blob_path.write_bytes(b"short")
            with self.assertRaisesRegex(ArtifactError, "size mismatch"):
                artifact.load_manifest()

    def test_preverified_restore_requires_a_current_stat_bound_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = SemanticTensorArtifact(directory)
            artifact.root.mkdir(parents=True, exist_ok=True)
            artifact.blob_path.write_bytes(b"weight-bytes")
            digest = __import__("hashlib").sha256(b"weight-bytes").hexdigest()
            manifest = {"blob_sha256": digest}
            marker = artifact.blob_path.with_name(
                artifact.blob_path.name + ".coldsnap-verified.json"
            )
            publish_validation_record(
                artifact.blob_path,
                digest,
                artifact.blob_path.stat().st_size,
                record_path=marker,
            )
            self.assertEqual(artifact.verify_preverified(manifest)["sha256"], digest)
            artifact.blob_path.write_bytes(b"changed-data")
            with self.assertRaisesRegex(ArtifactError, "SHA-256 mismatch"):
                artifact.verify_preverified(manifest)

    def test_external_lock_does_not_require_writable_artifact_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "read-only-artifact"
            locks = Path(directory) / "runtime-locks"
            artifact = SemanticTensorArtifact(root, lock_root=locks)
            descriptor = artifact._lock(exclusive=False)
            artifact._unlock(descriptor)
            self.assertFalse(root.exists())
            self.assertTrue(artifact.lock_path.is_file())


class IdentityNormalizationTest(unittest.TestCase):
    def test_nested_hf_config_is_normalized_without_repr(self) -> None:
        class NestedConfig:
            def to_dict(self):
                return {"hidden_size": 2048}

            def __repr__(self):
                raise AssertionError("identity normalization must not call repr")

        class RootConfig:
            def to_dict(self):
                return {"text_config": NestedConfig(), "architectures": ["NewModel"]}

        model_config = SimpleNamespace(hf_config=RootConfig())
        first = sglang_identity._config_digest(model_config)
        second = sglang_identity._config_digest(model_config)
        self.assertEqual(first, second)

    def test_artifact_path_uses_worker_identity_not_tensor_rank(self) -> None:
        base = {"parallel": {"tp_rank": 0, "worker_id": "prefill-worker-0"}}
        other = {"parallel": {"tp_rank": 0, "worker_id": "decode-worker-0"}}
        with tempfile.TemporaryDirectory() as directory:
            first = sglang_identity.artifact_for(Path(directory), base)
            second = sglang_identity.artifact_for(Path(directory), other)
        self.assertIn("workers/prefill-worker-0", first.root.as_posix())
        self.assertIn("workers/decode-worker-0", second.root.as_posix())
        self.assertNotEqual(first.root, second.root)

    def test_gpu_capacity_ignores_one_page_reporting_drift(self) -> None:
        self.assertEqual(
            _capacity_class(130_663_170_048),
            _capacity_class(130_663_165_952),
        )

    def test_driver_identity_tracks_cuda_abi_major_not_patch_release(self) -> None:
        self.assertEqual(_driver_abi_major(13030), _driver_abi_major(13010))
        self.assertNotEqual(_driver_abi_major(13030), _driver_abi_major(12080))

    def test_process_admission_is_capability_based_not_architecture_allowlisted(self) -> None:
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["NewModelForCausalLM"]),
            is_generation=True,
        )
        sglang_identity.validate_model_config(model_config)

        model_config.hf_config.architectures = []
        sglang_identity.validate_model_config(model_config)

        model_config.is_generation = False
        with self.assertRaisesRegex(ArtifactError, "generation model"):
            sglang_identity.validate_model_config(model_config)


class FakeSaver:
    def __init__(self) -> None:
        self.events = []

    @contextmanager
    def region(
        self,
        *,
        tag,
        enable_cpu_backup=False,
        enable_disk_backup=False,
    ):
        self.events.append(
            (
                "region",
                tag,
                enable_cpu_backup,
                enable_disk_backup,
                os.environ.get("LD_PRELOAD"),
            )
        )
        yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        self.events.append(("cuda_graph", kwargs))
        yield

    @contextmanager
    def disable(self):
        yield

    def set_disk_backup_dir(self, path):
        self.events.append(("disk_dir", path))

    def pause(self, *, tag):
        self.events.append(("pause", tag))

    def resume(self, *, tag):
        self.events.append(("resume", tag))


class AdapterTest(unittest.TestCase):
    def test_subprocess_preload_composes_coldsnap_and_memory_saver(self) -> None:
        @contextmanager
        def configure_subprocess():
            previous = os.environ.get("LD_PRELOAD")
            os.environ["LD_PRELOAD"] = "/opt/tms/torch_memory_saver_hook_mode_preload.so"
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop("LD_PRELOAD", None)
                else:
                    os.environ["LD_PRELOAD"] = previous

        module = SimpleNamespace(
            torch_memory_saver=FakeSaver(),
            configure_subprocess=configure_subprocess,
        )
        settings = Settings(
            mode="capture",
            artifact_root=Path("/artifact"),
            live_backing="discard",
            live_root=Path("/runtime/live"),
            preverified=False,
        )
        inherited = "/opt/coldsnap/bridge.so:/opt/coldsnap/shim.so:/opt/nccl.so"
        with (
            patch.dict(os.environ, {"LD_PRELOAD": inherited}, clear=True),
            patch.dict(sys.modules, {"torch_memory_saver": module}),
        ):
            adapter = ColdSnapMemorySaverAdapter(settings)
            with adapter.configure_subprocess():
                self.assertEqual(
                    os.environ["LD_PRELOAD"],
                    inherited + ":/opt/tms/torch_memory_saver_hook_mode_preload.so",
                )
                self.assertEqual(
                    os.environ["COLDSNAP_SGLANG_TMS_PRELOAD"],
                    "/opt/tms/torch_memory_saver_hook_mode_preload.so",
                )
            self.assertEqual(os.environ["LD_PRELOAD"], inherited)
            self.assertNotIn("COLDSNAP_SGLANG_TMS_PRELOAD", os.environ)

    def test_sglang_tags_map_to_preserve_discard_and_graph_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            saver = FakeSaver()

            @contextmanager
            def configure_subprocess():
                yield

            module = SimpleNamespace(
                torch_memory_saver=saver,
                configure_subprocess=configure_subprocess,
            )
            settings = Settings(
                mode="capture",
                artifact_root=Path(directory),
                live_backing="disk",
                live_root=Path(directory) / "live",
                preverified=False,
            )
            preload = "/opt/tms/torch_memory_saver_hook_mode_preload.so"
            with (
                patch.dict(
                    os.environ,
                    {
                        "LD_PRELOAD": "/opt/coldsnap/shim.so:" + preload,
                        "COLDSNAP_SGLANG_TMS_PRELOAD": preload,
                    },
                    clear=True,
                ),
                patch.dict(sys.modules, {"torch_memory_saver": module}),
            ):
                adapter = ColdSnapMemorySaverAdapter(settings)
                with adapter.region("weights"):
                    pass
                with adapter.region("kv_cache"):
                    pass
                with adapter.cuda_graph(cuda_graph="graph"):
                    pass
                adapter.pause("weights")
                adapter.resume("kv_cache")

            self.assertIn(("region", "weights", False, True, preload), saver.events)
            self.assertIn(("region", "kv_cache", False, False, preload), saver.events)
            graph = next(event[1] for event in saver.events if event[0] == "cuda_graph")
            self.assertEqual(graph["tag"], "cuda_graph")
            self.assertFalse(graph["enable_cpu_backup"])
            self.assertIn(("pause", "weights"), saver.events)
            self.assertIn(("resume", "kv_cache"), saver.events)


class LifecycleHookTest(unittest.TestCase):
    def test_tokenizer_closes_health_before_distributed_release(self) -> None:
        class ServerStatus:
            Starting = "starting"
            Up = "up"

        tokenizer_module = types.ModuleType("sglang.srt.managers.tokenizer_manager")
        tokenizer_module.ServerStatus = ServerStatus
        manager = SimpleNamespace(server_status=ServerStatus.Up)
        events = []

        async def release(_manager, _request):
            events.append(("release", manager.server_status))
            return "released"

        with patch.dict(
            sys.modules,
            {"sglang.srt.managers.tokenizer_manager": tokenizer_module},
        ):
            self.assertEqual(
                asyncio.run(
                    sglang_plugin._gate_tokenizer_release(
                        release, manager, SimpleNamespace(tags=["weights"])
                    )
                ),
                "released",
            )

        self.assertEqual(manager.server_status, ServerStatus.Starting)
        self.assertEqual(events, [("release", ServerStatus.Starting)])

    def test_recovery_tokenizer_stays_unready_until_runtime_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                mode="capture",
                artifact_root=root / "semantic",
                live_backing="discard",
                live_root=root / "live",
                preverified=True,
                process_artifact_root=root,
            )

            class ServerStatus:
                Starting = "starting"
                Up = "up"

            tokenizer_module = types.ModuleType("sglang.srt.managers.tokenizer_manager")
            tokenizer_module.ServerStatus = ServerStatus
            manager = SimpleNamespace(server_status=ServerStatus.Up)
            events = []

            async def resume(_manager, _request):
                events.append(("resume", manager.server_status))
                return "resumed"

            with (
                patch.object(sglang_plugin, "_settings", settings),
                patch.dict(
                    sys.modules,
                    {"sglang.srt.managers.tokenizer_manager": tokenizer_module},
                ),
            ):
                self.assertEqual(
                    asyncio.run(
                        sglang_plugin._gate_tokenizer_resume(
                            resume,
                            manager,
                            SimpleNamespace(
                                tags=[
                                    "weights",
                                    sglang_plugin._RECOVERY_WEIGHTS_RESUME_TAG,
                                ]
                            ),
                        )
                    ),
                    "resumed",
                )
                self.assertEqual(manager.server_status, ServerStatus.Starting)
                self.assertEqual(
                    asyncio.run(
                        sglang_plugin._gate_tokenizer_resume(
                            resume,
                            manager,
                            SimpleNamespace(
                                tags=[
                                    "kv_cache",
                                    "cuda_graph",
                                    sglang_plugin._RECOVERY_RUNTIME_RESUME_TAG,
                                ]
                            ),
                        )
                    ),
                    "resumed",
                )
                self.assertEqual(manager.server_status, ServerStatus.Up)
            self.assertEqual(
                events,
                [
                    ("resume", ServerStatus.Starting),
                    ("resume", ServerStatus.Starting),
                ],
            )

    def test_recovery_hydrates_inside_distributed_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                mode="capture",
                artifact_root=root / "semantic",
                live_backing="discard",
                live_root=root / "live",
                preverified=True,
                process_artifact_root=root,
            )
            events = []
            response = SimpleNamespace(success=True, message="updated")
            updater = SimpleNamespace(
                update_weights_from_disk=lambda request: events.append(
                    ("hydrate", request.model_path, request.load_format)
                )
                or response
            )
            io_struct = types.ModuleType("sglang.srt.managers.io_struct")
            io_struct.UpdateWeightFromDiskReqInput = lambda **kwargs: SimpleNamespace(
                **kwargs
            )
            checkpoint = types.ModuleType("coldsnap_nccl_checkpoint")
            checkpoint._apply_restore_transport_environment = lambda: None
            runtime = SimpleNamespace(restore=lambda: {"state": "restored"})

            @contextmanager
            def allocator_context():
                yield

            with (
                patch.object(sglang_plugin, "_settings", settings),
                patch.object(sglang_plugin, "_nccl_runtime", return_value=runtime),
                patch.object(
                    sglang_plugin,
                    "_graph_recapture_allocator_context",
                    side_effect=allocator_context,
                ),
                patch.object(sglang_plugin, "_write_state"),
                patch.object(sglang_plugin, "_synchronize_tp_lifecycle"),
                patch.object(sglang_plugin, "_recapture_cuda_graphs"),
                patch.dict(
                    sys.modules,
                    {
                        "coldsnap_nccl_checkpoint": checkpoint,
                        "sglang.srt.managers.io_struct": io_struct,
                    },
                ),
                patch.dict(
                    os.environ,
                    {
                        "COLDSNAP_MODEL_ID": "Qwen/model",
                        "COLDSNAP_SGLANG_RECOVERY_LOAD_FORMAT": "safetensors",
                    },
                    clear=True,
                ),
            ):
                sglang_plugin._restored_weight_update_pending = False
                sglang_plugin._recovery_resume_observation = None
                self.assertEqual(
                    sglang_plugin._resume_memory(
                        lambda _updater, _request: events.append(("resume",)) or "ok",
                        updater,
                        SimpleNamespace(
                            tags=[
                                "weights",
                                sglang_plugin._RECOVERY_WEIGHTS_RESUME_TAG,
                            ]
                        ),
                    ),
                    "ok",
                )
                self.assertEqual(
                    sglang_plugin._resume_memory(
                        lambda _updater, _request: events.append(("resume-runtime",))
                        or "runtime-ok",
                        updater,
                        SimpleNamespace(
                            tags=[
                                "kv_cache",
                                "cuda_graph",
                                sglang_plugin._RECOVERY_RUNTIME_RESUME_TAG,
                            ]
                        ),
                    ),
                    "runtime-ok",
                )

            self.assertEqual(
                events,
                [
                    ("resume",),
                    ("hydrate", "Qwen/model", "safetensors"),
                    ("resume-runtime",),
                ],
            )
            self.assertIsNone(sglang_plugin._recovery_resume_observation)

    def test_native_tokenizer_lifecycle_is_not_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                mode="capture",
                artifact_root=root / "semantic",
                live_backing="discard",
                live_root=root / "live",
                preverified=True,
                process_artifact_root=root,
            )
            manager = SimpleNamespace(server_status="up")
            tokenizer_module = types.ModuleType("sglang.srt.managers.tokenizer_manager")
            tokenizer_module.ServerStatus = SimpleNamespace(Starting="starting", Up="up")

            async def original(_manager, _request):
                return "native"

            with (
                patch.object(sglang_plugin, "_settings", settings),
                patch.dict(
                    sys.modules,
                    {"sglang.srt.managers.tokenizer_manager": tokenizer_module},
                ),
            ):
                self.assertEqual(
                    asyncio.run(
                        sglang_plugin._gate_tokenizer_resume(original, manager, object())
                    ),
                    "native",
                )
            self.assertEqual(manager.server_status, "up")

    def test_native_resume_reopens_health_closed_at_capture(self) -> None:
        class ServerStatus:
            Starting = "starting"
            Up = "up"

        tokenizer_module = types.ModuleType("sglang.srt.managers.tokenizer_manager")
        tokenizer_module.ServerStatus = ServerStatus
        manager = SimpleNamespace(server_status=ServerStatus.Starting)

        async def original(_manager, _request):
            self.assertEqual(manager.server_status, ServerStatus.Starting)
            return "native"

        with patch.dict(
            sys.modules,
            {"sglang.srt.managers.tokenizer_manager": tokenizer_module},
        ):
            self.assertEqual(
                asyncio.run(
                    sglang_plugin._gate_tokenizer_resume(
                        original,
                        manager,
                        SimpleNamespace(tags=["kv_cache", "weights", "cuda_graph"]),
                    )
                ),
                "native",
            )
        self.assertEqual(manager.server_status, ServerStatus.Up)

    def test_weight_lifecycle_owner_is_resolved_by_capability(self) -> None:
        module = types.ModuleType("fake_sglang.weight_updater")

        class RenamedManager:
            def release_memory_occupation(self, recv_req):
                pass

            def resume_memory_occupation(self, recv_req):
                pass

            def update_weights_from_disk(self, recv_req):
                pass

        RenamedManager.__module__ = module.__name__
        RenamedManager.__qualname__ = "RenamedManager"
        module.RenamedManager = RenamedManager
        resolved = resolve_weight_updater_contract(module)
        self.assertIs(resolved.owner, RenamedManager)
        self.assertEqual(
            resolved.hook_prefix,
            "fake_sglang.weight_updater.RenamedManager",
        )

        module.OtherManager = type(
            "OtherManager",
            (),
            {
                "__module__": module.__name__,
                "release_memory_occupation": lambda self, recv_req: None,
                "resume_memory_occupation": lambda self, recv_req: None,
                "update_weights_from_disk": lambda self, recv_req: None,
            },
        )
        with self.assertRaisesRegex(SGLangCompatibilityError, "exactly one"):
            resolve_weight_updater_contract(module)

    def test_graphs_and_nccl_are_quiesced_after_sglang_pauses_memory(self) -> None:
        events = []
        runtime = SimpleNamespace(
            prepare=lambda: events.append("nccl-prepare") or {"state": "prepared"}
        )

        def original(_updater, _request):
            events.append("memory-release")
            return "released"

        with (
            patch.object(
                sglang_plugin,
                "_capture_checkpoint_optional_parameters",
                side_effect=lambda: events.append("optional-state-capture"),
            ),
            patch.object(
                sglang_plugin,
                "_export_registered_model_payloads",
                side_effect=lambda: events.append("payload-export"),
            ),
            patch.object(
                sglang_plugin,
                "_discard_cuda_graphs",
                side_effect=lambda _updater: events.append("graph-discard"),
            ),
            patch.object(sglang_plugin, "_nccl_runtime", return_value=runtime),
            patch.object(
                sglang_plugin,
                "_write_state",
                side_effect=lambda state, **_kwargs: events.append("state-" + state),
            ),
            patch.object(
                sglang_plugin,
                "_synchronize_tp_lifecycle",
                side_effect=lambda _updater: events.append("tp-barrier"),
            ),
            patch("coldsnap_core.topology.worker_id", return_value="worker-0"),
        ):
            result = sglang_plugin._release_memory(
                original,
                object(),
                SimpleNamespace(tags=["weights"]),
            )

        self.assertEqual(result, "released")
        self.assertEqual(
            events,
            [
                "optional-state-capture",
                "payload-export",
                "memory-release",
                "graph-discard",
                "nccl-prepare",
                "state-sleeping",
                "tp-barrier",
            ],
        )

    def test_exact_nccl_capture_retains_sglang_graph_executables(self) -> None:
        events = []
        runtime = SimpleNamespace(
            prepare=lambda: events.append("nccl-prepare") or {"state": "prepared"}
        )

        with (
            patch.object(sglang_plugin, "_capture_checkpoint_optional_parameters"),
            patch.object(sglang_plugin, "_export_registered_model_payloads"),
            patch.object(sglang_plugin, "_discard_cuda_graphs") as discard,
            patch.object(sglang_plugin, "_nccl_runtime", return_value=runtime),
            patch.object(sglang_plugin, "_write_state"),
            patch.object(sglang_plugin, "_synchronize_tp_lifecycle"),
            patch.object(sglang_plugin, "_wait_at_cuda_restore_hold"),
            patch("coldsnap_core.topology.worker_id", return_value="worker-0"),
            patch.dict(
                os.environ,
                {"COLDSNAP_GRAPH_POLICY": "preserve-nccl-exec"},
                clear=False,
            ),
        ):
            result = sglang_plugin._release_memory(
                lambda _updater, _request: "released",
                object(),
                SimpleNamespace(tags=["kv_cache", "weights", "cuda_graph"]),
            )

        self.assertEqual(result, "released")
        self.assertEqual(events, ["nccl-prepare"])
        discard.assert_not_called()

    def test_exact_nccl_graph_fallback_switches_sglang_to_eager(self) -> None:
        events = []
        request = SimpleNamespace(tags=["cuda_graph"])
        settings = Settings(
            mode="restore",
            artifact_root=Path("/artifact"),
            live_backing="discard",
            live_root=Path("/runtime/live"),
            preverified=True,
            async_graphs=True,
        )
        with (
            patch.object(sglang_plugin, "_settings", settings),
            patch.object(
                sglang_plugin,
                "_discard_cuda_graphs",
                side_effect=lambda _updater: (
                    events.append("graph-discard"),
                    sglang_plugin._graph_recapture_plan.append(object()),
                ),
            ),
            patch.object(
                sglang_plugin,
                "_publish_async_graph_status",
                side_effect=lambda phase: events.append("status-" + phase),
            ),
            patch.dict(
                os.environ,
                {"COLDSNAP_GRAPH_POLICY": "preserve-nccl-exec"},
                clear=False,
            ),
        ):
            sglang_plugin._graph_recapture_plan = []
            sglang_plugin._async_graph_phase = "ready"
            self.assertEqual(
                sglang_plugin._release_memory(
                    lambda _updater, _request: events.append("release") or "released",
                    object(),
                    request,
                ),
                "released",
            )
            self.assertEqual(
                sglang_plugin._resume_memory(
                    lambda _updater, _request: events.append("resume") or "resumed",
                    object(),
                    request,
                ),
                "resumed",
            )

        self.assertEqual(
            events,
            ["release", "graph-discard", "resume", "status-eager"],
        )
        self.assertEqual(sglang_plugin._async_graph_phase, "eager")
        sglang_plugin._graph_recapture_plan = []

    def test_checkpoint_optional_parameters_survive_weight_discard(self) -> None:
        class FakeTensor:
            def __init__(self, value, *, optional=False, device="cuda:0"):
                self.value = value
                self._skip_weight_check = optional
                self.shape = (1,)
                self.dtype = "float32"
                self.device = device

            def detach(self):
                return self

            def cpu(self):
                return FakeTensor(
                    self.value,
                    optional=self._skip_weight_check,
                    device="cpu",
                )

            def clone(self):
                return FakeTensor(
                    self.value,
                    optional=self._skip_weight_check,
                    device=self.device,
                )

            def numel(self):
                return 1

            def element_size(self):
                return 4

            def to(self, *, device):
                return FakeTensor(
                    self.value,
                    optional=self._skip_weight_check,
                    device=device,
                )

            def copy_(self, other):
                self.value = other.value

        optional = FakeTensor(1.0, optional=True)
        checkpointed = FakeTensor(7.0)

        class Model:
            def named_parameters(self, **_kwargs):
                return [("k_scale", optional), ("weight", checkpointed)]

        model = Model()
        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = contextmanager(lambda: (yield))
        with (
            patch.object(
                sglang_plugin,
                "_pending_models",
                [(model, {"engine": "sglang"}, object())],
            ),
            patch.object(sglang_plugin, "_checkpoint_optional_parameters", []),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            sglang_plugin._capture_checkpoint_optional_parameters()
            optional.value = 0.0
            checkpointed.value = 0.0
            sglang_plugin._restore_checkpoint_optional_parameters()

        self.assertEqual(optional.value, 1.0)
        self.assertEqual(checkpointed.value, 0.0)

    def test_model_payload_export_is_deferred_until_release(self) -> None:
        context = SimpleNamespace(identity={"engine": "sglang"}, artifact=object())
        model = object()
        with (
            patch.object(sglang_plugin, "_context", return_value=context),
            patch.object(sglang_plugin, "_register_model") as register,
            patch.object(sglang_plugin, "_export_model_payload") as export,
        ):
            self.assertIs(
                sglang_plugin._model_load(
                    lambda _loader, **_kwargs: model,
                    object(),
                    model_config=object(),
                ),
                model,
            )

        register.assert_called_once_with(context, model)
        export.assert_not_called()

    def test_restored_weight_resume_arms_allocator_preservation_for_hydration(self) -> None:
        events = []
        runtime = SimpleNamespace(
            restore=lambda: events.append("nccl-restore") or {"state": "restored"}
        )

        def original(_updater, _request):
            events.append("memory-resume")
            return "resumed"

        checkpoint = types.ModuleType("coldsnap_nccl_checkpoint")
        checkpoint._apply_restore_transport_environment = lambda: None
        with (
            patch.dict(sys.modules, {"coldsnap_nccl_checkpoint": checkpoint}),
            patch.object(sglang_plugin, "_nccl_runtime", return_value=runtime),
            patch.object(sglang_plugin, "_recapture_cuda_graphs") as recapture,
            patch.object(sglang_plugin, "_write_state") as write_state,
            patch.object(sglang_plugin, "_synchronize_tp_lifecycle") as barrier,
        ):
            sglang_plugin._restored_weight_update_pending = False
            result = sglang_plugin._resume_memory(
                original,
                object(),
                SimpleNamespace(tags=["weights"]),
            )

        self.assertEqual(result, "resumed")
        self.assertEqual(events, ["nccl-restore", "memory-resume"])
        self.assertTrue(sglang_plugin._restored_weight_update_pending)
        recapture.assert_called_once_with()
        write_state.assert_called_once()
        barrier.assert_called_once()

    def test_armed_weight_update_preserves_restored_allocator_once(self) -> None:
        events = []

        @contextmanager
        def allocator_context():
            events.append("allocator-enter")
            try:
                yield
            finally:
                events.append("allocator-exit")

        def original(_updater, _request):
            events.append("weight-update")
            return "updated"

        with patch.object(
            sglang_plugin,
            "_graph_recapture_allocator_context",
            side_effect=allocator_context,
        ):
            sglang_plugin._restored_weight_update_pending = True
            self.assertEqual(
                sglang_plugin._update_weights_after_restore(
                    original,
                    object(),
                    object(),
                ),
                "updated",
            )
            self.assertFalse(sglang_plugin._restored_weight_update_pending)
            self.assertEqual(
                sglang_plugin._update_weights_after_restore(
                    original,
                    object(),
                    object(),
                ),
                "updated",
            )

        self.assertEqual(
            events,
            ["allocator-enter", "weight-update", "allocator-exit", "weight-update"],
        )

    def test_dummy_loader_model_is_hydrated_before_startup_returns(self) -> None:
        context = SimpleNamespace(identity={"engine": "sglang"}, artifact=object())
        model = object()
        adapter = SimpleNamespace(restore_startup_model=Mock())
        settings = Settings(
            mode="capture",
            artifact_root=Path("/artifact"),
            live_backing="discard",
            live_root=Path("/runtime/live"),
            preverified=True,
            startup_provider="native",
        )
        with (
            patch.object(sglang_plugin, "_settings", settings),
            patch.object(sglang_plugin, "_memory_adapter", adapter),
            patch.object(sglang_plugin, "_context", return_value=context),
            patch.object(sglang_plugin, "_register_model") as register,
        ):
            self.assertIs(
                sglang_plugin._model_load(
                    lambda _loader, **_kwargs: model,
                    object(),
                    model_config=object(),
                ),
                model,
            )

        register.assert_called_once_with(context, model)
        adapter.restore_startup_model.assert_called_once_with(
            model, context.identity, context.artifact
        )

    def test_graph_lifecycle_is_capability_based_across_target_and_draft(self) -> None:
        events = []
        cleanup_events = []
        cuda_calls = []

        def original_torch_empty_cache():
            cuda_calls.append("original-empty-cache")

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda: cuda_calls.append("synchronize"),
                empty_cache=original_torch_empty_cache,
            )
        )

        class Backend:
            def __init__(self, name):
                self.name = name

            def cleanup(self):
                cleanup_events.append("cleanup-" + self.name)

        class Graph:
            def __init__(self, name):
                self.backend = Backend(name)

        class Runner:
            def __init__(self, name):
                self.name = name
                self.server_args = SimpleNamespace(
                    cuda_graph_config=SimpleNamespace(
                        decode=SimpleNamespace(backend="full"),
                        prefill=SimpleNamespace(backend="disabled"),
                    )
                )
                self.decode_cuda_graph_runner = Graph(name)
                self.prefill_cuda_graph_runner = object()

            def init_decode_cuda_graph(self):
                fake_torch.cuda.empty_cache()
                events.append("recapture-" + self.name)
                self.decode_cuda_graph_runner = object()

            def init_prefill_cuda_graph(self):
                raise AssertionError("disabled prefill graph must not be recaptured")

        target = Runner("target")
        draft = Runner("draft")
        updater = SimpleNamespace(
            tp_worker=SimpleNamespace(model_runner=target),
            draft_worker=SimpleNamespace(draft_runners=[draft]),
        )
        common = types.ModuleType("sglang.srt.utils.common")

        def original_empty_device_cache(device_module=None):
            del device_module
            cuda_calls.append("original-empty-cache")
            return True

        common.empty_device_cache = original_empty_device_cache
        utils = types.ModuleType("sglang.srt.utils")
        utils.common = common
        with (
            patch("gc.collect"),
            patch.object(
                sglang_plugin,
                "_reset_cuda_graph_pool",
                side_effect=lambda: cuda_calls.append("reset-graph-pool"),
            ),
            patch.dict(
                sys.modules,
                {
                    "sglang": types.ModuleType("sglang"),
                    "sglang.srt": types.ModuleType("sglang.srt"),
                    "sglang.srt.utils": utils,
                    "sglang.srt.utils.common": common,
                    "torch": fake_torch,
                },
            ),
        ):
            sglang_plugin._graph_recapture_plan = []
            sglang_plugin._discard_cuda_graphs(updater)
            self.assertIsNone(target.decode_cuda_graph_runner)
            self.assertIsNone(draft.decode_cuda_graph_runner)
            self.assertIsNotNone(target.prefill_cuda_graph_runner)
            self.assertIsNotNone(draft.prefill_cuda_graph_runner)
            sglang_plugin._recapture_cuda_graphs()
            self.assertIs(common.empty_device_cache, original_empty_device_cache)
            self.assertIs(fake_torch.cuda.empty_cache, original_torch_empty_cache)

        self.assertEqual(events, ["recapture-target", "recapture-draft"])
        self.assertEqual(cleanup_events, ["cleanup-target", "cleanup-draft"])
        self.assertEqual(
            cuda_calls,
            [
                "reset-graph-pool",
                "synchronize",
            ],
        )
        self.assertEqual(sglang_plugin._graph_recapture_plan, [])

    def test_restored_target_starts_eager_then_captures_graphs_when_idle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = root / "graphs.arm"
            ready = root / "graphs.json"
            arm.touch()
            settings = Settings(
                mode="capture",
                artifact_root=root / "semantic",
                live_backing="discard",
                live_root=root / "live",
                preverified=True,
                async_graphs=True,
                async_graph_arm_file=arm,
                async_graph_ready_file=ready,
                startup_provider="recovery",
            )
            runner = SimpleNamespace(
                server_args=SimpleNamespace(
                    speculative_algorithm="NONE",
                    cuda_graph_config=SimpleNamespace(
                        decode=SimpleNamespace(backend="full"),
                        prefill=SimpleNamespace(backend="disabled"),
                    ),
                ),
                decode_cuda_graph_runner=object(),
            )
            captures = []

            def initialize(owner, *, capture_decode_cuda_graph=True):
                captures.append(f"startup:{capture_decode_cuda_graph}")
                owner.decode_cuda_graph_runner = "graph" if capture_decode_cuda_graph else "eager"

            def capture_decode():
                captures.append("decode")
                runner.decode_cuda_graph_runner = "graph"

            runner.init_decode_cuda_graph = capture_decode

            scheduler = SimpleNamespace(
                draft_worker=None,
                server_args=SimpleNamespace(speculative_algorithm="NONE"),
                is_fully_idle=lambda: True,
            )

            def recapture():
                plan = list(sglang_plugin._graph_recapture_plan)
                for step in plan:
                    step.callback()
                sglang_plugin._graph_recapture_plan = []

            with (
                patch.object(sglang_plugin, "_settings", settings),
                patch.object(
                    sglang_plugin,
                    "_synchronize_tp_lifecycle",
                ) as barrier,
                patch.object(sglang_plugin, "_recapture_cuda_graphs", side_effect=recapture),
            ):
                sglang_plugin._graph_recapture_plan = []
                sglang_plugin._async_graph_eager_steps = 0
                sglang_plugin._async_graph_phase = "disabled"
                sglang_plugin._defer_startup_cuda_graphs(initialize, runner)
                self.assertIsNone(runner.decode_cuda_graph_runner)
                self.assertEqual(sglang_plugin._async_graph_phase, "eager")

                self.assertEqual(
                    sglang_plugin._record_async_graph_eager_step(
                        lambda _scheduler: "published", scheduler
                    ),
                    "published",
                )
                sglang_plugin._capture_async_graphs_when_idle(scheduler)

            self.assertEqual(captures, ["startup:False", "decode"])
            self.assertEqual(barrier.call_count, 2)
            self.assertEqual(sglang_plugin._async_graph_phase, "ready")
            self.assertEqual(sglang_plugin._graph_recapture_plan, [])
            self.assertEqual(
                json.loads(ready.read_text(encoding="utf-8"))["phase"],
                "ready",
            )

    def test_speculative_startup_retains_synchronous_graph_initialization(self) -> None:
        settings = Settings(
            mode="capture",
            artifact_root=Path("/artifact"),
            live_backing="discard",
            live_root=Path("/runtime/live"),
            preverified=True,
            async_graphs=True,
            async_graph_arm_file=Path("/artifact/graphs.arm"),
            async_graph_ready_file=Path("/artifact/graphs.json"),
            startup_provider="recovery",
        )
        runner = SimpleNamespace(
            server_args=SimpleNamespace(
                speculative_algorithm="DSPARK",
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend="full"),
                    prefill=SimpleNamespace(backend="disabled"),
                ),
            ),
            decode_cuda_graph_runner=None,
        )

        def capture(owner, *, capture_decode_cuda_graph=True):
            owner.decode_cuda_graph_runner = "captured"
            return "synchronous"

        with patch.object(sglang_plugin, "_settings", settings):
            sglang_plugin._graph_recapture_plan = []
            self.assertEqual(
                sglang_plugin._defer_startup_cuda_graphs(capture, runner),
                "synchronous",
            )

        self.assertEqual(runner.decode_cuda_graph_runner, "captured")
        self.assertEqual(sglang_plugin._graph_recapture_plan, [])


class SGLangSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        value = os.environ.get("SGLANG_SOURCE_ROOT")
        if not value:
            raise unittest.SkipTest("set SGLANG_SOURCE_ROOT to an SGLang source checkout")
        cls.python = Path(value) / "python"

    def _function_parameters(self, relative: str, class_name: str, name: str):
        tree = ast.parse((self.python / relative).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == name
                    ):
                        return {
                            arg.arg
                            for arg in (
                                child.args.posonlyargs + child.args.args + child.args.kwonlyargs
                            )
                        } | ({child.args.kwarg.arg} if child.args.kwarg else set())
        self.fail(f"missing {class_name}.{name} in {relative}")

    def test_adapter_and_process_snapshot_contracts_exist(self) -> None:
        adapter = "sglang/srt/utils/torch_memory_saver_adapter.py"
        self.assertTrue(
            {"enable"} <= self._function_parameters(adapter, "TorchMemorySaverAdapter", "create")
        )
        self.assertTrue(
            {"tag", "enable_cpu_backup"}
            <= self._function_parameters(adapter, "TorchMemorySaverAdapter", "region")
        )
        loader = "sglang/srt/model_loader/loader.py"
        self.assertTrue(
            {"model_config", "device_config"}
            <= self._function_parameters(loader, "DefaultModelLoader", "load_model")
        )
        updater = self.python / "sglang/srt/managers/scheduler_components/weight_updater.py"
        tree = ast.parse(updater.read_text(encoding="utf-8"))
        lifecycle_owners = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                child.name: {
                    arg.arg
                    for arg in (child.args.posonlyargs + child.args.args + child.args.kwonlyargs)
                }
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            required = {
                "release_memory_occupation",
                "resume_memory_occupation",
                "update_weights_from_disk",
            }
            if required <= methods.keys():
                lifecycle_owners.append(methods)
        self.assertEqual(len(lifecycle_owners), 1)
        for name in (
            "release_memory_occupation",
            "resume_memory_occupation",
            "update_weights_from_disk",
        ):
            self.assertIn("recv_req", lifecycle_owners[0][name])
        for tokenizer, owner, name in (
            (
                "sglang/srt/managers/tokenizer_control_mixin.py",
                "TokenizerControlMixin",
                "resume_memory_occupation",
            ),
            (
                "sglang/srt/managers/tokenizer_manager.py",
                "TokenizerManager",
                "update_weights_from_disk",
            ),
        ):
            self.assertIn(
                "obj",
                self._function_parameters(tokenizer, owner, name),
            )
        draft = "sglang/srt/speculative/base_spec_worker.py"
        self.assertTrue(
            {"recv_req"}
            <= self._function_parameters(draft, "BaseSpecWorker", "update_weights_from_disk")
        )

    def test_plugin_hooks_exist(self) -> None:
        plugins = (self.python / "sglang/srt/plugins/__init__.py").read_text(encoding="utf-8")
        self.assertIn('GENERAL_PLUGINS_GROUP = "sglang.srt.plugins"', plugins)
        self.assertIn("HookRegistry.apply_hooks()", plugins)

    def test_semantic_memory_tags_are_unchanged(self) -> None:
        tree = ast.parse((self.python / "sglang/srt/constants.py").read_text(encoding="utf-8"))
        constants = {}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("GPU_MEMORY_TYPE_")
            ):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(constants["GPU_MEMORY_TYPE_WEIGHTS"], "weights")
        self.assertEqual(constants["GPU_MEMORY_TYPE_KV_CACHE"], "kv_cache")
        self.assertEqual(constants["GPU_MEMORY_TYPE_CUDA_GRAPH"], "cuda_graph")


if __name__ == "__main__":
    unittest.main()

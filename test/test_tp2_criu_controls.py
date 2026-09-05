# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT / "shared"))
sys.path.insert(0, str(RUNTIME_ROOT / "engine"))
MODULE_PATH = RUNTIME_ROOT / "engine" / "coldsnap_engine_rank_n610.py"
SPEC = importlib.util.spec_from_file_location("coldsnap_engine_rank_n610", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
node = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(node)

N580_MODULE_PATH = RUNTIME_ROOT / "engine" / "coldsnap_engine_rank_n580.py"
N580_SPEC = importlib.util.spec_from_file_location("coldsnap_engine_rank_n580", N580_MODULE_PATH)
assert N580_SPEC is not None and N580_SPEC.loader is not None
n580_node = importlib.util.module_from_spec(N580_SPEC)
N580_SPEC.loader.exec_module(n580_node)


class TP2CriuControlsTest(unittest.TestCase):
    def test_n580_requires_current_process_template_placement_abi(self) -> None:
        n580_node._require_process_template_placement_abi(
            {"process_template_placement_abi": (n580_node.PROCESS_TEMPLATE_PLACEMENT_ABI)}
        )
        with self.assertRaisesRegex(RuntimeError, "recapture"):
            n580_node._require_process_template_placement_abi({"process_template_placement_abi": 1})

    def test_n580_restored_process_reports_a_missing_root_as_exited(self) -> None:
        process = n580_node._RestoredProcess(2**31 - 1)
        self.assertEqual(process.poll(), -1)
        self.assertEqual(process.returncode, -1)

    def test_n580_restore_rewrites_tcp_endpoints_and_preserves_http_port(self) -> None:
        args = SimpleNamespace(
            artifact_root=Path("/opt/coldsnap/capsule"),
            criu=Path("/opt/coldsnap/criu/bin/criu"),
            criu_rpc=Path("/usr/local/bin/coldsnap-criu-rpc"),
            criu_plugin_dir=Path("/opt/coldsnap/criu/plugins"),
            timeout=1800.0,
            ghost_limit=64 * 1024**2,
            decompress_threads=1,
            image_io_mode="direct",
            tcp_address_map=[
                "10.24.11.14=192.168.1.43",
                "10.24.11.16=192.168.1.44",
            ],
            tcp_port_shift=4096,
            http_port=8000,
            defer_network_unlock=True,
            serve_command=["--", "vllm", "serve", "model", "--port", "8000"],
        )

        command = n580_node._criu_command(
            args, "restore", Path("/opt/coldsnap/capsule/restore-attempt")
        )

        self.assertEqual(command.count("--tcp-address-map"), 2)
        self.assertIn("10.24.11.14=192.168.1.43", command)
        self.assertIn("10.24.11.16=192.168.1.44", command)
        self.assertEqual(command[command.index("--tcp-port-shift") + 1], "4096")
        self.assertEqual(command[command.index("--tcp-preserve-port") + 1], "8000")
        self.assertIn("--leave-stopped", command)
        self.assertIn("--defer-network-unlock", command)
        self.assertIn("--tcp-allow-empty-map", command)

        args.http_port = 8103
        args.serve_command = ["--", "vllm", "serve", "model", "--port", "8103"]
        args.captured_http_port = 8000
        migrated = n580_node._criu_command(
            args, "restore", Path("/opt/coldsnap/capsule/restore-attempt")
        )
        self.assertEqual(migrated[migrated.index("--tcp-port-shift") + 1], "4096")
        self.assertNotIn("--tcp-preserve-port", migrated)
        self.assertEqual(migrated[migrated.index("--tcp-port-map") + 1], "8000=8103")

    def test_n580_process_template_placement_is_generation_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capture.json").write_text(
                json.dumps(
                    {
                        "process_template_placement_abi": 2,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                artifact_root=root,
                engine="vllm",
                generation="capture-id",
                master_address="192.168.1.43",
                http_port=8103,
                tcp_port_shift=103,
                tcp_address_map=["10.24.11.14=192.168.1.43"],
                serve_command=[
                    "vllm",
                    "serve",
                    "model",
                    "--master-port",
                    "25000",
                    "--port",
                    "8103",
                ],
            )
            self.assertTrue(n580_node._write_process_template_placement(args))
            placement = json.loads((root / "restore-placement.json").read_text())
            self.assertEqual(
                set(placement),
                {
                    "format",
                    "kind",
                    "generation",
                    "master_address",
                    "master_port",
                    "http_port",
                },
            )
            self.assertEqual(placement["master_address"], "192.168.1.43")
            self.assertEqual(placement["master_port"], 25103)
            self.assertEqual(placement["http_port"], 8103)

    def test_old_n580_artifact_rejects_changed_host_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capture.json").write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(
                artifact_root=root,
                engine="vllm",
                generation="capture-id",
                master_address="192.168.1.43",
                http_port=8000,
                tcp_port_shift=1,
                tcp_address_map=["10.24.11.14=192.168.1.43"],
                serve_command=["vllm", "serve", "model", "--port", "8000"],
            )
            with self.assertRaisesRegex(RuntimeError, "recapture"):
                n580_node._write_process_template_placement(args)

    def test_n580_pre_worker_placement_includes_lazy_transport_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capture.json").write_text(
                json.dumps(
                    {
                        "process_template_placement_abi": 2,
                        "process_template_phase": n580_node.PRE_WORKER_IMPORT_PHASE,
                        "captured_driver_libraries_qualified": True,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                artifact_root=root,
                engine="vllm",
                generation="capture-id",
                master_address="192.168.1.43",
                http_port=8103,
                tcp_port_shift=103,
                tcp_address_map=["10.24.11.14=192.168.1.43"],
                serve_command=[
                    "vllm",
                    "serve",
                    "model",
                    "--master-port",
                    "25000",
                    "--port",
                    "8103",
                ],
            )

            self.assertTrue(n580_node._write_process_template_placement(args))
            placement = json.loads((root / "restore-placement.json").read_text())
            self.assertEqual(
                placement["tcp_address_map"],
                {"10.24.11.14": "192.168.1.43"},
            )
            self.assertEqual(placement["tcp_port_shift"], 103)

    def test_n580_portable_pre_exec_boundary_accepts_newer_driver(self) -> None:
        captured = {"gpus": [{"driver": "580.173.02"}]}
        current = {"gpus": [{"driver": "610.1"}]}
        portability = n580_node._validate_nvidia_driver_library_portability(captured, current)
        self.assertEqual(portability, "pre-exec-current-driver")

    def test_n580_pre_worker_import_cross_driver_is_fail_closed(self) -> None:
        captured = {"gpus": [{"driver": "580.173.02"}]}
        current = {"gpus": [{"driver": "610.1"}]}
        with self.assertRaisesRegex(RuntimeError, "explicit.*qualification override"):
            n580_node._validate_nvidia_driver_library_portability(
                captured,
                current,
                n580_node.PRE_WORKER_IMPORT_PHASE,
            )
        portability = n580_node._validate_nvidia_driver_library_portability(
            captured,
            current,
            n580_node.PRE_WORKER_IMPORT_PHASE,
            True,
        )
        self.assertEqual(portability, "pre-worker-import-forced-cross-driver")

    def test_n580_pre_worker_import_exact_driver_is_admitted(self) -> None:
        identity = {"gpus": [{"driver": "580.173.02"}]}
        portability = n580_node._validate_nvidia_driver_library_portability(
            identity,
            identity,
            n580_node.PRE_WORKER_IMPORT_PHASE,
        )
        self.assertEqual(portability, "pre-worker-import-exact-driver")

    def test_n580_pre_worker_reexec_admits_target_driver_clients(self) -> None:
        captured = {"gpus": [{"driver": "580.173.02"}]}
        current = {"gpus": [{"driver": "610.1"}]}
        portability = n580_node._validate_nvidia_driver_library_portability(
            captured,
            current,
            n580_node.PRE_WORKER_IMPORT_PHASE,
            False,
            True,
        )
        self.assertEqual(portability, "pre-worker-import-portable-worker-reexec")

    def test_n580_pre_worker_import_artifact_requires_qualification_evidence(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "qualification evidence"):
            n580_node._capture_process_template_phase(
                {"process_template_phase": n580_node.PRE_WORKER_IMPORT_PHASE}
            )
        self.assertEqual(
            n580_node._capture_process_template_phase(
                {
                    "process_template_phase": n580_node.PRE_WORKER_IMPORT_PHASE,
                    "captured_driver_libraries_qualified": True,
                }
            ),
            n580_node.PRE_WORKER_IMPORT_PHASE,
        )

    def test_n580_qualification_phase_is_vllm_only(self) -> None:
        with mock.patch.dict(
            os.environ,
            {n580_node.QUALIFICATION_PHASE_ENV: (n580_node.PRE_WORKER_IMPORT_PHASE)},
            clear=True,
        ):
            self.assertEqual(
                n580_node._process_template_phase(SimpleNamespace(engine="vllm")),
                n580_node.PRE_WORKER_IMPORT_PHASE,
            )
            with self.assertRaisesRegex(RuntimeError, "vLLM qualification"):
                n580_node._process_template_phase(SimpleNamespace(engine="sglang"))

    def test_n580_driver_library_audit_override_does_not_hide_device_state(self) -> None:
        record = {
            "pid": 42,
            "device_maps": [],
            "accelerator_fds": [],
        }
        maps = "0000-1000 r-xp 0 00:00 0 /usr/lib/libcuda.so.580\n"
        with (
            mock.patch.object(n580_node.precuda, "_audit_process_tree", return_value=[record]),
            mock.patch.object(n580_node.Path, "read_text", return_value=maps),
            mock.patch.object(n580_node, "_primary_context_active", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "accelerator state"):
                n580_node._audit_tree([42])
            admitted = n580_node._audit_tree([42], allow_captured_driver_libraries=True)
        self.assertEqual(
            admitted[0]["nvidia_driver_library_maps"],
            ["/usr/lib/libcuda.so.580"],
        )

    def test_n580_qualification_driver_libraries_are_sealed_and_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            artifact_root = root / "artifact"
            destination_root = root / "destination"
            library = source_root / "usr/lib/libcuda.so.580"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"capture-driver-library")
            audit = [{"nvidia_driver_library_maps": [str(library)]}]

            # The real audit records absolute container paths. Keep that invariant
            # while redirecting the test's source path through the helper boundary.
            with mock.patch.object(
                n580_node,
                "_driver_library_paths",
                return_value=[library],
            ):
                records = n580_node._capture_qualification_driver_libraries(
                    artifact_root,
                    audit,
                )
            original = "/usr/lib/libcuda.so.580"
            records[0]["path"] = original
            records[0]["artifact_path"] = f"{n580_node.QUALIFICATION_DRIVER_LIBRARY_ROOT}{original}"
            sealed = artifact_root / records[0]["artifact_path"]
            sealed.parent.mkdir(parents=True, exist_ok=True)
            sealed.write_bytes(library.read_bytes())

            restored = n580_node._restore_qualification_driver_libraries(
                artifact_root,
                records,
                destination_root=destination_root,
            )
            destination = destination_root / "usr/lib/libcuda.so.580"
            self.assertEqual(destination.read_bytes(), b"capture-driver-library")
            self.assertEqual(restored[0]["action"], "materialized")
            self.assertEqual(
                n580_node._restore_qualification_driver_libraries(
                    artifact_root,
                    records,
                    destination_root=destination_root,
                )[0]["action"],
                "resident",
            )

    def test_n580_qualification_refuses_different_resident_driver_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifact"
            destination_root = root / "destination"
            original = "/usr/lib/libnvidia-ml.so.580"
            relative = f"{n580_node.QUALIFICATION_DRIVER_LIBRARY_ROOT}{original}"
            sealed = artifact_root / relative
            sealed.parent.mkdir(parents=True)
            sealed.write_bytes(b"captured")
            destination = destination_root / original.removeprefix("/")
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"different")
            records = [
                {
                    "path": original,
                    "artifact_path": relative,
                    "bytes": len(b"captured"),
                    "mode": 0o644,
                    "sha256": hashlib.sha256(b"captured").hexdigest(),
                }
            ]
            with self.assertRaisesRegex(RuntimeError, "refuses to replace"):
                n580_node._restore_qualification_driver_libraries(
                    artifact_root,
                    records,
                    destination_root=destination_root,
                )

    def test_sglang_cuda_restore_hold_is_validated_and_released(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hold = root / "cuda-restore-hold"
            hold.mkdir()
            (hold / "worker-0.ready.json").write_text(
                json.dumps(
                    {
                        "format": 1,
                        "kind": "coldsnap-sglang-cuda-restore-hold",
                        "worker_id": "worker-0",
                        "pid": 42,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(artifact_root=root, worker_count=1, timeout=1)
            records = node._wait_sglang_cuda_restore_hold(
                args, SimpleNamespace(poll=lambda: None), None
            )
            self.assertEqual([record["worker_id"] for record in records], ["worker-0"])
            self.assertTrue(node._release_sglang_cuda_restore_hold(args))
            self.assertEqual((hold / "release").read_text(encoding="utf-8"), "1\n")

    def test_n580_parser_accepts_sglang_operation_timeout(self) -> None:
        args = n580_node._parser().parse_args(
            [
                "--engine",
                "sglang",
                "--artifact-root",
                "/artifact",
                "--rank",
                "0",
                "--worker-count",
                "1",
                "--generation",
                "capture",
                "--master-address",
                "127.0.0.1",
                "--image-id",
                "sha256:image",
                "--model",
                "org/model",
                "--model-revision",
                "revision",
                "--expected",
                "expected",
                "--prompt",
                "prompt",
                "--target-launcher",
                "/runtime/exec.py",
                "--plugin-root",
                "/plugin",
                "--nccl-active-runtime",
                "/nccl/active.json",
                "--criu",
                "/criu",
                "--criu-rpc",
                "/criu-rpc",
                "--cuda-checkpoint",
                "/cuda-checkpoint",
                "--native-hydration-library",
                "/libhydration.so",
                "--criu-plugin-dir",
                "/plugins",
                "--nvml-dlopen-shim",
                "/libnvml-shim.so",
                "--runtime-lib-dir",
                "/runtime/lib",
                "--coordinator-path",
                "/coordinator.json",
                "--activation-namespace",
                "activation",
                "--timeout",
                "5400",
                "capture",
                "--",
                "sglang",
                "serve",
            ]
        )
        self.assertEqual(args.timeout, 5400)
        command = n580_node._criu_command(args, "dump")
        timeout_option = command.index("--timeout")
        self.assertEqual(command[timeout_option + 1], "3600")

    def test_n580_child_activates_qualified_nccl_provider_and_nvml_shim(self) -> None:
        active = {
            "preload_order": [
                "/provider/bridge.so",
                "/provider/checkpoint.so",
                "/provider/nccl.so",
            ],
            "provider_id": "nccl-2.31.2-1+coldsnap.10",
            "provider_revision": 10,
            "bridge": {"abi": 1},
            "_resolved": {"checkpoint-shim": Path("/provider/checkpoint.so")},
        }
        args = SimpleNamespace(
            artifact_root=Path("/artifact"),
            engine="sglang",
            generation="capture",
            nccl_active_runtime=Path("/provider/active.json"),
            nvml_dlopen_shim=Path("/runtime/libcoldsnap_nvml_dlopen_shim.so"),
            plugin_root=Path("/plugin"),
            timeout=5400,
        )
        with mock.patch.object(
            n580_node.service_runtime,
            "_load_active_nccl_runtime",
            return_value=active,
        ):
            environment = n580_node._child_environment(args)

        self.assertEqual(
            environment["LD_PRELOAD"],
            ":".join(
                [
                    "/provider/bridge.so",
                    "/provider/checkpoint.so",
                    "/provider/nccl.so",
                    "/runtime/libcoldsnap_nvml_dlopen_shim.so",
                ]
            ),
        )
        self.assertEqual(
            environment["COLDSNAP_NCCL_CHECKPOINT_SHIM_PATH"],
            "/provider/checkpoint.so",
        )
        self.assertEqual(
            environment["COLDSNAP_NCCL_PROVIDER_ID"],
            "nccl-2.31.2-1+coldsnap.10",
        )
        self.assertEqual(environment["COLDSNAP_NCCL_PROVIDER_REVISION"], "10")
        self.assertEqual(environment["COLDSNAP_NCCL_DLSYM_BRIDGE_ABI"], "1")

    def test_sglang_async_graphs_are_armed_on_every_rank_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory) / "graphs.arm"
            coordinator = SimpleNamespace(
                set=mock.Mock(),
                wait=mock.Mock(return_value="1"),
            )
            args = SimpleNamespace(
                engine="sglang",
                activation_state="running",
                rank=1,
                world_size=2,
                timeout=30,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "COLDSNAP_ASYNC_CUDA_GRAPHS": "1",
                    "COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE": str(arm),
                },
                clear=True,
            ):
                self.assertTrue(node._arm_sglang_async_graphs(args, coordinator, "activation"))

            self.assertTrue(arm.is_file())
            coordinator.set.assert_called_once_with("activation:sglang-async-graphs-armed:1", "1")
            self.assertEqual(coordinator.wait.call_count, 2)

    def test_sglang_load_snapshot_backing_has_a_bounded_restore_contract(self) -> None:
        path = "/dev/shm/sglang_loads_4c00ec5647f7_b4fb99a3.shm"
        self.assertTrue(node._restorable_regular_backing(path, "sglang"))
        self.assertFalse(node._restorable_regular_backing(path, "vllm"))
        self.assertFalse(
            node._restorable_regular_backing(
                "/dev/shm/sglang_loads_../../escape_b4fb99a3.shm", "sglang"
            )
        )

    def test_n580_tree_observation_resamples_vanished_helpers(self) -> None:
        with (
            mock.patch.object(n580_node.base, "_process_tree", return_value=[513, 602]),
            mock.patch.object(
                n580_node,
                "_audit_tree",
                side_effect=[FileNotFoundError("/proc/602/maps"), [{"pid": 513}]],
            ) as audit,
            mock.patch.object(n580_node.Path, "is_dir", return_value=True),
            mock.patch.object(n580_node.time, "sleep"),
        ):
            tree, records = n580_node._observe_capture_tree(513)

        self.assertEqual(tree, [513, 602])
        self.assertEqual(records, [{"pid": 513}])
        self.assertEqual(audit.call_count, 2)

    def test_n580_tree_observation_reports_serving_root_exit(self) -> None:
        with (
            mock.patch.object(n580_node.base, "_process_tree", return_value=[513, 602]),
            mock.patch.object(
                n580_node,
                "_audit_tree",
                side_effect=FileNotFoundError("/proc/602/maps"),
            ),
            mock.patch.object(n580_node.Path, "is_dir", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "serving root exited"):
                n580_node._observe_capture_tree(513)

    def test_rank_http_and_collective_rpc_share_operation_timeout(self) -> None:
        args = SimpleNamespace(
            master_address="127.0.0.1",
            http_port=8000,
            timeout=1800.0,
        )
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        with mock.patch.object(node.urllib.request, "urlopen", return_value=response) as urlopen:
            node._collective_rpc(args, "checkpoint_prepare")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["timeout"], 1800)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 1800.0)

        args.timeout = 5400.0
        with mock.patch.object(node.service_runtime, "_request", return_value={}) as request_call:
            node._collective_rpc(args, "checkpoint_prepare")
        self.assertEqual(request_call.call_args.args[3]["timeout"], 3600)

    def test_checkpoint_uses_remaining_operation_deadline(self) -> None:
        args = SimpleNamespace(
            master_address="127.0.0.1",
            http_port=8000,
            timeout=1800.0,
        )
        with mock.patch.object(node, "_request", return_value={}) as request:
            node._checkpoint(args, "prepare")

        payload = request.call_args.args[3]
        self.assertGreaterEqual(payload["timeout"], 1799)
        self.assertLessEqual(payload["timeout"], 1800)
        self.assertGreaterEqual(request.call_args.kwargs["timeout"], 1799)
        self.assertLessEqual(request.call_args.kwargs["timeout"], 1800)

        args.timeout = 5400.0
        with mock.patch.object(node, "_request", return_value={}) as request:
            node._checkpoint(args, "prepare")
        self.assertEqual(request.call_args.args[3]["timeout"], 3600)
        self.assertEqual(request.call_args.kwargs["timeout"], 3600.0)

    def test_checkpoint_timing_uses_slowest_worker(self) -> None:
        response = {
            "results": [
                {"nccl_restore_seconds": 0.4},
                {"nccl_restore_seconds": 0.7},
                {"nccl_restore_seconds": float("inf")},
            ]
        }
        self.assertEqual(
            node._checkpoint_max_seconds(response, "nccl_restore_seconds"),
            0.7,
        )
        self.assertIsNone(node._checkpoint_max_seconds(response, "missing"))

    def test_retained_nccl_activation_is_generation_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            activation = root / "nccl-in-place-activation"
            args = SimpleNamespace(
                artifact_root=root,
                activation_namespace="restore-generation-1",
            )
            environment = {
                "COLDSNAP_GRAPH_POLICY": "preserve-nccl-exec",
                "COLDSNAP_NCCL_IN_PLACE_MODE": "net-reconnect-v1",
                "COLDSNAP_NCCL_IN_PLACE_ACTIVATION_PATH": str(activation),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                self.assertEqual(node._stage_in_place_nccl_activation(args), activation)
            self.assertEqual(activation.read_text(), "restore-generation-1\n")

    def test_nonretained_restore_does_not_stage_nccl_activation(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"COLDSNAP_GRAPH_POLICY": "recreate-from-plan"},
            clear=True,
        ):
            self.assertIsNone(
                node._stage_in_place_nccl_activation(
                    SimpleNamespace(artifact_root=Path("/artifact"))
                )
            )

    def test_long_operation_uses_coordinator_sized_wait_chunks(self) -> None:
        calls: list[tuple[str, float]] = []

        class Coordinator:
            def wait(self, key: str, timeout: float) -> str:
                calls.append((key, timeout))
                return "ready"

        self.assertEqual(
            node._coordinator_wait(Coordinator(), "release", 5400.0),
            "ready",
        )
        self.assertEqual(calls, [("release", 3600.0)])

    def test_n580_dump_snapshots_link_remaps_during_post_dump_notify(self) -> None:
        args = SimpleNamespace(
            artifact_root=Path("/artifact"),
            criu_rpc=Path("/usr/local/bin/coldsnap-criu-rpc"),
            criu=Path("/opt/coldsnap/criu/bin/criu"),
            criu_plugin_dir=Path("/opt/coldsnap/criu/plugins"),
            shared_memory_dir=Path("/dev/shm"),
            timeout=600,
            ghost_limit=64 * 1024**2,
            decompress_threads=1,
            image_io_mode="direct",
            compress_block_bytes=256 * 1024,
            compress_acceleration=1,
        )

        command = n580_node._criu_command(args, "dump")

        source_index = command.index("--link-remap-source-dir")
        target_index = command.index("--link-remap-snapshot-dir")
        self.assertEqual(command[source_index + 1], "/dev/shm")
        self.assertEqual(command[target_index + 1], "/artifact/shm-image")

    def test_n580_model_identity_accepts_only_the_pinned_hf_snapshot(self) -> None:
        model = "Qwen/Qwen3.8-27B-FP8"
        revision = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
        snapshot = f"/cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/{revision}"

        self.assertTrue(n580_node._model_identity_matches(snapshot, revision, model, revision))
        self.assertTrue(n580_node._model_identity_matches(model, revision, model, revision))
        self.assertFalse(n580_node._model_identity_matches(snapshot, "other", model, revision))
        self.assertFalse(
            n580_node._model_identity_matches(
                f"/cache/huggingface/hub/models--Other--Model/snapshots/{revision}",
                revision,
                model,
                revision,
            )
        )

    @staticmethod
    def _identity() -> dict[str, object]:
        return {
            "format": 1,
            "controller_abi": 8,
            "model": "model",
            "kernel": "6.17.0-1029-nvidia",
            "architecture": "aarch64",
            "cuda_userspace": "13.0",
            "unit": "unit-0",
            "unit_index": 0,
            "unit_count": 2,
            "gpus": [
                {
                    "driver": "610.43.02",
                    "name": "NVIDIA GB10",
                    "compute_capability": "12.1",
                }
            ],
            "serve_command": [
                "vllm",
                "serve",
                "model",
                "--master-addr",
                "10.24.11.13",
                "--master-port=25000",
            ],
            "runtime_sha256": {
                "criu": "old-criu",
                "criu_rpc": "old-rpc",
                "target_launcher": "target",
            },
        }

    @staticmethod
    def _active_runtime(root: Path) -> Path:
        provider = root / "providers/nccl-2.31.2-1+coldsnap.10/linux-x86_64-cuda13-glibc2.38"
        (provider / "lib").mkdir(parents=True)
        common = root / "common"
        common.mkdir()
        runtime = provider / "lib/libnccl.so.2.31.2"
        shim = provider / "lib/libnccl-checkpoint-shim.so"
        bridge = common / "libcoldsnap-nccl-dlsym.so"
        for path, value in ((runtime, b"runtime"), (shim, b"shim"), (bridge, b"bridge")):
            path.write_bytes(value)
            path.chmod(0o755)
        active_shim = common / "libcoldsnap-checkpoint-shim.so"
        active_shim.write_bytes(shim.read_bytes())
        active_shim.chmod(0o755)

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        supported = [
            {
                "version": 23102,
                "release": "2.31.2",
                "path": "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
                "soname": "libnccl.so.2",
                "build_id": "1" * 40,
                "sha256": "2" * 64,
            }
        ]
        provider_abi = {"major": 1, "minor": 0}
        capabilities = [
            "full-network-reset",
            "ib-roce-device-release",
            "synchronous-termination",
        ]
        qualification = {
            "state": "accepted",
            "policy": "production",
            "capabilities": capabilities,
            "transports": ["socket"],
            "checks": ["abi-contract", "socket-restore"],
        }
        files = [
            {
                "path": "lib/libnccl-checkpoint-shim.so",
                "role": "checkpoint-shim",
                "mode": "0755",
                "size": shim.stat().st_size,
                "sha256": digest(shim),
                "build_id": "4" * 40,
                "soname": "libnccl-checkpoint-shim.so",
            },
            {
                "path": "lib/libnccl.so.2.31.2",
                "role": "nccl-runtime",
                "mode": "0755",
                "size": runtime.stat().st_size,
                "sha256": digest(runtime),
                "build_id": "5" * 40,
                "soname": "libnccl.so.2",
            },
        ]
        provider_runtime = {
            "version": 23102,
            "release": "2.31.2",
            "soname": files[1]["soname"],
            "build_id": files[1]["build_id"],
            "sha256": files[1]["sha256"],
        }
        manifest = {
            "format": 2,
            "kind": "coldsnap-nccl-provider",
            "provider_id": "nccl-2.31.2-1+coldsnap.10",
            "provider_revision": 10,
            "platform_key": "linux-x86_64-cuda13-glibc2.38",
            "provider_nccl_runtime": provider_runtime,
            "provider_selection": "exact",
            "supported_nccl_runtimes": supported,
            "provider_abi": provider_abi,
            "checkpoint_abi": 100,
            "capabilities": capabilities,
            "limitations": [],
            "requirements": {},
            "files": files,
            "provenance": {},
            "qualification": qualification,
            "dlsym_bridge_abi": 1,
            "preload_order": [
                "coldsnap-dlsym-bridge",
                "checkpoint-shim",
                "nccl-runtime",
            ],
        }
        manifest_path = provider / "provider-manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        active = {
            "format": 2,
            "kind": "coldsnap-nccl-active-runtime",
            "provider_id": manifest["provider_id"],
            "provider_revision": manifest["provider_revision"],
            "platform_key": manifest["platform_key"],
            "provider_root": str(provider),
            "provider_manifest": str(manifest_path),
            "manifest_sha256": digest(manifest_path),
            "provider_abi": provider_abi,
            "checkpoint_abi": manifest["checkpoint_abi"],
            "provider_nccl_runtime": provider_runtime,
            "provider_selection": "exact",
            "supported_nccl_runtimes": supported,
            "capabilities": capabilities,
            "limitations": [],
            "qualification": qualification,
            "files": {
                "checkpoint-shim": {
                    "path": str(active_shim),
                    "sha256": digest(active_shim),
                    "build_id": files[0]["build_id"],
                    "soname": files[0]["soname"],
                },
                "nccl-runtime": {
                    "path": str(runtime),
                    "sha256": digest(runtime),
                    "build_id": files[1]["build_id"],
                    "soname": files[1]["soname"],
                },
            },
            "bridge": {"path": str(bridge), "abi": 1, "sha256": digest(bridge)},
            "preload_order": [str(bridge), str(active_shim), str(runtime)],
        }
        active_path = root / "active.json"
        active_path.write_text(json.dumps(active, sort_keys=True) + "\n")
        return active_path

    def test_rank_controller_uses_qualified_page_io_defaults(self) -> None:
        parser = node._parser()
        self.assertEqual(parser.get_default("criu_compress_block_bytes"), 256 * 1024)
        self.assertEqual(parser.get_default("criu_compress_acceleration"), 1)
        self.assertEqual(parser.get_default("criu_decompress_threads"), 1)
        self.assertEqual(parser.get_default("criu_image_io_mode"), "direct")

    def test_sglang_recovery_resume_is_split_into_weight_and_runtime_phases(self) -> None:
        native = node._sglang_memory_payload("native")
        weights = node._sglang_memory_payload("recovery", "weights")
        runtime = node._sglang_memory_payload("recovery", "runtime")
        self.assertEqual(native["tags"], ["kv_cache", "weights", "cuda_graph"])
        self.assertEqual(weights["tags"], ["weights", "coldsnap_recovery_weights"])
        self.assertEqual(
            runtime["tags"],
            ["kv_cache", "cuda_graph", "coldsnap_recovery_runtime"],
        )
        with self.assertRaisesRegex(ValueError, "requires weights or runtime"):
            node._sglang_memory_payload("recovery")

    def test_full_blob_restore_does_not_select_optional_native_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"format": 2, "kind": "snapshot"}) + "\n")
            marker = root / "activation-provider"
            marker.write_text("native\n")

            node._select_weight_provider(manifest, "native")

            self.assertFalse(marker.exists())
            with self.assertRaisesRegex(RuntimeError, "require the native"):
                node._select_weight_provider(manifest, "recovery")

    def test_recovery_capture_weight_provider_selection_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"format": 2, "kind": "snapshot", "weight_source": "safetensors"}) + "\n"
            )
            with self.assertRaisesRegex(RuntimeError, "does not contain a native replay manifest"):
                node._select_weight_provider(manifest, "native")

            (root / "native-manifest.json").write_text("{}\n")
            node._select_weight_provider(manifest, "native")
            self.assertEqual((root / "activation-provider").read_text(), "native\n")

            node._select_weight_provider(manifest, "recovery")
            self.assertEqual((root / "activation-provider").read_text(), "recovery\n")

            manifest.write_text(
                json.dumps({"format": 2, "kind": "snapshot", "weight_source": "unknown"}) + "\n"
            )
            with self.assertRaisesRegex(RuntimeError, "unsupported weight source"):
                node._select_weight_provider(manifest, "recovery")

    def test_exact_identity_does_not_require_upgrade_override(self) -> None:
        identity = self._identity()
        self.assertEqual(
            node._identity_compatibility(identity, identity, False),
            "exact-v8-placement-independent",
        )

    def test_provider_identity_substitution_is_rejected(self) -> None:
        captured = self._identity()
        captured["nccl_provider"] = {
            "provider_id": "nccl-2.31.2-1+coldsnap.10",
            "manifest_sha256": "a" * 64,
        }
        current = json.loads(json.dumps(captured))
        provider = current["nccl_provider"]
        assert isinstance(provider, dict)
        provider["manifest_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            node._identity_compatibility(captured, current, False)

    def test_active_runtime_resolves_preload_order_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._active_runtime(root)
            active = node._load_active_nccl_runtime(path)
            self.assertEqual(active["provider_id"], "nccl-2.31.2-1+coldsnap.10")
            environment = node._child_environment(
                SimpleNamespace(nccl_active_runtime=path), root / "job.json"
            )
            self.assertEqual(environment["LD_PRELOAD"], ":".join(active["preload_order"]))
            self.assertEqual(environment["COLDSNAP_NCCL_PROVIDER_REVISION"], "10")
            self.assertEqual(environment["COLDSNAP_NCCL_DLSYM_BRIDGE_ABI"], "1")

            runtime = active["_resolved"]["nccl-runtime"]
            assert isinstance(runtime, Path)
            runtime.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                node._load_active_nccl_runtime(path)

    def test_sglang_child_internalizes_stable_allocator_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active_path = self._active_runtime(root)
            args = SimpleNamespace(engine="sglang", nccl_active_runtime=active_path)
            with unittest.mock.patch.dict(
                os.environ,
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64,expandable_segments:True",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8",
                },
                clear=True,
            ):
                environment = node._child_environment(args, root / "job.json")

            self.assertEqual(
                environment["PYTORCH_CUDA_ALLOC_CONF"],
                "max_split_size_mb:64,expandable_segments:False",
            )
            self.assertEqual(
                environment["PYTORCH_ALLOC_CONF"],
                "garbage_collection_threshold:0.8,expandable_segments:False",
            )

    def test_explicit_override_allows_only_criu_runtime_changes(self) -> None:
        captured = self._identity()
        current = self._identity()
        runtime = current["runtime_sha256"]
        assert isinstance(runtime, dict)
        runtime["criu"] = "new-criu"
        runtime["criu_rpc"] = "new-rpc"
        runtime["criu_lz4"] = "lz4"
        self.assertEqual(
            node._identity_compatibility(captured, current, True),
            "portable-v8-explicit-criu-runtime-upgrade",
        )
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            node._identity_compatibility(captured, current, False)

    def test_override_rejects_non_criu_identity_change(self) -> None:
        captured = self._identity()
        current = self._identity()
        runtime = current["runtime_sha256"]
        assert isinstance(runtime, dict)
        runtime["target_launcher"] = "changed-target"
        with self.assertRaisesRegex(RuntimeError, "outside CRIU runtime"):
            node._identity_compatibility(captured, current, True)

    def test_portable_identity_accepts_placement_and_newer_driver(self) -> None:
        captured = self._identity()
        current = self._identity()
        current["serve_command"] = [
            "vllm",
            "serve",
            "model",
            "--master-address=10.24.11.17",
            "--master-port",
            "25100",
            "--port",
            "8103",
        ]
        # Keep the spelling stable; only the address and port values are
        # placement. A change from addr to address remains semantic.
        captured["serve_command"] = [
            "vllm",
            "serve",
            "model",
            "--master-address=10.24.11.13",
            "--master-port",
            "25000",
            "--port",
            "8000",
        ]
        gpus = current["gpus"]
        assert isinstance(gpus, list) and isinstance(gpus[0], dict)
        gpu = gpus[0]
        gpu["driver"] = "611.1.0"
        self.assertEqual(
            node._identity_compatibility(captured, current, False),
            "portable-v8-placement-independent",
        )

    def test_portable_identity_uses_capability_kernel_policy_by_default(self) -> None:
        captured = self._identity()
        current = json.loads(json.dumps(captured))
        current["kernel"] = "6.18.1-1001-nvidia"
        self.assertEqual(
            node._identity_compatibility(captured, current, False),
            "portable-v8-capability-kernel",
        )
        with self.assertRaisesRegex(RuntimeError, r"identity\.kernel"):
            node._identity_compatibility(captured, current, False, "exact")

    def test_portable_identity_reports_mismatched_field_paths_without_values(self) -> None:
        captured = self._identity()
        current = json.loads(json.dumps(captured))
        runtime = current["runtime_sha256"]
        assert isinstance(runtime, dict)
        runtime["target_launcher"] = "sensitive-current-value"
        with self.assertRaisesRegex(
            RuntimeError, r"identity\.runtime_sha256\.target_launcher"
        ) as raised:
            node._identity_compatibility(captured, current, False)
        self.assertNotIn("sensitive-current-value", str(raised.exception))

    def test_n580_identity_is_placement_independent_and_kernel_capability_based(self) -> None:
        captured = self._identity()
        captured.update(
            {
                "engine": "vllm",
                "rank": 0,
                "world_size": 2,
                "worker_count": 2,
                "image_id": "sha256:image",
                "model_revision": "revision",
            }
        )
        current = json.loads(json.dumps(captured))
        captured["serve_command"] = [
            "vllm",
            "serve",
            "model",
            "--master-address=10.24.11.14",
            "--master-port",
            "25000",
            "--port",
            "8000",
        ]
        current["serve_command"] = [
            "vllm",
            "serve",
            "model",
            "--master-address=192.168.1.41",
            "--master-port",
            "25100",
            "--port",
            "8103",
        ]
        current["kernel"] = "6.18.1-1001-nvidia"
        gpus = current["gpus"]
        assert isinstance(gpus, list) and isinstance(gpus[0], dict)
        gpus[0]["driver"] = "611.1.0"
        self.assertEqual(
            n580_node._identity_compatibility(captured, current),
            "n580-capability-kernel-v2",
        )
        with self.assertRaisesRegex(RuntimeError, r"identity\.kernel"):
            n580_node._identity_compatibility(captured, current, "exact")

        current = json.loads(json.dumps(captured))
        runtime = current["runtime_sha256"]
        assert isinstance(runtime, dict)
        runtime["criu_rpc"] = "release-matched-overlay"
        with self.assertRaisesRegex(RuntimeError, r"identity\.runtime_sha256\.criu_rpc"):
            n580_node._identity_compatibility(captured, current)
        self.assertEqual(
            n580_node._identity_compatibility(captured, current, "capability", True),
            "n580-placement-independent-v2+criu-rpc-overlay",
        )

        # The launcher comes from the ABI-checked, content-addressed activation
        # runtime and may be patched without rebuilding the capsule.
        current = json.loads(json.dumps(captured))
        runtime = current["runtime_sha256"]
        assert isinstance(runtime, dict)
        runtime["target_launcher"] = "release-matched-activation-overlay"
        self.assertEqual(
            n580_node._identity_compatibility(captured, current),
            "n580-placement-independent-v2",
        )

        # Capture-sensitive engine plugin code remains exact.
        runtime["plugin/coldsnap_vllm.py"] = "changed-plugin"
        with self.assertRaisesRegex(
            RuntimeError, r"identity\.runtime_sha256\.plugin/coldsnap_vllm\.py"
        ):
            n580_node._identity_compatibility(captured, current)

    def test_portable_identity_rejects_older_driver_or_different_gpu(self) -> None:
        captured = self._identity()
        current = self._identity()
        gpus = current["gpus"]
        assert isinstance(gpus, list) and isinstance(gpus[0], dict)
        gpu = gpus[0]
        gpu["driver"] = "609.99"
        with self.assertRaisesRegex(RuntimeError, "older than the captured"):
            node._identity_compatibility(captured, current, False)

        current = self._identity()
        gpus = current["gpus"]
        assert isinstance(gpus, list) and isinstance(gpus[0], dict)
        gpu = gpus[0]
        gpu["compute_capability"] = "12.2"
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            node._identity_compatibility(captured, current, False)

    def test_restore_passes_explicit_tcp_address_mapping_to_rpc(self) -> None:
        args = SimpleNamespace(
            criu_rpc=Path("/usr/local/bin/coldsnap-criu-rpc"),
            criu=Path("/opt/coldsnap/criu/bin/criu"),
            cuda_checkpoint=Path("/usr/local/bin/cuda-checkpoint"),
            artifact_root=Path("/opt/coldsnap/capsule"),
            timeout=1800.0,
            ghost_limit=64 * 1024**2,
            criu_decompress_threads=0,
            criu_image_io_mode="writeback",
            criu_compress_block_bytes=0,
            criu_compress_acceleration=1,
            tcp_address_map=[
                "10.24.11.13=10.24.11.17",
                "10.24.11.17=10.24.11.13",
            ],
            tcp_port_shift=4096,
            http_port=8011,
            leave_stopped=True,
            defer_network_unlock=True,
            serve_command=["--", "vllm", "serve", "model", "--port", "8011"],
        )
        command = node._criu_command(args, "restore", "restore.log")
        self.assertEqual(command.count("--tcp-address-map"), 2)
        self.assertIn("10.24.11.13=10.24.11.17", command)
        self.assertIn("10.24.11.17=10.24.11.13", command)
        self.assertIn("--tcp-port-shift", command)
        self.assertIn("4096", command)
        self.assertEqual(command[command.index("--tcp-preserve-port") + 1], "8011")
        self.assertIn("--leave-stopped", command)
        self.assertIn("--defer-network-unlock", command)
        args.http_port = 8103
        args.serve_command = ["--", "vllm", "serve", "model", "--port", "8103"]
        args.captured_http_port = 8011
        migrated = node._criu_command(args, "restore", "restore.log")
        self.assertNotIn("--tcp-preserve-port", migrated)
        self.assertEqual(migrated[migrated.index("--tcp-port-map") + 1], "8011=8103")
        args.timeout = 5400.0
        long_operation = node._criu_command(args, "restore", "restore.log")
        self.assertEqual(long_operation[long_operation.index("--timeout") + 1], "3600")
        dump = node._criu_command(args, "dump", "dump.log")
        self.assertNotIn("--tcp-address-map", dump)
        self.assertNotIn("--tcp-port-shift", dump)
        self.assertNotIn("--leave-stopped", dump)
        self.assertNotIn("--defer-network-unlock", dump)

        deferred = node._deferred_cuda_restore_command(args, 513)
        self.assertEqual(deferred[deferred.index("--action") + 1], "cuda-restore")
        self.assertEqual(deferred[deferred.index("--pid") + 1], "513")
        self.assertEqual(deferred[deferred.index("--timeout") + 1], "3600")
        self.assertIn("--cuda-process-tree", deferred)
        self.assertEqual(
            deferred[deferred.index("--cuda-processes-result") + 1],
            str(args.artifact_root / "dump-rpc.json"),
        )
        self.assertNotIn("--images-dir", deferred)

    def test_network_unlock_waits_for_every_unit_before_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            calls: list[tuple[str, str]] = []

            class Coordinator:
                def set(self, key: str, value: str) -> None:
                    calls.append(("set", key))

                def wait(self, key: str, timeout: float) -> str:
                    del timeout
                    calls.append(("wait", key))
                    return "1"

            def run_criu(command: list[str], environment: dict[str, str]) -> dict[str, float]:
                del command, environment
                (work / "network-unlock-ready").touch()
                deadline = time.monotonic() + 1
                while not (work / "network-unlock-release").is_file():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("release was not published")
                    time.sleep(0.001)
                return {"wall_seconds": 0.1}

            args = SimpleNamespace(
                artifact_root=root,
                activation_namespace="activation",
                rank=1,
                world_size=2,
                timeout=1.0,
            )
            with mock.patch.object(node.base, "_run_criu_profiled", side_effect=run_criu):
                profile, seconds = node._run_restore_with_network_unlock_barrier(
                    args, ["criu"], {}, Coordinator()
                )
            self.assertEqual(profile, {"wall_seconds": 0.1})
            self.assertGreaterEqual(seconds, 0)
            self.assertEqual(
                calls,
                [
                    ("set", "activation:network-unlock-ready:1"),
                    ("wait", "activation:network-unlock-ready:0"),
                    ("wait", "activation:network-unlock-ready:1"),
                ],
            )
            self.assertTrue((work / "network-unlock-release").is_file())

    def test_restore_stages_only_destination_transport_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "restore-transport-environment.json"
            environment = {
                "COLDSNAP_RESTORE_TRANSPORT_ENVIRONMENT_PATH": str(path),
                "COLDSNAP_EXPECTED_UNIT": "unit-1",
                "NCCL_IB_HCA": "roce0",
                "UCX_NET_DEVICES": "roce0:1",
                "VLLM_HOST_IP": "10.0.0.2",
                "HF_HOME": "/cache/huggingface",
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                result = node._stage_restore_transport_environment(
                    SimpleNamespace(artifact_root=root, rank=1)
                )
            self.assertEqual(result, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["unit"], "unit-1")
            self.assertEqual(
                payload["variables"],
                {
                    "NCCL_IB_HCA": "roce0",
                    "UCX_NET_DEVICES": "roce0:1",
                    "VLLM_HOST_IP": "10.0.0.2",
                },
            )

    def test_restore_stages_target_runtime_policy_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "restore-runtime-environment.json"
            environment = {
                "COLDSNAP_RESTORE_RUNTIME_ENVIRONMENT_PATH": str(path),
                "COLDSNAP_EXPECTED_UNIT": "unit-1",
                "COLDSNAP_ASYNC_CUDA_GRAPHS": "1",
                "COLDSNAP_ASYNC_CUDA_GRAPHS_GENERATION": "restore-7",
                "COLDSNAP_SHAPE_CALIBRATION": "0",
                "VLLM_ENABLE_STARTUP_PLAN": "1",
                "NCCL_IB_HCA": "roce0",
                "HF_HOME": "/cache/huggingface",
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                result = node._stage_restore_runtime_environment(
                    SimpleNamespace(artifact_root=root, rank=1)
                )
            self.assertEqual(result, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["unit"], "unit-1")
            self.assertEqual(
                payload["variables"],
                {
                    "COLDSNAP_ASYNC_CUDA_GRAPHS": "1",
                    "COLDSNAP_ASYNC_CUDA_GRAPHS_GENERATION": "restore-7",
                    "COLDSNAP_SHAPE_CALIBRATION": "0",
                    "VLLM_ENABLE_STARTUP_PLAN": "1",
                },
            )


if __name__ == "__main__":
    unittest.main()

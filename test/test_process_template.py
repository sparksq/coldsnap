# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

import coldsnap_vllm_process_template as process_template  # noqa: E402
import coldsnap_vllm_worker_reexec as worker_reexec  # noqa: E402


class ProcessTemplateTest(unittest.TestCase):
    def test_portable_worker_reexec_rebuilds_target_local_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = process_template.ProcessTemplateSettings(
                ready_dir=Path(directory),
                generation="generation-a",
            )
            record = process_template._portable_worker_lock_record(settings)
            first = worker_reexec._shared_lock(record)
            second = worker_reexec._shared_lock(record)
            self.assertTrue(first.acquire(timeout=1))
            self.assertFalse(second.acquire(block=False))
            first.release()
            self.assertTrue(second.acquire(timeout=1))
            second.release()

    def test_portable_worker_reexec_preserves_pipe_direction(self) -> None:
        reader, writer = multiprocessing.Pipe(duplex=False)
        try:
            record = process_template._connection_record(writer)
            self.assertFalse(record["readable"])
            self.assertTrue(record["writable"])
            self.assertTrue(os.get_inheritable(record["fd"]))
        finally:
            reader.close()
            writer.close()

    def test_settings_are_disabled_by_default_and_fail_closed_when_partial(self) -> None:
        self.assertFalse(process_template.process_template_settings_from_env({}).enabled)
        with self.assertRaisesRegex(
            process_template.VllmContractError, "must be supplied together"
        ):
            process_template.process_template_settings_from_env(
                {process_template.RELEASE_FILE_ENV: "/tmp/release"}
            )

    def test_restore_marker_must_match_the_captured_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "restore-generation"
            settings = process_template.ProcessTemplateSettings(
                generation="generation-restore",
                restore_marker_file=marker,
            )
            self.assertFalse(process_template._restored_generation(settings))
            marker.write_text("different-generation\n", encoding="utf-8")
            self.assertFalse(process_template._restored_generation(settings))
            marker.write_text("generation-restore\n", encoding="utf-8")
            self.assertTrue(process_template._restored_generation(settings))

    def test_restored_runtime_environment_replaces_capture_only_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "restore-generation"
            marker.write_text("generation-restore\n", encoding="utf-8")
            path = root / process_template.RESTORE_RUNTIME_ENVIRONMENT_FILENAME
            path.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "kind": "coldsnap-restore-runtime-environment",
                        "unit": "unit-0",
                        "variables": {
                            "COLDSNAP_ASYNC_CUDA_GRAPHS": "1",
                            "COLDSNAP_ASYNC_CUDA_GRAPHS_GENERATION": "restore-9",
                            "COLDSNAP_SHAPE_CALIBRATION": "0",
                        },
                    }
                ),
                encoding="utf-8",
            )
            settings = process_template.ProcessTemplateSettings(
                generation="generation-restore",
                restore_marker_file=marker,
            )
            with patch.dict(
                os.environ,
                {
                    "COLDSNAP_EXPECTED_UNIT": "unit-0",
                    process_template.RESTORE_RUNTIME_ENVIRONMENT_PATH_ENV: str(path),
                    "COLDSNAP_SHAPE_CALIBRATION": "1",
                },
                clear=True,
            ):
                variables = process_template._restored_runtime_environment(settings)
            self.assertEqual(
                variables,
                {
                    "COLDSNAP_ASYNC_CUDA_GRAPHS": "1",
                    "COLDSNAP_ASYNC_CUDA_GRAPHS_GENERATION": "restore-9",
                    "COLDSNAP_SHAPE_CALIBRATION": "0",
                },
            )

    def test_exact_target_restore_applies_runtime_environment_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "restore-generation"
            marker.write_text("generation-restore\n", encoding="utf-8")
            path = root / process_template.RESTORE_RUNTIME_ENVIRONMENT_FILENAME
            path.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "kind": "coldsnap-restore-runtime-environment",
                        "unit": "unit-0",
                        "variables": {
                            "COLDSNAP_ASYNC_CUDA_GRAPHS": "1",
                            "COLDSNAP_SHAPE_CALIBRATION": "0",
                        },
                    }
                ),
                encoding="utf-8",
            )
            settings = process_template.ProcessTemplateSettings(
                generation="generation-restore",
                restore_marker_file=marker,
            )
            with patch.dict(
                os.environ,
                {
                    "COLDSNAP_EXPECTED_UNIT": "unit-0",
                    process_template.RESTORE_RUNTIME_ENVIRONMENT_PATH_ENV: str(path),
                    "COLDSNAP_SHAPE_CALIBRATION": "1",
                    "COLDSNAP_DEFERRED_WARMUP": "capture-only",
                },
                clear=True,
            ):
                variables = process_template._apply_restored_runtime_environment(
                    settings
                )
                self.assertEqual(os.environ["COLDSNAP_SHAPE_CALIBRATION"], "0")
                self.assertEqual(os.environ["COLDSNAP_ASYNC_CUDA_GRAPHS"], "1")
                self.assertNotIn("COLDSNAP_DEFERRED_WARMUP", os.environ)

            self.assertEqual(
                variables,
                {
                    "COLDSNAP_ASYNC_CUDA_GRAPHS": "1",
                    "COLDSNAP_SHAPE_CALIBRATION": "0",
                },
            )

    def test_restored_runtime_environment_rejects_unknown_variable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "restore-generation"
            marker.write_text("generation-restore\n", encoding="utf-8")
            path = root / process_template.RESTORE_RUNTIME_ENVIRONMENT_FILENAME
            path.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "kind": "coldsnap-restore-runtime-environment",
                        "unit": "unit-0",
                        "variables": {"HF_TOKEN": "must-not-cross-boundary"},
                    }
                ),
                encoding="utf-8",
            )
            settings = process_template.ProcessTemplateSettings(
                generation="generation-restore",
                restore_marker_file=marker,
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "COLDSNAP_EXPECTED_UNIT": "unit-0",
                        process_template.RESTORE_RUNTIME_ENVIRONMENT_PATH_ENV: str(path),
                    },
                    clear=True,
                ),
                self.assertRaisesRegex(
                    process_template.VllmContractError, "invalid variable"
                ),
            ):
                process_template._restored_runtime_environment(settings)

    def test_restored_worker_uses_controller_staged_rendezvous_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "restore-generation"
            marker.write_text("generation-restore\n", encoding="utf-8")
            (root / process_template.RESTORE_PLACEMENT_FILENAME).write_text(
                json.dumps(
                    {
                        "format": 1,
                        "kind": "coldsnap-process-template-placement",
                        "generation": "generation-restore",
                        "master_address": "192.168.1.43",
                        "master_port": 25103,
                        "http_port": 8103,
                        "tcp_address_map": {"10.24.11.14": "192.168.1.43"},
                        "tcp_port_shift": 103,
                    }
                ),
                encoding="utf-8",
            )
            settings = process_template.ProcessTemplateSettings(
                generation="generation-restore",
                restore_marker_file=marker,
            )
            parallel = SimpleNamespace(master_addr="10.24.11.14", master_port=25000)
            configs = [
                {"vllm_config": SimpleNamespace(parallel_config=parallel)},
                {"vllm_config": SimpleNamespace(parallel_config=parallel)},
            ]
            lazy_queue = {
                "captured": "tcp://10.24.11.14:25000",
                "restored": "tcp://192.168.1.43:25103",
            }
            with (
                patch.dict(
                    os.environ,
                    {
                        "VLLM_HOST_IP": "10.24.11.14",
                        "NODE_IP": "10.24.11.14",
                    },
                    clear=True,
                ),
                patch.object(
                    process_template,
                    "_remap_lazy_input_queue",
                    return_value=lazy_queue,
                ),
            ):
                placement = process_template._apply_restored_placement(configs, settings)
                self.assertEqual(os.environ["MASTER_ADDR"], "192.168.1.43")
                self.assertEqual(os.environ["MASTER_PORT"], "25103")
                self.assertEqual(os.environ["VLLM_HOST_IP"], "192.168.1.43")
                self.assertEqual(os.environ["NODE_IP"], "192.168.1.43")
            self.assertEqual(placement["http_port"], 8103)
            self.assertEqual(parallel.master_addr, "192.168.1.43")
            self.assertEqual(parallel.master_port, 25103)
            self.assertEqual(placement["lazy_input_queue"], lazy_queue)
            self.assertEqual(
                placement["local_addresses"],
                {
                    "VLLM_HOST_IP": {
                        "captured": "10.24.11.14",
                        "restored": "192.168.1.43",
                    },
                    "NODE_IP": {
                        "captured": "10.24.11.14",
                        "restored": "192.168.1.43",
                    },
                },
            )

    def test_lazy_tcp_endpoint_follows_restore_address_and_port_shift(self) -> None:
        placement = {
            "tcp_address_map": {"10.24.11.14": "192.168.1.43"},
            "tcp_port_shift": 103,
        }
        self.assertEqual(
            process_template._remap_tcp_endpoint("tcp://10.24.11.14:25000", placement),
            "tcp://192.168.1.43:25103",
        )

    def test_lazy_input_queue_is_retained_by_worker_constructor(self) -> None:
        handle = SimpleNamespace(remote_subscribe_addr="tcp://10.24.11.14:25000")
        placement = {
            "tcp_address_map": {"10.24.11.14": "192.168.1.43"},
            "tcp_port_shift": 103,
        }
        observed: list[object] = []

        class WorkerProc:
            def __init__(self, input_shm_handle: object) -> None:
                self.input_shm_handle = input_shm_handle
                observed.append(process_template._active_input_queue_handle)
                observed.append(process_template._remap_lazy_input_queue(placement))

        process_template._install_worker_proc_handle_hook(
            process_template.ProcessTemplateSettings(
                phase=process_template.PRE_WORKER_IMPORT_PHASE
            ),
            SimpleNamespace(WorkerProc=WorkerProc),
        )
        worker = WorkerProc(handle)
        receipt = observed[1]
        self.assertIs(observed[0], handle)
        self.assertEqual(receipt["captured"], "tcp://10.24.11.14:25000")
        self.assertEqual(receipt["restored"], "tcp://192.168.1.43:25103")
        self.assertEqual(
            worker.input_shm_handle.remote_subscribe_addr,
            "tcp://192.168.1.43:25103",
        )
        self.assertIs(
            process_template._active_input_queue_handle,
            process_template._INPUT_QUEUE_HANDLE_UNSET,
        )

    def test_headless_worker_has_no_local_input_queue_handle(self) -> None:
        observed: list[dict[str, str | None]] = []

        class WorkerProc:
            def __init__(self, input_shm_handle: object | None) -> None:
                self.input_shm_handle = input_shm_handle
                observed.append(
                    process_template._remap_lazy_input_queue(
                        {"tcp_address_map": {}, "tcp_port_shift": 103}
                    )
                )

        process_template._install_worker_proc_handle_hook(
            process_template.ProcessTemplateSettings(
                phase=process_template.PRE_WORKER_IMPORT_PHASE
            ),
            SimpleNamespace(WorkerProc=WorkerProc),
        )
        worker = WorkerProc(None)
        self.assertIsNone(worker.input_shm_handle)
        self.assertEqual(observed, [{"captured": None, "restored": None}])
        self.assertIs(
            process_template._active_input_queue_handle,
            process_template._INPUT_QUEUE_HANDLE_UNSET,
        )

    def test_restored_placement_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "restore-generation"
            marker.write_text("generation-restore\n", encoding="utf-8")
            target = root / "elsewhere.json"
            target.write_text("{}\n", encoding="utf-8")
            (root / process_template.RESTORE_PLACEMENT_FILENAME).symlink_to(target)
            settings = process_template.ProcessTemplateSettings(
                generation="generation-restore",
                restore_marker_file=marker,
            )
            with self.assertRaisesRegex(process_template.VllmContractError, "small regular file"):
                process_template._restored_placement(settings)

    def test_worker_reinitializes_driver_clients_only_after_restore(self) -> None:
        events: list[str] = []

        class WorkerWrapperBase:
            rpc_rank = 0
            global_rank = 0

            def init_worker(self, all_kwargs) -> str:
                del all_kwargs
                events.append("init")
                return "initialized"

        config = SimpleNamespace(
            model_config=SimpleNamespace(model="model", revision="revision"),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
        settings = process_template.ProcessTemplateSettings(
            release_file=Path("/release"),
            ready_dir=Path("/ready"),
            generation="generation-restore",
            phase=process_template.PRE_WORKER_IMPORT_PHASE,
        )
        module = SimpleNamespace(WorkerWrapperBase=WorkerWrapperBase)
        with (
            patch.object(process_template, "_wait_for_release", return_value=0.1),
            patch.object(
                process_template,
                "_reinitialize_driver_clients_after_restore",
                side_effect=lambda _settings: events.append("reinit"),
            ),
        ):
            process_template._install_worker_import_barrier(settings, module)
            wrapper = WorkerWrapperBase()
            self.assertEqual(wrapper.init_worker([{"vllm_config": config}]), "initialized")
        self.assertEqual(events, ["reinit", "init"])

    def test_restored_accelerator_descriptors_are_discarded(self) -> None:
        descriptors = [Path("/proc/self/fd/4"), Path("/proc/self/fd/9")]

        with (
            patch.object(process_template, "_captured_accelerator_fds", {}),
            patch.object(
                process_template.Path,
                "iterdir",
                return_value=iter(descriptors),
            ),
            patch.object(
                process_template.os,
                "readlink",
                side_effect=["/dev/nvidiactl", "/tmp/regular"],
            ),
            patch.object(process_template.os, "close") as close,
        ):
            closed = process_template._close_restored_accelerator_fds()

        self.assertEqual(closed, ["/dev/nvidiactl"])
        close.assert_called_once_with(4)

    def test_restored_placeholder_descriptors_close_by_captured_fd_number(self) -> None:
        descriptor = Path("/proc/self/fd/27")
        with (
            patch.object(
                process_template,
                "_captured_accelerator_fds",
                {27: "/dev/nvidiactl"},
            ),
            patch.object(
                process_template.Path,
                "iterdir",
                return_value=iter([descriptor]),
            ),
            patch.object(process_template.os, "readlink", return_value="/dev/null"),
            patch.object(process_template.os, "close") as close,
        ):
            closed = process_template._close_restored_accelerator_fds()

        self.assertEqual(closed, ["/dev/nvidiactl (placeholder /dev/null)"])
        close.assert_called_once_with(27)

    def test_identifiable_placeholder_closes_without_captured_fd_record(self) -> None:
        descriptor = Path("/proc/self/fd/31")
        target = "/tmp/coldsnap-nvidia-placeholder-17-abcd (deleted)"
        with (
            patch.object(process_template, "_captured_accelerator_fds", {}),
            patch.object(
                process_template.Path,
                "iterdir",
                return_value=iter([descriptor]),
            ),
            patch.object(process_template.os, "readlink", return_value=target),
            patch.object(process_template.os, "close") as close,
        ):
            closed = process_template._close_restored_accelerator_fds()

        self.assertEqual(closed, [target])
        close.assert_called_once_with(31)

    def test_driver_reinitialization_is_once_per_process_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "restore-generation"
            marker.write_text("generation-once\n", encoding="utf-8")
            settings = process_template.ProcessTemplateSettings(
                generation="generation-once",
                restore_marker_file=marker,
            )
            with (
                patch.dict(
                    os.environ,
                    {"COLDSNAP_EXPORT_MODEL_PAYLOAD": "1"},
                    clear=True,
                ),
                patch.object(
                    process_template,
                    "_restore_reinitialized_generation",
                    "",
                ),
                patch.object(
                    process_template,
                    "_restore_reinitialization_result",
                    None,
                ),
                patch.object(
                    process_template,
                    "_close_restored_accelerator_fds",
                    return_value=["/dev/nvidiactl"],
                ) as close,
                patch.object(
                    process_template,
                    "_reset_restored_nvml_client",
                    return_value=True,
                ) as reset,
            ):
                first = process_template._reinitialize_driver_clients_after_restore(settings)
                second = process_template._reinitialize_driver_clients_after_restore(settings)

                self.assertEqual(os.environ["COLDSNAP_PROCESS_TEMPLATE_RESTORED"], "1")
                self.assertEqual(os.environ["COLDSNAP_EXPORT_MODEL_PAYLOAD"], "0")

            self.assertIs(first, second)
            close.assert_called_once_with()
            reset.assert_called_once_with()

    def test_worker_advertises_exact_generation_before_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = process_template.ProcessTemplateSettings(
                release_file=root / "release",
                ready_dir=root / "ready",
                generation="generation-7",
                timeout_seconds=2,
                poll_seconds=0.005,
            )
            worker = SimpleNamespace(
                rank=3,
                local_rank=1,
                model_config=SimpleNamespace(model="model", revision="revision"),
                parallel_config=SimpleNamespace(tensor_parallel_size=4),
            )

            def release() -> None:
                ready = settings.ready_dir / "rank-3.json"
                deadline = time.monotonic() + 1
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                settings.release_file.write_text("stale-generation\n", encoding="utf-8")
                time.sleep(0.02)
                settings.release_file.write_text("generation-7\n", encoding="utf-8")

            thread = threading.Thread(target=release)
            thread.start()
            waited = process_template._wait_for_release(worker, settings)
            thread.join(timeout=1)
            payload = json.loads((settings.ready_dir / "rank-3.json").read_text(encoding="utf-8"))
            self.assertGreater(waited, 0)
            self.assertEqual(payload["generation"], "generation-7")
            self.assertEqual(payload["rank"], 3)
            self.assertEqual(payload["tensor_parallel_size"], 4)
            self.assertEqual(payload["phase"], "pre_load")

    def test_pre_hydration_phase_publishes_layout_before_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                process_template.RELEASE_FILE_ENV: str(root / "release"),
                process_template.READY_DIR_ENV: str(root / "ready"),
                process_template.GENERATION_ENV: "generation-8",
                process_template.PHASE_ENV: "pre_hydration",
                process_template.TIMEOUT_ENV: "2",
                process_template.POLL_SECONDS_ENV: "0.005",
                "LOCAL_RANK": "0",
                "COLDSNAP_MODEL_ID": "model",
                "COLDSNAP_MODEL_REVISION": "revision",
                "COLDSNAP_TP_SIZE": "4",
            }

            def release() -> None:
                ready = root / "ready/rank-2.json"
                deadline = time.monotonic() + 1
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                (root / "release").write_text("generation-8\n", encoding="utf-8")

            thread = threading.Thread(target=release)
            thread.start()
            with patch.dict(process_template.os.environ, environment):
                waited = process_template.wait_for_hydration_release(
                    rank=2,
                    capture_id="capture",
                    allocation_count=492,
                    allocation_bytes=1000,
                    replay_allocation_count=488,
                )
            thread.join(timeout=1)
            self.assertIsNotNone(waited)
            payload = json.loads((root / "ready/rank-2.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "pre_hydration")
            self.assertEqual(payload["replay_allocation_count"], 488)

    def test_pre_device_hook_waits_before_cuda_initialization(self) -> None:
        events: list[str] = []

        class Worker:
            rank = 0
            local_rank = 0
            model_config = SimpleNamespace(model="model", revision="revision")
            parallel_config = SimpleNamespace(tensor_parallel_size=1)

            def init_device(self) -> str:
                events.append("init_device")
                return "initialized"

        settings = process_template.ProcessTemplateSettings(
            release_file=Path("/release"),
            ready_dir=Path("/ready"),
            generation="generation-9",
            phase=process_template.PRE_DEVICE_PHASE,
        )

        def wait(worker, resolved_settings):
            self.assertIsInstance(worker, Worker)
            self.assertIs(resolved_settings, settings)
            events.append("wait")
            return 1.25

        module = SimpleNamespace(Worker=Worker)
        with (
            patch.object(process_template.importlib, "import_module", return_value=module),
            patch.object(process_template, "_wait_for_release", side_effect=wait),
        ):
            process_template._install_worker_device_barrier(settings)
            worker = Worker()
            self.assertEqual(worker.init_device(), "initialized")
        self.assertEqual(events, ["wait", "init_device"])
        self.assertEqual(worker._coldsnap_process_template_wait_seconds, 1.25)

    def test_pre_device_payload_separates_context_from_device_fds(self) -> None:
        settings = process_template.ProcessTemplateSettings(
            release_file=Path("/release"),
            ready_dir=Path("/ready"),
            generation="generation-10",
            phase=process_template.PRE_DEVICE_PHASE,
        )
        worker = SimpleNamespace(
            rank=0,
            local_rank=0,
            model_config=SimpleNamespace(model="model", revision="revision"),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
        with (
            patch.object(process_template, "_cuda_initialized", return_value=False),
            patch.object(
                process_template,
                "_cuda_driver_context_present",
                return_value=False,
            ),
            patch.object(
                process_template,
                "_open_accelerator_fds",
                return_value=["/dev/nvidiactl"],
            ),
            patch.object(
                process_template,
                "_cuda_primary_context_active",
                return_value=False,
            ),
        ):
            payload = process_template._ready_payload(worker, settings)

        self.assertFalse(payload["cuda_initialized"])
        self.assertFalse(payload["cuda_driver_context_present"])
        self.assertEqual(payload["accelerator_fds"], ["/dev/nvidiactl"])
        self.assertTrue(payload["criu_cpu_only_candidate"])
        self.assertTrue(payload["criu_device_fds_require_validation"])

        with (
            patch.object(process_template, "_cuda_initialized", return_value=False),
            patch.object(
                process_template,
                "_cuda_driver_context_present",
                return_value=True,
            ),
            patch.object(
                process_template,
                "_cuda_primary_context_active",
                return_value=True,
            ),
            patch.object(process_template, "_open_accelerator_fds", return_value=[]),
        ):
            payload = process_template._ready_payload(worker, settings)
        self.assertFalse(payload["criu_cpu_only_candidate"])

    def test_cpu_only_process_audit_fails_closed_on_unknown_driver_state(self) -> None:
        with (
            patch.object(process_template, "_cuda_initialized", return_value=False),
            patch.object(
                process_template,
                "_cuda_driver_context_present",
                return_value=None,
            ),
            patch.object(
                process_template,
                "_open_accelerator_fds",
                return_value=["/dev/nvidiactl"],
            ),
            patch.object(
                process_template,
                "_cuda_primary_context_active",
                return_value=False,
            ),
        ):
            audit = process_template._cpu_only_process_audit()

        self.assertFalse(audit["criu_cpu_only_candidate"])
        self.assertTrue(audit["criu_device_fds_require_validation"])

    def test_fla_platform_probe_preloads_without_cuda_and_restores_functions(self) -> None:
        settings = process_template.ProcessTemplateSettings(
            release_file=Path("/release"),
            ready_dir=Path("/ready"),
            generation="generation-fla",
            phase=process_template.PRE_WORKER_IMPORT_PHASE,
        )
        calls: list[tuple[str, object]] = []

        def original_target():
            raise AssertionError("context-creating Triton probe must not run")

        def original_name(device=None):
            raise AssertionError(f"context-creating name probe ran for {device}")

        def original_capability(device=None):
            raise AssertionError(f"context-creating capability probe ran for {device}")

        cuda = SimpleNamespace(
            get_device_name=original_name,
            get_device_capability=original_capability,
        )
        active = SimpleNamespace(get_current_target=original_target)
        torch = SimpleNamespace(cuda=cuda)
        triton = SimpleNamespace(runtime=SimpleNamespace(driver=SimpleNamespace(active=active)))
        platform = SimpleNamespace(
            is_cuda=lambda: True,
            get_device_name=lambda device: calls.append(("name", device)) or "NVIDIA GB10",
            has_device_capability=lambda capability: (
                calls.append(("capability", capability)) or True
            ),
        )
        platforms = SimpleNamespace(current_platform=platform)
        fla = SimpleNamespace()

        def import_module(name: str):
            modules = {
                "torch": torch,
                "triton": triton,
                "vllm.platforms": platforms,
            }
            if name == process_template._FLA_UTILS_MODULE:
                self.assertEqual(active.get_current_target().backend, "cuda")
                self.assertEqual(cuda.get_device_name(0), "NVIDIA GB10")
                self.assertEqual(cuda.get_device_capability(0), (9, 0))
                return fla
            return modules[name]

        clean_audit = {
            "cuda_initialized": False,
            "cuda_driver_context_present": False,
            "cuda_primary_context_active": False,
            "accelerator_fds": ["/dev/nvidiactl"],
            "criu_cpu_only_candidate": True,
            "criu_device_fds_require_validation": True,
        }
        with (
            patch.object(
                process_template.importlib,
                "import_module",
                side_effect=import_module,
            ),
            patch.object(process_template, "_cuda_initialized", return_value=False),
            patch.object(
                process_template,
                "_cpu_only_process_audit",
                return_value=clean_audit,
            ),
        ):
            preloaded = process_template.preload_context_free_fla_platform_probe(settings)

        self.assertTrue(preloaded)
        self.assertIs(active.get_current_target, original_target)
        self.assertIs(cuda.get_device_name, original_name)
        self.assertIs(cuda.get_device_capability, original_capability)
        self.assertEqual(calls, [("name", 0), ("capability", 90)])

    def test_pre_cuda_template_fails_before_publishing_contaminated_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = process_template.ProcessTemplateSettings(
                release_file=root / "release",
                ready_dir=root / "ready",
                generation="generation-contaminated",
                phase=process_template.PRE_WORKER_IMPORT_PHASE,
            )
            payload = {
                "phase": process_template.PRE_WORKER_IMPORT_PHASE,
                "cuda_initialized": False,
                "cuda_driver_context_present": None,
                "cuda_primary_context_active": False,
                "criu_cpu_only_candidate": False,
            }

            with self.assertRaisesRegex(
                process_template.VllmContractError,
                "refusing to publish a CUDA-contaminated process template",
            ):
                process_template._wait_for_rank_release(0, settings, payload)

            self.assertFalse((root / "ready/rank-0.json").exists())

    def test_pre_cuda_template_allows_context_free_state_with_device_fds(self) -> None:
        payload = {
            "phase": process_template.PRE_DEVICE_PHASE,
            "cuda_initialized": False,
            "cuda_driver_context_present": False,
            "cuda_primary_context_active": False,
            "criu_cpu_only_candidate": True,
            "accelerator_fds": ["/dev/nvidiactl"],
            "criu_device_fds_require_validation": True,
        }

        process_template._require_cpu_only_template(payload)

    def test_pre_worker_import_hook_waits_before_class_resolution(self) -> None:
        events: list[str] = []

        class WorkerWrapperBase:
            rpc_rank = 0
            global_rank = 2

            def init_worker(self, all_kwargs) -> str:
                del all_kwargs
                events.append("resolve-worker-class")
                return "initialized"

        WorkerWrapperBase.__module__ = process_template._WORKER_BASE_MODULE
        config = SimpleNamespace(
            model_config=SimpleNamespace(model="model", revision="revision"),
            parallel_config=SimpleNamespace(tensor_parallel_size=4),
            load_config=SimpleNamespace(load_format="instanttensor"),
        )
        all_kwargs = [
            {"vllm_config": config, "rank": 2, "local_rank": 0},
        ]
        settings = process_template.ProcessTemplateSettings(
            release_file=Path("/release"),
            ready_dir=Path("/ready"),
            generation="generation-11",
            phase=process_template.PRE_WORKER_IMPORT_PHASE,
        )

        def wait(worker, resolved_settings):
            self.assertEqual(worker.rank, 2)
            self.assertEqual(worker.local_rank, 0)
            self.assertIs(resolved_settings, settings)
            events.append("wait")
            return 2.5

        module = SimpleNamespace(WorkerWrapperBase=WorkerWrapperBase)
        with patch.object(process_template, "_wait_for_release", side_effect=wait):
            process_template._install_worker_import_barrier(settings, module)
            wrapper = WorkerWrapperBase()
            self.assertEqual(wrapper.init_worker(all_kwargs), "initialized")

        self.assertEqual(events, ["wait", "resolve-worker-class"])
        self.assertEqual(wrapper._coldsnap_process_template_wait_seconds, 2.5)

    def test_pre_worker_restore_selects_restore_only_load_format(self) -> None:
        events: list[str] = []

        class WorkerWrapperBase:
            rpc_rank = 1
            global_rank = 0

            def init_worker(self, all_kwargs) -> str:
                events.append(all_kwargs[1]["vllm_config"].load_config.load_format)
                return "initialized"

        WorkerWrapperBase.__module__ = process_template._WORKER_BASE_MODULE
        load_config = SimpleNamespace(load_format="instanttensor")
        config = SimpleNamespace(
            model_config=SimpleNamespace(model="model", revision="revision"),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            load_config=load_config,
        )
        all_kwargs = [
            {"scheduler_config": SimpleNamespace()},
            {"vllm_config": config, "rank": 0, "local_rank": 0},
        ]
        settings = process_template.ProcessTemplateSettings(
            release_file=Path("/release"),
            ready_dir=Path("/ready"),
            generation="generation-restored",
            phase=process_template.PRE_WORKER_IMPORT_PHASE,
        )
        module = SimpleNamespace(WorkerWrapperBase=WorkerWrapperBase)
        with (
            patch.dict(
                os.environ,
                {process_template.RESTORE_LOAD_FORMAT_ENV: "coldsnap"},
                clear=False,
            ),
            patch.object(process_template, "_wait_for_release", return_value=0.2),
            patch.object(process_template, "_restored_generation", return_value=True),
            patch.object(
                process_template,
                "_reinitialize_driver_clients_after_restore",
                return_value=None,
            ),
        ):
            process_template._install_worker_import_barrier(settings, module)
            wrapper = WorkerWrapperBase()
            self.assertEqual(wrapper.init_worker(all_kwargs), "initialized")

        self.assertEqual(events, ["coldsnap"])
        self.assertEqual(load_config.load_format, "coldsnap")


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


RUNTIME_ENGINE = Path(__file__).resolve().parents[1] / "runtime" / "engine"
if str(RUNTIME_ENGINE) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ENGINE))

import coldsnap_engine_exec as engine_exec  # noqa: E402


class EngineExecTest(unittest.TestCase):
    def test_n580_pre_exec_boundary_is_context_free_and_restore_aware(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            marker = root / "restore-generation"
            release.write_text("capture-id\n", encoding="utf-8")
            marker.write_text("capture-id\n", encoding="utf-8")
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_PHASE": "pre_exec",
                "COLDSNAP_MODEL_LOAD_GENERATION": "capture-id",
                "COLDSNAP_MODEL_LOAD_READY_DIR": str(root / "ready"),
                "COLDSNAP_MODEL_LOAD_RELEASE_FILE": str(release),
                "COLDSNAP_MODEL_LOAD_TIMEOUT_SECONDS": "10",
                "COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE": str(marker),
                "COLDSNAP_UNIT_INDEX": "1",
                "COLDSNAP_MODEL_ID": "org/model",
                "COLDSNAP_MODEL_REVISION": "revision",
                "COLDSNAP_TP_SIZE": "2",
                "COLDSNAP_EXPORT_MODEL_PAYLOAD": "1",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(engine_exec, "_accelerator_state", return_value=([], [])),
            ):
                engine_exec._process_template_pre_exec()
                self.assertEqual(os.environ["COLDSNAP_EXPORT_MODEL_PAYLOAD"], "0")
                self.assertEqual(os.environ["COLDSNAP_PROCESS_TEMPLATE_RESTORED"], "1")

            payload = json.loads((root / "ready/rank-1.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "pre_exec")
            self.assertEqual(payload["rank"], 1)
            self.assertEqual(payload["model"], "org/model")
            self.assertTrue(payload["criu_cpu_only_candidate"])

            with patch.dict(os.environ, environment, clear=True):
                os.environ["COLDSNAP_PROCESS_TEMPLATE_RESTORED"] = "1"
                engine_exec._clear_process_template_barrier_environment()
                self.assertEqual(os.environ["COLDSNAP_PROCESS_TEMPLATE_RESTORED"], "1")
                for name in engine_exec._PROCESS_TEMPLATE_BARRIER_ENVIRONMENT:
                    self.assertNotIn(name, os.environ)

    def test_restored_vllm_pre_exec_selects_coldsnap_load_format(self) -> None:
        command = ["vllm", "serve", "org/model", "--load-format", "instanttensor"]
        restored = engine_exec._replace_load_format(command, "coldsnap")
        self.assertEqual(restored[-2:], ["--load-format", "coldsnap"])

    def test_pre_worker_import_retains_plugin_barrier_environment(self) -> None:
        environment = {
            "COLDSNAP_PROCESS_TEMPLATE_PHASE": "pre_worker_import",
            "COLDSNAP_MODEL_LOAD_GENERATION": "capture-id",
            "COLDSNAP_MODEL_LOAD_READY_DIR": "/artifact/ready",
            "COLDSNAP_MODEL_LOAD_RELEASE_FILE": "/artifact/release",
        }
        with patch.dict(os.environ, environment, clear=True):
            engine_exec._clear_process_template_barrier_environment()
            self.assertEqual(dict(os.environ), environment)

    def test_restored_pre_exec_applies_destination_transport_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restore-transport-environment.json"
            path.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "kind": "coldsnap-restore-transport-environment",
                        "unit": "unit-0",
                        "variables": {
                            "VLLM_HOST_IP": "10.24.11.13",
                            "NCCL_SOCKET_IFNAME": "enP7s7",
                        },
                    }
                ),
                encoding="utf-8",
            )
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                "COLDSNAP_RESTORE_TRANSPORT_ENVIRONMENT_PATH": str(path),
                "COLDSNAP_EXPECTED_UNIT": "unit-0",
                "VLLM_HOST_IP": "10.24.11.14",
                "NODE_IP": "10.24.11.14",
            }
            with patch.dict(os.environ, environment, clear=True):
                applied = engine_exec._apply_restore_transport_environment()
                self.assertEqual(applied, ["NCCL_SOCKET_IFNAME", "VLLM_HOST_IP"])
                self.assertEqual(os.environ["VLLM_HOST_IP"], "10.24.11.13")
                self.assertNotIn("NODE_IP", os.environ)

    def test_restored_pre_exec_applies_target_runtime_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "restore-generation"
            marker.write_text("capture-id\n", encoding="utf-8")
            path = root / "restore-runtime-environment.json"
            path.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "kind": "coldsnap-restore-runtime-environment",
                        "unit": "unit-0",
                        "variables": {
                            "COLDSNAP_DEFERRED_API_MM_WARMUP": "1",
                            "COLDSNAP_SHAPE_CALIBRATION": "0",
                            "COLDSNAP_WARMUP_GENERATION": "restore-9",
                        },
                    }
                ),
                encoding="utf-8",
            )
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                "COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE": str(marker),
                "COLDSNAP_RESTORE_RUNTIME_ENVIRONMENT_PATH": str(path),
                "COLDSNAP_EXPECTED_UNIT": "unit-0",
                "COLDSNAP_DEFERRED_API_MM_WARMUP": "0",
                "COLDSNAP_SHAPE_CALIBRATION": "1",
                "COLDSNAP_DEFERRED_WARMUP": "capture-only",
            }
            with patch.dict(os.environ, environment, clear=True):
                applied = engine_exec._apply_restore_runtime_environment()
                self.assertEqual(
                    applied,
                    [
                        "COLDSNAP_DEFERRED_API_MM_WARMUP",
                        "COLDSNAP_SHAPE_CALIBRATION",
                        "COLDSNAP_WARMUP_GENERATION",
                    ],
                )
                self.assertEqual(os.environ["COLDSNAP_DEFERRED_API_MM_WARMUP"], "1")
                self.assertEqual(os.environ["COLDSNAP_SHAPE_CALIBRATION"], "0")
                self.assertNotIn("COLDSNAP_DEFERRED_WARMUP", os.environ)

    def test_pre_exec_rejects_versioned_nvidia_driver_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_PHASE": "pre_exec",
                "COLDSNAP_MODEL_LOAD_GENERATION": "capture-id",
                "COLDSNAP_MODEL_LOAD_READY_DIR": str(root / "ready"),
                "COLDSNAP_MODEL_LOAD_RELEASE_FILE": str(root / "release"),
                "COLDSNAP_MODEL_LOAD_TIMEOUT_SECONDS": "10",
                "COLDSNAP_UNIT_INDEX": "0",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(
                    engine_exec,
                    "_accelerator_state",
                    return_value=(
                        ["/usr/lib/aarch64-linux-gnu/libcuda.so.580.173.02"],
                        [],
                    ),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "NVIDIA driver state"):
                    engine_exec._process_template_pre_exec()

    def test_capture_prefers_instanttensor_and_preserves_option_shape(self) -> None:
        with (
            patch.dict(os.environ, {engine_exec.CAPTURE_LOAD_FORMAT_ENV: "auto"}),
            patch.object(
                engine_exec,
                "_module_available",
                side_effect=lambda name: name == "instanttensor",
            ),
        ):
            command, selected = engine_exec._capture_load_format(
                ["vllm", "serve", "model", "--load-format=auto"]
            )

        self.assertEqual(selected, "instanttensor")
        self.assertIn("--load-format=instanttensor", command)

    def test_capture_falls_back_to_fastsafetensors(self) -> None:
        with (
            patch.dict(
                os.environ,
                {engine_exec.CAPTURE_LOAD_FORMAT_ENV: "instanttensor"},
            ),
            patch.object(
                engine_exec,
                "_module_available",
                side_effect=lambda name: name == "fastsafetensors",
            ),
        ):
            command, selected = engine_exec._capture_load_format(
                ["vllm", "serve", "model", "--load-format", "instanttensor"]
            )

        self.assertEqual(selected, "fastsafetensors")
        self.assertEqual(command[-2:], ["--load-format", "fastsafetensors"])

    def test_capture_rewrite_removes_shell_line_continuations_before_tokenizing(
        self,
    ) -> None:
        shell_command = [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            "vllm serve org/model \\\n  --host 0.0.0.0 \\\n  --load-format auto",
        ]

        rewritten = engine_exec._replace_load_format(shell_command, "instanttensor")

        self.assertEqual(
            shlex.split(rewritten[4]),
            [
                "vllm",
                "serve",
                "org/model",
                "--host",
                "0.0.0.0",
                "--load-format",
                "instanttensor",
            ],
        )
        self.assertNotIn("\n", shlex.split(rewritten[4]))

    def test_restored_sglang_command_uses_controller_staged_placement(self) -> None:
        command = [
            "bash",
            "-c",
            "sglang serve --port 8000 --dist-init-addr 10.24.11.14:25000",
        ]
        rewritten = engine_exec._replace_restore_placement(
            command,
            {
                "master_address": "192.168.1.43",
                "master_port": 25103,
                "http_port": 8103,
            },
        )
        self.assertEqual(
            shlex.split(rewritten[2]),
            [
                "sglang",
                "serve",
                "--port",
                "8103",
                "--dist-init-addr",
                "192.168.1.43:25103",
            ],
        )

    def test_restored_process_template_accepts_portable_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "restore-generation"
            marker.write_text("capture-id\n", encoding="utf-8")
            (root / "restore-placement.json").write_text(
                json.dumps(
                    {
                        "format": 1,
                        "kind": "coldsnap-process-template-placement",
                        "generation": "capture-id",
                        "master_address": "192.168.1.43",
                        "master_port": 25103,
                        "http_port": 8103,
                    }
                ),
                encoding="utf-8",
            )
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                "COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE": str(marker),
                "COLDSNAP_MODEL_LOAD_GENERATION": "capture-id",
            }
            with patch.dict(os.environ, environment, clear=True):
                placement = engine_exec._restored_placement()

            self.assertIsNotNone(placement)
            assert placement is not None
            self.assertEqual(placement["master_address"], "192.168.1.43")
            self.assertEqual(placement["master_port"], 25103)

    def test_capture_falls_back_to_safetensors(self) -> None:
        with (
            patch.dict(
                os.environ,
                {engine_exec.CAPTURE_LOAD_FORMAT_ENV: "coldsnap"},
            ),
            patch.object(engine_exec, "_module_available", return_value=False),
        ):
            command, selected = engine_exec._capture_load_format(["vllm", "serve", "model"])

        self.assertEqual(selected, "safetensors")
        self.assertEqual(command[-2:], ["--load-format", "safetensors"])

    def test_restore_does_not_rewrite_load_format(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            command, selected = engine_exec._capture_load_format(
                ["vllm", "serve", "model", "--load-format", "coldsnap"]
            )

        self.assertIsNone(selected)
        self.assertEqual(command[-2:], ["--load-format", "coldsnap"])

    def test_restored_sglang_native_provider_replaces_startup_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for worker in ("worker-0", "worker-1"):
                selector = root / "hydration" / worker / "activation-provider"
                selector.parent.mkdir(parents=True)
                selector.write_text("native\n", encoding="utf-8")
            environment = {
                "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                "COLDSNAP_PROCESS_ARTIFACT_ROOT": str(root),
                "COLDSNAP_EXECUTION_GRAPH": json.dumps(
                    {
                        "unit": "unit-0",
                        "by_process_slot": {
                            "0": "worker-0",
                            "1": "worker-1",
                        },
                        "groups": {},
                    }
                ),
            }
            with patch.dict(os.environ, environment, clear=True):
                self.assertEqual(engine_exec._sglang_startup_provider(), "native")

        self.assertEqual(
            engine_exec._replace_load_format(
                ["sglang", "serve", "--load-format", "safetensors"],
                "dummy",
            )[-2:],
            ["--load-format", "dummy"],
        )
        shell_command = [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            "sglang serve --model-path org/model",
        ]
        rewritten = engine_exec._replace_load_format(shell_command, "dummy")
        self.assertEqual(
            shlex.split(rewritten[4]),
            ["sglang", "serve", "--model-path", "org/model", "--load-format", "dummy"],
        )
        self.assertEqual(engine_exec._configured_load_format(rewritten), "dummy")

    def test_restored_sglang_rejects_mixed_unit_providers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for worker, provider in (
                ("worker-0", "native"),
                ("worker-1", "recovery"),
            ):
                selector = root / "hydration" / worker / "activation-provider"
                selector.parent.mkdir(parents=True)
                selector.write_text(provider + "\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "COLDSNAP_PROCESS_TEMPLATE_RESTORED": "1",
                    "COLDSNAP_PROCESS_ARTIFACT_ROOT": str(root),
                    "COLDSNAP_EXECUTION_GRAPH": json.dumps(
                        {
                            "unit": "unit-0",
                            "by_process_slot": {
                                "0": "worker-0",
                                "1": "worker-1",
                            },
                        }
                    ),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "mixed weight providers"):
                    engine_exec._sglang_startup_provider()


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "runtime"
    / "engine"
    / "coldsnap_n580_criu.py"
)
SPEC = importlib.util.spec_from_file_location("coldsnap_n580_criu", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
precuda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(precuda)


class VllmPreCudaCriuTest(unittest.TestCase):
    @staticmethod
    def _args(root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            artifact_root=root / "artifact",
            criu=Path("/opt/criu/criu/criu"),
            criu_rpc=Path("/usr/local/bin/coldsnap-criu-rpc"),
            runtime_lib_dir=Path("/opt/criu/lib"),
            target=root / "benchmarks/vllm_precuda_criu_target.py",
            plugin_root=root / "vllm_disk_sleep",
            image_id="sha256:image",
            model="Qwen/Qwen3.5-0.8B",
            model_revision="revision",
            gpu_memory_utilization=0.2,
            kv_cache_memory_bytes=256 * 1024**2,
            max_model_len=2048,
            max_num_seqs=2,
            max_num_batched_tokens=4096,
            prompt="The capital of France is",
            output_tokens=1,
            expected_text=" Paris",
            minimum_target_pid=512,
            shared_memory_dir=root / "shm",
            ghost_limit=64 * 1024**2,
            compress_block_bytes=256 * 1024,
            compress_acceleration=1,
            decompress_threads=1,
            image_io_mode="direct",
            timeout=600.0,
        )

    def test_criu_command_is_cpu_only_and_handles_process_tree_tcp(self) -> None:
        args = self._args(Path("/tmp"))
        dump_command = precuda._criu_command(args, "dump")
        restore_command = precuda._criu_command(
            args, "restore", Path("/tmp/restore-attempt")
        )

        self.assertIn("--cpu-only", dump_command)
        self.assertIn("--tcp-established", dump_command)
        self.assertIn("--network-lock", dump_command)
        self.assertNotIn("--nvidia-placeholder-fds", dump_command)
        self.assertNotIn("--cuda-checkpoint", dump_command)
        self.assertNotIn("--cuda-process-tree", dump_command)
        self.assertNotIn("--criu-plugin-dir", dump_command)
        self.assertIn("--nvidia-placeholder-fds", restore_command)
        self.assertNotIn("--network-lock", restore_command)
        self.assertIn("/tmp/restore-attempt/images", restore_command)

    def test_child_environment_is_pre_worker_and_drops_inherited_coldsnap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            args.target.parent.mkdir(parents=True)
            with patch.dict(
                precuda.os.environ,
                {
                    "COLDSNAP_CUDA_EPOCH_RUNTIME": "1",
                    "COLDSNAP_STALE_SETTING": "stale",
                    "PATH": "/bin",
                },
                clear=True,
            ):
                environment = precuda._child_environment(args, "generation")

        self.assertEqual(
            environment["COLDSNAP_PROCESS_TEMPLATE_PHASE"], "pre_worker_import"
        )
        self.assertEqual(environment["VLLM_PLUGINS"], "coldsnap")
        self.assertNotIn("COLDSNAP_CUDA_EPOCH_RUNTIME", environment)
        self.assertNotIn("COLDSNAP_STALE_SETTING", environment)

    def test_readiness_requires_explicit_context_free_evidence(self) -> None:
        args = self._args(Path("/tmp"))
        admitted = {
            "generation": "generation",
            "phase": "pre_worker_import",
            "cuda_initialized": False,
            "cuda_driver_context_present": False,
            "cuda_primary_context_active": False,
            "criu_cpu_only_candidate": True,
            "accelerator_fds": ["/dev/nvidiactl"],
        }
        precuda._validate_readiness(args, "generation", (admitted,))

        rejected = {**admitted, "cuda_driver_context_present": None}
        with self.assertRaisesRegex(RuntimeError, "not context-free"):
            precuda._validate_readiness(args, "generation", (rejected,))

        unsupported = {**admitted, "accelerator_fds": ["/dev/infiniband/uverbs0"]}
        with self.assertRaisesRegex(RuntimeError, "cannot externalize"):
            precuda._validate_readiness(args, "generation", (unsupported,))

        frontend = {
            **admitted,
            "role": "frontend",
            "cuda_initialized": True,
            "criu_cpu_only_candidate": False,
            "criu_frontend_driver_context_free_candidate": True,
        }
        precuda._validate_readiness(args, "generation", (frontend, admitted))

        contaminated_frontend = {
            **frontend,
            "cuda_primary_context_active": True,
        }
        with self.assertRaisesRegex(RuntimeError, "not context-free"):
            precuda._validate_readiness(
                args, "generation", (contaminated_frontend, admitted)
            )

    def test_regular_backing_restore_is_bounded_to_psm_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            args.artifact_root.mkdir()
            path = Path("/dev/shm/psm_deadbeef")
            manifest = args.artifact_root / "restore-regular-backings.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "files": [
                            {"path": str(path), "bytes": 17, "mode": 0o600}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(precuda.os, "open", return_value=91) as open_file,
                patch.object(precuda.os, "ftruncate") as truncate,
                patch.object(precuda.os, "fchmod"),
                patch.object(precuda.os, "close"),
            ):
                restored = precuda._restore_regular_backings(args)

            self.assertEqual(restored[0]["bytes"], 17)
            open_file.assert_called_once()
            truncate.assert_called_once_with(91, 17)

            manifest.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "files": [
                            {"path": "/tmp/not-allowed", "bytes": 0, "mode": 0o600}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                precuda._restore_regular_backings(args)

    def test_regular_backing_capture_ignores_a_vanished_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            args.artifact_root.mkdir()
            original_read_text = Path.read_text

            def read_text(path: Path, *call_args, **call_kwargs):
                if str(path) == "/proc/602/maps":
                    raise FileNotFoundError(str(path))
                return original_read_text(path, *call_args, **call_kwargs)

            with patch.object(precuda.Path, "read_text", read_text):
                records = precuda._capture_regular_backings(args, [602])

            self.assertEqual(records, [])
            self.assertEqual(
                json.loads(
                    (args.artifact_root / "restore-regular-backings.json").read_text()
                ),
                {"files": [], "format": 1},
            )

    def test_restore_attempt_clones_sealed_criu_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            images = args.artifact_root / "images"
            images.mkdir(parents=True)
            (images / "inventory.img").write_bytes(b"inventory")
            (args.artifact_root / "external-files.json").write_text(
                '{"format":1,"kind":"coldsnap-criu-external-files","files":[]}'
            )

            template = precuda._seal_criu_template(args)
            first = precuda._materialize_restore_attempt(args)
            second = precuda._materialize_restore_attempt(args)

            self.assertNotEqual(first, second)
            self.assertEqual(
                (first / "images/inventory.img").read_bytes(), b"inventory"
            )
            (first / "images/inventory.img").write_bytes(b"mutated")
            self.assertEqual(
                (template / "images/inventory.img").read_bytes(), b"inventory"
            )
            self.assertEqual(
                (second / "images/inventory.img").read_bytes(), b"inventory"
            )


if __name__ == "__main__":
    unittest.main()

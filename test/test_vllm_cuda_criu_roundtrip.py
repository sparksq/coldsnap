# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "runtime"
    / "engine"
    / "coldsnap_cuda_criu.py"
)
SPEC = importlib.util.spec_from_file_location(
    "coldsnap_cuda_criu", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
roundtrip = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roundtrip)


class VllmCudaCriuRoundtripTest(unittest.TestCase):
    def test_cgroup_io_stat_sums_all_devices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "io.stat"
            path.write_text(
                "8:0 rbytes=10 wbytes=20 rios=1 wios=2\n"
                "8:1 rbytes=30 wbytes=40 rios=3 wios=4\n",
                encoding="utf-8",
            )

            self.assertEqual(
                roundtrip._cgroup_io_stat(path),
                {"rbytes": 40, "wbytes": 60, "rios": 4, "wios": 6},
            )

    @staticmethod
    def _args(root: Path) -> SimpleNamespace:
        paths = {
            "target": root / "target.py",
            "criu": root / "criu",
            "criu_rpc": root / "coldsnap-criu-rpc",
            "cuda_checkpoint": root / "cuda-checkpoint",
        }
        for name, path in paths.items():
            path.write_bytes(name.encode())
        return SimpleNamespace(
            artifact_root=root / "artifact",
            criu=paths["criu"],
            criu_rpc=paths["criu_rpc"],
            cuda_checkpoint=paths["cuda_checkpoint"],
            target=paths["target"],
            image_id="sha256:image",
            model="Qwen/Qwen3.5-0.8B",
            model_revision="revision",
            gpu_memory_utilization=0.2,
            kv_cache_memory_bytes=256 * 1024**2,
            max_model_len=2048,
            max_num_seqs=2,
            max_num_batched_tokens=4096,
            prompt="prompt",
            output_tokens=1,
            graphs=False,
            multiprocess_engine=False,
            cuda_job_mode=False,
            shared_memory_dir=root / "shm",
            timeout=600.0,
        )

    def test_tp1_criu_command_uses_rpc_without_plugins(self) -> None:
        args = SimpleNamespace(
            criu=Path("/opt/criu/criu/criu"),
            criu_rpc=Path("/usr/local/bin/coldsnap-criu-rpc"),
            cuda_checkpoint=Path("/usr/local/bin/cuda-checkpoint"),
            artifact_root=Path("/artifact"),
            timeout=600.0,
            multiprocess_engine=False,
        )
        command = roundtrip._criu_rpc_command(args, "dump", "dump.log")
        self.assertEqual(command[0], "/usr/local/bin/coldsnap-criu-rpc")
        self.assertEqual(command[command.index("--criu") + 1], "/opt/criu/criu/criu")
        self.assertEqual(
            command[command.index("--cuda-checkpoint") + 1],
            "/usr/local/bin/cuda-checkpoint",
        )
        self.assertEqual(
            command[command.index("--external-files") + 1],
            "/artifact/external-files.json",
        )
        self.assertNotIn("--libdir", command)
        self.assertNotIn("cuda_plugin.so", command)

        args.multiprocess_engine = True
        command = roundtrip._criu_rpc_command(args, "dump", "dump.log")
        self.assertIn("--cuda-process-tree", command)
        self.assertIn("--tcp-established", command)
        self.assertEqual(command[command.index("--network-lock") + 1], "nftables")
        restore = roundtrip._criu_rpc_command(args, "restore", "restore.log")
        self.assertNotIn("--network-lock", restore)

    def test_criu_rpc_result_is_strictly_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            args.artifact_root.mkdir()
            path = args.artifact_root / "dump-rpc.json"
            path.write_text(
                '{"format":1,"kind":"coldsnap-criu-rpc-result",'
                '"action":"dump","external_file_count":3}'
            )
            self.assertEqual(
                roundtrip._criu_rpc_result(args, "dump")["external_file_count"],
                3,
            )
            path.write_text(
                '{"format":1,"kind":"coldsnap-criu-rpc-result",'
                '"action":"restore","external_file_count":3}'
            )
            with self.assertRaisesRegex(RuntimeError, "invalid CRIU RPC result"):
                roundtrip._criu_rpc_result(args, "dump")

    def test_identity_binds_runtime_binaries_without_gpu_uuid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            gpu = {
                "driver": "610.43.02",
                "name": "NVIDIA GB10",
                "compute_capability": "12.1",
            }
            with patch.object(roundtrip, "_gpu_facts", return_value=gpu):
                first = roundtrip._identity(args)
                args.target.write_text("changed")
                second = roundtrip._identity(args)
            self.assertEqual(first["gpu"], gpu)
            self.assertNotIn("uuid", first["gpu"])
            self.assertEqual(first["controller_abi"], roundtrip.CONTROLLER_ABI)
            self.assertNotIn("controller", first["runtime_sha256"])
            self.assertNotEqual(
                first["runtime_sha256"]["target"],
                second["runtime_sha256"]["target"],
            )

    def test_cuda_job_is_unique_and_persisted_in_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            args.artifact_root.mkdir()

            def launch(command, **kwargs):
                Path(command[-1]).write_bytes(b"unique-job")
                return SimpleNamespace(returncode=0)

            with patch.object(roundtrip.subprocess, "run", side_effect=launch):
                job = roundtrip._create_cuda_job(args)
            self.assertEqual(job.read_bytes(), b"unique-job")
            with self.assertRaisesRegex(FileExistsError, "reuse"):
                roundtrip._create_cuda_job(args)

    def test_cuda_job_template_resets_bytes_without_changing_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            args.artifact_root.mkdir()
            job = args.artifact_root / "cuda-checkpoint.job"
            job.write_bytes(b"capture-job-state")
            inode = job.stat().st_ino
            record = roundtrip._preserve_cuda_job(args, job)
            job.write_bytes(b"mutated-by-restore")
            self.assertEqual(roundtrip._reset_cuda_job(args, record), job)
            self.assertEqual(job.read_bytes(), b"capture-job-state")
            self.assertEqual(job.stat().st_ino, inode)

    def test_shared_memory_link_images_are_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            args.artifact_root.mkdir()
            args.shared_memory_dir.mkdir()
            source = args.shared_memory_dir / "link_remap.10"
            source.write_bytes(b"semaphore")
            self.assertEqual(roundtrip._snapshot_link_remaps(args), (1, 9))
            source.unlink()
            self.assertEqual(roundtrip._prepare_link_remaps(args), 1)
            self.assertEqual(source.read_bytes(), b"semaphore")


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "harnesses"
REPOSITORY_ROOT = BENCHMARK_ROOT.parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


target = _load(
    "nccl_cuda_checkpoint_target_test", BENCHMARK_ROOT / "nccl_cuda_checkpoint_target.py"
)
controller = _load("nccl_cuda_criu_node_test", BENCHMARK_ROOT / "nccl_cuda_criu_node.py")


class NcclCudaCheckpointTest(unittest.TestCase):
    def test_collective_values_are_rank_and_generation_specific(self) -> None:
        self.assertEqual(target._rank_value(0, 0), 1.0)
        self.assertEqual(target._rank_value(1, 1), 12.0)
        self.assertEqual(target._expected_total(2, 0), 3.0)
        self.assertEqual(target._expected_total(2, 1), 23.0)

    def test_target_command_scopes_io_uring_filter_to_child(self) -> None:
        args = SimpleNamespace(
            target=Path("/target.py"),
            rank=0,
            world_size=2,
            master_address="10.0.0.1",
            master_port=29620,
            artifact_root=Path("/snapshot"),
            timeout=60,
            allow_io_uring=False,
            nccl_checkpoint_shim=None,
            nccl_library=None,
            nccl_checkpoint_coordinator_path=None,
            require_nccl_ib_reset=False,
            cuda_graph=False,
            in_place_provider=False,
            in_place_activation=None,
            control_store="tcp",
            control_store_address_path=None,
            control_store_namespace="coldsnap-control-v1",
        )
        self.assertIn("--block-io-uring", controller._target_command(args))
        args.allow_io_uring = True
        self.assertNotIn("--block-io-uring", controller._target_command(args))
        args.nccl_checkpoint_shim = Path("/shim.so")
        args.nccl_library = Path("/libnccl.so.2")
        args.nccl_checkpoint_coordinator_path = Path("/runtime/coordinator-endpoint")
        self.assertIn("--nccl-checkpoint", controller._target_command(args))
        args.require_nccl_ib_reset = True
        self.assertIn("--require-nccl-ib-reset", controller._target_command(args))
        args.cuda_graph = True
        self.assertIn("--cuda-graph", controller._target_command(args))
        args.in_place_provider = True
        self.assertIn("--in-place-provider", controller._target_command(args))
        args.control_store = "coldsnap"
        args.control_store_address_path = Path("/runtime/control-address")
        command = controller._target_command(args)
        self.assertEqual(
            command[command.index("--control-store") + 1],
            "coldsnap",
        )
        self.assertIn("/runtime/control-address", command)
        environment = controller._target_environment(args)
        self.assertEqual(environment["LD_PRELOAD"], "/shim.so:/libnccl.so.2")
        self.assertEqual(
            environment["NCCL_CHECKPOINT_COORDINATOR_PATH"],
            "/runtime/coordinator-endpoint",
        )
        self.assertEqual(
            environment["COLDSNAP_NCCL_IN_PLACE_MODE"],
            "net-reconnect-v1",
        )
        self.assertEqual(
            environment["COLDSNAP_NCCL_IN_PLACE_ACTIVATION_PATH"],
            "/snapshot/in-place-activation",
        )

    def test_coldsnap_store_uses_native_coordinator_client(self) -> None:
        class FakeDist:
            class Store:
                pass

        with patch.object(target.CoordinatorClient, "from_path") as from_path:
            client = from_path.return_value
            store = target._coordinator_store(
                FakeDist,
                Path("/runtime/coordinator-endpoint"),
                "qualification",
                30.0,
            )
            store.set("key", b"value")
            store.add("counter", 2)
            store.append("key", b"-suffix")

        client.set.assert_called_once_with("qualification:key", b"value")
        client.add.assert_called_once_with("qualification:counter", 2)
        client.append.assert_called_once_with("qualification:key", b"-suffix")

    def test_in_place_capture_barrier_is_manager_side(self) -> None:
        args = SimpleNamespace(
            control_store_address_path=Path("/runtime/coordinator-endpoint"),
            control_store_namespace="qualification-2",
            rank=1,
            world_size=2,
            timeout=30.0,
        )
        with patch.object(controller.CoordinatorClient, "from_path") as from_path:
            client = from_path.return_value
            client.get.return_value = b"ready"
            controller._manager_barrier(args, "capture-detached")

        client.set.assert_called_once_with(
            "nccl-in-place-harness:qualification-2:capture-detached:rank-1",
            b"ready",
        )
        self.assertEqual(client.get.call_count, 2)

    def test_ib_reset_policy_is_transport_identity(self) -> None:
        self.assertIn("NCCL_NET", controller.TRANSPORT_ENVIRONMENT)
        self.assertIn("NCCL_NET_SHARED_COMMS", controller.TRANSPORT_ENVIRONMENT)
        self.assertIn("NCCL_IB_GID_INDEX", controller.TRANSPORT_ENVIRONMENT)
        self.assertIn("NCCL_IB_RELEASE_ON_FINALIZE", controller.TRANSPORT_ENVIRONMENT)
        self.assertIn("NCCL_CROSS_NIC", controller.TRANSPORT_ENVIRONMENT)
        self.assertIn("NCCL_RAS_ENABLE", controller.TRANSPORT_ENVIRONMENT)

    def test_runtime_identity_uses_current_criu_rpc_bridge(self) -> None:
        source = (BENCHMARK_ROOT / "nccl_cuda_criu_node.py").read_text()
        self.assertIn('"criu_rpc": args.criu_rpc.resolve()', source)
        self.assertIn('str(args.criu_rpc)', source)
        self.assertIn('"--external-files"', source)
        self.assertIn('"--cuda-checkpoint"', source)
        self.assertNotIn('args.plugin_dir / "cuda_plugin.so"', source)

    def test_prepare_and_restore_responses_include_provider_evidence(self) -> None:
        source = (BENCHMARK_ROOT / "nccl_cuda_checkpoint_target.py").read_text()
        self.assertEqual(
            source.count('response["nccl_checkpoint"] = checkpoint.version'),
            2,
        )

    def test_checkpoint_termination_patch_is_opt_in_and_portable(self) -> None:
        patch_source = (
            REPOSITORY_ROOT
            / "native/nccl/releases/2.31.2-1/patches/0002-checkpoint-termination.patch"
        ).read_text(encoding="utf-8")
        self.assertIn('std::strcmp(value, "destroy")', patch_source)
        self.assertIn('std::strcmp(value, "abort")', patch_source)
        self.assertIn('std::strcmp(value, "abort-sync")', patch_source)
        self.assertIn('resolveRealFunction("ncclCommAbort"', patch_source)
        self.assertIn('resolveRealFunction("ncclGroupStart"', patch_source)
        self.assertIn('resolveRealFunction("ncclGroupEnd"', patch_source)
        self.assertIn("std::map<uint64_t", patch_source)
        self.assertIn("terminationComms.emplace(commHash", patch_source)
        self.assertIn("real_group_start()", patch_source)
        self.assertIn("real_group_end()", patch_source)
        self.assertIn("COLDSNAP_NCCL_TRACE_COMMUNICATORS", patch_source)
        self.assertNotIn("ncclCheckpointCommReclaimSync", patch_source)
        self.assertNotIn("return commReclaim", patch_source)
        self.assertNotIn("Qwen", patch_source)
        self.assertNotIn("B12", patch_source)

    def test_vllm_image_builds_current_coldsnap_dlsym_bridge(self) -> None:
        bridge_source = (REPOSITORY_ROOT / "native/coldsnap_nccl_dlsym.c").read_text(
            encoding="utf-8"
        )
        dockerfile = (REPOSITORY_ROOT / "deploy/vllm/Dockerfile").read_text(
            encoding="utf-8"
        )
        dockerignore = (REPOSITORY_ROOT / ".dockerignore").read_text(
            encoding="utf-8"
        )
        self.assertIn("COLDSNAP_NCCL_CHECKPOINT_SHIM_PATH", bridge_source)
        self.assertNotIn("GPUSNAP_NCCL_CHECKPOINT_SHIM_PATH", bridge_source)
        self.assertIn("COPY native/coldsnap_nccl_dlsym.c", dockerfile)
        self.assertIn("-o /out/libcoldsnap-nccl-dlsym.so", dockerfile)
        self.assertIn(
            "COPY --from=native_builder /out/libcoldsnap-nccl-dlsym.so",
            dockerfile,
        )
        self.assertNotIn("coldsnap_nccl_dlsym.c", dockerignore)

    def test_dump_uses_nftables_but_restore_does_not_reselect_lock(self) -> None:
        args = SimpleNamespace(
            criu=Path("/criu"),
            criu_rpc=Path("/coldsnap-criu-rpc"),
            cuda_checkpoint=Path("/cuda-checkpoint"),
            artifact_root=Path("/snapshot"),
            ghost_limit=64 * 1024**2,
            network_lock="nftables",
            timeout=60,
        )
        dump = controller._criu_rpc_command(args, "dump", pid=42)
        restore = controller._criu_rpc_command(args, "restore")
        self.assertEqual(dump[dump.index("--network-lock") + 1], "nftables")
        self.assertNotIn("--network-lock", restore)
        self.assertIn("--tcp-established", dump)
        self.assertIn("--tcp-established", restore)

    def test_tcp_rows_and_conflict_preflight(self) -> None:
        table = """  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0D0B18C0:73B6 110B18C0:C6AA 01 00000000:00000000 00:00000000 00000000 0 0 12345
"""
        row = controller._tcp_rows(table, "ipv4")[0]
        self.assertEqual(row["inode"], "12345")
        self.assertEqual(row["state"], "01")
        with patch.object(controller, "_tcp_table", return_value=[row]):
            with self.assertRaisesRegex(RuntimeError, "unsafe restore"):
                controller._assert_tcp_identities_available([row])


    def test_native_make_target_owns_dlsym_bridge_output(self) -> None:
        source = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("native-nccl-dlsym:", source)
        self.assertIn("NCCL_DLSYM_LIBRARY := $(NATIVE_BUILD_DIR_ABS)/", source)
        self.assertIn("libcoldsnap-nccl-dlsym.so", source)


if __name__ == "__main__":
    unittest.main()

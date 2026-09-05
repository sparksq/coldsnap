# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import ctypes
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

import coldsnap_vllm_nccl_checkpoint as checkpoint  # noqa: E402


class _Function:
    def __init__(self, result: int = 0) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        if len(args) == 2:
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[0] = 2
            ctypes.cast(args[1], ctypes.POINTER(ctypes.c_int))[0] = 23102
        return self.result


class _NcclVersionFunction:
    def __init__(self, version: int = 23102, result: int = 0) -> None:
        self.version = version
        self.result = result

    def __call__(self, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = self.version
        return self.result


class _ProviderQueryFunction:
    def __init__(self, table: checkpoint._ProviderTable, result: int = 0) -> None:
        self.table = table
        self.result = result

    def __call__(self, _major, output):
        ctypes.cast(
            output,
            ctypes.POINTER(ctypes.POINTER(checkpoint._ProviderTable)),
        )[0] = ctypes.pointer(self.table)
        return self.result


class _InPlaceQueryFunction:
    def __init__(self, table: checkpoint._InPlaceProviderTable, result: int = 0) -> None:
        self.table = table
        self.result = result

    def __call__(self, _major, output):
        ctypes.cast(
            output,
            ctypes.POINTER(ctypes.POINTER(checkpoint._InPlaceProviderTable)),
        )[0] = ctypes.pointer(self.table)
        return self.result


class NcclCheckpointRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        checkpoint._RUNTIME = None
        environment = mock.patch.dict(
            checkpoint.os.environ,
            {
                "COLDSNAP_NCCL_PROVIDER_ID": "nccl-2.31.2-1+coldsnap.10",
                "COLDSNAP_NCCL_PROVIDER_REVISION": "10",
                "COLDSNAP_NCCL_DLSYM_BRIDGE_ABI": "1",
            },
            clear=False,
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _provider_library(
        self,
        *,
        abi_major: int = 1,
        struct_size: int | None = None,
        compiled_nccl: int = 23102,
        loaded_nccl: int = 23102,
        checkpoint_abi: int = 2,
        capability_mask: int | None = None,
        actual_nccl: int = 23102,
        prepare_result: int = 0,
        restore_result: int = 0,
        network_reset_result: int = 0,
    ):
        library = SimpleNamespace(
            ncclCheckpointGetVersion=_Function(),
            ncclGetVersion=_NcclVersionFunction(actual_nccl),
            coldsnapNcclDlsymBridgeAbi=_Function(1),
            coldsnapNcclDlsymRouteMask=_Function(63),
            coldsnapNcclDlsymRouteCount=_Function(17),
        )
        implementations = {
            "prepare": _Function(prepare_result),
            "restore": _Function(restore_result),
            "network_reset": _Function(network_reset_result),
            "network_init": _Function(),
        }
        operations = {
            name: checkpoint._ProviderOperation(implementation)
            for name, implementation in implementations.items()
        }

        def ib_status(net_refs, devices, pd_refs, mr_count):
            for output in (net_refs, devices, pd_refs, mr_count):
                output[0] = 0
            return 0

        def comm_get_real(synthetic, output):
            output[0] = synthetic
            return 0

        ib_callback = checkpoint._ProviderIbStatus(ib_status)
        comm_callback = checkpoint._ProviderCommGetReal(comm_get_real)
        if capability_mask is None:
            capability_mask = sum(checkpoint._PROVIDER_CAPABILITIES.values())
        table = checkpoint._ProviderTable(
            struct_size=(
                checkpoint._PROVIDER_REQUIRED_SIZE if struct_size is None else struct_size
            ),
            abi_major=abi_major,
            abi_minor=4,
            provider_revision=10,
            compiled_nccl_version=compiled_nccl,
            loaded_nccl_version=loaded_nccl,
            checkpoint_abi_version=checkpoint_abi,
            capability_mask=capability_mask,
            provider_id=b"nccl-2.31.2-1+coldsnap.10",
            prepare=operations["prepare"],
            restore=operations["restore"],
            network_reset=operations["network_reset"],
            network_init=operations["network_init"],
            ib_status=ib_callback,
            comm_get_real=comm_callback,
        )
        library.coldsnapNcclProviderQuery = _ProviderQueryFunction(table)
        library._provider_callbacks = [*operations.values(), ib_callback, comm_callback]
        library._provider_implementations = implementations
        library._provider_table = table
        return library

    def _in_place_library(self):
        library = self._provider_library()
        state = {"value": "active", "generation": 0}

        def transition(expected, target):
            def operation():
                if state["value"] != expected:
                    return 5
                state["value"] = target
                if target == "suspended":
                    state["generation"] += 1
                return 0

            return operation

        operations = {
            "communicator_suspend": checkpoint._InPlaceOperation(transition("active", "suspended")),
            "transport_detach": checkpoint._InPlaceOperation(
                transition("suspended", "transport-detached")
            ),
            "transport_reattach": checkpoint._InPlaceOperation(
                transition("transport-detached", "transport-reattached")
            ),
            "communicator_resume": checkpoint._InPlaceOperation(
                transition("transport-reattached", "active")
            ),
        }

        evidence_buffers = []

        def evidence():
            payload = json.dumps(
                {
                    "format": 1,
                    "kind": "coldsnap-nccl-in-place-evidence",
                    "mode": "net-reconnect-v1",
                    "portable": False,
                    "transport_epoch_portable": True,
                    "network_identity_portable": True,
                    "bootstrap_available": False,
                    "proxy_control_reconstructed": True,
                    "backend": "IB",
                    "qualified": False,
                    "state": state["value"],
                    "generation": state["generation"],
                    "communicators": 3,
                    "persistent_graph_references": 4,
                    "receive_endpoints": 8,
                    "send_endpoints": 8,
                    "transport_ownership": "coldsnap",
                }
            ).encode()
            buffer = ctypes.create_string_buffer(payload)
            evidence_buffers[:] = [buffer]
            return ctypes.addressof(buffer)

        evidence_callback = checkpoint._InPlaceEvidence(evidence)
        table = checkpoint._InPlaceProviderTable(
            struct_size=checkpoint._IN_PLACE_REQUIRED_SIZE,
            abi_major=1,
            abi_minor=0,
            communicator_suspend=operations["communicator_suspend"],
            transport_detach=operations["transport_detach"],
            transport_reattach=operations["transport_reattach"],
            communicator_resume=operations["communicator_resume"],
            evidence_json=evidence_callback,
        )
        library.coldsnapNcclInPlaceQuery = _InPlaceQueryFunction(table)
        library._in_place_table = table
        library._in_place_callbacks = [
            *operations.values(),
            evidence_callback,
            evidence_buffers,
        ]
        library._in_place_state = state
        return library

    def test_provider_abi_accepts_supported_and_larger_structures(self) -> None:
        for extra_size in (0, 64):
            library = self._provider_library(
                struct_size=checkpoint._PROVIDER_REQUIRED_SIZE + extra_size
            )
            with mock.patch.object(checkpoint.ctypes, "CDLL", return_value=library):
                runtime = checkpoint.NcclCheckpointRuntime()
            provider = runtime.version["provider"]
            self.assertEqual(provider["id"], "nccl-2.31.2-1+coldsnap.10")
            self.assertEqual(provider["abi_major"], 1)
            self.assertIn("full-network-reset", provider["capabilities"])

    def test_provider_abi_rejects_incompatible_identity(self) -> None:
        cases = (
            ({"abi_major": 2}, "ABI major mismatch"),
            ({"struct_size": checkpoint._PROVIDER_REQUIRED_SIZE - 1}, "truncated"),
            ({"compiled_nccl": 23101}, "compiled/loaded version mismatch"),
            ({"actual_nccl": 23101}, "loaded NCCL runtime identity"),
            ({"checkpoint_abi": 3}, "checkpoint ABI report"),
        )
        for options, message in cases:
            with self.subTest(message=message):
                library = self._provider_library(**options)
                with mock.patch.object(checkpoint.ctypes, "CDLL", return_value=library):
                    with self.assertRaisesRegex(checkpoint.NcclCheckpointError, message):
                        checkpoint.NcclCheckpointRuntime()

    def test_provider_abi_rejects_missing_required_capability(self) -> None:
        mask = (
            sum(checkpoint._PROVIDER_CAPABILITIES.values())
            & ~checkpoint._PROVIDER_CAPABILITIES["full-network-reset"]
        )
        library = self._provider_library(capability_mask=mask)
        with mock.patch.object(checkpoint.ctypes, "CDLL", return_value=library):
            with self.assertRaisesRegex(checkpoint.NcclCheckpointError, "full-network-reset"):
                checkpoint.NcclCheckpointRuntime()

    def test_provider_query_is_mandatory(self) -> None:
        with mock.patch.object(checkpoint.ctypes, "CDLL", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(
                checkpoint.NcclCheckpointError, "coldsnapNcclProviderQuery"
            ):
                checkpoint.NcclCheckpointRuntime()

    def test_runtime_validates_version_and_state_transitions(self) -> None:
        library = self._provider_library()
        with mock.patch.object(checkpoint.ctypes, "CDLL", return_value=library):
            runtime = checkpoint.NcclCheckpointRuntime()
        self.assertEqual(runtime.version["nccl"], 23102)
        self.assertEqual(runtime.dlsym_status(), {"route_mask": 63, "route_count": 17})
        prepared = runtime.prepare()
        self.assertEqual(prepared["status"], "prepared")
        self.assertGreaterEqual(prepared["nccl_prepare_seconds"], 0)
        self.assertGreaterEqual(prepared["nccl_network_reset_seconds"], 0)
        self.assertGreaterEqual(prepared["nccl_prepare_total_seconds"], 0)
        with self.assertRaisesRegex(checkpoint.NcclCheckpointError, "already"):
            runtime.prepare()
        restored = runtime.restore()
        self.assertEqual(restored["status"], "restored")
        self.assertGreaterEqual(restored["nccl_restore_seconds"], 0)
        self.assertGreaterEqual(restored["nccl_restore_total_seconds"], 0)
        with self.assertRaisesRegex(checkpoint.NcclCheckpointError, "not prepared"):
            runtime.restore()
        self.assertEqual(library._provider_implementations["network_reset"].calls, 1)

    def test_in_place_runtime_detaches_and_reattaches_without_destroy(self) -> None:
        library = self._in_place_library()
        environment = {
            "COLDSNAP_NCCL_IN_PLACE_MODE": "net-reconnect-v1",
            "COLDSNAP_NCCL_IN_PLACE_EXPERIMENT": "1",
        }
        with (
            mock.patch.dict(checkpoint.os.environ, environment, clear=False),
            mock.patch.object(checkpoint.ctypes, "CDLL", return_value=library),
        ):
            runtime = checkpoint.NcclCheckpointRuntime()
        prepared = runtime.prepare()
        restored = runtime.restore()
        self.assertEqual(prepared["status"], "prepared-in-place")
        self.assertEqual(prepared["in_place_evidence"]["state"], "transport-detached")
        self.assertEqual(restored["status"], "restored-in-place")
        self.assertEqual(restored["in_place_evidence"]["state"], "active")
        self.assertEqual(library._provider_implementations["prepare"].calls, 0)
        self.assertEqual(library._provider_implementations["restore"].calls, 0)
        self.assertEqual(library._provider_implementations["network_reset"].calls, 0)

    def test_runtime_reports_ib_status_when_reset_fails(self) -> None:
        library = self._provider_library(network_reset_result=3)
        with mock.patch.object(checkpoint.ctypes, "CDLL", return_value=library):
            runtime = checkpoint.NcclCheckpointRuntime()
        with self.assertRaisesRegex(
            checkpoint.NcclCheckpointError,
            r"ib_status=.*net_refs",
        ):
            runtime.prepare()

    def test_failed_restore_is_terminal_and_does_not_reenter_nccl(self) -> None:
        library = self._provider_library(restore_result=3)
        with mock.patch.object(checkpoint.ctypes, "CDLL", return_value=library):
            runtime = checkpoint.NcclCheckpointRuntime()
        runtime.prepare()
        with self.assertRaisesRegex(checkpoint.NcclCheckpointError, "ncclResult_t=3"):
            runtime.restore()
        with self.assertRaisesRegex(checkpoint.NcclCheckpointError, "terminal"):
            runtime.restore()
        self.assertEqual(library._provider_implementations["restore"].calls, 1)

    def test_runtime_waits_for_async_ib_reclaim(self) -> None:
        library = self._provider_library()
        statuses = iter(
            [
                {"net_refs": 8, "devices": 1, "pd_refs": 256, "mr_count": 768},
                {"net_refs": 8, "devices": 1, "pd_refs": 0, "mr_count": 0},
            ]
        )
        with mock.patch.object(checkpoint.ctypes, "CDLL", return_value=library):
            runtime = checkpoint.NcclCheckpointRuntime()
        with (
            mock.patch.object(runtime, "ib_status", side_effect=statuses),
            mock.patch.object(checkpoint.time, "sleep"),
        ):
            status, elapsed = runtime._wait_ib_quiescent()
        self.assertEqual(status["pd_refs"], 0)
        self.assertGreaterEqual(elapsed, 0)

    def test_runtime_unwraps_synthetic_communicator(self) -> None:
        library = self._provider_library()

        def unwrap(_synthetic, output):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 0x12340000
            return 0

        with mock.patch.object(checkpoint.ctypes, "CDLL", return_value=library):
            runtime = checkpoint.NcclCheckpointRuntime()
        with mock.patch.object(runtime, "_comm_get_real", side_effect=unwrap):
            self.assertEqual(runtime.unwrap_communicator(7), 0x12340000)
        self.assertEqual(runtime.unwrap_communicator(0), 0)

    def test_worker_wrapper_orders_vllm_and_nccl_boundaries(self) -> None:
        calls: list[str] = []
        runtime = mock.Mock()
        runtime.ib_status.side_effect = [
            {"net_refs": 8},
            {"net_refs": 4},
        ]
        runtime.prepare.side_effect = lambda: calls.append("nccl-prepare") or {}
        runtime.restore.side_effect = lambda: calls.append("nccl-restore") or {}

        def original_prepare(_worker):
            calls.append("vllm-prepare")

        def original_restore(_worker):
            calls.append("vllm-restore")

        with mock.patch.object(checkpoint, "_runtime", return_value=runtime):
            prepared = checkpoint._wrap_worker_method(original_prepare, "prepare")(object())
            restored = checkpoint._wrap_worker_method(original_restore, "restore")(object())
        self.assertEqual(
            calls,
            ["vllm-prepare", "nccl-prepare", "nccl-restore", "vllm-restore"],
        )
        self.assertGreaterEqual(prepared["engine_checkpoint_prepare_seconds"], 0)
        self.assertGreaterEqual(prepared["checkpoint_prepare_total_seconds"], 0)
        self.assertGreaterEqual(restored["engine_checkpoint_restore_seconds"], 0)
        self.assertGreaterEqual(restored["checkpoint_restore_total_seconds"], 0)

    def test_restore_applies_destination_transport_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restore-transport-environment.json"
            path.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "kind": "coldsnap-restore-transport-environment",
                        "unit": "unit-0",
                        "variables": {
                            "NCCL_IB_HCA": "destination-hca",
                            "NCCL_SOCKET_IFNAME": "destination-iface",
                            "OMPI_MCA_btl_tcp_if_include": "destination-iface",
                            "VLLM_HOST_IP": "10.0.0.2",
                        },
                    }
                ),
                encoding="utf-8",
            )
            environment = {
                "COLDSNAP_RESTORE_TRANSPORT_ENVIRONMENT_PATH": str(path),
                "COLDSNAP_EXPECTED_UNIT": "unit-0",
                "NCCL_IB_HCA": "captured-hca",
                "NCCL_DEBUG": "INFO",
                "VLLM_HOST_IP": "10.0.0.1",
                "UNRELATED": "preserved",
            }
            with mock.patch.dict(checkpoint.os.environ, environment, clear=True):
                applied = checkpoint._apply_restore_transport_environment()
                self.assertEqual(
                    applied,
                    [
                        "NCCL_IB_HCA",
                        "NCCL_SOCKET_IFNAME",
                        "OMPI_MCA_btl_tcp_if_include",
                        "VLLM_HOST_IP",
                    ],
                )
                self.assertEqual(checkpoint.os.environ["NCCL_IB_HCA"], "destination-hca")
                self.assertEqual(checkpoint.os.environ["VLLM_HOST_IP"], "10.0.0.2")
                self.assertNotIn("NCCL_DEBUG", checkpoint.os.environ)
                self.assertEqual(checkpoint.os.environ["UNRELATED"], "preserved")

    def test_worker_wrapper_reports_prepare_phase_on_failure(self) -> None:
        runtime = mock.Mock()
        runtime.ib_status.side_effect = [
            {"net_refs": 8},
            {"net_refs": 4},
        ]
        runtime.prepare.side_effect = checkpoint.NcclCheckpointError("not quiescent")

        with mock.patch.object(checkpoint, "_runtime", return_value=runtime):
            wrapped = checkpoint._wrap_worker_method(lambda _worker: None, "prepare")
            with self.assertRaisesRegex(
                checkpoint.NcclCheckpointError,
                r"not quiescent.*ib_before_vllm=.*8.*ib_after_vllm=.*4",
            ):
                wrapped(object())

    def test_destroy_recreate_provider_rejects_graph_retention_policy(self) -> None:
        environment = {
            "COLDSNAP_NCCL_CHECKPOINT": "1",
            "COLDSNAP_GRAPH_POLICY": "preserve-nccl-exec",
        }
        with mock.patch.dict("os.environ", environment, clear=False):
            with self.assertRaisesRegex(checkpoint.NcclCheckpointError, "exact in-place"):
                checkpoint.install_nccl_checkpoint_hooks()

    def test_graph_retention_hook_requires_controller_admission(self) -> None:
        environment = {
            "COLDSNAP_NCCL_CHECKPOINT": "1",
            "COLDSNAP_GRAPH_POLICY": "preserve-nccl-exec",
            "COLDSNAP_NCCL_IN_PLACE_MODE": "net-reconnect-v1",
            "COLDSNAP_NCCL_IN_PLACE_EXPERIMENT": "0",
        }
        with mock.patch.dict("os.environ", environment, clear=False):
            with self.assertRaisesRegex(checkpoint.NcclCheckpointError, "controller admission"):
                checkpoint.install_nccl_checkpoint_hooks()

    def test_worker_class_supports_bounded_upstream_rename(self) -> None:
        class Worker:
            def checkpoint_prepare(self):
                pass

            def checkpoint_restore(self):
                pass

        module = SimpleNamespace(Worker=Worker)
        with mock.patch.object(checkpoint.importlib, "import_module", return_value=module):
            self.assertIs(checkpoint._worker_class(), Worker)


if __name__ == "__main__":
    unittest.main()

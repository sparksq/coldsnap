# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Bridge vLLM's worker checkpoint boundary to NVIDIA's NCCL shim.

The upstream vLLM worker hooks prepare vLLM-owned device communicators, but
NVIDIA's process-global NCCL checkpoint shim must be called separately.  Keep
that version-sensitive bridge in one optional adapter rather than spreading it
through the benchmark controller or engine patches.
"""

from __future__ import annotations

import ctypes
import functools
import importlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable

from coldsnap_core.topology import worker_id


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


_TRANSPORT_ENVIRONMENT_NAMES = {
    "GLOO_SOCKET_IFNAME",
    "MN_IF_NAME",
    "NODE_IP",
    "TP_SOCKET_IFNAME",
    "UCX_NET_DEVICES",
    "VLLM_HOST_IP",
}
_TRANSPORT_ENVIRONMENT_PREFIXES = ("NCCL_", "OMPI_MCA_", "UCX_")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_PROVIDER_ABI_MAJOR = 1
_PROVIDER_CAPABILITIES = {
    "full-network-reset": 1 << 0,
    "ib-roce-device-release": 1 << 1,
    "ras-reset": 1 << 2,
    "synchronous-termination": 1 << 3,
    "coordinator-protocol-v1": 1 << 4,
    "private-dlopen-routing": 1 << 5,
    "communicator-unwrap": 1 << 6,
    "registration-window-replay": 1 << 7,
    "split-shrink-grow-replay": 1 << 8,
    "cuda-graph": 1 << 9,
    "nccl-device-api": 1 << 10,
    "communicator-suspend-in-place": 1 << 11,
    "transport-detach-in-place": 1 << 12,
    "graph-resource-retention": 1 << 13,
    "registration-window-in-place-replay": 1 << 14,
    "nvls-in-place-replay": 1 << 15,
    "device-api-state-retention": 1 << 16,
}
_PROVIDER_ID = re.compile(r"^nccl-[0-9]+\.[0-9]+\.[0-9]+-[0-9]+\+coldsnap\.[0-9]+$")

_ProviderOperation = ctypes.CFUNCTYPE(ctypes.c_int32)
_ProviderIbStatus = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
)
_ProviderCommGetReal = ctypes.CFUNCTYPE(
    ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
)


class _ProviderTable(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_major", ctypes.c_uint32),
        ("abi_minor", ctypes.c_uint32),
        ("provider_revision", ctypes.c_uint32),
        ("compiled_nccl_version", ctypes.c_int32),
        ("loaded_nccl_version", ctypes.c_int32),
        ("checkpoint_abi_version", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("capability_mask", ctypes.c_uint64),
        ("provider_id", ctypes.c_char_p),
        ("prepare", _ProviderOperation),
        ("restore", _ProviderOperation),
        ("network_reset", _ProviderOperation),
        ("network_init", _ProviderOperation),
        ("ib_status", _ProviderIbStatus),
        ("comm_get_real", _ProviderCommGetReal),
    ]


_PROVIDER_REQUIRED_SIZE = ctypes.sizeof(_ProviderTable)

_IN_PLACE_ABI_MAJOR = 1
_IN_PLACE_MODE = "net-reconnect-v1"
_InPlaceOperation = ctypes.CFUNCTYPE(ctypes.c_int32)
_InPlaceEvidence = ctypes.CFUNCTYPE(ctypes.c_void_p)


class _InPlaceProviderTable(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_major", ctypes.c_uint32),
        ("abi_minor", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("communicator_suspend", _InPlaceOperation),
        ("transport_detach", _InPlaceOperation),
        ("transport_reattach", _InPlaceOperation),
        ("communicator_resume", _InPlaceOperation),
        ("evidence_json", _InPlaceEvidence),
    ]


_IN_PLACE_REQUIRED_SIZE = ctypes.sizeof(_InPlaceProviderTable)


def _is_transport_environment(name: str) -> bool:
    return name in _TRANSPORT_ENVIRONMENT_NAMES or name.startswith(_TRANSPORT_ENVIRONMENT_PREFIXES)


def _apply_restore_transport_environment() -> list[str] | None:
    """Replace captured fabric variables with destination placement values."""

    raw_path = os.environ.get("COLDSNAP_RESTORE_TRANSPORT_ENVIRONMENT_PATH")
    if raw_path is None:
        return None
    path = Path(raw_path)
    try:
        if path.stat().st_size > 1024 * 1024:
            raise NcclCheckpointError("restore transport environment exceeds 1 MiB")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NcclCheckpointError(f"cannot read restore transport environment: {error}") from error
    expected_unit = os.environ.get("COLDSNAP_EXPECTED_UNIT")
    if (
        not isinstance(payload, dict)
        or payload.get("format") != 1
        or payload.get("kind") != "coldsnap-restore-transport-environment"
        or expected_unit is None
        or not expected_unit
        or payload.get("unit") != expected_unit
        or not isinstance(payload.get("variables"), dict)
    ):
        raise NcclCheckpointError("restore transport environment is invalid")
    variables = payload["variables"]
    for name, value in variables.items():
        if (
            not isinstance(name, str)
            or _ENVIRONMENT_NAME.fullmatch(name) is None
            or not _is_transport_environment(name)
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise NcclCheckpointError("restore transport environment contains an invalid variable")
    for name in tuple(os.environ):
        if _is_transport_environment(name) and name not in variables:
            del os.environ[name]
    os.environ.update(variables)
    return sorted(variables)


class NcclCheckpointError(RuntimeError):
    """Raised when the selected NCCL checkpoint runtime is unsafe or invalid."""


class NcclCheckpointRuntime:
    """Validated process-global NCCL checkpoint ABI with strict state checks."""

    def __init__(
        self,
        *,
        require_ib_reset: bool = True,
        require_network_reset: bool = True,
        required_capabilities: set[str] | None = None,
    ) -> None:
        library = ctypes.CDLL(None)
        in_place_mode = os.environ.get("COLDSNAP_NCCL_IN_PLACE_MODE", "").strip()
        if in_place_mode not in {"", _IN_PLACE_MODE}:
            raise NcclCheckpointError("unsupported NCCL in-place mode: " + repr(in_place_mode))
        provider_query = getattr(library, "coldsnapNcclProviderQuery", None)
        if provider_query is None:
            raise NcclCheckpointError("selected NCCL runtime lacks coldsnapNcclProviderQuery")
        provider_query.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.POINTER(_ProviderTable)),
        ]
        provider_query.restype = ctypes.c_int32
        table_pointer = ctypes.POINTER(_ProviderTable)()
        self._check(
            provider_query(_PROVIDER_ABI_MAJOR, ctypes.byref(table_pointer)),
            "coldsnapNcclProviderQuery",
        )
        if not table_pointer:
            raise NcclCheckpointError("NCCL provider query returned null")
        table = table_pointer.contents
        if table.struct_size < _PROVIDER_REQUIRED_SIZE:
            raise NcclCheckpointError(
                "NCCL provider structure is truncated: "
                f"size={table.struct_size} required={_PROVIDER_REQUIRED_SIZE}"
            )
        if table.abi_major != _PROVIDER_ABI_MAJOR:
            raise NcclCheckpointError(
                "NCCL provider ABI major mismatch: "
                f"reported={table.abi_major} required={_PROVIDER_ABI_MAJOR}"
            )
        if table.provider_id is None:
            raise NcclCheckpointError("NCCL provider ID is missing")
        try:
            provider_id = table.provider_id.decode("ascii")
        except UnicodeDecodeError as error:
            raise NcclCheckpointError("NCCL provider ID is not ASCII") from error
        if _PROVIDER_ID.fullmatch(provider_id) is None:
            raise NcclCheckpointError("NCCL provider ID is invalid")
        expected_provider_id = os.environ.get("COLDSNAP_NCCL_PROVIDER_ID")
        if expected_provider_id is None or provider_id != expected_provider_id:
            raise NcclCheckpointError(
                "loaded NCCL provider ID does not match the active runtime: "
                f"loaded={provider_id} expected={expected_provider_id}"
            )
        expected_provider_revision = os.environ.get("COLDSNAP_NCCL_PROVIDER_REVISION")
        if (
            expected_provider_revision is None
            or not expected_provider_revision.isdigit()
            or int(expected_provider_revision) <= 0
        ):
            raise NcclCheckpointError("expected NCCL provider revision is invalid")
        if table.provider_revision != int(expected_provider_revision):
            raise NcclCheckpointError(
                "loaded NCCL provider revision does not match the active runtime: "
                f"loaded={table.provider_revision} expected={expected_provider_revision}"
            )
        if table.compiled_nccl_version != table.loaded_nccl_version:
            raise NcclCheckpointError(
                "NCCL provider compiled/loaded version mismatch: "
                f"compiled={table.compiled_nccl_version} "
                f"loaded={table.loaded_nccl_version}"
            )

        requested = set(required_capabilities or ())
        if require_ib_reset:
            requested.add("ib-roce-device-release")
        if require_network_reset:
            requested.add("full-network-reset")
        unknown = sorted(requested.difference(_PROVIDER_CAPABILITIES))
        if unknown:
            raise NcclCheckpointError(
                "unknown required NCCL provider capabilities: " + ", ".join(unknown)
            )
        missing = sorted(
            name for name in requested if table.capability_mask & _PROVIDER_CAPABILITIES[name] == 0
        )
        if missing:
            raise NcclCheckpointError(
                "NCCL provider lacks required capabilities: " + ", ".join(missing)
            )
        for name in ("prepare", "restore", "network_reset", "network_init"):
            if not getattr(table, name):
                raise NcclCheckpointError(f"NCCL provider lacks required {name} operation")

        try:
            get_version = library.ncclCheckpointGetVersion
            get_actual_version = library.ncclGetVersion
        except AttributeError as error:
            raise NcclCheckpointError("NCCL provider lacks a required version export") from error
        get_version.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        get_version.restype = ctypes.c_int
        checkpoint_version = ctypes.c_int()
        named_nccl_version = ctypes.c_int()
        self._check(
            get_version(ctypes.byref(checkpoint_version), ctypes.byref(named_nccl_version)),
            "ncclCheckpointGetVersion",
        )
        get_actual_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_actual_version.restype = ctypes.c_int
        actual_nccl_version = ctypes.c_int()
        self._check(
            get_actual_version(ctypes.byref(actual_nccl_version)),
            "ncclGetVersion",
        )
        if checkpoint_version.value != table.checkpoint_abi_version:
            raise NcclCheckpointError(
                "NCCL checkpoint ABI report disagrees with provider: "
                f"reported={checkpoint_version.value} "
                f"provider={table.checkpoint_abi_version}"
            )
        if (
            named_nccl_version.value != table.loaded_nccl_version
            or actual_nccl_version.value != table.loaded_nccl_version
        ):
            raise NcclCheckpointError(
                "loaded NCCL runtime identity disagrees with provider: "
                f"checkpoint={named_nccl_version.value} "
                f"runtime={actual_nccl_version.value} "
                f"provider={table.loaded_nccl_version}"
            )

        self._provider_table = table_pointer
        self._prepare = table.prepare
        self._restore = table.restore
        self._network_reset = table.network_reset
        self._ib_reset = (
            table.network_reset
            if table.capability_mask & _PROVIDER_CAPABILITIES["ib-roce-device-release"]
            else None
        )
        self._ib_status = table.ib_status or None
        if self._ib_status is not None:
            self._ib_status.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            self._ib_status.restype = ctypes.c_int
        self._comm_get_real = table.comm_get_real or None
        if self._comm_get_real is not None:
            self._comm_get_real.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._comm_get_real.restype = ctypes.c_int
        self._dlsym_route_mask = getattr(library, "coldsnapNcclDlsymRouteMask", None)
        self._dlsym_route_count = getattr(library, "coldsnapNcclDlsymRouteCount", None)
        for function in (self._dlsym_route_mask, self._dlsym_route_count):
            if function is not None:
                function.argtypes = []
                function.restype = ctypes.c_uint

        self._in_place: _InPlaceProviderTable | None = None
        if in_place_mode:
            if not _env_bool("COLDSNAP_NCCL_IN_PLACE_EXPERIMENT", False):
                raise NcclCheckpointError(
                    "NCCL in-place mode requires controller admission"
                )
            query = getattr(library, "coldsnapNcclInPlaceQuery", None)
            if query is None:
                raise NcclCheckpointError(
                    "selected exact NCCL provider lacks coldsnapNcclInPlaceQuery"
                )
            query.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.POINTER(_InPlaceProviderTable)),
            ]
            query.restype = ctypes.c_int32
            pointer = ctypes.POINTER(_InPlaceProviderTable)()
            self._check(
                query(_IN_PLACE_ABI_MAJOR, ctypes.byref(pointer)),
                "coldsnapNcclInPlaceQuery",
            )
            if not pointer or pointer.contents.struct_size < _IN_PLACE_REQUIRED_SIZE:
                raise NcclCheckpointError("NCCL in-place provider ABI is truncated")
            if pointer.contents.abi_major != _IN_PLACE_ABI_MAJOR:
                raise NcclCheckpointError("NCCL in-place provider ABI major mismatch")
            for name in (
                "communicator_suspend",
                "transport_detach",
                "transport_reattach",
                "communicator_resume",
                "evidence_json",
            ):
                if not getattr(pointer.contents, name):
                    raise NcclCheckpointError(
                        f"NCCL in-place provider lacks required {name} operation"
                    )
            self._in_place = pointer.contents

        expected_bridge_abi = os.environ.get("COLDSNAP_NCCL_DLSYM_BRIDGE_ABI")
        bridge_abi_function = getattr(library, "coldsnapNcclDlsymBridgeAbi", None)
        if (
            expected_bridge_abi is None
            or not expected_bridge_abi.isdigit()
            or int(expected_bridge_abi) <= 0
            or bridge_abi_function is None
        ):
            raise NcclCheckpointError("expected NCCL dlsym bridge ABI is invalid")
        bridge_abi_function.argtypes = []
        bridge_abi_function.restype = ctypes.c_uint
        bridge_abi = int(bridge_abi_function())
        if bridge_abi != int(expected_bridge_abi):
            raise NcclCheckpointError(
                "NCCL dlsym bridge ABI mismatch: "
                f"loaded={bridge_abi} expected={expected_bridge_abi}"
            )
        known_mask = sum(_PROVIDER_CAPABILITIES.values())
        capabilities = sorted(
            name for name, bit in _PROVIDER_CAPABILITIES.items() if table.capability_mask & bit
        )
        self.version = {
            "checkpoint": checkpoint_version.value,
            "nccl": actual_nccl_version.value,
            "ib_reset": self._ib_reset is not None,
            "network_reset": True,
            "dlsym_bridge_abi": bridge_abi,
            "provider": {
                "id": provider_id,
                "revision": table.provider_revision,
                "abi_major": table.abi_major,
                "abi_minor": table.abi_minor,
                "struct_size": table.struct_size,
                "capability_mask": table.capability_mask,
                "unknown_capability_mask": table.capability_mask & ~known_mask,
                "capabilities": capabilities,
                "compiled_nccl": table.compiled_nccl_version,
                "loaded_nccl": table.loaded_nccl_version,
            },
            "in_place": {
                "enabled": self._in_place is not None,
                "mode": in_place_mode,
                "abi_major": (self._in_place.abi_major if self._in_place is not None else 0),
                "abi_minor": (self._in_place.abi_minor if self._in_place is not None else 0),
            },
        }
        self._prepared = False
        self._restore_attempted = False
        self._lock = threading.Lock()

    def ib_status(self) -> dict[str, int] | None:
        """Return checkpoint-relevant IB ownership counters when available."""

        if self._ib_status is None:
            return None
        values = [ctypes.c_int() for _ in range(4)]
        self._check(
            self._ib_status(*(ctypes.byref(value) for value in values)),
            "ncclCheckpointIbGetStatus",
        )
        return dict(
            zip(
                ("net_refs", "devices", "pd_refs", "mr_count"),
                (value.value for value in values),
                strict=True,
            )
        )

    def dlsym_status(self) -> dict[str, int] | None:
        """Return private-loader NCCL routing telemetry when available."""

        if self._dlsym_route_mask is None or self._dlsym_route_count is None:
            return None
        return {
            "route_mask": int(self._dlsym_route_mask()),
            "route_count": int(self._dlsym_route_count()),
        }

    def _wait_ib_quiescent(self) -> tuple[dict[str, int] | None, float]:
        """Wait for NCCL's asynchronous communicator reclaim to release IB."""

        initial = self.ib_status()
        if initial is None or (initial["pd_refs"] == 0 and initial["mr_count"] == 0):
            return initial, 0.0

        timeout = float(os.environ.get("COLDSNAP_NCCL_IB_QUIESCE_TIMEOUT_S", "30"))
        if timeout <= 0 or timeout > 600:
            raise NcclCheckpointError("COLDSNAP_NCCL_IB_QUIESCE_TIMEOUT_S must be in (0, 600]")
        started = time.monotonic()
        deadline = started + timeout
        status = initial
        while time.monotonic() < deadline:
            time.sleep(0.01)
            status = self.ib_status()
            if status is not None and (status["pd_refs"] == 0 and status["mr_count"] == 0):
                return status, time.monotonic() - started
        raise NcclCheckpointError(
            "NCCL IB resources did not quiesce after communicator prepare; "
            f"timeout_s={timeout}; initial={initial}; final={status}"
        )

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise NcclCheckpointError(f"{operation} failed with ncclResult_t={result}")

    def _in_place_evidence(self, expected_state: str) -> dict[str, Any]:
        if self._in_place is None:
            raise NcclCheckpointError("NCCL in-place provider is unavailable")
        raw_pointer = self._in_place.evidence_json()
        if not raw_pointer:
            raise NcclCheckpointError("NCCL in-place provider returned no evidence")
        raw = ctypes.string_at(raw_pointer)
        if len(raw) > 64 * 1024:
            raise NcclCheckpointError("NCCL in-place provider evidence is too large")
        try:
            evidence = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise NcclCheckpointError("NCCL in-place provider evidence is invalid") from error
        if (
            not isinstance(evidence, dict)
            or evidence.get("format") != 1
            or evidence.get("kind") != "coldsnap-nccl-in-place-evidence"
            or evidence.get("mode") != _IN_PLACE_MODE
            or evidence.get("state") != expected_state
            or evidence.get("transport_epoch_portable") is not True
            or evidence.get("network_identity_portable") is not True
            or evidence.get("proxy_control_reconstructed") is not True
            or evidence.get("bootstrap_available") is not False
            or evidence.get("transport_ownership") != "coldsnap"
            or not isinstance(evidence.get("communicators"), int)
            or evidence["communicators"] <= 0
            or not isinstance(evidence.get("persistent_graph_references"), int)
            or evidence["persistent_graph_references"] <= 0
            or evidence.get("backend") not in {"IB", "Socket"}
        ):
            raise NcclCheckpointError(
                "NCCL in-place provider evidence does not admit retained graphs"
            )
        # The low-level evidence bit is frozen into the exact .12 payload that
        # completed qualification. Production admission belongs to the signed
        # provider manifest, not to mutable target-process self-attestation.
        if evidence.get("qualified") is not False:
            raise NcclCheckpointError(
                "NCCL in-place provider evidence has an unexpected qualification state"
            )
        return evidence

    def prepare(self) -> dict[str, Any]:
        with self._lock:
            if self._prepared:
                raise NcclCheckpointError("NCCL checkpoint is already prepared")
            total_started = time.perf_counter()
            phase_started = time.perf_counter()
            if self._in_place is not None:
                self._check(
                    self._in_place.communicator_suspend(),
                    "ncclCommSuspend/in-place",
                )
                self._check(
                    self._in_place.transport_detach(),
                    "NCCL in-place transport detach",
                )
                evidence = self._in_place_evidence("transport-detached")
                prepare_seconds = time.perf_counter() - phase_started
                self._prepared = True
                self._restore_attempted = False
                return {
                    "status": "prepared-in-place",
                    "version": self.version,
                    "in_place_evidence": evidence,
                    "nccl_prepare_seconds": prepare_seconds,
                    "nccl_network_reset_seconds": 0.0,
                    "nccl_prepare_total_seconds": time.perf_counter() - total_started,
                }
            self._check(self._prepare(), "ncclCheckpointPrepare")
            prepare_seconds = time.perf_counter() - phase_started
            after_prepare = self.ib_status()
            before_reset, quiesce_s = self._wait_ib_quiescent()
            reset_started = time.perf_counter()
            if self._network_reset is not None:
                result = self._network_reset()
                if result != 0:
                    raise NcclCheckpointError(
                        "ncclCheckpointNetworkReset failed with "
                        f"ncclResult_t={result}; ib_status={before_reset}"
                    )
            reset_seconds = time.perf_counter() - reset_started
            self._prepared = True
            self._restore_attempted = False
            return {
                "status": "prepared",
                "version": self.version,
                "ib_after_prepare": after_prepare,
                "ib_before_reset": before_reset,
                "ib_after_reset": self.ib_status(),
                "ib_quiesce_s": quiesce_s,
                "nccl_prepare_seconds": prepare_seconds,
                "nccl_network_reset_seconds": reset_seconds,
                "nccl_prepare_total_seconds": time.perf_counter() - total_started,
            }

    def restore(self) -> dict[str, Any]:
        with self._lock:
            if not self._prepared:
                raise NcclCheckpointError("NCCL checkpoint was not prepared")
            if self._restore_attempted:
                raise NcclCheckpointError(
                    "NCCL checkpoint restore is terminal after its first attempt"
                )
            # Communicator reconstruction is not retry-safe. A failed attempt
            # may have consumed descriptors or partially rebuilt transports.
            self._restore_attempted = True
            total_started = time.perf_counter()
            phase_started = time.perf_counter()
            if self._in_place is not None:
                self._check(
                    self._in_place.transport_reattach(),
                    "NCCL in-place transport reattach",
                )
                self._check(
                    self._in_place.communicator_resume(),
                    "ncclCommResume/in-place",
                )
                evidence = self._in_place_evidence("active")
                restore_seconds = time.perf_counter() - phase_started
                self._prepared = False
                return {
                    "status": "restored-in-place",
                    "version": self.version,
                    "in_place_evidence": evidence,
                    "nccl_restore_seconds": restore_seconds,
                    "nccl_restore_total_seconds": time.perf_counter() - total_started,
                }
            self._check(self._restore(), "ncclCheckpointRestore")
            restore_seconds = time.perf_counter() - phase_started
            self._prepared = False
            return {
                "status": "restored",
                "version": self.version,
                "nccl_restore_seconds": restore_seconds,
                "nccl_restore_total_seconds": time.perf_counter() - total_started,
            }

    def unwrap_communicator(self, communicator: int) -> int:
        """Return the live real handle for an in-process NCCL extension.

        InstantTensor resolves NCCL through a private ``dlopen`` handle and
        therefore bypasses LD_PRELOAD interposition.  It may use the live real
        handle during one-time loading, before checkpoint prepare, while the
        application and shim continue retaining the stable synthetic handle.
        """

        if communicator == 0:
            return 0
        if self._comm_get_real is None:
            raise NcclCheckpointError("NCCL checkpoint runtime lacks ncclCheckpointCommGetReal")
        real = ctypes.c_void_p()
        self._check(
            self._comm_get_real(ctypes.c_void_p(communicator), ctypes.byref(real)),
            "ncclCheckpointCommGetReal",
        )
        if real.value is None:
            raise NcclCheckpointError("NCCL communicator resolved to null")
        return real.value


_RUNTIME: NcclCheckpointRuntime | None = None
_INSTALL_LOCK = threading.Lock()


def _worker_class() -> type[Any]:
    """Resolve the bounded worker-class rename across supported vLLM builds."""

    module = importlib.import_module("vllm.v1.worker.gpu_worker")
    for name in ("GPUWorker", "Worker"):
        candidate = getattr(module, name, None)
        if candidate is not None and all(
            callable(getattr(candidate, method, None))
            for method in ("checkpoint_prepare", "checkpoint_restore")
        ):
            return candidate
    raise NcclCheckpointError(
        "vLLM GPU worker exposes no checkpoint_prepare/checkpoint_restore contract"
    )


def _runtime() -> NcclCheckpointRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        required_capabilities = {
            value.strip()
            for value in os.environ.get("COLDSNAP_NCCL_REQUIRED_CAPABILITIES", "").split(",")
            if value.strip()
        }
        _RUNTIME = NcclCheckpointRuntime(
            require_ib_reset=_env_bool("COLDSNAP_NCCL_REQUIRE_IB_RESET", True),
            require_network_reset=_env_bool("COLDSNAP_NCCL_REQUIRE_NETWORK_RESET", True),
            required_capabilities=required_capabilities,
        )
    return _RUNTIME


def _wrap_worker_method(original: Callable[..., Any], operation: str) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if operation == "prepare":
            runtime = _runtime()
            ib_before_vllm = runtime.ib_status()
            total_started = time.perf_counter()
            phase_started = time.perf_counter()
            original(self, *args, **kwargs)
            engine_prepare_seconds = time.perf_counter() - phase_started
            ib_after_vllm = runtime.ib_status()
            try:
                result = runtime.prepare()
            except NcclCheckpointError as error:
                raise NcclCheckpointError(
                    f"{error}; ib_before_vllm={ib_before_vllm}; ib_after_vllm={ib_after_vllm}"
                ) from error
            return {
                **result,
                "ib_before_vllm_prepare": ib_before_vllm,
                "ib_after_vllm_prepare": ib_after_vllm,
                "engine_checkpoint_prepare_seconds": engine_prepare_seconds,
                "checkpoint_prepare_total_seconds": time.perf_counter() - total_started,
            }
        transport_environment = _apply_restore_transport_environment()
        total_started = time.perf_counter()
        result = _runtime().restore()
        phase_started = time.perf_counter()
        original(self, *args, **kwargs)
        engine_restore_seconds = time.perf_counter() - phase_started
        result = {
            **result,
            "engine_checkpoint_restore_seconds": engine_restore_seconds,
            "checkpoint_restore_total_seconds": time.perf_counter() - total_started,
        }
        if transport_environment is not None:
            result = {**result, "transport_environment": transport_environment}
        return result

    wrapped.__coldsnap_nccl_checkpoint__ = True
    return wrapped


def _worker_ib_status(_worker: Any) -> dict[str, Any]:
    """Expose read-only NCCL ownership telemetry through collective RPC."""

    runtime = _runtime()
    return {
        "worker_id": worker_id(),
        "version": runtime.version,
        "ib_status": runtime.ib_status(),
        "dlsym_status": runtime.dlsym_status(),
    }


def install_nccl_checkpoint_hooks() -> bool:
    """Install the NCCL bridge when enabled by the controller.

    NCCL supports collectives in CUDA graphs during normal execution.  The
    n610 in-place boundary retains graph-visible communicator resources while
    recreating their transport epoch.  The n580 and explicit recreate paths
    instead rebuild graph executables from the engine plan.
    """

    if not _env_bool("COLDSNAP_NCCL_CHECKPOINT", False):
        return False
    graph_policy = os.environ.get("COLDSNAP_GRAPH_POLICY", "recreate-from-plan")
    if graph_policy == "preserve-nccl-exec":
        if os.environ.get("COLDSNAP_NCCL_IN_PLACE_MODE", "").strip() != _IN_PLACE_MODE:
            raise NcclCheckpointError(
                "preserve-nccl-exec requires the exact in-place NCCL provider"
            )
        if not _env_bool("COLDSNAP_NCCL_IN_PLACE_EXPERIMENT", False):
            raise NcclCheckpointError("preserve-nccl-exec requires controller admission")
    elif graph_policy not in {"disabled", "recreate-from-plan"}:
        raise NcclCheckpointError(
            f"CUDA graph policy {graph_policy!r} requires resource retention "
            "that the selected destroy/recreate NCCL provider does not advertise"
        )

    with _INSTALL_LOCK:
        worker_class = _worker_class()
        prepare = worker_class.checkpoint_prepare
        restore = worker_class.checkpoint_restore
        if getattr(prepare, "__coldsnap_nccl_checkpoint__", False):
            return False
        worker_class.checkpoint_prepare = _wrap_worker_method(prepare, "prepare")
        worker_class.checkpoint_restore = _wrap_worker_method(restore, "restore")
        worker_class.coldsnap_nccl_ib_status = _worker_ib_status
    return True


def install_instanttensor_nccl_unwrap() -> bool:
    """Translate a synthetic communicator at InstantTensor's private ABI edge."""

    if not _env_bool("COLDSNAP_INSTANTTENSOR_NCCL_UNWRAP", False):
        return False
    if not _env_bool("COLDSNAP_NCCL_CHECKPOINT", False):
        raise NcclCheckpointError(
            "InstantTensor NCCL unwrapping requires COLDSNAP_NCCL_CHECKPOINT=1"
        )

    import instanttensor

    extension = instanttensor._C
    original = extension.open
    if getattr(original, "__coldsnap_nccl_unwrap__", False):
        return False

    @functools.wraps(original)
    def wrapped(
        filename: Any,
        device_idx: Any,
        nccl_communicator: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        communicator = _runtime().unwrap_communicator(nccl_communicator)
        return original(filename, device_idx, communicator, *args, **kwargs)

    wrapped.__coldsnap_nccl_unwrap__ = True
    extension.open = wrapped
    return True

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Environment-owned configuration for the out-of-tree SGLang plugin."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


class SettingsError(ValueError):
    pass


def _boolean(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Settings:
    mode: str
    artifact_root: Path | None
    live_backing: str
    live_root: Path | None
    preverified: bool
    lock_root: Path | None = None
    export_model_payload: bool = False
    process_artifact_root: Path | None = None
    async_graphs: bool = False
    async_graph_arm_file: Path | None = None
    async_graph_ready_file: Path | None = None
    shape_calibration: bool = False
    startup_provider: str = ""

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.environ.get("COLDSNAP_MODE", "off").strip().lower()
        if mode not in {"off", "capture"}:
            raise SettingsError("COLDSNAP_MODE must be off or capture")
        artifact_value = os.environ.get("COLDSNAP_ARTIFACT_DIR")
        artifact_root = Path(artifact_value).resolve() if artifact_value else None
        if mode != "off" and artifact_root is None:
            raise SettingsError("COLDSNAP_ARTIFACT_DIR is required when COLDSNAP_MODE is enabled")
        live_backing = os.environ.get("COLDSNAP_LIVE_BACKING", "disk").strip().lower()
        if live_backing not in {"disk", "cpu", "discard"}:
            raise SettingsError("COLDSNAP_LIVE_BACKING must be disk, cpu, or discard")
        runtime_value = os.environ.get("COLDSNAP_RUNTIME_DIR")
        runtime_root = (
            Path(runtime_value).resolve()
            if runtime_value
            else Path(os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir()))
            / f"coldsnap-sglang-{os.getuid()}"
        )
        live_root = runtime_root / "live" if mode != "off" else None
        lock_root = runtime_root / "locks" if mode != "off" else None
        process_root_value = os.environ.get("COLDSNAP_PROCESS_ARTIFACT_ROOT")
        process_artifact_root = Path(process_root_value).resolve() if process_root_value else None
        if mode != "off" and process_artifact_root is None:
            raise SettingsError(
                "COLDSNAP_PROCESS_ARTIFACT_ROOT is required when the SGLang plugin is enabled"
            )
        async_graphs = _boolean("COLDSNAP_ASYNC_CUDA_GRAPHS")
        arm_value = os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE")
        ready_value = os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS_READY_FILE")
        async_graph_arm_file = Path(arm_value).resolve() if arm_value else None
        async_graph_ready_file = Path(ready_value).resolve() if ready_value else None
        if async_graphs and (async_graph_arm_file is None or async_graph_ready_file is None):
            raise SettingsError("asynchronous CUDA graphs require arm and ready file paths")
        startup_provider = os.environ.get("COLDSNAP_SGLANG_STARTUP_PROVIDER", "").strip().lower()
        if startup_provider not in {"", "native", "recovery"}:
            raise SettingsError("COLDSNAP_SGLANG_STARTUP_PROVIDER must be native or recovery")
        return cls(
            mode=mode,
            artifact_root=artifact_root,
            live_backing=live_backing,
            live_root=live_root,
            preverified=_boolean("COLDSNAP_ARTIFACT_PREVERIFIED"),
            lock_root=lock_root,
            export_model_payload=_boolean("COLDSNAP_EXPORT_MODEL_PAYLOAD"),
            process_artifact_root=process_artifact_root,
            async_graphs=async_graphs,
            async_graph_arm_file=async_graph_arm_file,
            async_graph_ready_file=async_graph_ready_file,
            shape_calibration=_boolean("COLDSNAP_SHAPE_CALIBRATION"),
            startup_provider=startup_provider,
        )

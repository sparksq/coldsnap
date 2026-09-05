# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""SGLang-owned CUDA graph shape coverage for capture artifacts.

SGLang's startup graph builders already run the engine's warmup/capture loop.
ColdSnap verifies that every planned batch size produced an executable, records
the coverage, and later discards those executables before the process dump.
This compiles the shape corpus in the captured process without reconstructing
SGLang's forward inputs or borrowing vLLM internals.
"""

from __future__ import annotations

import importlib.metadata
import os
import time
from pathlib import Path
from typing import Any


class SGLangShapeCalibrationError(RuntimeError):
    pass


def _shape_integer(value: Any) -> int | None:
    """Return the capture size represented by an engine-owned shape key."""
    candidate = getattr(value, "size", value)
    try:
        shape = int(candidate)
    except (TypeError, ValueError):
        return None
    return shape if shape > 0 else None


def _integer_shapes(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, dict):
        values = value.keys()
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        return []
    result: list[int] = []
    for item in values:
        shape = _shape_integer(item)
        if shape is None:
            continue
        if shape not in result:
            result.append(shape)
    return sorted(result)


def _planned_shapes(graph_runner: Any) -> list[int]:
    for owner in (graph_runner, getattr(graph_runner, "backend", None)):
        if owner is None:
            continue
        for name in ("capture_bs", "capture_batch_sizes", "capture_sizes"):
            shapes = _integer_shapes(getattr(owner, name, None))
            if shapes:
                return shapes
    raise SGLangShapeCalibrationError(
        "SGLang CUDA graph runner exposes no recognized capture batch-size plan"
    )


def _captured_shapes(graph_runner: Any) -> list[int]:
    for owner in (graph_runner, getattr(graph_runner, "backend", None)):
        if owner is None:
            continue
        for name in ("graphs", "_graphs", "graph_runners", "graph_executables"):
            shapes = _integer_shapes(getattr(owner, name, None))
            if shapes:
                return shapes
    raise SGLangShapeCalibrationError(
        "SGLang CUDA graph runner exposes no recognized captured-shape inventory"
    )


def _phase_enabled(model_runner: Any, phase: str) -> bool:
    """Use SGLang's typed phase configuration to exclude eager fallbacks."""
    server_args = getattr(model_runner, "server_args", None)
    graph_config = getattr(server_args, "cuda_graph_config", None)
    phase_config = getattr(graph_config, phase, None)
    backend = getattr(phase_config, "backend", None)
    if backend is None:
        return True
    value = getattr(backend, "value", backend)
    return str(value).lower() != "disabled"


def _toolchain() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for package in ("sglang", "torch", "triton"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unknown"
    try:
        import torch

        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        result["cuda_architecture"] = f"sm_{major}{minor}"
        result["cuda"] = str(getattr(torch.version, "cuda", "unknown"))
    except Exception:
        result["cuda_architecture"] = "unknown"
        result["cuda"] = "unknown"
    return result


def calibrate_capture_shapes(model_runner: Any) -> dict[str, Any]:
    started = time.perf_counter()
    modes: list[dict[str, Any]] = []
    for phase in ("decode", "prefill"):
        if not _phase_enabled(model_runner, phase):
            continue
        graph_runner = getattr(model_runner, f"{phase}_cuda_graph_runner", None)
        if graph_runner is None:
            continue
        planned = _planned_shapes(graph_runner)
        captured = _captured_shapes(graph_runner)
        missing = sorted(set(planned) - set(captured))
        if missing:
            raise SGLangShapeCalibrationError(
                f"SGLang {phase} CUDA graph coverage omitted planned shapes {missing}"
            )
        modes.append(
            {
                "mode": phase,
                "planned_shapes": len(planned),
                "warmed_shapes": len(planned),
                "batch_sizes": planned,
                "captured_batch_sizes": captured,
            }
        )
    if not modes:
        raise SGLangShapeCalibrationError(
            "SGLang shape calibration found no enabled CUDA graph phase"
        )
    planned_total = sum(int(mode["planned_shapes"]) for mode in modes)
    warmed_total = sum(int(mode["warmed_shapes"]) for mode in modes)
    if planned_total != warmed_total:
        raise SGLangShapeCalibrationError(
            f"SGLang warmed {warmed_total} of {planned_total} planned shapes"
        )
    return {
        "kind": "coldsnap-shape-calibration",
        "schema": 1,
        "engine": "sglang",
        "strategy": "verify-engine-startup-capture",
        "modes": modes,
        "planned_shapes": planned_total,
        "warmed_shapes": warmed_total,
        "seconds": time.perf_counter() - started,
        "toolchain": _toolchain(),
        "cache_root": str(
            Path(os.environ.get("COLDSNAP_RUNTIME_CACHE_ROOT", "/var/cache/coldsnap/runtime"))
        ),
    }

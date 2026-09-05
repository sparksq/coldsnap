# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Compile-cache calibration over the shapes CUDA graph capture will use.

A CRIU-restored process cannot use compile-cache entries that were written to
disk after its snapshot. Two DS4F TP2 activations of one artifact reported the
same five `disk-cache-miss` events for the same kernel keys, and the second
activation missed them even though the first had already compiled and written
all five to the shared cache. Presence on disk is therefore not sufficient; the
kernels have to be compiled inside the process that gets captured, before the
dump, so the compiled state travels in the snapshot.

vLLM warms each capture descriptor with eager `_dummy_run` calls before
recording it (`GPUModelRunner._warmup_and_capture`). Running only that warmup
half compiles the same kernel variants while creating no graph executable, so
nothing retains NCCL communicator resources and the checkpoint boundary that
the arm-gated artifact depends on is unaffected.

The missing variants are shape-derived rather than new kernels. On DS4F they
were three `SparseNSAPagedStreamLogitsKernel` keys differing only in
`persistent_ctas` and two `SparseNSAFusedIndexerKernel` keys differing in
`ctas_per_group` and `pack_cap`.
"""

from __future__ import annotations

import inspect
import importlib.metadata
import logging
import os
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from coldsnap_vllm import VllmContractError


SHAPE_CALIBRATION_ENV = "COLDSNAP_SHAPE_CALIBRATION"

logger = logging.getLogger(__name__)


def _coverage_metadata() -> dict[str, Any]:
    toolchain: dict[str, str] = {}
    for package in ("vllm", "torch", "triton"):
        try:
            toolchain[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            toolchain[package] = "unknown"
    try:
        import torch

        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        toolchain["cuda_architecture"] = f"sm_{major}{minor}"
        toolchain["cuda"] = str(getattr(torch.version, "cuda", "unknown"))
    except Exception:
        toolchain["cuda_architecture"] = "unknown"
        toolchain["cuda"] = "unknown"
    return {
        "engine": "vllm",
        "toolchain": toolchain,
        "cache_root": str(
            Path(os.environ.get("COLDSNAP_RUNTIME_CACHE_ROOT", "/var/cache/coldsnap/runtime"))
        ),
    }


def shape_calibration_enabled(values: dict[str, str] | None = None) -> bool:
    source = os.environ if values is None else values
    return source.get(SHAPE_CALIBRATION_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def runner_generation(runner: Any) -> str:
    """Which model-runner capture API this vLLM build exposes.

    vLLM ships two runners side by side. The v1 runner drives capture from a
    `cudagraph_dispatcher` and warms shapes through `_dummy_run`; the v2 runner
    (`use_v2_model_runner=True`, which DS4F selects) owns capture inside a
    `CudaGraphManager` and warms through a per-descriptor forward function.
    Both must be supported, and an unrecognised runner must fail rather than be
    silently skipped.
    """
    if getattr(runner, "cudagraph_manager", None) is not None:
        return "v2"
    if getattr(runner, "cudagraph_dispatcher", None) is not None:
        return "v1"
    raise VllmContractError(
        "model runner exposes neither cudagraph_manager (v2) nor "
        "cudagraph_dispatcher (v1); cannot enumerate capture shapes"
    )


def _capture_plan(runner: Any) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
    """The (runtime mode, descriptors) pairs graph capture would record."""
    dispatcher = getattr(runner, "cudagraph_dispatcher", None)
    if dispatcher is None:
        raise VllmContractError("model runner lacks cudagraph_dispatcher")
    describe = getattr(dispatcher, "get_capture_descs", None)
    if not callable(describe):
        raise VllmContractError(
            "cudagraph_dispatcher lacks get_capture_descs"
        )
    plan: list[tuple[Any, tuple[Any, ...]]] = []
    for entry in describe():
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise VllmContractError(
                "get_capture_descs must yield (runtime_mode, descriptors)"
            )
        mode, descriptors = entry
        plan.append((mode, tuple(descriptors)))
    return tuple(plan)


def _warmup_count(runner: Any) -> int:
    """Match vLLM's own warmup depth, but never skip the shape entirely."""
    compilation_config = getattr(runner, "compilation_config", None)
    configured = getattr(compilation_config, "cudagraph_num_of_warmups", None)
    try:
        count = int(configured)
    except (TypeError, ValueError):
        count = 0
    return max(count, 1)


def _dummy_run_arguments(
    runner: Any,
    descriptor: Any,
    *,
    eager_mode: Any,
    force_attention: bool,
) -> dict[str, Any]:
    """Bind only the parameters this vLLM's ``_dummy_run`` actually accepts.

    The capture path passes several tuning arguments that have appeared and
    moved across releases. Binding structurally keeps an upstream signature
    change a startup failure rather than a silently different warmup.
    """
    dummy_run = getattr(runner, "_dummy_run", None)
    if not callable(dummy_run):
        raise VllmContractError("model runner lacks _dummy_run")
    try:
        parameters = inspect.signature(dummy_run).parameters
    except (TypeError, ValueError) as error:
        raise VllmContractError("cannot inspect _dummy_run") from error

    num_tokens = getattr(descriptor, "num_tokens", None)
    if num_tokens is None:
        raise VllmContractError("capture descriptor lacks num_tokens")

    candidates = {
        "cudagraph_runtime_mode": eager_mode,
        "force_attention": force_attention,
        "uniform_decode": getattr(descriptor, "uniform", False),
        "allow_microbatching": False,
        "skip_eplb": True,
        "remove_lora": False,
        "num_active_loras": getattr(descriptor, "num_active_loras", None),
    }
    arguments = {
        name: value
        for name, value in candidates.items()
        if name in parameters and value is not None
    }
    # The eager-mode argument is the one thing that must not be dropped: without
    # it _dummy_run could record a graph instead of warming the kernels.
    if "cudagraph_runtime_mode" not in arguments:
        raise VllmContractError(
            "_dummy_run does not accept cudagraph_runtime_mode; refusing to "
            "calibrate shapes without a guaranteed eager run"
        )
    arguments["num_tokens"] = num_tokens
    return arguments


#: The order vLLM's own capture loop uses. Anything unrecognised runs last so a
#: new mode is still calibrated rather than silently skipped.
CAPTURE_MODE_ORDER = ("PIECEWISE", "FULL")


def _ordered_capture_descs(
    descs_by_mode: dict[Any, Any],
) -> list[tuple[Any, Any]]:
    def rank(item: tuple[Any, Any]) -> int:
        name = getattr(item[0], "name", str(item[0]))
        try:
            return CAPTURE_MODE_ORDER.index(name)
        except ValueError:
            return len(CAPTURE_MODE_ORDER)

    return sorted(descs_by_mode.items(), key=rank)


def _capture_owner(manager: Any) -> type[Any]:
    """The class in the manager's MRO that owns the descriptor loop.

    `ModelCudaGraphManager.capture` prepares a per-descriptor forward function
    and delegates to its base, which is where the warmup and the recording are
    interleaved. Intercepting the base gives the loop and the forward factory
    without reconstructing any model plumbing.
    """
    for owner in type(manager).__mro__:
        candidate = owner.__dict__.get("capture")
        if candidate is None:
            continue
        try:
            parameters = list(inspect.signature(candidate).parameters)
        except (TypeError, ValueError):
            continue
        if len(parameters) >= 2 and parameters[1] == "create_forward_fn":
            return owner
    raise VllmContractError(
        "no CudaGraphManager base accepts create_forward_fn; the v2 capture "
        "loop has moved"
    )


def _stable_descriptor_value(value: Any) -> Any:
    """Turn descriptor fields into an identity-independent comparison value."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return (type(value).__module__, type(value).__qualname__, name)
    if isinstance(value, (tuple, list)):
        return tuple(_stable_descriptor_value(item) for item in value)
    raise VllmContractError(
        "capture descriptor contains an unsupported semantic field: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _descriptor_signature(mode: Any, descriptor: Any) -> tuple[Any, ...]:
    """Describe one capture shape without relying on Python object identity.

    The v2 speculative runner may rebuild equivalent frozen dataclass
    descriptors between capture passes.  Object IDs therefore over-count the
    shape corpus even though the semantic descriptor set is unchanged.
    """
    if is_dataclass(descriptor) and not isinstance(descriptor, type):
        names = tuple(field.name for field in fields(descriptor))
    else:
        try:
            names = tuple(
                sorted(name for name in vars(descriptor) if not name.startswith("_"))
            )
        except TypeError as error:
            raise VllmContractError(
                "capture descriptor exposes no semantic fields"
            ) from error
    if not names:
        raise VllmContractError("capture descriptor exposes no semantic fields")
    return (
        getattr(mode, "name", str(mode)),
        type(descriptor).__module__,
        type(descriptor).__qualname__,
        tuple(
            (name, _stable_descriptor_value(getattr(descriptor, name)))
            for name in names
        ),
    )


def _calibrate_v2(runner: Any) -> dict[str, Any]:
    """Run only the warmup half of the v2 manager's capture loop.

    vLLM's loop is `create_forward_fn(desc, warmup=True)` followed by
    `forward_fn(CUDAGraphMode.NONE)` and then the recording. Replacing the
    whole loop with just those two calls compiles the same kernels for the same
    descriptors while producing no graph executable, so the artifact keeps its
    zero-graph-allocation property and nothing retains NCCL resources.
    """
    manager = runner.cudagraph_manager
    descs_by_mode = getattr(manager, "_capture_descs", None)
    if not isinstance(descs_by_mode, dict):
        raise VllmContractError(
            "v2 CudaGraphManager lacks a _capture_descs mapping"
        )
    capture_model = getattr(runner, "capture_model", None)
    if not callable(capture_model):
        raise VllmContractError("v2 model runner lacks capture_model")

    initial_keys = {
        _descriptor_signature(mode, desc)
        for mode, descs in descs_by_mode.items()
        for desc in descs
    }
    owner = _capture_owner(manager)
    original = owner.__dict__["capture"]
    warmed: list[str] = []
    observed: dict[tuple[Any, ...], tuple[str, int]] = {}
    warmed_keys: set[tuple[Any, ...]] = set()

    def warmup_only(
        self: Any,
        create_forward_fn: Any,
        *,
        channel_id: str,
        progress_bar_desc: str = "",
    ) -> None:
        del channel_id, progress_bar_desc
        # vLLM captures PIECEWISE before FULL because PIECEWISE has the larger
        # activations, and the manager keeps a persistent hidden-state buffer
        # sized by whichever descriptor runs first. Iterating the mapping in
        # dict order instead can size that buffer too small and fail a later,
        # wider descriptor, so the upstream order is reproduced exactly.
        for mode, descs in _ordered_capture_descs(self._capture_descs):
            for desc in descs:
                mode_name = getattr(mode, "name", str(mode))
                signature = _descriptor_signature(mode, desc)
                observed[signature] = (mode_name, int(getattr(desc, "num_tokens", 0)))
                eager = getattr(type(getattr(desc, "cg_mode", mode)), "NONE", None)
                if eager is None:
                    raise VllmContractError(
                        "capture descriptor mode has no NONE member"
                    )
                forward_fn = create_forward_fn(desc, warmup=True)
                forward_fn(eager)
                warmed.append(mode_name)
                warmed_keys.add(signature)
        # Deliberately leave _graphs_captured untouched: nothing was recorded,
        # so a later real capture must still see work to do.

    started = time.perf_counter()
    owner.capture = warmup_only
    try:
        capture_model()
    finally:
        owner.capture = original
        # Drop any state the pass left behind so the activation's real capture
        # starts from a clean manager.
        clear = getattr(manager, "clear", None)
        if callable(clear):
            clear()

    expected_keys = initial_keys | set(observed)
    missing = expected_keys - warmed_keys
    if missing:
        raise VllmContractError(
            "shape calibration coverage differs from the planned descriptor set: "
            f"warmed {len(warmed_keys)} of {len(expected_keys)}, "
            f"missing={len(missing)}"
        )
    dynamic_shapes = len(expected_keys - initial_keys)
    observed_by_mode: dict[str, list[int]] = {}
    for mode_name, num_tokens in observed.values():
        observed_by_mode.setdefault(mode_name, []).append(num_tokens)
    return {
        "kind": "coldsnap-shape-calibration",
        "schema": 1,
        "runner": "v2",
        "modes": [
            {
                "mode": name,
                "shapes": len(observed_by_mode.get(name, [])),
                "planned_shapes": len(observed_by_mode.get(name, [])),
                "num_tokens": sorted(observed_by_mode.get(name, [])),
            }
            for name in sorted(observed_by_mode)
        ],
        "shapes": len(warmed_keys),
        "warmed_shapes": len(warmed_keys),
        "planned_shapes": len(expected_keys),
        "initial_planned_shapes": len(initial_keys),
        "dynamic_shapes": dynamic_shapes,
        "warmups_per_shape": 1,
        "warmup_invocations": len(warmed),
        "seconds": time.perf_counter() - started,
        **_coverage_metadata(),
    }


def calibrate_capture_shapes(runner: Any) -> dict[str, Any]:
    """Compile every capture shape eagerly, recording nothing.

    Returns telemetry describing what was exercised so an artifact can prove
    the corpus covered the graph-capture envelope.
    """
    if runner_generation(runner) == "v2":
        result = _calibrate_v2(runner)
        logger.info(
            "Calibrated %d/%d v2 CUDA graph capture shapes eagerly in %.3f s; "
            "no graph executable was created",
            result["shapes"],
            result["planned_shapes"],
            result["seconds"],
        )
        return result
    plan = _capture_plan(runner)
    warmups = _warmup_count(runner)
    started = time.perf_counter()
    modes: list[dict[str, Any]] = []
    total_shapes = 0
    for mode, descriptors in plan:
        if not descriptors:
            continue
        mode_name = getattr(mode, "name", str(mode))
        eager_mode = getattr(type(mode), "NONE", None)
        if eager_mode is None:
            raise VllmContractError(
                f"cudagraph runtime mode {mode_name!r} has no NONE member"
            )
        # vLLM forces the attention path for FULL capture; the DS4F misses were
        # all attention kernels, so this flag is what makes them compile.
        force_attention = mode_name == "FULL"
        phase = time.perf_counter()
        for descriptor in descriptors:
            arguments = _dummy_run_arguments(
                runner,
                descriptor,
                eager_mode=eager_mode,
                force_attention=force_attention,
            )
            for _ in range(warmups):
                runner._dummy_run(**arguments)
        modes.append(
            {
                "mode": mode_name,
                "shapes": len(descriptors),
                "force_attention": force_attention,
                "seconds": time.perf_counter() - phase,
                "num_tokens": [
                    int(getattr(descriptor, "num_tokens", 0))
                    for descriptor in descriptors
                ],
            }
        )
        total_shapes += len(descriptors)

    result = {
        "kind": "coldsnap-shape-calibration",
        "schema": 1,
        "runner": "v1",
        "modes": modes,
        "shapes": total_shapes,
        "warmed_shapes": total_shapes,
        "planned_shapes": total_shapes,
        "warmups_per_shape": warmups,
        "seconds": time.perf_counter() - started,
        **_coverage_metadata(),
    }
    logger.info(
        "Calibrated %d v1 CUDA graph capture shapes eagerly in %.3f s; "
        "no graph executable was created",
        total_shapes,
        result["seconds"],
    )
    return result

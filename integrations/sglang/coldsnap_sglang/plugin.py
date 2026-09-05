# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""ColdSnap SGLang process-snapshot registration and lifecycle hooks."""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coldsnap_core.artifact import ArtifactMetrics

from .adapter import create_adapter
from .compat import (
    contract_mode,
    install_failure_sentinel,
    resolve_weight_updater_contract,
    validate_contracts,
)
from .identity import artifact_for, build_identity, validate_model_config
from .settings import Settings


logger = logging.getLogger(__name__)
ARTIFACT_CHUNK_BYTES = 256 * 1024**2
OPTIONAL_PARAMETER_STATE_MAX_BYTES = 64 * 1024**2
_RECOVERY_WEIGHTS_RESUME_TAG = "coldsnap_recovery_weights"
_RECOVERY_RUNTIME_RESUME_TAG = "coldsnap_recovery_runtime"
_RECOVERY_RESUME_TAGS = {
    _RECOVERY_WEIGHTS_RESUME_TAG,
    _RECOVERY_RUNTIME_RESUME_TAG,
}


@dataclass(frozen=True)
class _ArtifactContext:
    identity: dict[str, Any]
    artifact: Any


@dataclass(frozen=True)
class _GraphRecaptureStep:
    owner: Any
    callback: Any
    attributes: tuple[str, ...]


_settings: Settings | None = None
_memory_adapter: Any | None = None
_pending_models: list[tuple[Any, dict[str, Any], Any]] = []
_nccl_runtime_instance: Any | None = None
_graph_recapture_plan: list[_GraphRecaptureStep] = []
_async_graph_eager_steps = 0
_async_graph_phase = "disabled"
_async_graph_error = ""
_shape_calibration: dict[str, Any] | None = None
_restored_weight_update_pending = False
_recovery_resume_observation: dict[str, Any] | None = None
_checkpoint_optional_parameters: list[tuple[Any, str, Any]] = []


def _speculative_enabled(owner: Any) -> bool:
    server_args = getattr(owner, "server_args", None)
    for name in ("speculative_algorithm", "spec_algorithm"):
        value = getattr(server_args, name, None)
        value = getattr(value, "value", value)
        if value is not None and str(value).strip().lower() not in {
            "",
            "none",
            "disabled",
            "false",
            "0",
        }:
            return True
    return False


def _async_graphs_enabled(owner: Any | None = None) -> bool:
    if _settings is None or not _settings.async_graphs:
        return False
    # Target-only graph capture is qualified first. Speculative workers have
    # additional graph ownership and scheduling dependencies; retain the
    # existing synchronous path until that capability is separately proven.
    if owner is not None and (
        getattr(owner, "draft_worker", None) is not None or _speculative_enabled(owner)
    ):
        return False
    return True


def _retained_nccl_graphs() -> bool:
    return os.environ.get("COLDSNAP_GRAPH_POLICY", "").strip() == "preserve-nccl-exec"


def _publish_async_graph_status(phase: str, error: str = "") -> None:
    if _settings is None or _settings.async_graph_ready_file is None:
        return
    worker = ""
    try:
        from coldsnap_core.topology import worker_id

        worker = worker_id()
    except Exception:
        pass
    path = _settings.async_graph_ready_file
    path.parent.mkdir(parents=True, exist_ok=True)
    world_size = int(os.environ.get("COLDSNAP_WORLD_SIZE", "1"))
    requested = os.environ.get(
        "COLDSNAP_GRAPH_POLICY_REQUESTED", "recreate-from-plan"
    )
    effective = os.environ.get("COLDSNAP_GRAPH_POLICY", "recreate-from-plan")
    decision = "requested_policy_is_supported"
    if requested == "preserve-nccl-exec":
        decision = (
            "experimental_exact_nccl_in_place"
            if effective == "preserve-nccl-exec"
            else "in_place_nccl_graph_resource_retention_is_unqualified"
        )
    elif requested == "preserve-exec":
        decision = (
            "provider_reconstructs_communicator"
            if world_size > 1
            else "graph_exec_preservation_is_unqualified"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    value = {
        "format": 1,
        "kind": "coldsnap-sglang-async-cuda-graphs",
        "capture_id": os.environ.get("COLDSNAP_CAPTURE_ID", ""),
        "generation": os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS_GENERATION", ""),
        "worker_id": worker,
        "pid": os.getpid(),
        "phase": phase,
        "eager_steps": _async_graph_eager_steps,
        "error": error,
        "shape_calibration": _shape_calibration,
        "graph_resource_audit": {
            "format": 1,
            "policy_requested": requested,
            "policy_effective": effective,
            "dependencies": {
                "nccl_communicators": 1 if world_size > 1 else 0,
                "registered_windows": 0,
                "device_api": False,
                "external_semaphores": 0,
                "provider_owned_allocations": 0,
                "unknown": effective not in {"disabled", "preserve-nccl-exec"},
            },
            "decision": decision,
        },
        "updated_unix_ns": time.time_ns(),
    }
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _context(model_config: Any) -> _ArtifactContext:
    assert _settings is not None and _settings.artifact_root is not None
    validate_model_config(model_config)
    identity = build_identity(model_config)
    return _ArtifactContext(
        identity=identity,
        artifact=artifact_for(
            _settings.artifact_root,
            identity,
            lock_root=_settings.lock_root,
        ),
    )


def _record(metrics: ArtifactMetrics) -> None:
    logger.info(
        "ColdSnap SGLang native payload %s: backend=%s bytes=%d storages=%d elapsed=%.3f s",
        metrics.operation,
        metrics.backend,
        metrics.bytes,
        metrics.storages,
        metrics.seconds,
    )


def _export_model_payload(context: _ArtifactContext, model: Any) -> None:
    assert _settings is not None
    if not _settings.export_model_payload:
        return
    _record(
        context.artifact.capture(
            model,
            identity=context.identity,
            chunk_bytes=ARTIFACT_CHUNK_BYTES,
            replace=True,
        )
    )


def _export_registered_model_payloads() -> None:
    """Capture the final, warmed model layouts immediately before hibernation.

    SGLang may resize or replace derived model buffers after the model loader
    returns (during warmup and CUDA-graph preparation).  Capturing in the
    loader hook therefore records a stale storage extent even though the
    registered tensor object remains valid.  The release hook is the first
    lifecycle point at which the complete serving layout is stable and all
    weight bytes are still resident.
    """
    if _settings is None or not _settings.export_model_payload:
        return
    if not _pending_models:
        raise RuntimeError(
            "ColdSnap SGLang cannot export a native payload without registered models"
        )
    for model, identity, artifact in _pending_models:
        _export_model_payload(_ArtifactContext(identity=identity, artifact=artifact), model)


def _adapter_create(original, enable: bool):
    global _memory_adapter
    del original, enable
    assert _settings is not None
    adapter = create_adapter(_settings)
    _memory_adapter = adapter
    for model, identity, artifact in _pending_models:
        adapter.register_model(model, identity, artifact)
    return adapter


def _register_model(context: _ArtifactContext, model: Any) -> None:
    record = (model, context.identity, context.artifact)
    if not any(existing[0] is model for existing in _pending_models):
        _pending_models.append(record)
    if _memory_adapter is not None:
        _memory_adapter.register_model(*record)


def _model_load(original, loader, *args, **kwargs):
    model_config = kwargs.get("model_config")
    if model_config is None:
        raise RuntimeError(
            "ColdSnap requires keyword model_config on DefaultModelLoader.load_model"
        )
    model = original(loader, *args, **kwargs)
    context = _context(model_config)
    _register_model(context, model)
    if _settings is not None and _settings.startup_provider == "native":
        if _memory_adapter is None:
            raise RuntimeError("ColdSnap SGLang native startup has no memory-saver adapter")
        _memory_adapter.restore_startup_model(model, context.identity, context.artifact)
    return model


def _tags_include_weights(request: Any) -> bool:
    tags = getattr(request, "tags", None)
    return not tags or "weights" in tags


def _phase_graph_enabled(owner: Any, phase: str) -> bool:
    server_args = getattr(owner, "server_args", None)
    graph_config = getattr(server_args, "cuda_graph_config", None)
    phase_config = getattr(graph_config, phase, None)
    backend = getattr(phase_config, "backend", None)
    value = getattr(backend, "value", backend)
    return value is not None and str(value).lower() != "disabled"


def _graph_model_runners(updater: Any) -> list[Any]:
    candidates: list[Any] = []

    def add(value: Any) -> None:
        if value is not None and all(value is not item for item in candidates):
            candidates.append(value)

    tp_worker = getattr(updater, "tp_worker", None)
    add(getattr(tp_worker, "model_runner", None))
    draft_worker = getattr(updater, "draft_worker", None)
    for owner in (draft_worker, getattr(draft_worker, "_draft_worker", None)):
        if owner is None:
            continue
        for name in ("model_runner", "draft_model_runner", "draft_runner"):
            add(getattr(owner, name, None))
        for runner in getattr(owner, "draft_runners", ()) or ():
            add(runner)
    return candidates


def _reset_cuda_graph_pool() -> None:
    """Forget SGLang's process-wide pool and its NCCL allocator binding."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        set_graph_pool_id,
    )
    from sglang.srt.model_executor.runner_utils.pool import (
        set_global_graph_memory_pool,
    )

    set_graph_pool_id(None)
    set_global_graph_memory_pool(None)


def _cleanup_cuda_graph_runner(runner: Any, cleaned_backends: set[int]) -> None:
    """Invoke SGLang's backend-owned executable and pool cleanup contract."""
    backend = getattr(runner, "backend", None)
    cleanup = getattr(backend, "cleanup", None)
    if not callable(cleanup):
        raise RuntimeError("ColdSnap SGLang CUDA graph runner has no backend cleanup capability")
    identity = id(backend)
    if identity not in cleaned_backends:
        cleanup()
        cleaned_backends.add(identity)


def _discard_cuda_graphs(updater: Any) -> None:
    """Destroy engine graph executables before NCCL communicator teardown."""
    global _graph_recapture_plan
    if _graph_recapture_plan:
        raise RuntimeError("ColdSnap SGLang CUDA graphs are already discarded")

    plan: list[_GraphRecaptureStep] = []
    cleaned_backends: set[int] = set()
    for runner in _graph_model_runners(updater):
        for phase in ("decode", "prefill"):
            attribute = f"{phase}_cuda_graph_runner"
            current = getattr(runner, attribute, None)
            callback = getattr(runner, f"init_{phase}_cuda_graph", None)
            if current is not None and callable(callback) and _phase_graph_enabled(runner, phase):
                _cleanup_cuda_graph_runner(current, cleaned_backends)
                setattr(runner, attribute, None)
                plan.append(
                    _GraphRecaptureStep(
                        owner=runner,
                        callback=callback,
                        attributes=(attribute,),
                    )
                )

    draft_worker = getattr(updater, "draft_worker", None)
    for owner in (draft_worker, getattr(draft_worker, "_draft_worker", None)):
        if owner is None:
            continue
        attributes = tuple(
            name
            for name in ("cuda_graph_runner", "cuda_graph_runner_for_draft_extend")
            if getattr(owner, name, None) is not None
        )
        callback = getattr(owner, "_capture_cuda_graphs", None)
        if attributes and callable(callback):
            for attribute in attributes:
                _cleanup_cuda_graph_runner(getattr(owner, attribute), cleaned_backends)
                setattr(owner, attribute, None)
            plan.append(
                _GraphRecaptureStep(
                    owner=owner,
                    callback=callback,
                    attributes=attributes,
                )
            )

    _graph_recapture_plan = plan
    if plan:
        # Backend cleanup releases graph executables before their output tensors
        # and pool references. PyTorch wrappers can still participate in
        # reference cycles, and SGLang's process-wide pool plus pynccl binding
        # are independent owners, so clear both before collection. The graph
        # memory-saver tag is already paused when this runs; empty_cache() is
        # invalid while those CUDA allocations are paused and SGLang will
        # reclaim/reuse them after the tag resumes during graph recapture.
        _reset_cuda_graph_pool()
        gc.collect()
        import torch

        torch.cuda.synchronize()
        logger.info(
            "ColdSnap SGLang discarded %d CUDA graph owner(s) for recapture",
            len(plan),
        )


@contextmanager
def _graph_recapture_allocator_context():
    """Recapture graphs without flushing a restored memory-saver pool."""
    import torch
    from sglang.srt.utils import common

    original_sglang_empty = common.empty_device_cache
    original_torch_empty = torch.cuda.empty_cache

    def preserve_restored_cache(device_module=None):
        del device_module
        return False

    common.empty_device_cache = preserve_restored_cache
    torch.cuda.empty_cache = preserve_restored_cache
    try:
        yield
    finally:
        torch.cuda.empty_cache = original_torch_empty
        common.empty_device_cache = original_sglang_empty


def _recapture_cuda_graphs() -> None:
    global _graph_recapture_plan
    plan = _graph_recapture_plan
    if not plan:
        return
    with _graph_recapture_allocator_context():
        for step in plan:
            step.callback()
            if not any(getattr(step.owner, name, None) is not None for name in step.attributes):
                raise RuntimeError(
                    "ColdSnap SGLang CUDA graph recapture did not restore its runner"
                )
    _graph_recapture_plan = []
    logger.info(
        "ColdSnap SGLang recaptured %d CUDA graph owner(s)",
        len(plan),
    )


def _defer_startup_cuda_graphs(
    original: Any,
    owner: Any,
    capture_decode_cuda_graph: bool = True,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Construct eager dispatch now and defer target decode graph capture."""
    global _async_graph_phase
    if (
        _settings is None
        or _settings.startup_provider not in {"native", "recovery"}
        or not _async_graphs_enabled(owner)
        or not capture_decode_cuda_graph
        or not _phase_graph_enabled(owner, "decode")
        # SGLang's combined initializer captures prefill before decode. Keep
        # that path synchronous until deferred prefill graphs are qualified.
        or _phase_graph_enabled(owner, "prefill")
    ):
        result = original(
            owner,
            *args,
            capture_decode_cuda_graph=capture_decode_cuda_graph,
            **kwargs,
        )
        if (
            _settings is not None
            and _settings.mode == "capture"
            and _settings.shape_calibration
            and _settings.async_graphs
        ):
            global _shape_calibration
            from .shape_calibration import calibrate_capture_shapes

            _shape_calibration = calibrate_capture_shapes(owner)
            logger.info(
                "ColdSnap SGLang calibrated %d/%d CUDA graph shapes in %.3f s",
                _shape_calibration["warmed_shapes"],
                _shape_calibration["planned_shapes"],
                _shape_calibration["seconds"],
            )
            _publish_async_graph_status("calibrated")
        return result
    attribute = "decode_cuda_graph_runner"
    if any(step.owner is owner and attribute in step.attributes for step in _graph_recapture_plan):
        raise RuntimeError("ColdSnap SGLang decode CUDA graph was deferred more than once")

    # This initializes graph-shared output, the eager runner, post-capture
    # pools, and all normal accounting while deliberately skipping decode
    # capture. A None graph runner is SGLang's supported eager fallback.
    result = original(owner, *args, capture_decode_cuda_graph=False, **kwargs)
    setattr(owner, attribute, None)

    def capture() -> None:
        owner.init_decode_cuda_graph()

    _graph_recapture_plan.append(
        _GraphRecaptureStep(
            owner=owner,
            callback=capture,
            attributes=(attribute,),
        )
    )
    _async_graph_phase = "eager"
    _publish_async_graph_status("eager")
    logger.info("ColdSnap SGLang deferred target decode CUDA graphs")
    return result


def _record_async_graph_eager_step(original, scheduler, *args, **kwargs):
    global _async_graph_eager_steps
    result = original(scheduler, *args, **kwargs)
    if _async_graphs_enabled(scheduler) and _graph_recapture_plan:
        _async_graph_eager_steps += 1
        _publish_async_graph_status("eager")
    return result


def _abandon_partial_graphs() -> None:
    global _graph_recapture_plan
    cleaned_backends: set[int] = set()
    for step in _graph_recapture_plan:
        for attribute in step.attributes:
            runner = getattr(step.owner, attribute, None)
            if runner is not None:
                try:
                    _cleanup_cuda_graph_runner(runner, cleaned_backends)
                except Exception:
                    logger.exception("ColdSnap SGLang could not clean a partial CUDA graph")
                setattr(step.owner, attribute, None)
    _graph_recapture_plan = []
    try:
        _reset_cuda_graph_pool()
    except Exception:
        logger.exception("ColdSnap SGLang could not reset a partial graph pool")


def _capture_async_graphs_when_idle(scheduler: Any) -> None:
    global _async_graph_phase, _async_graph_error
    if (
        not _async_graphs_enabled(scheduler)
        or not _graph_recapture_plan
        or _async_graph_phase not in {"disabled", "eager"}
        or _async_graph_eager_steps < 1
        or _settings is None
        or _settings.async_graph_arm_file is None
        or not _settings.async_graph_arm_file.is_file()
        or not scheduler.is_fully_idle()
    ):
        return
    _async_graph_phase = "capturing"
    _publish_async_graph_status("capturing")
    logger.info(
        "ColdSnap SGLang starting scheduler-idle CUDA graph capture after %d eager batch(es)",
        _async_graph_eager_steps,
    )
    try:
        _synchronize_tp_lifecycle(scheduler)
        _recapture_cuda_graphs()
        _synchronize_tp_lifecycle(scheduler)
    except BaseException as error:
        _async_graph_phase = "fallback_eager"
        _async_graph_error = f"{type(error).__name__}: {error}"
        _abandon_partial_graphs()
        _publish_async_graph_status(_async_graph_phase, _async_graph_error)
        logger.exception("ColdSnap SGLang CUDA graph capture failed; retaining eager dispatch")
        return
    _async_graph_phase = "ready"
    _async_graph_error = ""
    _publish_async_graph_status("ready")
    logger.info("ColdSnap SGLang CUDA graphs are ready")


def _capture_async_graphs_before_idle(original, scheduler, *args, **kwargs):
    _capture_async_graphs_when_idle(scheduler)
    return original(scheduler, *args, **kwargs)


def _state_path() -> Path:
    assert _settings is not None and _settings.process_artifact_root is not None
    from coldsnap_core.topology import worker_id

    root = _settings.process_artifact_root / "hibernate-states"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{worker_id()}.json"


def _write_state(state: str, *, nccl: dict[str, Any] | None = None) -> None:
    assert _settings is not None
    from coldsnap_core.topology import worker_id

    path = _state_path()
    value = {
        "format": 1,
        "kind": "coldsnap-sglang-live-process",
        "worker_id": worker_id(),
        "capture_id": os.environ.get("COLDSNAP_CAPTURE_ID", ""),
        "generation": os.environ.get("COLDSNAP_ASYNC_CUDA_GRAPHS_GENERATION", ""),
        "pid": os.getpid(),
        "state": state,
        "updated_unix": time.time(),
    }
    if nccl is not None:
        value["nccl"] = nccl
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _wait_at_cuda_restore_hold() -> None:
    """Leave an n610 capture parked at a CRIU-safe SGLang boundary."""
    if (
        _settings is None
        or _settings.process_artifact_root is None
        or os.environ.get("COLDSNAP_SGLANG_CRIU_HOLD", "0") != "1"
        or os.environ.get("COLDSNAP_CAPTURE_PREPARE_ONLY", "0") == "1"
    ):
        return

    from coldsnap_core.topology import worker_id

    root = _settings.process_artifact_root / "cuda-restore-hold"
    root.mkdir(parents=True, exist_ok=True)
    identity = worker_id()
    path = root / f"{identity}.ready.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "format": 1,
                "kind": "coldsnap-sglang-cuda-restore-hold",
                "worker_id": identity,
                "pid": os.getpid(),
            },
            stream,
            sort_keys=True,
            separators=(",", ":"),
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    release = root / "release"
    while not release.is_file():
        time.sleep(0.01)


def _nccl_runtime():
    global _nccl_runtime_instance
    if _nccl_runtime_instance is None:
        from coldsnap_nccl_checkpoint import NcclCheckpointRuntime

        required = {
            item.strip()
            for item in os.environ.get("COLDSNAP_NCCL_REQUIRED_CAPABILITIES", "").split(",")
            if item.strip()
        }
        _nccl_runtime_instance = NcclCheckpointRuntime(
            require_ib_reset=True,
            require_network_reset=True,
            required_capabilities=required,
        )
    return _nccl_runtime_instance


def _synchronize_tp_lifecycle(updater: Any) -> None:
    """Keep the HTTP-owning rank from acknowledging a partial TP transition."""
    import torch

    group = getattr(updater, "tp_cpu_group", None)
    if group is None:
        raise RuntimeError("ColdSnap SGLang lifecycle owner has no TP CPU group")
    torch.distributed.barrier(group=group)


def _capture_checkpoint_optional_parameters() -> None:
    """Preserve small parameters a checkpoint is explicitly allowed to omit."""
    global _checkpoint_optional_parameters
    captured: list[tuple[Any, str, Any]] = []
    total_bytes = 0
    seen_models: set[int] = set()
    for model, _identity, _artifact in _pending_models:
        if id(model) in seen_models:
            continue
        seen_models.add(id(model))
        try:
            parameters = model.named_parameters(remove_duplicate=False)
        except TypeError:
            parameters = model.named_parameters()
        for name, parameter in parameters:
            if not bool(getattr(parameter, "_skip_weight_check", False)):
                continue
            value = parameter.detach().cpu().clone()
            total_bytes += int(value.numel()) * int(value.element_size())
            if total_bytes > OPTIONAL_PARAMETER_STATE_MAX_BYTES:
                raise RuntimeError(
                    "ColdSnap SGLang checkpoint-optional parameter state exceeds "
                    f"{OPTIONAL_PARAMETER_STATE_MAX_BYTES} bytes"
                )
            captured.append((model, name, value))
    _checkpoint_optional_parameters = captured


def _restore_checkpoint_optional_parameters() -> None:
    """Restore checkpoint-optional values before SGLang post-processing."""
    if not _checkpoint_optional_parameters:
        return
    import torch

    by_model: dict[int, tuple[Any, dict[str, Any]]] = {}
    for model, name, value in _checkpoint_optional_parameters:
        record = by_model.setdefault(id(model), (model, {}))
        record[1][name] = value
    with torch.inference_mode():
        for model, saved in by_model.values():
            try:
                current = dict(model.named_parameters(remove_duplicate=False))
            except TypeError:
                current = dict(model.named_parameters())
            for name, value in saved.items():
                parameter = current.get(name)
                if parameter is None:
                    raise RuntimeError(
                        f"ColdSnap SGLang optional parameter disappeared: {name}"
                    )
                if tuple(parameter.shape) != tuple(value.shape) or parameter.dtype != value.dtype:
                    raise RuntimeError(
                        "ColdSnap SGLang optional parameter metadata changed for "
                        f"{name}"
                    )
                parameter.copy_(value.to(device=parameter.device))


def _release_memory(original, updater, request, *args, **kwargs):
    includes_weights = _tags_include_weights(request)
    tags = getattr(request, "tags", None)
    graph_only = bool(tags and "cuda_graph" in tags and not includes_weights)
    if includes_weights:
        _capture_checkpoint_optional_parameters()
        _export_registered_model_payloads()
    result = original(updater, request, *args, **kwargs)
    observation: dict[str, Any] | None = None
    if includes_weights:
        # Let SGLang quiesce requests, flush live allocator caches, and pause all
        # requested regions first. Its CUDA graph tag keeps the graph pool's
        # allocator records coherent while ColdSnap destroys the graph
        # executables that retain NCCL registrations. The paused pool is
        # resumed before ColdSnap asks SGLang to recapture those graphs.
        if not _retained_nccl_graphs():
            _discard_cuda_graphs(updater)
        observation = _nccl_runtime().prepare()
        observation["worker_id"] = __import__(
            "coldsnap_core.topology", fromlist=["worker_id"]
        ).worker_id()
    if graph_only and _retained_nccl_graphs():
        # The retained-graph acceptance fallback pauses this region through
        # SGLang's normal allocator contract, then destroys only the stale
        # executable owners. The matching resume starts in eager mode and the
        # existing scheduler-idle path recaptures the graph asynchronously.
        _discard_cuda_graphs(updater)
    if observation is not None:
        _write_state("sleeping", nccl=observation)
        _synchronize_tp_lifecycle(updater)
        _wait_at_cuda_restore_hold()
    return result


def _finish_memory_resume(updater: Any, observation: dict[str, Any]) -> None:
    global _async_graph_phase
    if _async_graphs_enabled(updater) and _graph_recapture_plan:
        _async_graph_phase = "eager"
        _publish_async_graph_status("eager")
    else:
        _recapture_cuda_graphs()
    _write_state("running", nccl=observation)
    _synchronize_tp_lifecycle(updater)


def _resume_memory(original, updater, request, *args, **kwargs):
    global _recovery_resume_observation, _restored_weight_update_pending
    observation: dict[str, Any] | None = None
    tags = getattr(request, "tags", None)
    recovery_weights = bool(tags and _RECOVERY_WEIGHTS_RESUME_TAG in tags)
    recovery_runtime = bool(tags and _RECOVERY_RUNTIME_RESUME_TAG in tags)
    if recovery_weights and recovery_runtime:
        raise RuntimeError("ColdSnap SGLang recovery resume phase is ambiguous")
    if recovery_weights or recovery_runtime:
        request.tags = [tag for tag in tags if tag not in _RECOVERY_RESUME_TAGS]
    includes_weights = _tags_include_weights(request)
    graph_only = bool(tags and "cuda_graph" in tags and not includes_weights)
    if recovery_weights and not includes_weights:
        raise RuntimeError("ColdSnap SGLang recovery weight phase omitted weights")
    if recovery_runtime and includes_weights:
        raise RuntimeError("ColdSnap SGLang recovery runtime phase included weights")
    if recovery_runtime:
        observation = _recovery_resume_observation
        if observation is None:
            raise RuntimeError(
                "ColdSnap SGLang recovery runtime resumed before weight hydration"
            )
    if includes_weights:
        from coldsnap_nccl_checkpoint import _apply_restore_transport_environment

        transport = _apply_restore_transport_environment()
        observation = _nccl_runtime().restore()
        _restored_weight_update_pending = not recovery_weights
        if transport is not None:
            observation["transport_environment"] = transport
    result = original(updater, request, *args, **kwargs)
    if graph_only and _retained_nccl_graphs() and _graph_recapture_plan:
        global _async_graph_phase
        _async_graph_phase = "eager"
        _publish_async_graph_status("eager")
    if observation is not None:
        if recovery_weights:
            from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

            model_path = os.environ.get("COLDSNAP_MODEL_ID", "").strip()
            if not model_path:
                raise RuntimeError("ColdSnap SGLang recovery has no pinned model path")
            _restore_checkpoint_optional_parameters()
            with _graph_recapture_allocator_context():
                response = updater.update_weights_from_disk(
                    UpdateWeightFromDiskReqInput(
                        model_path=model_path,
                        load_format=os.environ.get(
                            "COLDSNAP_SGLANG_RECOVERY_LOAD_FORMAT", "auto"
                        ).strip()
                        or "auto",
                        recapture_cuda_graph=False,
                    )
                )
            if not getattr(response, "success", False):
                raise RuntimeError(
                    "ColdSnap SGLang recovery hydration failed: "
                    + str(getattr(response, "message", response))
                )
            # Keep KV storage and CUDA graph allocations paused until every TP
            # worker has completed semantic hydration.  The controller issues
            # a second distributed resume only after this request returns.
            _recovery_resume_observation = observation
        else:
            _finish_memory_resume(updater, observation)
            if recovery_runtime:
                _recovery_resume_observation = None
    return result


def _update_weights_after_restore(original, updater, request, *args, **kwargs):
    global _restored_weight_update_pending
    if not _restored_weight_update_pending:
        return original(updater, request, *args, **kwargs)
    try:
        # SGLang acknowledges resume_memory_occupation before dispatching this
        # separate update request. Keep allocator preservation around the real
        # hydration operation rather than the earlier acknowledgement.
        with _graph_recapture_allocator_context():
            return original(updater, request, *args, **kwargs)
    finally:
        _restored_weight_update_pending = False


async def _gate_tokenizer_release(original, manager, *args, **kwargs):
    """Close HTTP generation before scheduler memory becomes unavailable."""
    from sglang.srt.managers.tokenizer_manager import ServerStatus

    # This status is captured with the process. Closing it before distributed
    # release prevents an external observer from enqueueing health generation
    # between CRIU process resume and the controller's first resume request.
    manager.server_status = ServerStatus.Starting
    return await original(manager, *args, **kwargs)


async def _gate_tokenizer_resume(original, manager, *args, **kwargs):
    """Keep HTTP health closed while recovery weights are placeholders."""
    request = kwargs.get("obj") or (args[0] if args else None)
    tags = getattr(request, "tags", None)
    recovery_weights = bool(tags and _RECOVERY_WEIGHTS_RESUME_TAG in tags)
    recovery_runtime = bool(tags and _RECOVERY_RUNTIME_RESUME_TAG in tags)
    from sglang.srt.managers.tokenizer_manager import ServerStatus
    managed_resume = (
        recovery_weights
        or recovery_runtime
        or manager.server_status == ServerStatus.Starting
    )
    if not managed_resume:
        return await original(manager, *args, **kwargs)

    # Reject health generation before the scheduler is resumed.  Otherwise an
    # already-running observer can enqueue a request in the small interval
    # between memory resume and update_weights_from_disk.
    manager.server_status = ServerStatus.Starting
    result = await original(manager, *args, **kwargs)
    # The weight phase deliberately leaves HTTP closed. The runtime phase is
    # dispatched only after every scheduler has hydrated its semantic weights.
    if not recovery_weights:
        manager.server_status = ServerStatus.Up
    return result


def _draft_update_weights(original, draft_worker, request, *args, **kwargs):
    del original, args, kwargs
    for runner in draft_worker.draft_runners:
        model_path = str(runner.model_config.model_path)
        success, message = runner.weight_updater.update_weights_from_disk(
            model_path,
            request.load_format or "auto",
            recapture_cuda_graph=request.recapture_cuda_graph,
        )
        if not success:
            return success, message
    return True, "Succeeded to update draft model weights from their pinned source."


def register() -> None:
    global _settings
    try:
        settings = Settings.from_env()
    except Exception as error:
        install_failure_sentinel(error)
        raise
    if not settings.enabled:
        logger.debug("ColdSnap SGLang plugin installed but disabled")
        return
    _settings = settings
    if settings.async_graphs:
        # This activation-local receipt makes hook installation observable to
        # the controller before the graph initializer advances it to eager.
        _publish_async_graph_status("configured")
    try:
        validate_contracts()
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        weight_updater = resolve_weight_updater_contract()

        hooks = (
            (
                "sglang.srt.utils.torch_memory_saver_adapter.TorchMemorySaverAdapter.create",
                _adapter_create,
            ),
            (
                "sglang.srt.model_loader.loader.DefaultModelLoader.load_model",
                _model_load,
            ),
            (
                "sglang.srt.model_loader.loader.DummyModelLoader.load_model",
                _model_load,
            ),
            (
                "sglang.srt.model_executor.model_runner.ModelRunner.init_cuda_graphs",
                _defer_startup_cuda_graphs,
            ),
            (
                "sglang.srt.managers.scheduler.Scheduler.process_batch_result",
                _record_async_graph_eager_step,
            ),
            (
                "sglang.srt.managers.scheduler.Scheduler.on_idle",
                _capture_async_graphs_before_idle,
            ),
            (
                f"{weight_updater.hook_prefix}.release_memory_occupation",
                _release_memory,
            ),
            (
                f"{weight_updater.hook_prefix}.resume_memory_occupation",
                _resume_memory,
            ),
            (
                f"{weight_updater.hook_prefix}.update_weights_from_disk",
                _update_weights_after_restore,
            ),
            (
                "sglang.srt.managers.tokenizer_manager.TokenizerManager.release_memory_occupation",
                _gate_tokenizer_release,
            ),
            (
                "sglang.srt.managers.tokenizer_manager.TokenizerManager.resume_memory_occupation",
                _gate_tokenizer_resume,
            ),
            (
                "sglang.srt.speculative.base_spec_worker.BaseSpecWorker.update_weights_from_disk",
                _draft_update_weights,
            ),
        )
        for target, hook in hooks:
            HookRegistry.register(target, hook, HookType.AROUND)
    except Exception as error:
        install_failure_sentinel(error)
        raise
    logger.info(
        "ColdSnap SGLang enabled: process_snapshot=%s contract=%s artifact_root=%s live_backing=%s",
        settings.process_artifact_root,
        contract_mode(),
        settings.artifact_root,
        settings.live_backing,
    )

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Reuse DSpark's validated graph-replay costs before eager-first inference.

Only small, versioned JSON curves are cached. vLLM remains responsible for
profiling, rank-zero curve selection, and construction of its runtime tables.
No CUDA or vLLM import occurs until a worker is ready to compile its model.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from coldsnap_import_hook import after_module_import
from coldsnap_startup_plan import _all_workers_admit_startup_plan


logger = logging.getLogger(__name__)
_MAX_RECORD_BYTES = 128 * 1024
_KIND = "coldsnap-dspark-calibration"
_STATE = "_coldsnap_adaptive_calibration"
_MARKER = "_coldsnap_calibration_observer"
_SPEC_FIELDS = (
    "method", "model", "revision", "code_revision", "num_speculative_tokens",
    "draft_tensor_parallel_size", "quantization", "moe_backend", "attention_backend",
    "kv_cache_dtype", "max_model_len", "enforce_eager", "disable_padded_drafter_batch",
    "use_local_argmax_reduction", "use_heterogeneous_vocab", "parallel_drafting",
    "num_speculative_tokens_per_batch_size", "adaptive_speculative_tokens_window",
    "adaptive_speculative_tokens_initial", "rejection_sample_method",
    "synthetic_acceptance_rates", "synthetic_acceptance_length",
    "enable_adaptive_verification", "draft_sample_method", "dspark_draft_topk",
)


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(getattr(value, "name", None), str):
        return value.name
    # torch dtypes are not JSON values; other unknown config objects fail closed.
    if type(value).__name__ == "dtype":
        return str(value)
    raise ValueError(f"unsupported calibration identity value: {type(value).__name__}")


def _fields(owner: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: _value(getattr(owner, name, None)) for name in names}


def _hardware(worker: Any) -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(worker.device)
    version = Path("/proc/driver/nvidia/version").read_text()
    match = re.search(r"\b(\d{3}\.\d+(?:\.\d+)?)\b", version)
    if match is None:
        raise ValueError("NVIDIA host driver identity is unavailable")
    return {
        "driver": match.group(1),
        **_fields(props, ("name", "major", "minor", "total_memory", "multi_processor_count")),
    }


def _identity(worker: Any) -> dict[str, Any]:
    import vllm.envs as envs

    image = os.environ.get("COLDSNAP_SOURCE_RUNTIME_IMAGE", "")
    model = os.environ.get("COLDSNAP_MODEL_ID", "")
    revision = os.environ.get("COLDSNAP_MODEL_REVISION", "")
    fingerprint = getattr(worker, "_coldsnap_startup_plan_initial_fingerprint", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image) or not model or not revision:
        raise ValueError("pinned source runtime/model identity is unavailable")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("initial startup-plan fingerprint is unavailable")
    runner = worker.model_runner
    config = worker.vllm_config
    spec = config.speculative_config
    if spec.method != "dspark" or not spec.enable_adaptive_verification:
        raise ValueError("unsupported adaptive verification implementation")
    if worker.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("calibration reuse requires pipeline parallel size one")
    # An independently loaded draft model also needs an immutable revision.
    draft = spec.draft_model_config
    draft_revision = getattr(getattr(draft, "hf_config", None), "_commit_hash", None)
    if spec.model and spec.model != model and not draft_revision:
        raise ValueError("draft model revision is unavailable")
    result = {
        "source_runtime_image": image,
        "model": model,
        "revision": revision,
        "draft_revision": draft_revision,
        "startup_fingerprint": fingerprint,
        "rank": int(worker.rank),
        "hardware": _hardware(worker),
        "parallel": _fields(worker.parallel_config, (
            "world_size", "tensor_parallel_size", "pipeline_parallel_size",
            "data_parallel_size", "enable_expert_parallel", "decode_context_parallel_size",
            "prefill_context_parallel_size", "disable_custom_all_reduce", "all2all_backend",
        )),
        "model_config": _fields(config.model_config, (
            "dtype", "quantization", "max_model_len", "enforce_eager",
        )),
        "configured_model_len": getattr(worker, "_coldsnap_startup_plan_configured_model_len", None),
        "graph": _fields(config.compilation_config, (
            "cudagraph_mode", "cudagraph_capture_sizes", "max_cudagraph_capture_size",
            "cudagraph_num_of_warmups", "cudagraph_specialize_lora", "custom_ops",
        )),
        "scheduler": _fields(config.scheduler_config, (
            "max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill",
            "async_scheduling", "max_num_partial_prefills", "max_long_partial_prefills",
        )),
        "cache": _fields(config.cache_config, ("cache_dtype", "block_size", "enable_prefix_caching")),
        "speculative": _fields(spec, _SPEC_FIELDS),
        "request_limits": _fields(runner.adaptive_verification.req_states, (
            "max_num_reqs", "max_num_batched_tokens", "num_speculative_steps",
        )),
        "profile_context_len": int(envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN),
        "performance_environment": {
            name: value for name, value in sorted(os.environ.items())
            if name.startswith("B12X_") or name in {
                "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_BREAKABLE_CUDAGRAPH", "CUTE_DSL_ARCH",
                "NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
            }
        },
    }
    if len(_json(result)) > _MAX_RECORD_BYTES // 2:
        raise ValueError("calibration identity exceeds size limit")
    return result


def _curve(value: Any) -> list[list[int | float]]:
    if not isinstance(value, (list, tuple)) or not 0 < len(value) <= 4096:
        raise ValueError("calibration curve must be nonempty and bounded")
    result = []
    previous = 0
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("invalid calibration point")
        x, y = point
        if type(x) is not int or not previous < x <= 1 << 24:
            raise ValueError("calibration coordinates must be positive, ordered and unique")
        if type(y) not in (int, float) or not math.isfinite(y) or y <= 0:
            raise ValueError("calibration timing must be finite and positive")
        result.append([x, float(y)])
        previous = x
    return result


def _curves(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"draft", "verify", "cudagraph_limit"}:
        raise ValueError("invalid calibration payload")
    limit = value["cudagraph_limit"]
    if type(limit) is not int or not 0 < limit <= 1 << 24:
        raise ValueError("invalid captured-token limit")
    return {"draft": _curve(value["draft"]), "verify": _curve(value["verify"]), "cudagraph_limit": limit}


def _load(state: dict[str, Any]) -> dict[str, Any]:
    with state["path"].open("rb") as stream:
        data = stream.read(_MAX_RECORD_BYTES + 1)
    if len(data) > _MAX_RECORD_BYTES:
        raise ValueError("calibration record exceeds size limit")
    record = json.loads(data)
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "kind", "identity", "curves", "sha256"}
        or type(record["schema"]) is not int or record["schema"] != 1
        or record["kind"] != _KIND or record["identity"] != state["identity"]
    ):
        raise ValueError("calibration schema or compatibility identity differs")
    curves = _curves(record["curves"])
    if record["sha256"] != _digest(curves):
        raise ValueError("calibration curve checksum differs")
    return curves


def _save(state: dict[str, Any], curves: dict[str, Any]) -> None:
    path = state["path"]
    data = _json({"schema": 1, "kind": _KIND, "identity": state["identity"],
                  "curves": curves, "sha256": _digest(curves)})
    if len(data) > _MAX_RECORD_BYTES:
        raise ValueError("calibration record exceeds size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".calibration-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _broadcast(value: Any) -> Any:
    from vllm.distributed.parallel_state import get_tp_group

    return get_tp_group().broadcast_object(value, src=0)


def prepare_calibration(worker: Any) -> dict[str, Any] | None:
    runner = worker.model_runner
    manager = getattr(runner, "adaptive_verification", None)
    if manager is None:
        return None
    state = getattr(manager, _STATE, None)
    if state is not None:
        return state
    state = {"decision": "synchronous", "reason": "not looked up", "saved": False}
    setattr(manager, _STATE, state)
    setattr(worker, _STATE, state)
    try:
        identity = _identity(worker)
        root = Path(os.environ.get("VLLM_CACHE_ROOT", "/var/cache/coldsnap/runtime/vllm"))
        if not root.is_absolute():
            raise ValueError("VLLM_CACHE_ROOT must be absolute")
        state.update(identity=identity, key=_digest(identity))
        state["path"] = root / "adaptive-calibration" / (state["key"] + ".json")
    except (AttributeError, OSError, TypeError, ValueError) as error:
        state["reason"] = str(error)

    original = manager.set_cost_curves

    @functools.wraps(original)
    def observe(draft_curve: Any, verify_curve: Any) -> Any:
        result = original(draft_curve, verify_curve)
        if state.get("installing"):
            return result
        # The original selects rank zero internally without changing its
        # caller's arguments. Persist the actual accepted curves on every rank.
        accepted = _broadcast({"draft": draft_curve, "verify": verify_curve,
                               "cudagraph_limit": manager._cudagraph_limit})
        if "path" in state and getattr(runner.cudagraph_manager, "_graphs_captured", False):
            try:
                curves = _curves(accepted)
                _save(state, curves)
                state.update(saved=True, curve_sha256=_digest(curves))
                logger.info("ColdSnap saved validated DSpark calibration %s", state["key"])
            except (OSError, TypeError, ValueError, RecursionError) as error:
                state["save_error"] = str(error)
                logger.warning("ColdSnap could not cache DSpark calibration: %s", error)
        return result

    manager.set_cost_curves = observe
    return state


def reuse_calibration(worker: Any) -> bool:
    state = prepare_calibration(worker)
    if state is None:
        return True
    curves = None
    if "path" in state:
        try:
            curves = _load(state)
        except (OSError, TypeError, ValueError, RecursionError) as error:
            state["reason"] = str(error)
    state["local_reason"] = "valid local record" if curves is not None else state["reason"]
    if not _all_workers_admit_startup_plan(worker, curves is not None):
        state["reason"] = "at least one worker lacks compatible validated curves"
        logger.info("ColdSnap DSpark calibration cache miss; calibrating synchronously on all ranks; local reason: %s", state["local_reason"])
        return False
    digest = _digest(curves)
    if not _all_workers_admit_startup_plan(worker, _broadcast(digest) == digest):
        state["reason"] = "workers have different validated curves"
        logger.info("ColdSnap DSpark calibration records disagree; calibrating synchronously on all ranks")
        return False
    manager = worker.model_runner.adaptive_verification
    manager._cudagraph_limit = curves["cudagraph_limit"]
    state["installing"] = True
    try:
        manager.set_cost_curves(curves["draft"], curves["verify"])
    finally:
        state.pop("installing", None)
    state.update(decision="reused", reason="all workers admitted matching curves", curve_sha256=digest)
    logger.info("ColdSnap reused validated DSpark calibration %s before eager inference", state["key"])
    return True


def calibration_status(worker: Any) -> dict[str, Any] | None:
    state = getattr(worker, _STATE, None)
    if state is None:
        return None
    return {key: state[key] for key in ("decision", "reason", "local_reason", "key", "saved", "curve_sha256", "save_error") if key in state}


@contextmanager
def preserve_calibration_during_shape_warmup(runner: Any) -> Iterator[None]:
    """Shape-only forwards cannot supply graph-replay calibration samples."""
    manager = getattr(runner, "adaptive_verification", None)
    if manager is None:
        yield
        return
    attributes = vars(manager)
    saved = {name: (name in attributes, attributes.get(name)) for name in (
        "batches_to_profile", "set_initial_cost_curves",
    )}
    manager.batches_to_profile = lambda capture_sizes: iter(())
    manager.set_initial_cost_curves = lambda samples: None
    try:
        yield
    finally:
        for name, (present, value) in saved.items():
            if present:
                setattr(manager, name, value)
            else:
                delattr(manager, name)


def _install_worker_observer(module: Any) -> None:
    global logger
    try:
        from vllm.logger import init_logger
    except ImportError:
        pass  # CPU-only contract tests do not require an installed vLLM.
    else:
        logger = init_logger(__name__)
    # Older supported images expose GPUWorker, newer images Worker (or an alias).
    worker_class = getattr(module, "Worker", None) or getattr(module, "GPUWorker", None)
    original = worker_class.compile_or_warm_up_model
    if getattr(original, _MARKER, False):
        return

    @functools.wraps(original)
    def compile_with_observer(worker: Any, *args: Any, **kwargs: Any) -> Any:
        prepare_calibration(worker)
        return original(worker, *args, **kwargs)

    setattr(compile_with_observer, _MARKER, True)
    worker_class.compile_or_warm_up_model = compile_with_observer


def install_calibration_hooks() -> None:
    # Also observe ordinary synchronous captures so they can seed later starts.
    after_module_import("vllm.v1.worker.gpu_worker", "dspark-calibration-worker", _install_worker_observer)

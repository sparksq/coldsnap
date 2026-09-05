# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Cooperative vLLM runtime resources for a reconstructible CUDA epoch.

The model and KV cache already have explicit CuMem allocation scopes.  vLLM's
eager runner constructs a smaller set of persistent input tensors before those
scopes, plus CUDA events, streams, and pinned CPU staging tensors.  A context
reset invalidates all of them even when the model allocation addresses are
stable.

This opt-in adapter puts runner-construction CUDA tensors in a tagged
``runtime`` CuMem pool, keeps runner CPU buffers pageable, and records the
small set of CUDA resources that can be recreated after VMM rebind.  It is a
strict one-device-per-process/eager/synchronous prototype for the admitted
vLLM V1 path and the pinned V2 DSpark/DeepSeek-V4 path, not generic
object-graph surgery.
"""

from __future__ import annotations

import faulthandler
import functools
import gc
import os
import sys
import threading
import traceback
import weakref
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from coldsnap_import_hook import after_module_import
from coldsnap_vllm import VllmContractError


CUDA_EPOCH_RUNTIME_ENV = "COLDSNAP_CUDA_EPOCH_RUNTIME"
CUDA_EPOCH_MEMORY_HISTORY_ENV = "COLDSNAP_CUDA_EPOCH_MEMORY_HISTORY"
RUNTIME_TAG = "runtime"
RUNTIME_KV_META_TAG = "runtime_kv_meta"
RUNTIME_EPOCH_TAG_PREFIX = "runtime_epoch_"
_POOL_SCOPE_ATTR = "_coldsnap_cuda_epoch_pool_scope"
_RUNNER_V1_MODULE = "vllm.v1.worker.gpu_model_runner"
_RUNNER_V2_MODULE = "vllm.v1.worker.gpu.model_runner"
_RUNNER_MODULES = (_RUNNER_V1_MODULE, _RUNNER_V2_MODULE)
_RUNNER_KIND_ATTR = "_coldsnap_cuda_epoch_runner_kind"
_RUNNER_V1 = "v1"
_RUNNER_V2 = "v2"
_RUNNER_MARKER = "_coldsnap_cuda_epoch_runtime_hook"
_EXECUTE_MARKER = "_coldsnap_cuda_epoch_execution_hook"
_WORKER_MODULE = "vllm.v1.worker.gpu_worker"
_WORKER_MARKER = "_coldsnap_cuda_epoch_weight_pool_hook"
_WORKER_INIT_MARKER = "_coldsnap_cuda_epoch_worker_init_hook"
_SEED_INIT_MARKER = "_coldsnap_cuda_epoch_seed_init_hook"
_WORKER_EPOCH_ATTR = "_coldsnap_cuda_epoch_lifecycle"
_WORKER_LAST_EPOCH_ATTR = "_coldsnap_cuda_epoch_last_inventory"
_WORKER_WAKE_MARKER = "_coldsnap_cuda_epoch_wake_hook"
_BUFFER_MODULE = "vllm.v1.utils"
_BUFFER_MARKER = "_coldsnap_cuda_epoch_pageable_buffer_hook"
_V2_BUFFER_MODULE = "vllm.v1.worker.gpu.buffer_utils"
_V2_UVA_BUFFER_MARKER = "_coldsnap_cuda_epoch_managed_uva_buffer_hook"
_V2_UVA_COPY_MARKER = "_coldsnap_cuda_epoch_managed_uva_copy_hook"
_ASYNC_OUTPUT_MODULE = "vllm.v1.worker.gpu.async_utils"
_ASYNC_OUTPUT_MARKER = "_coldsnap_cuda_epoch_async_output_hook"
_ASYNC_COPY_MARKER = "_coldsnap_cuda_epoch_pageable_output_copy_hook"
_FLASHINFER_PINNED_WORKSPACE_ATTR = "_pin_memory_int_workspace_buffer"
_FLASHINFER_PINNED_WORKSPACE_OWNERS = frozenset(
    {
        "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper",
        "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper",
    }
)
_ASYNC_OUTPUT_TYPE = "vllm.v1.worker.gpu.async_utils.AsyncOutput"
_ASYNC_OUTPUT_RESOURCE_ATTRS = (
    "sampler_output",
    "num_sampled_tokens",
    "routed_experts",
    "_has_fault",
    "main_stream",
    "copy_stream",
    "copy_event",
    "sampled_token_ids",
    "logprobs_tensors",
    "num_nans",
    "num_sampled_tokens_np",
    "routed_experts_cpu",
    "draft_token_ids_np",
    "draft_req_ids",
)
_ASYNC_OUTPUT_ARRAY_ATTRS = (
    "sampled_token_ids",
    "num_nans",
    "num_sampled_tokens_np",
    "draft_token_ids_np",
)

_lock = threading.RLock()
_runner_ref: weakref.ReferenceType[Any] | None = None
_installed = False
_execution_tls = threading.local()
_bootstrap_tls = threading.local()
_memory_history_enabled = False
_bootstrap_trace_tls = threading.local()
_bootstrap_allocation_traces: dict[int, dict[str, Any]] = {}


def _trace_cuda_bootstrap_allocations(torch: Any) -> None:
    """Trace only the known tiny tensor signature through admission inference."""
    if getattr(_bootstrap_trace_tls, "mode", None) is not None:
        return
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except ImportError:
        return

    class CudaBootstrapTraceMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            del types
            result = func(*args, **(kwargs or {}))
            stack: list[Any] = [result]
            while stack and len(_bootstrap_allocation_traces) < 128:
                value = stack.pop()
                if isinstance(value, dict):
                    stack.extend(value.values())
                    continue
                if isinstance(value, (list, tuple)):
                    stack.extend(value)
                    continue
                try:
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.device.type != "cuda"
                        or tuple(int(item) for item in value.shape) != (8,)
                    ):
                        continue
                    pointer = int(value.untyped_storage().data_ptr())
                    if pointer <= 0 or pointer in _bootstrap_allocation_traces:
                        continue
                    frames = [
                        {
                            "filename": frame.filename,
                            "line": frame.lineno,
                            "name": frame.name,
                        }
                        for frame in traceback.extract_stack(limit=32)[:-1]
                    ]
                    _bootstrap_allocation_traces[pointer] = {
                        "pointer": pointer,
                        "op": str(func),
                        "dtype": str(value.dtype),
                        "frames": frames,
                    }
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    continue
            return result

    mode = CudaBootstrapTraceMode()
    mode.__enter__()
    _bootstrap_trace_tls.mode = mode


def _stop_cuda_bootstrap_trace() -> None:
    mode = getattr(_bootstrap_trace_tls, "mode", None)
    if mode is None:
        return
    del _bootstrap_trace_tls.mode
    mode.__exit__(None, None, None)


def _enable_cuda_memory_history() -> bool:
    """Enable bounded allocation stacks before the first CUDA allocation."""
    global _memory_history_enabled
    if os.environ.get(CUDA_EPOCH_MEMORY_HISTORY_ENV, "0") != "1":
        return False
    if _memory_history_enabled:
        return True
    import torch

    recorder = getattr(getattr(torch.cuda, "memory", None), "_record_memory_history", None)
    if not callable(recorder):
        raise VllmContractError(
            "PyTorch 2.10+ CUDA allocation history is unavailable"
        )
    recorder(
        enabled="all",
        context="alloc",
        stacks="python",
        max_entries=100_000,
    )
    _memory_history_enabled = True
    _trace_cuda_bootstrap_allocations(torch)
    return True


class _RuntimePoolScope:
    """Keep late CUDA allocations inside VMM for one complete CUDA epoch."""

    def __init__(self, allocator: Any, *, initial_tag: str = RUNTIME_TAG) -> None:
        self.allocator = allocator
        self.initial_tag = initial_tag
        self.context: Any | None = None
        self.active = False
        self.epoch = 0
        self.tags: list[str] = []
        self.activate_device: Any | None = None

    def enter(self) -> str:
        if self.active:
            raise VllmContractError("vLLM CUDA epoch runtime pool is already active")
        tag = (
            self.initial_tag
            if self.epoch == 0
            else f"{RUNTIME_EPOCH_TAG_PREFIX}{self.epoch}"
        )
        context = self.allocator.use_memory_pool(tag=tag)
        context.__enter__()
        self.context = context
        self.active = True
        self.epoch += 1
        self.tags.append(tag)
        return tag

    def exit(self, exc_info: tuple[Any, Any, Any] = (None, None, None)) -> None:
        if not self.active or self.context is None:
            raise VllmContractError("vLLM CUDA epoch runtime pool is not active")
        context = self.context
        self.context = None
        self.active = False
        context.__exit__(*exc_info)


def _runtime_pool_scope(runner: Any) -> _RuntimePoolScope:
    scope = getattr(runner, _POOL_SCOPE_ATTR, None)
    if not isinstance(scope, _RuntimePoolScope):
        raise VllmContractError(
            "vLLM runner was not constructed inside a complete CUDA epoch pool"
        )
    return scope


def _cuda_device_index(device: Any) -> int:
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        prefix, separator, suffix = device.partition(":")
        if prefix == "cuda" and separator and suffix.isdecimal():
            return int(suffix)
        raise VllmContractError(
            f"CUDA epoch runner device must have an explicit index, got {device!r}"
        )
    device_index = getattr(device, "index", None)
    if isinstance(device_index, int):
        return device_index
    raise VllmContractError(
        f"CUDA epoch runner device must have an explicit index, got {device!r}"
    )


def _activate_cuda_execution_thread(runner: Any) -> None:
    """Establish fresh-context device and default-stream state on this thread."""
    scope = _runtime_pool_scope(runner)
    if not scope.active:
        raise VllmContractError("vLLM CUDA execution reached an inactive epoch pool")
    generations = getattr(_execution_tls, "generations", None)
    if generations is None:
        generations = {}
        _execution_tls.generations = generations
    identity = id(runner)
    import torch

    if _memory_history_enabled:
        _trace_cuda_bootstrap_allocations(torch)
    # vLLM can dispatch execute_model and sample_tokens on the same executor
    # thread.  Kernels between those boundaries may change the thread-local
    # current stream, so generation-only initialization is insufficient.
    torch.cuda.set_device(runner.device)
    torch.cuda.set_stream(torch.cuda.default_stream(runner.device))
    activate_device = scope.activate_device
    if activate_device is not None:
        # Keep native activation last: PyTorch device bookkeeping may otherwise
        # restore the CUDA Runtime context cached before cudaDeviceReset.
        activate_device(_cuda_device_index(runner.device))
    generations[identity] = scope.epoch


def _enabled() -> bool:
    return os.environ.get(CUDA_EPOCH_RUNTIME_ENV, "0").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _disable_vllm_pin_memory() -> None:
    """Make epoch-spanning vLLM CPU tensors independent of CUDA contexts."""
    from vllm.utils import torch_utils

    value = getattr(torch_utils, "PIN_MEMORY", None)
    if not isinstance(value, bool):
        raise VllmContractError("vLLM torch_utils.PIN_MEMORY is unavailable")
    torch_utils.PIN_MEMORY = False
    # Plugin loading can happen after a utility module copied PIN_MEMORY with a
    # from-import.  Patch only that exact vLLM convention; future imports see
    # the updated torch_utils value normally.
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("vllm.") or module is None:
            continue
        if isinstance(getattr(module, "PIN_MEMORY", None), bool):
            module.PIN_MEMORY = False


def _register_runner(runner: Any) -> None:
    global _runner_ref
    with _lock:
        current = _runner_ref() if _runner_ref is not None else None
        if current is not None and current is not runner:
            raise VllmContractError(
                "the CUDA epoch runtime prototype supports one vLLM runner per process"
            )
        _runner_ref = weakref.ref(runner)


def _current_runner() -> Any:
    with _lock:
        runner = _runner_ref() if _runner_ref is not None else None
    if runner is None:
        raise VllmContractError(
            "no admitted vLLM runner was constructed for the CUDA epoch"
        )
    return runner


def _runner_kind(runner: Any) -> str:
    kind = getattr(runner, _RUNNER_KIND_ATTR, None)
    if kind not in {_RUNNER_V1, _RUNNER_V2}:
        # Keep direct unit construction and the original V1 prototype usable.
        if all(
            hasattr(runner, name)
            for name in ("input_batch", "requests", "transfer_event")
        ):
            return _RUNNER_V1
        raise VllmContractError("vLLM runner has no admitted CUDA epoch kind")
    return kind


def _require_empty_graph_manager(manager: Any, owner: str) -> None:
    if manager is None:
        return
    graphs = getattr(manager, "graphs", None)
    if graphs:
        raise VllmContractError(
            f"CUDA epoch requires eager execution; {owner} retains CUDA graphs"
        )
    if getattr(manager, "pool", None) is not None:
        raise VllmContractError(
            f"CUDA epoch requires eager execution; {owner} retains a graph pool"
        )


def _require_runner_contract(runner: Any) -> None:
    parallel = getattr(runner, "parallel_config", None)
    tensor_parallel = int(getattr(parallel, "tensor_parallel_size", 0))
    pipeline_parallel = int(getattr(parallel, "pipeline_parallel_size", 0))
    data_parallel = int(getattr(parallel, "data_parallel_size", 1))
    if tensor_parallel < 1 or pipeline_parallel != 1 or data_parallel != 1:
        raise VllmContractError(
            "CUDA epoch runtime requires one-device-per-process TP with PP1 and DP1"
        )
    kind = _runner_kind(runner)
    if kind == _RUNNER_V1:
        if getattr(runner, "speculative_config", None) is not None:
            raise VllmContractError(
                "CUDA epoch V1 runtime does not reconstruct speculative decoding"
            )
        if bool(getattr(runner, "use_async_scheduling", True)):
            raise VllmContractError(
                "CUDA epoch runtime requires synchronous scheduling because "
                "PyTorch's process-global CUDA stream pool belongs to the old context"
            )
        if getattr(runner, "cudagraph_batch_sizes", None):
            raise VllmContractError("CUDA epoch runtime requires eager execution")
        required = (
            "async_output_copy_stream",
            "prepare_inputs_event",
            "transfer_event",
            "input_batch",
            "requests",
        )
    else:
        speculative = getattr(runner, "speculative_config", None)
        if speculative is not None and getattr(speculative, "method", None) != "dspark":
            raise VllmContractError(
                "CUDA epoch V2 runtime admits only DSpark speculative decoding"
            )
        scheduler = getattr(runner, "scheduler_config", None)
        if bool(getattr(scheduler, "async_scheduling", True)):
            raise VllmContractError(
                "CUDA epoch V2 runtime requires synchronous scheduling"
            )
        model_config = getattr(runner, "model_config", None)
        if not bool(getattr(model_config, "enforce_eager", False)):
            raise VllmContractError("CUDA epoch V2 runtime requires eager execution")
        required = (
            "output_copy_stream",
            "req_states",
            "execute_model_state",
            "encoder_cache",
            "cudagraph_manager",
            "draft_tokens_handler",
        )
        _require_empty_graph_manager(
            getattr(runner, "cudagraph_manager", None), "vLLM V2 runner"
        )
        speculator = getattr(runner, "speculator", None)
        if speculator is not None:
            managers = getattr(speculator, "get_cudagraph_managers", None)
            if not callable(managers):
                raise VllmContractError(
                    "CUDA epoch DSpark speculator lacks graph-manager inventory"
                )
            for index, manager in enumerate(managers()):
                _require_empty_graph_manager(manager, f"DSpark manager {index}")
            if getattr(speculator, "_captured_backbone_outputs", None):
                raise VllmContractError(
                    "CUDA epoch DSpark speculator retains captured graph outputs"
                )
        if getattr(runner, "_cudagraph_pool_anchor", None) is not None:
            raise VllmContractError(
                "CUDA epoch V2 runner retains a CUDA graph pool anchor"
            )
        if getattr(runner, "_sps_debug_events", None):
            raise VllmContractError(
                "CUDA epoch V2 runner cannot retain DSpark SPS timing events"
            )
    for name in required:
        if not hasattr(runner, name):
            raise VllmContractError(
                f"vLLM {kind.upper()} runner lacks CUDA epoch resource {name!r}"
            )


def _install_runner_hook(module: ModuleType) -> None:
    runner_kind = {
        _RUNNER_V1_MODULE: _RUNNER_V1,
        _RUNNER_V2_MODULE: _RUNNER_V2,
    }.get(module.__name__)
    if runner_kind is None:
        raise VllmContractError(
            f"unsupported vLLM CUDA epoch runner module {module.__name__!r}"
        )
    runner_class = getattr(module, "GPUModelRunner", None)
    if not isinstance(runner_class, type):
        raise VllmContractError(
            f"{module.__name__} does not expose GPUModelRunner"
        )
    original = getattr(runner_class, "__init__", None)
    if not callable(original):
        raise VllmContractError(
            f"{module.__name__}.GPUModelRunner lacks __init__"
        )
    if getattr(original, _RUNNER_MARKER, False):
        return

    @functools.wraps(original)
    def init_in_runtime_pool(runner: Any, *args: Any, **kwargs: Any) -> None:
        from vllm.device_allocator import get_mem_allocator_instance

        config = args[0] if args else kwargs.get("vllm_config")
        model_config = getattr(config, "model_config", None)
        if not bool(getattr(model_config, "enable_sleep_mode", False)):
            raise VllmContractError(
                "CUDA epoch runtime requires vLLM sleep mode during construction"
            )
        allocator = get_mem_allocator_instance()
        use_pool = getattr(allocator, "use_memory_pool", None)
        if not callable(use_pool):
            raise VllmContractError(
                "vLLM CUDA allocator lacks use_memory_pool for runtime tensors"
            )
        bootstrap_scope = getattr(_bootstrap_tls, "scope", None)
        if bootstrap_scope is not None:
            if runner_kind != _RUNNER_V2:
                raise VllmContractError(
                    "CUDA epoch bootstrap pool reached a non-V2 runner"
                )
            if (
                not isinstance(bootstrap_scope, _RuntimePoolScope)
                or not bootstrap_scope.active
                or bootstrap_scope.initial_tag != RUNTIME_TAG
            ):
                raise VllmContractError(
                    "CUDA epoch V2 seed bootstrap pool is invalid or inactive"
                )
            if bootstrap_scope.allocator is not allocator:
                raise VllmContractError(
                    "CUDA epoch V2 bootstrap and runner allocators differ"
                )
            del _bootstrap_tls.scope
            scope = bootstrap_scope
        else:
            scope = _RuntimePoolScope(allocator)
            scope.enter()
        try:
            original(runner, *args, **kwargs)
            setattr(runner, _RUNNER_KIND_ATTR, runner_kind)
            _require_runner_contract(runner)
            setattr(runner, _POOL_SCOPE_ATTR, scope)
            _register_runner(runner)
        except BaseException:
            scope.exit(sys.exc_info())
            raise

    setattr(init_in_runtime_pool, _RUNNER_MARKER, True)
    runner_class.__init__ = init_in_runtime_pool

    zero_meta = getattr(runner_class, "_init_kv_zero_meta", None)
    if not callable(zero_meta):
        raise VllmContractError(
            f"{module.__name__}.GPUModelRunner lacks _init_kv_zero_meta"
        )

    @functools.wraps(zero_meta)
    def init_kv_zero_meta_in_runtime_pool(
        runner: Any, *args: Any, **kwargs: Any
    ) -> Any:
        from vllm.device_allocator import get_mem_allocator_instance

        allocator = get_mem_allocator_instance()
        use_pool = getattr(allocator, "use_memory_pool", None)
        if not callable(use_pool):
            raise VllmContractError(
                "vLLM CUDA allocator lacks use_memory_pool for KV metadata"
            )
        # Use a distinct tag because vLLM stores one strong MemPool reference
        # per tag; re-entering `runtime` would overwrite the pool that owns the
        # runner-construction tensors.
        with use_pool(tag=RUNTIME_KV_META_TAG):
            return zero_meta(runner, *args, **kwargs)

    runner_class._init_kv_zero_meta = init_kv_zero_meta_in_runtime_pool

    def wrap_execution_entrypoint(name: str) -> None:
        original_entrypoint = getattr(runner_class, name, None)
        if not callable(original_entrypoint):
            raise VllmContractError(
                f"{module.__name__}.GPUModelRunner lacks {name}"
            )
        if getattr(original_entrypoint, _EXECUTE_MARKER, False):
            return

        @functools.wraps(original_entrypoint)
        def in_current_epoch(runner: Any, *args: Any, **kwargs: Any) -> Any:
            _activate_cuda_execution_thread(runner)
            trace_timeout = os.environ.get(
                "COLDSNAP_CUDA_EPOCH_TRACE_TIMEOUT_S"
            )
            armed = False
            if trace_timeout and _runtime_pool_scope(runner).epoch > 1:
                try:
                    timeout = float(trace_timeout)
                except ValueError as error:
                    raise VllmContractError(
                        "COLDSNAP_CUDA_EPOCH_TRACE_TIMEOUT_S must be numeric"
                    ) from error
                if timeout <= 0 or timeout > 600:
                    raise VllmContractError(
                        "COLDSNAP_CUDA_EPOCH_TRACE_TIMEOUT_S must be in (0, 600]"
                    )
                faulthandler.dump_traceback_later(timeout, repeat=False)
                armed = True
            try:
                return original_entrypoint(runner, *args, **kwargs)
            finally:
                if armed:
                    faulthandler.cancel_dump_traceback_later()

        setattr(in_current_epoch, _EXECUTE_MARKER, True)
        setattr(runner_class, name, in_current_epoch)

    wrap_execution_entrypoint("execute_model")
    wrap_execution_entrypoint("sample_tokens")


def _install_worker_hook(module: ModuleType) -> None:
    worker_class = getattr(module, "GPUWorker", None)
    if not isinstance(worker_class, type):
        worker_class = getattr(module, "Worker", None)
    if not isinstance(worker_class, type):
        raise VllmContractError(f"{module.__name__} does not expose GPUWorker")

    seed_init = getattr(module, "set_random_seed", None)
    original_init_device = getattr(worker_class, "init_device", None)
    if callable(seed_init) != callable(original_init_device):
        raise VllmContractError(
            f"{module.__name__} exposes an incomplete worker-init contract"
        )
    if callable(seed_init) and not getattr(seed_init, _SEED_INIT_MARKER, False):

        @functools.wraps(seed_init)
        def seed_init_in_runtime_pool(*args: Any, **kwargs: Any) -> Any:
            worker = getattr(_bootstrap_tls, "worker", None)
            if worker is None:
                return seed_init(*args, **kwargs)
            if getattr(_bootstrap_tls, "scope", None) is not None:
                raise VllmContractError(
                    "CUDA epoch V2 seed bootstrap pool is already active"
                )
            from vllm.device_allocator import get_mem_allocator_instance

            allocator = get_mem_allocator_instance()
            scope = _RuntimePoolScope(allocator)
            scope.enter()
            _bootstrap_tls.scope = scope
            try:
                return seed_init(*args, **kwargs)
            except BaseException:
                del _bootstrap_tls.scope
                scope.exit(sys.exc_info())
                raise

        setattr(seed_init_in_runtime_pool, _SEED_INIT_MARKER, True)
        module.set_random_seed = seed_init_in_runtime_pool

    if callable(original_init_device) and not getattr(
        original_init_device, _WORKER_INIT_MARKER, False
    ):

        @functools.wraps(original_init_device)
        def init_device_with_bootstrap_cleanup(
            worker: Any, *args: Any, **kwargs: Any
        ) -> Any:
            use_v2 = bool(
                getattr(
                    worker,
                    "use_v2_model_runner",
                    getattr(getattr(worker, "vllm_config", None), "use_v2_model_runner", False),
                )
            )
            if use_v2:
                if getattr(_bootstrap_tls, "worker", None) is not None:
                    raise VllmContractError(
                        "CUDA epoch V2 worker bootstrap is already active"
                    )
                _bootstrap_tls.worker = worker
            try:
                result = original_init_device(worker, *args, **kwargs)
            except BaseException:
                _stop_cuda_bootstrap_trace()
                scope = getattr(_bootstrap_tls, "scope", None)
                if scope is not None:
                    del _bootstrap_tls.scope
                    if scope.active:
                        scope.exit(sys.exc_info())
                raise
            finally:
                if getattr(_bootstrap_tls, "worker", None) is worker:
                    del _bootstrap_tls.worker
            scope = getattr(_bootstrap_tls, "scope", None)
            if scope is not None:
                del _bootstrap_tls.scope
                if scope.active:
                    scope.exit()
                raise VllmContractError(
                    "CUDA epoch V2 runner did not adopt its seed bootstrap pool"
                )
            return result

        setattr(init_device_with_bootstrap_cleanup, _WORKER_INIT_MARKER, True)
        worker_class.init_device = init_device_with_bootstrap_cleanup

    original = getattr(worker_class, "_maybe_get_memory_pool_context", None)
    if not callable(original):
        raise VllmContractError(
            f"{module.__name__}.GPUWorker lacks _maybe_get_memory_pool_context"
        )
    if not getattr(original, _WORKER_MARKER, False):

        @functools.wraps(original)
        def memory_pool_with_runtime(worker: Any, tag: str):
            if tag != "weights":
                return original(worker, tag)
            # Upstream asserts that current usage is zero before its first weight
            # pool.  The admitted runner's earlier `runtime` pool is deliberate,
            # and _register_runner still enforces the single-instance invariant
            # that assertion was protecting.
            runner = _current_runner()
            if getattr(worker, "model_runner", None) is not runner:
                raise VllmContractError(
                    "CUDA epoch weight pool does not belong to the admitted runner"
                )
            from vllm.device_allocator import get_mem_allocator_instance

            allocator = get_mem_allocator_instance()
            use_pool = getattr(allocator, "use_memory_pool", None)
            if not callable(use_pool):
                raise VllmContractError(
                    "vLLM CUDA allocator lacks use_memory_pool for weights"
                )
            return use_pool(tag=tag)

        setattr(memory_pool_with_runtime, _WORKER_MARKER, True)
        worker_class._maybe_get_memory_pool_context = memory_pool_with_runtime

    original_wake = getattr(worker_class, "wake_up", None)
    if not callable(original_wake):
        raise VllmContractError(f"{module.__name__}.GPUWorker lacks wake_up")
    if not getattr(original_wake, _WORKER_WAKE_MARKER, False):

        @functools.wraps(original_wake)
        def wake_up_in_epoch(worker: Any, *args: Any, **kwargs: Any) -> Any:
            lifecycle = getattr(worker, _WORKER_EPOCH_ATTR, None)
            if lifecycle is None:
                return original_wake(worker, *args, **kwargs)
            result = lifecycle.resume(
                lambda: original_wake(worker, *args, **kwargs)
            )
            setattr(worker, _WORKER_LAST_EPOCH_ATTR, lifecycle.inventory())
            setattr(worker, _WORKER_EPOCH_ATTR, None)
            return result

        setattr(wake_up_in_epoch, _WORKER_WAKE_MARKER, True)
        worker_class.wake_up = wake_up_in_epoch

    def epoch_reset(worker: Any) -> dict[str, Any]:
        if getattr(worker, _WORKER_EPOCH_ATTR, None) is not None:
            raise VllmContractError("vLLM worker already owns a CUDA epoch")
        from coldsnap_cuda_epoch import VllmCudaEpoch

        lifecycle = VllmCudaEpoch.capture(externalize_active=True)
        setattr(worker, _WORKER_LAST_EPOCH_ATTR, None)
        setattr(worker, _WORKER_EPOCH_ATTR, lifecycle)
        return lifecycle.reset()

    def epoch_resume(worker: Any) -> dict[str, Any]:
        lifecycle = getattr(worker, _WORKER_EPOCH_ATTR, None)
        if lifecycle is None:
            raise VllmContractError("vLLM worker has no captured CUDA epoch")
        worker.wake_up()
        inventory = getattr(worker, _WORKER_LAST_EPOCH_ATTR, None)
        if not isinstance(inventory, dict):
            raise VllmContractError("vLLM worker lost its completed CUDA epoch")
        return inventory

    def epoch_rebind(worker: Any) -> dict[str, Any]:
        lifecycle = getattr(worker, _WORKER_EPOCH_ATTR, None)
        if lifecycle is None:
            raise VllmContractError("vLLM worker has no captured CUDA epoch")
        return lifecycle.rebind()

    def epoch_context_probe(worker: Any) -> dict[str, Any]:
        lifecycle = getattr(worker, _WORKER_EPOCH_ATTR, None)
        if lifecycle is None:
            raise VllmContractError("vLLM worker has no captured CUDA epoch")
        return lifecycle.probe_context()

    def epoch_status(worker: Any) -> dict[str, Any] | None:
        lifecycle = getattr(worker, _WORKER_EPOCH_ATTR, None)
        return None if lifecycle is None else lifecycle.inventory()

    def epoch_last_status(worker: Any) -> dict[str, Any] | None:
        inventory = getattr(worker, _WORKER_LAST_EPOCH_ATTR, None)
        return inventory if isinstance(inventory, dict) else None

    def epoch_audit(worker: Any) -> dict[str, Any]:
        from coldsnap_cuda_epoch import allocations as current_allocations
        from vllm.device_allocator import get_mem_allocator_instance

        current = current_allocations(get_mem_allocator_instance())
        return audit_cuda_epoch_allocations(
            tuple((item.pointer, item.size) for item in current)
        )

    def epoch_nccl_probe(worker: Any) -> dict[str, Any]:
        """Run one tiny eager TP sum on the current CUDA epoch."""
        runner = getattr(worker, "model_runner", None)
        if runner is not _current_runner():
            raise VllmContractError(
                "CUDA epoch NCCL probe does not belong to the admitted runner"
            )
        _activate_cuda_execution_thread(runner)
        import torch
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        communicator = getattr(group, "device_communicator", None)
        all_reduce = getattr(communicator, "all_reduce", None)
        if not callable(all_reduce):
            raise VllmContractError(
                "CUDA epoch NCCL probe requires a TP device communicator"
            )
        rank = int(group.rank_in_group)
        world_size = int(group.world_size)
        value = torch.tensor([rank + 1.0], dtype=torch.float32, device=runner.device)
        reduced = all_reduce(value)
        torch.cuda.synchronize(runner.device)
        actual = float(reduced.item())
        expected = world_size * (world_size + 1) / 2
        if actual != expected:
            raise VllmContractError(
                f"CUDA epoch NCCL probe expected {expected}, got {actual}"
            )
        return {
            "rank": rank,
            "world_size": world_size,
            "value": actual,
        }

    worker_class.coldsnap_cuda_epoch_reset = epoch_reset
    worker_class.coldsnap_cuda_epoch_context_probe = epoch_context_probe
    worker_class.coldsnap_cuda_epoch_rebind = epoch_rebind
    worker_class.coldsnap_cuda_epoch_resume = epoch_resume
    worker_class.coldsnap_cuda_epoch_status = epoch_status
    worker_class.coldsnap_cuda_epoch_last_status = epoch_last_status
    worker_class.coldsnap_cuda_epoch_audit = epoch_audit
    worker_class.coldsnap_cuda_epoch_nccl_probe = epoch_nccl_probe


def _install_buffer_hook(module: ModuleType) -> None:
    buffer_class = getattr(module, "CpuGpuBuffer", None)
    if not isinstance(buffer_class, type):
        raise VllmContractError(f"{module.__name__} does not expose CpuGpuBuffer")
    original = getattr(buffer_class, "__init__", None)
    if not callable(original):
        raise VllmContractError(f"{module.__name__}.CpuGpuBuffer lacks __init__")
    if getattr(original, _BUFFER_MARKER, False):
        return

    @functools.wraps(original)
    def init_pageable(buffer: Any, *args: Any, **kwargs: Any) -> None:
        # CpuGpuBuffer's default is evaluated when vllm.v1.utils imports, so
        # changing its module-level PIN_MEMORY later is insufficient.
        kwargs["pin_memory"] = False
        original(buffer, *args, **kwargs)

    setattr(init_pageable, _BUFFER_MARKER, True)
    buffer_class.__init__ = init_pageable


def _install_v2_buffer_hook(module: ModuleType) -> None:
    """Replace context-bound UVA aliases with managed CUDA staging mirrors."""
    buffer_class = getattr(module, "UvaBuffer", None)
    pool_class = getattr(module, "UvaBufferPool", None)
    if not isinstance(buffer_class, type) or not isinstance(pool_class, type):
        raise VllmContractError(
            f"{module.__name__} lacks the V2 UVA buffer contract"
        )

    original_init = getattr(buffer_class, "__init__", None)
    if not callable(original_init):
        raise VllmContractError(f"{module.__name__}.UvaBuffer lacks __init__")
    if not getattr(original_init, _V2_UVA_BUFFER_MARKER, False):

        @functools.wraps(original_init)
        def init_managed_staging(buffer: Any, size: Any, dtype: Any) -> None:
            import torch

            buffer.cpu = torch.zeros(
                size, dtype=dtype, device="cpu", pin_memory=False
            )
            buffer.np = buffer.cpu.numpy()
            buffer.uva = torch.zeros(
                size,
                dtype=dtype,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

        setattr(init_managed_staging, _V2_UVA_BUFFER_MARKER, True)
        buffer_class.__init__ = init_managed_staging

    original_copy = getattr(pool_class, "copy_to_uva", None)
    if not callable(original_copy):
        raise VllmContractError(
            f"{module.__name__}.UvaBufferPool lacks copy_to_uva"
        )
    if not getattr(original_copy, _V2_UVA_COPY_MARKER, False):

        @functools.wraps(original_copy)
        def copy_to_managed_staging(pool: Any, value: Any) -> Any:
            import torch

            pool._curr = (pool._curr + 1) % pool.max_concurrency
            buffer = pool._uva_bufs[pool._curr]
            destination = buffer.cpu if isinstance(value, torch.Tensor) else buffer.np
            count = len(value)
            destination[:count] = value
            mirror = buffer.uva[:count]
            mirror.copy_(buffer.cpu[:count], non_blocking=False)
            return mirror

        setattr(copy_to_managed_staging, _V2_UVA_COPY_MARKER, True)
        pool_class.copy_to_uva = copy_to_managed_staging


def _install_async_output_hook(module: ModuleType) -> None:
    """Make V2 D2H output staging pageable and release terminal resources."""
    original_copy = getattr(module, "async_copy_to_np", None)
    if not callable(original_copy):
        raise VllmContractError(f"{module.__name__} lacks async_copy_to_np")
    if not getattr(original_copy, _ASYNC_COPY_MARKER, False):

        @functools.wraps(original_copy)
        def copy_to_pageable_np(tensor: Any):
            # V2's helper normally requests a nonblocking D2H copy. PyTorch
            # backs that NumPy array with pinned CPU storage, and the worker's
            # response path can retain the backing Tensor after AsyncOutput is
            # serialized to the executor. A process-wide CUDA context reset
            # cannot admit that allocator-owned registration. Cooperative
            # epoch mode already requires synchronous scheduling, so perform
            # this tiny output copy synchronously into ordinary pageable RAM.
            return tensor.to("cpu", non_blocking=False).numpy()

        setattr(copy_to_pageable_np, _ASYNC_COPY_MARKER, True)
        module.async_copy_to_np = copy_to_pageable_np

    output_class = getattr(module, "AsyncOutput", None)
    if not isinstance(output_class, type):
        raise VllmContractError(f"{module.__name__} does not expose AsyncOutput")
    original = getattr(output_class, "get_output", None)
    if not callable(original):
        raise VllmContractError(f"{module.__name__}.AsyncOutput lacks get_output")
    if getattr(original, _ASYNC_OUTPUT_MARKER, False):
        return

    @functools.wraps(original)
    def get_output_and_release_epoch_resources(output: Any, *args: Any, **kwargs: Any):
        try:
            return original(output, *args, **kwargs)
        finally:
            # get_output has synchronized copy_event and converted every value
            # needed by ModelRunnerOutput into ordinary Python/CPU state. The
            # executor can briefly retain the completed AsyncOutput in a list;
            # do not let its pinned ndarray bases, CUDA tensors, stream, or
            # event leak across a later process-wide CUDA context reset.
            _release_async_output_resources(output)

    setattr(get_output_and_release_epoch_resources, _ASYNC_OUTPUT_MARKER, True)
    output_class.get_output = get_output_and_release_epoch_resources


def install_cuda_epoch_runtime_hooks() -> bool:
    """Install the opt-in V1/V2 runner allocation hooks before worker import."""
    global _installed
    if not _enabled():
        return False
    if _installed:
        return True
    _enable_cuda_memory_history()
    _disable_vllm_pin_memory()
    for runner_module in _RUNNER_MODULES:
        after_module_import(
            runner_module,
            f"cuda-epoch-runtime-{runner_module.rsplit('.', 1)[-1]}",
            _install_runner_hook,
        )
    after_module_import(
        _WORKER_MODULE,
        "cuda-epoch-weight-pool",
        _install_worker_hook,
    )
    after_module_import(
        _BUFFER_MODULE,
        "cuda-epoch-pageable-buffers",
        _install_buffer_hook,
    )
    after_module_import(
        _V2_BUFFER_MODULE,
        "cuda-epoch-managed-uva-buffers",
        _install_v2_buffer_hook,
    )
    after_module_import(
        _ASYNC_OUTPUT_MODULE,
        "cuda-epoch-v2-pageable-output",
        _install_async_output_hook,
    )
    _installed = True
    return True


@dataclass(frozen=True)
class _ResourceRecipe:
    name: str
    kind: str
    blocking: bool = False
    target: Any | None = None
    attribute: str | None = None
    count: int = 1


@dataclass
class _PinnedWorkspaceRecipe:
    owner: Any
    owner_type: str
    attribute: str
    pageable: Any
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Any
    bytes: int
    rehydrated: bool = False

    def rehydrate(self, torch: Any) -> None:
        if self.rehydrated:
            raise VllmContractError(
                f"CUDA epoch workspace {self.owner_type}.{self.attribute} "
                "was already rehydrated"
            )
        if getattr(self.owner, self.attribute, None) is not self.pageable:
            raise VllmContractError(
                f"CUDA epoch workspace {self.owner_type}.{self.attribute} changed "
                "while CUDA was reset"
            )
        pinned = torch.empty_strided(
            self.shape,
            self.stride,
            dtype=self.dtype,
            device="cpu",
            pin_memory=True,
        )
        pinned.copy_(self.pageable)
        setattr(self.owner, self.attribute, pinned)
        self.pageable = None
        self.rehydrated = True


@dataclass(frozen=True)
class _ReleasedAsyncOutput:
    owner_type: str
    pinned_array_bytes: int


def _release_async_output_resources(output: Any) -> int:
    pinned_array_bytes = 0
    for attribute in _ASYNC_OUTPUT_ARRAY_ATTRS:
        value = getattr(output, attribute, None)
        pinned_array_bytes += int(getattr(value, "nbytes", 0) or 0)
    for attribute in _ASYNC_OUTPUT_RESOURCE_ATTRS:
        if hasattr(output, attribute):
            setattr(output, attribute, None)
    if hasattr(output, "prompt_logprobs_dict"):
        output.prompt_logprobs_dict = {}
    return pinned_array_bytes


def _release_completed_v2_async_outputs() -> tuple[_ReleasedAsyncOutput, ...]:
    """Drop worker-side copies already handed across the executor boundary."""
    released: list[_ReleasedAsyncOutput] = []
    for output in gc.get_objects():
        try:
            if _type_name(output) != _ASYNC_OUTPUT_TYPE:
                continue
            event = getattr(output, "copy_event", None)
            has_resources = event is not None or any(
                getattr(output, attribute, None) is not None
                for attribute in _ASYNC_OUTPUT_RESOURCE_ATTRS
            )
            if not has_resources:
                continue
            if not bool(getattr(output, "copy_event_recorded", False)):
                raise VllmContractError(
                    "CUDA epoch found an incomplete V2 asynchronous output"
                )
            if event is None:
                raise VllmContractError(
                    "CUDA epoch V2 asynchronous output lost its copy event"
                )
            event.synchronize()
            released.append(
                _ReleasedAsyncOutput(
                    owner_type=_type_name(output),
                    pinned_array_bytes=_release_async_output_resources(output),
                )
            )
        except (ReferenceError, TypeError):
            continue
    return tuple(released)


def _dehydrate_flashinfer_workspaces() -> tuple[_PinnedWorkspaceRecipe, ...]:
    """Replace FlashInfer's context-bound pinned workspaces with CPU copies."""
    try:
        import torch
    except ImportError:
        return ()

    recipes: list[_PinnedWorkspaceRecipe] = []
    try:
        for owner in gc.get_objects():
            owner_class = type(owner)
            owner_type = f"{owner_class.__module__}.{owner_class.__qualname__}"
            if owner_type not in _FLASHINFER_PINNED_WORKSPACE_OWNERS:
                continue
            value = getattr(owner, _FLASHINFER_PINNED_WORKSPACE_ATTR, None)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise VllmContractError(
                    f"FlashInfer CUDA epoch workspace {owner_type}."
                    f"{_FLASHINFER_PINNED_WORKSPACE_ATTR} is not a tensor"
                )
            if value.device.type != "cpu":
                raise VllmContractError(
                    f"FlashInfer CUDA epoch workspace {owner_type} is not on CPU"
                )
            if not value.is_pinned():
                continue
            shape = tuple(int(item) for item in value.shape)
            stride = tuple(int(item) for item in value.stride())
            pageable = torch.empty_strided(
                shape,
                stride,
                dtype=value.dtype,
                device="cpu",
                pin_memory=False,
            )
            pageable.copy_(value)
            setattr(owner, _FLASHINFER_PINNED_WORKSPACE_ATTR, pageable)
            storage = value.untyped_storage()
            recipes.append(
                _PinnedWorkspaceRecipe(
                    owner=owner,
                    owner_type=owner_type,
                    attribute=_FLASHINFER_PINNED_WORKSPACE_ATTR,
                    pageable=pageable,
                    shape=shape,
                    stride=stride,
                    dtype=value.dtype,
                    bytes=int(storage.nbytes()),
                )
            )
    except BaseException:
        for recipe in reversed(recipes):
            recipe.rehydrate(torch)
        raise
    return tuple(recipes)


def _v1_resource_recipes(runner: Any) -> tuple[_ResourceRecipe, ...]:
    recipes: list[_ResourceRecipe] = []
    stream = runner.async_output_copy_stream
    event = runner.prepare_inputs_event
    if (stream is None) != (event is None):
        raise VllmContractError(
            "vLLM async output stream and prepare event must be present together"
        )
    if stream is not None:
        recipes.append(_ResourceRecipe("async_output_copy_stream", "cuda_stream"))
        recipes.append(
            _ResourceRecipe("prepare_inputs_event", "cuda_event", blocking=True)
        )
    if runner.transfer_event is None:
        raise VllmContractError("vLLM runner transfer_event is unexpectedly absent")
    recipes.append(_ResourceRecipe("transfer_event", "torch_event"))
    return tuple(recipes)


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _direct_resource(
    name: str,
    kind: str,
    target: Any,
    attribute: str,
    *,
    blocking: bool = False,
) -> _ResourceRecipe:
    return _ResourceRecipe(
        name,
        kind,
        blocking=blocking,
        target=target,
        attribute=attribute,
    )


def _sequence_resource(
    name: str,
    kind: str,
    values: Any,
) -> _ResourceRecipe:
    if not isinstance(values, list) or not values or any(item is None for item in values):
        raise VllmContractError(
            f"CUDA epoch resource {name} must be a non-empty initialized list"
        )
    return _ResourceRecipe(name, kind, target=values, count=len(values))


def _deepseek_v4_resource_recipes(runner: Any) -> list[_ResourceRecipe]:
    roots = [getattr(runner, "model", None)]
    speculator = getattr(runner, "speculator", None)
    roots.append(getattr(speculator, "model", None))
    modules: list[Any] = []
    seen_modules: set[int] = set()
    for root in roots:
        get_modules = getattr(root, "modules", None)
        if not callable(get_modules):
            continue
        for owner in get_modules():
            if id(owner) in seen_modules:
                continue
            seen_modules.add(id(owner))
            if not _type_name(owner).startswith("vllm.models.deepseek_v4."):
                continue
            if not all(
                hasattr(owner, attribute)
                for attribute in ("aux_stream_list", "ln_events", "attn_event_pool")
            ):
                continue
            modules.append(owner)

    recipes: list[_ResourceRecipe] = []
    seen_sequences: set[int] = set()
    for index, owner in enumerate(modules):
        resources = (
            ("aux_stream_list", "cuda_stream_list"),
            ("ln_events", "cuda_event_list"),
        )
        event_pool = owner.attn_event_pool
        captured = getattr(event_pool, "_captured_event_sets", None)
        if captured:
            raise VllmContractError(
                "CUDA epoch DeepSeek-V4 attention retains captured CUDA events"
            )
        resources += (("attn_event_pool.default_events", "cuda_event_list"),)
        for attribute, kind in resources:
            if attribute == "attn_event_pool.default_events":
                values = getattr(event_pool, "default_events", None)
            else:
                values = getattr(owner, attribute)
            if values is None or id(values) in seen_sequences:
                continue
            seen_sequences.add(id(values))
            recipes.append(
                _sequence_resource(
                    f"deepseek_v4_attention_{index}.{attribute}", kind, values
                )
            )
    return recipes


def _v2_resource_recipes(runner: Any) -> tuple[_ResourceRecipe, ...]:
    recipes = [
        _ResourceRecipe("output_copy_stream", "cuda_stream"),
        _ResourceRecipe("main_stream", "cuda_current_stream"),
    ]
    handler = runner.draft_tokens_handler
    if _type_name(handler) != (
        "vllm.v1.worker.gpu.spec_decode.utils.DraftTokensHandler"
    ):
        raise VllmContractError(
            "CUDA epoch V2 runner has an unsupported draft-token handler"
        )
    recipes.extend(
        (
            _direct_resource(
                "draft_tokens_handler.copy_stream",
                "cuda_stream",
                handler,
                "copy_stream",
            ),
            _direct_resource(
                "draft_tokens_handler.copy_event",
                "cuda_event",
                handler,
                "copy_event",
                blocking=True,
            ),
        )
    )
    capacity = getattr(runner, "verification_capacity_manager", None)
    if capacity is not None:
        if not _type_name(capacity).startswith(
            "vllm.v1.worker.gpu.spec_decode.capacity."
        ):
            raise VllmContractError(
                "CUDA epoch V2 runner has an unsupported capacity manager"
            )
        recipes.extend(
            (
                _direct_resource(
                    "verification_capacity_manager.copy_stream",
                    "cuda_stream",
                    capacity,
                    "copy_stream",
                ),
                _direct_resource(
                    "verification_capacity_manager.copy_event",
                    "cuda_event",
                    capacity,
                    "copy_event",
                    blocking=True,
                ),
            )
        )
    torch_utils = sys.modules.get("vllm.utils.torch_utils")
    if torch_utils is not None and getattr(torch_utils, "_aux_stream", None) is not None:
        recipes.append(
            _direct_resource(
                "vllm.utils.torch_utils._aux_stream",
                "cuda_stream",
                torch_utils,
                "_aux_stream",
            )
        )
    recipes.extend(_deepseek_v4_resource_recipes(runner))
    return tuple(recipes)


def _resource_recipes(runner: Any) -> tuple[_ResourceRecipe, ...]:
    if _runner_kind(runner) == _RUNNER_V2:
        return _v2_resource_recipes(runner)
    return _v1_resource_recipes(runner)


def _quiescent_v1_runner(runner: Any) -> None:
    requests = runner.requests
    if any(getattr(request, "generator", None) is not None for request in requests.values()):
        raise VllmContractError(
            "CUDA epoch does not yet reconstruct per-request CUDA generators"
        )
    input_batch = runner.input_batch
    if getattr(input_batch, "generators", None):
        raise VllmContractError(
            "CUDA epoch does not yet reconstruct input-batch CUDA generators"
        )
    # No request is alive, so references to per-request async copies are stale
    # bookkeeping rather than semantic state.
    input_batch.sampled_token_ids_cpu = None
    input_batch.async_copy_ready_event = None


def _quiescent_v2_runner(runner: Any) -> None:
    capacity = getattr(runner, "verification_capacity_manager", None)
    if capacity is not None and bool(getattr(capacity, "copy_event_pending", False)):
        flush = getattr(capacity, "_flush_draft_token_capacity_copy", None)
        if not callable(flush):
            raise VllmContractError(
                "CUDA epoch DSpark capacity manager cannot drain its pending copy"
            )
        flush()
        if bool(getattr(capacity, "copy_event_pending", False)):
            raise VllmContractError(
                "CUDA epoch DSpark capacity copy remained pending after drain"
            )
    handler = runner.draft_tokens_handler
    copy_event = getattr(handler, "copy_event", None)
    if copy_event is None:
        raise VllmContractError("CUDA epoch DSpark draft copy event is absent")
    # The engine has consumed any completed DraftTokenIds before entering the
    # collective sleep RPC. Synchronization is idempotent and prevents a stale
    # event generation from crossing the context boundary.
    copy_event.synchronize()


def _quiescent_runner(runner: Any) -> None:
    # A synchronous LLM.generate() can return after the engine scheduler has
    # finished a request but before the worker consumes the next
    # finished_req_ids update.  Those CachedRequestState objects are ordinary
    # CPU semantic state and are removed on the next execute_model call; they
    # are not evidence of in-flight CUDA work after llm.sleep() returned.
    if _runner_kind(runner) == _RUNNER_V2:
        _quiescent_v2_runner(runner)
    else:
        _quiescent_v1_runner(runner)
    for name in ("execute_model_state", "kv_connector_output"):
        if getattr(runner, name, None) is not None:
            raise VllmContractError(
                f"CUDA epoch requires runner {name} to be empty"
            )
    encoder_cache = getattr(runner, "encoder_cache", None)
    if encoder_cache is not None and len(encoder_cache):
        raise VllmContractError(
            "CUDA epoch requires an empty multimodal encoder cache"
        )


def _tensor_extents(
    objects: list[Any],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    import torch

    cuda: set[tuple[int, int]] = set()
    pinned: set[tuple[int, int]] = set()
    for value in objects:
        try:
            if not isinstance(value, torch.Tensor):
                continue
            storage = value.untyped_storage()
            pointer = int(storage.data_ptr())
            size = int(storage.nbytes())
            if pointer <= 0 or size <= 0:
                continue
            if value.device.type == "cuda":
                cuda.add((pointer, size))
            elif value.device.type == "cpu" and value.is_pinned():
                pinned.add((pointer, size))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            continue
    return cuda, pinned


def _live_tensor_extents() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return unique live CUDA and pinned-CPU storage extents."""
    return _tensor_extents(gc.get_objects())


def _tensor_diagnostics(
    extents: set[tuple[int, int]],
    *,
    device_type: str,
    require_pinned: bool = False,
) -> list[dict[str, Any]]:
    """Describe a bounded storage sample without retaining it on success."""
    try:
        import torch
    except ImportError:
        return []

    remaining = set(sorted(extents)[:3])
    diagnostics: list[dict[str, Any]] = []
    for value in gc.get_objects():
        if not remaining:
            break
        try:
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != device_type
                or (require_pinned and not value.is_pinned())
            ):
                continue
            storage = value.untyped_storage()
            extent = (int(storage.data_ptr()), int(storage.nbytes()))
            if extent not in remaining or (
                require_pinned and not value.is_pinned()
            ):
                continue
            owners: list[str] = []
            referrer_types: set[str] = set()
            for referrer in gc.get_referrers(value):
                if referrer is diagnostics:
                    continue
                referrer_types.add(
                    f"{type(referrer).__module__}.{type(referrer).__qualname__}"
                )
                if not isinstance(referrer, dict):
                    if not isinstance(referrer, (list, tuple)):
                        continue
                    indexes = [
                        index
                        for index, candidate in enumerate(referrer)
                        if candidate is value
                    ][:4]
                    if indexes:
                        owners.extend(
                            f"{type(referrer).__module__}."
                            f"{type(referrer).__qualname__}[{index}]"
                            for index in indexes
                        )
                    continue
                matches = [
                    key
                    for key, candidate in tuple(referrer.items())
                    if candidate is value
                ][:4]
                attributes = [str(key) for key in matches]
                if not attributes:
                    continue
                module_name = referrer.get("__name__")
                if isinstance(module_name, str) and "__package__" in referrer:
                    owners.extend(f"{module_name}.{name}" for name in attributes)
                for owner in gc.get_referrers(referrer):
                    try:
                        if getattr(owner, "__dict__", None) is not referrer:
                            continue
                    except (AttributeError, ReferenceError, RuntimeError, TypeError):
                        continue
                    owner_type = type(owner)
                    owners.extend(
                        f"{owner_type.__module__}.{owner_type.__qualname__}.{name}"
                        for name in attributes
                    )
                    if len(owners) >= 8:
                        break
                if len(owners) >= 8:
                    break
            diagnostics.append(
                {
                    "extent": extent,
                    "shape": tuple(int(item) for item in value.shape),
                    "stride": tuple(int(item) for item in value.stride()),
                    "dtype": str(value.dtype),
                    "owners": sorted(set(owners))[:8],
                    "referrer_types": sorted(referrer_types)[:8],
                }
            )
            remaining.remove(extent)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            continue
    return diagnostics


def _pinned_tensor_diagnostics(
    extents: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    return _tensor_diagnostics(
        extents,
        device_type="cpu",
        require_pinned=True,
    )


def _cuda_tensor_diagnostics(
    extents: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    return _tensor_diagnostics(extents, device_type="cuda")


def _cuda_allocation_history(
    extents: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Resolve live extents to bounded caching-allocator allocation stacks."""
    if not _memory_history_enabled or not extents:
        return []
    try:
        import torch

        snapshot = torch.cuda.memory._snapshot()
    except (AttributeError, RuntimeError):
        return []
    remaining = set(extents)
    result: list[dict[str, Any]] = []
    for segment in snapshot.get("segments", ()):
        cursor = int(segment.get("address", 0))
        for block in segment.get("blocks", ()):
            address = int(block.get("address", cursor))
            size = int(block.get("size", 0))
            cursor = address + size
            matches = [
                extent
                for extent in remaining
                if address <= extent[0] and extent[0] + extent[1] <= cursor
            ]
            if not matches:
                continue
            history = block.get("history") or ()
            frames = block.get("frames") or ()
            if history:
                # Active blocks normally carry one allocation-history entry.
                # Select the entry whose subextent contains the tensor when
                # PyTorch records more than one reuse within a cached block.
                for entry in reversed(history):
                    entry_address = int(entry.get("addr", address))
                    entry_size = int(entry.get("real_size", size))
                    if any(
                        entry_address <= extent[0]
                        and extent[0] + extent[1] <= entry_address + entry_size
                        for extent in matches
                    ):
                        frames = entry.get("frames") or frames
                        break
            formatted_frames = [
                {
                    "filename": str(frame.get("filename", "")),
                    "line": int(frame.get("line", 0)),
                    "name": str(frame.get("name", "")),
                }
                for frame in frames
                if isinstance(frame, dict)
            ][-24:]
            for extent in matches:
                result.append(
                    {
                        "extent": extent,
                        "block_address": address,
                        "block_size": size,
                        "requested_size": int(block.get("requested_size", 0)),
                        "state": str(block.get("state", "")),
                        "frames": formatted_frames,
                    }
                )
                remaining.remove(extent)
    return sorted(result, key=lambda item: item["extent"])


def _bootstrap_traces_for(
    extents: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    return [
        _bootstrap_allocation_traces[pointer]
        for pointer, _ in extents
        if pointer in _bootstrap_allocation_traces
    ]


def _live_tensor_audit(
    allocations: tuple[tuple[int, int], ...],
) -> tuple[
    set[tuple[int, int]],
    set[tuple[int, int]],
    list[tuple[int, int]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    # Keep this snapshot alive only while diagnostics run. Some extension
    # tensors are visible to Python's cyclic GC without another durable Python
    # owner; a second scan after releasing this list loses their shape/type.
    objects = gc.get_objects()
    cuda_extents, pinned_extents = _tensor_extents(objects)
    uncovered = sorted(
        extent for extent in cuda_extents if not _covered(extent, allocations)
    )
    pinned_diagnostics = _pinned_tensor_diagnostics(pinned_extents)
    cuda_diagnostics = _cuda_tensor_diagnostics(set(uncovered))
    return (
        cuda_extents,
        pinned_extents,
        uncovered,
        pinned_diagnostics,
        cuda_diagnostics,
    )


def audit_cuda_epoch_allocations(
    allocations: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    """Audit live CUDA storage coverage without changing the CUDA epoch."""
    _stop_cuda_bootstrap_trace()
    runner = _current_runner()
    _require_runner_contract(runner)
    pool_scope = _runtime_pool_scope(runner)
    if not pool_scope.active:
        raise VllmContractError(
            "vLLM CUDA epoch runtime pool is inactive during audit"
        )
    _quiescent_runner(runner)
    b12x_cache_reset = _invalidate_b12x_context()
    gc.collect()
    cuda_extents, _, uncovered, _, diagnostics = _live_tensor_audit(allocations)
    allocation_history = _cuda_allocation_history(set(uncovered))
    bootstrap_traces = _bootstrap_traces_for(uncovered)
    return {
        "live_cuda_storage_count": len(cuda_extents),
        "unmanaged_cuda_storage_count": len(uncovered),
        "unmanaged_cuda_storage_sample": uncovered[:3],
        "unmanaged_cuda_storage_diagnostics": diagnostics,
        "unmanaged_cuda_allocation_history": allocation_history,
        "unmanaged_cuda_bootstrap_traces": bootstrap_traces,
        "cuda_memory_history_enabled": _memory_history_enabled,
        "b12x_cache_reset": b12x_cache_reset,
    }


def _invalidate_triton_context() -> tuple[int, int, int]:
    """Drop every Triton device cache and autotune choice for this context."""
    try:
        from triton.compiler.compiler import CompiledKernel
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
    except ImportError:
        return 0, 0, 0

    invalidated_kernels = 0
    invalidated_functions = 0
    invalidated_autotuners = 0
    cleared_kernel_caches: set[int] = set()
    seen: set[int] = set()
    stack: list[Any] = []
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("vllm.") or module is None:
            continue
        try:
            stack.extend(vars(module).values())
        except (AttributeError, RuntimeError, TypeError):
            continue

    while stack:
        value = stack.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            if isinstance(value, JITFunction):
                changed = False
                for cache_data in value.device_caches.values():
                    kernel_cache = cache_data[0]
                    identity = id(kernel_cache)
                    if identity in cleared_kernel_caches:
                        continue
                    cleared_kernel_caches.add(identity)
                    invalidated_kernels += len(kernel_cache)
                    kernel_cache.clear()
                    changed = True
                if changed:
                    invalidated_functions += 1
            elif isinstance(value, Autotuner):
                if value.cache:
                    invalidated_autotuners += 1
                value.cache.clear()
                value.best_config = None
            if type(value).__module__.startswith("triton."):
                wrapped = getattr(value, "fn", None)
                if wrapped is not None and wrapped is not value:
                    stack.append(wrapped)
        except (AttributeError, ImportError, ReferenceError, RuntimeError, TypeError):
            continue

    # A caller can retain a compiled kernel outside a module's JIT wrapper.
    # Its run property will lazily load the cubin again.
    for value in gc.get_objects():
        try:
            if isinstance(value, CompiledKernel):
                value.module = None
                value.function = None
                value._run = None
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            continue
    return invalidated_kernels, invalidated_functions, invalidated_autotuners


def _invalidate_b12x_context() -> bool:
    """Clear imported B12x code caches and sparse-MLA pointer state."""
    invalidated = False
    b12x = sys.modules.get("b12x")
    clear_all = getattr(b12x, "clear_all_caches", None)
    if callable(clear_all):
        clear_all()
        invalidated = True
    mla = sys.modules.get("vllm.v1.attention.backends.mla.b12x_mla_sparse")
    implementation = getattr(mla, "B12xMLASparseImpl", None)
    reset_bindings = getattr(implementation, "reset_kv_cache_binding_state", None)
    if callable(reset_bindings):
        reset_bindings()
        invalidated = True
    return invalidated


def _resource_target(recipe: _ResourceRecipe, runner: Any) -> tuple[Any, str]:
    target = runner if recipe.target is None else recipe.target
    attribute = recipe.name if recipe.attribute is None else recipe.attribute
    return target, attribute


def _clear_resource(recipe: _ResourceRecipe, runner: Any) -> None:
    if recipe.kind in {"cuda_stream_list", "cuda_event_list"}:
        values = recipe.target
        if not isinstance(values, list) or len(values) != recipe.count:
            raise VllmContractError(
                f"CUDA epoch resource list {recipe.name} changed before reset"
            )
        values[:] = [None] * recipe.count
        return
    target, attribute = _resource_target(recipe, runner)
    if recipe.kind == "cuda_current_stream":
        values = getattr(target, "__dict__", None)
        if not isinstance(values, dict):
            raise VllmContractError(
                f"CUDA epoch cached stream owner {recipe.name} has no dictionary"
            )
        values.pop(attribute, None)
        return
    setattr(target, attribute, None)


def _rebuild_resource(recipe: _ResourceRecipe, runner: Any, torch: Any) -> None:
    if recipe.kind == "cuda_stream_list":
        values = recipe.target
        if not isinstance(values, list) or len(values) != recipe.count:
            raise VllmContractError(
                f"CUDA epoch stream list {recipe.name} changed while reset"
            )
        values[:] = [
            torch.cuda.Stream(device=runner.device) for _ in range(recipe.count)
        ]
        return
    if recipe.kind == "cuda_event_list":
        values = recipe.target
        if not isinstance(values, list) or len(values) != recipe.count:
            raise VllmContractError(
                f"CUDA epoch event list {recipe.name} changed while reset"
            )
        values[:] = [torch.cuda.Event() for _ in range(recipe.count)]
        return
    if recipe.kind == "cuda_stream":
        value = torch.cuda.Stream(device=runner.device)
    elif recipe.kind == "cuda_current_stream":
        value = torch.cuda.current_stream(runner.device)
    elif recipe.kind == "cuda_event":
        value = torch.cuda.Event(blocking=recipe.blocking)
    elif recipe.kind == "torch_event":
        value = torch.Event()
    else:  # pragma: no cover - recipes are closed above.
        raise VllmContractError(
            f"unknown CUDA epoch resource kind {recipe.kind!r}"
        )
    target, attribute = _resource_target(recipe, runner)
    setattr(target, attribute, value)


def _covered(
    extent: tuple[int, int], allocations: tuple[tuple[int, int], ...]
) -> bool:
    pointer, size = extent
    end = pointer + size
    return any(start <= pointer and end <= start + length for start, length in allocations)


class VllmCudaRuntime:
    """Captured recipe for non-memory CUDA resources owned by one runner."""

    def __init__(
        self,
        runner: Any,
        resources: tuple[_ResourceRecipe, ...],
        pool_scope: _RuntimePoolScope,
        *,
        cuda_storage_count: int,
        cuda_storage_bytes: int,
        pinned_workspaces: tuple[_PinnedWorkspaceRecipe, ...] = (),
        released_async_outputs: tuple[_ReleasedAsyncOutput, ...] = (),
        b12x_cache_reset: bool = False,
    ) -> None:
        self._runner_ref = weakref.ref(runner)
        self.resources = resources
        self.pool_scope = pool_scope
        self.cuda_storage_count = cuda_storage_count
        self.cuda_storage_bytes = cuda_storage_bytes
        self.pinned_workspaces = pinned_workspaces
        self.released_async_outputs = released_async_outputs
        self.triton_kernel_count = 0
        self.triton_function_count = 0
        self.triton_autotuner_count = 0
        self.compiler_cache_reset = False
        self.b12x_cache_reset = b12x_cache_reset
        self._sealed = False

    def _runner(self) -> Any:
        runner = self._runner_ref()
        if runner is None:
            raise VllmContractError("vLLM CUDA epoch runner was destroyed")
        return runner

    def seal(self) -> None:
        """Close the epoch pool while all of its mapped blocks are still valid."""
        if self._sealed:
            raise VllmContractError("vLLM CUDA epoch runtime is already sealed")
        _quiescent_runner(self._runner())
        self.pool_scope.exit()
        self._sealed = True

    def prepare_reset(self) -> None:
        import torch

        _quiescent_runner(self._runner())
        if not self._sealed or self.pool_scope.active:
            raise VllmContractError(
                "vLLM CUDA epoch runtime pool was not sealed before reset"
            )
        runner = self._runner()
        for recipe in self.resources:
            _clear_resource(recipe, runner)
        self.b12x_cache_reset = (
            _invalidate_b12x_context() or self.b12x_cache_reset
        )
        (
            self.triton_kernel_count,
            self.triton_function_count,
            self.triton_autotuner_count,
        ) = _invalidate_triton_context()
        compiler_reset = getattr(getattr(torch, "compiler", None), "reset", None)
        if not callable(compiler_reset):
            raise VllmContractError(
                "PyTorch 2.10+ compiler cache reset is unavailable"
            )
        compiler_reset()
        self.compiler_cache_reset = True
        gc.collect()
        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
        if not callable(empty_host_cache):
            raise VllmContractError(
                "PyTorch 2.10+ host allocator cache reset is unavailable"
            )
        empty_host_cache()

    def rebuild(self, native: Any) -> None:
        import torch

        runner = self._runner()
        _quiescent_runner(runner)
        if not self._sealed or self.pool_scope.active:
            raise VllmContractError(
                "vLLM CUDA epoch runtime must be sealed before rebuild"
            )
        activate_device = getattr(native, "activate", None)
        if not callable(activate_device):
            raise VllmContractError(
                "native CUDA epoch thread activation is unavailable"
            )
        self.pool_scope.activate_device = activate_device
        activate_device(_cuda_device_index(runner.device))
        self.pool_scope.enter()
        try:
            _activate_cuda_execution_thread(runner)
            for workspace in self.pinned_workspaces:
                workspace.rehydrate(torch)
            for recipe in self.resources:
                _rebuild_resource(recipe, runner, torch)
            self._sealed = False
        except BaseException:
            self.pool_scope.exit(sys.exc_info())
            raise

    def inventory(self) -> dict[str, Any]:
        return {
            "resource_names": [item.name for item in self.resources],
            "resource_count": len(self.resources),
            "cuda_storage_count": self.cuda_storage_count,
            "cuda_storage_bytes": self.cuda_storage_bytes,
            "pinned_cpu_storage_count": 0,
            "dehydrated_pinned_workspace_count": len(self.pinned_workspaces),
            "dehydrated_pinned_workspace_bytes": sum(
                item.bytes for item in self.pinned_workspaces
            ),
            "dehydrated_pinned_workspace_owners": [
                item.owner_type for item in self.pinned_workspaces
            ],
            "released_async_output_count": len(self.released_async_outputs),
            "released_async_output_pinned_bytes": sum(
                item.pinned_array_bytes for item in self.released_async_outputs
            ),
            "runtime_pool_active": self.pool_scope.active,
            "runtime_pool_tags": list(self.pool_scope.tags),
            "sealed": self._sealed,
            "triton_kernel_count": self.triton_kernel_count,
            "triton_function_count": self.triton_function_count,
            "triton_autotuner_count": self.triton_autotuner_count,
            "compiler_cache_reset": self.compiler_cache_reset,
            "b12x_cache_reset": self.b12x_cache_reset,
        }


def capture_cuda_epoch_runtime(
    allocations: tuple[tuple[int, int], ...],
) -> VllmCudaRuntime | None:
    """Audit live tensors and capture the installed runner's resource recipe."""
    _stop_cuda_bootstrap_trace()
    if not _enabled():
        return None
    runner = _current_runner()
    _require_runner_contract(runner)
    pool_scope = _runtime_pool_scope(runner)
    if not pool_scope.active:
        raise VllmContractError(
            "vLLM CUDA epoch runtime pool is inactive during capture"
        )
    _quiescent_runner(runner)
    released_async_outputs = (
        _release_completed_v2_async_outputs()
        if _runner_kind(runner) == _RUNNER_V2
        else ()
    )
    pinned_workspaces = _dehydrate_flashinfer_workspaces()
    # B12x holds compiled/device cache entries below Python's ordinary object
    # ownership graph. They are explicitly reconstructible, so retire them at
    # the quiescent capture boundary before enforcing complete VMM coverage.
    # Waiting until prepare_reset would make the audit reject precisely the
    # cache state that this lifecycle hook exists to discard.
    b12x_cache_reset = _invalidate_b12x_context()
    try:
        gc.collect()
        (
            cuda_extents,
            pinned_extents,
            uncovered,
            pinned_diagnostics,
            cuda_diagnostics,
        ) = _live_tensor_audit(allocations)
        if pinned_extents:
            sample = sorted(pinned_extents)[:3]
            raise VllmContractError(
                "CUDA epoch found live pinned CPU storages after pageable admission: "
                f"count={len(pinned_extents)} sample={sample} "
                f"diagnostics={pinned_diagnostics}"
            )
        if uncovered:
            allocation_history = _cuda_allocation_history(set(uncovered))
            bootstrap_traces = _bootstrap_traces_for(uncovered)
            raise VllmContractError(
                "CUDA epoch found live CUDA storages outside managed VMM allocations: "
                f"count={len(uncovered)} sample={uncovered[:3]} "
                f"diagnostics={cuda_diagnostics} "
                f"allocation_history={allocation_history[:3]} "
                f"bootstrap_traces={bootstrap_traces[:3]}"
            )
    except BaseException:
        if pinned_workspaces:
            import torch

            for workspace in reversed(pinned_workspaces):
                workspace.rehydrate(torch)
        raise
    return VllmCudaRuntime(
        runner,
        _resource_recipes(runner),
        pool_scope,
        cuda_storage_count=len(cuda_extents),
        cuda_storage_bytes=sum(size for _, size in cuda_extents),
        pinned_workspaces=pinned_workspaces,
        released_async_outputs=released_async_outputs,
        b12x_cache_reset=b12x_cache_reset,
    )

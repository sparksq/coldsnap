# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Structural contracts for the SGLang process-snapshot surface we consume."""

from __future__ import annotations

import hashlib
import inspect
import types
from dataclasses import dataclass
from pathlib import Path


class SGLangCompatibilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Contract:
    target: object
    name: str
    parameters: frozenset[str]


@dataclass(frozen=True)
class WeightUpdaterContract:
    owner: type
    hook_prefix: str


CONTRACT_MODE = "process-snapshot-v1"
_CONTRACT_MODULES = (
    "sglang.srt.constants",
    "sglang.srt.distributed.device_communicators.pynccl_allocator",
    "sglang.srt.managers.scheduler_components.weight_updater",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.tokenizer_manager",
    "sglang.srt.managers.tokenizer_control_mixin",
    "sglang.srt.model_loader.loader",
    "sglang.srt.model_executor.model_runner",
    "sglang.srt.model_executor.model_runner_components.cuda_graph_setup",
    "sglang.srt.model_executor.runner_backend.base_cuda_graph_backend",
    "sglang.srt.model_executor.runner_utils.pool",
    "sglang.srt.plugins.hook_registry",
    "sglang.srt.speculative.base_spec_worker",
    "sglang.srt.utils.common",
    "sglang.srt.utils.torch_memory_saver_adapter",
)


def contract_mode() -> str:
    return CONTRACT_MODE


def contract_digest() -> str:
    """Hash the exact SGLang source surface that gives snapshots meaning."""
    digest = hashlib.sha256()
    digest.update(CONTRACT_MODE.encode("utf-8"))
    digest.update(b"\0")
    for module_name in _CONTRACT_MODULES:
        module = __import__(module_name, fromlist=["__name__"])
        path = Path(inspect.getsourcefile(module) or "")
        if not path.is_file():
            raise SGLangCompatibilityError(f"cannot locate SGLang contract module {module_name}")
        digest.update(module_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require(contract: Contract) -> None:
    try:
        signature = inspect.signature(contract.target)
    except (TypeError, ValueError) as error:
        raise SGLangCompatibilityError(f"cannot inspect SGLang contract {contract.name}") from error
    missing = contract.parameters - set(signature.parameters)
    if missing:
        raise SGLangCompatibilityError(
            f"SGLang contract {contract.name} is missing parameters {sorted(missing)}"
        )


def resolve_weight_updater_contract(
    module: types.ModuleType | None = None,
) -> WeightUpdaterContract:
    """Resolve the one SGLang component that owns the live weight lifecycle.

    SGLang may rename the scheduler component without changing the process
    snapshot capability ColdSnap consumes.  Resolve the owner structurally and
    fail closed if the surface is missing or ambiguous.
    """
    if module is None:
        from sglang.srt.managers.scheduler_components import weight_updater as module

    required = (
        "release_memory_occupation",
        "resume_memory_occupation",
        "update_weights_from_disk",
    )
    candidates_by_identity = {
        id(value): value
        for value in vars(module).values()
        if inspect.isclass(value)
        and value.__module__ == module.__name__
        and all(callable(getattr(value, name, None)) for name in required)
    }
    candidates = list(candidates_by_identity.values())
    if len(candidates) != 1:
        names = sorted(candidate.__qualname__ for candidate in candidates)
        raise SGLangCompatibilityError(
            f"expected exactly one SGLang weight lifecycle owner; found {names or 'none'}"
        )
    owner = candidates[0]
    return WeightUpdaterContract(
        owner=owner,
        hook_prefix=f"{owner.__module__}.{owner.__qualname__}",
    )


def validate_contracts() -> None:
    from sglang.srt.constants import (
        GPU_MEMORY_TYPE_CUDA_GRAPH,
        GPU_MEMORY_TYPE_KV_CACHE,
        GPU_MEMORY_TYPE_WEIGHTS,
    )
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_loader.loader import DefaultModelLoader, DummyModelLoader
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        set_graph_pool_id,
    )
    from sglang.srt.model_executor.runner_utils.pool import (
        set_global_graph_memory_pool,
    )
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

    weight_updater = resolve_weight_updater_contract()

    if (GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH) != (
        "weights",
        "kv_cache",
        "cuda_graph",
    ):
        raise SGLangCompatibilityError("SGLang memory-region tags changed")
    if not hasattr(HookRegistry, "register") or not hasattr(HookType, "AROUND"):
        raise SGLangCompatibilityError("SGLang hook registry contract is unavailable")
    if not callable(set_graph_pool_id) or not callable(set_global_graph_memory_pool):
        raise SGLangCompatibilityError("SGLang CUDA graph pool reset is unavailable")
    contracts = (
        Contract(
            TorchMemorySaverAdapter.create,
            "adapter.create",
            frozenset({"enable"}),
        ),
        Contract(
            TorchMemorySaverAdapter.region,
            "adapter.region",
            frozenset({"tag", "enable_cpu_backup"}),
        ),
        Contract(
            TorchMemorySaverAdapter.cuda_graph,
            "adapter.cuda_graph",
            frozenset({"kwargs"}),
        ),
        Contract(
            TorchMemorySaverAdapter.pause,
            "adapter.pause",
            frozenset({"tag"}),
        ),
        Contract(
            TorchMemorySaverAdapter.resume,
            "adapter.resume",
            frozenset({"tag"}),
        ),
        Contract(
            DefaultModelLoader.load_model,
            "loader.load_model",
            frozenset({"model_config", "device_config"}),
        ),
        Contract(
            DummyModelLoader.load_model,
            "dummy_loader.load_model",
            frozenset({"model_config", "device_config"}),
        ),
        Contract(
            ModelRunner.init_cuda_graphs,
            "model_runner.init_cuda_graphs",
            frozenset({"capture_decode_cuda_graph"}),
        ),
        Contract(
            ModelRunner.init_decode_cuda_graph,
            "model_runner.init_decode_cuda_graph",
            frozenset(),
        ),
        Contract(
            ModelRunner.init_prefill_cuda_graph,
            "model_runner.init_prefill_cuda_graph",
            frozenset({"force_for_draft_worker"}),
        ),
        Contract(
            Scheduler.process_batch_result,
            "scheduler.process_batch_result",
            frozenset({"batch", "result"}),
        ),
        Contract(
            Scheduler.on_idle,
            "scheduler.on_idle",
            frozenset(),
        ),
        Contract(
            weight_updater.owner.release_memory_occupation,
            "weight_updater.release_memory_occupation",
            frozenset({"recv_req"}),
        ),
        Contract(
            TokenizerManager.resume_memory_occupation,
            "tokenizer_manager.resume_memory_occupation",
            frozenset({"obj"}),
        ),
        Contract(
            weight_updater.owner.resume_memory_occupation,
            "weight_updater.resume_memory_occupation",
            frozenset({"recv_req"}),
        ),
        Contract(
            BaseSpecWorker.update_weights_from_disk,
            "draft_worker.update_weights_from_disk",
            frozenset({"recv_req"}),
        ),
    )
    for contract in contracts:
        _require(contract)


def install_failure_sentinel(error: BaseException) -> None:
    """Make explicit activation fatal despite SGLang logging plugin errors."""
    from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

    message = f"ColdSnap SGLang compatibility validation failed: {error}"

    def fail_create(enable: bool):
        del enable
        raise SGLangCompatibilityError(message)

    TorchMemorySaverAdapter.create = staticmethod(fail_create)

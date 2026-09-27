# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Attach the shared external-resource lifecycle to vLLM worker checkpoints."""

from __future__ import annotations

import functools
from typing import Any

from coldsnap_core.checkpoint import CheckpointResourceError, REGISTRY
from coldsnap_import_hook import after_module_import


def install_reader_adapters() -> bool:
    from coldsnap_b12x_checkpoint import install_disk_row_cache_adapter

    return after_module_import(
        "b12x.sequence._shared.disk_table", "coldsnap-checkpoint-resources",
        install_disk_row_cache_adapter,
    )


def _install_worker_hooks(module: Any) -> bool:
    worker_class = next((
        cls for name in ("GPUWorker", "Worker")
        if (cls := getattr(module, name, None)) is not None
        and all(callable(getattr(cls, method, None)) for method in ("checkpoint_prepare", "checkpoint_restore"))
    ), None)
    if worker_class is None:
        raise CheckpointResourceError("vLLM worker has no checkpoint prepare/restore contract")
    if getattr(worker_class.checkpoint_prepare, "__coldsnap_checkpoint_resources__", False):
        return False
    original_prepare = worker_class.checkpoint_prepare
    original_restore = worker_class.checkpoint_restore

    @functools.wraps(original_prepare)
    def prepare(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # Requests have been paused and vLLM/NCCL have quiesced GPU work.
        result = original_prepare(self, *args, **kwargs)
        resources = REGISTRY.prepare()
        return {**(result if isinstance(result, dict) else {}), "checkpoint_resources": resources}

    @functools.wraps(original_restore)
    def restore(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # Called after CUDA restoration, before either native or recovery wake.
        result = original_restore(self, *args, **kwargs)
        resources = REGISTRY.restore()
        return {**(result if isinstance(result, dict) else {}), "checkpoint_resources": resources}

    prepare.__coldsnap_checkpoint_resources__ = True
    worker_class.checkpoint_prepare = prepare
    worker_class.checkpoint_restore = restore
    return True


def install_checkpoint_resource_hooks() -> bool:
    # Register after the NCCL bridge so its evidence is retained in our result.
    return after_module_import(
        "vllm.v1.worker.gpu_worker", "coldsnap-checkpoint-resources", _install_worker_hooks,
    )

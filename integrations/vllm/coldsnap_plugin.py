# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""ColdSnap vLLM plugin entrypoint for capture, restore, and live sleep."""

from __future__ import annotations

from coldsnap_disk_backend import DiskCuMemBackend, _env_bool
from coldsnap_vllm import (
    force_cross_rank_rpc_over_tcp,
    override_default_sleep_backend,
    register_sleep_backend,
)


_REGISTERED = False


def _force_cross_rank_rpc_over_tcp() -> None:
    """Avoid a pre-load POSIX SHM handle that long model loads can outlive."""
    if not _env_bool("COLDSNAP_FORCE_RPC_TCP", False):
        return

    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    force_cross_rank_rpc_over_tcp(MessageQueue)


def register() -> None:
    """Register the named backend and install the enabled restore hooks."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    from coldsnap_vllm_process_template import (
        install_process_template_hook,
        preload_context_free_fla_platform_probe,
    )

    # This must precede imports that can resolve a Qwen model class. It is a
    # no-op unless the explicit pre-worker-import process-template phase is on.
    preload_context_free_fla_platform_probe()
    from vllm.device_allocator.sleep_mode_backend import (
        SleepModeBackend,
        SleepModeBackendFactory,
    )
    from coldsnap_startup_plan import install_startup_plan_memory_fallback
    from coldsnap_vllm_deferred_api_warmup import install_deferred_api_warmup_hook
    from coldsnap_vllm_async_graphs import install_async_graph_capture_hooks
    from coldsnap_vllm_deferred_warmup import install_deferred_warmup_hooks
    from coldsnap_vllm_cuda_runtime import install_cuda_epoch_runtime_hooks
    from coldsnap_vllm_graphs import install_graph_capture_hooks
    from coldsnap_vllm_kv_capacity import install_kv_capacity_guard
    from coldsnap_vllm_kv_payload import install_kv_payload_tracking_hooks
    from coldsnap_vllm_nccl_checkpoint import (
        install_instanttensor_nccl_unwrap,
        install_nccl_checkpoint_hooks,
    )
    install_graph_capture_hooks()
    install_cuda_epoch_runtime_hooks()
    install_kv_payload_tracking_hooks()
    install_deferred_api_warmup_hook()
    # Install deferred warmup before deferred graph capture so the nested
    # engine-step hooks always finish warmup before capturing graphs.
    install_deferred_warmup_hooks()
    install_kv_capacity_guard()
    install_async_graph_capture_hooks()
    install_startup_plan_memory_fallback()
    install_process_template_hook()
    install_nccl_checkpoint_hooks()
    install_instanttensor_nccl_unwrap()
    _force_cross_rank_rpc_over_tcp()

    from coldsnap_recovery_loader import (
        install_model_payload_capture_hook,
        install_recovery_aware_loader,
    )
    from coldsnap_synthetic_loader import install_synthetic_weight_loader

    install_model_payload_capture_hook()
    install_recovery_aware_loader()
    install_synthetic_weight_loader()

    backend_class = globals().get("RegisteredDiskWeightPoolBackend")
    if backend_class is None:
        backend_class = type(
            "RegisteredDiskWeightPoolBackend",
            (DiskCuMemBackend, SleepModeBackend),
            {"__module__": __name__},
        )
        globals()["RegisteredDiskWeightPoolBackend"] = backend_class
    register_sleep_backend(SleepModeBackendFactory, backend_class)
    if _env_bool("COLDSNAP_VLLM_OVERRIDE_CUMEM", False):
        override_default_sleep_backend(SleepModeBackendFactory, backend_class)

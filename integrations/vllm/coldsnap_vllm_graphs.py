# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Scoped CUDA-graph allocation hooks for vLLM capture paths."""

from __future__ import annotations

import functools
import importlib
import sys
from contextlib import AbstractContextManager, ExitStack
from typing import Any

from coldsnap_import_hook import after_module_import
from coldsnap_graph_memory import (
    DisabledCudaGraphController,
    NativeCudaGraphController,
    graph_controller_from_env,
)
from coldsnap_vllm import VllmContractError


GraphController = NativeCudaGraphController | DisabledCudaGraphController

_controller: GraphController = DisabledCudaGraphController()
_installed = False
_BREAKABLE_MODULE = "vllm.compilation.breakable_cudagraph"


def get_graph_controller() -> GraphController:
    return _controller


def _pool_id(value: Any) -> int:
    return 0 if value is None else id(value)


def _install_torch_graph_hook(controller: GraphController) -> None:
    import torch

    original = torch.cuda.graph
    if getattr(original, "_coldsnap_graph_region_hook", False):
        return

    @functools.wraps(original)
    def graph_with_region(*args: Any, **kwargs: Any) -> AbstractContextManager[Any]:
        pool = kwargs.get("pool")
        if pool is None and len(args) >= 2:
            pool = args[1]

        class ScopedGraph:
            def __enter__(self) -> Any:
                self._stack = ExitStack()
                try:
                    value = self._stack.enter_context(original(*args, **kwargs))
                    self._stack.enter_context(
                        controller.capture_region(pool_id=_pool_id(pool))
                    )
                    return value
                except BaseException:
                    self._stack.__exit__(*sys.exc_info())
                    raise

            def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
                return self._stack.__exit__(exc_type, exc, traceback)

        return ScopedGraph()

    graph_with_region._coldsnap_graph_region_hook = True  # type: ignore[attr-defined]
    graph_with_region._coldsnap_original = original  # type: ignore[attr-defined]
    torch.cuda.graph = graph_with_region


def _install_breakable_graph_hook(
    controller: GraphController, module: Any | None = None
) -> None:
    module = module or importlib.import_module(_BREAKABLE_MODULE)
    capture_class = getattr(module, "BreakableCUDAGraphCapture", None)
    if capture_class is None:
        raise VllmContractError(
            "vLLM breakable_cudagraph module lacks BreakableCUDAGraphCapture"
        )
    if getattr(capture_class, "_coldsnap_graph_region_hook", False):
        return

    original_begin = getattr(capture_class, "_begin_segment", None)
    original_end = getattr(capture_class, "_end_segment", None)
    if not callable(original_begin) or not callable(original_end):
        raise VllmContractError(
            "vLLM BreakableCUDAGraphCapture lacks segment lifecycle methods"
        )

    @functools.wraps(original_begin)
    def begin_with_region(instance: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(instance, "_coldsnap_region_context", None) is not None:
            raise VllmContractError(
                "nested vLLM breakable CUDA graph region is unsupported"
            )
        result = original_begin(instance, *args, **kwargs)
        region = controller.capture_region(
            pool_id=_pool_id(getattr(instance, "pool", None))
        )
        try:
            region.__enter__()
        except BaseException:
            try:
                original_end(instance)
            finally:
                raise
        instance._coldsnap_region_context = region
        return result

    @functools.wraps(original_end)
    def end_with_region(instance: Any, *args: Any, **kwargs: Any) -> Any:
        region = getattr(instance, "_coldsnap_region_context", None)
        if region is None:
            return original_end(instance, *args, **kwargs)
        instance._coldsnap_region_context = None
        try:
            region.__exit__(*sys.exc_info())
        except BaseException:
            try:
                original_end(instance, *args, **kwargs)
            finally:
                raise
        return original_end(instance, *args, **kwargs)

    capture_class._begin_segment = begin_with_region
    capture_class._end_segment = end_with_region
    capture_class._coldsnap_graph_region_hook = True


def install_graph_capture_hooks(
    controller: GraphController | None = None,
) -> GraphController:
    """Install stable capture boundaries before any engine graph capture."""
    global _controller, _installed
    if controller is None:
        controller = graph_controller_from_env()
    if not controller.enabled:
        _controller = controller
        return controller
    if _installed:
        if controller is not _controller:
            raise VllmContractError(
                "CUDA graph capture hooks are already bound to another controller"
            )
        return controller

    _install_torch_graph_hook(controller)
    after_module_import(
        _BREAKABLE_MODULE,
        "native-graph-regions",
        lambda module: _install_breakable_graph_hook(controller, module),
    )
    _controller = controller
    _installed = True
    return controller

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Post-import hooks that keep vLLM plugin registration CPU-only.

vLLM loads general plugins in API, engine, and worker processes. Importing a
GPU-worker module while registering the plugin can initialize CUDA in a parent
that should remain fork- or CRIU-safe. This module lets adapters patch a vLLM
module immediately after the process that owns it imports it naturally.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import threading
from collections.abc import Callable
from types import ModuleType
from typing import Any


PostImportCallback = Callable[[ModuleType], Any]


class PostImportHookError(RuntimeError):
    """Raised when a requested hook cannot be made ordering-safe."""


_lock = threading.RLock()
_callbacks: dict[str, dict[str, PostImportCallback]] = {}
_running: set[tuple[str, str]] = set()
_completed: set[tuple[str, str]] = set()


def _module_is_initializing(module: ModuleType) -> bool:
    spec = getattr(module, "__spec__", None)
    return bool(getattr(spec, "_initializing", False))


def _remove_finder_if_idle_locked() -> None:
    if _callbacks:
        return
    try:
        sys.meta_path.remove(_finder)
    except ValueError:
        pass


def _run_callback(
    module_name: str,
    key: str,
    callback: PostImportCallback,
    module: ModuleType,
) -> None:
    identity = (module_name, key)
    try:
        callback(module)
    except BaseException:
        with _lock:
            _running.discard(identity)
        raise
    with _lock:
        _running.discard(identity)
        _completed.add(identity)


def _run_pending(module_name: str, module: ModuleType) -> None:
    with _lock:
        pending = _callbacks.pop(module_name, {})
        for key in pending:
            _running.add((module_name, key))
        _remove_finder_if_idle_locked()
    for key, callback in pending.items():
        _run_callback(module_name, key, callback, module)


class _PostImportLoader(importlib.abc.Loader):
    def __init__(self, module_name: str, delegate: importlib.abc.Loader) -> None:
        self.module_name = module_name
        self.delegate = delegate

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create_module = getattr(self.delegate, "create_module", None)
        if callable(create_module):
            return create_module(spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        exec_module = getattr(self.delegate, "exec_module", None)
        if not callable(exec_module):
            raise PostImportHookError(
                f"loader for {self.module_name!r} cannot execute modules"
            )
        exec_module(module)
        _run_pending(self.module_name, module)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        with _lock:
            if fullname not in _callbacks:
                return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None:
            return None
        loader = spec.loader
        if loader is None:
            raise PostImportHookError(
                f"cannot attach a post-import hook to namespace module {fullname!r}"
            )
        if not isinstance(loader, _PostImportLoader):
            spec.loader = _PostImportLoader(fullname, loader)
        return spec


_finder = _PostImportFinder()


def after_module_import(
    module_name: str,
    key: str,
    callback: PostImportCallback,
) -> bool:
    """Run ``callback`` once after ``module_name`` is fully imported.

    Returns ``True`` when this call installed or ran a new callback, and
    ``False`` when the same module/key pair was already pending or complete.
    Callback failures propagate from the target import, making compatibility
    failures explicit instead of leaving a partially patched worker alive.
    """
    if not module_name or not key:
        raise ValueError("module_name and key must be non-empty")
    identity = (module_name, key)
    immediate: ModuleType | None = None
    with _lock:
        if (
            identity in _completed
            or identity in _running
            or key in _callbacks.get(module_name, {})
        ):
            return False
        module = sys.modules.get(module_name)
        if module is not None and not _module_is_initializing(module):
            immediate = module
            _running.add(identity)
        else:
            if module is not None:
                loader = getattr(getattr(module, "__spec__", None), "loader", None)
                if not isinstance(loader, _PostImportLoader):
                    raise PostImportHookError(
                        f"{module_name!r} is already initializing without a "
                        "coldsnap loader; post-import ordering cannot be guaranteed"
                    )
            _callbacks.setdefault(module_name, {})[key] = callback
            if _finder not in sys.meta_path:
                sys.meta_path.insert(0, _finder)
    if immediate is not None:
        _run_callback(module_name, key, callback, immediate)
    return True

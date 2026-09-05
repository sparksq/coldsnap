# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Opt-in early registration for context-free vLLM process templates.

The pinned distributed vLLM CLI does not load general plugins early enough to
interpose before model resolution and worker initialization.  Python imports
``sitecustomize`` from the explicitly mounted ColdSnap plugin path.  Install a
small import hook only for process-template launches, then register ColdSnap
after the CLI module or worker wrapper itself is loaded.  Helper interpreters
that never import either module remain untouched.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType
from typing import Any


_TRIGGERS = {
    "vllm.entrypoints.cli.main",
    "vllm.v1.worker.worker_base",
}


class _ColdSnapLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self._wrapped, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        from coldsnap_plugin import register

        register()


class _ColdSnapFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> Any:
        if fullname not in _TRIGGERS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _ColdSnapLoader(spec.loader)
        return spec


if (
    os.environ.get("COLDSNAP_MODEL_LOAD_RELEASE_FILE")
    and "coldsnap" in os.environ.get("VLLM_PLUGINS", "").split(",")
):
    sys.meta_path.insert(0, _ColdSnapFinder())

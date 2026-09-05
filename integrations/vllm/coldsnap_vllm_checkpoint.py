# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Version-bounded access to vLLM's process-checkpoint lifecycle hooks."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


class CheckpointHookError(RuntimeError):
    """Raised when a vLLM object does not expose a safe checkpoint boundary."""


@dataclass(frozen=True)
class CheckpointHookCall:
    operation: str
    dispatch: str
    response: Any


def _rpc(target: Any) -> tuple[Any, str] | None:
    collective_rpc = getattr(target, "collective_rpc", None)
    if callable(collective_rpc):
        return collective_rpc, "collective_rpc"

    engine = getattr(target, "llm_engine", None)
    collective_rpc = getattr(engine, "collective_rpc", None)
    if callable(collective_rpc):
        return collective_rpc, "llm_engine.collective_rpc"
    return None


def call_checkpoint_hook(target: Any, operation: str) -> CheckpointHookCall:
    """Call a synchronous checkpoint hook collectively on every worker.

    Current ``AsyncLLM`` exposes native async methods, while synchronous
    ``LLM`` exposes the worker hook through ``collective_rpc``.  Keep that
    version-sensitive choice here rather than in the process-checkpoint
    controller.
    """

    if operation not in {"checkpoint_prepare", "checkpoint_restore"}:
        raise ValueError(f"unsupported checkpoint operation: {operation}")

    direct = getattr(target, operation, None)
    if callable(direct) and not inspect.iscoroutinefunction(direct):
        response = direct()
        if inspect.isawaitable(response):
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise CheckpointHookError(
                f"{operation} returned an awaitable in a synchronous context"
            )
        return CheckpointHookCall(operation, "direct", response)

    rpc = _rpc(target)
    if rpc is None:
        raise CheckpointHookError(
            "vLLM object exposes neither a synchronous checkpoint hook nor "
            "collective_rpc"
        )
    collective_rpc, dispatch = rpc
    return CheckpointHookCall(operation, dispatch, collective_rpc(operation))


async def call_checkpoint_hook_async(
    target: Any, operation: str
) -> CheckpointHookCall:
    """Call the async engine hook, with collective RPC as a fallback."""

    if operation not in {"checkpoint_prepare", "checkpoint_restore"}:
        raise ValueError(f"unsupported checkpoint operation: {operation}")

    direct = getattr(target, operation, None)
    if callable(direct):
        response = direct()
        if inspect.isawaitable(response):
            response = await response
        return CheckpointHookCall(operation, "direct", response)

    rpc = _rpc(target)
    if rpc is None:
        raise CheckpointHookError(
            "vLLM object exposes neither an async checkpoint hook nor "
            "collective_rpc"
        )
    collective_rpc, dispatch = rpc
    response = collective_rpc(operation)
    if inspect.isawaitable(response):
        response = await response
    return CheckpointHookCall(operation, dispatch, response)


def checkpoint_prepare(target: Any) -> CheckpointHookCall:
    return call_checkpoint_hook(target, "checkpoint_prepare")


def checkpoint_restore(target: Any) -> CheckpointHookCall:
    return call_checkpoint_hook(target, "checkpoint_restore")

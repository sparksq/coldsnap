# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Engine-neutral execution-graph helpers for accelerator worker processes."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any


EXECUTION_GRAPH_ENV = "COLDSNAP_EXECUTION_GRAPH"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _is_world_group(group: Any) -> bool:
    return (
        isinstance(group, dict)
        and isinstance(group.get("kind"), str)
        and group["kind"].endswith(":world")
        and isinstance(group.get("ranks"), dict)
    )


def execution_graph(
    environment: Mapping[str, str] | None = None,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    values = os.environ if environment is None else environment
    raw = values.get(EXECUTION_GRAPH_ENV)
    if not raw:
        if required:
            raise RuntimeError(f"{EXECUTION_GRAPH_ENV} is required")
        return None
    try:
        graph = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{EXECUTION_GRAPH_ENV} is invalid JSON") from error
    if (
        not isinstance(graph, dict)
        or not isinstance(graph.get("unit"), str)
        or _ID.fullmatch(graph["unit"]) is None
        or not isinstance(graph.get("by_process_slot"), dict)
        or not graph["by_process_slot"]
        or not isinstance(graph.get("groups"), dict)
    ):
        raise RuntimeError(f"{EXECUTION_GRAPH_ENV} has an invalid structure")
    workers = graph["by_process_slot"].values()
    if not all(isinstance(worker, str) and _ID.fullmatch(worker) for worker in workers):
        raise RuntimeError(f"{EXECUTION_GRAPH_ENV} contains an invalid worker")
    return graph


def worker_id(
    graph: dict[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> str:
    values = os.environ if environment is None else environment
    graph = execution_graph(values) if graph is None else graph
    assert graph is not None
    by_slot = graph["by_process_slot"]
    local_rank = values.get("LOCAL_RANK")
    if local_rank is not None and local_rank in by_slot:
        return str(by_slot[local_rank])
    if len(by_slot) == 1:
        return str(next(iter(by_slot.values())))

    global_rank = values.get("RANK")
    matches = {
        group.get("ranks", {}).get(global_rank)
        for group in graph["groups"].values()
        if global_rank is not None
        and _is_world_group(group)
        and group.get("ranks", {}).get(global_rank) is not None
    }
    if len(matches) != 1:
        raise RuntimeError("cannot resolve ColdSnap worker from local or group rank")
    result = matches.pop()
    if not isinstance(result, str) or _ID.fullmatch(result) is None:
        raise RuntimeError("ColdSnap worker identity is invalid")
    return result


def global_rank(
    graph: dict[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    *,
    required: bool = True,
) -> int | None:
    """Resolve one worker's global rank without assuming rank-per-unit.

    Inference engines do not consistently retain ``RANK`` in accelerator
    worker environments, especially after an n580 process-template exec.  The
    execution graph is the durable source of truth: resolve the logical worker
    first, then invert its engine-owned ``*:world`` rank namespace.  An
    explicit global rank remains valid but must agree with the graph.
    """
    values = os.environ if environment is None else environment
    explicit_rank: int | None = None
    for name in ("RANK", "COLDSNAP_RANK", "COLDSNAP_EXPECTED_RANK"):
        raw = values.get(name)
        if raw is None:
            continue
        try:
            rank = int(raw)
        except ValueError as error:
            raise RuntimeError(f"{name} is not a valid rank") from error
        if rank < 0:
            raise RuntimeError(f"{name} is not a valid rank")
        explicit_rank = rank
        break

    graph = execution_graph(values, required=False) if graph is None else graph
    topology_rank: int | None = None
    if graph is not None:
        worker = worker_id(graph, values)
        matches: set[int] = set()
        for group in graph["groups"].values():
            if not _is_world_group(group):
                continue
            for rank_text, member in group["ranks"].items():
                if member != worker:
                    continue
                try:
                    rank = int(rank_text)
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        "ColdSnap world group contains an invalid rank"
                    ) from error
                if rank < 0 or str(rank) != str(rank_text):
                    raise RuntimeError("ColdSnap world group contains an invalid rank")
                matches.add(rank)
        if len(matches) > 1:
            raise RuntimeError(
                "ColdSnap worker has conflicting global ranks in world groups"
            )
        topology_rank = next(iter(matches), None)

    if (
        explicit_rank is not None
        and topology_rank is not None
        and explicit_rank != topology_rank
    ):
        raise RuntimeError("explicit global rank disagrees with execution topology")
    if topology_rank is not None:
        return topology_rank
    if explicit_rank is not None:
        return explicit_rank

    if required:
        raise RuntimeError("cannot resolve ColdSnap global rank")
    return None


def world_group(
    rank: int,
    graph: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    graph = execution_graph() if graph is None else graph
    assert graph is not None
    matches: list[tuple[str, dict[str, Any]]] = []
    for group_id, group in graph["groups"].items():
        if (
            isinstance(group_id, str)
            and _is_world_group(group)
            and isinstance(group.get("size"), int)
            and str(rank) in group["ranks"]
        ):
            matches.append((group_id, group))
    if len(matches) != 1:
        raise RuntimeError("runtime rank does not identify exactly one world group")
    return matches[0]

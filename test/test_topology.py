# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations" / "core"))

from coldsnap_core.topology import (  # noqa: E402
    execution_graph,
    global_rank,
    world_group,
    worker_id,
)


class ExecutionTopologyTest(unittest.TestCase):
    @staticmethod
    def _graph() -> dict[str, object]:
        return {
            "unit": "unit-b",
            "by_process_slot": {"0": "worker-2", "1": "worker-3"},
            "groups": {
                "world": {
                    "kind": "torch:world",
                    "size": 4,
                    "ranks": {"2": "worker-2", "3": "worker-3"},
                },
                "expert": {
                    "kind": "vllm:expert",
                    "size": 2,
                    "ranks": {"0": "worker-2", "1": "worker-3"},
                },
            },
        }

    def test_local_process_slot_selects_worker(self) -> None:
        environment = {
            "COLDSNAP_EXECUTION_GRAPH": json.dumps(self._graph()),
            "LOCAL_RANK": "1",
        }
        graph = execution_graph(environment)
        self.assertEqual(worker_id(graph, environment), "worker-3")
        self.assertEqual(global_rank(graph, environment), 3)

    def test_global_group_rank_is_fallback_when_local_rank_is_absent(self) -> None:
        environment = {
            "COLDSNAP_EXECUTION_GRAPH": json.dumps(self._graph()),
            "RANK": "2",
        }
        graph = execution_graph(environment)
        self.assertEqual(worker_id(graph, environment), "worker-2")
        group_id, group = world_group(2, graph)
        self.assertEqual(group_id, "world")
        self.assertEqual(group["size"], 4)

    def test_engine_world_group_supplies_rank_without_rank_environment(self) -> None:
        graph_value = self._graph()
        graph_value["by_process_slot"] = {"0": "worker-2"}
        graph_value["groups"]["world"]["kind"] = "vllm:world"
        environment = {"COLDSNAP_EXECUTION_GRAPH": json.dumps(graph_value)}
        graph = execution_graph(environment)

        self.assertEqual(worker_id(graph, environment), "worker-2")
        self.assertEqual(global_rank(graph, environment), 2)
        group_id, _group = world_group(2, graph)
        self.assertEqual(group_id, "world")

    def test_explicit_global_rank_must_match_execution_graph(self) -> None:
        environment = {
            "COLDSNAP_EXECUTION_GRAPH": json.dumps(self._graph()),
            "LOCAL_RANK": "0",
            "RANK": "3",
        }
        graph = execution_graph(environment)

        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            global_rank(graph, environment)

    def test_local_rank_does_not_substitute_for_global_topology(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot resolve"):
            global_rank(None, {"LOCAL_RANK": "2"})
        self.assertIsNone(global_rank(None, {}, required=False))

    def test_external_service_units_can_reuse_distributed_rank(self) -> None:
        prefill = self._graph()
        decode = self._graph()
        prefill["unit"] = "prefill-unit"
        prefill["by_process_slot"] = {"0": "prefill-worker"}
        decode["unit"] = "decode-unit"
        decode["by_process_slot"] = {"0": "decode-worker"}
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=True):
            self.assertEqual(worker_id(prefill), "prefill-worker")
            self.assertEqual(worker_id(decode), "decode-worker")


if __name__ == "__main__":
    unittest.main()

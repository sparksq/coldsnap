# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nccl-payload-release-plan.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("nccl_payload_release_plan", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NcclPayloadReleasePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.planner = _load_planner()

    def test_all_releases_expand_to_two_native_architectures(self) -> None:
        releases = self.planner._all_releases()
        plan = self.planner._make_plan(releases)
        self.assertTrue(plan["has_changes"])
        self.assertEqual(len(plan["release_matrix"]["include"]), 2)
        self.assertEqual(len(plan["build_matrix"]["include"]), 4)
        self.assertEqual(
            {item["platform"] for item in plan["build_matrix"]["include"]},
            {"linux/amd64", "linux/arm64"},
        )
        tags = {
            item["release"]: item["provider_tag"]
            for item in plan["release_matrix"]["include"]
        }
        self.assertEqual(tags["2.31.2-1"], "2.31.2-1.coldsnap.12")
        self.assertEqual(tags["2.30.7-1"], "2.30.7-1.coldsnap.1")

    def test_capability_admission_change_does_not_rebuild_payload(self) -> None:
        releases = self.planner._all_releases()
        selected = self.planner._select_changed(
            releases,
            ["native/nccl/releases/2.31.2-1/qualification.json"],
        )
        self.assertEqual(selected, set())

    def test_release_build_input_selects_only_that_release(self) -> None:
        releases = self.planner._all_releases()
        selected = self.planner._select_changed(
            releases,
            ["native/nccl/releases/2.31.2-1/coldsnap_provider.cc"],
        )
        self.assertEqual(selected, {"2.31.2-1"})

    def test_shared_build_input_selects_every_release(self) -> None:
        releases = self.planner._all_releases()
        selected = self.planner._select_changed(
            releases,
            ["native/nccl_checkpoint_coord/kv_store_client.cc"],
        )
        self.assertEqual(selected, set(releases))

    def test_publish_workflow_is_manual_repository_owned_and_immutable(self) -> None:
        workflow = (ROOT / ".github/workflows/publish-nccl.yml").read_text()
        self.assertIn("outside the generic scitrera-repo-tools", workflow)
        self.assertIn("DOCKERHUB_TOKEN", workflow)
        self.assertNotIn("packages: write", workflow)
        self.assertIn("already exists; bump provider_revision", workflow)
        self.assertIn("push-by-digest=true", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn('tags: ["v*.*.*"]', workflow)

        dockerfile = (ROOT / "deploy/nccl/Dockerfile.payload").read_text()
        self.assertIn('io.sparksq.coldsnap.cuda.gencode="${NCCL_NVCC_GENCODE}"', dockerfile)
        self.assertIn("COPY deploy/nccl/README.md /README.md", dockerfile)
        self.assertIn("COPY native/nccl /source/coldsnap/native/nccl", dockerfile)
        self.assertIn(
            "COPY native/nccl_checkpoint_coord /source/coldsnap/native/nccl_checkpoint_coord",
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()

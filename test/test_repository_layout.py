# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import importlib.util
import unittest
from pathlib import Path, PurePosixPath


SPEC = importlib.util.spec_from_file_location(
    "repository_layout", Path(__file__).resolve().parents[1] / "scripts/check-repository-layout.py"
)
assert SPEC is not None and SPEC.loader is not None
layout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(layout)


class RepositoryLayoutTest(unittest.TestCase):
    def test_maintained_harnesses_and_provider_admission_are_allowed(self):
        self.assertEqual(layout.violations([
            PurePosixPath("benchmarks/README.md"),
            PurePosixPath("benchmarks/harnesses/README.md"),
            PurePosixPath("benchmarks/harnesses/measure.py"),
            PurePosixPath("benchmarks/harnesses/run.sh"),
            PurePosixPath("native/nccl/releases/2.31.2-1/qualification.json"),
        ]), [])

    def test_forced_add_cannot_reintroduce_archives_or_results(self):
        for name in (
            "benchmarks/results/run.json",
            "benchmarks/2026-09-04-run/output.log",
            "benchmarks/archive/tools/old.py",
            "benchmarks/harnesses/output.json",
            "benchmarks/harnesses/runs/old.py",
            "docs/archive/plan.md",
            "native/nccl/experiments/old/provider.cc",
            "internal/dockerapi/client.go",
            "deploy/vllm/nccl/Dockerfile.in-place-experiment",
            "deploy/vllm/nccl/Dockerfile.exact-target",
        ):
            with self.subTest(path=name):
                self.assertTrue(layout.violations([PurePosixPath(name)]))


if __name__ == "__main__":
    unittest.main()

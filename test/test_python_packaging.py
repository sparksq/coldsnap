# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "integrations" / "core"
SGLANG_ROOT = ROOT / "integrations" / "sglang"
VLLM_ROOT = ROOT / "integrations" / "vllm"

sys.path.insert(0, str(CORE_ROOT))
sys.path.insert(0, str(SGLANG_ROOT))
sys.path.insert(0, str(VLLM_ROOT))

import coldsnap_sglang  # noqa: E402
from render_runtime_metadata import load_project, render  # noqa: E402


def _pyproject(path: Path) -> dict:
    with path.open("rb") as source:
        return tomllib.load(source)


class PythonPackagingTest(unittest.TestCase):
    def test_root_is_a_non_publishable_workspace(self) -> None:
        metadata = _pyproject(ROOT / "pyproject.toml")
        self.assertFalse(metadata["tool"]["uv"]["package"])
        self.assertEqual(metadata["tool"]["setuptools"]["packages"], [])
        self.assertEqual(
            set(metadata["tool"]["uv"]["workspace"]["members"]),
            {
                "integrations/core",
                "integrations/sglang",
                "integrations/vllm",
            },
        )

    def test_controller_release_archives_include_legal_files(self) -> None:
        workflow = (ROOT / ".github/workflows/publish-go.yml").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            workflow.count('cp "$GITHUB_WORKSPACE/LICENSE" "$stage/"'), 4
        )
        self.assertEqual(
            workflow.count(
                'cp "$GITHUB_WORKSPACE/THIRD_PARTY_NOTICES.md" "$stage/"'
            ),
            4,
        )

    def test_release_packages_own_their_metadata(self) -> None:
        workspace = _pyproject(ROOT / "pyproject.toml")["project"]
        core = _pyproject(CORE_ROOT / "pyproject.toml")["project"]
        sglang = _pyproject(SGLANG_ROOT / "pyproject.toml")["project"]
        vllm = _pyproject(VLLM_ROOT / "pyproject.toml")["project"]
        self.assertEqual(workspace["version"], core["version"])
        self.assertEqual(sglang["version"], coldsnap_sglang.__version__)
        self.assertEqual(sglang["dependencies"], [f"coldsnap-core=={core['version']}"])
        for project in (workspace, core, sglang, vllm):
            self.assertEqual(project["license"], "AGPL-3.0-only")
        canonical_license = (ROOT / "LICENSE").read_bytes()
        for package_root, project in (
            (CORE_ROOT, core),
            (SGLANG_ROOT, sglang),
            (VLLM_ROOT, vllm),
        ):
            self.assertEqual(project["license-files"], ["LICENSE"])
            self.assertEqual((package_root / "LICENSE").read_bytes(), canonical_license)

    def test_vllm_manifest_covers_every_flat_runtime_module(self) -> None:
        metadata = _pyproject(VLLM_ROOT / "pyproject.toml")
        declared = set(metadata["tool"]["setuptools"]["py-modules"])
        present = {path.stem for path in VLLM_ROOT.glob("coldsnap_*.py")} | {"sitecustomize"}
        self.assertEqual(declared, present)

    def test_vllm_discovery_metadata_is_generated_from_project(self) -> None:
        metadata = _pyproject(VLLM_ROOT / "pyproject.toml")["project"]
        self.assertEqual(list(VLLM_ROOT.glob("*.dist-info")), [])
        project = load_project(VLLM_ROOT / "pyproject.toml")
        generated = render(project)
        prefix = f"coldsnap_vllm-{metadata['version']}.dist-info"
        installed = generated[f"{prefix}/METADATA"].decode()
        entry_points = generated[f"{prefix}/entry_points.txt"].decode()
        self.assertIn(f"Name: {metadata['name']}\n", installed)
        self.assertIn(f"Version: {metadata['version']}\n", installed)
        self.assertIn("License-Expression: AGPL-3.0-only\n", installed)
        self.assertEqual(
            entry_points,
            "[vllm.general_plugins]\n"
            f"coldsnap = {metadata['entry-points']['vllm.general_plugins']['coldsnap']}\n",
        )

    def test_sglang_discovery_metadata_is_generated_from_project(self) -> None:
        metadata = _pyproject(SGLANG_ROOT / "pyproject.toml")["project"]
        project = load_project(SGLANG_ROOT / "pyproject.toml")
        generated = render(project)
        prefix = f"coldsnap_sglang-{metadata['version']}.dist-info"
        self.assertEqual(
            generated[f"{prefix}/entry_points.txt"].decode(),
            "[sglang.srt.plugins]\n"
            f"coldsnap = {metadata['entry-points']['sglang.srt.plugins']['coldsnap']}\n",
        )

    def test_sglang_runtime_image_contains_plugin_but_omits_activation_controllers(self) -> None:
        dockerfile = (ROOT / "deploy" / "sglang" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        for instruction in (
            "COPY integrations/sglang/coldsnap_sglang /out/plugin/coldsnap_sglang",
            'io.sparksq.coldsnap.runtime="sglang-cuda-criu-v1"',
            "SGLANG_PLUGINS=coldsnap",
        ):
            self.assertIn(instruction, dockerfile)
        for controller in (
            "coldsnap_activation_logs.py",
            "coldsnap_service_runtime.py",
            "coldsnap_engine_rank_n610.py",
            "coldsnap_engine_rank_n580.py",
            "coldsnap-engine-rank-n610",
            "coldsnap-engine-rank-n580",
            "COPY --from=coldsnap_builder /out/coldsnap-sglang-adapter",
        ):
            self.assertNotIn(controller, dockerfile)

    def test_runtime_images_keep_coldsnap_and_third_party_licenses_separate(self) -> None:
        for engine in ("vllm", "sglang"):
            dockerfile = (ROOT / "deploy" / engine / "Dockerfile").read_text(
                encoding="utf-8"
            )
            for instruction in (
                'io.sparksq.coldsnap.license="AGPL-3.0-only"',
                "LICENSE THIRD_PARTY_NOTICES.md /opt/coldsnap/licenses/coldsnap/",
                "third_party/licenses/criu/COPYING "
                "/opt/coldsnap/licenses/third-party/criu/COPYING",
                "third_party/licenses/nccl/LICENSE.txt "
                "/opt/coldsnap/licenses/third-party/nccl/LICENSE.txt",
                "--from=go_criu_source LICENSE "
                "/opt/coldsnap/licenses/third-party/go-criu/LICENSE",
                "--from=cuda_checkpoint_source LICENSE "
                "/opt/coldsnap/licenses/third-party/cuda-checkpoint/LICENSE",
            ):
                self.assertIn(instruction, dockerfile)
            self.assertNotIn(
                'org.opencontainers.image.licenses="AGPL-3.0-only"', dockerfile
            )

    def test_vllm_image_preserves_artifact_bound_core_target(self) -> None:
        dockerfile = (ROOT / "deploy" / "vllm" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "COPY integrations/core/coldsnap_core /out/plugin/coldsnap_core",
            dockerfile,
        )
        self.assertIn(
            "COPY --from=python_plugin_builder /out/plugin /opt/coldsnap/plugin",
            dockerfile,
        )
        self.assertIn("render_runtime_metadata.py", dockerfile)

    def test_runtime_sources_have_explicit_shared_and_engine_ownership(self) -> None:
        runtime = ROOT / "runtime"
        self.assertFalse((runtime / "python").exists())
        self.assertEqual(
            {path.name for path in (runtime / "shared").glob("*.py")},
            {
                "checkpointctl.py",
                "coldsnap_coord.py",
            },
        )
        self.assertEqual(
            {path.name for path in (runtime / "engine").glob("*.py")},
            {
                "coldsnap_activation_logs.py",
                "coldsnap_service_runtime.py",
                "coldsnap_engine_exec.py",
                "coldsnap_lifecycle.py",
                "coldsnap_engine_rank_n610.py",
                "coldsnap_n580_criu.py",
                "coldsnap_engine_rank_n580.py",
                "coldsnap_cuda_criu.py",
                "vllm_cuda_snapshot_target.py",
            },
        )

    def test_vllm_image_preserves_capture_runtime_but_omits_activation_controllers(self) -> None:
        dockerfile = (ROOT / "deploy" / "vllm" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        expected = {
            "COPY runtime/shared/coldsnap_coord.py "
            "/opt/coldsnap/runtime/coldsnap_coord.py",
            "COPY runtime/engine/coldsnap_cuda_criu.py "
            "/opt/coldsnap/runtime/coldsnap_cuda_criu.py",
            "COPY runtime/engine/coldsnap_engine_exec.py "
            "/opt/coldsnap/runtime/coldsnap-engine-exec.py",
        }
        for instruction in expected:
            self.assertIn(instruction, dockerfile)
        for controller in (
            "coldsnap_activation_logs.py",
            "coldsnap_service_runtime.py",
            "coldsnap_engine_rank_n610.py",
            "coldsnap_engine_rank_n580.py",
            "coldsnap-engine-rank-n610",
            "coldsnap-engine-rank-n580",
            "COPY --from=coldsnap_builder /out/coldsnap-vllm-adapter",
        ):
            self.assertNotIn(controller, dockerfile)

        evaluation = (
            ROOT / "deploy" / "vllm" / "Dockerfile.criu-rpc"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "COPY runtime/engine/vllm_cuda_snapshot_target.py "
            "/opt/coldsnap/runtime/vllm_cuda_snapshot_target.py",
            evaluation,
        )

    def test_n580_uses_neutral_service_runtime(self) -> None:
        controller = (
            ROOT / "runtime" / "engine" / "coldsnap_engine_rank_n580.py"
        ).read_text(encoding="utf-8")
        self.assertIn("import coldsnap_service_runtime as service_runtime", controller)
        self.assertNotIn("import coldsnap_engine_rank_n610", controller)

    def test_release_coordinates_are_separate(self) -> None:
        versions = (ROOT / "versions.yaml").read_text(encoding="utf-8")
        self.assertIn("integrations/core/pyproject.toml", versions)
        self.assertIn("integrations/sglang/pyproject.toml", versions)
        self.assertIn("integrations/vllm/pyproject.toml", versions)
        for coordinate, package_root in (("coldsnap-vllm", VLLM_ROOT), ("coldsnap-sglang", SGLANG_ROOT)):
            package_version = _pyproject(package_root / "pyproject.toml")["project"]["version"]
            self.assertIn(f"{coordinate}: {package_version}\n", versions)


if __name__ == "__main__":
    unittest.main()

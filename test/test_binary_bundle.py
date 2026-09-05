# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_binary_bundle", ROOT / "scripts/verify-binary-bundle.py"
)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class BinaryBundleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.version = "0.3.20"
        self.commit = "a" * 40
        self.manifest = {
            "format": 1,
            "kind": "coldsnap-controller-binary-bundle",
            "version": self.version,
            "commit": self.commit,
            "platform": "linux-amd64",
            "sha256": {},
        }
        self.set_arch("amd64")
        patcher = mock.patch.object(bundle.subprocess, "run")
        self.run_command = patcher.start()
        self.addCleanup(patcher.stop)
        self.run_command.return_value = SimpleNamespace(
            stdout=json.dumps({"version": self.version, "commit": self.commit})
        )

    def set_arch(self, arch):
        self.manifest["platform"] = f"linux-{arch}"
        for name in bundle.BINARIES:
            data = bytearray(64)
            data[:6] = b"\x7fELF\x02\x01"
            data[18:20] = bundle.ELF_MACHINES[arch].to_bytes(2, "little")
            (self.root / name).write_bytes(data)
            self.manifest["sha256"][name] = hashlib.sha256(data).hexdigest()
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def verify(self, arch="amd64"):
        bundle.verify(self.root, version=self.version, commit=self.commit, arch=arch)

    def test_both_native_platforms_verify_hashes_and_execute_the_controller(self):
        for arch in ("amd64", "arm64"):
            with self.subTest(arch=arch):
                self.set_arch(arch)
                self.verify(arch)
        self.run_command.assert_called_with(
            [str(self.root / "coldsnap"), "version", "--json"],
            check=True, capture_output=True, text=True, timeout=30,
        )

    def test_wrong_release_identity_is_rejected_before_execution(self):
        for key, value in (
            ("format", 2), ("kind", "other"), ("version", "0.3.19"),
            ("commit", "b" * 40), ("platform", "linux-arm64"),
        ):
            with self.subTest(key=key):
                original = self.manifest[key]
                self.manifest[key] = value
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "identity"):
                    self.verify()
                self.manifest[key] = original
        self.run_command.assert_not_called()

    def test_non_object_manifest_is_rejected(self):
        (self.root / "manifest.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "identity"):
            self.verify()

    def test_missing_or_extra_checksum_is_rejected(self):
        hashes = self.manifest["sha256"].copy()
        for inventory in ({}, {**hashes, "unexpected": "a" * 64}):
            with self.subTest(inventory=inventory):
                self.manifest["sha256"] = inventory
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "exactly four"):
                    self.verify()

    def test_non_hex_checksum_is_rejected(self):
        self.manifest["sha256"]["coldsnap"] = "z" * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "invalid checksum"):
            self.verify()

    def test_corrupted_executable_is_rejected(self):
        (self.root / "coldsnap-vllm-adapter").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.verify()
        self.run_command.assert_not_called()

    def test_correct_hash_cannot_hide_the_wrong_elf_architecture(self):
        self.set_arch("arm64")
        self.manifest["platform"] = "linux-amd64"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "ELF"):
            self.verify()

    def test_symlinked_manifest_or_executable_is_rejected(self):
        for name in ("manifest.json", "coldsnap"):
            with self.subTest(name=name):
                path = self.root / name
                target = path.with_suffix(".real")
                path.rename(target)
                path.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "regular file"):
                    self.verify()
                path.unlink()
                target.rename(path)

    def test_controller_identity_must_match_even_when_all_hashes_match(self):
        self.run_command.return_value.stdout = json.dumps({"version": self.version, "commit": "b" * 40})
        with self.assertRaisesRegex(ValueError, "controller executable"):
            self.verify()

    def test_full_commit_is_required(self):
        self.commit = "short"
        with self.assertRaisesRegex(ValueError, "full release commit"):
            self.verify()


class BinaryBundleWorkflowTest(unittest.TestCase):
    def test_oidc_replaces_secret_gates_and_is_limited_to_publishing_jobs(self):
        workflow = (ROOT / ".github/workflows/build-docker.yml").read_text()
        self.assertNotIn("DOCKERHUB_TOKEN", workflow)
        self.assertNotIn("DOCKERHUB_USERNAME", workflow)
        self.assertNotIn("publish_dockerhub", workflow)
        self.assertNotIn("packages: write", workflow)
        self.assertEqual(workflow.count("id-token: write"), 2)
        self.assertEqual(workflow.count("DOCKERHUB_OIDC_CONNECTIONID:"), 2)
        self.assertEqual(workflow.count("14f6b5b4-89d1-444f-911b-98df94b7ac9d"), 2)
        for job in ("build", "publish"):
            block = re.split(r"\n  [a-z][a-z-]*:\n", workflow.split(f"\n  {job}:\n", 1)[1])[0]
            self.assertIn("id-token: write", block)

    def test_backfill_builds_the_tag_commit_and_publishes_only_verified_digests(self):
        workflow = (ROOT / ".github/workflows/build-docker.yml").read_text()
        self.assertIn("release_tag:", workflow)
        self.assertIn("ref: refs/tags/${{ inputs.release_tag || github.ref_name }}", workflow)
        self.assertIn("COMMIT=${{ needs.release-version.outputs.commit }}", workflow)
        self.assertNotIn("github.sha", workflow)
        self.assertIn("context: release-source", workflow)
        self.assertIn("python3 scripts/verify-binary-bundle.py", workflow)
        self.assertIn("needs: [release-version, build]", workflow)
        self.assertIn("already exists; refusing to replace", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertNotIn("eval ", workflow)
        self.assertNotIn("value=latest", workflow)


if __name__ == "__main__":
    unittest.main()

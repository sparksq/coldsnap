# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NcclProviderSourceContractTest(unittest.TestCase):
    def test_runtime_image_has_no_flat_nccl_filename_selection(self) -> None:
        dockerfile = (ROOT / "deploy/vllm/Dockerfile").read_text()
        self.assertIn("coldsnap nccl-provider install-active", dockerfile)
        self.assertIn("COPY --from=nccl_provider", dockerfile)
        self.assertNotIn("COPY --from=nccl_runtime", dockerfile)
        self.assertIsNone(re.search(r"find\s+/opt/coldsnap/nccl[^\n]*libnccl", dockerfile))
        self.assertNotIn("ln -sfn", dockerfile)
        self.assertIn("VLLM_IMAGE must be digest-pinned", dockerfile)

    def test_provider_builder_observes_target_and_assembles_qualified_bytes(self) -> None:
        provider = (ROOT / "deploy/nccl/Dockerfile.provider").read_text()
        payload = (ROOT / "deploy/nccl/Dockerfile.payload").read_text()
        self.assertIn("scripts/nccl-target-observe.py", provider)
        self.assertIn("coldsnap nccl-provider assemble", provider)
        self.assertIn("coldsnap nccl-provider verify", provider)
        self.assertIn("NCCL_RELEASE_TAG", provider)
        self.assertIn("NCCL_PAYLOAD_IMAGE must be digest-pinned", provider)
        self.assertIn("FROM ${NCCL_PAYLOAD_IMAGE} AS provider_payload", provider)
        self.assertIn("FROM ${TARGET_IMAGE} AS provider_assembly", provider)
        self.assertIn("COLDSNAP_ALLOW_LOCAL_NCCL_PAYLOAD", provider)
        self.assertIn("COLDSNAP_NCCL_POLICY", provider)
        self.assertIn("NCCL_BUILD_IMAGE must be digest-pinned", payload)
        self.assertIn("FROM ${NCCL_BUILD_IMAGE} AS provider_payload_build", payload)
        self.assertIn("NVCC=/usr/local/bin/coldsnap-nccl-nvcc", payload)
        self.assertIn("patch --fuzz=0", payload)
        self.assertIn("COPY --from=nccl_release", payload)
        self.assertIn("linux/amd64", (ROOT / ".github/workflows/publish-nccl.yml").read_text())
        self.assertIn("linux/arm64", (ROOT / ".github/workflows/publish-nccl.yml").read_text())
        self.assertIn("COPY LICENSE THIRD_PARTY_NOTICES.md ./", provider)
        self.assertIn(
            "COPY third_party/licenses/nccl/LICENSE.txt "
            "./third_party/licenses/nccl/LICENSE.txt",
            provider,
        )
        self.assertIn("FROM scratch", provider)
        self.assertIn("FROM scratch AS provider_payload_oci", payload)

        observer = (ROOT / "scripts/nccl-target-observe.py").read_text()
        self.assertIn('default="exact"', observer)
        self.assertIn('"nccl_policy": args.nccl_policy', observer)

        dockerignore = (ROOT / ".dockerignore").read_text()
        self.assertIn("!scripts/nccl-target-observe.py", dockerignore)
        self.assertIn("!scripts/nccl-nvcc-reproducible.sh", dockerignore)
        self.assertNotIn("benchmarks/results/", provider)
        self.assertNotIn("benchmarks/results/", payload)
        self.assertNotIn("benchmarks/results/", dockerignore)
        self.assertNotIn("--evidence", provider)

    def test_in_place_graph_provider_is_owned_by_qualified_release(self) -> None:
        release = ROOT / "native/nccl/releases/2.31.2-1"
        source = (release / "coldsnap_in_place_provider.cc").read_text()
        patch_source = (release / "patches/0004-build-in-place-provider.patch").read_text()
        transport_patch = (
            release / "patches/0005-exact-nccl-net-reattach.patch"
        ).read_text()
        control_patch = (
            release / "patches/0006-exact-nccl-control-reattach.patch"
        ).read_text()
        payload_dockerfile = (ROOT / "deploy/nccl/Dockerfile.payload").read_text()
        provider_dockerfile = (ROOT / "deploy/nccl/Dockerfile.provider").read_text()
        patch_series = (release / "patches/series").read_text().splitlines()
        production_source = (
            ROOT / "native/nccl/releases/2.31.2-1/coldsnap_provider.cc"
        ).read_text()

        self.assertIn('COLDSNAP_NCCL_IN_PLACE_MODE', source)
        self.assertIn('net-reconnect-v1', source)
        self.assertIn('COLDSNAP_NCCL_IN_PLACE_ACTIVATION_PATH', source)
        self.assertIn('NCCL_IB_DISABLE', source)
        self.assertIn('NCCL_IB_RELEASE_ON_FINALIZE', source)
        self.assertIn('"portable\\\":false', source)
        self.assertIn('"transport_epoch_portable\\\":true', source)
        self.assertIn('"network_identity_portable\\\":true', source)
        self.assertIn('"bootstrap_available\\\":false', source)
        self.assertIn('"proxy_control_reconstructed\\\":true', source)
        self.assertIn('"qualified\\\":false', source)
        self.assertIn("real->localPersistentRefs", source)
        self.assertIn("ncclCheckpointNetDetach", source)
        self.assertIn("ncclCheckpointNetResetGlobals", source)
        self.assertIn("ncclCheckpointNetReinitialize", source)
        self.assertIn("ncclCheckpointNetReattach", source)
        self.assertIn("ncclCheckpointControlDetach", source)
        self.assertIn("ncclCheckpointProxyReattach", source)
        self.assertIn("closeSend", transport_patch)
        self.assertIn("closeRecv", transport_patch)
        self.assertIn("ncclCheckpointIbReset", transport_patch)
        self.assertIn("regMrDmaBuf", transport_patch)
        self.assertIn("ncclCheckpointNetRecvEndpoints", transport_patch)
        self.assertIn("ncclCheckpointNetSendEndpoints", transport_patch)
        self.assertIn("ncclCheckpointNetReattach", transport_patch)
        self.assertIn("ncclProxyCheckpointDetach", control_patch)
        self.assertIn("ncclProxyCheckpointResume", control_patch)
        self.assertIn("ncclCheckpointControlDetach", control_patch)
        self.assertIn("ncclCheckpointProxyReattach", control_patch)
        self.assertIn("ncclParamBootstrapNetEnable", control_patch)
        self.assertIn("coldsnapNcclInPlaceQuery", patch_source)
        self.assertIn("coldsnap_in_place_provider.cc", patch_source)
        self.assertIn("0005-exact-nccl-net-reattach.patch", patch_series)
        self.assertIn("0006-exact-nccl-control-reattach.patch", patch_series)
        self.assertIn("COPY --from=nccl_release", payload_dockerfile)
        self.assertIn("*.cc", payload_dockerfile)
        self.assertIn("nccl-provider assemble", provider_dockerfile)
        self.assertNotIn(
            'extern "C" int32_t coldsnapNcclInPlaceQuery', production_source
        )

    def test_adapter_passes_only_active_provider_record(self) -> None:
        adapter = (ROOT / "internal/inferenceadapter/adapter.go").read_text()
        self.assertIn('"--nccl-active-runtime", ncclActiveRuntimePath', adapter)
        self.assertNotIn('"--nccl-runtime-dir"', adapter)
        self.assertNotIn("/opt/coldsnap/nccl/libnccl-checkpoint-shim.so", adapter)

    def test_sglang_unifies_bundled_nccl_with_the_active_provider(self) -> None:
        dockerfile = (ROOT / "deploy/sglang/Dockerfile").read_text()
        self.assertIn("*/nvidia/nccl/lib", dockerfile)
        self.assertIn('EP_SUPPRESS_NCCL_CHECK=0', dockerfile)
        self.assertIn("import deep_ep", dockerfile)
        self.assertNotIn("EP_SUPPRESS_NCCL_CHECK=1", dockerfile)
        self.assertIn('active["files"]["nccl-runtime"]["path"]', dockerfile)

    def test_release_recipes_bind_outputs_to_capability_admission(self) -> None:
        releases = {
            "2.31.2-1": ("nccl-2.31.2-1+coldsnap.12", 23102, True, True, 13),
            "2.30.7-1": ("nccl-2.30.7-1+coldsnap.1", 23007, True, True, 9),
        }
        for release, (provider_id, version, reproducible_nvcc, strip, capability_count) in releases.items():
            with self.subTest(release=release):
                root = ROOT / "native/nccl/releases" / release
                recipe = json.loads((root / "recipe.json").read_text())
                qualification = json.loads((root / "qualification.json").read_text())
                self.assertEqual(recipe["format"], 2)
                self.assertEqual(qualification["format"], 2)
                self.assertEqual(recipe["provider_id"], provider_id)
                self.assertEqual(recipe["nccl_version_code"], version)
                self.assertIs(recipe["build"]["use_reproducible_nvcc"], reproducible_nvcc)
                self.assertIs(recipe["build"]["strip_unneeded"], strip)
                self.assertEqual(recipe["checkpoint_abi"], 100)
                self.assertRegex(
                    recipe["builder"]["payload_build_image"],
                    r"^.+@sha256:[0-9a-f]{64}$",
                )
                self.assertEqual(
                    recipe["builder"]["base_image_policy"],
                    "digest-pinned-multiarch-payload-and-target-images",
                )
                self.assertEqual(
                    recipe["builder"]["payload_platforms"],
                    ["linux/amd64", "linux/arm64"],
                )
                self.assertEqual(
                    recipe["builder"]["payload_repository"],
                    "docker.io/scitrera/coldsnap-nccl",
                )
                self.assertEqual(
                    recipe["builder"]["payload_dockerfile"],
                    "deploy/nccl/Dockerfile.payload",
                )
                self.assertTrue(recipe["build"]["nvcc_gencode"])
                self.assertEqual(len(recipe["capabilities"]), capability_count)
                self.assertEqual(
                    {item["role"] for item in recipe["outputs"]},
                    {"nccl-runtime", "checkpoint-shim"},
                )
                for output in recipe["outputs"]:
                    self.assertEqual(set(output), {"path", "role", "soname"})
                for item in recipe["inputs"]["patches"]:
                    payload = ROOT.joinpath(item["path"]).read_bytes()
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
                self.assertEqual(qualification["state"], "accepted")
                self.assertEqual(qualification["policy"], "production")
                self.assertEqual(
                    qualification["capabilities"], recipe["capabilities"]
                )
                self.assertEqual(qualification["transports"], ["ib-roce", "socket"])
                self.assertTrue(qualification["checks"])
                self.assertNotIn("evidence", qualification)
                self.assertNotIn("valid_until", qualification)

    def test_nccl_nvcc_wrapper_derives_a_unique_stable_output_seed(self) -> None:
        wrapper = ROOT / "scripts/nccl-nvcc-reproducible.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = root / "arguments"
            compiler = root / "nvcc"
            compiler.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" >\"$COLDSNAP_NVCC_ARGUMENTS\"\n"
            )
            compiler.chmod(0o755)
            output = "/opt/nccl/build/obj/device/common.o"
            subprocess.run(
                [str(wrapper), "-dc", "common.cu", "-o", output],
                env={
                    "COLDSNAP_REAL_NVCC": str(compiler),
                    "COLDSNAP_NVCC_ARGUMENTS": str(arguments),
                },
                check=True,
            )
            actual = arguments.read_text().splitlines()
        expected_seed = hashlib.sha256(output.encode()).hexdigest()
        self.assertEqual(actual[0], f"--frandom-seed={expected_seed}")
        self.assertEqual(actual[1:], ["-dc", "common.cu", "-o", output])

    def test_criu_host_network_unlock_is_owned_by_pinned_runtime_image(self) -> None:
        self.assertEqual(list((ROOT / "deploy/vllm/criu/patches").glob("*.patch")), [])
        dockerfile = (ROOT / "deploy/vllm/Dockerfile").read_text()
        self.assertNotIn("notify-host-network-unlock.patch", dockerfile)
        self.assertNotIn("patch -d /opt/coldsnap/criu-source", dockerfile)
        self.assertNotIn("--from=criu_source", dockerfile)
        self.assertNotIn("/opt/coldsnap/criu-source", dockerfile)
        self.assertNotIn("make -C /opt/coldsnap/criu-source", dockerfile)
        self.assertIn("COPY --from=criu_image /usr/sbin/criu", dockerfile)
        self.assertNotIn("criu-plugin.h", dockerfile)
        self.assertIn(
            "COPY --from=criu_image /usr/lib/criu/coldsnap_nvidia_reset_plugin.so",
            dockerfile,
        )
        self.assertIn("CRIU_IMAGE must be digest-pinned", dockerfile)


if __name__ == "__main__":
    unittest.main()

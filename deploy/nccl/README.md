<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# NCCL provider images

ColdSnap separates the reusable compiled NCCL payload from provider assembly:

- `Dockerfile.payload` builds the patched NCCL runtime and checkpoint shim from
  a locked upstream archive and a release-owned patch/source context.
- `Dockerfile.provider` combines one payload with an observed, digest-pinned
  vLLM or SGLang target image and emits the manifest-verified provider consumed
  by the engine image.

These two stages are shared by vLLM and SGLang: compilation produces a reusable
payload, while assembly binds it to the exact observed engine image. The
Sparkrun ColdSnap plugin uses these same Dockerfiles, then passes the assembled
provider to `deploy/vllm/Dockerfile` or `deploy/sglang/Dockerfile`. There is no
separate vLLM-specific NCCL build directory. The retired exact-target
qualification image depended on an older builder's private intermediate layout
and is archived outside the public source tree. Current locked releases,
capability admission records, and provider-selection tests remain under
`native/nccl/` and `test/`.

The repository-owned `publish-nccl.yml` workflow builds native `linux/amd64`
and `linux/arm64` payloads and publishes one OCI index. An operator explicitly
dispatches one named release (or `all`) after configuring `DOCKERHUB_USERNAME`
and `DOCKERHUB_TOKEN`. Release tags do not trigger this workflow. The planning
script also supports comparing source refs for release review. The immutable tag is derived from the provider
ID, for example:

```text
docker.io/scitrera/coldsnap-nccl:2.31.2-1.coldsnap.12
```

The `2.31.2-1` alias advances to the newest provider revision for that exact
upstream NCCL release. It refuses to overwrite an existing immutable revision
tag; change `provider_revision` when payload inputs change.

Both architectures use the digest-pinned CUDA toolchain and GPU code-generation
set in the release recipe. OCI labels record the provider ID and ColdSnap Git
revision. The image carries separate ColdSnap and NCCL license files.

## Provider assembly

Pull the versioned payload, resolve it to a digest, and supply it while observing
the target inference image:

```bash
docker pull docker.io/scitrera/coldsnap-nccl:2.31.2-1.coldsnap.12
payload_ref="$(docker image inspect \
  docker.io/scitrera/coldsnap-nccl:2.31.2-1.coldsnap.12 \
  --format '{{index .RepoDigests 0}}')"
docker build \
  --file deploy/nccl/Dockerfile.provider \
  --build-arg "TARGET_IMAGE=${TARGET_IMAGE}" \
  --build-arg "NCCL_PAYLOAD_IMAGE=${payload_ref}" \
  --build-arg NCCL_RELEASE_TAG=2.31.2-1 \
  --build-arg COLDSNAP_NCCL_POLICY=exact \
  --tag coldsnap-nccl-provider:local \
  .
```

`TARGET_IMAGE` must be digest-pinned. Docker selects the payload matching the
target platform from the OCI index.

## Local source-build fallback

A platform that lacks a published payload can build `Dockerfile.payload` with
the exact archive from `source.lock`, the selected release directory as the
`nccl_release` build context, and the recipe's build arguments. Tag that image
locally, then assemble with:

```bash
docker build \
  --file deploy/nccl/Dockerfile.provider \
  --build-arg "TARGET_IMAGE=${TARGET_IMAGE}" \
  --build-arg NCCL_PAYLOAD_IMAGE=coldsnap-nccl-payload:local \
  --build-arg COLDSNAP_ALLOW_LOCAL_NCCL_PAYLOAD=1 \
  --build-arg NCCL_RELEASE_TAG=2.31.2-1 \
  --tag coldsnap-nccl-provider:local \
  .
```

Local output hashes and ELF build IDs are measured into the assembled manifest
and enforced thereafter. They are integrity metadata, not historical
qualification gates.

The scratch payload image carries `/README.md`, ColdSnap's license and
third-party notices, NVIDIA NCCL's separate BSD-3-Clause license, and the
ColdSnap source/build inputs needed to reconstruct the patched payload under
`/source/coldsnap`. OCI labels identify the provider, source revision, target
platform, exact CUDA `NVCC_GENCODE` set, and aggregate license expression.
Managers must verify that `io.sparksq.coldsnap.cuda.gencode` contains the
target GPU's `sm_*` code before assembly. A missing or incompatible label is a
payload miss and should use the locked target-architecture local build.

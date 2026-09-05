<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# ColdSnap vLLM runtime image

This image is the capture base for `coldsnap-vllm-adapter`. Build it from the
exact digest-pinned vLLM image used by the recipe. It installs the ColdSnap vLLM
plugin, native direct-I/O hydrator, native coordinator, CRIU/CUDA process
checkpoint runtime, and qualified NCCL checkpoint libraries.
The manager-side engine adapter and its embedded rank-controller activation
pack are not installed in the final image. Capture and restore stage the pack
separately and mount it read-only, so newly derived capsules do not duplicate
patchable controller code.

The build requires these external contexts:

- `criu_image`: the digest-pinned `ghcr.io/sparksq/criu` runtime, including the
  host-network unlock notification used to coordinate distributed TCP repair
  and the separately built n580 reset plugin;
- `go_criu_source`: the matching Go CRIU RPC fork;
- `cuda_checkpoint_source`: NVIDIA's `cuda-checkpoint` source package; and
- `nccl_provider`: one verified provider directory produced by
  `deploy/nccl/Dockerfile.provider` for the base-image NCCL identity.

The image always builds `libcoldsnap-nccl-dlsym.so` from the bridge source in
this repository. This keeps its configuration ABI synchronized with the
controller and prevents a stale pre-rename bridge from silently bypassing NCCL
checkpoint interposition for private `dlopen` users.

The vLLM integration and shared core are separate source distributions. Image
assembly places both below `/opt/coldsnap/plugin`, preserving the historical
`coldsnap_core/...` runtime paths used by plugin and artifact digests.
The image build also renders vLLM discovery metadata from the integration's
`pyproject.toml`; no generated `.dist-info` tree is kept in source control.

Use the verified provider context when building locally:

```bash
make vllm-runtime-image \
  VLLM_BASE_IMAGE=<repository@sha256> \
  CUDA_CHECKPOINT_SOURCE_ROOT=/path/to/gpu-checkpoint-restore \
  NCCL_PROVIDER_CATALOG=/path/to/provider-catalog
```

The target rejects a mutable `VLLM_BASE_IMAGE`. Sparkrun's ColdSnap builder
observes that image, chooses the matching locked release recipe, builds the
capability-admitted provider, and supplies it as the `nccl_provider` context
automatically. `CRIU_IMAGE` also must be digest-pinned; the Makefile supplies a
fixed multi-architecture `ghcr.io/sparksq/criu` digest. The
runtime and GPL-2.0-only n580 reset plugin are copied from that image, so normal
ColdSnap image conversion does not fetch or compile CRIU or compile any CRIU
plugin from headers.
The result is a reusable ColdSnap-enabled base. Capture then derives one
immutable OCI capsule per launch unit from it. The capsule contains process and
non-weight state plus explicitly bounded derived/JIT cache seeds, while optional
full native weight packs remain separate and can be distributed through Hugging
Face. Capsule construction uses a Docker BuildKit named context for the cache
rootfs so cache files are placed at their captured absolute paths without also
duplicating them below `/opt/coldsnap/capsule`.

## Plugin-free CUDA/CRIU evaluation image

Build the evaluation image from the published CRIU runtime and the public,
revision-pinned go-criu fork. `make` materializes
`https://github.com/sparksq/go-criu.git` at the declared immutable revision
under `build/sources/go-criu`; the NVIDIA checkpoint source must be supplied:

```bash
make vllm-cuda-criu-image \
  CUDA_CHECKPOINT_SOURCE_ROOT=/path/to/gpu-checkpoint-restore
```

Override `CRIU_IMAGE`, `GO_CRIU_SOURCE_URL`, `GO_CRIU_SOURCE_REVISION`, or
`VLLM_CUDA_CRIU_IMAGE` when the runtime digest, source revision, or desired
output tag differs. Docker receives the immutable CRIU OCI image and pinned
go-criu checkout as named build contexts and installs no CRIU plugins. The
published CRIU package and default go-criu fork are public, so a clean build
does not require GitHub credentials for either.

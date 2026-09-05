<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# ColdSnap SGLang runtime image

This derived image is the capture and capsule base for the controller-managed
SGLang process-snapshot path. Build it from the exact digest-pinned SGLang image
used by the recipe. It installs the SGLang plugin, native hydration runtime,
CRIU/CUDA checkpoint tooling, coordinator, and a qualified NCCL provider.
The manager-side adapter and its embedded rank-controller activation pack stay
outside the image; capture and restore stage the pack and mount it read-only.

The build takes the same named contexts as the vLLM image: `criu_image`,
`go_criu_source`, `cuda_checkpoint_source`, and `nccl_provider`. The SGLang base
must already provide `torch_memory_saver`; image construction validates the
process-snapshot hooks and fails closed if its SGLang contract has changed.
The CRIU context supplies both the CRIU executable and the separately
GPL-2.0-only n580 reset plugin; the ColdSnap source build does not compile that
in-process plugin.

```bash
make sglang-runtime-image \
  SGLANG_BASE_IMAGE=<repository@sha256> \
  CUDA_CHECKPOINT_SOURCE_ROOT=/path/to/gpu-checkpoint-restore \
  NCCL_PROVIDER_CATALOG=/path/to/provider-catalog
```

See [NCCL provider assembly](../nccl/README.md) for the catalog inputs.

Capture and restore are driven by `coldsnap-sglang-adapter` (normally through
Sparkrun), and SGLang
uses its own in-place disk loader when the shared native model payload is not
available.

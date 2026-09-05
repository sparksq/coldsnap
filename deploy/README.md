<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Deployment assets

- [`binaries/`](binaries/README.md) builds the architecture-matched controller and adapter bundle
  used as a verified backup to GitHub release attachments.
- `vllm/` builds the ColdSnap-enabled vLLM runtime and CRIU RPC evaluation image.
- `sglang/` builds the ColdSnap-enabled SGLang runtime.
- `nccl/` builds the engine-neutral NCCL payload and verified provider.

These definitions consume source from `runtime/`, `integrations/`, and
`native/`. Benchmark harnesses and qualification results are intentionally
excluded from production image contexts.

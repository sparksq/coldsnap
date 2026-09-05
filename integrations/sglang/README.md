<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# ColdSnap for SGLang

`coldsnap-sglang` is the in-container SGLang integration used by ColdSnap's
controller-managed process snapshots. It is inert unless explicitly enabled by
the controller. When enabled, both `COLDSNAP_ARTIFACT_DIR` and
`COLDSNAP_PROCESS_ARTIFACT_ROOT` are required.

The plugin registers address-stable memory, model-payload, NCCL checkpoint, and
in-place recovery hooks. Host execution and credentials stay behind the
controller's manager-provider boundary.

The qualified Qwen TP2 path supports native and recovery weights on n580 and
n610. The n610 default retains qualified CUDA graph executables and restores
NCCL transport in place; explicit graph recreation remains available. See
[`docs/sglang-plugin.md`](../../docs/sglang-plugin.md) for the lifecycle and
qualification boundary.

The n580 path restores a context-free pre-exec template and initializes a fresh
engine with either pinned safetensors or a native semantic payload. Graph
recreation and speculative-worker qualification have separate boundaries.

<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# ColdSnap SGLang integration

`coldsnap-sglang` is an out-of-tree SGLang plugin for ColdSnap's portable
process-snapshot path. It uses SGLang's public plugin hooks, memory-saver
regions, and weight-update endpoints; it does not patch SGLang and does not
open SSH connections or own cluster credentials.

The `n610` driver captures the initialized SGLang process and CUDA state,
discards weight and KV payloads from the residual capsule, and later restores
the process at its stable virtual addresses. The `n580` driver uses a smaller
context-free pre-exec process template and reconstructs the engine around
either a pinned Hugging Face checkpoint or a native semantic model payload. KV
is recreated in both modes.

## Components

| Component | Responsibility |
| --- | --- |
| `coldsnap-sglang` Python plugin | Registers final model storages, integrates SGLang memory regions, and places NCCL prepare/restore at the release/resume boundary. |
| `coldsnap-sglang-adapter` | Validates requests and artifacts, prepares capsules and weight providers, and drives capture, restore, publish, and lifecycle operations. |
| Adapter-supplied `coldsnap-engine-rank-n580` / `coldsnap-engine-rank-n610` | Run the driver-qualified node-local process transition and engine-specific HTTP activation sequence from a read-only, content-addressed mount. |
| Sparkrun ColdSnap plugin | Detects hardware, prepares the derived image, distributes models and capsules, supplies host operations, and owns user-facing workflow. |

Both Go adapter commands reuse engine-neutral orchestration under
`internal/inferenceadapter`, with thin vLLM and SGLang entrypoints selecting
their engine policy and operation timeout.

## Admission and compatibility

Admission is based on the runtime contract and captured identity, not a list of
model architecture names. Any generation model can enter the path when the
installed SGLang build exposes the required loader, hook registry,
memory-saver, weight-updater, and speculative-worker contracts.

Artifacts bind the exact engine/plugin source contract, model and revision,
tensor layout, execution topology, CUDA ABI, GPU architecture/capacity class,
NCCL provider, and process snapshot driver. They do not bind hostname, IP,
GPU UUID, or an architecture allowlist. A different topology or engine build
requires a new capture.

The current qualified TP2 matrix uses Qwen3.8 27B FP8 across two GB10 nodes and
covers native and recovery weight providers on both snapshot drivers. Detailed
measurements and pinned inputs are maintained outside the public repository.

The n610 default, `preserve-nccl-exec`, retains qualified graph executables and
reconstructs NCCL transport in place. The n580 default and explicit
`recreate-from-plan` path reconstruct graphs. Target-only asynchronous decode
graph reconstruction starts with eager dispatch and captures after a completed
request at a scheduler-idle boundary. Speculative/draft graph reconstruction
retains synchronous initialization until separately qualified.

Capture-time shape calibration is also engine-native. SGLang enumerates and
warms its own decode and prefill graph plan before snapshotting, then publishes
coverage for the controller to bind into the artifact. ColdSnap does not reuse
vLLM's graph descriptors or architecture-specific assumptions. An unrecognized
plan or a planned/warmed mismatch fails capture.

## Capture and restore sequence

Capture starts SGLang normally with `--enable-memory-saver`. The plugin records
the final target and speculative model layouts. When the controller calls
`/release_memory_occupation`, SGLang first quiesces requests, flushes live
allocator caches, and pauses the requested `weights`, `kv_cache`, and
`cuda_graph` regions. Under `preserve-nccl-exec`, ColdSnap retains the graph
executables and prepares the provider's in-place NCCL lifecycle. Under
`recreate-from-plan`, it destroys target and draft graph executables and resets
their process-wide graph-pool bindings before preparing NCCL.
A TP CPU-group barrier prevents the HTTP-owning rank from acknowledging a
partially transitioned TP group. The rank controller assembles optional native
model payloads and captures the remaining process/CUDA state.

For `n610`, restore reconstructs the initialized process and CUDA context from
its capsule. The plugin restores NCCL before SGLang resumes its memory regions.
Native mode hydrates registered storages from staged packs during resume.
Recovery first resumes weight storage and hydrates it through SGLang's
`update_weights_from_disk` contract, then resumes KV-cache and CUDA-graph
regions in a second distributed phase. Tokenizer readiness remains closed until
that sequence completes, preventing health traffic from racing hydration.
Small parameters which SGLang explicitly permits checkpoints to omit are kept
in residual state and restored before post-load processing.

For `n580`, restore reconstructs only the context-free pre-exec template. In
recovery mode SGLang loads weights directly from the pinned snapshot during
normal construction. In native mode a no-checkpoint loader allocates the same
model structure and the plugin hydrates its registered persistent parameters
and buffers from the staged semantic payload. PyTorch non-persistent buffers
are intentionally excluded because SGLang derives them for the fresh runtime
shape. Both paths initialize new destination communicators and then serve.

For speculative decoding, ColdSnap registers both the target and draft models.
Recovery updates each draft runner from its own pinned model path instead of
reusing the target model path.

## Image construction

`deploy/sglang/Dockerfile` derives a ColdSnap-enabled image from a digest-pinned
SGLang image. It installs the SGLang plugin, controller binaries, CRIU/CUDA
tools, native hydration support, and one immutable qualified NCCL provider.

Sparkrun's `builder: coldsnap` selects `deploy/sglang/Dockerfile` when the
materialized runtime is SGLang. The derived image advertises
`io.sparksq.coldsnap.runtime=sglang-cuda-criu-v1` and support for `n580` and `n610`.

## Configuration

The runtime image sets `SGLANG_PLUGINS=coldsnap`. Controller-owned settings are
injected into the container and should not be put in recipes:

| Setting | Meaning |
| --- | --- |
| `COLDSNAP_MODE` | `off` or `capture`; a restored capsule continues the captured process. |
| `COLDSNAP_ARTIFACT_DIR` | Per-capsule semantic layout root. |
| `COLDSNAP_PROCESS_ARTIFACT_ROOT` | Process artifact and hydration root. |
| `COLDSNAP_LIVE_BACKING` | `discard`, `cpu`, or `disk`; normal capsule capture uses `discard`. |
| `COLDSNAP_RUNTIME_DIR` | Writable locks and transient runtime state. |
| `COLDSNAP_ARTIFACT_PREVERIFIED` | Trust controller verification of immutable staged payloads. |
| `COLDSNAP_EXPORT_MODEL_PAYLOAD` | Export native model payload components during capture. |
| `COLDSNAP_SGLANG_STARTUP_PROVIDER` | Controller-selected `native` or `recovery` startup path. |
| `COLDSNAP_SGLANG_RECOVERY_LOAD_FORMAT` | SGLang load format used for recovery hydration. |
| `COLDSNAP_ASYNC_CUDA_GRAPHS` | Enable target-only eager startup followed by scheduler-idle decode graph capture. |
| `COLDSNAP_SHAPE_CALIBRATION` | Controller-owned capture-time calibration of SGLang's complete engine graph plan. |

## Verification

Run the local contract and packaging checks with:

```bash
python3 -m unittest discover -s test -p 'test_sglang_plugin.py' -q
go test ./cmd/coldsnap-sglang-adapter ./internal/snapshot \
  ./internal/inferenceadapter
```

The target-only publish-ready TP2 recipe is
`coldsnap-recipes/qwen3.8-27b-fp8-coldsnap-tp2-sglang.yaml`; the separately qualified
speculative recipe is
`coldsnap-recipes/qwen3.8-27b-nvfp4-dspark-coldsnap-tp2-sglang.yaml` in the companion
[sparkrun-recipes](https://github.com/sparksq/sparkrun-recipes) repository,
exposed through the plugin's `@coldsnap` registry.

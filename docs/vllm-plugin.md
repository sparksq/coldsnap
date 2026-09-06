<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# ColdSnap vLLM integration

`coldsnap-vllm` integrates vLLM's model loading, memory allocation, CUDA graphs,
and worker lifecycle with ColdSnap capture and restore. This guide covers the
engine integration for runtime developers. Recipe and CLI configuration belong
in the [Sparkrun reference](sparkrun-usage.md).

The `n610` snapshot driver captures initialized process and CUDA state. The
`n580` driver uses a context-free pre-exec template and reconstructs the engine
on the destination. Both paths use the same artifact, model-payload, and
manager-provider contracts, but their residuals and replay metadata remain
driver-specific.

## Components

| Component | Responsibility |
| --- | --- |
| `coldsnap-vllm` Python integration | Registers the vLLM plugin and connects model loading, stable allocations, sleep/wake, graph policy, and NCCL checkpoint hooks. |
| `coldsnap-vllm-adapter` | Validates requests and artifacts, selects providers, and drives distributed capture, restore, publication, and lifecycle operations. |
| Adapter-supplied `coldsnap-engine-rank-n580` / `coldsnap-engine-rank-n610` | Sequence node-local process transitions and readiness from the adapter's read-only activation-runtime mount. |
| Shared `coldsnap_core` and native libraries | Provide tensor/layout contracts, memory regions, payload capture and hydration, and validation support. |
| Placement manager | Supplies complete placement and launch requests, stages payloads, and implements authorized host operations. |

The Go adapter shares engine-neutral orchestration in
`internal/inferenceadapter` with the SGLang adapter. Engine-specific storage and
loading behavior stays behind the runtime integration's extension points.

## Admission and compatibility

The runtime must provide the vLLM loader, allocator, worker, and graph contracts
expected by the installed integration. Artifacts bind the runtime identity,
model revision, tensor layout, execution topology, CUDA ABI, GPU class, NCCL
provider, and snapshot driver. A model name alone does not establish
compatibility. See [capability, graph, and memory contracts](capability-graph-memory-contracts.md).

The n610 default preserves qualified distributed CUDA graph executables while
reconstructing NCCL transport in place. The n580 path and explicit
`recreate-from-plan` policy reconstruct graphs. Asynchronous recreation can
start with eager dispatch and capture graphs after readiness at the supported
engine boundary. Capture-time calibration records the engine's supported graph
plan; it does not assume that all runtime shapes are interchangeable.

## Capture and restore sequence

Capture starts a real vLLM workload with sleep mode enabled. The adapter prefers
InstantTensor when installed, then FastSafetensors, then the configured
`auto`/`safetensors` fallback. ColdSnap observes checkpoint-to-destination writes
and the final stable allocation layout while vLLM retains responsibility for
sharding, packing, quantization finalizers, and model-specific processing.

After response validation, capture records recovery metadata and optionally
exports native model bytes. The checkpoint boundary releases reload-owned
weight payloads, discards configured KV contents, and applies the selected
CUDA-graph and NCCL policy. Non-model residual bytes and replay maps stay with
the driver-qualified capsule; large native weight packs stay separate.

For **n610**, restore reconstructs the initialized process and CUDA state,
restores distributed communication, and hydrates the captured allocation
layout. Native mode reads the model pack plus capsule residual. Recovery mode
uses vLLM's in-place loading path to reconstruct model bytes from the pinned
safetensors, with the same residual metadata.

For **n580**, restore resumes the pre-exec template, applies the destination
placement, and starts vLLM with the ColdSnap loader. Capsule metadata guides
reconstruction of the allocation layout before native or recovery hydration.
The destination creates fresh CUDA and communication state.

Both paths remap and validate discarded KV memory and must pass health and
the expected-response check before reporting readiness. The restore load
format is `coldsnap`, regardless of the normal loader used during capture.
See [native hydration and calibration](native-hydration.md) for the detailed
startup path.

## Native payloads and recovery caching

The vLLM writer separates model extents from non-model residuals and produces
content-addressed worker packs. Recovery remains available from the immutable
Hugging Face model revision when an optional native pack is absent.

`policy.weights.native.materialize` controls the recovery-time native cache:
`async` is the vLLM default, `required` waits for a verified pack before restore
success, and `off` disables that writer. The artifact must already declare the
expected native payload inventory. This writes model payloads; it does not
rebuild the capsule or make residuals portable across incompatible runtimes.
See [model payloads and residuals](model-payload-layout.md).

## Image construction

`deploy/vllm/Dockerfile` derives the runtime from a digest-pinned vLLM image. It
installs the vLLM integration, shared core, native hydration and memory
libraries, CRIU/CUDA checkpoint tools, coordinator, and a qualified NCCL
provider. The image exposes the plugin through vLLM's `vllm.general_plugins`
entry point with `VLLM_PLUGINS=coldsnap`.

The matching Go adapter supplies the small activation-runtime pack separately.
Its rank controllers are mounted read-only during capture and restore rather
than copied into each new capsule. Follow the
[vLLM runtime build guide](../deploy/vllm/README.md) for pinned build inputs and
the `make vllm-runtime-image` target.

## Configuration

The adapter projects request policy into runtime settings. Configure the
request's policy rather than duplicating adapter-owned values in launch
environment overrides.

| Setting | Role |
| --- | --- |
| `VLLM_PLUGINS=coldsnap` | Enables plugin discovery in the runtime image. |
| `--enable-sleep-mode` | Enables the vLLM lifecycle needed for capture and live sleep/wake. |
| `COLDSNAP_VLLM_OVERRIDE_CUMEM` | Selects the ColdSnap allocation backend on vLLM lines without a named backend selector. |
| `COLDSNAP_RECOVERY_WEIGHT_SOURCE=safetensors` | Enables model-source recovery with separate non-model residuals. |
| `COLDSNAP_EXPORT_MODEL_PAYLOAD` | Enables capture of the optional external native model pack. |
| `COLDSNAP_DISCARD_REGIONS` | Selects supported discardable memory regions, including KV cache. |
| `COLDSNAP_DISK_SLEEP_DIR` | Identifies the local artifact root used by the live weight lifecycle. |

The [configuration reference](configuration.md#vllm) documents I/O policies,
distributed hydration, validation, graph controls, and startup guards.

## Verification

From the ColdSnap repository, run the integration contract and adapter checks:

```bash
python3 -m unittest discover -s test -p 'test_vllm_contract.py' -q
go test ./cmd/coldsnap-vllm-adapter ./internal/snapshot \
  ./internal/inferenceadapter
```

Changes to loading, graph, memory, or checkpoint behavior also need the relevant
tests under `test/` and GPU qualification on the intended engine, model, driver,
and topology. Contract tests alone do not establish successful GPU restore or
startup performance. Use the [benchmark harnesses](../benchmarks/harnesses/README.md)
for matched hardware measurements.

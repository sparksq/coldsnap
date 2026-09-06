<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# ColdSnap

ColdSnap is a manager-driven snapshot and memory-lifecycle system for distributed
vLLM and SGLang inference. Capture pays model construction, compilation,
calibration, and CUDA-graph setup once; restore reconstructs a compatible serving
workload from driver-qualified OCI capsules while keeping the large model weights
separate.

The project is pre-release research software. Its compatibility claims are limited
to explicitly qualified engine, model, CUDA, driver, hardware, and topology
combinations.

## Architecture

ColdSnap is a set of layers rather than a cluster daemon:

```text
Sparkrun reference manager or another placement manager
  -> coldsnap Go controller
    -> coldsnap-vllm-adapter or coldsnap-sglang-adapter
      -> per-unit Python engine integration
        -> CRIU, CUDA checkpointing, NCCL provider, and native memory helpers
```

- The manager owns scheduling, hardware discovery, credentials, image and model
  staging, host transport, prepare-before-evict ordering, logs, and workload
  identity. Sparkrun is the reference implementation of this manager boundary and
  currently uses SSH from its control node to coordinate placed GPU hosts. SSH
  belongs to Sparkrun, not the ColdSnap controller or adapters; a Kubernetes
  operator can implement the same boundary with a different transport.
- The Go controller owns strict request, artifact, compatibility, and receipt
  contracts. Engine-specific Go adapters coordinate capture and restore across
  launch units and supply a small content-addressed activation runtime read-only
  to each rank container.
- Shared Python runtime code and the vLLM or SGLang plugin integrate with engine
  loading, memory, graph, validation, and lifecycle semantics.

`coldsnap capabilities` is the authoritative machine-readable description of
supported request, artifact, receipt, provider, coordinator, and snapshot-driver
formats. See [the architecture document](docs/architecture.md) for component and
protocol details.

## Artifacts and weight providers

Each capture separates portable model data from driver/runtime state:

| Asset | Scope | Purpose |
| --- | --- | --- |
| OCI capsule | Launch unit and snapshot driver | Process, CUDA/NCCL, residual, and derived-cache state |
| Activation runtime | Adapter release and driver ABI | Patchable rank controller, staged per host and mounted read-only |
| Native model payload | Accelerator worker | Optional content-addressed model bytes; reusable between driver variants when byte-identical |
| Pinned Hugging Face snapshot | Model revision | Required safetensors recovery source when no native payload is available |

The default `weights: auto` policy prefers a verified native payload and falls
back to the optimized recovery loader over the pinned safetensors. No native blob
registry is required. KV payloads are discarded and remapped rather than stored;
compiler and CUDA caches are capsule seed data.

ColdSnap currently has two explicit NVIDIA snapshot drivers:

| Driver | Minimum host driver | Capture boundary |
| --- | ---: | --- |
| `n580` | 580 | Pre-CUDA process template followed by fresh CUDA reconstruction |
| `n610` | 610 | Initialized-CUDA process state with CUDA checkpoint restoration |

Artifacts, capsules, residuals, and address maps remain snapshot-driver qualified.
Sparkrun probes every placed host and selects the newest driver supported by the
whole placement, so heterogeneous hosts use one common snapshot method.

## Quick start

The examples below require Sparkrun with the first-party
[ColdSnap plugin](https://github.com/sparksq/sparkrun-coldsnap-plugin). The plugin
is developed separately; a plain Sparkrun installation may not include this
preview. Follow its [preview setup](https://github.com/sparksq/sparkrun-coldsnap-plugin/blob/main/DEV_PREVIEW.md)
and verify `sparkrun coldsnap --version` and `sparkrun coldsnap --help` first.

Sparkrun is the current reference manager. It derives
ColdSnap requests from placement and a recipe's top-level `coldsnap:` block,
then coordinates the selected hosts over SSH from the control node:

```bash
sparkrun coldsnap materialize --cluster two-node \
  @coldsnap/qwen3.8-27b-fp8-coldsnap-tp2-vllm
sparkrun run --cluster two-node \
  @coldsnap/qwen3.8-27b-fp8-coldsnap-tp2-vllm
```

This example uses a published recipe from the plugin's `@coldsnap` registry.
Choose a recipe qualified for your hardware and placement. `materialize`
prepares published artifacts and may perform a verification restore; it is not
a startup benchmark. To create your own capsule instead, use
`sparkrun coldsnap capture recipe.yaml --cluster two-node` with a fully pinned
recipe as described in [Sparkrun reference](docs/sparkrun-usage.md).

`sparkrun run` automatically chooses the ColdSnap execution strategy, prepares and
verifies capsules and optional native payloads, and replaces the serving workload
only after preparation succeeds. Promotion and lifecycle operations are explicit:

```bash
sparkrun coldsnap publish recipe.yaml --cluster two-node
sparkrun coldsnap publish-native recipe.yaml --cluster two-node \
  --hf-repo example/model-native
sparkrun coldsnap restore recipe.yaml --cluster two-node
sparkrun coldsnap warm recipe.yaml --cluster two-node
sparkrun coldsnap sleep recipe.yaml --cluster two-node
sparkrun coldsnap wake recipe.yaml --cluster two-node
sparkrun coldsnap status recipe.yaml --cluster two-node
sparkrun coldsnap delete recipe.yaml --cluster two-node
```

For qualified vLLM workloads, `sleep` releases managed weights and graphs and
discards configured KV payloads while retaining the live process, CUDA context,
NCCL communicator, and a fixed memory floor. `warm` restores an n610 capsule up to
the hydration boundary; `wake` hydrates and validates it. These are node-local
lifecycle states, not portable snapshots after a container or host restart.

### Manager-runtime compatibility

ColdSnap 0.3.20 requires a manager implementing `runtime-v1`, supplied by the
Sparkrun ColdSnap plugin 0.1.1. Upgrade the controller and plugin together.
The [runtime-neutral manager interface](docs/runtime-neutral-managers.md)
keeps Docker execution in the manager and preserves existing artifact formats.
Docker is the reference backend; Kubernetes is not yet implemented or qualified.

### Developer quick start

Build the controller and engine adapters locally, add them to `PATH`, and inspect
the machine-readable interface:

```bash
make build
export PATH="$PWD/bin:$PATH"
coldsnap capabilities
```

The direct `coldsnap capture|publish-native|publish|restore|sleep|wake|status`
commands accept strict JSON requests. They are manager-facing primitives and
require a manager host-provider; ColdSnap does not discover or connect to remote
hosts itself. See [basic usage](docs/usage.md) and the
[operator integration contract](docs/operator-integration.md).

## Performance measurement

Use the [maintained benchmark harnesses](benchmarks/harnesses/README.md) for
matched vanilla, recovery, and native measurements on your qualified runtime.
The external TTFT observer measures Docker `State.StartedAt` through the first
non-empty streamed model token and validates the completed response. Artifact
preparation and verification before container start are separate phases.

Historical results and raw logs are maintained privately outside this source
repository. Performance depends on the exact engine, model, driver, topology,
storage, and cache state; rerun the matched cases for each release candidate.

## Current qualification boundary

- vLLM is the broadest-qualified engine. SGLang has its own Go adapter and Python
  integration; Qwen TP2 native and recovery restores are qualified on both drivers.
- Current end-to-end evidence covers ARM64 DGX Spark GB10, CUDA 13, two-host TP2,
  and NVIDIA 580/610 driver families. The topology format represents multi-GPU
  launch units and TP, PP, DP, EP, CP, and disaggregated groups, but broader
  topologies still require qualification.
- n610 preserves distributed CUDA graph executables while reconstructing fresh
  NCCL transport endpoints in place. n580 uses eager-first graph recreation
  after readiness.
- NCCL integration is provider- and runtime-qualified, not a generic NCCL ABI.
- Recovery performance depends on model replay and first-inference work.
  Every release candidate needs
  cache-cleared native and recovery restore checks because portability and
  activation-runtime changes can affect this boundary.

## Development

```bash
uv sync --all-packages
make check
make build
```

Generated binaries and native assets stay under `bin/` and `build/`. Runtime
images are built on the exact digest-pinned vLLM or SGLang base and include the
matching Python integration, native helpers, NCCL provider, and GPL-contained
CRIU runtime. Controller and adapter release binaries are distributed separately
through GitHub release attachments and a licensed multi-architecture Docker Hub
binary bundle.

Useful references:

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Basic usage](docs/usage.md)
- [Configuration](docs/configuration.md)
- [Build and distribution](docs/distribution.md)
- [Execution topology](docs/execution-topology.md)
- [SGLang integration](docs/sglang-plugin.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

Copyright 2026 Scitrera LLC and Fox Engine Ltd.

ColdSnap is licensed under the [GNU Affero General Public License, version 3
only](LICENSE). The NVIDIA reset plugin is maintained in the public CRIU fork
under GPL-2.0-only; CRIU and other third-party components retain their own
licenses and notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Contributions are governed by [CONTRIBUTING.md](CONTRIBUTING.md) and the
[ColdSnap Contributor License Agreement](CLA.md).

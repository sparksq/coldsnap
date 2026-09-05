<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# ColdSnap documentation

Start with [basic usage](usage.md), covering the direct controller interface
and the [Sparkrun manager path](usage.md#sparkrun-usage).

## Current operation

- [Supported configuration and defaults](configuration.md)
- [Recovery-aware operation and capsules](recovery-aware-operations.md)
- [Build and binary distribution](distribution.md)
- [SGLang integration](sglang-plugin.md)

## Current architecture and contracts

- [System architecture and implementation ownership](architecture.md)
- [Operator integration contract](operator-integration.md)
- [Manager host-provider protocol](host-provider-protocol.md)
- [Runtime-neutral manager contract](runtime-neutral-managers.md)
- [Execution topology and artifact ownership](execution-topology.md)
- [Capability, graph, and memory contracts](capability-graph-memory-contracts.md)
- [Shared model payloads and driver residuals](model-payload-layout.md)
- [Native hydration and startup calibration](native-hydration.md)
- [NCCL provider manifest format](nccl-provider-format.md)

## Development

- [Repository content and publication policy](repository-layout.md)
- [Maintained benchmark harnesses](../benchmarks/README.md)
- [Repository maintenance tools](../scripts/README.md)

Historical plans, experiment reports, and raw qualification runs are retained
outside the public repository. Current schemas and release-owned provider
admission records define compatibility.

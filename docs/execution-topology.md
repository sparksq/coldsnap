<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Execution topology and artifact ownership

ColdSnap request format 4 and artifact format 9 separate physical launch
boundaries from accelerator workers. This is the format contract for
multi-GPU hosts and for engine parallelism beyond tensor parallelism.

The central rule is:

> A launch unit is a container or process-tree boundary. A worker is an
> accelerator-owning engine process. They are not assumed to be one-to-one.

For example, TP4 across two hosts with two GPUs per host is two launch units
and four workers. Each vLLM unit starts one process tree with two visible
devices; vLLM creates two local workers inside that tree. ColdSnap creates one
capsule per unit, one content-addressed model payload per worker for native
replay, and one recovery replay plan per worker.

## Schema layers

`launch.units[]` records placement and process-tree activation:

- stable unit ID and ordered unit index;
- destination host and visible device selectors;
- digest-pinned image, command, environment, and mounts.

`launch.execution.workers[]` records portable engine process slots:

- stable worker ID;
- owning unit and service;
- process slot within the unit;
- device-slot indexes into the owning unit's `devices` list.

`launch.execution.groups[]` records ordered rank namespaces. Group kinds are
adapter-owned, namespaced strings such as `vllm:world`, `vllm:tensor`, or
`sglang:decode`. ColdSnap does not enumerate parallel modes and does not infer
group membership by multiplying TP, PP, DP, EP, or CP values.

Worker global rank is resolved from the worker's engine-owned `*:world` group.
Runtime `RANK` is accepted when an engine provides it, but it must agree with
the execution graph. `LOCAL_RANK` identifies only a process slot within a unit;
it is never substituted for portable global identity. Rank-qualified worker
paths therefore stay stable even when an engine does not retain distributed
rank environment variables after an exec boundary.

`launch.execution.services[]` separates independently addressable engines and
cooperating roles. Data-parallel replicas and prefill/decode services can be
represented as separate service domains, including multiple units on the same
host.

`launch.execution.adapter` is an opaque, engine-owned topology payload with a
canonical SHA-256 digest. The adapter can add new dimensions, coordinates, or
runtime semantics without a ColdSnap core format change. Restore requires the
same execution graph and adapter digest as capture.

## Ownership

Artifact objects use typed owners:

| Owner | Examples | Lifecycle |
| --- | --- | --- |
| `unit/<id>` | OCI capsule, CRIU/CUDA state, derived cache | One per process tree |
| `worker/<id>` | Native weight pack, recovery replay plan | One per accelerator worker |
| `service/<id>` | Future service-level state | One per engine/service domain |
| `group/<id>` | Future collective-group state | One per ordered rank namespace |

This prevents two costly mistakes: duplicating a unit capsule for every local
GPU, and downloading every worker's model payload to every host. The manager
resolves each worker through its owning unit and stages that pack only on the
unit's destination host.

## Representative layouts

These counts describe the format, not qualification claims. The exact group
membership and service decomposition remain engine-owned.

| Workload | Launch units | Workers | Typical service/group representation |
| --- | ---: | ---: | --- |
| TP2, two 1-GPU hosts | 2 | 2 | One service, ordered world/tensor group |
| TP4, two 2-GPU hosts | 2 | 4 | One service, two workers per unit |
| TP8, one 8-GPU host | 1 | 8 | One service, eight workers in one unit |
| TP4 + DP2 | Placement-dependent | 8 | Two DP service domains, explicit groups per replica |
| TP4 + PP2 | Placement-dependent | 8 | Explicit tensor and pipeline rank namespaces |
| TP4 + EP2 or CP2 | Placement-dependent | Engine-defined | Overlapping explicit groups; EP/CP are not automatically multiplied |
| SGLang prefill/decode | Placement-dependent | Engine-defined | Separate prefill and decode services plus transport groups |

Every materially different execution topology requires its own capture. That
is a compatibility requirement, not a format limitation: TP, PP, DP, EP, CP,
and disaggregated service layouts all fit the same unit/worker/group/service
schema.

## Placement portability

Unit IDs, worker IDs, group order, service roles, device-slot shape, and the
adapter topology digest are capture identity. Hostnames, IP addresses, GPU
UUIDs, local device ordinals, mount sources, and communication-interface
environment are destination placement and may change.

Compatibility still requires the captured process architecture, GPU model and
compute capability, capsule-pinned CUDA/NCCL userspace, and a destination
NVIDIA driver at least as new as the captured minimum by default. Setting
`policy.compatibility.enforce_captured_driver_floor=false` relaxes that captured
floor, but never the selected snapshot driver's own minimum. The captured kernel is
provenance; default admission runs the capsule-pinned CRIU capability probe,
while exact uname equality is an opt-in policy. Unit device counts and worker
ownership must continue to match even when physical hosts or device selectors
change.

## Current qualification boundary

The integrated qualification covers vLLM and SGLang Qwen TP2 plus vLLM
DeepSeek TP2 on two GB10 hosts with one GPU per host. The generalized schema, vLLM materializer, capsule
builder, model-payload staging, runtime worker identity, and CPU-only topology
matrix cover multi-worker units including TP4 across two 2-GPU hosts. A real
multi-GPU capture/restore remains a separate hardware qualification step.

SGLang uses the same worker-safe identity model and can represent its parallel
and prefill/decode roles without another core format change. Its end-to-end
CUDA snapshot qualification remains narrower than vLLM's.

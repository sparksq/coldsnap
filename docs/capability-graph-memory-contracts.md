<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Capability, graph, and memory contracts

Status: implemented contract groundwork; hardware promotion remains feature-specific.

ColdSnap admits a snapshot from evidence about the resources that snapshot
actually depends on. A snapshot-driver name is an implementation selection,
not proof that every CUDA, CRIU, or NCCL resource family works in the current
environment.

## Host feature admission

Artifact format 9 adds a typed `requires` inventory. A destination publishes a
format-1 `coldsnap-host-feature-profile` containing one result per bounded
feature ID. Each result retains one of four distinct states:

- `passed`: the exact probe completed successfully;
- `failed`: the probe ran and failed, with its failure phase and reason;
- `unsupported`: the resource family is a known implementation limit;
- `unqualified`: ColdSnap has no accepted evidence for the resource family.

The n580 base contract requires a CRIU process template, fresh CUDA state, and
private exact-address VMM mapping. The n610 base contract additionally requires
the CUDA checkpoint API and a complete allocation round trip, initialized CUDA
state, a CRIU process tree, and classic CUDA IPC.

The release-matched `checkpointctl.py profile` probe runs in a disposable,
network-isolated capsule container. It performs:

- a CUDA checkpoint state-transition and allocation-content round trip;
- a classic CUDA IPC process-pair canary;
- a private VMM reserve, map, unmap, exact re-reserve, and remap canary;
- explicit known-unsupported records for managed memory and exported VMM handles.

The capsule-pinned CRIU binary separately runs `criu check`; that result is
then attached to the corresponding structural feature. Selected NCCL provider
identity remains an independent exact-version check.

Only an admitted profile is cached. Its key includes the host boot ID, NVIDIA
driver and device identity, capsule digest, snapshot driver, device ordinal,
and release-matched probe digest. A change to any of those inputs forces a new
probe. Model-weight payload identity remains content-addressed and independent
of the host profile.

Artifact format 9 is the only accepted artifact format. Earlier pre-release
formats fail validation and must be recaptured rather than normalized into a
new feature or graph contract.

Deep, multi-node evidence is represented by a typed qualification record. A
record must bind an exact environment and workload scope and include at least
two restores from one unchanged artifact before it can supply additional
passed features.

## NCCL graph preservation boundary

NCCL supports normal CUDA Graph capture. ColdSnap has two qualified lifecycle
boundaries: the portable destroy/recreate path reconstructs graph executables
from the engine plan, while the n610 exact-version path keeps graph-visible
communicator resources and reconstructs their transport epoch in place.

The provider ABI exposes explicit capabilities for the in-place lifecycle:

- communicator suspension;
- transport detachment and reattachment;
- graph-resource retention;
- registration-window replay;
- NVLS replay;
- NCCL device-API state retention.

These capabilities live behind the separate optional
`coldsnapNcclInPlaceQuery` ABI. Providers that do not export that ABI neither
acquire its semantics by rebuilding against a newer header nor satisfy an
in-place request.

The minimal two-rank CUDA/CRIU harness captures an all-reduce in a CUDA Graph,
demands the optional in-place ABI, records provider evidence, and replays the
original executable after restore. It remains useful as a focused provider
test in addition to the real-engine qualification.

Qualification covered reconstruction of NCCL's built-in Socket and IB/RoCE NET
data transports, repeated replay of a captured CUDA graph, zero-TCP capture,
rank-swapped restore, and real vLLM/SGLang TP2 engines. Socket bootstrap is not
yet reconstructed, and shared or remote NET proxy ownership remains outside
the admitted envelope. Driver 580 rejected the initialized-CUDA checkpoint.
The accepted capability record makes `preserve-nccl-exec` the n610 default;
n580 continues to use `recreate-from-plan`.

## Explicit graph policies

`policy.process.graph_policy` accepts three requested policies when asynchronous
graphs are enabled:

- `preserve-exec` preserves a graph executable only when all dependencies are
  proven stable;
- `recreate-from-plan` is the n580 default and the safe fallback;
- `preserve-nccl-exec` is the n610 distributed default and requests the exact
  in-place NCCL lifecycle.

Every format-9 artifact records the requested and effective policies, a
resource audit, a decision code, and a versioned deterministic engine recipe.
`preserve-exec` currently downgrades to `recreate-from-plan`: distributed
captures record the reconstructed communicator as the reason, while local
captures record that graph-executable preservation has not yet been promoted
for either engine adapter. `preserve-nccl-exec` requires a matching
exact-version in-place provider. A process with no NCCL communicator uses the
recreation path instead.
Unknown dependencies always force recreation. Validation recomputes the recipe
digest from the execution graph, audit, policy, and calibration evidence and
rejects tampering before activation.

Both the vLLM and SGLang integration report graph policy and resource-audit
evidence. Shape calibration remains engine-owned in both integrations and is
included in the graph-plan digest.

## Descriptive memory lifecycle

The engine-neutral memory contract now provides:

- `MemoryLayoutDescriptor`, with layout kind, logical size, allocation and
  mapping granularity, stable address, grouping, registration mode, bounded
  engine extension, and deterministic digest;
- `MemoryResidencyStats`, separating logical, mapped, reserved, preserved,
  discardable, and externally hydrated bytes;
- extended `ProviderCapabilities` for exact reservation, granular or batched
  mapping, physical-backing release, distributed atomic mapping, revisioned
  limits, recipe replay, registration ownership, and checkpoint quiescence;
- immutable `MemoryCheckpointPlan` records with provider/layout identity,
  policies, bounded memory and transport actions, byte expectations,
  participants, quiescence evidence, and terminal/retry semantics;
- cross-rank plan agreement and capability gates that fail before a provider
  operation can release memory. Plan generation, provider ABI, requested
  regions/policies, and every action-specific capability are revalidated after
  prepare and before commit.

Existing vLLM CuMem and native graph-memory providers expose read-only layout
and residency information. Providers that cannot describe their allocator
state report the optional capability as unavailable. Existing release, remap,
discard, and hydration paths are unchanged. No external allocator, provider
discovery system, daemon, or second control protocol is introduced.

## Qualification status

The software contracts are covered by Go and Python tests, including rejection
of old artifact formats, exact negative-state reporting, deterministic layout
digests, cross-rank agreement, fail-before-release behavior, graph-policy
downgrade, and the two-rank harness protocol.

Hardware qualification is tracked separately because a passing unit test is
not a CUDA/NCCL compatibility claim. Accepted capabilities and completed check
classes are recorded in each release-owned NCCL `qualification.json`; those
records do not depend on retaining raw benchmark reports.

Performance runs and detailed hardware evidence remain private release-review
inputs. Use the [maintained harnesses](../benchmarks/harnesses/README.md) to
repeat matched measurements for a candidate runtime.

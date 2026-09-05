<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Shared model payloads and driver residuals

ColdSnap artifact format 9 separates stable model bytes from snapshot-specific
runtime state. This avoids storing essentially the same large allocation image
once per NVIDIA driver, capture, or capsule.

For every execution worker, capture produces:

- `model-weights.pack`: model-owned byte ranges with deterministic 4 KiB zero
  padding between ranges for direct I/O. Extents are ordered by their SHA-256
  and size, so allocation order and CUDA virtual addresses cannot affect the
  object identity. This file is
  external to the capsule and is addressed as
  `model-payloads/sha256/<digest>.pack` in the configured Hugging Face
  repository.
- `weights.blob`: all non-model bytes within the stable weight allocations.
  This residual is driver/runtime specific and remains in the worker's unit
  capsule.
- `native-manifest.json`: the allocation map plus independent model and
  residual extent maps. It binds each file offset to the captured stable CUDA
  virtual address.
- the normal recovery manifest: the same residual map plus the pinned Hugging
  Face model identity used by recovery-loader fallback.

The capsule also contains CRIU/CUDA/NCCL state, the process address space,
derived-cache seeds, and other driver-qualified runtime state. Capsule and
native replay compatibility therefore remain driver specific even when two
artifacts reference the same model payload digest.

## Capture and publication

`COLDSNAP_EXPORT_MODEL_PAYLOAD=1` makes the vLLM disk backend write the external
model payload while it writes the much smaller residual blob. The adapter
hashes the completed payload and derives its repository path from that digest;
recipes and capture IDs cannot choose the committed object path. If multiple
workers have identical payloads, the artifact may reference the same object
from multiple worker owners and publication uploads it once.

Capture observes the configured vLLM loader's actual checkpoint-to-destination
writes. For n580 this observer wraps InstantTensor (or the selected normal
loader) without replacing its I/O implementation; n610 uses the same observer
through the recovery-aware loader. Both paths therefore classify model-owned
bytes from loader behavior rather than parameter names or driver-specific
addresses.

`coldsnap publish-native` and `sparkrun coldsnap publish-native` publish these
model payload objects. The command name describes the native restore provider;
it does not mean the published object includes driver residual state.

## Restore

Native restore preparation stages only the payload assigned to each worker on
the host that owns that worker. A first-use SHA-256 check produces a versioned
validation record. Activation bind-mounts the payload beside the
worker's capsule-resident residual and native manifest.

Hydration first remaps the captured allocations, then copies the model extents
from the shared payload and the remaining extents from the residual. The maps
must be disjoint and must cover every byte of every preserved allocation;
restore fails closed on gaps, overlaps, size changes, identity changes, or a
payload whose artifact path does not match its SHA-256.

If the artifact lacks a qualified native replay provider, or an optional model
payload cannot be staged in `auto` mode, the pinned safetensors recovery path
remains available. An n580 activation restores its context-free pre-exec
template and explicitly selects the ColdSnap loader before vLLM starts. That
loader reconstructs the stable allocation layout from pinned checkpoint
metadata, then the native provider hydrates the shared pack plus the
driver-specific residual through the normal level-1 wake path. A recovery
activation uses the same allocation path but hydrates model bytes from the
pinned safetensors files.

### Recovery read-through cache

A vLLM recovery restore may also produce the artifact's already-defined native
model payload without rebuilding its capsule. vLLM defaults materialization to
`async`: first-token readiness is governed by the recovery loader, then each
worker copies its immutable resident model bytes into the content-addressed
node-local cache in the background. `required` performs the same work before
restore reports success, while `off` disables it. SGLang defaults to `off` and
rejects `async` or `required` because its integration does not yet implement
the canonical model-payload writer; an already available native SGLang payload
can still be staged and restored.

The cache lives at
`<remote-state-root>/model-payloads/sha256/<digest>.pack`; only the worker that
owns a payload writes or stages that payload. A per-digest file lock makes
same-host duplicate workers converge on one object. The completed file is
accepted on later restores only after a mode-protected validation record binds
its digest to the current device, inode, size, and mtime. Stale or absent
evidence is repaired only after a successful full SHA-256 verification. A
partial write is kept under a temporary name and never becomes eligible for
native replay.

The coordinator's materialization primitive uses the exact canonical writer
used during capture. It does not by itself regenerate CRIU state, residual
overlays, address maps, or OCI capsules. A manager may build an additional
target-local residual overlay after the payload is materialized, as described
below. Capsules captured before canonical materialization support was included
in the vLLM integration must be recaptured once; after that, missing published
model payloads can be reconstructed from safetensors on any compatible node.

The engine integration owns construction because it has the live allocation
map. The shared Go payload verifier owns admission. Sparkrun and the Go adapter
both invoke the same release-matched verifier on the data-owning host, so the
full SHA-256, stable identity fields, and validation-record acceptance rules do
not drift between manager and controller implementations.

### Target-local residual overlays

Portable n580 capsules deliberately re-execute the context-free process
template against the destination's driver and CUDA userspace. That maximizes
portability, but it repeats more vLLM initialization than an exact-target
capture. A manager can treat an explicit native-materialization operation as a
one-time opportunity to capture target-local residual state after the shared
model payload is present.

Sparkrun compares the resulting artifact with the portable descriptor. The
model-payload objects must have the same owner, role, size, and SHA-256; they
remain shared and are never copied into the overlay. If the capsule and replay
objects are also identical, the temporary capture is discarded. Otherwise,
Sparkrun stores a local overlay descriptor bound to the portable descriptor,
snapshot driver, rank-ordered manager host identities, accelerator inventory,
and exact installed NVIDIA driver versions. Ordinary restore selects it only on
the same matching target that owns the local capsule images. Missing, stale, or
invalid overlay metadata falls back to the portable artifact.

This is an acceleration cache, not a new portable publication. Its target-local
capsules and residual maps can change after a driver/runtime update and are
removed with the recipe's local ColdSnap artifacts.

## Sharing boundary

Content identity, not model ID alone, determines reuse. Tensor-parallel shards
normally produce one digest per worker; data-parallel workers may legitimately
share one digest and object. Different drivers or runtime builds share a model
payload only when the exact ordered model bytes match. Their residual blobs,
manifests, address maps, runtime state, and capsules are never inferred to be
interchangeable from that match.

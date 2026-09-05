<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Recovery-aware operations

ColdSnap artifacts have two independent parts:

1. A per-unit process capsule contains CRIU state, non-weight CUDA residuals,
   derived-cache seed data, the exact runtime bundle, and acceptance evidence.
   The capsule may be delivered in a digest-pinned per-unit OCI image.
2. Model weights have two providers. An optional shared worker payload contains the
   final captured allocation layout and gives the full-blob activation path.
   The required recovery provider reconstructs those allocations from the
   original model's immutable Hugging Face safetensors revision.

The native model payloads are deliberately optional. There is no required ColdSnap blob
registry. A deployment may publish them to a pinned Hugging Face repository,
keep them only in the node cache, or omit them. `auto` prefers a verified native
pack and falls back to safetensors; `native` fails if a pack is unavailable;
`recovery` always uses safetensors; and `cache-only-auto` never contacts Hugging
Face while looking for the optional pack.

## Operation request

`coldsnap capture`, `coldsnap publish-native`, `coldsnap publish`,
`coldsnap restore`, and the request-selected `sleep`, `wake`, and `status`
operations accept a strict, engine-neutral JSON request with
`--request-json PATH` or `--request-json -`. The request binds:

- the model ID and immutable revision;
- launch units with their destination placement, visible devices,
  digest-pinned capture image, command, environment, and mounts;
- accelerator workers, ordered process groups, service domains, and an opaque
  engine-adapter topology digest;
- process, weight-provider, cache, and validation policies; and
- the output or committed artifact path.

Unknown fields, incomplete unit/worker inventories, mutable image references, missing
recovery identity, and trailing JSON fail before an engine adapter is invoked.
The vLLM and SGLang adapter entrypoints share the schema and provider-selection
contract without putting engine-specific cases into the coordinator or artifact
code. Each entrypoint selects its engine policy behind the common
`internal/inferenceadapter` orchestration boundary.

The committed format-9 artifact separates portable process identity from placement.
Unit/worker/group/service topology, the adapter topology digest, semantic unit commands,
non-transport environment, and mount targets remain exact. The capsule digest
pins vLLM, NCCL, CUDA userspace, and process state. A destination may change
hostname, IP, GPU UUID/ordinal, mount source, and NCCL/UCX/interface values when
it has the same process architecture, GPU model, and compute capability and an
NVIDIA driver at least as new as capture by default. Disabling
`policy.compatibility.enforce_captured_driver_floor` relaxes only the captured
floor, never the selected snapshot driver's minimum. The kernel release is provenance:
default admission runs the capsule-pinned `criu check` against the destination
kernel, while `policy.compatibility.kernel: exact` opts into uname equality.
Restore remaps TCP endpoints
and uses coordinated CRIU, CUDA, and process-release barriers; the restored
worker applies the orchestrator's destination transport environment before
the capsule-pinned NCCL runtime rebuilds communicators.

This policy was qualified by swapping logical TP0 and TP1 between two hosts for
Qwen3.8-27B-FP8 TP2. Both ranks completed TCP remap, coordinated CRIU/CUDA
barriers, NCCL checkpoint restore, recovery hydration, and the exact acceptance
response. The capture boundary reported zero remaining NCCL IB devices, MRs,
network references, and PD references.

## Sparkrun recipes

Sparkrun's ColdSnap plugin owns a top-level `coldsnap:` recipe item. The plugin
parses, validates, and exports that item; it is not stored under `metadata` and
does not leak into a runtime's unknown configuration fields.

```yaml
recipe_version: "2"
model: Qwen/Qwen3.5-0.8B
model_revision: <immutable-model-commit>
runtime: vllm-distributed
container: org/coldsnap-vllm@sha256:<image-digest>
defaults:
  tensor_parallel: 2

coldsnap:
  capsule:
    repository: ghcr.io/example/qwen-coldsnap-capsules
```

This intentionally shows the useful minimum. The omitted process, weights,
cache, and validation fields use the canonical defaults. Local artifact paths
are managed by Sparkrun from the intent and recipe fingerprint. Publication
also packages the small descriptor JSON as OCI in the capsule repository. A
portable recipe may pin the returned reference explicitly:

```yaml
coldsnap:
  artifact:
    reference: oci://ghcr.io/example/qwen-coldsnap-capsules@sha256:<descriptor-digest>
  capsule:
    repository: ghcr.io/example/qwen-coldsnap-capsules
```

Without an explicit reference, the capture controller uses its promoted local
generation and another controller falls back to the stable OCI descriptor tag
derived from the capsule repository and recipe fingerprint.

Sparkrun materializes its resolved placement and vLLM launch commands directly;
there is no second ColdSnap profile file. Before a restore, it downloads and
fully SHA-256 verifies worker packs concurrently on their owning unit hosts when
no valid validation record exists. That first staging pass writes a versioned
sidecar. Unchanged later restores validate the record plus device, inode, size,
and mtime without rereading the large files. Missing or stale evidence is
repaired by a full SHA-256 pass; ctime is diagnostic only. ColdSnap independently
matches the record's digest to the committed artifact before choosing the
provider.

```bash
sparkrun coldsnap capture recipe.yaml --cluster two-node
sparkrun coldsnap publish-native recipe.yaml --cluster two-node \
  --hf-repo example/qwen-native
sparkrun coldsnap publish recipe.yaml --cluster two-node
sparkrun coldsnap restore recipe.yaml --cluster two-node
sparkrun coldsnap native-status recipe.yaml --cluster two-node
sparkrun coldsnap restore recipe.yaml --cluster two-node --dry-run
sparkrun run recipe.yaml --cluster two-node
```

The explicit restore remains a diagnostic/manual surface. Normal
`sparkrun run` treats the presence of `coldsnap:` as recipe-local opt-in to the
restore strategy. It fails before replacement when there is no committed
local or OCI artifact. For `auto`, Sparkrun first stages optional model payloads, then
invokes `coldsnap restore --prepare-only` to strictly validate the artifact and
ensure its digest-pinned unit capsules are resident. Verified model payloads
supersede normal model preparation; when they are unavailable, Sparkrun next
prepares the pinned Hugging Face snapshot for safetensors recovery. Only a
successful receipt can cross Sparkrun's later eviction hook and activate.

An optional top-level request `workload` identity lets an orchestrator supply
its canonical cluster, intent, recipe, runtime, model, and served-model values.
The vLLM adapter then names restored containers `<cluster_id>_node_<unit-index>` and
adds normal Sparkrun lifecycle labels. Direct ColdSnap requests may omit it.

`--dry-run` renders the operation through the current recipe integration. Live
capture, publication, and restore use the installed engine adapter:
`coldsnap-vllm-adapter`/`COLDSNAP_VLLM_ADAPTER` or
`coldsnap-sglang-adapter`/`COLDSNAP_SGLANG_ADAPTER`. The shared adapter is
topology-neutral: it derives launch units, workers, ordered groups, services,
hosts, device assignments, image identity, model identity, commands, and
mounts entirely from the request. B12x
behavior remains behind the vLLM plugin's storage-adapter extension points;
the adapter contains no DS4F model cases.

Capture performs an exact validation request, writes a recovery residual and
replay plan, optionally writes one model-only payload per worker while the CUDA
allocations are resident, and captures the process with CRIU/CUDA checkpoint.
After the captured process exits, ColdSnap preserves its stdout/stderr as
`capture.log` and truncates `target.log` in place. Each restore resets
`target.log` again before CRIU starts and writes a structured restore-boundary
marker as its first line. Capture diagnostics therefore remain in the capsule,
while the workload log exposed to an orchestrator contains only the current
activation.
It then builds one local OCI image per unit from the exact digest-pinned base image.
The OCI build excludes `model-weights.pack`, retains the residual blob and small native manifest,
copies only the declared derived/JIT cache directories back to their exact
absolute paths, and records the resulting image digest in the committed
artifact. Cache paths are restricted to known vLLM/SGLang compiler-cache roots;
they may not overlap a launch mount. In particular, the Hugging Face cache and
local model mounts cannot become capsule seed data. The committed capture
descriptor records each local `sha256:<image-id>` and is immediately usable on
the original unit hosts. Capture never pushes OCI images.

Capsule and model-payload publication remain independent. An explicit
`coldsnap publish` operation verifies an accepted local descriptor and every
resident unit image on its original capture host, pushes those capsules to
`policy.capsule.repository`, and writes a separate descriptor containing
immutable repository digests.
An explicit `coldsnap publish-native` operation rehashes the accepted
capture-local model payloads on their original hosts, uploads distinct content objects to the
configured Hugging Face revision using controller-held authentication, and
writes a separate descriptor with the pinned provider.
After the final upload, the adapter resolves the Hub branch to its immutable
commit and records that commit—not the mutable upload branch—in the artifact.

## Placement portability contract

Artifact format 9 records a fail-closed, per-unit platform inventory before
workload eviction. Restore requires the same process architecture, GPU model,
CUDA compute capability, logical execution graph, adapter topology
digest, model revision, and
semantic serve command. Mount targets and non-transport environment remain
exact; mount sources may change. The capsule itself pins CUDA, NCCL, vLLM,
Python, and the remaining process userspace. A destination NVIDIA driver is
admitted only when its numeric version meets the selected snapshot driver's
minimum and, by default, the captured floor. The latter can be relaxed with
`policy.compatibility.enforce_captured_driver_floor=false`.
Captured kernel release remains recorded, but the default `capability` policy
admits a different release only after the exact CRIU binary in each assigned
capsule passes `criu check` on that destination. Set
`policy.compatibility.kernel` to `exact` for strict uname equality.

Hostname, management/fabric IPs, GPU UUID, device ordinal, master address, and
master port are placement rather than artifact identity. ColdSnap rewrites only
the decoded CRIU TCP endpoint records needed for the captured-to-destination IP
mapping. The graph-recreation path resets NCCL network state before fresh
communicator initialization; `preserve-nccl-exec` reconstructs the admitted
transport endpoints in place while retaining graph-visible resources.

CRIU leaves every restored process tree stopped. After every peer endpoint
exists, ColdSnap restores CUDA in each unit and stops the tree again; only an
all-unit CUDA barrier releases normal execution. The restored workers then
replace captured NCCL/UCX/interface variables with the rank-bound destination
transport environment before the capsule-pinned NCCL runtime initializes its
new network state.

Restore verifies the committed artifact, admits compatible replacement hosts,
selects a provider, validates staged model-payload trust receipts on their
destination nodes, and launches each capsule. The same captured process can
activate either path:
a tiny worker-local provider marker selects the staged full pack, while absence of
that pack selects the pinned safetensors replay plan. The serving unit performs the exact
post-restore validation request before the adapter reports success.

The adapter requires the manager host provider for host Docker and filesystem
operations. Sparkrun serves that provider over a private
operation-scoped Unix socket and may use SSH underneath its own cluster
transport; ColdSnap itself sees only the provider contract. Any other caller
must implement the same manager-provider protocol and generate the strict request consumed by
`coldsnap capture|publish-native|publish|restore|sleep|wake|status --request-json`.
The native coordinator traffic used by restored ranks is authenticated CSKV.
A future node service or Kubernetes provider can replace the manager's current
host transport without changing request or artifact formats.

## Runtime and capsule construction

Build a ColdSnap-enabled vLLM base with `make vllm-runtime-image`, or use
Sparkrun's `coldsnap` builder with a digest-pinned vLLM or SGLang recipe image.
Both paths assemble the qualified CRIU, Go CRIU, CUDA checkpoint, and NCCL
runtime inputs. See [`deploy/vllm/README.md`](../deploy/vllm/README.md) and
[`deploy/sglang/README.md`](../deploy/sglang/README.md). The resulting image
carries the capture-sensitive ColdSnap runtime. The matching engine adapter
stages its small content-addressed n580/n610 activation controllers separately
and mounts them read-only. Capsules therefore omit those controller scripts.
The activated container view provides these stable entry points:

- `/usr/local/bin/coldsnap-engine-rank-n580` or
  `coldsnap-engine-rank-n610`, supplied by the adapter, for either engine's
  per-unit capture or restore;
- `/usr/local/bin/coldsnap-coordinator` for activation-scoped CSKV;
- `/usr/local/bin/coldsnap-criu-rpc` and `cuda-checkpoint` for process state;
- `/opt/coldsnap/plugin` for the selected engine integration; and
- `/opt/coldsnap/native`, `/opt/coldsnap/nccl`, and `/opt/coldsnap/criu` for the
  sealed native runtime.

The vLLM capture command must use a safetensors-backed format and include
`--enable-sleep-mode`. Capture prefers installed loaders in this order:
`instanttensor`, `fastsafetensors`, then the recipe's `auto`/`safetensors`
fallback. ColdSnap observes the ordinary vLLM copy operations without replacing
that fast initial I/O path. Immediately before checkpoint, it changes the live
vLLM load config to `coldsnap`; restore commands are also normalized to
`--load-format coldsnap`. Thus native-payload fallback remains recovery-capable
without charging the recovery iterator to capture startup.

## Native coordinator

The NCCL checkpoint shim talks to the Go `coldsnap-coordinator` using CSKV
version 1, an authenticated length-prefixed
binary protocol scoped to one activation. The endpoint descriptor is mode 0600
and contains the address, scope, and random token. Supported operations are set,
blocking get, delete, health, atomic add, and append. Keys and values are opaque
bytes, preserving the NCCL shim's existing state machine without a text-protocol
translation layer.

The Go and Python coordinator tests share protocol fixtures with the native
client contract. Hardware qualification separately exercises checkpoint,
transport release, repeated restore, and exact-response inference with the
release-owned NCCL provider. Historical timing runs do not define coordinator
compatibility.

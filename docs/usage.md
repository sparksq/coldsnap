<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Basic capture, publish, and restore usage

ColdSnap has two equivalent entry points:

- `coldsnap capture|publish-native|publish|restore|sleep|wake|status --request-json` consumes a strict JSON operation
  request directly.
- `sparkrun coldsnap capture|publish-native|publish|restore|warm|sleep|wake|status` turns a Sparkrun recipe and placement
  plan into that same request, then invokes ColdSnap.

The current end-to-end operation adapters support vLLM and SGLang through the
same engine-neutral request, artifact, and manager-provider contracts. Qwen
SGLang TP2 native and recovery capsule restores are qualified on both snapshot
drivers; the n580 boundary performs a normal pinned SGLang startup and is not
an initialized-CUDA resume.

Direct requests must choose a snapshot driver explicitly. Use `n610` for the
initialized-CUDA snapshot path on NVIDIA driver 610 or newer, or `n580` for the
pre-CUDA process-template path on NVIDIA driver 580 or newer. Sparkrun probes
every placed host and selects the newest driver supported by all units; recipes
do not need to name it.

## Prerequisites

On the controller host:

1. Build ColdSnap and put `coldsnap` plus the matching vLLM or SGLang adapter
   on `PATH`.

   ```bash
   make build
   export PATH="$PWD/bin:$PATH"
   ```

   If the adapter is installed elsewhere, set `COLDSNAP_VLLM_ADAPTER` to its
   absolute path, or `COLDSNAP_SGLANG_ADAPTER` for the SGLang adapter.

2. Start a manager host-provider for the selected placement and export its
   operation-scoped endpoint and token before invoking a direct request:

   ```bash
   export COLDSNAP_HOST_PROVIDER_SOCKET=/run/user/$UID/coldsnap/provider.sock
   export COLDSNAP_HOST_PROVIDER_TOKEN=operation-scoped-secret
   ```

   Sparkrun creates this provider automatically. Other callers must implement
   the documented protocol; ColdSnap does not open direct SSH sessions.

3. Give that remote account permission to run Docker and GPU containers. Each
   launch-unit host also needs the qualified CRIU/CUDA checkpoint runtime, NVIDIA
   Container Toolkit, and access to the model cache or mount declared in the
   request.

4. Build a ColdSnap-enabled vLLM image from the exact digest-pinned runtime
   used for capture. Follow [the vLLM runtime image guide](../deploy/vllm/README.md).
   Push it or load it onto every capture host. Portable requests use a reference
   of the form `repository@sha256:<64 lowercase hex digits>`. A locally prepared
   capture may instead use the exact bare `sha256:<image-id>` after the
   orchestrator verifies it on the assigned host. Mutable tags are rejected.

5. Pin `model.revision` to an immutable Hugging Face commit. A branch such as
   `main` is not a reproducible recovery source.

The artifact manifest is written on the controller. Remote process state is
written below the user cache, normally `~/.cache/coldsnap`, by default. Capture
refuses to reuse existing state or overwrite an artifact, so use a new request
ID and output path for each capture attempt.

## Direct ColdSnap usage

The smallest direct example is TP1. Replace the host, image digest, model
commit, and mount source below with real values. Keep the service on port 8000
for this basic command.

Create `capture-request.json`:

```json
{
  "format": 4,
  "kind": "coldsnap-operation-request",
  "operation": "capture",
  "id": "qwen08b-tp1-capture-v1",
  "snapshot_driver": {"id": "n610"},
  "output": "./artifacts/qwen08b-tp1.json",
  "launch": {
    "engine": "vllm",
    "model": {
      "id": "Qwen/Qwen3.5-0.8B",
      "revision": "<immutable-hugging-face-commit>",
      "source": "huggingface"
    },
    "units": [
      {
        "id": "unit-0",
        "index": 0,
        "host": "gpu-a.example",
        "devices": ["0"],
        "image": "registry.example/coldsnap-vllm@sha256:<64-hex-digest>",
        "image_digest": "sha256:<same-64-hex-digest>",
        "command": [
          "bash",
          "--noprofile",
          "--norc",
          "-c",
          "exec vllm serve Qwen/Qwen3.5-0.8B --load-format instanttensor --enable-sleep-mode --port 8000"
        ],
        "environment": {
          "HF_HOME": "/cache/huggingface"
        },
        "mounts": [
          {
            "source": "/srv/huggingface-cache",
            "target": "/cache/huggingface"
          }
        ]
      }
    ],
    "execution": {
      "workers": [
        {
          "id": "worker-0",
          "unit": "unit-0",
          "service": "model",
          "process_slot": 0,
          "device_slots": [0]
        }
      ],
      "groups": [
        {
          "id": "world",
          "kind": "vllm:world",
          "service": "model",
          "members": ["worker-0"]
        }
      ],
      "services": [
        {
          "id": "model",
          "role": "vllm:serve",
          "workers": ["worker-0"]
        }
      ],
      "adapter": {
        "schema": "vllm:direct-v1",
        "digest": "sha256:236933adde8b8b19c2ad3b79d2797964764ac4f7aaf767d4d2431784588a1613",
        "payload": {
          "dimensions": {
            "tensor": 1,
            "pipeline": 1,
            "data": 1,
            "expert": 1,
            "context": 1
          },
          "runtime": "vllm-direct"
        }
      }
    }
  },
  "policy": {
    "process": {
      "backend": "cuda-criu",
      "kv_discard": true,
      "async_graphs": true
    },
    "weights": {
      "mode": "recovery",
      "recovery": {
        "enabled": true,
        "source": "huggingface-safetensors",
        "loader_backend": "auto"
      }
    },
    "cache": {
      "seed": true,
      "paths": [
        "/var/cache/coldsnap/runtime"
      ]
    },
    "capsule": {},
    "compatibility": {
      "enforce_captured_driver_floor": true
    }
  },
  "validation": {
    "health_path": "/health",
    "prompt": "Reply with exactly: coldsnap-cuda-snapshot-ok",
    "expected": "coldsnap-cuda-snapshot-ok"
  }
}
```

The example expands common policy fields for reference. Direct requests may omit
`format`, `kind`, `policy`, and `validation`; ColdSnap applies the canonical
defaults before strict validation and before computing the committed request
digest. The defaults are `cuda-criu`, semantic KV discard, asynchronous CUDA
graphs, automatic native-then-recovery weight selection, derived-cache seeding,
captured-driver floor enforcement, and the
`/health`/`coldsnap-cuda-snapshot-ok` acceptance check. A partial policy may
override only the values which differ; for example,
`"cache": {"seed": false}` disables cache capture without requiring an explicit
empty path list.

`policy.compatibility.enforce_captured_driver_floor` defaults to `true`. Setting
it to `false` allows restore on a driver older than the one used for capture,
but never below the selected snapshot driver's own minimum (`580` for `n580`,
`610` for `n610`). Architecture, GPU model/compute capability, capsule, CUDA
userspace, and NCCL-provider checks remain enforced. Kernel compatibility
defaults to `"capability"`: ColdSnap records the captured release and runs the
exact CRIU binary pinned by each capsule through `criu check` on its destination.
Set `policy.compatibility.kernel` to `"exact"` to require uname equality.

ColdSnap routes supported compiler and JIT caches below the canonical
`/var/cache/coldsnap/runtime` root. This includes the vLLM, Torch, Triton,
FlashInfer, CuteDSL, and CUDA driver caches; the CUDA driver cache is enabled
and bounded to 1 GiB. These are adapter-owned launch settings, so callers
should not set those cache environment variables themselves. Sparkrun capture
may attach a per-unit copy of its resolved runtime-cache leaf at this root.
Capture warmup updates that copy and the finished tree is baked into the OCI
capsule. Restore uses the capsule copy and does not require a host runtime-cache
mount.

ColdSnap rank controllers run as root inside their privileged containers so
they can drive CRIU and CUDA checkpoint operations. They do not retain root
ownership on the host: capture roots, staged runtime caches, derived-cache
copies, failure reports, and materialized native packs are normalized back to
the manager's UID/GID before the operation returns. Sparkrun also mounts the
prepared Hugging Face snapshot and local model inputs read-only. If a manager
implements the host-provider protocol directly, the account serving that
provider is the ownership identity used for these managed paths.

The rank controllers are part of the engine adapter's content-addressed
activation runtime, not new capsule contents. Capture and restore stage that
small pack through the manager provider and mount it read-only at the stable
entrypoint paths. This preserves existing compatible capsule digests and lets
future capsules omit duplicated controller scripts.

Capture the initialized service:

```bash
coldsnap capture --request-json ./capture-request.json
```

The vLLM operation adapter allows 45 minutes by default for capture,
publication, or restore. This accommodates a cache-cold DeepSeek V4 Flash TP2
capture whose model construction, graph warmup, model-payload write, and process
snapshot exceeded 30 minutes during qualification. Set
`COLDSNAP_ADAPTER_TIMEOUT` to a positive Go duration such as `60m` when a
larger model or slower storage needs more time; the same value remains a
fail-closed deadline for every rank.

This mode does not create or require a large ColdSnap weight blob. It stores
the process, CUDA/NCCL, hydration metadata, and bounded derived-cache seeds in a
per-unit OCI capsule. Capture always leaves that capsule in the unit host's
local Docker image store; it never pushes as a side effect. The original,
pinned safetensors remain the recovery provider.

To make an accepted capture portable, copy the capture request to
`publish-request.json` and make these changes while retaining the complete
launch, policy, and validation objects:

```json
{
  "operation": "publish",
  "id": "qwen08b-tp1-publish-v1",
  "artifact": "./artifacts/qwen08b-tp1.json",
  "output": "./artifacts/qwen08b-tp1-published.json",
  "policy": {
    "capsule": {
      "repository": "docker.io/example/qwen-coldsnap-capsules"
    }
  }
}
```

The policy fragment above replaces only `policy.capsule`; retain the other
policy sections from capture. Publish verifies each local image identity on its
original unit host (the publish request cannot substitute hosts), tags and
pushes it, then writes a new descriptor containing
immutable repository digests. It never overwrites the working local artifact:

```bash
coldsnap publish --request-json ./publish-request.json
```

With Sparkrun's host provider, registry login is required only on the
controller. Sparkrun uses its selected cluster transport and controller-side
Docker client, so Docker forwards controller-held registry authorization when
pushing capsules and pulling them into restore hosts. No credentials are
installed on unit hosts. Direct development must supply an equivalent
manager-provider implementation.

For restore, copy the capture request to `restore-request.json` and make these
changes:

```json
{
  "operation": "restore",
  "id": "qwen08b-tp1-restore-v1",
  "artifact": "./artifacts/qwen08b-tp1-published.json"
}
```

The fragment shows only the fields that change; keep the complete `launch`,
`policy`, and `validation` objects from the capture request, and remove
`output`. Use the local artifact instead when restoring on the original unit
hosts without publication. The restore request ID must be new. The model,
immutable revision, execution graph and adapter digest, semantic unit commands,
non-transport environment, and mount targets must match the capture. The
capsule digest pins vLLM, NCCL, CUDA userspace, CRIU state, and the rest of the
captured process image, so the recipe's capture-time builder-image digest is
not destination identity.

A replacement unit must have the same process architecture, GPU model, and
CUDA compute capability. Its destination kernel must pass the capsule-pinned
CRIU capability probe (or match exactly under the opt-in exact policy). Its
NVIDIA driver must meet the selected snapshot driver's minimum and, by default,
the captured floor. Set `policy.compatibility.enforce_captured_driver_floor=false`
to relax only the latter. Hostname,
IP address, GPU UUID/device ordinal,
mount source, and NCCL/UCX/interface environment are placement. ColdSnap
rewrites captured TCP endpoints into the destination unit map, rotates their
nonprivileged ports per activation, restores all CRIU trees stopped, restores
CUDA only after every peer socket exists, and releases all trees only after
every rank has restored CUDA. Immediately before NCCL reconstructs
communicators, the restored worker replaces its captured transport environment
with the destination values supplied to the capsule by the orchestrator.

Restore and validate the service:

```bash
coldsnap restore --request-json ./restore-request.json --prepare-only
coldsnap restore --request-json ./restore-request.json
```

To stop an n610 restore before hydration, add this request field while keeping
the operation `restore`:

```json
{
  "lifecycle": {"activation_state": "warm"}
}
```

For direct live control, copy the complete restore request, retain its exact
artifact, launch topology, validation policy, and `workload.cluster_id`, give
the request a fresh `id`, and change `operation` to `sleep`, `wake`, or
`status`. Invoke the matching command, for example:

```bash
coldsnap sleep --request-json ./sleep-request.json
coldsnap status --request-json ./status-request.json
coldsnap wake --request-json ./wake-request.json
```

The prepare-only command is safe before workload replacement. It validates the
descriptor and launch contract, selects and verifies the weight provider, and
ensures every local-only capsule exists or every digest-pinned registry capsule
is pulled and inspectable. It emits a JSON `coldsnap-restore-preparation`
report and starts no serving workload. Preparation may run disposable
feature-probe containers. When requested with `--receipt-json`, the controller
also writes the authoritative operation receipt.

When recovery is selected, `policy.weights.native.materialize` controls the
node-local native read-through cache:

- `async` (the vLLM default) returns the restored service after validation and writes
  each worker's native payload in the background;
- `required` does not report restore success until every assigned worker's
  content-addressed payload and validation record are complete;
- `off` performs recovery without populating the native cache.

For this **restore-time read-through policy**, SGLang defaults to `off` and
rejects explicit `async` or `required`: its integration does not implement that
recovery-time writer. This is distinct from capture-time native pack generation
and the explicit Sparkrun preparation command described below. Native SGLang
restore supports packs captured locally or published separately.

Materialization requires the committed artifact to declare its native replay
provider and content-addressed per-worker model-payload inventory. Artifacts
captured with `weights.mode: auto` have that inventory. A deliberately
recovery-only artifact cannot establish the expected digest and size of a new
native payload, so `required` fails before launch instead of reporting an
unverifiable success; recapture with `auto` or `native` to enable this path.

The next restore checks
`<remote-state-root>/model-payloads/sha256/<digest>.pack` before capture-local or
published providers. A verified cache hit selects native hydration and avoids
another safetensors recovery. This only writes model payloads; it reuses the
existing driver-qualified residual overlay and capsule.

With a local capsule, restore on the same unit host where capture built the
image. A published descriptor can restore on replacement hosts: prepare-only
pulls its digest-pinned rank images before activation.

### Optional native model payloads

Recovery mode reads the original Hugging Face safetensors through the optimized
ColdSnap recovery loader. To create the faster native layout as an optional
provider, capture with `mode: auto` (the default). Native publication is a
separate post-acceptance operation and is never a capture side effect.

Create `publish-native-request.json` from the complete capture request, retain
its launch and validation identity, and change these fields:

```json
{
  "operation": "publish-native",
  "id": "qwen08b-tp1-publish-native-v1",
  "artifact": "./artifacts/qwen08b-tp1.json",
  "output": "./artifacts/qwen08b-tp1-native.json",
  "policy": {
    "weights": {
      "mode": "auto",
      "native": {
        "repository": "example/qwen08b-coldsnap-weights",
        "revision": "main"
      }
    }
  }
}
```

Run it through the authenticated manager host-provider:

```bash
coldsnap publish-native --request-json ./publish-native-request.json
```

Configure Hugging Face authentication on the manager host-provider before
invocation; it owns publication credentials. The operation first rehashes every
capture-local model payload on its owning unit host,
uploads it directly from that host, then records the repository's
resolved immutable commit—not the mutable upload revision—in a new artifact.
The Sparkrun-held token crosses the selected transport on stdin and is neither
placed in ColdSnap's environment/argv nor installed on the unit host. The
provider runs the `hf` client from each unit's
already-qualified capsule with the payload bind-mounted read-only, so the host
needs no Python or Hugging Face installation. Its Xet working cache stays in
the transient helper container.

Before invoking the direct
adapter for a native restore, the caller must download each model payload onto its worker
host and populate `weights.native.staged` with its absolute path, byte count,
and SHA-256. Sparkrun performs that staging automatically, so it is the simpler
entry point for `auto` or `native` selection. Recovery-only operation needs no
model payload repository and no ColdSnap blob registry.

The first successful full-payload verification creates a mode-0600 validation
record next to the payload. Its receipt binds the committed digest to the
regular file's device, inode, size, and nanosecond mtime; ctime is retained only
as diagnostic metadata because safe ownership normalization may change it
without changing content. Subsequent prepare-only and activation calls accept
matching evidence without re-reading the large payload. Missing or stale
evidence triggers a full SHA-256 verification and an atomic record refresh;
digest mismatch fails closed.

## Sparkrun usage

Sparkrun owns cluster discovery, rank placement, and vLLM command generation.
Its first-party ColdSnap plugin owns the top-level `coldsnap:` recipe item and
materializes the direct JSON request; a separate ColdSnap profile file is not
needed.

### Install and verify the manager plugin

ColdSnap 0.3.20 requires plugin 0.1.1's `runtime-v1` host-provider capability.
Upgrade both together; an older provider fails capability admission before
engine operations. Snapshot/artifact formats are unchanged by this release.

The authoritative plugin source is the separate
[sparkrun-coldsnap-plugin repository](https://github.com/sparksq/sparkrun-coldsnap-plugin),
not an in-tree ColdSnap package or an arbitrary Sparkrun checkout. Sparkrun
distributions that include it vendor a pinned snapshot; the plugin repository's
[preview guide](https://github.com/sparksq/sparkrun-coldsnap-plugin/blob/main/DEV_PREVIEW.md)
documents setup before general availability. Its development environment uses
`source dev.sh` with `SPARKRUN_BRANCH=develop-next` for the documented preview,
or `SPARKRUN_CHECKOUT` for an explicit compatible host checkout. This assembles
the host with the live plugin and activates the plugin repository's `.venv`.

Check the resulting installation before operating a cluster:

```bash
sparkrun --version
sparkrun coldsnap --version
sparkrun coldsnap --help
sparkrun registry list
```

The plugin declares `@coldsnap` recipes from
[sparkrun-recipes](https://github.com/sparksq/sparkrun-recipes), using its
`coldsnap-recipes/` subdirectory. The matched `@coldsnap-vanilla` registry uses
`vanilla-recipes/` and is disabled by default. On a compatible host, these are
plugin-provided registry overlays; do not hand-edit `registries.yaml` to install
them. Use normal Sparkrun registry commands to change enablement.

For an already-published, hardware-compatible recipe, prepare it before serving:

```bash
sparkrun coldsnap materialize --cluster two-node \
  @coldsnap/qwen3.8-27b-fp8-coldsnap-tp2-vllm
sparkrun run --cluster two-node \
  @coldsnap/qwen3.8-27b-fp8-coldsnap-tp2-vllm
```

Materialization prepares capsules, caches, and optional native payloads/residual
overlays. It can include a disposable verification restore, so allow for that
work and do not count it as serving-startup timing. The manager acquires the
matching ColdSnap controller tools and creates the host-provider automatically;
the manual controller/provider setup above is for direct API callers.

Sparkrun plugin 0.1.2 adds explicit SGLang `materialize` support on
n580 and n610 using ColdSnap 0.3.20. It generates a
fresh local capture using the existing capture-time pack writer, verifies a
native restore, stops the verification workload, then makes the paired local
capsule and native payload available to subsequent normal `sparkrun run` calls.
The source descriptor is not rewritten, and normal recovery restores retain
materialization `off`; no asynchronous/write-behind writer is involved.

For SGLang, both materialization options default to `required` after resolving
`auto`. `--native-weights off --residual-overlay required` creates and verifies
recovery-only local runtime state. Requesting only native weights still creates
their matching capsule/replay metadata; newly captured packs are never assumed
interchangeable with an older capture's pack. SGLang's capture boundary is
unchanged—this is not vLLM's n580 pre-worker-import optimization.

The local artifact is selected only for the matching source, driver version,
hardware and rank-ordered hosts. Explicit restore `--artifact` bypasses local
selection. Repeating `materialize` verifies the existing local result without
recapturing. Allow enough disk space for capture and verification, and expect
capture to replace any existing deployment of the same recipe.

### Capture your own recipe

For a TP2 capture with automatic native-payload preference and safetensors
fallback, add this policy to a normal Sparkrun recipe:

```yaml
recipe_version: "2"
model: Qwen/Qwen3.5-0.8B
model_revision: <immutable-hugging-face-commit>
runtime: vllm-distributed
container: registry.example/coldsnap-vllm@sha256:<64-hex-digest>

defaults:
  tensor_parallel: 2

coldsnap: {}
```

The minimal recipe inherits the ColdSnap defaults: `cuda-criu`, KV discard,
asynchronous CUDA graphs, automatic native-payload preference with safetensors
recovery, derived-cache seeding, and the model-independent validation prompt
`Reply with exactly: coldsnap-cuda-snapshot-ok` with expected response
`coldsnap-cuda-snapshot-ok`. n610 selects retained NCCL graph execution by
default; n580 selects graph recreation. Add a nested setting only when the
recipe needs to deviate from those defaults.

One deliberate manager default differs from direct requests: Sparkrun sets
`coldsnap.compatibility.enforce_captured_driver_floor` to `false` unless the
recipe opts in. The selected snapshot driver's minimum still applies on every
host. Set this field to `true` to also enforce the capture machine's driver
floor; direct ColdSnap requests default to that stricter policy.

The selected Sparkrun cluster must provide two GPU ranks for this recipe and
must resolve the image to the digest shown above. Inspect the exact ColdSnap
request without starting containers:

```bash
sparkrun coldsnap capture recipe.yaml --cluster two-node \
  --dry-run
```

Capture, optionally publish, and restore:

```bash
sparkrun coldsnap capture recipe.yaml --cluster two-node
sparkrun coldsnap publish-native recipe.yaml --cluster two-node \
  --hf-repo example/qwen08b-coldsnap-weights
sparkrun coldsnap publish recipe.yaml --cluster two-node
sparkrun coldsnap restore recipe.yaml --cluster two-node
sparkrun run recipe.yaml --cluster two-node
```

After capture, `sparkrun run` can restore the local descriptor on its original
unit hosts. `sparkrun coldsnap publish-native` verifies and uploads the optional
model payloads, records the Hub's immutable commit, and activates the new local
descriptor. `sparkrun coldsnap publish` pushes the already-accepted capsules
and promotes a portable descriptor only after every rank has a registry digest.
Then `sparkrun run` is the normal serving path on either original or compatible
replacement hosts. Top-level
`coldsnap:` selects the recipe-local ColdSnap execution strategy; recipes
without it keep the standard runtime path. Restore preparation proceeds in
this order:

1. read and validate the committed current descriptor;
2. stage and verify optional model payloads;
3. obtain ColdSnap's prepare-only receipt while validating/pulling the
   descriptor's per-unit OCI capsules;
4. reconcile those capsules through normal Sparkrun container preparation;
5. prepare the pinned HF snapshot only when recovery was selected;
6. let Sparkrun evict the superseded deployment; and
7. activate restore into canonical Sparkrun containers.

Because activation uses `<cluster_id>_node_<rank>` plus Sparkrun labels, the
restored service participates in normal `status`, `logs`, `stop`, `--ensure`,
proxy discovery, and job-metadata flows.

### Warm and live sleep/wake

An n610 capsule can be restored into a deliberately non-serving warm state:

```bash
sparkrun coldsnap warm recipe.yaml --cluster two-node
sparkrun coldsnap status recipe.yaml --cluster two-node
sparkrun coldsnap wake recipe.yaml --cluster two-node
```

Warm preparation and replacement are identical to a normal restore through
CRIU process restoration, distributed CUDA restoration, and NCCL checkpoint
reconstruction. The rank controllers then publish durable warm state before
the collective `/wake_up` boundary. Model allocations therefore remain
unhydrated and discardable KV remains unmapped until `wake` performs hydration,
health admission, and the recipe's exact-response validation. Snapshot driver
`n580` rejects this state because its portable boundary is intentionally
pre-CUDA and cannot provide the same short activation path.

An already running ColdSnap workload can enter and leave vLLM level-1 sleep
without replacing its containers:

```bash
sparkrun coldsnap sleep recipe.yaml --cluster two-node
sparkrun coldsnap status recipe.yaml --cluster two-node
sparkrun coldsnap wake recipe.yaml --cluster two-node
```

Sparkrun resolves the single running cluster ID matching the recipe intent;
ambiguous or absent workloads fail without mutation. ColdSnap then verifies
the exact capture, driver, unit, and Sparkrun labels on every container. Sleep
runs the validation canary, releases managed weight and graph allocations, and
discards configured KV payloads. Wake hydrates/remaps them collectively and
must pass the same canary before returning. Every transition checks the
per-worker hibernation records, so a partial distributed state is rejected.

The retained vLLM processes, CUDA contexts, NCCL communicators, and other
runtime allocations still consume a fixed amount of GPU memory. Sleep means
"model payload offloaded", not zero VRAM. It also cannot survive removal of the
live containers; use the published capsules for recovery after replacement or
host restart.

Sparkrun stages managed captures under
`~/.cache/sparkrun/coldsnap/artifacts/<intent-id>/<recipe-fingerprint>/drivers/<n580-or-n610>/pending/`.
After a successful committed capture, it atomically installs an immutable
generation and promotes `current.json`; restore selects that active descriptor
automatically. `--output` and `--artifact` remain explicit path overrides; an
explicit capture output is not promoted into or pruned by the managed store.

The controller retains the five most recently activated immutable descriptor
generations per intent and recipe fingerprint. To change this local policy,
configure Sparkrun itself rather than adding anything to the recipe:

```yaml
plugins:
  coldsnap:
    artifact_generations: 5
```

`0` keeps only `current.json`; `unlimited` disables generation pruning. Pruning
is locked, happens only after successful promotion, and never removes the
current descriptor or an in-flight capture. It does not delete remote rank
artifacts, OCI capsules, or Hugging Face model payloads.

`--weights recovery` can override the recipe for a single command. Use `--host`
instead of `--cluster` when that is how the Sparkrun placement is normally
selected.

After publishing model payloads, Sparkrun records their immutable provider in the
artifact; it does not add mutable publication intent to the recipe. A portable
recipe may optionally declare that pinned provider explicitly:

```yaml
coldsnap:
  weights:
    native:
      repository: example/qwen08b-coldsnap-weights
      revision: <immutable-hugging-face-commit>
```

Per-worker file names, sizes, and digests come from the committed artifact;
`files_by_worker` is not an accepted recipe field.

During restore, Sparkrun stages every model payload before activation. The first
use calculates its full SHA-256 and writes a stat-bound preverification marker;
unchanged later uses validate only that marker and the current file identity.
`auto` falls back to the pinned safetensors if a pack is
unavailable; `native` makes a missing pack fatal; `recovery` skips native
staging; and `cache-only-auto` checks only the existing Hugging Face cache.

To distribute non-weight process state, also set:

```yaml
coldsnap:
  capsule:
    repository: ghcr.io/example/qwen-coldsnap-capsules
```

The repository is a destination used only by the explicit
`sparkrun coldsnap publish` command. Capture never pushes it automatically.
Sparkrun also packages the small committed JSON descriptor as OCI in this
repository and reports its immutable reference. A portable recipe may pin it:

```yaml
coldsnap:
  artifact:
    reference: oci://ghcr.io/example/qwen-coldsnap-capsules@sha256:<descriptor-digest>
  capsule:
    repository: ghcr.io/example/qwen-coldsnap-capsules
```

When `artifact.reference` is omitted, the capture controller prefers its local
managed descriptor. A different controller with no local generation falls back
to the stable descriptor tag derived from the capsule repository and recipe
fingerprint.
The generated fallback tag and every local managed descriptor path include the
selected snapshot driver, so an n580 artifact cannot shadow an n610 artifact
for the same recipe fingerprint.

The large model payloads remain separate from the OCI capsules. This permits
small per-unit capsules plus safetensors fallback when model payloads are not
published or not locally available. When capsules were already published,
`publish-native` also refreshes the recipe's stable OCI descriptor tag with the
new pinned native provider; otherwise run capsule `publish` afterward.

## Operational checks

- Use a unique capture ID and output path. A restore also gets a unique request
  ID; ColdSnap retains the original capture identity from the artifact.
- Keep an unpublished capture artifact on the controller. A published
  descriptor can instead be delivered through its OCI reference; embedded
  provider identities should not be edited manually.
- Capture, native publication, capsule publication, and restore must use
  compatible vLLM commands, model commits, and topologies. The capsule digest
  pins vLLM, CUDA, and NCCL userspace; the
  capture-time builder image is not destination identity.
- Derived-cache paths cannot overlap model/Hugging Face mounts. The defaults are
  deliberately bounded to known compiler and kernel caches.
- Direct JSON decoding rejects unknown fields. Use Sparkrun `--dry-run` to
  inspect a generated request when diagnosing schema or placement failures.
- Capsules and native replay manifests are driver-bound. Shared model payloads
  are content-addressed and may be referenced by n580 and n610 artifacts when
  their exact bytes match. An n580 artifact is never admitted by an n610
  request (or vice versa), even on hardware capable of running both paths.
- A locally retained capsule is not portable. Publish capsules or preload their
  digest-pinned images before restoring onto replacement hosts.
- Portable placement may change hostnames, IPs, GPU UUIDs, and ordinals. It
  requires exact process architecture, GPU model, and compute capability; the
  destination kernel must pass capsule-pinned CRIU capability admission unless
  exact kernel policy is selected. The destination NVIDIA driver must meet the
  snapshot-driver minimum and the captured floor unless that floor is explicitly
  relaxed by policy. Logical unit/worker/group/service topology
  remains unchanged.

For the artifact/provider model and detailed lifecycle, read
[Recovery-aware operations and OCI capsules](recovery-aware-operations.md).
For all supported policy fields, read [Configuration](configuration.md).

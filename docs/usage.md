<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Controller usage and requests

The `coldsnap` controller consumes strict operation requests from a placement manager.

`coldsnap capture|publish-native|publish|restore|sleep|wake|status --request-json`
accepts a request directly. For recipe and CLI workflows, use the
[Sparkrun reference](sparkrun-usage.md).

The current end-to-end operation adapters support vLLM and SGLang through the
same engine-neutral request, artifact, and manager-provider contracts. Qwen
SGLang TP2 native and recovery capsule restores are qualified on both snapshot
drivers; the n580 boundary performs a normal pinned SGLang startup and is not
an initialized-CUDA resume.

Direct requests must choose a snapshot driver explicitly. Use `n610` for the
initialized-CUDA snapshot path on NVIDIA driver 610 or newer, or `n580` for the
pre-CUDA process-template path on NVIDIA driver 580 or newer. The caller must
verify that every placed launch unit supports the selected driver.

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

   Implement the [manager host-provider protocol](host-provider-protocol.md).
   ColdSnap does not open direct SSH sessions.

3. Give that remote account permission to run Docker and GPU containers. Each
   launch-unit host also needs the qualified CRIU/CUDA checkpoint runtime, NVIDIA
   Container Toolkit, and access to the model cache or mount declared in the
   request.

4. Build a ColdSnap-enabled image from the exact digest-pinned engine runtime
   used for capture. Follow the [vLLM runtime image guide](../deploy/vllm/README.md)
   or [SGLang runtime image guide](../deploy/sglang/README.md).
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
should not set those cache environment variables themselves. A manager may
attach a per-unit copy of its resolved runtime-cache leaf at this root.
Capture warmup updates that copy and the finished tree is baked into the OCI
capsule. Restore uses the capsule copy and does not require a host runtime-cache
mount.

ColdSnap rank controllers run as root inside their privileged containers so
they can drive CRIU and CUDA checkpoint operations. They do not retain root
ownership on the host: capture roots, staged runtime caches, derived-cache
copies, failure reports, and materialized native packs are normalized back to
the manager's UID/GID before the operation returns. Mount prepared Hugging Face
snapshots and local model inputs read-only. The account serving the
host-provider is the ownership identity used for these managed paths.

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

Registry authentication belongs to the manager host-provider. Its implementation
must authorize capsule pushes and pulls without serializing credentials into
ColdSnap requests or installing them on the GPU hosts.

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
and explicit preparation through a fresh capture. Native SGLang restore
supports packs captured locally or published separately.

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
The manager-held token crosses the selected transport on stdin and is neither
placed in ColdSnap's environment/argv nor installed on the unit host. The
provider runs the `hf` client from each unit's
already-qualified capsule with the payload bind-mounted read-only, so the host
needs no Python or Hugging Face installation. Its Xet working cache stays in
the transient helper container.

Before invoking the direct
adapter for a native restore, the caller must download each model payload onto its worker
host and populate `weights.native.staged` with its absolute path, byte count,
and SHA-256. This staging is the manager
responsibility for `auto` or `native` selection. Recovery-only operation needs no
model payload repository and no ColdSnap blob registry.

The first successful full-payload verification creates a mode-0600 validation
record next to the payload. Its receipt binds the committed digest to the
regular file's device, inode, size, and nanosecond mtime; ctime is retained only
as diagnostic metadata because safe ownership normalization may change it
without changing content. Subsequent prepare-only and activation calls accept
matching evidence without re-reading the large payload. Missing or stale
evidence triggers a full SHA-256 verification and an atomic record refresh;
digest mismatch fails closed.

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
- Direct JSON decoding rejects unknown fields. Inspect the exact request JSON
  generated by your manager when diagnosing schema or placement failures.
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

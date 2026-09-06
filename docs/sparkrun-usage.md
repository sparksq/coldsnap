<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Sparkrun reference

Sparkrun owns cluster discovery, rank placement, and engine command generation.
Its first-party ColdSnap plugin owns the top-level `coldsnap:` recipe item and
materializes the direct JSON request; a separate ColdSnap profile file is not
needed.

## Setup and published capsules

Install or update Sparkrun through [sparkrun.dev](https://sparkrun.dev/).
The ColdSnap plugin ships with Sparkrun. During the integration preview,
enable it explicitly:

```bash
sparkrun setup features enable plugins.coldsnap
sparkrun coldsnap --version
sparkrun registry update
sparkrun recipe search @coldsnap
```

The enable step is temporary; ColdSnap will be enabled by default once the
integration matures. Enabling the plugin contributes the `@coldsnap` registry
from [sparkrun-recipes](https://github.com/sparksq/sparkrun-recipes/tree/main/coldsnap-recipes).

For a published capsule, materialize on the destination placement, then run:

```bash
sparkrun coldsnap materialize --cluster two-node \
  @coldsnap/qwen3.8-27b-fp8-coldsnap-tp2-vllm
sparkrun run --cluster two-node \
  @coldsnap/qwen3.8-27b-fp8-coldsnap-tp2-vllm
```

Use the [community quickstart](https://coldsnap.sh/docs/getting-started/quickstart/)
for the complete first-run workflow. The sections below cover recipe policy,
artifact selection, and the details behind those commands.

## Capture your own recipe

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

Capture, run and measure, then publish when satisfied:

```bash
sparkrun coldsnap capture recipe.yaml --cluster two-node
sparkrun run recipe.yaml --cluster two-node

# Publish after validating responses and startup performance.
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

## Warm and live sleep/wake

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

## Local artifacts and retention

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

`--weights recovery` can override the recipe for a single command. Use `--hosts`
instead of `--cluster` when that is how the Sparkrun placement is normally
selected.

## Published providers and portable descriptors

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

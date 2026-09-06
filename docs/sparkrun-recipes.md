<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Sparkrun recipe policy and weights

Sparkrun's ColdSnap plugin owns a top-level `coldsnap:` recipe item. The plugin
parses, validates, and exports that item; it is not stored under `metadata` and
does not leak into a runtime's unknown configuration fields.

## Recipe and artifact identity

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

## Restore preparation and staging

Sparkrun materializes its resolved placement and engine launch commands directly;
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
sparkrun run recipe.yaml --cluster two-node

# Publish after checking responses and startup performance.
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

## Inspect generated requests

`--dry-run` renders the operation through the current recipe integration. Live
capture, publication, and restore use the installed engine adapter:
`coldsnap-vllm-adapter`/`COLDSNAP_VLLM_ADAPTER` or
`coldsnap-sglang-adapter`/`COLDSNAP_SGLANG_ADAPTER`. The shared adapter is
topology-neutral: it derives launch units, workers, ordered groups, services,
hosts, device assignments, image identity, model identity, commands, and
mounts entirely from the request. B12x
behavior remains behind the vLLM plugin's storage-adapter extension points;
the adapter contains no DS4F model cases.

## Choose a weight provider

The default `auto` policy prefers verified native packs and falls back to the
pinned Hugging Face safetensors. Set `native` to require a compatible pack,
`recovery` to select safetensors, or `cache-only-auto` to check only the existing
cache for optional native packs.

Compare providers one run at a time on the same prepared placement:

```bash
sparkrun coldsnap restore recipe.yaml --cluster two-node --weights recovery
# Measure the response and startup time, then stop this run before the next.
sparkrun stop recipe.yaml --cluster two-node
sparkrun coldsnap restore recipe.yaml --cluster two-node --weights native
```

Explicit `native` fails when the matching pack is missing. The command-level
override leaves the recipe unchanged. See [artifact selection and publication](sparkrun-usage.md#published-providers-and-portable-descriptors)
for pinned providers and descriptors.

## Materialization by engine

`sparkrun coldsnap materialize` prepares artifacts for the selected hosts.
It is distinct from vLLM's optional native-cache writer during a recovery
restore. The explicit command can prepare native weights and target-local
residual state before later `sparkrun run` calls.

The plugin's SGLang materialization support, introduced in 0.1.2, creates and
verifies a complete local capture. Its native pack and replay metadata remain
paired with that capture. It does not replace a portable capsule's pack with
unrelated newly captured data. Native and residual preparation both default to
`required` for SGLang:

```bash
sparkrun coldsnap materialize recipe.yaml --cluster two-node
```

To prepare recovery-only local SGLang state:

```bash
sparkrun coldsnap materialize recipe.yaml --cluster two-node \
  --native-weights off --residual-overlay required
```

The local result is selected only for its matching source, snapshot driver,
driver version, hardware, and ordered hosts. Repeating materialization verifies
an existing matching result. A restore with an explicit `--artifact` bypasses
automatic local selection.

During ordinary recovery restore, vLLM supports `--materialize-native off`,
`async`, and `required`; its default is `async`. SGLang uses `off` and rejects
`async` and `required` on that restore-time option. Use the explicit
materialization command above to prepare SGLang packs instead.

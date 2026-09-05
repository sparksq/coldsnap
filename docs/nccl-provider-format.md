<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# NCCL provider manifest format 2

ColdSnap selects an NCCL checkpoint provider from a local catalog with:

```text
coldsnap nccl-provider inspect --provider <provider-root>
coldsnap nccl-provider verify --provider <provider-root>
coldsnap nccl-provider assemble --source-root <checkout> --recipe <recipe.json> \
  --qualification <qualification.json> --payload <build-output> \
  --target <target.json> --output <provider-root>
coldsnap nccl-provider select --catalog <catalog> --target <target.json>
coldsnap nccl-provider materialize --catalog <catalog> --target <target.json> --output <clean-root>
```

The catalog hierarchy is
`providers/<provider-id>/<platform-key>/provider-manifest.json`. Provider IDs
and platform keys must agree with their directory names. A selector validates
the entire catalog before returning one exact match; corrupt nonmatching
providers are not ignored.

Manifest JSON is decoded with unknown fields disabled. Format 2 binds:

- the provider ID/revision and exact platform key;
- the patched provider NCCL version, release, SONAME, build ID, and SHA-256;
- whether that payload was selected exactly or as a fallback upgrade;
- a finite set of complete base-image NCCL identities, separately from the
  patched provider payload;
- provider, NVIDIA checkpoint, and dlsym-bridge ABIs;
- sorted capabilities and explicit limitations;
- architecture/CUDA requirements and libc/libstdc++ floors;
- every regular payload's path, role, mode, size, SHA-256, and ELF identity;
- source/build provenance and an accepted capability-admission policy; and
- the logical preload order `coldsnap-dlsym-bridge`, `checkpoint-shim`,
  `nccl-runtime`.

Verification rejects traversal, symlinks, hard links, special files, undeclared
files, incorrect modes/sizes/hashes, unsupported ELF machines, and build-ID or
SONAME disagreement. Provider code is never loaded by the selector.

Assembly also verifies every release-owned patch digest and its exact series
order. Output contracts declare paths, roles, and SONAMEs. Assembly inspects the
actual ELF architecture, SONAME, and build ID, computes fresh payload hashes,
and binds those values into the immutable manifest. It does not compare a new
build with historical compiler-output hashes.

The compiled payload is distributed independently from target-image assembly.
Each immutable provider-revision tag in `docker.io/scitrera/coldsnap-nccl` is a
native `linux/amd64` and `linux/arm64` OCI index. The release recipe pins the
multiarch CUDA build image, supported host platforms, and GPU code-generation
set. Consumers resolve the versioned OCI tag to a digest before assembly;
Docker selects the matching host architecture. Local source builds remain an
explicit fallback and produce equally verifiable manifests, but are not
mistaken for historical golden binaries.

The release-owned `qualification.json` is the admission boundary for features.
It must explicitly accept the provider ID, policy, complete capability set,
transport classes, and completed check classes. Adding a capability therefore
requires updating the admission record after its checks pass. The record has no
benchmark-report dependency or wall-clock expiry; Git history and release
review preserve the decision, while CI checks the machine-readable contract.

Target observations use kind `coldsnap-nccl-provider-target`, format 1. They
bind a digest-pinned base image, the complete observed system NCCL identity,
the exact platform key, observed CUDA/libc/libstdc++ values, required provider
and bridge ABIs, sorted required capabilities, qualification policy, and
transport class. The default `nccl_policy` is `exact`: the provider NCCL release
must match the target release and a fallback-built provider is rejected. The
base path, build ID, and SHA-256 remain diagnostic provenance, but are not ABI
selection keys; same-release rebuilds with the same SONAME are accepted.

An orchestrator may explicitly set `nccl_policy` to
`match-or-latest-qualified`. Assembly then permits a newer provider in the same
NCCL major series, while still binding the complete base and provider
identities into the build record and manifest. It does not permit downgrades or
cross-major substitution. The orchestrator is responsible for selecting only
an accepted release qualified for the requested capabilities and transport. Restore
does not re-run this policy: the artifact and active runtime retain the exact
provider payload selected at image-build time.

The image installer writes `/opt/coldsnap/nccl/active.json` after installing one
provider below its immutable ID/platform path. The separately built bridge hash
is recorded there. Rank launch code revalidates this record, the provider
manifest, every loadable file, and the loaded provider NCCL version before
capture or restore.

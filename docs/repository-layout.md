<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Repository content policy

The public tree contains maintained source, reproducible build inputs,
regression tests, current documentation, and required license/provenance files.

| Area | Retained content |
| --- | --- |
| Root and `.github/` | Project identity, contribution terms, licenses, dependency manifests/locks, build and release automation. |
| `cmd/`, `internal/` | Current Go commands, implementation contracts, and tests, including the separate CRIU RPC module. |
| `integrations/`, `runtime/` | Packaged engine integration and operational runtime assets, metadata, and tests. |
| `native/` | Native helpers, provider ABI, locked release sources and patches, and machine-readable NCCL admission records. |
| `deploy/` | Current binary/runtime builds and the shared, engine-neutral NCCL payload/provider builds. |
| `test/` | Current regression/contract tests and deterministic fixtures. |
| `scripts/` | Maintained repository checks and release/provider tooling. |
| `docs/` | Current operator guides and architectural contracts. |
| `benchmarks/` | Maintained harnesses and their usage documentation only. |
| `third_party/` | Required upstream notices and verbatim license texts. |

Archive historical results, logs, private host inventories, old plans, and
superseded experiments outside the public repository before removing them from
the source tree. Archive their dedicated tests as well; retain tests that still
exercise maintained runtime code. Release-owned `qualification.json` files
are build/admission inputs and must remain with their matching NCCL recipes.

Generated binaries, native libraries, wheels, caches, editor settings, and local
agent state are excluded from source control. Local build products belong in
`bin/`, `build/`, or `dist/`. Store benchmark outputs outside the checkout.
`make repository-layout` rejects archived roots and benchmark result files even
if they were forcibly added despite ignore rules.

## First public publication

A cleanup commit removes files from its tree, but ancestors, tags, other
branches, and release attachments can still expose development history.
Retain that history in a private archive. To publish only a reviewed release
tree, export the committed review branch with `git archive`, initialize a new
repository from that export, and publish only its new root and subsequent
release refs. Do not mirror development refs or change the private repository's
visibility as a substitute for exporting the clean tree.

Before publication, run the repository checks and scan the exact export for
secrets. Confirm the selected public dependency revisions and image digests are
accessible to an unauthenticated consumer. Build and hardware qualification
remain separate from source-tree cleanup.

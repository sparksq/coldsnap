<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Controller binary bundles

`docker.io/scitrera/coldsnap-binaries:<version>` is the plugin's second remote
acquisition source after GitHub release attachments. Each Linux AMD64/ARM64
image contains the controller, vLLM and SGLang adapters, CRIU RPC helper, license
notices, and `/opt/coldsnap/bin/manifest.json`. The manifest binds all four
executable hashes to the source release version, full commit, and platform.

The repository-owned [Docker workflow](../../.github/workflows/build-docker.yml)
uses Docker Hub OIDC connection `14f6b5b4-89d1-444f-911b-98df94b7ac9d` for the
`scitrera` organization. Only the build and manifest-publication jobs request
`id-token: write`; no Docker Hub password or token secret is used. Docker's
[OIDC setup](https://docs.docker.com/enterprise/security/oidc-connections/create-manage/)
requires `docker/login-action` 4.5.0 or newer. The workflow pins 4.6.0 by commit.

The Docker connection must permit pushes to `scitrera/coldsnap-binaries` from
`sparksq/coldsnap`. Allow the `refs/heads/main` subject for manual backfills and
the relevant `refs/tags/v*` subjects for automatic release-tag pushes. The OIDC
subject comes from the workflow's ref, not the source tag checked out by a
manual run. A rejected connection fails the workflow; it no longer silently
skips publication when token secrets are absent.

## Publish or backfill a release

A source tag push triggers publication automatically. To publish an existing
release using the current workflow without moving the Git tag:

```bash
gh workflow run build-docker.yml --repo sparksq/coldsnap --ref main \
  -f release_tag=v0.3.20
```

The workflow resolves the tag to a commit, checks its version catalog, and tests
both Go modules. Native runners build and pull each platform digest, verify
the manifest, ELF architecture, controller identity, and legal files, and only
then publish the versioned multi-architecture tag. The verification helper comes
from the workflow revision, but all build inputs and embedded identities come
from the selected source release. A workflow-only update therefore cannot stamp
an incompatible new commit into a previously pinned controller release.

Existing version tags are never replaced. Backfills publish only the exact
version tag; they do not move `latest` or minor-version aliases. To inspect:

```bash
docker buildx imagetools inspect scitrera/coldsnap-binaries:0.3.20
docker run --rm scitrera/coldsnap-binaries:0.3.20 version --json
```

The bundle is not an inference runtime image and does not require a GPU.
NCCL payload publication is a separate [manual workflow](../nccl/README.md).

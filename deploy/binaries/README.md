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

The same repository and version tag also contain `darwin/amd64` and
`darwin/arm64` descriptors. Each Darwin payload contains only the native
controller, both engine adapters, its three-file checksum manifest, and license
notices. It is an OCI file bundle, not a Linux wrapper image or a runnable
macOS container. The plugin downloads these layers directly from the registry,
without Docker or ORAS, and verifies descriptor digests, image-config platform,
bundle identity, hashes, and Mach-O CPU type before executing the controller.
GitHub's matching `*_darwin_<arch>.tar.gz` assets remain the first source.

Select controllers using the control node's OS and CPU, not Docker Desktop's
Linux VM platform. Linux target CRIU and payload-verifier binaries remain a
separate acquisition even when a Mac and its GPU targets share ARM64 CPUs.
Native Windows controllers are not supported.

The repository-owned [Docker workflow](../../.github/workflows/build-docker.yml)
uses Docker Hub OIDC connection `14f6b5b4-89d1-444f-911b-98df94b7ac9d` for the
`scitrera` organization. Only the build and manifest-publication jobs request
`id-token: write`; no Docker Hub password or token secret is used. Docker's
[OIDC setup](https://docs.docker.com/enterprise/security/oidc-connections/create-manage/)
requires `docker/login-action` 4.5.0 or newer. The workflow pins 4.6.0 by commit.

The Docker connection must grant read/write access to
`scitrera/coldsnap-binaries`. This freshly created GitHub repository uses the
[immutable-ID subject format](https://github.blog/changelog/2026-04-23-immutable-subject-claims-for-github-actions-oidc-tokens/).
Its subject rules are:

```text
repo:sparksq@317042314/coldsnap@1357825449:ref:refs/heads/main
repo:sparksq@317042314/coldsnap@1357825449:ref:refs/tags/v*
```

The first permits manual backfills from `main`; the second permits release-tag
pushes. A legacy rule such as `repo:sparksq/coldsnap:ref:refs/heads/main` does not
match this repository. Confirm the current prefix with:

```bash
gh api repos/sparksq/coldsnap/actions/oidc/customization/sub --jq .sub_claim_prefix
```

The OIDC subject comes from the workflow's ref, not the source tag checked out
by a manual run. If Docker reports `access_denied`, inspect the connection's
Failures table and verify its activation, subject rules, and resource scopes.
A rejected connection fails the workflow; it no longer silently skips
publication when token secrets are absent.

## Publish or backfill a release

A source tag push triggers publication automatically. To publish an existing
release using the current workflow without moving the Git tag:

```bash
gh workflow run build-docker.yml --repo sparksq/coldsnap --ref main \
  -f release_tag=v0.3.20
```

The workflow resolves the tag to a commit, checks its version catalog, and tests
both Go modules. Native Linux runners build and pull each Linux platform digest, verify
the manifest, ELF architecture, controller identity, and legal files, and only
then publish the versioned multi-architecture tag. The verification helper comes
from the workflow revision, but all build inputs and embedded identities come
from the selected source release. A workflow-only update therefore cannot stamp
an incompatible new commit into a previously pinned controller release.

The reusable [macOS workflow](../../.github/workflows/test-macos.yml) runs the
controller tests (including cancellation cleanup) and builds and executes the
three binaries on Intel and Apple Silicon runners. `scripts/package-macos-bundle.py`
packages their bytes into deterministic Darwin OCI layouts. The publication job
uses ORAS to upload those verified blobs and manifests by digest, reads them
back for byte comparison, then adds both Darwin descriptors to the same release
index as the two Linux descriptors. The version tag is withheld unless all four
platform jobs succeed. macOS payloads never increase Linux download sizes.

Existing version tags are never replaced. Backfills publish only the exact
version tag; they do not move `latest` or minor-version aliases. To inspect:

```bash
docker buildx imagetools inspect scitrera/coldsnap-binaries:0.3.20
docker run --rm scitrera/coldsnap-binaries:0.3.20 version --json
```

The bundle is not an inference runtime image and does not require a GPU.
NCCL payload publication is a separate [manual workflow](../nccl/README.md).

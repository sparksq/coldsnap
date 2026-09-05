<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Build and distribution model

ColdSnap has two deliberately separate build products. They solve different
problems and must not be treated as one install.

The manager plugin is a separate project:
[sparkrun-coldsnap-plugin](https://github.com/sparksq/sparkrun-coldsnap-plugin).
Its version and Sparkrun's vendored plugin pin are not controlled by this
repository's `versions.yaml`. Verify the host/plugin pair with
`sparkrun coldsnap --version`; see [manager setup](usage.md#install-and-verify-the-manager-plugin).

## Controller release binaries

Sparkrun runs the engine-neutral `coldsnap` controller on its control machine.
That executable delegates engine operations to matching
`coldsnap-vllm-adapter` and `coldsnap-sglang-adapter` binaries. None needs CUDA
or engine Python packages on the controller.

The `coldsnap` coordinate in `versions.yaml` is authoritative for this Go tool
release. The generated `.github/workflows/publish-go.yml` cross-compiles the
controller, both engine adapters, and the CRIU RPC helper for Linux AMD64 and
ARM64. A `vX.Y.Z` tag must match the declared version. The workflow runs the Go
tests, stamps the version and commit into each applicable binary, and attaches
these archives to the GitHub release:

```text
coldsnap_<version>_linux_<arch>.tar.gz
coldsnap-vllm-adapter_<version>_linux_<arch>.tar.gz
coldsnap-sglang-adapter_<version>_linux_<arch>.tar.gz
coldsnap-criu-rpc_<version>_linux_<arch>.tar.gz
checksums.txt
```

Every binary archive also contains the canonical AGPL `LICENSE` and
`THIRD_PARTY_NOTICES.md`; the generated workflow stages them from the tagged
source tree before creating the archive.

The repository-owned `.github/workflows/build-docker.yml` also publishes the same
release as the multi-architecture
`docker.io/scitrera/coldsnap-binaries:<version>` image. The image contains all
four executables, a per-platform SHA-256 manifest, the canonical AGPL license,
the Apache-2.0 go-criu license, third-party notices, and the project README.
Its OCI metadata records the source, release version, commit, applicable
licenses, and copyright holders. This is a backup distribution transport for
managers; it is not an engine runtime image and is never used as a serving
container.

Both native build jobs and the manifest merge consume one validated version
from `versions.yaml`. Tag builds reject a tag that differs from that version;
manual dispatch uses the selected source revision's declared version.

Regenerate and verify release automation with the pinned
`scitrera-repo-tools` version:

```bash
python scripts/update-versions.py --check
python scripts/generate-ci-gha.py --check
```

To cut a release, update `versions.yaml`, synchronize version declarations,
regenerate workflows with `python scripts/generate-ci-gha.py --force`, merge the
commit, and push the matching signed tag:

```bash
git tag -s vX.Y.Z
git push origin vX.Y.Z
```

The generator owns version checking and Go publication. Docker binary-bundle
and NCCL payload workflows are maintained directly in this repository and
checked with `actionlint`.

Sparkrun resolves the four executables as one indivisible tool set. It first
tries the GitHub release archives and their `checksums.txt`, then the matching
Docker Hub binary-bundle image, and finally a source-pinned Docker build. Every
path records and rechecks the extracted binary hashes and verifies
`coldsnap version --json` before use. A private GitHub repository is supported
through `GH_TOKEN`, `GITHUB_TOKEN`, or an existing `gh auth login` session.
Docker Hub publication requires the repository secrets `DOCKERHUB_USERNAME`
and `DOCKERHUB_TOKEN` (or accessible organization secrets). Without them, the
optional binary-bundle workflow warns and skips publication; GitHub binary
releases remain independent. Public bundle pulls require no registry credentials.

`make build` remains the development build. It writes `coldsnap`, both engine
adapters, and `coldsnap-coordinator` to `bin/`. The CRIU RPC helper belongs to
its own Go module and is built separately by the runtime-image and release
workflows. Local builds are not release artifacts.

## Python distributions

The root `pyproject.toml` is a non-publishable uv workspace. It gives each
Python responsibility an explicit package boundary:

| Distribution | Source | Version ownership |
| --- | --- | --- |
| `coldsnap-core` | `integrations/core/` | `coldsnap` release coordinate |
| `coldsnap-sglang` | `integrations/sglang/` | `coldsnap-sglang` release coordinate |
| `coldsnap-vllm` | `integrations/vllm/` | `coldsnap-vllm` release coordinate |

The three coordinates normally advance together, but they are intentionally
separate. A plugin-only compatibility patch can bump its engine distribution
without renumbering the controller or the other plugin. Version synchronization
also updates both plugins' exact `coldsnap-core` dependency when the core
coordinate changes.

Each package declares `AGPL-3.0-only` as its PEP 639 license expression and
embeds an exact copy of the canonical license in wheels and source
distributions.

Build the complete workspace into `build/python-dist/` with
`make python-package-build`. These wheels are development and integration
artifacts for now; the release workflow does not publish them to PyPI. Runtime
image construction still copies exact sources into the container so captured
plugin digests remain deterministic.

Source-assembled runtimes generate their minimal discovery metadata from the
matching engine `pyproject.toml`. Run `make python-runtime-metadata` to inspect
the generated `.dist-info` files under `build/python-runtime/`; they are also
rendered automatically by image and exact-runtime-bundle assembly.

## Engine runtime images

The controller release is not the vLLM or SGLang runtime. Capture requires an
engine image containing CRIU, CUDA checkpoint tooling, the ColdSnap engine
plugin, native hydration libraries, the qualified NCCL provider, the
coordinator, and the rank entrypoint.

Sparkrun's `coldsnap` builder selects `deploy/vllm/Dockerfile` or
`deploy/sglang/Dockerfile` from the materialized recipe runtime. Both are
multi-stage:

1. Go programs are compiled in a Go builder stage.
2. CRIU and the GPL-2.0-only n580 reset plugin are copied from a digest-pinned
   `ghcr.io/sparksq/criu` image; the Go CRIU RPC runner is built from its pinned
   public source fork. ColdSnap does not compile CRIU plugins.
3. AGPL-licensed CUDA/native libraries are compiled against the target image's CUDA stack.
4. Python plugin sources and discovery metadata are assembled from the package
   manifests without downloading Python dependencies.
5. The final stage starts from the exact digest-pinned engine image and installs
   only the runtime outputs and matching engine plugin.

The derived image identity includes the base-image digest, source commits,
builder schema, and CUDA architecture. Capsules captured from it retain those
capture-sensitive runtime bytes. Capsule restore therefore does not rebuild
ColdSnap and does not download release binaries inside the serving container.
The matching engine adapter embeds a separate, small activation-runtime pack;
it is content-addressed on each host and mounted read-only at the rank-controller
entrypoints. New images omit those scripts, while the mount safely shadows the
copies retained by older ABI-compatible capsules.

SGLang follows the same model: source and native runtime assembly in Docker on
top of a digest-pinned SGLang image. Its controller-side executable shares the
strict request, manager-provider, cancellation, and credential boundary while
using SGLang-specific launch and lifecycle policy.

## Compatibility rule

The controller and adapter are one release unit. Sparkrun must never combine a
controller from one release with an adapter from another. The adapter's
activation runtime must match the selected snapshot-driver ABI. Runtime/capsule
compatibility remains separately enforced by the artifact schema, engine
identity, CUDA/driver admission, topology, and capsule contents; a compatible
controller patch does not by itself require recapture.

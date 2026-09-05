<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# NCCL checkpoint providers

ColdSnap treats every patched NCCL runtime and checkpoint shim as one immutable,
exact-version provider. Common controller code selects providers from an
external catalog through `coldsnap nccl-provider`; compiled NVIDIA/NCCL
libraries are not committed here.

The stable process boundary is defined by
[`abi/coldsnap_nccl_provider.h`](abi/coldsnap_nccl_provider.h). ABI-major 1 uses
a size-versioned immutable function table. Consumers reject compiled/loaded NCCL
version mismatches and missing capabilities before communicator creation.

Provider catalog layout and manifest format 2 are implemented by
`internal/ncclprovider`. Selection matches the admitted NCCL release, SONAME,
and platform contract. Observed file hashes and build IDs remain diagnostic
provenance rather than ABI selection keys.

The repository workflow can publish the compiled payload as a native
`linux/amd64` plus `linux/arm64` OCI index in
`docker.io/scitrera/coldsnap-nccl`. Publication is explicitly dispatched through
the repository workflow; release tags do not publish NCCL payloads. Immutable tags use the
provider revision (`2.31.2-1.coldsnap.12`); the shorter upstream-release tag is
an advancing alias. See [`deploy/nccl`](../../deploy/nccl/README.md) for the
manual workflow and local source-build fallback.

The checked-in 2.31.2-1 and 2.30.7-1 recipes are locked. The 2.31.2-1 provider
also owns the n610 in-place communicator and transport lifecycle used to retain
CUDA graph executables. Its low-level experiment-named environment switch and
`qualified:false` self-report are retained ABI details; production admission is
carried by the release-owned `qualification.json` capability policy and the
assembled provider manifest. New capabilities must be listed in both the recipe
and admission record after their declared checks pass. Compiled payload hashes
and build IDs are measured during assembly and then enforced for distribution
integrity; they are not compared with a historical golden build. Qualification
records have no benchmark-file dependency or time-based expiry. No compiled
provider payload is committed here.

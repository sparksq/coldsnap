<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# NCCL v2.30.7-1 provider source review

Comparison base: NVIDIA NCCL `v2.30.7-1` at
`73cf112295c33aee2b895f329f592f2a9b4b0f97`, against the previously
qualified `v2.31.2-1` provider source.

## Patch application

The three release-owned patches apply to the locked 2.30.7 source in `series`
order with GNU patch `--fuzz=0`. No 2.31.2 patch file is reused. The logical
reset and termination operations are unchanged; the rebase adjusts only
release-specific source context and line placement.

## Private lifetime surfaces

| Surface | 2.30.7 review | Classification |
| --- | --- | --- |
| `bootstrapNetInitDone`, interface name/address | Same process-global discovery lifetime; reset remains valid | manually reset |
| Socket `netRefCount`, `ncclNetIfs`, device PCI paths | Same ownership and finalize contract | manually reset |
| IB `netRefCount`, `pdRefs`, `mrCache` | Fields and last-reference semantics are present and unchanged for the reset path | manually reset |
| IB async event thread | 2.30.7 has the same single detached-thread design; the provider converts it to one joinable thread per discovered device | manually terminated |
| RAS `rasInitRefCount`, `rasTerminate` | Same zero-reference termination boundary | manually reset |
| `setCommAbortFlags`, `commReclaim`, `ncclCommFinalizeAsyncJob` | Synchronous reclaim call remains available with the same job ownership | manually terminated |
| Checkpoint shim handle table and replay records | Required communicator unwrap, registration, window, and split/shrink/grow records are present | manually replayed |

`src/include/comm.h` differs by 71 lines between the releases (60 additions,
11 deletions in 2.31.2), and `src/transport/net_ib/common.h` differs by 191
lines (165 additions, 26 deletions in 2.31.2). None of those additions removes
or changes the private fields used by this provider. This is source review, not
qualification: runtime acceptance still requires the complete socket, IB/RoCE,
RAS, resource-audit, repeat-restore, and real-vLLM matrix.

## API policy

- NVIDIA checkpoint shim wrappers are regenerated from the exact 2.30.7
  headers.
- The ColdSnap provider query and checkpoint entry points are manually
  exported.
- CUDA graph and NCCL device API families remain explicitly unsupported.
- No public or private API is admitted by a semantic version range.

## Reproducible build policy

Builds use the digest-pinned CUDA target image, locked source archive, and
release-owned patches with `--fuzz=0`. The release recipe owns GPU code
generation. `scripts/nccl-nvcc-reproducible.sh` supplies a stable per-output
`--frandom-seed`, and `strip --strip-unneeded` removes non-loadable local
symbol metadata from published shared libraries.

Provider assembly measures current output hashes and ELF build IDs; it does
not compare them with old lab outputs. The adjacent `qualification.json`
records the accepted capabilities and completed check classes. Detailed build
comparisons and hardware run records are retained privately.

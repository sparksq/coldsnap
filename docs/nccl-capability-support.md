<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# NCCL checkpoint capability support

The 2.32.3-1 provider revision 2 extends the existing CRIU-based reconstruction
and in-place lifecycle. Its production admission is accepted for the declared
capabilities, with ARM64 GPU/CRIU and x86 build/ABI evidence below. Active
GIN/RMA, CFT/NVLS, and in-place window/CE resources remain gated. Each new
resource family needs its own restore semantics and qualification evidence.

| Resource or API | Current implementation | Remaining qualification or implementation |
| --- | --- | --- |
| Ordinary built-in NET collectives | Reconstruction and in-place detach/reattach; preflight validates all communicators before mutation | Full ARM64 payload passed cross-host Socket/IB CUDA/CRIU, repeated immutable-artifact restore and eager TP2 engine replay |
| Launch-completion event | Passed through; grouped operations, captured stream dependencies, and repeated CUDA/CRIU restores passed on two 610 hosts | Engine coverage and additional hardware/driver combinations |
| Per-communicator host NVLS mode | Recorded/replayed with communicator configuration; retained in-place | Disabled mode persists through repeated cross-host CUDA/CRIU restores; enabled component combinations need NVLS hardware |
| Progress counters | Shared monitor teardown/restart, drained DMA, stable in-place counter addresses, destination clock recalibration | Full ARM64 payload passed cross-host CUDA/CRIU with restored counters and eager TP2 engine replay; additional hardware coverage remains |
| RAS control plane | Detach registrations, close sockets and join thread, exchange fresh destination listeners on restore | Full ARM64 payload passed cross-host Socket/IB with diagnostics enabled |
| Registration handles and local windows | Reconstruction replay preserves synthetic handles; real local windows passed four registration flags through repeated CUDA/CRIU restores | Active cross-host transport registration/zero-copy and GIN traffic; in-place re-registration remains blocked |
| Idle built-in RMA/GIN discovery contexts | Guarded detach/reinitialize with exclusive ownership and destination device-count checks; no active RMA or GIN traffic admitted | Two-host IB in-place CUDA/CRIU, repeated restore, rank reassignment and unchanged graph replay passed; full ARM64 payload matrix also passed |
| GIN/RMA and device APIs | Resource inventory and early rejection; CFT endpoint exports are marked restore-unsafe | Track/rebind contexts, GDRCopy mappings, windows, signals/counters and external device state; separate backend qualification |
| CFT counted operations and multicast/NVLS retention | Resource inventory and early rejection | Preserve or reconstruct logical endpoints, multicast mappings and graph-visible registrations on capable hardware |
| Copy-engine collective resources | In-place use is rejected pending retention support | Test reconstruction and implement in-place resource ownership, including hierarchical RMA contexts |
| TLS | Recipe enables OpenSSL 3; connection inventory, quiescent reset, private entropy context and mandatory restore reseeding; provider rejects backend mismatch | Final ARM64 payload passed encrypted Socket/IB CUDA/CRIU in both modes and eager TP2 engine replay; additional OpenSSL/runtime combinations remain |
| Rubin and additional GPU partition modes | Current payload does not emit sm_107 | Compatible pinned toolchain, device code, and hardware/topology-specific restore qualification |

The implementation order is early detection and API regressions, existing
lifecycle qualification, progress counters, then window/GIN/RMA/CFT/NVLS
extensions and optional TLS/hardware coverage. Local progress-counter work can
proceed while qualification hardware is occupied, but it does not bypass
production admission.

## Resource preflight ABI

The optional `coldsnapNcclResourcesQuery(1, &table)` export returns the
size-versioned `coldsnap_nccl_resources_v1` table defined in the
[provider header](../native/nccl/abi/coldsnap_nccl_provider.h). Existing provider
and in-place table layouts remain unchanged; older providers may lack this export.

Call `inspect(mode)` while the workload is quiescent, using
`COLDSNAP_NCCL_CHECKPOINT_RECREATE` or `COLDSNAP_NCCL_CHECKPOINT_IN_PLACE`.
The callback returns an NCCL result and records JSON in `evidence_json()` even
when resource admission fails. The JSON pointer is thread-local and valid until
the next inspection on that thread. Invalid mode arguments or failure to resolve
the required runtime entry points clear the evidence. An incomplete communicator
inspection reports `complete:false`; neither empty nor incomplete evidence admits
a checkpoint.

Each communicator reports its hash/rank, registrations and runtime windows,
NET/NVLS/IPC registrations, graph references, GIN/RMA/CFT/NVLS/copy-engine state,
host NVLS configuration, and monitor ownership/calibration. The `blockers` array
contains stable resource-specific reasons. `admitted:true` means only that this
local resource preflight passed; it does not replace provider, artifact,
hardware, topology, or transport qualification.

Both lifecycle implementations invoke preflight internally, so an older
consumer cannot skip the gate by omitting the optional query. Preparation checks
all communicators before mutation. Operational failures during a later detach
or reattach remain fatal for that activation; callers must not resume a workload
with a partially detached transport.

## Qualification progress

The current twelve-patch, OpenSSL-enabled payload builds for ARM64 and x86.
Both OCI artifacts passed descriptor/layer hash checks, ELF architecture checks,
and inspection of all seven recipe SM targets plus compute-120 PTX. The exact
stripped ARM64 libraries passed this matrix on the four GB10 hosts:

| Test boundary | Final payload results |
| --- | --- |
| Native lifecycle, driver 580 | Eleven scenarios: baseline reconstruction, retained graphs, registration replay, progress counters, four real local window flags, and encrypted reconstruction/retention. |
| Encrypted Socket/IB, driver 610 | Eight rank captures and sixteen CUDA/CRIU restores; both lifecycle modes, retained graphs, fresh handshakes, and unchanged original images. |
| Real local windows, driver 610 | Four captures and eight CUDA/CRIU restores; flags 0/1/2/4, two reconstruction cycles per activation, preserved bytes/handles, clean exits. |
| Pre-CUDA process boundary, driver 580 | Socket/IB: four rank captures and eight restores; CUDA initialization and correct collectives occur after CRIU restore. |
| Eager TP2 engine with encryption, driver 610 | Two rank captures and four restores; exact responses, clean shutdown, and all 576 original snapshot files unchanged. |
| X86 CPU ABI | Exact exported library hashes, provider/in-place/resource tables, TLS backend, and encryption configuration pass; GPU execution remains unqualified. |

These are 36 rank restores, including eight restores at a pre-CUDA process
boundary rather than CUDA checkpoint coverage. The private engine harness
configures a public test PSK; application key configuration remains the
application's responsibility. These final-payload results support the accepted
production record. X86 GPU execution and multiple independently captured ranks
sharing one GPU remain outside this qualification. Rank-swap evidence below
belongs to earlier payloads and is not claimed in the final admission checks.

Earlier ten-patch development and full ARM64 payloads passed the following
additional matrix on their own library hashes. These historical runs do not
implicitly qualify a later payload:

| Test boundary | Cases and results |
| --- | --- |
| Driver 610.43.02, two hosts, CUDA/CRIU | Socket and IB/RoCE, each with reconstruction and in-place graph retention. Per build: eight rank captures and twenty-four rank restores, including two restores from the original images and a swapped-host restore. |
| Driver 580.173.02, native lifecycle | Five scenarios with two communicators: reconstruction, in-place graphs, registration replay, and progress-enabled reconstruction/in-place. Two lifecycle cycles per scenario. |
| Driver 580.173.02, full ARM64 payload, context-free CRIU | Socket and IB/RoCE on two hosts: four rank captures and eight restores. Capture precedes CUDA initialization and has no accelerator FDs. Each restored process initializes NCCL, verifies collectives, and completes two communicator reconstruction cycles. |

Captured image hashes remain unchanged, all targets exit cleanly, and progress
monitor registrations resume. In-place CUDA/CRIU replay retains the same graph
object. The context-free test exercises the pre-CUDA process boundary. The ten-patch
full ARM64 payload also passed an eager TP2 inference-engine capture and two
restores on driver 610: two rank captures and four rank restores, with exact
response matching, complete IB resource release, and clean worker shutdown.
Both activations used disposable copies with separately probed TCP port
rotations; all 576 original snapshot files retained their capture-time hashes. The pinned x86 payload also builds successfully, with matching runtime/provider
ABI queries and a passing private-loader CPU regression. Both payloads contain
all seven recipe SM targets plus compute-120 PTX. X86 GPU qualification needs
a compatible driver.

The IB tests exposed idle RMA/GIN contexts and a process-static DMA-BUF
capability cache that outlived its device. The ten-patch lifecycle handles the
idle contexts and reprobes capabilities after device reconstruction. A CPU
regression also tests changed kernel-module availability. Earlier build results
are kept separately from the final ten-patch payload evidence.

A separate native API probe passed two cross-host CUDA/CRIU restores in both
reconstruction and retained-graph modes. Each rank owns two communicators and
uses grouped operations with a caller-owned launch-completion event. The tests
also verify restored progress monitor registrations and the disabled host-NVLS
configuration. These add four rank captures and eight rank restores on the full
ARM64 payload.

The same API probe failed with CUDA launch error 719 when two independently
captured ranks shared one GPU. Both reconstruction and retained-graph cases
failed; the root cause is not isolated. That CUDA/CRIU topology remains
unqualified even though the ordinary single-GPU library lifecycle tests pass.

The four available GB10 hosts have the same hardware, split between drivers
580 and 610. The two-host window probe reports no GPU Direct RDMA support for CUDA
VMM. NCCL therefore disables cross-host symmetric windows and active GIN/RMA on
these hosts. A successful no-op registration with a synthetic shim handle is
not positive window evidence. Those resource families need capable hardware in
addition to their remaining lifecycle implementation.

CFT has a separate toolchain and device gate. NCCL compiles its capability
queries only with CUDA 13.3 or later; the current payload uses CUDA 13.0. Direct
driver queries on both 610 hosts (CUDA driver API 13.3) return zero support for
logical endpoint unicast, multicast, counted operations, and owner-device
access. The 580 hosts (driver API 13.0) reject these attributes as invalid;
that older-driver result alone does not establish a hardware limitation.
The [CUDA attribute definitions](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/cuda_8h_source.html)
and [CFT capability documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/compute-fabric-transport.html)
describe these independent queries. A newer toolkit alone cannot enable CFT
on the tested 610 host configuration.

Single-rank local windows do work on this hardware. The newer window/CFT
shim, paired with the ten-patch full ARM64 runtime, passed default, collective
symmetric, strict-ordering, and GIN-only registration flags. Driver 580 native
tests completed two reconstruction cycles per case. Driver 610 CUDA/CRIU tests
completed four captures and eight restores from unchanged original images,
with two reconstruction cycles per activation, preserved allocation bytes and
collective outputs, and successful deregistration of the original synthetic
handle. Resource inventories confirmed real runtime windows before and after
restore. These tests do not exercise active GIN traffic, cross-host windows, or
strict-ordering semantics. This development-shim evidence is separate from the
older packaged-payload evidence above.

Patch 11 fixes failed-registration output cleanup and synthetic no-op window
lifetime while preserving deferred grouped/nonblocking output storage. The CFT
resource detector also distinguishes the uninitialized device runtime from the
invalid endpoint sentinel; endpoint zero remains a valid, gated resource.
CPU regressions compile the actual patched shim and resource provider.

Retained NCCL 2.32 proxy shared memory measured 68,423,680 bytes, exceeding the
previous 64 MiB CRIU ghost-file limit. NCCL-capable controller defaults are now
128 MiB; explicit limits remain configurable.

Private-library loading also needs the current common dlsym bridge. When a
symbol is absent from the checkpoint shim, the bridge now consults the selected
preloaded runtime before the image's private library. This keeps unwrapped APIs
such as `ncclGetVersion` and `ncclMemAlloc` consistent with communicator calls.
A CPU regression reproduces the previous mixed-version behavior; the exact
engine image now reports 23203 through private loading and routes communicator
creation to the shim. The bridge ABI remains 1.

## Encrypted socket lifecycle

The new recipe builds with `TLS_BACKEND=OPENSSL3`. Target images must provide
`libssl.so.3` and `libcrypto.so.3`; provider image assembly checks their presence.
The provider also checks that its compiled TLS setting matches the loaded
runtime. Existing release recipes retain their original backend setting.
Applications select encryption through NCCL's process-wide `ncclSetEncryption`
API before opening communicators. Merely installing this payload does not
configure a key or enable encryption for an application.

Both checkpoint paths now require every NCCL SSL connection to close before
reset. Reset discards NCCL's SSL contexts and BIO method, preserves the configured
PSK, and marks reseeding mandatory. Reconnection requests live entropy before
creating new contexts. NCCL uses its own OpenSSL library context, keeping the
application's default context untouched. The library context remains allocated
because persistent application threads may own its thread-local random state.
OpenSSL's [library-context API](https://docs.openssl.org/3.0/man3/SSL_CTX_new/)
and [reseed API](https://docs.openssl.org/3.0/man3/EVP_RAND/) provide these semantics.

The optional runtime export `ncclCheckpointCryptStatus` reports whether TLS was
compiled, live connection count, context readiness, and pending reseeding.
It never exposes key material. A real-source CPU regression exercises actual
OpenSSL handshakes and bidirectional encrypted data across repeated resets,
rejects reset with live connections, and injects an entropy failure to verify
that reconnection stays closed until a successful retry.

The TLS development runtime and backend-checked shim passed four two-host
CUDA/CRIU cases on driver 610: Socket and IB, each with reconstruction and
retained graphs. Eight rank captures and sixteen rank restores preserved the
source image hashes, produced correct collectives, and exited cleanly. Each
activation also completes a second library lifecycle cycle. Encrypted sockets
were absent at capture and present again after restore. On IB this qualifies
encryption of socket control traffic; it does not imply encryption of RDMA data.
These are development-build results, separate from the older payload evidence.

The same TLS development pair also passed eager TP2 inference on the two 610
hosts: two rank captures and four rank restores, exact response matching,
clean shutdown, and all 576 original snapshot files unchanged. A qualification
hook configures a public test PSK and verifies that worker crypto objects are
drained with reseeding pending at capture. This tests encrypted NCCL control
connections inside the engine; it does not add application key configuration
or qualify another OpenSSL version.

## CUDA 13.3 qualification variant

A separate private ARM64 SM121 build uses a pinned CUDA 13.3.1 image and
matching CUDA 13.3 runtime. It compiles NCCL's CFT discovery code and newer
CUDA-dependent paths, and passed all eleven native lifecycle scenarios on
driver 610.43.02. Encrypted Socket/IB reconstruction and retained graphs also
passed: eight rank captures and sixteen CUDA/CRIU restores, unchanged original
images, correct collectives, and clean exits. Both hosts report runtime/driver
API version 13030. This is separate evidence from the canonical CUDA 13.0
multi-architecture payload; it does not qualify a CUDA 13.3 inference-engine
image or additional GPU targets.
NVIDIA lists 610.43.02 for this toolkit in its
[CUDA 13.3 release notes](https://docs.nvidia.com/cuda/archive/13.3.1/cuda-toolkit-release-notes/index.html).

The compiled NCCL capability query reports CFT unicast/multicast/counted support
false on both hosts. A local communicator reports basic device API support,
without multimem, host RMA, or GIN in that probe configuration. Both drivers
also report no GPU or host DMA-BUF export capability; requiring NCCL's internal
DMA-BUF backend fails explicitly during initialization. Its driver-version
requirement alone is therefore insufficient.

Copy-engine tests must confirm actual selection: the ordinary path needs
symmetric windows and suitable local GPU connectivity, and hierarchical CE
needs host RMA. A one-rank copy or an accepted ZERO CTA policy does not prove
that NCCL used its copy-engine collective implementation. CUDA 13.3 broadens
compiler/runtime and unsupported-feature coverage on these hosts; it does not
qualify the still-gated active resources.

## Local regression coverage

The [native target](../test/native/nccl_provider_lifecycle.cc) creates two
communicators per process and uses two processes on one GPU with Socket NET.
Its [runner](../benchmarks/harnesses/nccl_provider_lifecycle_smoke.py) builds the
target, runs a private coordinator, bounds execution, cleans up owned processes,
and records the tested library hashes. The full CUDA/CRIU target also records
resource inventories before capture and after restore, and binds progress-counter
settings into its runtime identity. The native runner does not run CRIU or
emulate a separate-GPU or cross-host qualification. Its `window-0`, `window-1`,
`window-2`, and `window-4` modes use one rank and require real runtime windows;
a successful synthetic no-op handle cannot pass those cases. Set
`COLDSNAP_NCCL_TEST_TLS=1` with a TLS-enabled build to add encrypted reconstruction
and retained-graph native tests.

```bash
COLDSNAP_NCCL_TEST_SOURCE=/path/to/patched-and-built/nccl \
NCCL_SOURCE_ROOT=/path/to/upstream/nccl \
  python3 -m unittest discover -s test -p 'test_nccl*.py' -v
```

Promotion requires an accepted release-owned admission record with actual
checks for the added capability. Device API, NVLS, and registration/window
in-place feature identities already exist in the capability contract; extend
those contracts with evidence instead of introducing driver-name shortcuts.

<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# NCCL 2.32.3-1 provider port

Provider `nccl-2.32.3-1+coldsnap.2` carries forward the six-patch lifecycle from
`nccl-2.31.2-1+coldsnap.12`, followed by resource-preflight, progress/RAS, idle-plugin,
IB capability-cache lifecycle, window bookkeeping, and TLS lifecycle patches
(twelve patches total). Its source lock pins upstream commit
`12df1a11afad322be5a204a2db890161cbf8131d` and the SHA-256 of the
[upstream release archive](https://github.com/NVIDIA/nccl/archive/refs/tags/v2.32.3-1.tar.gz).
The tag is lightweight; the existing lock schema's `signed_tag` field names the
tag and does not constitute signature verification.

## Changes requiring a port

- **Socket discovery is shared with Socket RMA.** The NET reset now checks NET
  references, checks RMA references under the RMA mutex, and resets the common
  device cache under its own mutex. It cannot free discovery data while either
  transport still owns it. The new Socket GIN/RMA backend is not independently
  qualified by this port.
- **Bootstrap sends can be asynchronous.** In-place control detach drains
  outstanding sends before closing bootstrap sockets and detaching the proxy
  control plane. The new proxy loop rebuilds its active polling arrays from
  the persistent descriptor table on each iteration; the existing detach and
  reattach descriptor updates remain applicable.
- **The upstream checkpoint shim uses public runtime-property queries.** The
  port retains `ncclCommQueryProperties`, grouped finalization, and separation
  of blocking and nonblocking communicators. It preserves upstream's bounded
  configuration copy instead of overwriting its size/version fields. The
  release-owned in-place and resource-inspection providers add private
  communicator and plugin-header access with object-specific include paths.
- **IB rail discovery has a device-dependent cache.** Device count is recomputed
  on initialization after a network reset. The fixed environment policy can
  remain cached; the source host's device count cannot.
- **Resource preflight runs before mutation.** The optional resources ABI emits
  a per-communicator inventory. Both lifecycle entry points reject unsupported
  GIN/RMA/CFT and exported device resources; in-place additionally rejects
  registration/window, NVLS, and copy-engine state without retention support.
  Every communicator's NET connections are validated before the first detach.
  Exporting a raw CFT endpoint ID marks its owning communicator restore-unsafe.
- **Idle RMA/GIN contexts exist without active RMA traffic.** In-place capture
  validates exclusive shared-resource ownership and absence of device/RMA work,
  then finalizes the built-in plugin contexts without changing their logical
  plugin ownership. Restore initializes fresh contexts and checks device counts
  against the retained topology. The IB GIN device-index cache is reset alongside
  NET discovery. Active traffic, external plugins, windows, and shared ownership
  remain gated.
- **IB capability caches follow device lifetime.** DMA-BUF support is probed
  again after device reconstruction instead of retaining process-static once
  flags alongside cleared per-device results. Kernel-module availability is
  invalidated at network reset and checked on the destination. A CPU regression
  exercises the patched production cache code across changed capabilities.
- **Window shim bookkeeping preserves deferred outputs.** Failed registration
  clears its output, null output arguments are rejected, and successful
  deregistration retires synthetic no-op records. Grouped and nonblocking
  registration retain their output storage until completion. A CPU regression
  compiles the actual shim against a controlled runtime backend; this does not
  claim support for active hardware windows. Resource inspection also handles
  the initialized device-runtime sentinel correctly: invalid CFT endpoint IDs
  mean absent resources, while endpoint ID zero can be valid.
- **Progress counters have an explicit lifecycle.** Capture unregisters every
  communicator from its shared per-device monitor, drains DMA, and joins the
  monitor threads. In-place restore retains counter-buffer addresses, starts a
  new diagnostic counter epoch, recalibrates CPU/GPU time, and re-registers the
  monitors. Reconstruction resets the process-static calibration cache before
  recreating communicators. Partial resume retries do not double-register a
  communicator.
- **RAS is rebuilt alongside the transport.** In-place capture releases RAS
  communicator registrations and shuts down its sockets/thread. Restore exchanges
  fresh RAS listener addresses through the existing activation-scoped coordinator.
  RAS deregistration is idempotent, and restored host identity is refreshed at
  the quiescent boundary. This also releases RAS for plain Socket in-place
  capture, independently of IB cleanup.
- **NCCL now includes Apache-2.0 code and retained BSD-3-Clause portions.** The
  bundled license retains the older release's notice and adds the complete new
  upstream license text. Modified upstream files carry modification notices.
  The source lock, third-party notices, and aggregate OCI license labels reflect
  both licenses. The recipe now enables the optional OpenSSL 3 backend and requires matching runtime libraries.

The provider ABI remains 1.0, checkpoint ABI remains 100, and the recipe carries
the same 13 capabilities as 2.31.2-1. Exact compiled/loaded version admission now
expects `23203`. Device API state retention, NVLS in-place replay, registration
window in-place replay, and shared proxy state remain outside those capabilities.
Existing release sources and their qualification records are unchanged.

## Validation and admission

Local validation on 2026-09-28 used Linux aarch64, one GB10 GPU, driver 580.95.05,
and CUDA 13.0:

- The locked archive hash matched. All three release patch series applied to
  their exact upstream commits with zero fuzz via `test_nccl_release_source.py`.
- Patched NCCL and the checkpoint shim compiled successfully for `sm_121`.
  This was a local source build, not the pinned multi-architecture payload build.
- The runtime provider query returned the expected identity, version pair,
  revision, and capability mask.
- Two local Socket ranks, each owning two communicators, produced the expected
  all-reduce results across two reconstruction or in-place cycles. One CUDA graph
  executable covering both communicators was retained across in-place cycles.
- The tests exercise `nvlsHostMode` preservation and caller-owned
  `launchCompletionEvent` dependencies after grouped collectives, including
  captured stream waits. Captured events are not queried from the host.
- Registration and live-graph preflight failures leave both communicators usable.
  The reconstruction scenario replays a live registration and deregisters its
  original synthetic handle after two cycles.
- Progress-enabled tests verify one shared monitor per GPU, registration of
  both communicators, and recalibration after restore. The native target asserts
  that no progress-monitor or RAS threads remain at the capture boundary.
- Initial validation passed all 27 NCCL Python tests with the optional native
  GPU test enabled. The expanded suite has 37 tests: 34 CPU/source checks pass and three GPU tests are opt-in.
  Later GPU retries encountered `cudaSetDevice` out-of-memory before NCCL
  initialization. Subsequent remote GPU validation below passed.
- A local PyTorch target completed in-place restoration, resource inspection,
  and unchanged graph replay with progress counters. It exposed a test-target
  shutdown ordering issue (the graph remained live during communicator teardown).
  The target now releases its graph first; subsequent remote graph tests verified
  clean shutdown. Local evidence does not constitute production qualification.

The local tests exercise library lifecycle behavior without CRIU or CUDA
checkpoint/restore. They do not qualify cross-host operation, separate GPUs,
IB/RoCE, n610 restore, or an inference engine. The Socket RMA backend reported
GDRCopy unavailable on this host and was not exercised.

Subsequent two-host 610.43.02 qualification completed CUDA/CRIU capture,
repeated restore, and swapped-host restore for Socket reconstruction, Socket
in-place graph retention, and IB/RoCE reconstruction, all with progress counters.
IB in-place preflight exposed the idle-plugin contexts addressed by patch 9.
That build passed all five native lifecycle scenarios on driver 580.173.02 and
completed two-host IB in-place CUDA/CRIU capture. Restore exposed a stale
DMA-BUF capability cache during RMA context reconstruction, addressed by patch
10. The ten-patch build passes the source and CPU regression checks, all five
native scenarios on driver 580.173.02, and two-host IB in-place CUDA/CRIU with
two restores plus swapped-host restore. Graph identity and image hashes remain
unchanged, and targets exit cleanly. Socket reconstruction, Socket in-place
and IB reconstruction also pass the complete matrix on those same binaries:
eight rank captures and twenty-four rank restores in total. Earlier build
results remain separate from this evidence. The graph test also exposed a
68,423,680-byte retained NCCL shared-memory file, so NCCL-capable CRIU controller
defaults increased from 64 MiB to 128 MiB. Explicit limits remain configurable.

The digest-pinned, stripped ARM64 payload with all recipe gencode targets also
passed the complete four-case driver-610 matrix on its own measured hashes:
eight rank captures and twenty-four rank restores. Its five native driver-580
scenarios passed. An additional full-payload, two-host context-free CRIU test on
driver 580.173.02 passed Socket and IB/RoCE: four rank captures and eight restores
from unchanged images. Capture had no accelerator FDs and preceded CUDA
initialization; each restored process initialized NCCL and verified collectives
through two reconstruction cycles. This does not substitute for an engine test.
The pinned full-gencode x86 payload also builds successfully. Both architectures
contain all seven recipe SM targets and compute-120 PTX; OCI descriptor and
layer hashes verify. The x86 provider returns the expected ABI, revision,
compiled/loaded version pair and resource-query table in a CPU-only test. Its
private-loader regression, including `dlerror` behavior, passes. The available
x86 builder has driver 550.107.02, so x86 GPU qualification remains pending.

A separate two-host native API test passed real CUDA/CRIU capture and two
unchanged-artifact restores for reconstruction and retained graphs: four rank
captures and eight rank restores. It exercises grouped launch-completion events,
two communicators per rank, restored progress registrations, and retained
host-NVLS configuration. The same independently captured two-rank test sharing
one GPU failed with CUDA launch error 719 in both modes; that topology remains
unqualified and its cause is not isolated.

The ten-patch full ARM64 payload subsequently passed an eager TP2 engine
capture and two independent activations from the original snapshot: two rank
captures and four rank restores. Each activation returned the exact captured
response, released all workers cleanly, and used fresh probed TCP endpoints in
a disposable artifact copy. All 576 original snapshot files remained unchanged.
The workers reported NCCL 23203 and zero IB resources at the capture boundary.

The later eleventh patch fixes window handle bookkeeping and passes a CPU
regression for no-op, failed, grouped, and nonblocking registration. The GPU
evidence above is bound to the preceding ten-patch payload; rebuilt-payload
qualification is recorded separately. With the additional CFT sentinel fix,
a development shim passed four single-rank real-window reconstruction cases
on driver 580: default, collective-symmetric, strict-ordering, and GIN-only
flags, each with two cycles, byte preservation and original-handle cleanup.
There was no active GIN traffic; this is local library lifecycle evidence.

`qualification.json` is accepted for the recipe's declared capabilities on
the final twelve-patch payload evidence below: ABI/provider-format checks,
artifact round trip, Socket and IB/RoCE restore, repeated restore, retained CUDA
graph replay, local windows, TLS, and a real TP2 engine run. ARM64 has GPU/CRIU
evidence; x86 has build and CPU ABI evidence. The record does not claim x86 GPU
qualification, active gated resource families, or final-payload rank swap.
Earlier-payload rank-swap results are retained as historical evidence only.

To rerun source and metadata checks with an upstream checkout containing all
three locked tags:

```bash
NCCL_SOURCE_ROOT=/path/to/nccl \
  python3 -m unittest discover -s test -p 'test_nccl*.py' -v
```

Set `COLDSNAP_NCCL_TEST_SOURCE` to a built, patched NCCL source tree to also run
`test_nccl_provider_lifecycle_gpu.py`. It needs CUDA, a C++ compiler, and the built
`bin/coldsnap-coordinator`. Without those opt-ins, the corresponding source and
GPU tests are skipped. The standalone runner is
[`nccl_provider_lifecycle_smoke.py`](../../../../benchmarks/harnesses/nccl_provider_lifecycle_smoke.py).

The remaining implementation and qualification stages are tracked in
[`nccl-capability-support.md`](../../../../docs/nccl-capability-support.md).

A subsequent development-shim test adds real local-window CUDA/CRIU evidence:
four single-rank captures and eight restores on driver 610.43.02, covering flags
0, 1, 2, and 4. Each activation preserves input bytes, runs correct collectives,
reconstructs twice, and deregisters the original synthetic handle. Original
images remain unchanged and exits are clean. The shim includes patch 11 and the
CFT sentinel fix; the runtime is the ten-patch full ARM64 payload. This is not
active GIN traffic, cross-host window, or strict-ordering semantics qualification.
The maintained native harness now includes these four local-window modes.

Patch 12 gives encrypted sockets an explicit checkpoint lifecycle. Live SSL
objects prevent reset. Quiescent reset frees SSL contexts, retains the API PSK,
and requires live entropy before any restored handshake. A private OpenSSL
library context isolates this from the application's own TLS users. Provider
identity rejects a compiled/loaded TLS backend mismatch. The release recipe,
payload build, workflow, and assembly validation carry the backend setting.

Actual OpenSSL CPU regressions pass with the backend enabled and disabled,
including refusal of live-session reset and entropy failure/retry. The final
backend-checked development pair passed Socket/IB reconstruction and retained
CUDA graphs on two 610 hosts: eight captures and sixteen immutable-image
restores, correct collectives, clean exits, and fresh encrypted connections.
The final pair also passed eleven native GPU scenarios on 580.173.02 and an
eager TP2 engine capture with two restores on the 610 pair: two rank captures,
four rank restores, exact inference matching, clean shutdown, and 576 unchanged
original snapshot files. A qualification hook configures the public test PSK
and observes drained crypto objects with reseeding pending at capture.
The subsequent canonical twelve-patch payload builds for both architectures.
Exact packaged ARM64 libraries passed eleven native scenarios and 36 rank
restores: sixteen encrypted cross-host, eight local-window, eight pre-CUDA,
and four encrypted engine restores. Original snapshots remain unchanged and
all activations exit cleanly. X86 exact-library CPU ABI/TLS checks pass;
x86 GPU qualification remains unavailable on its installed driver. Production
admission accepts the tested capability scope above. A separate CUDA 13.3.1 SM121 variant
also passed eleven native scenarios and sixteen encrypted cross-host CUDA/CRIU
restores with its matching runtime. Actual capability queries still report no
CFT support on either 610 host; requiring the unavailable internal DMA-BUF
backend is rejected. These results do not qualify active new transport
resources, a CUDA 13.3 engine image, or another architecture.

## Accepted payload identity

The release payload retains the exact libraries used in the final matrix.
Packaging updates the committed source, admission record and image metadata;
it does not rebuild or replace these qualified library bytes.

| Architecture | Library | SHA-256 |
| --- | --- | --- |
| ARM64 | NCCL runtime | `d5f5e6ae5c184e00cd3e61e5cd025125d737eb768e5197acbf8533de39e14e84` |
| ARM64 | Checkpoint shim | `dc172e7e2db2e0ee3c8f7913aae6eaf60d108defef866c5dd9268bb7b063aec6` |
| x86-64 | NCCL runtime | `71ad8a4058ebbd266bb85b6d0abc81397e34243235fdd20eb1959d4102e7cf3e` |
| x86-64 | Checkpoint shim | `ee64e8563825d6418f4eb774613b09e2c46140b70da5c045647e321b05871e35` |

The canonical payload uses the recipe's pinned CUDA 13.0 toolchain and OpenSSL 3.
The CUDA 13.3 experiment is separate and is not the published release.

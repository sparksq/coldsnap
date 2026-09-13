<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Native capture, hydration, and startup calibration

The optional native hydrator writes an immutable snapshot blob directly into
engine-owned final CUDA allocations. It is below the vLLM adapter boundary and
therefore has no dependency on vLLM types. The engine-neutral library default
remains `python`; the qualified vLLM process adapter selects `direct`.

The staged `buffered` and `direct` backends use an aligned pinned-host ring,
bounded concurrent reads, `cudaMemcpyAsync`, and per-stage CUDA events. They
can verify CRC32 or CRC32+SHA256 inline before publishing a restore. The `gds`
backend uses a runtime-loaded cuFile API and writes directly to device memory;
because those bytes do not pass through host memory, it requires an immutable
artifact that has already passed full verification.

Build both native providers:

```bash
make native
```

Calibrate a bounded 512 MiB transfer on the local GPU:

```bash
.venv/bin/python benchmarks/harnesses/hydration_backends.py \
  --library build/native/libcoldsnap_hydration.so \
  --bytes 536870912 --chunk-bytes 67108864 --queue-depth 2
```

The current vLLM adapter selects the qualified `direct` backend and queue depth
for capture and restore. An experimental engine/plugin launch outside that
adapter can select the same runtime settings with
`COLDSNAP_HYDRATION_BACKEND` and `COLDSNAP_HYDRATION_QUEUE_DEPTH`. The
manager-driven request-JSON CLI is the operating surface.

`COLDSNAP_HYDRATION_BACKEND=gds` additionally requires
`COLDSNAP_HIBERNATE_VERIFY_MODE=preverified` and its existing trust-marker
admission. GDS
availability is filesystem/device-specific; a present `libcufile` is only a
dependency probe, not proof that a particular snapshot path supports GDS.

The same library exposes a separately versioned capture ABI. Its `buffered`
and `direct` backends copy final CUDA allocations into a bounded aligned
pinned-host ring, overlap device-to-host copies with positional file writes,
and calculate requested per-extent CRC32/SHA-256 digests from the staged bytes.
Every successful capture calls `fdatasync`; preverified capture additionally
rereads the durable file and compares its digests before publication. Set
`COLDSNAP_CAPTURE_BACKEND=auto` to prefer direct I/O for a fully aligned layout
and otherwise retain the native pipeline with buffered I/O, including when the
target filesystem rejects a direct open. Explicit `direct` remains fail-closed.
`python` disables native capture without changing hydration selection.

Managed engine launches use native capture `auto`. vLLM applies it to the full
live weight blob, recovery/native residuals, and exported model packs. SGLang
uses it through the engine-neutral semantic artifact writer. Both engines retain
`python` as an explicit compatibility fallback.

vLLM recovery and native providers share one residual file. The union of their
required GPU ranges is split at provider boundaries and written once; each
manifest references only its own exact pieces. New `shared-padded-v1` residual
layouts align every file offset and zero-pad extent tails to 4096 bytes. These
padding bytes never read beyond a GPU source or hydrate into a destination.
Logical extent CRCs remain reusable across both views; existing contiguous
manifests retain their original layout rules. The native capture ABI's optional
`COLDSNAP_CAPTURE_PAD_EXTENTS` flag enables direct I/O for these layouts, including
irregular logical lengths. It requires a library implementing that flag; older
libraries reject it rather than silently changing the layout.

Preverified capture performs one readback per unique residual object. The model
pack retains its canonical fingerprint pass and write-time extent CRC/SHA checks;
its sole post-sync readback validates whole-file SHA-256 (including padding) and
publishes the canonical payload validation receipt. Capsule construction supplies
the recorded expected size/digest to that validator, allowing receipt reuse when
the file's device, inode, size and mtime match. A missing/stale receipt still
causes full SHA validation. The generic native writer also combines extent and
whole-file readback checks into a single file pass when both are requested.

`verification_seconds` on the recovery manifest/sleep report includes both unique
objects; `verification_objects` breaks out their readback/admission wall times.
These times include reads and hashing. `phase_seconds.checksum_s` counts write-time
hashing separately, and `canonical_fingerprint_s` includes the model ordering
pass. `capture_backend` reports the actual native/Python and direct/buffered
selection. Preverified Python hydration now skips unused CRC calculation just
as native hydration does; inline hydration still checks CRCs.

Fresh n580 restores resume at the context-free pre-exec boundary and select the
ColdSnap loader before vLLM starts. During initial worker loading, the loader
honors the activation-local provider selector. Native startup uses the capture's
`native-bootstrap-index.json` to construct neutral checkpoint tensors, runs
vLLM's normal weight finalizers, then hydrates the prepared model tensors directly
from `model-weights.pack` before worker loading returns. It validates captured
artifact ownership and semantic model ranges without requiring identical CUDA
addresses or copying capture-time runtime residuals into the new process.
Adjacent unrelated tensors have independent semantic identities, so allocator
placement does not change those identities. Initial hydration retains the usual
preverified admission and performs no extra payload hash or sleep/wake cycle.
The recovery selection continues to use the configured safetensors transport.

For the qualified V4.1 B12x image, initial native hydration then refreshes
block32 linear scales/tiled copies and mHC broadcast tensors derived during
neutral initialization. These plain tensor attributes are not registered model
parameters. The refresh copies rebuilt values into existing storage before
warmup, retaining packed-weight objects, kernel plans and addresses. It does
not rerun MoE finalizers or replace already hydrated prepared expert weights.
WO-A additionally retains original FP8 prefill weights after converting its
registered weight to BF16. Both representations belong to the native payload;
the packed values and dense scale storage have explicit semantic extents.

Capture records bootstrap metadata while native export is enabled, including
after pre-exec has cleared its phase flag. Native startup fails explicitly when
that index is absent; captures produced without it must be recreated. Legacy
version-1 indexes remain readable for ordinary tensors. Version 2 also preserves
vLLM's `Source.file_weight_filter` contract: selected tensors carry the original
`FileTensorSource` path, offset, shape and dtype on meta tensors. Both native
bootstrap and recovery pass those descriptors through without staging,
broadcasting, sampling, or replacing their data with zero tensors. Disk-backed
Engram therefore remains owned by vLLM's file reader. The recovery transport's
byte accounting excludes these metadata-only entries from explicit reads.

The startup matrix keeps vLLM intact and compares full versus text-only model
initialization in both in-process and ordinary multiprocess `LLM()` topologies:

```bash
.venv/bin/python benchmarks/harnesses/vllm_startup_matrix.py \
  --cases inproc-full inproc-text multiproc-full multiproc-text
```

For cache-hit measurements, bind writable persistent directories and set
`VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, and `TORCHINDUCTOR_CACHE_DIR` before
the fresh process starts. The matrix records those paths in its output so a
cache claim remains auditable.

Add `serve-full` and `serve-text` to measure HTTP readiness and streaming TTFT
through the standard vLLM server.

Dense B12x MXFP8 kernel owners also contribute their packed values and both
scale layouts. Runtime LM-head quantization releases its registered source
parameters; omitting the packed owner would restore a neutral output head
during n580 native startup. These extents are shared with the common native
layout used by n610. Captures made without these extents require recapture
when the model uses this kernel.

### Fresh DSv4 native startup

The legacy DSv4 B12x stack and the newer `B12X` attention backend retain FP8
linear scales and fused WO-A/WO-B packed layouts outside registered model
tensors. After hydrating a fresh n580 model, ColdSnap repacks those derivatives
from restored checkpoint parameters and copies them into the existing execution
storage. It also refreshes the first-layer mHC broadcast. The V4.1 block32
refresh remains supported. This does not change native payload identity or
re-run MoE quantization; n610 restores these derivatives from captured runtime
state and retains its existing wake path.

Compiler-cache identity uses the configured model length only in a copy passed to
`ModelConfig.compute_hash`. Warmup and CUDA-graph capture keep the fitted runtime
length, matching the attention buffers already allocated by vLLM. Raising the
live limit during warmup can exceed newer DSv4 C128A metadata buffer capacity.

Before warmup, ColdSnap also synchronizes a v2 ModelState's cached sequence
limit when it shares the fitted model configuration and runner limit. Older
runners without that state retain their existing behavior. Shape calibration
accepts both v2 graph-manager callback APIs and preserves already-retained
graphs and resources.

Fused DSv4 WO ownership also suppresses standalone refresh of its child linear
layers. Newer vLLM disables their warmup providers without the old skip flag;
the fused owner remains responsible for both packed derivatives.

n610 saves `checkpoint-input.json` before CUDA checkpoint, preserving the
already-collected preparation, graph, region, and acceptance evidence if the
driver fails. It adds no tensor hashing or worker RPC at this boundary.

Newer generic DSv4 B12x experts expose `_prepared()` and keep a second prepared
owner on the layer. ColdSnap includes that owner's weights and runtime scales
in native payloads and retains the layer cache during recovery so vLLM reuses
the captured storage. The legacy lookup API and V4.1 model-owned experts retain
their respective lifecycles. Captures made before this discovery fix need
recapture; a small native pack cannot hydrate their omitted MoE weights.


The n610 launcher now disables PyTorch expandable segments for vLLM as well as
SGLang, preserving other allocator options and honoring both configuration
variable spellings. Full DS4F capture with the newer PyTorch 2.13 image fails
inside CUDA checkpoint with expandable segments enabled despite ample available
host memory; the same workload checkpoints with stable segments. This policy
is specific to n610's live CUDA checkpoint; n580's context-free launch is unchanged.

During recovery, newer DSv4 model-wide post-load finalizers are deferred until
vLLM completes layerwise reload and restores live parameter storage. Leaf
quantization finalizers still run online. This prevents mHC refresh from copying
out of a meta tensor, and preserves existing execution-buffer addresses.

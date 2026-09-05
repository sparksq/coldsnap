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
live weight blob, while SGLang uses it through the engine-neutral semantic
artifact writer. Both paths retain their existing Python implementation as an
explicit compatibility fallback.

Fresh n580 restores resume at the context-free pre-exec boundary and select the
ColdSnap loader before vLLM starts. The loader reconstructs the stable model
layout from checkpoint metadata and lets the chosen native or recovery
provider hydrate the final allocations; the retired pre-worker synthetic-loader
bootstrap is not part of current artifacts.

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

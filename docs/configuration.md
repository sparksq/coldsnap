<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# ColdSnap configuration

ColdSnap is the project and product name. The public executable, vLLM plugin
entry point, and SGLang plugin entry point are all `coldsnap`.

Python modules, native symbols, serialized artifact kinds, and `COLDSNAP_*`
environment variables form versioned runtime and artifact contracts. Managers
should configure the request policy and let the selected engine adapter
project its runtime settings.

## Controller policy

`policy.compatibility.kernel` accepts `capability` (default) or `exact`.
Capability mode keeps the captured kernel release as provenance and runs the
CRIU executable pinned by each capsule through `criu check` on its destination.
Exact mode additionally requires the destination uname release to equal the
captured value. Both modes retain exact process architecture, GPU model/compute
capability, snapshot-driver, runtime-provider, and topology checks.

The release-matched engine adapter also owns the activation-runtime pack. It
stages its content-addressed n580/n610 rank controllers through the manager
provider and mounts them read-only; this is not a recipe environment setting.

## vLLM

Enable the plugin with:

```bash
export VLLM_PLUGINS=coldsnap
export PYTHONPATH=/opt/coldsnap-vllm
```

The normal launch surface is:

| Setting | Supported values and meaning |
| --- | --- |
| `COLDSNAP_VLLM_OVERRIDE_CUMEM` | `1` selects ColdSnap's CuMem backend on vLLM lines without a named backend selector. |
| `COLDSNAP_DISK_SLEEP_DIR` | Required local artifact root for live weight sleep/wake. |
| `COLDSNAP_DISCARD_REGIONS` | Empty (disabled) or `kv_cache`. KV discard always means semantic payload selection with fixed zero/readback validation. |
| `COLDSNAP_HIBERNATE_READ_MODE` | `direct` or `buffered`; default `direct`. |
| `COLDSNAP_HIBERNATE_WRITE_MODE` | `direct` or `buffered`; default `direct`. |
| `COLDSNAP_HIBERNATE_VERIFY_MODE` | `inline` or `preverified`; default `inline`. Preverified mode requires an immutable, externally verified artifact. |
| `COLDSNAP_HIBERNATE_REUSE_BLOB` | Reuse an exact restored blob generation; default enabled. |
| `COLDSNAP_CAPTURE_BACKEND` | `python`, `auto`, `buffered`, or `direct`. Managed launches use `auto`; explicit `direct` rejects unaligned extents. |
| `COLDSNAP_HYDRATION_BACKEND` | `python`, `auto`, `buffered`, `direct`, or `gds`. |
| `COLDSNAP_HYDRATION_NATIVE_LIBRARY` | Native hydration library for non-Python backends. |
| `COLDSNAP_HYDRATION_NATIVE_SHA256` | Expected digest for the native library. |
| `COLDSNAP_KV_CAPACITY_GUARD` | Explicit guard override. It is enabled automatically with the CuMem override and never changes an explicit `--kv-cache-memory-bytes`. |

### CRIU process-image I/O

ColdSnap captures CRIU page images with 256 KiB LZ4 blocks at acceleration 1.
The patched CRIU layout preserves the exact compressed length while padding
each nonempty payload to a page boundary. Restore uses one decompression worker
and direct page-image I/O. CRIU automatically falls back to buffered reads for
older packed artifacts that do not advertise the padded layout.

These are ColdSnap-owned defaults for both native model-payload and safetensors-recovery
artifacts; they are not recipe environment settings. The vLLM adapter passes
them explicitly, and the rank controller and standalone CRIU RPC runner use the
same values when invoked directly. Benchmark measurements for these defaults
are maintained outside the public source tree.

### Recovery from original safetensors

The restore-only `coldsnap` load format omits reload-owned model weight payloads
from the rank-local artifact and refills the captured stable CUDA virtual-address pool
through vLLM's supported in-place reload path. It retains a residual blob for
the exact non-weight bytes that share the released weight-pool allocations,
including allocator gaps and auxiliary state. Reload-owned weights include both
registered Parameters and canonical packed expert owners exposed by the active
backend. vLLM remains responsible for
tensor-parallel sharding, model-specific packing, quantization finalizers, and
copying processed weights back into the captured kernel tensor storages.

For capture, launch with `--load-format instanttensor` when available (the
adapter falls back to `fastsafetensors`, then `auto`/`safetensors`) and enable
sleep mode. The capture observer records immutable safetensors ranges and exact
vLLM copy operations, then arms `coldsnap` in the live load config before the
snapshot. Recovery restore is therefore always `coldsnap` regardless of the
capture loader. Set:

| Setting | Supported values and meaning |
| --- | --- |
| `COLDSNAP_RECOVERY_WEIGHT_SOURCE` | `safetensors` enables hybrid residual snapshots and model-source weight recovery. The residual blob contains only bytes outside reload-owned model weight spans; unset retains the full allocation blob path. |
| `COLDSNAP_EXPORT_MODEL_PAYLOAD` | `1` emits a model-only worker payload and split native manifest during recovery capture. An n580 capture also records a capsule-local metadata-only bootstrap index used to recreate its pre-CUDA allocation layout. OCI capsules exclude `model-weights.pack` while retaining the driver-specific residual blob, address maps, and bootstrap metadata; the adapter publishes the payload by SHA-256 and can activate both parts together. |
| `COLDSNAP_MODEL_ID` | Required model identifier bound into the recovery manifest. |
| `COLDSNAP_MODEL_REVISION` | Required immutable Hugging Face revision bound into the recovery manifest. |
| `COLDSNAP_MODEL_METADATA_SHA256` | Optional controller-produced model metadata digest, also bound when present. |
| `COLDSNAP_RECOVERY_LOADER_BACKEND` | Adapter-owned projection of `policy.weights.recovery.loader_backend`: `direct`, `buffered`, `auto` (default), `mmap`, or `torch`. Do not set it in rank environment. `auto` inspects the filesystem behind each opened safetensors file: qualified local ext/xfs/btrfs storage selects the direct-interior/buffered-edge path, known remote filesystems select buffered reads, and unknown filesystems use the conservative buffered path. Receipts record the requested mode, filesystem and mount source, effective mode, reason, and any direct-open fallback. Explicit `direct` remains fail-closed. `mmap` and `torch` remain qualification transports. |
| `COLDSNAP_STARTUP_PLAN_MAX_SHORTFALL_BYTES` | Adapter-owned vLLM startup-plan guard. Managed runtime-cache restores set a 512 MiB bound. A smaller current free-memory value remains admissible only by reducing KV capacity by exactly that shortfall; larger drift falls back to vLLM's full memory profile. |
| `COLDSNAP_RECOVERY_LOADER_DISTRIBUTED` | `1` lets TP ranks divide physical safetensors ranges and directly broadcast their contiguous CUDA slab regions. Restore probes current per-rank disk throughput and accounts for mandatory rank-exclusive reads before assigning shared ranges; capture-time rates remain diagnostic input. `mmap` ignores this control because vLLM demand-pages rank-local TP/EP slices. |
| `COLDSNAP_RECOVERY_LOADER_CHUNK_BYTES` | Native staged-I/O chunk size; default 64 MiB and must be 4096-byte aligned. |
| `COLDSNAP_RECOVERY_LOADER_STAGING_BYTES` | Maximum physical-order CUDA staging slab; standalone default 256 MiB, managed adapter value 1 GiB. Concurrent source/receive slabs are bounded to 2 GiB per rank. |
| `COLDSNAP_RECOVERY_LOADER_COLLECTIVE_BYTES` | Maximum temporary coalescing buffer for fallback broadcasts and direct-broadcast chunk ceiling. Standalone default 64 MiB, managed adapter value 1 GiB; must be 4096-byte aligned. |
| `COLDSNAP_RECOVERY_LOADER_VERIFY_BYTES` | Optional head/tail bytes verified for each tensor after hydration and TP broadcast; 0 disables it and the maximum is 4096. Managed launches use 0 and enable model-level samples instead. |
| `COLDSNAP_RECOVERY_LOADER_VERIFY_MODEL_BYTES` | Head/tail bytes sampled from live model parameters and buffers before sleep and compared after recovery; 0 disables it and the maximum is 4096. Managed launches use 64 bytes. A mismatch aborts wake with the exact tensor names. |
| `COLDSNAP_RECOVERY_DERIVED_BUFFER_MAX_BYTES` | Maximum total size of the small exact derived-buffer capsule retained inside the CRIU process image (runtime attention scales and RoPE caches no larger than 1 MiB); default 64 MiB, maximum 1 GiB. Larger deterministic RoPE caches are rebuilt. Checkpoint-backed model weights are never included. Capture fails rather than silently exceeding the bound. |
| `COLDSNAP_RECOVERY_LOADER_QUEUE_DEPTH` | Native staged-I/O queue depth from 1 through 64; default 4. |
| `COLDSNAP_RECOVERY_LOADER_TRANSPORT_PIPELINE_DEPTH` | `1` (standalone default) serializes physical-batch hydration and TP transport. `2` (managed adapter value) uses a second bounded source slab to hydrate batch N+1 while batch N is packed/broadcast and requires one additional staging slab of CUDA headroom per rank. |
| `COLDSNAP_RESTORE_TRANSPORT_ENVIRONMENT_PATH` | Adapter-owned path to a typed, rank-bound destination transport-environment handoff inside the capsule. Do not set it in a recipe or rank environment. On portable restore the fresh controller writes current `NCCL_*`, `UCX_*`, `OMPI_MCA_*`, and named interface/IP values there; the restored worker replaces captured transport values immediately before NCCL network initialization. |

Optional model-payload admission uses a versioned validation record. The
staging layer performs one full SHA-256 verification and atomically writes a
mode-0600 record beside the pack. ColdSnap checks the committed digest plus the
current device, inode, size, and nanosecond mtime during prepare-only and
activation. Ctime is diagnostic only. An unchanged pack is admitted without
another full read; missing or stale evidence causes a full hash and record
refresh, while a digest mismatch fails closed.

The canonical implementation is `internal/payloadvalidation`. Managed Python
workers invoke it through `/usr/local/bin/coldsnap payload verify`; the engine
adapter helper exposes the same result as `payload-verify`. Long-running local
services can use `coldsnap payload serve --socket ... --token-file ...`, and a
client can select it with
`coldsnap payload verify --rpc-socket ... --rpc-token-file ... --rpc-timeout 30m`.
The token file must be a private regular file. The Python transport controls are:

| Setting | Supported values and meaning |
| --- | --- |
| `COLDSNAP_PAYLOAD_VALIDATION_TRANSPORT` | `auto`, `python`, `cli`, or `rpc`. `auto` prefers a configured RPC socket, then a configured CLI command, and otherwise uses Python. |
| `COLDSNAP_PAYLOAD_VALIDATION_COMMAND` | Shell-style argument prefix whose first item is a clean absolute executable; for example `/usr/local/bin/coldsnap payload verify`. It is parsed without a shell. |
| `COLDSNAP_PAYLOAD_VALIDATION_RPC_SOCKET` | Private Unix socket for the Go validation service. |
| `COLDSNAP_PAYLOAD_VALIDATION_RPC_TOKEN_FILE` | Absolute file containing the operation-scoped RPC token. |
| `COLDSNAP_PAYLOAD_VALIDATION_TIMEOUT_SECONDS` | Positive CLI/RPC deadline, default 1800 seconds and maximum 86400. |

CLI and RPC are transport adapters over the same Go `Validate` function and
produce the same admission object and canonical adjacent validation record.

The residual is hydrated before vLLM reload so its load kernels see the captured
auxiliary state, and once again afterward so any non-weight buffers touched
by loading return to their exact captured contents. Its size is model- and
allocator-dependent, but excludes every byte covered by a reload-owned model
weight tensor.

This mode assumes the pinned Hugging Face snapshot is already available at the
same cache path visible to the restored process. The live snapshot still owns
its CUDA virtual addresses and is not a portable model checkpoint. A recovery
manifest cannot be mixed with another model revision. The adapter's explicit
native activation marker may select the
portable native manifest produced by the same recovery capture; no other blob
generation is accepted.

A recovery loader must be qualified per model/backend combination. Current Qwen
and DeepSeek TP2 native/recovery paths have exact-response coverage on both
snapshot drivers. Detailed performance logs remain development artifacts, not
configuration or compatibility inputs.

These lifecycle settings are used by the reusable-process controllers, not as
general tuning knobs:

- `COLDSNAP_ASYNC_CUDA_GRAPHS=1` serves eagerly, then captures after the first
  completed request at a scheduler-idle boundary. `COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE`
  can hold capture until an external coordinator arms all ranks; the ready file
  and generation settings publish the result.
- `COLDSNAP_SHAPE_CALIBRATION=1` is capture-time work that compiles graph shapes
  before the process artifact is produced. `policy.process.shape_calibration`
  accepts `auto` (default), `enabled`, or `disabled`; `auto` enables calibration
  for capture whenever asynchronous graphs are enabled and leaves restore and
  materialization untouched. vLLM and SGLang each enumerate and warm their own
  engine-native graph plans. Capture fails on incomplete coverage, and the
  artifact records planned/warmed counts, mode details, toolchain identity,
  cache root, and elapsed time for every unit.
- vLLM startup-plan persistence is enabled automatically when the canonical
  managed runtime cache is seeded. Capture records vLLM's exact admitted KV
  capacity and device-independent sizing fingerprint; restore may reuse that
  plan only after vLLM's own fingerprint and free-memory admission checks pass.
  This preserves the recipe's `gpu_memory_utilization` and automatic context
  sizing while avoiding redundant profiling and CUDA-graph memory estimation.
- vLLM multimodal processor warmup is moved off text readiness during restore.
  A timer submits it to the renderer's existing serialized executor after a
  short grace period, so text inference can become ready first without parking
  work on the multimodal executor. The first real multimodal request cancels a
  pending timer and uses vLLM's normal request path immediately. If speculative
  warmup has already started, vLLM's executor serializes the request behind it.
- The n580 and explicit `recreate-from-plan` vLLM restore paths automatically defer
  artifact-redundant profiling and both
  pre-KV and runtime-dependent kernel warmup when the managed startup-plan
  cache supplies an admitted explicit KV size. The controller arms that work
  only after the acceptance response, so it cannot delay first-token
  readiness; CUDA graph recreation remains ordered after deferred warmup. The
  n610 default preserves the qualified graph executables and therefore does not
  schedule that recapture path unless retained-graph acceptance fails.
- Identity, topology, NCCL checkpoint, process-template,
  and startup-plan variables are controller-owned. Use the manager recipe or
  strict operation request instead of assembling those variables independently.

## SGLang

SGLang intentionally exposes a smaller, process-snapshot-only surface:

| Setting | Supported values and meaning |
| --- | --- |
| `COLDSNAP_MODE` | `off` or `capture`; default `off`. Restored capsules continue the captured process. |
| `COLDSNAP_ARTIFACT_DIR` | Required semantic-layout root when enabled. |
| `COLDSNAP_PROCESS_ARTIFACT_ROOT` | Required process artifact and hydration root when enabled. |
| `COLDSNAP_LIVE_BACKING` | `discard`, `disk`, or `cpu`; default `disk`. |
| `COLDSNAP_RUNTIME_DIR` | Writable root for locks and ephemeral live backing. |
| `COLDSNAP_ARTIFACT_PREVERIFIED` | Admit a stat-bound, externally verified immutable blob. |
| `COLDSNAP_EXPORT_MODEL_PAYLOAD` | Export native per-worker weight payload components during capture. |
| `COLDSNAP_SHAPE_CALIBRATION` | Controller-owned capture-time graph-plan calibration. It is enabled by `policy.process.shape_calibration` under the same async-graph rule as vLLM. |

Artifact chunks are fixed at 256 MiB and native hydration queue depth at four.
Runtime contracts and captured identity fail closed; there is no model-family
allowlist or environment bypass.

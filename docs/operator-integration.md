<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Operator integration contract

ColdSnap's public CLI is a controller primitive for placement-aware
orchestrators and integrations. Human
users normally use the orchestration layer's commands. Integrations should use
strict request and receipt JSON and must not parse progress messages.

## Controller discovery

An integration should first run:

```bash
coldsnap capabilities
```

The result identifies the controller build, request, artifact, receipt, and
manager-provider format versions; registered engine adapters; snapshot drivers;
operation replay semantics; and host-provider status. This is a static contract,
not an assertion that every external adapter executable or host prerequisite
is installed. Admission must still verify the requested adapter, driver,
assets, and target hosts.

## Operation invocation

Send one immutable request on stdin and reserve stdout for its result:

```bash
coldsnap restore \
  --request-json - \
  --receipt-json - < restore-request.json > restore-receipt.json
```

Before starting the process, the orchestrator should open an operation-scoped
host provider and set its private `COLDSNAP_HOST_PROVIDER_SOCKET` and
`COLDSNAP_HOST_PROVIDER_TOKEN`. The provider supplies node execution, upload, typed image/workload operations,
and credentialed OCI/Hugging Face operations through its own transport. Engine
adapters require the `runtime-v1` capability. See the
[manager host-provider protocol](host-provider-protocol.md).

With `--receipt-json -`, ColdSnap directs adapter output and progress to stderr
and writes exactly one validated operation receipt to stdout. The process exit
status remains authoritative: failed operations return nonzero and their
receipt has `state: failed`. The receipt binds the operation ID and canonical
request SHA-256, records the selected engine and snapshot driver, and describes
the operation's replay semantics.

Receipt format 2 also carries a bounded `timing` envelope. It contains a span
tree plus an explicit clock inventory rather than one flattened controller
duration. Controller and engine-adapter phases are recorded separately, and
runtime-unit reports add per-unit CRIU, CUDA restore, NCCL, graph-arm, weight
wake, health, and acceptance spans when the selected driver exposes those
boundaries. vLLM hibernation records add one clock per worker with native or
recovery hydration, model-payload/residual I/O and CUDA-copy attribution,
discard/remap work, and recovery-loader/adapter-replay subphases. SGLang NCCL
provider restore timing is likewise imported from its worker state. Each clock
includes its own Unix origin. A manager may use those origins to place the
spans in one visual tree, but must not add or subtract durations from different
clocks: placement-manager, controller, adapter, remote-unit, and remote-worker
clocks are independent.

Worker phase records are aggregate profiler counters and may overlap. Their
spans carry `placement: aggregate` and start at the worker clock origin; they
are suitable for attribution and regression comparison, not for reconstructing
a sequential critical path. Rank-controller timeline pairs such as CRIU begin
and end, graph arm, health, and acceptance retain reported placement. On n580
and explicit `recreate-from-plan` paths, deferred CUDA graph capture completion
is intentionally not awaited by restore and is therefore not guaranteed to
appear in the activation receipt. The n610 default restores retained graphs
before acceptance.

Managers that want live phase boundaries may additionally request the bounded
NDJSON side channel:

```bash
coldsnap restore \
  --request-json - \
  --receipt-json /run/coldsnap/restore-receipt.json \
  --timing-events - < restore-request.json
```

`--timing-events` accepts an absolute, previously absent file path or `-` for
stdout and requires `--receipt-json`. The receipt and event stream cannot both
use stdout. Each event is one complete JSON line and carries timing-event
format 1, a global sequence,
operation ID, and canonical request SHA-256. The lifecycle is `stream_start`,
one or more `clock`, `span_start`/`span_complete`, then `stream_complete`.
Runtime telemetry learned after remote collection may appear as a
`span_complete` without an earlier start. The final event binds its state and
the canonical SHA-256 of the receipt's complete timing envelope.

The NDJSON stream is advisory and append-only; the atomically published JSON
receipt remains authoritative. A manager may render validated events as they
arrive, but must reconcile against the final receipt and fall back to that
receipt if the event stream is absent, truncated, malformed, or has a digest
mismatch. Event-output failures emit a timing warning and do not change the
snapshot operation result. Streams are capped at 8,192 events and 64 KiB per
event.

Consumers should retain the outer manager-measured controller span as the
authoritative end-to-end duration and treat nested foreign-clock spans as
diagnostic attribution. Missing optional runtime-unit telemetry does not fail
an otherwise valid capture or restore; the controller and adapter spans remain
available and emit a timing warning in progress output.

For a durable local handoff, pass a receipt path instead. Publication is
atomic. A failed receipt may be replaced by a later successful run of the same
request; a successful receipt is never downgraded; and ColdSnap refuses a path
already bound to a different request before invoking the engine adapter.

The supported controller sequence is:

1. Resolve placement, select a snapshot driver, and construct the complete
   launch topology.
2. Stage the digest-pinned capsule and optional native model payload for each
   worker. For native payloads, stage the selected engine adapter as the
   release-matched Go verifier on each data-owning host and retain its JSON
   admission result. The adapter stages its complete content-addressed
   activation runtime through the manager provider during prepare-only and
   repeats the same verifier admission before use.
3. Call restore with `--prepare-only` and require a successful receipt.
4. Only then evict or replace the old workload.
5. Call restore without `--prepare-only` and retain its receipt as the
   activation record.
6. Treat the restored containers as ordinary orchestrator-owned workloads for
   logs, status, stop, and replacement.

- ColdSnap owns a separate default policy profile for every snapshot driver.
  Request fields omitted by an orchestrator are resolved from the selected
  `n580` or `n610` profile before validation. Both drivers default to `auto`,
  preferring an available native provider and falling back to the pinned
  safetensors model. Other qualified settings may still diverge without
  changing the request format.
- Operation progress always names the selected snapshot driver. This remains
  visible on stderr when stdout is reserved for a JSON receipt.
- `operation-timing-spans` in `coldsnap capabilities` indicates receipt format
  2 timing support. Managers should negotiate this feature rather than infer it
  from human-readable version output.
- `operation-timing-events-ndjson-v1` and protocol
  `timing_event_format: 1` indicate the optional live stream. Managers must
  continue consuming the final receipt even when they use this feature.

Engine-adapter capabilities also report `native_materialization` and
`default_native_materialization`. vLLM currently supports `off`, `async`, and
`required` with default `off`; SGLang supports only `off`. Both drivers use these
defaults. Managers should explicitly request `off` for ordinary restores when
compatibility with older controllers (which defaulted vLLM to `async`) is needed.
Managers should reject unsupported modes before launch and must still rely on the adapter's
capability check as the authoritative boundary.

Capture, publication, and lifecycle requests use the same envelope. The
capability response declares whether an operation is safe to repeat,
content-idempotent, convergent, replacement-oriented, or conflicts with an
existing output.

## Orchestrator responsibilities

### Controller and target platforms

The CLI and both engine adapters run on the manager's control node, using
Linux or macOS binaries for that node's AMD64/ARM64 platform. Their authenticated
Unix-socket provider and POSIX cancellation contract apply on both systems.
GPU targets remain Linux. A manager on macOS must supply the release-matched
Linux `COLDSNAP_TARGET_PAYLOAD_VERIFIER` and `COLDSNAP_TARGET_CRIU_RPC` files;
same CPU architecture does not make a Mach-O controller usable as an ELF worker.
The Sparkrun plugin acquires these independently. Native Windows requires a
separate IPC/process-lifecycle port and is not supported.

### Cancellation and coordinator ownership

The adapter owns its operation's short-lived coordinator and endpoint files.
It cleans them up after successful use, after failed startup (including a lost
launch reply or partial endpoint distribution), and when the operation is
cancelled. Cleanup uses an uncancelled, bounded context and reports failures
with the exact workload/endpoint and host. A cleanup failure makes the operation
fail; an incompletely cleaned restore attempt is not automatically retried.
Failed restore units and temporary capture units are also removed, while
successful serving restores remain running. Reusable artifacts are retained.

A manager must keep its host-provider server and transport session alive until
the controller **and its adapter child** have finished remote cleanup. Closing
the socket as soon as the manager receives an interrupt prevents even an
uncancelled adapter cleanup context from reaching the target hosts.

The controller forwards cancellation to its adapter with SIGTERM, allowing up
to four minutes for existing cleanup/ownership-repair budgets before forcing
termination. The Sparkrun plugin places the controller tree in a separate
process group, forwards manager cancellation, and waits up to five minutes
before closing the provider. The plugin development revision adds an explicit
escape hatch: a second Ctrl-C (or repeated SIGTERM) immediately forces the
controller tree to terminate. Plugin 0.1.5 instead ignores repeated interrupts
during its grace period. Forced termination is reported as **cleanup unconfirmed**,
not success; unreachable hosts, SIGKILL, or manager crashes can still require
manual recovery scoped to the exact operation-owned containers.

The controller development revision also closes an individual host-provider
connection when its call context is cancelled, waking blocked reads/writes
without shutting down the provider. Fresh, bounded cleanup contexts can still
use it. ColdSnap 0.3.22 lacks this cancellation wakeup and can wait on a capsule
pull until the adapter's forced-shutdown deadline.

These guarantees require the matching controller and plugin shutdown changes;
an older controller's immediate adapter kill bypasses adapter cleanup. A normal
Sparkrun serving-container stop is not a substitute for coordinator teardown.

### Placement and lifecycle responsibilities

ColdSnap does not own scheduling or global cluster state. The caller owns:

- placement and stable launch-unit/worker identities;
- hardware discovery and explicit `n580` or `n610` selection;
- image, model-snapshot, capsule, and per-worker native-payload staging;
- workload exclusion, eviction ordering, and rollback policy;
- registry and Hugging Face credentials;
- retention policy for operation receipts and artifact generations;
- log collection and service exposure.

Artifacts remain placement-independent within their declared compatibility
set. Hostnames, addresses, GPU UUIDs, and local device ordinals are activation
inputs rather than permanent worker identity. Capsules and residual state are
snapshot-driver-specific; exact content-addressed model payload bytes can be
shared by driver variants.

An orchestrator may set
`policy.compatibility.enforce_captured_driver_floor=false` when it has selected
one snapshot driver for the full placement. This relaxes only the floor recorded
from the capture host; ColdSnap still enforces the selected snapshot driver's
minimum NVIDIA driver and every other platform and runtime compatibility check.
Kernel policy is independently `capability` by default: the adapter executes
the capsule-pinned CRIU capability probe on each destination. An orchestrator
may request `policy.compatibility.kernel=exact` when uname equality is a
deployment requirement.

## Kubernetes direction and current boundary

The engine adapter now uses a typed manager runtime rather than constructing
Docker commands. Docker is the current reference manager backend. A Kubernetes manager still needs node-side CRIU/CUDA integration and
hardware qualification, not just a Pod API mapping. See the
[runtime-neutral manager contract](runtime-neutral-managers.md).

The request, artifact, receipt, and timing-event formats are manager-neutral
and can be stored in CRD status or an external object store. A
Kubernetes controller can therefore reconcile the same prepare, replace,
restore, sleep, wake, and status
operations without changing engine or storage formats. It may implement the
manager-provider protocol with Kubernetes exec or a separately deployed node
service. No Kubernetes operator or node service is shipped by this repository.
The operator remains responsible for
transport security, reconciliation, and status.

The SGLang adapter enters through this same manager-provider boundary. Engine
code never owns SSH, registry credentials, or Kubernetes calls; those
responsibilities remain in the manager provider.

An orchestrator may add registries and caches as distribution mechanisms
without making them part of artifact identity.

<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Startup readiness and first-token timing

Development source adds rank-0 startup measurements for running restores with
both engines (vLLM and SGLang) and both snapshot drivers (n580 and n610). These
changes are not part of the existing ColdSnap 0.3.21 / plugin 0.1.4 releases.
Test with this controller source, the matching canonical plugin development
branch, and Sparkrun's matching `develop-next` source. A local controller build
can be selected with `sparkrun coldsnap restore --coldsnap-binary /path/to/coldsnap`
(and its sibling tools), or `COLDSNAP_BINARY` in the benchmark harness.
Installing the plugin alone does not replace its release-pinned controller.

All three durations start at the serving rank-0 container's actual Docker
`State.StartedAt`, not the beginning of the CLI command or an SSH request:

| Metric | End boundary, observed on rank 0 |
| --- | --- |
| TTR port-open | First observation of the serving port in TCP LISTEN state |
| TTR HTTP-ready | First observed HTTP 200 from `/health` |
| Startup TTFT | First non-empty streamed `content`, `reasoning`, or `reasoning_content` delta |

The port probe is passive: it reads `/proc/net/tcp` and `/proc/net/tcp6`, without
opening a connection. HTTP polling and the streaming request run locally on
rank 0, so these durations do not subtract timestamps from different machines.
Clock steps detected during the acceptance request invalidate its timing.

## One acceptance inference, not two

The existing acceptance inference now streams. It retains the same prompt,
temperature zero, 64-token maximum, and exact final-content validation. Empty
role/usage events do not count as tokens. The controller records the first text
timestamp but still consumes the complete response and requires a finish reason
and stream terminator before accepting it. The service readiness barrier still
requires the validated final reply, not merely a first token. A failed attempt
does not supply a successful timing if acceptance is retried.

Successful running restores print a line such as:

```text
ColdSnap startup (rank0-acceptance-v1, rank 0, container start): TTR port-open 2.100s; TTR HTTP-ready 2.200s; TTFT 2.850s (response validated)
```

Those numbers are illustrative, not qualification results. Port-open and HTTP
health can precede usable inference; their gap to TTFT is measured, not assumed
to be less than a second. Missing timestamps are `unavailable`, never zero.
Warm restores do not run inference and do not claim a startup TTFT. Wake and
capture are not presented as Docker-start restore measurements.

The rank-0 runtime report contains
`post_restore_response.coldsnap_acceptance`, including the prompt hash, sampling
settings, first-token field, request-only TTFT, and readiness observations.
Operation receipts export `runtime.startup_port_open`,
`runtime.startup_http_ready`, and `runtime.startup_ttft` spans under a rank-local
clock whose origin is Docker start. These spans overlap: do not sum them.
Acquisition, image distribution, verification, and other work before container
start remain separate from startup TTFT and total CLI latency.

## Sparkrun integration and comparison

The canonical plugin requests the manager's optional container start timestamp
and passes the validated observation to new Sparkrun hosts through
`ActivationResult.startup_observation`. The host reuses it instead of sending
another inference request. Older hosts remain supported; older controllers
without streaming receipts do not gain an invented TTFT.

Normal Docker vLLM/SGLang launches in the matching upstream implementation also
perform one rank-local streaming readiness probe. Ordinary readiness stops
after first text; qualification can consume and validate the full reply. Use
matched prompts, sampling settings, engines, topology, and cache policy when
comparing normal loading with ColdSnap. Fast `sparkrun run --no-follow` remains
non-blocking; it does not wait for a new inference readiness check.

Both paths observe events by polling, not kernel event tracing. ColdSnap starts
its observers before restore work; ordinary Sparkrun starts its probe when the
post-launch readiness wait starts. An already-ready endpoint is therefore
recorded when first observed, potentially later than its actual transition.
`observer_started_unix_ns` is retained to make late observers auditable.

## Qualification profiles

The maintained harness supports `COLDSNAP_TTFT_MEASUREMENT=rank0`. ColdSnap cases
read their built-in acceptance receipt, with no second inference. Normal cases
use the same upstream head-local probe with full exact-response validation.
Both use the qualification prompt and 64-token maximum. See the
[harness instructions](../benchmarks/harnesses/README.md).

The previous external observer remains the default (`external-stream-v1`),
with its existing control-node observation and 128-token maximum. New local
results carry `rank0-acceptance-v1` or `sparkrun-rank0-v1` provenance. Do not mix
these profiles in one comparison or relabel historical results. Switching
published qualification numbers requires a fresh matched matrix; implementing
the measurement does not establish new performance numbers.

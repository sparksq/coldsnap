<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Manager host-provider protocol

ColdSnap request execution is transport-neutral. A placement manager opens an operation-scoped host session through its own cluster
transport and exposes that session to the engine adapter over a local Unix
socket. A future Kubernetes operator can implement the same protocol with a
node API, exec subresource, or DaemonSet. ColdSnap 0.3.20 adds typed runtime
operations so managers need not execute Docker commands supplied by ColdSnap.
See [runtime-neutral managers](runtime-neutral-managers.md) for the exact
contract and remaining Kubernetes prerequisites; Kubernetes is not yet qualified.

The provider is activation state, not artifact identity. Its socket, bearer
token, transport name, SSH options, host aliases, and credentials must never be
serialized into a ColdSnap request or artifact.

## Invocation

The manager starts its provider before invoking `coldsnap`, keeps it alive until
the process exits, and supplies:

```text
COLDSNAP_HOST_PROVIDER_SOCKET=/absolute/private/provider.sock
COLDSNAP_HOST_PROVIDER_TOKEN=<operation-scoped-random-token>
```

ColdSnap forwards the environment to its engine adapter. All controller-side
engine executables use the shared `internal/adaptercli` entrypoint, which
performs the capability handshake before constructing engine-specific code or
allowing any host mutation. The session identity is the
operation request `id`; the provider must authorize only the hosts named by
that request's launch units. The socket directory should be controller-private
and the socket mode `0600`.

This provider is required. ColdSnap has no built-in SSH, agent, or alternate
transport selection; a manager may use any substrate behind this local
protocol without exposing its transport details to ColdSnap.

## Wire contract

Protocol format 1 uses one newline-terminated JSON request and response per
Unix-socket connection. Each request contains `format`, a unique RPC `id`, the
operation `session`, the bearer `token`, and `operation`. Responses echo
`format` and `id`, and contain `ok`. Provider failures use `ok: false` and an
`error`; a remotely executed command uses `ok: true` and reports its process
`exit_code` separately.

The capability handshake is `operation: capabilities`. Format 1 defines:

| Capability | Manager operation |
| --- | --- |
| `exec` | Execute exact argv on one authorized host, with optional binary stdin and combined output. |
| `upload` | Copy controller-local paths to one authorized host. |
| `runtime-v1` | Typed image/workload operations using the manager's runtime backend. |
| `oci-pull` / `oci-push` | Use controller-held registry authentication against the selected host's image store. |
| `huggingface-publish` | Publish one host-local native payload using manager-held Hugging Face authentication. |
| `huggingface-resolve` | Resolve the published mutable revision to its immutable commit. |

Binary `input`, `output`, and `error_output` use standard JSON base64 encoding.
Arguments remain a JSON string array through the manager boundary; the
transport implementation owns any final shell quoting needed by its substrate.
Format 1 messages are bounded at 64 MiB because bulk checkpoint and model data
move through files, registries, or object stores rather than this control
channel.

## Lifetime and failure behavior

Provider calls may be concurrent. Cancelling the ColdSnap process must close
the provider and terminate provider-owned child sessions so long-running image
or file operations are not orphaned. The provider must fail closed for an
unknown capability, mismatched token/session, unrecognized host, malformed
binary encoding, unknown JSON field, trailing message, oversized message,
non-private socket, or response identity mismatch. Capabilities are
operation-scoped: engine adapters require `exec` and `runtime-v1`; publication adds
only the registry or Hugging Face capabilities it actually consumes.

Registry and Hugging Face credentials stay with the manager. Its authenticated
helpers must receive them through private, operation-scoped channels. These credentials must not
enter ColdSnap child environments, workload metadata, or persistent GPU-host
configuration. See the [reference credential handling](sparkrun-integration.md#credentials-and-host-storage).

## Typed runtime boundary

The outer authenticated envelope carries `operation: runtime` and a nested
`runtime` object with an `action`. `internal/hostops/runtime.go` defines the Go
types; [runtime-neutral managers](runtime-neutral-managers.md) lists the actions,
specification requirements, identity mapping, and failure codes. Controller
capabilities advertise `manager-runtime-v1`; a provider lacking `runtime-v1`
is rejected before engine operations. There is no adapter-side Docker fallback.

For startup measurement, `workload-inspect` accepts optional
`include_start_time: true`. The manager then includes the serving container's
actual RFC3339Nano Docker `State.StartedAt` as `runtime.workload.started_at`.
Ordinary inspection omits this field to remain compatible with older strict
decoders. Do not synthesize a start timestamp from the inspection time. Missing
support makes startup timing unavailable; it does not invalidate an otherwise
successful restore. See [startup timing](startup-timing.md).

Runtime responses carry the typed result under `runtime`. Failures may include
`error_code` alongside `error`: `not_found` for image/workload absence,
`path_not_found` for a missing path inside an existing workload, or
`runtime_failed`. Only the path-specific error skips an optional cache seed.
The manager must reject unsupported requirements instead of silently discarding
them. Logical names must resolve across provider sessions for later lifecycle
operations. Registry runtime actions still require `oci-pull` or `oci-push`
authority; runtime access alone does not grant publication permission.

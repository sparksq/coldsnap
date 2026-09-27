<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# ColdSnap core

`coldsnap-core` contains the engine-neutral semantic artifact, hydration,
memory-region, and payload-trust contracts shared by the vLLM and SGLang
integrations. It does not register an engine plugin by itself.

## External checkpoint resources

`coldsnap_core.checkpoint` provides a process-local registry for engine adapters
that must close external handles before a snapshot. Resources implement
`prepare()` and `restore()`, retaining graph-visible storage while rebuilding
only their handles. Registration is weak while active and retained during
suspension. Prepare rolls back completed resources in reverse order if a later
resource fails; rollback or restore failure is terminal. Engines must quiesce
requests before preparation and resume resources before admitting inference.

The registry rejects remaining io_uring descriptors and kernel workers. A
reader adapter must retire both its ring and its submitting thread when needed;
closing an FD alone is not a sufficient checkpoint contract. The vLLM B12x
adapter is the first implementation, for disk-backed PLE/Engram row caches.

<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# ColdSnap for vLLM

`coldsnap-vllm` contains the flat modules and entry point loaded by vLLM runtime
images. Its version follows the `coldsnap-vllm` coordinate in `versions.yaml`,
independently of the controller release. Shared `coldsnap_core` is installed separately
at the same runtime path used by existing artifact identities.

`pyproject.toml` is the sole source for distribution name, version, summary,
and vLLM entry-point metadata. `render_runtime_metadata.py` materializes the
minimal `.dist-info` directory needed by source-assembled runtime images and
exact runtime bundles; generated metadata is not checked in here.

## DSpark adaptive verification with deferred graphs

When async graphs are enabled, DSpark must have calibrated verification costs
before its first eager request. ColdSnap observes successful vLLM graph-replay
calibration, including synchronous captures, and writes bounded JSON curves to
`$VLLM_CACHE_ROOT/adaptive-calibration/`. The ordinary runtime cache seed carries
these records into activations; no separate cache transfer is needed.

Reuse requires the captured source runtime image digest, model and draft
revision, startup fingerprint, rank/topology, GPU and host driver, graph sizes,
request limits, speculative settings, and profiling context to match. Every
worker must admit a valid record and agree on the curve payload. A missing,
corrupt, incompatible, or inconsistent record makes all workers run vLLM's normal
synchronous graph capture and calibration. Cache write failures do not invalidate
successful calibration.

On a hit, vLLM rebuilds its cost tables from the validated curves before eager
inference. The existing scheduler-idle capture then records real graphs and
refreshes the calibration. Shape-only eager warmup preserves these tables rather
than deriving costs from forwards without graph replay. Worker graph status
includes `adaptive_calibration` with the admission decision, key, curve checksum,
and persistence result. No model-weight hashing is added by this cache.

The default cache root is `/var/cache/coldsnap/runtime/vllm`, and each record is
named `<identity-sha256>.json`. These files are derived runtime cache data,
separate from model weights and residual tensor blobs. Capture embeds them in
the OCI capsule; their contents contribute to the capsule digest.

For vLLM with the n580 snapshot driver, `sparkrun coldsnap materialize` captures
and verifies target-local runtime state when there is no matching local overlay.
That capture packages newly measured curves alongside local residuals while
retaining the portable artifact's weight-provider references. Target selection
includes the actual NVIDIA driver and rank-ordered hosts; the `n580` snapshot
driver name alone does not establish calibration compatibility. An existing
matching overlay is reused, without a separate calibration refresh step.

An ordinary restore can generate fresh curves after a cache miss, but those
writes remain in its container layer and do not update the persistent capsule.
The worker's `saved` status confirms a cache-file write, not overlay promotion.

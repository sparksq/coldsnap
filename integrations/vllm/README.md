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

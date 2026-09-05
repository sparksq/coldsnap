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

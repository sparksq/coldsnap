<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Runtime assets

This directory contains source files that are installed into ColdSnap-enabled
engine images or staged into exact runtime bundles.

- `shared/` contains the dependency-light checkpoint capability probe and
  coordinator client shared by the engine runtimes.
- `engine/` contains shared vLLM/SGLang rank launch, command wrapping, and
  CUDA/CRIU process-checkpoint helpers. Explicitly vLLM-named files there are
  standalone vLLM qualification targets, not shared adapter entrypoints.

Repository ownership does not change installed paths. Image and runtime-bundle
assembly continue to place these helpers at their established container and
host-runtime targets.

These files are operational runtime inputs. Maintained qualification harnesses
belong under `benchmarks/harnesses/`; results and historical experiments are
stored outside the public repository.

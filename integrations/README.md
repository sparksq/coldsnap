<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Engine integrations

- `core/` owns the engine-neutral `coldsnap-core` distribution and
  `coldsnap_core` package shared by engine integrations.
- `vllm/` owns the `coldsnap-vllm` distribution, flat runtime modules, and
  distribution metadata used by vLLM plugin discovery.
- `sglang/` owns the `coldsnap-sglang` distribution and plugin package.

The root `pyproject.toml` is a non-publishable uv workspace, not an engine
plugin. Each integration has independent package metadata. `versions.yaml`
owns the release coordinates: core follows `coldsnap`, while vLLM and SGLang
use their respective plugin coordinates. Both plugins pin the matching core
distribution. See [distribution](../docs/distribution.md).

The vLLM image copies shared core into
`/opt/coldsnap/plugin/coldsnap_core`. That preserves the runtime paths and
source-digest names bound by existing artifacts even though repository source
ownership is no longer nested under the vLLM integration.

Runtime-independent controller code remains in Go under `cmd/` and `internal/`.

Build all three Python distributions into `build/python-dist/` with
`make python-package-build`.

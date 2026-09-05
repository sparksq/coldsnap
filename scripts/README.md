<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Repository maintenance scripts

Two wrappers run the SHA-pinned `scitrera-repo-tools` release automation
declared in `versions.yaml`:

- `update-versions.py` synchronizes version declarations;
- `generate-ci-gha.py --check` checks version/Go workflow drift; pass `--force`
  to regenerate. Docker bundle and NCCL publication workflows are repository-owned.

Repository-owned checks and provider tools are:

- `check-repository-layout.py` enforces the documented root ownership and keeps
  generated outputs, archived history, benchmark results, and Python bytecode
  out of version control;
- `check-markdown-links.py` verifies that repository-local documentation links
  remain valid when files are reorganized;
- `check-license-headers.py` enforces the first-party SPDX header policy while
  excluding legal instruments, generated data, and verbatim third-party texts;
- `nccl-payload-release-plan.py` selects locked provider payloads for publication;
- `nccl-target-observe.py` records the target image's NCCL/runtime identity;
- `nccl-provider-runtime-probe.py` checks a built provider's runtime contract;
- `nccl-nvcc-reproducible.sh` supplies deterministic per-output NVCC seeds.

The two wrappers prefer an installed tool package and otherwise use `uv` to execute the
pinned source. They are repository maintenance tools, not ColdSnap runtime
inputs. The two scitrera-repo-tools-derived wrappers retain BSD-3-Clause
licensing; its verbatim license is stored at
`third_party/licenses/scitrera-repo-tools/LICENSE`. The remaining maintenance
scripts are ColdSnap source under AGPL-3.0-only.

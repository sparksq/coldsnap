<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Third-party license texts

This directory retains verbatim license texts for third-party binaries that
ColdSnap copies into derived runtime images or NCCL provider packages. These
files remain under the named third parties' licenses; the ColdSnap AGPL license
does not replace or relicense them.

- `licenses/criu/COPYING` is CRIU's GPL-2.0-only/LGPL-2.1-only license notice.
- `licenses/go-criu/LICENSE` is the Apache-2.0 license from the pinned
  `sparksq/go-criu` fork and is included in controller binary bundles.
- `licenses/nccl/LICENSE.txt` is NVIDIA NCCL's BSD-3-Clause license.
- `licenses/scitrera-repo-tools/LICENSE` retains attribution for the two
  repository-maintenance wrappers derived from that project.

Runtime images copy the go-criu and cuda-checkpoint licenses directly from
their pinned named Docker build contexts, so the exact source used for each
build supplies the corresponding license bytes.

<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Third-party notices

ColdSnap's own source is licensed under AGPL-3.0-only. Runtime images and
capsules are aggregates that also contain independently licensed components;
the ColdSnap license does not replace or relicense those components. Derived
images install the corresponding texts below
`/opt/coldsnap/licenses/third-party`, separately from
`/opt/coldsnap/licenses/coldsnap`.

## CRIU

ColdSnap runtime images copy the CRIU executable from the separately built
[`sparksq/criu`](https://github.com/sparksq/criu) OCI image. CRIU is
GPL-2.0-only, except for its `lib/` directory, which is LGPL-2.1-only. The
verbatim upstream notice and license text is retained at
`third_party/licenses/criu/COPYING` and is installed as
`/opt/coldsnap/licenses/third-party/criu/COPYING`.

CRIU remains a separate executable driven through its RPC interface; this
repository does not claim CRIU under the ColdSnap AGPL license.

The n580 NVIDIA reset plugin is a true in-process CRIU plugin: it includes
CRIU's GPL plugin header and is loaded into CRIU. Its source and build therefore
live in the public `sparksq/criu` fork under GPL-2.0-only. ColdSnap runtime
images copy that separately licensed plugin from the digest-pinned CRIU OCI
image; this repository neither builds it from CRIU headers nor presents it as
AGPL-licensed ColdSnap source. Its copyright remains with Scitrera LLC and Fox
Engine Ltd.

## go-criu

The `coldsnap-criu-rpc` binary uses the Apache-2.0
[`sparksq/go-criu`](https://github.com/sparksq/go-criu) fork. Builds resolve
the public fork at an immutable revision. Runtime images copy the exact
checkout's `LICENSE` to
`/opt/coldsnap/licenses/third-party/go-criu/LICENSE`.
Controller binary-bundle images carry the same license text at
`/usr/share/licenses/coldsnap/third-party/go-criu/LICENSE`.

## NVIDIA cuda-checkpoint

Runtime images copy NVIDIA's `cuda-checkpoint` executable from a pinned named
build context. The exact source checkout's `LICENSE` is installed separately
at `/opt/coldsnap/licenses/third-party/cuda-checkpoint/LICENSE`.

## NVIDIA NCCL

ColdSnap's versioned NCCL providers patch and rebuild NVIDIA NCCL, which is
BSD-3-Clause. The upstream license is retained at
`third_party/licenses/nccl/LICENSE.txt` and embedded as a separately declared,
manifest-verified provider file at
`licenses/third-party/nccl/LICENSE.txt`.

## torch_memory_saver

The CUDA virtual-memory allocation/remap mechanics in
`native/coldsnap_graph_memory.cpp` are independently adapted from
torch_memory_saver.

MIT License

Copyright (c) 2024 fzyzcjy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## InstantTensor

The aligned pinned-host staging ring, bounded asynchronous read/copy pipeline,
and runtime cuFile binding in `native/coldsnap_hydration.cpp` are independently
adapted from InstantTensor:

https://github.com/scitix/InstantTensor

Reference revision: f5a445ef2ab03c0ee92e6bf2862d7ba54ea03f33

Copyright 2026 Yitao Yuan

Licensed under the Apache License, Version 2.0. A copy of the license is
available at https://www.apache.org/licenses/LICENSE-2.0.

<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Native components

This directory contains native code compiled into ColdSnap runtime images or
loaded by engine integrations:

- `coldsnap_hydration.*`: native capture and file-to-GPU hydration transports;
- `coldsnap_cuda_epoch.*`: CUDA allocation epoch and virtual-memory support;
- `coldsnap_graph_memory.*`: stable-address CUDA graph memory support;
- `coldsnap_nccl_dlsym.c`: NCCL symbol-routing bridge used by checkpointed
  InstantTensor communication paths;
- `nccl_checkpoint_coord/`: native client for the ColdSnap coordinator
  protocol used by the NCCL checkpoint runtime.
- `nccl/`: provider ABI and release-owned NCCL sources, locked patches, and
  capability-admission records;
- `coldsnap_nvidia_mmap_shim.c` and `coldsnap_nvml_dlopen_shim.c`: narrow
  NVIDIA device-mapping and NVML loading compatibility helpers.

Local `make native` output belongs under `build/native/`, including the NCCL
bridge. Runtime image builds compile these sources in container build stages;
generated libraries do not belong in this directory.

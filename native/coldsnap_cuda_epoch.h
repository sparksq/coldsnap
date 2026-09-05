// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One sleeping vLLM CuMem allocation. handle_box_address is the host pointer
// stored in vLLM's fourth HandleType field; the native allocator allocated the
// box once and its regular free callback continues to own it.
struct coldsnap_cuda_epoch_allocation {
    uintptr_t address;
    uint64_t mapped_bytes;
    uintptr_t handle_box_address;
    int device;
};

// Diagnostic-only result for a post-reset CUDA context capability probe.
// Every operation result is the numeric CUresult/cudaError_t value returned by
// the corresponding API. COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN distinguishes an
// operation that was deliberately skipped from CUDA_SUCCESS/cudaSuccess.
#define COLDSNAP_CUDA_EPOCH_CONTEXT_PROBE_ABI 1U
#define COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN INT32_MIN

struct coldsnap_cuda_epoch_context_probe {
    uint32_t abi_version;
    uint32_t struct_size;
    int device;

    int cu_init;
    int cu_device_get;
    int cu_ctx_get_current;
    int current_context_present;
    int cu_ctx_clear_current;

    int cu_primary_get_state;
    unsigned int primary_flags;
    int primary_active;
    int cu_primary_retain;
    int cu_primary_release;

    int cu_private_create;
    int cu_private_get_current;
    int private_context_matched;
    int cu_private_alloc;
    int cu_private_memset;
    int cu_private_copy_to_host;
    int cu_private_synchronize;
    int cu_private_free;
    int driver_payload_verified;

    int runtime_get_device;
    int runtime_device;
    int runtime_malloc;
    int runtime_memset;
    int runtime_copy_to_host;
    int runtime_synchronize;
    int runtime_free;
    int runtime_payload_verified;

    int cu_private_destroy;
    int cu_ctx_clear_after_probe;
};

// Reset a quiesced device primary context. All CUDA data must already have
// been externalized and every managed VMM allocation must be asleep.
int coldsnap_cuda_epoch_reset_device(int device);

// Probe whether a new primary context can be retained after reset. Only when
// primary retention fails, try a disposable private driver context and tiny
// (4 KiB) driver/runtime allocations. This never changes rebind policy and
// never publishes the private context to vLLM.
int coldsnap_cuda_epoch_probe_context(
    int device,
    struct coldsnap_cuda_epoch_context_probe* result);

// Stable names for numeric results recorded by the context probe.
const char* coldsnap_cuda_epoch_driver_result_name(int result);
const char* coldsnap_cuda_epoch_runtime_result_name(int result);

// Establish a fresh primary context, map new physical allocations into the VMM
// address reservations retained by vLLM sleep, and write the new handles into
// vLLM's existing host-side handle boxes. This call is transactional: a
// failure removes every physical mapping and handle it created.
int coldsnap_cuda_epoch_rebind(
    const struct coldsnap_cuda_epoch_allocation* allocations,
    size_t allocation_count);

// Select the fresh primary context published by the last successful rebind on
// the calling thread. This does not retain another reference to the context.
int coldsnap_cuda_epoch_activate_device(int device);

// Minimal hydration operations that deliberately stay on the fresh driver
// context. CUDA-runtime pinned-pointer and event caches belong to the destroyed
// epoch and cannot be consulted while the plugin is rebuilding memory.
int coldsnap_cuda_epoch_copy_from_host(
    uintptr_t destination,
    uintptr_t source,
    uint64_t bytes);
int coldsnap_cuda_epoch_fill_zero(uintptr_t destination, uint64_t bytes);
int coldsnap_cuda_epoch_synchronize(void);

// Thread-local diagnostic for the last failed epoch operation.
const char* coldsnap_cuda_epoch_last_error(void);

#ifdef __cplusplus
}
#endif

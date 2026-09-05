// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum coldsnap_graph_allocation_state {
    COLDSNAP_GRAPH_ALLOCATION_ACTIVE = 0,
    COLDSNAP_GRAPH_ALLOCATION_PAUSED = 1,
    COLDSNAP_GRAPH_ALLOCATION_PARTIAL = 2,
};

// Scope CUDA runtime allocations on the calling thread into one semantic
// graph-pool region. device < 0 accepts the current CUDA device.
int coldsnap_graph_region_push(const char* tag, int device, uint64_t pool_id);
int coldsnap_graph_region_pop(void);

// Release and restore replaceable physical backing while retaining virtual
// address reservations. Operations are idempotent and retryable after failure.
int coldsnap_graph_pause(const char* tag, int device);
int coldsnap_graph_resume(const char* tag, int device);

// Inspect provider state without exposing internal allocation objects.
int coldsnap_graph_region_stats(
    const char* tag,
    int device,
    uint64_t* allocation_count,
    uint64_t* raw_bytes,
    uint64_t* mapped_bytes,
    uint64_t* paused_count);

int coldsnap_graph_allocation_at(
    const char* tag,
    int device,
    uint64_t index,
    uintptr_t* address,
    uint64_t* raw_bytes,
    uint64_t* mapped_bytes,
    int* allocation_device,
    uint64_t* pool_id,
    int* state);

// True only when this library owns the process-wide cudaMalloc resolution.
// Loading it later with dlopen is insufficient for CUDA graph interception.
int coldsnap_graph_interposition_active(void);

// Thread-local diagnostic for the last failed native operation.
const char* coldsnap_graph_last_error(void);

#ifdef __cplusplus
}
#endif

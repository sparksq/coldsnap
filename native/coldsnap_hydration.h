// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLDSNAP_HYDRATION_ABI_VERSION 1U
#define COLDSNAP_CAPTURE_ABI_VERSION 1U

enum coldsnap_hydration_backend {
    COLDSNAP_HYDRATION_BUFFERED = 1,
    COLDSNAP_HYDRATION_DIRECT = 2,
    COLDSNAP_HYDRATION_GDS = 3,
};

enum coldsnap_hydration_verification {
    COLDSNAP_HYDRATION_VERIFY_NONE = 0,
    COLDSNAP_HYDRATION_VERIFY_CRC32 = 1,
    COLDSNAP_HYDRATION_VERIFY_CRC32_SHA256 = 2,
};

enum coldsnap_hydration_flags {
    COLDSNAP_HYDRATION_REGISTER_DEVICE_BUFFERS = 1U << 0,
};

struct coldsnap_hydration_extent {
    uint64_t file_offset;
    uintptr_t destination;
    uint64_t length;
    uint32_t verification;
    uint32_t expected_crc32;
    uint8_t expected_sha256[32];
};

struct coldsnap_hydration_options {
    uint32_t abi_version;
    uint32_t backend;
    uint32_t queue_depth;
    uint32_t flags;
    uint64_t chunk_bytes;
    int32_t cuda_device;
    uint32_t reserved;
};

struct coldsnap_hydration_result {
    uint32_t abi_version;
    uint32_t backend;
    uint64_t bytes;
    uint64_t chunks;
    uint64_t verified_extents;
    uint64_t initialization_ns;
    uint64_t io_service_ns;
    uint64_t io_wait_ns;
    uint64_t checksum_ns;
    uint64_t cuda_enqueue_ns;
    uint64_t cuda_synchronize_ns;
    uint64_t total_ns;
};

enum coldsnap_capture_backend {
    COLDSNAP_CAPTURE_AUTO = 0,
    COLDSNAP_CAPTURE_BUFFERED = 1,
    COLDSNAP_CAPTURE_DIRECT = 2,
};

enum coldsnap_capture_checksum {
    COLDSNAP_CAPTURE_CHECKSUM_NONE = 0,
    COLDSNAP_CAPTURE_CHECKSUM_CRC32 = 1,
    COLDSNAP_CAPTURE_CHECKSUM_CRC32_SHA256 = 2,
};

enum coldsnap_capture_flags {
    COLDSNAP_CAPTURE_FILE_SHA256 = 1U << 0,
    COLDSNAP_CAPTURE_VERIFY_READBACK = 1U << 1,
};

struct coldsnap_capture_extent {
    uintptr_t source;
    uint64_t file_offset;
    uint64_t length;
    uint32_t checksum;
    uint32_t reserved;
};

struct coldsnap_capture_digest {
    uint32_t crc32;
    uint32_t reserved;
    uint8_t sha256[32];
};

struct coldsnap_capture_options {
    uint32_t abi_version;
    uint32_t backend;
    uint32_t queue_depth;
    uint32_t flags;
    uint64_t chunk_bytes;
    int32_t cuda_device;
    uint32_t reserved;
};

struct coldsnap_capture_result {
    uint32_t abi_version;
    uint32_t backend;
    uint64_t bytes;
    uint64_t file_bytes;
    uint64_t chunks;
    uint64_t checksummed_extents;
    uint64_t initialization_ns;
    uint64_t io_service_ns;
    uint64_t io_wait_ns;
    uint64_t checksum_ns;
    uint64_t cuda_enqueue_ns;
    uint64_t cuda_synchronize_ns;
    uint64_t durability_ns;
    uint64_t verification_ns;
    uint64_t total_ns;
    uint8_t file_sha256[32];
};

// Hydrate CUDA destinations from one immutable file. Extents are processed in
// caller order. Staged backends can verify bytes inline; GDS requires a
// separately established trust boundary and therefore accepts VERIFY_NONE.
int coldsnap_hydrate_file(
    const char* path,
    const struct coldsnap_hydration_extent* extents,
    size_t extent_count,
    const struct coldsnap_hydration_options* options,
    struct coldsnap_hydration_result* result);

// Capture CUDA sources into one durable file. Extents must be ordered and
// non-overlapping in file layout. The staged pipeline returns the requested
// per-extent digests without rereading device memory; optional readback
// verifies the durable file before it is admitted by the caller.
int coldsnap_capture_file(
    const char* path,
    const struct coldsnap_capture_extent* extents,
    size_t extent_count,
    const struct coldsnap_capture_options* options,
    struct coldsnap_capture_digest* digests,
    struct coldsnap_capture_result* result);

// This is a best-effort dependency probe. A successful GDS probe does not
// guarantee that the filesystem, device, or target allocation is compatible.
int coldsnap_hydration_backend_available(uint32_t backend);
int coldsnap_capture_backend_available(uint32_t backend);

const char* coldsnap_hydration_last_error(void);

#ifdef __cplusplus
}
#endif

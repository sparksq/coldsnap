// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Preserve the source-line identities used by the qualified NCCL provider.
#line 1
#ifndef COLDSNAP_NCCL_PROVIDER_H_
#define COLDSNAP_NCCL_PROVIDER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLDSNAP_NCCL_PROVIDER_ABI_MAJOR 1u
#define COLDSNAP_NCCL_PROVIDER_ABI_MINOR 0u

/* Keep capability bit assignments stable across provider releases. */
enum coldsnap_nccl_provider_capability {
  COLDSNAP_NCCL_CAP_FULL_NETWORK_RESET = UINT64_C(1) << 0,
  COLDSNAP_NCCL_CAP_IB_ROCE_DEVICE_RELEASE = UINT64_C(1) << 1,
  COLDSNAP_NCCL_CAP_RAS_RESET = UINT64_C(1) << 2,
  COLDSNAP_NCCL_CAP_SYNCHRONOUS_TERMINATION = UINT64_C(1) << 3,
  COLDSNAP_NCCL_CAP_COORDINATOR_PROTOCOL_V1 = UINT64_C(1) << 4,
  COLDSNAP_NCCL_CAP_PRIVATE_DLOPEN_ROUTING = UINT64_C(1) << 5,
  COLDSNAP_NCCL_CAP_COMMUNICATOR_UNWRAP = UINT64_C(1) << 6,
  COLDSNAP_NCCL_CAP_REGISTRATION_WINDOW_REPLAY = UINT64_C(1) << 7,
  COLDSNAP_NCCL_CAP_SPLIT_SHRINK_GROW_REPLAY = UINT64_C(1) << 8,
  COLDSNAP_NCCL_CAP_CUDA_GRAPH = UINT64_C(1) << 9,
  COLDSNAP_NCCL_CAP_NCCL_DEVICE_API = UINT64_C(1) << 10,
  COLDSNAP_NCCL_CAP_COMMUNICATOR_SUSPEND_IN_PLACE = UINT64_C(1) << 11,
  COLDSNAP_NCCL_CAP_TRANSPORT_DETACH_IN_PLACE = UINT64_C(1) << 12,
  COLDSNAP_NCCL_CAP_GRAPH_RESOURCE_RETENTION = UINT64_C(1) << 13,
  COLDSNAP_NCCL_CAP_REGISTRATION_WINDOW_IN_PLACE_REPLAY = UINT64_C(1) << 14,
  COLDSNAP_NCCL_CAP_NVLS_IN_PLACE_REPLAY = UINT64_C(1) << 15,
  COLDSNAP_NCCL_CAP_DEVICE_API_STATE_RETENTION = UINT64_C(1) << 16,
};

/* Keep the qualified provider_v1 declarations at their original debug lines. */
#line 27

typedef int32_t (*coldsnap_nccl_provider_operation_fn)(void);
typedef int32_t (*coldsnap_nccl_provider_ib_status_fn)(
    int32_t* net_refs, int32_t* devices, int32_t* pd_refs, int32_t* mr_count);
typedef int32_t (*coldsnap_nccl_provider_comm_get_real_fn)(
    void* synthetic_comm, void** real_comm);

/*
 * A query returns a pointer to provider-owned immutable storage. Consumers must
 * validate struct_size before reading optional tail fields and must reject an
 * ABI-major mismatch. Adding fields to the tail is ABI-minor compatible.
 */
struct coldsnap_nccl_provider_v1 {
  uint32_t struct_size;
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t provider_revision;
  int32_t compiled_nccl_version;
  int32_t loaded_nccl_version;
  uint32_t checkpoint_abi_version;
  uint32_t reserved0;
  uint64_t capability_mask;
  const char* provider_id;
  coldsnap_nccl_provider_operation_fn prepare;
  coldsnap_nccl_provider_operation_fn restore;
  coldsnap_nccl_provider_operation_fn network_reset;
  coldsnap_nccl_provider_operation_fn network_init;
  coldsnap_nccl_provider_ib_status_fn ib_status;
  coldsnap_nccl_provider_comm_get_real_fn comm_get_real;
};

/* Returns the provider runtime's ncclResult_t value (zero is success). */
int32_t coldsnapNcclProviderQuery(
    uint32_t requested_abi_major,
    const struct coldsnap_nccl_provider_v1** provider);

/*
 * Optional research ABI for a provider that retains graph-visible
 * communicator identity. It is intentionally separate from provider_v1 so
 * current destroy/recreate providers do not acquire stronger semantics merely
 * by rebuilding against a newer header. Evidence is immutable provider-owned
 * UTF-8 JSON and must describe communicator/device identities and retained
 * transport/registration ownership before promotion.
 */
#define COLDSNAP_NCCL_IN_PLACE_ABI_MAJOR 1u
#define COLDSNAP_NCCL_IN_PLACE_ABI_MINOR 0u

typedef const char* (*coldsnap_nccl_provider_evidence_fn)(void);

struct coldsnap_nccl_in_place_v1 {
  uint32_t struct_size;
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t reserved0;
  coldsnap_nccl_provider_operation_fn communicator_suspend;
  coldsnap_nccl_provider_operation_fn transport_detach;
  coldsnap_nccl_provider_operation_fn transport_reattach;
  coldsnap_nccl_provider_operation_fn communicator_resume;
  coldsnap_nccl_provider_evidence_fn evidence_json;
};

/* Optional export; absence means the exact provider has no in-place mode. */
int32_t coldsnapNcclInPlaceQuery(
    uint32_t requested_abi_major,
    const struct coldsnap_nccl_in_place_v1** provider);

/* Keep the existing C linkage trailer at its original debug line. */
#line 69

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* COLDSNAP_NCCL_PROVIDER_H_ */

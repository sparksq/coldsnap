// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Experimental provider_v1 identity for the exact NCCL in-place transport
// implementation. The stronger lifecycle remains available only through the
// separate coldsnapNcclInPlaceQuery ABI.
#include "coldsnap_nccl_provider.h"
#include "shim_core.h"

#include <mutex>

using namespace nccl_checkpoint;

extern "C" ncclResult_t ncclCheckpointNetworkReset(void);
extern "C" ncclResult_t ncclCheckpointNetworkInit(void);
extern "C" ncclResult_t ncclCheckpointIbGetStatus(
    int* net_refs, int* devices, int* pd_refs, int* mr_count);
extern "C" ncclResult_t ncclCheckpointCommGetReal(
    ncclComm_t synthetic_comm, ncclComm_t* real_comm);

namespace {

int32_t providerPrepare() { return ncclCheckpointPrepare(); }
int32_t providerRestore() { return ncclCheckpointRestore(); }
int32_t providerNetworkReset() { return ncclCheckpointNetworkReset(); }
int32_t providerNetworkInit() { return ncclCheckpointNetworkInit(); }

int32_t providerIbStatus(
    int32_t* net_refs, int32_t* devices, int32_t* pd_refs, int32_t* mr_count) {
  static_assert(sizeof(int32_t) == sizeof(int));
  return ncclCheckpointIbGetStatus(
      reinterpret_cast<int*>(net_refs), reinterpret_cast<int*>(devices),
      reinterpret_cast<int*>(pd_refs), reinterpret_cast<int*>(mr_count));
}

int32_t providerCommGetReal(void* synthetic_comm, void** real_comm) {
  return ncclCheckpointCommGetReal(
      reinterpret_cast<ncclComm_t>(synthetic_comm),
      reinterpret_cast<ncclComm_t*>(real_comm));
}

constexpr uint64_t kProviderCapabilities =
    COLDSNAP_NCCL_CAP_FULL_NETWORK_RESET |
    COLDSNAP_NCCL_CAP_IB_ROCE_DEVICE_RELEASE |
    COLDSNAP_NCCL_CAP_RAS_RESET |
    COLDSNAP_NCCL_CAP_SYNCHRONOUS_TERMINATION |
    COLDSNAP_NCCL_CAP_COORDINATOR_PROTOCOL_V1 |
    COLDSNAP_NCCL_CAP_PRIVATE_DLOPEN_ROUTING |
    COLDSNAP_NCCL_CAP_COMMUNICATOR_UNWRAP |
    COLDSNAP_NCCL_CAP_REGISTRATION_WINDOW_REPLAY |
    COLDSNAP_NCCL_CAP_SPLIT_SHRINK_GROW_REPLAY |
    COLDSNAP_NCCL_CAP_CUDA_GRAPH |
    COLDSNAP_NCCL_CAP_COMMUNICATOR_SUSPEND_IN_PLACE |
    COLDSNAP_NCCL_CAP_TRANSPORT_DETACH_IN_PLACE |
    COLDSNAP_NCCL_CAP_GRAPH_RESOURCE_RETENTION;

struct coldsnap_nccl_provider_v1 g_provider = {
    sizeof(struct coldsnap_nccl_provider_v1),
    COLDSNAP_NCCL_PROVIDER_ABI_MAJOR,
    COLDSNAP_NCCL_PROVIDER_ABI_MINOR,
    2,
    NCCL_VERSION_CODE,
    0,
    NCCL_CHECKPOINT_VERSION_CODE,
    0,
    kProviderCapabilities,
    "nccl-2.32.3-1+coldsnap.2",
    providerPrepare,
    providerRestore,
    providerNetworkReset,
    providerNetworkInit,
    providerIbStatus,
    providerCommGetReal,
};

std::once_flag g_provider_init_once;
ncclResult_t g_provider_init_result = ncclInternalError;

void initializeProviderIdentity() {
  using get_version_t = ncclResult_t (*)(int*);
  static get_version_t real_get_version = nullptr;
  g_provider_init_result = resolveRealFunction("ncclGetVersion", &real_get_version);
  if (g_provider_init_result != ncclSuccess) return;
  g_provider_init_result = real_get_version(&g_provider.loaded_nccl_version);
  if (g_provider_init_result != ncclSuccess) return;
  if (g_provider.loaded_nccl_version != g_provider.compiled_nccl_version) {
    WARN("provider compiled NCCL version %d does not match loaded version %d",
         g_provider.compiled_nccl_version, g_provider.loaded_nccl_version);
    g_provider_init_result = ncclInvalidUsage;
    return;
  }
  using crypt_status_t = ncclResult_t (*)(int*, size_t*, int*, int*);
  crypt_status_t crypt_status = nullptr;
  g_provider_init_result = resolveRealFunction("ncclCheckpointCryptStatus", &crypt_status);
  if (g_provider_init_result != ncclSuccess) return;
  int compiled, ready, reseed;
  size_t active;
  g_provider_init_result = crypt_status(&compiled, &active, &ready, &reseed);
  if (g_provider_init_result != ncclSuccess) return;
#if defined(NCCL_TLS_BACKEND_OPENSSL3)
  constexpr int expected_tls = 1;
#else
  constexpr int expected_tls = 0;
#endif
  if (compiled != expected_tls) {
    WARN("provider TLS backend does not match loaded NCCL runtime");
    g_provider_init_result = ncclInvalidUsage;
  }
}

}  // namespace

extern "C" int32_t coldsnapNcclProviderQuery(
    uint32_t requested_abi_major,
    const struct coldsnap_nccl_provider_v1** provider) {
  if (provider == nullptr) return ncclInvalidArgument;
  *provider = nullptr;
  if (requested_abi_major != COLDSNAP_NCCL_PROVIDER_ABI_MAJOR) {
    return ncclInvalidArgument;
  }
  std::call_once(g_provider_init_once, initializeProviderIdentity);
  if (g_provider_init_result != ncclSuccess) return g_provider_init_result;
  *provider = &g_provider;
  return ncclSuccess;
}

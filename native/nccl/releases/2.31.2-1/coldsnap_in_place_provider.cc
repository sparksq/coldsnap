// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Exact-NCCL NET transport reattachment experiment. Graph-visible
// communicator allocations remain stable while the matching NCCL build closes
// and recreates its transport epoch through the ColdSnap coordinator.

#include "coldsnap_nccl_provider.h"
#include "shim_core.h"
#include "kv_store_client.h"
#include "net.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

using namespace nccl_checkpoint;

namespace {

enum class InPlaceState {
  active,
  suspended,
  transport_detached,
  transport_reattached,
};

std::mutex g_mutex;
InPlaceState g_state = InPlaceState::active;
std::string g_evidence;
uint64_t g_generation = 0;
int g_communicators = 0;
uint64_t g_persistent_refs = 0;
int g_receive_endpoints = 0;
int g_send_endpoints = 0;
std::string g_transport;

struct coldsnapNcclNetEndpoint {
  uint32_t structSize;
  uint32_t version;
  int32_t senderRank;
  int32_t receiverRank;
  int32_t channelId;
  int32_t connIndex;
  ncclNetHandle_t handle;
};

using net_detach_t = ncclResult_t (*)(ncclComm_t);
using net_reset_globals_t = ncclResult_t (*)();
using net_reinitialize_t = ncclResult_t (*)(ncclComm_t);
using net_endpoints_t = ncclResult_t (*)(ncclComm_t, coldsnapNcclNetEndpoint*, int, int*);
using net_reattach_t = ncclResult_t (*)(ncclComm_t, coldsnapNcclNetEndpoint*, int);
using control_lifecycle_t = ncclResult_t (*)(ncclComm_t);

net_detach_t g_net_detach = nullptr;
net_reset_globals_t g_net_reset_globals = nullptr;
net_reinitialize_t g_net_reinitialize = nullptr;
net_endpoints_t g_net_recv_endpoints = nullptr;
net_endpoints_t g_net_send_endpoints = nullptr;
net_reattach_t g_net_reattach = nullptr;
control_lifecycle_t g_control_detach = nullptr;
control_lifecycle_t g_proxy_reattach = nullptr;

const char* stateString(InPlaceState state) {
  switch (state) {
    case InPlaceState::active:
      return "active";
    case InPlaceState::suspended:
      return "suspended";
    case InPlaceState::transport_detached:
      return "transport-detached";
    case InPlaceState::transport_reattached:
      return "transport-reattached";
  }
  return "unknown";
}

bool experimentEnabled() {
  const char* mode = std::getenv("COLDSNAP_NCCL_IN_PLACE_MODE");
  return mode != nullptr && std::strcmp(mode, "net-reconnect-v1") == 0;
}

bool transportConfigured() {
  const char* ib_disable = std::getenv("NCCL_IB_DISABLE");
  if (ib_disable != nullptr && std::strcmp(ib_disable, "1") == 0) return true;
  const char* release = std::getenv("NCCL_IB_RELEASE_ON_FINALIZE");
  return release != nullptr && std::strcmp(release, "1") == 0;
}

void updateEvidence() {
  char evidence[1024];
  std::snprintf(
      evidence, sizeof(evidence),
      "{\"format\":1,\"kind\":\"coldsnap-nccl-in-place-evidence\","
      "\"mode\":\"net-reconnect-v1\",\"portable\":false,"
      "\"transport_epoch_portable\":true,\"network_identity_portable\":true,"
      "\"bootstrap_available\":false,\"proxy_control_reconstructed\":true,"
      "\"backend\":\"%s\","
      "\"qualified\":false,\"state\":\"%s\",\"generation\":%llu,"
      "\"communicators\":%d,\"persistent_graph_references\":%llu,"
      "\"receive_endpoints\":%d,\"send_endpoints\":%d,"
      "\"transport_ownership\":\"coldsnap\","
      "\"limitation\":\"experimental exact-NCCL built-in NET transport; socket bootstrap is retired after capture; one communicator and process-local proxy ownership are required\"}",
      g_transport.empty() ? "unknown" : g_transport.c_str(), stateString(g_state),
      static_cast<unsigned long long>(g_generation),
      g_communicators, static_cast<unsigned long long>(g_persistent_refs),
      g_receive_endpoints, g_send_endpoints);
  g_evidence = evidence;
}

ncclResult_t resolveTransportFunctions() {
  NCCLCHECK(resolveRealFunction("ncclCheckpointNetDetach", &g_net_detach));
  NCCLCHECK(resolveRealFunction("ncclCheckpointNetResetGlobals", &g_net_reset_globals));
  NCCLCHECK(resolveRealFunction("ncclCheckpointNetReinitialize", &g_net_reinitialize));
  NCCLCHECK(resolveRealFunction("ncclCheckpointNetRecvEndpoints", &g_net_recv_endpoints));
  NCCLCHECK(resolveRealFunction("ncclCheckpointNetSendEndpoints", &g_net_send_endpoints));
  NCCLCHECK(resolveRealFunction("ncclCheckpointNetReattach", &g_net_reattach));
  NCCLCHECK(resolveRealFunction("ncclCheckpointControlDetach", &g_control_detach));
  NCCLCHECK(resolveRealFunction("ncclCheckpointProxyReattach", &g_proxy_reattach));
  return ncclSuccess;
}

bool validActivation(const std::string& value) {
  if (value.empty() || value.size() > 160) return false;
  for (char byte : value) {
    if ((byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
        (byte >= '0' && byte <= '9') || byte == '-' || byte == '_' || byte == '.') {
      continue;
    }
    return false;
  }
  return true;
}

ncclResult_t activationId(std::string* activation) {
  const char* path = std::getenv("COLDSNAP_NCCL_IN_PLACE_ACTIVATION_PATH");
  if (path != nullptr && *path != '\0') {
    std::ifstream stream(path);
    if (!stream || !std::getline(stream, *activation)) {
      WARN("cannot read ColdSnap in-place activation from %s", path);
      return ncclInvalidArgument;
    }
  } else {
    const char* value = std::getenv("COLDSNAP_NCCL_IN_PLACE_ACTIVATION");
    if (value != nullptr) *activation = value;
  }
  if (!validActivation(*activation)) {
    WARN("ColdSnap in-place transport requires a valid activation id");
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

std::string endpointKey(const char* kind, const coldsnapNcclNetEndpoint& endpoint) {
  char key[192];
  std::snprintf(key, sizeof(key), "%s/%d/%d/%d/%d", kind, endpoint.senderRank,
                endpoint.receiverRank, endpoint.channelId, endpoint.connIndex);
  return key;
}

bool sameEndpoint(const coldsnapNcclNetEndpoint& left, const coldsnapNcclNetEndpoint& right) {
  return left.structSize == right.structSize && left.version == right.version &&
         left.senderRank == right.senderRank && left.receiverRank == right.receiverRank &&
         left.channelId == right.channelId && left.connIndex == right.connIndex;
}

ncclResult_t activeCommunicators(std::vector<ncclComm_t>* active) {
  if (active == nullptr) return ncclInvalidArgument;
  active->clear();
  std::unordered_set<uint64_t> hashes;
  std::unordered_set<ncclProxyState*> proxies;
  NCCLCHECK(g_commHandles.forEachHandle(
      [&](ncclComm_t synthetic, const CommHandleEntry* entry) {
        if (entry->userState != comm_user_active || entry->realHandle == nullptr) {
          return ncclSuccess;
        }
        ncclComm_t real = nullptr;
        NCCLCHECK(g_commHandles.toReal(synthetic, &real));
        if (real == nullptr || real->endMagic != NCCL_MAGIC || real->proxyState == nullptr) {
          return ncclInternalError;
        }
        if (!hashes.insert(real->commHash).second) {
          WARN("in-place NET requires a unique hash for each active communicator");
          return ncclInvalidUsage;
        }
        if (!proxies.insert(real->proxyState).second) {
          WARN("in-place NET does not yet support communicators sharing a proxy state");
          return ncclInvalidUsage;
        }
        active->push_back(real);
        return ncclSuccess;
      }));
  if (active->empty()) {
    WARN("in-place diagnostic has no active communicators");
    return ncclInvalidUsage;
  }
  std::sort(active->begin(), active->end(), [](ncclComm_t left, ncclComm_t right) {
    return left->commHash < right->commHash;
  });
  return ncclSuccess;
}

ncclResult_t inspectCommunicators() {
  using async_error_t = ncclResult_t (*)(ncclComm_t, ncclResult_t*);
  static async_error_t real_async_error = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommGetAsyncError", &real_async_error));

  std::vector<ncclComm_t> active;
  NCCLCHECK(activeCommunicators(&active));
  uint64_t persistent_refs = 0;
  for (ncclComm_t real : active) {
    ncclResult_t async_error = ncclSuccess;
    NCCLCHECK(real_async_error(real, &async_error));
    NCCLCHECK(async_error);
    const char* backend = real->ncclNet == nullptr ? nullptr : real->ncclNet->name;
    if (backend == nullptr ||
        (std::strcmp(backend, "Socket") != 0 && std::strcmp(backend, "IB") != 0)) {
      WARN("in-place NET requires an exact built-in Socket or IB backend");
      return ncclInvalidUsage;
    }
    if (!g_transport.empty() && g_transport != backend) {
      WARN("in-place NET does not support mixed backends in one process");
      return ncclInvalidUsage;
    }
    g_transport = backend;
    persistent_refs += real->localPersistentRefs;
  }
  g_communicators = static_cast<int>(active.size());
  g_persistent_refs = persistent_refs;
  return ncclSuccess;
}

int32_t communicatorSuspend() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!experimentEnabled() || !transportConfigured()) {
    WARN("in-place NET reconnect requires COLDSNAP_NCCL_IN_PLACE_MODE=net-reconnect-v1; "
         "IB additionally requires NCCL_IB_RELEASE_ON_FINALIZE=1");
    return ncclInvalidUsage;
  }
  if (g_state != InPlaceState::active) return ncclInvalidUsage;
  NCCLCHECK(inspectCommunicators());
  g_generation++;
  g_state = InPlaceState::suspended;
  updateEvidence();
  return ncclSuccess;
}

int32_t transportDetach() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_state != InPlaceState::suspended) return ncclInvalidUsage;
  NCCLCHECK(resolveTransportFunctions());
  g_receive_endpoints = 0;
  g_send_endpoints = 0;
  std::vector<ncclComm_t> active;
  NCCLCHECK(activeCommunicators(&active));
  for (ncclComm_t real : active) {
    CUDACHECK(cudaSetDevice(real->cudaDev));
    NCCLCHECK(g_net_detach(real));
    NCCLCHECK(g_control_detach(real));
  }
  NCCLCHECK(g_net_reset_globals());
  g_state = InPlaceState::transport_detached;
  updateEvidence();
  return ncclSuccess;
}

int32_t transportReattach() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_state != InPlaceState::transport_detached) return ncclInvalidUsage;
  NCCLCHECK(resolveTransportFunctions());
  std::string activation;
  NCCLCHECK(activationId(&activation));
  KVStoreClient kv;
  if (!kv.connect_from_env()) return ncclInvalidArgument;

  std::vector<ncclComm_t> active;
  NCCLCHECK(activeCommunicators(&active));
  for (ncclComm_t real : active) {
    CUDACHECK(cudaSetDevice(real->cudaDev));
    NCCLCHECK(g_net_reinitialize(real));
    NCCLCHECK(g_proxy_reattach(real));

    int receiveCount = 0;
    int sendCount = 0;
    NCCLCHECK(g_net_recv_endpoints(real, nullptr, 0, &receiveCount));
    NCCLCHECK(g_net_send_endpoints(real, nullptr, 0, &sendCount));
    std::vector<coldsnapNcclNetEndpoint> receives(receiveCount);
    std::vector<coldsnapNcclNetEndpoint> sends(sendCount);
    NCCLCHECK(g_net_recv_endpoints(real, receives.data(), receiveCount, &receiveCount));
    NCCLCHECK(g_net_send_endpoints(real, sends.data(), sendCount, &sendCount));

    char prefix[320];
    std::snprintf(prefix, sizeof(prefix), "in-place/%s/%016llx", activation.c_str(),
                  static_cast<unsigned long long>(real->commHash));
    kv.set_prefix(prefix);
    for (const auto& endpoint : receives) {
      const std::string key = endpointKey("recv", endpoint);
      if (!kv.set(key.c_str(), &endpoint, sizeof(endpoint))) return ncclInternalError;
    }
    char readyKey[64];
    std::snprintf(readyKey, sizeof(readyKey), "ready/%d", real->rank);
    const uint8_t ready = 1;
    if (!kv.set(readyKey, &ready, sizeof(ready))) return ncclInternalError;
    for (int rank = 0; rank < real->nRanks; rank++) {
      std::snprintf(readyKey, sizeof(readyKey), "ready/%d", rank);
      uint8_t peerReady = 0;
      size_t length = 0;
      if (!kv.get(readyKey, &peerReady, sizeof(peerReady), &length) ||
          length != sizeof(peerReady) || peerReady != 1) {
        return ncclInternalError;
      }
    }
    for (auto& endpoint : sends) {
      const coldsnapNcclNetEndpoint expected = endpoint;
      const std::string key = endpointKey("recv", endpoint);
      size_t length = 0;
      if (!kv.get(key.c_str(), &endpoint, sizeof(endpoint), &length) ||
          length != sizeof(endpoint) || !sameEndpoint(expected, endpoint)) {
        WARN("invalid or missing ColdSnap in-place endpoint %s", key.c_str());
        return ncclInternalError;
      }
    }
    NCCLCHECK(g_net_reattach(real, sends.data(), sendCount));
    g_receive_endpoints += receiveCount;
    g_send_endpoints += sendCount;
  }
  g_state = InPlaceState::transport_reattached;
  updateEvidence();
  return ncclSuccess;
}

int32_t communicatorResume() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_state != InPlaceState::transport_reattached) return ncclInvalidUsage;
  NCCLCHECK(inspectCommunicators());
  g_state = InPlaceState::active;
  updateEvidence();
  return ncclSuccess;
}

const char* evidenceJson() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_evidence.empty()) updateEvidence();
  return g_evidence.c_str();
}

struct coldsnap_nccl_in_place_v1 g_provider = {
    sizeof(struct coldsnap_nccl_in_place_v1),
    COLDSNAP_NCCL_IN_PLACE_ABI_MAJOR,
    COLDSNAP_NCCL_IN_PLACE_ABI_MINOR,
    0,
    communicatorSuspend,
    transportDetach,
    transportReattach,
    communicatorResume,
    evidenceJson,
};

}  // namespace

extern "C" int32_t coldsnapNcclInPlaceQuery(
    uint32_t requested_abi_major,
    const struct coldsnap_nccl_in_place_v1** provider) {
  if (provider == nullptr) return ncclInvalidArgument;
  *provider = nullptr;
  if (requested_abi_major != COLDSNAP_NCCL_IN_PLACE_ABI_MAJOR) {
    return ncclInvalidArgument;
  }
  if (!experimentEnabled()) {
    WARN("experimental in-place ABI requested without "
         "COLDSNAP_NCCL_IN_PLACE_MODE=net-reconnect-v1");
    return ncclInvalidUsage;
  }
  if (!transportConfigured()) {
    WARN("net-reconnect-v1 requires NCCL_IB_RELEASE_ON_FINALIZE=1 unless NCCL_IB_DISABLE=1");
    return ncclInvalidUsage;
  }
  *provider = &g_provider;
  return ncclSuccess;
}

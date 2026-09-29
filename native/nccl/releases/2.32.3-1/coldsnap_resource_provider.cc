// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

#include "coldsnap_nccl_provider.h"
#include "comm.h"
#include "register.h"
#undef WARN
#undef INFO
#undef TRACE
#undef NCCLCHECK
#undef NCCLCHECKGOTO
#undef CUDACHECK
#include "shim_core.h"

#include <sstream>
#include <string>
#include <unordered_set>

using namespace nccl_checkpoint;

namespace {
thread_local std::string evidence;

bool hasCftEndpoints(const ncclDevrState& state) {
  // Before device-runtime initialization, the whole structure is zeroed.
  // Afterwards, a missing endpoint uses NCCL_LE_ID_INVALID; zero is a valid ID.
  return state.bigSize != 0 &&
         (state.le[0].baseId != NCCL_LE_ID_INVALID || state.le[1].baseId != NCCL_LE_ID_INVALID);
}

int32_t inspect(uint32_t mode) {
  if (mode != COLDSNAP_NCCL_CHECKPOINT_RECREATE && mode != COLDSNAP_NCCL_CHECKPOINT_IN_PLACE) {
    evidence.clear();
    return ncclInvalidArgument;
  }
  const bool inPlace = mode == COLDSNAP_NCCL_CHECKPOINT_IN_PLACE;
  using query_error_t = ncclResult_t (*)(ncclComm_t, ncclResult_t*);
  using validate_net_t = ncclResult_t (*)(ncclComm_t);
  using progress_status_t = ncclResult_t (*)(ncclComm_t, int*, int*);
  static progress_status_t progressStatus = nullptr;
  static query_error_t queryError = nullptr;
  static validate_net_t validateNet = nullptr;
  evidence.clear();
  NCCLCHECK(resolveRealFunction("ncclCheckpointProgressStatus", &progressStatus));
  NCCLCHECK(resolveRealFunction("ncclCommGetAsyncError", &queryError));
  if (inPlace) NCCLCHECK(resolveRealFunction("ncclCheckpointNetPreflight", &validateNet));

  std::ostringstream rows;
  bool first = true;
  bool blocked = false;
  std::unordered_set<uint64_t> hashes;
  std::unordered_set<ncclProxyState*> proxies;
  const ncclResult_t result = g_commHandles.forEachHandle(
      [&](ncclComm_t synthetic, const CommHandleEntry* entry) {
        ncclComm_t real = entry->realHandle;
        if (real == nullptr) return ncclSuccess;
        ncclResult_t asyncError = ncclSuccess;
        NCCLCHECK(queryError(real, &asyncError));
        if (asyncError != ncclSuccess || real->endMagic != NCCL_MAGIC) return ncclInvalidUsage;
        std::ostringstream reasons;
        bool firstReason = true;
        auto reject = [&](const char* reason) {
          if (!firstReason) reasons << ',';
          reasons << '"' << reason << '"';
          firstReason = false;
          blocked = true;
        };
        const auto& gin = real->sharedRes->ginState;
        const auto& rma = real->rmaState;
        const bool ginActive = gin.connected || gin.devComms != nullptr || gin.proxyThreadsCreated;
        const bool rmaActive = rma.rmaProxyState.connected || rma.rmaProxyState.rmaProxyCtxCount != 0 ||
                               rma.rmaCeState.initialized;
        const bool counters = real->hostCountersBlock != nullptr || real->deviceCountersBlock != nullptr;
        int monitors = 0, monitorRegistered = 0;
        NCCLCHECK(progressStatus(real, &monitors, &monitorRegistered));
        const bool cft = hasCftEndpoints(real->devrState);
        int registeredNet = 0, registeredNvls = 0, registeredIpc = 0;
        for (int i = 0; i < real->regCache.population; ++i) {
          const ncclReg* reg = real->regCache.slots[i];
          if (reg == nullptr) continue;
          registeredNet += reg->netHandleHead != nullptr;
          registeredNvls += reg->nvlsUbReg != nullptr;
          registeredIpc += (reg->state & IPC_REG_COMPLETE) != 0;
        }
        if (entry->restoreUnsafe) reject("exported-resource-not-restorable");
        if (ginActive) reject("gin-state-not-restorable");
        if (rmaActive) reject("rma-state-not-restorable");
        if (cft) reject("cft-state-not-restorable");
        if ((real->hostCountersBlock == nullptr) != (real->deviceCountersBlock == nullptr))
          reject("incomplete-progress-counter-storage");
        if (!inPlace && real->localPersistentRefs != 0) reject("destroy-graphs-before-recreate");
        if (inPlace) {
          if (entry->userState != comm_user_active) reject("communicator-not-active");
          if (!entry->registrations.empty() || real->regCache.population != 0)
            reject("registration-in-place-replay-unavailable");
          if (!entry->windows.empty() || real->devrState.winSortedCount != 0)
            reject("window-in-place-replay-unavailable");
          if (real->nvlsResources != nullptr) reject("nvls-in-place-replay-unavailable");
          if (real->ceColl.initialized) reject("copy-engine-in-place-replay-unavailable");
          if (!hashes.insert(real->commHash).second) reject("duplicate-communicator-hash");
          if (real->proxyState == nullptr || !proxies.insert(real->proxyState).second)
            reject("shared-or-missing-proxy-state");
          // Validate every NET connection before any communicator is detached.
          if (firstReason && validateNet(real) != ncclSuccess) reject("net-transport-not-detachable");
        }
        if (!first) rows << ',';
        first = false;
        rows << "{\"comm_hash\":\"" << std::hex << real->commHash << std::dec
             << "\",\"rank\":" << real->rank
             << ",\"registrations\":" << entry->registrations.size()
             << ",\"registration_cache_entries\":" << real->regCache.population
             << ",\"net_registered_buffers\":" << registeredNet
             << ",\"nvls_registered_buffers\":" << registeredNvls
             << ",\"ipc_registered_buffers\":" << registeredIpc
             << ",\"windows\":" << entry->windows.size()
             << ",\"runtime_windows\":" << real->devrState.winSortedCount
             << ",\"persistent_graph_references\":" << real->localPersistentRefs
             << ",\"gin_active\":" << (ginActive ? "true" : "false")
             << ",\"rma_active\":" << (rmaActive ? "true" : "false")
             << ",\"rma_plugin_context\":" << (real->rmaContext != nullptr ? "true" : "false")
             << ",\"gin_plugin_contexts\":" << gin.numActiveBackends
             << ",\"cft_active\":" << (cft ? "true" : "false")
             << ",\"nvls_active\":" << (real->nvlsResources != nullptr ? "true" : "false")
             << ",\"nvls_host_mode\":" << real->config.nvlsHostMode
             << ",\"copy_engine_active\":" << (real->ceColl.initialized ? "true" : "false")
             << ",\"progress_counters\":" << (counters ? "true" : "false")
             << ",\"progress_monitors\":" << monitors
             << ",\"progress_monitor_registered\":" << monitorRegistered
             << ",\"progress_timer_offset_ns\":" << real->gpuTimerOffsetNs
             << ",\"blockers\":[" << reasons.str() << "]}";
        return ncclSuccess;
      });
  std::ostringstream report;
  report << "{\"format\":1,\"kind\":\"coldsnap-nccl-resource-inventory\",\"mode\":\""
         << (inPlace ? "in-place" : "recreate") << "\",\"complete\":"
         << (result == ncclSuccess ? "true" : "false") << ",\"admitted\":"
         << (result == ncclSuccess && !blocked ? "true" : "false")
         << ",\"communicators\":[" << rows.str() << "]}";
  evidence = report.str();
  if (result != ncclSuccess) return result;
  if (blocked) {
    WARN("checkpoint resource preflight refused: %s", evidence.c_str());
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

const char* evidenceJson() { return evidence.c_str(); }
const coldsnap_nccl_resources_v1 provider = {
    sizeof(coldsnap_nccl_resources_v1), 1, 0, 0, inspect, evidenceJson};
}  // namespace

// Internal entry point used by both checkpoint lifecycles, even without an ABI consumer.
extern "C" ncclResult_t coldsnapNcclCheckpointPreflight(uint32_t mode) {
  return static_cast<ncclResult_t>(inspect(mode));
}

extern "C" int32_t coldsnapNcclResourcesQuery(
    uint32_t requested_abi_major, const coldsnap_nccl_resources_v1** output) {
  if (output == nullptr) return ncclInvalidArgument;
  *output = nullptr;
  if (requested_abi_major != 1) return ncclInvalidArgument;
  const coldsnap_nccl_provider_v1* identity = nullptr;
  const int32_t result = coldsnapNcclProviderQuery(1, &identity);
  if (result != ncclSuccess) return result;
  *output = &provider;
  return ncclSuccess;
}

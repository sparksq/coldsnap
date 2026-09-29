// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// One-GPU lifecycle regressions; an external controller may run CRIU.
#include "coldsnap_nccl_provider.h"
#include <nccl.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>

namespace fs = std::filesystem;

static void check(int result, const char* operation) {
  if (result != 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}
#define CHECK(call) check((call), #call)
static void require(bool condition, const char* reason) {
  if (!condition) throw std::runtime_error(reason);
}

static void requireMonitorsStopped() {
  for (const auto& task : fs::directory_iterator("/proc/self/task")) {
    std::ifstream nameFile(task.path() / "comm");
    std::string name;
    std::getline(nameFile, name);
    require(name.find("NCCL CtrMon") != 0 && name != "NCCL RAS",
            "diagnostic threads must stop before the checkpoint boundary");
  }
}

static void tlsStatus(const char* phase, bool detached) {
  using status_t = ncclResult_t (*)(int*, size_t*, int*, int*);
  auto status = reinterpret_cast<status_t>(dlsym(RTLD_DEFAULT, "ncclCheckpointCryptStatus"));
  require(status != nullptr, "TLS lifecycle export unavailable");
  int compiled, ready, reseed;
  size_t active;
  CHECK(status(&compiled, &active, &ready, &reseed));
  require(compiled == 1, "TLS-enabled runtime required");
  if (detached) require(active == 0 && ready == 0 && reseed == 1, "TLS capture boundary retains live state");
  else require(active > 0 && ready == 1 && reseed == 0, "encrypted connections must be reestablished");
  std::cout << "{\"phase\":\"" << phase << "\",\"tls_compiled\":" << compiled
            << ",\"tls_active_connections\":" << active << ",\"tls_contexts_ready\":" << ready
            << ",\"tls_reseed_pending\":" << reseed << "}" << std::endl;
}

// Keep the optional CRIU pause outside timed barriers: one immutable capture
// may be restored long after its original wall/monotonic-clock deadlines.
static void checkpointPause(const fs::path& directory, int rank, int cycle) {
  if (cycle != 0 || std::getenv("COLDSNAP_NCCL_TEST_CRIU_HOLD") == nullptr) return;
  std::ofstream(directory / ("capture-ready-" + std::to_string(rank))).close();
  while (!fs::exists(directory / "restore-continue")) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

static void barrier(const fs::path& directory, int rank, const std::string& phase) {
  std::ofstream(directory / (phase + "-" + std::to_string(rank))).close();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
  while (!fs::exists(directory / (phase + "-" + std::to_string(1 - rank)))) {
    if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("barrier timeout: " + phase);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

// A single rank can create real local windows without GPU Direct RDMA. This
// tests registration replay, not active GIN traffic or strict-ordering semantics.
static int windowLifecycle(const fs::path& directory, int flags,
                           const coldsnap_nccl_provider_v1* provider,
                           const coldsnap_nccl_resources_v1* resources) {
  require(flags == 0 || flags == NCCL_WIN_COLL_SYMMETRIC ||
          flags == NCCL_WIN_STRICT_ORDERING || flags == NCCL_WIN_GIN_ONLY,
          "unsupported window test flags");
  ncclUniqueId uid;
  CHECK(ncclGetUniqueId(&uid));
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.nvlsHostMode = ncclNvlsHostModeDisable;
  ncclComm_t comm = nullptr;
  CHECK(ncclCommInitRankConfig(&comm, 1, uid, 0, &config));
  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  constexpr size_t count = 16384;
  constexpr size_t bytes = count * sizeof(float);
  float *input = nullptr, *output = nullptr;
  CHECK(ncclMemAlloc(reinterpret_cast<void**>(&input), bytes));
  CHECK(ncclMemAlloc(reinterpret_cast<void**>(&output), bytes));
  std::vector<float> expected(count), actual(count);
  for (size_t i = 0; i < count; ++i) expected[i] = float(i % 17) + 0.25f;
  CHECK(cudaMemcpy(input, expected.data(), bytes, cudaMemcpyHostToDevice));
  ncclWindow_t window = nullptr;
  CHECK(ncclCommWindowRegister(comm, input, bytes, &window, flags));
  require(window != nullptr, "registration must return a synthetic handle");
  const ncclWindow_t original = window;
  auto inventory = [&]() {
    CHECK(resources->inspect(COLDSNAP_NCCL_CHECKPOINT_RECREATE));
    std::cout << resources->evidence_json() << std::endl;
  };
  auto verify = [&](const char* phase) {
    CHECK(cudaMemcpy(actual.data(), input, bytes, cudaMemcpyDeviceToHost));
    require(actual == expected, "window input bytes changed");
    CHECK(ncclAllReduce(input, output, count, ncclFloat, ncclSum, comm, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(actual.data(), output, bytes, cudaMemcpyDeviceToHost));
    require(actual == expected, "window collective mismatch");
    std::cout << "{\"rank\":0,\"phase\":\"" << phase << "\",\"window_flags\":"
              << flags << ",\"window_bytes_preserved\":true}" << std::endl;
  };
  inventory(); verify("baseline");
  const coldsnap_nccl_in_place_v1* ip = nullptr;
  CHECK(coldsnapNcclInPlaceQuery(1, &ip));
  require(ip->communicator_suspend() == ncclInvalidUsage,
          "in-place window must fail before mutation");
  std::cout << resources->evidence_json() << std::endl;
  verify("after-rejection");
  for (int cycle = 0; cycle < 2; ++cycle) {
    inventory();
    CHECK(provider->prepare());
    CHECK(provider->network_reset());
    requireMonitorsStopped();
    checkpointPause(directory, 0, cycle);
    CHECK(provider->restore());
    require(window == original, "synthetic handle changed");
    inventory(); verify("restored");
  }
  inventory(); // Include any internal symmetric-kernel resource window.
  CHECK(ncclCommWindowDeregister(comm, original));
  inventory();
  CHECK(ncclCommDestroy(comm));
  CHECK(cudaStreamDestroy(stream));
  CHECK(ncclMemFree(input)); CHECK(ncclMemFree(output));
  std::cout << "{\"rank\":0,\"passed\":true,\"window_flags\":" << flags
            << ",\"original_window_deregistered\":true}" << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "usage: target DIRECTORY RANK MODE");
    const fs::path directory = argv[1];
    const int rank = std::stoi(argv[2]);
    const std::string mode = argv[3];
    require(rank == 0 || rank == 1, "rank must be 0 or 1");
    const bool inPlace = mode == "inplace" || mode == "progress-inplace" || mode == "tls-inplace";
    const bool tls = mode == "tls-inplace" || mode == "tls-recreate";
    const bool progress = mode == "progress-inplace" || mode == "progress-recreate";
    if (tls) {
      ncclEncryptionConfig_t encryption = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
      encryption.mode = NCCL_ENCRYPTION_MODE_PSK;
      encryption.psk = "public-test-only-psk-0123456789abcdef";
      CHECK(ncclSetEncryption(&encryption));
    }
    CHECK(cudaSetDevice(0));
    const coldsnap_nccl_provider_v1* provider = nullptr;
    CHECK(coldsnapNcclProviderQuery(1, &provider));
    require(provider->compiled_nccl_version == 23203 && provider->loaded_nccl_version == 23203,
            "expected exact NCCL 2.32.3");
    require(provider->provider_revision == 2, "expected provider revision 2");
    const coldsnap_nccl_resources_v1* resources = nullptr;
    CHECK(coldsnapNcclResourcesQuery(1, &resources));
    require(resources->struct_size >= sizeof(*resources) && resources->abi_major == 1, "resource ABI");
    const coldsnap_nccl_resources_v1* invalid = resources;
    require(coldsnapNcclResourcesQuery(2, &invalid) == ncclInvalidArgument && invalid == nullptr,
            "resource ABI mismatch must clear output");
    require(resources->inspect(99) == ncclInvalidArgument, "invalid mode must fail");
    if (mode.rfind("window-", 0) == 0) {
      require(rank == 0, "window test has one rank");
      return windowLifecycle(directory, std::stoi(mode.substr(7)), provider, resources);
    }

    ncclUniqueId uid;
    if (rank == 0) {
      CHECK(ncclGetUniqueId(&uid));
      std::ofstream file(directory / "uid", std::ios::binary);
      file.write(reinterpret_cast<const char*>(&uid), sizeof(uid));
    }
    barrier(directory, rank, "uid");
    std::ifstream file(directory / "uid", std::ios::binary);
    file.read(reinterpret_cast<char*>(&uid), sizeof(uid));
    require(file.gcount() == sizeof(uid), "unique id must be complete");
    file.close();

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.nvlsHostMode = ncclNvlsHostModeDisable;
    ncclComm_t comm = nullptr;
    CHECK(ncclCommInitRankConfig(&comm, 2, uid, rank, &config));
    // Two independent communicators share one per-device progress monitor.
    ncclUniqueId extraId;
    if (rank == 0) {
      CHECK(ncclGetUniqueId(&extraId));
      std::ofstream extraFile(directory / "extra-uid", std::ios::binary);
      extraFile.write(reinterpret_cast<const char*>(&extraId), sizeof(extraId));
    }
    barrier(directory, rank, "extra-uid");
    std::ifstream extraFile(directory / "extra-uid", std::ios::binary);
    extraFile.read(reinterpret_cast<char*>(&extraId), sizeof(extraId));
    require(extraFile.gcount() == sizeof(extraId), "extra unique id must be complete");
    extraFile.close();
    ncclComm_t extraComm = nullptr;
    CHECK(ncclCommInitRankConfig(&extraComm, 2, extraId, rank, &config));
    cudaStream_t stream;
    cudaEvent_t event;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    float *input, *output;
    CHECK(cudaMalloc(&input, 4096));
    CHECK(cudaMalloc(&output, 4096));
    const float value = rank + 1;
    CHECK(cudaMemcpy(input, &value, sizeof(value), cudaMemcpyHostToDevice));
    ncclCollConfig_t collective = NCCL_COLLCONFIG_INITIALIZER;
    collective.launchCompletionEvent = event;
    bool eventCaptured = false;
    auto enqueue = [&]() {
      CHECK(ncclGroupStart());
      CHECK(ncclAllReduceConfig(input, output, 1, ncclFloat, ncclSum, comm, stream, &collective));
      CHECK(ncclAllReduce(input, output + 1, 1, ncclFloat, ncclSum, extraComm, stream));
      CHECK(ncclGroupEnd());
      CHECK(cudaStreamWaitEvent(stream, event, 0));
    };
    auto verify = [&](const char* phase) {
      CHECK(cudaStreamSynchronize(stream));
      // Captured events express graph dependencies and are not host-queryable.
      if (!eventCaptured) CHECK(cudaEventQuery(event));
      float actual[2];
      CHECK(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost));
      require(actual[0] == 3.0f && actual[1] == 3.0f, "both allreduces must produce 3");
      std::cout << "{\"rank\":" << rank << ",\"phase\":\"" << phase << "\",\"allreduce\":3}" << std::endl;
    };
    enqueue(); verify("baseline");
    if (tls) tlsStatus("tls-baseline", false);

    void* registration = nullptr;
    if (mode == "registration") {
      CHECK(ncclCommRegister(comm, input, 4096, &registration));
      require(resources->inspect(COLDSNAP_NCCL_CHECKPOINT_IN_PLACE) == ncclInvalidUsage,
              "in-place registration must fail before mutation");
      std::cout << resources->evidence_json() << std::endl;
      const coldsnap_nccl_in_place_v1* ip = nullptr;
      CHECK(coldsnapNcclInPlaceQuery(1, &ip));
      require(ip->communicator_suspend() == ncclInvalidUsage, "suspend must enforce preflight");
      enqueue(); verify("after-rejection");
      // The registration remains live and is replayed by the reconstruction path.
    }
    {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t executable = nullptr;
      const coldsnap_nccl_in_place_v1* ip = nullptr;
      if (inPlace) {
        CHECK(coldsnapNcclInPlaceQuery(1, &ip));
        CHECK(cudaEventDestroy(event));
        CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        collective.launchCompletionEvent = event;
        eventCaptured = true;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        enqueue();
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
        CHECK(cudaGraphLaunch(executable, stream)); verify("graph-baseline");
        require(provider->prepare() == ncclInvalidUsage, "recreate must reject live persistent graphs");
        CHECK(cudaGraphLaunch(executable, stream)); verify("graph-after-rejection");
      }
      for (int cycle = 0; cycle < 2; ++cycle) {
        barrier(directory, rank, "prepare" + std::to_string(cycle));
        CHECK(resources->inspect(inPlace ? COLDSNAP_NCCL_CHECKPOINT_IN_PLACE : COLDSNAP_NCCL_CHECKPOINT_RECREATE));
        std::cout << resources->evidence_json() << std::endl;
        if (inPlace) {
          const std::string activation = "cycle-" + std::to_string(cycle);
          setenv("COLDSNAP_NCCL_IN_PLACE_ACTIVATION", activation.c_str(), 1);
          CHECK(ip->communicator_suspend());
          CHECK(ip->transport_detach());
          requireMonitorsStopped();
          if (tls) tlsStatus("tls-detached", true);
          checkpointPause(directory, rank, cycle);
          barrier(directory, rank, "detached" + std::to_string(cycle));
          CHECK(ip->transport_reattach());
          CHECK(ip->communicator_resume());
          CHECK(cudaGraphLaunch(executable, stream));
        } else {
          CHECK(provider->prepare());
          CHECK(provider->network_reset());
          requireMonitorsStopped();
          if (tls) tlsStatus("tls-detached", true);
          checkpointPause(directory, rank, cycle);
          barrier(directory, rank, "detached" + std::to_string(cycle));
          CHECK(provider->restore());
          enqueue();
        }
        verify("restored");
        if (tls) tlsStatus("tls-restored", false);
        CHECK(resources->inspect(inPlace ? COLDSNAP_NCCL_CHECKPOINT_IN_PLACE : COLDSNAP_NCCL_CHECKPOINT_RECREATE));
        std::cout << resources->evidence_json() << std::endl;
      }
      if (executable != nullptr) CHECK(cudaGraphExecDestroy(executable));
      if (graph != nullptr) CHECK(cudaGraphDestroy(graph));
    }
    if (registration != nullptr) CHECK(ncclCommDeregister(comm, registration));
    barrier(directory, rank, "destroy");
    CHECK(ncclCommDestroy(comm));
    CHECK(ncclCommDestroy(extraComm));
    CHECK(cudaEventDestroy(event));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(input)); CHECK(cudaFree(output));
    std::cout << "{\"rank\":" << rank << ",\"passed\":true,\"progress\":"
              << (progress ? "true" : "false") << "}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}

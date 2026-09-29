// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

#include "shim_core.h"
#include <stdexcept>

using namespace nccl_checkpoint;
extern "C" {
void window_test_mode(int);
void window_test_complete();
ncclWindow_t window_test_last();
}

static void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  ncclComm_t comm = g_commHandles.makeSynthetic(
      reinterpret_cast<ncclComm_t>(0x100000), new CommInitParams{});
  ncclComm_t other = g_commHandles.makeSynthetic(
      reinterpret_cast<ncclComm_t>(0x100001), new CommInitParams{});
  int data = 42;
  ncclWindow_t win = reinterpret_cast<ncclWindow_t>(0xbad);

  // Failing registration must not publish a handle already removed from the map.
  window_test_mode(1);
  require(ncclCommWindowRegister(comm, &data, sizeof(data), &win, 0) == ncclInvalidArgument,
          "registration error propagated");
  require(win == nullptr, "failed registration exposed a dangling synthetic window");
  require(g_commHandles[comm].windows.empty(), "failed registration retained bookkeeping");
  require(ncclCommWindowRegister(comm, &data, sizeof(data), nullptr, 0) == ncclInvalidArgument,
          "null output must not crash");

  // A no-op registration has an owned replay record until deregistered.
  window_test_mode(0);
  require(ncclCommWindowRegister(comm, &data, sizeof(data), &win, 0) == ncclSuccess,
          "unsupported window registration");
  require(g_windowHandles.checkHandle(win), "synthetic record missing");
  require(ncclCommWindowDeregister(other, win) == ncclInvalidArgument,
          "foreign communicator accepted");
  require(ncclCommWindowDeregister(comm, win) == ncclSuccess, "no-op window deregistration");
  require(!g_windowHandles.checkHandle(win) && g_commHandles[comm].windows.empty(),
          "no-op window leaked bookkeeping");

  // Grouped success and nonblocking registration may write the output later.
  for (int mode : {2, 3, 4}) {
    window_test_mode(mode);
    require(ncclCommWindowRegister(comm, &data, sizeof(data), &win, 0) ==
                (mode == 3 ? ncclInProgress : ncclSuccess), "registration status");
    require(g_windowHandles.checkHandle(win), "deferred output storage was freed too early");
    if (mode != 4) window_test_complete();
    window_test_mode(5);
    require(ncclCommWindowDeregister(comm, win) == ncclInvalidArgument,
            "deregister failure status");
    require(g_windowHandles.checkHandle(win), "deregister failure lost live handle");
    window_test_mode(0);
    require(ncclCommWindowDeregister(comm, win) == ncclSuccess, "real deregistration");
    require(window_test_last() == reinterpret_cast<ncclWindow_t>(0x200000),
            "runtime handle not translated");
    require(!g_windowHandles.checkHandle(win) && g_commHandles[comm].windows.empty(),
            "real window retained bookkeeping");
  }
  g_commHandles.remove(comm);
  g_commHandles.remove(other);
}

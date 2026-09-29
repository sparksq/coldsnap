// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

#include <nccl.h>
#include <cassert>

extern "C" {
static int mode = 0;
static ncclWindow_t* pending = nullptr;
static ncclWindow_t last = nullptr;

void window_test_mode(int value) { mode = value; }
void window_test_complete() {
  assert(pending);
  *pending = reinterpret_cast<ncclWindow_t>(0x200000);
  pending = nullptr;
}
ncclWindow_t window_test_last() { return last; }

ncclResult_t ncclCommWindowRegister(ncclComm_t, void*, size_t, ncclWindow_t* out, int) {
  if (!out) return ncclInvalidArgument;
  *out = nullptr;
  if (mode == 1) return ncclInvalidArgument;
  if (mode == 2 || mode == 3) {
    pending = out;
    return mode == 2 ? ncclSuccess : ncclInProgress;
  }
  if (mode == 4) *out = reinterpret_cast<ncclWindow_t>(0x200000);
  return ncclSuccess;
}

ncclResult_t ncclCommWindowDeregister(ncclComm_t, ncclWindow_t win) {
  last = win;
  return mode == 5 ? ncclInvalidArgument : ncclSuccess;
}

// Logging must not initialize CUDA during this CPU-only regression.
cudaError_t cudaGetDevice(int* dev) {
  *dev = 0;
  return cudaSuccess;
}
}

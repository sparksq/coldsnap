// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Runtime ABI fixture for the actual release-owned provider identity source.
#include <cstddef>
extern "C" {
int cudaGetDevice(int* device) { *device = 0; return 0; }
int ncclGetVersion(int* version) { *version = 23203; return 0; }
int ncclCheckpointCryptStatus(int* compiled, size_t* active, int* ready, int* reseed) {
  *compiled = TEST_TLS; *active = 0; *ready = 0; *reseed = 0; return 0;
}
int ncclCheckpointPrepare() { return 5; }
int ncclCheckpointRestore() { return 5; }
int ncclCheckpointNetworkReset() { return 5; }
int ncclCheckpointNetworkInit() { return 5; }
int ncclCheckpointIbGetStatus(int*, int*, int*, int*) { return 5; }
int ncclCheckpointCommGetReal(void*, void**) { return 5; }
}

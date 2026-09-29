// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// CPU-only dependency fixture for the actual patched net_ib/gdr.cc.
#pragma once
#include <cerrno>
#include <mutex>

using ncclResult_t = int;
constexpr int ncclSuccess = 0, ncclSystemError = 2, ncclInvalidArgument = 4;
struct ibv_context {};
struct ibv_pd {};
struct ncclIbDev { ibv_context* context; int dmaBufSupported; };
struct ncclIbMergedDev { struct { int ndevs; int devs[2]; } vProps; };
extern ncclIbDev ncclIbDevs[2];
extern ncclIbMergedDev ncclIbMergedDevs[2];
extern int ncclNIbDevs, ncclNMergedIbDevs;
extern int capabilityProbes;
extern bool dmaBufAvailable, peerMemAvailable;

inline int fixtureAccess(const char*, int) { return peerMemAvailable ? 0 : -1; }
#define access fixtureAccess
#define F_OK 0
#define NCCLCHECKGOTO(call, result, label) do { result = (call); if (result != ncclSuccess) goto label; } while (0)
inline ncclResult_t wrap_ibv_alloc_pd(ibv_pd** pd, ibv_context*) {
  static ibv_pd value;
  *pd = &value;
  return ncclSuccess;
}
inline ncclResult_t wrap_ibv_dealloc_pd(ibv_pd*) { return ncclSuccess; }
inline void* wrap_direct_ibv_reg_dmabuf_mr(ibv_pd*, unsigned long long, unsigned long long,
                                         unsigned long long, int, int) {
  ++capabilityProbes;
  errno = dmaBufAvailable ? EBADF : EOPNOTSUPP;
  return nullptr;
}

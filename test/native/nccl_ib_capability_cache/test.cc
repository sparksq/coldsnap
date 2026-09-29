// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

#include "common.h"
#include <stdexcept>

ncclIbDev ncclIbDevs[2] = {};
ncclIbMergedDev ncclIbMergedDevs[2] = {};
int ncclNIbDevs = 1, ncclNMergedIbDevs = 1;
int capabilityProbes = 0;
bool dmaBufAvailable = true, peerMemAvailable = false;

// The test compiles the release-patched production source, without a CUDA GPU.
#include "gdr.cc"

static void require(bool condition, const char* reason) {
  if (!condition) throw std::runtime_error(reason);
}
int main() {
  ncclIbMergedDevs[0].vProps.ndevs = 1;
  require(ncclIbDmaBufSupport(0) == ncclSuccess, "initial device probe");
  require(ncclIbDmaBufSupport(0) == ncclSuccess && capabilityProbes == 1, "cache within device lifetime");
  // Match the device reconstruction performed by IB finalization/discovery.
  ncclIbDevs[0] = {};
  require(ncclIbDmaBufSupport(0) == ncclSuccess && capabilityProbes == 2, "reprobe reopened device");
  ncclIbDevs[0] = {};
  dmaBufAvailable = false;
  require(ncclIbDmaBufSupport(0) == ncclSystemError && capabilityProbes == 3, "destination lacks DMA-BUF");
  require(ncclIbDmaBufSupport(0) == ncclSystemError && capabilityProbes == 3, "cache negative result");
  require(ncclIbDmaBufSupport(-1) == ncclInvalidArgument, "reject negative device");
  require(ncclIbDmaBufSupport(1) == ncclInvalidArgument, "reject absent device");
  ncclIbMergedDevs[0].vProps.ndevs = 0;
  require(ncclIbDmaBufSupport(0) == ncclInvalidArgument, "reject empty merged device");
  require(capabilityProbes == 3, "invalid devices must not probe");

  require(ncclIbGdrSupport() == ncclSystemError, "source lacks peermem");
  require(ncclIbPeerMemSupport() == ncclSystemError, "source lacks NVIDIA peermem");
  peerMemAvailable = true;
  require(ncclIbGdrSupport() == ncclSystemError, "module result is cached within epoch");
  ncclIbGdrCheckpointReset();
  require(ncclIbGdrSupport() == ncclSuccess, "destination module reprobe");
  require(ncclIbPeerMemSupport() == ncclSuccess, "destination NVIDIA module reprobe");
  peerMemAvailable = false;
  ncclIbGdrCheckpointReset();
  require(ncclIbGdrSupport() == ncclSystemError, "next destination module absent");
  require(ncclIbPeerMemSupport() == ncclSystemError, "next destination NVIDIA module absent");
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

#include "coldsnap_resource_provider.cc"
#include <stdexcept>
static void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
int main() {
  ncclDevrState state{};
  require(!hasCftEndpoints(state), "uninitialized device runtime has no endpoints");
  state.bigSize = 4096;
  state.le[0].baseId = state.le[1].baseId = NCCL_LE_ID_INVALID;
  require(!hasCftEndpoints(state), "invalid endpoint sentinel treated as active CFT");
  state.le[0].baseId = 0;
  require(hasCftEndpoints(state), "endpoint zero is valid");
  state.le[0].baseId = NCCL_LE_ID_INVALID;
  state.le[1].baseId = 123;
  require(hasCftEndpoints(state), "counted endpoint is active");
}

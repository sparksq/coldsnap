// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Preserve the source-line identities used by the qualified NCCL provider.
#line 1
#pragma once

// ColdSnap coordinator transport for NVIDIA NCCL's checkpoint shim.
//
// The class name and method surface intentionally match NCCL's
// contrib/nccl_checkpoint KVStoreClient so the checkpoint state machine is
// unchanged. This implementation has no external datastore dependency.

#include <cstddef>

namespace nccl_checkpoint {

class KVStoreClient {
 public:
  KVStoreClient();
  ~KVStoreClient();

  KVStoreClient(const KVStoreClient&) = delete;
  KVStoreClient& operator=(const KVStoreClient&) = delete;
  KVStoreClient(KVStoreClient&&) = delete;
  KVStoreClient& operator=(KVStoreClient&&) = delete;

  bool connect(const char* host, int port);

  // Reads a four-field ColdSnap endpoint from the file named by
  // NCCL_CHECKPOINT_COORDINATOR_PATH:
  //   coldsnap-coord-v1 <host>:<port> <scope> <token>
  bool connect_from_env();

  void disconnect();
  bool is_connected() const;
  void set_prefix(const char* prefix);
  bool set(const char* key, const void* data, size_t len);
  bool get(const char* key, void* buf, size_t buf_len, size_t* out_len);
  void del(const char* key);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace nccl_checkpoint

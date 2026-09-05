// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Preserve the source-line identities used by the qualified NCCL provider.
#line 1
#include "kv_store_client.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr char kMagic[] = {'C', 'S', 'K', 'V'};
constexpr uint8_t kVersion = 1;
constexpr uint8_t kOpSet = 1;
constexpr uint8_t kOpGet = 2;
constexpr uint8_t kOpDelete = 3;
constexpr uint8_t kOpHealth = 4;
constexpr uint8_t kStatusOk = 0;
constexpr uint32_t kMaximumResponseBytes = 16U * 1024U * 1024U;
constexpr int kDefaultTimeoutSeconds = 300;
constexpr int kConnectRetryMilliseconds = 100;
constexpr const char* kPathEnvironment = "NCCL_CHECKPOINT_COORDINATOR_PATH";
constexpr const char* kTimeoutEnvironment = "NCCL_CHECKPOINT_COORDINATOR_TIMEOUT";

#pragma pack(push, 1)
struct RequestHeader {
  char magic[4];
  uint8_t version;
  uint8_t operation;
  uint16_t flags;
  uint32_t timeout_ms;
  uint32_t token_length;
  uint32_t key_length;
  uint32_t value_length;
  uint32_t total_length;
};

struct ResponseHeader {
  char magic[4];
  uint8_t version;
  uint8_t status;
  uint16_t flags;
  uint32_t value_length;
};
#pragma pack(pop)

static_assert(sizeof(RequestHeader) == 28, "unexpected coordinator request header size");
static_assert(sizeof(ResponseHeader) == 12, "unexpected coordinator response header size");

std::chrono::milliseconds coordinator_timeout() {
  const char* value = std::getenv(kTimeoutEnvironment);
  if (value == nullptr || *value == '\0') {
    return std::chrono::seconds(kDefaultTimeoutSeconds);
  }
  char* end = nullptr;
  const double seconds = std::strtod(value, &end);
  if (end == value || *end != '\0' || seconds < 0.0 || seconds > 4294967.295) {
    return std::chrono::seconds(kDefaultTimeoutSeconds);
  }
  return std::chrono::milliseconds(static_cast<long long>(seconds * 1000.0));
}

bool send_all(int socket, const void* value, size_t length) {
  const auto* current = static_cast<const char*>(value);
  while (length != 0) {
    const ssize_t written = send(socket, current, length, MSG_NOSIGNAL);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return false;
    current += written;
    length -= static_cast<size_t>(written);
  }
  return true;
}

bool receive_all(int socket, void* value, size_t length) {
  auto* current = static_cast<char*>(value);
  while (length != 0) {
    const ssize_t received = recv(socket, current, length, 0);
    if (received < 0 && errno == EINTR) continue;
    if (received <= 0) return false;
    current += received;
    length -= static_cast<size_t>(received);
  }
  return true;
}

bool split_address(const std::string& address, std::string* host, int* port) {
  size_t separator = std::string::npos;
  if (!address.empty() && address.front() == '[') {
    const auto bracket = address.rfind("]:");
    if (bracket == std::string::npos) return false;
    *host = address.substr(1, bracket - 1);
    separator = bracket + 1;
  } else {
    separator = address.rfind(':');
    if (separator == std::string::npos) return false;
    *host = address.substr(0, separator);
  }
  if (host->empty()) return false;
  char* end = nullptr;
  const long parsed = std::strtol(address.c_str() + separator + 1, &end, 10);
  if (*end != '\0' || parsed <= 0 || parsed > 65535) return false;
  *port = static_cast<int>(parsed);
  return true;
}

int open_socket(const std::string& host, int port, std::chrono::milliseconds timeout) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* addresses = nullptr;
  const std::string service = std::to_string(port);
  const int lookup = getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
  if (lookup != 0) {
    std::fprintf(stderr, "ColdSnap coordinator: resolve %s: %s\n", host.c_str(), gai_strerror(lookup));
    return -1;
  }
  int connected = -1;
  for (addrinfo* address = addresses; address != nullptr; address = address->ai_next) {
    const int candidate = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (candidate < 0) continue;
    timeval value{};
    value.tv_sec = static_cast<time_t>(timeout.count() / 1000);
    value.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
    setsockopt(candidate, SOL_SOCKET, SO_RCVTIMEO, &value, sizeof(value));
    setsockopt(candidate, SOL_SOCKET, SO_SNDTIMEO, &value, sizeof(value));
    if (::connect(candidate, address->ai_addr, address->ai_addrlen) == 0) {
      connected = candidate;
      break;
    }
    close(candidate);
  }
  freeaddrinfo(addresses);
  return connected;
}

}  // namespace

namespace nccl_checkpoint {

struct KVStoreClient::Impl {
  std::string host;
  int port = 0;
  std::string scope;
  std::string token;
  std::string prefix;
  bool connected = false;

  std::string key(const char* value) const {
    std::string result = scope;
    if (!prefix.empty()) result += "/" + prefix;
    if (value != nullptr && *value != '\0') result += "/" + std::string(value);
    return result;
  }

  bool request(uint8_t operation, const std::string& key, const void* input, size_t input_length,
               std::chrono::milliseconds wait, std::vector<char>* output) const {
    if (!connected || token.empty()) return false;
    if (key.size() > std::numeric_limits<uint32_t>::max() ||
        input_length > std::numeric_limits<uint32_t>::max() ||
        token.size() > std::numeric_limits<uint32_t>::max()) {
      std::fprintf(stderr, "ColdSnap coordinator: request is too large\n");
      return false;
    }
    const size_t total = token.size() + key.size() + input_length;
    if (total > std::numeric_limits<uint32_t>::max()) return false;
    const auto network_timeout = wait + std::chrono::seconds(5);
    const int socket = open_socket(host, port, network_timeout);
    if (socket < 0) {
      std::fprintf(stderr, "ColdSnap coordinator: connect to %s:%d failed: %s\n", host.c_str(), port,
                   std::strerror(errno));
      return false;
    }
    RequestHeader request{};
    std::memcpy(request.magic, kMagic, sizeof(kMagic));
    request.version = kVersion;
    request.operation = operation;
    request.timeout_ms = htonl(static_cast<uint32_t>(wait.count()));
    request.token_length = htonl(static_cast<uint32_t>(token.size()));
    request.key_length = htonl(static_cast<uint32_t>(key.size()));
    request.value_length = htonl(static_cast<uint32_t>(input_length));
    request.total_length = htonl(static_cast<uint32_t>(total));
    bool ok = send_all(socket, &request, sizeof(request)) && send_all(socket, token.data(), token.size()) &&
              send_all(socket, key.data(), key.size()) && send_all(socket, input, input_length);
    ResponseHeader response{};
    ok = ok && receive_all(socket, &response, sizeof(response));
    if (!ok || std::memcmp(response.magic, kMagic, sizeof(kMagic)) != 0 || response.version != kVersion) {
      std::fprintf(stderr, "ColdSnap coordinator: incomplete or invalid response\n");
      close(socket);
      return false;
    }
    const uint32_t value_length = ntohl(response.value_length);
    if (value_length > kMaximumResponseBytes) {
      std::fprintf(stderr, "ColdSnap coordinator: response exceeds %u bytes\n", kMaximumResponseBytes);
      close(socket);
      return false;
    }
    output->resize(value_length);
    ok = receive_all(socket, output->data(), output->size());
    close(socket);
    if (!ok || response.status != kStatusOk) {
      std::string detail(output->begin(), output->end());
      std::fprintf(stderr, "ColdSnap coordinator: operation %u failed with status %u%s%s\n", operation,
                   response.status, detail.empty() ? "" : ": ", detail.c_str());
      return false;
    }
    return true;
  }
};

KVStoreClient::KVStoreClient() : impl_(new Impl{}) {}

KVStoreClient::~KVStoreClient() {
  disconnect();
  delete impl_;
}

bool KVStoreClient::connect(const char* host, int port) {
  disconnect();
  if (host == nullptr || *host == '\0' || port <= 0 || port > 65535) return false;
  const char* scope = std::getenv("NCCL_CHECKPOINT_COORDINATOR_SCOPE");
  const char* token = std::getenv("NCCL_CHECKPOINT_COORDINATOR_TOKEN");
  if (scope == nullptr || token == nullptr || *scope == '\0' || *token == '\0') {
    std::fprintf(stderr, "ColdSnap coordinator: scope/token environment is missing\n");
    return false;
  }
  impl_->host = host;
  impl_->port = port;
  impl_->scope = scope;
  impl_->token = token;
  impl_->connected = true;
  std::vector<char> response;
  if (!impl_->request(kOpHealth, "", nullptr, 0, std::chrono::milliseconds(0), &response)) {
    disconnect();
    return false;
  }
  return true;
}

bool KVStoreClient::connect_from_env() {
  const char* path = std::getenv(kPathEnvironment);
  if (path == nullptr || *path == '\0') {
    std::fprintf(stderr, "KVStoreClient: %s not set\n", kPathEnvironment);
    return false;
  }
  std::ifstream stream(path);
  if (!stream) {
    std::fprintf(stderr, "KVStoreClient: cannot open %s: %s\n", path, std::strerror(errno));
    return false;
  }
  std::string format;
  std::string address;
  std::string scope;
  std::string token;
  std::string extra;
  if (!(stream >> format >> address >> scope >> token) || stream >> extra || format != "coldsnap-coord-v1") {
    std::fprintf(stderr, "KVStoreClient: invalid ColdSnap coordinator endpoint in %s\n", path);
    return false;
  }
  std::string host;
  int port = 0;
  if (!split_address(address, &host, &port)) {
    std::fprintf(stderr, "KVStoreClient: invalid coordinator address in %s\n", path);
    return false;
  }
  disconnect();
  impl_->host = host;
  impl_->port = port;
  impl_->scope = scope;
  impl_->token = token;
  impl_->connected = true;

  const auto timeout = coordinator_timeout();
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    std::vector<char> response;
    if (impl_->request(kOpHealth, "", nullptr, 0, std::chrono::milliseconds(0), &response)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(kConnectRetryMilliseconds));
  } while (std::chrono::steady_clock::now() < deadline);
  std::fprintf(stderr, "KVStoreClient: timed out connecting to ColdSnap coordinator\n");
  disconnect();
  return false;
}

void KVStoreClient::disconnect() {
  impl_->connected = false;
  impl_->host.clear();
  impl_->port = 0;
  impl_->scope.clear();
  impl_->token.clear();
}

bool KVStoreClient::is_connected() const { return impl_->connected; }

void KVStoreClient::set_prefix(const char* prefix) { impl_->prefix = prefix != nullptr ? prefix : ""; }

bool KVStoreClient::set(const char* key, const void* data, size_t len) {
  std::vector<char> response;
  return impl_->request(kOpSet, impl_->key(key), data, len, std::chrono::milliseconds(0), &response);
}

bool KVStoreClient::get(const char* key, void* buf, size_t buf_len, size_t* out_len) {
  std::vector<char> response;
  if (!impl_->request(kOpGet, impl_->key(key), nullptr, 0, coordinator_timeout(), &response)) return false;
  if (response.size() > buf_len) {
    std::fprintf(stderr, "ColdSnap coordinator: value is too large: %zu > %zu bytes\n", response.size(), buf_len);
    return false;
  }
  if (!response.empty()) std::memcpy(buf, response.data(), response.size());
  if (out_len != nullptr) *out_len = response.size();
  return true;
}

void KVStoreClient::del(const char* key) {
  std::vector<char> response;
  (void)impl_->request(kOpDelete, impl_->key(key), nullptr, 0, std::chrono::milliseconds(0), &response);
}

}  // namespace nccl_checkpoint

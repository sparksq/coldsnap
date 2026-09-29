// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Compile the exact patched NCCL implementation; substitute only OS adapters.
#include "misc/crypt.cc"
#include <array>
#include <cstdarg>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <dlfcn.h>
#include <sys/socket.h>
#include <unistd.h>

uint32_t ncclDebugLevelMask = 0;
uint64_t ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;
void ncclDebugLogInternal(ncclDebugLogLevel, unsigned long, const char*, const char*, int, const char* fmt, ...) {
  va_list args; va_start(args, fmt); vfprintf(stderr, fmt, args); va_end(args); fputc('\n', stderr);
}
int64_t ncclParamPollTimeOut() { return 0; }
void ncclOsPollSocket(ncclSocketDescriptor, int) {}
ncclOsLibraryHandle ncclOsDlopen(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* name) { return dlsym(handle, name); }
void ncclOsDlclose(ncclOsLibraryHandle handle) { if (handle != nullptr) dlclose(handle); }
const char* ncclOsDlerror() { return dlerror(); }
const char* ncclSocketToString(const ncclSocketAddress*, char*, const int) { return "local-test-socket"; }
ncclResult_t ncclOsSocketProgressOpt(int op, ncclSocket* sock, void* data, int size,
                                    int* offset, int, int* closed) {
  *closed = 0;
  ssize_t n = op == NCCL_SOCKET_SEND ?
    send(sock->socketDescriptor, static_cast<char*>(data) + *offset, size - *offset, MSG_DONTWAIT | MSG_NOSIGNAL) :
    recv(sock->socketDescriptor, static_cast<char*>(data) + *offset, size - *offset, MSG_DONTWAIT);
  if (n > 0) *offset += n;
  else if (n == 0) *closed = 1;
  else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return ncclRemoteError;
  return ncclSuccess;
}
static void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
static void check(ncclResult_t value) { require(value == ncclSuccess, "unexpected NCCL error"); }

#if defined(NCCL_TLS_BACKEND_OPENSSL3)
static int reseeds = 0;
static bool failEntropy = false;
static int reseed(EVP_RAND_CTX* ctx, int prediction, const unsigned char* ent, size_t entLen,
                  const unsigned char* additional, size_t additionalLen) {
  require(ctx == RAND_get0_primary(ncclCryptLibCtx), "must reseed the private context");
  require(prediction == 1, "must request live entropy");
  reseeds++;
  if (failEntropy) return 0;
  return EVP_RAND_reseed(ctx, prediction, ent, entLen, additional, additionalLen);
}
static void transfer(ncclSocket* sender, ncclSocket* receiver) {
  std::vector<unsigned char> sent(256 * 1024), received(sent.size());
  for (size_t i = 0; i < sent.size(); ++i) sent[i] = (i * 31 + 7) % 251;
  int written = 0, read = 0;
  for (int n = 0; n < 100000 && read < int(sent.size()); ++n) {
    check(ncclCryptSocketSend(sender, sent.data(), sent.size(), &written, nullptr));
    check(ncclCryptSocketRecv(receiver, received.data(), received.size(), &read, nullptr));
  }
  require(read == int(sent.size()) && sent == received, "encrypted transfer mismatch");
}
static std::array<unsigned char, 32> connection(int cycle) {
  int fds[2]; require(socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0, fds) == 0, "socketpair");
  ncclSocket client{}, server{};
  client.socketDescriptor = fds[0]; server.socketDescriptor = fds[1];
  client.magic = server.magic = NCCL_SOCKET_MAGIC;
  client.type = server.type = ncclSocketTypeNetSocket;
  if (cycle == 1) {
    failEntropy = true;
    require(ncclCryptStartConnect(&client) == ncclSystemError, "entropy failure must refuse a new connection");
    require(client.crypto == nullptr && ncclCryptActiveConnections == 0 &&
            ncclCryptNeedsReseed && !ncclCryptLibReady, "entropy failure must remain retryable and closed");
    failEntropy = false;
  }
  check(ncclCryptStartConnect(&client));
  bool clientReady = false, serverReady = false;
  for (int n = 0; n < 100000 && !(clientReady && serverReady); ++n) {
    if (!clientReady) {
      ncclResult_t rc = ncclCryptConnectHello(&client);
      require(rc == ncclSuccess || rc == ncclInProgress, "client handshake failed");
      clientReady = rc == ncclSuccess;
    }
    if (!serverReady) {
      ncclCryptHelloVerdict verdict;
      ncclResult_t rc = ncclCryptAcceptHello(&server, &verdict);
      require(rc == ncclSuccess || rc == ncclInProgress, "server handshake failed");
      require(verdict != NCCL_CRYPT_HELLO_VERDICT_RESET, "PSK peer rejected");
      serverReady = verdict == NCCL_CRYPT_HELLO_VERDICT_READY;
    }
  }
  require(clientReady && serverReady, "handshake timeout");
  require(ncclCryptActiveConnections == 2, "live connection inventory");
  SSL* original = client.crypto->ssl;
  require(ncclCheckpointCryptReset() == ncclInvalidUsage, "reset must reject live SSL state");
  require(client.crypto->ssl == original && ncclCryptLibReady, "rejected reset mutated live state");
  transfer(&client, &server); transfer(&server, &client);
  std::array<unsigned char, 32> random{};
  require(SSL_get_client_random(client.crypto->ssl, random.data(), random.size()) == random.size(), "client random");
  ncclCryptFree(client.crypto); ncclCryptFree(server.crypto);
  close(fds[0]); close(fds[1]);
  require(ncclCryptActiveConnections == 0, "connection count must drain");
  check(ncclCheckpointCryptReset()); check(ncclCheckpointCryptReset());
  require(!ncclCryptLibReady && ncclCryptNeedsReseed, "capture retains a mandatory reseed boundary");
  return random;
}
#endif

int main() {
  try {
    check(ncclCheckpointCryptReset());
    int compiled = -1, ready = -1, pending = -1; size_t active = 99;
    check(ncclCheckpointCryptStatus(&compiled, &active, &ready, &pending));
    require(active == 0 && ready == 0 && pending == 0, "initial crypto state");
    require(ncclCheckpointCryptStatus(nullptr, &active, &ready, &pending) == ncclInvalidArgument,
            "status validates outputs");
    ncclEncryptionConfig_t config = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
    config.mode = NCCL_ENCRYPTION_MODE_PSK;
    config.psk = "public-test-only-psk-0123456789abcdef";
    check(ncclSetEncryption(&config));
    bool encrypted = false;
#if defined(NCCL_TLS_BACKEND_OPENSSL3)
    require(compiled == 1, "TLS compiled inventory");
    check(ncclGetCryptConnectionMode(&encrypted)); require(encrypted, "TLS must be active");
    require(ncclCryptLibCtx != nullptr && ncclCryptLibCtx != OSSL_LIB_CTX_get0_global_default(), "isolated context");
    ncclCryptOpenSsl.pfn_EVP_RAND_reseed = reseed;
    auto first = connection(0);
    auto second = connection(1);
    auto third = connection(2);
    require(first != second && second != third && first != third, "fresh handshake randomness");
    require(reseeds == 3, "each restore reseeds, including the injected failure retry");
    unsigned char applicationRandom[32];
    require(RAND_bytes(applicationRandom, sizeof(applicationRandom)) == 1, "application RNG remains usable");
    require(ncclCryptApiPskLen == strlen(config.psk) &&
            memcmp(ncclCryptApiPsk, config.psk, ncclCryptApiPskLen) == 0, "PSK survives lifecycle");
#else
    require(compiled == 0, "disabled backend inventory");
    require(ncclGetCryptConnectionMode(&encrypted) == ncclInvalidUsage, "missing TLS backend must refuse PSK");
#endif
    check(ncclSetEncryption(nullptr));
    check(ncclGetCryptConnectionMode(&encrypted)); require(!encrypted, "explicit plaintext reset");
    check(ncclCheckpointCryptReset());
    std::cout << "TLS lifecycle passed (compiled=" << compiled << ")" << std::endl;
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 1; }
}

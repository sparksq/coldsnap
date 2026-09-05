// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

//
// The aligned pinned-buffer ring, bounded I/O window, asynchronous CUDA-copy
// pipeline, and optional runtime cuFile binding are adapted from concepts in
// InstantTensor. This implementation has a separate ABI and writes directly
// into engine-owned final allocations.

#include "coldsnap_hydration.h"

#include <cuda_runtime_api.h>
#include <cufile.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

thread_local std::string last_error;

uint64_t elapsed_ns(Clock::time_point started) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started)
            .count());
}

std::string system_error(const char* operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class FileDescriptor {
  public:
    explicit FileDescriptor(int value) : value_(value) {}
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    ~FileDescriptor() {
        if (value_ >= 0) {
            ::close(value_);
        }
    }
    int get() const { return value_; }

  private:
    int value_;
};

struct Job {
    size_t extent_index;
    uint64_t file_offset;
    uintptr_t destination;
    size_t length;
};

struct CaptureJob {
    size_t extent_index;
    uint64_t file_offset;
    uintptr_t source;
    size_t length;
};

std::vector<Job> make_jobs(
    const coldsnap_hydration_extent* extents,
    size_t extent_count,
    uint64_t chunk_bytes) {
    std::vector<Job> jobs;
    for (size_t index = 0; index < extent_count; ++index) {
        const auto& extent = extents[index];
        uint64_t copied = 0;
        while (copied < extent.length) {
            const uint64_t count = std::min(chunk_bytes, extent.length - copied);
            jobs.push_back(Job{
                index,
                extent.file_offset + copied,
                extent.destination + copied,
                static_cast<size_t>(count),
            });
            copied += count;
        }
    }
    return jobs;
}

std::vector<CaptureJob> make_capture_jobs(
    const coldsnap_capture_extent* extents,
    size_t extent_count,
    uint64_t chunk_bytes) {
    std::vector<CaptureJob> jobs;
    for (size_t index = 0; index < extent_count; ++index) {
        const auto& extent = extents[index];
        uint64_t copied = 0;
        while (copied < extent.length) {
            const uint64_t count = std::min(chunk_bytes, extent.length - copied);
            jobs.push_back(CaptureJob{
                index,
                extent.file_offset + copied,
                extent.source + copied,
                static_cast<size_t>(count),
            });
            copied += count;
        }
    }
    return jobs;
}

uint64_t pread_exact(int fd, void* destination, size_t length, uint64_t offset) {
    const auto started = Clock::now();
    size_t completed = 0;
    auto* bytes = static_cast<uint8_t*>(destination);
    while (completed < length) {
        const auto remaining = length - completed;
        const auto current_offset = offset + completed;
        if (current_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
            throw std::runtime_error("file offset exceeds off_t");
        }
        const ssize_t count = ::pread(
            fd,
            bytes + completed,
            remaining,
            static_cast<off_t>(current_offset));
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(system_error("pread"));
        }
        if (count == 0) {
            throw std::runtime_error("snapshot ended before the requested extent");
        }
        completed += static_cast<size_t>(count);
    }
    return elapsed_ns(started);
}

uint64_t pwrite_exact(int fd, const void* source, size_t length, uint64_t offset) {
    const auto started = Clock::now();
    size_t completed = 0;
    const auto* bytes = static_cast<const uint8_t*>(source);
    while (completed < length) {
        const auto remaining = length - completed;
        const auto current_offset = offset + completed;
        if (current_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
            throw std::runtime_error("file offset exceeds off_t");
        }
        const ssize_t count = ::pwrite(
            fd,
            bytes + completed,
            remaining,
            static_cast<off_t>(current_offset));
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(system_error("pwrite"));
        }
        if (count == 0) {
            throw std::runtime_error("pwrite returned no data");
        }
        completed += static_cast<size_t>(count);
    }
    return elapsed_ns(started);
}

struct DigestContextDeleter {
    void operator()(EVP_MD_CTX* context) const { EVP_MD_CTX_free(context); }
};

struct CheckState {
    uint32_t mode = COLDSNAP_HYDRATION_VERIFY_NONE;
    uLong crc = crc32(0L, Z_NULL, 0);
    std::unique_ptr<EVP_MD_CTX, DigestContextDeleter> sha;
};

std::vector<CheckState> make_check_states(
    const coldsnap_hydration_extent* extents,
    size_t extent_count) {
    std::vector<CheckState> states(extent_count);
    for (size_t index = 0; index < extent_count; ++index) {
        const uint32_t mode = extents[index].verification;
        if (mode > COLDSNAP_HYDRATION_VERIFY_CRC32_SHA256) {
            throw std::runtime_error("unsupported extent verification mode");
        }
        states[index].mode = mode;
        if (mode == COLDSNAP_HYDRATION_VERIFY_CRC32_SHA256) {
            states[index].sha.reset(EVP_MD_CTX_new());
            if (!states[index].sha ||
                EVP_DigestInit_ex(states[index].sha.get(), EVP_sha256(), nullptr) != 1) {
                throw std::runtime_error("cannot initialize SHA-256 verifier");
            }
        }
    }
    return states;
}

void update_checksum(CheckState& state, const void* data, size_t length) {
    if (state.mode == COLDSNAP_HYDRATION_VERIFY_NONE) {
        return;
    }
    const auto* bytes = static_cast<const Bytef*>(data);
    size_t completed = 0;
    while (completed < length) {
        const size_t count = std::min<size_t>(
            length - completed,
            std::numeric_limits<uInt>::max());
        state.crc = crc32(state.crc, bytes + completed, static_cast<uInt>(count));
        completed += count;
    }
    if (state.mode == COLDSNAP_HYDRATION_VERIFY_CRC32_SHA256 &&
        EVP_DigestUpdate(state.sha.get(), data, length) != 1) {
        throw std::runtime_error("cannot update SHA-256 verifier");
    }
}

uint64_t finish_checksums(
    std::vector<CheckState>& states,
    const coldsnap_hydration_extent* extents) {
    uint64_t verified = 0;
    for (size_t index = 0; index < states.size(); ++index) {
        auto& state = states[index];
        if (state.mode == COLDSNAP_HYDRATION_VERIFY_NONE) {
            continue;
        }
        if (static_cast<uint32_t>(state.crc) != extents[index].expected_crc32) {
            throw std::runtime_error(
                "CRC32 mismatch for extent " + std::to_string(index));
        }
        if (state.mode == COLDSNAP_HYDRATION_VERIFY_CRC32_SHA256) {
            std::array<uint8_t, 32> actual{};
            unsigned int length = 0;
            if (EVP_DigestFinal_ex(state.sha.get(), actual.data(), &length) != 1 ||
                length != actual.size()) {
                throw std::runtime_error("cannot finalize SHA-256 verifier");
            }
            if (!std::equal(
                    actual.begin(),
                    actual.end(),
                    extents[index].expected_sha256)) {
                throw std::runtime_error(
                    "SHA-256 mismatch for extent " + std::to_string(index));
            }
        }
        ++verified;
    }
    return verified;
}

std::vector<CheckState> make_capture_check_states(
    const coldsnap_capture_extent* extents,
    size_t extent_count) {
    std::vector<CheckState> states(extent_count);
    for (size_t index = 0; index < extent_count; ++index) {
        const uint32_t mode = extents[index].checksum;
        if (mode > COLDSNAP_CAPTURE_CHECKSUM_CRC32_SHA256) {
            throw std::runtime_error("unsupported capture checksum mode");
        }
        states[index].mode = mode;
        if (mode == COLDSNAP_CAPTURE_CHECKSUM_CRC32_SHA256) {
            states[index].sha.reset(EVP_MD_CTX_new());
            if (!states[index].sha ||
                EVP_DigestInit_ex(states[index].sha.get(), EVP_sha256(), nullptr) != 1) {
                throw std::runtime_error("cannot initialize capture SHA-256");
            }
        }
    }
    return states;
}

uint64_t finish_capture_checksums(
    std::vector<CheckState>& states,
    coldsnap_capture_digest* digests) {
    uint64_t checksummed = 0;
    for (size_t index = 0; index < states.size(); ++index) {
        auto& state = states[index];
        if (state.mode == COLDSNAP_CAPTURE_CHECKSUM_NONE) {
            continue;
        }
        digests[index].crc32 = static_cast<uint32_t>(state.crc);
        if (state.mode == COLDSNAP_CAPTURE_CHECKSUM_CRC32_SHA256) {
            unsigned int length = 0;
            if (EVP_DigestFinal_ex(
                    state.sha.get(), digests[index].sha256, &length) != 1 ||
                length != sizeof(digests[index].sha256)) {
                throw std::runtime_error("cannot finalize capture SHA-256");
            }
        }
        ++checksummed;
    }
    return checksummed;
}

std::unique_ptr<EVP_MD_CTX, DigestContextDeleter> make_sha256_context(
    const char* operation) {
    std::unique_ptr<EVP_MD_CTX, DigestContextDeleter> context(EVP_MD_CTX_new());
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error(std::string("cannot initialize ") + operation);
    }
    return context;
}

void update_zeros(EVP_MD_CTX* context, uint64_t length) {
    static const std::array<uint8_t, 4096> zeros{};
    while (length > 0) {
        const size_t count = static_cast<size_t>(
            std::min<uint64_t>(length, zeros.size()));
        if (EVP_DigestUpdate(context, zeros.data(), count) != 1) {
            throw std::runtime_error("cannot update capture file SHA-256 padding");
        }
        length -= count;
    }
}

void finish_sha256(EVP_MD_CTX* context, uint8_t destination[32], const char* operation) {
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(context, destination, &length) != 1 || length != 32) {
        throw std::runtime_error(std::string("cannot finalize ") + operation);
    }
}

class Stage {
  public:
    explicit Stage(size_t size) : size_(size) {
        void* value = nullptr;
        const int result = ::posix_memalign(&value, 4096, size);
        if (result != 0) {
            throw std::runtime_error(
                "cannot allocate aligned host stage: " +
                std::string(std::strerror(result)));
        }
        data_ = value;
        try {
            check_cuda(
                cudaHostRegister(data_, size_, cudaHostRegisterDefault),
                "cudaHostRegister");
            registered_ = true;
            check_cuda(
                cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
                "cudaEventCreateWithFlags");
        } catch (...) {
            if (registered_) {
                cudaHostUnregister(data_);
            }
            std::free(data_);
            throw;
        }
    }
    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;
    ~Stage() {
        if (read.valid()) {
            read.wait();
        }
        if (write.valid()) {
            write.wait();
        }
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
        if (registered_) {
            cudaHostUnregister(data_);
        }
        std::free(data_);
    }
    void* data() const { return data_; }
    cudaEvent_t event() const { return event_; }

    std::future<uint64_t> read;
    std::future<uint64_t> write;
    Job job{};
    CaptureJob capture_job{};
    bool active = false;
    bool write_active = false;

  private:
    void* data_ = nullptr;
    size_t size_ = 0;
    bool registered_ = false;
    cudaEvent_t event_ = nullptr;
};

class StagedTransport {
  public:
    StagedTransport(size_t stage_size, size_t depth)
        : stage_size_(stage_size) {
        stages_.reserve(depth);
        for (size_t index = 0; index < depth; ++index) {
            stages_.push_back(std::make_unique<Stage>(stage_size));
        }
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }
    StagedTransport(const StagedTransport&) = delete;
    StagedTransport& operator=(const StagedTransport&) = delete;
    ~StagedTransport() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    bool compatible(size_t stage_size, size_t depth) const {
        return stage_size_ == stage_size && stages_.size() >= depth;
    }
    Stage& stage(size_t index) { return *stages_.at(index); }
    cudaStream_t stream() const { return stream_; }

  private:
    size_t stage_size_ = 0;
    std::vector<std::unique_ptr<Stage>> stages_;
    cudaStream_t stream_ = nullptr;
};

class CaptureOperationGuard {
  public:
    CaptureOperationGuard(StagedTransport& transport, size_t depth)
        : transport_(transport), depth_(depth) {}
    CaptureOperationGuard(const CaptureOperationGuard&) = delete;
    CaptureOperationGuard& operator=(const CaptureOperationGuard&) = delete;
    ~CaptureOperationGuard() {
        // A failed checksum or I/O operation must not leave a device-to-host
        // copy or positional write using a stage after the capture fd closes.
        (void)cudaStreamSynchronize(transport_.stream());
        for (size_t index = 0; index < depth_; ++index) {
            Stage& stage = transport_.stage(index);
            if (stage.write_active && stage.write.valid()) {
                stage.write.wait();
            }
            stage.write_active = false;
        }
    }

  private:
    StagedTransport& transport_;
    size_t depth_;
};

thread_local std::unique_ptr<StagedTransport> staged_transport;

StagedTransport& reusable_staged_transport(size_t stage_size, size_t depth) {
    if (!staged_transport || !staged_transport->compatible(stage_size, depth)) {
        staged_transport = std::make_unique<StagedTransport>(stage_size, depth);
    }
    return *staged_transport;
}

void validate_direct_jobs(const std::vector<Job>& jobs) {
    for (const auto& job : jobs) {
        if (job.file_offset % 4096 != 0 || job.length % 4096 != 0) {
            throw std::runtime_error(
                "direct hydration requires 4096-byte aligned offsets and lengths");
        }
    }
}

void hydrate_staged(
    const char* path,
    const coldsnap_hydration_extent* extents,
    size_t extent_count,
    const coldsnap_hydration_options& options,
    coldsnap_hydration_result& result) {
    const auto initialization_started = Clock::now();
    const int flags = O_RDONLY |
        (options.backend == COLDSNAP_HYDRATION_DIRECT ? O_DIRECT : 0);
    FileDescriptor fd(::open(path, flags));
    if (fd.get() < 0) {
        throw std::runtime_error(system_error("open hydration blob"));
    }
    const auto jobs = make_jobs(extents, extent_count, options.chunk_bytes);
    if (options.backend == COLDSNAP_HYDRATION_DIRECT) {
        validate_direct_jobs(jobs);
    }
    const size_t depth = std::min<size_t>(options.queue_depth, jobs.size());
    StagedTransport& transport = reusable_staged_transport(
        static_cast<size_t>(options.chunk_bytes), depth);
    cudaStream_t stream = transport.stream();
    auto checks = make_check_states(extents, extent_count);
    result.initialization_ns = elapsed_ns(initialization_started);

    size_t next_job = 0;
    auto submit = [&](Stage& stage, const Job& job) {
        stage.job = job;
        stage.active = true;
        stage.read = std::async(
            std::launch::async,
            [fd_value = fd.get(), pointer = stage.data(), job]() {
                return pread_exact(
                    fd_value, pointer, job.length, job.file_offset);
            });
    };
    for (size_t index = 0; index < depth; ++index) {
        submit(transport.stage(index), jobs[next_job++]);
    }

    size_t completed = 0;
    while (completed < jobs.size()) {
        Stage& stage = transport.stage(completed % depth);
        const auto wait_started = Clock::now();
        result.io_service_ns += stage.read.get();
        result.io_wait_ns += elapsed_ns(wait_started);

        const auto checksum_started = Clock::now();
        update_checksum(
            checks[stage.job.extent_index], stage.data(), stage.job.length);
        result.checksum_ns += elapsed_ns(checksum_started);

        const auto copy_started = Clock::now();
        check_cuda(
            cudaMemcpyAsync(
                reinterpret_cast<void*>(stage.job.destination),
                stage.data(),
                stage.job.length,
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync");
        check_cuda(cudaEventRecord(stage.event(), stream), "cudaEventRecord");
        result.cuda_enqueue_ns += elapsed_ns(copy_started);
        ++completed;

        if (next_job < jobs.size()) {
            const auto synchronize_started = Clock::now();
            check_cuda(
                cudaEventSynchronize(stage.event()),
                "cudaEventSynchronize");
            result.cuda_synchronize_ns += elapsed_ns(synchronize_started);
            submit(stage, jobs[next_job++]);
        }
    }
    const auto synchronize_started = Clock::now();
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    result.cuda_synchronize_ns += elapsed_ns(synchronize_started);
    result.verified_extents = finish_checksums(checks, extents);
}

void validate_direct_capture_jobs(const std::vector<CaptureJob>& jobs) {
    for (const auto& job : jobs) {
        if (job.file_offset % 4096 != 0 || job.length % 4096 != 0) {
            throw std::runtime_error(
                "direct capture requires 4096-byte aligned offsets and lengths");
        }
    }
}

bool direct_capture_jobs_compatible(const std::vector<CaptureJob>& jobs) {
    return std::all_of(jobs.begin(), jobs.end(), [](const CaptureJob& job) {
        return job.file_offset % 4096 == 0 && job.length % 4096 == 0;
    });
}

void prepare_capture_file(int fd, uint64_t file_bytes) {
    if (file_bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        throw std::runtime_error("capture file size exceeds off_t");
    }
    if (::ftruncate(fd, static_cast<off_t>(file_bytes)) != 0) {
        throw std::runtime_error(system_error("ftruncate capture blob"));
    }
    const int status = ::posix_fallocate(fd, 0, static_cast<off_t>(file_bytes));
    if (status != 0 && status != EINVAL && status != EOPNOTSUPP &&
        status != ENOSYS) {
        throw std::runtime_error(
            "posix_fallocate capture blob: " + std::string(std::strerror(status)));
    }
}

void verify_capture_file(
    const char* path,
    const coldsnap_capture_extent* extents,
    size_t extent_count,
    const coldsnap_capture_options& options,
    const coldsnap_capture_digest* expected,
    const coldsnap_capture_result& result) {
    const int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW |
        (options.backend == COLDSNAP_CAPTURE_DIRECT ? O_DIRECT : 0);
    FileDescriptor fd(::open(path, flags));
    if (fd.get() < 0) {
        throw std::runtime_error(system_error("open capture blob for verification"));
    }
    const auto jobs = make_capture_jobs(extents, extent_count, options.chunk_bytes);
    std::vector<CheckState> checks = make_capture_check_states(extents, extent_count);
    StagedTransport& transport = reusable_staged_transport(
        static_cast<size_t>(options.chunk_bytes), 1);
    void* buffer = transport.stage(0).data();
    for (const auto& job : jobs) {
        pread_exact(fd.get(), buffer, job.length, job.file_offset);
        update_checksum(checks[job.extent_index], buffer, job.length);
    }
    std::vector<coldsnap_capture_digest> observed(extent_count);
    finish_capture_checksums(checks, observed.data());
    for (size_t index = 0; index < extent_count; ++index) {
        if (extents[index].checksum == COLDSNAP_CAPTURE_CHECKSUM_NONE) {
            continue;
        }
        if (observed[index].crc32 != expected[index].crc32) {
            throw std::runtime_error(
                "capture readback CRC32 mismatch for extent " + std::to_string(index));
        }
        if (extents[index].checksum == COLDSNAP_CAPTURE_CHECKSUM_CRC32_SHA256 &&
            !std::equal(
                std::begin(observed[index].sha256),
                std::end(observed[index].sha256),
                expected[index].sha256)) {
            throw std::runtime_error(
                "capture readback SHA-256 mismatch for extent " + std::to_string(index));
        }
    }
    if ((options.flags & COLDSNAP_CAPTURE_FILE_SHA256) != 0) {
        auto sha = make_sha256_context("capture readback file SHA-256");
        uint64_t offset = 0;
        while (offset < result.file_bytes) {
            const size_t count = static_cast<size_t>(std::min<uint64_t>(
                options.chunk_bytes, result.file_bytes - offset));
            pread_exact(fd.get(), buffer, count, offset);
            if (EVP_DigestUpdate(sha.get(), buffer, count) != 1) {
                throw std::runtime_error("cannot update capture readback file SHA-256");
            }
            offset += count;
        }
        std::array<uint8_t, 32> observed_file{};
        finish_sha256(
            sha.get(), observed_file.data(), "capture readback file SHA-256");
        if (!std::equal(
                observed_file.begin(), observed_file.end(), result.file_sha256)) {
            throw std::runtime_error("capture readback file SHA-256 mismatch");
        }
    }
}

void capture_staged(
    const char* path,
    const coldsnap_capture_extent* extents,
    size_t extent_count,
    const coldsnap_capture_options& options,
    coldsnap_capture_digest* digests,
    coldsnap_capture_result& result) {
    const auto initialization_started = Clock::now();
    const auto jobs = make_capture_jobs(extents, extent_count, options.chunk_bytes);
    uint32_t backend = options.backend;
    if (backend == COLDSNAP_CAPTURE_AUTO) {
        backend = direct_capture_jobs_compatible(jobs)
            ? COLDSNAP_CAPTURE_DIRECT
            : COLDSNAP_CAPTURE_BUFFERED;
    }
    const int base_flags = O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW;
    int descriptor = ::open(
        path,
        base_flags | (backend == COLDSNAP_CAPTURE_DIRECT ? O_DIRECT : 0),
        0600);
    if (descriptor < 0 && options.backend == COLDSNAP_CAPTURE_AUTO &&
        backend == COLDSNAP_CAPTURE_DIRECT &&
        (errno == EINVAL || errno == EOPNOTSUPP || errno == ENOTSUP)) {
        backend = COLDSNAP_CAPTURE_BUFFERED;
        descriptor = ::open(path, base_flags, 0600);
    }
    FileDescriptor fd(descriptor);
    if (fd.get() < 0) {
        throw std::runtime_error(system_error("open capture blob"));
    }
    result.backend = backend;
    prepare_capture_file(fd.get(), result.file_bytes);
    if (backend == COLDSNAP_CAPTURE_DIRECT) {
        validate_direct_capture_jobs(jobs);
    }
    const size_t depth = std::min<size_t>(options.queue_depth, jobs.size());
    StagedTransport& transport = reusable_staged_transport(
        static_cast<size_t>(options.chunk_bytes), depth);
    cudaStream_t stream = transport.stream();
    CaptureOperationGuard operation_guard(transport, depth);
    auto checks = make_capture_check_states(extents, extent_count);
    std::unique_ptr<EVP_MD_CTX, DigestContextDeleter> file_sha;
    if ((options.flags & COLDSNAP_CAPTURE_FILE_SHA256) != 0) {
        file_sha = make_sha256_context("capture file SHA-256");
    }
    result.initialization_ns = elapsed_ns(initialization_started);

    auto wait_write = [&](Stage& stage) {
        if (!stage.write_active) {
            return;
        }
        const auto wait_started = Clock::now();
        stage.write_active = false;
        result.io_service_ns += stage.write.get();
        result.io_wait_ns += elapsed_ns(wait_started);
    };
    auto submit_copy = [&](Stage& stage, const CaptureJob& job) {
        stage.capture_job = job;
        const auto copy_started = Clock::now();
        check_cuda(
            cudaMemcpyAsync(
                stage.data(),
                reinterpret_cast<const void*>(job.source),
                job.length,
                cudaMemcpyDeviceToHost,
                stream),
            "cudaMemcpyAsync device-to-host");
        check_cuda(cudaEventRecord(stage.event(), stream), "cudaEventRecord capture");
        result.cuda_enqueue_ns += elapsed_ns(copy_started);
    };

    size_t next_job = 0;
    for (size_t index = 0; index < depth; ++index) {
        Stage& stage = transport.stage(index);
        wait_write(stage);
        submit_copy(stage, jobs[next_job++]);
    }

    uint64_t file_position = 0;
    size_t completed = 0;
    while (completed < jobs.size()) {
        Stage& stage = transport.stage(completed % depth);
        if (completed >= depth) {
            wait_write(stage);
            if (next_job >= jobs.size()) {
                throw std::runtime_error("capture staging schedule is inconsistent");
            }
            submit_copy(stage, jobs[next_job++]);
        }
        const auto synchronize_started = Clock::now();
        check_cuda(
            cudaEventSynchronize(stage.event()),
            "cudaEventSynchronize capture");
        result.cuda_synchronize_ns += elapsed_ns(synchronize_started);

        const auto checksum_started = Clock::now();
        const CaptureJob job = stage.capture_job;
        update_checksum(checks[job.extent_index], stage.data(), job.length);
        if (file_sha) {
            if (job.file_offset < file_position) {
                throw std::runtime_error("capture jobs are not ordered by file offset");
            }
            update_zeros(file_sha.get(), job.file_offset - file_position);
            if (EVP_DigestUpdate(file_sha.get(), stage.data(), job.length) != 1) {
                throw std::runtime_error("cannot update capture file SHA-256");
            }
            file_position = job.file_offset + job.length;
        }
        result.checksum_ns += elapsed_ns(checksum_started);

        stage.write = std::async(
            std::launch::async,
            [fd_value = fd.get(), pointer = stage.data(), job]() {
                return pwrite_exact(fd_value, pointer, job.length, job.file_offset);
            });
        stage.write_active = true;
        ++completed;
    }
    for (size_t index = 0; index < depth; ++index) {
        wait_write(transport.stage(index));
    }
    result.checksummed_extents = finish_capture_checksums(checks, digests);
    if (file_sha) {
        update_zeros(file_sha.get(), result.file_bytes - file_position);
        finish_sha256(file_sha.get(), result.file_sha256, "capture file SHA-256");
    }

    const auto durability_started = Clock::now();
    if (::fdatasync(fd.get()) != 0) {
        throw std::runtime_error(system_error("fdatasync capture blob"));
    }
    result.durability_ns = elapsed_ns(durability_started);
    if ((options.flags & COLDSNAP_CAPTURE_VERIFY_READBACK) != 0) {
        (void)::posix_fadvise(fd.get(), 0, 0, POSIX_FADV_DONTNEED);
        const auto verification_started = Clock::now();
        verify_capture_file(path, extents, extent_count, options, digests, result);
        result.verification_ns = elapsed_ns(verification_started);
    }
}

struct CufileSymbols {
    using DriverOpen = CUfileError_t (*)();
    using HandleRegister = CUfileError_t (*)(CUfileHandle_t*, CUfileDescr_t*);
    using HandleDeregister = void (*)(CUfileHandle_t);
    using BufferRegister = CUfileError_t (*)(const void*, size_t, int);
    using BufferDeregister = CUfileError_t (*)(const void*);
    using Read = ssize_t (*)(CUfileHandle_t, void*, size_t, off_t, off_t);

    void* library = nullptr;
    DriverOpen driver_open_fn = nullptr;
    HandleRegister handle_register_fn = nullptr;
    HandleDeregister handle_deregister_fn = nullptr;
    BufferRegister buffer_register_fn = nullptr;
    BufferDeregister buffer_deregister_fn = nullptr;
    Read read_fn = nullptr;
    bool driver_open = false;
    std::mutex mutex;
};

CufileSymbols cufile;

template <typename T>
T resolve_symbol(void* library, const char* name) {
    void* symbol = ::dlsym(library, name);
    if (symbol == nullptr) {
        throw std::runtime_error(
            std::string("libcufile lacks ") + name + ": " + ::dlerror());
    }
    return reinterpret_cast<T>(symbol);
}

void load_cufile_symbols() {
    if (cufile.library != nullptr) {
        return;
    }
    void* library = nullptr;
    for (const char* candidate : {"libcufile.so.0", "libcufile.so"}) {
        library = ::dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
        if (library != nullptr) {
            break;
        }
    }
    if (library == nullptr) {
        throw std::runtime_error(
            std::string("cannot load libcufile: ") + ::dlerror());
    }
    try {
        auto driver_open = resolve_symbol<CufileSymbols::DriverOpen>(
            library, "cuFileDriverOpen");
        auto handle_register = resolve_symbol<CufileSymbols::HandleRegister>(
            library, "cuFileHandleRegister");
        auto handle_deregister = resolve_symbol<CufileSymbols::HandleDeregister>(
            library, "cuFileHandleDeregister");
        auto buffer_register = resolve_symbol<CufileSymbols::BufferRegister>(
            library, "cuFileBufRegister");
        auto buffer_deregister = resolve_symbol<CufileSymbols::BufferDeregister>(
            library, "cuFileBufDeregister");
        auto read = resolve_symbol<CufileSymbols::Read>(library, "cuFileRead");
        cufile.driver_open_fn = driver_open;
        cufile.handle_register_fn = handle_register;
        cufile.handle_deregister_fn = handle_deregister;
        cufile.buffer_register_fn = buffer_register;
        cufile.buffer_deregister_fn = buffer_deregister;
        cufile.read_fn = read;
        cufile.library = library;
    } catch (...) {
        ::dlclose(library);
        throw;
    }
}

void check_cufile(CUfileError_t status, const char* operation) {
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuFile error " +
            std::to_string(static_cast<int>(status.err)) + " and CUDA error " +
            std::to_string(static_cast<int>(status.cu_err)));
    }
}

uint64_t initialize_cufile() {
    const auto started = Clock::now();
    std::lock_guard<std::mutex> guard(cufile.mutex);
    load_cufile_symbols();
    if (!cufile.driver_open) {
        check_cufile(cufile.driver_open_fn(), "cuFileDriverOpen");
        cufile.driver_open = true;
    }
    return elapsed_ns(started);
}

class CufileHandle {
  public:
    explicit CufileHandle(int fd) {
        CUfileDescr_t descriptor{};
        descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        descriptor.handle.fd = fd;
        check_cufile(
            cufile.handle_register_fn(&handle_, &descriptor),
            "cuFileHandleRegister");
    }
    CufileHandle(const CufileHandle&) = delete;
    CufileHandle& operator=(const CufileHandle&) = delete;
    ~CufileHandle() {
        if (handle_ != nullptr) {
            cufile.handle_deregister_fn(handle_);
        }
    }
    CUfileHandle_t get() const { return handle_; }

  private:
    CUfileHandle_t handle_ = nullptr;
};

class RegisteredBuffers {
  public:
    RegisteredBuffers(
        const coldsnap_hydration_extent* extents,
        size_t extent_count,
        bool enabled) {
        if (!enabled) {
            return;
        }
        try {
            for (size_t index = 0; index < extent_count; ++index) {
                const auto& extent = extents[index];
                check_cufile(
                    cufile.buffer_register_fn(
                        reinterpret_cast<const void*>(extent.destination),
                        static_cast<size_t>(extent.length),
                        0),
                    "cuFileBufRegister");
                pointers_.push_back(extent.destination);
            }
        } catch (...) {
            deregister();
            throw;
        }
    }
    RegisteredBuffers(const RegisteredBuffers&) = delete;
    RegisteredBuffers& operator=(const RegisteredBuffers&) = delete;
    ~RegisteredBuffers() { deregister(); }

  private:
    void deregister() noexcept {
        for (auto pointer = pointers_.rbegin(); pointer != pointers_.rend(); ++pointer) {
            cufile.buffer_deregister_fn(reinterpret_cast<const void*>(*pointer));
        }
        pointers_.clear();
    }

    std::vector<uintptr_t> pointers_;
};

uint64_t cufile_read_job(
    CUfileHandle_t handle,
    Job job,
    uintptr_t destination_base,
    int cuda_device) {
    if (cuda_device >= 0) {
        check_cuda(cudaSetDevice(cuda_device), "cudaSetDevice in GDS reader");
    }
    const uint64_t destination_offset = job.destination - destination_base;
    if (job.file_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        destination_offset >
            static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        job.length > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
        throw std::runtime_error("GDS job exceeds cuFile offset or length limits");
    }
    const auto started = Clock::now();
    const ssize_t result = cufile.read_fn(
        handle,
        reinterpret_cast<void*>(destination_base),
        job.length,
        static_cast<off_t>(job.file_offset),
        static_cast<off_t>(destination_offset));
    if (result != static_cast<ssize_t>(job.length)) {
        throw std::runtime_error(
            "cuFileRead returned " + std::to_string(result) +
            " bytes; expected " + std::to_string(job.length));
    }
    return elapsed_ns(started);
}

void hydrate_gds(
    const char* path,
    const coldsnap_hydration_extent* extents,
    size_t extent_count,
    const coldsnap_hydration_options& options,
    coldsnap_hydration_result& result) {
    for (size_t index = 0; index < extent_count; ++index) {
        if (extents[index].verification != COLDSNAP_HYDRATION_VERIFY_NONE) {
            throw std::runtime_error(
                "GDS hydration requires preverified extents");
        }
    }
    result.initialization_ns = initialize_cufile();
    FileDescriptor fd(::open(path, O_RDONLY | O_DIRECT));
    if (fd.get() < 0) {
        throw std::runtime_error(system_error("open GDS hydration blob"));
    }
    CufileHandle handle(fd.get());
    RegisteredBuffers buffers(
        extents,
        extent_count,
        (options.flags & COLDSNAP_HYDRATION_REGISTER_DEVICE_BUFFERS) != 0);
    const auto jobs = make_jobs(extents, extent_count, options.chunk_bytes);
    const size_t depth = std::min<size_t>(options.queue_depth, jobs.size());
    std::vector<std::future<uint64_t>> pending;
    pending.reserve(depth);
    size_t next_job = 0;
    while (next_job < depth) {
        const Job job = jobs[next_job++];
        const uintptr_t destination_base =
            extents[job.extent_index].destination;
        pending.push_back(std::async(
            std::launch::async,
            [handle_value = handle.get(), job, destination_base,
             device = options.cuda_device]() {
                return cufile_read_job(
                    handle_value, job, destination_base, device);
            }));
    }
    size_t completed = 0;
    while (completed < jobs.size()) {
        const size_t slot = completed % depth;
        const auto wait_started = Clock::now();
        result.io_service_ns += pending[slot].get();
        result.io_wait_ns += elapsed_ns(wait_started);
        ++completed;
        if (next_job < jobs.size()) {
            const Job job = jobs[next_job++];
            const uintptr_t destination_base =
                extents[job.extent_index].destination;
            pending[slot] = std::async(
                std::launch::async,
                [handle_value = handle.get(), job, destination_base,
                 device = options.cuda_device]() {
                    return cufile_read_job(
                        handle_value, job, destination_base, device);
                });
        }
    }
    const auto synchronize_started = Clock::now();
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after GDS");
    result.cuda_synchronize_ns += elapsed_ns(synchronize_started);
}

void validate_request(
    const char* path,
    const coldsnap_hydration_extent* extents,
    size_t extent_count,
    const coldsnap_hydration_options* options,
    coldsnap_hydration_result* result) {
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error("hydration path is empty");
    }
    if (extents == nullptr || extent_count == 0) {
        throw std::runtime_error("hydration has no extents");
    }
    if (options == nullptr || result == nullptr) {
        throw std::runtime_error("hydration options and result are required");
    }
    if (options->abi_version != COLDSNAP_HYDRATION_ABI_VERSION) {
        throw std::runtime_error("unsupported hydration ABI version");
    }
    if (options->backend < COLDSNAP_HYDRATION_BUFFERED ||
        options->backend > COLDSNAP_HYDRATION_GDS) {
        throw std::runtime_error("unsupported hydration backend");
    }
    if (options->queue_depth == 0 || options->queue_depth > 64) {
        throw std::runtime_error("hydration queue depth must be between 1 and 64");
    }
    if (options->chunk_bytes == 0 || options->chunk_bytes % 4096 != 0 ||
        options->chunk_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(
            "hydration chunk size must be positive and 4096-byte aligned");
    }
    if ((options->flags & ~COLDSNAP_HYDRATION_REGISTER_DEVICE_BUFFERS) != 0 ||
        options->reserved != 0) {
        throw std::runtime_error("unsupported hydration options or flags");
    }
    for (size_t index = 0; index < extent_count; ++index) {
        const auto& extent = extents[index];
        if (extent.destination == 0 || extent.length == 0 ||
            extent.length > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            extent.file_offset > std::numeric_limits<uint64_t>::max() - extent.length ||
            extent.destination > std::numeric_limits<uintptr_t>::max() - extent.length) {
            throw std::runtime_error(
                "invalid hydration extent " + std::to_string(index));
        }
    }
}

void validate_capture_request(
    const char* path,
    const coldsnap_capture_extent* extents,
    size_t extent_count,
    const coldsnap_capture_options* options,
    const coldsnap_capture_digest* digests,
    const coldsnap_capture_result* result) {
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error("capture path is empty");
    }
    if (extents == nullptr || extent_count == 0) {
        throw std::runtime_error("capture has no extents");
    }
    if (options == nullptr || digests == nullptr || result == nullptr) {
        throw std::runtime_error("capture options, digests, and result are required");
    }
    if (options->abi_version != COLDSNAP_CAPTURE_ABI_VERSION) {
        throw std::runtime_error("unsupported capture ABI version");
    }
    if (options->backend > COLDSNAP_CAPTURE_DIRECT) {
        throw std::runtime_error("unsupported capture backend");
    }
    if (options->queue_depth == 0 || options->queue_depth > 64) {
        throw std::runtime_error("capture queue depth must be between 1 and 64");
    }
    if (options->chunk_bytes == 0 || options->chunk_bytes % 4096 != 0 ||
        options->chunk_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(
            "capture chunk size must be positive and 4096-byte aligned");
    }
    const uint32_t supported_flags =
        COLDSNAP_CAPTURE_FILE_SHA256 | COLDSNAP_CAPTURE_VERIFY_READBACK;
    if ((options->flags & ~supported_flags) != 0 || options->reserved != 0) {
        throw std::runtime_error("unsupported capture options or flags");
    }
    bool has_checksum = false;
    uint64_t previous_end = 0;
    for (size_t index = 0; index < extent_count; ++index) {
        const auto& extent = extents[index];
        if (extent.source == 0 || extent.length == 0 || extent.reserved != 0 ||
            extent.checksum > COLDSNAP_CAPTURE_CHECKSUM_CRC32_SHA256 ||
            extent.length > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            extent.file_offset > std::numeric_limits<uint64_t>::max() - extent.length ||
            extent.source > std::numeric_limits<uintptr_t>::max() - extent.length ||
            extent.file_offset < previous_end) {
            throw std::runtime_error(
                "invalid or overlapping capture extent " + std::to_string(index));
        }
        has_checksum = has_checksum ||
            extent.checksum != COLDSNAP_CAPTURE_CHECKSUM_NONE;
        previous_end = extent.file_offset + extent.length;
    }
    if ((options->flags & COLDSNAP_CAPTURE_VERIFY_READBACK) != 0 &&
        !has_checksum && (options->flags & COLDSNAP_CAPTURE_FILE_SHA256) == 0) {
        throw std::runtime_error(
            "capture readback requires an extent checksum or file SHA-256");
    }
}

}  // namespace

extern "C" int coldsnap_hydrate_file(
    const char* path,
    const coldsnap_hydration_extent* extents,
    size_t extent_count,
    const coldsnap_hydration_options* options,
    coldsnap_hydration_result* result) {
    last_error.clear();
    const auto started = Clock::now();
    try {
        validate_request(path, extents, extent_count, options, result);
        *result = {};
        result->abi_version = COLDSNAP_HYDRATION_ABI_VERSION;
        result->backend = options->backend;
        const auto jobs = make_jobs(extents, extent_count, options->chunk_bytes);
        result->chunks = jobs.size();
        for (size_t index = 0; index < extent_count; ++index) {
            result->bytes += extents[index].length;
        }
        if (options->cuda_device >= 0) {
            check_cuda(cudaSetDevice(options->cuda_device), "cudaSetDevice");
        }
        if (options->backend == COLDSNAP_HYDRATION_GDS) {
            hydrate_gds(path, extents, extent_count, *options, *result);
        } else {
            hydrate_staged(path, extents, extent_count, *options, *result);
        }
        result->total_ns = elapsed_ns(started);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
    } catch (...) {
        last_error = "unknown native hydration failure";
    }
    if (result != nullptr) {
        result->total_ns = elapsed_ns(started);
    }
    return -1;
}

extern "C" int coldsnap_capture_file(
    const char* path,
    const coldsnap_capture_extent* extents,
    size_t extent_count,
    const coldsnap_capture_options* options,
    coldsnap_capture_digest* digests,
    coldsnap_capture_result* result) {
    last_error.clear();
    const auto started = Clock::now();
    try {
        validate_capture_request(
            path, extents, extent_count, options, digests, result);
        *result = {};
        std::memset(digests, 0, extent_count * sizeof(*digests));
        result->abi_version = COLDSNAP_CAPTURE_ABI_VERSION;
        result->backend = options->backend;
        const auto jobs = make_capture_jobs(extents, extent_count, options->chunk_bytes);
        result->chunks = jobs.size();
        for (size_t index = 0; index < extent_count; ++index) {
            if (result->bytes > std::numeric_limits<uint64_t>::max() - extents[index].length) {
                throw std::runtime_error("capture byte count overflows uint64");
            }
            result->bytes += extents[index].length;
            result->file_bytes = extents[index].file_offset + extents[index].length;
        }
        if (options->cuda_device >= 0) {
            check_cuda(cudaSetDevice(options->cuda_device), "cudaSetDevice capture");
        }
        capture_staged(path, extents, extent_count, *options, digests, *result);
        result->total_ns = elapsed_ns(started);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
    } catch (...) {
        last_error = "unknown native capture failure";
    }
    if (result != nullptr) {
        result->total_ns = elapsed_ns(started);
    }
    return -1;
}

extern "C" int coldsnap_hydration_backend_available(uint32_t backend) {
    last_error.clear();
    if (backend == COLDSNAP_HYDRATION_BUFFERED ||
        backend == COLDSNAP_HYDRATION_DIRECT) {
        return 1;
    }
    if (backend != COLDSNAP_HYDRATION_GDS) {
        last_error = "unsupported hydration backend";
        return 0;
    }
    try {
        std::lock_guard<std::mutex> guard(cufile.mutex);
        load_cufile_symbols();
        return 1;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

extern "C" int coldsnap_capture_backend_available(uint32_t backend) {
    last_error.clear();
    if (backend == COLDSNAP_CAPTURE_AUTO ||
        backend == COLDSNAP_CAPTURE_BUFFERED ||
        backend == COLDSNAP_CAPTURE_DIRECT) {
        return 1;
    }
    last_error = "unsupported capture backend";
    return 0;
}

extern "C" const char* coldsnap_hydration_last_error(void) {
    return last_error.c_str();
}

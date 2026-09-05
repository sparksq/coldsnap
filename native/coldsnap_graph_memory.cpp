// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

//
// CUDA VMM allocation/remap mechanics are independently adapted from
// torch_memory_saver (Copyright (c) 2024 fzyzcjy, MIT). See
// THIRD_PARTY_NOTICES.md. The lifecycle, error, enumeration, and retry model
// in this file is coldsnap-specific.
#include "coldsnap_graph_memory.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

enum class AllocationState {
    Active,
    Paused,
    Partial,
};

struct Allocation {
    void* address = nullptr;
    size_t raw_size = 0;
    size_t mapped_size = 0;
    int device = -1;
    std::string tag;
    uint64_t pool_id = 0;
    CUmemGenericAllocationHandle handle = 0;
    bool handle_valid = false;
    bool mapped = false;
    bool accessible = false;
};

struct RegionScope {
    std::string tag;
    int device = -1;
    uint64_t pool_id = 0;
};

std::mutex allocation_mutex;
std::unordered_map<void*, Allocation> allocations;
std::atomic<uint64_t> active_scope_count{0};
thread_local std::vector<RegionScope> region_scopes;
thread_local std::string last_error;

using CudaMalloc = cudaError_t (*)(void**, size_t);
using CudaFree = cudaError_t (*)(void*);

std::mutex forwarding_mutex;
CudaMalloc real_cuda_malloc = nullptr;
CudaFree real_cuda_free = nullptr;

void set_error(const std::string& message) {
    last_error = message;
}

std::string driver_error(CUresult result, const char* operation) {
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    std::ostringstream stream;
    stream << operation << " failed";
    if (name != nullptr) {
        stream << " with " << name;
    }
    if (description != nullptr) {
        stream << ": " << description;
    }
    return stream.str();
}

std::string runtime_error(cudaError_t result, const char* operation) {
    std::ostringstream stream;
    stream << operation << " failed with " << cudaGetErrorName(result)
           << ": " << cudaGetErrorString(result);
    return stream.str();
}

template <typename Function>
Function resolve_forwarded(const char* name, Function* cache) {
    std::lock_guard<std::mutex> guard(forwarding_mutex);
    if (*cache != nullptr) {
        return *cache;
    }
    dlerror();
    void* value = dlsym(RTLD_NEXT, name);
    const char* error = dlerror();
    if (value == nullptr || error != nullptr) {
        set_error(
            std::string("dlsym(RTLD_NEXT, ") + name + ") failed: "
            + (error == nullptr ? "unknown error" : error));
        return nullptr;
    }
    *cache = reinterpret_cast<Function>(value);
    return *cache;
}

AllocationState state_of(const Allocation& allocation) {
    if (
        allocation.mapped
        && allocation.handle_valid
        && allocation.accessible
    ) {
        return AllocationState::Active;
    }
    if (
        !allocation.mapped
        && !allocation.handle_valid
        && !allocation.accessible
    ) {
        return AllocationState::Paused;
    }
    return AllocationState::Partial;
}

int public_state(const Allocation& allocation) {
    switch (state_of(allocation)) {
    case AllocationState::Active:
        return COLDSNAP_GRAPH_ALLOCATION_ACTIVE;
    case AllocationState::Paused:
        return COLDSNAP_GRAPH_ALLOCATION_PAUSED;
    case AllocationState::Partial:
        return COLDSNAP_GRAPH_ALLOCATION_PARTIAL;
    }
    return COLDSNAP_GRAPH_ALLOCATION_PARTIAL;
}

bool matches(const Allocation& allocation, const char* tag, int device) {
    return (
        (tag == nullptr || tag[0] == '\0' || allocation.tag == tag)
        && (device < 0 || allocation.device == device)
    );
}

class DeviceGuard {
public:
    explicit DeviceGuard(int requested_device) {
        cudaError_t result = cudaGetDevice(&original_device_);
        if (result != cudaSuccess) {
            set_error(runtime_error(result, "cudaGetDevice"));
            return;
        }
        valid_ = true;
        if (requested_device != original_device_) {
            result = cudaSetDevice(requested_device);
            if (result != cudaSuccess) {
                set_error(runtime_error(result, "cudaSetDevice"));
                valid_ = false;
                return;
            }
            changed_ = true;
        }
    }

    ~DeviceGuard() {
        if (valid_ && changed_) {
            cudaSetDevice(original_device_);
        }
    }

    bool valid() const {
        return valid_;
    }

private:
    int original_device_ = -1;
    bool valid_ = false;
    bool changed_ = false;
};

bool synchronize_device(int device) {
    DeviceGuard guard(device);
    if (!guard.valid()) {
        return false;
    }
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        set_error(runtime_error(result, "cudaDeviceSynchronize"));
        return false;
    }
    return true;
}

bool synchronize_matching_devices(const char* tag, int device) {
    std::set<int> devices;
    {
        std::lock_guard<std::mutex> guard(allocation_mutex);
        for (const auto& item : allocations) {
            if (matches(item.second, tag, device)) {
                devices.insert(item.second.device);
            }
        }
    }
    for (int allocation_device : devices) {
        if (!synchronize_device(allocation_device)) {
            return false;
        }
    }
    return true;
}

bool initialize_driver() {
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuInit"));
        return false;
    }
    return true;
}

bool allocation_properties(
    int device,
    size_t raw_size,
    CUmemAllocationProp* properties,
    size_t* mapped_size
) {
    if (!initialize_driver()) {
        return false;
    }
    CUdevice cuda_device;
    CUresult result = cuDeviceGet(&cuda_device, device);
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuDeviceGet"));
        return false;
    }

    *properties = {};
    properties->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties->location.id = cuda_device;
    properties->requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    size_t granularity = 0;
    result = cuMemGetAllocationGranularity(
        &granularity,
        properties,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM
    );
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemGetAllocationGranularity"));
        return false;
    }
    if (granularity == 0) {
        set_error("cuMemGetAllocationGranularity returned zero");
        return false;
    }
    if (raw_size > std::numeric_limits<size_t>::max() - (granularity - 1)) {
        set_error("CUDA graph allocation size overflows VMM granularity");
        return false;
    }
    *mapped_size = ((raw_size + granularity - 1) / granularity) * granularity;
    return true;
}

bool set_access(const Allocation& allocation) {
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = allocation.device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUresult result = cuMemSetAccess(
        reinterpret_cast<CUdeviceptr>(allocation.address),
        allocation.mapped_size,
        &access,
        1
    );
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemSetAccess"));
        return false;
    }
    return true;
}

cudaError_t allocate_scoped(void** pointer, size_t raw_size, const RegionScope& scope) {
    int current_device = -1;
    cudaError_t runtime_result = cudaGetDevice(&current_device);
    if (runtime_result != cudaSuccess) {
        set_error(runtime_error(runtime_result, "cudaGetDevice"));
        return runtime_result;
    }
    if (scope.device >= 0 && scope.device != current_device) {
        set_error(
            "CUDA graph allocation occurred on device "
            + std::to_string(current_device)
            + " inside a region scoped to device "
            + std::to_string(scope.device));
        return cudaErrorInvalidDevice;
    }

    CUmemAllocationProp properties = {};
    size_t mapped_size = 0;
    if (!allocation_properties(
            current_device, raw_size, &properties, &mapped_size)) {
        return cudaErrorUnknown;
    }

    Allocation allocation;
    allocation.raw_size = raw_size;
    allocation.mapped_size = mapped_size;
    allocation.device = current_device;
    allocation.tag = scope.tag;
    allocation.pool_id = scope.pool_id;

    CUresult result = cuMemCreate(
        &allocation.handle,
        mapped_size,
        &properties,
        0
    );
    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        set_error(driver_error(result, "cuMemCreate"));
        return cudaErrorMemoryAllocation;
    }
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemCreate"));
        return cudaErrorUnknown;
    }
    allocation.handle_valid = true;

    CUdeviceptr address = 0;
    result = cuMemAddressReserve(&address, mapped_size, 0, 0, 0);
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemAddressReserve"));
        cuMemRelease(allocation.handle);
        return cudaErrorUnknown;
    }
    allocation.address = reinterpret_cast<void*>(address);

    result = cuMemMap(address, mapped_size, 0, allocation.handle, 0);
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemMap"));
        cuMemAddressFree(address, mapped_size);
        cuMemRelease(allocation.handle);
        return cudaErrorUnknown;
    }
    allocation.mapped = true;

    if (!set_access(allocation)) {
        cuMemUnmap(address, mapped_size);
        cuMemAddressFree(address, mapped_size);
        cuMemRelease(allocation.handle);
        return cudaErrorUnknown;
    }
    allocation.accessible = true;

    {
        std::lock_guard<std::mutex> guard(allocation_mutex);
        auto inserted = allocations.emplace(allocation.address, allocation);
        if (!inserted.second) {
            set_error("duplicate CUDA graph virtual address reservation");
            cuMemUnmap(address, mapped_size);
            cuMemAddressFree(address, mapped_size);
            cuMemRelease(allocation.handle);
            return cudaErrorUnknown;
        }
    }

    *pointer = allocation.address;
    return cudaSuccess;
}

bool pause_allocation(Allocation* allocation) {
    DeviceGuard guard(allocation->device);
    if (!guard.valid()) {
        return false;
    }

    if (allocation->mapped) {
        allocation->accessible = false;
        CUresult result = cuMemUnmap(
            reinterpret_cast<CUdeviceptr>(allocation->address),
            allocation->mapped_size
        );
        if (result != CUDA_SUCCESS) {
            set_error(driver_error(result, "cuMemUnmap"));
            return false;
        }
        allocation->mapped = false;
    }

    if (allocation->handle_valid) {
        CUresult result = cuMemRelease(allocation->handle);
        if (result != CUDA_SUCCESS) {
            set_error(driver_error(result, "cuMemRelease"));
            return false;
        }
        allocation->handle_valid = false;
        allocation->handle = 0;
    }
    return state_of(*allocation) == AllocationState::Paused;
}

bool resume_allocation(Allocation* allocation) {
    DeviceGuard guard(allocation->device);
    if (!guard.valid()) {
        return false;
    }

    if (state_of(*allocation) == AllocationState::Active) {
        return true;
    }
    if (allocation->mapped) {
        set_error("cannot recover a partially mapped CUDA graph allocation");
        return false;
    }
    if (allocation->handle_valid) {
        CUresult result = cuMemRelease(allocation->handle);
        if (result != CUDA_SUCCESS) {
            set_error(driver_error(result, "cuMemRelease(retry)"));
            return false;
        }
        allocation->handle_valid = false;
        allocation->handle = 0;
    }

    CUmemAllocationProp properties = {};
    size_t mapped_size = 0;
    if (!allocation_properties(
            allocation->device,
            allocation->raw_size,
            &properties,
            &mapped_size)) {
        return false;
    }
    if (mapped_size != allocation->mapped_size) {
        set_error("CUDA VMM granularity changed while restoring graph backing");
        return false;
    }

    CUmemGenericAllocationHandle new_handle = 0;
    CUresult result = cuMemCreate(
        &new_handle,
        allocation->mapped_size,
        &properties,
        0
    );
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemCreate(restore)"));
        return false;
    }

    result = cuMemMap(
        reinterpret_cast<CUdeviceptr>(allocation->address),
        allocation->mapped_size,
        0,
        new_handle,
        0
    );
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemMap(restore)"));
        cuMemRelease(new_handle);
        return false;
    }

    allocation->handle = new_handle;
    allocation->handle_valid = true;
    allocation->mapped = true;
    allocation->accessible = false;
    if (!set_access(*allocation)) {
        cuMemUnmap(
            reinterpret_cast<CUdeviceptr>(allocation->address),
            allocation->mapped_size
        );
        cuMemRelease(new_handle);
        allocation->handle = 0;
        allocation->handle_valid = false;
        allocation->mapped = false;
        return false;
    }
    allocation->accessible = true;
    return true;
}

std::vector<Allocation> matching_allocations(const char* tag, int device) {
    std::vector<Allocation> result;
    std::lock_guard<std::mutex> guard(allocation_mutex);
    for (const auto& item : allocations) {
        if (matches(item.second, tag, device)) {
            result.push_back(item.second);
        }
    }
    std::sort(
        result.begin(),
        result.end(),
        [](const Allocation& left, const Allocation& right) {
            return left.address < right.address;
        }
    );
    return result;
}

}  // namespace

extern "C" cudaError_t cudaMalloc(void** pointer, size_t size) {
    if (pointer == nullptr) {
        set_error("cudaMalloc received a null output pointer");
        return cudaErrorInvalidValue;
    }
    if (region_scopes.empty() || size == 0) {
        CudaMalloc forwarded = resolve_forwarded(
            "cudaMalloc", &real_cuda_malloc);
        if (forwarded == nullptr) {
            return cudaErrorUnknown;
        }
        return forwarded(pointer, size);
    }
    return allocate_scoped(pointer, size, region_scopes.back());
}

extern "C" cudaError_t cudaFree(void* pointer) {
    bool managed = false;
    {
        std::lock_guard<std::mutex> guard(allocation_mutex);
        managed = allocations.count(pointer) != 0;
    }
    if (!managed) {
        CudaFree forwarded = resolve_forwarded("cudaFree", &real_cuda_free);
        if (forwarded == nullptr) {
            return cudaErrorUnknown;
        }
        return forwarded(pointer);
    }

    Allocation allocation;
    {
        std::lock_guard<std::mutex> guard(allocation_mutex);
        allocation = allocations.at(pointer);
    }
    if (!synchronize_device(allocation.device)) {
        return cudaErrorUnknown;
    }

    std::lock_guard<std::mutex> guard(allocation_mutex);
    auto found = allocations.find(pointer);
    if (found == allocations.end()) {
        set_error("managed CUDA graph allocation disappeared during cudaFree");
        return cudaErrorUnknown;
    }
    Allocation& current = found->second;
    DeviceGuard device_guard(current.device);
    if (!device_guard.valid()) {
        return cudaErrorUnknown;
    }
    if (current.mapped) {
        current.accessible = false;
        CUresult result = cuMemUnmap(
            reinterpret_cast<CUdeviceptr>(current.address),
            current.mapped_size
        );
        if (result != CUDA_SUCCESS) {
            set_error(driver_error(result, "cuMemUnmap(cudaFree)"));
            return cudaErrorUnknown;
        }
        current.mapped = false;
    }
    if (current.handle_valid) {
        CUresult result = cuMemRelease(current.handle);
        if (result != CUDA_SUCCESS) {
            set_error(driver_error(result, "cuMemRelease(cudaFree)"));
            return cudaErrorUnknown;
        }
        current.handle_valid = false;
        current.handle = 0;
    }
    CUresult result = cuMemAddressFree(
        reinterpret_cast<CUdeviceptr>(current.address),
        current.mapped_size
    );
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemAddressFree"));
        return cudaErrorUnknown;
    }
    allocations.erase(found);
    return cudaSuccess;
}

extern "C" int coldsnap_graph_region_push(
    const char* tag,
    int device,
    uint64_t pool_id
) {
    if (tag == nullptr || tag[0] == '\0') {
        set_error("CUDA graph region tag must not be empty");
        return -1;
    }
    if (device >= 0) {
        int current_device = -1;
        cudaError_t result = cudaGetDevice(&current_device);
        if (result != cudaSuccess) {
            set_error(runtime_error(result, "cudaGetDevice"));
            return -1;
        }
        if (current_device != device) {
            set_error(
                "CUDA graph region device does not match the current CUDA device");
            return -1;
        }
    }
    region_scopes.push_back(RegionScope{tag, device, pool_id});
    active_scope_count.fetch_add(1, std::memory_order_acq_rel);
    return 0;
}

extern "C" int coldsnap_graph_region_pop(void) {
    if (region_scopes.empty()) {
        set_error("CUDA graph region stack underflow");
        return -1;
    }
    region_scopes.pop_back();
    active_scope_count.fetch_sub(1, std::memory_order_acq_rel);
    return 0;
}

extern "C" int coldsnap_graph_pause(const char* tag, int device) {
    if (active_scope_count.load(std::memory_order_acquire) != 0) {
        set_error("cannot pause CUDA graph backing during active graph capture");
        return -1;
    }
    if (!synchronize_matching_devices(tag, device)) {
        return -1;
    }

    std::lock_guard<std::mutex> guard(allocation_mutex);
    for (auto& item : allocations) {
        Allocation& allocation = item.second;
        if (!matches(allocation, tag, device)) {
            continue;
        }
        if (state_of(allocation) == AllocationState::Paused) {
            continue;
        }
        if (!pause_allocation(&allocation)) {
            return -1;
        }
    }
    return 0;
}

extern "C" int coldsnap_graph_resume(const char* tag, int device) {
    if (active_scope_count.load(std::memory_order_acquire) != 0) {
        set_error("cannot resume CUDA graph backing during active graph capture");
        return -1;
    }

    {
        std::lock_guard<std::mutex> guard(allocation_mutex);
        for (auto& item : allocations) {
            Allocation& allocation = item.second;
            if (!matches(allocation, tag, device)) {
                continue;
            }
            if (!resume_allocation(&allocation)) {
                return -1;
            }
        }
    }
    if (!synchronize_matching_devices(tag, device)) {
        return -1;
    }
    return 0;
}

extern "C" int coldsnap_graph_region_stats(
    const char* tag,
    int device,
    uint64_t* allocation_count,
    uint64_t* raw_bytes,
    uint64_t* mapped_bytes,
    uint64_t* paused_count
) {
    if (
        allocation_count == nullptr
        || raw_bytes == nullptr
        || mapped_bytes == nullptr
        || paused_count == nullptr
    ) {
        set_error("CUDA graph stats received a null output pointer");
        return -1;
    }
    *allocation_count = 0;
    *raw_bytes = 0;
    *mapped_bytes = 0;
    *paused_count = 0;

    std::lock_guard<std::mutex> guard(allocation_mutex);
    for (const auto& item : allocations) {
        const Allocation& allocation = item.second;
        if (!matches(allocation, tag, device)) {
            continue;
        }
        ++*allocation_count;
        *raw_bytes += allocation.raw_size;
        *mapped_bytes += allocation.mapped_size;
        if (state_of(allocation) != AllocationState::Active) {
            ++*paused_count;
        }
    }
    return 0;
}

extern "C" int coldsnap_graph_allocation_at(
    const char* tag,
    int device,
    uint64_t index,
    uintptr_t* address,
    uint64_t* raw_bytes,
    uint64_t* mapped_bytes,
    int* allocation_device,
    uint64_t* pool_id,
    int* state
) {
    if (
        address == nullptr
        || raw_bytes == nullptr
        || mapped_bytes == nullptr
        || allocation_device == nullptr
        || pool_id == nullptr
        || state == nullptr
    ) {
        set_error("CUDA graph allocation inspection received a null output pointer");
        return -1;
    }
    std::vector<Allocation> found = matching_allocations(tag, device);
    if (index >= found.size()) {
        set_error("CUDA graph allocation index is out of range");
        return -1;
    }
    const Allocation& allocation = found[index];
    *address = reinterpret_cast<uintptr_t>(allocation.address);
    *raw_bytes = allocation.raw_size;
    *mapped_bytes = allocation.mapped_size;
    *allocation_device = allocation.device;
    *pool_id = allocation.pool_id;
    *state = public_state(allocation);
    return 0;
}

extern "C" int coldsnap_graph_interposition_active(void) {
    dlerror();
    void* resolved = dlsym(RTLD_DEFAULT, "cudaMalloc");
    if (dlerror() != nullptr || resolved == nullptr) {
        return 0;
    }
    return resolved == reinterpret_cast<void*>(&cudaMalloc) ? 1 : 0;
}

extern "C" const char* coldsnap_graph_last_error(void) {
    return last_error.c_str();
}

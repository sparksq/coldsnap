// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

#include "coldsnap_cuda_epoch.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

thread_local std::string last_error;
std::mutex published_contexts_mutex;
std::unordered_map<int, CUcontext> published_contexts;

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

struct ReboundAllocation {
    coldsnap_cuda_epoch_allocation input{};
    CUmemGenericAllocationHandle handle = 0;
    bool handle_valid = false;
    bool mapped = false;
};

bool select_context(
    int device,
    std::unordered_map<int, CUcontext>* contexts
) {
    auto found = contexts->find(device);
    if (found == contexts->end()) {
        CUdevice cuda_device = 0;
        CUresult result = cuDeviceGet(&cuda_device, device);
        if (result != CUDA_SUCCESS) {
            set_error(driver_error(result, "cuDeviceGet(epoch)"));
            return false;
        }
        CUcontext context = nullptr;
        result = cuDevicePrimaryCtxRetain(&context, cuda_device);
        if (result != CUDA_SUCCESS) {
            set_error(driver_error(result, "cuDevicePrimaryCtxRetain(epoch)"));
            return false;
        }
        found = contexts->emplace(device, context).first;
    }
    CUresult result = cuCtxSetCurrent(found->second);
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuCtxSetCurrent(epoch)"));
        return false;
    }
    return true;
}

bool allocation_properties(
    int device,
    CUmemAllocationProp* properties,
    size_t* granularity
) {
    *properties = {};
    properties->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties->location.id = device;
    properties->requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    properties->allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

    CUresult result = cuMemGetAllocationGranularity(
        granularity,
        properties,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM
    );
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemGetAllocationGranularity(epoch)"));
        return false;
    }
    if (*granularity == 0) {
        set_error("cuMemGetAllocationGranularity(epoch) returned zero");
        return false;
    }
    return true;
}

void rollback(
    std::vector<ReboundAllocation>* rebound,
    std::unordered_map<int, CUcontext>* contexts
) {
    for (auto item = rebound->rbegin(); item != rebound->rend(); ++item) {
        if (!select_context(item->input.device, contexts)) {
            continue;
        }
        if (item->mapped) {
            (void)cuMemUnmap(
                static_cast<CUdeviceptr>(item->input.address),
                static_cast<size_t>(item->input.mapped_bytes)
            );
            item->mapped = false;
        }
        if (item->handle_valid) {
            (void)cuMemRelease(item->handle);
            item->handle_valid = false;
        }
        if (item->input.handle_box_address != 0) {
            auto* handle_box = reinterpret_cast<CUmemGenericAllocationHandle*>(
                item->input.handle_box_address);
            *handle_box = 0;
        }
    }
}

void initialize_context_probe(
    int device,
    coldsnap_cuda_epoch_context_probe* probe
) {
    *probe = {};
    probe->abi_version = COLDSNAP_CUDA_EPOCH_CONTEXT_PROBE_ABI;
    probe->struct_size = sizeof(*probe);
    probe->device = device;
    probe->cu_init = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_device_get = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_ctx_get_current = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->current_context_present = -1;
    probe->cu_ctx_clear_current = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_primary_get_state = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->primary_active = -1;
    probe->cu_primary_retain = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_primary_release = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_private_create = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_private_get_current = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->private_context_matched = -1;
    probe->cu_private_alloc = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_private_memset = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_private_copy_to_host = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_private_synchronize = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_private_free = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->driver_payload_verified = -1;
    probe->runtime_get_device = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->runtime_device = -1;
    probe->runtime_malloc = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->runtime_memset = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->runtime_copy_to_host = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->runtime_synchronize = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->runtime_free = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->runtime_payload_verified = -1;
    probe->cu_private_destroy = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
    probe->cu_ctx_clear_after_probe = COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN;
}

}  // namespace

extern "C" int coldsnap_cuda_epoch_reset_device(int device) {
    last_error.clear();
    if (device < 0) {
        set_error("CUDA epoch device must be non-negative");
        return -1;
    }
    // Stop future admitted execution threads from selecting the context that
    // this reset is about to destroy. Epoch admission guarantees quiescence,
    // so no caller may be concurrently using the removed context.
    {
        std::lock_guard<std::mutex> lock(published_contexts_mutex);
        published_contexts.erase(device);
    }
    cudaError_t result = cudaSetDevice(device);
    if (result != cudaSuccess) {
        set_error(runtime_error(result, "cudaSetDevice(epoch reset)"));
        return -1;
    }
    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        set_error(runtime_error(result, "cudaDeviceSynchronize(epoch reset)"));
        return -1;
    }
    result = cudaDeviceReset();
    if (result != cudaSuccess) {
        set_error(runtime_error(result, "cudaDeviceReset(epoch)"));
        return -1;
    }
    using publish_mappings_function = int (*)();
    auto publish_mappings = reinterpret_cast<publish_mappings_function>(
        dlsym(RTLD_DEFAULT, "coldsnap_nvidia_mmap_publish_after_reset")
    );
    if (publish_mappings != nullptr) {
        int publish_result = publish_mappings();
        if (publish_result != 0) {
            std::ostringstream stream;
            stream << "publish NVIDIA mmap descriptors after CUDA reset failed with "
                   << publish_result;
            set_error(stream.str());
            return -1;
        }
    }
    return 0;
}

extern "C" const char* coldsnap_cuda_epoch_driver_result_name(int result) {
    if (result == COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN) {
        return "NOT_RUN";
    }
    const char* name = nullptr;
    if (cuGetErrorName(static_cast<CUresult>(result), &name) != CUDA_SUCCESS
        || name == nullptr) {
        return "CUDA_ERROR_UNKNOWN_RESULT";
    }
    return name;
}

extern "C" const char* coldsnap_cuda_epoch_runtime_result_name(int result) {
    if (result == COLDSNAP_CUDA_EPOCH_PROBE_NOT_RUN) {
        return "NOT_RUN";
    }
    const char* name = cudaGetErrorName(static_cast<cudaError_t>(result));
    return name == nullptr ? "cudaErrorUnknownResult" : name;
}

extern "C" int coldsnap_cuda_epoch_probe_context(
    int device,
    coldsnap_cuda_epoch_context_probe* probe
) {
    last_error.clear();
    if (probe == nullptr) {
        set_error("CUDA epoch context probe requires an output record");
        return -1;
    }
    initialize_context_probe(device, probe);
    if (device < 0) {
        set_error("CUDA epoch context probe device must be non-negative");
        return -1;
    }

    CUresult driver_result = cuInit(0);
    probe->cu_init = static_cast<int>(driver_result);
    if (driver_result != CUDA_SUCCESS) {
        return 0;
    }
    CUdevice cuda_device = 0;
    driver_result = cuDeviceGet(&cuda_device, device);
    probe->cu_device_get = static_cast<int>(driver_result);
    if (driver_result != CUDA_SUCCESS) {
        return 0;
    }

    CUcontext inherited_context = nullptr;
    driver_result = cuCtxGetCurrent(&inherited_context);
    probe->cu_ctx_get_current = static_cast<int>(driver_result);
    if (driver_result == CUDA_SUCCESS) {
        probe->current_context_present = inherited_context == nullptr ? 0 : 1;
    }
    driver_result = cuCtxSetCurrent(nullptr);
    probe->cu_ctx_clear_current = static_cast<int>(driver_result);
    if (driver_result != CUDA_SUCCESS) {
        return 0;
    }

    unsigned int primary_flags = 0;
    int primary_active = 0;
    driver_result = cuDevicePrimaryCtxGetState(
        cuda_device,
        &primary_flags,
        &primary_active
    );
    probe->cu_primary_get_state = static_cast<int>(driver_result);
    if (driver_result == CUDA_SUCCESS) {
        probe->primary_flags = primary_flags;
        probe->primary_active = primary_active;
    }

    CUcontext primary_context = nullptr;
    driver_result = cuDevicePrimaryCtxRetain(&primary_context, cuda_device);
    probe->cu_primary_retain = static_cast<int>(driver_result);
    if (driver_result == CUDA_SUCCESS) {
        probe->cu_primary_release = static_cast<int>(
            cuDevicePrimaryCtxRelease(cuda_device));
        probe->cu_ctx_clear_after_probe = static_cast<int>(
            cuCtxSetCurrent(nullptr));
        return 0;
    }

    // The legacy private-context entry point has a stable three-argument ABI.
    // Resolve it directly because CUDA 13 maps cuCtxCreate to the new v4 ABI,
    // while the driver-580 experiment specifically needs a call available on
    // both older and newer driver branches.
    using context_create_v2_function = CUresult (*)(
        CUcontext*, unsigned int, CUdevice);
    auto create_private = reinterpret_cast<context_create_v2_function>(
        dlsym(RTLD_DEFAULT, "cuCtxCreate_v2"));
    if (create_private == nullptr) {
        probe->cu_private_create = static_cast<int>(CUDA_ERROR_NOT_SUPPORTED);
        probe->cu_ctx_clear_after_probe = static_cast<int>(
            cuCtxSetCurrent(nullptr));
        return 0;
    }

    CUcontext private_context = nullptr;
    driver_result = create_private(&private_context, 0, cuda_device);
    probe->cu_private_create = static_cast<int>(driver_result);
    if (driver_result != CUDA_SUCCESS) {
        probe->cu_ctx_clear_after_probe = static_cast<int>(
            cuCtxSetCurrent(nullptr));
        return 0;
    }

    CUcontext selected_context = nullptr;
    driver_result = cuCtxGetCurrent(&selected_context);
    probe->cu_private_get_current = static_cast<int>(driver_result);
    if (driver_result == CUDA_SUCCESS) {
        probe->private_context_matched = selected_context == private_context ? 1 : 0;
    }

    constexpr size_t probe_bytes = 4096;
    constexpr std::uint8_t probe_pattern = 0xA5;
    std::array<std::uint8_t, probe_bytes> host_payload{};
    CUdeviceptr driver_pointer = 0;
    driver_result = cuMemAlloc(&driver_pointer, probe_bytes);
    probe->cu_private_alloc = static_cast<int>(driver_result);
    if (driver_result == CUDA_SUCCESS) {
        driver_result = cuMemsetD8(driver_pointer, probe_pattern, probe_bytes);
        probe->cu_private_memset = static_cast<int>(driver_result);
        if (driver_result == CUDA_SUCCESS) {
            driver_result = cuMemcpyDtoH(
                host_payload.data(),
                driver_pointer,
                probe_bytes
            );
            probe->cu_private_copy_to_host = static_cast<int>(driver_result);
        }
        if (driver_result == CUDA_SUCCESS) {
            driver_result = cuCtxSynchronize();
            probe->cu_private_synchronize = static_cast<int>(driver_result);
        }
        if (probe->cu_private_copy_to_host == static_cast<int>(CUDA_SUCCESS)) {
            probe->driver_payload_verified = std::all_of(
                host_payload.begin(),
                host_payload.end(),
                [](std::uint8_t value) { return value == probe_pattern; }
            ) ? 1 : 0;
        }
        probe->cu_private_free = static_cast<int>(cuMemFree(driver_pointer));
    }

    const bool driver_usable =
        probe->cu_private_get_current == static_cast<int>(CUDA_SUCCESS)
        && probe->private_context_matched == 1
        && probe->cu_private_alloc == static_cast<int>(CUDA_SUCCESS)
        && probe->cu_private_memset == static_cast<int>(CUDA_SUCCESS)
        && probe->cu_private_copy_to_host == static_cast<int>(CUDA_SUCCESS)
        && probe->cu_private_synchronize == static_cast<int>(CUDA_SUCCESS)
        && probe->cu_private_free == static_cast<int>(CUDA_SUCCESS)
        && probe->driver_payload_verified == 1;

    // Only touch the restored CUDA Runtime after the private driver context
    // has proven usable. This is deliberately destructive diagnostic state:
    // the disposable restored worker is expected to exit after the probe.
    if (driver_usable) {
        int runtime_device = -1;
        cudaError_t runtime_result = cudaGetDevice(&runtime_device);
        probe->runtime_get_device = static_cast<int>(runtime_result);
        if (runtime_result == cudaSuccess) {
            probe->runtime_device = runtime_device;
            void* runtime_pointer = nullptr;
            runtime_result = cudaMalloc(&runtime_pointer, probe_bytes);
            probe->runtime_malloc = static_cast<int>(runtime_result);
            if (runtime_result == cudaSuccess) {
                runtime_result = cudaMemset(
                    runtime_pointer,
                    probe_pattern,
                    probe_bytes
                );
                probe->runtime_memset = static_cast<int>(runtime_result);
                host_payload.fill(0);
                if (runtime_result == cudaSuccess) {
                    runtime_result = cudaMemcpy(
                        host_payload.data(),
                        runtime_pointer,
                        probe_bytes,
                        cudaMemcpyDeviceToHost
                    );
                    probe->runtime_copy_to_host = static_cast<int>(runtime_result);
                }
                if (runtime_result == cudaSuccess) {
                    runtime_result = cudaDeviceSynchronize();
                    probe->runtime_synchronize = static_cast<int>(runtime_result);
                }
                if (probe->runtime_copy_to_host == static_cast<int>(cudaSuccess)) {
                    probe->runtime_payload_verified = std::all_of(
                        host_payload.begin(),
                        host_payload.end(),
                        [](std::uint8_t value) { return value == probe_pattern; }
                    ) ? 1 : 0;
                }
                probe->runtime_free = static_cast<int>(cudaFree(runtime_pointer));
            }
        }
    }

    probe->cu_private_destroy = static_cast<int>(cuCtxDestroy(private_context));
    probe->cu_ctx_clear_after_probe = static_cast<int>(cuCtxSetCurrent(nullptr));
    return 0;
}

extern "C" int coldsnap_cuda_epoch_rebind(
    const coldsnap_cuda_epoch_allocation* input,
    size_t allocation_count
) {
    last_error.clear();
    if (input == nullptr || allocation_count == 0) {
        set_error("CUDA epoch rebind requires at least one allocation");
        return -1;
    }
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuInit(epoch)"));
        return -1;
    }

    // A runtime reset may leave the destroyed primary context selected in the
    // calling thread's driver context slot. Clear it before retaining a fresh
    // primary context.
    result = cuCtxSetCurrent(nullptr);
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuCtxSetCurrent(null epoch)"));
        return -1;
    }

    std::unordered_map<int, CUcontext> contexts;
    std::vector<ReboundAllocation> rebound;
    rebound.reserve(allocation_count);
    for (size_t index = 0; index < allocation_count; ++index) {
        const auto& allocation = input[index];
        if (
            allocation.device < 0
            || allocation.address == 0
            || allocation.mapped_bytes == 0
            || allocation.handle_box_address == 0
        ) {
            set_error("CUDA epoch allocation fields must be positive");
            rollback(&rebound, &contexts);
            return -1;
        }
        if (!select_context(allocation.device, &contexts)) {
            rollback(&rebound, &contexts);
            return -1;
        }

        CUmemAllocationProp properties{};
        size_t granularity = 0;
        if (!allocation_properties(
                allocation.device, &properties, &granularity)) {
            rollback(&rebound, &contexts);
            return -1;
        }
        if (
            allocation.address % granularity != 0
            || allocation.mapped_bytes % granularity != 0
        ) {
            std::ostringstream stream;
            stream << "CUDA epoch allocation " << index
                   << " is not aligned to VMM granularity " << granularity;
            set_error(stream.str());
            rollback(&rebound, &contexts);
            return -1;
        }

        ReboundAllocation current{};
        current.input = allocation;
        result = cuMemCreate(
            &current.handle,
            static_cast<size_t>(allocation.mapped_bytes),
            &properties,
            0
        );
        if (result != CUDA_SUCCESS) {
            set_error(driver_error(result, "cuMemCreate(epoch)"));
            rebound.push_back(current);
            rollback(&rebound, &contexts);
            return -1;
        }
        current.handle_valid = true;
        result = cuMemMap(
            static_cast<CUdeviceptr>(allocation.address),
            static_cast<size_t>(allocation.mapped_bytes),
            0,
            current.handle,
            0
        );
        if (result != CUDA_SUCCESS) {
            std::ostringstream stream;
            stream << driver_error(result, "cuMemMap(epoch retained address)")
                   << " at index " << index << " address 0x" << std::hex
                   << allocation.address;
            set_error(stream.str());
            rebound.push_back(current);
            rollback(&rebound, &contexts);
            return -1;
        }
        current.mapped = true;

        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = allocation.device;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        result = cuMemSetAccess(
            static_cast<CUdeviceptr>(allocation.address),
            static_cast<size_t>(allocation.mapped_bytes),
            &access,
            1
        );
        if (result != CUDA_SUCCESS) {
            set_error(driver_error(result, "cuMemSetAccess(epoch)"));
            rebound.push_back(current);
            rollback(&rebound, &contexts);
            return -1;
        }

        auto* handle_box = reinterpret_cast<CUmemGenericAllocationHandle*>(
            allocation.handle_box_address);
        *handle_box = current.handle;
        rebound.push_back(current);
    }
    // Publish only after every mapping and mutable handle box has committed.
    // A failed transaction must never expose one of its rollback contexts to
    // an executor thread.
    {
        std::lock_guard<std::mutex> lock(published_contexts_mutex);
        for (const auto& item : contexts) {
            published_contexts[item.first] = item.second;
        }
    }
    return 0;
}

extern "C" int coldsnap_cuda_epoch_activate_device(int device) {
    last_error.clear();
    if (device < 0) {
        set_error("CUDA epoch device must be non-negative");
        return -1;
    }
    CUcontext context = nullptr;
    {
        std::lock_guard<std::mutex> lock(published_contexts_mutex);
        const auto found = published_contexts.find(device);
        if (found == published_contexts.end()) {
            std::ostringstream stream;
            stream << "CUDA epoch device " << device
                   << " has no context published by rebind";
            set_error(stream.str());
            return -1;
        }
        context = found->second;
    }
    // Refresh CUDA Runtime's per-thread device state first.  PyTorch caches a
    // selected device independently from the driver's current-context slot;
    // after cudaDeviceReset both views must be pointed at the new epoch.
    const cudaError_t runtime_result = cudaSetDevice(device);
    if (runtime_result != cudaSuccess) {
        set_error(runtime_error(runtime_result, "cudaSetDevice(epoch activation)"));
        return -1;
    }
    const CUresult result = cuCtxSetCurrent(context);
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuCtxSetCurrent(epoch activation)"));
        return -1;
    }
    return 0;
}

extern "C" const char* coldsnap_cuda_epoch_last_error(void) {
    return last_error.c_str();
}

extern "C" int coldsnap_cuda_epoch_copy_from_host(
    uintptr_t destination,
    uintptr_t source,
    uint64_t bytes
) {
    last_error.clear();
    if (destination == 0 || source == 0 || bytes == 0) {
        set_error("CUDA epoch host copy fields must be positive");
        return -1;
    }
    CUresult result = cuMemcpyHtoD(
        static_cast<CUdeviceptr>(destination),
        reinterpret_cast<const void*>(source),
        static_cast<size_t>(bytes)
    );
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemcpyHtoD(epoch)"));
        return -1;
    }
    return 0;
}

extern "C" int coldsnap_cuda_epoch_fill_zero(
    uintptr_t destination,
    uint64_t bytes
) {
    last_error.clear();
    if (destination == 0 || bytes == 0) {
        set_error("CUDA epoch memset fields must be positive");
        return -1;
    }
    CUresult result = cuMemsetD8(
        static_cast<CUdeviceptr>(destination),
        0,
        static_cast<size_t>(bytes)
    );
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuMemsetD8(epoch)"));
        return -1;
    }
    return 0;
}

extern "C" int coldsnap_cuda_epoch_synchronize(void) {
    last_error.clear();
    CUresult result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {
        set_error(driver_error(result, "cuCtxSynchronize(epoch)"));
        return -1;
    }
    return 0;
}

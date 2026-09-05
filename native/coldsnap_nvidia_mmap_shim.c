// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

//
// Record the descriptor used for each NVIDIA device VMA. CRIU's device hook
// receives an O_PATH descriptor from /proc/PID/map_files, which identifies the
// mapping but cannot replay mmap. The reset-epoch runtime asks this opt-in vLLM
// shim to publish surviving mmap-capable descriptors only after cudaDeviceReset
// has completed.  The shim retains a duplicate of the exact open-file
// description while its VMA is alive; remembering only the descriptor number
// is unsafe because CUDA can close and reuse that number before reset.

#define _GNU_SOURCE
#define _LARGEFILE64_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define FD_BROKER_ENVIRONMENT "COLDSNAP_CRIU_NVIDIA_FD_BROKER_SOCKET"
#define FD_BROKER_PROTOCOL "COLDSNAP_FD_BROKER_V1"
#define MAX_NVIDIA_MAPPINGS 4096

typedef void* (*mmap_function)(void*, size_t, int, int, int, off_t);
typedef void* (*mmap64_function)(void*, size_t, int, int, int, off64_t);
typedef int (*munmap_function)(void*, size_t);

static mmap_function next_mmap = NULL;
static mmap64_function next_mmap64 = NULL;
static munmap_function next_munmap = NULL;
static __thread int resolving = 0;

struct nvidia_mapping {
    uintptr_t address;
    size_t length;
    int protection;
    int preserved_descriptor;
    int preserve_error;
};

static pthread_mutex_t mappings_mutex = PTHREAD_MUTEX_INITIALIZER;
static struct nvidia_mapping mappings[MAX_NVIDIA_MAPPINGS];
static size_t mapping_count = 0;

static int nvidia_descriptor(int descriptor) {
    if (descriptor < 0) {
        return 0;
    }
    char proc_path[64];
    int bytes = snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", descriptor);
    if (bytes <= 0 || (size_t)bytes >= sizeof(proc_path)) {
        return 0;
    }
    char target[256];
    ssize_t target_bytes = readlink(proc_path, target, sizeof(target) - 1);
    if (target_bytes <= 0 || (size_t)target_bytes >= sizeof(target)) {
        return 0;
    }
    target[target_bytes] = '\0';
    const char prefix[] = "/dev/nvidia";
    return strncmp(target, prefix, sizeof(prefix) - 1) == 0;
}

static int connect_broker(void) {
    const char* path = getenv(FD_BROKER_ENVIRONMENT);
    if (path == NULL || path[0] != '/' ||
        strlen(path) >= sizeof(((struct sockaddr_un*)0)->sun_path)) {
        return -EINVAL;
    }
    int descriptor = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (descriptor < 0) {
        return -errno;
    }
    struct sockaddr_un address = { .sun_family = AF_UNIX };
    memcpy(address.sun_path, path, strlen(path) + 1);
    if (connect(descriptor, (const struct sockaddr*)&address, sizeof(address)) != 0) {
        int result = -errno;
        close(descriptor);
        return result;
    }
    return descriptor;
}

static int preserve_mapping(uintptr_t address, int source_fd) {
    int socket_fd = connect_broker();
    if (socket_fd < 0) {
        return socket_fd;
    }
    char request[128];
    int request_bytes = snprintf(
        request,
        sizeof(request),
        FD_BROKER_PROTOCOL " PUT M %llu\n",
        (unsigned long long)address
    );
    if (request_bytes <= 0 || (size_t)request_bytes >= sizeof(request)) {
        close(socket_fd);
        return -EINVAL;
    }
    char control[CMSG_SPACE(sizeof(int))] = {0};
    struct iovec vector = {
        .iov_base = request,
        .iov_len = (size_t)request_bytes,
    };
    struct msghdr message = {
        .msg_iov = &vector,
        .msg_iovlen = 1,
        .msg_control = control,
        .msg_controllen = sizeof(control),
    };
    struct cmsghdr* header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(header), &source_fd, sizeof(source_fd));
    if (sendmsg(socket_fd, &message, MSG_NOSIGNAL) != request_bytes) {
        int result = errno == 0 ? -EIO : -errno;
        close(socket_fd);
        return result;
    }
    char response[64];
    ssize_t response_bytes = recv(socket_fd, response, sizeof(response) - 1, 0);
    int saved_errno = errno;
    close(socket_fd);
    if (response_bytes < 0) {
        return -saved_errno;
    }
    response[response_bytes] = '\0';
    return strncmp(response, "OK", 2) == 0 ? 0 : -EIO;
}

static void resolve_mmap_symbols(void) {
    if ((next_mmap != NULL && next_munmap != NULL) || resolving) {
        return;
    }
    resolving = 1;
    if (next_mmap == NULL) {
        next_mmap = (mmap_function)dlsym(RTLD_NEXT, "mmap");
    }
    if (next_mmap64 == NULL) {
        next_mmap64 = (mmap64_function)dlsym(RTLD_NEXT, "mmap64");
    }
    if (next_munmap == NULL) {
        next_munmap = (munmap_function)dlsym(RTLD_NEXT, "munmap");
    }
    resolving = 0;
}

static uintptr_t range_end(uintptr_t address, size_t length) {
    if (length > UINTPTR_MAX - address) {
        return UINTPTR_MAX;
    }
    return address + length;
}

static int ranges_overlap(
    uintptr_t left_address,
    size_t left_length,
    uintptr_t right_address,
    size_t right_length
) {
    return left_address < range_end(right_address, right_length) &&
           right_address < range_end(left_address, left_length);
}

static void forget_overlapping_mappings_locked(
    uintptr_t address,
    size_t length
) {
    size_t index = 0;
    while (index < mapping_count) {
        if (!ranges_overlap(
                mappings[index].address,
                mappings[index].length,
                address,
                length
            )) {
            ++index;
            continue;
        }
        if (mappings[index].preserved_descriptor >= 0) {
            close(mappings[index].preserved_descriptor);
        }
        mappings[index] = mappings[mapping_count - 1];
        --mapping_count;
    }
}

static int mapping_still_nvidia(uintptr_t address) {
    FILE* stream = fopen("/proc/self/maps", "re");
    if (stream == NULL) {
        return -errno;
    }
    int result = 0;
    char line[1024];
    while (fgets(line, sizeof(line), stream) != NULL) {
        unsigned long long start = 0;
        unsigned long long end = 0;
        unsigned long long offset = 0;
        unsigned int device_major = 0;
        unsigned int device_minor = 0;
        unsigned long inode = 0;
        char permissions[5] = {0};
        char path[256] = {0};
        int fields = sscanf(
            line,
            "%llx-%llx %4s %llx %x:%x %lu %255s",
            &start,
            &end,
            permissions,
            &offset,
            &device_major,
            &device_minor,
            &inode,
            path
        );
        if (fields >= 7 && start == (unsigned long long)address) {
            result = fields == 8 && strncmp(path, "/dev/nvidia", 11) == 0;
            break;
        }
    }
    if (ferror(stream)) {
        result = -EIO;
    }
    fclose(stream);
    return result;
}

static void record_if_nvidia(
    void* result,
    size_t length,
    int protection,
    int descriptor
) {
    if (result == MAP_FAILED || getenv(FD_BROKER_ENVIRONMENT) == NULL ||
        !nvidia_descriptor(descriptor)) {
        return;
    }
    uintptr_t address = (uintptr_t)result;
    int preserved_descriptor = fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
    int preserve_error = preserved_descriptor < 0 ? errno : 0;
    pthread_mutex_lock(&mappings_mutex);
    forget_overlapping_mappings_locked(address, length);
    if (mapping_count < MAX_NVIDIA_MAPPINGS) {
        mappings[mapping_count++] = (struct nvidia_mapping){
            .address = address,
            .length = length,
            .protection = protection,
            .preserved_descriptor = preserved_descriptor,
            .preserve_error = preserve_error,
        };
    } else {
        if (preserved_descriptor >= 0) {
            close(preserved_descriptor);
        }
        static const char warning[] =
            "coldsnap NVIDIA mmap shim mapping registry is full\n";
        ssize_t ignored = write(STDERR_FILENO, warning, sizeof(warning) - 1);
        (void)ignored;
    }
    pthread_mutex_unlock(&mappings_mutex);
}

// Called cooperatively by libcoldsnap_cuda_epoch immediately after
// cudaDeviceReset. Returns zero only when every still-live NVIDIA VMA has a
// surviving original descriptor that can be transferred to the broker.
int coldsnap_nvidia_mmap_publish_after_reset(void) {
    int first_error = 0;
    size_t published = 0;
    pthread_mutex_lock(&mappings_mutex);
    for (size_t index = 0; index < mapping_count; ++index) {
        int live = mapping_still_nvidia(mappings[index].address);
        if (live < 0) {
            if (first_error == 0) {
                first_error = live;
            }
            continue;
        }
        if (live == 0) {
            if (mappings[index].preserved_descriptor >= 0) {
                close(mappings[index].preserved_descriptor);
                mappings[index].preserved_descriptor = -1;
            }
            continue;
        }
        int descriptor = mappings[index].preserved_descriptor;
        if (!nvidia_descriptor(descriptor)) {
            if (first_error == 0) {
                first_error = mappings[index].preserve_error == 0
                    ? -ENODEV
                    : -mappings[index].preserve_error;
            }
            continue;
        }
        int result = preserve_mapping(
            mappings[index].address,
            descriptor
        );
        if (result == 0) {
            close(descriptor);
            mappings[index].preserved_descriptor = -1;
        }
        if (result != 0) {
            if (first_error == 0) {
                first_error = result;
            }
            continue;
        }
        ++published;
    }
    pthread_mutex_unlock(&mappings_mutex);
    if (first_error != 0) {
        dprintf(
            STDERR_FILENO,
            "coldsnap NVIDIA mmap shim published %zu mappings before error %d\n",
            published,
            first_error
        );
    }
    return first_error;
}

int munmap(void* address, size_t length) {
    resolve_mmap_symbols();
    if (next_munmap == NULL) {
        errno = ENOSYS;
        return -1;
    }
    int result = next_munmap(address, length);
    if (result == 0 && length != 0) {
        pthread_mutex_lock(&mappings_mutex);
        forget_overlapping_mappings_locked((uintptr_t)address, length);
        pthread_mutex_unlock(&mappings_mutex);
    }
    return result;
}

void* mmap(
    void* address,
    size_t length,
    int protection,
    int flags,
    int descriptor,
    off_t offset
) {
    resolve_mmap_symbols();
    if (next_mmap == NULL) {
        errno = ENOSYS;
        return MAP_FAILED;
    }
    void* result = next_mmap(address, length, protection, flags, descriptor, offset);
    record_if_nvidia(result, length, protection, descriptor);
    return result;
}

void* mmap64(
    void* address,
    size_t length,
    int protection,
    int flags,
    int descriptor,
    off64_t offset
) {
    resolve_mmap_symbols();
    if (next_mmap64 == NULL) {
        if (next_mmap == NULL) {
            errno = ENOSYS;
            return MAP_FAILED;
        }
        void* result = next_mmap(
            address, length, protection, flags, descriptor, (off_t)offset
        );
        record_if_nvidia(result, length, protection, descriptor);
        return result;
    }
    void* result = next_mmap64(
        address, length, protection, flags, descriptor, offset
    );
    record_if_nvidia(result, length, protection, descriptor);
    return result;
}

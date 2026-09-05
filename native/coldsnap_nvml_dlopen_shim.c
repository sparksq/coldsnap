// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

//
// Give restored process templates a fresh NVML link-map. CUDA can establish a
// new context after a CPU-only CRIU restore on driver 580, but dlopen by NCCL
// otherwise reuses NVML globals captured before restore. This opt-in shim is
// inactive until the exact ColdSnap restore-generation marker is present.

#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GENERATION_ENVIRONMENT "COLDSNAP_MODEL_LOAD_GENERATION"
#define RESTORE_MARKER_ENVIRONMENT \
    "COLDSNAP_PROCESS_TEMPLATE_RESTORE_MARKER_FILE"

typedef void* (*dlopen_function)(const char*, int);

static dlopen_function next_dlopen = NULL;
static __thread int resolving_dlopen = 0;

static void resolve_dlopen(void) {
    if (next_dlopen != NULL || resolving_dlopen) {
        return;
    }
    resolving_dlopen = 1;
    next_dlopen = (dlopen_function)dlsym(RTLD_NEXT, "dlopen");
    resolving_dlopen = 0;
}

static int restored_generation(void) {
    const char* marker = getenv(RESTORE_MARKER_ENVIRONMENT);
    const char* generation = getenv(GENERATION_ENVIRONMENT);
    if (marker == NULL || marker[0] != '/' || generation == NULL ||
        generation[0] == '\0') {
        return 0;
    }
    size_t generation_bytes = strlen(generation);
    if (generation_bytes >= 512) {
        return 0;
    }
    int descriptor = open(marker, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        return 0;
    }
    char payload[513];
    ssize_t bytes = read(descriptor, payload, sizeof(payload) - 1);
    int saved_errno = errno;
    close(descriptor);
    errno = saved_errno;
    if (bytes < 0) {
        return 0;
    }
    while (bytes > 0 &&
           (payload[bytes - 1] == '\n' || payload[bytes - 1] == '\r')) {
        --bytes;
    }
    payload[bytes] = '\0';
    return (size_t)bytes == generation_bytes &&
           memcmp(payload, generation, generation_bytes) == 0;
}

static int mapped_nvml_path(char* destination, size_t destination_bytes) {
    FILE* stream = fopen("/proc/self/maps", "re");
    if (stream == NULL) {
        return -errno;
    }
    int result = -ENOENT;
    char line[2048];
    while (fgets(line, sizeof(line), stream) != NULL) {
        unsigned long long start = 0;
        unsigned long long end = 0;
        unsigned long long offset = 0;
        unsigned int device_major = 0;
        unsigned int device_minor = 0;
        unsigned long inode = 0;
        char permissions[5];
        char path[1024];
        int fields = sscanf(
            line,
            "%llx-%llx %4s %llx %x:%x %lu %1023s",
            &start,
            &end,
            permissions,
            &offset,
            &device_major,
            &device_minor,
            &inode,
            path
        );
        if (fields != 8 || path[0] != '/' ||
            strstr(path, "/libnvidia-ml.so") == NULL ||
            access(path, R_OK) != 0) {
            continue;
        }
        size_t bytes = strlen(path);
        if (bytes >= destination_bytes) {
            result = -ENAMETOOLONG;
            break;
        }
        memcpy(destination, path, bytes + 1);
        result = 0;
        break;
    }
    if (result == -ENOENT && ferror(stream)) {
        result = -EIO;
    }
    fclose(stream);
    if (result == -ENOENT) {
        const char* candidates[] = {
            "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib64/libnvidia-ml.so.1",
            "/usr/lib/libnvidia-ml.so.1",
        };
        for (size_t index = 0;
             index < sizeof(candidates) / sizeof(candidates[0]);
             ++index) {
            char* resolved = realpath(candidates[index], NULL);
            if (resolved == NULL) {
                continue;
            }
            size_t bytes = strlen(resolved);
            if (bytes < destination_bytes && access(resolved, R_OK) == 0) {
                memcpy(destination, resolved, bytes + 1);
                free(resolved);
                return 0;
            }
            free(resolved);
        }
    }
    return result;
}

static int copy_file(int source, int destination) {
    char buffer[1 << 16];
    for (;;) {
        ssize_t bytes = read(source, buffer, sizeof(buffer));
        if (bytes == 0) {
            return 0;
        }
        if (bytes < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -errno;
        }
        ssize_t written = 0;
        while (written < bytes) {
            ssize_t count = write(
                destination,
                buffer + written,
                (size_t)(bytes - written)
            );
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count <= 0) {
                return count == 0 ? -EIO : -errno;
            }
            written += count;
        }
    }
}

static int private_nvml_copy(char* destination, size_t destination_bytes) {
    char source_path[1024];
    int result = mapped_nvml_path(source_path, sizeof(source_path));
    if (result != 0) {
        return result;
    }
    const char pattern[] = "/tmp/coldsnap-restored-libnvidia-ml-XXXXXX.so";
    if (sizeof(pattern) > destination_bytes) {
        return -ENAMETOOLONG;
    }
    memcpy(destination, pattern, sizeof(pattern));
    int destination_fd = mkstemps(destination, 3);
    if (destination_fd < 0) {
        return -errno;
    }
    int source_fd = open(source_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (source_fd < 0) {
        result = -errno;
    } else {
        result = copy_file(source_fd, destination_fd);
        close(source_fd);
    }
    int saved_errno = errno;
    close(destination_fd);
    errno = saved_errno;
    if (result != 0) {
        unlink(destination);
    }
    return result;
}

void* dlopen(const char* filename, int flags) {
    resolve_dlopen();
    if (next_dlopen == NULL) {
        errno = ENOSYS;
        return NULL;
    }
    if (filename == NULL || strstr(filename, "libnvidia-ml.so") == NULL ||
        !restored_generation()) {
        return next_dlopen(filename, flags);
    }
    char private_path[256];
    int result = private_nvml_copy(private_path, sizeof(private_path));
    if (result != 0) {
        dprintf(
            STDERR_FILENO,
            "coldsnap NVML dlopen shim could not copy NVML: %d\n",
            result
        );
        return next_dlopen(filename, flags);
    }
    void* handle = next_dlopen(private_path, flags);
    int saved_errno = errno;
    unlink(private_path);
    errno = saved_errno;
    if (handle != NULL) {
        dprintf(
            STDERR_FILENO,
            "coldsnap NVML dlopen shim loaded a fresh restore-epoch library\n"
        );
    }
    return handle;
}

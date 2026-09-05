// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

typedef void *(*dlsym_fn)(void *, const char *);

static _Atomic unsigned int route_mask = 0;
static _Atomic unsigned int route_count = 0;

unsigned int coldsnapNcclDlsymBridgeAbi(void) {
  return 1U;
}

static unsigned int route_bit(const char *name) {
  if (strcmp(name, "ncclAllGather") == 0) return 1U;
  if (strcmp(name, "ncclCommCount") == 0) return 2U;
  if (strcmp(name, "ncclCommUserRank") == 0) return 4U;
  if (strcmp(name, "ncclCommInitRank") == 0) return 8U;
  if (strcmp(name, "ncclCommDestroy") == 0) return 16U;
  if (strcmp(name, "ncclAllReduce") == 0) return 32U;
  return 0U;
}

static dlsym_fn real_dlsym(void) {
  static dlsym_fn function = NULL;
  if (function == NULL) {
    function = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (function == NULL) {
      fputs("coldsnap NCCL dlsym bridge could not resolve GLIBC_2.34 dlsym\n", stderr);
      abort();
    }
  }
  return function;
}

static int is_nccl_call(const char *name) {
  return name != NULL && strncmp(name, "nccl", 4) == 0;
}

unsigned int coldsnapNcclDlsymRouteMask(void) {
  return atomic_load(&route_mask);
}

unsigned int coldsnapNcclDlsymRouteCount(void) {
  return atomic_load(&route_count);
}

void *dlsym(void *handle, const char *name) {
  dlsym_fn next = real_dlsym();
  if (handle != RTLD_DEFAULT && handle != RTLD_NEXT &&
      is_nccl_call(name)) {
    const char *path = getenv("COLDSNAP_NCCL_CHECKPOINT_SHIM_PATH");
    if (path != NULL && path[0] != '\0') {
      void *shim = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
      if (shim != NULL) {
        void *interposed = next(shim, name);
        if (interposed != NULL) {
          atomic_fetch_or(&route_mask, route_bit(name));
          atomic_fetch_add(&route_count, 1U);
          return interposed;
        }
      }
    }
  }
  return next(handle, name);
}

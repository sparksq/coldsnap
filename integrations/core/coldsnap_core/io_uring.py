# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Keep non-checkpointable rings out of event loops without blocking readers."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
import ctypes.util
import errno
from typing import Any, Callable


def block_io_uring_for_current_thread() -> None:
    library_name = ctypes.util.find_library("seccomp")
    if library_name is None:
        raise RuntimeError("libseccomp is required to block io_uring")
    seccomp = ctypes.CDLL(library_name, use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]

    allow = 0x7FFF0000
    deny = 0x00050000 | errno.EPERM
    context = seccomp.seccomp_init(allow)
    if not context:
        raise OSError(ctypes.get_errno(), "seccomp_init failed")
    try:
        for name in (b"io_uring_setup", b"io_uring_enter", b"io_uring_register"):
            number = seccomp.seccomp_syscall_resolve_name(name)
            if number < 0:
                raise RuntimeError(f"libseccomp cannot resolve {name.decode()}")
            result = seccomp.seccomp_rule_add(context, deny, number, 0)
            if result != 0:
                raise OSError(-result, f"seccomp_rule_add failed for {name.decode()}")
        result = seccomp.seccomp_load(context)
        if result != 0:
            raise OSError(-result, "seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)


def without_io_uring(factory: Callable[[], Any]) -> Any:
    """Construct on a short-lived, restricted thread, then transfer ownership.

    Libuv falls back to epoll when ring creation fails. Its current releases do
    not expose a switch for disabling epoll-batching rings. The seccomp filter
    is thread-local (no TSYNC); caller and reader threads remain unrestricted.
    Only use factories whose results permit use on another thread, such as an
    event loop that has not yet run. Joining also removes the helper before a
    process checkpoint.
    """
    def create() -> Any:
        block_io_uring_for_current_thread()
        return factory()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="coldsnap-epoll") as executor:
        return executor.submit(create).result()

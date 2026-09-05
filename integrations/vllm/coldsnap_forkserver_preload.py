# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Pre-CUDA import payload for the experimental vLLM forkserver template.

This module is imported inside Python's single-purpose forkserver process.
Importing vLLM here lets later children inherit its read-only Python/module
state.  The template must fail before serving children if import initialized a
CUDA context; forking a CUDA-initialized process is not a supported boundary.
"""

from __future__ import annotations

import os
import time


_started = time.perf_counter()

from vllm import LLM, SamplingParams  # noqa: E402, F401

import torch  # noqa: E402


if torch.cuda.is_initialized():
    raise RuntimeError("vLLM forkserver preload initialized CUDA")

os.environ["COLDSNAP_FORKSERVER_PRELOADED"] = "1"
os.environ["COLDSNAP_FORKSERVER_PRELOAD_SECONDS"] = str(
    time.perf_counter() - _started
)
os.environ["COLDSNAP_FORKSERVER_PRELOAD_PID"] = str(os.getpid())

# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Payload-free neutral tensors for final-layout model construction."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from typing import Any


MATERIALIZER_ENV = "COLDSNAP_SYNTHETIC_MATERIALIZER"
DENSE_PATTERNS_ENV = "COLDSNAP_SYNTHETIC_DENSE_PATTERNS"


@dataclass(frozen=True)
class MaterializerStats:
    tensors: int
    logical_bytes: int
    physical_source_bytes: int
    dense_tensors: int


class NeutralTensorMaterializer:
    """Supply broadcast-zero views while vLLM builds its normal final layout.

    The normal model ``load_weights`` implementation remains in control of
    parameter allocation, packing, and finalization. Expanded scalar sources
    carry valid neutral values without allocating checkpoint-sized CPU
    storage. Known incompatible weight names can opt into dense zero tensors.
    """

    def __init__(
        self,
        *,
        mode: str | None = None,
        dense_patterns: tuple[str, ...] | None = None,
    ) -> None:
        self.mode = mode or os.environ.get(MATERIALIZER_ENV, "expanded_zero")
        if self.mode not in {"expanded_zero", "dense_zero"}:
            raise ValueError(
                f"{MATERIALIZER_ENV} must be expanded_zero or dense_zero"
            )
        if dense_patterns is None:
            raw_patterns = os.environ.get(DENSE_PATTERNS_ENV, "")
            dense_patterns = tuple(
                value.strip() for value in raw_patterns.split(",") if value.strip()
            )
        if any(not pattern for pattern in dense_patterns):
            raise ValueError("synthetic dense fallback patterns cannot be empty")
        self.dense_patterns = dense_patterns
        self._tensors = 0
        self._logical_bytes = 0
        self._physical_source_bytes = 0
        self._dense_tensors = 0

    def uses_dense_source(self, name: str) -> bool:
        return self.mode == "dense_zero" or any(
            fnmatch.fnmatchcase(name, pattern) for pattern in self.dense_patterns
        )

    def tensor(
        self,
        torch_module: Any,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> Any:
        logical_elements = 1
        for dimension in shape:
            logical_elements *= dimension
        scalar = torch_module.zeros((), dtype=dtype, device="cpu")
        element_bytes = int(scalar.element_size())
        self._tensors += 1
        self._logical_bytes += logical_elements * element_bytes
        if self.uses_dense_source(name):
            value = torch_module.zeros(shape, dtype=dtype, device="cpu")
            self._physical_source_bytes += logical_elements * element_bytes
            self._dense_tensors += 1
            return value
        self._physical_source_bytes += element_bytes
        return scalar.expand(shape)

    @property
    def stats(self) -> MaterializerStats:
        return MaterializerStats(
            tensors=self._tensors,
            logical_bytes=self._logical_bytes,
            physical_source_bytes=self._physical_source_bytes,
            dense_tensors=self._dense_tensors,
        )

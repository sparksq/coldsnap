# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

from coldsnap_materializer import NeutralTensorMaterializer  # noqa: E402


class FakeTensor:
    def __init__(self, shape, element_bytes, *, expanded=False) -> None:
        self.shape = shape
        self.element_bytes = element_bytes
        self.expanded = expanded

    def element_size(self) -> int:
        return self.element_bytes

    def expand(self, shape):
        return FakeTensor(shape, self.element_bytes, expanded=True)


class FakeTorch:
    def __init__(self, element_bytes=2) -> None:
        self.element_bytes = element_bytes
        self.zero_shapes = []

    def zeros(self, shape, *, dtype, device):
        del dtype, device
        self.zero_shapes.append(shape)
        return FakeTensor(shape, self.element_bytes)


class NeutralTensorMaterializerTests(unittest.TestCase):
    def test_expanded_zero_uses_scalar_physical_storage(self) -> None:
        torch = FakeTorch(element_bytes=2)
        materializer = NeutralTensorMaterializer(mode="expanded_zero")

        tensor = materializer.tensor(torch, "model.weight", (8, 4), "bf16")

        self.assertTrue(tensor.expanded)
        self.assertEqual(torch.zero_shapes, [()])
        self.assertEqual(materializer.stats.logical_bytes, 64)
        self.assertEqual(materializer.stats.physical_source_bytes, 2)
        self.assertEqual(materializer.stats.dense_tensors, 0)

    def test_name_pattern_provides_explicit_dense_fallback(self) -> None:
        torch = FakeTorch(element_bytes=2)
        materializer = NeutralTensorMaterializer(
            mode="expanded_zero", dense_patterns=("*.packed_weight",)
        )

        tensor = materializer.tensor(
            torch, "layer.packed_weight", (8, 4), "bf16"
        )

        self.assertFalse(tensor.expanded)
        self.assertEqual(torch.zero_shapes, [(), (8, 4)])
        self.assertEqual(materializer.stats.physical_source_bytes, 64)
        self.assertEqual(materializer.stats.dense_tensors, 1)

    def test_unknown_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "expanded_zero or dense_zero"):
            NeutralTensorMaterializer(mode="meta")


if __name__ == "__main__":
    unittest.main()

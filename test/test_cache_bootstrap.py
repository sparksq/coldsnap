# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
from coldsnap_cache_bootstrap import seed_cache  # noqa: E402


class CacheBootstrapTest(unittest.TestCase):
    def test_copies_into_fresh_writable_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "committed"
            source.mkdir()
            (source / "kernel.o").write_bytes(b"compiled")
            destination = root / "worker"

            self.assertGreaterEqual(seed_cache(source, destination), 0)
            self.assertEqual((destination / "kernel.o").read_bytes(), b"compiled")
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                seed_cache(source, destination)


if __name__ == "__main__":
    unittest.main()

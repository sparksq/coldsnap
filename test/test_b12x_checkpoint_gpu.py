# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Real B12x/CUDA reader tests: COLDSNAP_TEST_B12X_CHECKPOINT=1.

COLDSNAP_TEST_CHECKPOINT_PAUSE_DIR optionally lets an external CUDA/CRIU
controller checkpoint the prepared process before publishing a resume file.
"""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("COLDSNAP_TEST_B12X_CHECKPOINT"), "B12x CUDA image required")
class B12xCheckpointGpuTest(unittest.TestCase):
    def test_native_reader_reopens_without_replacing_graph_storage(self):
        import torch
        from b12x.sequence._shared import disk_table
        from coldsnap_core.checkpoint import CheckpointResourceError, ResourceRegistry, assert_no_io_uring
        import coldsnap_b12x_checkpoint as adapter

        import uvloop
        from coldsnap_vllm_resources import _install_event_loop_hooks
        _install_event_loop_hooks(uvloop)
        loop = uvloop.new_event_loop()
        self.addCleanup(loop.close)
        import signal
        loop.add_signal_handler(signal.SIGUSR1, lambda: None)
        loop.remove_signal_handler(signal.SIGUSR1)
        assert_no_io_uring()
        registry = ResourceRegistry()
        with tempfile.TemporaryDirectory() as directory, patch.object(adapter, "REGISTRY", registry), patch.dict(os.environ, {"B12X_DISK_BACKEND": "io_uring"}):
            adapter.install_disk_row_cache_adapter(disk_table)
            weights = torch.arange(8*16, dtype=torch.uint8).reshape(8, 16)
            scales = (torch.arange(8*4, dtype=torch.uint8)+128).reshape(8, 4)
            source = Path(directory)/"table.bin"
            source.write_bytes(weights.numpy().tobytes()+scales.numpy().tobytes())
            cache = disk_table.DiskRowCache(device="cuda:0", max_lookups=4, table_rows=8,
                                          shard_start=0, shard_end=8, shard_rows=4,
                                          weight_row_bytes=16, scale_row_bytes=4, queue_depth=4)
            try:
                for shard in (0, 1):
                    cache.add_shard(shard, str(source), shard*4*16)
                    cache.add_shard(shard, str(source), 128+shard*4*4, scale=True)
                cache.freeze()
                ids = torch.tensor([7, 0, 3, 3], dtype=torch.int64, device="cuda")
                with cache.transaction():
                    cache.read_rows(ids, 4)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(cache.weight.cpu(), weights[[7, 0, 3, 3]]))
                self.assertTrue(torch.equal(cache.scale.cpu(), scales[[7, 0, 3, 3]]))
                output, scale_output = torch.empty_like(cache.weight), torch.empty_like(cache.scale)
                output.copy_(cache.weight)
                scale_output.copy_(cache.scale)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output.copy_(cache.weight)
                    scale_output.copy_(cache.scale)
                torch.cuda.synchronize()
                before = adapter._buffer_identity(cache)
                with self.assertRaises(CheckpointResourceError):
                    assert_no_io_uring()
                prepared = registry.prepare()
                assert_no_io_uring()
                with self.assertRaisesRegex(CheckpointResourceError, "prepared"):
                    cache.stats()
                print(json.dumps({"phase": "prepared", "pid": os.getpid(), "resources": prepared, "buffer_addresses": before}), flush=True)
                pause = os.environ.get("COLDSNAP_TEST_CHECKPOINT_PAUSE_DIR")
                if pause:
                    root = Path(pause)
                    root.mkdir(parents=True, exist_ok=True)
                    (root/"prepared.json").write_text(json.dumps({"pid": os.getpid(), "buffers": before}))
                    deadline = time.monotonic()+240
                    while not (root/"resume").exists():
                        if time.monotonic() > deadline:
                            self.fail("timed out waiting for external checkpoint controller")
                        time.sleep(0.1)
                async def loop_alive():
                    import asyncio
                    await asyncio.sleep(0)
                    return type(asyncio.get_running_loop()).__module__
                self.assertEqual(loop.run_until_complete(loop_alive()), "uvloop")
                restored = registry.restore()
                self.assertEqual(adapter._buffer_identity(cache), before)
                ids.copy_(torch.tensor([1, 6, 2, 0], dtype=torch.int64, device="cuda"))
                with cache.transaction():
                    cache.read_rows(ids, 4)
                    graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(output.cpu(), weights[[1, 6, 2, 0]]))
                self.assertTrue(torch.equal(scale_output.cpu(), scales[[1, 6, 2, 0]]))
                print(json.dumps({"phase": "restored", "resources": restored, "graph_replay": True,
                                  "weights_and_scales_correct": True, "addresses_preserved": True}), flush=True)
                if pause:
                    (Path(pause)/"passed.json").write_text(json.dumps({"passed": True, "resources": restored}))
            finally:
                cache.close()
                assert_no_io_uring()


if __name__ == "__main__":
    unittest.main()

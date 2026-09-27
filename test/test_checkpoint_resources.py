# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "integrations/core"), str(ROOT / "integrations/vllm")]
from coldsnap_core import checkpoint  # noqa: E402
import coldsnap_vllm_resources as vllm_resources  # noqa: E402


class Resource:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls
        self.fail_prepare = self.fail_restore = False

    def prepare(self):
        self.calls.append((self.name, "prepare"))
        if self.fail_prepare:
            raise RuntimeError("prepare failed")
        return {"phase": "prepared"}

    def restore(self):
        self.calls.append((self.name, "restore"))
        if self.fail_restore:
            raise RuntimeError("restore failed")
        return {"phase": "restored"}


class CheckpointResourcesTest(unittest.TestCase):
    def setUp(self):
        self.registry = checkpoint.ResourceRegistry()
        self.calls = []

    def resource(self, name):
        resource = Resource(name, self.calls)
        self.registry.register(resource, kind=name)
        return resource

    def test_prepare_is_idempotent_and_restore_is_ordered(self):
        a, b = self.resource("a"), self.resource("b")
        report = self.registry.prepare()
        self.assertIs(self.registry.prepare(), report)
        self.assertEqual(len(report["resources"]), 2)
        with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "register.*prepared"):
            self.resource("late")
        self.assertEqual(self.registry.restore()["phase"], "restored")
        self.assertEqual(self.calls, [("a", "prepare"), ("b", "prepare"), ("a", "restore"), ("b", "restore")])
        with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "restore.*active"):
            self.registry.restore()
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)

    def test_failed_prepare_rolls_back_completed_resources_in_reverse(self):
        resources = [self.resource(name) for name in ("a", "b", "c")]
        resources[-1].fail_prepare = True
        with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "prepare failed"):
            self.registry.prepare()
        self.assertEqual(self.calls, [("a", "prepare"), ("b", "prepare"), ("c", "prepare"), ("b", "restore"), ("a", "restore")])
        resources[-1].fail_prepare = False
        self.registry.prepare()
        self.registry.restore()

    def test_rollback_failure_and_restore_failure_are_terminal(self):
        a, b = self.resource("a"), self.resource("b")
        a.fail_restore, b.fail_prepare = True, True
        with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "rollback failed"):
            self.registry.prepare()
        with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "while failed"):
            self.registry.prepare()
        registry = checkpoint.ResourceRegistry()
        registry.register(a, kind="a")
        registry.prepare()
        with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "restore failed"):
            registry.restore()
        with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "while failed"):
            registry.restore()

    def test_registry_does_not_extend_active_resource_lifetime(self):
        resource = self.resource("gone")
        reference = weakref.ref(resource)
        del resource
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(self.registry.prepare()["resources"], [])
        self.registry.restore()

    def test_unmanaged_ring_rejection_rolls_back_managed_resources(self):
        resource = self.resource("managed")
        with patch.object(checkpoint, "assert_no_io_uring", side_effect=checkpoint.CheckpointResourceError("PID 9 fd 12")):
            with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "PID 9 fd 12"):
                self.registry.prepare()
        self.assertEqual(resource.calls, [("managed", "prepare"), ("managed", "restore")])

    def test_descriptor_guard_checks_each_process_and_ignores_vanished_fds(self):
        def entries(path):
            return [path / "12"] if path.name == "fd" else []
        with patch.object(Path, "iterdir", entries), patch.object(checkpoint.os, "readlink", return_value="anon_inode:[io_uring]"):
            with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "PID 9 fd 12, PID 10 fd 12"):
                checkpoint.assert_no_io_uring([9, 10])
        with patch.object(Path, "iterdir", entries), patch.object(checkpoint.os, "readlink", side_effect=FileNotFoundError):
            checkpoint.assert_no_io_uring([9])
        with patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "cannot inspect"):
                checkpoint.assert_no_io_uring([9])

    def test_kernel_worker_guard_rejects_a_worker_after_all_rings_are_closed(self):
        def entries(path):
            return [] if path.name == "fd" else [Path("/proc/9/task/12")]
        with patch.object(Path, "iterdir", entries), patch.object(Path, "read_text", return_value="iou-wrk-9\n"):
            with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "PID 9 thread 12"):
                checkpoint.assert_no_io_uring([9])

    def test_event_loop_factory_restricts_only_a_joined_constructor_thread(self):
        from coldsnap_core import io_uring
        caller = threading.get_ident()
        calls = []
        def restrict():
            calls.append(("filter", threading.current_thread()))
        def factory():
            calls.append(("factory", threading.current_thread()))
            return "epoll-loop"
        def run(main, *, loop_factory, debug=False):
            return main, loop_factory(), debug
        module = SimpleNamespace(new_event_loop=factory, run=run)
        self.assertTrue(vllm_resources._install_event_loop_hooks(module))
        self.assertFalse(vllm_resources._install_event_loop_hooks(module))
        with patch.object(io_uring, "block_io_uring_for_current_thread", restrict):
            self.assertEqual(module.new_event_loop(), "epoll-loop")
            self.assertEqual(module.run("coroutine", debug=True), ("coroutine", "epoll-loop", True))
            with self.assertRaisesRegex(ValueError, "constructor failed"):
                io_uring.without_io_uring(lambda: (_ for _ in ()).throw(ValueError("constructor failed")))
        self.assertEqual([kind for kind, _ in calls], ["filter", "factory", "filter", "factory", "filter"])
        for _, thread in calls:
            self.assertNotEqual(thread.ident, caller)
            self.assertFalse(thread.is_alive())

    def test_tcp_store_factory_preserves_arguments_and_backend_on_restricted_thread(self):
        from coldsnap_core import io_uring
        caller = threading.current_thread()
        calls = []
        result = object()
        def factory(*args, **kwargs):
            calls.append((threading.current_thread(), args, kwargs))
            return result
        module = SimpleNamespace(_create_c10d_store=factory)
        self.assertTrue(vllm_resources._install_tcp_store_hooks(module))
        self.assertFalse(vllm_resources._install_tcp_store_hooks(module))
        with patch.object(io_uring, "block_io_uring_for_current_thread") as restrict:
            self.assertIs(module._create_c10d_store("host", 42, use_libuv=True), result)
        restrict.assert_called_once_with()
        thread, args, kwargs = calls[0]
        self.assertIsNot(thread, caller)
        self.assertFalse(thread.is_alive())
        self.assertEqual(args, ("host", 42))
        self.assertEqual(kwargs, {"use_libuv": True})
        with self.assertRaisesRegex(checkpoint.CheckpointResourceError, "TCP-store factory"):
            vllm_resources._install_tcp_store_hooks(SimpleNamespace())

    def test_event_loop_and_tcp_store_hooks_only_apply_to_initialized_checkpoints(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(vllm_resources, "after_module_import") as hook:
            self.assertFalse(vllm_resources.install_checkpoint_event_loops())
            hook.assert_not_called()
            os.environ["COLDSNAP_CHECKPOINT_EVENT_LOOP"] = "epoll"
            hook.return_value = True
            self.assertTrue(vllm_resources.install_checkpoint_event_loops())
            self.assertEqual([call.args[0] for call in hook.call_args_list], ["uvloop", "torch.distributed.rendezvous"])

    def test_worker_wrapper_preserves_engine_evidence_and_orders_resources(self):
        calls = self.calls
        resource = self.resource("reader")
        class Worker:
            def checkpoint_prepare(self):
                calls.append(("engine", "prepare"))
                return {"nccl": "prepared"}
            def checkpoint_restore(self):
                calls.append(("engine", "restore"))
                return {"nccl": "restored"}
        with patch.object(vllm_resources, "REGISTRY", self.registry):
            module = SimpleNamespace(GPUWorker=Worker)
            self.assertTrue(vllm_resources._install_worker_hooks(module))
            self.assertFalse(vllm_resources._install_worker_hooks(module))
            worker = Worker()
            self.assertEqual(worker.checkpoint_prepare()["nccl"], "prepared")
            report = worker.checkpoint_restore()
            self.assertEqual(report["nccl"], "restored")
            self.assertEqual(report["checkpoint_resources"]["phase"], "restored")
        self.assertEqual(resource.calls, [("engine", "prepare"), ("reader", "prepare"), ("engine", "restore"), ("reader", "restore")])


if __name__ == "__main__":
    unittest.main()

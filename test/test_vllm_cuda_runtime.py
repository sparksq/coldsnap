# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
import coldsnap_vllm_cuda_runtime as runtime  # noqa: E402


class FakeAllocator:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    @contextmanager
    def use_memory_pool(self, tag: str):
        self.events.append(("enter", tag))
        try:
            yield
        finally:
            self.events.append(("exit", tag))


def fake_runner_class() -> type:
    class FakeRunner:
        def __init__(self, vllm_config, device="cuda:0") -> None:
            self.vllm_config = vllm_config
            self.device = device
            self.parallel_config = SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
            )
            self.speculative_config = None
            self.use_async_scheduling = False
            self.cudagraph_batch_sizes = []
            self.async_output_copy_stream = None
            self.prepare_inputs_event = None
            self.transfer_event = object()
            self.input_batch = SimpleNamespace(
                num_reqs=0,
                sampled_token_ids_cpu=None,
                async_copy_ready_event=None,
                generators={},
            )
            self.requests = {}
            self.execute_model_state = None
            self.kv_connector_output = None
            self.encoder_cache = {}

        def _init_kv_zero_meta(self) -> str:
            return "initialized"

        def execute_model(self) -> str:
            return "executed"

        def sample_tokens(self) -> str:
            return "sampled"

    return FakeRunner


class FakeCudaResource:
    def __init__(self) -> None:
        self.synchronized = 0

    def synchronize(self) -> None:
        self.synchronized += 1


class DraftTokensHandler:
    def __init__(self) -> None:
        self.copy_stream = FakeCudaResource()
        self.copy_event = FakeCudaResource()


DraftTokensHandler.__module__ = "vllm.v1.worker.gpu.spec_decode.utils"


def fake_v2_runner_class() -> type:
    class FakeV2Runner:
        def __init__(self, vllm_config, device="cuda:0") -> None:
            self.vllm_config = vllm_config
            self.device = device
            self.parallel_config = SimpleNamespace(
                tensor_parallel_size=2,
                pipeline_parallel_size=1,
                data_parallel_size=1,
            )
            self.model_config = vllm_config.model_config
            self.scheduler_config = SimpleNamespace(async_scheduling=False)
            self.speculative_config = SimpleNamespace(method="dspark")
            self.output_copy_stream = FakeCudaResource()
            self.req_states = SimpleNamespace(num_reqs=0)
            self.execute_model_state = None
            self.encoder_cache = {}
            self.cudagraph_manager = SimpleNamespace(graphs={}, pool=None)
            self._cudagraph_pool_anchor = None
            self._sps_debug_events = None
            self.draft_tokens_handler = DraftTokensHandler()
            self.verification_capacity_manager = None
            self.speculator = SimpleNamespace(
                get_cudagraph_managers=lambda: (),
                _captured_backbone_outputs=[],
                model=None,
            )
            self.model = None

        def _init_kv_zero_meta(self) -> str:
            return "initialized-v2"

        def execute_model(self) -> str:
            return "executed-v2"

        def sample_tokens(self) -> str:
            return "sampled-v2"

    return FakeV2Runner


def admit_runner(allocator: FakeAllocator | None = None):
    runner = fake_runner_class()(SimpleNamespace())
    scope = runtime._RuntimePoolScope(allocator or FakeAllocator())
    scope.enter()
    setattr(runner, runtime._POOL_SCOPE_ATTR, scope)
    runtime._register_runner(runner)
    return runner, scope


def admit_v2_runner(allocator: FakeAllocator | None = None):
    config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=True, enforce_eager=True)
    )
    runner = fake_v2_runner_class()(config)
    setattr(runner, runtime._RUNNER_KIND_ATTR, runtime._RUNNER_V2)
    scope = runtime._RuntimePoolScope(allocator or FakeAllocator())
    scope.enter()
    setattr(runner, runtime._POOL_SCOPE_ATTR, scope)
    runtime._register_runner(runner)
    return runner, scope


class VllmCudaRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        current = runtime._runner_ref() if runtime._runner_ref is not None else None
        scope = getattr(current, runtime._POOL_SCOPE_ATTR, None)
        if isinstance(scope, runtime._RuntimePoolScope) and scope.active:
            scope.exit()
        runtime._runner_ref = None
        runtime._execution_tls.generations = {}
        bootstrap = getattr(runtime._bootstrap_tls, "scope", None)
        if bootstrap is not None:
            del runtime._bootstrap_tls.scope
            if bootstrap.active:
                bootstrap.exit()
        if hasattr(runtime._bootstrap_tls, "worker"):
            del runtime._bootstrap_tls.worker

    def test_runner_construction_uses_runtime_pool(self) -> None:
        allocator = FakeAllocator()
        device_allocator = ModuleType("vllm.device_allocator")
        device_allocator.get_mem_allocator_instance = lambda: allocator
        runner_module = ModuleType("vllm.v1.worker.gpu_model_runner")
        runner_module.GPUModelRunner = fake_runner_class()
        runtime._install_runner_hook(runner_module)
        config = SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=True)
        )

        with patch.dict(sys.modules, {"vllm.device_allocator": device_allocator}):
            runner = runner_module.GPUModelRunner(config)
            self.assertEqual(runner._init_kv_zero_meta(), "initialized")
            runtime._runtime_pool_scope(runner).exit()

        self.assertEqual(
            allocator.events,
            [
                ("enter", runtime.RUNTIME_TAG),
                ("enter", runtime.RUNTIME_KV_META_TAG),
                ("exit", runtime.RUNTIME_KV_META_TAG),
                ("exit", runtime.RUNTIME_TAG),
            ],
        )
        self.assertIs(runtime._current_runner(), runner)

    def test_runner_construction_requires_sleep_mode(self) -> None:
        allocator = FakeAllocator()
        device_allocator = ModuleType("vllm.device_allocator")
        device_allocator.get_mem_allocator_instance = lambda: allocator
        runner_module = ModuleType("vllm.v1.worker.gpu_model_runner")
        runner_module.GPUModelRunner = fake_runner_class()
        runtime._install_runner_hook(runner_module)
        config = SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=False)
        )

        with (
            patch.dict(sys.modules, {"vllm.device_allocator": device_allocator}),
            self.assertRaisesRegex(runtime.VllmContractError, "sleep mode"),
        ):
            runner_module.GPUModelRunner(config)

    def test_v2_runner_construction_admits_eager_dspark(self) -> None:
        allocator = FakeAllocator()
        device_allocator = ModuleType("vllm.device_allocator")
        device_allocator.get_mem_allocator_instance = lambda: allocator
        runner_module = ModuleType("vllm.v1.worker.gpu.model_runner")
        runner_module.GPUModelRunner = fake_v2_runner_class()
        runtime._install_runner_hook(runner_module)
        config = SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=True, enforce_eager=True)
        )

        with patch.dict(sys.modules, {"vllm.device_allocator": device_allocator}):
            runner = runner_module.GPUModelRunner(config)
            self.assertEqual(runner._init_kv_zero_meta(), "initialized-v2")
            runtime._runtime_pool_scope(runner).exit()

        self.assertEqual(runtime._runner_kind(runner), runtime._RUNNER_V2)
        self.assertEqual(
            allocator.events,
            [
                ("enter", runtime.RUNTIME_TAG),
                ("enter", runtime.RUNTIME_KV_META_TAG),
                ("exit", runtime.RUNTIME_KV_META_TAG),
                ("exit", runtime.RUNTIME_TAG),
            ],
        )

    def test_v2_seed_bootstrap_pool_is_adopted_by_runner(self) -> None:
        allocator = FakeAllocator()
        device_allocator = ModuleType("vllm.device_allocator")
        device_allocator.get_mem_allocator_instance = lambda: allocator
        runner_module = ModuleType("vllm.v1.worker.gpu.model_runner")
        runner_module.GPUModelRunner = fake_v2_runner_class()
        runtime._install_runner_hook(runner_module)
        worker_module = ModuleType("vllm.v1.worker.gpu_worker")
        events = []

        def set_random_seed(seed) -> None:
            events.append(("seed", seed))

        config = SimpleNamespace(
            use_v2_model_runner=True,
            model_config=SimpleNamespace(enable_sleep_mode=True, enforce_eager=True),
        )

        class FakeWorker:
            def __init__(self) -> None:
                self.vllm_config = config
                self.use_v2_model_runner = True

            def init_device(self) -> None:
                worker_module.set_random_seed(7)
                self.model_runner = runner_module.GPUModelRunner(config)

            def _maybe_get_memory_pool_context(self, _tag: str):
                raise AssertionError("upstream weight-pool path reached")

            def wake_up(self) -> None:
                pass

        worker_module.set_random_seed = set_random_seed
        worker_module.GPUWorker = FakeWorker
        runtime._install_worker_hook(worker_module)

        with patch.dict(sys.modules, {"vllm.device_allocator": device_allocator}):
            worker = worker_module.GPUWorker()
            worker.init_device()

        scope = runtime._runtime_pool_scope(worker.model_runner)
        self.assertTrue(scope.active)
        self.assertEqual(events, [("seed", 7)])
        self.assertEqual(allocator.events, [("enter", runtime.RUNTIME_TAG)])
        self.assertIsNone(getattr(runtime._bootstrap_tls, "scope", None))

    def test_v2_runner_rejects_non_dspark_speculation(self) -> None:
        runner, _ = admit_v2_runner()
        runner.speculative_config.method = "mtp"
        with self.assertRaisesRegex(runtime.VllmContractError, "only DSpark"):
            runtime._require_runner_contract(runner)

    def test_v2_resource_recipe_covers_copy_and_current_streams(self) -> None:
        runner, _ = admit_v2_runner()
        runner.main_stream = FakeCudaResource()

        recipes = runtime._resource_recipes(runner)

        self.assertEqual(
            [recipe.name for recipe in recipes],
            [
                "output_copy_stream",
                "main_stream",
                "draft_tokens_handler.copy_stream",
                "draft_tokens_handler.copy_event",
            ],
        )
        runtime._quiescent_runner(runner)
        self.assertEqual(runner.draft_tokens_handler.copy_event.synchronized, 1)

    def test_deepseek_v4_shared_streams_and_event_pools_rebuild_in_place(self) -> None:
        class DeepseekV4B12xMLAAttention:
            def __init__(self, aux_streams) -> None:
                self.aux_stream_list = aux_streams
                self.ln_events = [FakeCudaResource() for _ in range(4)]
                self.attn_event_pool = SimpleNamespace(
                    default_events=[FakeCudaResource() for _ in range(3)],
                    _captured_event_sets=[],
                )

        DeepseekV4B12xMLAAttention.__module__ = (
            "vllm.models.deepseek_v4.nvidia.b12x"
        )
        shared_streams = [FakeCudaResource() for _ in range(3)]
        owners = [
            DeepseekV4B12xMLAAttention(shared_streams),
            DeepseekV4B12xMLAAttention(shared_streams),
        ]
        runner, _ = admit_v2_runner()
        runner.model = SimpleNamespace(modules=lambda: iter(owners))

        recipes = runtime._deepseek_v4_resource_recipes(runner)

        self.assertEqual(len(recipes), 5)
        self.assertEqual(
            sum(recipe.kind == "cuda_stream_list" for recipe in recipes), 1
        )
        for recipe in recipes:
            runtime._clear_resource(recipe, runner)
        self.assertEqual(shared_streams, [None, None, None])
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=lambda **kwargs: ("stream", kwargs),
                Event=lambda **kwargs: ("event", kwargs),
            )
        )
        for recipe in recipes:
            runtime._rebuild_resource(recipe, runner, fake_torch)

        self.assertTrue(all(value[0] == "stream" for value in shared_streams))
        self.assertTrue(
            all(value[0] == "event" for value in owners[0].ln_events)
        )

    def test_worker_weight_pool_accepts_admitted_runtime_usage(self) -> None:
        allocator = FakeAllocator()
        runner = fake_runner_class()(SimpleNamespace())
        runtime._register_runner(runner)
        device_allocator = ModuleType("vllm.device_allocator")
        device_allocator.get_mem_allocator_instance = lambda: allocator
        worker_module = ModuleType("vllm.v1.worker.gpu_worker")

        class FakeWorker:
            def __init__(self) -> None:
                self.model_runner = runner

            def _maybe_get_memory_pool_context(self, tag: str):
                raise AssertionError(f"upstream path reached for {tag}")

            def wake_up(self) -> None:
                pass

        worker_module.GPUWorker = FakeWorker
        runtime._install_worker_hook(worker_module)

        with (
            patch.dict(sys.modules, {"vllm.device_allocator": device_allocator}),
            worker_module.GPUWorker()._maybe_get_memory_pool_context("weights"),
        ):
            pass

        self.assertEqual(
            allocator.events,
            [("enter", "weights"), ("exit", "weights")],
        )

    def test_worker_exposes_collective_epoch_reset_and_resume(self) -> None:
        events: list[object] = []
        lifecycle = SimpleNamespace(
            reset=lambda: events.append("reset") or {"state": "RESET"},
            probe_context=lambda: events.append("probe_context")
            or {"schema": 1, "policy_changed": False},
            rebind=lambda: events.append("rebind") or {"state": "REBOUND"},
            resume=lambda callback: events.append("resume") or callback(),
            inventory=lambda: {"state": "ACTIVE"},
        )

        class FakeEpoch:
            @staticmethod
            def capture(*, externalize_active: bool):
                events.append(("capture", externalize_active))
                return lifecycle

        class FakeWorker:
            def _maybe_get_memory_pool_context(self, _tag: str):
                return None

            def wake_up(self) -> None:
                events.append("wake_up")

        worker_module = ModuleType("vllm.v1.worker.gpu_worker")
        worker_module.GPUWorker = FakeWorker
        epoch_module = ModuleType("coldsnap_cuda_epoch")
        epoch_module.VllmCudaEpoch = FakeEpoch
        runtime._install_worker_hook(worker_module)
        worker = FakeWorker()

        self.assertTrue(
            callable(worker.coldsnap_cuda_epoch_nccl_probe)
        )
        self.assertTrue(callable(worker.coldsnap_cuda_epoch_context_probe))

        with patch.dict(sys.modules, {"coldsnap_cuda_epoch": epoch_module}):
            self.assertEqual(
                worker.coldsnap_cuda_epoch_reset(), {"state": "RESET"}
            )
            self.assertEqual(
                worker.coldsnap_cuda_epoch_status(), {"state": "ACTIVE"}
            )
            self.assertEqual(
                worker.coldsnap_cuda_epoch_context_probe(),
                {"schema": 1, "policy_changed": False},
            )
            self.assertEqual(
                worker.coldsnap_cuda_epoch_rebind(), {"state": "REBOUND"}
            )
            self.assertEqual(
                worker.coldsnap_cuda_epoch_resume(), {"state": "ACTIVE"}
            )
            self.assertEqual(
                worker.coldsnap_cuda_epoch_last_status(), {"state": "ACTIVE"}
            )

        self.assertEqual(
            events,
            [
                ("capture", True),
                "reset",
                "probe_context",
                "rebind",
                "resume",
                "wake_up",
            ],
        )
        self.assertIsNone(worker.coldsnap_cuda_epoch_status())

    def test_runner_contract_allows_cross_process_tensor_parallelism(self) -> None:
        runner = fake_runner_class()(SimpleNamespace())
        runner.parallel_config.tensor_parallel_size = 2
        runtime._require_runner_contract(runner)

    def test_cpu_gpu_buffer_forces_pageable_backing(self) -> None:
        buffer_module = ModuleType("vllm.v1.utils")
        calls: list[dict[str, object]] = []

        class FakeBuffer:
            def __init__(self, *args, **kwargs) -> None:
                calls.append(kwargs)

        buffer_module.CpuGpuBuffer = FakeBuffer
        runtime._install_buffer_hook(buffer_module)

        buffer_module.CpuGpuBuffer(4, dtype="int32", pin_memory=True)

        self.assertEqual(calls, [{"dtype": "int32", "pin_memory": False}])

    def test_v2_uva_buffers_use_pageable_cpu_and_managed_cuda_mirror(self) -> None:
        class FakeTensor:
            def __init__(self, size, *, device, pinned=False):
                self.values = [0] * int(size)
                self.device = device
                self.pinned = pinned

            def numpy(self):
                return self

            def __len__(self):
                return len(self.values)

            def __getitem__(self, key):
                view = FakeTensor(len(self.values[key]), device=self.device)
                view.values = self.values[key]
                view._parent = self
                view._key = key
                return view

            def __setitem__(self, key, value):
                values = value.values if isinstance(value, FakeTensor) else list(value)
                self.values[key] = values

            def copy_(self, source, non_blocking=False):
                self.values = list(source.values)
                if hasattr(self, "_parent"):
                    self._parent.values[self._key] = self.values
                self.non_blocking = non_blocking
                return self

        fake_torch = ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.device = lambda kind, index: f"{kind}:{index}"
        fake_torch.cuda = SimpleNamespace(current_device=lambda: 0)
        fake_torch.zeros = lambda size, **kwargs: FakeTensor(
            size,
            device=kwargs["device"],
            pinned=kwargs.get("pin_memory", False),
        )

        module = ModuleType("vllm.v1.worker.gpu.buffer_utils")

        class UvaBuffer:
            def __init__(self, size, dtype):
                raise AssertionError("original UVA constructor must not run")

        class UvaBufferPool:
            def __init__(self, size, dtype):
                self.max_concurrency = 2
                self._curr = 0
                self._uva_bufs = [UvaBuffer(size, dtype) for _ in range(2)]

            def copy_to_uva(self, value):
                raise AssertionError("original UVA copy must not run")

        module.UvaBuffer = UvaBuffer
        module.UvaBufferPool = UvaBufferPool
        with patch.dict(sys.modules, {"torch": fake_torch}):
            runtime._install_v2_buffer_hook(module)
            pool = UvaBufferPool(8, "int32")
            mirror = pool.copy_to_uva([1, 2, 3])

        selected = pool._uva_bufs[1]
        self.assertFalse(selected.cpu.pinned)
        self.assertEqual(selected.uva.device, "cuda:0")
        self.assertEqual(mirror.values, [1, 2, 3])
        self.assertFalse(mirror.non_blocking)

    def test_v2_async_output_releases_completed_cuda_and_pinned_state(self) -> None:
        output_module = ModuleType("vllm.v1.worker.gpu.async_utils")

        class FakeTensor:
            def __init__(self) -> None:
                self.to_calls = []

            def to(self, device, *, non_blocking):
                self.to_calls.append((device, non_blocking))
                return self

            def numpy(self):
                return "pageable-array"

        def async_copy_to_np(tensor):
            return tensor.to("cpu", non_blocking=True).numpy()

        class FakeAsyncOutput:
            def __init__(self) -> None:
                for attribute in (
                    "sampler_output",
                    "num_sampled_tokens",
                    "routed_experts",
                    "_has_fault",
                    "main_stream",
                    "copy_stream",
                    "copy_event",
                    "sampled_token_ids",
                    "logprobs_tensors",
                    "num_nans",
                    "num_sampled_tokens_np",
                    "routed_experts_cpu",
                    "draft_token_ids_np",
                    "draft_req_ids",
                ):
                    setattr(self, attribute, object())
                self.prompt_logprobs_dict = {0: object()}
                self.model_runner_output = object()

            def get_output(self):
                return self.model_runner_output

        output_module.AsyncOutput = FakeAsyncOutput
        output_module.async_copy_to_np = async_copy_to_np
        runtime._install_async_output_hook(output_module)
        tensor = FakeTensor()
        output = FakeAsyncOutput()

        array = output_module.async_copy_to_np(tensor)
        result = output.get_output()

        self.assertEqual(array, "pageable-array")
        self.assertEqual(tensor.to_calls, [("cpu", False)])
        self.assertIs(result, output.model_runner_output)
        self.assertIsNone(output.sampled_token_ids)
        self.assertIsNone(output.copy_stream)
        self.assertIsNone(output.copy_event)
        self.assertEqual(output.prompt_logprobs_dict, {})

    def test_capture_releases_worker_side_completed_v2_async_output(self) -> None:
        class AsyncOutput:
            pass

        AsyncOutput.__module__ = "vllm.v1.worker.gpu.async_utils"
        AsyncOutput.__qualname__ = "AsyncOutput"
        output = AsyncOutput()
        output.copy_event_recorded = True
        output.copy_event = FakeCudaResource()
        output.sampled_token_ids = SimpleNamespace(nbytes=40)
        output.prompt_logprobs_dict = {0: object()}

        with patch.object(runtime.gc, "get_objects", return_value=[output]):
            records = runtime._release_completed_v2_async_outputs()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].pinned_array_bytes, 40)
        self.assertIsNone(output.sampled_token_ids)
        self.assertIsNone(output.copy_event)
        self.assertEqual(output.prompt_logprobs_dict, {})

    def test_capture_audits_storage_and_records_resources(self) -> None:
        runner, _ = admit_runner()
        with (
            patch.dict(os.environ, {runtime.CUDA_EPOCH_RUNTIME_ENV: "1"}),
            patch.object(
                runtime,
                "_live_tensor_audit",
                return_value=(
                    {(0x101000, 4096), (0x201000, 8192)},
                    set(),
                    [],
                    [],
                    [],
                ),
            ),
            patch.object(runtime, "_invalidate_b12x_context", return_value=True),
        ):
            captured = runtime.capture_cuda_epoch_runtime(
                ((0x100000, 65536), (0x200000, 65536))
            )

        self.assertIsNotNone(captured)
        assert captured is not None
        self.assertEqual(captured.cuda_storage_count, 2)
        self.assertEqual(
            captured.inventory()["resource_names"],
            ["transfer_event"],
        )
        self.assertTrue(captured.inventory()["b12x_cache_reset"])

    def test_capture_allows_completed_request_cache_and_clears_async_event(self) -> None:
        runner, _ = admit_runner()
        runner.requests = {"finished": SimpleNamespace(generator=None)}
        runner.input_batch.num_reqs = 1
        runner.input_batch.sampled_token_ids_cpu = object()
        runner.input_batch.async_copy_ready_event = object()
        with (
            patch.dict(os.environ, {runtime.CUDA_EPOCH_RUNTIME_ENV: "1"}),
            patch.object(
                runtime,
                "_live_tensor_audit",
                return_value=(set(), set(), [], [], []),
            ),
        ):
            captured = runtime.capture_cuda_epoch_runtime(((0x100000, 65536),))

        self.assertIsNotNone(captured)
        self.assertIsNone(runner.input_batch.sampled_token_ids_cpu)
        self.assertIsNone(runner.input_batch.async_copy_ready_event)

    def test_capture_rejects_unmanaged_cuda_storage(self) -> None:
        runner, _ = admit_runner()
        with (
            patch.dict(os.environ, {runtime.CUDA_EPOCH_RUNTIME_ENV: "1"}),
            patch.object(
                runtime,
                "_live_tensor_audit",
                return_value=(
                    {(0x300000, 4096)},
                    set(),
                    [(0x300000, 4096)],
                    [],
                    [],
                ),
            ),
            self.assertRaisesRegex(runtime.VllmContractError, "outside managed VMM"),
        ):
            runtime.capture_cuda_epoch_runtime(((0x100000, 65536),))

    def test_pre_sleep_audit_reports_unmanaged_cuda_storage(self) -> None:
        runner, _ = admit_runner()
        with (
            patch.dict(os.environ, {runtime.CUDA_EPOCH_RUNTIME_ENV: "1"}),
            patch.object(
                runtime,
                "_live_tensor_audit",
                return_value=(
                    {(0x101000, 4096), (0x300000, 32)},
                    set(),
                    [(0x300000, 32)],
                    [],
                    [{"x": 1}],
                ),
            ),
            patch.object(runtime, "_invalidate_b12x_context", return_value=True),
        ):
            audit = runtime.audit_cuda_epoch_allocations(((0x100000, 65536),))

        self.assertEqual(audit["live_cuda_storage_count"], 2)
        self.assertEqual(audit["unmanaged_cuda_storage_count"], 1)
        self.assertEqual(audit["unmanaged_cuda_storage_sample"], [(0x300000, 32)])
        self.assertEqual(audit["unmanaged_cuda_storage_diagnostics"], [{"x": 1}])
        self.assertTrue(audit["b12x_cache_reset"])
        self.assertIsNotNone(runner)

    def test_cuda_diagnostics_do_not_require_pinned_storage(self) -> None:
        fake_torch = ModuleType("torch")

        class Tensor:
            device = SimpleNamespace(type="cuda")
            shape = (8,)
            dtype = "torch.int32"

            def untyped_storage(self):
                return SimpleNamespace(data_ptr=lambda: 0x300000, nbytes=lambda: 32)

            def is_pinned(self):
                return False

            def stride(self):
                return (1,)

        fake_torch.Tensor = Tensor
        tensor = Tensor()
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(runtime.gc, "get_objects", return_value=[tensor]),
            patch.object(runtime.gc, "get_referrers", return_value=[]),
        ):
            diagnostics = runtime._cuda_tensor_diagnostics({(0x300000, 32)})

        self.assertEqual(diagnostics[0]["shape"], (8,))
        self.assertEqual(diagnostics[0]["dtype"], "torch.int32")

    def test_cuda_allocation_history_resolves_tensor_subextent(self) -> None:
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            memory=SimpleNamespace(
                _snapshot=lambda: {
                    "segments": [
                        {
                            "address": 0x300000,
                            "blocks": [
                                {
                                    "size": 1024,
                                    "requested_size": 32,
                                    "state": "active_allocated",
                                    "history": [
                                        {
                                            "addr": 0x300000,
                                            "real_size": 32,
                                            "frames": [
                                                {
                                                    "filename": "bootstrap.py",
                                                    "line": 17,
                                                    "name": "allocate",
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            )
        )
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(runtime, "_memory_history_enabled", True),
        ):
            history = runtime._cuda_allocation_history({(0x300000, 32)})

        self.assertEqual(history[0]["requested_size"], 32)
        self.assertEqual(history[0]["frames"][0]["filename"], "bootstrap.py")

    def test_capture_rejects_async_scheduling(self) -> None:
        runner, _ = admit_runner()
        runner.use_async_scheduling = True
        runner.async_output_copy_stream = object()
        runner.prepare_inputs_event = object()
        with (
            patch.dict(os.environ, {runtime.CUDA_EPOCH_RUNTIME_ENV: "1"}),
            self.assertRaisesRegex(
                runtime.VllmContractError, "synchronous scheduling"
            ),
        ):
            runtime.capture_cuda_epoch_runtime(((0x100000, 65536),))

    def test_capture_rejects_live_pinned_storage(self) -> None:
        runner, _ = admit_runner()
        with (
            patch.dict(os.environ, {runtime.CUDA_EPOCH_RUNTIME_ENV: "1"}),
            patch.object(
                runtime,
                "_live_tensor_audit",
                return_value=(
                    set(),
                    {(0x400000, 4096)},
                    [],
                    [],
                    [],
                ),
            ),
            self.assertRaisesRegex(runtime.VllmContractError, "pinned CPU"),
        ):
            runtime.capture_cuda_epoch_runtime(((0x100000, 65536),))

    def test_flashinfer_pinned_workspaces_are_dehydrated_and_rebuilt(self) -> None:
        class FakeTensor:
            def __init__(self, *, pinned: bool, data: bytes = b"workspace") -> None:
                self.device = SimpleNamespace(type="cpu")
                self.shape = (len(data),)
                self.dtype = "uint8"
                self._pinned = pinned
                self.data = data

            def stride(self):
                return (1,)

            def is_pinned(self):
                return self._pinned

            def copy_(self, source):
                self.data = source.data
                return self

            def untyped_storage(self):
                return SimpleNamespace(nbytes=lambda: len(self.data))

        class BatchPrefillWithPagedKVCacheWrapper:
            pass

        BatchPrefillWithPagedKVCacheWrapper.__module__ = "flashinfer.prefill"
        BatchPrefillWithPagedKVCacheWrapper.__qualname__ = (
            "BatchPrefillWithPagedKVCacheWrapper"
        )
        owner = BatchPrefillWithPagedKVCacheWrapper()
        original = FakeTensor(pinned=True)
        owner._pin_memory_int_workspace_buffer = original
        fake_torch = ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.empty_strided = lambda _shape, _stride, **kwargs: FakeTensor(
            pinned=bool(kwargs["pin_memory"]), data=b""
        )

        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(runtime.gc, "get_objects", return_value=[owner]),
        ):
            recipes = runtime._dehydrate_flashinfer_workspaces()
            pageable = owner._pin_memory_int_workspace_buffer
            self.assertFalse(pageable.is_pinned())
            self.assertEqual(pageable.data, b"workspace")
            self.assertEqual(len(recipes), 1)
            self.assertEqual(recipes[0].bytes, len(b"workspace"))
            recipes[0].rehydrate(fake_torch)

        rebuilt = owner._pin_memory_int_workspace_buffer
        self.assertTrue(rebuilt.is_pinned())
        self.assertEqual(rebuilt.data, b"workspace")

    def test_rebuild_replaces_streams_and_events(self) -> None:
        allocator = FakeAllocator()
        runner, scope = admit_runner(allocator)
        fake_torch = ModuleType("torch")
        thread_events: list[object] = []
        fake_torch.cuda = SimpleNamespace(
            Stream=lambda **kwargs: ("stream", kwargs),
            Event=lambda **kwargs: ("cuda-event", kwargs),
            default_stream=lambda device: ("default-stream", device),
            set_device=lambda device: thread_events.append(("device", device)),
            set_stream=lambda stream: thread_events.append(("stream", stream)),
        )
        fake_torch.Event = lambda: ("torch-event", {})
        fake_torch._C = SimpleNamespace(_host_emptyCache=lambda: None)
        fake_torch.compiler = SimpleNamespace(
            reset=lambda: thread_events.append("compiler-reset")
        )
        captured = runtime.VllmCudaRuntime(
            runner,
            runtime._resource_recipes(runner),
            scope,
            cuda_storage_count=1,
            cuda_storage_bytes=4096,
        )
        native = SimpleNamespace(
            activate=lambda device: thread_events.append(("activate", device))
        )

        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(
                runtime,
                "_invalidate_triton_context",
                return_value=(4, 3, 2),
            ),
        ):
            captured.seal()
            captured.prepare_reset()
            captured.rebuild(native)

        self.assertIsNone(runner.async_output_copy_stream)
        self.assertIsNone(runner.prepare_inputs_event)
        self.assertEqual(runner.transfer_event, ("torch-event", {}))
        self.assertTrue(scope.active)
        self.assertEqual(captured.inventory()["triton_kernel_count"], 4)
        self.assertEqual(captured.inventory()["triton_function_count"], 3)
        self.assertEqual(captured.inventory()["triton_autotuner_count"], 2)
        self.assertTrue(captured.inventory()["compiler_cache_reset"])
        self.assertEqual(
            thread_events,
            [
                "compiler-reset",
                ("activate", 0),
                ("device", "cuda:0"),
                ("stream", ("default-stream", "cuda:0")),
                ("activate", 0),
            ],
        )
        self.assertEqual(
            allocator.events,
            [
                ("enter", runtime.RUNTIME_TAG),
                ("exit", runtime.RUNTIME_TAG),
                ("enter", f"{runtime.RUNTIME_EPOCH_TAG_PREFIX}1"),
            ],
        )


if __name__ == "__main__":
    unittest.main()

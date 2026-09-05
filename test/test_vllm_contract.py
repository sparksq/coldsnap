# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Source-level contracts for an explicitly selected vLLM checkout.

These tests avoid importing vLLM's compiled extension and GPU dependencies.
Set VLLM_SOURCE_ROOT to exercise them against every supported checkout in CI.
"""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
from coldsnap_vllm import supported_prepare_parameters  # noqa: E402


class VllmSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configured = os.environ.get("VLLM_SOURCE_ROOT")
        if not configured:
            raise unittest.SkipTest("set VLLM_SOURCE_ROOT to a vLLM source checkout")
        cls.root = Path(configured).resolve()
        if not (cls.root / "vllm").is_dir():
            raise RuntimeError(f"VLLM_SOURCE_ROOT has no vllm package: {cls.root}")

    @classmethod
    def parse(cls, relative: str) -> ast.Module:
        path = cls.root / relative
        if not path.is_file():
            raise AssertionError(f"required vLLM source is absent: {path}")
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    @staticmethod
    def find_class(module: ast.Module, name: str) -> ast.ClassDef:
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        raise AssertionError(f"class {name} is absent")

    @staticmethod
    def find_method(owner: ast.ClassDef, name: str) -> ast.FunctionDef:
        for node in owner.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"method {owner.name}.{name} is absent")

    def test_default_loader_prepare_contract_is_understood(self) -> None:
        module = self.parse("vllm/model_executor/model_loader/default_loader.py")
        loader = self.find_class(module, "DefaultModelLoader")
        source = self.find_class(loader, "Source")
        source_fields = {
            node.target.id
            for node in source.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        prepare = self.find_method(loader, "_prepare_weights")
        parameter_names = [argument.arg for argument in prepare.args.args[1:]]
        unknown = set(parameter_names) - supported_prepare_parameters()
        self.assertEqual(
            unknown,
            set(),
            f"coldsnap does not adapt _prepare_weights parameters: {sorted(unknown)}",
        )
        required_source_fields = {
            "model_or_path",
            "subfolder",
            "revision",
            "fall_back_to_pt",
            "allow_patterns_overrides",
            "prefix",
        }
        self.assertEqual(required_source_fields - source_fields, set())

    def test_public_model_loader_registry_exists(self) -> None:
        module = self.parse("vllm/model_executor/model_loader/__init__.py")
        functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("register_model_loader", functions)

    def test_optional_rpc_transport_hook_has_required_parameters(self) -> None:
        module = self.parse("vllm/distributed/device_communicators/shm_broadcast.py")
        queue = self.find_class(module, "MessageQueue")
        initializer = self.find_method(queue, "__init__")
        parameters = {argument.arg for argument in initializer.args.args[1:]}
        self.assertEqual({"n_reader", "n_local_reader"} - parameters, set())

    def test_sleep_backend_exposes_public_named_registration(self) -> None:
        module = self.parse("vllm/device_allocator/sleep_mode_backend.py")
        factory = self.find_class(module, "SleepModeBackendFactory")
        self.find_method(factory, "register_backend")
        self.find_method(factory, "get_backend_class")
        self.find_method(factory, "create_backend")

        model_config = self.find_class(self.parse("vllm/config/model.py"), "ModelConfig")
        fields = {
            node.target.id
            for node in model_config.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertIn("sleep_mode_backend", fields)

    def test_process_checkpoint_worker_hooks_exist(self) -> None:
        module = self.parse("vllm/v1/worker/gpu_worker.py")
        worker = self.find_class(module, "Worker")
        self.find_method(worker, "init_device")
        self.find_method(worker, "load_model")
        self.find_method(worker, "wake_up")
        self.find_method(worker, "checkpoint_prepare")
        self.find_method(worker, "checkpoint_restore")

        worker_methods = {node.name for node in worker.body if isinstance(node, ast.FunctionDef)}
        if "_warmup_kernels_once" not in worker_methods:
            compile_or_warm_up = self.find_method(worker, "compile_or_warm_up_model")
            calls = {
                ast.unparse(node)
                for node in ast.walk(compile_or_warm_up)
                if isinstance(node, ast.Call)
            }
            self.assertIn("kernel_warmup(self)", calls)
            imports = {
                alias.name
                for node in module.body
                if isinstance(node, ast.ImportFrom)
                and node.module == "vllm.model_executor.warmup.kernel_warmup"
                for alias in node.names
            }
            self.assertIn("kernel_warmup", imports)

    def test_worker_publishes_allocator_independent_capacity_inputs(self) -> None:
        module = self.parse("vllm/v1/worker/gpu_worker.py")
        worker = self.find_class(module, "Worker")
        init_device = self.find_method(worker, "init_device")
        determine = self.find_method(worker, "determine_available_memory")

        def assigned_self_attributes(function: ast.FunctionDef) -> set[str]:
            names: set[str] = set()
            for node in ast.walk(function):
                targets: list[ast.expr] = []
                if isinstance(node, ast.Assign):
                    targets.extend(node.targets)
                elif isinstance(node, ast.AnnAssign):
                    targets.append(node.target)
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        names.add(target.attr)
            return names

        self.assertIn("requested_memory", assigned_self_attributes(init_device))
        required = {
            "total_consumed",
            "peak_activation_memory",
            "available_kv_cache_memory_bytes",
        }
        self.assertEqual(
            required - assigned_self_attributes(determine),
            set(),
        )
        calls = {
            ast.unparse(node.func) for node in ast.walk(determine) if isinstance(node, ast.Call)
        }
        self.assertIn("reserve_mm_ipc_gpu_memory", calls)

    def test_runner_families_publish_semantic_kv_payload_tensors(self) -> None:
        for relative in (
            "vllm/v1/worker/gpu/model_runner.py",
            "vllm/v1/worker/gpu_model_runner.py",
        ):
            with self.subTest(relative=relative):
                runner = self.find_class(self.parse(relative), "GPUModelRunner")
                self.find_method(runner, "initialize_kv_cache")
                assignments = [
                    node
                    for node in ast.walk(runner)
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                ]
                published = False
                for assignment in assignments:
                    targets = (
                        assignment.targets
                        if isinstance(assignment, ast.Assign)
                        else [assignment.target]
                    )
                    if any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == "kv_caches"
                        for target in targets
                    ):
                        published = True
                        break
                self.assertTrue(
                    published,
                    f"{relative} no longer publishes self.kv_caches",
                )

    def test_kv_post_wake_follows_backend_resume(self) -> None:
        worker = self.find_class(self.parse("vllm/v1/worker/gpu_worker.py"), "Worker")
        wake = self.find_method(worker, "wake_up")
        calls = [
            (node.lineno, ast.unparse(node.func))
            for node in ast.walk(wake)
            if isinstance(node, ast.Call)
        ]

        def first_line(suffix: str) -> int:
            for line, name in calls:
                if name.endswith(suffix):
                    return line
            raise AssertionError(f"Worker.wake_up no longer calls {suffix}")

        self.assertLess(
            first_line("._get_sleep_mode_backend().resume"),
            first_line(".model_runner.post_kv_cache_wake_up"),
        )

    def test_worker_plugin_load_precedes_concrete_class_resolution(self) -> None:
        module = self.parse("vllm/v1/worker/worker_base.py")
        wrapper = self.find_class(module, "WorkerWrapperBase")
        init_worker = self.find_method(wrapper, "init_worker")
        calls = [
            (node.lineno, node.func.id)
            for node in ast.walk(init_worker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"load_general_plugins", "resolve_obj_by_qualname"}
        ]
        plugin_lines = [line for line, name in calls if name == "load_general_plugins"]
        resolve_lines = [line for line, name in calls if name == "resolve_obj_by_qualname"]
        self.assertEqual(len(plugin_lines), 1)
        self.assertGreaterEqual(len(resolve_lines), 1)
        self.assertLess(plugin_lines[0], min(resolve_lines))

    def test_direct_cuda_graph_capture_sites_are_structurally_adapted(self) -> None:
        relative = "vllm/compilation/breakable_cudagraph.py"
        module = self.parse(relative)
        capture = self.find_class(module, "BreakableCUDAGraphCapture")
        begin = self.find_method(capture, "_begin_segment")
        end = self.find_method(capture, "_end_segment")

        begin_calls = [
            node
            for node in ast.walk(begin)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "capture_begin"
        ]
        end_calls = [
            node
            for node in ast.walk(end)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "capture_end"
        ]
        self.assertGreaterEqual(len(begin_calls), 1)
        self.assertEqual(len(end_calls), 1)

        direct_capture_files: set[str] = set()
        for path in (self.root / "vllm").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"capture_begin", "capture_end"}
                for node in ast.walk(tree)
            ):
                direct_capture_files.add(str(path.relative_to(self.root)))
        self.assertEqual(direct_capture_files, {relative})

    def test_deepseek_v4_snapshot_boundary_precedes_finalizers(self) -> None:
        module = self.parse("vllm/models/deepseek_v4/nvidia/model.py")
        model = self.find_class(module, "DeepseekV4ForCausalLM")
        load_weights = self.find_method(model, "load_weights")
        calls = [
            (ast.unparse(node.func), node.lineno)
            for node in ast.walk(load_weights)
            if isinstance(node, ast.Call)
        ]

        def position(suffix: str) -> int:
            for value, lineno in calls:
                if value.endswith(suffix):
                    return lineno
            raise AssertionError(f"DeepSeek V4 load_weights no longer calls {suffix}")

        load = position("loader.load_weights")
        direct_finalizers = {
            value: lineno
            for value, lineno in calls
            if value.endswith(
                (
                    "self.model.finalize_mega_moe_weights",
                    "self.model.finalize_mhc_broadcast_weights",
                )
            )
        }
        if direct_finalizers:
            self.assertEqual(len(direct_finalizers), 2)
            self.assertTrue(all(load < lineno for lineno in direct_finalizers.values()))
            return

        process = position("self.process_weights_after_loading")
        self.assertLess(load, process)
        finalizer_method = self.find_method(model, "process_weights_after_loading")
        delegated_calls = {
            ast.unparse(node.func)
            for node in ast.walk(finalizer_method)
            if isinstance(node, ast.Call)
        }
        self.assertIn("self.model.finalize_mega_moe_weights", delegated_calls)
        self.assertIn("self.model.finalize_mhc_broadcast_weights", delegated_calls)

    def test_qwen35_snapshot_boundary_is_auto_weights_loader(self) -> None:
        module = self.parse("vllm/model_executor/models/qwen3_5.py")
        base = self.find_class(module, "Qwen3_5ForCausalLMBase")
        load_weights = self.find_method(base, "load_weights")
        constructor_names = {
            ast.unparse(node.func) for node in ast.walk(load_weights) if isinstance(node, ast.Call)
        }
        self.assertIn("AutoWeightsLoader", constructor_names)
        self.assertIn("loader.load_weights", constructor_names)

    def test_deferred_graph_capture_has_safe_upstream_boundaries(self) -> None:
        worker_module = self.parse("vllm/v1/worker/gpu_worker.py")
        workers = [
            node
            for node in worker_module.body
            if isinstance(node, ast.ClassDef) and node.name in {"Worker", "GPUWorker"}
        ]
        self.assertEqual(len(workers), 1)
        worker = workers[0]
        compile_or_warm_up = self.find_method(worker, "compile_or_warm_up_model")
        capture_calls = [
            node
            for node in ast.walk(compile_or_warm_up)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "capture_model"
        ]
        self.assertEqual(len(capture_calls), 1)

        dispatcher = self.find_class(
            self.parse("vllm/v1/cudagraph_dispatcher.py"),
            "CudagraphDispatcher",
        )
        self.find_method(dispatcher, "dispatch")
        manager = self.find_class(
            self.parse("vllm/v1/worker/gpu/cudagraph_utils.py"),
            "CudaGraphManager",
        )
        self.find_method(manager, "dispatch")

        engine = self.find_class(self.parse("vllm/v1/engine/core.py"), "EngineCoreProc")
        process_step = self.find_method(engine, "_process_engine_step")
        output_lines = [
            node.lineno
            for node in ast.walk(process_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "put_nowait"
        ]
        post_step_lines = [
            node.lineno
            for node in ast.walk(process_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "post_step"
        ]
        self.assertTrue(output_lines)
        self.assertEqual(len(post_step_lines), 1)
        self.assertLess(min(output_lines), post_step_lines[0])

    def test_shape_calibration_mirrors_the_graph_capture_warmup(self) -> None:
        """The eager sweep must stay a faithful copy of vLLM's warmup half.

        Shape calibration reproduces the `_dummy_run` calls that
        `_warmup_and_capture` performs before recording a graph. If the
        dispatcher accessor or those parameters move, the sweep would compile a
        different set of kernels than capture needs and the activation misses
        would return silently.
        """
        module = self.parse("vllm/v1/worker/gpu_model_runner.py")
        runner = self.find_class(module, "GPUModelRunner")

        dummy_run = self.find_method(runner, "_dummy_run")
        parameters = {argument.arg for argument in dummy_run.args.args[1:]}
        parameters |= {argument.arg for argument in dummy_run.args.kwonlyargs}
        required = {
            "num_tokens",
            "cudagraph_runtime_mode",
            "force_attention",
            "uniform_decode",
        }
        self.assertEqual(
            required - parameters,
            set(),
            "coldsnap shape calibration needs these _dummy_run parameters",
        )

        capture_model = self.find_method(runner, "capture_model")
        accessors = {
            node.func.attr
            for node in ast.walk(capture_model)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("get_capture_descs", accessors)

        warmup = self.find_method(runner, "_warmup_and_capture")
        eager_warmup = [
            node
            for node in ast.walk(warmup)
            if isinstance(node, ast.keyword)
            and node.arg == "cudagraph_runtime_mode"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "NONE"
        ]
        self.assertTrue(
            eager_warmup,
            "_warmup_and_capture no longer warms with CUDAGraphMode.NONE",
        )

    def test_v2_shape_calibration_mirrors_the_manager_warmup(self) -> None:
        """The v2 runner owns capture in a CudaGraphManager, not a dispatcher.

        DS4F selects `use_v2_model_runner=True`, so this is the path that
        actually runs in production. Calibration replaces the manager's capture
        loop with only its warmup half, which requires the factory parameter,
        the descriptor mapping, and the eager warmup call to stay put.
        """
        module = self.parse("vllm/v1/worker/gpu/cudagraph_utils.py")
        manager = self.find_class(module, "CudaGraphManager")

        capture = self.find_method(manager, "capture")
        parameters = [argument.arg for argument in capture.args.args]
        self.assertEqual(
            parameters[1],
            "create_forward_fn",
            "coldsnap intercepts CudaGraphManager.capture by its factory argument",
        )

        # The descriptor mapping the calibration pass enumerates.
        descriptor_reads = {
            node.attr
            for node in ast.walk(capture)
            if isinstance(node, ast.Attribute) and node.attr == "_capture_descs"
        }
        self.assertIn("_capture_descs", descriptor_reads)

        # The warmup half: create_forward_fn(desc, warmup=True) then an eager call.
        warmup_calls = [
            node
            for node in ast.walk(capture)
            if isinstance(node, ast.Call)
            and any(
                keyword.arg == "warmup"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
        ]
        self.assertTrue(
            warmup_calls,
            "CudaGraphManager.capture no longer builds a warmup forward function",
        )
        eager_calls = [
            node
            for node in ast.walk(capture)
            if isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr == "NONE"
        ]
        self.assertTrue(
            eager_calls,
            "CudaGraphManager.capture no longer warms with CUDAGraphMode.NONE",
        )

        # ModelCudaGraphManager is the subclass that builds the factory and
        # delegates to the base loop intercepted above.
        self.find_class(module, "ModelCudaGraphManager")


if __name__ == "__main__":
    unittest.main()

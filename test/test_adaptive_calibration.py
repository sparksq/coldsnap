# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Cache admission, collective consistency, and graph-only calibration provenance."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
import coldsnap_vllm_calibration as calibration

CURVES = {"draft": [[1, 1.0], [6, 3.0]], "verify": [[6, 4.0], [36, 7.0]], "cudagraph_limit": 36}


class Manager:
    def __init__(self):
        self.cost_tables = None
        self._cudagraph_limit = 36
        self.calls = []

    def set_cost_curves(self, draft, verify):
        self.calls.append((draft, verify, self._cudagraph_limit))
        self.cost_tables = (draft, verify)

    def batches_to_profile(self, sizes):
        self._cudagraph_limit = max(sizes)
        return iter([{"num_tokens": size} for size in sizes])

    def set_initial_cost_curves(self, samples):
        raise AssertionError("shape-only warmup must not recalibrate")


class AdaptiveCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.identity = {"image": "immutable", "rank": 0}
        self.enterContext(patch.dict(os.environ, {"VLLM_CACHE_ROOT": self.temporary.name}))
        self.enterContext(patch.object(calibration, "_identity", side_effect=lambda worker: copy.deepcopy(self.identity)))
        self.broadcast = self.enterContext(patch.object(calibration, "_broadcast", side_effect=lambda value: value))
        self.worker = self.make_worker()

    def make_worker(self):
        return SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=1),
            model_runner=SimpleNamespace(
                adaptive_verification=Manager(),
                cudagraph_manager=SimpleNamespace(_graphs_captured=True),
            ),
        )

    def save(self, worker=None):
        worker = worker or self.worker
        state = calibration.prepare_calibration(worker)
        worker.model_runner.adaptive_verification.set_cost_curves(CURVES["draft"], CURVES["verify"])
        return state

    def test_synchronous_capture_saves_and_next_worker_seeds_before_inference(self):
        state = self.save()
        self.assertTrue(state["saved"])
        worker = self.make_worker()
        self.assertIsNone(worker.model_runner.adaptive_verification.cost_tables)
        self.assertTrue(calibration.reuse_calibration(worker))
        self.assertEqual(worker.model_runner.adaptive_verification.cost_tables, (CURVES["draft"], CURVES["verify"]))
        self.assertEqual(calibration.calibration_status(worker)["decision"], "reused")
        self.assertFalse(calibration.calibration_status(worker)["saved"])
        self.assertEqual(state["path"].parent.name, "adaptive-calibration")

    def test_persists_rank_zero_accepted_values_not_local_timings(self):
        self.broadcast.side_effect = lambda value: CURVES
        state = calibration.prepare_calibration(self.worker)
        self.worker.model_runner.adaptive_verification.set_cost_curves([[1, 99.0]], [[1, 123.0]])
        self.assertEqual(calibration._load(state), CURVES)

    def test_no_graph_record_during_eager_only_warmup(self):
        self.worker.model_runner.cudagraph_manager._graphs_captured = False
        state = self.save()
        self.assertFalse(state["saved"])
        self.assertFalse(state["path"].exists())

    def test_readonly_cache_does_not_break_successful_calibration(self):
        with patch.object(calibration, "_save", side_effect=PermissionError("read only")):
            self.save()
        self.assertIsNotNone(self.worker.model_runner.adaptive_verification.cost_tables)
        self.assertIn("read only", calibration.calibration_status(self.worker)["save_error"])

    def test_upstream_rejection_is_not_cached(self):
        self.worker.model_runner.adaptive_verification.set_cost_curves = lambda *args: (_ for _ in ()).throw(RuntimeError("upstream rejected"))
        state = calibration.prepare_calibration(self.worker)
        with self.assertRaisesRegex(RuntimeError, "upstream rejected"):
            self.worker.model_runner.adaptive_verification.set_cost_curves([], [])
        self.assertFalse(state["path"].exists())
        self.broadcast.assert_not_called()

    def test_missing_or_changed_identity_falls_back(self):
        self.assertFalse(calibration.reuse_calibration(self.worker))
        self.save()
        self.identity["image"] = "different"
        self.assertFalse(calibration.reuse_calibration(self.make_worker()))

    def test_one_worker_miss_makes_local_hit_fall_back_before_curve_broadcast(self):
        self.save()
        self.broadcast.reset_mock()
        worker = self.make_worker()
        with patch.object(calibration, "_all_workers_admit_startup_plan", return_value=False) as admit:
            self.assertFalse(calibration.reuse_calibration(worker))
        admit.assert_called_once_with(worker, True)
        self.broadcast.assert_not_called()
        self.assertIsNone(worker.model_runner.adaptive_verification.cost_tables)

    def test_different_rank_payload_falls_back_before_installing_tables(self):
        self.save()
        self.broadcast.side_effect = lambda value: "different-rank-zero-digest"
        worker = self.make_worker()
        self.assertFalse(calibration.reuse_calibration(worker))
        self.assertIsNone(worker.model_runner.adaptive_verification.cost_tables)

    def test_invalid_records_fall_back(self):
        state = self.save()
        valid = state["path"].read_bytes()
        record = json.loads(valid)
        bad_records = [b"{", b"x" * (calibration._MAX_RECORD_BYTES + 1)]
        for key, value in (("schema", True), ("schema", 2), ("identity", {}), ("sha256", "wrong")):
            bad = copy.deepcopy(record)
            bad[key] = value
            bad_records.append(json.dumps(bad).encode())
        for draft in ([], [[1, float("nan")]], [[1, float("inf")]], [[1, 0]], [[1, -2]], [[0, 2]], [[True, 2]], [[2, 2], [1, 2]], [[1, 2], [1, 3]]):
            bad = copy.deepcopy(record)
            bad["curves"]["draft"] = draft
            bad_records.append(json.dumps(bad).encode())
        for payload in bad_records:
            with self.subTest(payload=payload[:100]):
                state["path"].write_bytes(payload)
                worker = self.make_worker()
                self.assertFalse(calibration.reuse_calibration(worker))
                self.assertIsNone(worker.model_runner.adaptive_verification.cost_tables)

    def test_shape_only_pass_preserves_tables_limit_and_methods_even_on_error(self):
        manager = self.worker.model_runner.adaptive_verification
        manager.cost_tables = tables = object()
        with self.assertRaisesRegex(RuntimeError, "warmup"):
            with calibration.preserve_calibration_during_shape_warmup(self.worker.model_runner):
                self.assertEqual(list(manager.batches_to_profile([8])), [])
                manager.set_initial_cost_curves([])
                raise RuntimeError("warmup")
        self.assertIs(manager.cost_tables, tables)
        self.assertEqual(manager._cudagraph_limit, 36)
        self.assertNotIn("batches_to_profile", vars(manager))
        self.assertNotIn("set_initial_cost_curves", vars(manager))

    def test_install_observer_is_independent_of_async_setting(self):
        worker = self.worker
        class Worker:
            def compile_or_warm_up_model(self):
                worker.model_runner.adaptive_verification.set_cost_curves(CURVES["draft"], CURVES["verify"])
                return "compiled"
        module = SimpleNamespace(Worker=Worker)
        calibration._install_worker_observer(module)
        calibration._install_worker_observer(module)
        self.assertEqual(Worker.compile_or_warm_up_model(worker), "compiled")
        self.assertTrue(calibration.calibration_status(worker)["saved"])

    def test_nonadaptive_older_images_are_unchanged(self):
        worker = SimpleNamespace(model_runner=SimpleNamespace())
        self.assertTrue(calibration.reuse_calibration(worker))
        self.assertIsNone(calibration.calibration_status(worker))
        self.broadcast.assert_not_called()


class CalibrationIdentityTest(unittest.TestCase):
    def setUp(self):
        spec = SimpleNamespace(method="dspark", enable_adaptive_verification=True, model=None, draft_model_config=None, num_speculative_tokens=5)
        config = SimpleNamespace(
            speculative_config=spec,
            model_config=SimpleNamespace(dtype="bfloat16", max_model_len=250000),
            compilation_config=SimpleNamespace(cudagraph_mode="FULL_AND_PIECEWISE", cudagraph_capture_sizes=[6, 12, 36]),
            scheduler_config=SimpleNamespace(max_num_seqs=6, max_num_batched_tokens=8192),
            cache_config=SimpleNamespace(block_size=256, cache_dtype="fp8"),
        )
        self.worker = SimpleNamespace(vllm_config=config, rank=0,
            parallel_config=SimpleNamespace(world_size=4, tensor_parallel_size=4, pipeline_parallel_size=1, enable_expert_parallel=True),
            model_runner=SimpleNamespace(adaptive_verification=SimpleNamespace(req_states=SimpleNamespace(max_num_reqs=6, max_num_batched_tokens=8192, num_speculative_steps=5))),
            _coldsnap_startup_plan_initial_fingerprint="initial",
            _coldsnap_startup_plan_configured_model_len=250000,
        )
        self.envs = SimpleNamespace(VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=8192)
        self.enterContext(patch.dict(sys.modules, {"vllm": SimpleNamespace(envs=self.envs), "vllm.envs": self.envs}))
        self.enterContext(patch.dict(os.environ, {"COLDSNAP_SOURCE_RUNTIME_IMAGE": "sha256:" + "a" * 64, "COLDSNAP_MODEL_ID": "model", "COLDSNAP_MODEL_REVISION": "revision"}, clear=True))
        self.hardware = self.enterContext(patch.object(calibration, "_hardware", return_value={"driver": "580.1", "name": "GB10"}))

    def test_each_material_dimension_changes_key(self):
        initial = calibration._digest(calibration._identity(self.worker))
        dimensions = (
            (self.worker, "rank", 1), (self.worker.parallel_config, "tensor_parallel_size", 2),
            (self.worker.parallel_config, "enable_expert_parallel", False),
            (self.worker.vllm_config.compilation_config, "cudagraph_capture_sizes", [6, 24]),
            (self.worker.vllm_config.scheduler_config, "max_num_seqs", 4),
            (self.worker.vllm_config.scheduler_config, "max_num_batched_tokens", 4096),
            (self.worker.vllm_config.speculative_config, "num_speculative_tokens", 4),
            (self.worker.vllm_config.cache_config, "cache_dtype", "auto"),
            (self.envs, "VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", 1024),
        )
        for owner, name, value in dimensions:
            with self.subTest(name=name), patch.object(owner, name, value):
                self.assertNotEqual(initial, calibration._digest(calibration._identity(self.worker)))
        for name in ("COLDSNAP_MODEL_ID", "COLDSNAP_MODEL_REVISION", "COLDSNAP_SOURCE_RUNTIME_IMAGE"):
            value = "sha256:" + "b" * 64 if "IMAGE" in name else "other"
            with self.subTest(name=name), patch.dict(os.environ, {name: value}):
                self.assertNotEqual(initial, calibration._digest(calibration._identity(self.worker)))
        self.hardware.return_value = {"driver": "610.2", "name": "GB10"}
        self.assertNotEqual(initial, calibration._digest(calibration._identity(self.worker)))

    def test_identity_does_not_include_credentials_or_activation_paths(self):
        initial = calibration._identity(self.worker)
        with patch.dict(os.environ, {"HF_TOKEN": "secret", "COLDSNAP_CAPTURE_ID": "different", "VLLM_CACHE_ROOT": "/different"}):
            self.assertEqual(initial, calibration._identity(self.worker))

    def test_no_reuse_without_pinned_runtime_identity(self):
        with patch.dict(os.environ, {"COLDSNAP_SOURCE_RUNTIME_IMAGE": "image:latest"}):
            with self.assertRaisesRegex(ValueError, "pinned source"):
                calibration._identity(self.worker)


if __name__ == "__main__":
    unittest.main()

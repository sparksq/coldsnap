# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Compile-cache calibration over CUDA graph capture shapes."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))

from coldsnap_vllm import VllmContractError  # noqa: E402
from coldsnap_vllm_shape_calibration import (  # noqa: E402
    SHAPE_CALIBRATION_ENV,
    calibrate_capture_shapes,
    runner_generation,
    shape_calibration_enabled,
)


class FakeMode:
    """Stand-in for vLLM's CUDAGraphMode enum members."""

    def __init__(self, name: str) -> None:
        self.name = name


FakeMode.NONE = FakeMode("NONE")


def descriptor(num_tokens: int, uniform: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        num_tokens=num_tokens, uniform=uniform, num_active_loras=0
    )


class FakeRunner:
    def __init__(self, plan, *, warmups: int = 1, parameters=None) -> None:
        self.cudagraph_dispatcher = SimpleNamespace(
            get_capture_descs=lambda: plan
        )
        self.compilation_config = SimpleNamespace(
            cudagraph_num_of_warmups=warmups
        )
        self.calls: list[dict] = []
        self._parameters = parameters

    def _dummy_run(
        self,
        num_tokens,
        cudagraph_runtime_mode=None,
        force_attention=False,
        uniform_decode=False,
        allow_microbatching=False,
        skip_eplb=True,
        remove_lora=False,
        num_active_loras=None,
    ):
        self.calls.append(
            {
                "num_tokens": num_tokens,
                "mode": cudagraph_runtime_mode,
                "force_attention": force_attention,
                "uniform_decode": uniform_decode,
            }
        )


class ShapeCalibrationTest(unittest.TestCase):
    def test_enabled_only_for_truthy_values(self) -> None:
        for value in ("1", "true", "YES", "on"):
            self.assertTrue(shape_calibration_enabled({SHAPE_CALIBRATION_ENV: value}))
        for value in ("0", "false", "", "no"):
            self.assertFalse(shape_calibration_enabled({SHAPE_CALIBRATION_ENV: value}))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(shape_calibration_enabled())

    def test_every_capture_shape_runs_eagerly_and_records_nothing(self) -> None:
        full = FakeMode("FULL")
        piecewise = FakeMode("PIECEWISE")
        runner = FakeRunner(
            [
                (piecewise, [descriptor(1), descriptor(8)]),
                (full, [descriptor(16), descriptor(32), descriptor(64)]),
            ]
        )
        result = calibrate_capture_shapes(runner)

        self.assertEqual(result["shapes"], 5)
        self.assertEqual(len(runner.calls), 5)
        # Every run must be eager: recording a graph here would retain NCCL
        # communicator resources and break the checkpoint boundary.
        self.assertTrue(all(call["mode"] is FakeMode.NONE for call in runner.calls))
        self.assertEqual(
            [call["num_tokens"] for call in runner.calls], [1, 8, 16, 32, 64]
        )
        # force_attention mirrors vLLM: only FULL forces the attention path,
        # which is where the DS4F misses were.
        self.assertEqual(
            [call["force_attention"] for call in runner.calls],
            [False, False, True, True, True],
        )
        modes = {entry["mode"]: entry for entry in result["modes"]}
        self.assertEqual(modes["PIECEWISE"]["num_tokens"], [1, 8])
        self.assertEqual(modes["FULL"]["shapes"], 3)
        self.assertTrue(modes["FULL"]["force_attention"])

    def test_warmup_count_is_honoured_but_never_zero(self) -> None:
        mode = FakeMode("FULL")
        runner = FakeRunner([(mode, [descriptor(4)])], warmups=3)
        calibrate_capture_shapes(runner)
        self.assertEqual(len(runner.calls), 3)

        # A configured zero would skip the shape entirely and leave the kernel
        # uncompiled, which is the whole failure being fixed.
        skipped = FakeRunner([(mode, [descriptor(4)])], warmups=0)
        result = calibrate_capture_shapes(skipped)
        self.assertEqual(len(skipped.calls), 1)
        self.assertEqual(result["warmups_per_shape"], 1)

    def test_missing_dispatcher_contract_fails_closed(self) -> None:
        runner = FakeRunner([])
        runner.cudagraph_dispatcher = None
        with self.assertRaisesRegex(VllmContractError, "cudagraph_dispatcher"):
            calibrate_capture_shapes(runner)

        runner = FakeRunner([])
        runner.cudagraph_dispatcher = SimpleNamespace()
        with self.assertRaisesRegex(VllmContractError, "get_capture_descs"):
            calibrate_capture_shapes(runner)

    def test_dummy_run_without_eager_mode_is_refused(self) -> None:
        mode = FakeMode("FULL")
        runner = FakeRunner([(mode, [descriptor(4)])])

        # An upstream _dummy_run that lost cudagraph_runtime_mode could silently
        # record a graph instead of warming; refuse rather than guess.
        def narrowed(num_tokens, uniform_decode=False):
            runner.calls.append({"num_tokens": num_tokens})

        runner._dummy_run = narrowed
        with self.assertRaisesRegex(
            VllmContractError, "cudagraph_runtime_mode"
        ):
            calibrate_capture_shapes(runner)
        self.assertEqual(runner.calls, [])

    def test_mode_without_none_member_fails_closed(self) -> None:
        class ModeWithoutNone:
            name = "FULL"

        runner = FakeRunner([(ModeWithoutNone(), [descriptor(4)])])
        with self.assertRaisesRegex(VllmContractError, "no NONE member"):
            calibrate_capture_shapes(runner)

    def test_empty_descriptor_sets_are_skipped(self) -> None:
        runner = FakeRunner([(FakeMode("FULL"), [])])
        result = calibrate_capture_shapes(runner)
        self.assertEqual(result["shapes"], 0)
        self.assertEqual(result["modes"], [])
        self.assertEqual(runner.calls, [])


class FakeDesc:
    def __init__(self, num_tokens: int, cg_mode) -> None:
        self.num_tokens = num_tokens
        self.cg_mode = cg_mode


class FakeCudaGraphManager:
    """Mirrors the v2 base whose capture() takes a forward-function factory."""

    def __init__(self, capture_descs) -> None:
        self._capture_descs = capture_descs
        self.graphs = {"stale": object()}
        self._graphs_captured = False
        self.recorded: list[int] = []

    def capture(self, create_forward_fn, *, channel_id, progress_bar_desc=""):
        for _mode, descs in self._capture_descs.items():
            for desc in descs:
                forward_fn = create_forward_fn(desc, warmup=True)
                forward_fn(FakeMode.NONE)
                # The real loop records here; a calibration pass must not.
                self.recorded.append(desc.num_tokens)
        self._graphs_captured = True

    def clear(self) -> None:
        self.graphs.clear()
        self._graphs_captured = False


class FakeModelCudaGraphManager(FakeCudaGraphManager):
    pass


class FakeV2Runner:
    def __init__(self, manager) -> None:
        self.cudagraph_manager = manager
        self.warmed: list[tuple[int, object]] = []

    def capture_model(self) -> int:
        def create_forward_fn(desc, warmup):
            def forward_fn(cg_mode):
                self.warmed.append((desc.num_tokens, cg_mode, warmup))

            return forward_fn

        self.cudagraph_manager.capture(
            create_forward_fn, channel_id="test", progress_bar_desc="x"
        )
        return 0


class V2ShapeCalibrationTest(unittest.TestCase):
    def _runner(self) -> FakeV2Runner:
        piecewise = FakeMode("PIECEWISE")
        full = FakeMode("FULL")
        manager = FakeModelCudaGraphManager(
            {
                piecewise: [FakeDesc(1, piecewise), FakeDesc(8, piecewise)],
                full: [FakeDesc(16, full)],
            }
        )
        return FakeV2Runner(manager)

    def test_generation_detection_prefers_v2_and_fails_on_neither(self) -> None:
        self.assertEqual(runner_generation(self._runner()), "v2")
        self.assertEqual(
            runner_generation(FakeRunner([(FakeMode("FULL"), [descriptor(1)])])),
            "v1",
        )
        with self.assertRaisesRegex(VllmContractError, "neither"):
            runner_generation(SimpleNamespace())

    def test_v2_warms_every_shape_and_records_no_graph(self) -> None:
        runner = self._runner()
        result = calibrate_capture_shapes(runner)

        self.assertEqual(result["runner"], "v2")
        self.assertEqual(result["shapes"], 3)
        self.assertEqual(result["planned_shapes"], 3)
        self.assertEqual(
            [tokens for tokens, _, _ in runner.warmed], [1, 8, 16]
        )
        # Every warmup runs eagerly, and with vLLM's warmup=True input state.
        self.assertTrue(
            all(mode is FakeMode.NONE for _, mode, _ in runner.warmed)
        )
        self.assertTrue(all(warmup for _, _, warmup in runner.warmed))
        # The recording half must not have run.
        self.assertEqual(runner.cudagraph_manager.recorded, [])
        self.assertFalse(runner.cudagraph_manager._graphs_captured)

    def test_v2_warms_piecewise_before_full_regardless_of_mapping_order(
        self,
    ) -> None:
        """Order is load-bearing, not cosmetic.

        The manager keeps a persistent hidden-state buffer sized by whichever
        descriptor runs first. Warming a narrow FULL shape before a wider
        PIECEWISE one sizes that buffer too small and the wider descriptor then
        fails to expand into it.
        """
        piecewise = FakeMode("PIECEWISE")
        full = FakeMode("FULL")
        # Deliberately insert FULL first, as a dict iteration would yield it.
        manager = FakeModelCudaGraphManager(
            {full: [FakeDesc(48, full)], piecewise: [FakeDesc(64, piecewise)]}
        )
        runner = FakeV2Runner(manager)
        calibrate_capture_shapes(runner)
        self.assertEqual([tokens for tokens, _, _ in runner.warmed], [64, 48])

    def test_v2_accepts_repeated_engine_capture_passes_as_full_coverage(self) -> None:
        runner = self._runner()
        capture_once = runner.capture_model

        def capture_twice() -> int:
            capture_once()
            capture_once()
            return 0

        runner.capture_model = capture_twice
        result = calibrate_capture_shapes(runner)

        self.assertEqual(result["planned_shapes"], 3)
        self.assertEqual(result["warmed_shapes"], 3)
        self.assertEqual(result["warmup_invocations"], 6)

    def test_v2_accepts_reconstructed_equivalent_descriptors(self) -> None:
        runner = self._runner()
        original_plan = runner.cudagraph_manager._capture_descs
        capture_once = runner.capture_model

        def capture_twice_with_fresh_descriptors() -> int:
            capture_once()
            runner.cudagraph_manager._capture_descs = {
                mode: [FakeDesc(desc.num_tokens, desc.cg_mode) for desc in descs]
                for mode, descs in original_plan.items()
            }
            capture_once()
            return 0

        runner.capture_model = capture_twice_with_fresh_descriptors
        result = calibrate_capture_shapes(runner)

        self.assertEqual(result["planned_shapes"], 3)
        self.assertEqual(result["warmed_shapes"], 3)
        self.assertEqual(result["warmup_invocations"], 6)
        self.assertEqual(result["dynamic_shapes"], 0)

    def test_v2_counts_runtime_generated_descriptors_in_the_plan(self) -> None:
        runner = self._runner()
        original_plan = runner.cudagraph_manager._capture_descs
        capture_once = runner.capture_model

        def capture_with_dynamic_descriptor() -> int:
            capture_once()
            runner.cudagraph_manager._capture_descs = {
                **original_plan,
                FakeMode("DYNAMIC"): [
                    FakeDesc(32, FakeMode("DYNAMIC")),
                ],
            }
            capture_once()
            return 0

        runner.capture_model = capture_with_dynamic_descriptor
        result = calibrate_capture_shapes(runner)

        self.assertEqual(result["initial_planned_shapes"], 3)
        self.assertEqual(result["planned_shapes"], 4)
        self.assertEqual(result["warmed_shapes"], 4)
        self.assertEqual(result["dynamic_shapes"], 1)
        self.assertEqual(result["warmup_invocations"], 7)

    def test_v2_restores_capture_and_clears_manager_state(self) -> None:
        runner = self._runner()
        owner = FakeCudaGraphManager
        original = owner.__dict__["capture"]
        calibrate_capture_shapes(runner)

        # The class-level interception must not outlive the pass, or the real
        # capture at activation would silently do nothing.
        self.assertIs(owner.__dict__["capture"], original)
        self.assertEqual(runner.cudagraph_manager.graphs, {})

    def test_v2_restores_capture_even_when_the_pass_fails(self) -> None:
        runner = self._runner()
        owner = FakeCudaGraphManager
        original = owner.__dict__["capture"]

        def explode() -> int:
            raise RuntimeError("warmup exploded")

        runner.capture_model = explode
        with self.assertRaisesRegex(RuntimeError, "warmup exploded"):
            calibrate_capture_shapes(runner)
        self.assertIs(owner.__dict__["capture"], original)

    def test_v2_requires_a_descriptor_mapping_and_capture_model(self) -> None:
        runner = self._runner()
        runner.cudagraph_manager._capture_descs = None
        with self.assertRaisesRegex(VllmContractError, "_capture_descs"):
            calibrate_capture_shapes(runner)

        runner = self._runner()
        runner.capture_model = None
        with self.assertRaisesRegex(VllmContractError, "capture_model"):
            calibrate_capture_shapes(runner)


if __name__ == "__main__":
    unittest.main()

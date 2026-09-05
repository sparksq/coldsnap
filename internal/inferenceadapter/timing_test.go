// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/snapshot"
)

func TestRuntimeRestoreTimingNormalizesIntoForeignUnitClock(t *testing.T) {
	collector := operationtiming.New(time.Now())
	ctx := operationtiming.WithCollector(context.Background(), collector)
	ctx, parent := operationtiming.Start(ctx, "units.ready", nil)
	total, criu, wake, health := 12.0, 4.0, 5.0, 1.0
	origin := time.Now().UnixNano()
	report := runtimeTimingReport{
		Kind: "coldsnap-vllm-cuda-criu-restore-ready", Engine: "vllm", Rank: 0,
		RestoreControllerSeconds: &total, CRIURestoreSeconds: &criu,
		WakeUpSeconds: &wake, HealthWaitSeconds: &health,
		Timeline: []runtimeTimingMark{
			{Name: "restore_begin", UnixNS: origin, SinceRestoreBegin: 0},
			{Name: "criu_restore_end", UnixNS: origin + int64(4*time.Second), SinceRestoreBegin: 4},
			{Name: "wake_up_begin", UnixNS: origin + int64(5*time.Second), SinceRestoreBegin: 5},
			{Name: "wake_up_end", UnixNS: origin + int64(10*time.Second), SinceRestoreBegin: 10},
			{Name: "health_wait_begin", UnixNS: origin + int64(10*time.Second), SinceRestoreBegin: 10},
			{Name: "health_ready", UnixNS: origin + int64(11*time.Second), SinceRestoreBegin: 11},
		},
	}
	unit := snapshot.LaunchUnit{ID: "unit-0", Host: "node-a"}
	if err := importRuntimeRestoreTiming(
		collector, operationtiming.ParentID(ctx), unit, "vllm",
		[]snapshot.Worker{{ID: "worker-0", Unit: unit.ID, ProcessSlot: 0, DeviceSlots: []int{0}}}, report,
	); err != nil {
		t.Fatal(err)
	}
	parent.End(nil)
	timing := collector.Export()
	if err := timing.Validate(); err != nil {
		t.Fatal(err)
	}
	names := make(map[string]snapshot.TimingSpan)
	for _, span := range timing.Spans {
		names[span.Name] = span
	}
	if names["runtime.restore"].Clock != "unit-unit-0" ||
		names["runtime.restore"].Parent != names["units.ready"].ID ||
		names["runtime.weights_wake"].DurationSeconds != 5 ||
		names["runtime.health_wait"].DurationSeconds != 1 {
		t.Fatalf("normalized timing = %#v", timing)
	}
}

func TestRuntimeRestoreTimingIncludesWorkerHydrationAndNCCLPhases(t *testing.T) {
	collector := operationtiming.New(time.Now())
	total, checkpoint, nccl, engineRestore, resume := 20.0, 3.0, 0.4, 1.2, 10.5
	origin := time.Now().UnixNano()
	raw := func(value any) json.RawMessage {
		payload, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return payload
	}
	report := runtimeTimingReport{
		Kind: "coldsnap-vllm-cuda-criu-restore-ready", Engine: "vllm", Rank: 0,
		RestoreControllerSeconds: &total, CheckpointRestoreSeconds: &checkpoint,
		NCCLRestoreSeconds: &nccl, EngineCheckpointRestore: &engineRestore,
		Timeline: []runtimeTimingMark{
			{Name: "restore_begin", UnixNS: origin, SinceRestoreBegin: 0},
			{Name: "rank_barrier_end", UnixNS: origin + int64(4*time.Second), SinceRestoreBegin: 4},
			{Name: "distributed_process_release_end", UnixNS: origin + int64(5*time.Second), SinceRestoreBegin: 5},
			{Name: "async_graph_arm_begin", UnixNS: origin + int64(5*time.Second), SinceRestoreBegin: 5},
			{Name: "async_graph_arm_end", UnixNS: origin + int64(5500*time.Millisecond), SinceRestoreBegin: 5.5},
			{Name: "checkpoint_restore_begin", UnixNS: origin + int64(6*time.Second), SinceRestoreBegin: 6},
			{Name: "checkpoint_restore_end", UnixNS: origin + int64(9*time.Second), SinceRestoreBegin: 9},
		},
		HibernateStates: map[string]runtimeHibernateState{
			"worker-0": {
				State: "running", WorkerID: "worker-0",
				UpdatedUnix:   float64(time.Now().UnixNano()) / float64(time.Second),
				ResumeSeconds: &resume, HydrationBackend: "recovery-safetensors",
				ReadIOMode: "buffered", WeightRecoverySource: "safetensors",
				PhaseSeconds: map[string]json.RawMessage{
					"map_seconds":          raw(0.1),
					"model_reload_seconds": raw(6.0),
					"residual_restore_before": raw(map[string]any{
						"seconds": 0.2, "io_service_seconds": 0.08,
						"cuda_copy_seconds": 0.05,
					}),
					"recovery_loader": raw(map[string]any{
						"total_seconds": 10.5, "loader_seconds": 5.5,
						"io_seconds":         2.0,
						"collective_seconds": 1.0, "backend": "torch",
						"adapter_replay": map[string]any{
							"total_seconds": 5.0, "io_wait_seconds": 1.5,
							"broadcast_seconds": 0.7, "copy_seconds": 0.8,
						},
					}),
				},
			},
		},
	}
	unit := snapshot.LaunchUnit{ID: "unit-0", Host: "node-a"}
	worker := snapshot.Worker{ID: "worker-0", Unit: unit.ID, ProcessSlot: 0, DeviceSlots: []int{0}}
	if err := importRuntimeRestoreTiming(collector, "", unit, "vllm", []snapshot.Worker{worker}, report); err != nil {
		t.Fatal(err)
	}
	timing := collector.Export()
	if err := timing.Validate(); err != nil {
		t.Fatal(err)
	}
	names := make(map[string]snapshot.TimingSpan)
	for _, span := range timing.Spans {
		names[span.Name] = span
	}
	if names["runtime.worker_resume"].Clock != "worker-unit-0-worker-0" ||
		names["runtime.worker_resume"].DurationSeconds != 10.5 ||
		names["runtime.recovery_loader"].DurationSeconds != 10.5 ||
		names["runtime.hydration.residual_before.io_service_total"].DurationSeconds != 0.08 ||
		names["runtime.hydration.residual_before.cuda_copy"].DurationSeconds != 0.05 ||
		names["runtime.adapter_replay.io_wait"].DurationSeconds != 1.5 ||
		names["runtime.nccl_restore"].DurationSeconds != 0.4 ||
		names["runtime.engine_checkpoint_restore"].DurationSeconds != 1.2 ||
		names["runtime.async_graph_arm"].DurationSeconds != 0.5 ||
		names["runtime.distributed_process_release"].DurationSeconds != 1 {
		t.Fatalf("normalized timing = %#v", timing)
	}
	if names["runtime.nccl_restore"].Parent != names["runtime.checkpoint_restore"].ID ||
		names["runtime.recovery_loader"].Parent != names["runtime.worker_resume"].ID ||
		names["runtime.hydration.residual_before.io_service_total"].Parent != names["runtime.worker_resume"].ID ||
		names["runtime.hydration.residual_before.io_service_total"].Attributes["composition"] != "non_additive" ||
		names["runtime.hydration.residual_before.io_service_total"].Attributes["timing_semantics"] != "cumulative_service_time" {
		t.Fatalf("normalized timing parents = %#v", timing)
	}
}

func TestWorkerHydrationServiceTimeIsNotNestedUnderShorterWallPhase(t *testing.T) {
	collector := operationtiming.New(time.Now())
	raw := func(value any) json.RawMessage {
		payload, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return payload
	}
	importWorkerPhaseTiming(
		collector, "worker", "clock", "unit-0", "worker-0",
		map[string]json.RawMessage{
			"model_restore": raw(map[string]any{
				"seconds": 1.5, "io_service_seconds": 3.7,
				"io_wait_seconds": 1.1,
			}),
		},
	)
	spans := make(map[string]snapshot.TimingSpan)
	for _, span := range collector.Export().Spans {
		spans[span.Name] = span
	}
	phase := spans["runtime.hydration.model_payload"]
	service := spans["runtime.hydration.model_payload.io_service_total"]
	wait := spans["runtime.hydration.model_payload.io_wait"]
	if phase.DurationSeconds != 1.5 || service.DurationSeconds != 3.7 || wait.DurationSeconds != 1.1 {
		t.Fatalf("hydration timing = %#v", spans)
	}
	if service.Parent != "worker" || wait.Parent != phase.ID || service.Attributes["overlaps"] != phase.Name {
		t.Fatalf("hydration timing composition = %#v", spans)
	}
}

func TestRuntimeRestoreTimingDoesNotDoubleCountAggregateRecoveryTotal(t *testing.T) {
	collector := operationtiming.New(time.Now())
	raw := func(value any) json.RawMessage {
		payload, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return payload
	}
	phases := map[string]json.RawMessage{
		"recovery_loader": raw(map[string]any{
			"total_seconds":  7.0,
			"loader_seconds": 2.0,
			"adapter_replay": map[string]any{"total_seconds": 5.0},
		}),
	}
	importWorkerPhaseTiming(collector, "worker", "clock", "unit-0", "worker-0", phases)
	timing := collector.Export()
	found := false
	for _, span := range timing.Spans {
		if span.Name == "runtime.recovery_loader" && span.DurationSeconds != 7.0 {
			t.Fatalf("aggregate recovery duration = %v, want 7", span.DurationSeconds)
		}
		if span.Name == "runtime.recovery_loader" {
			found = true
		}
	}
	if !found {
		t.Fatal("aggregate recovery timing span is missing")
	}
}

func TestRuntimeRestoreTimingIncludesSGLangWorkerNCCLWithoutResumeTotal(t *testing.T) {
	collector := operationtiming.New(time.Now())
	total := 4.0
	raw := func(value any) json.RawMessage {
		payload, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return payload
	}
	report := runtimeTimingReport{
		Kind: "coldsnap-sglang-cuda-criu-restore-ready", Engine: "sglang", Rank: 0,
		RestoreControllerSeconds: &total,
		HibernateStates: map[string]runtimeHibernateState{
			"worker-0": {
				State: "running", WorkerID: "worker-0",
				UpdatedUnix: float64(time.Now().UnixNano()) / float64(time.Second),
				NCCL: map[string]json.RawMessage{
					"nccl_restore_seconds":       raw(0.25),
					"nccl_restore_total_seconds": raw(0.3),
				},
			},
		},
	}
	unit := snapshot.LaunchUnit{ID: "unit-0", Host: "node-a"}
	worker := snapshot.Worker{ID: "worker-0", Unit: unit.ID, ProcessSlot: 0, DeviceSlots: []int{0}}
	if err := importRuntimeRestoreTiming(collector, "", unit, "sglang", []snapshot.Worker{worker}, report); err != nil {
		t.Fatal(err)
	}
	timing := collector.Export()
	if err := timing.Validate(); err != nil {
		t.Fatal(err)
	}
	names := make(map[string]snapshot.TimingSpan)
	for _, span := range timing.Spans {
		names[span.Name] = span
	}
	if names["runtime.worker_nccl_restore"].DurationSeconds != 0.3 ||
		names["runtime.nccl_restore"].DurationSeconds != 0.25 ||
		names["runtime.worker_nccl_restore"].Clock != "worker-unit-0-worker-0" {
		t.Fatalf("normalized timing = %#v", timing)
	}
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/sparksq/coldsnap/internal/capsule"
	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/snapshot"
)

type runtimeTimingMark struct {
	Name              string  `json:"name"`
	UnixNS            int64   `json:"unix_ns"`
	SinceRestoreBegin float64 `json:"since_restore_begin_s"`
}

type runtimeTimingReport struct {
	Kind                        string                           `json:"kind"`
	Engine                      string                           `json:"engine"`
	Rank                        int                              `json:"rank"`
	RestoreControllerSeconds    *float64                         `json:"restore_controller_seconds"`
	CRIURestoreSeconds          *float64                         `json:"criu_restore_seconds"`
	NetworkUnlockBarrierSeconds *float64                         `json:"network_unlock_barrier_seconds"`
	RankBarrierWaitSeconds      *float64                         `json:"rank_barrier_wait_seconds"`
	DeferredCUDARestoreSeconds  *float64                         `json:"deferred_cuda_restore_seconds"`
	CUDARestoreBarrierSeconds   *float64                         `json:"cuda_restore_barrier_wait_seconds"`
	CheckpointRestoreSeconds    *float64                         `json:"checkpoint_restore_seconds"`
	WakeUpSeconds               *float64                         `json:"wake_up_seconds"`
	HealthWaitSeconds           *float64                         `json:"health_wait_seconds"`
	NCCLRestoreSeconds          *float64                         `json:"nccl_restore_seconds"`
	EngineCheckpointRestore     *float64                         `json:"engine_checkpoint_restore_seconds"`
	HibernateStates             map[string]runtimeHibernateState `json:"hibernate_states"`
	Timeline                    []runtimeTimingMark              `json:"timeline"`
	PostRestoreResponse         struct {
		Acceptance *streamAcceptanceTiming `json:"coldsnap_acceptance"`
	} `json:"post_restore_response"`
}

type runtimeCaptureTimingReport struct {
	Kind                     string   `json:"kind"`
	Engine                   string   `json:"engine"`
	Rank                     *int     `json:"rank"`
	CaptureControllerSeconds *float64 `json:"capture_controller_seconds"`
	DumpSeconds              *float64 `json:"dump_seconds"`
	DiskSleepSeconds         *float64 `json:"disk_sleep_seconds"`
	NCCLPrepareSeconds       *float64 `json:"nccl_prepare_seconds"`
	NCCLNetworkResetSeconds  *float64 `json:"nccl_network_reset_seconds"`
	NCCLIBQuiesceSeconds     *float64 `json:"nccl_ib_quiesce_seconds"`
	EngineCheckpointPrepare  *float64 `json:"engine_checkpoint_prepare_seconds"`
	ImageBytes               int64    `json:"image_bytes"`
}

type runtimeHibernateState struct {
	State                string                     `json:"state"`
	WorkerID             string                     `json:"worker_id"`
	UpdatedUnix          float64                    `json:"updated_unix"`
	ResumeSeconds        *float64                   `json:"resume_seconds"`
	HydrationBackend     string                     `json:"hydration_backend"`
	ReadIOMode           string                     `json:"read_io_mode"`
	WeightRecoverySource string                     `json:"weight_recovery_source"`
	PhaseSeconds         map[string]json.RawMessage `json:"phase_seconds"`
	NCCL                 map[string]json.RawMessage `json:"nccl"`
}

func (adapter Adapter) collectCaptureTimingReports(
	ctx context.Context,
	request snapshot.Request,
	roots []string,
) error {
	collector := operationtiming.FromContext(ctx)
	if collector == nil {
		return nil
	}
	parent := operationtiming.ParentID(ctx)
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		if unit.Index >= len(roots) || roots[unit.Index] == "" {
			return fmt.Errorf("unit %s capture timing root is unavailable", unit.ID)
		}
		payload, err := adapter.Remote.Run(ctx, unit.Host, "cat", roots[unit.Index]+"/capture.json")
		if err != nil {
			return fmt.Errorf("read unit %s capture timing report: %w", unit.ID, err)
		}
		var report runtimeCaptureTimingReport
		if err := json.Unmarshal(payload, &report); err != nil {
			return fmt.Errorf("decode unit %s capture timing report: %w", unit.ID, err)
		}
		if !strings.HasPrefix(report.Kind, "coldsnap-") || !strings.Contains(report.Kind, "capture") ||
			(report.Rank != nil && *report.Rank < 0) {
			return fmt.Errorf("unit %s capture timing report identity is invalid", unit.ID)
		}
		total := positiveTimingValue(report.CaptureControllerSeconds)
		if total <= 0 {
			return fmt.Errorf("unit %s capture timing has no positive duration", unit.ID)
		}
		clockID := "unit-" + unit.ID
		collector.AddClock(snapshot.TimingClock{
			ID: clockID, Source: "runtime-unit", OriginUnixNS: time.Now().Add(-time.Duration(total * float64(time.Second))).UnixNano(),
			Unit: unit.ID,
		})
		rootID := "runtime-" + unit.ID
		rank := unit.Index
		if report.Rank != nil {
			rank = *report.Rank
		}
		attributes := map[string]string{
			"unit": unit.ID, "host": unit.Host, "engine": request.Launch.Engine,
			"rank": fmt.Sprintf("%d", rank), "placement": "estimated",
		}
		if report.ImageBytes > 0 {
			attributes["image_bytes"] = fmt.Sprintf("%d", report.ImageBytes)
		}
		collector.AddSpan(snapshot.TimingSpan{
			ID: rootID, Parent: parent, Name: "runtime.capture", Clock: clockID,
			DurationSeconds: total, Status: "ok", Attributes: attributes,
		})
		next := 0
		add := func(name string, duration float64) {
			if duration <= 0 {
				return
			}
			next++
			collector.AddSpan(snapshot.TimingSpan{
				ID: fmt.Sprintf("runtime-%s-%d", unit.ID, next), Parent: rootID,
				Name: name, Clock: clockID,
				DurationSeconds: duration, Status: "ok",
				Attributes: map[string]string{"unit": unit.ID, "placement": "aggregate"},
			})
		}
		add("runtime.criu_dump", positiveTimingValue(report.DumpSeconds))
		add("runtime.disk_sleep", positiveTimingValue(report.DiskSleepSeconds))
		add("runtime.nccl_prepare", positiveTimingValue(report.NCCLPrepareSeconds))
		add("runtime.nccl_network_reset", positiveTimingValue(report.NCCLNetworkResetSeconds))
		add("runtime.nccl_ib_quiesce", positiveTimingValue(report.NCCLIBQuiesceSeconds))
		add("runtime.engine_checkpoint_prepare", positiveTimingValue(report.EngineCheckpointPrepare))
		return nil
	})
}

func (adapter Adapter) collectRestoreTimingReports(
	ctx context.Context,
	request snapshot.Request,
	containers []string,
) error {
	collector := operationtiming.FromContext(ctx)
	parent := operationtiming.ParentID(ctx)
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		if unit.Index >= len(containers) || containers[unit.Index] == "" {
			return fmt.Errorf("unit %s timing container is unavailable", unit.ID)
		}
		path := fmt.Sprintf("%s/restore-ready-unit-%s.json", capsule.Root, unit.ID)
		payload, err := adapter.execWorkload(
			ctx, unit.Host, containers[unit.Index], "cat", path,
		)
		if err != nil {
			return fmt.Errorf("read unit %s restore timing report: %w", unit.ID, err)
		}
		var report runtimeTimingReport
		if err := json.Unmarshal(payload, &report); err != nil {
			return fmt.Errorf("decode unit %s restore timing report: %w", unit.ID, err)
		}
		if !strings.HasPrefix(report.Kind, "coldsnap-") || !strings.Contains(report.Kind, "restore") || report.Rank < 0 {
			return fmt.Errorf("unit %s restore timing report identity is invalid", unit.ID)
		}
		if collector != nil {
			if err := importRuntimeRestoreTiming(
				collector, parent, unit, request.Launch.Engine,
				request.Launch.Execution.UnitWorkers(unit.ID), report,
			); err != nil {
				return err
			}
		}
		if unit.Index == 0 && lifecycleActivationState(request) != "warm" {
			if err := adapter.collectStartupTTFT(ctx, request, unit, containers[unit.Index], report); err != nil {
				fmt.Fprintf(adapter.Output, "ColdSnap startup TTFT: unavailable (%v)\n", err)
			}
		}
		return nil
	})
}

func importRuntimeRestoreTiming(
	collector *operationtiming.Collector,
	parent string,
	unit snapshot.LaunchUnit,
	engine string,
	workers []snapshot.Worker,
	report runtimeTimingReport,
) error {
	total := positiveTimingValue(report.RestoreControllerSeconds)
	markers := make(map[string]runtimeTimingMark, len(report.Timeline))
	for _, mark := range report.Timeline {
		if mark.Name == "" || mark.UnixNS <= 0 || !finiteTimingValue(mark.SinceRestoreBegin) {
			return errors.New("runtime restore timing marker is invalid")
		}
		markers[mark.Name] = mark
		if mark.SinceRestoreBegin > total {
			total = mark.SinceRestoreBegin
		}
	}
	if total <= 0 {
		return errors.New("runtime restore timing has no positive duration")
	}
	originUnixNS := time.Now().Add(-time.Duration(total * float64(time.Second))).UnixNano()
	placement := "estimated"
	if begin, ok := markers["restore_begin"]; ok {
		originUnixNS = begin.UnixNS - int64(begin.SinceRestoreBegin*float64(time.Second))
		placement = "reported"
	}
	clockID := "unit-" + unit.ID
	collector.AddClock(snapshot.TimingClock{
		ID: clockID, Source: "runtime-unit", OriginUnixNS: originUnixNS, Unit: unit.ID,
	})
	rootID := "runtime-" + unit.ID
	attributes := map[string]string{
		"unit": unit.ID, "host": unit.Host, "engine": engine,
		"rank": fmt.Sprintf("%d", report.Rank), "placement": placement,
	}
	collector.AddSpan(snapshot.TimingSpan{
		ID: rootID, Parent: parent, Name: "runtime.restore", Clock: clockID,
		DurationSeconds: total, Status: "ok", Attributes: attributes,
	})
	next := 0
	add := func(parentID, name string, start, duration float64, attributes map[string]string) string {
		if duration <= 0 || !finiteTimingValue(start) || !finiteTimingValue(duration) {
			return ""
		}
		next++
		if attributes == nil {
			attributes = map[string]string{"unit": unit.ID}
		}
		id := fmt.Sprintf("runtime-%s-%d", unit.ID, next)
		collector.AddSpan(snapshot.TimingSpan{
			ID: id, Parent: parentID,
			Name: name, Clock: clockID, StartOffsetSeconds: start,
			DurationSeconds: duration, Status: "ok",
			Attributes: attributes,
		})
		return id
	}
	addPair := func(parentID, name, beginName, endName string, fallback *float64) string {
		begin, beginOK := markers[beginName]
		end, endOK := markers[endName]
		if beginOK && endOK && end.SinceRestoreBegin >= begin.SinceRestoreBegin {
			return add(parentID, name, begin.SinceRestoreBegin, end.SinceRestoreBegin-begin.SinceRestoreBegin, nil)
		}
		duration := positiveTimingValue(fallback)
		if duration > 0 {
			return add(parentID, name, 0, duration, map[string]string{
				"unit": unit.ID, "placement": "aggregate",
			})
		}
		return ""
	}
	addPair(rootID, "runtime.criu_restore", "restore_begin", "criu_restore_end", report.CRIURestoreSeconds)
	addPair(rootID, "runtime.cuda_restore", "deferred_cuda_restore_begin", "deferred_cuda_restore_end", report.DeferredCUDARestoreSeconds)
	checkpointID := addPair(rootID, "runtime.checkpoint_restore", "checkpoint_restore_begin", "checkpoint_restore_end", report.CheckpointRestoreSeconds)
	addPair(rootID, "runtime.weights_wake", "wake_up_begin", "wake_up_end", report.WakeUpSeconds)
	addPair(rootID, "runtime.health_wait", "health_wait_begin", "health_ready", report.HealthWaitSeconds)
	addPair(rootID, "runtime.async_graph_arm", "async_graph_arm_begin", "async_graph_arm_end", nil)
	addPair(rootID, "runtime.distributed_process_release", "rank_barrier_end", "distributed_process_release_end", nil)
	addPair(rootID, "runtime.distributed_process_resume", "distributed_cuda_restore_barrier_end", "distributed_process_resume_end", nil)
	addPair(rootID, "runtime.sglang_cuda_hold_release", "distributed_cuda_restore_barrier_end", "distributed_sglang_cuda_hold_release_end", nil)
	checkpointParent := rootID
	checkpointStart := markerOffset(markers, "checkpoint_restore_begin")
	if checkpointID != "" {
		checkpointParent = checkpointID
	}
	if duration := positiveTimingValue(report.NCCLRestoreSeconds); duration > 0 {
		add(checkpointParent, "runtime.nccl_restore", checkpointStart, duration, map[string]string{
			"unit": unit.ID, "placement": "aggregate",
		})
	}
	if duration := positiveTimingValue(report.EngineCheckpointRestore); duration > 0 {
		add(checkpointParent, "runtime.engine_checkpoint_restore", checkpointStart, duration, map[string]string{
			"unit": unit.ID, "placement": "aggregate",
		})
	}
	if duration := positiveTimingValue(report.NetworkUnlockBarrierSeconds); duration > 0 {
		add(rootID, "runtime.network_unlock_barrier", 0, duration, nil)
	}
	if duration := positiveTimingValue(report.RankBarrierWaitSeconds); duration > 0 {
		start := markerOffset(markers, "criu_restore_end")
		add(rootID, "runtime.rank_barrier_wait", start, duration, nil)
	}
	if duration := positiveTimingValue(report.CUDARestoreBarrierSeconds); duration > 0 {
		start := markerOffset(markers, "deferred_cuda_restore_end")
		add(rootID, "runtime.cuda_barrier_wait", start, duration, nil)
	}
	if begin, beginOK := markers["health_ready"]; beginOK {
		if end, endOK := markers["acceptance_ready"]; endOK && end.SinceRestoreBegin >= begin.SinceRestoreBegin {
			add(rootID, "runtime.acceptance", begin.SinceRestoreBegin, end.SinceRestoreBegin-begin.SinceRestoreBegin, nil)
		}
	}
	return importWorkerRestoreTiming(collector, rootID, unit, engine, workers, report.HibernateStates)
}

func importWorkerRestoreTiming(
	collector *operationtiming.Collector,
	parent string,
	unit snapshot.LaunchUnit,
	engine string,
	workers []snapshot.Worker,
	states map[string]runtimeHibernateState,
) error {
	expected := make(map[string]snapshot.Worker, len(workers))
	for _, worker := range workers {
		expected[worker.ID] = worker
	}
	for key, state := range states {
		worker, ok := expected[key]
		if !ok || state.WorkerID != key {
			return fmt.Errorf("unit %s runtime worker timing identity is invalid", unit.ID)
		}
		resume := positiveTimingValue(state.ResumeSeconds)
		ncclTotal := rawTimingValue(state.NCCL["nccl_restore_total_seconds"])
		if ncclTotal <= 0 {
			ncclTotal = rawTimingValue(state.NCCL["nccl_restore_seconds"])
		}
		if resume <= 0 && ncclTotal <= 0 {
			continue
		}
		clockDuration := math.Max(resume, ncclTotal)
		originUnixNS := time.Now().Add(-time.Duration(clockDuration * float64(time.Second))).UnixNano()
		if state.UpdatedUnix > clockDuration && finiteTimingValue(state.UpdatedUnix) && state.UpdatedUnix < float64(math.MaxInt64)/float64(time.Second) {
			originUnixNS = int64((state.UpdatedUnix - clockDuration) * float64(time.Second))
		}
		clockID := "worker-" + unit.ID + "-" + worker.ID
		collector.AddClock(snapshot.TimingClock{
			ID: clockID, Source: "runtime-worker", OriginUnixNS: originUnixNS,
			Unit: unit.ID, Worker: worker.ID,
		})
		rootID := "runtime-" + unit.ID + "-" + worker.ID
		attributes := map[string]string{
			"unit": unit.ID, "host": unit.Host, "engine": engine,
			"worker": worker.ID, "process_slot": fmt.Sprintf("%d", worker.ProcessSlot),
			"device_slots": fmt.Sprint(worker.DeviceSlots), "placement": "reported",
		}
		if state.WeightRecoverySource != "" {
			attributes["weight_provider"] = state.WeightRecoverySource
		}
		if state.HydrationBackend != "" {
			attributes["hydration_backend"] = state.HydrationBackend
		}
		if state.ReadIOMode != "" {
			attributes["read_io_mode"] = state.ReadIOMode
		}
		workerParent := parent
		if resume > 0 {
			collector.AddSpan(snapshot.TimingSpan{
				ID: rootID, Parent: parent, Name: "runtime.worker_resume", Clock: clockID,
				DurationSeconds: resume, Status: "ok", Attributes: attributes,
			})
			workerParent = rootID
			importWorkerPhaseTiming(collector, rootID, clockID, unit.ID, worker.ID, state.PhaseSeconds)
		}
		if ncclTotal > 0 {
			ncclID := rootID + "-nccl"
			collector.AddSpan(snapshot.TimingSpan{
				ID: ncclID, Parent: workerParent, Name: "runtime.worker_nccl_restore", Clock: clockID,
				DurationSeconds: ncclTotal, Status: "ok", Attributes: map[string]string{
					"unit": unit.ID, "worker": worker.ID, "placement": "aggregate",
				},
			})
			if provider := rawTimingValue(state.NCCL["nccl_restore_seconds"]); provider > 0 {
				collector.AddSpan(snapshot.TimingSpan{
					ID: ncclID + "-provider", Parent: ncclID, Name: "runtime.nccl_restore", Clock: clockID,
					DurationSeconds: provider, Status: "ok", Attributes: map[string]string{
						"unit": unit.ID, "worker": worker.ID, "placement": "aggregate",
					},
				})
			}
		}
	}
	return nil
}

func importWorkerPhaseTiming(
	collector *operationtiming.Collector,
	parent, clock, unit, worker string,
	phases map[string]json.RawMessage,
) {
	next := 0
	add := func(parentID, name string, duration float64, extra map[string]string) string {
		if duration <= 0 || !finiteTimingValue(duration) {
			return ""
		}
		next++
		id := fmt.Sprintf("runtime-%s-%s-phase-%d", unit, worker, next)
		attributes := map[string]string{
			"unit": unit, "worker": worker, "placement": "aggregate",
		}
		for key, value := range extra {
			attributes[key] = value
		}
		collector.AddSpan(snapshot.TimingSpan{
			ID: id, Parent: parentID, Name: name, Clock: clock,
			DurationSeconds: duration, Status: "ok", Attributes: attributes,
		})
		return id
	}
	addCumulativeService := func(name string, duration float64, overlaps string) string {
		return add(parent, name, duration, map[string]string{
			"timing_semantics": "cumulative_service_time",
			"composition":      "non_additive",
			"overlaps":         overlaps,
		})
	}
	flat := []struct{ key, name string }{
		{"map_seconds", "runtime.hydration.map"},
		{"disk_read_wait_seconds", "runtime.hydration.io_wait"},
		{"checksum_seconds", "runtime.hydration.checksum"},
		{"cuda_copy_seconds", "runtime.hydration.cuda_copy"},
		{"cuda_enqueue_seconds", "runtime.hydration.cuda_enqueue"},
		{"native_initialization_seconds", "runtime.hydration.native_initialize"},
		{"native_synchronize_seconds", "runtime.hydration.native_sync"},
		{"native_total_seconds", "runtime.hydration.native"},
		{"initialization_seconds", "runtime.hydration.initialize"},
		{"model_reload_seconds", "runtime.hydration.model_reload"},
		{"discard_cache_release_seconds", "runtime.hydration.discard_cache_release"},
		{"discard_remap_seconds", "runtime.hydration.discard_remap"},
		{"discard_map_seconds", "runtime.hydration.discard_map"},
		{"discard_zero_seconds", "runtime.hydration.discard_zero"},
		{"discard_verification_seconds", "runtime.hydration.discard_verify"},
		{"graph_remap_seconds", "runtime.hydration.graph_remap"},
		{"sync_seconds", "runtime.hydration.sync"},
	}
	for _, metric := range flat {
		add(parent, metric.name, rawTimingValue(phases[metric.key]), nil)
	}
	addCumulativeService(
		"runtime.hydration.io_service_total",
		rawTimingValue(phases["disk_read_service_seconds"]),
		"runtime.worker_resume",
	)
	for _, metric := range []struct{ key, name string }{
		{"model_restore", "runtime.hydration.model_payload"},
		{"residual_restore", "runtime.hydration.residual"},
		{"residual_restore_before", "runtime.hydration.residual_before"},
		{"residual_restore_after", "runtime.hydration.residual_after"},
	} {
		object := rawTimingObject(phases[metric.key])
		phaseID := add(parent, metric.name, rawTimingValue(object["seconds"]), nil)
		if phaseID == "" {
			phaseID = parent
		}
		for _, detail := range []struct{ key, suffix string }{
			{"io_wait_seconds", ".io_wait"},
			{"checksum_seconds", ".checksum"},
			{"cuda_enqueue_seconds", ".cuda_enqueue"},
			{"cuda_copy_seconds", ".cuda_copy"},
			{"initialization_seconds", ".initialize"},
			{"cuda_synchronize_seconds", ".cuda_sync"},
		} {
			add(phaseID, metric.name+detail.suffix, rawTimingValue(object[detail.key]), nil)
		}
		// Native io_service_seconds is the sum of each read's service time.
		// Queue-depth concurrency means it may exceed the phase's elapsed wall
		// time, so model it as an explicitly non-additive sibling rather than
		// an impossible child span.
		addCumulativeService(
			metric.name+".io_service_total",
			rawTimingValue(object["io_service_seconds"]),
			metric.name,
		)
	}
	recovery := rawTimingObject(phases["recovery_loader"])
	recoveryAttributes := make(map[string]string)
	if backend := rawTimingString(recovery["backend"]); backend != "" {
		recoveryAttributes["backend"] = backend
	}
	replay := rawTimingObject(recovery["adapter_replay"])
	replayTotal := rawTimingValue(replay["total_seconds"])
	recoveryTotal := normalizedRecoveryTimingTotal(phases)
	recoveryID := add(parent, "runtime.recovery_loader", recoveryTotal, recoveryAttributes)
	if recoveryID == "" {
		recoveryID = parent
	}
	for _, metric := range []struct{ key, name string }{
		{"io_seconds", "runtime.recovery_loader.io"},
		{"collective_seconds", "runtime.recovery_loader.collective"},
		{"verification_seconds", "runtime.recovery_loader.verification"},
	} {
		add(recoveryID, metric.name, rawTimingValue(recovery[metric.key]), nil)
	}
	replayID := add(recoveryID, "runtime.recovery_loader.adapter_replay", replayTotal, nil)
	if replayID == "" {
		replayID = recoveryID
	}
	for _, metric := range []struct{ key, name string }{
		{"staging_seconds", "runtime.adapter_replay.staging"},
		{"status_seconds", "runtime.adapter_replay.status"},
		{"io_seconds", "runtime.adapter_replay.io"},
		{"io_wait_seconds", "runtime.adapter_replay.io_wait"},
		{"io_overlap_seconds", "runtime.adapter_replay.io_overlap"},
		{"broadcast_seconds", "runtime.adapter_replay.broadcast"},
		{"copy_seconds", "runtime.adapter_replay.copy"},
		{"replay_sync_seconds", "runtime.adapter_replay.sync"},
		{"finalizer_seconds", "runtime.adapter_replay.finalizer"},
	} {
		add(replayID, metric.name, rawTimingValue(replay[metric.key]), nil)
	}
}

func normalizedRecoveryTimingTotal(phases map[string]json.RawMessage) float64 {
	recovery := rawTimingObject(phases["recovery_loader"])
	replay := rawTimingObject(recovery["adapter_replay"])
	replayTotal := rawTimingValue(replay["total_seconds"])
	total := rawTimingValue(recovery["total_seconds"])
	return math.Max(total, replayTotal)
}

func rawTimingObject(value json.RawMessage) map[string]json.RawMessage {
	if len(value) == 0 {
		return nil
	}
	var result map[string]json.RawMessage
	if json.Unmarshal(value, &result) != nil {
		return nil
	}
	return result
}

func rawTimingValue(value json.RawMessage) float64 {
	if len(value) == 0 {
		return 0
	}
	var result float64
	if json.Unmarshal(value, &result) != nil || !finiteTimingValue(result) || result <= 0 {
		return 0
	}
	return result
}

func rawTimingString(value json.RawMessage) string {
	if len(value) == 0 {
		return ""
	}
	var result string
	if json.Unmarshal(value, &result) != nil {
		return ""
	}
	return result
}

func markerOffset(markers map[string]runtimeTimingMark, name string) float64 {
	if mark, ok := markers[name]; ok {
		return mark.SinceRestoreBegin
	}
	return 0
}

func positiveTimingValue(value *float64) float64 {
	if value == nil || !finiteTimingValue(*value) || *value <= 0 {
		return 0
	}
	return *value
}

func finiteTimingValue(value float64) bool {
	return value >= 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

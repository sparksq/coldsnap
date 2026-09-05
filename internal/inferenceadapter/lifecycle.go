// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/sparksq/coldsnap/internal/capsule"
	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/snapshot"
	activationruntime "github.com/sparksq/coldsnap/runtime/engine"
)

const lifecycleMarkerPath = "/run/coldsnap/lifecycle.json"

const (
	lifecycleMarkerWriteAttempts = 5
	lifecycleMarkerRetryDelay    = 100 * time.Millisecond
)

type lifecycleContainerInspect = hostops.WorkloadInfo

type lifecycleControlResult struct {
	Action   string  `json:"action"`
	Sleeping bool    `json:"is_sleeping"`
	Accepted bool    `json:"accepted"`
	Seconds  float64 `json:"seconds"`
}

type LifecycleWorkerEvidence struct {
	WorkerID   string `json:"worker_id"`
	State      string `json:"state"`
	Capture    string `json:"capture_id"`
	PID        int    `json:"pid"`
	Generation string `json:"generation"`
}

type LifecycleUnitReport struct {
	Unit        string                    `json:"unit"`
	Host        string                    `json:"host"`
	Container   string                    `json:"container"`
	ContainerID string                    `json:"container_id"`
	Workers     []LifecycleWorkerEvidence `json:"workers"`
}

type LifecycleReport struct {
	Format      int                   `json:"format"`
	Kind        string                `json:"kind"`
	Engine      string                `json:"engine"`
	OperationID string                `json:"operation_id"`
	Operation   string                `json:"operation"`
	State       string                `json:"state"`
	ClusterID   string                `json:"cluster_id"`
	CaptureID   string                `json:"capture_id"`
	Driver      string                `json:"snapshot_driver"`
	Sleeping    bool                  `json:"is_sleeping"`
	Accepted    bool                  `json:"accepted"`
	Seconds     float64               `json:"seconds"`
	UpdatedUnix float64               `json:"updated_unix"`
	Error       string                `json:"error,omitempty"`
	Units       []LifecycleUnitReport `json:"units"`
}

func lifecycleActivationState(request snapshot.Request) string {
	if request.Lifecycle.ActivationState == "warm" {
		return "warm"
	}
	return "running"
}

func (adapter Adapter) runLifecycle(ctx context.Context, request snapshot.Request) error {
	var artifact snapshot.Artifact
	if err := operationtiming.Measure(ctx, "artifact.read_verify", nil, func(_ context.Context) error {
		artifactPath, err := artifactInputPath(request.Artifact)
		if err != nil {
			return err
		}
		artifact, err = snapshot.ReadArtifact(artifactPath)
		if err != nil {
			return err
		}
		if err := validateArtifactDriver(request, artifact); err != nil {
			return err
		}
		return compatibleLaunch(artifact.Launch, request.Launch)
	}); err != nil {
		return err
	}
	containers, err := operationtiming.MeasureValue(ctx, "lifecycle.containers", nil, func(phase context.Context) ([]lifecycleContainerInspect, error) {
		return adapter.inspectLifecycleContainers(phase, request, artifact)
	})
	if err != nil {
		return err
	}
	control, err := operationtiming.MeasureValue(ctx, "lifecycle.backend", map[string]string{
		"action": request.Operation,
	}, func(phase context.Context) (lifecycleControlResult, error) {
		return adapter.controlLifecycleBackend(phase, request, request.Operation)
	})
	if err != nil {
		adapter.writeFailedLifecycleState(ctx, request, artifact, err)
		return err
	}
	units, err := operationtiming.MeasureValue(ctx, "lifecycle.evidence", nil, func(phase context.Context) ([]LifecycleUnitReport, error) {
		return adapter.collectLifecycleEvidence(phase, request, artifact, containers)
	})
	if err != nil {
		adapter.writeFailedLifecycleState(ctx, request, artifact, err)
		return err
	}
	state, err := lifecycleState(request.Operation, control.Sleeping, units)
	if err != nil {
		adapter.writeFailedLifecycleState(ctx, request, artifact, err)
		return err
	}
	if request.Operation == "wake" && adapter.engine() == "vllm" && request.Policy.Process.AsyncGraphs {
		if err := operationtiming.Measure(ctx, "lifecycle.graphs_arm", nil, func(phase context.Context) error {
			return adapter.armLifecycleGraphs(phase, request)
		}); err != nil {
			adapter.writeFailedLifecycleState(ctx, request, artifact, err)
			return err
		}
	}
	report := LifecycleReport{
		Format: 1, Kind: "coldsnap-inference-lifecycle", Engine: adapter.engine(), OperationID: request.ID,
		Operation: request.Operation, State: state, ClusterID: request.Workload.ClusterID,
		CaptureID: artifact.CaptureID, Driver: artifact.Driver.ID,
		Sleeping: control.Sleeping, Accepted: control.Accepted, Seconds: control.Seconds,
		UpdatedUnix: float64(time.Now().UnixNano()) / 1e9, Units: units,
	}
	if request.Operation != "status" {
		if err := operationtiming.Measure(ctx, "lifecycle.persist", nil, func(phase context.Context) error {
			return adapter.writeLifecycleMarkers(phase, request, report)
		}); err != nil {
			return err
		}
	} else if markerState, markerErr := adapter.lifecycleMarkerState(ctx, request, artifact); markerErr == nil {
		if control.Sleeping && markerState == "warm" {
			report.State = "warm"
		} else if markerState == "failed" {
			report.State = "failed"
		}
	}
	return json.NewEncoder(adapter.Output).Encode(report)
}

func lifecycleState(operation string, sleeping bool, units []LifecycleUnitReport) (string, error) {
	expected := "running"
	if sleeping {
		expected = "sleeping"
	}
	for _, unit := range units {
		for _, worker := range unit.Workers {
			if worker.State != expected {
				return "", fmt.Errorf("worker %s lifecycle state is %s, expected %s", worker.WorkerID, worker.State, expected)
			}
		}
	}
	if operation == "sleep" && !sleeping {
		return "", errors.New("backend remained awake after sleep")
	}
	if operation == "wake" && sleeping {
		return "", errors.New("backend remained sleeping after wake")
	}
	return expected, nil
}

func (adapter Adapter) inspectLifecycleContainers(
	ctx context.Context, request snapshot.Request, artifact snapshot.Artifact,
) ([]lifecycleContainerInspect, error) {
	containers := make([]lifecycleContainerInspect, len(request.Launch.Units))
	err := parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		name := restoreContainerName(request, unit.Index)
		value, err := adapter.inspectWorkload(ctx, unit.Host, name)
		if err != nil {
			return fmt.Errorf("inspect lifecycle unit %s: %w", unit.ID, err)
		}
		expected := map[string]string{
			"io.sparksq.coldsnap.managed":  "true",
			"io.sparksq.coldsnap.rank":     strconv.Itoa(unit.Index),
			"io.sparksq.coldsnap.unit":     unit.ID,
			"io.sparksq.coldsnap.capture":  artifact.CaptureID,
			"io.sparksq.coldsnap.driver":   artifact.Driver.ID,
			"io.sparksq.coldsnap.workload": request.Workload.ClusterID,
		}
		if value.ID == "" || !value.Running || value.Paused {
			return fmt.Errorf("lifecycle unit %s container is not running and unpaused", unit.ID)
		}
		for key, wanted := range expected {
			if value.Labels[key] != wanted {
				return fmt.Errorf("lifecycle unit %s container label %s does not match", unit.ID, key)
			}
		}
		containers[unit.Index] = value
		return nil
	})
	return containers, err
}

func (adapter Adapter) controlLifecycleBackend(
	ctx context.Context, request snapshot.Request, action string,
) (lifecycleControlResult, error) {
	head := request.Launch.Units[0]
	model := request.Workload.ServedModelName
	if model == "" {
		model = request.Launch.Model.ID
	}
	arguments := []string{
		"python3",
		activationruntime.LifecycleTarget, "control", adapter.engine(), action, strconv.Itoa(httpPort(head.Command)), model,
		request.Validation.HealthPath, request.Validation.Prompt, request.Validation.Expected,
		commandLoadFormat(head.Command),
		strconv.Itoa(max(1, int(adapter.Timeout.Seconds()))),
	}
	output, err := adapter.execWorkload(ctx, head.Host, restoreContainerName(request, 0), arguments...)
	if err != nil {
		return lifecycleControlResult{}, fmt.Errorf("%s %s lifecycle backend: %w", action, adapter.engine(), err)
	}
	var result lifecycleControlResult
	if err := json.Unmarshal(output, &result); err != nil {
		return lifecycleControlResult{}, fmt.Errorf("decode %s lifecycle backend result: %w", action, err)
	}
	if result.Action != action || result.Seconds < 0 {
		return lifecycleControlResult{}, fmt.Errorf("%s lifecycle backend result is invalid", action)
	}
	return result, nil
}

func (adapter Adapter) collectLifecycleEvidence(
	ctx context.Context,
	request snapshot.Request,
	artifact snapshot.Artifact,
	containers []lifecycleContainerInspect,
) ([]LifecycleUnitReport, error) {
	result := make([]LifecycleUnitReport, len(request.Launch.Units))
	err := parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		workers := request.Launch.Execution.UnitWorkers(unit.ID)
		workerIDs := make([]string, len(workers))
		for index, worker := range workers {
			workerIDs[index] = worker.ID
		}
		expected, err := json.Marshal(workerIDs)
		if err != nil {
			return err
		}
		name := restoreContainerName(request, unit.Index)
		output, err := adapter.execWorkload(
			ctx, unit.Host, name, "python3",
			activationruntime.LifecycleTarget, "evidence", artifact.CaptureID, string(expected),
		)
		if err != nil {
			return fmt.Errorf("read lifecycle evidence for unit %s: %w", unit.ID, err)
		}
		var evidence []LifecycleWorkerEvidence
		if err := json.Unmarshal(output, &evidence); err != nil {
			return fmt.Errorf("decode lifecycle evidence for unit %s: %w", unit.ID, err)
		}
		if len(evidence) != len(workerIDs) {
			return fmt.Errorf("unit %s lifecycle evidence is incomplete", unit.ID)
		}
		result[unit.Index] = LifecycleUnitReport{
			Unit: unit.ID, Host: unit.Host, Container: name,
			ContainerID: containers[unit.Index].ID, Workers: evidence,
		}
		return nil
	})
	return result, err
}

// synchronizeRestoreLifecycleEvidence aligns an n580 template's captured
// sleep evidence with the validated post-restore service. Unlike n610, n580
// recovery boots CUDA and weights normally rather than calling /wake_up, so
// the backend has no lifecycle transition in which to rewrite these files.
func (adapter Adapter) synchronizeRestoreLifecycleEvidence(
	ctx context.Context, request snapshot.Request, artifact snapshot.Artifact, containers []string, state string,
) error {
	evidenceState := "running"
	if state == "warm" {
		evidenceState = "sleeping"
	}
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		workers := request.Launch.Execution.UnitWorkers(unit.ID)
		workerIDs := make([]string, len(workers))
		for index, worker := range workers {
			workerIDs[index] = worker.ID
		}
		expected, err := json.Marshal(workerIDs)
		if err != nil {
			return err
		}
		if _, err := adapter.execWorkload(
			ctx, unit.Host, containers[unit.Index], "python3",
			activationruntime.LifecycleTarget, "synchronize-evidence", evidenceState, artifact.CaptureID, string(expected),
		); err != nil {
			return fmt.Errorf("synchronize restored lifecycle evidence for unit %s: %w", unit.ID, err)
		}
		return nil
	})
}

func (adapter Adapter) armLifecycleGraphs(ctx context.Context, request snapshot.Request) error {
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		if _, err := adapter.execWorkload(
			ctx, unit.Host, restoreContainerName(request, unit.Index),
			"touch", capsule.Root+"/async-graphs-arm",
		); err != nil {
			return fmt.Errorf("arm asynchronous graphs for unit %s: %w", unit.ID, err)
		}
		return nil
	})
}

func (adapter Adapter) writeRestoreLifecycleState(
	ctx context.Context, request snapshot.Request, artifact snapshot.Artifact,
	containerNames []string, state string,
) error {
	units := make([]LifecycleUnitReport, len(request.Launch.Units))
	for _, unit := range request.Launch.Units {
		units[unit.Index] = LifecycleUnitReport{
			Unit: unit.ID, Host: unit.Host, Container: containerNames[unit.Index],
		}
	}
	report := LifecycleReport{
		Format: 1, Kind: "coldsnap-inference-lifecycle", Engine: adapter.engine(), OperationID: request.ID,
		Operation: "restore", State: state, ClusterID: request.Workload.ClusterID,
		CaptureID: artifact.CaptureID, Driver: artifact.Driver.ID,
		Sleeping: state == "warm", UpdatedUnix: float64(time.Now().UnixNano()) / 1e9,
		Units: units,
	}
	return adapter.writeLifecycleMarkers(ctx, request, report)
}

func (adapter Adapter) writeLifecycleMarkers(
	ctx context.Context, request snapshot.Request, report LifecycleReport,
) error {
	payload, err := json.Marshal(report)
	if err != nil {
		return err
	}
	payload = append(payload, '\n')
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		var lastErr error
		for attempt := 1; attempt <= lifecycleMarkerWriteAttempts; attempt++ {
			if _, err := adapter.execWorkloadInput(
				ctx, unit.Host, payload,
				restoreContainerName(request, unit.Index), "python3",
				activationruntime.LifecycleTarget, "write-marker",
			); err == nil {
				return nil
			} else {
				lastErr = err
			}
			if attempt < lifecycleMarkerWriteAttempts {
				if err := waitContext(ctx, lifecycleMarkerRetryDelay); err != nil {
					return fmt.Errorf("write lifecycle marker for unit %s: %w", unit.ID, err)
				}
			}
		}
		return fmt.Errorf(
			"write lifecycle marker for unit %s after %d attempts: %w",
			unit.ID, lifecycleMarkerWriteAttempts, lastErr,
		)
	})
}

func (adapter Adapter) lifecycleMarkerState(
	ctx context.Context, request snapshot.Request, artifact snapshot.Artifact,
) (string, error) {
	states := make([]string, len(request.Launch.Units))
	err := parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		output, err := adapter.execWorkload(
			ctx, unit.Host, restoreContainerName(request, unit.Index),
			"cat", lifecycleMarkerPath,
		)
		if err != nil {
			return err
		}
		var marker LifecycleReport
		if err := json.Unmarshal(output, &marker); err != nil {
			return err
		}
		if marker.Kind != "coldsnap-inference-lifecycle" || marker.Engine != adapter.engine() || marker.ClusterID != request.Workload.ClusterID ||
			marker.CaptureID != artifact.CaptureID || marker.Driver != artifact.Driver.ID {
			return errors.New("lifecycle marker identity differs")
		}
		states[unit.Index] = marker.State
		return nil
	})
	if err != nil {
		return "", err
	}
	for _, state := range states[1:] {
		if state != states[0] {
			return "", errors.New("lifecycle markers disagree across units")
		}
	}
	return states[0], nil
}

func (adapter Adapter) writeFailedLifecycleState(
	ctx context.Context, request snapshot.Request, artifact snapshot.Artifact,
	cause error,
) {
	markerCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
	defer cancel()
	report := LifecycleReport{
		Format: 1, Kind: "coldsnap-inference-lifecycle", Engine: adapter.engine(), OperationID: request.ID,
		Operation: request.Operation, State: "failed", ClusterID: request.Workload.ClusterID,
		CaptureID: artifact.CaptureID, Driver: artifact.Driver.ID,
		UpdatedUnix: float64(time.Now().UnixNano()) / 1e9, Error: cause.Error(),
	}
	_ = adapter.writeLifecycleMarkers(markerCtx, request, report)
}

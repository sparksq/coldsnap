// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/snapshot"
	activationruntime "github.com/sparksq/coldsnap/runtime/engine"
)

type lifecycleRemote struct {
	mutex              sync.Mutex
	sleeping           bool
	markers            map[string][]byte
	markerFailuresLeft int
	markerCalls        int
	restoreSyncs       []string
}

func (remote *lifecycleRemote) Run(
	_ context.Context, host string, arguments ...string,
) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	if len(arguments) >= 5 && arguments[0] == "docker" && arguments[1] == "inspect" {
		rank, err := strconv.Atoi(host[len(host)-1:])
		if err != nil {
			return nil, err
		}
		value := lifecycleContainerInspect{
			ID: "container-" + strconv.Itoa(rank), Running: true,
			Labels: map[string]string{
				"io.sparksq.coldsnap.managed":  "true",
				"io.sparksq.coldsnap.rank":     strconv.Itoa(rank),
				"io.sparksq.coldsnap.unit":     "unit-" + strconv.Itoa(rank),
				"io.sparksq.coldsnap.capture":  "generic-capture",
				"io.sparksq.coldsnap.driver":   "n610",
				"io.sparksq.coldsnap.workload": "sparkrun_deadbeef_01234567",
			},
		}
		return json.Marshal(value)
	}
	if len(arguments) >= 8 && arguments[0] == "docker" && arguments[1] == "exec" &&
		arguments[4] == activationruntime.LifecycleTarget && arguments[5] == "control" {
		action := arguments[7]
		switch action {
		case "sleep":
			remote.sleeping = true
		case "wake":
			remote.sleeping = false
		case "status":
		default:
			return nil, errors.New("unexpected lifecycle action")
		}
		return json.Marshal(lifecycleControlResult{
			Action: action, Sleeping: remote.sleeping, Accepted: action != "status", Seconds: 1,
		})
	}
	if len(arguments) >= 8 && arguments[0] == "docker" && arguments[1] == "exec" &&
		arguments[4] == activationruntime.LifecycleTarget && arguments[5] == "evidence" {
		var workers []string
		if err := json.Unmarshal([]byte(arguments[7]), &workers); err != nil {
			return nil, err
		}
		state := "running"
		if remote.sleeping {
			state = "sleeping"
		}
		evidence := make([]LifecycleWorkerEvidence, len(workers))
		for index, worker := range workers {
			evidence[index] = LifecycleWorkerEvidence{
				WorkerID: worker, State: state, Capture: arguments[6], PID: 100 + index, Generation: "generation",
			}
		}
		return json.Marshal(evidence)
	}
	if len(arguments) == 9 && arguments[0] == "docker" && arguments[1] == "exec" &&
		arguments[4] == activationruntime.LifecycleTarget && arguments[5] == "synchronize-evidence" {
		remote.restoreSyncs = append(remote.restoreSyncs, host+"\x00"+arguments[6]+"\x00"+arguments[7]+"\x00"+arguments[8])
		return nil, nil
	}
	if len(arguments) == 5 && slices.Equal(arguments[:2], []string{"docker", "exec"}) && arguments[3] == "cat" {
		marker := remote.markers[host]
		if len(marker) == 0 {
			return nil, errors.New("marker absent")
		}
		return marker, nil
	}
	if slices.Contains(arguments, "/opt/coldsnap/capsule/async-graphs-arm") {
		return nil, nil
	}
	return nil, fmt.Errorf("unexpected remote command: %#v", arguments)
}

func (*lifecycleRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (remote *lifecycleRemote) RunInput(
	_ context.Context, host string, input []byte, arguments ...string,
) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	if len(arguments) != 7 || arguments[0] != "docker" || arguments[1] != "exec" ||
		arguments[5] != activationruntime.LifecycleTarget || arguments[6] != "write-marker" {
		return nil, fmt.Errorf("unexpected lifecycle marker command: %#v", arguments)
	}
	remote.markerCalls++
	if remote.markerFailuresLeft > 0 {
		remote.markerFailuresLeft--
		return nil, errors.New("transient docker exec failure")
	}
	if remote.markers == nil {
		remote.markers = make(map[string][]byte)
	}
	remote.markers[host] = slices.Clone(input)
	return nil, nil
}

func TestLifecycleMarkerWriteRetriesTransientDockerExecFailure(t *testing.T) {
	request := lifecycleRequest()
	remote := &lifecycleRemote{markerFailuresLeft: 2}
	adapter := fixtureAdapter(Adapter{Remote: remote, Timeout: time.Minute})
	report := LifecycleReport{
		Format: 1, Kind: "coldsnap-inference-lifecycle", Engine: "vllm", OperationID: request.ID,
		Operation: "restore", State: "running", ClusterID: request.Workload.ClusterID,
		CaptureID: "generic-capture", Driver: "n610",
	}
	if err := adapter.writeLifecycleMarkers(context.Background(), request, report); err != nil {
		t.Fatal(err)
	}
	if remote.markerCalls < len(request.Launch.Units)+2 {
		t.Fatalf("marker calls = %d, want at least %d", remote.markerCalls, len(request.Launch.Units)+2)
	}
	if len(remote.markers) != len(request.Launch.Units) {
		t.Fatalf("markers = %#v", remote.markers)
	}
}

func (*lifecycleRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func lifecycleRequest() snapshot.Request {
	request := validRequest(2)
	request.Operation = "sleep"
	request.Artifact = "/tmp/artifact.json"
	request.Output = ""
	request.Workload = snapshot.WorkloadIdentity{
		ClusterID: "sparkrun_deadbeef_01234567", IntentID: "deadbeef",
		Recipe: "qwen", Runtime: "vllm-distributed", Model: request.Launch.Model.ID,
		ServedModelName: "model",
	}
	return request
}

func TestLifecycleBackendAndEvidenceAreTopologyGeneric(t *testing.T) {
	request := lifecycleRequest()
	remote := &lifecycleRemote{}
	adapter := fixtureAdapter(Adapter{Remote: remote, Timeout: time.Minute})
	artifact := snapshot.Artifact{CaptureID: "generic-capture", Driver: validSnapshotDriverContract()}
	containers, err := adapter.inspectLifecycleContainers(context.Background(), request, artifact)
	if err != nil {
		t.Fatal(err)
	}
	control, err := adapter.controlLifecycleBackend(context.Background(), request, "sleep")
	if err != nil {
		t.Fatal(err)
	}
	units, err := adapter.collectLifecycleEvidence(context.Background(), request, artifact, containers)
	if err != nil {
		t.Fatal(err)
	}
	state, err := lifecycleState("sleep", control.Sleeping, units)
	if err != nil || state != "sleeping" {
		t.Fatalf("sleep state = %q, %v", state, err)
	}
	if len(units) != 2 || units[1].Workers[0].WorkerID != "worker-1" {
		t.Fatalf("lifecycle units = %#v", units)
	}

	report := LifecycleReport{
		Format: 1, Kind: "coldsnap-inference-lifecycle", Engine: "vllm", OperationID: request.ID,
		Operation: "sleep", State: state, ClusterID: request.Workload.ClusterID,
		CaptureID: artifact.CaptureID, Driver: artifact.Driver.ID, Sleeping: true,
	}
	if err := adapter.writeLifecycleMarkers(context.Background(), request, report); err != nil {
		t.Fatal(err)
	}
	markerState, err := adapter.lifecycleMarkerState(context.Background(), request, artifact)
	if err != nil || markerState != "sleeping" {
		t.Fatalf("marker state = %q, %v", markerState, err)
	}

	control, err = adapter.controlLifecycleBackend(context.Background(), request, "wake")
	if err != nil {
		t.Fatal(err)
	}
	units, err = adapter.collectLifecycleEvidence(context.Background(), request, artifact, containers)
	if err != nil {
		t.Fatal(err)
	}
	if state, err = lifecycleState("wake", control.Sleeping, units); err != nil || state != "running" {
		t.Fatalf("wake state = %q, %v", state, err)
	}
}

func TestLifecycleStateRejectsPartialDistributedTransition(t *testing.T) {
	units := []LifecycleUnitReport{
		{Workers: []LifecycleWorkerEvidence{{WorkerID: "worker-0", State: "sleeping"}}},
		{Workers: []LifecycleWorkerEvidence{{WorkerID: "worker-1", State: "running"}}},
	}
	if _, err := lifecycleState("sleep", true, units); err == nil {
		t.Fatal("partial distributed sleep was accepted")
	}
}

func TestN580RestoreSynchronizesCapturedSleepEvidence(t *testing.T) {
	request := lifecycleRequest()
	remote := &lifecycleRemote{}
	adapter := fixtureAdapter(Adapter{Remote: remote, Timeout: time.Minute})
	artifact := snapshot.Artifact{CaptureID: "generic-capture", Driver: validSnapshotDriverContract()}
	containers := []string{"restore-0", "restore-1"}
	if err := adapter.synchronizeRestoreLifecycleEvidence(
		context.Background(), request, artifact, containers, "running",
	); err != nil {
		t.Fatal(err)
	}
	slices.Sort(remote.restoreSyncs)
	want := []string{
		"node-0\x00running\x00generic-capture\x00[\"worker-0\"]",
		"node-1\x00running\x00generic-capture\x00[\"worker-1\"]",
	}
	if !slices.Equal(remote.restoreSyncs, want) {
		t.Fatalf("restore evidence syncs = %#v", remote.restoreSyncs)
	}
}

func TestRestoreLifecycleEvidenceUsesVersionedActivationHelper(t *testing.T) {
	request := lifecycleRequest()
	remote := &lifecycleRemote{}
	artifact := snapshot.Artifact{CaptureID: "portable-capture", Driver: validSnapshotDriverContract()}
	if err := (fixtureAdapter(Adapter{Remote: remote})).synchronizeRestoreLifecycleEvidence(
		context.Background(), request, artifact, []string{"restore-0", "restore-1"}, "running",
	); err != nil {
		t.Fatal(err)
	}
	for _, invocation := range remote.restoreSyncs {
		if !strings.Contains(invocation, "\x00portable-capture\x00") {
			t.Fatalf("restore lifecycle helper did not receive portable capture identity: %q", invocation)
		}
	}
}

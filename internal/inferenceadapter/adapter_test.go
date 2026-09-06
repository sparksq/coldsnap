// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/sparksq/coldsnap/internal/hostops"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/capsule"
	"github.com/sparksq/coldsnap/internal/ncclprovider"
	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/payloadvalidation"
	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

const testDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
const publishedTestDigest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

func validSnapshotDriverContract() snapshotdriver.Contract {
	contract, err := snapshotdriver.Lookup(snapshotdriver.N610)
	if err != nil {
		panic(err)
	}
	return contract
}

func validGraphPolicy(launch snapshot.LaunchSpec) snapshot.GraphPolicyRecord {
	return validGraphPolicyForDriver(launch, snapshotdriver.N610)
}

func validGraphPolicyForDriver(
	launch snapshot.LaunchSpec, driverID string,
) snapshot.GraphPolicyRecord {
	policy, err := snapshot.DefaultSnapshotPolicyForDriver(driverID)
	if err != nil {
		panic(err)
	}
	record, err := snapshot.NewGraphPolicyRecord(snapshot.Request{
		Launch: launch, Policy: policy,
	}, nil)
	if err != nil {
		panic(err)
	}
	return record
}

type fakeRemote struct {
	presentInfiniband bool
	driver            string
}

type localCommandRemote struct{}

func (localCommandRemote) Run(ctx context.Context, _ string, arguments ...string) ([]byte, error) {
	if len(arguments) >= 2 && arguments[1] == "payload-verify" {
		var stdout, stderr bytes.Buffer
		if err := payloadvalidation.Execute(arguments[2:], &stdout, &stderr); err != nil {
			return nil, fmt.Errorf("%w: %s", err, strings.TrimSpace(stderr.String()))
		}
		return stdout.Bytes(), nil
	}
	command := exec.CommandContext(ctx, arguments[0], arguments[1:]...)
	output, err := command.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%w: %s", err, strings.TrimSpace(string(output)))
	}
	return output, nil
}

func (localCommandRemote) RunCombined(ctx context.Context, host string, arguments ...string) ([]byte, error) {
	return localCommandRemote{}.Run(ctx, host, arguments...)
}

func (localCommandRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, errors.New("unexpected input command")
}

func (localCommandRemote) Upload(context.Context, string, []string, string, bool) error {
	return errors.New("unexpected upload")
}

func TestRemoteModelPayloadObjectCreatesStatBoundAdmission(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, modelPayloadName)
	if err := os.WriteFile(path, []byte("payload"), 0o600); err != nil {
		t.Fatal(err)
	}
	adapter := fixtureAdapter(Adapter{Remote: localCommandRemote{}})
	unit := snapshot.LaunchUnit{ID: "unit-0", Host: "local"}
	object, err := adapter.remoteModelPayloadObject(
		context.Background(), unit, snapshot.WorkerOwner("worker-0"), path,
		"pending-content-address", 0, "",
	)
	if err != nil {
		t.Fatal(err)
	}
	markerPath := path + ".coldsnap-validation.json"
	markerInfo, err := os.Stat(markerPath)
	if err != nil {
		t.Fatal(err)
	}
	if markerInfo.Mode().Perm() != 0o600 {
		t.Fatalf("marker mode = %o", markerInfo.Mode().Perm())
	}
	markerPayload, err := os.ReadFile(markerPath)
	if err != nil {
		t.Fatal(err)
	}
	var marker map[string]any
	if err := json.Unmarshal(markerPayload, &marker); err != nil {
		t.Fatal(err)
	}
	expected, _ := marker["expected"].(map[string]any)
	if marker["kind"] != "coldsnap-payload-validation" || marker["provider"] != "sha256-cache-v1" ||
		expected["sha256"] != object.SHA256 {
		t.Fatalf("marker = %#v, object = %#v", marker, object)
	}
	reused, err := adapter.remoteModelPayloadObject(
		context.Background(), unit, object.Owner, path, object.Path, object.Bytes, object.SHA256,
	)
	if err != nil || reused != object {
		t.Fatalf("reused object = %#v, error = %v", reused, err)
	}
	if err := os.WriteFile(path, []byte("changed"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := adapter.remoteModelPayloadObject(
		context.Background(), unit, object.Owner, path, object.Path, object.Bytes, object.SHA256,
	); err == nil {
		t.Fatal("changed model payload was admitted by stale marker")
	}
}

type ncclProviderRemote struct {
	providerByHost map[string]string
}

type capturedNCCLRemote struct {
	reports map[string][]map[string]any
}

func (remote capturedNCCLRemote) Run(
	_ context.Context, host string, arguments ...string,
) ([]byte, error) {
	if len(arguments) != 2 || arguments[0] != "cat" || !strings.HasSuffix(arguments[1], "/capture.json") {
		return nil, fmt.Errorf("unexpected remote command: %v", arguments)
	}
	return json.Marshal(map[string]any{
		"nccl_workers_before_sleep": map[string]any{"results": remote.reports[host]},
	})
}

func (capturedNCCLRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (capturedNCCLRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, nil
}

func (capturedNCCLRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func (remote ncclProviderRemote) Run(
	_ context.Context, host string, arguments ...string,
) ([]byte, error) {
	if !slices.Contains(arguments, "verify-active") {
		return nil, fmt.Errorf("unexpected remote command: %v", arguments)
	}
	record := ncclprovider.ActiveRecord{
		Format: ncclprovider.ActiveFormat, Kind: ncclprovider.ActiveKind,
		ProviderID: remote.providerByHost[host], ProviderRevision: 1,
		PlatformKey: "linux-aarch64-cuda13", ManifestSHA256: strings.Repeat("a", 64),
	}
	return json.Marshal(record)
}

func (ncclProviderRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (ncclProviderRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, nil
}

func (ncclProviderRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

type packIdentityRemote struct {
	identity snapshot.PreparedPayloadValidation
	calls    [][]string
}

type workloadLogRemote struct {
	mutex sync.Mutex
	calls [][]string
}

type materializationRemote struct {
	mutex  sync.Mutex
	inputs map[string][]byte
}

type tcpPortProbeRemote struct {
	mutex         sync.Mutex
	rejectedShift uint16
	fatalShift    uint16
	shifts        []uint16
	preserved     []bool
	imagesDirs    []string
	portMaps      []string
	allowEmpty    []bool
}

func (remote *tcpPortProbeRemote) Run(
	_ context.Context, _ string, arguments ...string,
) ([]byte, error) {
	index := slices.Index(arguments, "tcp-probe")
	if index < 0 {
		return nil, fmt.Errorf("unexpected remote command: %v", arguments)
	}
	for _, expected := range []string{
		"--pull", "never", "--network", "host",
	} {
		if !slices.Contains(arguments, expected) {
			return nil, fmt.Errorf("TCP probe command lacks %q: %v", expected, arguments)
		}
	}
	imagesIndex := slices.Index(arguments, "--images-dir")
	if imagesIndex < 0 || imagesIndex+1 >= len(arguments) {
		return nil, errors.New("TCP probe command lacks an images directory")
	}
	shiftIndex := slices.Index(arguments, "--tcp-port-shift")
	if shiftIndex < 0 || shiftIndex+1 >= len(arguments) {
		return nil, errors.New("TCP probe command lacks a port shift")
	}
	parsed, err := strconv.ParseUint(arguments[shiftIndex+1], 10, 16)
	if err != nil {
		return nil, err
	}
	shift := uint16(parsed)
	remote.mutex.Lock()
	remote.shifts = append(remote.shifts, shift)
	remote.preserved = append(remote.preserved, slices.Contains(arguments, "--tcp-preserve-port"))
	remote.imagesDirs = append(remote.imagesDirs, arguments[imagesIndex+1])
	portMap := ""
	if index := slices.Index(arguments, "--tcp-port-map"); index >= 0 && index+1 < len(arguments) {
		portMap = arguments[index+1]
	}
	remote.portMaps = append(remote.portMaps, portMap)
	remote.allowEmpty = append(remote.allowEmpty, slices.Contains(arguments, "--tcp-allow-empty-map"))
	remote.mutex.Unlock()
	if shift == remote.fatalShift {
		return nil, errors.New("probe helper is incompatible")
	}
	if shift == remote.rejectedShift {
		return json.Marshal(map[string]any{
			"format": 1, "kind": "coldsnap-criu-tcp-port-probe",
			"port_shift": shift, "endpoints": 4, "available": false,
			"collision": "address already in use",
		})
	}
	return json.Marshal(map[string]any{
		"format": 1, "kind": "coldsnap-criu-tcp-port-probe",
		"port_shift": shift, "endpoints": 4, "available": true,
	})
}

func (*tcpPortProbeRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (*tcpPortProbeRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, nil
}

func (*tcpPortProbeRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func (remote *materializationRemote) Run(
	_ context.Context, _ string, arguments ...string,
) ([]byte, error) {
	if slices.Equal(arguments, []string{"stat", "-c", "%u:%g", "/cache/coldsnap/model-payloads"}) {
		return []byte("1000:1000\n"), nil
	}
	return nil, nil
}

func (*materializationRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (remote *materializationRemote) RunInput(
	_ context.Context, host string, input []byte, arguments ...string,
) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	if len(arguments) != 2 || arguments[0] != "tee" {
		return nil, fmt.Errorf("unexpected input command: %v", arguments)
	}
	if remote.inputs == nil {
		remote.inputs = make(map[string][]byte)
	}
	remote.inputs[host] = slices.Clone(input)
	return nil, nil
}

func (*materializationRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func (remote *workloadLogRemote) Run(
	_ context.Context, host string, arguments ...string,
) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	remote.calls = append(remote.calls, append([]string{host}, arguments...))
	if slices.Contains(arguments, "readlink") {
		return []byte(containerTargetLogPath + "\n"), nil
	}
	return []byte("ok\n"), nil
}

func (*workloadLogRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (*workloadLogRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, nil
}

func (*workloadLogRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func (remote *packIdentityRemote) Run(
	_ context.Context, _ string, arguments ...string,
) ([]byte, error) {
	remote.calls = append(remote.calls, slices.Clone(arguments))
	if len(arguments) >= 2 && arguments[1] == "payload-verify" {
		return []byte(fmt.Sprintf(
			`{"format":1,"kind":"coldsnap-payload-validation-result","decision":"accept","bytes":%d,"sha256":%q,"validation":{"record":%q,"provider":%q,"content_evidence":%q,"reason":%q,"device":%d,"inode":%d,"size":%d,"mtime_ns":%d,"ctime_ns":%d}}`,
			remote.identity.Size, testDigest, remote.identity.Record, remote.identity.Provider,
			remote.identity.ContentEvidence, remote.identity.Reason,
			remote.identity.Device, remote.identity.Inode, remote.identity.Size,
			remote.identity.MTimeNS, remote.identity.CTimeNS,
		)), nil
	}
	return nil, errors.New("unexpected remote command")
}

func (*packIdentityRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (*packIdentityRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, nil
}

func (*packIdentityRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func (remote fakeRemote) Run(
	_ context.Context, _ string, arguments ...string,
) ([]byte, error) {
	if slices.Equal(arguments, []string{"test", "-e", "/dev/infiniband"}) && !remote.presentInfiniband {
		return nil, errors.New("absent")
	}
	if slices.Equal(arguments, []string{"uname", "-m"}) {
		return []byte("aarch64\n"), nil
	}
	if slices.Equal(arguments, []string{"uname", "-r"}) {
		return []byte("6.17.0-1029-nvidia\n"), nil
	}
	if len(arguments) > 0 && arguments[0] == "nvidia-smi" {
		driver := remote.driver
		if driver == "" {
			driver = "611.1.0"
		}
		return []byte(driver + ", NVIDIA GB10, 12.1\n"), nil
	}
	return []byte("ok\n"), nil
}

func (fakeRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (fakeRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, nil
}

func (fakeRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func validRequest(worldSize int) snapshot.Request {
	topology, err := snapshot.NewAdapterTopology("vllm.parallel:v1", map[string]any{
		"dimensions": map[string]int{"tensor": worldSize},
	})
	if err != nil {
		panic(err)
	}
	units := make([]snapshot.LaunchUnit, worldSize)
	execution := snapshot.ExecutionGraph{
		Adapter:  topology,
		Services: []snapshot.ServiceDomain{{ID: "model", Role: "vllm:serve"}},
		Groups:   []snapshot.ProcessGroup{{ID: "world", Kind: "torch:world", Service: "model"}},
	}
	for rank := range worldSize {
		unitID := "unit-" + strconv.Itoa(rank)
		workerID := "worker-" + strconv.Itoa(rank)
		units[rank] = snapshot.LaunchUnit{
			ID: unitID, Index: rank, Host: "node-" + strconv.Itoa(rank), Devices: []string{"0"},
			Image: "registry/vllm@" + testDigest, ImageDigest: testDigest,
			Command: []string{
				"bash", "--noprofile", "--norc", "-c",
				"exec vllm serve model --load-format coldsnap --enable-sleep-mode",
			},
		}
		execution.Workers = append(execution.Workers, snapshot.Worker{
			ID: workerID, Unit: unitID, Service: "model", ProcessSlot: 0, DeviceSlots: []int{0},
		})
		execution.Services[0].Workers = append(execution.Services[0].Workers, workerID)
		execution.Groups[0].Members = append(execution.Groups[0].Members, workerID)
	}
	return snapshot.Request{
		Format: snapshot.RequestFormat, Kind: snapshot.RequestKind, Operation: "capture",
		Driver: snapshotdriver.Selection{ID: snapshotdriver.N610},
		ID:     "generic-capture", Output: "/tmp/artifact.json",
		Launch: snapshot.LaunchSpec{
			Engine: "vllm", Model: snapshot.ModelSource{ID: "org/model", Revision: "commit"},
			Units: units, Execution: execution,
		},
		Policy: snapshot.SnapshotPolicy{
			Process: snapshot.ProcessPolicy{Backend: "cuda-criu", KVDiscard: true},
			Weights: snapshot.WeightPolicy{
				Mode: "recovery", Recovery: snapshot.RecoveryPolicy{
					Enabled: true, Source: "huggingface-safetensors", LoaderBackend: "direct",
				},
			},
			Compatibility: snapshot.CompatibilityPolicy{
				EnforceCapturedDriverFloor: true,
				Kernel:                     snapshot.KernelCompatibilityCapability,
			},
		},
		Validation: snapshot.ValidationPolicy{HealthPath: "/health", Prompt: "say exact", Expected: "exact"},
	}
}

func validPortableCompatibility(worldSize int) snapshot.ArtifactCompatibility {
	compatibility := snapshot.ArtifactCompatibility{Policy: snapshot.PortabilityPolicy}
	for rank := range worldSize {
		compatibility.Units = append(compatibility.Units, snapshot.UnitPlatformCompatibility{
			Unit: "unit-" + strconv.Itoa(rank), Architecture: "aarch64", Kernel: "6.17.0-1029-nvidia",
			NVIDIADriverCaptured: "610.43.02", NVIDIADriverMin: "610.43.02", CUDAUserspace: "13.0",
			Devices: []snapshot.DeviceCompatibility{{Slot: 0, GPUName: "NVIDIA GB10", ComputeCapability: "12.1"}},
		})
	}
	return compatibility
}

func capturedNCCLFixture(t *testing.T, request snapshot.Request) ([]snapshot.RuntimeProviderBinding, map[string][]map[string]any) {
	t.Helper()
	capabilities := []string{"communicator-unwrap", "synchronous-termination"}
	bindings := make([]snapshot.RuntimeProviderBinding, 0, len(request.Launch.Units))
	reports := make(map[string][]map[string]any, len(request.Launch.Units))
	for _, unit := range request.Launch.Units {
		active := ncclprovider.ActiveRecord{
			ProviderID: "nccl-test", ProviderRevision: 11,
			ProviderABI: ncclprovider.ABIVersion{Major: 1}, CheckpointABI: 100,
			ProviderNCCLRuntime: ncclprovider.ProviderNCCLRuntimeIdentity{Version: 23102},
			Capabilities:        slices.Clone(capabilities), Bridge: ncclprovider.ActiveBridge{ABI: 1},
		}
		binding, err := snapshot.NewRuntimeProviderBinding(
			snapshot.UnitOwner(unit.ID), ncclProviderKind, ncclProviderSchema, active,
		)
		if err != nil {
			t.Fatal(err)
		}
		bindings = append(bindings, binding)
		for _, worker := range request.Launch.Execution.UnitWorkers(unit.ID) {
			reports[unit.Host] = append(reports[unit.Host], map[string]any{
				"worker_id": worker.ID,
				"version": map[string]any{
					"checkpoint": 100, "nccl": 23102, "dlsym_bridge_abi": 1,
					"provider": map[string]any{
						"id": "nccl-test", "revision": 11, "abi_major": 1, "abi_minor": 0,
						"capabilities": capabilities, "compiled_nccl": 23102, "loaded_nccl": 23102,
						"capability_mask": 0, "unknown_capability_mask": 0,
					},
				},
			})
		}
	}
	return bindings, reports
}

func TestValidateCapturedNCCLWorkersAggregatesPerUnitEvidence(t *testing.T) {
	request := validRequest(2)
	bindings, reports := capturedNCCLFixture(t, request)
	adapter := fixtureAdapter(Adapter{Remote: capturedNCCLRemote{reports: reports}})
	if err := adapter.validateCapturedNCCLWorkers(
		context.Background(), request, []string{"/capture/unit-0", "/capture/unit-1"}, bindings,
	); err != nil {
		t.Fatal(err)
	}

	reports[request.Launch.Units[1].Host] = nil
	err := adapter.validateCapturedNCCLWorkers(
		context.Background(), request, []string{"/capture/unit-0", "/capture/unit-1"}, bindings,
	)
	if err == nil || !strings.Contains(err.Error(), "1 unique NCCL workers, want 2") {
		t.Fatalf("missing worker error = %v", err)
	}
}

func TestValidateCapturedNCCLWorkersAllowsIdenticalCollectiveReports(t *testing.T) {
	request := validRequest(2)
	bindings, reports := capturedNCCLFixture(t, request)
	global := append(slices.Clone(reports["node-0"]), reports["node-1"]...)
	reports["node-0"] = global
	reports["node-1"] = slices.Clone(global)
	adapter := fixtureAdapter(Adapter{Remote: capturedNCCLRemote{reports: reports}})
	if err := adapter.validateCapturedNCCLWorkers(
		context.Background(), request, []string{"/capture/unit-0", "/capture/unit-1"}, bindings,
	); err != nil {
		t.Fatal(err)
	}
}

func multiWorkerRequest(t *testing.T) snapshot.Request {
	t.Helper()
	request := validRequest(2)
	request.Launch.Units[0].Devices = []string{"0", "1"}
	request.Launch.Units[1].Devices = []string{"0", "1"}
	topology, err := snapshot.NewAdapterTopology("vllm.parallel:v1", map[string]any{
		"dimensions": map[string]int{"tensor": 4, "pipeline": 1, "data": 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	request.Launch.Execution = snapshot.ExecutionGraph{
		Adapter: topology,
		Services: []snapshot.ServiceDomain{{
			ID: "model", Role: "vllm:serve", Workers: []string{"worker-0", "worker-1", "worker-2", "worker-3"},
		}},
		Groups: []snapshot.ProcessGroup{{
			ID: "world", Kind: "torch:world", Service: "model", Members: []string{"worker-0", "worker-1", "worker-2", "worker-3"},
		}},
		Workers: []snapshot.Worker{
			{ID: "worker-0", Unit: "unit-0", Service: "model", ProcessSlot: 0, DeviceSlots: []int{0}},
			{ID: "worker-1", Unit: "unit-0", Service: "model", ProcessSlot: 1, DeviceSlots: []int{1}},
			{ID: "worker-2", Unit: "unit-1", Service: "model", ProcessSlot: 0, DeviceSlots: []int{0}},
			{ID: "worker-3", Unit: "unit-1", Service: "model", ProcessSlot: 1, DeviceSlots: []int{1}},
		},
	}
	return request
}

func TestUnitCommandSeparatesUnitAndWorkerWorlds(t *testing.T) {
	request := multiWorkerRequest(t)
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[1], request.Launch.Units[1].Image,
		"capture-unit-1", "capture", "activation", request.ID, "/state/unit-1", "/state/endpoint",
		"recovery", nil, request.Launch.Units[1].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{
		`"device=0,1"`, "COLDSNAP_WORLD_SIZE=4", "COLDSNAP_TP_SIZE=4",
		"COLDSNAP_EXPECTED_UNIT=unit-1",
		"COLDSNAP_HIBERNATE_STATE_DIR=/opt/coldsnap/capsule/hibernate-states",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("unit command lacks %q: %#v", expected, command)
		}
	}
	if slices.ContainsFunc(command, func(argument string) bool {
		return strings.HasPrefix(argument, "COLDSNAP_EXPECTED_RANK=") ||
			strings.HasPrefix(argument, "COLDSNAP_HIBERNATE_STATE_PATH=")
	}) {
		t.Fatalf("unit command retains one-worker environment: %#v", command)
	}
	worldOption := slices.Index(command, "--world-size")
	rankOption := slices.Index(command, "--rank")
	if worldOption < 0 || command[worldOption+1] != "2" || rankOption < 0 || command[rankOption+1] != "1" {
		t.Fatalf("controller unit topology is incorrect: %#v", command)
	}
	graphArgument := ""
	for _, argument := range command {
		if value, ok := strings.CutPrefix(argument, "COLDSNAP_EXECUTION_GRAPH="); ok {
			graphArgument = value
		}
	}
	var graph struct {
		Unit          string            `json:"unit"`
		ByProcessSlot map[string]string `json:"by_process_slot"`
		Groups        map[string]struct {
			Kind  string            `json:"kind"`
			Size  int               `json:"size"`
			Ranks map[string]string `json:"ranks"`
		} `json:"groups"`
	}
	if err := json.Unmarshal([]byte(graphArgument), &graph); err != nil {
		t.Fatalf("decode execution graph: %v; command=%#v", err, command)
	}
	if graph.Unit != "unit-1" || graph.ByProcessSlot["0"] != "worker-2" ||
		graph.ByProcessSlot["1"] != "worker-3" || graph.Groups["world"].Size != 4 ||
		graph.Groups["world"].Ranks["2"] != "worker-2" || graph.Groups["world"].Ranks["3"] != "worker-3" {
		t.Fatalf("unit worker map = %#v", graph)
	}
}

func TestConfiguredUsesQualifiedOperationTimeout(t *testing.T) {
	configured, err := (fixtureAdapter(Adapter{Remote: fakeRemote{}})).configured(validRequest(1))
	if err != nil {
		t.Fatal(err)
	}
	if configured.Timeout != 45*time.Minute {
		t.Fatalf("timeout = %s, want 45m", configured.Timeout)
	}
}

func TestRankCommandIsTopologyNeutralAndUsesQualifiedRecoveryDefaults(t *testing.T) {
	request := validRequest(4)
	request.Launch.Units[3].Environment = map[string]string{
		"NCCL_CHECKPOINT_TERMINATION": "abort",
		"NCCL_CUMEM_ENABLE":           "0",
	}
	adapter := fixtureAdapter(Adapter{
		Remote: fakeRemote{presentInfiniband: true}, Timeout: 20 * time.Minute,
	})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[3], request.Launch.Units[3].Image,
		"capture-rank-3", "capture", "activation", request.ID, "/state/rank3", "/state/endpoint",
		"recovery", nil, request.Launch.Units[3].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{
		"device=0", "COLDSNAP_WORLD_SIZE=4", "COLDSNAP_TP_SIZE=4",
		"COLDSNAP_CAPTURE_ID=generic-capture",
		"COLDSNAP_RECOVERY_LOADER_BACKEND=direct", "--world-size", "4", "--rank", "3",
		"COLDSNAP_RECOVERY_LOADER_STATUS_GROUP=device",
		"VLLM_ENABLE_STARTUP_PLAN=0", "COLDSNAP_DEFERRED_API_MM_WARMUP=0",
		"NCCL_CHECKPOINT_TERMINATION=destroy", "NCCL_CUMEM_ENABLE=1",
		"COLDSNAP_NCCL_REQUIRE_NETWORK_RESET=1",
		"COLDSNAP_RESTORE_RUNTIME_ENVIRONMENT_PATH=/opt/coldsnap/capsule/restore-runtime-environment.json",
		"COLDSNAP_RESTORE_TRANSPORT_ENVIRONMENT_PATH=/opt/coldsnap/capsule/restore-transport-environment.json",
		"PYTHONDONTWRITEBYTECODE=1",
		"/dev/infiniband:/dev/infiniband",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("command lacks %q: %#v", expected, command)
		}
	}
	for option, expected := range map[string]string{
		"--criu-compress-block-bytes":  "262144",
		"--criu-compress-acceleration": "1",
		"--criu-decompress-threads":    "1",
		"--criu-image-io-mode":         "direct",
	} {
		index := slices.Index(command, option)
		if index < 0 || index+1 >= len(command) || command[index+1] != expected {
			t.Fatalf("command option %s does not equal %q: %#v", option, expected, command)
		}
	}
	for _, rejected := range []string{
		"NCCL_CHECKPOINT_TERMINATION=abort", "NCCL_CUMEM_ENABLE=0",
	} {
		if slices.Contains(command, rejected) {
			t.Fatalf("command preserves unsafe ColdSnap runtime value %q: %#v", rejected, command)
		}
	}
	artifactOption := slices.Index(command, "--artifact-root")
	operation := slices.Index(command, "capture")
	if artifactOption < 0 || operation < artifactOption {
		t.Fatalf("rank controller options must precede operation: %#v", command)
	}
}

func TestCaptureAndRestorePreserveCommandWhileCheckpointForcesColdSnap(t *testing.T) {
	request := validRequest(1)
	request.Launch.Units[0].Command[4] =
		"exec vllm serve model --load-format instanttensor --enable-sleep-mode"
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})

	capture, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"capture-rank-0", "capture", "activation", request.ID, "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(capture, "COLDSNAP_CAPTURE_LOAD_FORMAT=instanttensor") {
		t.Fatalf("capture command lacks the recipe load format: %#v", capture)
	}
	if !strings.Contains(capture[len(capture)-1], "--load-format instanttensor") {
		t.Fatalf("capture command did not preserve InstantTensor: %#v", capture)
	}

	restore, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"restore-rank-0", "restore", "activation", request.ID, "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(restore, "COLDSNAP_CAPTURE_LOAD_FORMAT=instanttensor") {
		t.Fatalf("restore command retained capture load policy: %#v", restore)
	}
	if !strings.Contains(restore[len(restore)-1], "--load-format instanttensor") {
		t.Fatalf("restore command changed captured process identity: %#v", restore)
	}
}

func TestN580RankCommandUsesRecoveryWeightLoadPath(t *testing.T) {
	request := validRequest(1)
	request.Driver = snapshotdriver.Selection{ID: snapshotdriver.N580}
	request.Launch.Units[0].Command[4] =
		"exec vllm serve model --load-format instanttensor --enable-sleep-mode"
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"restore-rank-0", "restore", "activation", request.ID, "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest,
		[]string{"10.24.11.14=192.168.1.43"}, 4096, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{
		"COLDSNAP_LOAD_FORMAT=coldsnap", "COLDSNAP_RECOVERY_WEIGHT_SOURCE=safetensors",
		"/usr/local/bin/coldsnap-engine-rank-n580",
		"--worker-count", "1", "--generation", request.ID,
		"--defer-network-unlock", "--tcp-address-map", "10.24.11.14=192.168.1.43",
		"--tcp-port-shift", "4096",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("n580 command lacks %q: %#v", expected, command)
		}
	}
	if slices.Contains(command, "--leave-stopped") {
		t.Fatalf("n580 command exposed the n610 controller-level stop policy: %#v", command)
	}
}

func TestN580CaptureUsesFastLoaderAndArmsRecoveryMetadata(t *testing.T) {
	request := validRequest(1)
	request.Driver = snapshotdriver.Selection{ID: snapshotdriver.N580}
	request.Policy.Weights.Mode = "auto"
	request.Launch.Units[0].Command[4] =
		"exec vllm serve model --load-format instanttensor --enable-sleep-mode"
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"capture-rank-0", "capture", "activation", request.ID, "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{
		"COLDSNAP_LOAD_FORMAT=coldsnap",
		"COLDSNAP_CAPTURE_LOAD_FORMAT=instanttensor",
		"COLDSNAP_RECOVERY_WEIGHT_SOURCE=safetensors",
		"COLDSNAP_EXPORT_MODEL_PAYLOAD=1",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("n580 capture command lacks %q: %#v", expected, command)
		}
	}
}

func TestN580TargetLocalCaptureUsesExactTargetProcessTemplate(t *testing.T) {
	request := validRequest(1)
	request.Driver = snapshotdriver.Selection{ID: snapshotdriver.N580}
	request.Policy.Process.ArtifactScope = snapshot.ArtifactScopeTargetLocal
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"capture-rank-0", "capture", "activation", request.ID, "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{
		"COLDSNAP_N580_QUALIFICATION_PROCESS_TEMPLATE_PHASE=pre_worker_import",
		"COLDSNAP_N580_QUALIFICATION_ALLOW_CAPTURED_DRIVER_LIBRARIES=1",
		"COLDSNAP_N580_PORTABLE_WORKER_REEXEC=0",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("target-local n580 command lacks %q: %#v", expected, command)
		}
	}
}

func TestN580NativeModelPayloadUsesWorkerScopedStaging(t *testing.T) {
	request := validRequest(1)
	request.Operation = "restore"
	request.Driver = snapshotdriver.Selection{ID: snapshotdriver.N580}
	directory := t.TempDir()
	request.Artifact = directory + "/n580.json"
	n580, err := snapshotdriver.Lookup(snapshotdriver.N580)
	if err != nil {
		t.Fatal(err)
	}
	compatibility := validPortableCompatibility(1)
	compatibility.Units[0].NVIDIADriverCaptured = "580.95.05"
	compatibility.Units[0].NVIDIADriverMin = "580.95.05"
	artifact := snapshot.Artifact{
		Format: snapshot.ArtifactFormat, Kind: snapshot.ArtifactKind, State: "committed",
		Driver: n580, Requires: n580.BaseRequirements.Clone(), Graph: validGraphPolicyForDriver(request.Launch, snapshotdriver.N580),
		CaptureID: request.ID, RequestSHA256: testDigest, Launch: request.Launch,
		Compatibility: compatibility,
		Capsule: snapshot.Capsule{
			Images: []snapshot.CapsuleImage{{
				Unit: "unit-0", Reference: testDigest, Digest: testDigest,
				Root: "/opt/coldsnap/capsule", Driver: n580.Binding(),
			}},
			Objects: []snapshot.Object{{
				Role: "oci-capsule", Owner: snapshot.UnitOwner("unit-0"),
				Path: "units/unit-0/capsule.oci", Bytes: 100, SHA256: testDigest,
			}},
		},
		Weights: snapshot.WeightProviders{
			Native: &snapshot.NativeProvider{Driver: n580.Binding()},
			ModelPayloads: &snapshot.ModelPayloadProvider{Objects: []snapshot.Object{{
				Role: "model-weight-payload", Owner: snapshot.WorkerOwner("worker-0"),
				Path: modelPayloadObjectPath(testDigest), Bytes: 100, SHA256: testDigest,
			}}},
			Recovery: snapshot.RecoveryProvider{
				Source: "huggingface-safetensors", ModelID: request.Launch.Model.ID,
				Revision: request.Launch.Model.Revision, Driver: n580.Binding(), LoadPath: "coldsnap-replay",
				ReplayPlan: []snapshot.Object{{
					Role: "safetensors-replay-plan", Owner: snapshot.WorkerOwner("worker-0"),
					Path:  "drivers/n580/units/unit-0/hydration/worker-0/manifest.json",
					Bytes: 10, SHA256: testDigest,
				}},
			},
		},
		Acceptance: snapshot.ArtifactAcceptance{Accepted: true, Expected: "exact"},
	}
	payload, err := snapshot.Encode(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(request.Artifact, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	target, err := modelPayloadTarget(request, "worker-0")
	if err != nil {
		t.Fatal(err)
	}
	if target != "/var/cache/coldsnap/model-payloads/worker-0.pack" {
		t.Fatalf("n580 model payload target = %q", target)
	}
}

func TestRecoveryCapableCaptureFormats(t *testing.T) {
	for _, loadFormat := range []string{
		"auto", "safetensors", "fastsafetensors", "instanttensor", "coldsnap",
	} {
		command := []string{
			"bash", "--noprofile", "--norc", "-c",
			"exec vllm serve model --load-format " + loadFormat + " --enable-sleep-mode",
		}
		if !recoveryCapableCommand("vllm", command) {
			t.Fatalf("load format %q was rejected", loadFormat)
		}
	}
	if recoveryCapableCommand("vllm", []string{
		"bash", "--noprofile", "--norc", "-c",
		"exec vllm serve model --load-format pt --enable-sleep-mode",
	}) {
		t.Fatal("pt capture load format was accepted")
	}
}

func TestCaptureWeightPolicyExportsModelPayloadForPreferredModes(t *testing.T) {
	tests := []struct {
		mode string
		want bool
	}{
		{mode: "auto", want: true},
		{mode: "cache-only-auto", want: true},
		{mode: "native", want: true},
		{mode: "recovery", want: false},
	}
	for _, test := range tests {
		policy := snapshot.WeightPolicy{Mode: test.mode}
		if got := wantsModelPayload(policy); got != test.want {
			t.Errorf("wantsModelPayload(mode=%q) = %t, want %t", test.mode, got, test.want)
		}
	}

	policy := snapshot.WeightPolicy{
		Mode:   "recovery",
		Native: snapshot.NativePolicy{Repository: "org/native", Revision: "main"},
	}
	if !wantsModelPayload(policy) {
		t.Fatal("explicit native provider did not export a model payload")
	}
}

func TestRankCommandUsesTypedRecoveryBackend(t *testing.T) {
	request := validRequest(1)
	request.Policy.Weights.Recovery.LoaderBackend = "torch"
	request.Launch.Units[0].Environment = map[string]string{
		"COLDSNAP_RECOVERY_LOADER_BACKEND": "mmap",
	}
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"capture-rank-0", "capture", "activation", request.ID, "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(command, "COLDSNAP_RECOVERY_LOADER_BACKEND=torch") {
		t.Fatalf("command does not use the typed recovery backend: %#v", command)
	}
	if slices.Contains(command, "COLDSNAP_RECOVERY_LOADER_BACKEND=mmap") {
		t.Fatalf("command preserves a launch-environment recovery backend: %#v", command)
	}
}

func TestSGLangRankCommandInternalizesStableAllocatorPolicy(t *testing.T) {
	request := validRequest(1)
	request.Launch.Units[0].Command[4] += " --port 8011"
	request.Launch.Units[0].Environment = map[string]string{
		"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64,expandable_segments:True",
		"PYTORCH_ALLOC_CONF":      "expandable_segments:True,garbage_collection_threshold:0.8",
	}
	adapter := fixtureAdapter(Adapter{Engine: "sglang", Remote: fakeRemote{}, Timeout: 90 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"capture-rank-0", "capture", "activation", request.ID, "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{
		"PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:False",
		"PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,expandable_segments:False",
		"NCCL_CHECKPOINT_COORDINATOR_TIMEOUT=3600",
		"NCCL_CHECKPOINT_TERMINATION=abort-sync",
		"COLDSNAP_SGLANG_CRIU_HOLD=1",
		"/usr/local/bin/coldsnap-engine-rank-n610",
		"--worker-count", "1",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("SGLang command lacks stable allocator setting %q: %#v", expected, command)
		}
	}
	portOption := slices.Index(command, "--http-port")
	if portOption < 0 || portOption+1 >= len(command) || command[portOption+1] != "8011" {
		t.Fatalf("SGLang command does not preserve the resolved HTTP port: %#v", command)
	}
	for _, rejected := range []string{
		"PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True",
		"PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8",
	} {
		if slices.Contains(command, rejected) {
			t.Fatalf("SGLang command preserves expandable allocator setting %q: %#v", rejected, command)
		}
	}
}

func TestSGLangN580UsesExplicitDriverController(t *testing.T) {
	request := validRequest(1)
	request.Driver = snapshotdriver.Selection{ID: snapshotdriver.N580}
	request.Launch.Engine = "sglang"
	request.Launch.Units[0].Command[4] =
		"exec sglang serve --model-path org/model --load-format safetensors"
	adapter := fixtureAdapter(Adapter{Engine: "sglang", Remote: fakeRemote{}, Timeout: 90 * time.Minute})
	if _, err := adapter.configured(request); err != nil {
		t.Fatalf("configure SGLang n580: %v", err)
	}
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"capture-rank-0", "capture", "activation", request.ID, "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{
		"/usr/local/bin/coldsnap-engine-rank-n580",
		"--engine", "sglang", "COLDSNAP_UNIT_INDEX=0",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("SGLang n580 command lacks %q: %#v", expected, command)
		}
	}
}

func TestSGLangRecoveryLoadPathReflectsSnapshotBoundary(t *testing.T) {
	if got := recoveryLoadPath("sglang"); got != "sglang-inplace-disk" {
		t.Fatalf("n610 SGLang recovery load path = %q", got)
	}
	if got := processTemplateRecoveryLoadPath("sglang"); got != "sglang-startup-disk" {
		t.Fatalf("n580 SGLang recovery load path = %q", got)
	}
	if got := processTemplateRecoveryLoadPath("vllm"); got != "coldsnap-replay" {
		t.Fatalf("n580 vLLM recovery load path = %q", got)
	}
}

func TestRecoveryRankMountsLocalModelPayloadMaterialization(t *testing.T) {
	request := validRequest(1)
	request.Operation = "restore"
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"restore-rank-0", "restore", "activation", request.ID, "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest, nil, 0,
		&materializationMount{
			cacheRoot: "/host/coldsnap/model-payloads", controlPath: "/host/coldsnap/control.json",
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{
		"/host/coldsnap/model-payloads:/var/cache/coldsnap/model-payloads",
		"/host/coldsnap/control.json:/run/coldsnap/model-payload-materialization.json:ro",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("materialization command lacks %q: %#v", expected, command)
		}
	}
}

func TestPrepareModelPayloadMaterializationWritesPerUnitWorkerControls(t *testing.T) {
	request := multiWorkerRequest(t)
	objects := make([]snapshot.Object, 0, len(request.Launch.Execution.Workers))
	for _, worker := range request.Launch.Execution.Workers {
		objects = append(objects, snapshot.Object{
			Role: "model-weight-payload", Owner: snapshot.WorkerOwner(worker.ID),
			Path: modelPayloadObjectPath(testDigest), Bytes: 100, SHA256: testDigest,
		})
	}
	remote := &materializationRemote{}
	adapter := fixtureAdapter(Adapter{Remote: remote, StateRoot: "/cache/coldsnap"})
	mounts, err := adapter.prepareModelPayloadMaterialization(
		context.Background(), request, "restore-namespace", restorePreparation{
			materialize: objects, materializationMode: "required",
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(mounts) != len(request.Launch.Units) || len(remote.inputs) != len(request.Launch.Units) {
		t.Fatalf("materialization mounts=%#v inputs=%#v", mounts, remote.inputs)
	}
	for _, unit := range request.Launch.Units {
		var control materializationControl
		if err := json.Unmarshal(remote.inputs[unit.Host], &control); err != nil {
			t.Fatal(err)
		}
		if control.Mode != "required" || control.OperationID != request.ID ||
			control.OwnerUID != 1000 || control.OwnerGID != 1000 ||
			len(control.Workers) != len(request.Launch.Execution.UnitWorkers(unit.ID)) {
			t.Fatalf("unit %s materialization control = %#v", unit.ID, control)
		}
		for _, worker := range request.Launch.Execution.UnitWorkers(unit.ID) {
			target, ok := control.Workers[worker.ID]
			if !ok || target.Path != "sha256/"+strings.TrimPrefix(testDigest, "sha256:")+".pack" ||
				target.Bytes != 100 || target.SHA256 != testDigest {
				t.Fatalf("worker %s materialization target = %#v", worker.ID, target)
			}
		}
	}
}

func TestRequiredModelPayloadMaterializationRejectsRecoveryOnlyArtifact(t *testing.T) {
	artifact := snapshot.Artifact{Weights: snapshot.WeightProviders{
		Recovery: snapshot.RecoveryProvider{Source: "huggingface-safetensors"},
	}}
	selection := snapshot.Selection{Provider: "recovery"}

	objects, mode, err := modelPayloadMaterialization(artifact, selection, "required")

	if err == nil || !strings.Contains(err.Error(), "artifact has no native model-payload inventory") {
		t.Fatalf("required materialization error = %v", err)
	}
	if objects != nil || mode != "" {
		t.Fatalf("required materialization result = %#v, %q", objects, mode)
	}
}

func TestAsyncModelPayloadMaterializationAllowsRecoveryOnlyArtifact(t *testing.T) {
	artifact := snapshot.Artifact{Weights: snapshot.WeightProviders{
		Recovery: snapshot.RecoveryProvider{Source: "huggingface-safetensors"},
	}}

	objects, mode, err := modelPayloadMaterialization(
		artifact, snapshot.Selection{Provider: "recovery"}, "async",
	)

	if err != nil || objects != nil || mode != "" {
		t.Fatalf("async materialization result = %#v, %q, %v", objects, mode, err)
	}
}

func TestEngineMaterializationCapabilitiesAreExplicit(t *testing.T) {
	for _, test := range []struct {
		engine, requested, expected string
		wantError                   bool
	}{
		{engine: "vllm", requested: "", expected: "async"},
		{engine: "vllm", requested: "required", expected: "required"},
		{engine: "sglang", requested: "", expected: "off"},
		{engine: "sglang", requested: "off", expected: "off"},
		{engine: "sglang", requested: "async", wantError: true},
		{engine: "sglang", requested: "required", wantError: true},
	} {
		actual, err := engineMaterializationMode(test.engine, test.requested)
		if test.wantError {
			if err == nil || !strings.Contains(err.Error(), "does not support") {
				t.Fatalf("%s %q error = %v", test.engine, test.requested, err)
			}
			continue
		}
		if err != nil || actual != test.expected {
			t.Fatalf("%s %q = %q, %v; want %q", test.engine, test.requested, actual, err, test.expected)
		}
	}
}

func TestRankCommandInternalizesAndMountsStagedRuntimeCache(t *testing.T) {
	request := validRequest(1)
	request.Policy.Cache = snapshot.CachePolicy{
		Seed: true, Paths: []string{snapshot.CanonicalRuntimeCachePath},
		Staged: []snapshot.PreparedCache{{Unit: "unit-0", Path: "/host/staged/rank0"}},
	}
	request.Launch.Units[0].Environment = map[string]string{
		"CUDA_CACHE_DISABLE": "1", "CUDA_CACHE_MAXSIZE": "1", "CUDA_CACHE_PATH": "/recipe/cache",
	}
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"capture-rank-0", "capture", "activation", request.ID, "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{
		"CUDA_CACHE_DISABLE=0", "CUDA_CACHE_MAXSIZE=1073741824",
		"CUDA_CACHE_PATH=/var/cache/coldsnap/runtime/cuda",
		"VLLM_ENABLE_STARTUP_PLAN=1",
		"COLDSNAP_STARTUP_PLAN_MAX_SHORTFALL_BYTES=536870912",
		"COLDSNAP_DEFERRED_WARMUP=0",
		"COLDSNAP_DEFERRED_WARMUP_ARM_FILE=/opt/coldsnap/capsule/async-graphs-arm",
		"COLDSNAP_FULLY_WARM_FILE=/opt/coldsnap/capsule/fully-warm.json",
		"COLDSNAP_DEFERRED_API_MM_WARMUP=0",
		"COLDSNAP_API_MM_WARMUP_DELAY_SECONDS=0",
		"COLDSNAP_API_MM_WARM_FILE=/opt/coldsnap/capsule/api-mm-warm.json",
		"COLDSNAP_WARMUP_GENERATION=generic-capture",
		"VLLM_CACHE_ROOT=/var/cache/coldsnap/runtime/vllm",
		"TORCHINDUCTOR_CACHE_DIR=/var/cache/coldsnap/runtime/inductor",
		"TRITON_CACHE_DIR=/var/cache/coldsnap/runtime/triton",
		"TORCH_HOME=/var/cache/coldsnap/runtime/torch",
		"TORCH_EXTENSIONS_DIR=/var/cache/coldsnap/runtime/torch_extensions",
		"TVM_FFI_CACHE_DIR=/var/cache/coldsnap/runtime/tvm_ffi",
		"XDG_CACHE_HOME=/var/cache/coldsnap/runtime",
		"/host/staged/rank0:/var/cache/coldsnap/runtime",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("command lacks managed CUDA cache setting %q: %#v", expected, command)
		}
	}
	for _, rejected := range []string{
		"CUDA_CACHE_DISABLE=1", "CUDA_CACHE_MAXSIZE=1", "CUDA_CACHE_PATH=/recipe/cache",
	} {
		if slices.Contains(command, rejected) {
			t.Fatalf("command preserves recipe CUDA cache setting %q: %#v", rejected, command)
		}
	}
}

func TestRestoreRankCommandUsesArtifactCaptureIdentity(t *testing.T) {
	request := validRequest(1)
	request.ID = "restore-attempt-1"
	request.Operation = "restore"
	request.Policy.Process.AsyncGraphs = true
	request.Policy.Cache = snapshot.CachePolicy{
		Seed: true, Paths: []string{snapshot.CanonicalRuntimeCachePath},
	}
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"restore-rank-0", "restore", "activation", "original-capture", "/state/rank0", "/state/endpoint",
		"recovery", nil, "sha256:captured-base", []string{"10.0.0.1=10.0.0.2"}, 4096, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(command, "COLDSNAP_CAPTURE_ID=original-capture") {
		t.Fatalf("command does not use artifact capture identity: %#v", command)
	}
	if slices.Contains(command, "COLDSNAP_CAPTURE_ID=restore-attempt-1") {
		t.Fatalf("command incorrectly uses restore request identity: %#v", command)
	}
	for _, expected := range []string{
		"--allow-compatible-criu-runtime", "--leave-stopped", "--defer-network-unlock",
		"--tcp-address-map", "10.0.0.1=10.0.0.2",
		"--tcp-port-shift", "4096",
		"sha256:captured-base",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("portable restore command lacks %q: %#v", expected, command)
		}
	}
	if !slices.Contains(command, "VLLM_CACHE_ROOT=/var/cache/coldsnap/runtime/vllm") {
		t.Fatalf("restore command does not address the capsule runtime cache: %#v", command)
	}
	for _, expected := range []string{
		"VLLM_ENABLE_STARTUP_PLAN=1",
		"COLDSNAP_STARTUP_PLAN_MAX_SHORTFALL_BYTES=536870912",
		"COLDSNAP_DEFERRED_WARMUP=1",
		"COLDSNAP_DEFERRED_WARMUP_ARM_FILE=/opt/coldsnap/capsule/async-graphs-arm",
		"COLDSNAP_FULLY_WARM_FILE=/opt/coldsnap/capsule/fully-warm.json",
		"COLDSNAP_DEFERRED_API_MM_WARMUP=1",
		"COLDSNAP_API_MM_WARMUP_DELAY_SECONDS=30",
		"COLDSNAP_API_MM_WARM_FILE=/opt/coldsnap/capsule/api-mm-warm.json",
		"COLDSNAP_WARMUP_GENERATION=restore-attempt-1",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("restore command lacks %q: %#v", expected, command)
		}
	}
	for _, argument := range command {
		if strings.HasSuffix(argument, ":/var/cache/coldsnap/runtime") {
			t.Fatalf("restore command attached a host runtime cache: %#v", command)
		}
	}
}

func TestRetainedNCCLGraphRankCommandSelectsExactProviderLifecycle(t *testing.T) {
	request := validRequest(2)
	request.Operation = "restore"
	request.Policy.Process.AsyncGraphs = true
	request.Policy.Process.GraphPolicy = snapshot.GraphPreserveNCCLExec
	request.Policy.Cache = snapshot.CachePolicy{
		Seed: true, Paths: []string{snapshot.CanonicalRuntimeCachePath},
	}
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"restore-rank-0", "restore", "activation", "original-capture", "/state/rank0", "/state/endpoint",
		"recovery", nil, request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{
		"COLDSNAP_GRAPH_POLICY=preserve-nccl-exec",
		"COLDSNAP_NCCL_IN_PLACE_EXPERIMENT=1",
		"COLDSNAP_NCCL_IN_PLACE_MODE=net-reconnect-v1",
		"COLDSNAP_NCCL_IN_PLACE_ACTIVATION_PATH=/opt/coldsnap/capsule/nccl-in-place-activation",
		"COLDSNAP_DEFERRED_WARMUP=0",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("retained graph command lacks %q: %#v", expected, command)
		}
	}
	required := "COLDSNAP_NCCL_REQUIRED_CAPABILITIES=communicator-suspend-in-place,full-network-reset,graph-resource-retention,ib-roce-device-release,synchronous-termination,transport-detach-in-place"
	if !slices.Contains(command, required) {
		t.Fatalf("retained graph command lacks exact provider capabilities: %#v", command)
	}
}

func TestRestoreRankUsesCanonicalSparkrunIdentityWhenSupplied(t *testing.T) {
	request := validRequest(1)
	request.Operation = "restore"
	request.Workload = snapshot.WorkloadIdentity{
		ClusterID: "sparkrun_deadbeef_01234567", IntentID: "deadbeef",
		Recipe: "catalog/qwen", Runtime: "vllm-distributed", Model: "org/model",
		ServedModelName: "qwen",
	}
	if got := restoreContainerName(request, 0); got != "sparkrun_deadbeef_01234567_node_0" {
		t.Fatalf("restore container name = %q", got)
	}
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		restoreContainerName(request, 0), "restore", "activation", "original-capture",
		"/state/rank0", "/state/endpoint", "recovery", nil,
		request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, label := range []string{
		"io.sparksq.coldsnap.capture=original-capture",
		"io.sparksq.coldsnap.driver=n610",
		"io.sparksq.coldsnap.managed=true",
		"io.sparksq.coldsnap.workload=sparkrun_deadbeef_01234567",
		"io.sparksq.coldsnap.rank=0",
		"io.sparksq.coldsnap.unit=unit-0",
	} {
		if !slices.Contains(command, label) {
			t.Fatalf("restore command lacks lifecycle label %q: %#v", label, command)
		}
	}
	for _, expected := range []string{
		"COLDSNAP_EXPORT_MODEL_PAYLOAD=0",
		"COLDSNAP_HIBERNATE_REUSE_BLOB=1",
	} {
		if !slices.Contains(command, expected) {
			t.Fatalf("restore command lacks lifecycle setting %q: %#v", expected, command)
		}
	}
	activation := slices.Index(command, "--activation-state")
	if activation < 0 || activation+1 >= len(command) || command[activation+1] != "running" {
		t.Fatalf("ordinary restore lacks explicit running activation state: %#v", command)
	}
}

func TestWarmRestoreRankHoldsBeforeHydration(t *testing.T) {
	request := validRequest(1)
	request.Operation = "restore"
	request.Lifecycle.ActivationState = "warm"
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Timeout: 20 * time.Minute})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"restore-rank-0", "restore", "activation", "original-capture",
		"/state/rank0", "/state/endpoint", "recovery", nil,
		request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	option := slices.Index(command, "--activation-state")
	if option < 0 || command[option+1] != "warm" {
		t.Fatalf("warm activation command = %#v", command)
	}
}

func TestExposeWorkloadLogsLinksEveryRestoredRankToOrchestratorPath(t *testing.T) {
	request := validRequest(2)
	request.Workload = snapshot.WorkloadIdentity{
		ClusterID: "sparkrun_deadbeef_01234567", IntentID: "deadbeef",
		Recipe: "catalog/qwen", Runtime: "vllm-distributed", Model: "org/model",
		LogPath: "/tmp/sparkrun_serve.log",
	}
	remote := &workloadLogRemote{}
	containers := []string{
		"sparkrun_deadbeef_01234567_node_0",
		"sparkrun_deadbeef_01234567_node_1",
	}
	if err := (fixtureAdapter(Adapter{Remote: remote})).exposeWorkloadLogs(
		context.Background(), request, containers,
	); err != nil {
		t.Fatal(err)
	}
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	for rank, container := range containers {
		host := request.Launch.Units[rank].Host
		expected := [][]string{
			{host, "docker", "exec", container, "test", "-f", containerTargetLogPath},
			{host, "docker", "exec", container, "ln", "-sfnT", containerTargetLogPath, request.Workload.LogPath},
			{host, "docker", "exec", container, "readlink", request.Workload.LogPath},
		}
		for _, command := range expected {
			if !slices.ContainsFunc(remote.calls, func(call []string) bool { return slices.Equal(call, command) }) {
				t.Fatalf("rank %d log command missing: %#v calls=%#v", rank, command, remote.calls)
			}
		}
	}
}

func TestExposeWorkloadLogsIsOptionalForDirectColdSnapUse(t *testing.T) {
	if err := (fixtureAdapter(Adapter{})).exposeWorkloadLogs(context.Background(), validRequest(1), nil); err != nil {
		t.Fatal(err)
	}
}

type capsulePrepareRemote struct {
	mutex sync.Mutex
	calls [][]string
	seen  map[string]bool
}

func (remote *capsulePrepareRemote) Run(_ context.Context, host string, arguments ...string) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	remote.calls = append(remote.calls, append([]string{host}, arguments...))
	if len(arguments) >= 4 && arguments[1] == "image" && arguments[2] == "inspect" {
		key := host + "\x00" + arguments[3]
		if !remote.seen[key] {
			return nil, errors.New("not resident")
		}
		return json.Marshal(hostops.ImageInfo{ID: testDigest})
	}
	if len(arguments) >= 3 && arguments[1] == "pull" {
		remote.seen[host+"\x00"+arguments[2]] = true
	}
	return []byte("ok\n"), nil
}

func (*capsulePrepareRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}
func (*capsulePrepareRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, nil
}
func (*capsulePrepareRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

type authenticatedCapsulePrepareRemote struct {
	*capsulePrepareRemote
	pulls []string
}

func (remote *authenticatedCapsulePrepareRemote) PullDockerImage(
	_ context.Context, host, reference string,
) error {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	remote.pulls = append(remote.pulls, host+"\x00"+reference)
	remote.seen[host+"\x00"+reference] = true
	return nil
}

func TestPrepareCapsuleImagesPullsBeforeActivation(t *testing.T) {
	request := validRequest(2)
	remote := &capsulePrepareRemote{seen: make(map[string]bool)}
	images := []snapshot.CapsuleImage{
		{Unit: "unit-0", Reference: "registry/capsule0@" + testDigest, Digest: testDigest},
		{Unit: "unit-1", Reference: "registry/capsule1@" + testDigest, Digest: testDigest},
	}
	adapter := fixtureAdapter(Adapter{Remote: remote})
	if err := adapter.prepareCapsuleImages(context.Background(), request, images); err != nil {
		t.Fatal(err)
	}
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	for _, image := range images {
		pull := false
		for _, call := range remote.calls {
			if slices.Contains(call, "pull") && slices.Contains(call, image.Reference) {
				pull = true
			}
		}
		if !pull {
			t.Fatalf("capsule was not pulled before activation: %s calls=%#v", image.Reference, remote.calls)
		}
	}
}

func TestPrepareCapsuleImagesUsesControllerAuthenticatedPuller(t *testing.T) {
	request := validRequest(2)
	remote := &authenticatedCapsulePrepareRemote{
		capsulePrepareRemote: &capsulePrepareRemote{seen: make(map[string]bool)},
	}
	images := []snapshot.CapsuleImage{
		{Unit: "unit-0", Reference: "registry/capsule0@" + testDigest, Digest: testDigest},
		{Unit: "unit-1", Reference: "registry/capsule1@" + testDigest, Digest: testDigest},
	}
	adapter := fixtureAdapter(Adapter{Remote: remote})
	if err := adapter.prepareCapsuleImages(context.Background(), request, images); err != nil {
		t.Fatal(err)
	}
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	for _, image := range images {
		pulled := false
		for _, pull := range remote.pulls {
			pulled = pulled || strings.HasSuffix(pull, "\x00"+image.Reference)
		}
		if !pulled {
			t.Fatalf("capsule lacked a controller-authenticated pull: %s pulls=%#v", image.Reference, remote.pulls)
		}
	}
	for _, call := range remote.calls {
		if slices.Contains(call, "pull") {
			t.Fatalf("restore used rank-local Docker credentials: %#v", remote.calls)
		}
	}
}

func TestPrepareCapsuleImagesReportsPerUnitPullProgress(t *testing.T) {
	request := validRequest(2)
	remote := &authenticatedCapsulePrepareRemote{
		capsulePrepareRemote: &capsulePrepareRemote{seen: make(map[string]bool)},
	}
	images := []snapshot.CapsuleImage{
		{Unit: "unit-0", Reference: "registry/capsule0@" + testDigest, Digest: testDigest},
		{Unit: "unit-1", Reference: "registry/capsule1@" + testDigest, Digest: testDigest},
	}
	collector := operationtiming.New(time.Now())
	ctx := operationtiming.WithCollector(context.Background(), collector)
	if err := (fixtureAdapter(Adapter{Remote: remote})).prepareCapsuleImages(ctx, request, images); err != nil {
		t.Fatal(err)
	}
	spans := collector.Export().Spans
	for _, unit := range request.Launch.Units {
		for _, name := range []string{"capsule.prepare", "capsule.inspect", "capsule.pull", "capsule.verify"} {
			if !slices.ContainsFunc(spans, func(span snapshot.TimingSpan) bool {
				return span.Name == name && span.Attributes["unit"] == unit.ID && span.Attributes["host"] == unit.Host
			}) {
				t.Fatalf("%s timing for %s on %s is missing: %#v", name, unit.ID, unit.Host, spans)
			}
		}
	}
}

func TestMaterializationStatusImportsDetailedRuntimePhases(t *testing.T) {
	request := validRequest(1)
	unit := request.Launch.Units[0]
	worker := request.Launch.Execution.Workers[0]
	object := snapshot.Object{
		Role: "model-weight-payload", Owner: snapshot.WorkerOwner(worker.ID),
		Path:  "model-payloads/sha256/" + strings.TrimPrefix(testDigest, "sha256:") + ".pack",
		Bytes: 100, SHA256: testDigest,
	}
	payload := []byte(fmt.Sprintf(`{
		"state":"ready",
		"worker_id":%q,
		"path":"/cache/model.pack",
		"sha256":%q,
		"bytes":100,
		"started_unix_ns":1700000000000000000,
		"seconds":12.5,
		"phase_seconds":{
			"canonical_fingerprint_s":2.0,
			"cuda_copy_s":3.0,
			"disk_write_s":5.0,
			"sync_s":0.5,
			"payload_verification_s":1.0,
			"validation_record_s":0.1
		}
	}`, worker.ID, testDigest))
	collector := operationtiming.New(time.Now())
	ctx := operationtiming.WithCollector(context.Background(), collector)
	ctx, parent := operationtiming.Start(ctx, "materialization.verify", nil)
	if err := importMaterializationTiming(ctx, unit, worker, object, payload); err != nil {
		t.Fatal(err)
	}
	parent.End(nil)
	spans := collector.Export().Spans
	for _, name := range []string{
		"runtime.materialization",
		"runtime.materialization.canonicalize",
		"runtime.materialization.gpu_to_host",
		"runtime.materialization.write_service",
		"runtime.materialization.fdatasync",
		"runtime.materialization.post_write_verify",
		"runtime.materialization.validation_publish",
	} {
		if !slices.ContainsFunc(spans, func(span snapshot.TimingSpan) bool {
			return span.Name == name && span.Attributes["worker"] == worker.ID
		}) {
			t.Fatalf("materialization span %s is missing: %#v", name, spans)
		}
	}
}

func TestTransientNCCLInitializationFailureClassification(t *testing.T) {
	for _, output := range []string{
		"RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)",
		"torch.distributed.DistBackendError: ncclUnhandledCudaError: Call to CUDA function failed.",
	} {
		if !isTransientNCCLInitializationFailure([]byte(output)) {
			t.Fatalf("transient NCCL initialization failure was not classified: %q", output)
		}
	}
}

func TestTransientNCCLInitializationFailureDoesNotMatchGenericFailures(t *testing.T) {
	for _, output := range []string{
		"NCCL error: remote process exited or there was a network error",
		"RuntimeError: CUDA out of memory",
		"one or more CUDA graph workers failed capture",
	} {
		if isTransientNCCLInitializationFailure([]byte(output)) {
			t.Fatalf("unrelated failure was classified as transient NCCL initialization: %q", output)
		}
	}
}

func TestVersionedNCCLProviderBindingsArePerUnitAndExact(t *testing.T) {
	request := validRequest(2)
	remote := ncclProviderRemote{providerByHost: map[string]string{
		"node-0": "nccl-2.31.2-1+coldsnap.10",
		"node-1": "nccl-2.31.2-1+coldsnap.10",
	}}
	adapter := fixtureAdapter(Adapter{Remote: remote})
	bindings, err := adapter.verifyNCCLProviders(
		context.Background(), request.Launch, nil, snapshot.RuntimeProviders{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(bindings) != 2 || bindings[0].Owner != snapshot.UnitOwner("unit-0") ||
		bindings[1].Owner != snapshot.UnitOwner("unit-1") {
		t.Fatalf("NCCL provider bindings = %#v", bindings)
	}
	images := map[string]snapshot.CapsuleImage{
		"unit-0": {Unit: "unit-0", Reference: "capsule-0@" + testDigest},
		"unit-1": {Unit: "unit-1", Reference: "capsule-1@" + testDigest},
	}
	if _, err := adapter.verifyNCCLProviders(
		context.Background(), request.Launch, images,
		snapshot.RuntimeProviders{Bindings: bindings},
	); err != nil {
		t.Fatal(err)
	}
	for index := range bindings {
		formatted, err := json.MarshalIndent(json.RawMessage(bindings[index].Payload), "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		bindings[index].Payload = formatted
	}
	if _, err := adapter.verifyNCCLProviders(
		context.Background(), request.Launch, images,
		snapshot.RuntimeProviders{Bindings: bindings},
	); err != nil {
		t.Fatalf("formatted equivalent NCCL binding was rejected: %v", err)
	}
	remote.providerByHost["node-1"] = "nccl-2.30.7-1+coldsnap.10"
	if _, err := adapter.verifyNCCLProviders(
		context.Background(), request.Launch, images,
		snapshot.RuntimeProviders{Bindings: bindings},
	); err == nil || !strings.Contains(err.Error(), "differs from captured") {
		t.Fatalf("changed NCCL provider error = %v", err)
	}
}

type capsulePublishRemote struct {
	mutex  sync.Mutex
	calls  [][]string
	pushes []string
}

func (remote *capsulePublishRemote) Run(_ context.Context, _ string, arguments ...string) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	remote.calls = append(remote.calls, slices.Clone(arguments))
	if len(arguments) > 2 && arguments[1] == "image" && arguments[2] == "inspect" {
		return json.Marshal(hostops.ImageInfo{ID: testDigest, RepoDigests: []string{"scitrera/coldsnap-test@" + publishedTestDigest}})
	}
	return []byte("ok\n"), nil
}

func (*capsulePublishRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (*capsulePublishRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, nil
}

func (*capsulePublishRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func (remote *capsulePublishRemote) PushDockerImage(_ context.Context, host, tag string) error {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	remote.pushes = append(remote.pushes, host+" "+tag)
	return nil
}

func TestPublishPromotesAcceptedLocalCapsulesIntoNewDescriptor(t *testing.T) {
	request := validRequest(2)
	request.Operation = "publish"
	directory := t.TempDir()
	request.Artifact = directory + "/local.json"
	request.Output = directory + "/published.json"
	request.Policy.Capsule = snapshot.CapsulePolicy{
		Repository: "docker.io/scitrera/coldsnap-test",
	}
	artifact := snapshot.Artifact{
		Format: snapshot.ArtifactFormat, Kind: snapshot.ArtifactKind, State: "committed",
		Driver: validSnapshotDriverContract(), Requires: validSnapshotDriverContract().BaseRequirements.Clone(),
		Graph:     validGraphPolicy(request.Launch),
		CaptureID: "generic-capture", RequestSHA256: testDigest, Launch: request.Launch,
		Compatibility: validPortableCompatibility(2),
		Capsule: snapshot.Capsule{
			Images: []snapshot.CapsuleImage{
				{Unit: "unit-0", Reference: testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule", Driver: validSnapshotDriverContract().Binding()},
				{Unit: "unit-1", Reference: testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule", Driver: validSnapshotDriverContract().Binding()},
			},
			Objects: []snapshot.Object{
				{Role: "oci-capsule", Owner: snapshot.UnitOwner("unit-0"), Path: "units/unit-0/capsule.oci", Bytes: 100, SHA256: testDigest},
				{Role: "oci-capsule", Owner: snapshot.UnitOwner("unit-1"), Path: "units/unit-1/capsule.oci", Bytes: 100, SHA256: testDigest},
			},
		},
		Weights: snapshot.WeightProviders{Recovery: snapshot.RecoveryProvider{
			Source: "huggingface-safetensors", ModelID: request.Launch.Model.ID,
			Revision: request.Launch.Model.Revision, Driver: validSnapshotDriverContract().Binding(),
			LoadPath: "coldsnap-replay",
			ReplayPlan: []snapshot.Object{
				{Role: "safetensors-replay-plan", Owner: snapshot.WorkerOwner("worker-0"), Path: "drivers/n610/units/unit-0/hydration/worker-0/manifest.json", Bytes: 10, SHA256: testDigest},
				{Role: "safetensors-replay-plan", Owner: snapshot.WorkerOwner("worker-1"), Path: "drivers/n610/units/unit-1/hydration/worker-1/manifest.json", Bytes: 10, SHA256: testDigest},
			},
		}},
		Acceptance: snapshot.ArtifactAcceptance{Accepted: true, Expected: "exact"},
	}
	payload, err := snapshot.Encode(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(request.Artifact, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	target, err := modelPayloadTarget(request, "worker-0")
	if err != nil {
		t.Fatal(err)
	}
	if target != "/opt/coldsnap/capsule/hydration/worker-0/model-weights.pack" {
		t.Fatalf("driver-qualified model payload target = %q", target)
	}
	remote := &capsulePublishRemote{}
	if err := (fixtureAdapter(Adapter{Remote: remote})).Run(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	published, err := snapshot.ReadArtifact(request.Output)
	if err != nil {
		t.Fatal(err)
	}
	for index, image := range published.Capsule.Images {
		if image.Reference != "docker.io/scitrera/coldsnap-test@"+publishedTestDigest || image.Unit != "unit-"+strconv.Itoa(index) {
			t.Fatalf("published image %d = %#v", index, image)
		}
	}
	for index, object := range published.Capsule.Objects {
		if object.Owner != snapshot.UnitOwner("unit-"+strconv.Itoa(index)) || object.SHA256 != publishedTestDigest {
			t.Fatalf("published object %d = %#v", index, object)
		}
	}
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	for _, call := range remote.calls {
		if slices.Contains(call, "push") {
			t.Fatalf("publish used rank-local Docker credentials: %#v", remote.calls)
		}
	}
	if len(remote.pushes) != 2 {
		t.Fatalf("controller-authenticated pushes = %#v", remote.pushes)
	}
}

func TestPublishRejectsReplacementHostForLocalCapsule(t *testing.T) {
	request := validRequest(1)
	request.Operation = "publish"
	request.Launch.Units[0].Host = "replacement"
	directory := t.TempDir()
	request.Artifact = directory + "/local.json"
	request.Output = directory + "/published.json"
	request.Policy.Capsule.Repository = "docker.io/scitrera/coldsnap-test"
	artifact := snapshot.Artifact{
		Format: snapshot.ArtifactFormat, Kind: snapshot.ArtifactKind, State: "committed",
		Driver: validSnapshotDriverContract(), Requires: validSnapshotDriverContract().BaseRequirements.Clone(),
		Graph:     validGraphPolicy(validRequest(1).Launch),
		CaptureID: "generic-capture", RequestSHA256: testDigest,
		Launch:        validRequest(1).Launch,
		Compatibility: validPortableCompatibility(1),
		Capsule: snapshot.Capsule{
			Images: []snapshot.CapsuleImage{{
				Unit: "unit-0", Reference: testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule",
				Driver: validSnapshotDriverContract().Binding(),
			}},
			Objects: []snapshot.Object{{Role: "oci-capsule", Owner: snapshot.UnitOwner("unit-0"), Path: "units/unit-0/capsule.oci", Bytes: 100, SHA256: testDigest}},
		},
		Weights: snapshot.WeightProviders{Recovery: snapshot.RecoveryProvider{
			Source: "huggingface-safetensors", ModelID: request.Launch.Model.ID,
			Revision: request.Launch.Model.Revision, Driver: validSnapshotDriverContract().Binding(),
			LoadPath:   "coldsnap-replay",
			ReplayPlan: []snapshot.Object{{Role: "safetensors-replay-plan", Owner: snapshot.WorkerOwner("worker-0"), Path: "drivers/n610/units/unit-0/hydration/worker-0/manifest.json", Bytes: 10, SHA256: testDigest}},
		}},
		Acceptance: snapshot.ArtifactAcceptance{Accepted: true, Expected: "exact"},
	}
	payload, err := snapshot.Encode(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(request.Artifact, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	err = (fixtureAdapter(Adapter{Remote: &capsulePublishRemote{}})).Run(context.Background(), request)
	if err == nil || !strings.Contains(err.Error(), "belongs to capture host") {
		t.Fatalf("publish host validation error = %v", err)
	}
}

func TestCompatibleLaunchAllowsNewHostsButNotNewProcessIdentity(t *testing.T) {
	captured := validRequest(2).Launch
	requested := validRequest(2).Launch
	captured.Units[0].Command = []string{"bash", "-c", "vllm serve model --master-addr 10.24.11.13 --master-port=25000 --port 8000"}
	requested.Units[0].Command = []string{"bash", "-c", "vllm serve model --master-addr 10.24.11.17 --master-port=25100 --port=8103"}
	requested.Units[0].Host = "replacement-a"
	requested.Units[1].Host = "replacement-b"
	requested.Units[0].Image = "different-builder-image@" + publishedTestDigest
	requested.Units[0].ImageDigest = publishedTestDigest
	captured.Units[0].Environment = map[string]string{
		"NCCL_SOCKET_IFNAME": "captured-iface", "MODEL_SETTING": "same",
	}
	requested.Units[0].Environment = map[string]string{
		"NCCL_SOCKET_IFNAME": "destination-iface", "MODEL_SETTING": "same",
	}
	captured.Units[0].Mounts = []snapshot.Mount{{Source: "/capture/cache", Target: "/cache"}}
	requested.Units[0].Mounts = []snapshot.Mount{{Source: "/restore/cache", Target: "/cache"}}
	if err := compatibleLaunch(captured, requested); err != nil {
		t.Fatal(err)
	}
	requested.Units[1].Command = []string{"different"}
	if err := compatibleLaunch(captured, requested); err == nil {
		t.Fatal("changed process command was accepted")
	}
	requested = captured
	requested.Units = slices.Clone(captured.Units)
	requested.Units[0].Environment = maps.Clone(captured.Units[0].Environment)
	requested.Units[0].Environment["MODEL_SETTING"] = "different"
	if err := compatibleLaunch(captured, requested); err == nil {
		t.Fatal("changed non-transport environment was accepted")
	}
	requested.Units[0].Environment["MODEL_SETTING"] = "same"
	requested.Units[0].Mounts = []snapshot.Mount{{Source: "/restore/cache", Target: "/different"}}
	if err := compatibleLaunch(captured, requested); err == nil {
		t.Fatal("changed mount target was accepted")
	}
}

func TestPortablePlatformAcceptsSameKernelGPUAndNewerDriver(t *testing.T) {
	request := validRequest(2)
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}})
	if err := adapter.verifyPortablePlatforms(
		context.Background(), request, validPortableCompatibility(2),
	); err != nil {
		t.Fatal(err)
	}
}

func TestPortablePlatformValidatesEveryDeviceInMultiWorkerUnit(t *testing.T) {
	request := multiWorkerRequest(t)
	compatibility := snapshot.ArtifactCompatibility{Policy: snapshot.PortabilityPolicy}
	for _, unit := range request.Launch.Units {
		compatibility.Units = append(compatibility.Units, snapshot.UnitPlatformCompatibility{
			Unit: unit.ID, Architecture: "aarch64", Kernel: "6.17.0-1029-nvidia",
			NVIDIADriverCaptured: "610.43.02", NVIDIADriverMin: "610.43.02", CUDAUserspace: "13.0",
			Devices: []snapshot.DeviceCompatibility{
				{Slot: 0, GPUName: "NVIDIA GB10", ComputeCapability: "12.1"},
				{Slot: 1, GPUName: "NVIDIA GB10", ComputeCapability: "12.1"},
			},
		})
	}
	if err := (fixtureAdapter(Adapter{Remote: fakeRemote{}})).verifyPortablePlatforms(
		context.Background(), request, compatibility,
	); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyStagedPacksUsesPayloadValidationAndIgnoresCTime(t *testing.T) {
	identity := snapshot.PreparedPayloadValidation{
		Record:   "/cache/native/rank0.pack.coldsnap-validation.json",
		Provider: "sha256-cache-v1", ContentEvidence: "cached-full-sha256", Reason: "cached_validation",
		Device: 1, Inode: 2, Size: 100, MTimeNS: 3, CTimeNS: 4,
	}
	remote := &packIdentityRemote{identity: identity}
	request := validRequest(1)
	pack := snapshot.PreparedPayload{
		Worker: "worker-0", Path: "/cache/native/rank0.pack", Bytes: 100, SHA256: testDigest,
		Validation: &identity,
	}
	if err := (fixtureAdapter(Adapter{Remote: remote})).verifyStagedPayloads(
		context.Background(), request, []snapshot.PreparedPayload{pack},
	); err != nil {
		t.Fatal(err)
	}
	if len(remote.calls) != 1 || len(remote.calls[0]) < 2 || remote.calls[0][1] != "payload-verify" {
		t.Fatalf("verification calls = %#v", remote.calls)
	}

	remote.identity.CTimeNS++
	if err := (fixtureAdapter(Adapter{Remote: remote})).verifyStagedPayloads(
		context.Background(), request, []snapshot.PreparedPayload{pack},
	); err != nil {
		t.Fatalf("ctime-only change was rejected: %v", err)
	}
}

func TestPortablePlatformUsesCapabilityKernelPolicyAndRejectsOtherRegressions(t *testing.T) {
	request := validRequest(1)
	output := &bytes.Buffer{}
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, Output: output})
	compatibility := validPortableCompatibility(1)
	compatibility.Units[0].Kernel = "different"
	if err := adapter.verifyPortablePlatforms(context.Background(), request, compatibility); err != nil {
		t.Fatalf("capability kernel compatibility error = %v", err)
	}
	if !strings.Contains(output.String(), "using capsule-pinned CRIU capability admission") {
		t.Fatalf("kernel compatibility warning = %q", output.String())
	}
	request.Policy.Compatibility.Kernel = snapshot.KernelCompatibilityExact
	if err := adapter.verifyPortablePlatforms(context.Background(), request, compatibility); err == nil ||
		!strings.Contains(err.Error(), "under exact policy") {
		t.Fatalf("exact kernel compatibility error = %v", err)
	}
	request.Policy.Compatibility.Kernel = snapshot.KernelCompatibilityCapability

	compatibility = validPortableCompatibility(1)
	compatibility.Units[0].Devices[0].GPUName = "different"
	if err := adapter.verifyPortablePlatforms(context.Background(), request, compatibility); err == nil ||
		!strings.Contains(err.Error(), "GPU model") {
		t.Fatalf("GPU compatibility error = %v", err)
	}

	compatibility = validPortableCompatibility(1)
	compatibility.Units[0].NVIDIADriverMin = "612.0"
	if err := adapter.verifyPortablePlatforms(context.Background(), request, compatibility); err == nil ||
		!strings.Contains(err.Error(), "older than captured") {
		t.Fatalf("driver compatibility error = %v", err)
	}
}

func TestPortablePlatformCanRelaxCapturedDriverFloor(t *testing.T) {
	request := validRequest(1)
	request.Driver.ID = snapshotdriver.N580
	request.Policy.Compatibility.EnforceCapturedDriverFloor = false
	compatibility := validPortableCompatibility(1)
	if err := (fixtureAdapter(Adapter{Remote: fakeRemote{driver: "580.95.05"}})).verifyPortablePlatforms(
		context.Background(), request, compatibility,
	); err != nil {
		t.Fatal(err)
	}
}

func TestPortablePlatformAlwaysEnforcesSnapshotDriverMinimum(t *testing.T) {
	request := validRequest(1)
	request.Driver.ID = snapshotdriver.N610
	request.Policy.Compatibility.EnforceCapturedDriverFloor = false
	compatibility := validPortableCompatibility(1)
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{driver: "580.95.05"}})
	if err := adapter.verifyPortablePlatforms(context.Background(), request, compatibility); err == nil ||
		!strings.Contains(err.Error(), "snapshot driver n610 minimum 610") {
		t.Fatalf("snapshot driver compatibility error = %v", err)
	}
}

func TestPlacementTCPAddressMapSwapsLogicalRanks(t *testing.T) {
	captured := validRequest(2).Launch
	requested := validRequest(2).Launch
	captured.Units[0].Host = "10.24.11.13"
	captured.Units[1].Host = "10.24.11.17"
	requested.Units[0].Host = "10.24.11.17"
	requested.Units[1].Host = "10.24.11.13"
	mapping, err := placementTCPAddressMap(captured, requested)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(mapping, []string{
		"10.24.11.13=10.24.11.17",
		"10.24.11.17=10.24.11.13",
	}) {
		t.Fatalf("placement TCP address map = %#v", mapping)
	}
}

func TestPlacementTCPAddressMapUsesOrchestratorCommunicationAddresses(t *testing.T) {
	captured := validRequest(1).Launch
	requested := validRequest(1).Launch
	captured.Units[0].Environment = map[string]string{"VLLM_HOST_IP": "10.0.0.1"}
	requested.Units[0].Environment = map[string]string{"VLLM_HOST_IP": "10.0.0.2"}
	mapping, err := placementTCPAddressMap(captured, requested)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(mapping, []string{"10.0.0.1=10.0.0.2"}) {
		t.Fatalf("placement TCP address map = %#v", mapping)
	}
}

func TestPlacementTCPAddressMapAllowsMultipleUnitsPerHost(t *testing.T) {
	captured := validRequest(2).Launch
	requested := validRequest(2).Launch
	for index := range captured.Units {
		captured.Units[index].Host = "10.0.0.1"
		requested.Units[index].Host = "10.0.0.2"
	}
	mapping, err := placementTCPAddressMap(captured, requested)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(mapping, []string{"10.0.0.1=10.0.0.2"}) {
		t.Fatalf("placement TCP address map = %#v", mapping)
	}
}

func TestPlacementTCPAddressMapPreservesIdentityForFreshPortRotation(t *testing.T) {
	captured := validRequest(2).Launch
	requested := validRequest(2).Launch
	captured.Units[0].Host, requested.Units[0].Host = "10.24.11.13", "10.24.11.13"
	captured.Units[1].Host, requested.Units[1].Host = "10.24.11.17", "10.24.11.17"
	mapping, err := placementTCPAddressMap(captured, requested)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(mapping, []string{
		"10.24.11.13=10.24.11.13",
		"10.24.11.17=10.24.11.17",
	}) {
		t.Fatalf("identity placement TCP address map = %#v", mapping)
	}
}

func TestPortableTCPPortShiftRejectsOccupiedCandidateAcrossUnits(t *testing.T) {
	request := validRequest(2)
	images := make(map[string]snapshot.CapsuleImage)
	for _, unit := range request.Launch.Units {
		images[unit.ID] = snapshot.CapsuleImage{
			Unit: unit.ID, Reference: unit.Image, Digest: unit.ImageDigest,
			Root: capsule.Root, Driver: validSnapshotDriverContract().Binding(),
		}
	}
	namespace := "restore-port-admission"
	first := portableTCPPortShift(namespace)
	second := portableTCPPortShiftCandidate(namespace, 1)
	remote := &tcpPortProbeRemote{rejectedShift: first}
	output := &bytes.Buffer{}
	adapter := fixtureAdapter(Adapter{Remote: remote, Output: output})
	selected, err := adapter.selectPortableTCPPortShift(
		context.Background(), request, request.Launch, images,
		[]string{"10.24.11.13=10.24.11.13", "10.24.11.17=10.24.11.17"},
		namespace,
	)
	if err != nil {
		t.Fatal(err)
	}
	if selected != second {
		t.Fatalf("selected TCP port shift = %d, want %d", selected, second)
	}
	if !strings.Contains(output.String(), "after rejecting 1 occupied candidate") {
		t.Fatalf("TCP port admission progress = %q", output.String())
	}
	counts := map[uint16]int{}
	remote.mutex.Lock()
	for _, shift := range remote.shifts {
		counts[shift]++
	}
	remote.mutex.Unlock()
	if counts[first] != 2 || counts[second] != 2 {
		t.Fatalf("TCP port probes by candidate = %#v", counts)
	}
}

func TestPortableTCPPortShiftDoesNotRetryStructuralProbeFailure(t *testing.T) {
	request := validRequest(2)
	images := make(map[string]snapshot.CapsuleImage)
	for _, unit := range request.Launch.Units {
		images[unit.ID] = snapshot.CapsuleImage{
			Unit: unit.ID, Reference: unit.Image, Digest: unit.ImageDigest,
			Root: capsule.Root, Driver: validSnapshotDriverContract().Binding(),
		}
	}
	namespace := "restore-port-structural-failure"
	first := portableTCPPortShift(namespace)
	remote := &tcpPortProbeRemote{fatalShift: first}
	adapter := fixtureAdapter(Adapter{Remote: remote, Output: &bytes.Buffer{}})
	_, err := adapter.selectPortableTCPPortShift(
		context.Background(), request, request.Launch, images,
		[]string{"10.24.11.13=10.24.11.13", "10.24.11.17=10.24.11.17"},
		namespace,
	)
	if err == nil || !strings.Contains(err.Error(), "probe helper is incompatible") {
		t.Fatalf("structural TCP probe error = %v", err)
	}
	remote.mutex.Lock()
	probes := len(remote.shifts)
	remote.mutex.Unlock()
	if probes != 2 {
		t.Fatalf("structural TCP probe was retried: %d calls", probes)
	}
}

func TestPortableHTTPPortPolicy(t *testing.T) {
	captured := validRequest(2).Launch
	requested := validRequest(2).Launch
	for index := range captured.Units {
		captured.Units[index].Command = []string{"vllm", "serve", "model", "--port", "8000"}
		requested.Units[index].Command = []string{"vllm", "serve", "model", "--port=8103"}
	}
	shift, preserve, err := portableHTTPPortPolicy(captured, requested)
	if err != nil || shift != 103 || preserve {
		t.Fatalf("portable HTTP port policy = shift %d, preserve %t, %v", shift, preserve, err)
	}
	shift, preserve, err = portableHTTPPortPolicy(captured, captured)
	if err != nil || shift != 0 || !preserve {
		t.Fatalf("unchanged HTTP port policy = shift %d, preserve %t, %v", shift, preserve, err)
	}
	requested.Units[1].Command = []string{"vllm", "serve", "model", "--port", "8104"}
	if _, _, err := portableHTTPPortPolicy(captured, requested); err == nil ||
		!strings.Contains(err.Error(), "inconsistent") {
		t.Fatalf("inconsistent HTTP port policy error = %v", err)
	}
	captured.Units[0].Command = []string{"vllm", "serve", "model", "--port", "80"}
	requested.Units[0].Command = []string{"vllm", "serve", "model", "--port", "8080"}
	if _, _, err := portableHTTPPortPolicy(captured, requested); err == nil ||
		!strings.Contains(err.Error(), "privileged") {
		t.Fatalf("privileged HTTP port policy error = %v", err)
	}
}

func TestCapsuleTCPImagesPathUsesSnapshotDriverLayout(t *testing.T) {
	n610 := snapshot.CapsuleImage{Root: capsule.Root, Driver: validSnapshotDriverContract().Binding()}
	if path := capsuleTCPImagesPath(n610); path != "/opt/coldsnap/capsule/images" {
		t.Fatalf("n610 TCP images path = %q", path)
	}
	n580Contract, err := snapshotdriver.Lookup(snapshotdriver.N580)
	if err != nil {
		t.Fatal(err)
	}
	n580 := snapshot.CapsuleImage{Root: capsule.Root, Driver: n580Contract.Binding()}
	if path := capsuleTCPImagesPath(n580); path != "/opt/coldsnap/capsule/template/images" {
		t.Fatalf("n580 TCP images path = %q", path)
	}
}

func TestPortableTCPPortShiftMapsRequestedHTTPPort(t *testing.T) {
	configureTestCRIURPC(t)
	request := validRequest(2)
	captured := request.Launch
	captured.Units = slices.Clone(request.Launch.Units)
	images := make(map[string]snapshot.CapsuleImage)
	for index := range request.Launch.Units {
		captured.Units[index].Command = []string{"vllm", "serve", "model", "--port", "8000"}
		request.Launch.Units[index].Command = []string{"vllm", "serve", "model", "--port", "8103"}
		unit := request.Launch.Units[index]
		images[unit.ID] = snapshot.CapsuleImage{
			Unit: unit.ID, Reference: unit.Image, Digest: unit.ImageDigest,
			Root: capsule.Root, Driver: validSnapshotDriverContract().Binding(),
		}
	}
	remote := &tcpPortProbeRemote{}
	adapter := fixtureAdapter(Adapter{Remote: remote, Output: &bytes.Buffer{}})
	selected, err := adapter.selectPortableTCPPortShift(
		context.Background(), request, captured, images,
		[]string{"10.24.11.13=10.24.11.13", "10.24.11.17=10.24.11.17"},
		"fixed-http-port",
	)
	if err != nil || selected != 103 {
		t.Fatalf("selected fixed TCP port shift = %d, %v", selected, err)
	}
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	if !slices.Equal(remote.shifts, []uint16{103, 103}) {
		t.Fatalf("fixed TCP probes = %#v", remote.shifts)
	}
	if slices.Contains(remote.preserved, true) {
		t.Fatalf("changed HTTP port was preserved: %#v", remote.preserved)
	}
	if !slices.Equal(remote.portMaps, []string{"8000=8103", "8000=8103"}) {
		t.Fatalf("fixed TCP port mappings = %#v", remote.portMaps)
	}
}

func TestPortableTCPPortShiftRejectsOccupiedRequestedHTTPPort(t *testing.T) {
	configureTestCRIURPC(t)
	request := validRequest(2)
	captured := request.Launch
	captured.Units = slices.Clone(request.Launch.Units)
	images := make(map[string]snapshot.CapsuleImage)
	for index := range request.Launch.Units {
		captured.Units[index].Command = []string{"vllm", "serve", "model", "--port", "8000"}
		request.Launch.Units[index].Command = []string{"vllm", "serve", "model", "--port", "8103"}
		unit := request.Launch.Units[index]
		images[unit.ID] = snapshot.CapsuleImage{
			Unit: unit.ID, Reference: unit.Image, Digest: unit.ImageDigest,
			Root: capsule.Root, Driver: validSnapshotDriverContract().Binding(),
		}
	}
	remote := &tcpPortProbeRemote{rejectedShift: 103}
	adapter := fixtureAdapter(Adapter{Remote: remote, Output: &bytes.Buffer{}})
	_, err := adapter.selectPortableTCPPortShift(
		context.Background(), request, captured, images,
		[]string{"10.24.11.13=10.24.11.13", "10.24.11.17=10.24.11.17"},
		"occupied-fixed-http-port",
	)
	if err == nil || !strings.Contains(err.Error(), "requested serving-port mapping is unavailable") {
		t.Fatalf("occupied requested HTTP port error = %v", err)
	}
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	if !slices.Equal(remote.shifts, []uint16{103, 103}) {
		t.Fatalf("occupied fixed TCP probes = %#v", remote.shifts)
	}
}

func configureTestCRIURPC(t *testing.T) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "coldsnap-criu-rpc")
	if err := os.WriteFile(path, []byte("test-criu-rpc"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv(criuRPCEnvironment, path)
}

func TestNumericUser(t *testing.T) {
	user, err := numericUser([]byte("1000\n"), []byte("1001\n"))
	if err != nil || user != "1000:1001" {
		t.Fatalf("numericUser = %q, %v", user, err)
	}
	for _, invalid := range [][2]string{{"user", "1001"}, {"1000", "group"}, {"01", "2"}} {
		if _, err := numericUser([]byte(invalid[0]), []byte(invalid[1])); err == nil {
			t.Fatalf("numericUser accepted %#v", invalid)
		}
	}
}

func TestCaptureCommandMustBeRecoveryCapable(t *testing.T) {
	if recoveryCapableCommand("vllm", []string{"vllm", "serve", "model"}) {
		t.Fatal("plain vLLM command was accepted")
	}
	if !recoveryCapableCommand("vllm", validRequest(1).Launch.Units[0].Command) {
		t.Fatal("ColdSnap load format command was rejected")
	}
}

type cacheRemote struct {
	mutex sync.Mutex
	calls [][]string
}

func (remote *cacheRemote) Run(
	_ context.Context, _ string, arguments ...string,
) ([]byte, error) {
	remote.mutex.Lock()
	remote.calls = append(remote.calls, slices.Clone(arguments))
	remote.mutex.Unlock()
	joined := strings.Join(arguments, "\x00")
	if strings.Contains(joined, "docker\x00cp") && strings.Contains(joined, "/root/.triton/cache") {
		return nil, &hostops.RuntimeError{Code: "path_not_found", Message: "cache absent"}
	}
	if len(arguments) > 0 && arguments[0] == "find" {
		return nil, nil
	}
	if slices.Equal(arguments, []string{"id", "-u"}) || slices.Equal(arguments, []string{"id", "-g"}) {
		return []byte("1000\n"), nil
	}
	return []byte("ok\n"), nil
}

func TestNormalizeManagedPathOwnershipUsesManagerIdentity(t *testing.T) {
	request := validRequest(1)
	remote := &cacheRemote{}
	adapter := fixtureAdapter(Adapter{Remote: remote})
	if err := adapter.normalizeManagedPathOwnership(
		context.Background(), request.Launch.Units[0], request.Launch.Units[0].Image,
		"/cache/coldsnap/capture-rank-0",
	); err != nil {
		t.Fatal(err)
	}
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	found := false
	for _, call := range remote.calls {
		if len(call) >= 18 && call[0] == "docker" && call[1] == "run" &&
			slices.Contains(call, "python3") && slices.Contains(call, conditionalOwnershipProgram) &&
			slices.Contains(call, "1000") {
			found = true
		}
		if len(call) > 1 && call[0] == "docker" && call[1] == "exec" {
			t.Fatalf("post-capture assembly used docker exec: %#v", call)
		}
		if slices.Contains(call, "chmod") {
			t.Fatalf("post-capture assembly changed captured mode bits: %#v", call)
		}
	}
	if !found {
		t.Fatalf("ownership sidecar call missing: %#v", remote.calls)
	}
}

func TestNormalizeManagedPathOwnershipRejectsBroadOrAmbiguousBind(t *testing.T) {
	request := validRequest(1)
	for _, path := range []string{"/", "/cache/coldsnap/../other", "/cache/coldsnap:other"} {
		remote := &cacheRemote{}
		adapter := fixtureAdapter(Adapter{Remote: remote})
		if err := adapter.normalizeManagedPathOwnership(
			context.Background(), request.Launch.Units[0], request.Launch.Units[0].Image, path,
		); err == nil || !strings.Contains(err.Error(), "managed writable path is unsafe") {
			t.Fatalf("path %q error = %v", path, err)
		}
		if len(remote.calls) != 0 {
			t.Fatalf("unsafe path %q reached remote transport: %#v", path, remote.calls)
		}
	}
}

func TestNormalizeCapturePathOwnershipIncludesStagedRuntimeCache(t *testing.T) {
	request := validRequest(1)
	request.Policy.Cache.Staged = []snapshot.PreparedCache{{
		Unit: request.Launch.Units[0].ID, Path: "/cache/coldsnap/runtime-cache-staging/unit-0",
	}}
	remote := &cacheRemote{}
	adapter := fixtureAdapter(Adapter{Remote: remote, Output: io.Discard})
	if err := adapter.normalizeCapturePathOwnership(
		context.Background(), request, []string{"/cache/coldsnap/captures/unit-0"},
	); err != nil {
		t.Fatal(err)
	}
	var normalized []string
	for _, call := range remote.calls {
		if len(call) > 10 && call[0] == "docker" && call[1] == "run" {
			normalized = append(normalized, call[10])
		}
	}
	slices.Sort(normalized)
	want := []string{
		"/cache/coldsnap/captures/unit-0:" + containerManagedOwnershipRoot,
		"/cache/coldsnap/runtime-cache-staging/unit-0:" + containerManagedOwnershipRoot,
	}
	if !slices.Equal(normalized, want) {
		t.Fatalf("normalized bind inventory = %#v, want %#v; calls = %#v", normalized, want, remote.calls)
	}
}

func TestMissingWorkloadCopySourceIsNarrow(t *testing.T) {
	if !missingWorkloadCopySource(&hostops.RuntimeError{Code: "path_not_found", Message: "cache absent"}) {
		t.Fatal("Docker missing-source error was not recognized")
	}
	if missingWorkloadCopySource(errors.New("permission denied")) {
		t.Fatal("unrelated Docker copy error was accepted")
	}
	if missingWorkloadCopySource(errors.New("no such file or directory")) {
		t.Fatal("generic filesystem error was accepted")
	}
}

func (*cacheRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (*cacheRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, nil
}

func (*cacheRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func TestStageCacheSeedsCopiesDeclaredExistingDirectories(t *testing.T) {
	request := validRequest(1)
	request.Policy.Cache = snapshot.CachePolicy{
		Seed: true, Paths: []string{snapshot.CanonicalRuntimeCachePath},
	}
	remote := &cacheRemote{}
	adapter := fixtureAdapter(Adapter{Remote: remote, StateRoot: "/var/lib/coldsnap"})
	roots, err := adapter.stageCacheSeeds(context.Background(), request, []string{"capture-rank-0"})
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 1 || roots[0] != "/var/lib/coldsnap/capsule-cache-seeds/generic-capture/units/unit-0" {
		t.Fatalf("roots = %#v", roots)
	}
	copies := 0
	foundExisting := false
	for _, call := range remote.calls {
		if len(call) > 1 && call[0] == "docker" && call[1] == "cp" {
			copies++
			if slices.Contains(call, "capture-rank-0:/var/cache/coldsnap/runtime") {
				foundExisting = true
			}
		}
	}
	if copies != 1 || !foundExisting {
		t.Fatalf("docker copies = %d, calls = %#v", copies, remote.calls)
	}
}

type publicationRemote struct {
	mutex   sync.Mutex
	calls   [][]string
	uploads []string
}

func (remote *publicationRemote) Run(
	_ context.Context, _ string, arguments ...string,
) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	remote.calls = append(remote.calls, slices.Clone(arguments))
	if len(arguments) >= 2 && arguments[0] == "python3" && arguments[1] == "-c" {
		return []byte("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"), nil
	}
	return []byte("ok\n"), nil
}

func (*publicationRemote) RunCombined(context.Context, string, ...string) ([]byte, error) {
	return nil, nil
}

func (*publicationRemote) RunInput(context.Context, string, []byte, ...string) ([]byte, error) {
	return nil, nil
}

func (*publicationRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func (remote *publicationRemote) PublishHuggingFaceFile(
	_ context.Context, host, image, repository, revision, source, destination string,
) error {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	remote.uploads = append(remote.uploads, strings.Join([]string{host, image, repository, revision, source, destination}, " "))
	return nil
}

func (remote *publicationRemote) ResolveHuggingFaceRevision(
	_ context.Context, host, image, repository, revision string,
) (string, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	remote.calls = append(remote.calls, []string{"resolve", host, image, repository, revision})
	return "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", nil
}

func TestPublishModelPayloadsRecordsResolvedHubCommit(t *testing.T) {
	request := validRequest(2)
	request.Policy.Weights.Native = snapshot.NativePolicy{
		Repository: "org/native-packs", Revision: "main",
	}
	remote := &publicationRemote{}
	revision, err := (fixtureAdapter(Adapter{Remote: remote})).publishModelPayloads(
		context.Background(),
		request,
		map[string]string{"worker-0": "/captures/rank0.pack", "worker-1": "/captures/rank1.pack"},
		map[string]snapshot.Object{
			"worker-0": {Owner: snapshot.WorkerOwner("worker-0"), Path: "worker-0.weights", Bytes: 100},
			"worker-1": {Owner: snapshot.WorkerOwner("worker-1"), Path: "worker-1.weights", Bytes: 100},
		},
		[]snapshot.CapsuleImage{
			{Unit: "unit-0", Reference: testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule"},
			{Unit: "unit-1", Reference: testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule"},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if revision != "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" {
		t.Fatalf("revision = %q", revision)
	}
	if len(remote.uploads) != 2 || len(remote.calls) != 1 || remote.calls[0][0] != "resolve" {
		t.Fatalf("publication uploads=%#v calls=%#v", remote.uploads, remote.calls)
	}
}

type nativePublicationRemote struct {
	publicationRemote
	files map[string][]byte
}

func (remote *nativePublicationRemote) Run(
	_ context.Context, host string, arguments ...string,
) ([]byte, error) {
	if result, ok := activationProbeFixture(arguments); ok {
		return result, nil
	}
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	remote.calls = append(remote.calls, slices.Clone(arguments))
	if remote.files == nil {
		remote.files = make(map[string][]byte)
	}
	if len(arguments) > 0 && slices.Contains([]string{"install", "test", "chmod"}, arguments[0]) {
		return nil, nil
	}
	if len(arguments) == 4 && arguments[0] == "mv" && arguments[1] == "-f" {
		data, ok := remote.files[host+"\x00"+arguments[2]]
		if !ok {
			return nil, errors.New("temporary activation file is absent")
		}
		remote.files[host+"\x00"+arguments[3]] = data
		delete(remote.files, host+"\x00"+arguments[2])
		return nil, nil
	}
	if len(arguments) > 0 && arguments[0] == "find" {
		root := strings.TrimSuffix(arguments[1], "/hydration")
		return []byte(root + "/hydration/generation/manifest.json\n"), nil
	}
	if len(arguments) == 2 && arguments[0] == "cat" && strings.HasSuffix(arguments[1], "/manifest.json") {
		worker := "worker-1"
		if strings.Contains(arguments[1], "/unit-0/") {
			worker = "worker-0"
		}
		return []byte(`{"worker_id":"` + worker + `"}`), nil
	}
	if slices.Equal(arguments[:min(len(arguments), 2)], []string{"stat", "-c"}) {
		return []byte("100\n"), nil
	}
	if len(arguments) > 0 && arguments[0] == "sha256sum" {
		if data, ok := remote.files[host+"\x00"+arguments[1]]; ok {
			digest := sha256.Sum256(data)
			return []byte(hex.EncodeToString(digest[:]) + "  " + arguments[1] + "\n"), nil
		}
		return []byte(strings.TrimPrefix(testDigest, "sha256:") + "  " + arguments[1] + "\n"), nil
	}
	if len(arguments) >= 2 && arguments[1] == "payload-verify" {
		return []byte(`{"format":1,"kind":"coldsnap-payload-validation-result","decision":"accept","bytes":100,"sha256":"` + testDigest + `","validation":{"record":"` + arguments[5] + `","provider":"sha256-cache-v1","content_evidence":"cached-full-sha256","reason":"cached_validation","device":1,"inode":2,"size":100,"mtime_ns":3}}`), nil
	}
	if slices.Equal(arguments[:min(len(arguments), 3)], []string{"docker", "image", "inspect"}) {
		return json.Marshal(hostops.ImageInfo{ID: testDigest})
	}
	return nil, fmt.Errorf("unexpected remote command: %v", arguments)
}

func (remote *nativePublicationRemote) RunInput(
	_ context.Context, host string, input []byte, arguments ...string,
) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	if len(arguments) != 2 || arguments[0] != "tee" {
		return nil, fmt.Errorf("unexpected input command: %v", arguments)
	}
	if remote.files == nil {
		remote.files = make(map[string][]byte)
	}
	remote.files[host+"\x00"+arguments[1]] = slices.Clone(input)
	return nil, nil
}

func TestPublishNativePromotesVerifiedCaptureLocalPacks(t *testing.T) {
	configureNativeActivationTools(t)
	request := validRequest(2)
	request.Operation = "publish-native"
	request.Policy.Weights.Native = snapshot.NativePolicy{Repository: "org/native-packs", Revision: "main"}
	directory := t.TempDir()
	request.Artifact = directory + "/local.json"
	request.Output = directory + "/native-published.json"
	artifact := snapshot.Artifact{
		Format: snapshot.ArtifactFormat, Kind: snapshot.ArtifactKind, State: "committed",
		Driver: validSnapshotDriverContract(), Requires: validSnapshotDriverContract().BaseRequirements.Clone(),
		Graph:     validGraphPolicy(request.Launch),
		CaptureID: "generic-capture", RequestSHA256: testDigest, Launch: request.Launch,
		Compatibility: validPortableCompatibility(2),
		Capsule: snapshot.Capsule{
			Images: []snapshot.CapsuleImage{
				{Unit: "unit-0", Reference: testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule", Driver: validSnapshotDriverContract().Binding()},
				{Unit: "unit-1", Reference: testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule", Driver: validSnapshotDriverContract().Binding()},
			},
			Objects: []snapshot.Object{
				{Role: "oci-capsule", Owner: snapshot.UnitOwner("unit-0"), Path: "units/unit-0/capsule.oci", Bytes: 100, SHA256: testDigest},
				{Role: "oci-capsule", Owner: snapshot.UnitOwner("unit-1"), Path: "units/unit-1/capsule.oci", Bytes: 100, SHA256: testDigest},
			},
		},
		Weights: snapshot.WeightProviders{
			Native: &snapshot.NativeProvider{Driver: validSnapshotDriverContract().Binding()},
			ModelPayloads: &snapshot.ModelPayloadProvider{Objects: []snapshot.Object{
				{Role: "model-weight-payload", Owner: snapshot.WorkerOwner("worker-0"), Path: modelPayloadObjectPath(testDigest), Bytes: 100, SHA256: testDigest},
				{Role: "model-weight-payload", Owner: snapshot.WorkerOwner("worker-1"), Path: modelPayloadObjectPath(testDigest), Bytes: 100, SHA256: testDigest},
			}},
			Recovery: snapshot.RecoveryProvider{
				Source: "huggingface-safetensors", ModelID: request.Launch.Model.ID,
				Revision: request.Launch.Model.Revision, Driver: validSnapshotDriverContract().Binding(),
				LoadPath: "coldsnap-replay",
				ReplayPlan: []snapshot.Object{
					{Role: "safetensors-replay-plan", Owner: snapshot.WorkerOwner("worker-0"), Path: "drivers/n610/units/unit-0/hydration/worker-0/manifest.json", Bytes: 10, SHA256: testDigest},
					{Role: "safetensors-replay-plan", Owner: snapshot.WorkerOwner("worker-1"), Path: "drivers/n610/units/unit-1/hydration/worker-1/manifest.json", Bytes: 10, SHA256: testDigest},
				},
			},
		},
		Acceptance: snapshot.ArtifactAcceptance{Accepted: true, Expected: "exact"},
	}
	payload, err := snapshot.Encode(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(request.Artifact, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	remote := &nativePublicationRemote{}
	if err := (fixtureAdapter(Adapter{Remote: remote, StateRoot: "/cache/coldsnap"})).Run(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	published, err := snapshot.ReadArtifact(request.Output)
	if err != nil {
		t.Fatal(err)
	}
	if published.Weights.ModelPayloads.Repository != "org/native-packs" ||
		published.Weights.ModelPayloads.Revision != "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" {
		t.Fatalf("published model payload provider = %#v", published.Weights.ModelPayloads)
	}
	if len(remote.uploads) != 1 {
		t.Fatalf("native uploads = %#v", remote.uploads)
	}
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

const testDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func validLaunch(worldSize int) LaunchSpec {
	topology, err := NewAdapterTopology("vllm.parallel:v1", map[string]any{
		"dimensions": map[string]int{"tensor": worldSize},
	})
	if err != nil {
		panic(err)
	}
	launch := LaunchSpec{
		Engine: "vllm",
		Model:  ModelSource{ID: "Qwen/Qwen3.5-0.8B", Revision: "0123456789abcdef", Source: "huggingface"},
		Execution: ExecutionGraph{
			Adapter:  topology,
			Services: []ServiceDomain{{ID: "model", Role: "vllm:serve"}},
			Groups:   []ProcessGroup{{ID: "world", Kind: "torch:world", Service: "model"}},
		},
	}
	for rank := 0; rank < worldSize; rank++ {
		unitID := "unit-" + string(rune('a'+rank))
		workerID := "worker-" + string(rune('a'+rank))
		launch.Units = append(launch.Units, LaunchUnit{
			ID: unitID, Index: rank, Host: "node-" + string(rune('a'+rank)), Devices: []string{"0"},
			Image: "registry/vllm@" + testDigest, ImageDigest: testDigest,
			Command:     []string{"vllm", "serve", "Qwen/Qwen3.5-0.8B"},
			Environment: map[string]string{"NCCL_DEBUG": "WARN"},
		})
		launch.Execution.Workers = append(launch.Execution.Workers, Worker{
			ID: workerID, Unit: unitID, Service: "model", ProcessSlot: 0, DeviceSlots: []int{0},
		})
		launch.Execution.Services[0].Workers = append(launch.Execution.Services[0].Workers, workerID)
		launch.Execution.Groups[0].Members = append(launch.Execution.Groups[0].Members, workerID)
	}
	return launch
}

func validCompatibility(worldSize int) ArtifactCompatibility {
	compatibility := ArtifactCompatibility{Policy: PortabilityPolicy}
	for rank := 0; rank < worldSize; rank++ {
		compatibility.Units = append(compatibility.Units, UnitPlatformCompatibility{
			Unit: "unit-" + string(rune('a'+rank)), Architecture: "aarch64", Kernel: "6.17.0-1029-nvidia",
			NVIDIADriverCaptured: "610.43.02", NVIDIADriverMin: "610.43.02", CUDAUserspace: "13.0",
			Devices: []DeviceCompatibility{{Slot: 0, GPUName: "NVIDIA GB10", ComputeCapability: "12.1"}},
		})
	}
	return compatibility
}

func validRequest() Request {
	return Request{
		Format: RequestFormat, Kind: RequestKind, Operation: "restore", ID: "qwen35-tp2-restore",
		Driver:   snapshotdriver.Selection{ID: snapshotdriver.N610},
		Artifact: "/artifacts/qwen35/manifest.json", Launch: validLaunch(2),
		Policy: SnapshotPolicy{
			Process: ProcessPolicy{Backend: "cuda-criu", KVDiscard: true, AsyncGraphs: true},
			Weights: WeightPolicy{
				Mode: "auto", Native: NativePolicy{Repository: "org/qwen35-native", Revision: "abc123"},
				Recovery: RecoveryPolicy{
					Enabled: true, Source: "huggingface-safetensors", LoaderBackend: "direct",
				},
			},
			Cache: CachePolicy{
				Seed: true,
				Paths: []string{
					"/root/.cache/flashinfer", "/root/.triton/cache", "/tmp/torchinductor_root",
				},
			},
			Capsule: CapsulePolicy{Repository: "registry.example/coldsnap/qwen35"},
			Compatibility: CompatibilityPolicy{
				EnforceCapturedDriverFloor: true,
				Kernel:                     KernelCompatibilityCapability,
			},
		},
		Validation: ValidationPolicy{HealthPath: "/health", Prompt: "hello", Expected: "world"},
	}
}

func TestRequestRequiresCapsuleRepositoryForPublish(t *testing.T) {
	request := validRequest()
	request.Operation = "publish"
	request.Output = "/artifacts/qwen35/published.json"
	request.Policy.Capsule = CapsulePolicy{}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "capsule repository") {
		t.Fatalf("request validation error = %v", err)
	}
}

func TestRequestRequiresExplicitSnapshotDriver(t *testing.T) {
	request := validRequest()
	request.Driver = snapshotdriver.Selection{}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "required") {
		t.Fatalf("expected explicit snapshot driver rejection, got %v", err)
	}

	request.Driver.ID = "auto"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("expected auto snapshot driver rejection, got %v", err)
	}
}

func TestLifecycleRequestsRequireExactWorkloadAndArtifact(t *testing.T) {
	for _, operation := range []string{"sleep", "wake", "status"} {
		request := validRequest()
		request.Operation = operation
		request.Workload = WorkloadIdentity{
			ClusterID: "sparkrun_deadbeef_01234567", IntentID: "deadbeef",
			Recipe: "qwen", Runtime: "vllm-distributed", Model: request.Launch.Model.ID,
		}
		if err := request.Validate(); err != nil {
			t.Fatalf("%s lifecycle request: %v", operation, err)
		}
		request.Workload.ClusterID = ""
		if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "workload.cluster_id") {
			t.Fatalf("%s workload validation error = %v", operation, err)
		}
	}
}

func TestWarmActivationIsN610RestoreOnly(t *testing.T) {
	request := validRequest()
	request.Lifecycle.ActivationState = "warm"
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
	request.Driver.ID = snapshotdriver.N580
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "n610") {
		t.Fatalf("n580 warm validation error = %v", err)
	}
	request.Driver.ID = snapshotdriver.N610
	request.Operation = "sleep"
	request.Workload = WorkloadIdentity{
		ClusterID: "sparkrun_deadbeef_01234567", IntentID: "deadbeef",
		Recipe: "qwen", Runtime: "vllm-distributed", Model: request.Launch.Model.ID,
	}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "only for restore") {
		t.Fatalf("sleep warm validation error = %v", err)
	}
}

func TestWarmActivationRejectsRequiredReadThroughMaterialization(t *testing.T) {
	request := validRequest()
	request.Lifecycle.ActivationState = "warm"
	request.Policy.Weights.Native.Materialize = "required"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "before hydration") {
		t.Fatalf("warm materialization validation error = %v", err)
	}
}

func TestPublishRequiresArtifactAndOutput(t *testing.T) {
	request := validRequest()
	request.Operation = "publish"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "artifact and output") {
		t.Fatalf("request validation error = %v", err)
	}
}

func TestPublishNativeRequiresArtifactOutputAndRepository(t *testing.T) {
	request := validRequest()
	request.Operation = "publish-native"
	request.Output = "/artifacts/qwen35/native-published.json"
	request.Policy.Weights.Native = NativePolicy{}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "model payload repository") {
		t.Fatalf("publish-native repository validation error = %v", err)
	}
	request.Policy.Weights.Native = NativePolicy{Repository: "org/qwen35-native", Revision: "main"}
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestRequestRejectsTaggedCapsuleRepository(t *testing.T) {
	request := validRequest()
	request.Policy.Capsule.Repository = "registry.example/coldsnap/qwen:latest"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "capsule repository") {
		t.Fatalf("tagged capsule repository validation error = %v", err)
	}
}

func TestRequestSupportsGenericTopology(t *testing.T) {
	request := validRequest()
	request.Launch = validLaunch(4)
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestNativeMaterializationModeIsTyped(t *testing.T) {
	request := validRequest()
	request.Policy.Weights.Native.Materialize = "async"
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
	request.Policy.Weights.Native.Materialize = "required"
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
	request.Policy.Weights.Native.Materialize = "eventually"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "materialization mode") {
		t.Fatalf("materialization validation error = %v", err)
	}
}

func TestExecutionGraphRepresentsTwoUnitsWithFourWorkers(t *testing.T) {
	request := validRequest()
	request.Launch = multiWorkerLaunch(t, 2, 2)
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
	if len(request.Launch.Units) != 2 || len(request.Launch.Execution.Workers) != 4 {
		t.Fatalf("launch topology = %#v", request.Launch)
	}
	if workers := request.Launch.Execution.UnitWorkers("unit-a"); len(workers) != 2 || workers[0].ID != "worker-0" || workers[1].ID != "worker-1" {
		t.Fatalf("unit-a workers = %#v", workers)
	}
}

func TestExecutionGraphRepresentsSingleUnitTP8(t *testing.T) {
	request := validRequest()
	request.Launch = multiWorkerLaunch(t, 1, 8)
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestExecutionGraphCarriesIndependentParallelRankNamespaces(t *testing.T) {
	request := validRequest()
	request.Launch = multiWorkerLaunch(t, 2, 2)
	request.Launch.Execution.Groups = append(request.Launch.Execution.Groups,
		ProcessGroup{ID: "pipeline-0", Kind: "vllm:pipeline", Service: "model", Members: []string{"worker-0", "worker-2"}},
		ProcessGroup{ID: "expert-0", Kind: "vllm:expert", Service: "model", Members: []string{"worker-1", "worker-3"}},
		ProcessGroup{ID: "decode-context-0", Kind: "vllm:decode-context", Service: "model", Members: []string{"worker-0", "worker-1"}},
	)
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestExecutionGraphDoesNotInferCartesianParallelism(t *testing.T) {
	request := validRequest()
	request.Launch = multiWorkerLaunch(t, 2, 2)
	topology, err := NewAdapterTopology("vllm.parallel:v1", map[string]any{
		"dimensions": map[string]int{
			"tensor": 4, "pipeline": 2, "data": 3, "expert": 2,
		},
		"note": "group membership, not dimension multiplication, defines workers",
	})
	if err != nil {
		t.Fatal(err)
	}
	request.Launch.Execution.Adapter = topology
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
	if got := len(request.Launch.Execution.Workers); got != 4 {
		t.Fatalf("worker count = %d, want observed count 4", got)
	}
}

func TestExecutionGraphRejectsAdapterTopologyDigestMismatch(t *testing.T) {
	request := validRequest()
	request.Launch.Execution.Adapter.Payload = json.RawMessage(`{"dimensions":{"tensor":8}}`)
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("adapter topology validation error = %v", err)
	}
}

func TestExecutionGraphRejectsSharedDeviceSlots(t *testing.T) {
	request := validRequest()
	request.Launch = multiWorkerLaunch(t, 1, 2)
	request.Launch.Execution.Workers[1].DeviceSlots = []int{0}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "shared device slot") {
		t.Fatalf("shared device validation error = %v", err)
	}
}

func TestExecutionGraphRequiresOneWorldRankPerWorker(t *testing.T) {
	request := validRequest()
	request.Launch.Execution.Groups[0].Kind = "torch:tensor"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "exactly one engine world group") {
		t.Fatalf("missing world rank validation error = %v", err)
	}

	request = validRequest()
	duplicate := request.Launch.Execution.Groups[0]
	duplicate.ID = "world-duplicate"
	duplicate.Kind = "vllm:world"
	request.Launch.Execution.Groups = append(request.Launch.Execution.Groups, duplicate)
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "exactly one engine world group") {
		t.Fatalf("duplicate world rank validation error = %v", err)
	}
}

func multiWorkerLaunch(t *testing.T, unitCount, workersPerUnit int) LaunchSpec {
	t.Helper()
	topology, err := NewAdapterTopology("vllm.parallel:v1", map[string]any{
		"dimensions": map[string]int{"tensor": unitCount * workersPerUnit},
	})
	if err != nil {
		t.Fatal(err)
	}
	launch := LaunchSpec{
		Engine: "vllm",
		Model:  ModelSource{ID: "Qwen/Qwen3.5-0.8B", Revision: "0123456789abcdef", Source: "huggingface"},
		Execution: ExecutionGraph{
			Adapter:  topology,
			Services: []ServiceDomain{{ID: "model", Role: "vllm:serve"}},
			Groups:   []ProcessGroup{{ID: "world", Kind: "torch:world", Service: "model"}},
		},
	}
	workerIndex := 0
	for unitIndex := 0; unitIndex < unitCount; unitIndex++ {
		unitID := "unit-" + string(rune('a'+unitIndex))
		unit := LaunchUnit{
			ID: unitID, Index: unitIndex, Host: "node-" + string(rune('a'+unitIndex)),
			Image: "registry/vllm@" + testDigest, ImageDigest: testDigest,
			Command: []string{"vllm", "serve", "Qwen/Qwen3.5-0.8B"},
		}
		for processSlot := 0; processSlot < workersPerUnit; processSlot++ {
			unit.Devices = append(unit.Devices, string(rune('0'+processSlot)))
			workerID := "worker-" + string(rune('0'+workerIndex))
			launch.Execution.Workers = append(launch.Execution.Workers, Worker{
				ID: workerID, Unit: unitID, Service: "model", ProcessSlot: processSlot,
				DeviceSlots: []int{processSlot},
			})
			launch.Execution.Services[0].Workers = append(launch.Execution.Services[0].Workers, workerID)
			launch.Execution.Groups[0].Members = append(launch.Execution.Groups[0].Members, workerID)
			workerIndex++
		}
		launch.Units = append(launch.Units, unit)
	}
	return launch
}

func TestRequestDoesNotRequireB12xEnvironment(t *testing.T) {
	request := validRequest()
	request.Launch.Units[0].Environment = map[string]string{"A_GENERIC_ENGINE_FLAG": "1"}
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestRequestRejectsEnforceEagerWithAsyncGraphs(t *testing.T) {
	request := validRequest()
	request.Launch.Units[0].Command = []string{"bash", "-c", "vllm serve model --enforce-eager"}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "asynchronous CUDA graphs") {
		t.Fatalf("eager/async-graph validation error = %v", err)
	}
	request.Policy.Process.AsyncGraphs = false
	if err := request.Validate(); err != nil {
		t.Fatalf("eager launch without async graphs rejected: %v", err)
	}
}

func TestRequestAcceptsOptionalOrchestratorWorkloadIdentity(t *testing.T) {
	request := validRequest()
	request.Workload = WorkloadIdentity{
		ClusterID: "sparkrun_deadbeef_01234567", IntentID: "deadbeef",
		Recipe: "catalog/qwen", Runtime: "vllm-distributed", Model: request.Launch.Model.ID,
		ServedModelName: "qwen", LogPath: "/tmp/sparkrun_serve.log",
	}
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
	request.Workload.ClusterID = ""
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "cluster_id") {
		t.Fatalf("partial workload identity error = %v", err)
	}
	request.Workload = WorkloadIdentity{
		ClusterID: "sparkrun_deadbeef_01234567", IntentID: "deadbeef",
		Recipe: "catalog/qwen", Runtime: "vllm-distributed", Model: request.Launch.Model.ID,
		LogPath: "tmp/serve.log",
	}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "log_path") {
		t.Fatalf("relative workload log path error = %v", err)
	}
}

func TestLaunchAcceptsLocalImageIDAndRejectsUnpinnedImageAndDuplicateDevice(t *testing.T) {
	request := validRequest()
	request.Launch.Units[0].Image = "registry/vllm:latest"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "not pinned") {
		t.Fatalf("mutable image validation error = %v", err)
	}
	request = validRequest()
	request.Launch.Units[0].Image = testDigest
	if err := request.Validate(); err != nil {
		t.Fatalf("local content-addressed image validation error = %v", err)
	}
	request = validRequest()
	request.Launch.Units[0].Image = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "not pinned") {
		t.Fatalf("mismatched local image validation error = %v", err)
	}
	request = validRequest()
	request.Launch.Units[1].Host = request.Launch.Units[0].Host
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "duplicated") {
		t.Fatalf("duplicate device validation error = %v", err)
	}
}

func TestRequestRequiresCompleteStagedNativeInventory(t *testing.T) {
	request := validRequest()
	request.Policy.Weights.Native.Staged = []PreparedPayload{{
		Worker: "worker-a", Path: "/cache/native/rank0.pack", Bytes: 100, SHA256: testDigest,
	}}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "every launch worker") {
		t.Fatalf("request validation error = %v", err)
	}
}

func TestRequestRejectsInvalidStagedNativeValidation(t *testing.T) {
	request := validRequest()
	request.Policy.Weights.Native.Staged = []PreparedPayload{
		{
			Worker: "worker-a", Path: "/cache/native/rank0.pack", Bytes: 100, SHA256: testDigest,
			Validation: &PreparedPayloadValidation{
				Record:   "/cache/native/rank0.pack.coldsnap-validation.json",
				Provider: "sha256-cache-v1", ContentEvidence: "cached-full-sha256",
				Device: 1, Inode: 2, Size: 99, MTimeNS: 3, CTimeNS: 4,
			},
		},
		{
			Worker: "worker-b", Path: "/cache/native/rank1.pack", Bytes: 100, SHA256: testDigest,
		},
	}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "validation evidence") {
		t.Fatalf("request validation error = %v", err)
	}
}

func TestDecodeRequestAcceptsPayloadValidationDiagnostics(t *testing.T) {
	request := validRequest()
	request.Policy.Weights.Native.Staged = []PreparedPayload{
		{
			Worker: "worker-a", Path: "/cache/native/rank0.pack", Bytes: 100, SHA256: testDigest,
			Validation: &PreparedPayloadValidation{
				Record:   "/cache/native/rank0.pack.coldsnap-validation.json",
				Provider: "sha256-cache-v1", ContentEvidence: "full-sha256",
				Device: 1, Inode: 2, Size: 100, MTimeNS: 3, CTimeNS: 4,
				UID: 1000, GID: 1001, Mode: 0o400, BytesHashed: 100,
			},
		},
		{
			Worker: "worker-b", Path: "/cache/native/rank1.pack", Bytes: 100, SHA256: testDigest,
		},
	}
	payload, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeRequest(bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	validation := decoded.Policy.Weights.Native.Staged[0].Validation
	if validation == nil || validation.UID != 1000 || validation.GID != 1001 || validation.Mode != 0o400 {
		t.Fatalf("payload validation diagnostics = %#v", validation)
	}
}

func TestRequestRejectsUnsupportedRecoveryLoaderBackend(t *testing.T) {
	request := validRequest()
	request.Policy.Weights.Recovery.LoaderBackend = "swizzler"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "recovery loader backend") {
		t.Fatalf("request validation error = %v", err)
	}
}

func TestRequestRejectsUnsafeOrMountedCacheSeedPaths(t *testing.T) {
	request := validRequest()
	request.Policy.Cache.Paths = []string{"/root/.cache/huggingface"}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "cache path inventory") {
		t.Fatalf("unsafe cache path validation error = %v", err)
	}

	request = validRequest()
	request.Launch.Units[0].Mounts = []Mount{{
		Source: "/var/cache/flashinfer", Target: "/root/.cache/flashinfer",
	}}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "overlaps launch mount") {
		t.Fatalf("mounted cache path validation error = %v", err)
	}
}

func TestRequestRejectsOverlappingCacheSeedPaths(t *testing.T) {
	request := validRequest()
	request.Policy.Cache.Paths = []string{"/root/.cache/flashinfer", "/root/.cache/flashinfer/generated"}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "must not overlap") {
		t.Fatalf("overlapping cache path validation error = %v", err)
	}
}

func TestCaptureAcceptsCompleteStagedRuntimeCacheInventory(t *testing.T) {
	request := validRequest()
	request.Operation = "capture"
	request.Output = "/artifacts/qwen35/capture.json"
	request.Artifact = ""
	request.Policy.Cache = CachePolicy{
		Seed:  true,
		Paths: []string{CanonicalRuntimeCachePath},
		Staged: []PreparedCache{
			{Unit: "unit-a", Path: "/var/cache/sparkrun/coldsnap/request/rank0"},
			{Unit: "unit-b", Path: "/var/cache/sparkrun/coldsnap/request/rank1"},
		},
	}
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestRequestRejectsIncompleteOrRestoreStagedRuntimeCaches(t *testing.T) {
	request := validRequest()
	request.Operation = "capture"
	request.Output = "/artifacts/qwen35/capture.json"
	request.Artifact = ""
	request.Policy.Cache = CachePolicy{
		Seed: true, Paths: []string{CanonicalRuntimeCachePath},
		Staged: []PreparedCache{{Unit: "unit-a", Path: "/var/cache/sparkrun/coldsnap/request/rank0"}},
	}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "every launch unit") {
		t.Fatalf("incomplete staged cache error = %v", err)
	}

	request.Policy.Cache.Paths = []string{CanonicalRuntimeCachePath + "/vllm"}
	request.Policy.Cache.Staged = []PreparedCache{
		{Unit: "unit-a", Path: "/var/cache/sparkrun/coldsnap/request/rank0"},
		{Unit: "unit-b", Path: "/var/cache/sparkrun/coldsnap/request/rank1"},
	}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "canonical runtime cache") {
		t.Fatalf("nested cache inventory error = %v", err)
	}

	request = validRequest()
	request.Policy.Cache = CachePolicy{
		Seed: true, Paths: []string{CanonicalRuntimeCachePath},
		Staged: []PreparedCache{
			{Unit: "unit-a", Path: "/var/cache/sparkrun/coldsnap/request/rank0"},
			{Unit: "unit-b", Path: "/var/cache/sparkrun/coldsnap/request/rank1"},
		},
	}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "only for capture") {
		t.Fatalf("restore staged cache error = %v", err)
	}
}

func TestDecodeRequestRejectsUnknownFields(t *testing.T) {
	_, err := DecodeRequest(strings.NewReader(`{"format":1,"kind":"coldsnap-operation-request","unknown":1}`))
	if err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("DecodeRequest error = %v", err)
	}
}

func TestDecodeRequestAppliesCanonicalPolicyDefaults(t *testing.T) {
	request := validRequest()
	payload, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(payload, &document); err != nil {
		t.Fatal(err)
	}
	delete(document, "format")
	delete(document, "kind")
	delete(document, "policy")
	delete(document, "validation")
	payload, err = json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}

	decoded, err := DecodeRequest(strings.NewReader(string(payload)))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Format != RequestFormat || decoded.Kind != RequestKind ||
		decoded.Policy.Process.Backend != "cuda-criu" ||
		!decoded.Policy.Process.KVDiscard || !decoded.Policy.Process.AsyncGraphs ||
		decoded.Policy.Process.GraphPolicy != GraphPreserveNCCLExec ||
		decoded.Policy.Weights.Mode != "auto" ||
		decoded.Policy.Weights.Native.Materialize != "" ||
		!decoded.Policy.Weights.Recovery.Enabled ||
		decoded.Policy.Weights.Recovery.Source != "huggingface-safetensors" ||
		decoded.Policy.Weights.Recovery.LoaderBackend != "auto" ||
		decoded.Policy.Compatibility.Kernel != KernelCompatibilityCapability ||
		!decoded.Policy.Cache.Seed ||
		len(decoded.Policy.Cache.Paths) != 1 ||
		decoded.Policy.Cache.Paths[0] != CanonicalRuntimeCachePath ||
		decoded.Validation.HealthPath != "/health" ||
		decoded.Validation.Prompt != "Reply with exactly: coldsnap-cuda-snapshot-ok" ||
		decoded.Validation.Expected != "coldsnap-cuda-snapshot-ok" {
		t.Fatalf("decoded defaults = %#v", decoded)
	}
}

func TestSnapshotPolicyDefaultsAreSelectedAndClonedPerDriver(t *testing.T) {
	n580, err := DefaultSnapshotPolicyForDriver(snapshotdriver.N580)
	if err != nil {
		t.Fatal(err)
	}
	if n580.Process.ArtifactScope != ArtifactScopePortable {
		t.Fatalf("n580 artifact scope = %q, want portable", n580.Process.ArtifactScope)
	}
	n610, err := DefaultSnapshotPolicyForDriver(snapshotdriver.N610)
	if err != nil {
		t.Fatal(err)
	}
	if n580.Process.Backend != "cuda-criu" || n610.Process.Backend != "cuda-criu" ||
		!n580.Process.KVDiscard || !n610.Process.AsyncGraphs ||
		n580.Process.GraphPolicy != GraphRecreateFromPlan ||
		n610.Process.GraphPolicy != GraphPreserveNCCLExec ||
		n580.Compatibility.Kernel != KernelCompatibilityCapability ||
		n610.Compatibility.Kernel != KernelCompatibilityCapability ||
		!n580.Compatibility.EnforceCapturedDriverFloor || !n610.Compatibility.EnforceCapturedDriverFloor {
		t.Fatalf("driver defaults = n580 %#v, n610 %#v", n580, n610)
	}
	if n580.Weights.Mode != "auto" || n610.Weights.Mode != "auto" {
		t.Fatalf("driver weight defaults = n580 %q, n610 %q", n580.Weights.Mode, n610.Weights.Mode)
	}
	n580.Cache.Paths[0] = "/changed"
	second, err := DefaultSnapshotPolicyForDriver(snapshotdriver.N580)
	if err != nil {
		t.Fatal(err)
	}
	if second.Cache.Paths[0] != CanonicalRuntimeCachePath {
		t.Fatalf("driver defaults share mutable cache paths: %#v", second.Cache.Paths)
	}
	if _, err := DefaultSnapshotPolicyForDriver("future"); err == nil {
		t.Fatal("unknown snapshot driver defaults were accepted")
	}
}

func TestDecodeRequestFillsSparsePolicyFromSelectedDriver(t *testing.T) {
	request := validRequest()
	request.Driver.ID = snapshotdriver.N580
	payload, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(payload, &document); err != nil {
		t.Fatal(err)
	}
	document["policy"] = map[string]any{
		"weights": map[string]any{
			"native": map[string]any{},
		},
		"cache": map[string]any{
			"seed":  true,
			"paths": []string{CanonicalRuntimeCachePath},
		},
		"capsule": map[string]any{},
	}
	delete(document, "validation")
	payload, err = json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}

	decoded, err := DecodeRequest(bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Driver.ID != snapshotdriver.N580 ||
		decoded.Policy.Process.Backend != "cuda-criu" ||
		!decoded.Policy.Process.KVDiscard || !decoded.Policy.Process.AsyncGraphs ||
		decoded.Policy.Weights.Mode != "auto" ||
		decoded.Policy.Weights.Native.Materialize != "" ||
		!decoded.Policy.Weights.Recovery.Enabled ||
		decoded.Policy.Weights.Recovery.LoaderBackend != "auto" ||
		decoded.Policy.Compatibility.Kernel != KernelCompatibilityCapability ||
		!decoded.Policy.Compatibility.EnforceCapturedDriverFloor ||
		decoded.Validation.Expected != "coldsnap-cuda-snapshot-ok" {
		t.Fatalf("decoded sparse request = %#v", decoded)
	}
}

func TestRequestAcceptsOptInExactKernelCompatibility(t *testing.T) {
	request := validRequest()
	request.Policy.Compatibility.Kernel = KernelCompatibilityExact
	payload, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeRequest(bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Policy.Compatibility.Kernel != KernelCompatibilityExact {
		t.Fatalf("kernel compatibility = %q", decoded.Policy.Compatibility.Kernel)
	}
	request.Policy.Compatibility.Kernel = "uname-prefix"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "kernel compatibility") {
		t.Fatalf("unsupported kernel compatibility error = %v", err)
	}
}

func TestDecodeRequestCanRelaxCapturedDriverFloor(t *testing.T) {
	request := validRequest()
	request.Policy.Compatibility.EnforceCapturedDriverFloor = false
	payload, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeRequest(bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Policy.Compatibility.EnforceCapturedDriverFloor {
		t.Fatal("explicit captured-driver floor relaxation was not preserved")
	}
}

func TestDecodeRequestPreservesExplicitFalsePolicyOverrides(t *testing.T) {
	request := validRequest()
	payload, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(payload, &document); err != nil {
		t.Fatal(err)
	}
	document["policy"] = map[string]any{
		"process": map[string]any{"async_graphs": false},
		"cache":   map[string]any{"seed": false},
	}
	delete(document, "validation")
	payload, err = json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}

	decoded, err := DecodeRequest(strings.NewReader(string(payload)))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Policy.Process.AsyncGraphs || !decoded.Policy.Process.KVDiscard ||
		decoded.Policy.Process.Backend != "cuda-criu" || decoded.Policy.Cache.Seed ||
		len(decoded.Policy.Cache.Paths) != 0 {
		t.Fatalf("decoded overrides = %#v", decoded.Policy)
	}
}

func TestRequestSHAIsStable(t *testing.T) {
	request := validRequest()
	first, err := RequestSHA256(request)
	if err != nil {
		t.Fatal(err)
	}
	second, err := RequestSHA256(request)
	if err != nil {
		t.Fatal(err)
	}
	if first != second || !strings.HasPrefix(first, "sha256:") {
		t.Fatalf("unstable request digest: %q %q", first, second)
	}
}

func TestDistributedGraphPolicyRecordsConservativeDowngrade(t *testing.T) {
	request := validRequest()
	request.Policy.Process.GraphPolicy = GraphPreserveExec
	record, err := NewGraphPolicyRecord(request, nil)
	if err != nil {
		t.Fatal(err)
	}
	if record.Requested != GraphPreserveExec || record.Effective != GraphRecreateFromPlan ||
		record.Decision != "provider_reconstructs_communicator" || record.Audit.NCCLCommunicators != 1 ||
		!record.Audit.UnknownDependencies {
		t.Fatalf("graph policy record = %#v", record)
	}
	request.Launch.Engine = "sglang"
	record, err = NewGraphPolicyRecord(request, nil)
	if err != nil {
		t.Fatal(err)
	}
	if record.Recipe.Engine != "sglang" || !strings.HasPrefix(record.Recipe.PlanSHA256, "sha256:") {
		t.Fatalf("SGLang graph recipe = %#v", record.Recipe)
	}
}

func TestSingleWorkerGraphPreservationRemainsFailClosed(t *testing.T) {
	request := validRequest()
	request.Launch.Units = request.Launch.Units[:1]
	request.Launch.Execution.Workers = request.Launch.Execution.Workers[:1]
	request.Launch.Execution.Groups[0].Members = request.Launch.Execution.Groups[0].Members[:1]
	request.Policy.Process.GraphPolicy = GraphPreserveExec
	record, err := NewGraphPolicyRecord(request, nil)
	if err != nil {
		t.Fatal(err)
	}
	if record.Effective != GraphRecreateFromPlan ||
		record.Decision != graphDecisionPreserveUnqualified {
		t.Fatalf("single-worker graph policy record = %#v", record)
	}
}

func TestDistributedRecreateGraphPolicyStillRecordsUnknownDependencies(t *testing.T) {
	request := validRequest()
	request.Policy.Process.GraphPolicy = GraphRecreateFromPlan
	record, err := NewGraphPolicyRecord(request, nil)
	if err != nil {
		t.Fatal(err)
	}
	if record.Effective != GraphRecreateFromPlan || record.Audit.NCCLCommunicators != 1 ||
		!record.Audit.UnknownDependencies {
		t.Fatalf("graph policy record = %#v", record)
	}
}

func TestPreserveNCCLGraphPolicyUsesInPlaceProvider(t *testing.T) {
	request := validRequest()
	request.Policy.Process.GraphPolicy = GraphPreserveNCCLExec
	record, err := NewGraphPolicyRecord(request, nil)
	if err != nil {
		t.Fatal(err)
	}
	if record.Effective != GraphPreserveNCCLExec ||
		record.Decision != graphDecisionNCCLInPlaceQualified ||
		record.Audit.UnknownDependencies || record.Audit.NCCLCommunicators != 1 ||
		record.Recipe.Mode != "engine-native-retained" {
		t.Fatalf("in-place graph policy record = %#v", record)
	}
}

func TestPreserveNCCLGraphPolicyRecreatesWithoutACommunicator(t *testing.T) {
	request := validRequest()
	request.Policy.Process.GraphPolicy = GraphPreserveNCCLExec
	request.Launch.Execution.Workers = request.Launch.Execution.Workers[:1]
	record, err := NewGraphPolicyRecord(request, nil)
	if err != nil {
		t.Fatal(err)
	}
	if record.Requested != GraphPreserveNCCLExec ||
		record.Effective != GraphRecreateFromPlan ||
		record.Decision != graphDecisionPreserveUnqualified ||
		!record.Audit.UnknownDependencies || record.Audit.NCCLCommunicators != 0 ||
		record.Recipe.Mode != "engine-native-async" {
		t.Fatalf("NCCL-free graph policy record = %#v", record)
	}
}

func TestGraphRecipeDigestRejectsTampering(t *testing.T) {
	artifact := validArtifact()
	artifact.Graph.Audit.RegisteredWindows++
	if err := artifact.Validate(); err == nil || !strings.Contains(err.Error(), "digest") {
		t.Fatalf("tampered graph recipe validation error = %v", err)
	}
}

func TestGraphPolicyDecisionRejectsInconsistentAudit(t *testing.T) {
	artifact := validArtifact()
	artifact.Graph.Decision = graphDecisionReconstructsNCCL
	if err := artifact.Validate(); err == nil || !strings.Contains(err.Error(), "decision") {
		t.Fatalf("inconsistent graph decision validation error = %v", err)
	}
}

func TestArtifactRejectsUnsafeObjectAndCapsulePaths(t *testing.T) {
	artifact := validArtifact()
	artifact.Capsule.Objects[0].Path = "../rank0/criu.pack"
	if err := artifact.Validate(); err == nil || !strings.Contains(err.Error(), "object record") {
		t.Fatalf("traversal validation error = %v", err)
	}
	artifact = validArtifact()
	artifact.Capsule.Images[0].Root = "/opt/coldsnap/../capsule"
	if err := artifact.Validate(); err == nil || !strings.Contains(err.Error(), "capsule image") {
		t.Fatalf("capsule root validation error = %v", err)
	}
}

func TestArtifactAcceptsDigestBoundUnitRuntimeProvider(t *testing.T) {
	artifact := validArtifact()
	binding, err := NewRuntimeProviderBinding(
		UnitOwner("unit-a"), "collective:nccl", "nccl:provider-v1",
		map[string]any{"provider_id": "nccl-2.31.2", "revision": "2.31.2-1"},
	)
	if err != nil {
		t.Fatal(err)
	}
	artifact.Runtime.Bindings = []RuntimeProviderBinding{binding}
	if err := artifact.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestArtifactCarriesEngineOwnedShapeCalibrationCoverage(t *testing.T) {
	artifact := validArtifact()
	evidence := func(engine string) json.RawMessage {
		return json.RawMessage(fmt.Sprintf(`{
			"format":1,
			"kind":"coldsnap-shape-calibration-coverage",
			"engine":%q,
			"planned_shapes":3,
			"warmed_shapes":3,
			"seconds":1.25,
			"details":[{
				"kind":"coldsnap-shape-calibration",
				"engine":%q,
				"planned_shapes":3,
				"warmed_shapes":3,
				"modes":[{"mode":"decode","planned_shapes":3}],
				"toolchain":{"vllm":"test","cuda_architecture":"sm_121"},
				"cache_root":"/var/cache/coldsnap/runtime"
			}]
		}`, engine, engine))
	}
	artifact.ShapeCalibration = &ShapeCalibrationCoverage{
		Policy: "engine-owned-v1",
		Units: []UnitShapeCalibration{
			{Unit: "unit-a", Engine: "vllm", PlannedShapes: 3, WarmedShapes: 3, Seconds: 1.25, Evidence: evidence("vllm")},
			{Unit: "unit-b", Engine: "vllm", PlannedShapes: 3, WarmedShapes: 3, Seconds: 1.25, Evidence: evidence("vllm")},
		},
	}
	graph, err := NewGraphPolicyRecord(Request{
		Launch: artifact.Launch, Policy: newDefaultSnapshotPolicy(),
	}, artifact.ShapeCalibration)
	if err != nil {
		t.Fatal(err)
	}
	artifact.Graph = graph
	if err := artifact.Validate(); err != nil {
		t.Fatal(err)
	}

	artifact.ShapeCalibration.Units[1].WarmedShapes = 2
	if err := artifact.Validate(); err == nil || !strings.Contains(err.Error(), "shape-calibration") {
		t.Fatalf("incomplete shape calibration validation error = %v", err)
	}
}

func TestSGLangN580ArtifactUsesInPlaceDiskRecovery(t *testing.T) {
	artifact := validArtifact()
	driver, err := snapshotdriver.Lookup(snapshotdriver.N580)
	if err != nil {
		t.Fatal(err)
	}
	binding := driver.Binding()
	artifact.Driver = driver
	artifact.Requires = driver.BaseRequirements.Clone()
	artifact.Launch.Engine = "sglang"
	graph, err := NewGraphPolicyRecord(Request{
		Launch: artifact.Launch, Policy: newDefaultSnapshotPolicy(),
	}, artifact.ShapeCalibration)
	if err != nil {
		t.Fatal(err)
	}
	artifact.Graph = graph
	for index := range artifact.Capsule.Images {
		artifact.Capsule.Images[index].Driver = binding
	}
	artifact.Weights.Native.Driver = binding
	artifact.Weights.Recovery.Driver = binding
	artifact.Weights.Recovery.LoadPath = "sglang-startup-disk"
	if err := artifact.Validate(); err != nil {
		t.Fatal(err)
	}

	artifact.Weights.Recovery.LoadPath = "coldsnap-replay"
	if err := artifact.Validate(); err == nil || !strings.Contains(err.Error(), "sglang-startup-disk") {
		t.Fatalf("unexpected SGLang n580 recovery path validation error: %v", err)
	}
}

func TestArtifactRejectsWorkerOwnedOrTamperedRuntimeProvider(t *testing.T) {
	artifact := validArtifact()
	binding, err := NewRuntimeProviderBinding(
		WorkerOwner("worker-a"), "collective:nccl", "nccl:provider-v1",
		map[string]any{"provider_id": "nccl-2.31.2"},
	)
	if err == nil || !strings.Contains(err.Error(), "identity") {
		t.Fatalf("worker-owned binding error = %v", err)
	}
	binding, err = NewRuntimeProviderBinding(
		UnitOwner("unit-a"), "collective:nccl", "nccl:provider-v1",
		map[string]any{"provider_id": "nccl-2.31.2"},
	)
	if err != nil {
		t.Fatal(err)
	}
	binding.Payload = json.RawMessage(`{"provider_id":"nccl-2.30.7"}`)
	artifact.Runtime.Bindings = []RuntimeProviderBinding{binding}
	if err := artifact.Validate(); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("tampered binding validation error = %v", err)
	}
}

func TestReadArtifactRejectsOldFormat(t *testing.T) {
	artifact := validArtifact()
	artifact.Format = ArtifactFormat - 1
	payload, err := Encode(artifact)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "artifact.json")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadArtifact(path); err == nil || !strings.Contains(err.Error(), "committed ColdSnap v9") {
		t.Fatalf("old artifact format error = %v", err)
	}
}

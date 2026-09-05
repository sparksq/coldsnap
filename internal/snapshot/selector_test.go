// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import (
	"strings"
	"testing"

	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

const workerBDigest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

func validArtifact() Artifact {
	launch := validLaunch(2)
	driver, err := snapshotdriver.Lookup(snapshotdriver.N610)
	if err != nil {
		panic(err)
	}
	binding := driver.Binding()
	graph, err := NewGraphPolicyRecord(Request{
		Launch: launch, Policy: newDefaultSnapshotPolicy(),
	}, nil)
	if err != nil {
		panic(err)
	}
	return Artifact{
		Format: ArtifactFormat, Kind: ArtifactKind, State: "committed", CaptureID: "qwen35-capture-1",
		Driver:        driver,
		Requires:      driver.BaseRequirements.Clone(),
		Graph:         graph,
		RequestSHA256: testDigest, Launch: launch, Compatibility: validCompatibility(2),
		Capsule: Capsule{
			Images: []CapsuleImage{
				{Unit: "unit-a", Reference: "org/capsule@" + testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule", Driver: binding},
				{Unit: "unit-b", Reference: "org/capsule@" + testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule", Driver: binding},
			},
			Objects: []Object{
				{Role: "criu-images", Owner: UnitOwner("unit-a"), Path: "units/unit-a/criu.pack", Bytes: 10, SHA256: testDigest},
				{Role: "criu-images", Owner: UnitOwner("unit-b"), Path: "units/unit-b/criu.pack", Bytes: 11, SHA256: testDigest},
			},
		},
		Weights: WeightProviders{
			Native: &NativeProvider{Driver: binding},
			ModelPayloads: &ModelPayloadProvider{
				Repository: "org/qwen-native", Revision: "commit1",
				Objects: []Object{
					{Role: "model-weight-payload", Owner: WorkerOwner("worker-a"), Path: "model-payloads/sha256/" + strings.TrimPrefix(testDigest, "sha256:") + ".pack", Bytes: 100, SHA256: testDigest},
					{Role: "model-weight-payload", Owner: WorkerOwner("worker-b"), Path: "model-payloads/sha256/" + strings.TrimPrefix(workerBDigest, "sha256:") + ".pack", Bytes: 101, SHA256: workerBDigest},
				},
			},
			Recovery: RecoveryProvider{
				Source: "huggingface-safetensors", ModelID: launch.Model.ID, Revision: launch.Model.Revision,
				Driver: binding, LoadPath: "coldsnap-replay",
				ReplayPlan: []Object{
					{Role: "safetensors-replay-plan", Owner: WorkerOwner("worker-a"), Path: "workers/worker-a/replay.json", Bytes: 20, SHA256: testDigest},
					{Role: "safetensors-replay-plan", Owner: WorkerOwner("worker-b"), Path: "workers/worker-b/replay.json", Bytes: 21, SHA256: testDigest},
				},
			},
		},
		Acceptance: ArtifactAcceptance{Accepted: true, Expected: "world"},
	}
}

func TestSelectVerifiedNative(t *testing.T) {
	artifact := validArtifact()
	selection, err := SelectWeights(artifact, "auto", ProviderInventory{ModelPayloads: map[string]PreparedPayload{
		"worker-a": {Worker: "worker-a", Path: "/cache/r0", Bytes: 100, SHA256: testDigest},
		"worker-b": {Worker: "worker-b", Path: "/cache/r1", Bytes: 101, SHA256: workerBDigest},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if selection.Provider != "native" || len(selection.ModelPayloads) != 2 || selection.FetchRequired {
		t.Fatalf("selection = %#v", selection)
	}
}

func TestAutoFallsBackOnlyWhenNativeMissing(t *testing.T) {
	selection, err := SelectWeights(validArtifact(), "cache-only-auto", ProviderInventory{})
	if err != nil {
		t.Fatal(err)
	}
	if selection.Provider != "recovery" {
		t.Fatalf("selection = %#v", selection)
	}
}

func TestDigestMismatchFailsClosed(t *testing.T) {
	_, err := SelectWeights(validArtifact(), "auto", ProviderInventory{ModelPayloads: map[string]PreparedPayload{
		"worker-a": {Worker: "worker-a", Path: "/cache/r0", Bytes: 100, SHA256: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
	}})
	if err == nil || !strings.Contains(err.Error(), "identity verification") {
		t.Fatalf("SelectWeights error = %v", err)
	}
}

func TestAutoCanRequestOptionalFetch(t *testing.T) {
	selection, err := SelectWeights(validArtifact(), "auto", ProviderInventory{AllowNativeFetch: true})
	if err != nil {
		t.Fatal(err)
	}
	if selection.Provider != "native" || !selection.FetchRequired {
		t.Fatalf("selection = %#v", selection)
	}
}

func TestArtifactSeparatesUnitCapsulesFromWorkerWeightPacks(t *testing.T) {
	launch := multiWorkerLaunch(t, 2, 2)
	driver, err := snapshotdriver.Lookup(snapshotdriver.N610)
	if err != nil {
		t.Fatal(err)
	}
	binding := driver.Binding()
	graph, err := NewGraphPolicyRecord(Request{
		Launch: launch, Policy: newDefaultSnapshotPolicy(),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	artifact := Artifact{
		Format: ArtifactFormat, Kind: ArtifactKind, State: "committed", CaptureID: "tp4-two-unit",
		Driver:        driver,
		Requires:      driver.BaseRequirements.Clone(),
		Graph:         graph,
		RequestSHA256: testDigest, Launch: launch,
		Compatibility: ArtifactCompatibility{Policy: PortabilityPolicy, Units: []UnitPlatformCompatibility{
			{Unit: "unit-a", Architecture: "aarch64", Kernel: "6.17.0", NVIDIADriverCaptured: "610.0", NVIDIADriverMin: "610.0", CUDAUserspace: "13.0", Devices: []DeviceCompatibility{
				{Slot: 0, GPUName: "NVIDIA GB10", ComputeCapability: "12.1"},
				{Slot: 1, GPUName: "NVIDIA GB10", ComputeCapability: "12.1"},
			}},
			{Unit: "unit-b", Architecture: "aarch64", Kernel: "6.17.0", NVIDIADriverCaptured: "610.0", NVIDIADriverMin: "610.0", CUDAUserspace: "13.0", Devices: []DeviceCompatibility{
				{Slot: 0, GPUName: "NVIDIA GB10", ComputeCapability: "12.1"},
				{Slot: 1, GPUName: "NVIDIA GB10", ComputeCapability: "12.1"},
			}},
		}},
		Capsule: Capsule{
			Images: []CapsuleImage{
				{Unit: "unit-a", Reference: "org/capsule@" + testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule", Driver: binding},
				{Unit: "unit-b", Reference: "org/capsule@" + testDigest, Digest: testDigest, Root: "/opt/coldsnap/capsule", Driver: binding},
			},
			Objects: []Object{
				{Role: "oci-capsule", Owner: UnitOwner("unit-a"), Path: "units/unit-a/capsule.oci", Bytes: 10, SHA256: testDigest},
				{Role: "oci-capsule", Owner: UnitOwner("unit-b"), Path: "units/unit-b/capsule.oci", Bytes: 10, SHA256: testDigest},
			},
		},
		Weights: WeightProviders{
			Native:        &NativeProvider{Driver: binding},
			ModelPayloads: &ModelPayloadProvider{Objects: workerObjects(launch, "model-weight-payload", ".weights")},
			Recovery: RecoveryProvider{
				Source: "huggingface-safetensors", ModelID: launch.Model.ID, Revision: launch.Model.Revision,
				Driver:     binding,
				LoadPath:   "coldsnap-replay",
				ReplayPlan: workerObjects(launch, "safetensors-replay-plan", ".json"),
			},
		},
		Acceptance: ArtifactAcceptance{Accepted: true, Expected: "ok"},
	}
	if err := artifact.Validate(); err != nil {
		t.Fatal(err)
	}
	if len(artifact.Capsule.Images) != 2 || len(artifact.Weights.ModelPayloads.Objects) != 4 {
		t.Fatalf("artifact ownership cardinality = %#v", artifact)
	}
}

func TestSharedModelPayloadPathRequiresIdenticalObjectIdentity(t *testing.T) {
	artifact := validArtifact()
	shared := artifact.Weights.ModelPayloads.Objects[0]
	shared.Owner = artifact.Weights.ModelPayloads.Objects[1].Owner
	artifact.Weights.ModelPayloads.Objects[1] = shared
	if err := artifact.Validate(); err != nil {
		t.Fatalf("shared content-addressed payload was rejected: %v", err)
	}
	artifact.Weights.ModelPayloads.Objects[1].Bytes++
	if err := artifact.Validate(); err == nil || !strings.Contains(err.Error(), "object record") {
		t.Fatalf("inconsistent shared payload identity error = %v", err)
	}
}

func TestModelPayloadPathMustMatchDigest(t *testing.T) {
	artifact := validArtifact()
	artifact.Weights.ModelPayloads.Objects[0].Path = "model-payloads/sha256/not-the-digest.pack"
	if err := artifact.Validate(); err == nil || !strings.Contains(err.Error(), "content-addressed") {
		t.Fatalf("model payload path error = %v", err)
	}
}

func workerObjects(launch LaunchSpec, role, suffix string) []Object {
	objects := make([]Object, 0, len(launch.Execution.Workers))
	for _, worker := range launch.Execution.Workers {
		path := "workers/" + worker.ID + suffix
		if role == "model-weight-payload" {
			path = "model-payloads/sha256/" + strings.TrimPrefix(testDigest, "sha256:") + ".pack"
		}
		objects = append(objects, Object{
			Role: role, Owner: WorkerOwner(worker.ID), Path: path,
			Bytes: 100, SHA256: testDigest,
		})
	}
	return objects
}

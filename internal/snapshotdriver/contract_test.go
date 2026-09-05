// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshotdriver

import "testing"

func TestRegisteredContracts(t *testing.T) {
	for _, id := range []string{N580, N610} {
		contract, err := Lookup(id)
		if err != nil {
			t.Fatal(err)
		}
		if err := contract.Validate(); err != nil {
			t.Fatalf("contract %s: %v", id, err)
		}
	}
}

func TestSelectionRejectsAutoAndUnknown(t *testing.T) {
	for _, id := range []string{"", "auto", "n999"} {
		if err := (Selection{ID: id}).Validate(); err == nil {
			t.Fatalf("selection %q unexpectedly passed", id)
		}
	}
}

func TestFeatureAdmissionPreservesNegativeStatus(t *testing.T) {
	requirements := NewRequirements(FeatureCUDAClassicIPC, FeatureCUDAVMMExactAddress)
	profile := FeatureProfile{
		Format: FeatureProfileFormat,
		Kind:   "coldsnap-host-feature-profile",
		Identity: map[string]any{
			"boot_id": "test",
		},
		Features: []FeatureResult{
			{ID: FeatureCUDAClassicIPC, Status: FeatureFailed, Probe: "cuda-ipc-v1", FailurePhase: "open", Reason: "CUDA_ERROR_INVALID_HANDLE", Seconds: 0.1},
			{ID: FeatureCUDAVMMExactAddress, Status: FeaturePassed, Probe: "cuda-vmm-exact-v1", Seconds: 0.1},
		},
	}
	if err := requirements.Admit(profile); err == nil || err.Error() != "required feature cuda-classic-ipc is failed: phase open: CUDA_ERROR_INVALID_HANDLE" {
		t.Fatalf("admission error = %v", err)
	}
}

func TestFeatureAdmissionDistinguishesMissingQualification(t *testing.T) {
	requirements := NewRequirements(FeatureCUDAClassicIPC)
	profile := FeatureProfile{
		Format: FeatureProfileFormat, Kind: "coldsnap-host-feature-profile",
		Identity: map[string]any{"boot_id": "test"},
		Features: []FeatureResult{
			{ID: FeatureCUDAUVMCheckpoint, Status: FeatureUnsupported, Probe: "known-negative-v1", Reason: "unsupported by CUDA checkpoint", Seconds: 0},
		},
	}
	if err := requirements.Admit(profile); err == nil || err.Error() != "required feature cuda-classic-ipc is unqualified: destination profile contains no result" {
		t.Fatalf("admission error = %v", err)
	}
}

func TestQualificationRequiresExactEnvironmentAndRepeatedRestoreEvidence(t *testing.T) {
	profile := FeatureProfile{
		Format: 1, Kind: "coldsnap-host-feature-profile",
		Identity: map[string]any{"driver": "610.43.02", "provider": "nccl-2.31.2-1+coldsnap.11"},
	}
	record := QualificationRecord{
		Format: 1, Kind: "coldsnap-qualification-record", ID: "tp2-graph-socket-v1", Accepted: true,
		Environment: map[string]any{"driver": "610.43.02", "provider": "nccl-2.31.2-1+coldsnap.11"},
		Scope: QualificationScope{
			Engine: "vllm", ModelFamily: "qwen3.8", Connector: "none",
			TopologySHA256: "sha256:test", GraphMode: "full", Transport: "socket",
			NCCLProvider: "nccl-2.31.2-1+coldsnap.11",
		},
		Features: []FeatureID{FeatureCUDAClassicIPC}, Restores: 2,
	}
	qualified, err := ApplyQualification(profile, record)
	if err != nil || len(qualified.Features) != 1 || qualified.Features[0].Status != FeaturePassed {
		t.Fatalf("qualified profile = %#v, %v", qualified, err)
	}
	record.Environment["driver"] = "610.99.01"
	if _, err := ApplyQualification(profile, record); err == nil {
		t.Fatal("mismatched qualification environment unexpectedly passed")
	}
}

func TestQualificationPromotesOnlyUnqualifiedResults(t *testing.T) {
	profile := FeatureProfile{
		Format: 1, Kind: "coldsnap-host-feature-profile",
		Identity: map[string]any{"driver": "610.43.02", "provider": "nccl-2.31.2-1+coldsnap.11"},
		Features: []FeatureResult{
			{ID: FeatureNCCLGraphRetention, Status: FeatureUnqualified, Probe: "known-boundary-v1", Reason: "no accepted deep qualification"},
		},
	}
	record := QualificationRecord{
		Format: 1, Kind: "coldsnap-qualification-record", ID: "tp2-graph-socket-v1", Accepted: true,
		Environment: map[string]any{"driver": "610.43.02", "provider": "nccl-2.31.2-1+coldsnap.11"},
		Scope: QualificationScope{
			Engine: "vllm", ModelFamily: "qwen3.8", Connector: "none",
			TopologySHA256: "sha256:test", GraphMode: "full", Transport: "socket",
			NCCLProvider: "nccl-2.31.2-1+coldsnap.11",
		},
		Features: []FeatureID{FeatureNCCLGraphRetention}, Restores: 2,
	}
	qualified, err := ApplyQualification(profile, record)
	if err != nil || qualified.Features[0].Status != FeaturePassed ||
		qualified.Features[0].Probe != "qualification:tp2-graph-socket-v1" {
		t.Fatalf("qualified profile = %#v, %v", qualified, err)
	}
	profile.Features[0].Status = FeatureFailed
	profile.Features[0].Reason = "current probe failed"
	if _, err := ApplyQualification(profile, record); err == nil {
		t.Fatal("qualification unexpectedly overrode a current failed probe")
	}
}

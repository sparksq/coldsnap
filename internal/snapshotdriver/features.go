// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshotdriver

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
)

const FeatureProfileFormat = 1

// FeatureID is a bounded identity for one independently probed resource
// family. Driver names are deliberately not capabilities: a driver contract
// is admitted by matching these requirements against destination evidence.
type FeatureID string

const (
	FeatureCRIUProcessTemplate       FeatureID = "criu-process-template"
	FeatureCRIUProcessTree           FeatureID = "criu-process-tree"
	FeatureCUDAFreshState            FeatureID = "cuda-fresh-state"
	FeatureCUDAInitializedState      FeatureID = "cuda-initialized-state"
	FeatureCUDACheckpointAPI         FeatureID = "cuda-checkpoint-api"
	FeatureCUDACheckpointRoundTrip   FeatureID = "cuda-checkpoint-roundtrip"
	FeatureCUDAClassicIPC            FeatureID = "cuda-classic-ipc"
	FeatureCUDAVMMExactAddress       FeatureID = "cuda-vmm-exact-address"
	FeatureCUDAUVMCheckpoint         FeatureID = "cuda-uvm-checkpoint"
	FeatureCUDAVMMExportedCheckpoint FeatureID = "cuda-vmm-exported-handle-checkpoint"
	FeatureNCCLCommSuspendInPlace    FeatureID = "nccl-communicator-suspend-in-place"
	FeatureNCCLTransportDetach       FeatureID = "nccl-transport-detach-in-place"
	FeatureNCCLGraphRetention        FeatureID = "nccl-graph-resource-retention"
	FeatureNCCLRegistrationReplay    FeatureID = "nccl-registration-window-in-place-replay"
	FeatureNCCLNVLSReplay            FeatureID = "nccl-nvls-in-place-replay"
	FeatureNCCLDeviceAPIRetention    FeatureID = "nccl-device-api-state-retention"
)

var knownFeatures = []FeatureID{
	FeatureCRIUProcessTemplate,
	FeatureCRIUProcessTree,
	FeatureCUDAFreshState,
	FeatureCUDAInitializedState,
	FeatureCUDACheckpointAPI,
	FeatureCUDACheckpointRoundTrip,
	FeatureCUDAClassicIPC,
	FeatureCUDAVMMExactAddress,
	FeatureCUDAUVMCheckpoint,
	FeatureCUDAVMMExportedCheckpoint,
	FeatureNCCLCommSuspendInPlace,
	FeatureNCCLTransportDetach,
	FeatureNCCLGraphRetention,
	FeatureNCCLRegistrationReplay,
	FeatureNCCLNVLSReplay,
	FeatureNCCLDeviceAPIRetention,
}

type FeatureStatus string

const (
	FeatureUnsupported FeatureStatus = "unsupported"
	FeatureUnqualified FeatureStatus = "unqualified"
	FeatureFailed      FeatureStatus = "failed"
	FeaturePassed      FeatureStatus = "passed"
)

type FeatureRequirement struct {
	ID FeatureID `json:"id"`
}

type Requirements struct {
	Format   int                  `json:"format"`
	Features []FeatureRequirement `json:"features"`
}

type FeatureResult struct {
	ID           FeatureID      `json:"id"`
	Status       FeatureStatus  `json:"status"`
	Probe        string         `json:"probe"`
	FailurePhase string         `json:"failure_phase,omitempty"`
	Reason       string         `json:"reason,omitempty"`
	Seconds      float64        `json:"seconds"`
	Evidence     map[string]any `json:"evidence,omitempty"`
}

type FeatureProfile struct {
	Format        int             `json:"format"`
	Kind          string          `json:"kind"`
	Identity      map[string]any  `json:"identity"`
	Features      []FeatureResult `json:"features"`
	StartedUnix   float64         `json:"started_unix"`
	CompletedUnix float64         `json:"completed_unix"`
}

// QualificationRecord is expensive, immutable evidence produced outside fast
// placement admission. Its environment identity is exact by default; a future
// compatibility rule must be another bounded enum, never an implicit range.
type QualificationRecord struct {
	Format      int                `json:"format"`
	Kind        string             `json:"kind"`
	ID          string             `json:"id"`
	Accepted    bool               `json:"accepted"`
	Environment map[string]any     `json:"environment"`
	Scope       QualificationScope `json:"scope"`
	Features    []FeatureID        `json:"features"`
	Restores    int                `json:"unchanged_artifact_restores"`
}

type QualificationScope struct {
	Engine         string `json:"engine"`
	ModelFamily    string `json:"model_family"`
	Connector      string `json:"connector"`
	TopologySHA256 string `json:"topology_sha256"`
	GraphMode      string `json:"graph_mode"`
	Transport      string `json:"transport"`
	NCCLProvider   string `json:"nccl_provider"`
}

func (record QualificationRecord) Validate() error {
	if record.Format != 1 || record.Kind != "coldsnap-qualification-record" ||
		record.ID == "" || !record.Accepted || len(record.Environment) == 0 ||
		record.Scope.Engine == "" || record.Scope.ModelFamily == "" ||
		record.Scope.TopologySHA256 == "" || record.Scope.GraphMode == "" ||
		record.Scope.Transport == "" || record.Scope.NCCLProvider == "" ||
		record.Restores < 2 || len(record.Features) == 0 {
		return errors.New("qualification record is incomplete or unaccepted")
	}
	previous := FeatureID("")
	for _, feature := range record.Features {
		if !validFeature(feature) || feature <= previous {
			return errors.New("qualification feature inventory is unknown, duplicated, or unsorted")
		}
		previous = feature
	}
	return nil
}

// ApplyQualification adds passed results only when the destination immutable
// identity exactly matches the accepted evidence. It does not weaken exact
// NCCL provider selection or replace the fast disposable results.
func ApplyQualification(
	profile FeatureProfile, record QualificationRecord,
) (FeatureProfile, error) {
	if err := profile.Validate(); err != nil {
		return FeatureProfile{}, err
	}
	if err := record.Validate(); err != nil {
		return FeatureProfile{}, err
	}
	actual, err := json.Marshal(profile.Identity)
	if err != nil {
		return FeatureProfile{}, err
	}
	expected, err := json.Marshal(record.Environment)
	if err != nil {
		return FeatureProfile{}, err
	}
	var actualValue, expectedValue any
	if json.Unmarshal(actual, &actualValue) != nil || json.Unmarshal(expected, &expectedValue) != nil ||
		!deepJSONEqual(actualValue, expectedValue) {
		return FeatureProfile{}, errors.New("qualification environment identity differs from destination profile")
	}
	provided := make(map[FeatureID]int, len(profile.Features))
	for index, feature := range profile.Features {
		provided[feature.ID] = index
	}
	profile.Features = slices.Clone(profile.Features)
	for _, feature := range record.Features {
		if index, ok := provided[feature]; ok {
			switch profile.Features[index].Status {
			case FeaturePassed:
				continue
			case FeatureUnqualified:
				profile.Features[index] = FeatureResult{
					ID: feature, Status: FeaturePassed,
					Probe: "qualification:" + record.ID,
				}
				continue
			default:
				return FeatureProfile{}, fmt.Errorf(
					"qualification %s cannot override current %s result for feature %s",
					record.ID, profile.Features[index].Status, feature,
				)
			}
		}
		profile.Features = append(profile.Features, FeatureResult{
			ID: feature, Status: FeaturePassed,
			Probe: "qualification:" + record.ID,
		})
		provided[feature] = len(profile.Features) - 1
	}
	slices.SortFunc(profile.Features, func(a, b FeatureResult) int {
		return strings.Compare(string(a.ID), string(b.ID))
	})
	return profile, profile.Validate()
}

func deepJSONEqual(first, second any) bool {
	firstJSON, firstErr := json.Marshal(first)
	secondJSON, secondErr := json.Marshal(second)
	return firstErr == nil && secondErr == nil && string(firstJSON) == string(secondJSON)
}

func validFeature(id FeatureID) bool { return slices.Contains(knownFeatures, id) }

func NewRequirements(ids ...FeatureID) Requirements {
	features := make([]FeatureRequirement, 0, len(ids))
	for _, id := range ids {
		features = append(features, FeatureRequirement{ID: id})
	}
	slices.SortFunc(features, func(a, b FeatureRequirement) int {
		return strings.Compare(string(a.ID), string(b.ID))
	})
	return Requirements{Format: FeatureProfileFormat, Features: features}
}

func (requirements Requirements) Clone() Requirements {
	requirements.Features = slices.Clone(requirements.Features)
	return requirements
}

func (requirements Requirements) Validate() error {
	if requirements.Format != FeatureProfileFormat || len(requirements.Features) == 0 {
		return errors.New("snapshot feature requirements are empty or use an unsupported format")
	}
	var previous FeatureID
	for _, feature := range requirements.Features {
		if !validFeature(feature.ID) || feature.ID <= previous {
			return errors.New("snapshot feature requirements are unknown, duplicated, or unsorted")
		}
		previous = feature.ID
	}
	return nil
}

func (requirements Requirements) ContainsAll(base Requirements) bool {
	provided := make(map[FeatureID]bool, len(requirements.Features))
	for _, feature := range requirements.Features {
		provided[feature.ID] = true
	}
	for _, feature := range base.Features {
		if !provided[feature.ID] {
			return false
		}
	}
	return true
}

func (profile FeatureProfile) Validate() error {
	if profile.Format != FeatureProfileFormat || profile.Kind != "coldsnap-host-feature-profile" || len(profile.Identity) == 0 {
		return errors.New("host feature profile identity or format is invalid")
	}
	if profile.StartedUnix < 0 || profile.CompletedUnix < profile.StartedUnix {
		return errors.New("host feature profile timestamps are invalid")
	}
	seen := make(map[FeatureID]bool, len(profile.Features))
	for _, feature := range profile.Features {
		if !validFeature(feature.ID) || seen[feature.ID] || feature.Probe == "" || feature.Seconds < 0 {
			return errors.New("host feature profile result is invalid or duplicated")
		}
		if !slices.Contains([]FeatureStatus{FeatureUnsupported, FeatureUnqualified, FeatureFailed, FeaturePassed}, feature.Status) {
			return fmt.Errorf("host feature %s has unknown status %q", feature.ID, feature.Status)
		}
		if feature.Status != FeaturePassed && feature.Reason == "" {
			return fmt.Errorf("host feature %s status %s has no reason", feature.ID, feature.Status)
		}
		seen[feature.ID] = true
	}
	return nil
}

// Admit requires accepted evidence for every feature used by an artifact. A
// negative result retains its exact status so operators can distinguish an
// implementation limit from missing qualification or a failed attempt.
func (requirements Requirements) Admit(profile FeatureProfile) error {
	if err := requirements.Validate(); err != nil {
		return err
	}
	if err := profile.Validate(); err != nil {
		return err
	}
	provided := make(map[FeatureID]FeatureResult, len(profile.Features))
	for _, feature := range profile.Features {
		provided[feature.ID] = feature
	}
	for _, requirement := range requirements.Features {
		result, ok := provided[requirement.ID]
		if !ok {
			return fmt.Errorf("required feature %s is unqualified: destination profile contains no result", requirement.ID)
		}
		if result.Status != FeaturePassed {
			message := result.Reason
			if result.FailurePhase != "" {
				message = "phase " + result.FailurePhase + ": " + message
			}
			return fmt.Errorf("required feature %s is %s: %s", requirement.ID, result.Status, message)
		}
	}
	return nil
}

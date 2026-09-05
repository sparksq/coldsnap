// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshotdriver

import (
	"errors"
	"fmt"
	"slices"
)

const ABI = 1

const (
	N580 = "n580"
	N610 = "n610"
)

// Selection is the explicit snapshot implementation requested by a caller.
// ColdSnap core deliberately has no "auto" value: orchestrators must resolve
// their hardware policy before constructing an operation request.
type Selection struct {
	ID string `json:"id"`
}

// Binding is the compact contract identity copied onto driver-sensitive
// capsule and weight-provider records.
type Binding struct {
	ID  string `json:"id"`
	ABI int    `json:"abi"`
}

// Contract is the artifact-stable identity of one snapshot implementation.
// It describes the lifecycle boundary rather than the installed kernel driver
// version, which remains separately recorded as capture evidence.
type Contract struct {
	ID                       string       `json:"id"`
	ABI                      int          `json:"abi"`
	CaptureBoundary          string       `json:"capture_boundary"`
	MinimumNVIDIADriverMajor int          `json:"minimum_nvidia_driver_major"`
	BaseRequirements         Requirements `json:"base_requirements"`
}

var contracts = map[string]Contract{
	N580: {
		ID: N580, ABI: ABI, CaptureBoundary: "pre-cuda-process-template",
		MinimumNVIDIADriverMajor: 580,
		BaseRequirements: NewRequirements(
			FeatureCRIUProcessTemplate, FeatureCUDAFreshState, FeatureCUDAVMMExactAddress,
		),
	},
	N610: {
		ID: N610, ABI: ABI, CaptureBoundary: "initialized-cuda-process-state",
		MinimumNVIDIADriverMajor: 610,
		BaseRequirements: NewRequirements(
			FeatureCRIUProcessTree, FeatureCUDACheckpointAPI,
			FeatureCUDACheckpointRoundTrip, FeatureCUDAClassicIPC,
			FeatureCUDAInitializedState,
			FeatureCUDAVMMExactAddress,
		),
	},
}

func Lookup(id string) (Contract, error) {
	contract, ok := contracts[id]
	if !ok {
		return Contract{}, fmt.Errorf("unsupported snapshot driver %q", id)
	}
	contract.BaseRequirements = contract.BaseRequirements.Clone()
	return contract, nil
}

func Contracts() []Contract {
	ids := make([]string, 0, len(contracts))
	for id := range contracts {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	values := make([]Contract, 0, len(ids))
	for _, id := range ids {
		contract, _ := Lookup(id)
		values = append(values, contract)
	}
	return values
}

func (selection Selection) Validate() error {
	if selection.ID == "" {
		return errors.New("snapshot driver is required")
	}
	_, err := Lookup(selection.ID)
	return err
}

func (contract Contract) Validate() error {
	expected, err := Lookup(contract.ID)
	if err != nil {
		return err
	}
	if contract.ABI != expected.ABI || contract.CaptureBoundary != expected.CaptureBoundary ||
		contract.MinimumNVIDIADriverMajor != expected.MinimumNVIDIADriverMajor ||
		!slices.Equal(contract.BaseRequirements.Features, expected.BaseRequirements.Features) {
		return fmt.Errorf("snapshot driver %s contract differs from ABI %d", contract.ID, expected.ABI)
	}
	if err := contract.BaseRequirements.Validate(); err != nil {
		return fmt.Errorf("snapshot driver %s requirements: %w", contract.ID, err)
	}
	return nil
}
func (contract Contract) Binding() Binding {
	return Binding{ID: contract.ID, ABI: contract.ABI}
}

func (binding Binding) Validate() error {
	contract, err := Lookup(binding.ID)
	if err != nil {
		return err
	}
	if binding.ABI != contract.ABI {
		return fmt.Errorf("snapshot driver %s ABI %d is unsupported; expected ABI %d", binding.ID, binding.ABI, contract.ABI)
	}
	return nil
}

func (binding Binding) Matches(contract Contract) bool {
	return binding.ID == contract.ID && binding.ABI == contract.ABI
}

func Resolve(selection Selection) (Contract, error) {
	if err := selection.Validate(); err != nil {
		return Contract{}, err
	}
	return Lookup(selection.ID)
}

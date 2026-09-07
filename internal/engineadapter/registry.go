// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Package engineadapter defines the controller-to-engine adapter boundary.
// Adapters are separate executables which consume the stable snapshot request
// protocol; registering one does not add engine-specific code to the CLI.
package engineadapter

import (
	"fmt"
	"slices"
)

type Descriptor struct {
	Engine                       string   `json:"engine"`
	Executable                   string   `json:"executable"`
	Environment                  string   `json:"environment"`
	Operations                   []string `json:"operations"`
	PrepareRestore               bool     `json:"prepare_restore"`
	ControllerModel              string   `json:"controller_model"`
	NativeMaterialization        []string `json:"native_materialization"`
	DefaultNativeMaterialization string   `json:"default_native_materialization"`
}

var registered = map[string]Descriptor{
	"vllm": {
		Engine: "vllm", Executable: "coldsnap-vllm-adapter", Environment: "COLDSNAP_VLLM_ADAPTER",
		Operations:     []string{"capture", "publish", "publish-native", "restore", "sleep", "status", "wake"},
		PrepareRestore: true, ControllerModel: "external-process",
		NativeMaterialization: []string{"off", "async", "required"}, DefaultNativeMaterialization: "off",
	},
	"sglang": {
		Engine: "sglang", Executable: "coldsnap-sglang-adapter", Environment: "COLDSNAP_SGLANG_ADAPTER",
		Operations:     []string{"capture", "publish", "publish-native", "restore", "sleep", "status", "wake"},
		PrepareRestore: true, ControllerModel: "external-process",
		NativeMaterialization: []string{"off"}, DefaultNativeMaterialization: "off",
	},
}

func Lookup(engine string) (Descriptor, error) {
	descriptor, ok := registered[engine]
	if !ok {
		return Descriptor{}, fmt.Errorf("unsupported engine adapter %q", engine)
	}
	descriptor.Operations = slices.Clone(descriptor.Operations)
	descriptor.NativeMaterialization = slices.Clone(descriptor.NativeMaterialization)
	return descriptor, nil
}

func Descriptors() []Descriptor {
	engines := make([]string, 0, len(registered))
	for engine := range registered {
		engines = append(engines, engine)
	}
	slices.Sort(engines)
	descriptors := make([]Descriptor, 0, len(engines))
	for _, engine := range engines {
		descriptor, _ := Lookup(engine)
		descriptors = append(descriptors, descriptor)
	}
	return descriptors
}

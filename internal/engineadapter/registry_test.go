// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package engineadapter

import "testing"

func TestRegistryIsStableAndReturnsCopies(t *testing.T) {
	descriptors := Descriptors()
	if len(descriptors) != 2 || descriptors[0].Engine != "sglang" || descriptors[1].Engine != "vllm" {
		t.Fatalf("descriptors = %#v", descriptors)
	}
	if len(descriptors[0].Operations) != 7 || !descriptors[0].PrepareRestore || descriptors[0].ControllerModel != "external-process" {
		t.Fatalf("SGLang controller capabilities are incomplete: %#v", descriptors[0])
	}
	if len(descriptors[0].NativeMaterialization) != 1 || descriptors[0].NativeMaterialization[0] != "off" ||
		descriptors[0].DefaultNativeMaterialization != "off" ||
		len(descriptors[1].NativeMaterialization) != 3 || descriptors[1].DefaultNativeMaterialization != "off" {
		t.Fatalf("native materialization capabilities = %#v", descriptors)
	}
	descriptors[1].Operations[0] = "mutated"
	descriptors[1].NativeMaterialization[0] = "mutated"
	descriptor, err := Lookup("vllm")
	if err != nil {
		t.Fatal(err)
	}
	if descriptor.Operations[0] != "capture" || descriptor.NativeMaterialization[0] != "off" {
		t.Fatalf("registry was mutated: %#v", descriptor)
	}
}

func TestUnknownEngineIsRejected(t *testing.T) {
	if _, err := Lookup("unknown"); err == nil {
		t.Fatal("expected unsupported engine error")
	}
}

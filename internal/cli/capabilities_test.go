// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package cli

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"
)

func TestCapabilitiesDescribeControllerBoundary(t *testing.T) {
	stdout := &bytes.Buffer{}
	command := NewRoot(Dependencies{Stdin: &bytes.Buffer{}, Stdout: stdout, Stderr: &bytes.Buffer{}})
	command.SetArgs([]string{"capabilities"})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	var capabilities controllerCapabilities
	decoder := json.NewDecoder(stdout)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&capabilities); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(capabilities.Features, "manager-runtime-v1") {
		t.Fatal("typed runtime feature is absent")
	}
	if capabilities.Kind != "coldsnap-controller-capabilities" || capabilities.Protocols.ReceiptFormat == 0 ||
		capabilities.Protocols.TimingEventFormat != 1 || capabilities.Protocols.HostProviderFormat != 1 ||
		capabilities.Protocols.PayloadValidationRPCFormat != 1 || len(capabilities.SnapshotDrivers) != 2 {
		t.Fatalf("capabilities = %#v", capabilities)
	}
	if len(capabilities.Protocols.AcceptedRequestFormats) != 1 ||
		capabilities.Protocols.AcceptedRequestFormats[0] != capabilities.Protocols.RequestFormat ||
		len(capabilities.Protocols.AcceptedArtifactFormats) != 1 ||
		capabilities.Protocols.AcceptedArtifactFormats[0] != capabilities.Protocols.ArtifactFormat {
		t.Fatalf("accepted protocol formats = %#v", capabilities.Protocols)
	}
	if len(capabilities.HostTransports) != 1 || capabilities.HostTransports[0].Name != "manager-provider" ||
		!capabilities.HostTransports[0].EngineAdapterRequests || capabilities.HostTransports[0].Status != "required" {
		t.Fatalf("transport capabilities = %#v", capabilities.HostTransports)
	}
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package cli

import (
	"encoding/json"

	"github.com/sparksq/coldsnap/internal/buildinfo"
	"github.com/sparksq/coldsnap/internal/engineadapter"
	"github.com/sparksq/coldsnap/internal/hostprovider"
	"github.com/sparksq/coldsnap/internal/payloadvalidation"
	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
	"github.com/spf13/cobra"
)

const capabilitiesFormat = 1

type operationCapability struct {
	Operation       string `json:"operation"`
	ReplaySemantics string `json:"replay_semantics"`
	PrepareOnly     bool   `json:"prepare_only,omitempty"`
}

type transportCapability struct {
	Name                  string `json:"name"`
	EngineAdapterRequests bool   `json:"engine_adapter_requests"`
	Security              string `json:"security"`
	Status                string `json:"status"`
}

type protocolCapabilities struct {
	RequestFormat              int   `json:"request_format"`
	AcceptedRequestFormats     []int `json:"accepted_request_formats"`
	ArtifactFormat             int   `json:"artifact_format"`
	AcceptedArtifactFormats    []int `json:"accepted_artifact_formats"`
	ReceiptFormat              int   `json:"receipt_format"`
	TimingEventFormat          int   `json:"timing_event_format"`
	HostProviderFormat         int   `json:"host_provider_format"`
	PayloadValidationRPCFormat int   `json:"payload_validation_rpc_format"`
}

type controllerCapabilities struct {
	Format          int                        `json:"format"`
	Kind            string                     `json:"kind"`
	Controller      buildinfo.Info             `json:"controller"`
	Protocols       protocolCapabilities       `json:"protocols"`
	Operations      []operationCapability      `json:"operations"`
	EngineAdapters  []engineadapter.Descriptor `json:"engine_adapters"`
	SnapshotDrivers []snapshotdriver.Contract  `json:"snapshot_drivers"`
	HostTransports  []transportCapability      `json:"host_transports"`
	Features        []string                   `json:"features"`
}

func currentCapabilities() controllerCapabilities {
	operations := make([]operationCapability, 0, 8)
	for _, operation := range []string{"capture", "publish", "publish-native", "restore", "sleep", "status", "wake"} {
		operations = append(operations, operationCapability{
			Operation: operation, ReplaySemantics: snapshot.ReplaySemantics(operation, false),
		})
	}
	operations = append(operations, operationCapability{
		Operation: "restore", ReplaySemantics: snapshot.ReplaySemantics("restore", true), PrepareOnly: true,
	})
	return controllerCapabilities{
		Format: capabilitiesFormat, Kind: "coldsnap-controller-capabilities", Controller: buildinfo.Current(),
		Protocols: protocolCapabilities{
			RequestFormat: snapshot.RequestFormat, ArtifactFormat: snapshot.ArtifactFormat,
			AcceptedRequestFormats:     []int{snapshot.RequestFormat},
			AcceptedArtifactFormats:    []int{snapshot.ArtifactFormat},
			ReceiptFormat:              snapshot.OperationReceiptFormat,
			TimingEventFormat:          snapshot.TimingEventFormat,
			HostProviderFormat:         hostprovider.ProtocolFormat,
			PayloadValidationRPCFormat: payloadvalidation.RPCFormat,
		},
		Operations: operations, EngineAdapters: engineadapter.Descriptors(), SnapshotDrivers: snapshotdriver.Contracts(),
		HostTransports: []transportCapability{
			{Name: "manager-provider", EngineAdapterRequests: true, Security: "operation-scoped-unix-token", Status: "required"},
		},
		Features: []string{
			"content-addressed-native-payloads", "driver-specific-residual-overlays",
			"content-addressed-activation-runtime", "manager-host-provider", "manager-runtime-v1", "native-payload-verifier-go-v1",
			"native-payload-verifier-rpc-v1",
			"operation-receipts", "operation-timing-events-ndjson-v1", "operation-timing-spans",
			"placement-independent-artifacts", "prepare-only-restore",
		},
	}
}

func newCapabilitiesCommand() *cobra.Command {
	return &cobra.Command{
		Use: "capabilities", Short: "Print the machine-readable ColdSnap controller contract", Args: cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			encoder := json.NewEncoder(command.OutOrStdout())
			encoder.SetEscapeHTML(false)
			encoder.SetIndent("", "  ")
			return encoder.Encode(currentCapabilities())
		},
	}
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package cli

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/sparksq/coldsnap/internal/buildinfo"
	"github.com/spf13/cobra"
)

// Streams are the process streams passed across the engine-adapter boundary.
type Streams struct {
	Stdin  io.Reader
	Stdout io.Writer
	Stderr io.Writer
}

type Dependencies struct {
	RequestRunner RequestRunner
	Stdin         io.Reader
	Stdout        io.Writer
	Stderr        io.Writer
}

// NewRoot returns the manager-facing ColdSnap CLI. Every lifecycle operation
// consumes a validated request document; host coordination belongs to the
// manager that supplies the operation-scoped host provider.
func NewRoot(dependencies Dependencies) *cobra.Command {
	if dependencies.RequestRunner == nil {
		dependencies.RequestRunner = EngineAdapterRunner{}
	}
	root := &cobra.Command{
		Use:           "coldsnap",
		Short:         "Capture, publish, and restore GPU model state",
		SilenceErrors: true,
		SilenceUsage:  true,
	}
	root.SetIn(dependencies.Stdin)
	root.SetOut(dependencies.Stdout)
	root.SetErr(dependencies.Stderr)
	root.AddCommand(
		newVersionCommand(), newCapabilitiesCommand(), newNCCLProviderCommand(), newPayloadCommand(),
	)
	for _, operation := range []struct {
		name, short string
	}{
		{"capture", "Capture a snapshot through an engine adapter"},
		{"publish", "Publish accepted snapshot capsules"},
		{"publish-native", "Publish accepted native model payloads"},
		{"sleep", "Offload a live inference workload"},
		{"wake", "Hydrate and validate a live inference workload"},
		{"status", "Show live lifecycle status"},
	} {
		root.AddCommand(newRequestCommand(dependencies, operation.name, operation.short, false))
	}
	root.AddCommand(newRequestCommand(dependencies, "restore", "Restore a snapshot through an engine adapter", true))
	return root
}

func newRequestCommand(dependencies Dependencies, operation, short string, allowPrepare bool) *cobra.Command {
	var requestJSON string
	var receiptJSON string
	var timingEvents string
	var prepareOnly bool
	command := &cobra.Command{
		Use: operation, Short: short, Args: cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			if requestJSON == "" {
				return fmt.Errorf("--request-json is required")
			}
			return runSnapshotRequest(command.Context(), dependencies, requestJSON, operation, prepareOnly, receiptJSON, timingEvents)
		},
	}
	command.Flags().StringVar(&requestJSON, "request-json", "", "ColdSnap operation request path, or - for stdin")
	command.Flags().StringVar(&receiptJSON, "receipt-json", "", "write a stable operation receipt to a path, or - for stdout")
	command.Flags().StringVar(&timingEvents, "timing-events", "", "stream advisory timing events as NDJSON to a path, or - for stdout")
	if allowPrepare {
		command.Flags().BoolVar(&prepareOnly, "prepare-only", false, "verify and prepare restore assets without activating workloads")
	}
	return command
}

func newVersionCommand() *cobra.Command {
	var outputJSON bool
	command := &cobra.Command{
		Use: "version", Short: "Print the ColdSnap version", Args: cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			info := buildinfo.Current()
			if !outputJSON {
				_, err := fmt.Fprintln(command.OutOrStdout(), info.Version)
				return err
			}
			encoder := json.NewEncoder(command.OutOrStdout())
			encoder.SetEscapeHTML(false)
			return encoder.Encode(info)
		},
	}
	command.Flags().BoolVar(&outputJSON, "json", false, "print machine-readable version information")
	return command
}

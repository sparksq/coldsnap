// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package cli

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"

	"github.com/sparksq/coldsnap/internal/ncclprovider"
	"github.com/spf13/cobra"
)

func newNCCLProviderCommand() *cobra.Command {
	command := &cobra.Command{
		Use:   "nccl-provider",
		Short: "Inspect, select, materialize, and verify immutable NCCL providers",
		Args:  cobra.NoArgs,
	}
	command.AddCommand(
		newNCCLProviderAssembleCommand(),
		newNCCLProviderInspectCommand(),
		newNCCLProviderSelectCommand(false),
		newNCCLProviderSelectCommand(true),
		newNCCLProviderInstallActiveCommand(),
		newNCCLProviderVerifyActiveCommand(),
		newNCCLProviderVerifyCommand(),
	)
	return command
}

func newNCCLProviderAssembleCommand() *cobra.Command {
	var sourceRoot, recipe, qualification, payload, target, output string
	command := &cobra.Command{
		Use:   "assemble",
		Short: "Assemble qualified build outputs into an immutable NCCL provider",
		Args:  cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			if sourceRoot == "" || recipe == "" || qualification == "" || payload == "" || target == "" || output == "" {
				return errors.New("--source-root, --recipe, --qualification, --payload, --target, and --output are required")
			}
			verification, err := ncclprovider.Assemble(ncclprovider.AssemblyOptions{
				SourceRoot: sourceRoot, RecipePath: recipe, QualificationPath: qualification,
				PayloadRoot: payload, TargetPath: target, Output: output,
			})
			if err != nil {
				return err
			}
			return writeNCCLProviderJSON(command, verification)
		},
	}
	flags := command.Flags()
	flags.StringVar(&sourceRoot, "source-root", "", "repository source root for recipe inputs")
	flags.StringVar(&recipe, "recipe", "", "locked provider recipe JSON")
	flags.StringVar(&qualification, "qualification", "", "accepted provider qualification JSON")
	flags.StringVar(&payload, "payload", "", "compiled provider payload root")
	flags.StringVar(&target, "target", "", "strict target-observation JSON path")
	flags.StringVar(&output, "output", "", "clean provider output directory")
	return command
}

func newNCCLProviderVerifyActiveCommand() *cobra.Command {
	var active string
	command := &cobra.Command{
		Use:   "verify-active",
		Short: "Verify an installed active NCCL provider and its resolved payloads",
		Args:  cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			if active == "" {
				return errors.New("--active is required")
			}
			record, err := ncclprovider.VerifyActive(active)
			if err != nil {
				return err
			}
			return writeNCCLProviderJSON(command, record)
		},
	}
	command.Flags().StringVar(&active, "active", "", "active.json path")
	return command
}

func newNCCLProviderInstallActiveCommand() *cobra.Command {
	var provider, bridge, output string
	var bridgeABI uint32
	command := &cobra.Command{
		Use:   "install-active",
		Short: "Install one provider and common bridge into an immutable runtime layout",
		Args:  cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			if provider == "" || bridge == "" || output == "" {
				return errors.New("--provider, --bridge, and --output are required")
			}
			record, err := ncclprovider.InstallActive(provider, bridge, output, bridgeABI)
			if err != nil {
				return err
			}
			return writeNCCLProviderJSON(command, record)
		},
	}
	flags := command.Flags()
	flags.StringVar(&provider, "provider", "", "verified materialized provider root")
	flags.StringVar(&bridge, "bridge", "", "platform-specific dlsym bridge")
	flags.StringVar(&output, "output", "", "active NCCL runtime root")
	flags.Uint32Var(&bridgeABI, "bridge-abi", 1, "ColdSnap dlsym bridge ABI")
	return command
}

func newNCCLProviderInspectCommand() *cobra.Command {
	var provider string
	command := &cobra.Command{
		Use:   "inspect",
		Short: "Strictly parse an NCCL provider manifest without loading its code",
		Args:  cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			if provider == "" {
				return errors.New("--provider is required")
			}
			loaded, err := ncclprovider.LoadManifest(filepath.Join(provider, ncclprovider.ManifestFilename))
			if err != nil {
				return err
			}
			return writeNCCLProviderJSON(command, loaded)
		},
	}
	command.Flags().StringVar(&provider, "provider", "", "provider root containing provider-manifest.json")
	return command
}

func newNCCLProviderVerifyCommand() *cobra.Command {
	var provider string
	command := &cobra.Command{
		Use:   "verify",
		Short: "Verify every manifest-bound file and ELF identity in a provider",
		Args:  cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			if provider == "" {
				return errors.New("--provider is required")
			}
			verification, err := ncclprovider.Verify(provider)
			if err != nil {
				return err
			}
			return writeNCCLProviderJSON(command, verification)
		},
	}
	command.Flags().StringVar(&provider, "provider", "", "provider root containing provider-manifest.json")
	return command
}

func newNCCLProviderSelectCommand(materialize bool) *cobra.Command {
	var catalog, targetPath, output string
	use, short := "select", "Select exactly one provider for a target observation"
	if materialize {
		use, short = "materialize", "Atomically materialize exactly one selected provider"
	}
	command := &cobra.Command{
		Use:   use,
		Short: short,
		Args:  cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			if catalog == "" || targetPath == "" {
				return errors.New("--catalog and --target are required")
			}
			if materialize && output == "" {
				return errors.New("--output is required")
			}
			target, err := ncclprovider.LoadTarget(targetPath)
			if err != nil {
				return err
			}
			var selection ncclprovider.Selection
			if materialize {
				selection, err = ncclprovider.Materialize(catalog, target, output)
			} else {
				selection, err = ncclprovider.Select(catalog, target)
			}
			if err != nil {
				return err
			}
			return writeNCCLProviderJSON(command, selection)
		},
	}
	flags := command.Flags()
	flags.StringVar(&catalog, "catalog", "", "local provider catalog root")
	flags.StringVar(&targetPath, "target", "", "strict target-observation JSON path")
	if materialize {
		flags.StringVar(&output, "output", "", "clean provider materialization destination")
	}
	return command
}

func writeNCCLProviderJSON(command *cobra.Command, value any) error {
	encoder := json.NewEncoder(command.OutOrStdout())
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return fmt.Errorf("write NCCL provider result: %w", err)
	}
	return nil
}

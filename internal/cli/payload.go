// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package cli

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/sparksq/coldsnap/internal/payloadvalidation"
	"github.com/spf13/cobra"
)

func newPayloadCommand() *cobra.Command {
	command := &cobra.Command{
		Use: "payload", Short: "Validate immutable native payloads", Args: cobra.NoArgs,
	}
	command.AddCommand(newPayloadVerifyCommand(), newPayloadServeCommand())
	return command
}

func newPayloadVerifyCommand() *cobra.Command {
	var options payloadvalidation.Options
	var rpcSocket string
	var rpcTokenFile string
	var rpcTimeout time.Duration
	command := &cobra.Command{
		Use: "verify", Short: "Admit one immutable payload", Args: cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			var admission payloadvalidation.Admission
			var err error
			if rpcSocket == "" {
				if rpcTokenFile != "" {
					return errors.New("--rpc-token-file requires --rpc-socket")
				}
				admission, err = payloadvalidation.Validate(options)
			} else {
				if rpcTimeout <= 0 || rpcTimeout > 24*time.Hour {
					return errors.New("--rpc-timeout must be between zero and 24h")
				}
				token, tokenErr := payloadvalidation.LoadRPCToken(rpcTokenFile)
				if tokenErr != nil {
					return tokenErr
				}
				ctx, cancel := context.WithTimeout(command.Context(), rpcTimeout)
				defer cancel()
				admission, err = payloadvalidation.ValidateRPC(
					ctx, rpcSocket, token, options,
				)
			}
			if err != nil {
				return err
			}
			encoder := json.NewEncoder(command.OutOrStdout())
			encoder.SetEscapeHTML(false)
			return encoder.Encode(admission)
		},
	}
	command.Flags().StringVar(&options.Path, "path", "", "absolute native model payload path")
	command.Flags().StringVar(
		&options.Record, "record", "", "validation record path (default: canonical adjacent path)",
	)
	command.Flags().StringVar(
		&options.ExpectedSHA256, "expected-sha256", "", "expected canonical sha256:<hex> digest",
	)
	command.Flags().Int64Var(
		&options.ExpectedBytes, "expected-bytes", 0, "expected byte size; zero discovers the identity",
	)
	command.Flags().StringVar(
		&options.Worker, "worker", "", "optional execution worker identity",
	)
	command.Flags().StringVar(
		&rpcSocket, "rpc-socket", "", "private payload validation RPC Unix socket",
	)
	command.Flags().StringVar(
		&rpcTokenFile, "rpc-token-file", "", "absolute file containing the RPC token",
	)
	command.Flags().DurationVar(
		&rpcTimeout, "rpc-timeout", 30*time.Minute, "RPC validation deadline",
	)
	return command
}

func newPayloadServeCommand() *cobra.Command {
	var socket string
	var tokenFile string
	var maximumConcurrent int
	command := &cobra.Command{
		Use: "serve", Short: "Serve payload admission over a private Unix socket", Args: cobra.NoArgs,
		RunE: func(command *cobra.Command, _ []string) error {
			if socket == "" || !filepath.IsAbs(socket) || filepath.Clean(socket) != socket {
				return errors.New("--socket must be a clean absolute path")
			}
			token, err := payloadvalidation.LoadRPCToken(tokenFile)
			if err != nil {
				return err
			}
			parent := filepath.Dir(socket)
			if err := os.MkdirAll(parent, 0o700); err != nil {
				return fmt.Errorf("create payload validation RPC directory: %w", err)
			}
			information, err := os.Lstat(parent)
			if err != nil || !information.IsDir() || information.Mode()&os.ModeSymlink != 0 ||
				information.Mode().Perm()&0o077 != 0 {
				return errors.New("payload validation RPC directory must be a private real directory")
			}
			if _, err := os.Lstat(socket); err == nil {
				return errors.New("payload validation RPC socket path already exists")
			} else if !errors.Is(err, os.ErrNotExist) {
				return fmt.Errorf("inspect payload validation RPC socket path: %w", err)
			}
			address, err := net.ResolveUnixAddr("unix", socket)
			if err != nil {
				return fmt.Errorf("resolve payload validation RPC socket: %w", err)
			}
			listener, err := net.ListenUnix("unix", address)
			if err != nil {
				return fmt.Errorf("listen for payload validation RPC: %w", err)
			}
			defer listener.Close()
			listener.SetUnlinkOnClose(true)
			if err := os.Chmod(socket, 0o600); err != nil {
				return fmt.Errorf("secure payload validation RPC socket: %w", err)
			}
			fmt.Fprintf(command.ErrOrStderr(), "payload validation RPC listening on %s\n", socket)
			return (payloadvalidation.RPCServer{
				Token: token, MaxConcurrent: maximumConcurrent,
			}).Serve(command.Context(), listener)
		},
	}
	command.Flags().StringVar(&socket, "socket", "", "private Unix socket to create")
	command.Flags().StringVar(&tokenFile, "token-file", "", "absolute file containing the RPC token")
	command.Flags().IntVar(
		&maximumConcurrent, "max-concurrent", 2, "maximum concurrent payload hashes",
	)
	return command
}

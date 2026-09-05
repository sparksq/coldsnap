// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/sparksq/coldsnap/internal/adaptercli"
	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/inferenceadapter"
)

const defaultSGLangOperationTimeout = 90 * time.Minute

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	config := adaptercli.Config{
		Command: "coldsnap-sglang-adapter",
		Engine:  "sglang",
		Operations: []string{
			"capture", "publish", "publish-native", "restore", "sleep", "wake", "status",
		},
		PrepareRestore: true,
		DefaultTimeout: defaultSGLangOperationTimeout,
		NewAdapter: func(
			remote hostops.Remote,
			stateRoot string,
			timeout time.Duration,
			output io.Writer,
		) adaptercli.Adapter {
			return inferenceadapter.Adapter{
				Engine: "sglang", Remote: remote, StateRoot: stateRoot, Timeout: timeout, Output: output,
			}
		},
	}
	if err := adaptercli.Execute(ctx, config, os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, config.Command+":", err)
		os.Exit(1)
	}
}

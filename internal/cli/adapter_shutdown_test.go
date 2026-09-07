// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package cli

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestAdapterShutdownChild(t *testing.T) {
	mode := os.Getenv("COLDSNAP_SHUTDOWN_TEST_CHILD")
	if mode == "" {
		return
	}
	if mode == "stuck" {
		signal.Ignore(syscall.SIGTERM)
	} else {
		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM)
		defer stop()
		fmt.Println("ready")
		<-ctx.Done()
		time.Sleep(50 * time.Millisecond)
		if err := os.WriteFile(os.Getenv("COLDSNAP_SHUTDOWN_TEST_MARKER"), []byte("cleaned"), 0600); err != nil {
			os.Exit(2)
		}
		os.Exit(1)
	}
	fmt.Println("ready")
	for {
		time.Sleep(time.Second)
	}
}

func TestAdapterCancellationWaitsForCleanupAndBoundsStuckChild(t *testing.T) {
	for _, mode := range []string{"graceful", "stuck"} {
		t.Run(mode, func(t *testing.T) {
			marker := filepath.Join(t.TempDir(), "cleaned")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			command := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestAdapterShutdownChild$")
			command.Env = append(os.Environ(), "COLDSNAP_SHUTDOWN_TEST_CHILD="+mode, "COLDSNAP_SHUTDOWN_TEST_MARKER="+marker)
			grace := 200 * time.Millisecond
			if mode == "graceful" {
				grace = 2 * time.Second // Leave headroom for loaded native CI runners.
			}
			configureAdapterShutdown(command, grace)
			stdout, err := command.StdoutPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := command.Start(); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = command.Process.Kill() })
			ready := make(chan string, 1)
			go func() { line, _ := bufio.NewReader(stdout).ReadString('\n'); ready <- line }()
			select {
			case line := <-ready:
				if line != "ready\n" {
					t.Fatalf("child output: %q", line)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("child not ready")
			}
			cancel()
			done := make(chan error, 1)
			go func() { done <- command.Wait() }()
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("cancelled child reported success")
				}
				if mode == "stuck" {
					failure := finishEngineAdapterRun(validOperationRequest("restore"), err, nil)
					if !strings.Contains(failure.Error(), "remote cleanup is unconfirmed") {
						t.Fatalf("forced shutdown was not surfaced: %v", failure)
					}
				}
			case <-time.After(5 * time.Second):
				t.Fatal("shutdown not bounded")
			}
			_, err = os.Stat(marker)
			if (mode == "graceful") != (err == nil) {
				t.Fatalf("mode=%s cleanup marker: %v", mode, err)
			}
		})
	}
}

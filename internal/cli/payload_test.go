// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/payloadvalidation"
)

func TestPayloadVerifyCommandUsesCanonicalAdmission(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.pack")
	payload := []byte("cli payload")
	if err := os.WriteFile(path, payload, 0o400); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(payload)
	stdout := &bytes.Buffer{}
	command := NewRoot(Dependencies{
		Stdin: &bytes.Buffer{}, Stdout: stdout, Stderr: &bytes.Buffer{},
	})
	command.SetArgs([]string{
		"payload", "verify", "--path", path,
		"--expected-sha256", "sha256:" + hex.EncodeToString(digest[:]),
		"--expected-bytes", "11", "--worker", "worker-cli",
	})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	var admission payloadvalidation.Admission
	if err := json.Unmarshal(stdout.Bytes(), &admission); err != nil {
		t.Fatal(err)
	}
	if admission.Decision != "accept" || admission.Worker != "worker-cli" ||
		admission.Validation.Record != path+payloadvalidation.RecordSuffix {
		t.Fatalf("admission = %#v", admission)
	}
}

func TestPayloadCommandsServeAndUseCanonicalRPC(t *testing.T) {
	root := t.TempDir()
	if err := os.Chmod(root, 0o700); err != nil {
		t.Fatal(err)
	}
	tokenPath := filepath.Join(root, "validator.token")
	if err := os.WriteFile(tokenPath, []byte("0123456789abcdef\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	socketPath := filepath.Join(root, "validator.sock")
	serverErrors := &bytes.Buffer{}
	server := NewRoot(Dependencies{
		Stdin: &bytes.Buffer{}, Stdout: &bytes.Buffer{}, Stderr: serverErrors,
	})
	server.SetArgs([]string{
		"payload", "serve", "--socket", socketPath, "--token-file", tokenPath,
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- server.ExecuteContext(ctx) }()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if information, err := os.Lstat(socketPath); err == nil && information.Mode()&os.ModeSocket != 0 {
			break
		}
		select {
		case err := <-done:
			t.Fatalf("payload RPC server exited early: %v (%s)", err, serverErrors.String())
		default:
		}
		if time.Now().After(deadline) {
			t.Fatalf("payload RPC socket was not created (%s)", serverErrors.String())
		}
		time.Sleep(10 * time.Millisecond)
	}

	payload := []byte("RPC CLI payload")
	path := filepath.Join(root, "model.pack")
	if err := os.WriteFile(path, payload, 0o400); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(payload)
	stdout := &bytes.Buffer{}
	client := NewRoot(Dependencies{
		Stdin: &bytes.Buffer{}, Stdout: stdout, Stderr: &bytes.Buffer{},
	})
	client.SetArgs([]string{
		"payload", "verify", "--path", path,
		"--expected-sha256", "sha256:" + hex.EncodeToString(digest[:]),
		"--expected-bytes", "15", "--rpc-socket", socketPath,
		"--rpc-token-file", tokenPath, "--rpc-timeout", "5s",
	})
	if err := client.Execute(); err != nil {
		t.Fatal(err)
	}
	var admission payloadvalidation.Admission
	if err := json.Unmarshal(stdout.Bytes(), &admission); err != nil {
		t.Fatal(err)
	}
	if admission.Decision != "accept" || admission.Validation.BytesHashed != int64(len(payload)) {
		t.Fatalf("RPC admission = %#v", admission)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("payload RPC server did not stop")
	}
}

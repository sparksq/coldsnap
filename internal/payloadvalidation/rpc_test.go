// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package payloadvalidation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRPCUsesCanonicalValidatorAndRecord(t *testing.T) {
	root := t.TempDir()
	payload := []byte("payload admitted over RPC")
	path := filepath.Join(root, "model.pack")
	if err := os.WriteFile(path, payload, 0o400); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(payload)
	expected := "sha256:" + hex.EncodeToString(digest[:])
	socket := filepath.Join(root, "validator.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(socket, 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- (RPCServer{Token: "test-test-test-test", MaxConcurrent: 1}).Serve(ctx, listener)
	}()

	callContext, callCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer callCancel()
	admission, err := ValidateRPC(callContext, socket, "test-test-test-test", Options{
		Path: path, ExpectedSHA256: expected, ExpectedBytes: int64(len(payload)), Worker: "worker-2",
	})
	if err != nil {
		t.Fatal(err)
	}
	if admission.Decision != "accept" || admission.Worker != "worker-2" ||
		admission.Validation.BytesHashed != int64(len(payload)) ||
		admission.Validation.Record != path+RecordSuffix {
		t.Fatalf("admission = %#v", admission)
	}
	if _, err := os.Stat(path + RecordSuffix); err != nil {
		t.Fatalf("canonical validation record is absent: %v", err)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("payload validation RPC server did not stop")
	}
}

func TestRPCRejectsWrongToken(t *testing.T) {
	root := t.TempDir()
	socket := filepath.Join(root, "validator.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(socket, 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- (RPCServer{Token: "test-test-test-test"}).Serve(ctx, listener) }()

	callContext, callCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer callCancel()
	_, err = ValidateRPC(
		callContext, socket, "fedcba9876543210", Options{Path: filepath.Join(root, "missing")},
	)
	if err == nil || !strings.Contains(err.Error(), "authentication") {
		t.Fatalf("wrong-token error = %v", err)
	}
	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestLoadRPCTokenRequiresPrivateRegularFile(t *testing.T) {
	root := t.TempDir()
	private := filepath.Join(root, "private.token")
	if err := os.WriteFile(private, []byte("test-test-test-test\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if token, err := LoadRPCToken(private); err != nil || token != "test-test-test-test" {
		t.Fatalf("private token = %q, %v", token, err)
	}

	shared := filepath.Join(root, "shared.token")
	if err := os.WriteFile(shared, []byte("test-test-test-test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadRPCToken(shared); err == nil || !strings.Contains(err.Error(), "private") {
		t.Fatalf("shared token error = %v", err)
	}

	link := filepath.Join(root, "linked.token")
	if err := os.Symlink(private, link); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadRPCToken(link); err == nil {
		t.Fatal("symlinked token was accepted")
	}
}

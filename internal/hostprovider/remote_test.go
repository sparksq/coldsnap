// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package hostprovider

import (
	"context"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/sparksq/coldsnap/internal/hostops"
)

type providerHarness struct {
	t        *testing.T
	listener net.Listener
	token    string
	session  string
	mutex    sync.Mutex
	requests []Request
}

func newProviderHarness(t *testing.T) *providerHarness {
	t.Helper()
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	socket := filepath.Join(directory, "provider.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(socket, 0o600); err != nil {
		t.Fatal(err)
	}
	harness := &providerHarness{t: t, listener: listener, token: "secret", session: "operation-1"}
	t.Cleanup(func() { _ = listener.Close() })
	go harness.serve()
	return harness
}

func (harness *providerHarness) serve() {
	for {
		connection, err := harness.listener.Accept()
		if err != nil {
			return
		}
		go harness.handle(connection)
	}
}

func (harness *providerHarness) handle(connection net.Conn) {
	defer connection.Close()
	var request Request
	if err := json.NewDecoder(connection).Decode(&request); err != nil {
		harness.t.Errorf("decode provider request: %v", err)
		return
	}
	harness.mutex.Lock()
	harness.requests = append(harness.requests, request)
	harness.mutex.Unlock()
	response := Response{Format: ProtocolFormat, ID: request.ID, OK: true}
	if request.Token != harness.token || request.Session != harness.session {
		response.OK, response.Error = false, "unauthorized provider session"
	} else {
		switch request.Operation {
		case "capabilities":
			response.Provider = "test-provider"
			response.Capabilities = []string{
				"exec", "huggingface-publish", "huggingface-resolve", "oci-pull", "oci-push", "runtime-v1", "upload",
			}
		case "exec":
			response.Output = append([]byte("output:"), request.Input...)
			if slices.Contains(request.Arguments, "fail") {
				response.ExitCode = 17
				response.ErrorOutput = []byte("command failed")
			}
		case "huggingface-resolve":
			response.Value = "0123456789012345678901234567890123456789"
		case "runtime":
			response.Runtime = &hostops.RuntimeResponse{Value: "workload-id"}
		}
	}
	if err := json.NewEncoder(connection).Encode(response); err != nil {
		harness.t.Errorf("encode provider response: %v", err)
	}
}

func (harness *providerHarness) remote(t *testing.T) *Remote {
	t.Helper()
	remote, err := New(context.Background(), harness.listener.Addr().String(), harness.token, harness.session)
	if err != nil {
		t.Fatal(err)
	}
	return remote
}

func TestManagerProviderPreservesArgumentsInputAndFailures(t *testing.T) {
	harness := newProviderHarness(t)
	remote := harness.remote(t)
	output, err := remote.RunInput(
		context.Background(), "node-a", []byte("payload\n"), "command", "space separated", "*.json",
	)
	if err != nil || string(output) != "output:payload\n" {
		t.Fatalf("output=%q err=%v", output, err)
	}
	if _, err := remote.Run(context.Background(), "node-a", "command", "fail"); err == nil {
		t.Fatal("nonzero provider command was accepted")
	}
	harness.mutex.Lock()
	defer harness.mutex.Unlock()
	request := harness.requests[1]
	if request.Host != "node-a" || !slices.Equal(request.Arguments, []string{"command", "space separated", "*.json"}) ||
		string(request.Input) != "payload\n" {
		t.Fatalf("provider request = %#v", request)
	}
}

func TestManagerProviderExposesCredentialedAndUploadCapabilities(t *testing.T) {
	harness := newProviderHarness(t)
	remote := harness.remote(t)
	ctx := context.Background()
	if _, err := remote.Runtime(ctx, "node-a", hostops.RuntimeRequest{Action: hostops.RuntimeImagePull, Image: "registry/image@sha256:abc"}); err != nil {
		t.Fatal(err)
	}
	if _, err := remote.Runtime(ctx, "node-a", hostops.RuntimeRequest{Action: hostops.RuntimeImagePush, Image: "registry/image:tag"}); err != nil {
		t.Fatal(err)
	}
	if err := remote.Upload(ctx, "node-a", []string{"/tmp/a"}, "/tmp/b", true); err != nil {
		t.Fatal(err)
	}
	if err := remote.PublishHuggingFaceFile(
		ctx, "node-a", "capsule@sha256:abc", "org/model", "staging", "/packs/native", "worker.pack",
	); err != nil {
		t.Fatal(err)
	}
	revision, err := remote.ResolveHuggingFaceRevision(ctx, "node-a", "capsule@sha256:abc", "org/model", "staging")
	if err != nil || revision != "0123456789012345678901234567890123456789" {
		t.Fatalf("revision=%q err=%v", revision, err)
	}
}

func TestManagerProviderRoundTripsTypedRuntimeOperation(t *testing.T) {
	harness := newProviderHarness(t)
	remote := harness.remote(t)
	response, err := remote.Runtime(context.Background(), "node-a", hostops.RuntimeRequest{
		Action: hostops.RuntimeWorkloadRun,
		Workload: &hostops.WorkloadSpec{
			Name: "rank-0", Image: "registry/capsule@sha256:abc", Detached: true,
			GPUs: []string{"0"}, Network: "host", Privileged: true,
		},
	})
	if err != nil || response.Value != "workload-id" {
		t.Fatalf("response=%#v err=%v", response, err)
	}
	harness.mutex.Lock()
	defer harness.mutex.Unlock()
	request := harness.requests[len(harness.requests)-1]
	if request.Operation != "runtime" || request.Runtime == nil ||
		request.Runtime.Action != hostops.RuntimeWorkloadRun || request.Runtime.Workload.Name != "rank-0" {
		t.Fatalf("runtime request = %#v", request)
	}
}

func TestManagerProviderNegotiatesCapabilitiesPerOperation(t *testing.T) {
	harness := newProviderHarness(t)
	harness.listener.Close()
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	socket := filepath.Join(directory, "provider.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	if err := os.Chmod(socket, 0o600); err != nil {
		t.Fatal(err)
	}
	go func() {
		connection, acceptErr := listener.Accept()
		if acceptErr != nil {
			return
		}
		defer connection.Close()
		var request Request
		_ = json.NewDecoder(connection).Decode(&request)
		_ = json.NewEncoder(connection).Encode(Response{
			Format: ProtocolFormat, ID: request.ID, OK: true, Provider: "incomplete", Capabilities: []string{"exec"},
		})
	}()
	remote, err := New(context.Background(), socket, "secret", "operation-1")
	if err != nil {
		t.Fatal(err)
	}
	if err := remote.Require("exec"); err != nil {
		t.Fatal(err)
	}
	if err := remote.Require("oci-pull"); err == nil {
		t.Fatal("unadvertised operation capability was accepted")
	}
}

func TestManagerProviderRejectsNonPrivateOrSymlinkSocket(t *testing.T) {
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	socket := filepath.Join(directory, "provider.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	if err := os.Chmod(socket, 0o666); err != nil {
		t.Fatal(err)
	}
	if _, err := New(context.Background(), socket, "secret", "operation-1"); err == nil {
		t.Fatal("non-private provider socket was accepted")
	}
	if err := os.Chmod(socket, 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(directory, "provider-link.sock")
	if err := os.Symlink(socket, link); err != nil {
		t.Fatal(err)
	}
	if _, err := New(context.Background(), link, "secret", "operation-1"); err == nil {
		t.Fatal("symlink provider socket was accepted")
	}
}

func TestManagerProviderRejectsTrailingResponse(t *testing.T) {
	directory := t.TempDir()
	if err := os.Chmod(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	socket := filepath.Join(directory, "provider.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	if err := os.Chmod(socket, 0o600); err != nil {
		t.Fatal(err)
	}
	go func() {
		connection, acceptErr := listener.Accept()
		if acceptErr != nil {
			return
		}
		defer connection.Close()
		var request Request
		_ = json.NewDecoder(connection).Decode(&request)
		response := Response{
			Format: ProtocolFormat, ID: request.ID, OK: true,
			Provider: "trailing", Capabilities: []string{"exec"},
		}
		payload, _ := json.Marshal(response)
		_, _ = connection.Write(append(payload, []byte("\n{}\n")...))
	}()
	if _, err := New(context.Background(), socket, "secret", "operation-1"); err == nil {
		t.Fatal("provider response with trailing data was accepted")
	}
}

func TestManagerProviderCrossLanguageContract(t *testing.T) {
	socket := os.Getenv("COLDSNAP_TEST_HOST_PROVIDER_SOCKET")
	if socket == "" {
		t.Skip("set COLDSNAP_TEST_HOST_PROVIDER_SOCKET from a manager-provider test")
	}
	remote, err := New(
		context.Background(),
		socket,
		os.Getenv("COLDSNAP_TEST_HOST_PROVIDER_TOKEN"),
		os.Getenv("COLDSNAP_TEST_HOST_PROVIDER_SESSION"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := remote.Require("exec"); err != nil {
		t.Fatal(err)
	}
	payload := []byte("cross-language\x00payload\n")
	output, err := remote.RunInput(
		context.Background(),
		os.Getenv("COLDSNAP_TEST_HOST_PROVIDER_HOST"),
		payload,
		"contract-command",
		"space separated",
		"*.json",
		"",
	)
	if err != nil {
		t.Fatal(err)
	}
	if string(output) != "contract:"+string(payload) {
		t.Fatalf("cross-language output = %q", output)
	}
}

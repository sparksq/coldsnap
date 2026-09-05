// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package cli

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

const requestTestDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

type recordingRequestRunner struct {
	requests []snapshot.Request
	prepared []snapshot.Request
	err      error
	output   string
}

func (runner *recordingRequestRunner) PrepareRequest(
	_ context.Context,
	request snapshot.Request,
	streams Streams,
) error {
	runner.prepared = append(runner.prepared, request)
	_, _ = streams.Stdout.Write([]byte(runner.output))
	return runner.err
}

func (runner *recordingRequestRunner) RunRequest(
	_ context.Context,
	request snapshot.Request,
	streams Streams,
) error {
	runner.requests = append(runner.requests, request)
	_, _ = streams.Stdout.Write([]byte(runner.output))
	return runner.err
}

func TestReceiptStdoutIsStrictAndAdapterProgressMovesToStderr(t *testing.T) {
	runner := &recordingRequestRunner{output: "adapter progress\n"}
	request := validOperationRequest("restore")
	payload, err := snapshot.Encode(request)
	if err != nil {
		t.Fatal(err)
	}
	stdout, stderr := &bytes.Buffer{}, &bytes.Buffer{}
	command := NewRoot(Dependencies{
		RequestRunner: runner, Stdin: bytes.NewReader(payload),
		Stdout: stdout, Stderr: stderr,
	})
	command.SetArgs([]string{"restore", "--request-json", "-", "--receipt-json", "-"})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	receipt, err := snapshot.DecodeOperationReceipt(bytes.NewReader(stdout.Bytes()))
	if err != nil {
		t.Fatalf("stdout is not one strict receipt: %v\n%s", err, stdout.String())
	}
	if receipt.State != "succeeded" || receipt.OperationID != request.ID {
		t.Fatalf("receipt = %#v", receipt)
	}
	if receipt.Format != 2 || receipt.Timing.Format != 1 || len(receipt.Timing.Spans) != 1 ||
		receipt.Timing.Spans[0].Name != "controller.restore" {
		t.Fatalf("receipt timing = %#v", receipt.Timing)
	}
	expectedProgress := "ColdSnap: snapshot driver n610; running restore\n" + runner.output
	if stderr.String() != expectedProgress {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestTimingEventsStdoutStreamsNDJSONAndRetainsReceiptFile(t *testing.T) {
	runner := &recordingRequestRunner{output: "adapter progress\n"}
	request := validOperationRequest("restore")
	payload, err := snapshot.Encode(request)
	if err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Join(t.TempDir(), "operation.json")
	stdout, stderr := &bytes.Buffer{}, &bytes.Buffer{}
	command := NewRoot(Dependencies{
		RequestRunner: runner, Stdin: bytes.NewReader(payload),
		Stdout: stdout, Stderr: stderr,
	})
	command.SetArgs([]string{
		"restore", "--request-json", "-", "--receipt-json", receiptPath,
		"--timing-events", "-",
	})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := snapshot.DecodeOperationReceipt(file)
	_ = file.Close()
	if err != nil {
		t.Fatal(err)
	}
	validator := snapshot.TimingEventStreamValidator{
		OperationID: request.ID, RequestSHA256: receipt.RequestSHA256,
	}
	lines := strings.Split(strings.TrimSpace(stdout.String()), "\n")
	for _, line := range lines {
		event, err := snapshot.DecodeTimingEvent(strings.NewReader(line))
		if err != nil {
			t.Fatal(err)
		}
		if err := validator.Accept(event); err != nil {
			t.Fatalf("accept timing event %s: %v\n%s", line, err, stdout.String())
		}
		if event.Event == snapshot.TimingEventStreamComplete {
			timingDigest, err := snapshot.OperationTimingSHA256(receipt.Timing)
			if err != nil || event.TimingSHA256 != timingDigest {
				t.Fatalf("stream digest=%q receipt digest=%q error=%v", event.TimingSHA256, timingDigest, err)
			}
		}
	}
	if !validator.Complete() || !strings.Contains(stderr.String(), runner.output) {
		t.Fatalf("events complete=%v stderr=%q\n%s", validator.Complete(), stderr.String(), stdout.String())
	}
}

func TestTimingEventsRequireAnAuthoritativeReceipt(t *testing.T) {
	request := validOperationRequest("restore")
	payload, err := snapshot.Encode(request)
	if err != nil {
		t.Fatal(err)
	}
	command := NewRoot(Dependencies{
		RequestRunner: &recordingRequestRunner{}, Stdin: bytes.NewReader(payload),
		Stdout: &bytes.Buffer{}, Stderr: &bytes.Buffer{},
	})
	command.SetArgs([]string{"restore", "--request-json", "-", "--timing-events", "-"})
	if err := command.Execute(); err == nil || !strings.Contains(err.Error(), "requires --receipt-json") {
		t.Fatalf("error = %v", err)
	}
}

func TestFailedTimingEventStreamBindsFailedReceipt(t *testing.T) {
	runner := &recordingRequestRunner{err: errors.New("adapter failed")}
	request := validOperationRequest("restore")
	payload, err := snapshot.Encode(request)
	if err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Join(t.TempDir(), "operation.json")
	stdout := &bytes.Buffer{}
	command := NewRoot(Dependencies{
		RequestRunner: runner, Stdin: bytes.NewReader(payload),
		Stdout: stdout, Stderr: &bytes.Buffer{},
	})
	command.SetArgs([]string{
		"restore", "--request-json", "-", "--receipt-json", receiptPath,
		"--timing-events", "-",
	})
	if err := command.Execute(); err == nil || !strings.Contains(err.Error(), "adapter failed") {
		t.Fatalf("error = %v", err)
	}
	file, err := os.Open(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := snapshot.DecodeOperationReceipt(file)
	_ = file.Close()
	if err != nil || receipt.State != "failed" {
		t.Fatalf("receipt = %#v, error = %v", receipt, err)
	}
	validator := snapshot.TimingEventStreamValidator{
		OperationID: request.ID, RequestSHA256: receipt.RequestSHA256,
	}
	for _, line := range strings.Split(strings.TrimSpace(stdout.String()), "\n") {
		event, err := snapshot.DecodeTimingEvent(strings.NewReader(line))
		if err != nil {
			t.Fatal(err)
		}
		if err := validator.Accept(event); err != nil {
			t.Fatalf("accept timing event %s: %v", line, err)
		}
		if event.Event == snapshot.TimingEventStreamComplete && event.State != "failed" {
			t.Fatalf("completion = %#v", event)
		}
	}
	if !validator.Complete() {
		t.Fatal("failed stream did not complete")
	}
}

func TestRequestProgressAlwaysNamesSnapshotDriver(t *testing.T) {
	for _, driver := range []string{snapshotdriver.N580, snapshotdriver.N610} {
		request := validOperationRequest("restore")
		request.Driver.ID = driver
		payload, err := snapshot.Encode(request)
		if err != nil {
			t.Fatal(err)
		}
		stdout, stderr := &bytes.Buffer{}, &bytes.Buffer{}
		command := NewRoot(Dependencies{
			RequestRunner: &recordingRequestRunner{},
			Stdin:         bytes.NewReader(payload), Stdout: stdout, Stderr: stderr,
		})
		command.SetArgs([]string{"restore", "--request-json", "-"})
		if err := command.Execute(); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(stderr.String(), "snapshot driver "+driver+";") {
			t.Fatalf("%s progress = %q", driver, stderr.String())
		}
	}
}

func TestEngineAdapterTimingCrossesTheProcessBoundary(t *testing.T) {
	adapterPath := filepath.Join(t.TempDir(), "test-vllm-adapter")
	adapterScript := `#!/bin/sh
set -eu
timing_path=
timing_fd=
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--timing-json" ]; then
        timing_path=$2
        shift 2
    elif [ "$1" = "--timing-events-fd" ]; then
        timing_fd=$2
        shift 2
    else
        shift
    fi
done
test -n "$timing_path"
test "$timing_fd" = 3
sed 's/^.*$//' >/dev/null
printf '%s\n' '{"format":1,"clocks":[{"id":"engine-adapter","source":"engine-adapter","origin_unix_ns":1000000000}],"spans":[{"id":"engine-adapter-1","name":"engine_adapter.restore","clock":"engine-adapter","start_offset_seconds":0,"duration_seconds":0.25,"status":"ok"}]}' >"$timing_path"
printf '%s\n' \
    "{\"format\":1,\"kind\":\"coldsnap-timing-event\",\"sequence\":1,\"event\":\"stream_start\",\"operation_id\":\"$TEST_OPERATION_ID\",\"request_sha256\":\"$TEST_REQUEST_SHA\"}" \
    "{\"format\":1,\"kind\":\"coldsnap-timing-event\",\"sequence\":2,\"event\":\"clock\",\"operation_id\":\"$TEST_OPERATION_ID\",\"request_sha256\":\"$TEST_REQUEST_SHA\",\"clock\":{\"id\":\"engine-adapter\",\"source\":\"engine-adapter\",\"origin_unix_ns\":1000000000}}" \
    "{\"format\":1,\"kind\":\"coldsnap-timing-event\",\"sequence\":3,\"event\":\"span_start\",\"operation_id\":\"$TEST_OPERATION_ID\",\"request_sha256\":\"$TEST_REQUEST_SHA\",\"span\":{\"id\":\"engine-adapter-1\",\"name\":\"engine_adapter.restore\",\"clock\":\"engine-adapter\",\"start_offset_seconds\":0}}" \
    "{\"format\":1,\"kind\":\"coldsnap-timing-event\",\"sequence\":4,\"event\":\"span_complete\",\"operation_id\":\"$TEST_OPERATION_ID\",\"request_sha256\":\"$TEST_REQUEST_SHA\",\"span\":{\"id\":\"engine-adapter-1\",\"name\":\"engine_adapter.restore\",\"clock\":\"engine-adapter\",\"start_offset_seconds\":0,\"duration_seconds\":0.25,\"status\":\"ok\"}}" \
    "{\"format\":1,\"kind\":\"coldsnap-timing-event\",\"sequence\":5,\"event\":\"stream_complete\",\"operation_id\":\"$TEST_OPERATION_ID\",\"request_sha256\":\"$TEST_REQUEST_SHA\",\"state\":\"succeeded\",\"timing_sha256\":\"$TEST_TIMING_SHA\"}" >&3
`
	if err := os.WriteFile(adapterPath, []byte(adapterScript), 0o700); err != nil {
		t.Fatal(err)
	}
	request := validOperationRequest("restore")
	requestDigest, err := snapshot.RequestSHA256(request)
	if err != nil {
		t.Fatal(err)
	}
	adapterTiming := snapshot.OperationTiming{
		Format: snapshot.OperationTimingFormat,
		Clocks: []snapshot.TimingClock{{
			ID: "engine-adapter", Source: "engine-adapter", OriginUnixNS: 1_000_000_000,
		}},
		Spans: []snapshot.TimingSpan{{
			ID: "engine-adapter-1", Name: "engine_adapter.restore", Clock: "engine-adapter",
			DurationSeconds: 0.25, Status: "ok",
		}},
	}
	adapterTimingDigest, err := snapshot.OperationTimingSHA256(adapterTiming)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("COLDSNAP_VLLM_ADAPTER", adapterPath)
	t.Setenv("TEST_OPERATION_ID", request.ID)
	t.Setenv("TEST_REQUEST_SHA", requestDigest)
	t.Setenv("TEST_TIMING_SHA", adapterTimingDigest)
	var eventOutput bytes.Buffer
	eventWriter := operationtiming.NewNDJSONWriter(&eventOutput, request.ID, requestDigest)
	eventWriter.Start()
	collector := operationtiming.NewWithSink(time.Now(), eventWriter)
	ctx := operationtiming.WithCollector(context.Background(), collector)
	ctx, root := operationtiming.Start(ctx, "controller.restore", nil)
	if err := runEngineAdapter(ctx, request, Streams{
		Stdout: &bytes.Buffer{}, Stderr: &bytes.Buffer{},
	}, false); err != nil {
		t.Fatal(err)
	}
	root.End(nil)
	timing := collector.Export()
	timingDigest, err := snapshot.OperationTimingSHA256(timing)
	if err != nil {
		t.Fatal(err)
	}
	eventWriter.Complete("succeeded", timingDigest)
	if err := eventWriter.Error(); err != nil {
		t.Fatal(err)
	}
	if err := timing.Validate(); err != nil {
		t.Fatal(err)
	}
	spans := make(map[string]snapshot.TimingSpan, len(timing.Spans))
	for _, span := range timing.Spans {
		spans[span.Name] = span
	}
	process := spans["engine_adapter.process"]
	adapter := spans["engine_adapter.restore"]
	if process.ID == "" || adapter.ID == "" || adapter.Parent != process.ID ||
		adapter.Clock != "adapter-engine-adapter" {
		t.Fatalf("merged timing = %#v", timing)
	}
	validator := snapshot.TimingEventStreamValidator{
		OperationID: request.ID, RequestSHA256: requestDigest,
	}
	startedAdapter, completedAdapter := false, false
	for _, line := range strings.Split(strings.TrimSpace(eventOutput.String()), "\n") {
		event, err := snapshot.DecodeTimingEvent(strings.NewReader(line))
		if err != nil {
			t.Fatal(err)
		}
		if err := validator.Accept(event); err != nil {
			t.Fatalf("accept timing event %s: %v", line, err)
		}
		if event.Span != nil && event.Span.ID == adapter.ID {
			startedAdapter = startedAdapter || event.Event == snapshot.TimingEventSpanStart
			completedAdapter = completedAdapter || event.Event == snapshot.TimingEventSpanComplete
		}
	}
	if !validator.Complete() || !startedAdapter || !completedAdapter {
		t.Fatalf("adapter stream complete=%v started=%v completed=%v\n%s", validator.Complete(), startedAdapter, completedAdapter, eventOutput.String())
	}
}

func TestFailedReceiptCanBecomeSuccessfulButCannotChangeRequest(t *testing.T) {
	receiptPath := t.TempDir() + "/restore-receipt.json"
	request := validOperationRequest("restore")
	run := func(request snapshot.Request, runner *recordingRequestRunner) error {
		payload, err := snapshot.Encode(request)
		if err != nil {
			t.Fatal(err)
		}
		command := NewRoot(Dependencies{
			RequestRunner: runner, Stdin: bytes.NewReader(payload),
			Stdout: &bytes.Buffer{}, Stderr: &bytes.Buffer{},
		})
		command.SetArgs([]string{"restore", "--request-json", "-", "--receipt-json", receiptPath})
		return command.Execute()
	}
	failing := &recordingRequestRunner{err: errors.New("remote admission denied")}
	if err := run(request, failing); err == nil || !strings.Contains(err.Error(), "remote admission denied") {
		t.Fatalf("failure = %v", err)
	}
	file, err := os.Open(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := snapshot.DecodeOperationReceipt(file)
	_ = file.Close()
	if err != nil || receipt.State != "failed" {
		t.Fatalf("failed receipt = %#v, %v", receipt, err)
	}
	if err := run(request, &recordingRequestRunner{}); err != nil {
		t.Fatal(err)
	}
	file, err = os.Open(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err = snapshot.DecodeOperationReceipt(file)
	_ = file.Close()
	if err != nil || receipt.State != "succeeded" {
		t.Fatalf("successful receipt = %#v, %v", receipt, err)
	}
	if err := run(request, &recordingRequestRunner{err: errors.New("later transient failure")}); err == nil {
		t.Fatal("expected later transient failure")
	}
	file, err = os.Open(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	preserved, err := snapshot.DecodeOperationReceipt(file)
	_ = file.Close()
	if err != nil || preserved.State != "succeeded" || preserved.CompletedAt != receipt.CompletedAt {
		t.Fatalf("successful receipt was downgraded: %#v, %v", preserved, err)
	}
	different := request
	different.ID = "a-different-restore"
	differentRunner := &recordingRequestRunner{}
	if err := run(different, differentRunner); err == nil || !strings.Contains(err.Error(), "different request") {
		t.Fatalf("different request error = %v", err)
	}
	if len(differentRunner.requests) != 0 {
		t.Fatal("different request ran before receipt conflict was detected")
	}
}

func TestRequestJSONDelegatesValidatedRequest(t *testing.T) {
	runner := &recordingRequestRunner{}
	request := validOperationRequest("restore")
	payload, err := snapshot.Encode(request)
	if err != nil {
		t.Fatal(err)
	}
	command := NewRoot(Dependencies{
		RequestRunner: runner,
		Stdin:         bytes.NewReader(payload),
		Stdout:        &bytes.Buffer{},
		Stderr:        &bytes.Buffer{},
	})
	command.SetArgs([]string{"restore", "--request-json", "-"})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	if len(runner.requests) != 1 || runner.requests[0].ID != request.ID {
		t.Fatalf("unexpected requests: %#v", runner.requests)
	}
}

func TestPublishRequestJSONDelegatesValidatedRequest(t *testing.T) {
	runner := &recordingRequestRunner{}
	request := validOperationRequest("publish")
	payload, err := snapshot.Encode(request)
	if err != nil {
		t.Fatal(err)
	}
	command := NewRoot(Dependencies{
		RequestRunner: runner,
		Stdin:         bytes.NewReader(payload),
		Stdout:        &bytes.Buffer{},
		Stderr:        &bytes.Buffer{},
	})
	command.SetArgs([]string{"publish", "--request-json", "-"})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	if len(runner.requests) != 1 || runner.requests[0].Operation != "publish" {
		t.Fatalf("unexpected requests: %#v", runner.requests)
	}
}

func TestPublishNativeRequestJSONDelegatesValidatedRequest(t *testing.T) {
	runner := &recordingRequestRunner{}
	request := validOperationRequest("publish-native")
	payload, err := snapshot.Encode(request)
	if err != nil {
		t.Fatal(err)
	}
	command := NewRoot(Dependencies{
		RequestRunner: runner,
		Stdin:         bytes.NewReader(payload),
		Stdout:        &bytes.Buffer{},
		Stderr:        &bytes.Buffer{},
	})
	command.SetArgs([]string{"publish-native", "--request-json", "-"})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	if len(runner.requests) != 1 || runner.requests[0].Operation != "publish-native" {
		t.Fatalf("unexpected requests: %#v", runner.requests)
	}
}

func TestRequestJSONRejectsCommandOperationMismatch(t *testing.T) {
	runner := &recordingRequestRunner{}
	request := validOperationRequest("restore")
	payload, err := snapshot.Encode(request)
	if err != nil {
		t.Fatal(err)
	}
	command := NewRoot(Dependencies{
		RequestRunner: runner,
		Stdin:         bytes.NewReader(payload),
		Stdout:        &bytes.Buffer{},
		Stderr:        &bytes.Buffer{},
	})
	command.SetArgs([]string{"capture", "--request-json", "-"})
	if err := command.Execute(); err == nil {
		t.Fatal("expected operation mismatch")
	}
	if len(runner.requests) != 0 {
		t.Fatalf("request runner called for mismatch: %#v", runner.requests)
	}
}

func TestPrepareOnlyDelegatesWithoutRunningRestore(t *testing.T) {
	runner := &recordingRequestRunner{}
	request := validOperationRequest("restore")
	payload, err := snapshot.Encode(request)
	if err != nil {
		t.Fatal(err)
	}
	command := NewRoot(Dependencies{
		RequestRunner: runner,
		Stdin:         bytes.NewReader(payload),
		Stdout:        &bytes.Buffer{},
		Stderr:        &bytes.Buffer{},
	})
	command.SetArgs([]string{"restore", "--request-json", "-", "--prepare-only"})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	if len(runner.prepared) != 1 || len(runner.requests) != 0 {
		t.Fatalf("prepare-only delegated incorrectly: prepared=%d run=%d", len(runner.prepared), len(runner.requests))
	}
}

func TestLifecycleRequestCommandsDelegateStrictOperations(t *testing.T) {
	for _, operation := range []string{"sleep", "wake", "status"} {
		runner := &recordingRequestRunner{}
		request := validOperationRequest(operation)
		payload, err := snapshot.Encode(request)
		if err != nil {
			t.Fatal(err)
		}
		command := NewRoot(Dependencies{
			RequestRunner: runner,
			Stdin:         bytes.NewReader(payload), Stdout: &bytes.Buffer{}, Stderr: &bytes.Buffer{},
		})
		command.SetArgs([]string{operation, "--request-json", "-"})
		if err := command.Execute(); err != nil {
			t.Fatalf("%s request: %v", operation, err)
		}
		if len(runner.requests) != 1 || runner.requests[0].Operation != operation {
			t.Fatalf("%s requests = %#v", operation, runner.requests)
		}
	}
}

func validOperationRequest(operation string) snapshot.Request {
	topology, err := snapshot.NewAdapterTopology("vllm.parallel:v1", map[string]any{
		"dimensions": map[string]int{"tensor": 2},
	})
	if err != nil {
		panic(err)
	}
	request := snapshot.Request{
		Format:    snapshot.RequestFormat,
		Kind:      snapshot.RequestKind,
		Operation: operation,
		ID:        "sparkrun-qwen35-tp2-" + operation,
		Driver:    snapshotdriver.Selection{ID: snapshotdriver.N610},
		Launch: snapshot.LaunchSpec{
			Engine: "vllm",
			Model: snapshot.ModelSource{
				ID: "Qwen/Qwen3.5-0.8B", Revision: "0123456789abcdef", Source: "huggingface",
			},
			Units: []snapshot.LaunchUnit{
				{ID: "unit-a", Index: 0, Host: "node-a", Devices: []string{"0"}, Image: "org/capsule@" + requestTestDigest, ImageDigest: requestTestDigest, Command: []string{"vllm", "serve"}},
				{ID: "unit-b", Index: 1, Host: "node-b", Devices: []string{"0"}, Image: "org/capsule@" + requestTestDigest, ImageDigest: requestTestDigest, Command: []string{"vllm", "serve", "--headless"}},
			},
			Execution: snapshot.ExecutionGraph{
				Workers: []snapshot.Worker{
					{ID: "worker-a", Unit: "unit-a", Service: "model", ProcessSlot: 0, DeviceSlots: []int{0}},
					{ID: "worker-b", Unit: "unit-b", Service: "model", ProcessSlot: 0, DeviceSlots: []int{0}},
				},
				Groups:   []snapshot.ProcessGroup{{ID: "world", Kind: "torch:world", Service: "model", Members: []string{"worker-a", "worker-b"}}},
				Services: []snapshot.ServiceDomain{{ID: "model", Role: "vllm:serve", Workers: []string{"worker-a", "worker-b"}}},
				Adapter:  topology,
			},
		},
		Policy: snapshot.SnapshotPolicy{
			Process: snapshot.ProcessPolicy{Backend: "cuda-criu", KVDiscard: true, AsyncGraphs: true},
			Weights: snapshot.WeightPolicy{
				Mode: "auto",
				Recovery: snapshot.RecoveryPolicy{
					Enabled: true, Source: "huggingface-safetensors", LoaderBackend: "direct",
				},
			},
			Cache: snapshot.CachePolicy{Seed: true},
			Compatibility: snapshot.CompatibilityPolicy{
				EnforceCapturedDriverFloor: true,
				Kernel:                     snapshot.KernelCompatibilityCapability,
			},
		},
		Validation: snapshot.ValidationPolicy{HealthPath: "/health", Prompt: "hello", Expected: "world"},
	}
	if operation == "restore" {
		request.Artifact = "/opt/coldsnap/capsule/artifact.json"
	} else if operation == "sleep" || operation == "wake" || operation == "status" {
		request.Artifact = "/opt/coldsnap/capsule/artifact.json"
		request.Workload = snapshot.WorkloadIdentity{
			ClusterID: "sparkrun_deadbeef_01234567", IntentID: "deadbeef",
			Recipe: "qwen", Runtime: "vllm-distributed", Model: request.Launch.Model.ID,
		}
	} else if operation == "publish" {
		request.Artifact = "/opt/coldsnap/capsule/artifact.json"
		request.Output = "/tmp/coldsnap-published.json"
		request.Policy.Capsule = snapshot.CapsulePolicy{
			Repository: "registry.example/coldsnap/qwen",
		}
	} else if operation == "publish-native" {
		request.Artifact = "/opt/coldsnap/capsule/artifact.json"
		request.Output = "/tmp/coldsnap-native-published.json"
		request.Policy.Weights.Native = snapshot.NativePolicy{Repository: "org/qwen-native", Revision: "main"}
	} else {
		request.Output = "/tmp/coldsnap-output"
	}
	return request
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package hostprovider

import (
	"bytes"
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/hostops"
)

// The Python provider suite drives this against both a recording backend and
// an opt-in local Docker backend, exercising the actual wire contract.
func TestManagerRuntimeCrossLanguageContract(t *testing.T) {
	socket := os.Getenv("COLDSNAP_TEST_HOST_PROVIDER_SOCKET")
	if socket == "" {
		t.Skip("requires the Python manager-provider test")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	remote, err := New(ctx, socket, os.Getenv("COLDSNAP_TEST_HOST_PROVIDER_TOKEN"), os.Getenv("COLDSNAP_TEST_HOST_PROVIDER_SESSION"))
	if err != nil {
		t.Fatal(err)
	}
	host := os.Getenv("COLDSNAP_TEST_HOST_PROVIDER_HOST")
	name := os.Getenv("COLDSNAP_TEST_WORKLOAD_NAME")
	image := os.Getenv("COLDSNAP_TEST_WORKLOAD_IMAGE")
	imageInfo, err := remote.Runtime(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeImageInspect, Image: image})
	if err != nil || imageInfo.Image == nil || imageInfo.Image.ID == "" {
		t.Fatalf("image=%#v err=%v", imageInfo, err)
	}
	run, err := remote.Runtime(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadRun, Workload: &hostops.WorkloadSpec{
		Name: name, Image: image, Detached: true, PullPolicy: "never", Network: "none",
		Labels:     map[string]string{"io.sparksq.coldsnap.workload": "runtime-contract", "io.sparksq.coldsnap.rank": "0"},
		Entrypoint: "sh", Command: []string{"-c", "echo runtime-ready; echo capsule > /tmp/marker; exec sleep 60"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		cleanup, stop := context.WithTimeout(context.Background(), 10*time.Second)
		defer stop()
		if _, err := remote.Runtime(cleanup, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadRemove, Name: name}); err != nil {
			t.Error(err)
		}
	}()
	if run.Value == "" {
		t.Fatal("no workload handle")
	}
	inspection, err := remote.Runtime(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadInspect, Name: name})
	if err != nil || inspection.Workload == nil {
		t.Fatalf("inspect=%#v err=%v", inspection, err)
	}
	if inspection.Workload.ID != run.Value || !inspection.Workload.Running || inspection.Workload.Labels["sparkrun.cluster_id"] != "runtime-contract" {
		t.Fatalf("workload=%#v", inspection.Workload)
	}
	if inspection.Workload.StartedAt != "" {
		t.Fatal("default inspection leaked an opt-in field to legacy clients")
	}
	started, err := remote.Runtime(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadInspect, Name: name, IncludeStartTime: true})
	if err != nil || started.Workload == nil {
		t.Fatalf("startup inspect=%#v err=%v", started, err)
	}
	if _, err := time.Parse(time.RFC3339Nano, started.Workload.StartedAt); err != nil {
		t.Fatalf("invalid container started_at: %v", err)
	}
	payload := []byte("binary\x00input\n")
	execution, err := remote.Runtime(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadExec, Name: name, Execution: &hostops.Execution{Command: []string{"cat"}, Input: payload}})
	if err != nil || !bytes.Equal(execution.Output, payload) {
		t.Fatalf("exec=%#v err=%v", execution, err)
	}
	logs, err := remote.Runtime(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadLogs, Name: name, Tail: 5})
	if err != nil || !bytes.Contains(logs.Output, []byte("runtime-ready")) {
		t.Fatalf("logs=%#v err=%v", logs, err)
	}
	_, err = remote.Runtime(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadCopyFrom, Name: name, Path: "/tmp/missing-contract-file", Destination: os.Getenv("COLDSNAP_TEST_COPY_TARGET")})
	var failure *hostops.RuntimeError
	if !errors.As(err, &failure) || failure.Code != "path_not_found" {
		t.Fatalf("missing path error=%v", err)
	}
}

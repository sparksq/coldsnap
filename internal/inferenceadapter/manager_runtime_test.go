// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"encoding/json"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

type recordingManagerRuntime struct{ calls []hostops.RuntimeRequest }

func (runtime *recordingManagerRuntime) Runtime(_ context.Context, _ string, request hostops.RuntimeRequest) (hostops.RuntimeResponse, error) {
	runtime.calls = append(runtime.calls, request)
	switch request.Action {
	case hostops.RuntimeWorkloadRun:
		return hostops.RuntimeResponse{Value: "opaque-manager-handle"}, nil
	case hostops.RuntimeWorkloadInspect:
		return hostops.RuntimeResponse{Workload: &hostops.WorkloadInfo{ID: "opaque-manager-handle", State: "running", Running: true, Image: "capsule@digest"}}, nil
	case hostops.RuntimeImageInspect:
		return hostops.RuntimeResponse{Image: &hostops.ImageInfo{ID: "content-id"}}, nil
	case hostops.RuntimeWorkloadExec:
		return hostops.RuntimeResponse{Output: request.Execution.Input}, nil
	case hostops.RuntimeWorkloadLogs:
		return hostops.RuntimeResponse{Output: []byte("log")}, nil
	}
	return hostops.RuntimeResponse{}, nil
}

func TestAdapterWorkloadLifecycleUsesOpaqueManagerRuntime(t *testing.T) {
	backend := &recordingManagerRuntime{}
	adapter := Adapter{Runtime: backend} // Deliberately no node command transport or Docker fixture.
	ctx := context.Background()
	id, err := adapter.runWorkload(ctx, "node", &hostops.WorkloadSpec{Name: "logical-unit", Image: "capsule", Detached: true}, true)
	if err != nil || string(id) != "opaque-manager-handle" {
		t.Fatalf("launch: %s %v", id, err)
	}
	payload := []byte("payload\x00")
	output, err := adapter.execWorkloadInput(ctx, "node", payload, "logical-unit", "cat")
	if err != nil || string(output) != string(payload) {
		t.Fatalf("exec: %q %v", output, err)
	}
	if _, err = adapter.inspectWorkload(ctx, "node", "logical-unit"); err != nil {
		t.Fatal(err)
	}
	if _, err = adapter.logsWorkload(ctx, "node", "logical-unit", 10); err != nil {
		t.Fatal(err)
	}
	if _, err = adapter.copyWorkload(ctx, "node", "logical-unit", "/source", "/target"); err != nil {
		t.Fatal(err)
	}
	if _, err = adapter.inspectImage(ctx, "node", "capsule"); err != nil {
		t.Fatal(err)
	}
	actions := make([]string, len(backend.calls))
	for i, call := range backend.calls {
		actions[i] = call.Action
	}
	if !slices.Equal(actions, []string{"workload-remove", "workload-run", "workload-exec", "workload-inspect", "workload-logs", "workload-copy-from", "image-inspect"}) {
		t.Fatal(actions)
	}
}

func TestRankWorkloadSpecIsNeutralForBothEnginesAndDrivers(t *testing.T) {
	for _, engine := range []string{"vllm", "sglang"} {
		for _, driver := range []string{"n580", "n610"} {
			t.Run(engine+"/"+driver, func(t *testing.T) {
				request := validRequest(1)
				request.Driver.ID = driver
				request.Workload.ClusterID = "manager-workload"
				request.Launch.Units[0].Devices = []string{"0", "1"}
				adapter := Adapter{Engine: engine, Remote: &fakeRemote{}, StateRoot: t.TempDir()}
				spec, err := adapter.rankWorkloadSpec(context.Background(), request, request.Launch.Units[0], "capsule", "unit", "restore", "namespace", "capture", "/artifact", "/endpoint", "recovery", nil, "", nil, 0, nil)
				if err != nil {
					t.Fatal(err)
				}
				if !spec.Privileged || !spec.MemlockUnlimited || !spec.SeccompUnconfined || spec.Network != "host" || spec.SharedMemoryBytes != 32<<30 || !slices.Equal(spec.GPUs, []string{"0", "1"}) {
					t.Fatalf("lost launch requirement: %#v", spec)
				}
				if spec.Command[0] != "/usr/local/bin/coldsnap-engine-rank-"+driver {
					t.Fatal(spec.Command)
				}
				for key := range spec.Labels {
					if strings.HasPrefix(key, "sparkrun.") {
						t.Fatal("manager label leaked", key)
					}
				}
				// Every spec must remain ordinary JSON; there are no runtime CLI tokens.
				payload, err := json.Marshal(spec)
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(string(payload), "--gpus") || strings.Contains(string(payload), "docker run") {
					t.Fatal(string(payload))
				}
				if driver == snapshotdriver.N610 && spec.Labels["io.sparksq.coldsnap.workload"] != "manager-workload" {
					t.Fatal(spec.Labels)
				}
			})
		}
	}
}

func TestEngineOrchestrationCannotConstructDockerCommands(t *testing.T) {
	for _, root := range []string{".", "../capsule"} {
		entries, err := os.ReadDir(root)
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			if !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
				continue
			}
			path := filepath.Join(root, entry.Name())
			file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			ast.Inspect(file, func(node ast.Node) bool {
				value, ok := node.(*ast.BasicLit)
				if !ok || value.Kind != token.STRING {
					return true
				}
				text, _ := strconv.Unquote(value.Value)
				if text == "docker" || strings.Contains(text, "internal/runtimetest") {
					t.Errorf("runtime implementation leaked into %s: %s", path, text)
				}
				return true
			})
		}
	}
}

func TestManagerRuntimeIsRequiredWithoutDockerFallback(t *testing.T) {
	_, err := (Adapter{}).managerRuntime()
	if err == nil {
		t.Fatal("missing manager accepted")
	}
	if missingWorkloadCopySource(&hostops.RuntimeError{Code: "not_found", Message: "no workload"}) || missingWorkloadCopySource(errors.New("Could not find the file in container")) {
		t.Fatal("untyped or workload absence treated as optional cache miss")
	}
}

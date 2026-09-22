// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

type filesystemDiagnosticRemote struct {
	fakeRemote
	writeErr error
	probeErr error
	output   string
	calls    [][]string
	hosts    []string
	deadline time.Time
}

func (remote *filesystemDiagnosticRemote) Run(ctx context.Context, host string, args ...string) ([]byte, error) {
	remote.calls = append(remote.calls, slices.Clone(args))
	remote.hosts = append(remote.hosts, host)
	if args[0] == "python3" {
		remote.deadline, _ = ctx.Deadline()
		return []byte(remote.output), remote.probeErr
	}
	return nil, remote.writeErr
}

func (remote *filesystemDiagnosticRemote) RunInput(ctx context.Context, host string, _ []byte, args ...string) ([]byte, error) {
	return remote.Run(ctx, host, args...)
}

func spaceObservation(path string, available, inodes, total int64) string {
	data, _ := json.Marshal([]map[string]any{{
		"path": path, "existing_path": "/cache", "uid": 1000,
		"available_bytes": available, "available_inodes": inodes, "total_inodes": total,
	}})
	return string(data)
}

func TestFilesystemFailureDiagnostic(t *testing.T) {
	cause := errors.New("install: cannot change permissions: No such file or directory")
	for _, tc := range []struct {
		name, output, want string
		probeErr           error
	}{
		{"full-blocks", spaceObservation("/cache/new", 0, 100, 200), "no disk space", nil},
		{"negative-availability", spaceObservation("/cache/new", -1, 100, 200), "no disk space", nil},
		{"full-inodes", spaceObservation("/cache/new", 8192, 0, 200), "no free inodes", nil},
		{"both-full", spaceObservation("/cache/new", 0, 0, 200), "no disk space and no free inodes", nil},
		{"healthy", spaceObservation("/cache/new", 8192, 100, 200), "", nil},
		{"unknown-inode-pool", spaceObservation("/cache/new", 8192, 0, 0), "", nil},
		{"unrelated-path", spaceObservation("/other", 0, 0, 200), "", nil},
		{"incomplete", `[{"path":"/cache/new","existing_path":"/cache"}]`, "", nil},
		{"malformed", "not JSON", "", nil},
		{"unavailable", "", "", errors.New("python3 unavailable")},
		{"probe-timeout", "", "", context.DeadlineExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			remote := &filesystemDiagnosticRemote{writeErr: cause, output: tc.output, probeErr: tc.probeErr}
			adapter := Adapter{Remote: remote}
			err := adapter.ensurePrivateDirectories(context.Background(), "node-a", "/cache/new")
			if !errors.Is(err, cause) {
				t.Fatalf("lost original error: %v", err)
			}
			if tc.want == "" {
				if err != cause {
					t.Fatalf("unconfirmed diagnosis replaced error: %v", err)
				}
			} else if !strings.Contains(err.Error(), tc.want+" available to uid 1000 on node-a") ||
				!strings.Contains(err.Error(), "free storage and retry") ||
				!strings.Contains(err.Error(), cause.Error()) {
				t.Fatalf("missing actionable diagnostic: %v", err)
			}
			if len(remote.calls) != 2 || !slices.Equal(remote.hosts, []string{"node-a", "node-a"}) {
				t.Fatalf("unexpected probes: %v %v", remote.hosts, remote.calls)
			}
			if remote.deadline.IsZero() || time.Until(remote.deadline) > 3*time.Second {
				t.Fatal("failure probe lacks bounded deadline")
			}
		})
	}
}

func TestFilesystemDiagnosticDoesNotProbeSuccessfulOrCanceledOperations(t *testing.T) {
	for _, name := range []string{"success", "canceled", "deadline", "context-canceled"} {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var cause error
			switch name {
			case "canceled":
				cause = context.Canceled
			case "deadline":
				cause = context.DeadlineExceeded
			case "context-canceled":
				cancel()
				cause = errors.New("remote operation interrupted")
			}
			remote := &filesystemDiagnosticRemote{writeErr: cause}
			err := (Adapter{Remote: remote}).ensurePrivateDirectories(ctx, "node-a", "/cache/new")
			if err != cause || len(remote.calls) != 1 {
				t.Fatalf("error=%v calls=%v", err, remote.calls)
			}
		})
	}
}

func TestCoordinatorDirectoryExhaustionDiagnosticSharedByBothDrivers(t *testing.T) {
	for _, engine := range []string{"vllm", "sglang"} {
		for _, driver := range []string{snapshotdriver.N580, snapshotdriver.N610} {
			t.Run(engine+"/"+driver, func(t *testing.T) {
				request := validRequest(2)
				request.Launch.Engine = engine
				request.Driver.ID = driver
				cause := errors.New("install: cannot change permissions: No such file or directory")
				root := "/cache/operations/test-operation"
				remote := &filesystemDiagnosticRemote{writeErr: cause, output: spaceObservation(root, 0, 100, 200)}
				adapter := Adapter{Remote: remote, StateRoot: "/cache"}
				paths, name, err := adapter.startCoordinator(context.Background(), request, "test-operation", "")
				if !errors.Is(err, cause) || !strings.Contains(err.Error(), "create coordinator state: no disk space") ||
					paths != nil || name != "" {
					t.Fatalf("coordinator startup: %v %q %v", paths, name, err)
				}
			})
		}
	}
}

func TestFilesystemSpaceProbeResolvesAncestorAndEffectiveUserAvailability(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	root := t.TempDir()
	// The requested target is missing beneath a real symlink and contains shell
	// metacharacters. The exact-argv probe must resolve the relevant filesystem
	// without executing any portion of that path.
	target := filepath.Join(root, "mounted-cache")
	if err := os.Mkdir(target, 0700); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "cache-link")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	missing := filepath.Join(link, "missing", "op $(touch injected)")
	for _, uid := range []int{0, 1000} {
		t.Run(strconv.Itoa(uid), func(t *testing.T) {
			prefix := `import os, types
real_statvfs = os.statvfs
def simulated_statvfs(path):
    real_statvfs(path)
    return types.SimpleNamespace(f_frsize=4096, f_bfree=7, f_bavail=0, f_ffree=8, f_favail=0, f_files=100)
os.statvfs = simulated_statvfs
os.geteuid = lambda: ` + strconv.Itoa(uid) + "\n"
			output, err := exec.Command(python, "-c", prefix+filesystemSpaceProbe, missing).CombinedOutput()
			if err != nil {
				t.Fatalf("probe: %s %v", output, err)
			}
			var observations []filesystemSpace
			if err := json.Unmarshal(output, &observations); err != nil || len(observations) != 1 {
				t.Fatalf("decode: %s %v", output, err)
			}
			observed := observations[0]
			if observed.Path != missing || observed.ExistingPath != target || *observed.UID != uint32(uid) {
				t.Fatalf("wrong target/identity: %+v", observed)
			}
			wantBytes, wantInodes := int64(0), int64(0)
			if uid == 0 {
				wantBytes, wantInodes = 7*4096, 8
			}
			if *observed.AvailableBytes != wantBytes || *observed.AvailableInodes != wantInodes {
				t.Fatalf("wrong effective-user availability: %s", output)
			}
		})
	}
}

func TestStateFileWriteReportsExhaustionWithoutRetryingTheWrite(t *testing.T) {
	cause := errors.New("tee failed")
	remote := &filesystemDiagnosticRemote{writeErr: cause, output: spaceObservation("/cache/state.json", 0, 100, 200)}
	err := (Adapter{Remote: remote}).writeStateFile(context.Background(), "node-a", []byte("state"), "/cache/state.json")
	if !errors.Is(err, cause) || !strings.HasPrefix(err.Error(), "no disk space") || len(remote.calls) != 2 || remote.calls[0][0] != "tee" {
		t.Fatalf("write error=%v calls=%v", err, remote.calls)
	}
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"errors"
	"io"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

type coordinatorCleanupRemote struct {
	fakeRemote
	phase          string
	cancel         context.CancelFunc
	alive          bool
	name           string
	peer           string
	removed        []string
	endpoints      []string
	cleanupFailure bool
}

func (remote *coordinatorCleanupRemote) Run(ctx context.Context, host string, args ...string) ([]byte, error) {
	if args[0] == "rm" {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if _, ok := ctx.Deadline(); !ok {
			return nil, errors.New("cleanup lacks deadline")
		}
		remote.endpoints = append(remote.endpoints, host+":"+args[len(args)-1])
		return nil, nil
	}
	if args[0] == "id" {
		return []byte("1000"), nil
	}
	if args[0] == "cat" {
		if remote.phase == "cancel" {
			remote.cancel()
			return nil, context.Canceled
		}
		return []byte("endpoint"), nil
	}
	if host == remote.peer && args[0] == remote.phase {
		return nil, errors.New("peer staging failed")
	}
	return nil, nil
}

func (remote *coordinatorCleanupRemote) RunInput(ctx context.Context, host string, _ []byte, args ...string) ([]byte, error) {
	return remote.Run(ctx, host, args...)
}

func (remote *coordinatorCleanupRemote) Runtime(ctx context.Context, host string, request hostops.RuntimeRequest) (hostops.RuntimeResponse, error) {
	if ctx.Err() != nil {
		return hostops.RuntimeResponse{}, ctx.Err()
	}
	switch request.Action {
	case hostops.RuntimeWorkloadRun:
		remote.name, remote.alive = request.Workload.Name, true
		if remote.phase == "launch" {
			return hostops.RuntimeResponse{}, errors.New("lost launch reply")
		}
		return hostops.RuntimeResponse{Value: "test-container-id"}, nil
	case hostops.RuntimeWorkloadRemove:
		remote.removed = append(remote.removed, host+":"+request.Name)
		if remote.alive && remote.cleanupFailure {
			return hostops.RuntimeResponse{}, errors.New("removal denied")
		}
		remote.alive = false
		return hostops.RuntimeResponse{}, nil
	default:
		return hostops.RuntimeResponse{}, errors.New("unexpected runtime operation")
	}
}

func TestCoordinatorStartupFailureCleansPartialState(t *testing.T) {
	for _, engine := range []string{"vllm", "sglang"} {
		for _, driver := range []string{snapshotdriver.N580, snapshotdriver.N610} {
			for _, phase := range []string{"launch", "cancel", "timeout", "install", "tee", "chmod"} {
				t.Run(engine+"/"+driver+"/"+phase, func(t *testing.T) {
					request := validRequest(2)
					request.Launch.Engine = engine
					contract, err := snapshotdriver.Lookup(driver)
					if err != nil {
						t.Fatal(err)
					}
					request.Driver = snapshotdriver.Selection{ID: contract.ID}
					ctx, cancel := context.WithCancel(context.Background())
					defer cancel()
					remote := &coordinatorCleanupRemote{phase: phase, cancel: cancel, peer: request.Launch.Units[1].Host}
					adapter := Adapter{Remote: remote, Runtime: remote, StateRoot: "/cache/coldsnap", Timeout: time.Second, PollInterval: time.Millisecond, Output: io.Discard}
					if phase == "timeout" {
						adapter.Timeout = time.Nanosecond
					}
					paths, name, err := adapter.startCoordinator(ctx, request, "test-operation", "")
					if err == nil || name != "" || paths != nil || remote.alive {
						t.Fatalf("startup result %v %q %v alive=%v", paths, name, err, remote.alive)
					}
					if phase == "cancel" && !errors.Is(err, context.Canceled) {
						t.Fatalf("lost cancellation: %v", err)
					}
					if len(remote.removed) != 2 || len(remote.endpoints) != 2 {
						t.Fatalf("cleanup: %#v %#v", remote.removed, remote.endpoints)
					}
					for _, removed := range remote.removed {
						if removed != request.Launch.Units[0].Host+":"+remote.name {
							t.Fatalf("unrelated removal: %s", removed)
						}
					}
					for i, endpoint := range remote.endpoints {
						expected := request.Launch.Units[i].Host + ":" + filepath.Join(adapter.StateRoot, "operations/test-operation/coordinator.endpoint")
						if endpoint != expected {
							t.Fatalf("unrelated endpoint removal: %s", endpoint)
						}
					}
				})
			}
		}
	}
}

func TestCoordinatorSuccessfulStartupTransfersCleanupOwnership(t *testing.T) {
	request := validRequest(2)
	remote := &coordinatorCleanupRemote{peer: request.Launch.Units[1].Host}
	adapter := Adapter{Remote: remote, Runtime: remote, StateRoot: "/cache/coldsnap", Timeout: time.Second}
	ctx, cancel := context.WithCancel(context.Background())
	paths, name, err := adapter.startCoordinator(ctx, request, "test-operation", "")
	if err != nil || !remote.alive || len(remote.endpoints) != 0 {
		t.Fatalf("startup: %v alive=%v", err, remote.alive)
	}
	cancel()
	if err := adapter.cleanupCoordinator(ctx, request, paths, name); err != nil {
		t.Fatal(err)
	}
	if remote.alive || len(remote.endpoints) != 2 {
		t.Fatalf("incomplete cleanup: %#v", remote)
	}
}

func TestCoordinatorCleanupFailurePreservesPrimaryErrorAndIdentity(t *testing.T) {
	request := validRequest(2)
	remote := &coordinatorCleanupRemote{phase: "launch", cleanupFailure: true}
	adapter := Adapter{Remote: remote, Runtime: remote, StateRoot: "/cache/coldsnap", Timeout: time.Second}
	_, _, err := adapter.startCoordinator(context.Background(), request, "test-operation", "")
	if !errors.Is(err, errCleanupIncomplete) || !strings.Contains(err.Error(), "lost launch reply") ||
		!strings.Contains(err.Error(), remote.name) || !strings.Contains(err.Error(), "removal denied") {
		t.Fatalf("cleanup failure: %v", err)
	}
	if len(remote.endpoints) != 2 {
		t.Fatalf("endpoint cleanup not attempted: %#v", remote.endpoints)
	}
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package hostops

import (
	"context"
)

// Runtime is the manager-owned image and workload lifecycle boundary. Engine
// adapters describe the required result; a manager may realize it with Docker,
// a CRI implementation, Kubernetes, or another compatible substrate.
type Runtime interface {
	Runtime(context.Context, string, RuntimeRequest) (RuntimeResponse, error)
}

const (
	RuntimeImageInspect     = "image-inspect"
	RuntimeImagePull        = "image-pull"
	RuntimeImagePush        = "image-push"
	RuntimeImageTag         = "image-tag"
	RuntimeImageRemove      = "image-remove"
	RuntimeImageBuild       = "image-build"
	RuntimeWorkloadRun      = "workload-run"
	RuntimeWorkloadRemove   = "workload-remove"
	RuntimeWorkloadInspect  = "workload-inspect"
	RuntimeWorkloadExec     = "workload-exec"
	RuntimeWorkloadLogs     = "workload-logs"
	RuntimeWorkloadCopyFrom = "workload-copy-from"
)

type RuntimeRequest struct {
	Action           string        `json:"action"`
	Image            string        `json:"image,omitempty"`
	Source           string        `json:"source,omitempty"`
	Target           string        `json:"target,omitempty"`
	Name             string        `json:"name,omitempty"`
	Build            *ImageBuild   `json:"build,omitempty"`
	Workload         *WorkloadSpec `json:"workload,omitempty"`
	Execution        *Execution    `json:"execution,omitempty"`
	Path             string        `json:"path,omitempty"`
	Destination      string        `json:"destination,omitempty"`
	Tail             int           `json:"tail,omitempty"`
	IncludeStartTime bool          `json:"include_start_time,omitempty"`
}

type RuntimeResponse struct {
	Output   []byte        `json:"output,omitempty"`
	Value    string        `json:"value,omitempty"`
	Image    *ImageInfo    `json:"image,omitempty"`
	Workload *WorkloadInfo `json:"workload,omitempty"`
}

type ImageInfo struct {
	// ID is the OCI image-config digest (sha256:...), not a runtime-specific
	// image handle. Existing local-only capsule identities use this digest.
	ID          string   `json:"id"`
	RepoDigests []string `json:"repo_digests,omitempty"`
	Size        int64    `json:"size,omitempty"`
}

type ImageBuild struct {
	Dockerfile []byte            `json:"dockerfile"`
	Context    string            `json:"context"`
	Contexts   map[string]string `json:"contexts,omitempty"`
	Arguments  map[string]string `json:"arguments,omitempty"`
	Tag        string            `json:"tag"`
	Pull       bool              `json:"pull,omitempty"`
}

type Mount struct {
	Source   string `json:"source"`
	Target   string `json:"target"`
	ReadOnly bool   `json:"read_only,omitempty"`
}

type Device struct {
	Source string `json:"source"`
	Target string `json:"target"`
}

type WorkloadSpec struct {
	Name              string            `json:"name"`
	Image             string            `json:"image"`
	Detached          bool              `json:"detached,omitempty"`
	RemoveAfterExit   bool              `json:"remove_after_exit,omitempty"`
	PullPolicy        string            `json:"pull_policy,omitempty"`
	Network           string            `json:"network,omitempty"`
	Privileged        bool              `json:"privileged,omitempty"`
	SeccompUnconfined bool              `json:"seccomp_unconfined,omitempty"`
	MemlockUnlimited  bool              `json:"memlock_unlimited,omitempty"`
	SharedMemoryBytes int64             `json:"shared_memory_bytes,omitempty"`
	GPUs              []string          `json:"gpus,omitempty"`
	User              string            `json:"user,omitempty"`
	Entrypoint        string            `json:"entrypoint,omitempty"`
	Environment       map[string]string `json:"environment,omitempty"`
	Labels            map[string]string `json:"labels,omitempty"`
	Mounts            []Mount           `json:"mounts,omitempty"`
	Devices           []Device          `json:"devices,omitempty"`
	Command           []string          `json:"command,omitempty"`
	Input             []byte            `json:"input,omitempty"`
	Combined          bool              `json:"combined,omitempty"`
}

type Execution struct {
	Command  []string `json:"command"`
	Input    []byte   `json:"input,omitempty"`
	Combined bool     `json:"combined,omitempty"`
	User     string   `json:"user,omitempty"`
}

type WorkloadInfo struct {
	// ID is an opaque, stable workload instance identity. Managers resolve the
	// logical request name across sessions, and report serving-process state.
	ID string `json:"id"`
	// StartedAt is optional RFC3339Nano on the workload host's wall clock.
	// It must be the actual serving-container start, never inspection time.
	StartedAt string            `json:"started_at,omitempty"`
	Image     string            `json:"image,omitempty"`
	State     string            `json:"state,omitempty"`
	ExitCode  int               `json:"exit_code,omitempty"`
	Running   bool              `json:"running,omitempty"`
	Paused    bool              `json:"paused,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// RuntimeError carries machine-readable failure classification across the manager boundary.
// path_not_found concerns a file inside an existing workload; not_found concerns
// the workload/image itself. Other errors must not be treated as cache misses.
type RuntimeError struct {
	Code    string
	Message string
}

func (err *RuntimeError) Error() string { return err.Message }

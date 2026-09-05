// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package runtimetest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/sparksq/coldsnap/internal/hostops"
)

// DockerRuntime translates typed operations into historical argv for existing
// regression fakes. Only tests import this package; production runtime execution
// belongs to the manager.
type DockerRuntime struct {
	Remote  CommandRemote
	Command string
}

func NewDockerRuntime(remote CommandRemote, command string) *DockerRuntime {
	if command == "" {
		command = "docker"
	}
	return &DockerRuntime{Remote: remote, Command: command}
}

func (runtime *DockerRuntime) Runtime(ctx context.Context, host string, request hostops.RuntimeRequest) (hostops.RuntimeResponse, error) {
	if runtime == nil || runtime.Remote == nil {
		return hostops.RuntimeResponse{}, errors.New("docker runtime remote is nil")
	}
	switch request.Action {
	case hostops.RuntimeImageInspect:
		output, err := runtime.Remote.Run(ctx, host, runtime.Command, "image", "inspect", request.Image, "--format", `{"id":{{json .Id}},"repo_digests":{{json .RepoDigests}},"size":{{json .Size}}}`)
		if err != nil {
			return hostops.RuntimeResponse{}, err
		}
		var info hostops.ImageInfo
		if err := json.Unmarshal(bytes.TrimSpace(output), &info); err != nil {
			return hostops.RuntimeResponse{}, fmt.Errorf("decode Docker image inspection: %w", err)
		}
		return hostops.RuntimeResponse{Image: &info}, nil
	case hostops.RuntimeImagePull, hostops.RuntimeImagePush:
		verb := strings.TrimPrefix(request.Action, "image-")
		if verb == "pull" {
			if puller, ok := runtime.Remote.(interface {
				PullDockerImage(context.Context, string, string) error
			}); ok {
				return hostops.RuntimeResponse{}, puller.PullDockerImage(ctx, host, request.Image)
			}
		}
		if verb == "push" {
			if pusher, ok := runtime.Remote.(interface {
				PushDockerImage(context.Context, string, string) error
			}); ok {
				return hostops.RuntimeResponse{}, pusher.PushDockerImage(ctx, host, request.Image)
			}
		}
		_, err := runtime.Remote.Run(ctx, host, runtime.Command, verb, request.Image)
		return hostops.RuntimeResponse{}, err
	case hostops.RuntimeImageTag:
		_, err := runtime.Remote.Run(ctx, host, runtime.Command, "tag", request.Source, request.Target)
		return hostops.RuntimeResponse{}, err
	case hostops.RuntimeImageRemove:
		_, err := runtime.Remote.Run(ctx, host, runtime.Command, "image", "rm", request.Image)
		return hostops.RuntimeResponse{}, err
	case hostops.RuntimeImageBuild:
		if request.Build == nil {
			return hostops.RuntimeResponse{}, errors.New("docker image build specification is absent")
		}
		arguments := []string{runtime.Command, "build"}
		if !request.Build.Pull {
			arguments = append(arguments, "--pull=false")
		}
		keys := make([]string, 0, len(request.Build.Contexts))
		for key := range request.Build.Contexts {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			arguments = append(arguments, "--build-context", key+"="+request.Build.Contexts[key])
		}
		keys = keys[:0]
		for key := range request.Build.Arguments {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			arguments = append(arguments, "--build-arg", key+"="+request.Build.Arguments[key])
		}
		arguments = append(arguments, "--file", "-", "--tag", request.Build.Tag, request.Build.Context)
		_, err := runtime.Remote.RunInput(ctx, host, request.Build.Dockerfile, arguments...)
		return hostops.RuntimeResponse{}, err
	case hostops.RuntimeWorkloadRun:
		if request.Workload == nil {
			return hostops.RuntimeResponse{}, errors.New("docker workload specification is absent")
		}
		arguments, err := DockerRunArguments(runtime.Command, *request.Workload)
		if err != nil {
			return hostops.RuntimeResponse{}, err
		}
		var output []byte
		if request.Workload.Input != nil {
			output, err = runtime.Remote.RunInput(ctx, host, request.Workload.Input, arguments...)
		} else if request.Workload.Combined {
			output, err = runCombined(runtime.Remote, ctx, host, arguments...)
		} else {
			output, err = runtime.Remote.Run(ctx, host, arguments...)
		}
		if err != nil {
			return hostops.RuntimeResponse{Output: output}, err
		}
		response := hostops.RuntimeResponse{Output: output}
		if request.Workload.Detached {
			response.Value = strings.TrimSpace(string(output))
		}
		return response, nil
	case hostops.RuntimeWorkloadRemove:
		_, err := runtime.Remote.Run(ctx, host, runtime.Command, "rm", "-f", request.Name)
		return hostops.RuntimeResponse{}, err
	case hostops.RuntimeWorkloadInspect:
		output, err := runtime.Remote.Run(ctx, host, runtime.Command, "inspect", "--format", `{"id":{{json .Id}},"image":{{json .Config.Image}},"state":{{json .State.Status}},"exit_code":{{json .State.ExitCode}},"running":{{json .State.Running}},"paused":{{json .State.Paused}},"labels":{{json .Config.Labels}}}`, request.Name)
		if err != nil {
			return hostops.RuntimeResponse{}, err
		}
		var info hostops.WorkloadInfo
		if err := json.Unmarshal(bytes.TrimSpace(output), &info); err != nil {
			return hostops.RuntimeResponse{}, fmt.Errorf("decode Docker workload inspection: %w", err)
		}
		return hostops.RuntimeResponse{Workload: &info}, nil
	case hostops.RuntimeWorkloadExec:
		if request.Execution == nil || len(request.Execution.Command) == 0 {
			return hostops.RuntimeResponse{}, errors.New("docker workload execution is absent")
		}
		arguments := []string{runtime.Command, "exec"}
		if request.Execution.Input != nil {
			arguments = append(arguments, "-i")
		}
		if request.Execution.User != "" {
			arguments = append(arguments, "--user", request.Execution.User)
		}
		arguments = append(arguments, request.Name)
		arguments = append(arguments, request.Execution.Command...)
		var output []byte
		var err error
		if request.Execution.Input != nil {
			output, err = runtime.Remote.RunInput(ctx, host, request.Execution.Input, arguments...)
		} else if request.Execution.Combined {
			output, err = runCombined(runtime.Remote, ctx, host, arguments...)
		} else {
			output, err = runtime.Remote.Run(ctx, host, arguments...)
		}
		return hostops.RuntimeResponse{Output: output}, err
	case hostops.RuntimeWorkloadLogs:
		arguments := []string{runtime.Command, "logs"}
		if request.Tail > 0 {
			arguments = append(arguments, "--tail", strconv.Itoa(request.Tail))
		}
		arguments = append(arguments, request.Name)
		output, err := runCombined(runtime.Remote, ctx, host, arguments...)
		return hostops.RuntimeResponse{Output: output}, err
	case hostops.RuntimeWorkloadCopyFrom:
		_, err := runtime.Remote.Run(ctx, host, runtime.Command, "cp", request.Name+":"+request.Path, request.Destination)
		return hostops.RuntimeResponse{}, err
	default:
		return hostops.RuntimeResponse{}, fmt.Errorf("unsupported Docker runtime action %q", request.Action)
	}
}

// DockerRunArguments is kept as the deterministic Docker manager backend and
// as a useful contract oracle for tests. It is not used by engine adapters to
// communicate with managers.
func DockerRunArguments(command string, spec hostops.WorkloadSpec) ([]string, error) {
	if command == "" || spec.Image == "" {
		return nil, errors.New("docker workload command and image are required")
	}
	arguments := []string{command, "run"}
	if spec.Input != nil {
		arguments = append(arguments, "-i")
	}
	if spec.Detached {
		arguments = append(arguments, "-d")
	}
	if spec.RemoveAfterExit {
		arguments = append(arguments, "--rm")
	}
	if spec.Name != "" {
		arguments = append(arguments, "--name", spec.Name)
	}
	if spec.PullPolicy != "" {
		if !slices.Contains([]string{"always", "missing", "never"}, spec.PullPolicy) {
			return nil, fmt.Errorf("unsupported image pull policy %q", spec.PullPolicy)
		}
		arguments = append(arguments, "--pull", spec.PullPolicy)
	}
	if spec.Network != "" && spec.Network != "default" {
		if spec.Network != "host" && spec.Network != "none" {
			return nil, fmt.Errorf("unsupported workload network %q", spec.Network)
		}
		arguments = append(arguments, "--network", spec.Network)
	}
	if len(spec.GPUs) > 0 {
		selection := "device=" + strings.Join(spec.GPUs, ",")
		if len(spec.GPUs) > 1 {
			selection = `"` + selection + `"`
		}
		arguments = append(arguments, "--gpus", selection)
	}
	if spec.Privileged {
		arguments = append(arguments, "--privileged")
	}
	if spec.SeccompUnconfined {
		arguments = append(arguments, "--security-opt", "seccomp=unconfined")
	}
	if spec.MemlockUnlimited {
		arguments = append(arguments, "--ulimit", "memlock=-1:-1")
	}
	if spec.SharedMemoryBytes > 0 {
		arguments = append(arguments, "--shm-size", strconv.FormatInt(spec.SharedMemoryBytes, 10))
	}
	if spec.User != "" {
		arguments = append(arguments, "--user", spec.User)
	}
	labelKeys := sortedKeys(spec.Labels)
	for _, key := range labelKeys {
		arguments = append(arguments, "--label", key+"="+spec.Labels[key])
	}
	environmentKeys := sortedKeys(spec.Environment)
	for _, key := range environmentKeys {
		arguments = append(arguments, "-e", key+"="+spec.Environment[key])
	}
	for _, mount := range spec.Mounts {
		value := mount.Source + ":" + mount.Target
		if mount.ReadOnly {
			value += ":ro"
		}
		arguments = append(arguments, "-v", value)
	}
	for _, device := range spec.Devices {
		target := device.Target
		if target == "" {
			target = device.Source
		}
		arguments = append(arguments, "--device", device.Source+":"+target)
	}
	if spec.Entrypoint != "" {
		arguments = append(arguments, "--entrypoint", spec.Entrypoint)
	}
	arguments = append(arguments, spec.Image)
	arguments = append(arguments, spec.Command...)
	return arguments, nil
}

func sortedKeys(values map[string]string) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

type CommandRemote interface {
	Run(context.Context, string, ...string) ([]byte, error)
	RunInput(context.Context, string, []byte, ...string) ([]byte, error)
}

func runCombined(remote CommandRemote, ctx context.Context, host string, args ...string) ([]byte, error) {
	if combined, ok := remote.(interface {
		RunCombined(context.Context, string, ...string) ([]byte, error)
	}); ok {
		return combined.RunCombined(ctx, host, args...)
	}
	return remote.Run(ctx, host, args...)
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"errors"
	"strings"

	"github.com/sparksq/coldsnap/internal/hostops"
)

func (adapter Adapter) managerRuntime() (hostops.Runtime, error) {
	if adapter.Runtime != nil {
		return adapter.Runtime, nil
	}
	if runtime, ok := adapter.Remote.(hostops.Runtime); ok {
		return runtime, nil
	}
	return nil, errors.New("manager does not provide the runtime-v1 capability")
}

func (adapter Adapter) runtimeCall(
	ctx context.Context, host string, request hostops.RuntimeRequest,
) (hostops.RuntimeResponse, error) {
	runtime, err := adapter.managerRuntime()
	if err != nil {
		return hostops.RuntimeResponse{}, err
	}
	return runtime.Runtime(ctx, host, request)
}

func (adapter Adapter) runtimeBackend() hostops.Runtime {
	return adapterRuntime{adapter}
}

type adapterRuntime struct{ adapter Adapter }

func (runtime adapterRuntime) Runtime(ctx context.Context, host string, request hostops.RuntimeRequest) (hostops.RuntimeResponse, error) {
	return runtime.adapter.runtimeCall(ctx, host, request)
}

func (adapter Adapter) runWorkload(ctx context.Context, host string, spec *hostops.WorkloadSpec, replace bool) ([]byte, error) {
	if spec == nil {
		return nil, errors.New("workload specification is absent")
	}
	if replace && spec.Name != "" {
		if _, err := adapter.removeWorkload(ctx, host, spec.Name); err != nil {
			return nil, err
		}
	}
	response, err := adapter.runtimeCall(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadRun, Workload: spec})
	if err != nil {
		return response.Output, err
	}
	if spec.Detached {
		if strings.TrimSpace(response.Value) == "" {
			return nil, errors.New("manager returned no workload ID")
		}
		return []byte(response.Value), nil
	}
	return response.Output, nil
}

func (adapter Adapter) removeWorkload(ctx context.Context, host, name string) ([]byte, error) {
	_, err := adapter.runtimeCall(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadRemove, Name: name})
	return nil, err
}

func (adapter Adapter) execWorkload(ctx context.Context, host, name string, command ...string) ([]byte, error) {
	return adapter.execWorkloadInput(ctx, host, nil, name, command...)
}

func (adapter Adapter) execWorkloadInput(ctx context.Context, host string, input []byte, name string, command ...string) ([]byte, error) {
	response, err := adapter.runtimeCall(ctx, host, hostops.RuntimeRequest{
		Action: hostops.RuntimeWorkloadExec, Name: name, Execution: &hostops.Execution{Command: command, Input: input},
	})
	return response.Output, err
}

func (adapter Adapter) logsWorkload(ctx context.Context, host, name string, tail int) ([]byte, error) {
	response, err := adapter.runtimeCall(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadLogs, Name: name, Tail: tail})
	return response.Output, err
}

func (adapter Adapter) copyWorkload(ctx context.Context, host, name, path, destination string) ([]byte, error) {
	_, err := adapter.runtimeCall(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadCopyFrom, Name: name, Path: path, Destination: destination})
	return nil, err
}

func (adapter Adapter) inspectWorkload(ctx context.Context, host, name string) (hostops.WorkloadInfo, error) {
	response, err := adapter.runtimeCall(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeWorkloadInspect, Name: name})
	if err != nil {
		return hostops.WorkloadInfo{}, err
	}
	if response.Workload == nil || response.Workload.ID == "" {
		return hostops.WorkloadInfo{}, errors.New("manager returned no workload identity")
	}
	return *response.Workload, nil
}

func (adapter Adapter) workloadImage(ctx context.Context, host, name string) ([]byte, error) {
	info, err := adapter.inspectWorkload(ctx, host, name)
	return []byte(info.Image), err
}

func (adapter Adapter) inspectImage(ctx context.Context, host, image string) (hostops.ImageInfo, error) {
	response, err := adapter.runtimeCall(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeImageInspect, Image: image})
	if err != nil {
		return hostops.ImageInfo{}, err
	}
	if response.Image == nil || response.Image.ID == "" {
		return hostops.ImageInfo{}, errors.New("manager returned no image identity")
	}
	return *response.Image, nil
}

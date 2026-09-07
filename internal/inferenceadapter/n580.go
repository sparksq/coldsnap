// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/sparksq/coldsnap/internal/capsule"
	"github.com/sparksq/coldsnap/internal/fsutil"
	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

// captureN580 captures a context-free process template, then lets the original
// tree continue through a real engine load and acceptance request. The accepted
// service also emits the same recovery metadata and content-addressed model
// payload used by n610. Recovery restore follows vLLM's own loader, while a
// staged native provider bootstraps the deterministic allocation layout and
// hydrates this common payload through the capsule-local n580 address map.
func (adapter Adapter) captureN580(ctx context.Context, request snapshot.Request) (operationErr error) {
	for _, unit := range request.Launch.Units {
		if !recoveryCapableCommand(adapter.engine(), unit.Command) {
			return fmt.Errorf(
				"n580 capture unit %s command is not recovery-capable for %s",
				unit.ID, adapter.engine(),
			)
		}
	}
	if err := operationtiming.Measure(ctx, "compatibility.verify_hosts", nil, func(phase context.Context) error {
		return adapter.verifySnapshotDriverHosts(phase, request)
	}); err != nil {
		return err
	}
	if _, err := operationtiming.MeasureValue(ctx, "activation_runtime.prepare", nil, func(phase context.Context) (string, error) {
		return adapter.prepareActivationRuntime(phase, request)
	}); err != nil {
		return err
	}
	if err := operationtiming.Measure(ctx, "compatibility.features", nil, func(phase context.Context) error {
		return adapter.verifyCaptureFeatureAdmission(phase, request)
	}); err != nil {
		return err
	}
	fmt.Fprintf(adapter.Output, "ColdSnap n580: verifying versioned NCCL providers for %d launch unit(s)\n", len(request.Launch.Units))
	runtimeProviders, err := operationtiming.MeasureValue(ctx, "nccl.verify", nil, func(phase context.Context) ([]snapshot.RuntimeProviderBinding, error) {
		return adapter.verifyNCCLProviders(phase, request.Launch, nil, snapshot.RuntimeProviders{})
	})
	if err != nil {
		return err
	}
	namespace := operationName(request.ID, "capture-n580")
	fmt.Fprintf(adapter.Output, "ColdSnap n580: starting capture coordinator for %d launch unit(s) and %d worker(s)\n", len(request.Launch.Units), len(request.Launch.Execution.Workers))
	coordinatorContext, coordinatorSpan := operationtiming.Start(ctx, "coordinator.start", nil)
	endpointPaths, coordinatorName, err := adapter.startCoordinator(coordinatorContext, request, namespace, "")
	coordinatorSpan.End(err)
	if err != nil {
		return err
	}
	defer func() {
		operationErr = errors.Join(operationErr, adapter.cleanupCoordinator(ctx, request, endpointPaths, coordinatorName))
	}()

	roots := make([]string, len(request.Launch.Units))
	containers := make([]string, len(request.Launch.Units))
	ownershipNormalized := false
	defer func() {
		operationErr = errors.Join(operationErr, adapter.removeContainers(ctx, request, containers))
		if !ownershipNormalized {
			adapter.normalizeCapturePathOwnershipBestEffort(ctx, request, roots)
		}
	}()
	fmt.Fprintln(adapter.Output, "ColdSnap n580: launching context-free capture unit containers")
	if err := operationtiming.Measure(ctx, "units.launch", nil, func(phase context.Context) error {
		return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
			return operationtiming.Measure(phase, "unit.launch", map[string]string{
				"unit": unit.ID, "host": unit.Host,
			}, func(unitContext context.Context) error {
				root := filepath.Join(
					adapter.StateRoot, "captures", request.ID, "drivers", snapshotdriver.N580, "units", unit.ID,
				)
				if _, err := adapter.Remote.Run(unitContext, unit.Host, "test", "!", "-e", root); err != nil {
					return fmt.Errorf("unit %s n580 capture root already exists: %s", unit.ID, root)
				}
				if _, err := adapter.Remote.Run(unitContext, unit.Host, "install", "-d", "-m", "0700", root); err != nil {
					return fmt.Errorf("create unit %s n580 capture root: %w", unit.ID, err)
				}
				name := operationName(request.ID, "capture-n580-unit-"+unit.ID)
				roots[unit.Index], containers[unit.Index] = root, name
				command, err := adapter.rankWorkloadSpec(
					unitContext, request, unit, unit.Image, name, "capture", namespace,
					request.ID, root, endpointPaths[unit.Index], "recovery", nil,
					unit.ImageDigest, nil, 0, nil,
				)
				if err != nil {
					return err
				}
				output, err := adapter.runWorkload(unitContext, unit.Host, command, true)
				if err != nil {
					return fmt.Errorf("launch n580 capture unit %s: %w", unit.ID, err)
				}
				if strings.TrimSpace(string(output)) == "" {
					return fmt.Errorf("launch n580 capture unit %s returned no container ID", unit.ID)
				}
				return nil
			})
		})
	}); err != nil {
		adapter.collectFailureLogs(ctx, request, containers)
		return err
	}
	fmt.Fprintln(adapter.Output, "ColdSnap n580: waiting for context-free snapshots and post-capture acceptance")
	if err := operationtiming.Measure(ctx, "units.capture_ready", nil, func(phase context.Context) error {
		if err := adapter.waitCapture(phase, request, roots, containers); err != nil {
			return err
		}
		if err := adapter.collectCaptureTimingReports(phase, request, roots); err != nil {
			fmt.Fprintf(adapter.Output, "ColdSnap timing warning: %v\n", err)
		}
		return nil
	}); err != nil {
		adapter.collectFailureLogs(ctx, request, containers)
		return err
	}
	if capturePrepareOnly(request) {
		return errors.New("n580 does not support NCCL capture prepare-only mode")
	}
	if err := operationtiming.Measure(ctx, "capture.ownership", nil, func(phase context.Context) error {
		return adapter.normalizeCapturePathOwnership(phase, request, roots)
	}); err != nil {
		return err
	}
	ownershipNormalized = true
	fmt.Fprintln(adapter.Output, "ColdSnap n580: collecting derived-cache seeds")
	cacheRoots, err := operationtiming.MeasureValue(ctx, "cache.seed", nil, func(phase context.Context) ([]string, error) {
		return adapter.stageCacheSeeds(phase, request, containers)
	})
	if err != nil {
		return err
	}
	defer adapter.removeCacheSeeds(ctx, request, cacheRoots)
	fmt.Fprintln(adapter.Output, "ColdSnap n580: building driver-qualified local OCI capsules")
	artifact, err := operationtiming.MeasureValue(ctx, "capsules.construct", nil, func(phase context.Context) (snapshot.Artifact, error) {
		return adapter.constructArtifactN580(phase, request, roots, cacheRoots, runtimeProviders)
	})
	if err != nil {
		return err
	}
	var path string
	if err := operationtiming.Measure(ctx, "artifact.write", nil, func(_ context.Context) error {
		var err error
		path, err = artifactOutputPath(request.Output)
		if err != nil {
			return err
		}
		if err := fsutil.WriteJSONExclusive(path, artifact, 0o600); err != nil {
			return fmt.Errorf("write n580 ColdSnap artifact: %w", err)
		}
		return nil
	}); err != nil {
		return err
	}
	fmt.Fprintf(adapter.Output, "ColdSnap n580 artifact: %s\n", path)
	return nil
}

func (adapter Adapter) constructArtifactN580(
	ctx context.Context,
	request snapshot.Request,
	roots, cacheRoots []string,
	runtimeProviders []snapshot.RuntimeProviderBinding,
) (snapshot.Artifact, error) {
	driverContract, err := snapshotdriver.Resolve(request.Driver)
	if err != nil {
		return snapshot.Artifact{}, err
	}
	if driverContract.ID != snapshotdriver.N580 {
		return snapshot.Artifact{}, errors.New("n580 artifact construction requires the n580 driver")
	}
	if len(roots) != len(request.Launch.Units) || len(cacheRoots) != len(request.Launch.Units) {
		return snapshot.Artifact{}, errors.New("n580 capture root inventory differs from launch units")
	}
	shapeCalibration, err := adapter.capturedShapeCalibration(ctx, request, roots)
	if err != nil {
		return snapshot.Artifact{}, err
	}
	graphPolicy, err := snapshot.NewGraphPolicyRecord(request, shapeCalibration)
	if err != nil {
		return snapshot.Artifact{}, err
	}
	platforms := make([]snapshot.UnitPlatformCompatibility, len(request.Launch.Units))
	if err := parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		platform, err := adapter.capturedPlatform(ctx, unit, roots[unit.Index])
		if err != nil {
			return err
		}
		platforms[unit.Index] = platform
		return nil
	}); err != nil {
		return snapshot.Artifact{}, err
	}
	builder := capsule.Builder{Remote: adapter.Remote, Runtime: adapter.runtimeBackend()}
	results := make([]capsule.Result, len(request.Launch.Units))
	replay := make([]snapshot.Object, len(request.Launch.Execution.Workers))
	modelPayloads := make([]snapshot.Object, len(request.Launch.Execution.Workers))
	workerIndexes := make(map[string]int, len(request.Launch.Execution.Workers))
	for index, worker := range request.Launch.Execution.Workers {
		workerIndexes[worker.ID] = index
	}
	hasModelPayloads := wantsModelPayload(request.Policy.Weights)
	if err := parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		return operationtiming.Measure(ctx, "capsule.construct", map[string]string{
			"unit": unit.ID, "host": unit.Host,
		}, func(unitContext context.Context) error {
			manifests, err := adapter.hydrationManifests(unitContext, request.Launch, unit, roots[unit.Index])
			if err != nil {
				return fmt.Errorf("unit %s recovery manifests: %w", unit.ID, err)
			}
			for _, manifest := range manifests {
				workerIndex := workerIndexes[manifest.Worker]
				logicalManifest := filepath.ToSlash(filepath.Join(
					"drivers", driverContract.ID, "units", unit.ID, manifest.Relative,
				))
				replayObject, objectErr := adapter.remoteObject(
					unitContext, unit, snapshot.WorkerOwner(manifest.Worker), manifest.Path,
					"safetensors-replay-plan", logicalManifest,
				)
				if objectErr != nil {
					return objectErr
				}
				replay[workerIndex] = replayObject
				if hasModelPayloads {
					payload, err := adapter.remoteModelPayloadObject(
						unitContext, unit, snapshot.WorkerOwner(manifest.Worker),
						filepath.Join(filepath.Dir(manifest.Path), modelPayloadName),
						"pending-content-address", 0, "",
					)
					if err != nil {
						return err
					}
					payload.Path = modelPayloadObjectPath(payload.SHA256)
					modelPayloads[workerIndex] = payload
				}
			}
			result, err := builder.Build(unitContext, capsule.Spec{
				Host: unit.Host, BaseImage: unit.Image, BaseDigest: unit.ImageDigest,
				ArtifactRoot: roots[unit.Index], CacheRootFS: cacheRoots[unit.Index],
				CaptureID: request.ID, Unit: unit.ID, Driver: driverContract,
				Repository: request.Policy.Capsule.Repository,
			})
			if err != nil {
				return err
			}
			results[unit.Index] = result
			return nil
		})
	}); err != nil {
		return snapshot.Artifact{}, err
	}
	images := make([]snapshot.CapsuleImage, len(results))
	objects := make([]snapshot.Object, len(results))
	for index, result := range results {
		images[index], objects[index] = result.Image, result.Object
	}
	digest, err := snapshot.RequestSHA256(request)
	if err != nil {
		return snapshot.Artifact{}, err
	}
	artifact := snapshot.Artifact{
		Format: snapshot.ArtifactFormat, Kind: snapshot.ArtifactKind, State: "committed",
		CaptureID: request.ID, RequestSHA256: digest, Driver: driverContract,
		Requires: driverContract.BaseRequirements.Clone(),
		Launch:   request.Launch,
		Compatibility: snapshot.ArtifactCompatibility{
			Policy: snapshot.PortabilityPolicy, Units: platforms,
		},
		Capsule:          snapshot.Capsule{Images: images, Objects: objects},
		Runtime:          snapshot.RuntimeProviders{Bindings: runtimeProviders},
		ShapeCalibration: shapeCalibration,
		Graph:            graphPolicy,
		Weights: snapshot.WeightProviders{Recovery: snapshot.RecoveryProvider{
			Source: "huggingface-safetensors", ModelID: request.Launch.Model.ID,
			Revision: request.Launch.Model.Revision, Driver: driverContract.Binding(),
			LoadPath: processTemplateRecoveryLoadPath(adapter.engine()), ReplayPlan: replay,
		}},
		Acceptance: snapshot.ArtifactAcceptance{
			Accepted: true, Expected: request.Validation.Expected,
		},
	}
	if hasModelPayloads {
		artifact.Weights.ModelPayloads = &snapshot.ModelPayloadProvider{
			Repository: request.Policy.Weights.Native.Repository,
			Revision:   request.Policy.Weights.Native.Revision,
			Objects:    modelPayloads,
		}
		artifact.Weights.Native = &snapshot.NativeProvider{Driver: driverContract.Binding()}
	}
	if err := snapshot.ValidateArtifact(artifact); err != nil {
		return snapshot.Artifact{}, fmt.Errorf("constructed n580 artifact: %w", err)
	}
	return artifact, nil
}

func (adapter Adapter) verifySnapshotDriverHosts(ctx context.Context, request snapshot.Request) error {
	contract, err := snapshotdriver.Resolve(request.Driver)
	if err != nil {
		return err
	}
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		output, err := adapter.Remote.Run(
			ctx, unit.Host, "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits",
		)
		if err != nil {
			return fmt.Errorf("unit %s NVIDIA driver probe: %w", unit.ID, err)
		}
		lines := strings.Fields(string(output))
		if len(lines) < len(unit.Devices) {
			return fmt.Errorf("unit %s NVIDIA driver inventory is incomplete", unit.ID)
		}
		for _, value := range lines {
			majorText := strings.SplitN(value, ".", 2)[0]
			major, parseErr := strconv.Atoi(majorText)
			if parseErr != nil || major < contract.MinimumNVIDIADriverMajor {
				return fmt.Errorf(
					"unit %s NVIDIA driver %q does not satisfy snapshot driver %s minimum %d",
					unit.ID, value, contract.ID, contract.MinimumNVIDIADriverMajor,
				)
			}
		}
		return nil
	})
}

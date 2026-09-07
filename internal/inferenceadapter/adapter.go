// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net/netip"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sparksq/coldsnap/internal/capsule"
	"github.com/sparksq/coldsnap/internal/fsutil"
	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/ncclprovider"
	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

const (
	DefaultOperationTimeout         = 45 * time.Minute
	containerArtifactRoot           = capsule.Root
	containerRuntimeCacheRoot       = snapshot.CanonicalRuntimeCachePath
	containerCudaCachePath          = containerRuntimeCacheRoot + "/cuda"
	containerCaptureLogPath         = capsule.Root + "/capture.log"
	containerTargetLogPath          = capsule.Root + "/target.log"
	containerModelPayloadRoot       = "/var/cache/coldsnap/model-payloads"
	containerMaterializationControl = "/run/coldsnap/model-payload-materialization.json"
	coordinatorEndpoint             = "/run/coldsnap/coordinator.endpoint"
	modelPayloadName                = "model-weights.pack"
	defaultCRIUBlockBytes           = 256 * 1024
	defaultCRIUAcceleration         = 1
	defaultCRIUThreads              = 1
	defaultCRIUImageIOMode          = "direct"
	ncclProviderKind                = "collective:nccl"
	ncclProviderSchema              = "nccl:provider-v1"
	ncclActiveRuntimePath           = "/opt/coldsnap/nccl/active.json"
	restoreNCCLRetryLimit           = 1
)

var huggingFaceCommitPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)

type Adapter struct {
	Engine       string
	Remote       hostops.Remote
	Runtime      hostops.Runtime
	StateRoot    string
	Timeout      time.Duration
	PollInterval time.Duration
	Output       io.Writer
}

func (adapter Adapter) engine() string {
	if adapter.Engine == "" {
		return "vllm"
	}
	return adapter.Engine
}

type authenticatedNativePublisher interface {
	PublishHuggingFaceFile(context.Context, string, string, string, string, string, string) error
	ResolveHuggingFaceRevision(context.Context, string, string, string, string) (string, error)
}

type RestorePreparationReceipt struct {
	Format          int                     `json:"format"`
	Kind            string                  `json:"kind"`
	OperationID     string                  `json:"operation_id"`
	CaptureID       string                  `json:"capture_id"`
	Provider        string                  `json:"provider"`
	Reason          string                  `json:"reason"`
	Images          []snapshot.CapsuleImage `json:"images"`
	Driver          snapshotdriver.Binding  `json:"snapshot_driver"`
	Materialization string                  `json:"materialization,omitempty"`
}

type restorePreparation struct {
	artifact            snapshot.Artifact
	selection           snapshot.Selection
	tcpAddressMap       []string
	materialize         []snapshot.Object
	materializationMode string
}

type materializationMount struct {
	controlPath string
	cacheRoot   string
}

type materializationControl struct {
	Format      int                              `json:"format"`
	Kind        string                           `json:"kind"`
	OperationID string                           `json:"operation_id"`
	Mode        string                           `json:"mode"`
	OwnerUID    int64                            `json:"owner_uid"`
	OwnerGID    int64                            `json:"owner_gid"`
	Workers     map[string]materializationTarget `json:"workers"`
}

type materializationTarget struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

func (adapter Adapter) Run(ctx context.Context, request snapshot.Request) (operationErr error) {
	ctx, operationSpan := operationtiming.Start(ctx, "adapter."+request.Operation, map[string]string{
		"engine": request.Launch.Engine, "snapshot_driver": request.Driver.ID,
	})
	defer func() { operationSpan.End(operationErr) }()
	configured, err := adapter.configured(request)
	if err != nil {
		return err
	}
	adapter = configured
	switch request.Operation {
	case "capture":
		driver, driverErr := resolveProcessDriver(request.Driver)
		if driverErr != nil {
			return driverErr
		}
		return driver.Capture(ctx, adapter, request)
	case "publish":
		return adapter.publish(ctx, request)
	case "publish-native":
		return adapter.publishNative(ctx, request)
	case "restore":
		driver, driverErr := resolveProcessDriver(request.Driver)
		if driverErr != nil {
			return driverErr
		}
		return driver.Restore(ctx, adapter, request)
	case "sleep", "wake", "status":
		return adapter.runLifecycle(ctx, request)
	default:
		return fmt.Errorf("unsupported %s adapter operation %q", adapter.Engine, request.Operation)
	}
}

func (adapter Adapter) publishNative(ctx context.Context, request snapshot.Request) error {
	var artifact snapshot.Artifact
	if err := operationtiming.Measure(ctx, "artifact.read_verify", nil, func(_ context.Context) error {
		path, err := artifactInputPath(request.Artifact)
		if err != nil {
			return err
		}
		artifact, err = snapshot.ReadArtifact(path)
		if err != nil {
			return err
		}
		if err := validateArtifactDriver(request, artifact); err != nil {
			return err
		}
		return compatibleLaunch(artifact.Launch, request.Launch)
	}); err != nil {
		return err
	}
	if artifact.Weights.ModelPayloads == nil {
		return errors.New("accepted artifact has no capture-local model payloads")
	}
	if len(artifact.Weights.ModelPayloads.Objects) != len(request.Launch.Execution.Workers) {
		return errors.New("accepted artifact model payload inventory is incomplete")
	}
	if _, err := operationtiming.MeasureValue(ctx, "activation_runtime.prepare", nil, func(phase context.Context) (string, error) {
		return adapter.prepareActivationRuntime(phase, request)
	}); err != nil {
		return err
	}
	if err := operationtiming.Measure(ctx, "capsules.prepare", nil, func(phase context.Context) error {
		return adapter.prepareCapsuleImages(phase, request, artifact.Capsule.Images)
	}); err != nil {
		return fmt.Errorf("prepare model payload publication helpers: %w", err)
	}
	paths := make(map[string]string, len(request.Launch.Execution.Workers))
	payloads := make(map[string]snapshot.Object, len(request.Launch.Execution.Workers))
	for _, object := range artifact.Weights.ModelPayloads.Objects {
		workerID, ok := strings.CutPrefix(object.Owner, snapshot.WorkerOwnerPrefix)
		if !ok || payloads[workerID].Role != "" {
			return errors.New("accepted artifact model payload inventory is invalid")
		}
		payloads[workerID] = object
	}
	if err := operationtiming.Measure(ctx, "native.verify_local", nil, func(phase context.Context) error {
		for _, unit := range request.Launch.Units {
			captured := artifact.Launch.Units[unit.Index]
			if captured.Host != unit.Host {
				return fmt.Errorf(
					"unit %s capture-local model payloads belong to host %s, not %s",
					unit.ID, captured.Host, unit.Host,
				)
			}
			root := filepath.Join(
				adapter.StateRoot, "captures", artifact.CaptureID,
				"drivers", artifact.Driver.ID, "units", unit.ID,
			)
			manifests, manifestErr := adapter.hydrationManifests(phase, request.Launch, unit, root)
			if manifestErr != nil {
				return fmt.Errorf("locate capture-local model payloads for unit %s: %w", unit.ID, manifestErr)
			}
			for _, manifest := range manifests {
				expected := payloads[manifest.Worker]
				packPath := filepath.Join(filepath.Dir(manifest.Path), modelPayloadName)
				observed, objectErr := adapter.remoteModelPayloadObject(
					phase, unit, snapshot.WorkerOwner(manifest.Worker), packPath,
					expected.Path, expected.Bytes, expected.SHA256,
				)
				if objectErr != nil {
					return objectErr
				}
				if observed.Bytes != expected.Bytes || observed.SHA256 != expected.SHA256 {
					return fmt.Errorf("capture-local worker %s model payload changed after acceptance", manifest.Worker)
				}
				paths[manifest.Worker] = packPath
			}
		}
		return nil
	}); err != nil {
		return err
	}
	revision, err := operationtiming.MeasureValue(ctx, "native.publish", nil, func(phase context.Context) (string, error) {
		return adapter.publishModelPayloads(phase, request, paths, payloads, artifact.Capsule.Images)
	})
	if err != nil {
		return err
	}
	artifact.Weights.ModelPayloads.Repository = request.Policy.Weights.Native.Repository
	artifact.Weights.ModelPayloads.Revision = revision
	if err := snapshot.ValidateArtifact(artifact); err != nil {
		return fmt.Errorf("published native artifact: %w", err)
	}
	var output string
	if err := operationtiming.Measure(ctx, "artifact.write", nil, func(_ context.Context) error {
		var err error
		output, err = artifactOutputPath(request.Output)
		if err != nil {
			return err
		}
		if err := fsutil.WriteJSONExclusive(output, artifact, 0o600); err != nil {
			return fmt.Errorf("write native-published ColdSnap artifact: %w", err)
		}
		return nil
	}); err != nil {
		return err
	}
	fmt.Fprintf(adapter.Output, "ColdSnap model payloads published: %s@%s\n", artifact.Weights.ModelPayloads.Repository, revision)
	fmt.Fprintf(adapter.Output, "ColdSnap native-published artifact: %s\n", output)
	return nil
}

func (adapter Adapter) publish(ctx context.Context, request snapshot.Request) error {
	var artifact snapshot.Artifact
	var imagesByUnit map[string]snapshot.CapsuleImage
	if err := operationtiming.Measure(ctx, "artifact.read_verify", nil, func(_ context.Context) error {
		path, err := artifactInputPath(request.Artifact)
		if err != nil {
			return err
		}
		artifact, err = snapshot.ReadArtifact(path)
		if err != nil {
			return err
		}
		if err := validateArtifactDriver(request, artifact); err != nil {
			return err
		}
		if err := compatibleLaunch(artifact.Launch, request.Launch); err != nil {
			return err
		}
		imagesByUnit, err = capsuleImagesByUnit(artifact.Capsule.Images)
		return err
	}); err != nil {
		return err
	}
	for _, unit := range request.Launch.Units {
		captured := artifact.Launch.Units[unit.Index]
		image := imagesByUnit[unit.ID]
		if captured.Host != unit.Host {
			return fmt.Errorf(
				"unit %s local OCI capsule belongs to capture host %s, not %s",
				unit.ID, captured.Host, unit.Host,
			)
		}
		if image.Reference != image.Digest {
			return fmt.Errorf("unit %s OCI capsule is already registry-backed: %s", unit.ID, image.Reference)
		}
	}

	published := make([]snapshot.CapsuleImage, len(request.Launch.Units))
	builder := capsule.Builder{Remote: adapter.Remote, Runtime: adapter.runtimeBackend()}
	if err := operationtiming.Measure(ctx, "capsules.publish", nil, func(phase context.Context) error {
		return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
			return operationtiming.Measure(phase, "capsule.publish", map[string]string{
				"unit": unit.ID, "host": unit.Host,
			}, func(unitContext context.Context) error {
				image, err := builder.Publish(unitContext, capsule.PublishSpec{
					Host: unit.Host, Source: imagesByUnit[unit.ID].Digest,
					CaptureID: artifact.CaptureID, Unit: unit.ID,
					Driver:     artifact.Driver,
					Repository: request.Policy.Capsule.Repository,
				})
				if err != nil {
					return err
				}
				published[unit.Index] = image
				return nil
			})
		})
	}); err != nil {
		return err
	}
	artifact.Capsule.Images = published
	seenObjects := make(map[string]bool, len(request.Launch.Units))
	for index := range artifact.Capsule.Objects {
		object := &artifact.Capsule.Objects[index]
		if object.Role != "oci-capsule" {
			continue
		}
		unitID, ok := strings.CutPrefix(object.Owner, snapshot.UnitOwnerPrefix)
		if !ok || seenObjects[unitID] {
			return errors.New("artifact OCI capsule object inventory is invalid")
		}
		unit, ok := launchUnit(request.Launch, unitID)
		if !ok {
			return errors.New("artifact OCI capsule object owner is unknown")
		}
		object.SHA256 = published[unit.Index].Digest
		seenObjects[unitID] = true
	}
	if len(seenObjects) != len(request.Launch.Units) {
		return errors.New("artifact OCI capsule object inventory is incomplete")
	}
	if err := snapshot.ValidateArtifact(artifact); err != nil {
		return fmt.Errorf("published artifact: %w", err)
	}
	var output string
	if err := operationtiming.Measure(ctx, "artifact.write", nil, func(_ context.Context) error {
		var err error
		output, err = artifactOutputPath(request.Output)
		if err != nil {
			return err
		}
		if err := fsutil.WriteJSONExclusive(output, artifact, 0o600); err != nil {
			return fmt.Errorf("write published ColdSnap artifact: %w", err)
		}
		return nil
	}); err != nil {
		return err
	}
	fmt.Fprintf(adapter.Output, "ColdSnap published artifact: %s\n", output)
	return nil
}

// PrepareRestore verifies every prerequisite that can fail before a serving
// workload is evicted. It is intentionally safe to repeat: a subsequent Run
// rechecks the committed descriptor and resident capsule identities.
func (adapter Adapter) PrepareRestore(
	ctx context.Context, request snapshot.Request, receiptOutput io.Writer,
) (operationErr error) {
	ctx, operationSpan := operationtiming.Start(ctx, "adapter.restore.prepare", map[string]string{
		"engine": request.Launch.Engine, "snapshot_driver": request.Driver.ID,
	})
	defer func() { operationSpan.End(operationErr) }()
	configured, err := adapter.configured(request)
	if err != nil {
		return err
	}
	adapter = configured
	if request.Operation != "restore" {
		return errors.New("prepare-only is supported only for restore requests")
	}
	driver, err := resolveProcessDriver(request.Driver)
	if err != nil {
		return err
	}
	prepared, err := driver.PrepareRestore(ctx, adapter, request)
	if err != nil {
		return err
	}
	receipt := RestorePreparationReceipt{
		Format: 1, Kind: "coldsnap-restore-preparation", OperationID: request.ID,
		CaptureID: prepared.artifact.CaptureID, Provider: prepared.selection.Provider,
		Reason: prepared.selection.Reason, Images: prepared.artifact.Capsule.Images,
		Driver:          prepared.artifact.Driver.Binding(),
		Materialization: prepared.materializationMode,
	}
	if receiptOutput == nil {
		return errors.New("restore preparation receipt output is nil")
	}
	if err := json.NewEncoder(receiptOutput).Encode(receipt); err != nil {
		return fmt.Errorf("write restore preparation receipt: %w", err)
	}
	return nil
}

func (adapter Adapter) configured(request snapshot.Request) (Adapter, error) {
	if adapter.Engine == "" {
		adapter.Engine = "vllm"
	}
	if adapter.Engine != "vllm" && adapter.Engine != "sglang" {
		return Adapter{}, fmt.Errorf("unsupported ColdSnap engine %q", adapter.Engine)
	}
	if request.Launch.Engine != adapter.Engine {
		return Adapter{}, fmt.Errorf("coldsnap-%s-adapter cannot run engine %q", adapter.Engine, request.Launch.Engine)
	}
	if adapter.Remote == nil {
		return Adapter{}, fmt.Errorf("%s adapter remote transport is nil", adapter.Engine)
	}
	if adapter.StateRoot == "" {
		adapter.StateRoot = "/var/lib/coldsnap"
	}
	if !filepath.IsAbs(adapter.StateRoot) || filepath.Clean(adapter.StateRoot) != adapter.StateRoot {
		return Adapter{}, errors.New("ColdSnap remote state root must be an absolute clean path")
	}
	if adapter.Timeout == 0 {
		adapter.Timeout = DefaultOperationTimeout
	}
	if adapter.PollInterval == 0 {
		adapter.PollInterval = 250 * time.Millisecond
	}
	runtime, err := adapter.managerRuntime()
	if err != nil {
		return Adapter{}, err
	}
	adapter.Runtime = runtime
	if adapter.Output == nil {
		adapter.Output = io.Discard
	}
	return adapter, nil
}

func (adapter Adapter) captureN610(ctx context.Context, request snapshot.Request) error {
	if err := operationtiming.Measure(ctx, "compatibility.verify_hosts", nil, func(phase context.Context) error {
		return adapter.verifySnapshotDriverHosts(phase, request)
	}); err != nil {
		return err
	}
	for _, unit := range request.Launch.Units {
		if !recoveryCapableCommand(adapter.engine(), unit.Command) {
			return fmt.Errorf(
				"capture unit %s command is not recovery-capable for %s",
				unit.ID, adapter.engine(),
			)
		}
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
	fmt.Fprintf(adapter.Output, "ColdSnap: verifying versioned NCCL providers for %d launch unit(s)\n", len(request.Launch.Units))
	runtimeProviders, err := operationtiming.MeasureValue(ctx, "nccl.verify", nil, func(phase context.Context) ([]snapshot.RuntimeProviderBinding, error) {
		return adapter.verifyNCCLProviders(phase, request.Launch, nil, snapshot.RuntimeProviders{})
	})
	if err != nil {
		return err
	}
	namespace := operationName(request.ID, "capture")
	fmt.Fprintf(adapter.Output, "ColdSnap: starting capture coordinator for %d launch unit(s) and %d worker(s)\n", len(request.Launch.Units), len(request.Launch.Execution.Workers))
	coordinatorContext, coordinatorSpan := operationtiming.Start(ctx, "coordinator.start", nil)
	endpointPaths, coordinatorName, err := adapter.startCoordinator(coordinatorContext, request, namespace, "")
	coordinatorSpan.End(err)
	if err != nil {
		return err
	}
	defer adapter.cleanupCoordinator(ctx, request, endpointPaths, coordinatorName)

	roots := make([]string, len(request.Launch.Units))
	containers := make([]string, len(request.Launch.Units))
	ownershipNormalized := false
	defer func() {
		adapter.removeContainers(ctx, request, containers)
		if !ownershipNormalized {
			adapter.normalizeCapturePathOwnershipBestEffort(ctx, request, roots)
		}
	}()
	fmt.Fprintf(adapter.Output, "ColdSnap: launching capture unit containers\n")
	if err := operationtiming.Measure(ctx, "units.launch", nil, func(phase context.Context) error {
		return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
			return operationtiming.Measure(phase, "unit.launch", map[string]string{
				"unit": unit.ID, "host": unit.Host,
			}, func(unitContext context.Context) error {
				root := filepath.Join(
					adapter.StateRoot, "captures", request.ID, "drivers", request.Driver.ID, "units", unit.ID,
				)
				if _, err := adapter.Remote.Run(unitContext, unit.Host, "test", "!", "-e", root); err != nil {
					return fmt.Errorf("unit %s capture root already exists: %s", unit.ID, root)
				}
				if _, err := adapter.Remote.Run(unitContext, unit.Host, "install", "-d", "-m", "0700", root); err != nil {
					return fmt.Errorf("create unit %s capture root: %w", unit.ID, err)
				}
				name := operationName(request.ID, "capture-unit-"+unit.ID)
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
					return fmt.Errorf("launch capture unit %s: %w", unit.ID, err)
				}
				if strings.TrimSpace(string(output)) == "" {
					return fmt.Errorf("launch capture unit %s returned no container ID", unit.ID)
				}
				return nil
			})
		})
	}); err != nil {
		adapter.collectFailureLogs(ctx, request, containers)
		return err
	}
	fmt.Fprintf(adapter.Output, "ColdSnap: waiting for %s warmup and process snapshot\n", adapter.engine())
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
	if err := operationtiming.Measure(ctx, "capture.ownership", nil, func(phase context.Context) error {
		return adapter.normalizeCapturePathOwnership(phase, request, roots)
	}); err != nil {
		return err
	}
	ownershipNormalized = true
	if capturePrepareOnly(request) {
		fmt.Fprintf(adapter.Output, "ColdSnap NCCL prepare-only reports: %s\n", strings.Join(roots, ", "))
		return nil
	}

	fmt.Fprintf(adapter.Output, "ColdSnap: collecting derived-cache seeds\n")
	cacheRoots, err := operationtiming.MeasureValue(ctx, "cache.seed", nil, func(phase context.Context) ([]string, error) {
		return adapter.stageCacheSeeds(phase, request, containers)
	})
	if err != nil {
		return err
	}
	defer adapter.removeCacheSeeds(ctx, request, cacheRoots)
	fmt.Fprintf(adapter.Output, "ColdSnap: building local OCI capsules\n")
	artifact, err := operationtiming.MeasureValue(ctx, "capsules.construct", nil, func(phase context.Context) (snapshot.Artifact, error) {
		return adapter.constructArtifact(phase, request, roots, cacheRoots, runtimeProviders)
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
			return fmt.Errorf("write ColdSnap artifact: %w", err)
		}
		return nil
	}); err != nil {
		return err
	}
	fmt.Fprintf(adapter.Output, "ColdSnap artifact: %s\n", path)
	return nil
}

func capturePrepareOnly(request snapshot.Request) bool {
	if len(request.Launch.Units) == 0 {
		return false
	}
	for _, rank := range request.Launch.Units {
		if rank.Environment["COLDSNAP_CAPTURE_PREPARE_ONLY"] != "1" {
			return false
		}
	}
	return true
}

type transientNCCLRestoreError struct {
	cause error
}

func (failure *transientNCCLRestoreError) Error() string { return failure.cause.Error() }
func (failure *transientNCCLRestoreError) Unwrap() error { return failure.cause }

func (adapter Adapter) restore(ctx context.Context, request snapshot.Request) error {
	for attempt := 0; ; attempt++ {
		err := operationtiming.Measure(ctx, "restore.attempt", map[string]string{
			"attempt": strconv.Itoa(attempt + 1),
		}, func(phase context.Context) error {
			return adapter.restoreAttempt(phase, request)
		})
		if err == nil {
			return nil
		}
		var transient *transientNCCLRestoreError
		if attempt >= restoreNCCLRetryLimit || !errors.As(err, &transient) {
			return err
		}
		fmt.Fprintf(
			adapter.Output,
			"ColdSnap: transient NCCL initialization failure; cleaned up the failed launch and retrying restore (%d/%d)\n",
			attempt+1,
			restoreNCCLRetryLimit,
		)
		if err := waitContext(ctx, time.Second); err != nil {
			return err
		}
	}
}

func (adapter Adapter) restoreAttempt(ctx context.Context, request snapshot.Request) error {
	prepared, err := operationtiming.MeasureValue(ctx, "restore.prepare", nil, func(phase context.Context) (restorePreparation, error) {
		return adapter.prepareRestore(phase, request)
	})
	if err != nil {
		return err
	}
	artifact, selection := prepared.artifact, prepared.selection

	namespace := operationName(request.ID, fmt.Sprintf("restore-%d", time.Now().UnixNano()))
	materializationMounts, err := operationtiming.MeasureValue(ctx, "materialization.prepare", nil, func(phase context.Context) (map[string]*materializationMount, error) {
		return adapter.prepareModelPayloadMaterialization(phase, request, namespace, prepared)
	})
	if err != nil {
		return err
	}
	imagesByUnit, err := capsuleImagesByUnit(artifact.Capsule.Images)
	if err != nil {
		return err
	}
	coordinatorContext, coordinatorSpan := operationtiming.Start(ctx, "coordinator.start", nil)
	endpointPaths, coordinatorName, err := adapter.startCoordinator(
		coordinatorContext, request, namespace, artifact.Capsule.Images[0].Reference,
	)
	coordinatorSpan.End(err)
	if err != nil {
		return err
	}
	// The restored service no longer consults the coordinator after every rank
	// has published restore-ready, so successful completion stops it below.
	defer adapter.cleanupCoordinator(ctx, request, endpointPaths, coordinatorName)
	portShift := uint16(0)
	if len(prepared.tcpAddressMap) != 0 {
		portShift, err = operationtiming.MeasureValue(ctx, "network.port_select", nil, func(phase context.Context) (uint16, error) {
			return adapter.selectPortableTCPPortShift(
				phase, request, artifact.Launch, imagesByUnit, prepared.tcpAddressMap, namespace,
			)
		})
		if err != nil {
			return err
		}
	}
	nativeByWorker := preparedPayloadsByWorker(selection.ModelPayloads)
	containers := make([]string, len(request.Launch.Units))
	if err := operationtiming.Measure(ctx, "units.launch", nil, func(phase context.Context) error {
		return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
			return operationtiming.Measure(phase, "unit.launch", map[string]string{
				"unit": unit.ID, "host": unit.Host,
			}, func(unitContext context.Context) error {
				image, ok := imagesByUnit[unit.ID]
				if !ok {
					return fmt.Errorf("capsule image inventory lacks unit %s", unit.ID)
				}
				name := restoreContainerName(request, unit.Index)
				var payloads []snapshot.PreparedPayload
				if selection.Provider == "native" {
					for _, worker := range request.Launch.Execution.UnitWorkers(unit.ID) {
						pack, found := nativeByWorker[worker.ID]
						if !found {
							return fmt.Errorf("model payload selection lacks worker %s", worker.ID)
						}
						payloads = append(payloads, pack)
					}
				}
				command, err := adapter.rankWorkloadSpec(
					unitContext, request, unit, image.Reference, name, "restore", namespace,
					artifact.CaptureID, capsule.Root, endpointPaths[unit.Index], selection.Provider, payloads,
					artifact.Launch.Units[unit.Index].ImageDigest, prepared.tcpAddressMap, portShift,
					materializationMounts[unit.ID],
				)
				if err != nil {
					return err
				}
				if _, err := adapter.runWorkload(unitContext, unit.Host, command, true); err != nil {
					return fmt.Errorf("launch restore unit %s: %w", unit.ID, err)
				}
				containers[unit.Index] = name
				return nil
			})
		})
	}); err != nil {
		adapter.collectFailureLogs(ctx, request, containers)
		adapter.removeContainers(ctx, request, containers)
		return err
	}
	if err := operationtiming.Measure(ctx, "units.ready", nil, func(phase context.Context) error {
		if err := adapter.waitRestore(phase, request, containers); err != nil {
			return err
		}
		if err := adapter.collectRestoreTimingReports(phase, request, containers); err != nil {
			fmt.Fprintf(adapter.Output, "ColdSnap timing warning: %v\n", err)
		}
		return nil
	}); err != nil {
		adapter.collectFailureLogs(ctx, request, containers)
		adapter.removeContainers(ctx, request, containers)
		return err
	}
	if prepared.materializationMode == "required" && lifecycleActivationState(request) != "warm" {
		if err := operationtiming.Measure(ctx, "materialization.verify", nil, func(phase context.Context) error {
			return adapter.verifyMaterializedModelPayloads(phase, request, prepared.materialize)
		}); err != nil {
			adapter.collectFailureLogs(ctx, request, containers)
			adapter.removeContainers(ctx, request, containers)
			return err
		}
	}
	if err := operationtiming.Measure(ctx, "workload.logs", nil, func(phase context.Context) error {
		return adapter.exposeWorkloadLogs(phase, request, containers)
	}); err != nil {
		adapter.collectFailureLogs(ctx, request, containers)
		adapter.removeContainers(ctx, request, containers)
		return err
	}
	state := lifecycleActivationState(request)
	if artifact.Driver.ID == snapshotdriver.N580 {
		if err := operationtiming.Measure(ctx, "lifecycle.synchronize", nil, func(phase context.Context) error {
			return adapter.synchronizeRestoreLifecycleEvidence(phase, request, artifact, containers, state)
		}); err != nil {
			adapter.collectFailureLogs(ctx, request, containers)
			adapter.removeContainers(ctx, request, containers)
			return err
		}
	}
	if err := operationtiming.Measure(ctx, "lifecycle.persist", nil, func(phase context.Context) error {
		return adapter.writeRestoreLifecycleState(phase, request, artifact, containers, state)
	}); err != nil {
		adapter.collectFailureLogs(ctx, request, containers)
		adapter.removeContainers(ctx, request, containers)
		return err
	}
	if state == "warm" {
		fmt.Fprintf(adapter.Output, "ColdSnap warm with %s weights pending hydration: %s\n", selection.Provider, selection.Reason)
	} else {
		fmt.Fprintf(adapter.Output, "ColdSnap restored with %s weights: %s\n", selection.Provider, selection.Reason)
	}
	return nil
}

func (adapter Adapter) prepareModelPayloadMaterialization(
	ctx context.Context,
	request snapshot.Request,
	namespace string,
	prepared restorePreparation,
) (map[string]*materializationMount, error) {
	result := make(map[string]*materializationMount)
	if len(prepared.materialize) == 0 {
		return result, nil
	}
	objectsByWorker := make(map[string]snapshot.Object, len(prepared.materialize))
	for _, object := range prepared.materialize {
		worker, ok := strings.CutPrefix(object.Owner, snapshot.WorkerOwnerPrefix)
		if !ok || objectsByWorker[worker].Role != "" {
			return nil, errors.New("model payload materialization inventory is invalid")
		}
		objectsByWorker[worker] = object
	}
	cacheRoot := filepath.Join(adapter.StateRoot, "model-payloads")
	controlRoot := filepath.Join(adapter.StateRoot, "operations", namespace, "materialization")
	for _, unit := range request.Launch.Units {
		if _, err := adapter.Remote.Run(
			ctx, unit.Host, "install", "-d", "-m", "0700",
			cacheRoot, filepath.Join(cacheRoot, "sha256"), controlRoot,
		); err != nil {
			return nil, fmt.Errorf("prepare unit %s model payload cache: %w", unit.ID, err)
		}
		ownerOutput, err := adapter.Remote.Run(ctx, unit.Host, "stat", "-c", "%u:%g", cacheRoot)
		if err != nil {
			return nil, fmt.Errorf("resolve unit %s model payload cache owner: %w", unit.ID, err)
		}
		owner := strings.SplitN(strings.TrimSpace(string(ownerOutput)), ":", 2)
		if len(owner) != 2 {
			return nil, fmt.Errorf("unit %s model payload cache owner is invalid", unit.ID)
		}
		ownerUID, uidErr := strconv.ParseInt(owner[0], 10, 64)
		ownerGID, gidErr := strconv.ParseInt(owner[1], 10, 64)
		if uidErr != nil || gidErr != nil || ownerUID < 0 || ownerGID < 0 {
			return nil, fmt.Errorf("unit %s model payload cache owner is invalid", unit.ID)
		}
		control := materializationControl{
			Format: 1, Kind: "coldsnap-model-payload-materialization",
			OperationID: request.ID, Mode: prepared.materializationMode,
			OwnerUID: ownerUID, OwnerGID: ownerGID,
			Workers: make(map[string]materializationTarget),
		}
		for _, worker := range request.Launch.Execution.UnitWorkers(unit.ID) {
			object, ok := objectsByWorker[worker.ID]
			if !ok {
				return nil, fmt.Errorf("model payload materialization lacks worker %s", worker.ID)
			}
			control.Workers[worker.ID] = materializationTarget{
				Path:  strings.TrimPrefix(object.Path, "model-payloads/"),
				Bytes: object.Bytes, SHA256: object.SHA256,
			}
		}
		payload, err := json.MarshalIndent(control, "", "  ")
		if err != nil {
			return nil, fmt.Errorf("encode unit %s model payload materialization: %w", unit.ID, err)
		}
		payload = append(payload, '\n')
		controlPath := filepath.Join(controlRoot, unit.ID+".json")
		if _, err := adapter.Remote.RunInput(ctx, unit.Host, payload, "tee", controlPath); err != nil {
			return nil, fmt.Errorf("write unit %s model payload materialization: %w", unit.ID, err)
		}
		if _, err := adapter.Remote.Run(ctx, unit.Host, "chmod", "0600", controlPath); err != nil {
			return nil, fmt.Errorf("protect unit %s model payload materialization: %w", unit.ID, err)
		}
		result[unit.ID] = &materializationMount{controlPath: controlPath, cacheRoot: cacheRoot}
	}
	return result, nil
}

func (adapter Adapter) verifyMaterializedModelPayloads(
	ctx context.Context, request snapshot.Request, objects []snapshot.Object,
) error {
	objectsByWorker := make(map[string]snapshot.Object, len(objects))
	for _, object := range objects {
		worker, ok := strings.CutPrefix(object.Owner, snapshot.WorkerOwnerPrefix)
		if !ok {
			return errors.New("materialized model payload owner is invalid")
		}
		objectsByWorker[worker] = object
	}
	return parallelWorkers(request.Launch, func(unit snapshot.LaunchUnit, worker snapshot.Worker) error {
		object, ok := objectsByWorker[worker.ID]
		if !ok {
			return fmt.Errorf("materialized model payload lacks worker %s", worker.ID)
		}
		path := filepath.Join(adapter.StateRoot, filepath.FromSlash(object.Path))
		record := path + ".coldsnap-validation.json"
		output, err := adapter.runPayloadVerifier(
			ctx, unit.Host, path, record, object.SHA256, object.Bytes, worker.ID,
		)
		if err != nil {
			return fmt.Errorf("verify worker %s materialized model payload: %w", worker.ID, err)
		}
		var admitted remotePayloadAdmission
		if err := json.Unmarshal(output, &admitted); err != nil || !admitted.valid() || admitted.Bytes != object.Bytes ||
			admitted.SHA256 != object.SHA256 || admitted.Validation.Provider != "sha256-cache-v1" {
			return fmt.Errorf("worker %s materialized model payload validation is invalid", worker.ID)
		}
		statusOutput, statusErr := adapter.Remote.Run(
			ctx,
			unit.Host,
			"cat",
			filepath.Join(adapter.StateRoot, "model-payloads", ".status", worker.ID+".json"),
		)
		if statusErr == nil {
			if err := importMaterializationTiming(ctx, unit, worker, object, statusOutput); err != nil {
				return err
			}
		}
		return nil
	})
}

func importMaterializationTiming(
	ctx context.Context,
	unit snapshot.LaunchUnit,
	worker snapshot.Worker,
	object snapshot.Object,
	payload []byte,
) error {
	var status struct {
		State         string             `json:"state"`
		WorkerID      string             `json:"worker_id"`
		Path          string             `json:"path"`
		SHA256        string             `json:"sha256"`
		Bytes         int64              `json:"bytes"`
		StartedUnixNS int64              `json:"started_unix_ns"`
		Seconds       float64            `json:"seconds"`
		PhaseSeconds  map[string]float64 `json:"phase_seconds"`
	}
	if err := json.Unmarshal(payload, &status); err != nil {
		return fmt.Errorf("decode worker %s materialization timing: %w", worker.ID, err)
	}
	if status.State != "ready" || status.WorkerID != worker.ID ||
		status.SHA256 != object.SHA256 || status.Bytes != object.Bytes {
		return fmt.Errorf("worker %s materialization status does not match the committed payload", worker.ID)
	}
	if status.StartedUnixNS <= 0 || status.Seconds < 0 || math.IsNaN(status.Seconds) || math.IsInf(status.Seconds, 0) {
		// A capsule captured before detailed materialization telemetry remains
		// usable; its content validation is authoritative and only its optional
		// phase spans are unavailable.
		return nil
	}
	collector := operationtiming.FromContext(ctx)
	if collector == nil {
		return nil
	}
	clockID := "materialization-" + worker.ID
	collector.AddClock(snapshot.TimingClock{
		ID: clockID, Source: "runtime-worker", OriginUnixNS: status.StartedUnixNS,
		Unit: unit.ID, Worker: worker.ID,
	})
	attributes := map[string]string{
		"unit": unit.ID, "worker": worker.ID, "host": unit.Host,
		"measurement": "runtime-wall",
	}
	totalID := clockID + "-total"
	collector.AddSpan(snapshot.TimingSpan{
		ID: totalID, Parent: operationtiming.ParentID(ctx), Name: "runtime.materialization",
		Clock: clockID, StartOffsetSeconds: 0, DurationSeconds: status.Seconds,
		Status: "ok", Attributes: attributes,
	})
	names := map[string]string{
		"canonical_fingerprint_s":   "runtime.materialization.canonicalize",
		"cuda_copy_s":               "runtime.materialization.gpu_to_host",
		"checksum_s":                "runtime.materialization.checksum",
		"disk_write_s":              "runtime.materialization.write_service",
		"preallocate_s":             "runtime.materialization.preallocate",
		"sync_s":                    "runtime.materialization.fdatasync",
		"payload_verification_s":    "runtime.materialization.post_write_verify",
		"ownership_normalization_s": "runtime.materialization.ownership",
		"validation_record_s":       "runtime.materialization.validation_publish",
		"existing_validation_s":     "runtime.materialization.existing_validation",
	}
	for raw, name := range names {
		duration, ok := status.PhaseSeconds[raw]
		if !ok || duration < 0 || math.IsNaN(duration) || math.IsInf(duration, 0) {
			continue
		}
		phaseAttributes := maps.Clone(attributes)
		phaseAttributes["measurement"] = "cumulative-phase"
		collector.AddSpan(snapshot.TimingSpan{
			ID: totalID + "-" + strings.TrimSuffix(raw, "_s"), Parent: totalID, Name: name,
			Clock: clockID, StartOffsetSeconds: 0, DurationSeconds: duration,
			Status: "ok", Attributes: phaseAttributes,
		})
	}
	return nil
}

func (adapter Adapter) exposeWorkloadLogs(
	ctx context.Context, request snapshot.Request, containers []string,
) error {
	if request.Workload.LogPath == "" {
		return nil
	}
	return parallelUnits(request.Launch.Units, func(rank snapshot.LaunchUnit) error {
		container := containers[rank.Index]
		if container == "" {
			return fmt.Errorf("expose rank %d workload log: restore container is empty", rank.Index)
		}
		if _, err := adapter.execWorkload(
			ctx, rank.Host, container, "test", "-f", containerTargetLogPath,
		); err != nil {
			return fmt.Errorf("expose rank %d workload log: target is unavailable: %w", rank.Index, err)
		}
		if _, err := adapter.execWorkload(
			ctx, rank.Host, container,
			"ln", "-sfnT", containerTargetLogPath, request.Workload.LogPath,
		); err != nil {
			return fmt.Errorf("expose rank %d workload log: %w", rank.Index, err)
		}
		resolved, err := adapter.execWorkload(
			ctx, rank.Host, container, "readlink", request.Workload.LogPath,
		)
		if err != nil {
			return fmt.Errorf("verify rank %d workload log: %w", rank.Index, err)
		}
		if strings.TrimSpace(string(resolved)) != containerTargetLogPath {
			return fmt.Errorf("verify rank %d workload log: unexpected target %q", rank.Index, strings.TrimSpace(string(resolved)))
		}
		return nil
	})
}

func (adapter Adapter) prepareRestore(ctx context.Context, request snapshot.Request) (restorePreparation, error) {
	var artifact snapshot.Artifact
	if err := operationtiming.Measure(ctx, "artifact.read_verify", nil, func(_ context.Context) error {
		path, err := artifactInputPath(request.Artifact)
		if err != nil {
			return err
		}
		artifact, err = snapshot.ReadArtifact(path)
		if err != nil {
			return err
		}
		if err := validateArtifactDriver(request, artifact); err != nil {
			return err
		}
		return compatibleLaunch(artifact.Launch, request.Launch)
	}); err != nil {
		return restorePreparation{}, err
	}
	if err := operationtiming.Measure(ctx, "compatibility.verify", nil, func(phase context.Context) error {
		return adapter.verifyPortablePlatforms(phase, request, artifact.Compatibility)
	}); err != nil {
		return restorePreparation{}, err
	}
	// Both qualified process drivers capture initialized distributed TCP
	// endpoints. n580's boundary is pre-CUDA, not pre-network. Even an identity
	// placement needs a fresh port generation so repeated restores cannot
	// collide with sockets left in TIME_WAIT.
	tcpAddressMap, err := operationtiming.MeasureValue(ctx, "network.address_map", nil, func(_ context.Context) ([]string, error) {
		return placementTCPAddressMap(artifact.Launch, request.Launch)
	})
	if err != nil {
		return restorePreparation{}, err
	}
	inventory := snapshot.ProviderInventory{ModelPayloads: make(map[string]snapshot.PreparedPayload)}
	for _, pack := range request.Policy.Weights.Native.Staged {
		inventory.ModelPayloads[pack.Worker] = pack
	}
	selection, err := operationtiming.MeasureValue(ctx, "weights.select", nil, func(_ context.Context) (snapshot.Selection, error) {
		return snapshot.SelectWeights(artifact, request.Policy.Weights.Mode, inventory)
	})
	if err != nil {
		return restorePreparation{}, err
	}
	if selection.FetchRequired {
		return restorePreparation{}, fmt.Errorf("model payloads must be staged before invoking the %s adapter", adapter.engine())
	}
	_, err = operationtiming.MeasureValue(ctx, "activation_runtime.prepare", nil, func(phase context.Context) (string, error) {
		return adapter.prepareActivationRuntime(phase, request)
	})
	if err != nil {
		return restorePreparation{}, err
	}
	if selection.Provider == "native" {
		if err := operationtiming.Measure(ctx, "weights.native_verify", nil, func(phase context.Context) error {
			return adapter.verifyStagedPayloads(phase, request, selection.ModelPayloads)
		}); err != nil {
			return restorePreparation{}, err
		}
	}
	materializationMode, err := engineMaterializationMode(
		adapter.engine(), request.Policy.Weights.Native.Materialize,
	)
	if err != nil {
		return restorePreparation{}, err
	}
	materialize, materializationMode, err := modelPayloadMaterialization(artifact, selection, materializationMode)
	if err != nil {
		return restorePreparation{}, err
	}
	if err := operationtiming.Measure(ctx, "capsules.prepare", nil, func(phase context.Context) error {
		return adapter.prepareCapsuleImages(phase, request, artifact.Capsule.Images)
	}); err != nil {
		return restorePreparation{}, err
	}
	imagesByUnit, err := capsuleImagesByUnit(artifact.Capsule.Images)
	if err != nil {
		return restorePreparation{}, err
	}
	if err := operationtiming.Measure(ctx, "compatibility.criu", nil, func(phase context.Context) error {
		return adapter.verifyCRIUCapabilities(phase, request, imagesByUnit)
	}); err != nil {
		return restorePreparation{}, err
	}
	if err := operationtiming.Measure(ctx, "compatibility.features", nil, func(phase context.Context) error {
		return adapter.verifyHostFeatureProfiles(phase, request, imagesByUnit, artifact.Requires)
	}); err != nil {
		return restorePreparation{}, err
	}
	if _, err := operationtiming.MeasureValue(ctx, "nccl.verify", nil, func(phase context.Context) ([]snapshot.RuntimeProviderBinding, error) {
		return adapter.verifyNCCLProviders(phase, request.Launch, imagesByUnit, artifact.Runtime)
	}); err != nil {
		return restorePreparation{}, fmt.Errorf("verify prepared capsule NCCL providers: %w", err)
	}
	return restorePreparation{
		artifact: artifact, selection: selection, tcpAddressMap: tcpAddressMap,
		materialize: materialize, materializationMode: func() string {
			if len(materialize) == 0 {
				return ""
			}
			return materializationMode
		}(),
	}, nil
}

func modelPayloadMaterialization(
	artifact snapshot.Artifact, selection snapshot.Selection, mode string,
) ([]snapshot.Object, string, error) {
	if mode == "" {
		mode = "off"
	}
	if selection.Provider != "recovery" || mode == "off" {
		return nil, "", nil
	}
	if artifact.Weights.Native == nil || artifact.Weights.ModelPayloads == nil {
		if mode == "required" {
			return nil, "", errors.New(
				"native materialization is required, but the artifact has no native model-payload inventory; " +
					"capture with weights mode auto or native before materializing",
			)
		}
		return nil, "", nil
	}
	return slices.Clone(artifact.Weights.ModelPayloads.Objects), mode, nil
}

func engineMaterializationMode(engine, requested string) (string, error) {
	if requested == "" {
		return "off", nil
	}
	if engine == "sglang" && requested != "off" {
		return "", fmt.Errorf(
			"SGLang does not support native model-payload materialization mode %q; use off or pre-publish native payloads",
			requested,
		)
	}
	return requested, nil
}

func (adapter Adapter) verifyPortablePlatforms(
	ctx context.Context, request snapshot.Request, compatibility snapshot.ArtifactCompatibility,
) error {
	if err := compatibility.Validate(request.Launch); err != nil {
		return fmt.Errorf("portable compatibility policy: %w", err)
	}
	driverContract, err := snapshotdriver.Resolve(request.Driver)
	if err != nil {
		return err
	}
	compatibilityByUnit := make(map[string]snapshot.UnitPlatformCompatibility, len(compatibility.Units))
	for _, platform := range compatibility.Units {
		compatibilityByUnit[platform.Unit] = platform
	}
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		captured := compatibilityByUnit[unit.ID]
		architecture, err := adapter.Remote.Run(ctx, unit.Host, "uname", "-m")
		if err != nil {
			return fmt.Errorf("unit %s architecture probe: %w", unit.ID, err)
		}
		kernel, err := adapter.Remote.Run(ctx, unit.Host, "uname", "-r")
		if err != nil {
			return fmt.Errorf("unit %s kernel probe: %w", unit.ID, err)
		}
		if strings.TrimSpace(string(architecture)) != captured.Architecture {
			return fmt.Errorf("unit %s process architecture differs from capture", unit.ID)
		}
		currentKernel := strings.TrimSpace(string(kernel))
		if currentKernel != captured.Kernel {
			if request.Policy.Compatibility.Kernel == snapshot.KernelCompatibilityExact {
				return fmt.Errorf(
					"unit %s kernel %s differs from captured %s under exact policy",
					unit.ID, currentKernel, captured.Kernel,
				)
			}
			output := adapter.Output
			if output == nil {
				output = io.Discard
			}
			fmt.Fprintf(
				output,
				"ColdSnap: unit %s kernel %s differs from captured %s; using capsule-pinned CRIU capability admission\n",
				unit.ID, currentKernel, captured.Kernel,
			)
		}
		var capturedMinimumDriver []int
		if request.Policy.Compatibility.EnforceCapturedDriverFloor {
			capturedMinimumDriver, err = numericVersion(captured.NVIDIADriverMin)
			if err != nil {
				return fmt.Errorf("unit %s captured NVIDIA driver: %w", unit.ID, err)
			}
		}
		for slot, selector := range unit.Devices {
			gpu, probeErr := adapter.Remote.Run(
				ctx, unit.Host, "nvidia-smi", "--id="+selector,
				"--query-gpu=driver_version,name,compute_cap", "--format=csv,noheader,nounits",
			)
			if probeErr != nil {
				return fmt.Errorf("unit %s device slot %d NVIDIA compatibility probe: %w", unit.ID, slot, probeErr)
			}
			fields := strings.Split(strings.TrimSpace(string(gpu)), ",")
			if len(fields) != 3 {
				return fmt.Errorf("unit %s device slot %d NVIDIA compatibility probe returned invalid output", unit.ID, slot)
			}
			for index := range fields {
				fields[index] = strings.TrimSpace(fields[index])
			}
			device := captured.Devices[slot]
			if fields[1] != device.GPUName || fields[2] != device.ComputeCapability {
				return fmt.Errorf("unit %s device slot %d GPU model or CUDA compute capability differs from capture", unit.ID, slot)
			}
			currentDriver, versionErr := numericVersion(fields[0])
			if versionErr != nil {
				return fmt.Errorf("unit %s device slot %d current NVIDIA driver: %w", unit.ID, slot, versionErr)
			}
			if currentDriver[0] < driverContract.MinimumNVIDIADriverMajor {
				return fmt.Errorf(
					"unit %s device slot %d NVIDIA driver %s does not satisfy snapshot driver %s minimum %d",
					unit.ID, slot, fields[0], driverContract.ID, driverContract.MinimumNVIDIADriverMajor,
				)
			}
			if request.Policy.Compatibility.EnforceCapturedDriverFloor &&
				compareNumericVersions(currentDriver, capturedMinimumDriver) < 0 {
				return fmt.Errorf(
					"unit %s device slot %d NVIDIA driver %s is older than captured minimum %s",
					unit.ID, slot, fields[0], captured.NVIDIADriverMin,
				)
			}
		}
		return nil
	})
}

func numericVersion(value string) ([]int, error) {
	parts := strings.Split(value, ".")
	if value == "" || len(parts) == 0 {
		return nil, errors.New("version is empty")
	}
	result := make([]int, len(parts))
	for index, part := range parts {
		if part == "" {
			return nil, fmt.Errorf("invalid numeric version %q", value)
		}
		component, err := strconv.Atoi(part)
		if err != nil || component < 0 {
			return nil, fmt.Errorf("invalid numeric version %q", value)
		}
		result[index] = component
	}
	return result, nil
}

func compareNumericVersions(left, right []int) int {
	length := max(len(left), len(right))
	for index := 0; index < length; index++ {
		var leftValue, rightValue int
		if index < len(left) {
			leftValue = left[index]
		}
		if index < len(right) {
			rightValue = right[index]
		}
		if leftValue < rightValue {
			return -1
		}
		if leftValue > rightValue {
			return 1
		}
	}
	return 0
}

func (adapter Adapter) prepareCapsuleImages(ctx context.Context, request snapshot.Request, images []snapshot.CapsuleImage) error {
	imagesByUnit, err := capsuleImagesByUnit(images)
	if err != nil {
		return err
	}
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		attributes := map[string]string{"unit": unit.ID, "host": unit.Host}
		return operationtiming.Measure(ctx, "capsule.prepare", attributes, func(unitContext context.Context) error {
			image, ok := imagesByUnit[unit.ID]
			if !ok {
				return fmt.Errorf("unit %s capsule image is missing", unit.ID)
			}
			resident, _ := operationtiming.MeasureValue(
				unitContext, "capsule.inspect", attributes,
				func(inspectContext context.Context) (bool, error) {
					_, inspectErr := adapter.inspectImage(inspectContext, unit.Host, image.Reference)
					// Absence is the normal signal to enter the pull path, not a
					// failed restore phase. Preserve the existing behavior where any
					// inspect failure is retried through a registry pull.
					return inspectErr == nil, nil
				},
			)
			if resident {
				return nil
			}
			if image.Reference == image.Digest {
				return fmt.Errorf("unit %s local-only capsule %s is not resident", unit.ID, image.Reference)
			}
			if err := operationtiming.Measure(unitContext, "capsule.pull", attributes, func(pullContext context.Context) error {
				_, pullErr := adapter.runtimeCall(pullContext, unit.Host, hostops.RuntimeRequest{Action: hostops.RuntimeImagePull, Image: image.Reference})
				return pullErr
			}); err != nil {
				return fmt.Errorf("pull unit %s capsule %s: %w", unit.ID, image.Reference, err)
			}
			if err := operationtiming.Measure(unitContext, "capsule.verify", attributes, func(verifyContext context.Context) error {
				_, verifyErr := adapter.inspectImage(verifyContext, unit.Host, image.Reference)
				return verifyErr
			}); err != nil {
				return fmt.Errorf("verify unit %s capsule %s: %w", unit.ID, image.Reference, err)
			}
			return nil
		})
	})
}

func (adapter Adapter) verifyNCCLProviders(
	ctx context.Context,
	launch snapshot.LaunchSpec,
	images map[string]snapshot.CapsuleImage,
	expected snapshot.RuntimeProviders,
) ([]snapshot.RuntimeProviderBinding, error) {
	expectedByOwner := make(map[string]snapshot.RuntimeProviderBinding, len(expected.Bindings))
	for _, binding := range expected.Bindings {
		if binding.Kind != ncclProviderKind {
			continue
		}
		if expectedByOwner[binding.Owner].Owner != "" {
			return nil, fmt.Errorf("duplicate expected NCCL provider for %s", binding.Owner)
		}
		expectedByOwner[binding.Owner] = binding
	}
	requireExpected := images != nil
	if requireExpected && len(expectedByOwner) != len(launch.Units) {
		return nil, errors.New("artifact must bind exactly one NCCL provider to every launch unit")
	}
	bindings := make([]snapshot.RuntimeProviderBinding, len(launch.Units))
	if err := parallelUnits(launch.Units, func(unit snapshot.LaunchUnit) error {
		image := unit.Image
		if prepared, ok := images[unit.ID]; ok {
			image = prepared.Reference
		}
		output, err := adapter.runWorkload(ctx, unit.Host, &hostops.WorkloadSpec{
			Image: image, RemoveAfterExit: true, Network: "none",
			Entrypoint: "/usr/local/bin/coldsnap",
			Command:    []string{"nccl-provider", "verify-active", "--active", ncclActiveRuntimePath},
		}, false)
		if err != nil {
			return fmt.Errorf("verify active NCCL runtime in %s: %w", image, err)
		}
		var active ncclprovider.ActiveRecord
		decoder := json.NewDecoder(bytes.NewReader(output))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&active); err != nil {
			return fmt.Errorf("decode verified active NCCL runtime: %w", err)
		}
		if err := decoder.Decode(&struct{}{}); err != io.EOF {
			return errors.New("verified active NCCL runtime contains trailing JSON")
		}
		binding, err := snapshot.NewRuntimeProviderBinding(
			snapshot.UnitOwner(unit.ID), ncclProviderKind, ncclProviderSchema, active,
		)
		if err != nil {
			return err
		}
		if wanted, ok := expectedByOwner[binding.Owner]; ok {
			// Both bindings have already validated their canonical payload digest.
			// RawMessage whitespace changes when an artifact is written with
			// indentation, so byte equality would reject the same JSON object.
			if wanted.Schema != binding.Schema || wanted.Digest != binding.Digest {
				return fmt.Errorf("NCCL provider differs from captured binding %s", binding.Owner)
			}
		} else if requireExpected {
			return fmt.Errorf("artifact has no NCCL provider for %s", binding.Owner)
		}
		bindings[unit.Index] = binding
		return nil
	}); err != nil {
		return nil, err
	}
	return bindings, nil
}

func restoreContainerName(request snapshot.Request, rank int) string {
	if request.Workload.ClusterID != "" {
		return request.Workload.ClusterID + "_node_" + strconv.Itoa(rank)
	}
	return operationName(request.ID, fmt.Sprintf("restore-rank-%d", rank))
}

func (adapter Adapter) startCoordinator(
	ctx context.Context, request snapshot.Request, namespace, image string,
) ([]string, string, error) {
	rank0 := request.Launch.Units[0]
	if image == "" {
		image = rank0.Image
	}
	root := filepath.Join(adapter.StateRoot, "operations", namespace)
	if _, err := adapter.Remote.Run(ctx, rank0.Host, "install", "-d", "-m", "0700", root); err != nil {
		return nil, "", fmt.Errorf("create coordinator state: %w", err)
	}
	endpoint := filepath.Join(root, "coordinator.endpoint")
	name := operationName(namespace, "coordinator")
	_, _ = adapter.removeWorkload(ctx, rank0.Host, name)
	uid, err := adapter.Remote.Run(ctx, rank0.Host, "id", "-u")
	if err != nil {
		return nil, "", fmt.Errorf("resolve coordinator uid: %w", err)
	}
	gid, err := adapter.Remote.Run(ctx, rank0.Host, "id", "-g")
	if err != nil {
		return nil, "", fmt.Errorf("resolve coordinator gid: %w", err)
	}
	user, err := numericUser(uid, gid)
	if err != nil {
		return nil, "", err
	}
	spec := &hostops.WorkloadSpec{
		Name: name, Image: image, Detached: true, Network: "host", User: user,
		Mounts:     []hostops.Mount{{Source: root, Target: "/run/coldsnap"}},
		Entrypoint: "/usr/local/bin/coldsnap-coordinator",
		Command: []string{"--listen", "0.0.0.0:0", "--advertise-host", rank0.Host,
			"--scope", namespace, "--endpoint-file", coordinatorEndpoint},
	}
	if _, err := adapter.runWorkload(ctx, rank0.Host, spec, false); err != nil {
		return nil, "", fmt.Errorf("start native ColdSnap coordinator: %w", err)
	}
	var payload []byte
	deadline := time.Now().Add(min(adapter.Timeout, 30*time.Second))
	for time.Now().Before(deadline) {
		if data, err := adapter.Remote.Run(ctx, rank0.Host, "cat", endpoint); err == nil && len(data) > 0 {
			payload = data
			break
		}
		if err := waitContext(ctx, adapter.PollInterval); err != nil {
			return nil, "", err
		}
	}
	if len(payload) == 0 {
		return nil, "", errors.New("native ColdSnap coordinator did not publish its endpoint")
	}
	paths := make([]string, len(request.Launch.Units))
	for _, rank := range request.Launch.Units {
		peerRoot := filepath.Join(adapter.StateRoot, "operations", namespace)
		peerEndpoint := filepath.Join(peerRoot, "coordinator.endpoint")
		if _, err := adapter.Remote.Run(ctx, rank.Host, "install", "-d", "-m", "0700", peerRoot); err != nil {
			return nil, "", err
		}
		if rank.Host != rank0.Host {
			if _, err := adapter.Remote.RunInput(ctx, rank.Host, payload, "tee", peerEndpoint); err != nil {
				return nil, "", fmt.Errorf("stage coordinator endpoint for rank %d: %w", rank.Index, err)
			}
			if _, err := adapter.Remote.Run(ctx, rank.Host, "chmod", "0600", peerEndpoint); err != nil {
				return nil, "", err
			}
		}
		paths[rank.Index] = peerEndpoint
	}
	return paths, name, nil
}

func numericUser(uid, gid []byte) (string, error) {
	values := []string{strings.TrimSpace(string(uid)), strings.TrimSpace(string(gid))}
	for _, value := range values {
		parsed, err := strconv.ParseUint(value, 10, 32)
		if err != nil || strconv.FormatUint(parsed, 10) != value {
			return "", errors.New("remote coordinator user identity is invalid")
		}
	}
	return values[0] + ":" + values[1], nil
}

func (adapter Adapter) stopCoordinator(
	ctx context.Context,
	request snapshot.Request,
	endpointPaths []string,
	name string,
) {
	if name != "" {
		_, _ = adapter.removeWorkload(ctx, request.Launch.Units[0].Host, name)
	}
	seen := make(map[string]bool)
	for _, rank := range request.Launch.Units {
		key := rank.Host + "\x00" + endpointPaths[rank.Index]
		if seen[key] {
			continue
		}
		seen[key] = true
		_, _ = adapter.Remote.Run(ctx, rank.Host, "rm", "-f", endpointPaths[rank.Index])
	}
}

func (adapter Adapter) cleanupCoordinator(
	parent context.Context,
	request snapshot.Request,
	endpointPaths []string,
	name string,
) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), 30*time.Second)
	defer cancel()
	adapter.stopCoordinator(ctx, request, endpointPaths, name)
}

func (adapter Adapter) removeContainers(
	parent context.Context,
	request snapshot.Request,
	containers []string,
) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), 30*time.Second)
	defer cancel()
	_ = parallelUnits(request.Launch.Units, func(rank snapshot.LaunchUnit) error {
		if containers[rank.Index] != "" {
			_, _ = adapter.removeWorkload(
				ctx, rank.Host, containers[rank.Index])
		}
		return nil
	})
}

const containerManagedOwnershipRoot = "/run/coldsnap/managed-ownership"

// normalizeManagedPathOwnership returns one exact writable host bind to the
// manager identity that invoked the adapter. The rank controller must remain
// root so it can drive CRIU and CUDA checkpoint operations, but root ownership
// must not escape through manager-owned bind mounts. A short, network-isolated
// sidecar changes only entries that actually differ, so a same-owner cleanup
// cannot mutate ctime on immutable payloads or invalidate cached validation
// evidence. CRIU-owned modes are left unchanged.
func (adapter Adapter) normalizeManagedPathOwnership(
	ctx context.Context,
	unit snapshot.LaunchUnit,
	image string,
	path string,
) error {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || path == string(filepath.Separator) ||
		strings.Contains(path, ":") {
		return fmt.Errorf("unit %s managed writable path is unsafe: %q", unit.ID, path)
	}
	if _, err := adapter.Remote.Run(ctx, unit.Host, "test", "-e", path); err != nil {
		return fmt.Errorf("unit %s managed writable path is unavailable: %s", unit.ID, path)
	}
	if _, err := adapter.Remote.Run(ctx, unit.Host, "test", "!", "-L", path); err != nil {
		return fmt.Errorf("unit %s managed writable path is a symbolic link: %s", unit.ID, path)
	}
	uid, err := adapter.Remote.Run(ctx, unit.Host, "id", "-u")
	if err != nil {
		return fmt.Errorf("resolve unit %s manager uid: %w", unit.ID, err)
	}
	gid, err := adapter.Remote.Run(ctx, unit.Host, "id", "-g")
	if err != nil {
		return fmt.Errorf("resolve unit %s manager gid: %w", unit.ID, err)
	}
	_, err = numericUser(uid, gid)
	if err != nil {
		return fmt.Errorf("resolve unit %s manager identity: %w", unit.ID, err)
	}
	if _, err := adapter.runWorkload(ctx, unit.Host, &hostops.WorkloadSpec{
		Image: image, RemoveAfterExit: true, PullPolicy: "never", Network: "none", User: "0:0",
		Mounts:     []hostops.Mount{{Source: path, Target: containerManagedOwnershipRoot}},
		Entrypoint: "python3",
		Command: []string{"-c", conditionalOwnershipProgram, containerManagedOwnershipRoot,
			strings.TrimSpace(string(uid)), strings.TrimSpace(string(gid))},
	}, false); err != nil {
		return fmt.Errorf("restore unit %s manager ownership for %s: %w", unit.ID, path, err)
	}
	return nil
}

const conditionalOwnershipProgram = `import os, sys
root, uid, gid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
paths = [root]
for directory, names, files in os.walk(root, topdown=False, followlinks=False):
    paths.extend(os.path.join(directory, name) for name in names)
    paths.extend(os.path.join(directory, name) for name in files)
    if directory != root:
        paths.append(directory)
for path in paths:
    value = os.lstat(path)
    if value.st_uid != uid or value.st_gid != gid:
        os.chown(path, uid, gid, follow_symlinks=False)
`

func (adapter Adapter) normalizeCapturePathOwnership(
	ctx context.Context,
	request snapshot.Request,
	roots []string,
) error {
	fmt.Fprintf(adapter.Output, "ColdSnap: restoring manager ownership for writable capture paths\n")
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		if unit.Index >= len(roots) || roots[unit.Index] == "" {
			return fmt.Errorf("unit %s capture root is unavailable", unit.ID)
		}
		paths := []string{roots[unit.Index]}
		if staged := stagedCachePath(request.Policy.Cache.Staged, unit.ID); staged != "" {
			paths = append(paths, staged)
		}
		for _, path := range paths {
			if err := adapter.normalizeManagedPathOwnership(ctx, unit, unit.Image, path); err != nil {
				return err
			}
		}
		return nil
	})
}

func (adapter Adapter) normalizeCapturePathOwnershipBestEffort(
	parent context.Context,
	request snapshot.Request,
	roots []string,
) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), 2*time.Minute)
	defer cancel()
	if err := adapter.normalizeCapturePathOwnership(ctx, request, roots); err != nil {
		fmt.Fprintf(adapter.Output, "ColdSnap: warning: capture ownership cleanup failed: %v\n", err)
	}
}

func (adapter Adapter) rankWorkloadSpec(
	ctx context.Context,
	request snapshot.Request,
	unit snapshot.LaunchUnit,
	image, name, operation, namespace, captureID, artifactRoot, endpointPath, provider string,
	modelPayloads []snapshot.PreparedPayload,
	identityImage string,
	tcpAddressMap []string,
	tcpPortShift uint16,
	materialization *materializationMount,
) (*hostops.WorkloadSpec, error) {
	spec := &hostops.WorkloadSpec{Name: name, Image: image, Detached: true}
	if operation == "restore" && request.Workload.ClusterID != "" {
		labels := map[string]string{
			"io.sparksq.coldsnap.capture":  captureID,
			"io.sparksq.coldsnap.driver":   request.Driver.ID,
			"io.sparksq.coldsnap.managed":  "true",
			"io.sparksq.coldsnap.workload": request.Workload.ClusterID,
			"io.sparksq.coldsnap.rank":     strconv.Itoa(unit.Index),
			"io.sparksq.coldsnap.unit":     unit.ID,
		}
		spec.Labels = labels
	}
	spec.GPUs = slices.Clone(unit.Devices)
	spec.Privileged, spec.SeccompUnconfined, spec.MemlockUnlimited = true, true, true
	spec.Network, spec.SharedMemoryBytes = "host", 32<<30

	activationPack, err := activationRuntimePack()
	if err != nil {
		return nil, err
	}
	for _, binding := range activationRuntimeBindings(adapter.StateRoot, activationPack) {
		spec.Mounts = append(spec.Mounts, hostops.Mount{Source: binding.HostPath, Target: binding.ContainerPath, ReadOnly: true})
	}
	if _, err := adapter.Remote.Run(ctx, unit.Host, "test", "-e", "/dev/infiniband"); err == nil {
		spec.Devices = append(spec.Devices, hostops.Device{Source: "/dev/infiniband", Target: "/dev/infiniband"})
	}
	values := make(map[string]string, len(unit.Environment)+32)
	for key, value := range unit.Environment {
		values[key] = value
	}
	workerMap, err := workerMapJSON(request.Launch, unit.ID)
	if err != nil {
		return nil, err
	}
	tpSize, err := tensorParallelSize(request.Launch)
	if err != nil {
		return nil, err
	}
	graphPolicy, err := snapshot.NewGraphPolicyRecord(request, nil)
	if err != nil {
		return nil, err
	}
	master := request.Launch.Units[0].Host
	if value := request.Launch.Units[0].Environment["MASTER_ADDR"]; value != "" {
		master = value
	}
	forced := map[string]string{
		"COLDSNAP_ENGINE":                       adapter.engine(),
		"COLDSNAP_ASYNC_CUDA_GRAPHS":            boolText(request.Policy.Process.AsyncGraphs),
		"COLDSNAP_GRAPH_POLICY":                 graphPolicy.Effective,
		"COLDSNAP_GRAPH_POLICY_REQUESTED":       graphPolicy.Requested,
		"COLDSNAP_ASYNC_CUDA_GRAPHS_ARM_FILE":   containerArtifactRoot + "/async-graphs-arm",
		"COLDSNAP_ASYNC_CUDA_GRAPHS_GENERATION": request.ID,
		"COLDSNAP_ASYNC_CUDA_GRAPHS_READY_FILE": containerArtifactRoot + "/async-graphs-ready.json",
		"COLDSNAP_CAPTURE_ID":                   captureID,
		"COLDSNAP_CAPTURE_BACKEND":              "auto",
		"COLDSNAP_DISCARD_REGIONS": func() string {
			if request.Policy.Process.KVDiscard {
				return "kv_cache"
			}
			return ""
		}(),
		"COLDSNAP_DISK_SLEEP_CHUNK_BYTES":                   strconv.Itoa(64 * 1024 * 1024),
		"COLDSNAP_DISK_SLEEP_DIR":                           containerArtifactRoot + "/hydration",
		"COLDSNAP_EXECUTION_GRAPH":                          workerMap,
		"COLDSNAP_EXPECTED_UNIT":                            unit.ID,
		"COLDSNAP_UNIT_INDEX":                               strconv.Itoa(unit.Index),
		"COLDSNAP_EXPORT_MODEL_PAYLOAD":                     boolText(operation == "capture" && wantsModelPayload(request.Policy.Weights)),
		"COLDSNAP_HIBERNATE_PIPELINE_DEPTH":                 "4",
		"COLDSNAP_HIBERNATE_READ_MODE":                      "direct",
		"COLDSNAP_HIBERNATE_REUSE_BLOB":                     boolText(operation == "restore"),
		"COLDSNAP_HIBERNATE_STATE_DIR":                      containerArtifactRoot + "/hibernate-states",
		"COLDSNAP_HIBERNATE_VERIFY_MODE":                    "preverified",
		"COLDSNAP_HIBERNATE_WRITE_MODE":                     "direct",
		"COLDSNAP_HYDRATION_BACKEND":                        "direct",
		"COLDSNAP_HYDRATION_NATIVE_LIBRARY":                 "/opt/coldsnap/native/libcoldsnap_hydration.so",
		"COLDSNAP_HYDRATION_QUEUE_DEPTH":                    "4",
		"COLDSNAP_KV_CAPACITY_GUARD":                        "1",
		"COLDSNAP_LOAD_FORMAT":                              "coldsnap",
		"COLDSNAP_MODEL_ID":                                 request.Launch.Model.ID,
		"COLDSNAP_MODEL_REVISION":                           request.Launch.Model.Revision,
		"COLDSNAP_NCCL_ACTIVE_RUNTIME_PATH":                 ncclActiveRuntimePath,
		"COLDSNAP_NCCL_CHECKPOINT":                          "1",
		"COLDSNAP_NCCL_REQUIRE_IB_RESET":                    "1",
		"COLDSNAP_NCCL_REQUIRE_NETWORK_RESET":               "1",
		"COLDSNAP_NCCL_REQUIRED_CAPABILITIES":               "full-network-reset,ib-roce-device-release,synchronous-termination",
		"COLDSNAP_PAYLOAD_VALIDATION_COMMAND":               "/usr/local/bin/coldsnap payload verify",
		"COLDSNAP_PAYLOAD_VALIDATION_TRANSPORT":             "cli",
		"COLDSNAP_RECOVERY_DERIVED_BUFFER_MAX_BYTES":        strconv.Itoa(64 * 1024 * 1024),
		"COLDSNAP_RECOVERY_LOADER_BACKEND":                  request.Policy.Weights.Recovery.LoaderBackend,
		"COLDSNAP_RECOVERY_LOADER_CHUNK_BYTES":              strconv.Itoa(64 * 1024 * 1024),
		"COLDSNAP_RECOVERY_LOADER_COLLECTIVE_BYTES":         strconv.Itoa(1024 * 1024 * 1024),
		"COLDSNAP_RECOVERY_LOADER_DISTRIBUTED":              "1",
		"COLDSNAP_RECOVERY_LOADER_QUEUE_DEPTH":              "4",
		"COLDSNAP_RECOVERY_LOADER_STATUS_GROUP":             "device",
		"COLDSNAP_RECOVERY_LOADER_STAGING_BYTES":            strconv.Itoa(1024 * 1024 * 1024),
		"COLDSNAP_RECOVERY_LOADER_TRANSPORT_PIPELINE_DEPTH": "2",
		"COLDSNAP_RECOVERY_LOADER_VERIFY_BYTES":             "0",
		"COLDSNAP_RECOVERY_LOADER_VERIFY_MODEL_BYTES":       "64",
		"COLDSNAP_RECOVERY_WEIGHT_SOURCE":                   "safetensors",
		"COLDSNAP_RESTORE_RUNTIME_ENVIRONMENT_PATH":         containerArtifactRoot + "/restore-runtime-environment.json",
		"COLDSNAP_RESTORE_TRANSPORT_ENVIRONMENT_PATH":       containerArtifactRoot + "/restore-transport-environment.json",
		"COLDSNAP_SHAPE_CALIBRATION": boolText(
			operation == "capture" && request.Policy.Process.AsyncGraphs &&
				request.Policy.Process.ShapeCalibration != "disabled",
		),
		"COLDSNAP_TP_SIZE":                    strconv.Itoa(tpSize),
		"COLDSNAP_VLLM_OVERRIDE_CUMEM":        "1",
		"COLDSNAP_WORLD_SIZE":                 strconv.Itoa(len(request.Launch.Execution.Workers)),
		"NCCL_CHECKPOINT_COORDINATOR_PATH":    coordinatorEndpoint,
		"NCCL_CHECKPOINT_COORDINATOR_TIMEOUT": strconv.Itoa(int(min(adapter.Timeout, time.Hour).Seconds())),
		"NCCL_CHECKPOINT_TERMINATION":         "destroy",
		"NCCL_CUMEM_ENABLE":                   "1",
		"NCCL_IB_RELEASE_ON_FINALIZE":         "1",
		"NCCL_RAS_ENABLE":                     "0",
		"PYTHONDONTWRITEBYTECODE":             "1",
		"PYTHONPATH":                          "/opt/coldsnap/plugin:/opt/coldsnap/runtime",
		"VLLM_PLUGINS":                        "coldsnap",
		"VLLM_SERVER_DEV_MODE":                "1",
	}
	if graphPolicy.Effective == snapshot.GraphPreserveNCCLExec {
		forced["COLDSNAP_NCCL_IN_PLACE_EXPERIMENT"] = "1"
		forced["COLDSNAP_NCCL_IN_PLACE_MODE"] = "net-reconnect-v1"
		forced["COLDSNAP_NCCL_IN_PLACE_ACTIVATION_PATH"] = containerArtifactRoot + "/nccl-in-place-activation"
		forced["COLDSNAP_NCCL_REQUIRED_CAPABILITIES"] = strings.Join([]string{
			"communicator-suspend-in-place",
			"full-network-reset",
			"graph-resource-retention",
			"ib-roce-device-release",
			"synchronous-termination",
			"transport-detach-in-place",
		}, ",")
	}
	if adapter.engine() == "vllm" {
		managedRuntimeCache := request.Policy.Cache.Seed &&
			cachePathsCover(request.Policy.Cache.Paths, containerRuntimeCacheRoot)
		forced["VLLM_ENABLE_STARTUP_PLAN"] = boolText(managedRuntimeCache)
		// GB10 unified-memory accounting can drift slightly between capture and
		// restore even on the same target with no competing workload. Reuse the
		// admitted plan within a bounded envelope by reducing KV bytes by the
		// observed shortfall; larger changes retain vLLM's full-profile fallback.
		forced["COLDSNAP_STARTUP_PLAN_MAX_SHORTFALL_BYTES"] = strconv.Itoa(512 * 1024 * 1024)
		forced["COLDSNAP_DEFERRED_WARMUP"] = boolText(
			operation == "restore" && managedRuntimeCache && request.Policy.Process.AsyncGraphs &&
				graphPolicy.Effective != snapshot.GraphPreserveNCCLExec,
		)
		forced["COLDSNAP_DEFERRED_WARMUP_ARM_FILE"] = containerArtifactRoot + "/async-graphs-arm"
		forced["COLDSNAP_FULLY_WARM_FILE"] = containerArtifactRoot + "/fully-warm.json"
		forced["COLDSNAP_DEFERRED_API_MM_WARMUP"] = boolText(operation == "restore")
		forced["COLDSNAP_API_MM_WARMUP_DELAY_SECONDS"] = func() string {
			if operation == "restore" {
				return "30"
			}
			return "0"
		}()
		forced["COLDSNAP_API_MM_WARM_FILE"] = containerArtifactRoot + "/api-mm-warm.json"
		forced["COLDSNAP_WARMUP_GENERATION"] = request.ID
		if request.Driver.ID == snapshotdriver.N580 &&
			request.Policy.Process.ArtifactScope == snapshot.ArtifactScopeTargetLocal {
			// A target-local overlay intentionally retains the matching target's
			// pre-worker process state. The portable artifact remains the fallback
			// for a different driver/runtime identity.
			forced["COLDSNAP_N580_QUALIFICATION_PROCESS_TEMPLATE_PHASE"] = "pre_worker_import"
			forced["COLDSNAP_N580_QUALIFICATION_ALLOW_CAPTURED_DRIVER_LIBRARIES"] = "1"
			forced["COLDSNAP_N580_PORTABLE_WORKER_REEXEC"] = "0"
		}
	}
	if operation == "capture" && adapter.engine() == "vllm" {
		forced["COLDSNAP_CAPTURE_LOAD_FORMAT"] = commandLoadFormat(unit.Command)
	}
	if adapter.engine() == "sglang" {
		for _, key := range []string{
			"COLDSNAP_DISK_SLEEP_CHUNK_BYTES",
			"COLDSNAP_DISK_SLEEP_DIR",
			"COLDSNAP_HIBERNATE_PIPELINE_DEPTH",
			"COLDSNAP_HIBERNATE_READ_MODE",
			"COLDSNAP_HIBERNATE_REUSE_BLOB",
			"COLDSNAP_HIBERNATE_STATE_DIR",
			"COLDSNAP_HIBERNATE_VERIFY_MODE",
			"COLDSNAP_HIBERNATE_WRITE_MODE",
			"COLDSNAP_KV_CAPACITY_GUARD",
			"COLDSNAP_LOAD_FORMAT",
			"COLDSNAP_RECOVERY_DERIVED_BUFFER_MAX_BYTES",
			"COLDSNAP_RECOVERY_LOADER_BACKEND",
			"COLDSNAP_RECOVERY_LOADER_CHUNK_BYTES",
			"COLDSNAP_RECOVERY_LOADER_COLLECTIVE_BYTES",
			"COLDSNAP_RECOVERY_LOADER_DISTRIBUTED",
			"COLDSNAP_RECOVERY_LOADER_QUEUE_DEPTH",
			"COLDSNAP_RECOVERY_LOADER_STATUS_GROUP",
			"COLDSNAP_RECOVERY_LOADER_STAGING_BYTES",
			"COLDSNAP_RECOVERY_LOADER_TRANSPORT_PIPELINE_DEPTH",
			"COLDSNAP_RECOVERY_LOADER_VERIFY_BYTES",
			"COLDSNAP_RECOVERY_LOADER_VERIFY_MODEL_BYTES",
			"COLDSNAP_RECOVERY_WEIGHT_SOURCE",
			"COLDSNAP_VLLM_OVERRIDE_CUMEM",
			"VLLM_PLUGINS",
			"VLLM_SERVER_DEV_MODE",
		} {
			delete(forced, key)
		}
		forced["COLDSNAP_ARTIFACT_DIR"] = containerArtifactRoot + "/semantic"
		forced["COLDSNAP_ARTIFACT_PREVERIFIED"] = "1"
		forced["COLDSNAP_LIVE_BACKING"] = "discard"
		forced["COLDSNAP_MODE"] = "capture"
		forced["COLDSNAP_PROCESS_ARTIFACT_ROOT"] = containerArtifactRoot
		forced["COLDSNAP_RUNTIME_DIR"] = "/run/coldsnap/sglang"
		forced["COLDSNAP_SGLANG_RECOVERY_LOAD_FORMAT"] = commandLoadFormat(unit.Command)
		// SGLang owns several communicators whose creation order is not a
		// portable cross-rank finalization order. ncclCommFinalize can therefore
		// deadlock even after every scheduler enters the checkpoint hook. The
		// qualified provider's synchronous abort path reclaims each local
		// communicator without a cross-rank finalize rendezvous, while retaining
		// the metadata required to reconstruct it during restore.
		forced["NCCL_CHECKPOINT_TERMINATION"] = "abort-sync"
		forced["SGLANG_PLUGINS"] = "coldsnap"
		if request.Driver.ID == snapshotdriver.N610 {
			forced["COLDSNAP_SGLANG_CRIU_HOLD"] = "1"
		}
		forced["PYTORCH_CUDA_ALLOC_CONF"] = withoutExpandableSegments(values["PYTORCH_CUDA_ALLOC_CONF"])
		if value, ok := values["PYTORCH_ALLOC_CONF"]; ok {
			forced["PYTORCH_ALLOC_CONF"] = withoutExpandableSegments(value)
		}
	}
	if request.Driver.ID == snapshotdriver.N580 {
		// n580 resumes from a pre-CUDA process template and deliberately lets
		// capture use vLLM's selected fast loader. The restored pre-worker
		// boundary switches only restored worker configs to ColdSnap, so the
		// capture emits the same residual/replay split as n610 without a full
		// weight blob.
		forced["COLDSNAP_LOAD_FORMAT"] = "coldsnap"
		forced["COLDSNAP_RECOVERY_WEIGHT_SOURCE"] = "safetensors"
	}
	// A seeded derived-cache tree is part of the portable capsule, not recipe
	// launch policy. Route every supported compiler/JIT cache below one
	// canonical root so Sparkrun can preseed it without depending on the image's
	// configured USER or HOME. A custom cache inventory can opt out by omitting
	// the canonical root.
	if request.Policy.Cache.Seed && cachePathsCover(request.Policy.Cache.Paths, containerRuntimeCacheRoot) {
		for key, value := range map[string]string{
			"CUTE_DSL_CACHE_DIR":                 containerRuntimeCacheRoot + "/cute_dsl",
			"CUDA_CACHE_DISABLE":                 "0",
			"CUDA_CACHE_MAXSIZE":                 strconv.Itoa(1024 * 1024 * 1024),
			"CUDA_CACHE_PATH":                    containerCudaCachePath,
			"FLASHINFER_CACHE_DIR":               containerRuntimeCacheRoot + "/flashinfer",
			"FLASHINFER_WORKSPACE_BASE":          containerRuntimeCacheRoot + "/flashinfer",
			"FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": containerRuntimeCacheRoot + "/flash_attn_cute_dsl",
			"TORCHINDUCTOR_CACHE_DIR":            containerRuntimeCacheRoot + "/inductor",
			"TORCH_EXTENSIONS_DIR":               containerRuntimeCacheRoot + "/torch_extensions",
			"TORCH_HOME":                         containerRuntimeCacheRoot + "/torch",
			"TRITON_CACHE_DIR":                   containerRuntimeCacheRoot + "/triton",
			"TVM_FFI_CACHE_DIR":                  containerRuntimeCacheRoot + "/tvm_ffi",
			"SGLANG_CACHE_DIR":                   containerRuntimeCacheRoot + "/sglang",
			"SGLANG_JIT_CACHE_DIR":               containerRuntimeCacheRoot + "/sglang_jit",
			"TILELANG_CACHE_DIR":                 containerRuntimeCacheRoot + "/tilelang",
			"VLLM_CACHE_ROOT":                    containerRuntimeCacheRoot + "/vllm",
			"XDG_CACHE_HOME":                     containerRuntimeCacheRoot,
		} {
			forced[key] = value
		}
	}
	for key, value := range forced {
		values[key] = value
	}
	spec.Environment = values
	for _, mount := range unit.Mounts {
		if mount.Target == containerArtifactRoot || mount.Target == coordinatorEndpoint ||
			mount.Target == containerModelPayloadRoot || mount.Target == containerMaterializationControl ||
			isActivationRuntimeTarget(mount.Target) ||
			strings.HasPrefix(mount.Target, containerArtifactRoot+"/") ||
			strings.HasPrefix(mount.Target, capsule.Root+"/") ||
			strings.HasPrefix(mount.Target, containerModelPayloadRoot+"/") {
			return nil, fmt.Errorf("unit %s mount conflicts with ColdSnap runtime: %s", unit.ID, mount.Target)
		}
		spec.Mounts = append(spec.Mounts, hostops.Mount{Source: mount.Source, Target: mount.Target, ReadOnly: mount.ReadOnly})
	}
	if operation == "capture" {
		if staged := stagedCachePath(request.Policy.Cache.Staged, unit.ID); staged != "" {
			spec.Mounts = append(spec.Mounts, hostops.Mount{Source: staged, Target: containerRuntimeCacheRoot})
		}
	}
	if operation == "capture" {
		spec.Mounts = append(spec.Mounts, hostops.Mount{Source: artifactRoot, Target: containerArtifactRoot})
	}
	spec.Mounts = append(spec.Mounts, hostops.Mount{Source: endpointPath, Target: coordinatorEndpoint, ReadOnly: true})
	if materialization != nil {
		spec.Mounts = append(spec.Mounts,
			hostops.Mount{Source: materialization.cacheRoot, Target: containerModelPayloadRoot},
			hostops.Mount{Source: materialization.controlPath, Target: containerMaterializationControl, ReadOnly: true},
		)
	}
	for _, modelPayload := range modelPayloads {
		target, err := modelPayloadTarget(request, modelPayload.Worker)
		if err != nil {
			return nil, err
		}
		spec.Mounts = append(spec.Mounts, hostops.Mount{Source: modelPayload.Path, Target: target, ReadOnly: true})
	}
	rankController := "/usr/local/bin/coldsnap-engine-rank-n610"
	targetLauncher := "/opt/coldsnap/runtime/coldsnap-engine-exec.py"
	if request.Driver.ID == snapshotdriver.N580 {
		rankController = "/usr/local/bin/coldsnap-engine-rank-n580"
	}
	command := []string{rankController,
		"--engine", adapter.engine(),
		"--artifact-root", func() string {
			if operation == "capture" {
				return containerArtifactRoot
			}
			return capsule.Root
		}(),
		"--rank", strconv.Itoa(unit.Index), "--world-size", strconv.Itoa(len(request.Launch.Units)),
		"--worker-count", strconv.Itoa(len(request.Launch.Execution.UnitWorkers(unit.ID))),
	}
	if request.Driver.ID == snapshotdriver.N580 {
		command = append(command,
			"--generation", captureID,
		)
	}
	command = append(command,
		"--master-address", master, "--http-port", strconv.Itoa(httpPort(unit.Command)),
		"--image-id", identityImage, "--model", request.Launch.Model.ID,
		"--model-revision", request.Launch.Model.Revision,
		"--prompt", request.Validation.Prompt, "--expected", request.Validation.Expected,
		"--weight-provider", provider,
		"--target-launcher", targetLauncher,
		"--plugin-root", "/opt/coldsnap/plugin", "--nccl-active-runtime", ncclActiveRuntimePath,
		"--criu", "/opt/coldsnap/criu/bin/criu", "--criu-rpc", "/usr/local/bin/coldsnap-criu-rpc",
		"--runtime-lib-dir", "/opt/coldsnap/criu/lib", "--cuda-checkpoint", "/usr/local/bin/cuda-checkpoint",
		"--native-hydration-library", "/opt/coldsnap/native/libcoldsnap_hydration.so",
		"--coordinator-path", coordinatorEndpoint, "--activation-namespace", namespace,
		"--timeout", strconv.Itoa(int(adapter.Timeout.Seconds())),
		"--kernel-compatibility", request.Policy.Compatibility.Kernel,
		"--criu-compress-block-bytes", strconv.Itoa(defaultCRIUBlockBytes),
		"--criu-compress-acceleration", strconv.Itoa(defaultCRIUAcceleration),
		"--criu-decompress-threads", strconv.Itoa(defaultCRIUThreads),
		"--criu-image-io-mode", defaultCRIUImageIOMode,
	)
	if operation == "restore" {
		command = append(command, "--activation-state", lifecycleActivationState(request))
	}
	if request.Driver.ID == snapshotdriver.N580 {
		command = append(command,
			"--criu-plugin-dir", "/opt/coldsnap/criu/plugins",
			"--nvml-dlopen-shim", "/opt/coldsnap/native/libcoldsnap_nvml_dlopen_shim.so",
		)
	}
	command = append(command, operation, "--")
	if operation == "restore" {
		insert := len(command) - 2
		portable := []string{"--allow-compatible-criu-runtime"}
		if request.Driver.ID == snapshotdriver.N610 {
			portable = append(portable, "--leave-stopped")
		}
		if len(tcpAddressMap) != 0 {
			portable = append(portable, "--defer-network-unlock")
		}
		for _, mapping := range tcpAddressMap {
			portable = append(portable, "--tcp-address-map", mapping)
		}
		if tcpPortShift != 0 {
			portable = append(portable, "--tcp-port-shift", strconv.Itoa(int(tcpPortShift)))
		}
		command = slices.Insert(command, insert, portable...)
	}
	// The restored process never execs this command, but the CUDA/CRIU process
	// artifact verifies it as part of the captured identity. Keep it byte-for-
	// byte semantic with capture; the checkpointed vLLM load_config was already
	// switched to ColdSnap immediately before sleep.
	command = append(command, unit.Command...)
	spec.Command = command
	return spec, nil
}

func withoutExpandableSegments(value string) string {
	settings := make([]string, 0, 4)
	for _, item := range strings.Split(value, ",") {
		item = strings.TrimSpace(item)
		if item == "" || strings.TrimSpace(strings.SplitN(item, ":", 2)[0]) == "expandable_segments" {
			continue
		}
		settings = append(settings, item)
	}
	return strings.Join(append(settings, "expandable_segments:False"), ",")
}

func cachePathsCover(paths []string, target string) bool {
	for _, path := range paths {
		if target == path || strings.HasPrefix(target, path+"/") {
			return true
		}
	}
	return false
}

func stagedCachePath(staged []snapshot.PreparedCache, unitID string) string {
	for _, cache := range staged {
		if cache.Unit == unitID {
			return cache.Path
		}
	}
	return ""
}

func (adapter Adapter) waitCapture(
	ctx context.Context, request snapshot.Request, roots, containers []string,
) error {
	deadline := time.Now().Add(adapter.Timeout)
	for time.Now().Before(deadline) {
		ready := 0
		for _, rank := range request.Launch.Units {
			if _, err := adapter.Remote.Run(ctx, rank.Host, "test", "-s", filepath.Join(roots[rank.Index], "capture.json")); err == nil {
				ready++
				continue
			}
			state, err := adapter.inspectWorkload(ctx, rank.Host, containers[rank.Index])
			if err == nil && state.State == "exited" && state.ExitCode != 0 {
				return fmt.Errorf("capture rank %d workload failed: %s (exit %d)", rank.Index, state.State, state.ExitCode)
			}
		}
		if ready == len(request.Launch.Units) {
			return nil
		}
		if err := waitContext(ctx, adapter.PollInterval); err != nil {
			return err
		}
	}
	return errors.New("timed out waiting for distributed vLLM capture")
}

func (adapter Adapter) waitRestore(
	ctx context.Context, request snapshot.Request, containers []string,
) error {
	deadline := time.Now().Add(adapter.Timeout)
	for time.Now().Before(deadline) {
		ready := 0
		for _, rank := range request.Launch.Units {
			path := fmt.Sprintf("%s/restore-ready-unit-%s.json", capsule.Root, rank.ID)
			if _, err := adapter.execWorkload(ctx, rank.Host, containers[rank.Index], "test", "-s", path); err == nil {
				ready++
				continue
			}
			state, err := adapter.inspectWorkload(ctx, rank.Host, containers[rank.Index])
			if err != nil || !state.Running {
				failure := fmt.Errorf("restore rank %d workload is not running: %s (exit %d; inspect error: %v)", rank.Index, state.State, state.ExitCode, err)
				if adapter.hasTransientNCCLInitializationFailure(ctx, request, containers) {
					return &transientNCCLRestoreError{cause: failure}
				}
				return failure
			}
		}
		if ready == len(request.Launch.Units) {
			return nil
		}
		if err := waitContext(ctx, adapter.PollInterval); err != nil {
			return err
		}
	}
	return errors.New("timed out waiting for distributed vLLM restore")
}

func (adapter Adapter) hasTransientNCCLInitializationFailure(
	ctx context.Context, request snapshot.Request, containers []string,
) bool {
	for _, unit := range request.Launch.Units {
		if unit.Index < 0 || unit.Index >= len(containers) || containers[unit.Index] == "" {
			continue
		}
		output, err := adapter.logsWorkload(
			ctx, unit.Host, containers[unit.Index], 400)
		if err == nil && isTransientNCCLInitializationFailure(output) {
			return true
		}
	}
	return false
}

func isTransientNCCLInitializationFailure(output []byte) bool {
	message := bytes.ToLower(output)
	return bytes.Contains(message, []byte("nccl error: unhandled cuda error")) ||
		bytes.Contains(message, []byte("ncclunhandledcudaerror"))
}

func (adapter Adapter) constructArtifact(
	ctx context.Context, request snapshot.Request, roots, cacheRoots []string,
	runtimeProviders []snapshot.RuntimeProviderBinding,
) (snapshot.Artifact, error) {
	driverContract, err := snapshotdriver.Resolve(request.Driver)
	if err != nil {
		return snapshot.Artifact{}, err
	}
	if err := adapter.validateCapturedNCCLWorkers(ctx, request, roots, runtimeProviders); err != nil {
		return snapshot.Artifact{}, err
	}
	shapeCalibration, err := adapter.capturedShapeCalibration(ctx, request, roots)
	if err != nil {
		return snapshot.Artifact{}, err
	}
	graphPolicy, err := snapshot.NewGraphPolicyRecord(request, shapeCalibration)
	if err != nil {
		return snapshot.Artifact{}, err
	}
	images := make([]snapshot.CapsuleImage, len(request.Launch.Units))
	objects := make([]snapshot.Object, len(request.Launch.Units))
	replay := make([]snapshot.Object, len(request.Launch.Execution.Workers))
	modelPayloads := make([]snapshot.Object, len(request.Launch.Execution.Workers))
	platforms := make([]snapshot.UnitPlatformCompatibility, len(request.Launch.Units))
	workerIndexes := make(map[string]int, len(request.Launch.Execution.Workers))
	for index, worker := range request.Launch.Execution.Workers {
		workerIndexes[worker.ID] = index
	}
	hasNative := wantsModelPayload(request.Policy.Weights)
	builder := capsule.Builder{Remote: adapter.Remote, Runtime: adapter.runtimeBackend()}
	if err := parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		return operationtiming.Measure(ctx, "capsule.construct", map[string]string{
			"unit": unit.ID, "host": unit.Host,
		}, func(unitContext context.Context) error {
			platform, err := adapter.capturedPlatform(unitContext, unit, roots[unit.Index])
			if err != nil {
				return fmt.Errorf("unit %s compatibility identity: %w", unit.ID, err)
			}
			platforms[unit.Index] = platform
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
				if hasNative {
					packPath := filepath.Join(filepath.Dir(manifest.Path), modelPayloadName)
					pack, packErr := adapter.remoteModelPayloadObject(
						unitContext, unit, snapshot.WorkerOwner(manifest.Worker), packPath,
						"pending-content-address", 0, "",
					)
					if packErr != nil {
						return packErr
					}
					pack.Path = modelPayloadObjectPath(pack.SHA256)
					modelPayloads[workerIndex] = pack
				}
			}
			result, err := builder.Build(unitContext, capsule.Spec{
				Host: unit.Host, BaseImage: unit.Image, BaseDigest: unit.ImageDigest,
				ArtifactRoot: roots[unit.Index], CaptureID: request.ID, Unit: unit.ID,
				Driver:      driverContract,
				Repository:  request.Policy.Capsule.Repository,
				CacheRootFS: cacheRoots[unit.Index],
			})
			if err != nil {
				return err
			}
			images[unit.Index], objects[unit.Index] = result.Image, result.Object
			return nil
		})
	}); err != nil {
		return snapshot.Artifact{}, err
	}
	modelPayloadRevision := request.Policy.Weights.Native.Revision
	digest, err := snapshot.RequestSHA256(request)
	if err != nil {
		return snapshot.Artifact{}, err
	}
	artifact := snapshot.Artifact{
		Format: snapshot.ArtifactFormat, Kind: snapshot.ArtifactKind, State: "committed",
		CaptureID: request.ID, RequestSHA256: digest, Driver: driverContract,
		Requires: driverContract.BaseRequirements.Clone(), Launch: request.Launch,
		Compatibility: snapshot.ArtifactCompatibility{
			Policy: snapshot.PortabilityPolicy,
			Units:  platforms,
		},
		Capsule:          snapshot.Capsule{Images: images, Objects: objects},
		Runtime:          snapshot.RuntimeProviders{Bindings: runtimeProviders},
		ShapeCalibration: shapeCalibration,
		Graph:            graphPolicy,
		Weights: snapshot.WeightProviders{Recovery: snapshot.RecoveryProvider{
			Source: "huggingface-safetensors", ModelID: request.Launch.Model.ID,
			Revision: request.Launch.Model.Revision, Driver: driverContract.Binding(),
			LoadPath: recoveryLoadPath(request.Launch.Engine), ReplayPlan: replay,
		}},
		Acceptance: snapshot.ArtifactAcceptance{Accepted: true, Expected: request.Validation.Expected},
	}
	if hasNative {
		artifact.Weights.ModelPayloads = &snapshot.ModelPayloadProvider{
			Repository: request.Policy.Weights.Native.Repository,
			Revision:   modelPayloadRevision,
			Objects:    modelPayloads,
		}
		artifact.Weights.Native = &snapshot.NativeProvider{Driver: driverContract.Binding()}
	}
	if err := snapshot.ValidateArtifact(artifact); err != nil {
		return snapshot.Artifact{}, fmt.Errorf("constructed artifact: %w", err)
	}
	return artifact, nil
}

func (adapter Adapter) capturedShapeCalibration(
	ctx context.Context,
	request snapshot.Request,
	roots []string,
) (*snapshot.ShapeCalibrationCoverage, error) {
	if !request.Policy.Process.AsyncGraphs || request.Policy.Process.ShapeCalibration == "disabled" {
		return nil, nil
	}
	if len(roots) != len(request.Launch.Units) {
		return nil, errors.New("shape-calibration root inventory differs from launch units")
	}
	units := make([]snapshot.UnitShapeCalibration, len(request.Launch.Units))
	if err := parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		payload, err := adapter.Remote.Run(
			ctx,
			unit.Host,
			"cat",
			filepath.Join(roots[unit.Index], "capture.json"),
		)
		if err != nil {
			return fmt.Errorf("read unit %s shape-calibration report: %w", unit.ID, err)
		}
		var report struct {
			ShapeCalibration json.RawMessage `json:"shape_calibration"`
		}
		if err := json.Unmarshal(payload, &report); err != nil {
			return fmt.Errorf("decode unit %s shape-calibration report: %w", unit.ID, err)
		}
		if len(report.ShapeCalibration) == 0 || bytes.Equal(report.ShapeCalibration, []byte("null")) {
			return fmt.Errorf("unit %s did not publish required shape-calibration coverage", unit.ID)
		}
		var evidence struct {
			Engine        string  `json:"engine"`
			PlannedShapes int     `json:"planned_shapes"`
			WarmedShapes  int     `json:"warmed_shapes"`
			Seconds       float64 `json:"seconds"`
		}
		if err := json.Unmarshal(report.ShapeCalibration, &evidence); err != nil {
			return fmt.Errorf("decode unit %s shape-calibration evidence: %w", unit.ID, err)
		}
		units[unit.Index] = snapshot.UnitShapeCalibration{
			Unit:          unit.ID,
			Engine:        evidence.Engine,
			PlannedShapes: evidence.PlannedShapes,
			WarmedShapes:  evidence.WarmedShapes,
			Seconds:       evidence.Seconds,
			Evidence:      append(json.RawMessage(nil), report.ShapeCalibration...),
		}
		return nil
	}); err != nil {
		return nil, err
	}
	coverage := &snapshot.ShapeCalibrationCoverage{
		Policy: "engine-owned-v1",
		Units:  units,
	}
	if err := coverage.Validate(request.Launch); err != nil {
		return nil, fmt.Errorf("captured shape calibration: %w", err)
	}
	return coverage, nil
}

type ncclWorkerObservation struct {
	WorkerID string `json:"worker_id"`
	Version  struct {
		Checkpoint     uint32 `json:"checkpoint"`
		NCCL           int    `json:"nccl"`
		DLSymBridgeABI uint32 `json:"dlsym_bridge_abi"`
		Provider       struct {
			ID                    string   `json:"id"`
			Revision              uint32   `json:"revision"`
			ABIMajor              uint32   `json:"abi_major"`
			ABIMinor              uint32   `json:"abi_minor"`
			CapabilityMask        uint64   `json:"capability_mask"`
			UnknownCapabilityMask uint64   `json:"unknown_capability_mask"`
			Capabilities          []string `json:"capabilities"`
			CompiledNCCL          int      `json:"compiled_nccl"`
			LoadedNCCL            int      `json:"loaded_nccl"`
		} `json:"provider"`
	} `json:"version"`
}

func (adapter Adapter) validateCapturedNCCLWorkers(
	ctx context.Context,
	request snapshot.Request,
	roots []string,
	bindings []snapshot.RuntimeProviderBinding,
) error {
	if len(roots) != len(request.Launch.Units) {
		return errors.New("capture root inventory differs from launch units")
	}
	observations := make(map[string]ncclWorkerObservation, len(request.Launch.Execution.Workers))
	for _, unit := range request.Launch.Units {
		payload, err := adapter.Remote.Run(
			ctx, unit.Host, "cat", filepath.Join(roots[unit.Index], "capture.json"),
		)
		if err != nil {
			return fmt.Errorf("read unit %s capture NCCL worker reports: %w", unit.ID, err)
		}
		var report struct {
			Workers struct {
				Results []ncclWorkerObservation `json:"results"`
			} `json:"nccl_workers_before_sleep"`
		}
		if err := json.Unmarshal(payload, &report); err != nil {
			return fmt.Errorf("decode unit %s capture NCCL worker reports: %w", unit.ID, err)
		}
		for _, observation := range report.Workers.Results {
			if previous, exists := observations[observation.WorkerID]; exists {
				// vLLM's collective RPC can report every worker in every unit
				// capture, while SGLang records only the workers local to a unit.
				// Accept repeated evidence only when every field agrees.
				if !reflect.DeepEqual(previous, observation) {
					return fmt.Errorf("capture reported conflicting NCCL worker %q", observation.WorkerID)
				}
				continue
			}
			observations[observation.WorkerID] = observation
		}
	}
	if len(observations) != len(request.Launch.Execution.Workers) {
		return fmt.Errorf(
			"capture reported %d unique NCCL workers, want %d",
			len(observations), len(request.Launch.Execution.Workers),
		)
	}
	activeByUnit := make(map[string]ncclprovider.ActiveRecord, len(bindings))
	for _, binding := range bindings {
		if binding.Kind != ncclProviderKind || binding.Schema != ncclProviderSchema {
			continue
		}
		unitID, ok := strings.CutPrefix(binding.Owner, snapshot.UnitOwnerPrefix)
		if !ok || activeByUnit[unitID].ProviderID != "" {
			return errors.New("capture NCCL provider bindings are invalid or duplicated")
		}
		var active ncclprovider.ActiveRecord
		decoder := json.NewDecoder(bytes.NewReader(binding.Payload))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&active); err != nil {
			return fmt.Errorf("decode unit %s NCCL provider binding: %w", unitID, err)
		}
		activeByUnit[unitID] = active
	}
	if len(activeByUnit) != len(request.Launch.Units) {
		return errors.New("capture lacks one verified NCCL provider per launch unit")
	}
	for workerID, observation := range observations {
		worker, ok := request.Launch.Execution.Worker(observation.WorkerID)
		if !ok || workerID == "" {
			return fmt.Errorf("capture reported invalid NCCL worker %q", observation.WorkerID)
		}
		active := activeByUnit[worker.Unit]
		provider := observation.Version.Provider
		if provider.ID != active.ProviderID || provider.Revision != active.ProviderRevision ||
			provider.ABIMajor != active.ProviderABI.Major || provider.ABIMinor != active.ProviderABI.Minor ||
			observation.Version.Checkpoint != active.CheckpointABI ||
			observation.Version.DLSymBridgeABI != active.Bridge.ABI ||
			provider.CompiledNCCL != provider.LoadedNCCL || observation.Version.NCCL != provider.LoadedNCCL ||
			provider.UnknownCapabilityMask != 0 || !slices.Equal(provider.Capabilities, active.Capabilities) ||
			active.ProviderNCCLRuntime.Version != provider.LoadedNCCL {
			return fmt.Errorf("worker %s loaded NCCL provider differs from verified unit %s binding", worker.ID, worker.Unit)
		}
	}
	return nil
}

func (adapter Adapter) capturedPlatform(
	ctx context.Context, unit snapshot.LaunchUnit, artifactRoot string,
) (snapshot.UnitPlatformCompatibility, error) {
	payload, err := adapter.Remote.Run(ctx, unit.Host, "cat", filepath.Join(artifactRoot, "capture.json"))
	if err != nil {
		return snapshot.UnitPlatformCompatibility{}, err
	}
	var report struct {
		Identity struct {
			Unit          string `json:"unit"`
			Architecture  string `json:"architecture"`
			Kernel        string `json:"kernel"`
			CUDAUserspace string `json:"cuda_userspace"`
			GPUs          []struct {
				Driver            string `json:"driver"`
				Name              string `json:"name"`
				ComputeCapability string `json:"compute_capability"`
			} `json:"gpus"`
		} `json:"identity"`
	}
	if err := json.Unmarshal(payload, &report); err != nil {
		return snapshot.UnitPlatformCompatibility{}, fmt.Errorf("decode capture report: %w", err)
	}
	identity := report.Identity
	if identity.Unit != unit.ID || len(identity.GPUs) != len(unit.Devices) {
		return snapshot.UnitPlatformCompatibility{}, errors.New("capture report unit or device inventory differs from launch unit")
	}
	result := snapshot.UnitPlatformCompatibility{
		Unit: unit.ID, Architecture: identity.Architecture, Kernel: identity.Kernel,
		NVIDIADriverCaptured: identity.GPUs[0].Driver,
		NVIDIADriverMin:      identity.GPUs[0].Driver, CUDAUserspace: identity.CUDAUserspace,
	}
	for slot, gpu := range identity.GPUs {
		if gpu.Driver != result.NVIDIADriverMin {
			return snapshot.UnitPlatformCompatibility{}, errors.New("capture report contains inconsistent NVIDIA driver identities")
		}
		result.Devices = append(result.Devices, snapshot.DeviceCompatibility{
			Slot: slot, GPUName: gpu.Name, ComputeCapability: gpu.ComputeCapability,
		})
	}
	if err := result.Validate(len(unit.Devices)); err != nil {
		return snapshot.UnitPlatformCompatibility{}, err
	}
	return result, nil
}

func (adapter Adapter) publishModelPayloads(
	ctx context.Context,
	request snapshot.Request,
	paths map[string]string,
	payloads map[string]snapshot.Object,
	images []snapshot.CapsuleImage,
) (string, error) {
	workerCount := len(request.Launch.Execution.Workers)
	if len(paths) != workerCount || len(payloads) != workerCount || len(images) != len(request.Launch.Units) {
		return "", errors.New("model payload publication inventory is incomplete")
	}
	imagesByUnit, err := capsuleImagesByUnit(images)
	if err != nil {
		return "", err
	}
	policy := request.Policy.Weights.Native
	publisher, ok := adapter.Remote.(authenticatedNativePublisher)
	if !ok {
		return "", errors.New("configured remote transport does not support authenticated model payload publication")
	}
	output := adapter.Output
	if output == nil {
		output = io.Discard
	}
	publishedObjects := make(map[string]bool, workerCount)
	for _, worker := range request.Launch.Execution.Workers {
		unit, ok := launchUnit(request.Launch, worker.Unit)
		if !ok {
			return "", fmt.Errorf("model payload worker %s references unknown unit %s", worker.ID, worker.Unit)
		}
		pack := payloads[worker.ID]
		if publishedObjects[pack.Path] {
			fmt.Fprintf(
				output, "ColdSnap: reusing published model payload %s for worker %s\n",
				pack.Path, worker.ID,
			)
			continue
		}
		fmt.Fprintf(
			output, "ColdSnap: publishing worker %s model payload (%d bytes) to %s/%s\n",
			worker.ID, pack.Bytes, policy.Repository, pack.Path,
		)
		if err := publisher.PublishHuggingFaceFile(
			ctx, unit.Host, imagesByUnit[unit.ID].Reference, policy.Repository, policy.Revision,
			paths[worker.ID], pack.Path,
		); err != nil {
			return "", fmt.Errorf("publish worker %s model payload: %w", worker.ID, err)
		}
		publishedObjects[pack.Path] = true
	}
	revision, err := publisher.ResolveHuggingFaceRevision(
		ctx, request.Launch.Units[0].Host, imagesByUnit[request.Launch.Units[0].ID].Reference, policy.Repository, policy.Revision,
	)
	if err != nil {
		return "", fmt.Errorf("resolve published native provider commit: %w", err)
	}
	revision = strings.TrimSpace(revision)
	if !huggingFaceCommitPattern.MatchString(revision) {
		return "", errors.New("published native provider returned an invalid commit")
	}
	return revision, nil
}

func (adapter Adapter) stageCacheSeeds(
	ctx context.Context,
	request snapshot.Request,
	containers []string,
) ([]string, error) {
	roots := make([]string, len(request.Launch.Units))
	if !request.Policy.Cache.Seed || len(request.Policy.Cache.Paths) == 0 {
		return roots, nil
	}
	err := parallelUnits(request.Launch.Units, func(rank snapshot.LaunchUnit) error {
		root := filepath.Join(
			adapter.StateRoot,
			"capsule-cache-seeds",
			request.ID,
			"units", rank.ID,
		)
		if _, err := adapter.Remote.Run(ctx, rank.Host, "test", "!", "-e", root); err != nil {
			return fmt.Errorf("rank %d cache seed root already exists: %s", rank.Index, root)
		}
		if _, err := adapter.Remote.Run(ctx, rank.Host, "install", "-d", "-m", "0700", root); err != nil {
			return fmt.Errorf("create rank %d cache seed root: %w", rank.Index, err)
		}
		roots[rank.Index] = root
		for _, cachePath := range request.Policy.Cache.Paths {
			destination := filepath.Join(root, strings.TrimPrefix(cachePath, "/"))
			if _, err := adapter.Remote.Run(ctx, rank.Host, "install", "-d", "-m", "0700", filepath.Dir(destination)); err != nil {
				return fmt.Errorf("create rank %d cache seed parent: %w", rank.Index, err)
			}
			if _, err := adapter.copyWorkload(ctx, rank.Host, containers[rank.Index], cachePath, destination); err != nil {
				if missingWorkloadCopySource(err) {
					continue
				}
				return fmt.Errorf("copy rank %d derived cache %s: %w", rank.Index, cachePath, err)
			}
		}
		// Workload copying preserves the container's numeric ownership. Normalize the
		// completed seed before host-side inspection and before it becomes a
		// an image build context.
		if err := adapter.normalizeManagedPathOwnership(ctx, rank, rank.Image, root); err != nil {
			return fmt.Errorf("normalize rank %d cache seed ownership: %w", rank.Index, err)
		}
		for _, arguments := range [][]string{
			{"find", root, "-xdev", "-type", "l", "-print", "-quit"},
			{"find", root, "-xdev", "!", "-type", "d", "!", "-type", "f", "-print", "-quit"},
		} {
			output, err := adapter.Remote.Run(ctx, rank.Host, arguments...)
			if err != nil {
				return fmt.Errorf("inspect rank %d cache seed: %w", rank.Index, err)
			}
			if strings.TrimSpace(string(output)) != "" {
				return fmt.Errorf("rank %d cache seed contains a link or special file: %s", rank.Index, strings.TrimSpace(string(output)))
			}
		}
		return nil
	})
	if err != nil {
		adapter.removeCacheSeeds(ctx, request, roots)
		return nil, err
	}
	return roots, nil
}

func missingWorkloadCopySource(err error) bool {
	var failure *hostops.RuntimeError
	return errors.As(err, &failure) && failure.Code == "path_not_found"
}

func (adapter Adapter) removeCacheSeeds(
	parent context.Context,
	request snapshot.Request,
	roots []string,
) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), 30*time.Second)
	defer cancel()
	_ = parallelUnits(request.Launch.Units, func(rank snapshot.LaunchUnit) error {
		if rank.Index < len(roots) && roots[rank.Index] != "" {
			if _, err := adapter.Remote.Run(ctx, rank.Host, "rm", "-rf", roots[rank.Index]); err == nil {
				return nil
			}
			if err := adapter.normalizeManagedPathOwnership(ctx, rank, rank.Image, roots[rank.Index]); err != nil {
				fmt.Fprintf(adapter.Output, "ColdSnap: warning: cache-seed ownership cleanup failed: %v\n", err)
				return nil
			}
			if _, err := adapter.Remote.Run(ctx, rank.Host, "rm", "-rf", roots[rank.Index]); err != nil {
				fmt.Fprintf(adapter.Output, "ColdSnap: warning: cache-seed removal failed for %s: %v\n", roots[rank.Index], err)
			}
		}
		return nil
	})
}

type hydrationManifest struct {
	Worker   string
	Path     string
	Relative string
}

func (adapter Adapter) hydrationManifests(
	ctx context.Context, launch snapshot.LaunchSpec, unit snapshot.LaunchUnit, root string,
) ([]hydrationManifest, error) {
	output, err := adapter.Remote.Run(ctx, unit.Host, "find", filepath.Join(root, "hydration"), "-mindepth", "2", "-maxdepth", "2", "-type", "f", "-name", "manifest.json", "-print")
	if err != nil {
		return nil, err
	}
	lines := strings.Fields(strings.TrimSpace(string(output)))
	expectedWorkers := launch.Execution.UnitWorkers(unit.ID)
	if len(lines) != len(expectedWorkers) {
		return nil, fmt.Errorf("capture contains %d hydration manifests for %d workers", len(lines), len(expectedWorkers))
	}
	expected := make(map[string]bool, len(expectedWorkers))
	for _, worker := range expectedWorkers {
		expected[worker.ID] = true
	}
	result := make([]hydrationManifest, 0, len(lines))
	seen := make(map[string]bool, len(lines))
	for _, path := range lines {
		if !strings.HasPrefix(path, root+string(filepath.Separator)) {
			return nil, errors.New("hydration manifest escaped capture root")
		}
		relative, relativeErr := filepath.Rel(root, path)
		if relativeErr != nil || strings.HasPrefix(relative, "..") {
			return nil, errors.New("hydration manifest escaped capture root")
		}
		payload, readErr := adapter.Remote.Run(ctx, unit.Host, "cat", path)
		if readErr != nil {
			return nil, fmt.Errorf("read hydration manifest %s: %w", path, readErr)
		}
		var identity struct {
			Worker string `json:"worker_id"`
		}
		if decodeErr := json.Unmarshal(payload, &identity); decodeErr != nil {
			return nil, fmt.Errorf("decode hydration manifest %s: %w", path, decodeErr)
		}
		if !expected[identity.Worker] || seen[identity.Worker] {
			return nil, fmt.Errorf("hydration manifest has invalid or duplicate worker %q", identity.Worker)
		}
		seen[identity.Worker] = true
		result = append(result, hydrationManifest{Worker: identity.Worker, Path: path, Relative: relative})
	}
	slices.SortFunc(result, func(left, right hydrationManifest) int {
		leftWorker, _ := launch.Execution.Worker(left.Worker)
		rightWorker, _ := launch.Execution.Worker(right.Worker)
		return leftWorker.ProcessSlot - rightWorker.ProcessSlot
	})
	return result, nil
}

func (adapter Adapter) remoteObject(
	ctx context.Context,
	unit snapshot.LaunchUnit,
	owner string,
	path, role, logicalPath string,
) (snapshot.Object, error) {
	sizeOutput, err := adapter.Remote.Run(ctx, unit.Host, "stat", "-c", "%s", path)
	if err != nil {
		return snapshot.Object{}, fmt.Errorf("stat unit %s %s: %w", unit.ID, role, err)
	}
	bytes, err := strconv.ParseInt(strings.TrimSpace(string(sizeOutput)), 10, 64)
	if err != nil || bytes <= 0 {
		return snapshot.Object{}, fmt.Errorf("unit %s %s has invalid size", unit.ID, role)
	}
	hashOutput, err := adapter.Remote.Run(ctx, unit.Host, "sha256sum", path)
	if err != nil {
		return snapshot.Object{}, fmt.Errorf("hash unit %s %s: %w", unit.ID, role, err)
	}
	fields := strings.Fields(string(hashOutput))
	if len(fields) == 0 || len(fields[0]) != 64 {
		return snapshot.Object{}, fmt.Errorf("unit %s %s has invalid SHA-256", unit.ID, role)
	}
	return snapshot.Object{
		Role: role, Owner: owner, Path: filepath.ToSlash(logicalPath), Bytes: bytes,
		SHA256: "sha256:" + fields[0],
	}, nil
}

func (adapter Adapter) remoteModelPayloadObject(
	ctx context.Context,
	unit snapshot.LaunchUnit,
	owner, path, logicalPath string,
	expectedBytes int64,
	expectedSHA256 string,
) (snapshot.Object, error) {
	record := path + ".coldsnap-validation.json"
	output, err := adapter.runPayloadVerifier(
		ctx, unit.Host, path, record, expectedSHA256, expectedBytes, "",
	)
	if err != nil {
		return snapshot.Object{}, fmt.Errorf("admit unit %s model-weight-payload: %w", unit.ID, err)
	}
	var admitted remotePayloadAdmission
	if err := json.Unmarshal(output, &admitted); err != nil {
		return snapshot.Object{}, fmt.Errorf("decode unit %s model-weight-payload admission: %w", unit.ID, err)
	}
	if !admitted.valid() || admitted.Bytes <= 0 || len(admitted.SHA256) != len("sha256:")+sha256.Size*2 ||
		!strings.HasPrefix(admitted.SHA256, "sha256:") || admitted.Validation.Provider != "sha256-cache-v1" {
		return snapshot.Object{}, fmt.Errorf("unit %s model-weight-payload admission is invalid", unit.ID)
	}
	return snapshot.Object{
		Role: "model-weight-payload", Owner: owner, Path: filepath.ToSlash(logicalPath),
		Bytes: admitted.Bytes, SHA256: admitted.SHA256,
	}, nil
}

func modelPayloadObjectPath(digest string) string {
	return filepath.ToSlash(filepath.Join(
		"model-payloads", "sha256", strings.TrimPrefix(digest, "sha256:")+".pack",
	))
}

func (adapter Adapter) verifyStagedPayloads(
	ctx context.Context, request snapshot.Request, payloads []snapshot.PreparedPayload,
) error {
	if len(payloads) != len(request.Launch.Execution.Workers) {
		return errors.New("model payload selection does not cover every worker")
	}
	payloadsByWorker := preparedPayloadsByWorker(payloads)
	return parallelWorkers(request.Launch, func(unit snapshot.LaunchUnit, worker snapshot.Worker) error {
		pack, ok := payloadsByWorker[worker.ID]
		if !ok {
			return fmt.Errorf("model payload selection lacks worker %s", worker.ID)
		}
		if pack.Validation != nil {
			validation, err := adapter.remotePayloadValidation(ctx, unit, pack)
			if err != nil {
				return fmt.Errorf("validate worker %s model payload: %w", worker.ID, err)
			}
			if validation.Provider != "sha256-cache-v1" || validation.Size != pack.Bytes {
				return fmt.Errorf("worker %s model payload validation result is invalid", worker.ID)
			}
			return nil
		}
		object, err := adapter.remoteObject(
			ctx, unit, snapshot.WorkerOwner(worker.ID), pack.Path,
			"model-weight-payload", modelPayloadObjectPath(pack.SHA256),
		)
		if err != nil {
			return err
		}
		if object.Bytes != pack.Bytes || object.SHA256 != pack.SHA256 {
			return fmt.Errorf("worker %s model payload changed after staging", worker.ID)
		}
		return nil
	})
}

type remotePayloadAdmission struct {
	Format     int                                `json:"format"`
	Kind       string                             `json:"kind"`
	Decision   string                             `json:"decision"`
	Bytes      int64                              `json:"bytes"`
	SHA256     string                             `json:"sha256"`
	Validation snapshot.PreparedPayloadValidation `json:"validation"`
}

func (admission remotePayloadAdmission) valid() bool {
	return admission.Format == 1 && admission.Kind == "coldsnap-payload-validation-result" &&
		admission.Decision == "accept"
}

func (adapter Adapter) remotePayloadValidation(
	ctx context.Context, rank snapshot.LaunchUnit, pack snapshot.PreparedPayload,
) (snapshot.PreparedPayloadValidation, error) {
	output, err := adapter.runPayloadVerifier(
		ctx, rank.Host, pack.Path, pack.Validation.Record, pack.SHA256, pack.Bytes, pack.Worker,
	)
	if err != nil {
		return snapshot.PreparedPayloadValidation{}, err
	}
	var admitted remotePayloadAdmission
	if err := json.Unmarshal(output, &admitted); err != nil {
		return snapshot.PreparedPayloadValidation{}, fmt.Errorf("decode model payload validation: %w", err)
	}
	validation := admitted.Validation
	if !admitted.valid() || admitted.Bytes != pack.Bytes || admitted.SHA256 != pack.SHA256 ||
		validation.Provider != "sha256-cache-v1" || validation.Device == 0 ||
		validation.Inode == 0 || validation.Size <= 0 || validation.MTimeNS <= 0 {
		return snapshot.PreparedPayloadValidation{}, errors.New("model payload validation is invalid")
	}
	return validation, nil
}

func (adapter Adapter) runPayloadVerifier(
	ctx context.Context,
	host, path, record, expectedSHA256 string,
	expectedBytes int64,
	worker string,
) ([]byte, error) {
	pack, err := activationRuntimePack()
	if err != nil {
		return nil, err
	}
	verifier, err := payloadVerifierActivationPath(adapter.StateRoot, pack)
	if err != nil {
		return nil, err
	}
	arguments := []string{
		verifier, "payload-verify",
		"--path", path,
		"--record", record,
		"--expected-sha256", expectedSHA256,
		"--expected-bytes", strconv.FormatInt(expectedBytes, 10),
	}
	if worker != "" {
		arguments = append(arguments, "--worker", worker)
	}
	return adapter.Remote.Run(ctx, host, arguments...)
}

func (adapter Adapter) collectFailureLogs(
	ctx context.Context, request snapshot.Request, containers []string,
) {
	for _, rank := range request.Launch.Units {
		if containers[rank.Index] == "" {
			continue
		}
		if output, err := adapter.logsWorkload(ctx, rank.Host, containers[rank.Index], 200); err == nil {
			fmt.Fprintf(adapter.Output, "rank %d logs:\n%s\n", rank.Index, output)
		}
		root := filepath.Join(
			adapter.StateRoot,
			"failures",
			request.ID,
			fmt.Sprintf("rank%d", rank.Index),
		)
		if _, err := adapter.Remote.Run(ctx, rank.Host, "install", "-d", "-m", "0700", root); err != nil {
			continue
		}
		type failureLog struct {
			source string
			label  string
		}
		var destinations []failureLog
		for _, item := range []failureLog{
			{source: containerArtifactRoot + "/controller-error.json", label: "current activation controller error"},
			{
				source: containerCaptureLogPath,
				label:  "capture-time target log",
			},
			{
				source: containerTargetLogPath,
				label:  "current activation target log",
			},
			{source: containerArtifactRoot + "/restore-rpc.json", label: "current activation restore RPC"},
			{source: containerArtifactRoot + "/work/restore.log", label: "current activation CRIU restore log"},
		} {
			source := item.source
			destination := filepath.Join(root, strings.ReplaceAll(strings.TrimPrefix(source, containerArtifactRoot+"/"), "/", "-"))
			if _, err := adapter.copyWorkload(ctx, rank.Host, containers[rank.Index], source, destination); err != nil {
				continue
			}
			destinations = append(destinations, failureLog{source: destination, label: item.label})
		}
		image := rank.Image
		if output, err := adapter.workloadImage(ctx, rank.Host, containers[rank.Index]); err == nil && strings.TrimSpace(string(output)) != "" {
			image = strings.TrimSpace(string(output))
		}
		if err := adapter.normalizeManagedPathOwnership(ctx, rank, image, root); err != nil {
			fmt.Fprintf(adapter.Output, "ColdSnap: warning: failure-artifact ownership cleanup failed: %v\n", err)
		}
		for _, destination := range destinations {
			if output, err := adapter.Remote.Run(ctx, rank.Host, "tail", "-c", "32768", destination.source); err == nil {
				fmt.Fprintf(
					adapter.Output, "rank %d %s [%s]:\n%s\n",
					rank.Index, filepath.Base(destination.source), destination.label, output,
				)
			}
		}
		fmt.Fprintf(adapter.Output, "rank %d failure artifacts: %s:%s\n", rank.Index, rank.Host, root)
	}
}

func compatibleLaunch(captured, requested snapshot.LaunchSpec) error {
	if captured.Engine != requested.Engine || captured.Model != requested.Model ||
		len(captured.Units) != len(requested.Units) ||
		captured.Execution.Adapter.Schema != requested.Execution.Adapter.Schema ||
		captured.Execution.Adapter.Digest != requested.Execution.Adapter.Digest ||
		!reflect.DeepEqual(captured.Execution.Workers, requested.Execution.Workers) ||
		!reflect.DeepEqual(captured.Execution.Groups, requested.Execution.Groups) ||
		!reflect.DeepEqual(captured.Execution.Services, requested.Execution.Services) {
		return errors.New("restore launch engine, model, or topology differs from the capture")
	}
	for index := range captured.Units {
		before, after := captured.Units[index], requested.Units[index]
		// The capsule digest, not the recipe's capture-time builder image, pins
		// the process userspace used for restore. Host, GPU ordinal, mount source,
		// and transport environment are destination placement. Mount targets,
		// non-transport environment, and the semantic serve command remain process
		// identity.
		if before.ID != after.ID || before.Index != after.Index || len(before.Devices) != len(after.Devices) ||
			!slices.Equal(portableCommand(before.Command), portableCommand(after.Command)) ||
			!maps.Equal(portableEnvironment(before.Environment), portableEnvironment(after.Environment)) ||
			!slices.Equal(portableMounts(before.Mounts), portableMounts(after.Mounts)) {
			return fmt.Errorf("restore launch identity differs for unit %s", before.ID)
		}
	}
	return nil
}

var placementCommandArgument = regexp.MustCompile(`(^|[[:space:]])(--(?:master-(?:addr(?:ess)?|port)|dist-init-addr|port))(?:[[:space:]]+|=)(?:'[^']*'|"[^"]*"|[^[:space:]]+)`)

var placementTransportEnvironmentNames = map[string]bool{
	"GLOO_SOCKET_IFNAME": true,
	"MN_IF_NAME":         true,
	"NODE_IP":            true,
	"TP_SOCKET_IFNAME":   true,
	"UCX_NET_DEVICES":    true,
	"VLLM_HOST_IP":       true,
}

func isPlacementTransportEnvironment(name string) bool {
	return placementTransportEnvironmentNames[name] ||
		strings.HasPrefix(name, "NCCL_") || strings.HasPrefix(name, "OMPI_MCA_") ||
		strings.HasPrefix(name, "UCX_")
}

func portableEnvironment(environment map[string]string) map[string]string {
	result := maps.Clone(environment)
	for name := range result {
		if isPlacementTransportEnvironment(name) {
			delete(result, name)
		}
	}
	return result
}

func portableMounts(mounts []snapshot.Mount) []snapshot.Mount {
	result := slices.Clone(mounts)
	for index := range result {
		result[index].Source = "<placement>"
	}
	slices.SortFunc(result, func(left, right snapshot.Mount) int {
		return strings.Compare(left.Target, right.Target)
	})
	return result
}

func portableCommand(command []string) []string {
	result := slices.Clone(command)
	for index, value := range result {
		result[index] = placementCommandArgument.ReplaceAllString(value, `${1}${2}=<placement>`)
	}
	for index := 0; index+1 < len(result); index++ {
		switch result[index] {
		case "--master-addr", "--master-address", "--master-port", "--dist-init-addr", "--port":
			result[index+1] = "<placement>"
		}
	}
	return result
}

func rankPlacementAddress(rank snapshot.LaunchUnit) (netip.Addr, error) {
	value := rank.Environment["VLLM_HOST_IP"]
	if value == "" {
		value = rank.Environment["NODE_IP"]
	}
	if value == "" {
		value = rank.Host
	}
	address, err := netip.ParseAddr(value)
	if err != nil || address.Zone() != "" {
		return netip.Addr{}, fmt.Errorf("rank %d placement address %q is not an IP address", rank.Index, value)
	}
	address = address.Unmap()
	if !address.IsValid() || address.IsUnspecified() || address.IsLoopback() ||
		address.IsMulticast() || address.IsLinkLocalUnicast() {
		return netip.Addr{}, fmt.Errorf("rank %d placement address %q is not portable", rank.Index, value)
	}
	return address, nil
}

func placementTCPAddressMap(captured, requested snapshot.LaunchSpec) ([]string, error) {
	if len(captured.Units) != len(requested.Units) {
		return nil, errors.New("restore launch rank count differs from the capture")
	}
	result := make([]string, 0, len(captured.Units))
	oldToNew := make(map[netip.Addr]netip.Addr, len(captured.Units))
	newToOld := make(map[netip.Addr]netip.Addr, len(captured.Units))
	for index := range captured.Units {
		oldAddress, err := rankPlacementAddress(captured.Units[index])
		if err != nil {
			return nil, fmt.Errorf("captured placement: %w", err)
		}
		newAddress, err := rankPlacementAddress(requested.Units[index])
		if err != nil {
			return nil, fmt.Errorf("restore placement: %w", err)
		}
		if previous, ok := oldToNew[oldAddress]; ok && previous != newAddress {
			return nil, errors.New("one captured host address maps to multiple restore hosts")
		}
		if previous, ok := newToOld[newAddress]; ok && previous != oldAddress {
			return nil, errors.New("multiple captured host addresses map to one restore host")
		}
		oldToNew[oldAddress], newToOld[newAddress] = newAddress, oldAddress
		if oldAddress.Is4() != newAddress.Is4() {
			return nil, fmt.Errorf("unit %s restore placement changes IP family", captured.Units[index].ID)
		}
		mapping := oldAddress.String() + "=" + newAddress.String()
		if !slices.Contains(result, mapping) {
			result = append(result, mapping)
		}
	}
	return result, nil
}

func portableTCPPortShift(activationNamespace string) uint16 {
	return portableTCPPortShiftCandidate(activationNamespace, 0)
}

func portableTCPPortShiftCandidate(activationNamespace string, attempt int) uint16 {
	identity := activationNamespace
	if attempt != 0 {
		identity += "\x00" + strconv.Itoa(attempt)
	}
	digest := sha256.Sum256([]byte(identity))
	// A non-zero rotation gives every activation a fresh
	// transport identity while remaining identical on every logical rank.
	return uint16(binary.BigEndian.Uint32(digest[:4])%(portableTCPPortCount-1) + 1)
}

const (
	portableTCPPortBase  = uint32(1024)
	portableTCPPortCount = uint32(65536) - portableTCPPortBase
)

func portableHTTPPortPolicy(captured, requested snapshot.LaunchSpec) (uint16, bool, error) {
	if len(captured.Units) != len(requested.Units) {
		return 0, false, errors.New("restore launch rank count differs from the capture")
	}
	var selected uint32
	for index := range captured.Units {
		before, after := httpPort(captured.Units[index].Command), httpPort(requested.Units[index].Command)
		if before < int(portableTCPPortBase) || after < int(portableTCPPortBase) {
			if before != after {
				return 0, false, fmt.Errorf(
					"unit %s cannot migrate a privileged serving port from %d to %d",
					captured.Units[index].ID, before, after,
				)
			}
			continue
		}
		beforeOffset := uint32(before) - portableTCPPortBase
		afterOffset := uint32(after) - portableTCPPortBase
		shift := (afterOffset + portableTCPPortCount - beforeOffset) % portableTCPPortCount
		if index == 0 {
			selected = shift
		} else if selected != shift {
			return 0, false, errors.New("restore units require inconsistent serving-port mappings")
		}
	}
	if selected == 0 {
		return 0, true, nil
	}
	return uint16(selected), false, nil
}

func (adapter Adapter) selectPortableTCPPortShift(
	ctx context.Context,
	request snapshot.Request,
	captured snapshot.LaunchSpec,
	imagesByUnit map[string]snapshot.CapsuleImage,
	addressMapping []string,
	activationNamespace string,
) (uint16, error) {
	const maximumCandidates = 32
	fixedShift, preserveHTTPPort, err := portableHTTPPortPolicy(captured, request.Launch)
	if err != nil {
		return 0, err
	}
	activationPack, err := activationRuntimePack()
	if err != nil {
		return 0, err
	}
	criuRPCBinding, hasCRIURPCOverlay := criuRPCActivationBinding(adapter.StateRoot, activationPack)
	portMapping := ""
	if fixedShift != 0 {
		portMapping = fmt.Sprintf(
			"%d=%d", httpPort(captured.Units[0].Command), httpPort(request.Launch.Units[0].Command),
		)
	}
	candidates := maximumCandidates
	if fixedShift != 0 {
		candidates = 1
	}
	var lastErr error
	seen := make(map[uint16]bool, candidates)
	for attempt := 0; attempt < candidates; attempt++ {
		candidate := portableTCPPortShiftCandidate(activationNamespace, attempt)
		if fixedShift != 0 {
			candidate = fixedShift
		}
		if seen[candidate] {
			continue
		}
		seen[candidate] = true
		var collisionMutex sync.Mutex
		var collisions []string
		err := parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
			image, ok := imagesByUnit[unit.ID]
			if !ok {
				return fmt.Errorf("capsule image inventory lacks unit %s", unit.ID)
			}
			requiresOverlay := fixedShift != 0 || image.Driver.ID == snapshotdriver.N580
			if requiresOverlay && !hasCRIURPCOverlay {
				return errors.New("portable n580 or serving-port restore requires the release-matched CRIU RPC helper")
			}
			spec := &hostops.WorkloadSpec{Image: image.Reference, RemoveAfterExit: true,
				PullPolicy: "never", Network: "host", Entrypoint: "/usr/local/bin/coldsnap-criu-rpc"}
			if hasCRIURPCOverlay {
				spec.Mounts = append(spec.Mounts, hostops.Mount{Source: criuRPCBinding.HostPath, Target: criuRPCBinding.ContainerPath, ReadOnly: true})
			}
			command := []string{"tcp-probe", "--images-dir", capsuleTCPImagesPath(image),
				"--tcp-port-shift", strconv.Itoa(int(candidate))}
			if image.Driver.ID == snapshotdriver.N580 {
				command = append(command, "--tcp-allow-empty-map")
			}
			if portMapping != "" {
				command = append(command, "--tcp-port-map", portMapping)
			}
			if preserveHTTPPort {
				command = append(command, "--tcp-preserve-port", strconv.Itoa(httpPort(unit.Command)))
			}
			for _, mapping := range addressMapping {
				command = append(command, "--tcp-address-map", mapping)
			}
			spec.Command = command
			output, err := adapter.runWorkload(ctx, unit.Host, spec, false)
			if err != nil {
				return fmt.Errorf("unit %s rejected TCP port shift %d: %w", unit.ID, candidate, err)
			}
			var receipt struct {
				Format    int    `json:"format"`
				Kind      string `json:"kind"`
				PortShift uint16 `json:"port_shift"`
				Endpoints int    `json:"endpoints"`
				Available *bool  `json:"available"`
				Collision string `json:"collision"`
			}
			minimumEndpoints := 1
			if image.Driver.ID == snapshotdriver.N580 {
				minimumEndpoints = 0
			}
			if err := json.Unmarshal(output, &receipt); err != nil ||
				receipt.Format != 1 || receipt.Kind != "coldsnap-criu-tcp-port-probe" ||
				receipt.PortShift != candidate || receipt.Endpoints < minimumEndpoints || receipt.Available == nil ||
				(!*receipt.Available && receipt.Collision == "") {
				return fmt.Errorf("unit %s returned an invalid TCP port probe receipt", unit.ID)
			}
			if !*receipt.Available {
				collisionMutex.Lock()
				collisions = append(collisions, fmt.Sprintf("unit %s: %s", unit.ID, receipt.Collision))
				collisionMutex.Unlock()
			}
			return nil
		})
		if err != nil {
			return 0, fmt.Errorf("inspect TCP port generation %d: %w", candidate, err)
		}
		if len(collisions) == 0 {
			if attempt != 0 {
				fmt.Fprintf(
					adapter.Output,
					"ColdSnap: selected TCP port shift %d after rejecting %d occupied candidate(s)\n",
					candidate,
					attempt,
				)
			}
			return candidate, nil
		}
		slices.Sort(collisions)
		lastErr = errors.New(strings.Join(collisions, "; "))
	}
	if fixedShift != 0 {
		return 0, fmt.Errorf(
			"requested serving-port mapping is unavailable across %d restore unit(s): %w",
			len(request.Launch.Units), lastErr,
		)
	}
	return 0, fmt.Errorf(
		"no collision-free TCP port generation found across %d restore unit(s) after %d candidates: %w",
		len(request.Launch.Units), maximumCandidates, lastErr,
	)
}

func capsuleTCPImagesPath(image snapshot.CapsuleImage) string {
	root := image.Root
	if root == "" {
		root = capsule.Root
	}
	if image.Driver.ID == snapshotdriver.N580 {
		return filepath.Join(root, "template", "images")
	}
	return filepath.Join(root, "images")
}

func modelPayloadTarget(request snapshot.Request, workerID string) (string, error) {
	artifactPath, err := artifactInputPath(request.Artifact)
	if err != nil {
		return "", err
	}
	artifact, err := snapshot.ReadArtifact(artifactPath)
	if err != nil {
		return "", err
	}
	worker, ok := artifact.Launch.Execution.Worker(workerID)
	if !ok {
		return "", fmt.Errorf("artifact has no worker %s", workerID)
	}
	// The n580 rank controller must choose and link the native payload before
	// CRIU releases the restored pre-CUDA process template. Keep its verified
	// worker-scoped staging mount independent of the replay-plan path used by
	// the worker after release.
	if artifact.Driver.ID == snapshotdriver.N580 && artifact.Weights.Native != nil {
		return filepath.Join(containerModelPayloadRoot, workerID+".pack"), nil
	}
	for _, object := range artifact.Weights.Recovery.ReplayPlan {
		if object.Owner != snapshot.WorkerOwner(workerID) {
			continue
		}
		prefix := filepath.ToSlash(filepath.Join(
			"drivers", artifact.Driver.ID, "units", worker.Unit,
		)) + "/"
		relative := strings.TrimPrefix(object.Path, prefix)
		if relative == object.Path || filepath.Base(relative) != "manifest.json" {
			return "", fmt.Errorf("worker %s recovery plan path is invalid", workerID)
		}
		return filepath.Join(capsule.Root, filepath.Dir(relative), modelPayloadName), nil
	}
	return "", fmt.Errorf("artifact has no recovery plan for worker %s", workerID)
}

func parallelUnits(units []snapshot.LaunchUnit, operation func(snapshot.LaunchUnit) error) error {
	var wait sync.WaitGroup
	errorsByUnit := make([]error, len(units))
	for _, unit := range units {
		unit := unit
		wait.Add(1)
		go func() {
			defer wait.Done()
			errorsByUnit[unit.Index] = operation(unit)
		}()
	}
	wait.Wait()
	var failures []string
	for index, err := range errorsByUnit {
		if err != nil {
			failures = append(failures, fmt.Sprintf("unit %s: %v", units[index].ID, err))
		}
	}
	if len(failures) != 0 {
		return errors.New(strings.Join(failures, "; "))
	}
	return nil
}

func parallelWorkers(
	launch snapshot.LaunchSpec,
	operation func(snapshot.LaunchUnit, snapshot.Worker) error,
) error {
	var wait sync.WaitGroup
	errorsByWorker := make([]error, len(launch.Execution.Workers))
	for index, worker := range launch.Execution.Workers {
		index, worker := index, worker
		unit, ok := launchUnit(launch, worker.Unit)
		if !ok {
			return fmt.Errorf("worker %s references unknown unit %s", worker.ID, worker.Unit)
		}
		wait.Add(1)
		go func() {
			defer wait.Done()
			errorsByWorker[index] = operation(unit, worker)
		}()
	}
	wait.Wait()
	var failures []string
	for index, err := range errorsByWorker {
		if err != nil {
			failures = append(failures, fmt.Sprintf("worker %s: %v", launch.Execution.Workers[index].ID, err))
		}
	}
	if len(failures) != 0 {
		return errors.New(strings.Join(failures, "; "))
	}
	return nil
}

func launchUnit(launch snapshot.LaunchSpec, unitID string) (snapshot.LaunchUnit, bool) {
	for _, unit := range launch.Units {
		if unit.ID == unitID {
			return unit, true
		}
	}
	return snapshot.LaunchUnit{}, false
}

func capsuleImagesByUnit(images []snapshot.CapsuleImage) (map[string]snapshot.CapsuleImage, error) {
	result := make(map[string]snapshot.CapsuleImage, len(images))
	for _, image := range images {
		if image.Unit == "" || result[image.Unit].Unit != "" {
			return nil, errors.New("capsule image inventory contains an invalid or duplicate unit")
		}
		result[image.Unit] = image
	}
	return result, nil
}

func preparedPayloadsByWorker(payloads []snapshot.PreparedPayload) map[string]snapshot.PreparedPayload {
	result := make(map[string]snapshot.PreparedPayload, len(payloads))
	for _, payload := range payloads {
		result[payload.Worker] = payload
	}
	return result
}

func workerMapJSON(launch snapshot.LaunchSpec, unitID string) (string, error) {
	type groupRanks struct {
		Kind  string         `json:"kind"`
		Size  int            `json:"size"`
		Ranks map[int]string `json:"ranks"`
	}
	value := struct {
		Unit          string                `json:"unit"`
		ByProcessSlot map[int]string        `json:"by_process_slot"`
		Groups        map[string]groupRanks `json:"groups"`
	}{Unit: unitID, ByProcessSlot: make(map[int]string), Groups: make(map[string]groupRanks)}
	unitWorkers := make(map[string]bool)
	for _, worker := range launch.Execution.UnitWorkers(unitID) {
		value.ByProcessSlot[worker.ProcessSlot] = worker.ID
		unitWorkers[worker.ID] = true
	}
	if len(unitWorkers) == 0 {
		return "", fmt.Errorf("unit %s has no workers", unitID)
	}
	for _, group := range launch.Execution.Groups {
		ranks := make(map[int]string)
		for rank, workerID := range group.Members {
			if unitWorkers[workerID] {
				ranks[rank] = workerID
			}
		}
		if len(ranks) != 0 {
			value.Groups[group.ID] = groupRanks{Kind: group.Kind, Size: len(group.Members), Ranks: ranks}
		}
	}
	payload, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("encode unit %s worker map: %w", unitID, err)
	}
	return string(payload), nil
}

func tensorParallelSize(launch snapshot.LaunchSpec) (int, error) {
	var topology struct {
		Dimensions map[string]int `json:"dimensions"`
	}
	if err := json.Unmarshal(launch.Execution.Adapter.Payload, &topology); err != nil {
		return 0, fmt.Errorf("decode vLLM adapter topology: %w", err)
	}
	if topology.Dimensions["tensor"] <= 0 {
		return 0, errors.New("vLLM adapter topology lacks a positive tensor dimension")
	}
	return topology.Dimensions["tensor"], nil
}

func artifactOutputPath(value string) (string, error) {
	path, err := filepath.Abs(value)
	if err != nil {
		return "", fmt.Errorf("resolve artifact output: %w", err)
	}
	if filepath.Ext(path) == ".json" {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			return "", err
		}
		return path, nil
	}
	if err := os.MkdirAll(path, 0o700); err != nil {
		return "", err
	}
	return filepath.Join(path, "artifact.json"), nil
}

func artifactInputPath(value string) (string, error) {
	path, err := filepath.Abs(value)
	if err != nil {
		return "", fmt.Errorf("resolve artifact input: %w", err)
	}
	if information, statErr := os.Stat(path); statErr == nil && information.IsDir() {
		path = filepath.Join(path, "artifact.json")
	}
	return path, nil
}

func operationName(id, suffix string) string {
	value := strings.ToLower(id + "-" + suffix)
	var result strings.Builder
	for _, character := range value {
		if character >= 'a' && character <= 'z' || character >= '0' && character <= '9' || character == '.' || character == '_' || character == '-' {
			result.WriteRune(character)
		} else {
			result.WriteByte('-')
		}
	}
	return strings.Trim(result.String(), ".-")
}

var portArgument = regexp.MustCompile(`(^|[[:space:]])--port(?:[[:space:]]+|=)(?:'([0-9]+)'|"([0-9]+)"|([0-9]+))`)

func httpPort(command []string) int {
	match := portArgument.FindStringSubmatch(strings.Join(command, " "))
	if len(match) != 0 {
		for _, value := range match[2:] {
			if value == "" {
				continue
			}
			if port, err := strconv.Atoi(value); err == nil && port > 0 && port < 65536 {
				return port
			}
		}
	}
	return 8000
}

func boolText(value bool) string {
	if value {
		return "1"
	}
	return "0"
}

func wantsModelPayload(policy snapshot.WeightPolicy) bool {
	return policy.Mode == "auto" || policy.Mode == "cache-only-auto" ||
		policy.Mode == "native" ||
		policy.Native.Repository != ""
}

var loadFormatArgument = regexp.MustCompile(`(^|[[:space:]])--load-format(?:[[:space:]]+|=)(?:'([^']*)'|"([^"]*)"|([^[:space:]]+))`)

func commandLoadFormat(command []string) string {
	match := loadFormatArgument.FindStringSubmatch(strings.Join(command, " "))
	if len(match) == 0 {
		return "auto"
	}
	for _, value := range match[2:] {
		if value != "" {
			return strings.ToLower(value)
		}
	}
	return "auto"
}

func recoveryLoadPath(engine string) string {
	if engine == "sglang" {
		return "sglang-inplace-disk"
	}
	return "coldsnap-replay"
}

func processTemplateRecoveryLoadPath(engine string) string {
	if engine == "sglang" {
		return "sglang-startup-disk"
	}
	return recoveryLoadPath(engine)
}

func recoveryCapableCommand(engine string, command []string) bool {
	loadFormat := commandLoadFormat(command)
	joined := strings.Join(command, " ")
	if engine == "sglang" {
		launcher := strings.Contains(joined, "sglang serve") ||
			strings.Contains(joined, "-m sglang.launch_server")
		return launcher && slices.Contains([]string{"auto", "safetensors"}, loadFormat)
	}
	return slices.Contains(
		[]string{"auto", "safetensors", "fastsafetensors", "instanttensor", "coldsnap"},
		loadFormat,
	) && strings.Contains(joined, "--enable-sleep-mode")
}

func waitContext(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

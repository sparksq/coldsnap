// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/sparksq/coldsnap/internal/canonicaljson"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

const (
	RequestFormat                 = 4
	RequestKind                   = "coldsnap-operation-request"
	ArtifactFormat                = 9
	ArtifactKind                  = "coldsnap-snapshot-artifact"
	PortabilityPolicy             = "placement-independent-v1"
	KernelCompatibilityCapability = "capability"
	KernelCompatibilityExact      = "exact"
	ArtifactScopePortable         = "portable"
	ArtifactScopeTargetLocal      = "target-local"
)

var (
	idPattern             = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
	digestPattern         = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	envPattern            = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	numericVersionPattern = regexp.MustCompile(`^[0-9]+(?:\.[0-9]+)*$`)
)

type Request struct {
	Format     int                      `json:"format"`
	Kind       string                   `json:"kind"`
	Operation  string                   `json:"operation"`
	ID         string                   `json:"id"`
	Driver     snapshotdriver.Selection `json:"snapshot_driver"`
	Artifact   string                   `json:"artifact,omitempty"`
	Output     string                   `json:"output,omitempty"`
	Launch     LaunchSpec               `json:"launch"`
	Policy     SnapshotPolicy           `json:"policy"`
	Validation ValidationPolicy         `json:"validation"`
	Workload   WorkloadIdentity         `json:"workload,omitempty"`
	Lifecycle  LifecyclePolicy          `json:"lifecycle,omitempty"`
}

// LifecyclePolicy selects the state in which a restore activation returns.
// The empty value resolves to the normal running activation state.
type LifecyclePolicy struct {
	ActivationState string `json:"activation_state,omitempty"`
}

// WorkloadIdentity lets an orchestrator make restored containers first-class
// members of its normal lifecycle. It is optional for direct ColdSnap usage;
// when present, the adapter uses the canonical container names and labels
// supplied by that orchestrator instead of operation-scoped names.
type WorkloadIdentity struct {
	ClusterID       string `json:"cluster_id,omitempty"`
	IntentID        string `json:"intent_id,omitempty"`
	Recipe          string `json:"recipe,omitempty"`
	Runtime         string `json:"runtime,omitempty"`
	Model           string `json:"model,omitempty"`
	ServedModelName string `json:"served_model_name,omitempty"`
	LogPath         string `json:"log_path,omitempty"`
}

type LaunchSpec struct {
	Engine    string         `json:"engine"`
	Model     ModelSource    `json:"model"`
	Units     []LaunchUnit   `json:"units"`
	Execution ExecutionGraph `json:"execution"`
}

type ModelSource struct {
	ID       string `json:"id"`
	Revision string `json:"revision"`
	Source   string `json:"source,omitempty"`
}

type LaunchUnit struct {
	ID          string            `json:"id"`
	Index       int               `json:"index"`
	Host        string            `json:"host"`
	Devices     []string          `json:"devices"`
	Image       string            `json:"image"`
	ImageDigest string            `json:"image_digest"`
	Command     []string          `json:"command"`
	Environment map[string]string `json:"environment,omitempty"`
	Mounts      []Mount           `json:"mounts,omitempty"`
}

type Mount struct {
	Source   string `json:"source"`
	Target   string `json:"target"`
	ReadOnly bool   `json:"read_only,omitempty"`
}

type SnapshotPolicy struct {
	Process       ProcessPolicy       `json:"process"`
	Weights       WeightPolicy        `json:"weights"`
	Cache         CachePolicy         `json:"cache"`
	Capsule       CapsulePolicy       `json:"capsule"`
	Compatibility CompatibilityPolicy `json:"compatibility"`
}

// CompatibilityPolicy controls restore admission checks which are stricter
// than the selected snapshot driver's own platform contract. Direct ColdSnap
// requests default to enforcing the driver version recorded at capture time.
type CompatibilityPolicy struct {
	EnforceCapturedDriverFloor bool `json:"enforce_captured_driver_floor"`
	// Kernel selects capability-based CRIU admission or the stricter exact
	// uname release check. The kernel release remains capture provenance in
	// either mode; it is not capsule identity in capability mode.
	Kernel string `json:"kernel"`
}

type ProcessPolicy struct {
	Backend          string `json:"backend"`
	KVDiscard        bool   `json:"kv_discard"`
	AsyncGraphs      bool   `json:"async_graphs"`
	GraphPolicy      string `json:"graph_policy,omitempty"`
	ShapeCalibration string `json:"shape_calibration"`
	ArtifactScope    string `json:"artifact_scope"`
}

type WeightPolicy struct {
	Mode     string         `json:"mode"`
	Native   NativePolicy   `json:"native"`
	Recovery RecoveryPolicy `json:"recovery"`
}

type NativePolicy struct {
	Repository    string            `json:"repository,omitempty"`
	Revision      string            `json:"revision,omitempty"`
	FilesByWorker map[string]string `json:"files_by_worker,omitempty"`
	Staged        []PreparedPayload `json:"staged,omitempty"`
	Materialize   string            `json:"materialize,omitempty"`
}

type RecoveryPolicy struct {
	Enabled       bool   `json:"enabled"`
	Source        string `json:"source"`
	LoaderBackend string `json:"loader_backend"`
}

type CachePolicy struct {
	Seed   bool            `json:"seed"`
	Paths  []string        `json:"paths,omitempty"`
	Staged []PreparedCache `json:"staged,omitempty"`
}

type PreparedCache struct {
	Unit string `json:"unit"`
	Path string `json:"path"`
}

// CapsulePolicy controls the per-unit OCI images which contain the process,
// CUDA, NCCL, and non-weight hydration state. Content-addressed model payloads
// remain a separate provider so capsules stay small and driver variants can
// share the same model bytes.
type CapsulePolicy struct {
	Repository string `json:"repository,omitempty"`
}

type ValidationPolicy struct {
	HealthPath string `json:"health_path"`
	Prompt     string `json:"prompt"`
	Expected   string `json:"expected"`
}

const CanonicalRuntimeCachePath = "/var/cache/coldsnap/runtime"

var defaultCachePaths = []string{CanonicalRuntimeCachePath}

var defaultSnapshotPolicies = map[string]SnapshotPolicy{
	snapshotdriver.N580: defaultN580SnapshotPolicy(),
	snapshotdriver.N610: defaultN610SnapshotPolicy(),
}

func newDefaultSnapshotPolicy() SnapshotPolicy {
	return SnapshotPolicy{
		Process: ProcessPolicy{
			Backend: "cuda-criu", KVDiscard: true, AsyncGraphs: true,
			GraphPolicy: GraphRecreateFromPlan, ShapeCalibration: "auto",
			ArtifactScope: ArtifactScopePortable,
		},
		Weights: WeightPolicy{
			Mode:   "auto",
			Native: NativePolicy{},
			Recovery: RecoveryPolicy{
				Enabled:       true,
				Source:        "huggingface-safetensors",
				LoaderBackend: "auto",
			},
		},
		Cache: CachePolicy{Seed: true, Paths: slices.Clone(defaultCachePaths)},
		Compatibility: CompatibilityPolicy{
			EnforceCapturedDriverFloor: true,
			Kernel:                     KernelCompatibilityCapability,
		},
	}
}

func defaultN580SnapshotPolicy() SnapshotPolicy {
	return newDefaultSnapshotPolicy()
}

func defaultN610SnapshotPolicy() SnapshotPolicy {
	policy := newDefaultSnapshotPolicy()
	policy.Process.GraphPolicy = GraphPreserveNCCLExec
	return policy
}

// DefaultSnapshotPolicyForDriver returns the policy owned by one snapshot
// implementation. Both current drivers prefer automatic native-provider
// selection with safetensors recovery. Other qualified settings may diverge
// without changing the request format.
func DefaultSnapshotPolicyForDriver(driverID string) (SnapshotPolicy, error) {
	policy, ok := defaultSnapshotPolicies[driverID]
	if !ok {
		return SnapshotPolicy{}, fmt.Errorf("unsupported snapshot driver %q", driverID)
	}
	policy.Cache.Paths = slices.Clone(policy.Cache.Paths)
	return policy, nil
}

// DefaultValidationPolicy provides a portable, model-independent acceptance
// check which recipes may override when necessary.
func DefaultValidationPolicy() ValidationPolicy {
	return ValidationPolicy{
		HealthPath: "/health",
		Prompt:     "Reply with exactly: coldsnap-cuda-snapshot-ok",
		Expected:   "coldsnap-cuda-snapshot-ok",
	}
}

func (request Request) Validate() error {
	if request.Format != RequestFormat || request.Kind != RequestKind {
		return fmt.Errorf("unsupported ColdSnap request format %d kind %q", request.Format, request.Kind)
	}
	if !slices.Contains([]string{"capture", "publish", "publish-native", "restore", "sleep", "wake", "status"}, request.Operation) {
		return fmt.Errorf("unsupported ColdSnap operation %q", request.Operation)
	}
	if !idPattern.MatchString(request.ID) {
		return errors.New("ColdSnap request ID is invalid")
	}
	if err := request.Driver.Validate(); err != nil {
		return fmt.Errorf("ColdSnap request: %w", err)
	}
	if request.Operation == "capture" && request.Output == "" {
		return errors.New("capture request requires output")
	}
	if slices.Contains([]string{"restore", "sleep", "wake", "status"}, request.Operation) && request.Artifact == "" {
		return fmt.Errorf("%s request requires artifact", request.Operation)
	}
	if (request.Operation == "publish" || request.Operation == "publish-native") &&
		(request.Artifact == "" || request.Output == "") {
		return fmt.Errorf("%s request requires artifact and output", request.Operation)
	}
	activationState := request.Lifecycle.ActivationState
	if activationState == "" {
		activationState = "running"
	}
	if activationState != "running" && activationState != "warm" {
		return errors.New("lifecycle activation_state must be running or warm")
	}
	if request.Operation != "restore" && activationState != "running" {
		return errors.New("warm activation_state is valid only for restore")
	}
	if activationState == "warm" {
		if request.Driver.ID != snapshotdriver.N610 {
			return errors.New("warm activation currently requires snapshot driver n610")
		}
		if request.Policy.Weights.Native.Materialize == "required" {
			return errors.New("warm activation cannot require native materialization before hydration")
		}
	}
	if slices.Contains([]string{"sleep", "wake", "status"}, request.Operation) && request.Workload.ClusterID == "" {
		return fmt.Errorf("%s request requires workload.cluster_id", request.Operation)
	}
	if err := request.Launch.Validate(); err != nil {
		return err
	}
	if err := request.Policy.Validate(); err != nil {
		return err
	}
	if request.Policy.Process.AsyncGraphs {
		for _, unit := range request.Launch.Units {
			for _, argument := range unit.Command {
				if strings.Contains(argument, "--enforce-eager") {
					return fmt.Errorf("launch unit %s enables --enforce-eager while asynchronous CUDA graphs are enabled", unit.ID)
				}
			}
		}
	}
	if request.Operation == "publish" {
		if request.Policy.Capsule.Repository == "" {
			return errors.New("publish request requires a capsule repository")
		}
	}
	if request.Operation == "publish-native" && request.Policy.Weights.Native.Repository == "" {
		return errors.New("publish-native request requires a model payload repository")
	}
	if request.Operation != "capture" && len(request.Policy.Cache.Staged) != 0 {
		return errors.New("staged derived caches are valid only for capture")
	}
	if len(request.Policy.Cache.Staged) != 0 {
		if len(request.Policy.Cache.Staged) != len(request.Launch.Units) {
			return errors.New("staged derived cache inventory must cover every launch unit")
		}
		covered := false
		for _, cachePath := range request.Policy.Cache.Paths {
			if cachePath == CanonicalRuntimeCachePath ||
				strings.HasPrefix(CanonicalRuntimeCachePath, cachePath+"/") {
				covered = true
				break
			}
		}
		if !covered {
			return errors.New("staged derived caches require the canonical runtime cache path")
		}
		unitIDs := make(map[string]bool, len(request.Launch.Units))
		for _, unit := range request.Launch.Units {
			unitIDs[unit.ID] = true
		}
		for _, staged := range request.Policy.Cache.Staged {
			if !unitIDs[staged.Unit] {
				return errors.New("staged derived cache unit is outside launch topology")
			}
		}
	}
	for _, unit := range request.Launch.Units {
		for _, mount := range unit.Mounts {
			for _, cachePath := range request.Policy.Cache.Paths {
				if pathsOverlap(mount.Target, cachePath) {
					return fmt.Errorf(
						"unit %s derived cache path overlaps launch mount %s",
						unit.ID,
						mount.Target,
					)
				}
			}
		}
	}
	workerIDs := make(map[string]bool, len(request.Launch.Execution.Workers))
	for _, worker := range request.Launch.Execution.Workers {
		workerIDs[worker.ID] = true
	}
	for worker := range request.Policy.Weights.Native.FilesByWorker {
		if !workerIDs[worker] {
			return errors.New("model payload filename worker is outside launch topology")
		}
	}
	if len(request.Policy.Weights.Native.FilesByWorker) != 0 &&
		len(request.Policy.Weights.Native.FilesByWorker) != len(request.Launch.Execution.Workers) {
		return errors.New("model payload filenames must cover every launch worker")
	}
	for _, pack := range request.Policy.Weights.Native.Staged {
		if !workerIDs[pack.Worker] {
			return errors.New("staged model payload worker is outside launch topology")
		}
	}
	if len(request.Policy.Weights.Native.Staged) != 0 &&
		len(request.Policy.Weights.Native.Staged) != len(request.Launch.Execution.Workers) {
		return errors.New("staged model payloads must cover every launch worker")
	}
	if request.Validation.HealthPath == "" || !strings.HasPrefix(request.Validation.HealthPath, "/") ||
		request.Validation.Prompt == "" || request.Validation.Expected == "" {
		return errors.New("validation policy requires an absolute health path, prompt, and expected text")
	}
	if err := request.Workload.Validate(); err != nil {
		return err
	}
	return nil
}

func (workload WorkloadIdentity) Validate() error {
	if workload.ClusterID == "" {
		if workload != (WorkloadIdentity{}) {
			return errors.New("workload identity requires cluster_id when any field is set")
		}
		return nil
	}
	if !idPattern.MatchString(workload.ClusterID) || workload.IntentID == "" || workload.Recipe == "" ||
		workload.Runtime == "" || workload.Model == "" {
		return errors.New("workload identity requires valid cluster_id, intent_id, recipe, runtime, and model")
	}
	for _, value := range []string{
		workload.IntentID, workload.Recipe, workload.Runtime, workload.Model,
		workload.ServedModelName, workload.LogPath,
	} {
		if strings.ContainsRune(value, '\x00') || strings.ContainsAny(value, "\r\n") {
			return errors.New("workload identity contains an invalid value")
		}
	}
	if workload.LogPath != "" &&
		(!filepath.IsAbs(workload.LogPath) || filepath.Clean(workload.LogPath) != workload.LogPath || workload.LogPath == "/") {
		return errors.New("workload log_path must be a clean absolute file path")
	}
	return nil
}

func (launch LaunchSpec) Validate() error {
	if launch.Engine != "vllm" && launch.Engine != "sglang" {
		return fmt.Errorf("unsupported inference engine %q", launch.Engine)
	}
	if launch.Model.ID == "" || launch.Model.Revision == "" || strings.ContainsRune(launch.Model.ID, '\x00') ||
		strings.ContainsRune(launch.Model.Revision, '\x00') {
		return errors.New("model ID and immutable revision are required")
	}
	if len(launch.Units) == 0 {
		return errors.New("launch contains no process-tree units")
	}
	devices := make(map[string]bool)
	unitIDs := make(map[string]bool, len(launch.Units))
	for index, unit := range launch.Units {
		if !idPattern.MatchString(unit.ID) || unit.Index != index || unitIDs[unit.ID] || unit.Host == "" || len(unit.Devices) == 0 || unit.Image == "" || len(unit.Command) == 0 {
			return fmt.Errorf("launch unit %q is incomplete or duplicated", unit.ID)
		}
		unitIDs[unit.ID] = true
		if strings.ContainsAny(unit.Image, " \t\r\n\x00") {
			return fmt.Errorf("launch unit %s image reference is invalid", unit.ID)
		}
		if !digestPattern.MatchString(unit.ImageDigest) {
			return fmt.Errorf("launch unit %s image digest is invalid", unit.ID)
		}
		// A registry reference pinned with @sha256 is portable. A bare Docker
		// image ID is equally immutable and is required for local-only capture
		// bases produced independently on each rank host. Distribution and
		// existence are orchestrator concerns; reject mutable tags here.
		if !imageBindsDigest(unit.Image, unit.ImageDigest) {
			return fmt.Errorf("launch unit %s image is not pinned to its digest", unit.ID)
		}
		seenDevices := make(map[string]bool)
		for _, deviceSelector := range unit.Devices {
			device := unit.Host + "\x00" + deviceSelector
			if deviceSelector == "" || strings.ContainsRune(deviceSelector, '\x00') || seenDevices[deviceSelector] || devices[device] {
				return fmt.Errorf("launch unit %s device assignment is invalid or duplicated", unit.ID)
			}
			seenDevices[deviceSelector] = true
			devices[device] = true
		}
		for _, argument := range unit.Command {
			if argument == "" || strings.ContainsRune(argument, '\x00') {
				return fmt.Errorf("launch unit %s command is invalid", unit.ID)
			}
		}
		for name, value := range unit.Environment {
			if !envPattern.MatchString(name) || strings.ContainsRune(value, '\x00') {
				return fmt.Errorf("launch unit %s environment is invalid", unit.ID)
			}
		}
		seenTargets := make(map[string]bool)
		for _, mount := range unit.Mounts {
			if !filepath.IsAbs(mount.Source) || filepath.Clean(mount.Source) != mount.Source ||
				strings.ContainsRune(mount.Source, '\x00') || !filepath.IsAbs(mount.Target) ||
				filepath.Clean(mount.Target) != mount.Target || seenTargets[mount.Target] {
				return fmt.Errorf("launch unit %s mount is invalid", unit.ID)
			}
			seenTargets[mount.Target] = true
		}
	}
	return launch.Execution.Validate(launch.Units)
}

func imageBindsDigest(reference, digest string) bool {
	if reference == digest {
		return true
	}
	separator := strings.LastIndexByte(reference, '@')
	return separator > 0 && reference[separator+1:] == digest
}

func (policy SnapshotPolicy) Validate() error {
	if policy.Process.Backend != "cuda-criu" {
		return fmt.Errorf("unsupported process snapshot backend %q", policy.Process.Backend)
	}
	artifactScope := policy.Process.ArtifactScope
	if artifactScope == "" {
		artifactScope = ArtifactScopePortable
	}
	if !slices.Contains(
		[]string{ArtifactScopePortable, ArtifactScopeTargetLocal},
		artifactScope,
	) {
		return fmt.Errorf("unsupported process artifact scope %q", policy.Process.ArtifactScope)
	}
	shapeCalibration := policy.Process.ShapeCalibration
	if shapeCalibration == "" {
		shapeCalibration = "auto"
	}
	if !slices.Contains([]string{"auto", "enabled", "disabled"}, shapeCalibration) {
		return fmt.Errorf("unsupported shape calibration policy %q", policy.Process.ShapeCalibration)
	}
	if shapeCalibration == "enabled" && !policy.Process.AsyncGraphs {
		return errors.New("shape calibration requires asynchronous CUDA graphs")
	}
	graphPolicy := policy.Process.GraphPolicy
	if graphPolicy == "" && policy.Process.AsyncGraphs {
		graphPolicy = GraphRecreateFromPlan
	}
	if policy.Process.AsyncGraphs {
		if !slices.Contains([]string{GraphPreserveExec, GraphRecreateFromPlan, GraphPreserveNCCLExec}, graphPolicy) {
			return fmt.Errorf("unsupported CUDA graph preservation policy %q", policy.Process.GraphPolicy)
		}
	} else if graphPolicy != "" {
		return errors.New("CUDA graph preservation policy requires asynchronous CUDA graphs")
	}
	if !slices.Contains([]string{"auto", "native", "recovery", "cache-only-auto"}, policy.Weights.Mode) {
		return fmt.Errorf("unsupported weight mode %q", policy.Weights.Mode)
	}
	if policy.Weights.Native.Repository != "" && policy.Weights.Native.Revision == "" {
		return errors.New("model payload repository must be pinned to a revision")
	}
	if !slices.Contains([]string{"", "off", "async", "required"}, policy.Weights.Native.Materialize) {
		return fmt.Errorf("unsupported native model payload materialization mode %q", policy.Weights.Native.Materialize)
	}
	if len(policy.Weights.Native.FilesByWorker) > 0 && policy.Weights.Native.Repository == "" {
		return errors.New("model payload filenames require a repository")
	}
	seenWorkers := make(map[string]bool)
	for worker, name := range policy.Weights.Native.FilesByWorker {
		if !idPattern.MatchString(worker) || name == "" || filepath.IsAbs(name) ||
			filepath.Clean(name) != name || strings.HasPrefix(name, "../") {
			return errors.New("model payload filename inventory is invalid")
		}
	}
	for _, pack := range policy.Weights.Native.Staged {
		if !idPattern.MatchString(pack.Worker) || seenWorkers[pack.Worker] ||
			!filepath.IsAbs(pack.Path) || pack.Bytes <= 0 || !digestPattern.MatchString(pack.SHA256) {
			return errors.New("staged model payload inventory is invalid")
		}
		if pack.Validation != nil &&
			(pack.Validation.Record != pack.Path+".coldsnap-validation.json" ||
				pack.Validation.Provider != "sha256-cache-v1" ||
				pack.Validation.ContentEvidence == "" ||
				pack.Validation.Device == 0 || pack.Validation.Inode == 0 ||
				pack.Validation.Size != pack.Bytes || pack.Validation.MTimeNS <= 0 ||
				pack.Validation.BytesHashed < 0 || pack.Validation.Seconds < 0) {
			return errors.New("staged model payload validation evidence is invalid")
		}
		seenWorkers[pack.Worker] = true
	}
	if policy.Weights.Recovery.Enabled && policy.Weights.Recovery.Source != "huggingface-safetensors" {
		return fmt.Errorf("unsupported recovery source %q", policy.Weights.Recovery.Source)
	}
	if !slices.Contains(
		[]string{"direct", "buffered", "auto", "mmap", "torch"},
		policy.Weights.Recovery.LoaderBackend,
	) {
		return fmt.Errorf("unsupported recovery loader backend %q", policy.Weights.Recovery.LoaderBackend)
	}
	if (policy.Weights.Mode == "recovery" || policy.Weights.Mode == "auto" || policy.Weights.Mode == "cache-only-auto") &&
		!policy.Weights.Recovery.Enabled {
		return fmt.Errorf("weight mode %q requires recovery fallback", policy.Weights.Mode)
	}
	if !policy.Cache.Seed && len(policy.Cache.Paths) != 0 {
		return errors.New("derived cache paths require cache seeding")
	}
	if !policy.Cache.Seed && len(policy.Cache.Staged) != 0 {
		return errors.New("staged derived caches require cache seeding")
	}
	seenCachePaths := make(map[string]bool)
	for _, cachePath := range policy.Cache.Paths {
		if !safeCachePath(cachePath) || seenCachePaths[cachePath] {
			return errors.New("derived cache path inventory is invalid")
		}
		for existing := range seenCachePaths {
			if strings.HasPrefix(cachePath, existing+"/") || strings.HasPrefix(existing, cachePath+"/") {
				return errors.New("derived cache paths must not overlap")
			}
		}
		seenCachePaths[cachePath] = true
	}
	seenCacheUnits := make(map[string]bool)
	for _, staged := range policy.Cache.Staged {
		if !idPattern.MatchString(staged.Unit) || seenCacheUnits[staged.Unit] || !filepath.IsAbs(staged.Path) ||
			filepath.Clean(staged.Path) != staged.Path || strings.ContainsRune(staged.Path, '\x00') {
			return errors.New("staged derived cache inventory is invalid")
		}
		seenCacheUnits[staged.Unit] = true
	}
	if policy.Capsule.Repository != "" && !validCapsuleRepository(policy.Capsule.Repository) {
		return errors.New("capsule repository is invalid")
	}
	if policy.Compatibility.Kernel != KernelCompatibilityCapability &&
		policy.Compatibility.Kernel != KernelCompatibilityExact {
		return fmt.Errorf("unsupported kernel compatibility policy %q", policy.Compatibility.Kernel)
	}
	return nil
}

func validCapsuleRepository(value string) bool {
	if value == "" || value != strings.ToLower(value) || strings.ContainsAny(value, "@ \t\r\n\x00") ||
		strings.HasPrefix(value, "/") || strings.HasSuffix(value, "/") || strings.Contains(value, "//") {
		return false
	}
	lastSlash := strings.LastIndexByte(value, '/')
	if strings.LastIndexByte(value, ':') > lastSlash {
		return false
	}
	for _, character := range value {
		if character < 'a' || character > 'z' {
			if character < '0' || character > '9' {
				if !strings.ContainsRune("._:/-", character) {
					return false
				}
			}
		}
	}
	return true
}

func pathsOverlap(first, second string) bool {
	return first == second || strings.HasPrefix(first, second+"/") || strings.HasPrefix(second, first+"/")
}

func safeCachePath(value string) bool {
	if !filepath.IsAbs(value) || filepath.Clean(value) != value || strings.ContainsRune(value, '\x00') {
		return false
	}
	for _, root := range []string{
		"/root/.cache/flashinfer", "/root/.cache/torch", "/root/.cache/torch_extensions",
		"/root/.cache/vllm", "/root/.triton/cache", "/tmp/coldsnap-derived-cache",
		"/tmp/torchinductor_root", "/var/cache/coldsnap",
	} {
		if value == root || strings.HasPrefix(value, root+"/") {
			return true
		}
	}
	return false
}

type Artifact struct {
	Format           int                         `json:"format"`
	Kind             string                      `json:"kind"`
	State            string                      `json:"state"`
	CaptureID        string                      `json:"capture_id"`
	RequestSHA256    string                      `json:"request_sha256"`
	Driver           snapshotdriver.Contract     `json:"snapshot_driver"`
	Requires         snapshotdriver.Requirements `json:"requires"`
	Launch           LaunchSpec                  `json:"launch"`
	Compatibility    ArtifactCompatibility       `json:"compatibility"`
	Capsule          Capsule                     `json:"capsule"`
	Runtime          RuntimeProviders            `json:"runtime"`
	ShapeCalibration *ShapeCalibrationCoverage   `json:"shape_calibration,omitempty"`
	Graph            GraphPolicyRecord           `json:"graph"`
	Weights          WeightProviders             `json:"weights"`
	Acceptance       ArtifactAcceptance          `json:"acceptance"`
}

type ShapeCalibrationCoverage struct {
	Policy string                 `json:"policy"`
	Units  []UnitShapeCalibration `json:"units"`
}

type UnitShapeCalibration struct {
	Unit          string          `json:"unit"`
	Engine        string          `json:"engine"`
	PlannedShapes int             `json:"planned_shapes"`
	WarmedShapes  int             `json:"warmed_shapes"`
	Seconds       float64         `json:"seconds"`
	Evidence      json.RawMessage `json:"evidence"`
}

const (
	GraphPreserveExec     = "preserve-exec"
	GraphRecreateFromPlan = "recreate-from-plan"
	GraphPreserveNCCLExec = "preserve-nccl-exec"
	GraphDisabled         = "disabled"

	graphDecisionSupported            = "requested_policy_is_supported"
	graphDecisionReconstructsNCCL     = "provider_reconstructs_communicator"
	graphDecisionPreserveUnqualified  = "graph_exec_preservation_is_unqualified"
	graphDecisionNCCLInPlaceQualified = "qualified_exact_nccl_in_place"
)

type GraphPolicyRecord struct {
	Format    int                `json:"format"`
	Requested string             `json:"policy_requested"`
	Effective string             `json:"policy_effective"`
	Decision  string             `json:"decision"`
	Audit     GraphResourceAudit `json:"resource_audit"`
	Recipe    GraphRecipe        `json:"recipe"`
}

type GraphResourceAudit struct {
	Format                   int  `json:"format"`
	UnknownDependencies      bool `json:"unknown_dependencies"`
	NCCLCommunicators        int  `json:"nccl_communicators"`
	RegisteredWindows        int  `json:"registered_windows"`
	DeviceAPI                bool `json:"device_api"`
	ExternalSemaphores       int  `json:"external_semaphores"`
	ProviderOwnedAllocations int  `json:"provider_owned_allocations"`
}

type GraphRecipe struct {
	Format                 int    `json:"format"`
	Kind                   string `json:"kind"`
	Engine                 string `json:"engine"`
	Mode                   string `json:"mode"`
	ShapeCalibrationPolicy string `json:"shape_calibration_policy"`
	CalibrationUnits       int    `json:"calibration_units"`
	PlanSHA256             string `json:"plan_sha256"`
}

func NewGraphPolicyRecord(
	request Request, calibration *ShapeCalibrationCoverage,
) (GraphPolicyRecord, error) {
	requested := request.Policy.Process.GraphPolicy
	if !request.Policy.Process.AsyncGraphs {
		requested = GraphDisabled
	} else if requested == "" {
		requested = GraphRecreateFromPlan
	}
	effective := requested
	decision := graphDecisionSupported
	audit := GraphResourceAudit{Format: 1}
	if requested != GraphDisabled {
		// Engine status reports identify known resources, but the artifact
		// builder cannot yet prove that every graph-reachable allocation and
		// registration has been enumerated. Preserve that uncertainty even
		// when the effective policy is already the safe recreate path.
		audit.UnknownDependencies = true
	}
	if requested != GraphDisabled && len(request.Launch.Execution.Workers) > 1 {
		audit.NCCLCommunicators = 1
	}
	if requested == GraphPreserveExec {
		effective = GraphRecreateFromPlan
		decision = graphDecisionPreserveUnqualified
		if audit.NCCLCommunicators > 0 {
			decision = graphDecisionReconstructsNCCL
		}
	}
	if requested == GraphPreserveNCCLExec {
		if audit.NCCLCommunicators == 0 {
			// The retained-NCCL policy has nothing to preserve for a TP1 process.
			// Keep local graph handling on the qualified recreation path.
			effective = GraphRecreateFromPlan
			decision = graphDecisionPreserveUnqualified
		} else {
			effective = GraphPreserveNCCLExec
			decision = graphDecisionNCCLInPlaceQualified
			audit.UnknownDependencies = false
		}
	}
	calibrationUnits := 0
	if calibration != nil {
		calibrationUnits = len(calibration.Units)
	}
	shapePolicy := request.Policy.Process.ShapeCalibration
	if shapePolicy == "" {
		shapePolicy = "auto"
	}
	record := GraphPolicyRecord{
		Format: 1, Requested: requested, Effective: effective, Decision: decision, Audit: audit,
		Recipe: GraphRecipe{
			Format: 1, Kind: "coldsnap-engine-graph-recipe", Engine: request.Launch.Engine,
			Mode: "engine-native-async", ShapeCalibrationPolicy: shapePolicy,
			CalibrationUnits: calibrationUnits,
		},
	}
	if requested == GraphDisabled {
		record.Recipe.Mode = GraphDisabled
	} else if effective == GraphPreserveNCCLExec {
		record.Recipe.Mode = "engine-native-retained"
	}
	digest, err := graphPlanSHA256(record, request.Launch, calibration)
	if err != nil {
		return GraphPolicyRecord{}, err
	}
	record.Recipe.PlanSHA256 = digest
	if err := record.Validate(request.Launch, calibration); err != nil {
		return GraphPolicyRecord{}, err
	}
	return record, nil
}

func graphPlanSHA256(
	record GraphPolicyRecord, launch LaunchSpec, calibration *ShapeCalibrationCoverage,
) (string, error) {
	plan := struct {
		Engine           string                    `json:"engine"`
		Launch           ExecutionGraph            `json:"execution"`
		Requested        string                    `json:"requested"`
		Effective        string                    `json:"effective"`
		Audit            GraphResourceAudit        `json:"audit"`
		ShapeCalibration *ShapeCalibrationCoverage `json:"shape_calibration,omitempty"`
	}{
		Engine: record.Recipe.Engine, Launch: launch.Execution,
		Requested: record.Requested, Effective: record.Effective, Audit: record.Audit,
		ShapeCalibration: calibration,
	}
	payload, err := json.Marshal(plan)
	if err != nil {
		return "", err
	}
	digest, err := canonicaljson.CanonicalSHA256(payload)
	if err != nil {
		return "", err
	}
	return "sha256:" + digest, nil
}

func (record GraphPolicyRecord) Validate(
	launch LaunchSpec, calibration *ShapeCalibrationCoverage,
) error {
	policies := []string{GraphPreserveExec, GraphRecreateFromPlan, GraphPreserveNCCLExec, GraphDisabled}
	decisions := []string{
		graphDecisionSupported, graphDecisionReconstructsNCCL,
		graphDecisionPreserveUnqualified,
		graphDecisionNCCLInPlaceQualified,
	}
	if record.Format != 1 || !slices.Contains(policies, record.Requested) ||
		!slices.Contains(policies, record.Effective) || !slices.Contains(decisions, record.Decision) ||
		record.Audit.Format != 1 || record.Audit.NCCLCommunicators < 0 ||
		record.Audit.RegisteredWindows < 0 || record.Audit.ExternalSemaphores < 0 ||
		record.Audit.ProviderOwnedAllocations < 0 {
		return errors.New("CUDA graph policy record is invalid")
	}
	if record.Requested != GraphDisabled && record.Effective != GraphRecreateFromPlan &&
		record.Effective != GraphPreserveNCCLExec {
		return errors.New("graph executable preservation has no qualified engine/provider path")
	}
	if record.Audit.UnknownDependencies && record.Effective != GraphRecreateFromPlan &&
		record.Effective != GraphDisabled {
		return errors.New("unknown CUDA graph dependencies require recreation")
	}
	expectedDecision := graphDecisionSupported
	switch record.Requested {
	case GraphDisabled:
		if record.Effective != GraphDisabled || record.Audit.UnknownDependencies ||
			record.Audit.NCCLCommunicators != 0 || record.Audit.RegisteredWindows != 0 ||
			record.Audit.DeviceAPI || record.Audit.ExternalSemaphores != 0 ||
			record.Audit.ProviderOwnedAllocations != 0 {
			return errors.New("disabled CUDA graphs have an inconsistent resource audit")
		}
	case GraphRecreateFromPlan:
		if record.Effective != GraphRecreateFromPlan {
			return errors.New("recreate-from-plan has an inconsistent effective policy")
		}
	case GraphPreserveExec:
		expectedDecision = graphDecisionPreserveUnqualified
		if record.Audit.NCCLCommunicators > 0 {
			expectedDecision = graphDecisionReconstructsNCCL
		}
	case GraphPreserveNCCLExec:
		if record.Audit.NCCLCommunicators == 0 {
			expectedDecision = graphDecisionPreserveUnqualified
			if record.Effective != GraphRecreateFromPlan || !record.Audit.UnknownDependencies {
				return errors.New("NCCL-free graph preservation has an inconsistent audit")
			}
		} else {
			expectedDecision = graphDecisionNCCLInPlaceQualified
			if record.Effective != GraphPreserveNCCLExec || record.Audit.UnknownDependencies ||
				record.Audit.NCCLCommunicators != 1 {
				return errors.New("NCCL graph preservation has an inconsistent audit")
			}
		}
	}
	if record.Decision != expectedDecision {
		return errors.New("CUDA graph policy decision differs from its resource audit")
	}
	if record.Recipe.Format != 1 || record.Recipe.Kind != "coldsnap-engine-graph-recipe" ||
		record.Recipe.Engine != launch.Engine || record.Recipe.Mode == "" ||
		!slices.Contains([]string{"auto", "enabled", "disabled"}, record.Recipe.ShapeCalibrationPolicy) ||
		record.Recipe.CalibrationUnits < 0 ||
		!digestPattern.MatchString(record.Recipe.PlanSHA256) {
		return errors.New("CUDA graph recipe is invalid")
	}
	expectedMode := "engine-native-async"
	if record.Effective == GraphDisabled {
		expectedMode = GraphDisabled
	} else if record.Effective == GraphPreserveNCCLExec {
		expectedMode = "engine-native-retained"
	}
	if record.Recipe.Mode != expectedMode {
		return errors.New("CUDA graph recipe mode differs from graph policy")
	}
	expectedUnits := 0
	if calibration != nil {
		expectedUnits = len(calibration.Units)
	}
	if record.Recipe.CalibrationUnits != expectedUnits {
		return errors.New("CUDA graph recipe calibration inventory differs from artifact evidence")
	}
	digest, err := graphPlanSHA256(record, launch, calibration)
	if err != nil {
		return err
	}
	if record.Recipe.PlanSHA256 != digest {
		return errors.New("CUDA graph recipe digest does not match its artifact inputs")
	}
	return nil
}

// ArtifactCompatibility records only the host properties that are outside a
// digest-pinned capsule and therefore must be revalidated before eviction.
// Hostnames, IP addresses, GPU UUIDs, and device ordinals are deliberately
// absent: they are placement, not artifact identity.
type ArtifactCompatibility struct {
	Policy string                      `json:"policy"`
	Units  []UnitPlatformCompatibility `json:"units"`
}

type UnitPlatformCompatibility struct {
	Unit                 string                `json:"unit"`
	Architecture         string                `json:"architecture"`
	Kernel               string                `json:"kernel"`
	NVIDIADriverCaptured string                `json:"nvidia_driver_captured"`
	NVIDIADriverMin      string                `json:"nvidia_driver_min"`
	CUDAUserspace        string                `json:"cuda_userspace"`
	Devices              []DeviceCompatibility `json:"devices"`
}

type DeviceCompatibility struct {
	Slot              int    `json:"slot"`
	GPUName           string `json:"gpu_name"`
	ComputeCapability string `json:"compute_capability"`
}

type Capsule struct {
	Images  []CapsuleImage `json:"images"`
	Objects []Object       `json:"objects"`
}

type CapsuleImage struct {
	Unit      string                 `json:"unit"`
	Reference string                 `json:"reference"`
	Digest    string                 `json:"digest"`
	Root      string                 `json:"root"`
	Driver    snapshotdriver.Binding `json:"snapshot_driver"`
}

type Object struct {
	Role   string `json:"role"`
	Owner  string `json:"owner"`
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type WeightProviders struct {
	ModelPayloads *ModelPayloadProvider `json:"model_payloads,omitempty"`
	Native        *NativeProvider       `json:"native,omitempty"`
	Recovery      RecoveryProvider      `json:"recovery"`
}

type NativeProvider struct {
	Driver snapshotdriver.Binding `json:"snapshot_driver"`
}

type ModelPayloadProvider struct {
	Repository string   `json:"repository,omitempty"`
	Revision   string   `json:"revision,omitempty"`
	Objects    []Object `json:"objects"`
}

type RecoveryProvider struct {
	Source     string                 `json:"source"`
	ModelID    string                 `json:"model_id"`
	Revision   string                 `json:"revision"`
	Driver     snapshotdriver.Binding `json:"snapshot_driver"`
	LoadPath   string                 `json:"load_path"`
	ReplayPlan []Object               `json:"replay_plan"`
}

type ArtifactAcceptance struct {
	Accepted bool   `json:"accepted"`
	Expected string `json:"expected"`
}

func (artifact Artifact) Validate() error {
	if artifact.Format != ArtifactFormat || artifact.Kind != ArtifactKind || artifact.State != "committed" {
		return fmt.Errorf("artifact is not a committed ColdSnap v%d snapshot", ArtifactFormat)
	}
	if !idPattern.MatchString(artifact.CaptureID) || !digestPattern.MatchString(artifact.RequestSHA256) {
		return errors.New("artifact identity is invalid")
	}
	if err := artifact.Driver.Validate(); err != nil {
		return fmt.Errorf("artifact snapshot driver: %w", err)
	}
	if err := artifact.Requires.Validate(); err != nil {
		return fmt.Errorf("artifact feature requirements: %w", err)
	}
	if !artifact.Requires.ContainsAll(artifact.Driver.BaseRequirements) {
		return errors.New("artifact feature requirements omit snapshot driver base requirements")
	}
	if err := artifact.Launch.Validate(); err != nil {
		return fmt.Errorf("artifact launch: %w", err)
	}
	if err := artifact.Compatibility.Validate(artifact.Launch); err != nil {
		return fmt.Errorf("artifact compatibility: %w", err)
	}
	if len(artifact.Capsule.Images) != len(artifact.Launch.Units) {
		return errors.New("artifact capsule image count differs from launch-unit count")
	}
	if len(artifact.Capsule.Objects) == 0 {
		return errors.New("artifact capsule contains no process or residual objects")
	}
	seenUnits := make(map[string]bool, len(artifact.Capsule.Images))
	for _, image := range artifact.Capsule.Images {
		if !idPattern.MatchString(image.Unit) || seenUnits[image.Unit] || !digestPattern.MatchString(image.Digest) ||
			!imageBindsDigest(image.Reference, image.Digest) || !filepath.IsAbs(image.Root) ||
			filepath.Clean(image.Root) != image.Root || strings.ContainsRune(image.Root, '\x00') ||
			!image.Driver.Matches(artifact.Driver) {
			return fmt.Errorf("artifact capsule image for unit %s is invalid", image.Unit)
		}
		seenUnits[image.Unit] = true
	}
	for _, unit := range artifact.Launch.Units {
		if !seenUnits[unit.ID] {
			return fmt.Errorf("artifact capsule has no image for unit %s", unit.ID)
		}
	}
	if err := validateObjects(artifact.Capsule.Objects, artifact.Launch, nil); err != nil {
		return fmt.Errorf("artifact capsule: %w", err)
	}
	if err := artifact.Runtime.Validate(artifact.Launch); err != nil {
		return fmt.Errorf("artifact runtime: %w", err)
	}
	if artifact.ShapeCalibration != nil {
		if err := artifact.ShapeCalibration.Validate(artifact.Launch); err != nil {
			return fmt.Errorf("artifact shape calibration: %w", err)
		}
	}
	if err := artifact.Graph.Validate(artifact.Launch, artifact.ShapeCalibration); err != nil {
		return fmt.Errorf("artifact CUDA graph policy: %w", err)
	}
	if artifact.Weights.Native != nil {
		native := artifact.Weights.Native
		if !native.Driver.Matches(artifact.Driver) {
			return errors.New("native provider snapshot driver differs from artifact")
		}
		if artifact.Weights.ModelPayloads == nil {
			return errors.New("native provider has no shared model payload provider")
		}
	}
	if provider := artifact.Weights.ModelPayloads; provider != nil {
		if provider.Repository != "" && provider.Revision == "" {
			return errors.New("model payload provider repository is not pinned")
		}
		if err := validateSharedWorkerObjects(provider.Objects, artifact.Launch, "model payload provider"); err != nil {
			return fmt.Errorf("model payload provider: %w", err)
		}
		for _, object := range provider.Objects {
			expectedPath := filepath.ToSlash(filepath.Join(
				"model-payloads", "sha256", strings.TrimPrefix(object.SHA256, "sha256:")+".pack",
			))
			if object.Role != "model-weight-payload" || object.Path != expectedPath {
				return errors.New("model payload provider object is not content-addressed")
			}
		}
	}
	recovery := artifact.Weights.Recovery
	if recovery.Source != "huggingface-safetensors" || recovery.ModelID != artifact.Launch.Model.ID ||
		recovery.Revision != artifact.Launch.Model.Revision || !recovery.Driver.Matches(artifact.Driver) {
		return errors.New("recovery provider does not match the captured model identity")
	}
	switch artifact.Driver.ID {
	case snapshotdriver.N610:
		expectedLoadPath := "coldsnap-replay"
		if artifact.Launch.Engine == "sglang" {
			expectedLoadPath = "sglang-inplace-disk"
		}
		if recovery.LoadPath != expectedLoadPath {
			return fmt.Errorf("n610 %s recovery provider must use %s", artifact.Launch.Engine, expectedLoadPath)
		}
		if err := validateWorkerObjects(recovery.ReplayPlan, artifact.Launch, "recovery provider"); err != nil {
			return fmt.Errorf("recovery provider: %w", err)
		}
	case snapshotdriver.N580:
		expectedLoadPath := "coldsnap-replay"
		if artifact.Launch.Engine == "sglang" {
			expectedLoadPath = "sglang-startup-disk"
		}
		if recovery.LoadPath != expectedLoadPath {
			return fmt.Errorf("n580 %s recovery provider must use %s", artifact.Launch.Engine, expectedLoadPath)
		}
		if err := validateWorkerObjects(recovery.ReplayPlan, artifact.Launch, "recovery provider"); err != nil {
			return fmt.Errorf("recovery provider: %w", err)
		}
	}
	if !artifact.Acceptance.Accepted || artifact.Acceptance.Expected == "" {
		return errors.New("artifact acceptance proof is missing")
	}
	return nil
}

func (coverage ShapeCalibrationCoverage) Validate(launch LaunchSpec) error {
	if coverage.Policy != "engine-owned-v1" || len(coverage.Units) != len(launch.Units) {
		return errors.New("engine-owned shape-calibration coverage is incomplete")
	}
	units := make(map[string]bool, len(launch.Units))
	for _, unit := range launch.Units {
		units[unit.ID] = true
	}
	seen := make(map[string]bool, len(coverage.Units))
	for _, unit := range coverage.Units {
		if !units[unit.Unit] || seen[unit.Unit] || unit.Engine != launch.Engine ||
			unit.PlannedShapes <= 0 || unit.WarmedShapes != unit.PlannedShapes ||
			unit.Seconds < 0 || math.IsNaN(unit.Seconds) || math.IsInf(unit.Seconds, 0) ||
			len(unit.Evidence) == 0 {
			return fmt.Errorf("unit %s shape-calibration coverage is invalid", unit.Unit)
		}
		var evidence struct {
			Format        int               `json:"format"`
			Kind          string            `json:"kind"`
			Engine        string            `json:"engine"`
			PlannedShapes int               `json:"planned_shapes"`
			WarmedShapes  int               `json:"warmed_shapes"`
			Seconds       float64           `json:"seconds"`
			Details       []json.RawMessage `json:"details"`
		}
		if err := json.Unmarshal(unit.Evidence, &evidence); err != nil ||
			evidence.Format != 1 || evidence.Kind != "coldsnap-shape-calibration-coverage" ||
			evidence.Engine != unit.Engine || evidence.PlannedShapes != unit.PlannedShapes ||
			evidence.WarmedShapes != unit.WarmedShapes || evidence.Seconds != unit.Seconds ||
			len(evidence.Details) == 0 {
			return fmt.Errorf("unit %s shape-calibration evidence is invalid", unit.Unit)
		}
		for _, raw := range evidence.Details {
			var detail struct {
				Kind          string           `json:"kind"`
				Engine        string           `json:"engine"`
				PlannedShapes int              `json:"planned_shapes"`
				WarmedShapes  int              `json:"warmed_shapes"`
				Modes         []map[string]any `json:"modes"`
				Toolchain     map[string]any   `json:"toolchain"`
				CacheRoot     string           `json:"cache_root"`
			}
			if err := json.Unmarshal(raw, &detail); err != nil ||
				detail.Kind != "coldsnap-shape-calibration" || detail.Engine != unit.Engine ||
				detail.PlannedShapes <= 0 || detail.WarmedShapes != detail.PlannedShapes ||
				len(detail.Modes) == 0 || len(detail.Toolchain) == 0 || !filepath.IsAbs(detail.CacheRoot) {
				return fmt.Errorf("unit %s engine shape-calibration detail is invalid", unit.Unit)
			}
		}
		seen[unit.Unit] = true
	}
	return nil
}

func (compatibility ArtifactCompatibility) Validate(launch LaunchSpec) error {
	if compatibility.Policy != PortabilityPolicy || len(compatibility.Units) != len(launch.Units) {
		return errors.New("portable platform inventory is missing or incomplete")
	}
	unitByID := make(map[string]LaunchUnit, len(launch.Units))
	for _, unit := range launch.Units {
		unitByID[unit.ID] = unit
	}
	seen := make(map[string]bool, len(compatibility.Units))
	for _, platform := range compatibility.Units {
		unit, ok := unitByID[platform.Unit]
		if !ok || seen[platform.Unit] {
			return fmt.Errorf("unit %s portable platform inventory is invalid or duplicated", platform.Unit)
		}
		seen[platform.Unit] = true
		if err := platform.Validate(len(unit.Devices)); err != nil {
			return fmt.Errorf("unit %s portable platform identity is invalid", platform.Unit)
		}
	}
	return nil
}

func (platform UnitPlatformCompatibility) Validate(deviceCount int) error {
	if !idPattern.MatchString(platform.Unit) || platform.Architecture == "" || platform.Kernel == "" ||
		platform.CUDAUserspace == "" || len(platform.Devices) != deviceCount ||
		!numericVersionPattern.MatchString(platform.NVIDIADriverCaptured) ||
		!numericVersionPattern.MatchString(platform.NVIDIADriverMin) ||
		!numericVersionPattern.MatchString(platform.CUDAUserspace) {
		return errors.New("portable platform identity fields are invalid")
	}
	for slot, device := range platform.Devices {
		if device.Slot != slot || device.GPUName == "" || !numericVersionPattern.MatchString(device.ComputeCapability) {
			return errors.New("portable device identity fields are invalid")
		}
	}
	return nil
}

func validateWorkerObjects(objects []Object, launch LaunchSpec, label string) error {
	return validateWorkerObjectsWithPathPolicy(objects, launch, label, false)
}

func validateSharedWorkerObjects(objects []Object, launch LaunchSpec, label string) error {
	return validateWorkerObjectsWithPathPolicy(objects, launch, label, true)
}

func validateWorkerObjectsWithPathPolicy(
	objects []Object, launch LaunchSpec, label string, allowSharedPaths bool,
) error {
	if len(objects) != len(launch.Execution.Workers) {
		return fmt.Errorf("%s must contain exactly one object per worker", label)
	}
	required := make(map[string]bool, len(launch.Execution.Workers))
	for _, worker := range launch.Execution.Workers {
		required[WorkerOwner(worker.ID)] = true
	}
	if err := validateObjectsWithPathPolicy(objects, launch, required, allowSharedPaths); err != nil {
		return err
	}
	return nil
}

func validateObjects(objects []Object, launch LaunchSpec, required map[string]bool) error {
	return validateObjectsWithPathPolicy(objects, launch, required, false)
}

func validateObjectsWithPathPolicy(
	objects []Object, launch LaunchSpec, required map[string]bool, allowSharedPaths bool,
) error {
	owners := launch.validOwners()
	seenOwners := make(map[string]bool)
	seenPaths := make(map[string]Object)
	for _, object := range objects {
		prior, shared := seenPaths[object.Path]
		if object.Role == "" || !owners[object.Owner] || object.Path == "" ||
			filepath.IsAbs(object.Path) || filepath.Clean(object.Path) != object.Path ||
			object.Path == "." || strings.HasPrefix(object.Path, ".."+string(filepath.Separator)) ||
			strings.ContainsRune(object.Path, '\x00') || object.Bytes <= 0 ||
			!digestPattern.MatchString(object.SHA256) || (!allowSharedPaths && shared) ||
			(shared && (prior.Role != object.Role || prior.Bytes != object.Bytes || prior.SHA256 != object.SHA256)) {
			return errors.New("object record is invalid")
		}
		if required != nil && required[object.Owner] && seenOwners[object.Owner] {
			return errors.New("provider contains duplicate worker objects")
		}
		seenOwners[object.Owner] = true
		seenPaths[object.Path] = object
	}
	for owner := range required {
		if !seenOwners[owner] {
			return fmt.Errorf("provider does not contain required owner %s", owner)
		}
	}
	return nil
}

func DecodeRequest(reader io.Reader) (Request, error) {
	payload, err := io.ReadAll(reader)
	if err != nil {
		return Request{}, fmt.Errorf("read ColdSnap request: %w", err)
	}
	policy := newDefaultSnapshotPolicy()
	var header struct {
		Driver snapshotdriver.Selection `json:"snapshot_driver"`
	}
	if err := json.Unmarshal(payload, &header); err == nil && header.Driver.ID != "" {
		if selected, selectErr := DefaultSnapshotPolicyForDriver(header.Driver.ID); selectErr == nil {
			policy = selected
		}
	}
	request := Request{
		Format:     RequestFormat,
		Kind:       RequestKind,
		Policy:     policy,
		Validation: DefaultValidationPolicy(),
	}
	if err := decodeStrict(bytes.NewReader(payload), &request); err != nil {
		return Request{}, fmt.Errorf("decode ColdSnap request: %w", err)
	}
	// cache.seed=false is a complete opt-out when paths are omitted. A plain
	// decode over a default-valued struct cannot distinguish omitted paths from
	// the default inventory, so retain this small presence check explicitly.
	var presence struct {
		Policy *struct {
			Process *struct {
				AsyncGraphs *bool   `json:"async_graphs"`
				GraphPolicy *string `json:"graph_policy"`
			} `json:"process"`
			Cache *struct {
				Seed  *bool     `json:"seed"`
				Paths *[]string `json:"paths"`
			} `json:"cache"`
		} `json:"policy"`
	}
	if err := json.Unmarshal(payload, &presence); err != nil {
		return Request{}, fmt.Errorf("decode ColdSnap request presence: %w", err)
	}
	if presence.Policy != nil && presence.Policy.Cache != nil &&
		presence.Policy.Cache.Seed != nil && !*presence.Policy.Cache.Seed &&
		presence.Policy.Cache.Paths == nil {
		request.Policy.Cache.Paths = nil
	}
	if presence.Policy != nil && presence.Policy.Process != nil &&
		presence.Policy.Process.AsyncGraphs != nil && !*presence.Policy.Process.AsyncGraphs &&
		presence.Policy.Process.GraphPolicy == nil {
		request.Policy.Process.GraphPolicy = ""
	}
	if err := request.Validate(); err != nil {
		return Request{}, fmt.Errorf("validate ColdSnap request: %w", err)
	}
	return request, nil
}

func ReadArtifact(path string) (Artifact, error) {
	file, err := os.Open(path)
	if err != nil {
		return Artifact{}, fmt.Errorf("open ColdSnap artifact: %w", err)
	}
	defer file.Close()
	var artifact Artifact
	if err := decodeStrict(file, &artifact); err != nil {
		return Artifact{}, fmt.Errorf("decode ColdSnap artifact: %w", err)
	}
	if err := artifact.Validate(); err != nil {
		return Artifact{}, fmt.Errorf("validate ColdSnap artifact: %w", err)
	}
	return artifact, nil
}

// ValidateArtifact exposes the strict committed-artifact gate to engine
// adapters which construct a manifest before publishing it.
func ValidateArtifact(artifact Artifact) error {
	return artifact.Validate()
}

func RequestSHA256(request Request) (string, error) {
	data, err := json.Marshal(request)
	if err != nil {
		return "", err
	}
	digest, err := canonicaljson.CanonicalSHA256(data)
	if err != nil {
		return "", err
	}
	return "sha256:" + digest, nil
}

func decodeStrict(reader io.Reader, destination any) error {
	decoder := json.NewDecoder(reader)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return errors.New("trailing JSON")
	}
	return nil
}

func Encode(value any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
	activationruntime "github.com/sparksq/coldsnap/runtime/engine"
)

type activationRuntimeRemote struct {
	mutex       sync.Mutex
	files       map[string][]byte
	inputWrites int
	shaCalls    int
	criuCalls   [][]string
	criuError   error
}

type featureProbeRemote struct {
	profile       []byte
	combinedCalls int
}

func (remote *featureProbeRemote) Run(
	_ context.Context, _ string, arguments ...string,
) ([]byte, error) {
	switch {
	case len(arguments) == 2 && arguments[0] == "cat" && arguments[1] == "/proc/sys/kernel/random/boot_id":
		return []byte("boot-1\n"), nil
	case len(arguments) > 0 && arguments[0] == "nvidia-smi":
		return []byte("610.43.02, GPU-test, NVIDIA GB10, 12.1\n"), nil
	case len(arguments) > 0 && arguments[0] == "cat":
		return nil, errors.New("not found")
	case len(arguments) > 0 && arguments[0] == "docker":
		return slices.Clone(remote.profile), nil
	case len(arguments) > 0 && slices.Contains([]string{"install", "chmod", "mv"}, arguments[0]):
		return nil, nil
	default:
		return nil, fmt.Errorf("unexpected command: %v", arguments)
	}
}

func (remote *featureProbeRemote) RunCombined(
	context.Context, string, ...string,
) ([]byte, error) {
	remote.combinedCalls++
	return append([]byte("informational container diagnostic\n"), remote.profile...), nil
}

func (*featureProbeRemote) RunInput(
	context.Context, string, []byte, ...string,
) ([]byte, error) {
	return nil, nil
}

func (*featureProbeRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func TestDecodeHostFeatureProfileAttachesOnlySelectedCRIUEvidence(t *testing.T) {
	profile := snapshotdriver.FeatureProfile{
		Format: 1, Kind: "coldsnap-host-feature-profile",
		Identity: map[string]any{"boot_id": "boot-1"},
		Features: []snapshotdriver.FeatureResult{
			{ID: snapshotdriver.FeatureCRIUProcessTemplate, Status: snapshotdriver.FeatureUnqualified, Probe: "capsule-criu-check-v1", Reason: "pending"},
			{ID: snapshotdriver.FeatureCRIUProcessTree, Status: snapshotdriver.FeatureUnqualified, Probe: "capsule-criu-check-v1", Reason: "pending"},
			{ID: snapshotdriver.FeatureCUDAVMMExactAddress, Status: snapshotdriver.FeaturePassed, Probe: "cuda-vmm-exact-v1"},
		},
	}
	payload, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := decodeHostFeatureProfile(payload, snapshotdriver.N610)
	if err != nil {
		t.Fatal(err)
	}
	statuses := make(map[snapshotdriver.FeatureID]snapshotdriver.FeatureStatus)
	for _, result := range decoded.Features {
		statuses[result.ID] = result.Status
	}
	if statuses[snapshotdriver.FeatureCRIUProcessTree] != snapshotdriver.FeaturePassed ||
		statuses[snapshotdriver.FeatureCRIUProcessTemplate] != snapshotdriver.FeatureUnqualified {
		t.Fatalf("CRIU profile statuses = %#v", statuses)
	}
}

func TestHostFeatureProfileKeepsProbeStderrSeparateFromJSON(t *testing.T) {
	profile := snapshotdriver.FeatureProfile{
		Format: 1,
		Kind:   "coldsnap-host-feature-profile",
		Identity: map[string]any{
			"boot_id": "boot-1",
		},
		Features: []snapshotdriver.FeatureResult{{
			ID:      snapshotdriver.FeatureCUDAFreshState,
			Status:  snapshotdriver.FeaturePassed,
			Probe:   "cuda-fresh-state-v1",
			Seconds: 0.1,
		}},
		StartedUnix:   1,
		CompletedUnix: 2,
	}
	payload, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	remote := &featureProbeRemote{profile: payload}
	request := validRequest(1)
	unit := request.Launch.Units[0]
	result, cached, err := (fixtureAdapter(Adapter{
		Remote: remote, StateRoot: "/cache/coldsnap",
	})).hostFeatureProfile(
		context.Background(), request, unit, 0, "0",
		snapshot.CapsuleImage{
			Unit: unit.ID, Reference: unit.Image, Digest: unit.ImageDigest,
		},
		activationRuntimeBinding{
			HostPath: "/cache/probe.py", ContainerPath: activationruntime.FeatureProbeTarget,
		},
		testDigest,
		snapshotdriver.NewRequirements(snapshotdriver.FeatureCUDAFreshState),
	)
	if err != nil {
		t.Fatal(err)
	}
	if cached || remote.combinedCalls != 0 || result.Kind != profile.Kind {
		t.Fatalf("cached=%t combined=%d profile=%#v", cached, remote.combinedCalls, result)
	}
}

func (remote *activationRuntimeRemote) key(host, path string) string { return host + "\x00" + path }

func (remote *activationRuntimeRemote) Run(
	_ context.Context, host string, arguments ...string,
) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	if remote.files == nil {
		remote.files = make(map[string][]byte)
	}
	switch {
	case len(arguments) >= 2 && arguments[0] == "sha256sum":
		remote.shaCalls++
		var output strings.Builder
		for _, path := range arguments[1:] {
			data, ok := remote.files[remote.key(host, path)]
			if !ok {
				return nil, errors.New("not found")
			}
			digest := sha256.Sum256(data)
			fmt.Fprintf(&output, "%s  %s\n", hex.EncodeToString(digest[:]), path)
		}
		return []byte(output.String()), nil
	case len(arguments) == 4 && arguments[0] == "mv" && arguments[1] == "-f":
		data, ok := remote.files[remote.key(host, arguments[2])]
		if !ok {
			return nil, errors.New("temporary file is absent")
		}
		remote.files[remote.key(host, arguments[3])] = data
		delete(remote.files, remote.key(host, arguments[2]))
		return nil, nil
	case len(arguments) > 0 && slices.Contains([]string{"install", "test", "chmod"}, arguments[0]):
		return nil, nil
	default:
		return nil, fmt.Errorf("unexpected command: %v", arguments)
	}
}

func (remote *activationRuntimeRemote) RunCombined(
	_ context.Context, host string, arguments ...string,
) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	remote.criuCalls = append(remote.criuCalls, append([]string{host}, arguments...))
	if remote.criuError != nil {
		return []byte("Does not look good."), remote.criuError
	}
	return []byte("Looks good."), nil
}

func (remote *activationRuntimeRemote) RunInput(
	_ context.Context, host string, input []byte, arguments ...string,
) ([]byte, error) {
	remote.mutex.Lock()
	defer remote.mutex.Unlock()
	if len(arguments) != 2 || arguments[0] != "tee" {
		return nil, fmt.Errorf("unexpected input command: %v", arguments)
	}
	if remote.files == nil {
		remote.files = make(map[string][]byte)
	}
	remote.files[remote.key(host, arguments[1])] = slices.Clone(input)
	remote.inputWrites++
	return nil, nil
}

func (*activationRuntimeRemote) Upload(context.Context, string, []string, string, bool) error {
	return nil
}

func TestPrepareActivationRuntimeStagesOncePerHostAndReusesDigest(t *testing.T) {
	remote := &activationRuntimeRemote{}
	request := validRequest(2)
	adapter := fixtureAdapter(Adapter{Remote: remote, StateRoot: "/cache/coldsnap"})
	digest, err := adapter.prepareActivationRuntime(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	pack, err := activationRuntimePack()
	if err != nil {
		t.Fatal(err)
	}
	if digest != pack.SHA256 || remote.inputWrites != len(pack.Files)*2 {
		t.Fatalf("digest=%q writes=%d pack=%#v", digest, remote.inputWrites, pack)
	}
	priorSHACalls := remote.shaCalls
	if _, err := adapter.prepareActivationRuntime(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if remote.inputWrites != len(pack.Files)*2 {
		t.Fatalf("cached activation runtime was rewritten %d times", remote.inputWrites)
	}
	if remote.shaCalls-priorSHACalls != 2 {
		t.Fatalf("warm cache digest calls = %d, want one per host", remote.shaCalls-priorSHACalls)
	}
}

func TestRankCommandMountsActivationRuntimeAndKernelPolicy(t *testing.T) {
	request := validRequest(1)
	request.Policy.Compatibility.Kernel = snapshot.KernelCompatibilityExact
	adapter := fixtureAdapter(Adapter{Remote: fakeRemote{}, StateRoot: "/cache/coldsnap"})
	command, err := rankCommandFixture(adapter,
		context.Background(), request, request.Launch.Units[0], request.Launch.Units[0].Image,
		"capture", "capture", "namespace", request.ID, "/capture", "/endpoint", "recovery",
		nil, request.Launch.Units[0].ImageDigest, nil, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	pack, err := activationRuntimePack()
	if err != nil {
		t.Fatal(err)
	}
	for _, binding := range activationRuntimeBindings(adapter.StateRoot, pack) {
		if !slices.Contains(command, binding.HostPath+":"+binding.ContainerPath+":ro") {
			t.Fatalf("activation runtime binding is absent: %s -> %s\n%v", binding.HostPath, binding.ContainerPath, command)
		}
	}
	index := slices.Index(command, "--kernel-compatibility")
	if index < 0 || index+1 >= len(command) || command[index+1] != snapshot.KernelCompatibilityExact {
		t.Fatalf("kernel compatibility command = %v", command)
	}
}

func TestActivationRuntimeTargetsRejectShadowingMounts(t *testing.T) {
	for _, path := range []string{
		"/usr/local/bin",
		"/usr/local/bin/coldsnap-engine-rank-n610",
		"/opt/coldsnap/runtime",
		"/opt/coldsnap/runtime/coldsnap_activation_logs.py",
		"/opt/coldsnap/runtime/coldsnap_service_runtime.py",
		"/opt/coldsnap/runtime/coldsnap-engine-exec.py",
	} {
		if !isActivationRuntimeTarget(path) {
			t.Fatalf("activation runtime shadowing path was accepted: %s", path)
		}
	}
	for _, path := range []string{"/model", "/opt/coldsnap/plugin", "/usr/local/lib"} {
		if isActivationRuntimeTarget(path) {
			t.Fatalf("unrelated mount was rejected: %s", path)
		}
	}
}

func TestVerifyCRIUCapabilitiesUsesPinnedCapsuleOnEveryUnit(t *testing.T) {
	remote := &activationRuntimeRemote{}
	request := validRequest(2)
	images := map[string]snapshot.CapsuleImage{
		"unit-0": {Unit: "unit-0", Reference: "registry/capsule0@" + testDigest},
		"unit-1": {Unit: "unit-1", Reference: "registry/capsule1@" + testDigest},
	}
	if err := (fixtureAdapter(Adapter{Remote: remote})).verifyCRIUCapabilities(
		context.Background(), request, images,
	); err != nil {
		t.Fatal(err)
	}
	if len(remote.criuCalls) != 2 {
		t.Fatalf("CRIU calls = %#v", remote.criuCalls)
	}
	for _, call := range remote.criuCalls {
		joined := strings.Join(call, " ")
		for _, expected := range []string{
			"--pull never", "--privileged", "seccomp=unconfined", "--network host",
			"--entrypoint /opt/coldsnap/criu/bin/criu", " check",
		} {
			if !strings.Contains(joined, expected) {
				t.Fatalf("CRIU call lacks %q: %v", expected, call)
			}
		}
	}
	remote.criuError = errors.New("exit status 1")
	if err := (fixtureAdapter(Adapter{Remote: remote})).verifyCRIUCapabilities(
		context.Background(), request, images,
	); err == nil || !strings.Contains(err.Error(), "Does not look good") {
		t.Fatalf("CRIU capability failure = %v", err)
	}
}

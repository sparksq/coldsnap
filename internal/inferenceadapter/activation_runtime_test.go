// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/sparksq/coldsnap/internal/buildinfo"
	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
	activationruntime "github.com/sparksq/coldsnap/runtime/engine"
)

type activationRuntimeRemote struct {
	platforms   map[string]string
	identity    []byte
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
	if slices.Equal(arguments, []string{"uname", "-sm"}) && remote.platforms != nil {
		return []byte(remote.platforms[host]), nil
	}
	if len(arguments) == 3 && arguments[1] == "version" && remote.identity != nil {
		return remote.identity, nil
	}
	if result, ok := activationProbeFixture(arguments); ok {
		return result, nil
	}
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

func activationProbeFixture(arguments []string) ([]byte, bool) {
	if slices.Equal(arguments, []string{"uname", "-sm"}) {
		machine := map[string]string{"amd64": "x86_64", "arm64": "aarch64"}[runtime.GOARCH]
		return []byte("Linux " + machine + "\n"), true
	}
	if len(arguments) == 3 && slices.Equal(arguments[1:], []string{"version", "--json"}) {
		data, _ := json.Marshal(buildinfo.Current())
		return data, true
	}
	return nil, false
}

func targetELFFixture(machine uint16) []byte {
	data := make([]byte, 64)
	copy(data, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	binary.LittleEndian.PutUint16(data[16:], 2)
	binary.LittleEndian.PutUint16(data[18:], machine)
	binary.LittleEndian.PutUint32(data[20:], 1)
	binary.LittleEndian.PutUint16(data[52:], 64)
	return data
}

func configureNativeActivationTools(t *testing.T) {
	t.Helper()
	machine := map[string]uint16{"amd64": 62, "arm64": 183}[runtime.GOARCH]
	directory := t.TempDir()
	for _, name := range []string{"verifier", "helper"} {
		if err := os.WriteFile(filepath.Join(directory, name), targetELFFixture(machine), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv(targetVerifierEnvironment, filepath.Join(directory, "verifier"))
	t.Setenv(targetCRIURPCEnvironment, filepath.Join(directory, "helper"))
}

func TestActivationTargetOverridesReplaceBothControllerExecutables(t *testing.T) {
	directory := t.TempDir()
	verifier := filepath.Join(directory, "verifier")
	helper := filepath.Join(directory, "helper")
	for _, path := range []string{verifier, helper} {
		if err := os.WriteFile(path, targetELFFixture(183), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv(targetVerifierEnvironment, verifier)
	t.Setenv(targetCRIURPCEnvironment, helper)
	pack, err := activationRuntimePack()
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, item := range pack.Files {
		if item.Name == activationruntime.PayloadVerifierFile || item.Name == "coldsnap-criu-rpc" {
			architecture, err := activationBinaryArchitecture(item.Data)
			if err != nil || architecture != "aarch64" {
				t.Fatalf("%s architecture=%s err=%v", item.Name, architecture, err)
			}
			count++
		}
	}
	if count != 2 {
		t.Fatalf("target executable count=%d", count)
	}
	t.Setenv(targetCRIURPCEnvironment, "")
	if _, err := activationRuntimePack(); err == nil {
		t.Fatal("partial target override accepted")
	}
}

func TestActivationRejectsForeignELFBeforeStaging(t *testing.T) {
	foreign := uint16(183)
	if runtime.GOARCH == "arm64" {
		foreign = 62
	}
	directory := t.TempDir()
	for _, name := range []string{"verifier", "helper"} {
		if err := os.WriteFile(filepath.Join(directory, name), targetELFFixture(foreign), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv(targetVerifierEnvironment, filepath.Join(directory, "verifier"))
	t.Setenv(targetCRIURPCEnvironment, filepath.Join(directory, "helper"))
	remote := &activationRuntimeRemote{}
	_, err := (Adapter{Remote: remote, StateRoot: directory}).prepareActivationRuntime(context.Background(), validRequest(2))
	if err == nil || !strings.Contains(err.Error(), "supply release-matched target tools") || remote.inputWrites != 0 {
		t.Fatalf("err=%v writes=%d", err, remote.inputWrites)
	}
}

func TestActivationBinaryRejectsInvalidELF(t *testing.T) {
	for _, data := range [][]byte{nil, []byte("#!/bin/sh\nexit 0\n"), targetELFFixture(40)} {
		if _, err := activationBinaryArchitecture(data); err == nil {
			t.Fatal("invalid ELF accepted")
		}
	}
}

func TestActivationAdmitsAllHostsBeforeStaging(t *testing.T) {
	configureNativeActivationTools(t)
	native := map[string]string{"amd64": "x86_64", "arm64": "aarch64"}[runtime.GOARCH]
	foreign := map[string]string{"amd64": "aarch64", "arm64": "x86_64"}[runtime.GOARCH]
	remote := &activationRuntimeRemote{platforms: map[string]string{"node-0": "Linux " + native, "node-1": "Linux " + foreign}}
	_, err := (Adapter{Remote: remote, StateRoot: t.TempDir()}).prepareActivationRuntime(context.Background(), validRequest(2))
	if err == nil || remote.inputWrites != 0 {
		t.Fatalf("err=%v writes=%d", err, remote.inputWrites)
	}
}

func TestActivationRejectsWrongReleaseIdentity(t *testing.T) {
	configureNativeActivationTools(t)
	for _, identity := range []string{`{"version":"0.0.1","commit":"wrong"}`, "not JSON"} {
		remote := &activationRuntimeRemote{identity: []byte(identity)}
		_, err := (Adapter{Remote: remote, StateRoot: t.TempDir()}).prepareActivationRuntime(context.Background(), validRequest(1))
		if err == nil || !strings.Contains(err.Error(), "identity") {
			t.Fatalf("err=%v", err)
		}
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
	configureNativeActivationTools(t)
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

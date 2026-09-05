// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package ncclprovider

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
)

type providerFixture struct {
	catalog  string
	root     string
	manifest Manifest
	target   Target
}

func newProviderFixture(t *testing.T, catalog, providerID string) providerFixture {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("provider ELF validation is Linux-only")
	}
	architecture := runtime.GOARCH
	switch architecture {
	case "arm64":
		architecture = "aarch64"
	case "amd64":
		architecture = "x86_64"
	}
	platformKey := "linux-" + architecture + "-cuda13-glibc2.38"
	if architecture != "aarch64" && architecture != "x86_64" {
		t.Skip("provider fixture requires a supported architecture")
	}
	if catalog == "" {
		catalog = t.TempDir()
	}
	root := filepath.Join(catalog, "providers", providerID, platformKey)
	if err := os.MkdirAll(filepath.Join(root, "lib"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "provenance"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, directory := range []string{"licenses/coldsnap", "licenses/third-party/nccl"} {
		if err := os.MkdirAll(filepath.Join(root, filepath.FromSlash(directory)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	buildSharedObject(t, filepath.Join(root, "lib/libnccl.so.2.31.2"), "libnccl.so.2")
	buildSharedObject(t, filepath.Join(root, "lib/libnccl-checkpoint-shim.so"), "libnccl-checkpoint-shim.so")
	contents := map[string]string{
		"licenses/coldsnap/LICENSE":                "AGPL-3.0-only test license\n",
		"licenses/coldsnap/THIRD_PARTY_NOTICES.md": "test third-party notices\n",
		"licenses/third-party/nccl/LICENSE.txt":    "BSD-3-Clause test license\n",
		"provenance/qualification.json":            `{"state":"accepted"}\n`,
		"provenance/source.lock":                   "locked source\n",
		"provenance/recipe.json":                   `{"recipe":"test"}\n`,
		"provenance/patch-series.sha256":           strings.Repeat("1", 64) + "  series\n",
		"provenance/build-record.json":             `{"builder":"test"}\n`,
	}
	for path, content := range contents {
		if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(path)), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	roles := map[string]string{
		"lib/libnccl.so.2.31.2":                    RoleNCCLRuntime,
		"lib/libnccl-checkpoint-shim.so":           RoleCheckpointShim,
		"licenses/coldsnap/LICENSE":                RoleLicense,
		"licenses/coldsnap/THIRD_PARTY_NOTICES.md": RoleThirdParty,
		"licenses/third-party/nccl/LICENSE.txt":    RoleNCCLLicense,
		"provenance/qualification.json":            "qualification-policy",
		"provenance/source.lock":                   "source-lock",
		"provenance/recipe.json":                   "recipe",
		"provenance/patch-series.sha256":           "patch-series",
		"provenance/build-record.json":             "build-record",
	}
	paths := make([]string, 0, len(roles))
	for path := range roles {
		paths = append(paths, path)
	}
	slices.Sort(paths)
	files := make([]ProviderFile, 0, len(paths))
	for _, path := range paths {
		absolute := filepath.Join(root, filepath.FromSlash(path))
		info, err := os.Stat(absolute)
		if err != nil {
			t.Fatal(err)
		}
		file := ProviderFile{
			Path: path, Role: roles[path], Mode: "0644", Size: info.Size(), SHA256: mustHash(t, absolute),
		}
		if file.Role == RoleNCCLRuntime || file.Role == RoleCheckpointShim {
			identity, err := inspectELF(absolute)
			if err != nil {
				t.Fatal(err)
			}
			file.Mode = "0755"
			file.BuildID = identity.BuildID
			file.SONAME = identity.SONAME
		}
		files = append(files, file)
	}
	digestForRole := func(role string) string {
		for _, file := range files {
			if file.Role == role {
				return file.SHA256
			}
		}
		t.Fatalf("missing fixture role %q", role)
		return ""
	}
	fileForRole := func(role string) ProviderFile {
		for _, file := range files {
			if file.Role == role {
				return file
			}
		}
		t.Fatalf("missing fixture role %q", role)
		return ProviderFile{}
	}
	runtimeIdentity := NCCLRuntimeIdentity{
		Version: 23102, Release: "2.31.2", Path: "/usr/lib/" + architecture + "-linux-gnu/libnccl.so.2",
		SONAME: "libnccl.so.2", BuildID: strings.Repeat("a", 40), SHA256: strings.Repeat("b", 64),
	}
	requirements := PlatformRequirements{
		Architecture: architecture, CUDAUserspace: "13.0", CUDARuntimeSONAME: "libcudart.so.13",
		GlibcMin: "2.38", LibstdcxxABIMin: "3.4.30",
	}
	revision, err := strconv.ParseUint(providerID[strings.LastIndex(providerID, ".")+1:], 10, 32)
	if err != nil {
		t.Fatal(err)
	}
	capabilities := []string{"communicator-unwrap", "full-network-reset", "ib-roce-device-release", "private-dlopen-routing", "ras-reset", "synchronous-termination"}
	manifest := Manifest{
		Format: ManifestFormat, Kind: ManifestKind, ProviderID: providerID, ProviderRevision: uint32(revision),
		PlatformKey: platformKey,
		ProviderNCCLRuntime: ProviderNCCLRuntimeIdentity{
			Version: 23102, Release: "2.31.2", SONAME: fileForRole(RoleNCCLRuntime).SONAME,
			BuildID: fileForRole(RoleNCCLRuntime).BuildID, SHA256: fileForRole(RoleNCCLRuntime).SHA256,
		},
		ProviderSelection:     ProviderSelectionExact,
		SupportedNCCLRuntimes: []NCCLRuntimeIdentity{runtimeIdentity},
		ProviderABI:           ABIVersion{Major: 1, Minor: 0}, CheckpointABI: 2,
		Capabilities: capabilities,
		Limitations:  []string{}, Requirements: requirements, Files: files,
		Provenance: Provenance{
			SourceLockSHA256: digestForRole("source-lock"), RecipeSHA256: digestForRole("recipe"),
			PatchSeriesSHA256: digestForRole("patch-series"), BuildRecordSHA256: digestForRole("build-record"),
		},
		Qualification: Qualification{
			State: "accepted", Policy: "production", Capabilities: slices.Clone(capabilities),
			Transports: []string{"ib-roce", "socket"}, Checks: []string{"abi-contract", "socket-restore"},
		},
		DLSymBridgeABI: 1, PreloadOrder: []string{RoleDLSymBridge, RoleCheckpointShim, RoleNCCLRuntime},
	}
	writeManifest(t, root, manifest)
	target := Target{
		Format: TargetFormat, Kind: TargetKind, BaseImageDigest: "sha256:" + strings.Repeat("c", 64),
		PlatformKey: platformKey, NCCL: runtimeIdentity, Requirements: requirements,
		NCCLPolicy:             NCCLPolicyExact,
		RequiredProviderABI:    ABIVersion{Major: 1, Minor: 0},
		RequiredCapabilities:   []string{"full-network-reset", "synchronous-termination"},
		RequiredDLSymBridgeABI: 1, QualificationPolicy: "production", Transport: "socket",
	}
	return providerFixture{catalog: catalog, root: root, manifest: manifest, target: target}
}

func buildSharedObject(t *testing.T, path, soname string) {
	t.Helper()
	command := exec.Command("cc", "-x", "c", "-shared", "-fPIC", "-Wl,--build-id=sha1", "-Wl,-soname,"+soname, "-o", path, "-")
	command.Stdin = strings.NewReader("int coldsnap_provider_fixture(void) { return 0; }\n")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build fixture shared object: %v: %s", err, output)
	}
	if err := os.Chmod(path, 0o755); err != nil {
		t.Fatal(err)
	}
}

func mustHash(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func writeManifest(t *testing.T, root string, manifest Manifest) {
	t.Helper()
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(filepath.Join(root, ManifestFilename), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyAndExactSelection(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	verification, err := Verify(fixture.root)
	if err != nil {
		t.Fatal(err)
	}
	if verification.ProviderID != fixture.manifest.ProviderID || len(verification.ManifestSHA256) != 64 {
		t.Fatalf("verification = %#v", verification)
	}
	selection, err := Select(fixture.catalog, fixture.target)
	if err != nil {
		t.Fatal(err)
	}
	if selection.ProviderID != fixture.manifest.ProviderID || selection.SourcePath != fixture.root {
		t.Fatalf("selection = %#v", selection)
	}
}

func TestSelectionNeverApproximatesNCCLVersion(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	for _, mutation := range []func(*Target){
		func(target *Target) { target.NCCL.Version, target.NCCL.Release = 23101, "2.31.1" },
		func(target *Target) { target.NCCL.Version, target.NCCL.Release = 23007, "2.30.7" },
	} {
		target := fixture.target
		mutation(&target)
		_, err := Select(fixture.catalog, target)
		var noMatch NoMatchError
		if !errors.As(err, &noMatch) || !strings.Contains(err.Error(), "NCCL version or SONAME mismatch") {
			t.Fatalf("wrong-version selection error = %v", err)
		}
	}
}

func TestSelectionAcceptsRebuiltBaseRuntimeWithSameABIIdentity(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	target := fixture.target
	target.NCCL.Path = "/opt/alternate/libnccl.so.2"
	target.NCCL.BuildID = strings.Repeat("d", 40)
	target.NCCL.SHA256 = strings.Repeat("e", 64)
	selection, err := Select(fixture.catalog, target)
	if err != nil || selection.ProviderID != fixture.manifest.ProviderID {
		t.Fatalf("same-ABI rebuilt runtime selection = %#v, %v", selection, err)
	}
}

func TestSelectionRejectsAmbiguity(t *testing.T) {
	catalog := t.TempDir()
	first := newProviderFixture(t, catalog, "nccl-2.31.2-1+coldsnap.10")
	newProviderFixture(t, catalog, "nccl-2.31.2-1+coldsnap.11")
	_, err := Select(catalog, first.target)
	if err == nil || !strings.Contains(err.Error(), "multiple exact") {
		t.Fatalf("ambiguity error = %v", err)
	}
}

func TestLoadManifestRejectsUnknownFieldsAndTraversal(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	path := filepath.Join(fixture.root, ManifestFilename)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatal(err)
	}
	payload["unexpected"] = true
	unknown, _ := json.Marshal(payload)
	if err := os.WriteFile(path, unknown, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadManifest(path); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown-field error = %v", err)
	}

	fixture = newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	fixture.manifest.Files[0].Path = "../escape"
	writeManifest(t, fixture.root, fixture.manifest)
	if _, err := LoadManifest(filepath.Join(fixture.root, ManifestFilename)); err == nil || !strings.Contains(err.Error(), "clean relative") {
		t.Fatalf("traversal error = %v", err)
	}
}

func TestManifestRequiresExplicitProviderRuntimeIdentityAndSelection(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	path := filepath.Join(fixture.root, ManifestFilename)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatal(err)
	}
	delete(payload, "provider_nccl_runtime")
	delete(payload, "provider_selection")
	incomplete, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(incomplete, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadManifest(path); err == nil || !strings.Contains(err.Error(), "provider_nccl_runtime") {
		t.Fatalf("missing provider identity error = %v", err)
	}
}

func TestManifestRejectsDuplicatedIdentityDisagreement(t *testing.T) {
	tests := map[string]func(*Manifest){
		"provider revision": func(manifest *Manifest) { manifest.ProviderRevision++ },
		"provider release":  func(manifest *Manifest) { manifest.ProviderNCCLRuntime.Release = "2.31.1" },
		"platform CUDA":     func(manifest *Manifest) { manifest.Requirements.CUDAUserspace = "13.1" },
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
			mutate(&fixture.manifest)
			writeManifest(t, fixture.root, fixture.manifest)
			if _, err := LoadManifest(filepath.Join(fixture.root, ManifestFilename)); err == nil || !strings.Contains(err.Error(), "disagrees") {
				t.Fatalf("duplicated identity disagreement error = %v", err)
			}
		})
	}
}

func TestVerifyRejectsTamperSymlinkHardlinkAndExtraFile(t *testing.T) {
	tests := map[string]func(*testing.T, providerFixture){
		"tamper": func(t *testing.T, fixture providerFixture) {
			if err := os.WriteFile(filepath.Join(fixture.root, "provenance/qualification.json"), []byte("changed\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		},
		"symlink": func(t *testing.T, fixture providerFixture) {
			if err := os.Symlink("provenance/qualification.json", filepath.Join(fixture.root, "unexpected-link")); err != nil {
				t.Fatal(err)
			}
		},
		"hardlink": func(t *testing.T, fixture providerFixture) {
			if err := os.Link(filepath.Join(fixture.root, "provenance/qualification.json"), filepath.Join(fixture.root, "hardlink")); err != nil {
				t.Fatal(err)
			}
		},
		"extra": func(t *testing.T, fixture providerFixture) {
			if err := os.WriteFile(filepath.Join(fixture.root, "extra.so"), []byte("extra"), 0o755); err != nil {
				t.Fatal(err)
			}
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
			mutate(t, fixture)
			if _, err := Verify(fixture.root); err == nil {
				t.Fatal("Verify accepted corrupt provider")
			}
		})
	}
}

func TestMaterializeIsAtomicAndExactlyIdempotent(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	output := filepath.Join(t.TempDir(), "provider")
	first, err := Materialize(fixture.catalog, fixture.target, output)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Materialize(fixture.catalog, fixture.target, output)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("materializations differ: %#v %#v", first, second)
	}
	verification, err := Verify(output)
	if err != nil {
		t.Fatal(err)
	}
	if verification.ManifestSHA256 != first.ManifestSHA256 {
		t.Fatalf("verification = %#v", verification)
	}
	if err := os.WriteFile(filepath.Join(output, "extra"), []byte("bad"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Materialize(fixture.catalog, fixture.target, output); err == nil || !strings.Contains(err.Error(), "not the exact selected provider") {
		t.Fatalf("nonexact reuse error = %v", err)
	}
}

func TestInstallActivePublishesImmutableResolvedPaths(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	bridge := filepath.Join(t.TempDir(), "libcoldsnap-nccl-dlsym.so")
	buildSharedObject(t, bridge, "libcoldsnap-nccl-dlsym.so")
	output := filepath.Join(t.TempDir(), "nccl")
	record, err := InstallActive(fixture.root, bridge, output, 1)
	if err != nil {
		t.Fatal(err)
	}
	if record.ProviderID != fixture.manifest.ProviderID || record.ManifestSHA256 == "" {
		t.Fatalf("active record = %#v", record)
	}
	if !strings.HasPrefix(record.ProviderRoot, filepath.Join(output, "providers")+string(os.PathSeparator)) {
		t.Fatalf("provider root is not immutable: %q", record.ProviderRoot)
	}
	wantPreload := []string{
		record.Bridge.Path,
		record.Files[RoleCheckpointShim].Path,
		record.Files[RoleNCCLRuntime].Path,
	}
	if !slices.Equal(record.PreloadOrder, wantPreload) {
		t.Fatalf("preload order = %#v", record.PreloadOrder)
	}
	if filepath.Base(record.Files[RoleCheckpointShim].Path) != ActiveCheckpointShimFilename ||
		strings.Contains(filepath.Base(record.Files[RoleCheckpointShim].Path), "libnccl") {
		t.Fatalf("checkpoint shim runtime path can be misidentified as NCCL: %q", record.Files[RoleCheckpointShim].Path)
	}
	providerShim := filepath.Join(record.ProviderRoot, "lib/libnccl-checkpoint-shim.so")
	if mustHash(t, providerShim) != mustHash(t, record.Files[RoleCheckpointShim].Path) {
		t.Fatal("active checkpoint shim does not preserve provider bytes")
	}
	data, err := os.ReadFile(filepath.Join(output, ActiveFilename))
	if err != nil {
		t.Fatal(err)
	}
	var saved ActiveRecord
	if err := decodeStrict(data, &saved); err != nil {
		t.Fatal(err)
	}
	if saved.ManifestSHA256 != record.ManifestSHA256 {
		t.Fatalf("saved active record = %#v", saved)
	}
	second, err := InstallActive(fixture.root, bridge, output, 1)
	if err != nil || second.ManifestSHA256 != record.ManifestSHA256 {
		t.Fatalf("idempotent install = %#v, %v", second, err)
	}
	verified, err := VerifyActive(filepath.Join(output, ActiveFilename))
	if err != nil || verified.ManifestSHA256 != record.ManifestSHA256 {
		t.Fatalf("verified active runtime = %#v, %v", verified, err)
	}
	if err := os.WriteFile(record.Bridge.Path, []byte("tampered"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyActive(filepath.Join(output, ActiveFilename)); err == nil || !strings.Contains(err.Error(), "bridge digest") {
		t.Fatalf("tampered active runtime error = %v", err)
	}
}

func TestInstallExactFileRejectsDigestDisagreement(t *testing.T) {
	source := filepath.Join(t.TempDir(), "bridge.so")
	if err := os.WriteFile(source, []byte("bridge"), 0o755); err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(t.TempDir(), "common", "bridge.so")
	err := installExactFile(source, destination, 0o755, strings.Repeat("0", 64))
	if err == nil || !strings.Contains(err.Error(), "changed while installing") {
		t.Fatalf("digest disagreement error = %v", err)
	}
	if _, statErr := os.Stat(destination); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("digest-disagreeing bridge was published: %v", statErr)
	}
}

func TestAssembleBindsQualifiedPayloadAndTarget(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	sourceRoot := t.TempDir()
	releaseRoot := filepath.Join(sourceRoot, "native/nccl/releases/2.31.2-1")
	patchRoot := filepath.Join(releaseRoot, "patches")
	if err := os.MkdirAll(patchRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	inputs := map[string]string{
		"LICENSE":                                                   "AGPL-3.0-only test license\n",
		"THIRD_PARTY_NOTICES.md":                                    "test third-party notices\n",
		"deploy/nccl/Dockerfile.payload":                            "FROM scratch\n",
		"deploy/nccl/Dockerfile.provider":                           "FROM scratch\n",
		"third_party/licenses/nccl/LICENSE.txt":                     "BSD-3-Clause test license\n",
		"native/nccl/abi/coldsnap_nccl_provider.h":                  "provider header\n",
		"native/nccl/releases/2.31.2-1/coldsnap_provider.cc":        "provider source\n",
		"native/nccl/releases/2.31.2-1/source.lock":                 "source lock\n",
		"native/nccl/releases/2.31.2-1/patches/0001-provider.patch": "patch\n",
		"native/nccl/releases/2.31.2-1/patches/series":              "0001-provider.patch\n",
	}
	for relative, content := range inputs {
		path := filepath.Join(sourceRoot, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runtimePath := filepath.Join(fixture.root, "lib/libnccl.so.2.31.2")
	shimPath := filepath.Join(fixture.root, "lib/libnccl-checkpoint-shim.so")
	runtimeELF, err := inspectELF(runtimePath)
	if err != nil {
		t.Fatal(err)
	}
	shimELF, err := inspectELF(shimPath)
	if err != nil {
		t.Fatal(err)
	}
	patchPath := filepath.Join(patchRoot, "0001-provider.patch")
	stripUnneeded := false
	recipe := providerRecipe{
		Format: recipeFormat, Kind: recipeKind, ProviderID: fixture.manifest.ProviderID,
		ProviderRevision: fixture.manifest.ProviderRevision, NCCLRelease: "2.31.2", NCCLVersionCode: 23102,
		ProviderABI: fixture.manifest.ProviderABI, CheckpointABI: fixture.manifest.CheckpointABI,
		DLSymBridgeABI: fixture.manifest.DLSymBridgeABI, Capabilities: fixture.manifest.Capabilities,
		Limitations: []string{},
		Builder: recipeBuilder{
			ProviderDockerfile: "deploy/nccl/Dockerfile.provider",
			PayloadDockerfile:  "deploy/nccl/Dockerfile.payload",
			PayloadRepository:  "docker.io/scitrera/coldsnap-nccl",
			PayloadBuildImage:  "registry.example/qualified-build@sha256:" + strings.Repeat("d", 64),
			PayloadPlatforms:   []string{"linux/amd64", "linux/arm64"},
			BaseImagePolicy:    "digest-pinned-multiarch-payload-and-target-images",
		},
		Inputs: recipeInputs{
			SourceLock:     "native/nccl/releases/2.31.2-1/source.lock",
			ProviderHeader: "native/nccl/abi/coldsnap_nccl_provider.h",
			ProviderSource: "native/nccl/releases/2.31.2-1/coldsnap_provider.cc",
			Patches:        []recipePatch{{Path: "native/nccl/releases/2.31.2-1/patches/0001-provider.patch", SHA256: mustHash(t, patchPath)}},
		},
		Build: recipeBuild{NCCLBuildJobsArgument: "NCCL_BUILD_JOBS", NVCCGencodeArgument: "NCCL_NVCC_GENCODE", NVCCGencode: "-gencode=arch=compute_80,code=sm_80", UseReproducibleNVCC: &stripUnneeded, StripUnneeded: &stripUnneeded, VerifyCompiledAndLoadedVersion: true},
		Outputs: []providerOutput{
			{Path: "lib/libnccl.so.2.31.2", Role: RoleNCCLRuntime, SONAME: runtimeELF.SONAME},
			{Path: "lib/libnccl-checkpoint-shim.so", Role: RoleCheckpointShim, SONAME: shimELF.SONAME},
		},
	}
	recipePath := filepath.Join(releaseRoot, "recipe.json")
	writeJSONFixture(t, recipePath, recipe)
	qualification := qualificationSource{
		Format: recipeFormat, Kind: qualifyKind, ProviderID: recipe.ProviderID, State: "accepted",
		Policy: "production", Capabilities: slices.Clone(recipe.Capabilities),
		Transports: []string{"ib-roce", "socket"}, Checks: []string{"abi-contract", "socket-restore"},
	}
	qualificationPath := filepath.Join(releaseRoot, "qualification.json")
	writeJSONFixture(t, qualificationPath, qualification)
	targetPath := filepath.Join(sourceRoot, "target.json")
	writeJSONFixture(t, targetPath, fixture.target)
	output := filepath.Join(t.TempDir(), "provider")
	verification, err := Assemble(AssemblyOptions{
		SourceRoot: sourceRoot, RecipePath: recipePath, QualificationPath: qualificationPath,
		PayloadRoot: fixture.root, TargetPath: targetPath, Output: output,
	})
	if err != nil {
		t.Fatal(err)
	}
	if verification.ProviderID != recipe.ProviderID || verification.Root != output {
		t.Fatalf("assembly verification = %#v", verification)
	}
	loaded, err := LoadManifest(filepath.Join(output, ManifestFilename))
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Manifest.SupportedNCCLRuntimes[0] != fixture.target.NCCL ||
		loaded.Manifest.ProviderNCCLRuntime.Version != recipe.NCCLVersionCode ||
		loaded.Manifest.ProviderSelection != ProviderSelectionExact ||
		!slices.Equal(loaded.Manifest.Qualification.Capabilities, recipe.Capabilities) {
		t.Fatalf("assembled manifest = %#v", loaded.Manifest)
	}
	for path, content := range map[string]string{
		"licenses/coldsnap/LICENSE":                inputs["LICENSE"],
		"licenses/coldsnap/THIRD_PARTY_NOTICES.md": inputs["THIRD_PARTY_NOTICES.md"],
		"licenses/third-party/nccl/LICENSE.txt":    inputs["third_party/licenses/nccl/LICENSE.txt"],
	} {
		data, err := os.ReadFile(filepath.Join(output, filepath.FromSlash(path)))
		if err != nil || string(data) != content {
			t.Fatalf("assembled license %s = %q, %v", path, data, err)
		}
	}
}

func TestProviderSelectionPolicyAllowsOnlyNewerSameMajorFallback(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	recipe := providerRecipe{NCCLRelease: "2.31.2", NCCLVersionCode: 23102}
	target := fixture.target
	target.NCCL.Release = "2.30.7"
	target.NCCL.Version = 23007

	if _, err := providerSelectionFor(recipe, target); err == nil || !strings.Contains(err.Error(), "exact target") {
		t.Fatalf("exact policy mismatch error = %v", err)
	}
	target.NCCLPolicy = NCCLPolicyMatchOrLatestQualified
	selection, err := providerSelectionFor(recipe, target)
	if err != nil || selection != ProviderSelectionFallbackUpgrade {
		t.Fatalf("fallback selection = %q, %v", selection, err)
	}

	for name, release := range map[string]string{"older": "2.29.9", "other-major": "3.0.0"} {
		t.Run(name, func(t *testing.T) {
			candidate := recipe
			candidate.NCCLRelease = release
			parts, _ := numericVersion(release)
			candidate.NCCLVersionCode = int(parts[0]*10000 + parts[1]*100 + parts[2])
			if _, err := providerSelectionFor(candidate, target); err == nil {
				t.Fatal("incompatible fallback was accepted")
			}
		})
	}
}

func TestExactCatalogSelectionUsesBaseIdentityNotProviderPayloadVersion(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	base := fixture.manifest.SupportedNCCLRuntimes[0]
	base.Version = 23007
	base.Release = "2.30.7"
	fixture.manifest.SupportedNCCLRuntimes = []NCCLRuntimeIdentity{base}
	fixture.manifest.ProviderSelection = ProviderSelectionFallbackUpgrade
	fixture.target.NCCL = base
	fixture.target.NCCLPolicy = NCCLPolicyMatchOrLatestQualified
	writeManifest(t, fixture.root, fixture.manifest)

	selection, err := Select(fixture.catalog, fixture.target)
	if err != nil || selection.ProviderID != fixture.manifest.ProviderID {
		t.Fatalf("fallback provider catalog selection = %#v, %v", selection, err)
	}
	fixture.target.NCCLPolicy = NCCLPolicyExact
	if _, err := Select(fixture.catalog, fixture.target); err == nil || !strings.Contains(err.Error(), "requires exact") {
		t.Fatalf("exact target accepted fallback provider: %v", err)
	}
}

func writeJSONFixture(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestSelectionRejectsMissingCapability(t *testing.T) {
	fixture := newProviderFixture(t, "", "nccl-2.31.2-1+coldsnap.10")
	target := fixture.target
	target.RequiredCapabilities = []string{"cuda-graph", "full-network-reset"}
	if _, err := Select(fixture.catalog, target); err == nil || !strings.Contains(err.Error(), "capability") {
		t.Fatalf("capability error = %v", err)
	}
}

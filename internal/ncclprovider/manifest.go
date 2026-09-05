// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Package ncclprovider validates and selects immutable NCCL checkpoint providers.
//
// Provider data is treated as untrusted input. JSON decoding is strict, paths
// are confined to a provider root, and provider code is inspected as ELF data
// rather than loaded into the controller process.
package ncclprovider

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const (
	ManifestFormat = 2
	ManifestKind   = "coldsnap-nccl-provider"
	TargetFormat   = 1
	TargetKind     = "coldsnap-nccl-provider-target"

	ManifestFilename = "provider-manifest.json"
	maxJSONBytes     = 4 << 20
)

const (
	NCCLPolicyExact                  = "exact"
	NCCLPolicyMatchOrLatestQualified = "match-or-latest-qualified"
	ProviderSelectionExact           = "exact"
	ProviderSelectionFallbackUpgrade = "fallback-upgrade"
)

const (
	RoleDLSymBridge    = "coldsnap-dlsym-bridge"
	RoleCheckpointShim = "checkpoint-shim"
	RoleNCCLRuntime    = "nccl-runtime"
	RoleLicense        = "license"
	RoleThirdParty     = "third-party-notices"
	RoleNCCLLicense    = "nccl-license"
)

var (
	providerIDPattern    = regexp.MustCompile(`^nccl-[0-9]+\.[0-9]+\.[0-9]+-[0-9]+\+coldsnap\.[0-9]+$`)
	platformKeyPattern   = regexp.MustCompile(`^linux-(?:aarch64|x86_64)-cuda[0-9]+(?:\.[0-9]+)?-glibc[0-9]+\.[0-9]+$`)
	platformParts        = regexp.MustCompile(`^linux-(aarch64|x86_64)-cuda([0-9]+(?:\.[0-9]+)?)-glibc([0-9]+\.[0-9]+)$`)
	releasePattern       = regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+$`)
	hexDigestPattern     = regexp.MustCompile(`^[0-9a-f]{64}$`)
	hexBuildIDPattern    = regexp.MustCompile(`^[0-9a-f]{8,128}$`)
	sonamePattern        = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9.+_-]*\.so(?:\.[0-9]+)*$`)
	capabilityPattern    = regexp.MustCompile(`^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$`)
	ociRepositoryPattern = regexp.MustCompile(`^[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?/[a-z0-9._-]+(?:/[a-z0-9._-]+)*$`)
	nvccGencodePattern   = regexp.MustCompile(`^[A-Za-z0-9_=,. -]+$`)
)

// ABIVersion is a size-independent major/minor compatibility declaration.
type ABIVersion struct {
	Major uint32 `json:"major"`
	Minor uint32 `json:"minor"`
}

// NCCLRuntimeIdentity is the complete base-image runtime identity for which a
// provider variant was assembled. It is deliberately distinct from the
// patched NCCL payload that the provider preloads.
type NCCLRuntimeIdentity struct {
	Version int    `json:"version"`
	Release string `json:"release"`
	Path    string `json:"path"`
	SONAME  string `json:"soname"`
	BuildID string `json:"build_id"`
	SHA256  string `json:"sha256"`
}

// ProviderNCCLRuntimeIdentity binds the patched NCCL payload. Its installed
// path is resolved by ActiveFile, so the immutable identity excludes a path.
type ProviderNCCLRuntimeIdentity struct {
	Version int    `json:"version"`
	Release string `json:"release"`
	SONAME  string `json:"soname"`
	BuildID string `json:"build_id"`
	SHA256  string `json:"sha256"`
}

// PlatformRequirements are provider build/runtime requirements. PlatformKey
// is still matched exactly; version floors are checked as an additional guard.
type PlatformRequirements struct {
	Architecture      string `json:"architecture"`
	CUDAUserspace     string `json:"cuda_userspace"`
	CUDARuntimeSONAME string `json:"cuda_runtime_soname"`
	GlibcMin          string `json:"glibc_min"`
	LibstdcxxABIMin   string `json:"libstdcxx_abi_min"`
}

// ProviderFile binds every regular payload file into the manifest.
type ProviderFile struct {
	Path    string `json:"path"`
	Role    string `json:"role"`
	Mode    string `json:"mode"`
	Size    int64  `json:"size"`
	SHA256  string `json:"sha256"`
	BuildID string `json:"build_id,omitempty"`
	SONAME  string `json:"soname,omitempty"`
}

// Provenance binds the provider to source, recipe, patch, and build records.
type Provenance struct {
	SourceLockSHA256  string `json:"source_lock_sha256"`
	RecipeSHA256      string `json:"recipe_sha256"`
	PatchSeriesSHA256 string `json:"patch_series_sha256"`
	BuildRecordSHA256 string `json:"build_record_sha256"`
}

// Qualification is a checked-in capability-admission policy. Payload and
// manifest hashes provide build integrity separately; qualification is not
// coupled to historical compiler output or benchmark-report files.
type Qualification struct {
	State        string   `json:"state"`
	Policy       string   `json:"policy"`
	Capabilities []string `json:"capabilities"`
	Transports   []string `json:"transports"`
	Checks       []string `json:"checks"`
}

// Manifest is format 2 of an immutable NCCL checkpoint provider variant.
type Manifest struct {
	Format                int                         `json:"format"`
	Kind                  string                      `json:"kind"`
	ProviderID            string                      `json:"provider_id"`
	ProviderRevision      uint32                      `json:"provider_revision"`
	PlatformKey           string                      `json:"platform_key"`
	ProviderNCCLRuntime   ProviderNCCLRuntimeIdentity `json:"provider_nccl_runtime"`
	ProviderSelection     string                      `json:"provider_selection"`
	SupportedNCCLRuntimes []NCCLRuntimeIdentity       `json:"supported_nccl_runtimes"`
	ProviderABI           ABIVersion                  `json:"provider_abi"`
	CheckpointABI         uint32                      `json:"checkpoint_abi"`
	Capabilities          []string                    `json:"capabilities"`
	Limitations           []string                    `json:"limitations"`
	Requirements          PlatformRequirements        `json:"requirements"`
	Files                 []ProviderFile              `json:"files"`
	Provenance            Provenance                  `json:"provenance"`
	Qualification         Qualification               `json:"qualification"`
	DLSymBridgeABI        uint32                      `json:"dlsym_bridge_abi"`
	PreloadOrder          []string                    `json:"preload_order"`
}

// Target is produced by probing a digest-pinned base image.
type Target struct {
	Format                 int                  `json:"format"`
	Kind                   string               `json:"kind"`
	BaseImageDigest        string               `json:"base_image_digest"`
	PlatformKey            string               `json:"platform_key"`
	NCCL                   NCCLRuntimeIdentity  `json:"nccl"`
	NCCLPolicy             string               `json:"nccl_policy"`
	Requirements           PlatformRequirements `json:"requirements"`
	RequiredProviderABI    ABIVersion           `json:"required_provider_abi"`
	RequiredCapabilities   []string             `json:"required_capabilities"`
	RequiredDLSymBridgeABI uint32               `json:"required_dlsym_bridge_abi"`
	QualificationPolicy    string               `json:"qualification_policy"`
	Transport              string               `json:"transport"`
}

// LoadedManifest retains the digest of the exact bytes that were parsed.
type LoadedManifest struct {
	Manifest Manifest `json:"manifest"`
	SHA256   string   `json:"manifest_sha256"`
	Path     string   `json:"path"`
}

// LoadManifest strictly parses a provider manifest without executing provider code.
func LoadManifest(path string) (LoadedManifest, error) {
	data, err := readBoundedRegular(path, maxJSONBytes)
	if err != nil {
		return LoadedManifest{}, fmt.Errorf("read provider manifest: %w", err)
	}
	var manifest Manifest
	if err := decodeStrict(data, &manifest); err != nil {
		return LoadedManifest{}, fmt.Errorf("decode provider manifest: %w", err)
	}
	if err := manifest.Validate(); err != nil {
		return LoadedManifest{}, fmt.Errorf("validate provider manifest: %w", err)
	}
	sum := sha256.Sum256(data)
	absolute, err := filepath.Abs(path)
	if err != nil {
		return LoadedManifest{}, fmt.Errorf("resolve provider manifest: %w", err)
	}
	return LoadedManifest{Manifest: manifest, SHA256: hex.EncodeToString(sum[:]), Path: absolute}, nil
}

// LoadTarget strictly parses a target observation.
func LoadTarget(path string) (Target, error) {
	data, err := readBoundedRegular(path, maxJSONBytes)
	if err != nil {
		return Target{}, fmt.Errorf("read NCCL provider target: %w", err)
	}
	var target Target
	if err := decodeStrict(data, &target); err != nil {
		return Target{}, fmt.Errorf("decode NCCL provider target: %w", err)
	}
	if err := target.Validate(); err != nil {
		return Target{}, fmt.Errorf("validate NCCL provider target: %w", err)
	}
	return target, nil
}

func decodeStrict(data []byte, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

func readBoundedRegular(path string, limit int64) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("path is not a regular file")
	}
	if info.Size() > limit {
		return nil, fmt.Errorf("file exceeds %d bytes", limit)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("file exceeds %d bytes", limit)
	}
	return data, nil
}

// Validate checks all manifest invariants that do not require filesystem access.
func (manifest Manifest) Validate() error {
	if manifest.Format != ManifestFormat || manifest.Kind != ManifestKind {
		return fmt.Errorf("expected format=%d kind=%q", ManifestFormat, ManifestKind)
	}
	if !providerIDPattern.MatchString(manifest.ProviderID) {
		return errors.New("provider_id is invalid")
	}
	if manifest.ProviderRevision == 0 {
		return errors.New("provider_revision must be positive")
	}
	providerRuntime, err := manifest.effectiveProviderNCCLRuntime()
	if err != nil {
		return err
	}
	if err := validateProviderIDFields(manifest.ProviderID, manifest.ProviderRevision, providerRuntime.Release); err != nil {
		return err
	}
	if err := providerRuntime.validate(); err != nil {
		return fmt.Errorf("provider_nccl_runtime: %w", err)
	}
	if manifest.ProviderSelection != ProviderSelectionExact && manifest.ProviderSelection != ProviderSelectionFallbackUpgrade {
		return errors.New("provider_selection is invalid")
	}
	if !platformKeyPattern.MatchString(manifest.PlatformKey) {
		return errors.New("platform_key is invalid")
	}
	if manifest.ProviderABI.Major == 0 {
		return errors.New("provider_abi.major must be positive")
	}
	if manifest.CheckpointABI == 0 || manifest.DLSymBridgeABI == 0 {
		return errors.New("checkpoint_abi and dlsym_bridge_abi must be positive")
	}
	if len(manifest.SupportedNCCLRuntimes) == 0 {
		return errors.New("supported_nccl_runtimes must not be empty")
	}
	seenRuntime := make(map[string]bool)
	for index, runtime := range manifest.SupportedNCCLRuntimes {
		if err := runtime.validate(); err != nil {
			return fmt.Errorf("supported_nccl_runtimes[%d]: %w", index, err)
		}
		key := runtime.identityKey()
		if seenRuntime[key] {
			return fmt.Errorf("supported_nccl_runtimes[%d] is duplicated", index)
		}
		seenRuntime[key] = true
	}
	if err := validateProviderSelection(manifest.ProviderSelection, providerRuntime, manifest.SupportedNCCLRuntimes); err != nil {
		return err
	}
	if err := validateSortedUniqueStrings("capabilities", manifest.Capabilities, capabilityPattern, false); err != nil {
		return err
	}
	if manifest.Limitations == nil {
		return errors.New("limitations must be explicitly present")
	}
	if err := validateSortedUniqueStrings("limitations", manifest.Limitations, capabilityPattern, true); err != nil {
		return err
	}
	if err := manifest.Requirements.validate(); err != nil {
		return fmt.Errorf("requirements: %w", err)
	}
	if err := validatePlatformKeyFields(manifest.PlatformKey, manifest.Requirements); err != nil {
		return err
	}
	if len(manifest.Files) == 0 {
		return errors.New("files must not be empty")
	}
	seenPath := make(map[string]bool)
	seenRole := make(map[string]bool)
	for index, file := range manifest.Files {
		if err := file.validate(); err != nil {
			return fmt.Errorf("files[%d]: %w", index, err)
		}
		if seenPath[file.Path] {
			return fmt.Errorf("files[%d].path is duplicated", index)
		}
		if seenRole[file.Role] {
			return fmt.Errorf("files[%d].role is duplicated", index)
		}
		seenPath[file.Path] = true
		seenRole[file.Role] = true
	}
	for _, role := range []string{
		RoleCheckpointShim, RoleNCCLRuntime, RoleLicense, RoleThirdParty, RoleNCCLLicense,
		"qualification-policy", "source-lock", "recipe", "patch-series", "build-record",
	} {
		if !seenRole[role] {
			return fmt.Errorf("files lacks required role %q", role)
		}
	}
	if err := validateDigest("provenance.source_lock_sha256", manifest.Provenance.SourceLockSHA256); err != nil {
		return err
	}
	if err := validateDigest("provenance.recipe_sha256", manifest.Provenance.RecipeSHA256); err != nil {
		return err
	}
	if err := validateDigest("provenance.patch_series_sha256", manifest.Provenance.PatchSeriesSHA256); err != nil {
		return err
	}
	if err := validateDigest("provenance.build_record_sha256", manifest.Provenance.BuildRecordSHA256); err != nil {
		return err
	}
	roleDigests := make(map[string]string, len(manifest.Files))
	for _, file := range manifest.Files {
		roleDigests[file.Role] = file.SHA256
	}
	for field, expected := range map[string]string{
		"source-lock": roleDigests["source-lock"], "recipe": roleDigests["recipe"],
		"patch-series": roleDigests["patch-series"], "build-record": roleDigests["build-record"],
	} {
		actual := map[string]string{
			"source-lock": manifest.Provenance.SourceLockSHA256, "recipe": manifest.Provenance.RecipeSHA256,
			"patch-series": manifest.Provenance.PatchSeriesSHA256, "build-record": manifest.Provenance.BuildRecordSHA256,
		}[field]
		if actual != expected {
			return fmt.Errorf("provenance %s digest does not match its file", field)
		}
	}
	if err := manifest.Qualification.validate(); err != nil {
		return fmt.Errorf("qualification: %w", err)
	}
	if !slices.Equal(manifest.Qualification.Capabilities, manifest.Capabilities) {
		return errors.New("qualification capabilities do not match provider capabilities")
	}
	wantPreload := []string{RoleDLSymBridge, RoleCheckpointShim, RoleNCCLRuntime}
	if !slices.Equal(manifest.PreloadOrder, wantPreload) {
		return fmt.Errorf("preload_order must equal %q", wantPreload)
	}
	return nil
}

func (runtime NCCLRuntimeIdentity) validate() error {
	if runtime.Version <= 0 || !releasePattern.MatchString(runtime.Release) {
		return errors.New("version/release is invalid")
	}
	if versionCode, err := ncclVersionCode(runtime.Release); err != nil || versionCode != runtime.Version {
		return errors.New("version disagrees with release")
	}
	if !filepath.IsAbs(runtime.Path) || filepath.Clean(runtime.Path) != runtime.Path {
		return errors.New("path must be an absolute clean path")
	}
	if !sonamePattern.MatchString(runtime.SONAME) {
		return errors.New("soname is invalid")
	}
	if !hexBuildIDPattern.MatchString(runtime.BuildID) {
		return errors.New("build_id is invalid")
	}
	return validateDigest("sha256", runtime.SHA256)
}

func (runtime ProviderNCCLRuntimeIdentity) validate() error {
	if runtime.Version <= 0 || !releasePattern.MatchString(runtime.Release) {
		return errors.New("version/release is invalid")
	}
	if versionCode, err := ncclVersionCode(runtime.Release); err != nil || versionCode != runtime.Version {
		return errors.New("version disagrees with release")
	}
	if !sonamePattern.MatchString(runtime.SONAME) {
		return errors.New("soname is invalid")
	}
	if !hexBuildIDPattern.MatchString(runtime.BuildID) {
		return errors.New("build_id is invalid")
	}
	return validateDigest("sha256", runtime.SHA256)
}

func ncclVersionCode(release string) (int, error) {
	parts, err := numericVersion(release)
	if err != nil || len(parts) != 3 || parts[0] > 99 || parts[1] > 99 || parts[2] > 99 {
		return 0, errors.New("NCCL release is invalid")
	}
	return int(parts[0]*10000 + parts[1]*100 + parts[2]), nil
}

func validateProviderSelection(selection string, provider ProviderNCCLRuntimeIdentity, bases []NCCLRuntimeIdentity) error {
	providerRelease, err := numericVersion(provider.Release)
	if err != nil || len(providerRelease) != 3 {
		return errors.New("provider NCCL release is invalid")
	}
	for _, base := range bases {
		baseRelease, err := numericVersion(base.Release)
		if err != nil || len(baseRelease) != 3 {
			return errors.New("supported base NCCL release is invalid")
		}
		switch selection {
		case ProviderSelectionExact:
			if provider.Version != base.Version || provider.Release != base.Release {
				return errors.New("exact provider selection disagrees with supported base NCCL")
			}
		case ProviderSelectionFallbackUpgrade:
			if providerRelease[0] != baseRelease[0] || provider.Version <= base.Version {
				return errors.New("fallback provider selection is not a newer same-major NCCL runtime")
			}
		}
	}
	return nil
}

func validateProviderIDFields(providerID string, revision uint32, providerRelease string) error {
	prefixless := strings.TrimPrefix(providerID, "nccl-")
	releaseEnd := strings.IndexByte(prefixless, '-')
	revisionStart := strings.LastIndex(providerID, "+coldsnap.")
	if releaseEnd <= 0 || revisionStart < 0 {
		return errors.New("provider_id is invalid")
	}
	parsedRevision, err := strconv.ParseUint(providerID[revisionStart+len("+coldsnap."):], 10, 32)
	if err != nil || uint32(parsedRevision) != revision {
		return errors.New("provider_revision disagrees with provider_id")
	}
	release := prefixless[:releaseEnd]
	if providerRelease != "" && providerRelease != release {
		return errors.New("provider NCCL release disagrees with provider_id")
	}
	return nil
}

func (manifest Manifest) effectiveProviderNCCLRuntime() (ProviderNCCLRuntimeIdentity, error) {
	if manifest.ProviderNCCLRuntime.Version == 0 {
		return ProviderNCCLRuntimeIdentity{}, errors.New("provider_nccl_runtime is required")
	}
	return manifest.ProviderNCCLRuntime, nil
}

func validatePlatformKeyFields(platformKey string, requirements PlatformRequirements) error {
	parts := platformParts.FindStringSubmatch(platformKey)
	if len(parts) != 4 {
		return errors.New("platform_key is invalid")
	}
	cuda := strings.TrimSuffix(requirements.CUDAUserspace, ".0")
	if parts[1] != requirements.Architecture || parts[2] != cuda || parts[3] != requirements.GlibcMin {
		return errors.New("platform_key disagrees with requirements")
	}
	return nil
}

func (runtime NCCLRuntimeIdentity) identityKey() string {
	data, _ := json.Marshal(runtime)
	return string(data)
}

func (requirements PlatformRequirements) validate() error {
	if requirements.Architecture != "aarch64" && requirements.Architecture != "x86_64" {
		return errors.New("architecture must be aarch64 or x86_64")
	}
	for name, value := range map[string]string{
		"cuda_userspace":    requirements.CUDAUserspace,
		"glibc_min":         requirements.GlibcMin,
		"libstdcxx_abi_min": requirements.LibstdcxxABIMin,
	} {
		if _, err := numericVersion(value); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	if !sonamePattern.MatchString(requirements.CUDARuntimeSONAME) {
		return errors.New("cuda_runtime_soname is invalid")
	}
	return nil
}

func (file ProviderFile) validate() error {
	if file.Path == "" || filepath.IsAbs(file.Path) || filepath.Clean(file.Path) != file.Path || file.Path == "." || file.Path == ".." || strings.HasPrefix(file.Path, "../") || strings.Contains(file.Path, `\`) {
		return errors.New("path must be a clean relative slash path")
	}
	if file.Path == ManifestFilename {
		return errors.New("provider manifest cannot list itself")
	}
	if !capabilityPattern.MatchString(file.Role) {
		return errors.New("role is invalid")
	}
	mode, err := strconv.ParseUint(file.Mode, 8, 32)
	if err != nil || len(file.Mode) != 4 || mode&^0o777 != 0 {
		return errors.New("mode must be four octal permission digits")
	}
	if mode&0o022 != 0 {
		return errors.New("mode must not be group/world writable")
	}
	if file.Size <= 0 {
		return errors.New("size must be positive")
	}
	if err := validateDigest("sha256", file.SHA256); err != nil {
		return err
	}
	if file.Role == RoleCheckpointShim || file.Role == RoleNCCLRuntime {
		if !hexBuildIDPattern.MatchString(file.BuildID) || !sonamePattern.MatchString(file.SONAME) {
			return errors.New("loadable payload requires valid build_id and soname")
		}
		if mode&0o111 == 0 {
			return errors.New("loadable payload must be executable")
		}
	} else if file.BuildID != "" || file.SONAME != "" {
		return errors.New("non-loadable payload must not declare ELF identity")
	}
	return nil
}

func (qualification Qualification) validate() error {
	if qualification.State != "accepted" {
		return errors.New("state must be accepted")
	}
	if !capabilityPattern.MatchString(qualification.Policy) {
		return errors.New("policy is invalid")
	}
	if err := validateSortedUniqueStrings("capabilities", qualification.Capabilities, capabilityPattern, false); err != nil {
		return err
	}
	if err := validateSortedUniqueStrings("transports", qualification.Transports, capabilityPattern, false); err != nil {
		return err
	}
	return validateSortedUniqueStrings("checks", qualification.Checks, capabilityPattern, false)
}

func validateSortedUniqueStrings(name string, values []string, pattern *regexp.Regexp, allowEmpty bool) error {
	if values == nil || (!allowEmpty && len(values) == 0) {
		return fmt.Errorf("%s must not be empty", name)
	}
	for index, value := range values {
		if !pattern.MatchString(value) {
			return fmt.Errorf("%s[%d] is invalid", name, index)
		}
		if index > 0 && values[index-1] >= value {
			return fmt.Errorf("%s must be sorted and unique", name)
		}
	}
	return nil
}

func validateDigest(name, value string) error {
	if !hexDigestPattern.MatchString(value) {
		return fmt.Errorf("%s must be a lowercase SHA-256", name)
	}
	return nil
}

func numericVersion(value string) ([]uint64, error) {
	parts := strings.Split(value, ".")
	if value == "" || len(parts) > 4 {
		return nil, errors.New("must be a numeric dotted version")
	}
	result := make([]uint64, len(parts))
	for index, part := range parts {
		if part == "" || (len(part) > 1 && part[0] == '0') {
			return nil, errors.New("must be a canonical numeric dotted version")
		}
		parsed, err := strconv.ParseUint(part, 10, 32)
		if err != nil {
			return nil, errors.New("must be a numeric dotted version")
		}
		result[index] = parsed
	}
	return result, nil
}

// Validate checks a target observation independently from any provider.
func (target Target) Validate() error {
	if target.Format != TargetFormat || target.Kind != TargetKind {
		return fmt.Errorf("expected format=%d kind=%q", TargetFormat, TargetKind)
	}
	if !strings.HasPrefix(target.BaseImageDigest, "sha256:") || !hexDigestPattern.MatchString(strings.TrimPrefix(target.BaseImageDigest, "sha256:")) {
		return errors.New("base_image_digest must be a sha256 digest")
	}
	if !platformKeyPattern.MatchString(target.PlatformKey) {
		return errors.New("platform_key is invalid")
	}
	if err := target.NCCL.validate(); err != nil {
		return fmt.Errorf("nccl: %w", err)
	}
	if target.NCCLPolicy != NCCLPolicyExact && target.NCCLPolicy != NCCLPolicyMatchOrLatestQualified {
		return errors.New("nccl_policy is invalid")
	}
	if err := target.Requirements.validate(); err != nil {
		return fmt.Errorf("requirements: %w", err)
	}
	if err := validatePlatformKeyFields(target.PlatformKey, target.Requirements); err != nil {
		return err
	}
	if target.RequiredProviderABI.Major == 0 || target.RequiredDLSymBridgeABI == 0 {
		return errors.New("required provider and dlsym bridge ABIs must be positive")
	}
	if err := validateSortedUniqueStrings("required_capabilities", target.RequiredCapabilities, capabilityPattern, false); err != nil {
		return err
	}
	if !capabilityPattern.MatchString(target.QualificationPolicy) || !capabilityPattern.MatchString(target.Transport) {
		return errors.New("qualification_policy and transport are invalid")
	}
	return nil
}

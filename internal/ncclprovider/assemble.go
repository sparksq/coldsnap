// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package ncclprovider

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

const (
	recipeFormat = 2
	recipeKind   = "coldsnap-nccl-provider-recipe"
	qualifyKind  = "coldsnap-nccl-provider-qualification-source"
)

// AssemblyOptions are the locked inputs used to turn build outputs into one
// immutable provider directory. SourceRoot resolves repository-relative paths
// declared by the recipe; only declared files are copied to Output.
type AssemblyOptions struct {
	SourceRoot        string
	RecipePath        string
	QualificationPath string
	PayloadRoot       string
	TargetPath        string
	Output            string
}

type providerRecipe struct {
	Format           int              `json:"format"`
	Kind             string           `json:"kind"`
	ProviderID       string           `json:"provider_id"`
	ProviderRevision uint32           `json:"provider_revision"`
	NCCLRelease      string           `json:"nccl_release"`
	NCCLVersionCode  int              `json:"nccl_version_code"`
	ProviderABI      ABIVersion       `json:"provider_abi"`
	CheckpointABI    uint32           `json:"checkpoint_abi"`
	DLSymBridgeABI   uint32           `json:"dlsym_bridge_abi"`
	Capabilities     []string         `json:"capabilities"`
	Limitations      []string         `json:"limitations"`
	Builder          recipeBuilder    `json:"builder"`
	Inputs           recipeInputs     `json:"inputs"`
	Build            recipeBuild      `json:"build"`
	Outputs          []providerOutput `json:"outputs"`
}

type recipeBuilder struct {
	ProviderDockerfile string   `json:"provider_dockerfile"`
	PayloadDockerfile  string   `json:"payload_dockerfile"`
	PayloadRepository  string   `json:"payload_repository"`
	PayloadBuildImage  string   `json:"payload_build_image"`
	PayloadPlatforms   []string `json:"payload_platforms"`
	BaseImagePolicy    string   `json:"base_image_policy"`
	MutableTagsAllowed bool     `json:"mutable_tags_allowed"`
}

type recipeInputs struct {
	SourceLock     string        `json:"source_lock"`
	SourceDiff     string        `json:"source_diff,omitempty"`
	ProviderHeader string        `json:"provider_header"`
	ProviderSource string        `json:"provider_source"`
	Patches        []recipePatch `json:"patches"`
}

type recipePatch struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type recipeBuild struct {
	NCCLBuildJobsArgument          string `json:"nccl_build_jobs_argument"`
	NVCCGencodeArgument            string `json:"nvcc_gencode_argument"`
	NVCCGencode                    string `json:"nvcc_gencode"`
	UseReproducibleNVCC            *bool  `json:"use_reproducible_nvcc"`
	StripUnneeded                  *bool  `json:"strip_unneeded"`
	VerifyCompiledAndLoadedVersion bool   `json:"verify_compiled_and_loaded_version"`
}

type providerOutput struct {
	Path   string `json:"path"`
	Role   string `json:"role"`
	SONAME string `json:"soname"`
}

type qualificationSource struct {
	Format       int      `json:"format"`
	Kind         string   `json:"kind"`
	ProviderID   string   `json:"provider_id"`
	State        string   `json:"state"`
	Policy       string   `json:"policy"`
	Capabilities []string `json:"capabilities"`
	Transports   []string `json:"transports"`
	Checks       []string `json:"checks"`
}

type providerBuildRecord struct {
	Format            int                         `json:"format"`
	Kind              string                      `json:"kind"`
	ProviderID        string                      `json:"provider_id"`
	PlatformKey       string                      `json:"platform_key"`
	BaseImageDigest   string                      `json:"base_image_digest"`
	BaseNCCL          NCCLRuntimeIdentity         `json:"base_nccl"`
	ProviderNCCL      ProviderNCCLRuntimeIdentity `json:"provider_nccl"`
	ProviderSelection string                      `json:"provider_selection"`
	RecipeSHA256      string                      `json:"recipe_sha256"`
	Payloads          []providerBuildPayload      `json:"payloads"`
}

type providerBuildPayload struct {
	Path    string `json:"path"`
	Role    string `json:"role"`
	Size    int64  `json:"size"`
	SHA256  string `json:"sha256"`
	BuildID string `json:"build_id"`
	SONAME  string `json:"soname"`
}

// Assemble verifies a release recipe, its capability-admission record, the
// target image observation, and the compiled ELF properties before atomically
// publishing one provider directory. Payload hashes are derived from this
// build for integrity; they are not compared with historical compiler output.
func Assemble(options AssemblyOptions) (Verification, error) {
	sourceRoot, err := realDirectory(options.SourceRoot, "source root")
	if err != nil {
		return Verification{}, err
	}
	payloadRoot, err := realDirectory(options.PayloadRoot, "payload root")
	if err != nil {
		return Verification{}, err
	}
	recipeData, err := readBoundedRegular(options.RecipePath, maxJSONBytes)
	if err != nil {
		return Verification{}, fmt.Errorf("read NCCL provider recipe: %w", err)
	}
	var recipe providerRecipe
	if err := decodeStrict(recipeData, &recipe); err != nil {
		return Verification{}, fmt.Errorf("decode NCCL provider recipe: %w", err)
	}
	qualificationData, err := readBoundedRegular(options.QualificationPath, maxJSONBytes)
	if err != nil {
		return Verification{}, fmt.Errorf("read NCCL provider qualification: %w", err)
	}
	var qualification qualificationSource
	if err := decodeStrict(qualificationData, &qualification); err != nil {
		return Verification{}, fmt.Errorf("decode NCCL provider qualification: %w", err)
	}
	target, err := LoadTarget(options.TargetPath)
	if err != nil {
		return Verification{}, err
	}
	if err := validateAssemblyMetadata(recipe, qualification, target); err != nil {
		return Verification{}, err
	}
	if err := validateRecipeInputs(sourceRoot, recipe); err != nil {
		return Verification{}, err
	}

	absoluteOutput, err := filepath.Abs(options.Output)
	if err != nil {
		return Verification{}, fmt.Errorf("resolve provider output: %w", err)
	}
	if _, err := os.Lstat(absoluteOutput); !errors.Is(err, os.ErrNotExist) {
		if err == nil {
			return Verification{}, errors.New("provider output already exists")
		}
		return Verification{}, fmt.Errorf("stat provider output: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(absoluteOutput), 0o755); err != nil {
		return Verification{}, fmt.Errorf("create provider output parent: %w", err)
	}
	temporary, err := os.MkdirTemp(filepath.Dir(absoluteOutput), ".nccl-provider-assembly-")
	if err != nil {
		return Verification{}, fmt.Errorf("create provider assembly directory: %w", err)
	}
	complete := false
	defer func() {
		if !complete {
			_ = os.RemoveAll(temporary)
		}
	}()

	files := make([]ProviderFile, 0, len(recipe.Outputs)+8)
	payloads := make([]providerBuildPayload, 0, len(recipe.Outputs))
	for _, expected := range recipe.Outputs {
		source, err := confinedPath(payloadRoot, expected.Path)
		if err != nil {
			return Verification{}, fmt.Errorf("expected output %q: %w", expected.Path, err)
		}
		info, err := os.Lstat(source)
		if err != nil || !info.Mode().IsRegular() {
			return Verification{}, fmt.Errorf("expected output %q is not a regular file", expected.Path)
		}
		digest, err := hashFile(source)
		if err != nil {
			return Verification{}, err
		}
		identity, err := inspectELF(source)
		if err != nil {
			return Verification{}, fmt.Errorf("inspect expected output %q: %w", expected.Path, err)
		}
		if identity.SONAME != expected.SONAME {
			return Verification{}, fmt.Errorf("provider output %q SONAME is %q, want %q", expected.Path, identity.SONAME, expected.SONAME)
		}
		if identity.Architecture != target.Requirements.Architecture {
			return Verification{}, fmt.Errorf("expected output %q architecture is %q, want %q", expected.Path, identity.Architecture, target.Requirements.Architecture)
		}
		if err := copyRegular(source, filepath.Join(temporary, filepath.FromSlash(expected.Path)), 0o755); err != nil {
			return Verification{}, err
		}
		files = append(files, ProviderFile{Path: expected.Path, Role: expected.Role, Mode: "0755", Size: info.Size(), SHA256: digest, BuildID: identity.BuildID, SONAME: identity.SONAME})
		payloads = append(payloads, providerBuildPayload{Path: expected.Path, Role: expected.Role, Size: info.Size(), SHA256: digest, BuildID: identity.BuildID, SONAME: identity.SONAME})
	}

	recipeDigest := digestBytes(recipeData)
	provenanceInputs := []struct {
		source, path, role string
		data               []byte
	}{
		{filepath.Join(sourceRoot, "LICENSE"), "licenses/coldsnap/LICENSE", RoleLicense, nil},
		{filepath.Join(sourceRoot, "THIRD_PARTY_NOTICES.md"), "licenses/coldsnap/THIRD_PARTY_NOTICES.md", RoleThirdParty, nil},
		{filepath.Join(sourceRoot, "third_party/licenses/nccl/LICENSE.txt"), "licenses/third-party/nccl/LICENSE.txt", RoleNCCLLicense, nil},
		{options.RecipePath, "provenance/recipe.json", "recipe", recipeData},
		{options.QualificationPath, "provenance/qualification.json", "qualification-policy", qualificationData},
	}
	sourceLock, err := confinedPath(sourceRoot, recipe.Inputs.SourceLock)
	if err != nil {
		return Verification{}, err
	}
	seriesPath := filepath.Join(filepath.Dir(filepath.Join(sourceRoot, filepath.FromSlash(recipe.Inputs.Patches[0].Path))), "series")
	provenanceInputs = append(provenanceInputs,
		struct {
			source, path, role string
			data               []byte
		}{sourceLock, "provenance/source.lock", "source-lock", nil},
		struct {
			source, path, role string
			data               []byte
		}{seriesPath, "provenance/patch-series", "patch-series", nil},
	)
	roleDigest := make(map[string]string)
	for _, input := range provenanceInputs {
		data := input.data
		if data == nil {
			data, err = readBoundedRegular(input.source, maxJSONBytes)
			if err != nil {
				return Verification{}, fmt.Errorf("read %s: %w", input.role, err)
			}
		}
		if err := writeRegular(filepath.Join(temporary, filepath.FromSlash(input.path)), data, 0o644); err != nil {
			return Verification{}, err
		}
		digest := digestBytes(data)
		roleDigest[input.role] = digest
		files = append(files, ProviderFile{Path: input.path, Role: input.role, Mode: "0644", Size: int64(len(data)), SHA256: digest})
	}

	slices.SortFunc(payloads, func(left, right providerBuildPayload) int { return strings.Compare(left.Path, right.Path) })
	providerRuntime, err := providerRuntimeIdentity(recipe, files)
	if err != nil {
		return Verification{}, err
	}
	providerSelection, err := providerSelectionFor(recipe, target)
	if err != nil {
		return Verification{}, err
	}
	buildRecord := providerBuildRecord{
		Format: 1, Kind: "coldsnap-nccl-provider-build-record", ProviderID: recipe.ProviderID,
		PlatformKey: target.PlatformKey, BaseImageDigest: target.BaseImageDigest,
		BaseNCCL: target.NCCL, ProviderNCCL: providerRuntime, ProviderSelection: providerSelection,
		RecipeSHA256: recipeDigest, Payloads: payloads,
	}
	buildRecordData, err := marshalCanonical(buildRecord)
	if err != nil {
		return Verification{}, err
	}
	buildRecordPath := "provenance/build-record.json"
	if err := writeRegular(filepath.Join(temporary, filepath.FromSlash(buildRecordPath)), buildRecordData, 0o644); err != nil {
		return Verification{}, err
	}
	buildRecordDigest := digestBytes(buildRecordData)
	files = append(files, ProviderFile{Path: buildRecordPath, Role: "build-record", Mode: "0644", Size: int64(len(buildRecordData)), SHA256: buildRecordDigest})
	slices.SortFunc(files, func(left, right ProviderFile) int { return strings.Compare(left.Path, right.Path) })

	manifest := Manifest{
		Format: ManifestFormat, Kind: ManifestKind, ProviderID: recipe.ProviderID,
		ProviderRevision: recipe.ProviderRevision, PlatformKey: target.PlatformKey,
		ProviderNCCLRuntime: providerRuntime, ProviderSelection: providerSelection,
		SupportedNCCLRuntimes: []NCCLRuntimeIdentity{target.NCCL}, ProviderABI: recipe.ProviderABI,
		CheckpointABI: recipe.CheckpointABI, Capabilities: recipe.Capabilities, Limitations: recipe.Limitations,
		Requirements: target.Requirements, Files: files,
		Provenance:     Provenance{SourceLockSHA256: roleDigest["source-lock"], RecipeSHA256: recipeDigest, PatchSeriesSHA256: roleDigest["patch-series"], BuildRecordSHA256: buildRecordDigest},
		Qualification:  Qualification{State: qualification.State, Policy: qualification.Policy, Capabilities: qualification.Capabilities, Transports: qualification.Transports, Checks: qualification.Checks},
		DLSymBridgeABI: recipe.DLSymBridgeABI,
		PreloadOrder:   []string{RoleDLSymBridge, RoleCheckpointShim, RoleNCCLRuntime},
	}
	manifestData, err := marshalCanonical(manifest)
	if err != nil {
		return Verification{}, err
	}
	if err := writeRegular(filepath.Join(temporary, ManifestFilename), manifestData, 0o644); err != nil {
		return Verification{}, err
	}
	verification, err := Verify(temporary)
	if err != nil {
		return Verification{}, fmt.Errorf("verify assembled provider: %w", err)
	}
	if err := os.Rename(temporary, absoluteOutput); err != nil {
		return Verification{}, fmt.Errorf("publish assembled provider: %w", err)
	}
	complete = true
	verification.Root = absoluteOutput
	return verification, nil
}

func validateAssemblyMetadata(recipe providerRecipe, qualification qualificationSource, target Target) error {
	if recipe.Format != recipeFormat || recipe.Kind != recipeKind || qualification.Format != recipeFormat || qualification.Kind != qualifyKind {
		return errors.New("NCCL provider recipe or qualification format is invalid")
	}
	if qualification.ProviderID != recipe.ProviderID || qualification.State != "accepted" || qualification.Policy != target.QualificationPolicy || !slices.Equal(qualification.Capabilities, recipe.Capabilities) {
		return errors.New("NCCL provider qualification does not admit this recipe and target policy")
	}
	if err := validateSortedUniqueStrings("qualification capabilities", qualification.Capabilities, capabilityPattern, false); err != nil {
		return err
	}
	if err := validateSortedUniqueStrings("qualification transports", qualification.Transports, capabilityPattern, false); err != nil {
		return err
	}
	if err := validateSortedUniqueStrings("qualification checks", qualification.Checks, capabilityPattern, false); err != nil {
		return err
	}
	if !slices.Contains(qualification.Transports, target.Transport) {
		return fmt.Errorf("NCCL provider is not qualified for transport %q", target.Transport)
	}
	if recipe.ProviderRevision == 0 || recipe.ProviderABI.Major == 0 || recipe.CheckpointABI == 0 || recipe.DLSymBridgeABI == 0 {
		return errors.New("NCCL provider recipe identity is incomplete")
	}
	if _, err := providerSelectionFor(recipe, target); err != nil {
		return err
	}
	if recipe.ProviderABI.Major != target.RequiredProviderABI.Major || recipe.ProviderABI.Minor < target.RequiredProviderABI.Minor || recipe.DLSymBridgeABI != target.RequiredDLSymBridgeABI {
		return errors.New("NCCL provider recipe does not satisfy target ABI requirements")
	}
	if err := validateProviderIDFields(recipe.ProviderID, recipe.ProviderRevision, recipe.NCCLRelease); err != nil {
		return err
	}
	if err := validateSortedUniqueStrings("capabilities", recipe.Capabilities, capabilityPattern, false); err != nil {
		return err
	}
	if err := validateSortedUniqueStrings("limitations", recipe.Limitations, capabilityPattern, true); err != nil {
		return err
	}
	for _, capability := range target.RequiredCapabilities {
		if !slices.Contains(recipe.Capabilities, capability) {
			return fmt.Errorf("NCCL provider recipe lacks target capability %q", capability)
		}
	}
	buildImageParts := strings.Split(recipe.Builder.PayloadBuildImage, "@sha256:")
	validBuildImage := len(buildImageParts) == 2 && buildImageParts[0] != "" && hexDigestPattern.MatchString(buildImageParts[1])
	validRepository := ociRepositoryPattern.MatchString(recipe.Builder.PayloadRepository)
	validBuilderPaths := recipe.Builder.ProviderDockerfile != "" && recipe.Builder.PayloadDockerfile != "" &&
		!filepath.IsAbs(recipe.Builder.ProviderDockerfile) && !filepath.IsAbs(recipe.Builder.PayloadDockerfile) &&
		filepath.Clean(recipe.Builder.ProviderDockerfile) == recipe.Builder.ProviderDockerfile &&
		filepath.Clean(recipe.Builder.PayloadDockerfile) == recipe.Builder.PayloadDockerfile
	validPlatforms := slices.Equal(recipe.Builder.PayloadPlatforms, []string{"linux/amd64", "linux/arm64"})
	if len(recipe.Outputs) != 2 || len(recipe.Inputs.Patches) == 0 || recipe.Build.UseReproducibleNVCC == nil || recipe.Build.StripUnneeded == nil || !nvccGencodePattern.MatchString(recipe.Build.NVCCGencode) || !recipe.Build.VerifyCompiledAndLoadedVersion || recipe.Builder.MutableTagsAllowed || recipe.Builder.BaseImagePolicy != "digest-pinned-multiarch-payload-and-target-images" || !validBuildImage || !validRepository || !validBuilderPaths || !validPlatforms {
		return errors.New("NCCL provider recipe build policy is incomplete")
	}
	seenRoles := map[string]bool{}
	for _, output := range recipe.Outputs {
		if output.Role != RoleNCCLRuntime && output.Role != RoleCheckpointShim {
			return fmt.Errorf("unexpected provider output role %q", output.Role)
		}
		if seenRoles[output.Role] {
			return fmt.Errorf("duplicated provider output role %q", output.Role)
		}
		seenRoles[output.Role] = true
		if output.Path == "" || filepath.IsAbs(output.Path) || filepath.Clean(output.Path) != output.Path || !sonamePattern.MatchString(output.SONAME) {
			return fmt.Errorf("provider output %q has an invalid contract", output.Path)
		}
	}
	return nil
}

func providerSelectionFor(recipe providerRecipe, target Target) (string, error) {
	providerVersionCode, err := ncclVersionCode(recipe.NCCLRelease)
	if err != nil || providerVersionCode != recipe.NCCLVersionCode {
		return "", errors.New("NCCL provider recipe version disagrees with its release")
	}
	if recipe.NCCLVersionCode == target.NCCL.Version && recipe.NCCLRelease == target.NCCL.Release {
		return ProviderSelectionExact, nil
	}
	if target.NCCLPolicy != NCCLPolicyMatchOrLatestQualified {
		return "", errors.New("NCCL provider recipe identity disagrees with the exact target")
	}
	providerVersion, err := numericVersion(recipe.NCCLRelease)
	if err != nil || len(providerVersion) != 3 {
		return "", errors.New("NCCL provider recipe release is invalid")
	}
	baseVersion, err := numericVersion(target.NCCL.Release)
	if err != nil || len(baseVersion) != 3 {
		return "", errors.New("target NCCL release is invalid")
	}
	if providerVersion[0] != baseVersion[0] || recipe.NCCLVersionCode <= target.NCCL.Version {
		return "", errors.New("fallback NCCL provider must be a newer release in the same major series")
	}
	return ProviderSelectionFallbackUpgrade, nil
}

func providerRuntimeIdentity(recipe providerRecipe, files []ProviderFile) (ProviderNCCLRuntimeIdentity, error) {
	for _, file := range files {
		if file.Role == RoleNCCLRuntime {
			return ProviderNCCLRuntimeIdentity{
				Version: recipe.NCCLVersionCode, Release: recipe.NCCLRelease,
				SONAME: file.SONAME, BuildID: file.BuildID, SHA256: file.SHA256,
			}, nil
		}
	}
	return ProviderNCCLRuntimeIdentity{}, errors.New("provider files lack the NCCL runtime payload")
}

func validateRecipeInputs(sourceRoot string, recipe providerRecipe) error {
	seriesDirectory := ""
	seriesNames := make([]string, 0, len(recipe.Inputs.Patches))
	for _, patch := range recipe.Inputs.Patches {
		path, err := confinedPath(sourceRoot, patch.Path)
		if err != nil {
			return fmt.Errorf("recipe patch %q: %w", patch.Path, err)
		}
		digest, err := hashFile(path)
		if err != nil || digest != patch.SHA256 {
			return fmt.Errorf("recipe patch %q digest mismatch", patch.Path)
		}
		if seriesDirectory == "" {
			seriesDirectory = filepath.Dir(path)
		} else if filepath.Dir(path) != seriesDirectory {
			return errors.New("recipe patches do not share one release-owned directory")
		}
		seriesNames = append(seriesNames, filepath.Base(path))
	}
	seriesData, err := readBoundedRegular(filepath.Join(seriesDirectory, "series"), maxJSONBytes)
	if err != nil {
		return fmt.Errorf("read patch series: %w", err)
	}
	seriesLines := strings.Split(strings.TrimSpace(string(seriesData)), "\n")
	if !slices.Equal(seriesLines, seriesNames) {
		return errors.New("patch series order disagrees with recipe")
	}
	for _, path := range []string{recipe.Builder.ProviderDockerfile, recipe.Builder.PayloadDockerfile, recipe.Inputs.SourceLock, recipe.Inputs.ProviderHeader, recipe.Inputs.ProviderSource} {
		resolved, err := confinedPath(sourceRoot, path)
		if err != nil {
			return err
		}
		if _, err := readBoundedRegular(resolved, maxJSONBytes); err != nil {
			return fmt.Errorf("read recipe input %q: %w", path, err)
		}
	}
	return nil
}

func realDirectory(path, name string) (string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve %s: %w", name, err)
	}
	info, err := os.Lstat(absolute)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("%s must be a real directory", name)
	}
	return absolute, nil
}

func confinedPath(root, relative string) (string, error) {
	if relative == "" || filepath.IsAbs(relative) || filepath.Clean(relative) != relative || relative == "." || relative == ".." || strings.HasPrefix(relative, "../") || strings.Contains(relative, `\`) {
		return "", errors.New("path must be a clean relative path")
	}
	return filepath.Join(root, filepath.FromSlash(relative)), nil
}

func marshalCanonical(value any) ([]byte, error) {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

func writeRegular(path string, data []byte, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, mode)
	if err != nil {
		return err
	}
	success := false
	defer func() {
		_ = file.Close()
		if !success {
			_ = os.Remove(path)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return err
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	success = true
	return nil
}

func digestBytes(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package ncclprovider

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// Selection is the machine-readable exact-selection result.
type Selection struct {
	ProviderID     string `json:"provider_id"`
	PlatformKey    string `json:"platform_key"`
	ManifestSHA256 string `json:"manifest_sha256"`
	SourcePath     string `json:"source_path"`
}

// NoMatchError includes deterministic rejection explanations for operators.
type NoMatchError struct {
	TargetPlatform string   `json:"target_platform"`
	Rejections     []string `json:"rejections"`
}

func (err NoMatchError) Error() string {
	return fmt.Sprintf("no exact NCCL provider for platform %q: %s", err.TargetPlatform, strings.Join(err.Rejections, "; "))
}

// Select validates an entire local catalog and returns exactly one provider.
func Select(catalog string, target Target) (Selection, error) {
	if err := target.Validate(); err != nil {
		return Selection{}, err
	}
	roots, err := catalogProviderRoots(catalog)
	if err != nil {
		return Selection{}, err
	}
	seen := make(map[string]string)
	var matches []Selection
	var rejections []string
	for _, root := range roots {
		verification, err := Verify(root)
		if err != nil {
			return Selection{}, fmt.Errorf("catalog provider %q is invalid: %w", root, err)
		}
		key := verification.ProviderID + "\x00" + verification.PlatformKey
		if previous, ok := seen[key]; ok {
			return Selection{}, fmt.Errorf("duplicate provider ID/platform %q/%q at %q and %q", verification.ProviderID, verification.PlatformKey, previous, root)
		}
		seen[key] = root
		loaded, err := LoadManifest(filepath.Join(root, ManifestFilename))
		if err != nil {
			return Selection{}, err
		}
		if reason := mismatchReason(loaded.Manifest, target); reason != "" {
			rejections = append(rejections, loaded.Manifest.ProviderID+"/"+loaded.Manifest.PlatformKey+": "+reason)
			continue
		}
		matches = append(matches, Selection{
			ProviderID: verification.ProviderID, PlatformKey: verification.PlatformKey,
			ManifestSHA256: verification.ManifestSHA256, SourcePath: verification.Root,
		})
	}
	slices.Sort(rejections)
	if len(matches) == 0 {
		if len(rejections) == 0 {
			rejections = []string{"catalog contains no providers"}
		}
		return Selection{}, NoMatchError{TargetPlatform: target.PlatformKey, Rejections: rejections}
	}
	if len(matches) != 1 {
		slices.SortFunc(matches, func(left, right Selection) int {
			return strings.Compare(left.ProviderID+"/"+left.PlatformKey, right.ProviderID+"/"+right.PlatformKey)
		})
		identities := make([]string, len(matches))
		for index, match := range matches {
			identities[index] = match.ProviderID + "/" + match.PlatformKey
		}
		return Selection{}, fmt.Errorf("multiple exact NCCL providers match target: %s", strings.Join(identities, ", "))
	}
	return matches[0], nil
}

func catalogProviderRoots(catalog string) ([]string, error) {
	absolute, err := filepath.Abs(catalog)
	if err != nil {
		return nil, fmt.Errorf("resolve catalog: %w", err)
	}
	providers := filepath.Join(absolute, "providers")
	providerEntries, err := os.ReadDir(providers)
	if err != nil {
		return nil, fmt.Errorf("read catalog providers: %w", err)
	}
	var roots []string
	for _, providerEntry := range providerEntries {
		if providerEntry.Type()&os.ModeSymlink != 0 || !providerEntry.IsDir() || !providerIDPattern.MatchString(providerEntry.Name()) {
			return nil, fmt.Errorf("catalog contains invalid provider entry %q", providerEntry.Name())
		}
		providerPath := filepath.Join(providers, providerEntry.Name())
		platformEntries, err := os.ReadDir(providerPath)
		if err != nil {
			return nil, err
		}
		if len(platformEntries) == 0 {
			return nil, fmt.Errorf("provider %q has no platforms", providerEntry.Name())
		}
		for _, platformEntry := range platformEntries {
			if platformEntry.Type()&os.ModeSymlink != 0 || !platformEntry.IsDir() || !platformKeyPattern.MatchString(platformEntry.Name()) {
				return nil, fmt.Errorf("provider %q contains invalid platform entry %q", providerEntry.Name(), platformEntry.Name())
			}
			root := filepath.Join(providerPath, platformEntry.Name())
			loaded, err := LoadManifest(filepath.Join(root, ManifestFilename))
			if err != nil {
				return nil, err
			}
			if loaded.Manifest.ProviderID != providerEntry.Name() || loaded.Manifest.PlatformKey != platformEntry.Name() {
				return nil, fmt.Errorf("provider manifest identity does not match catalog path %q", root)
			}
			roots = append(roots, root)
		}
	}
	slices.Sort(roots)
	return roots, nil
}

func mismatchReason(manifest Manifest, target Target) string {
	if manifest.PlatformKey != target.PlatformKey {
		return "platform key mismatch"
	}
	if manifest.Requirements.Architecture != target.Requirements.Architecture ||
		manifest.Requirements.CUDAUserspace != target.Requirements.CUDAUserspace ||
		manifest.Requirements.CUDARuntimeSONAME != target.Requirements.CUDARuntimeSONAME {
		return "architecture or CUDA userspace mismatch"
	}
	for _, pair := range [][2]string{
		{target.Requirements.GlibcMin, manifest.Requirements.GlibcMin},
		{target.Requirements.LibstdcxxABIMin, manifest.Requirements.LibstdcxxABIMin},
	} {
		ok, err := versionAtLeast(pair[0], pair[1])
		if err != nil || !ok {
			return "libc or libstdc++ ABI floor mismatch"
		}
	}
	if manifest.ProviderABI.Major != target.RequiredProviderABI.Major || manifest.ProviderABI.Minor < target.RequiredProviderABI.Minor {
		return "provider ABI mismatch"
	}
	if manifest.DLSymBridgeABI != target.RequiredDLSymBridgeABI {
		return "dlsym bridge ABI mismatch"
	}
	if !containsAll(manifest.Capabilities, target.RequiredCapabilities) {
		return "required capability missing"
	}
	if manifest.Qualification.Policy != target.QualificationPolicy ||
		!slices.Contains(manifest.Qualification.Transports, target.Transport) ||
		!containsAll(manifest.Qualification.Capabilities, target.RequiredCapabilities) {
		return "qualification scope mismatch"
	}
	if target.NCCLPolicy == NCCLPolicyExact && manifest.ProviderSelection != ProviderSelectionExact {
		return "target requires exact NCCL provider selection"
	}
	for _, supported := range manifest.SupportedNCCLRuntimes {
		if compatibleNCCLRuntime(supported, target.NCCL) {
			return ""
		}
	}
	return "NCCL version or SONAME mismatch"
}

// compatibleNCCLRuntime intentionally ignores path, build ID, and file hash.
// The provider supplies and verifies its own immutable NCCL payload; the base
// runtime only needs the admitted release and SONAME contract. Exact target
// observations remain in the manifest/build record for diagnostics.
func compatibleNCCLRuntime(supported, target NCCLRuntimeIdentity) bool {
	return supported.Version == target.Version && supported.Release == target.Release &&
		supported.SONAME == target.SONAME
}

func containsAll(have, required []string) bool {
	for _, value := range required {
		if !slices.Contains(have, value) {
			return false
		}
	}
	return true
}

func versionAtLeast(actual, minimum string) (bool, error) {
	left, err := numericVersion(actual)
	if err != nil {
		return false, err
	}
	right, err := numericVersion(minimum)
	if err != nil {
		return false, err
	}
	length := max(len(left), len(right))
	for index := range length {
		var a, b uint64
		if index < len(left) {
			a = left[index]
		}
		if index < len(right) {
			b = right[index]
		}
		if a != b {
			return a > b, nil
		}
	}
	return true, nil
}

var errAlreadyMaterialized = errors.New("provider is already materialized")

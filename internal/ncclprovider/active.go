// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package ncclprovider

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
)

const (
	ActiveFormat                 = 2
	ActiveKind                   = "coldsnap-nccl-active-runtime"
	ActiveFilename               = "active.json"
	ActiveCheckpointShimFilename = "libcoldsnap-checkpoint-shim.so"
)

// ActiveFile is one resolved, manifest-bound provider payload.
type ActiveFile struct {
	Path    string `json:"path"`
	SHA256  string `json:"sha256"`
	BuildID string `json:"build_id"`
	SONAME  string `json:"soname"`
}

// ActiveBridge binds the separately built, platform-specific dlsym bridge.
type ActiveBridge struct {
	Path   string `json:"path"`
	ABI    uint32 `json:"abi"`
	SHA256 string `json:"sha256"`
}

// ActiveRecord resolves one verified provider to immutable in-image paths.
// Consumers still verify the referenced manifest/files; this record is not a
// replacement for the provider trust boundary.
type ActiveRecord struct {
	Format                int                         `json:"format"`
	Kind                  string                      `json:"kind"`
	ProviderID            string                      `json:"provider_id"`
	ProviderRevision      uint32                      `json:"provider_revision"`
	PlatformKey           string                      `json:"platform_key"`
	ProviderRoot          string                      `json:"provider_root"`
	ProviderManifest      string                      `json:"provider_manifest"`
	ManifestSHA256        string                      `json:"manifest_sha256"`
	ProviderABI           ABIVersion                  `json:"provider_abi"`
	CheckpointABI         uint32                      `json:"checkpoint_abi"`
	ProviderNCCLRuntime   ProviderNCCLRuntimeIdentity `json:"provider_nccl_runtime"`
	ProviderSelection     string                      `json:"provider_selection"`
	SupportedNCCLRuntimes []NCCLRuntimeIdentity       `json:"supported_nccl_runtimes"`
	Capabilities          []string                    `json:"capabilities"`
	Limitations           []string                    `json:"limitations"`
	Qualification         Qualification               `json:"qualification"`
	Files                 map[string]ActiveFile       `json:"files"`
	Bridge                ActiveBridge                `json:"bridge"`
	PreloadOrder          []string                    `json:"preload_order"`
}

// LoadActive strictly parses an installed active runtime record without
// loading any of its native libraries.
func LoadActive(path string) (ActiveRecord, error) {
	data, err := readBoundedRegular(path, maxJSONBytes)
	if err != nil {
		return ActiveRecord{}, fmt.Errorf("read active NCCL runtime: %w", err)
	}
	var record ActiveRecord
	if err := decodeStrict(data, &record); err != nil {
		return ActiveRecord{}, fmt.Errorf("decode active NCCL runtime: %w", err)
	}
	return record, nil
}

// VerifyActive revalidates the complete provider tree, manifest, common
// bridge, resolved paths, and qualification policy described by active.json.
// It is safe to run before starting a worker because it never loads provider
// code into the controller process.
func VerifyActive(path string) (ActiveRecord, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return ActiveRecord{}, fmt.Errorf("stat active NCCL runtime: %w", err)
	}
	if !info.Mode().IsRegular() || info.Mode().Perm() != 0o644 {
		return ActiveRecord{}, errors.New("active NCCL runtime must be a mode-0644 regular file")
	}
	if err := rejectHardlink(info); err != nil {
		return ActiveRecord{}, fmt.Errorf("active NCCL runtime: %w", err)
	}
	record, err := LoadActive(path)
	if err != nil {
		return ActiveRecord{}, err
	}
	if record.Format != ActiveFormat || record.Kind != ActiveKind || record.ProviderRevision == 0 ||
		record.ProviderABI.Major == 0 || record.CheckpointABI == 0 || record.Bridge.ABI == 0 {
		return ActiveRecord{}, errors.New("active NCCL runtime identity is invalid")
	}
	for _, value := range []string{record.ProviderRoot, record.ProviderManifest, record.Bridge.Path} {
		if !filepath.IsAbs(value) || filepath.Clean(value) != value {
			return ActiveRecord{}, errors.New("active NCCL runtime contains a non-canonical path")
		}
	}
	verification, err := Verify(record.ProviderRoot)
	if err != nil {
		return ActiveRecord{}, fmt.Errorf("verify active NCCL provider: %w", err)
	}
	if verification.ProviderID != record.ProviderID || verification.PlatformKey != record.PlatformKey ||
		verification.ManifestSHA256 != record.ManifestSHA256 ||
		record.ProviderManifest != filepath.Join(verification.Root, ManifestFilename) {
		return ActiveRecord{}, errors.New("active NCCL runtime disagrees with its provider manifest")
	}
	loaded, err := LoadManifest(record.ProviderManifest)
	if err != nil {
		return ActiveRecord{}, err
	}
	manifest := loaded.Manifest
	if manifest.ProviderRevision != record.ProviderRevision || manifest.ProviderABI != record.ProviderABI ||
		manifest.CheckpointABI != record.CheckpointABI || manifest.DLSymBridgeABI != record.Bridge.ABI ||
		manifest.ProviderNCCLRuntime != record.ProviderNCCLRuntime ||
		manifest.ProviderSelection != record.ProviderSelection ||
		!reflect.DeepEqual(manifest.SupportedNCCLRuntimes, record.SupportedNCCLRuntimes) ||
		!slices.Equal(manifest.Capabilities, record.Capabilities) || !slices.Equal(manifest.Limitations, record.Limitations) ||
		!reflect.DeepEqual(manifest.Qualification, record.Qualification) {
		return ActiveRecord{}, errors.New("active NCCL runtime metadata disagrees with its provider manifest")
	}
	expectedFiles := make(map[string]ActiveFile, 2)
	for _, file := range manifest.Files {
		if file.Role == RoleCheckpointShim || file.Role == RoleNCCLRuntime {
			resolvedPath := filepath.Join(verification.Root, filepath.FromSlash(file.Path))
			if file.Role == RoleCheckpointShim {
				resolvedPath = filepath.Join(filepath.Dir(record.Bridge.Path), ActiveCheckpointShimFilename)
			}
			expectedFiles[file.Role] = ActiveFile{
				Path:   resolvedPath,
				SHA256: file.SHA256, BuildID: file.BuildID, SONAME: file.SONAME,
			}
		}
	}
	if !reflect.DeepEqual(record.Files, expectedFiles) {
		return ActiveRecord{}, errors.New("active NCCL runtime file resolution disagrees with its provider manifest")
	}
	shim := record.Files[RoleCheckpointShim]
	shimInfo, err := os.Lstat(shim.Path)
	if err != nil {
		return ActiveRecord{}, fmt.Errorf("stat active checkpoint shim: %w", err)
	}
	if !shimInfo.Mode().IsRegular() || shimInfo.Mode().Perm() != 0o755 {
		return ActiveRecord{}, errors.New("active checkpoint shim must be a mode-0755 regular file")
	}
	if err := rejectHardlink(shimInfo); err != nil {
		return ActiveRecord{}, fmt.Errorf("active checkpoint shim: %w", err)
	}
	shimDigest, err := hashFile(shim.Path)
	if err != nil || shimDigest != shim.SHA256 {
		return ActiveRecord{}, errors.New("active checkpoint shim digest mismatch")
	}
	shimELF, err := inspectELF(shim.Path)
	if err != nil || shimELF.BuildID != shim.BuildID || shimELF.SONAME != shim.SONAME {
		return ActiveRecord{}, errors.New("active checkpoint shim ELF identity mismatch")
	}
	bridgeInfo, err := os.Lstat(record.Bridge.Path)
	if err != nil {
		return ActiveRecord{}, fmt.Errorf("stat active NCCL bridge: %w", err)
	}
	if !bridgeInfo.Mode().IsRegular() || bridgeInfo.Mode().Perm() != 0o755 {
		return ActiveRecord{}, errors.New("active NCCL bridge must be a mode-0755 regular file")
	}
	if err := rejectHardlink(bridgeInfo); err != nil {
		return ActiveRecord{}, fmt.Errorf("active NCCL bridge: %w", err)
	}
	bridgeDigest, err := hashFile(record.Bridge.Path)
	if err != nil || bridgeDigest != record.Bridge.SHA256 {
		return ActiveRecord{}, errors.New("active NCCL bridge digest mismatch")
	}
	expectedPreload := []string{
		record.Bridge.Path,
		record.Files[RoleCheckpointShim].Path,
		record.Files[RoleNCCLRuntime].Path,
	}
	if !slices.Equal(record.PreloadOrder, expectedPreload) {
		return ActiveRecord{}, errors.New("active NCCL preload order is invalid")
	}
	return record, nil
}

// InstallActive verifies and installs one materialized provider plus the
// common bridge below outputRoot, then atomically publishes active.json.
func InstallActive(provider, bridge, outputRoot string, bridgeABI uint32) (ActiveRecord, error) {
	if bridgeABI == 0 {
		return ActiveRecord{}, errors.New("dlsym bridge ABI must be positive")
	}
	verification, err := Verify(provider)
	if err != nil {
		return ActiveRecord{}, err
	}
	loaded, err := LoadManifest(filepath.Join(verification.Root, ManifestFilename))
	if err != nil {
		return ActiveRecord{}, err
	}
	if loaded.SHA256 != verification.ManifestSHA256 {
		return ActiveRecord{}, errors.New("provider changed while preparing active runtime")
	}
	absoluteOutput, err := filepath.Abs(outputRoot)
	if err != nil {
		return ActiveRecord{}, err
	}
	if !filepath.IsAbs(absoluteOutput) || filepath.Clean(absoluteOutput) != absoluteOutput {
		return ActiveRecord{}, errors.New("active runtime output must be an absolute clean path")
	}
	providerRoot := filepath.Join(absoluteOutput, "providers", verification.ProviderID, verification.PlatformKey)
	selection := Selection{
		ProviderID: verification.ProviderID, PlatformKey: verification.PlatformKey,
		ManifestSHA256: verification.ManifestSHA256, SourcePath: verification.Root,
	}
	if err := installVerifiedProvider(selection, loaded, providerRoot); err != nil {
		return ActiveRecord{}, err
	}
	bridgeInfo, err := os.Lstat(bridge)
	if err != nil {
		return ActiveRecord{}, fmt.Errorf("stat dlsym bridge: %w", err)
	}
	if !bridgeInfo.Mode().IsRegular() || bridgeInfo.Mode().Perm() != 0o755 {
		return ActiveRecord{}, errors.New("dlsym bridge must be a mode-0755 regular file")
	}
	if err := rejectHardlink(bridgeInfo); err != nil {
		return ActiveRecord{}, fmt.Errorf("dlsym bridge: %w", err)
	}
	bridgeDigest, err := hashFile(bridge)
	if err != nil {
		return ActiveRecord{}, err
	}
	bridgePath := filepath.Join(absoluteOutput, "common", "libcoldsnap-nccl-dlsym.so")
	if err := installExactFile(bridge, bridgePath, 0o755, bridgeDigest); err != nil {
		return ActiveRecord{}, err
	}
	files := make(map[string]ActiveFile)
	for _, file := range loaded.Manifest.Files {
		if file.Role != RoleNCCLRuntime && file.Role != RoleCheckpointShim {
			continue
		}
		resolvedPath := filepath.Join(providerRoot, filepath.FromSlash(file.Path))
		if file.Role == RoleCheckpointShim {
			resolvedPath = filepath.Join(absoluteOutput, "common", ActiveCheckpointShimFilename)
			if err := installExactFile(
				filepath.Join(providerRoot, filepath.FromSlash(file.Path)),
				resolvedPath,
				0o755,
				file.SHA256,
			); err != nil {
				return ActiveRecord{}, fmt.Errorf("install active checkpoint shim: %w", err)
			}
		}
		files[file.Role] = ActiveFile{
			Path: resolvedPath, SHA256: file.SHA256,
			BuildID: file.BuildID, SONAME: file.SONAME,
		}
	}
	preload := []string{
		bridgePath,
		files[RoleCheckpointShim].Path,
		files[RoleNCCLRuntime].Path,
	}
	record := ActiveRecord{
		Format: ActiveFormat, Kind: ActiveKind, ProviderID: loaded.Manifest.ProviderID,
		ProviderRevision: loaded.Manifest.ProviderRevision, PlatformKey: loaded.Manifest.PlatformKey,
		ProviderRoot: providerRoot, ProviderManifest: filepath.Join(providerRoot, ManifestFilename),
		ManifestSHA256: loaded.SHA256, ProviderABI: loaded.Manifest.ProviderABI,
		CheckpointABI:         loaded.Manifest.CheckpointABI,
		ProviderNCCLRuntime:   loaded.Manifest.ProviderNCCLRuntime,
		ProviderSelection:     loaded.Manifest.ProviderSelection,
		SupportedNCCLRuntimes: slices.Clone(loaded.Manifest.SupportedNCCLRuntimes),
		Capabilities:          slices.Clone(loaded.Manifest.Capabilities), Limitations: slices.Clone(loaded.Manifest.Limitations),
		Qualification: loaded.Manifest.Qualification, Files: files,
		Bridge:       ActiveBridge{Path: bridgePath, ABI: bridgeABI, SHA256: bridgeDigest},
		PreloadOrder: preload,
	}
	if err := publishActiveRecord(filepath.Join(absoluteOutput, ActiveFilename), record); err != nil {
		return ActiveRecord{}, err
	}
	return record, nil
}

func installVerifiedProvider(selection Selection, loaded LoadedManifest, destination string) error {
	if err := existingMaterialization(destination, selection); err != nil {
		if errors.Is(err, errAlreadyMaterialized) {
			return nil
		}
		return err
	}
	parent := filepath.Dir(destination)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return err
	}
	temporary, err := os.MkdirTemp(parent, ".provider-")
	if err != nil {
		return err
	}
	complete := false
	defer func() {
		if !complete {
			_ = os.RemoveAll(temporary)
		}
	}()
	if err := copyRegular(filepath.Join(selection.SourcePath, ManifestFilename), filepath.Join(temporary, ManifestFilename), 0o644); err != nil {
		return err
	}
	for _, file := range loaded.Manifest.Files {
		mode, _ := parseMode(file.Mode)
		if err := copyRegular(filepath.Join(selection.SourcePath, filepath.FromSlash(file.Path)), filepath.Join(temporary, filepath.FromSlash(file.Path)), mode); err != nil {
			return err
		}
	}
	verification, err := Verify(temporary)
	if err != nil {
		return fmt.Errorf("verify installed provider: %w", err)
	}
	if verification.ManifestSHA256 != selection.ManifestSHA256 {
		return errors.New("installed provider manifest digest does not match selection")
	}
	if err := os.Rename(temporary, destination); err != nil {
		return err
	}
	complete = true
	return nil
}

func parseMode(value string) (os.FileMode, error) {
	var mode uint32
	if _, err := fmt.Sscanf(value, "%o", &mode); err != nil {
		return 0, err
	}
	return os.FileMode(mode), nil
}

func installExactFile(source, destination string, mode os.FileMode, digest string) error {
	if info, err := os.Lstat(destination); err == nil {
		if !info.Mode().IsRegular() || info.Mode().Perm() != mode {
			return errors.New("active runtime destination is not exact")
		}
		actual, hashErr := hashFile(destination)
		if hashErr != nil || actual != digest {
			return errors.New("active runtime destination is not exact")
		}
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(destination), ".bridge-")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	if err := temporary.Close(); err != nil {
		return err
	}
	_ = os.Remove(temporaryPath)
	if err := copyRegular(source, temporaryPath, mode); err != nil {
		return err
	}
	installedDigest, err := hashFile(temporaryPath)
	if err != nil {
		return err
	}
	if installedDigest != digest {
		return errors.New("file changed while installing active runtime")
	}
	if err := os.Rename(temporaryPath, destination); err != nil {
		_ = os.Remove(temporaryPath)
		return err
	}
	return nil
}

func publishActiveRecord(path string, record ActiveRecord) error {
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if existing, err := os.ReadFile(path); err == nil {
		if bytes.Equal(existing, data) {
			return nil
		}
		return errors.New("active runtime record already exists with different content")
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".active-")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	success := false
	defer func() {
		_ = temporary.Close()
		if !success {
			_ = os.Remove(temporaryPath)
		}
	}()
	if err := temporary.Chmod(0o644); err != nil {
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		return err
	}
	if err := temporary.Sync(); err != nil {
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return err
	}
	success = true
	return nil
}

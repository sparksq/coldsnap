// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package ncclprovider

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"syscall"
)

// Verification is the immutable identity of a validated provider directory.
type Verification struct {
	ProviderID     string `json:"provider_id"`
	PlatformKey    string `json:"platform_key"`
	ManifestSHA256 string `json:"manifest_sha256"`
	Root           string `json:"root"`
}

// Verify validates the manifest and every filesystem/ELF invariant without
// loading provider code.
func Verify(root string) (Verification, error) {
	absolute, err := filepath.Abs(root)
	if err != nil {
		return Verification{}, fmt.Errorf("resolve provider root: %w", err)
	}
	rootInfo, err := os.Lstat(absolute)
	if err != nil {
		return Verification{}, fmt.Errorf("stat provider root: %w", err)
	}
	if !rootInfo.IsDir() || rootInfo.Mode()&os.ModeSymlink != 0 {
		return Verification{}, errors.New("provider root must be a real directory")
	}
	manifestPath := filepath.Join(absolute, ManifestFilename)
	manifestInfo, err := os.Lstat(manifestPath)
	if err != nil {
		return Verification{}, fmt.Errorf("stat provider manifest: %w", err)
	}
	if !manifestInfo.Mode().IsRegular() || manifestInfo.Mode().Perm() != 0o644 {
		return Verification{}, errors.New("provider manifest must be a mode-0644 regular file")
	}
	if err := rejectHardlink(manifestInfo); err != nil {
		return Verification{}, fmt.Errorf("provider manifest: %w", err)
	}
	loaded, err := LoadManifest(manifestPath)
	if err != nil {
		return Verification{}, err
	}
	declared := make(map[string]ProviderFile, len(loaded.Manifest.Files))
	for _, file := range loaded.Manifest.Files {
		declared[filepath.FromSlash(file.Path)] = file
	}
	seen := make(map[string]bool, len(declared))
	err = filepath.WalkDir(absolute, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if path == absolute {
			return nil
		}
		relative, err := filepath.Rel(absolute, path)
		if err != nil {
			return err
		}
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("provider contains symlink %q", filepath.ToSlash(relative))
		}
		if info.IsDir() {
			return nil
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("provider contains special file %q", filepath.ToSlash(relative))
		}
		if relative == ManifestFilename {
			return nil
		}
		file, ok := declared[relative]
		if !ok {
			return fmt.Errorf("provider contains undeclared file %q", filepath.ToSlash(relative))
		}
		if err := verifyFile(path, info, file, loaded.Manifest.Requirements.Architecture); err != nil {
			return fmt.Errorf("verify %q: %w", file.Path, err)
		}
		seen[relative] = true
		return nil
	})
	if err != nil {
		return Verification{}, err
	}
	for path := range declared {
		if !seen[path] {
			return Verification{}, fmt.Errorf("provider lacks declared file %q", filepath.ToSlash(path))
		}
	}
	return Verification{
		ProviderID: loaded.Manifest.ProviderID, PlatformKey: loaded.Manifest.PlatformKey,
		ManifestSHA256: loaded.SHA256, Root: absolute,
	}, nil
}

func verifyFile(path string, info os.FileInfo, declared ProviderFile, architecture string) error {
	if err := rejectHardlink(info); err != nil {
		return err
	}
	mode, _ := strconv.ParseUint(declared.Mode, 8, 32)
	if info.Mode().Perm() != os.FileMode(mode) {
		return fmt.Errorf("mode is %04o, want %s", info.Mode().Perm(), declared.Mode)
	}
	if info.Size() != declared.Size {
		return fmt.Errorf("size is %d, want %d", info.Size(), declared.Size)
	}
	digest, err := hashFile(path)
	if err != nil {
		return err
	}
	if digest != declared.SHA256 {
		return fmt.Errorf("SHA-256 is %s, want %s", digest, declared.SHA256)
	}
	if declared.Role == RoleCheckpointShim || declared.Role == RoleNCCLRuntime {
		identity, err := inspectELF(path)
		if err != nil {
			return fmt.Errorf("inspect ELF: %w", err)
		}
		if identity.Architecture != architecture {
			return fmt.Errorf("ELF architecture is %q, want %q", identity.Architecture, architecture)
		}
		if identity.BuildID != declared.BuildID {
			return fmt.Errorf("ELF build ID is %q, want %q", identity.BuildID, declared.BuildID)
		}
		if identity.SONAME != declared.SONAME {
			return fmt.Errorf("ELF SONAME is %q, want %q", identity.SONAME, declared.SONAME)
		}
	}
	return nil
}

func rejectHardlink(info os.FileInfo) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return errors.New("cannot inspect link count")
	}
	if stat.Nlink != 1 {
		return fmt.Errorf("hard-link count is %d, want 1", stat.Nlink)
	}
	return nil
}

func hashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

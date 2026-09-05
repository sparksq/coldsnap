// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package ncclprovider

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
)

// Materialize selects and atomically copies only manifest-bound regular files.
func Materialize(catalog string, target Target, output string) (Selection, error) {
	selection, err := Select(catalog, target)
	if err != nil {
		return Selection{}, err
	}
	absoluteOutput, err := filepath.Abs(output)
	if err != nil {
		return Selection{}, fmt.Errorf("resolve materialization output: %w", err)
	}
	if err := existingMaterialization(absoluteOutput, selection); err != nil {
		if errors.Is(err, errAlreadyMaterialized) {
			return selection, nil
		}
		return Selection{}, err
	}
	parent := filepath.Dir(absoluteOutput)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return Selection{}, fmt.Errorf("create materialization parent: %w", err)
	}
	temporary, err := os.MkdirTemp(parent, ".nccl-provider-")
	if err != nil {
		return Selection{}, fmt.Errorf("create temporary materialization: %w", err)
	}
	complete := false
	defer func() {
		if !complete {
			_ = os.RemoveAll(temporary)
		}
	}()
	loaded, err := LoadManifest(filepath.Join(selection.SourcePath, ManifestFilename))
	if err != nil {
		return Selection{}, err
	}
	if err := copyRegular(filepath.Join(selection.SourcePath, ManifestFilename), filepath.Join(temporary, ManifestFilename), 0o644); err != nil {
		return Selection{}, err
	}
	for _, file := range loaded.Manifest.Files {
		mode, _ := strconv.ParseUint(file.Mode, 8, 32)
		if err := copyRegular(filepath.Join(selection.SourcePath, filepath.FromSlash(file.Path)), filepath.Join(temporary, filepath.FromSlash(file.Path)), os.FileMode(mode)); err != nil {
			return Selection{}, err
		}
	}
	verification, err := Verify(temporary)
	if err != nil {
		return Selection{}, fmt.Errorf("verify temporary materialization: %w", err)
	}
	if verification.ManifestSHA256 != selection.ManifestSHA256 {
		return Selection{}, errors.New("temporary materialization changed manifest identity")
	}
	if err := os.Rename(temporary, absoluteOutput); err != nil {
		if existingErr := existingMaterialization(absoluteOutput, selection); errors.Is(existingErr, errAlreadyMaterialized) {
			return selection, nil
		}
		return Selection{}, fmt.Errorf("publish materialized provider: %w", err)
	}
	complete = true
	return selection, nil
}

func existingMaterialization(output string, selection Selection) error {
	info, err := os.Lstat(output)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return errors.New("materialization destination exists and is not a real directory")
	}
	verification, err := Verify(output)
	if err != nil {
		return fmt.Errorf("materialization destination exists but is not the exact selected provider: %w", err)
	}
	if verification.ProviderID != selection.ProviderID || verification.PlatformKey != selection.PlatformKey || verification.ManifestSHA256 != selection.ManifestSHA256 {
		return errors.New("materialization destination contains a different provider")
	}
	return errAlreadyMaterialized
}

func copyRegular(source, destination string, mode os.FileMode) error {
	info, err := os.Lstat(source)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("source %q is not a regular file", source)
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return err
	}
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return err
	}
	success := false
	defer func() {
		_ = output.Close()
		if !success {
			_ = os.Remove(destination)
		}
	}()
	if err := output.Chmod(mode); err != nil {
		return err
	}
	if _, err := io.Copy(output, input); err != nil {
		return err
	}
	if err := output.Sync(); err != nil {
		return err
	}
	if err := output.Close(); err != nil {
		return err
	}
	success = true
	return nil
}

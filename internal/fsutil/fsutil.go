// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package fsutil

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const copyBufferBytes = 16 << 20

func ValidSHA256(value string) bool {
	if len(value) != sha256.Size*2 || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func FileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()
	digest := sha256.New()
	buffer := make([]byte, copyBufferBytes)
	if _, err := io.CopyBuffer(digest, file, buffer); err != nil {
		return "", fmt.Errorf("hash %s: %w", path, err)
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func WriteJSONAtomic(path string, value any, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create %s parent: %w", path, err)
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary file for %s: %w", path, err)
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err := temporary.Chmod(mode); err != nil {
		temporary.Close()
		return fmt.Errorf("chmod temporary file for %s: %w", path, err)
	}
	encoder := json.NewEncoder(temporary)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		temporary.Close()
		return fmt.Errorf("encode %s: %w", path, err)
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return fmt.Errorf("sync temporary file for %s: %w", path, err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close temporary file for %s: %w", path, err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("publish %s: %w", path, err)
	}
	return SyncDirectory(filepath.Dir(path))
}

func WriteFileAtomic(path string, data []byte, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create %s parent: %w", path, err)
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary file for %s: %w", path, err)
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err := temporary.Chmod(mode); err != nil {
		temporary.Close()
		return fmt.Errorf("chmod temporary file for %s: %w", path, err)
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return fmt.Errorf("write temporary file for %s: %w", path, err)
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return fmt.Errorf("sync temporary file for %s: %w", path, err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close temporary file for %s: %w", path, err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("publish %s: %w", path, err)
	}
	return SyncDirectory(filepath.Dir(path))
}

func WriteFileExclusive(path string, data []byte, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create %s parent: %w", path, err)
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary file for %s: %w", path, err)
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err := temporary.Chmod(mode); err != nil {
		temporary.Close()
		return fmt.Errorf("chmod temporary file for %s: %w", path, err)
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return fmt.Errorf("write temporary file for %s: %w", path, err)
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return fmt.Errorf("sync temporary file for %s: %w", path, err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close temporary file for %s: %w", path, err)
	}
	if err := os.Link(temporaryPath, path); err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("refusing to replace %s", path)
		}
		return fmt.Errorf("publish %s exclusively: %w", path, err)
	}
	if err := SyncDirectory(filepath.Dir(path)); err != nil {
		_ = os.Remove(path)
		return err
	}
	return nil
}

func WriteJSONExclusive(path string, value any, mode os.FileMode) error {
	var encoded strings.Builder
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		return fmt.Errorf("encode %s: %w", path, err)
	}
	return WriteFileExclusive(path, []byte(encoded.String()), mode)
}

func SyncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open directory %s: %w", path, err)
	}
	defer directory.Close()
	if err := directory.Sync(); err != nil {
		return fmt.Errorf("sync directory %s: %w", path, err)
	}
	return nil
}

func PublishDirectoryExclusive(temporary, output string) error {
	temporary = filepath.Clean(temporary)
	output = filepath.Clean(output)
	if !filepath.IsAbs(temporary) || !filepath.IsAbs(output) || output == string(filepath.Separator) || filepath.Dir(temporary) != filepath.Dir(output) {
		return fmt.Errorf("directory publication paths must be safe absolute siblings")
	}
	info, err := os.Lstat(temporary)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("temporary publication path is not a directory: %s", temporary)
	}
	lock := output + ".coldsnap-publish.lock"
	if err := os.Mkdir(lock, 0o700); err != nil {
		return fmt.Errorf("acquire publication lock %s: %w", lock, err)
	}
	defer os.Remove(lock)
	if _, err := os.Lstat(output); err == nil {
		return fmt.Errorf("refusing to replace destination directory: %s", output)
	} else if !os.IsNotExist(err) {
		return err
	}
	if err := os.Rename(temporary, output); err != nil {
		return fmt.Errorf("publish directory %s: %w", output, err)
	}
	if err := SyncDirectory(filepath.Dir(output)); err != nil {
		if rollbackErr := os.Rename(output, temporary); rollbackErr != nil {
			return fmt.Errorf("%w (also failed to roll directory publication back: %v)", err, rollbackErr)
		}
		_ = SyncDirectory(filepath.Dir(output))
		return err
	}
	return nil
}

func SafeRelative(value string) (string, error) {
	if value == "" || filepath.IsAbs(value) || value == "." || strings.ContainsRune(value, '\x00') {
		return "", fmt.Errorf("path must be safe and relative: %q", value)
	}
	cleaned := filepath.Clean(value)
	if cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path must be safe and relative: %q", value)
	}
	return filepath.ToSlash(cleaned), nil
}

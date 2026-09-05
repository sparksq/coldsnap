// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package fsutil

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestValidSHA256RequiresCanonicalLowercaseHex(t *testing.T) {
	if !ValidSHA256(strings.Repeat("a", 64)) {
		t.Fatal("canonical SHA-256 was rejected")
	}
	for _, invalid := range []string{strings.Repeat("A", 64), strings.Repeat("z", 64), "abc"} {
		if ValidSHA256(invalid) {
			t.Fatalf("noncanonical SHA-256 was accepted: %q", invalid)
		}
	}
}

func TestWriteFileExclusivePublishesExactlyOneConcurrentWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "value")
	var successes atomic.Int32
	var wait sync.WaitGroup
	for index := 0; index < 32; index++ {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			if err := WriteFileExclusive(path, []byte(fmt.Sprintf("writer-%d", index)), 0o600); err == nil {
				successes.Add(1)
			}
		}(index)
	}
	wait.Wait()
	if successes.Load() != 1 {
		t.Fatalf("successful writers = %d, want 1", successes.Load())
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatal("published file is empty")
	}
}

func TestWriteJSONExclusivePreservesExistingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "record.json")
	if err := WriteJSONExclusive(path, map[string]int{"generation": 1}, 0o600); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteJSONExclusive(path, map[string]int{"generation": 2}, 0o600); err == nil {
		t.Fatal("exclusive JSON publication replaced an existing record")
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatalf("existing record changed: before=%q after=%q", before, after)
	}
}

func TestPublishDirectoryExclusivePreservesExistingDestination(t *testing.T) {
	root := t.TempDir()
	first := filepath.Join(root, "first.tmp")
	output := filepath.Join(root, "published")
	if err := os.Mkdir(first, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(first, "value"), []byte("first"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := PublishDirectoryExclusive(first, output); err != nil {
		t.Fatal(err)
	}
	second := filepath.Join(root, "second.tmp")
	if err := os.Mkdir(second, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(second, "value"), []byte("second"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := PublishDirectoryExclusive(second, output); err == nil {
		t.Fatal("exclusive directory publication replaced an existing destination")
	}
	data, err := os.ReadFile(filepath.Join(output, "value"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "first" {
		t.Fatalf("published directory changed: %q", data)
	}
	if _, err := os.Stat(second); err != nil {
		t.Fatalf("failed publication removed its input: %v", err)
	}
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package canonicaljson

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"testing"
)

type goldenValue struct {
	Name      string          `json:"name"`
	Value     json.RawMessage `json:"value"`
	Fields    json.RawMessage `json:"fields"`
	Canonical string          `json:"canonical"`
	SHA256    string          `json:"sha256"`
	Digest    string          `json:"layout_digest"`
}

func TestCrossLanguageGoldenCanonicalContracts(t *testing.T) {
	data, err := os.ReadFile("../../test/fixtures/cross-language-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Format        int           `json:"format"`
		Kind          string        `json:"kind"`
		CanonicalJSON []goldenValue `json:"canonical_json"`
		MemoryLayouts []goldenValue `json:"memory_layouts"`
		Manifests     []goldenValue `json:"manifests"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Format != 1 || fixture.Kind != "coldsnap-cross-language-contract-fixtures" {
		t.Fatalf("fixture identity = %d %q", fixture.Format, fixture.Kind)
	}
	for _, item := range append(fixture.CanonicalJSON, fixture.Manifests...) {
		t.Run(item.Name, func(t *testing.T) {
			canonical, err := CanonicalJSON(item.Value)
			if err != nil {
				t.Fatal(err)
			}
			if string(canonical) != item.Canonical {
				t.Fatalf("canonical JSON = %s, want %s", canonical, item.Canonical)
			}
			digest := sha256.Sum256(canonical)
			if actual := hex.EncodeToString(digest[:]); actual != item.SHA256 {
				t.Fatalf("canonical SHA-256 = %s, want %s", actual, item.SHA256)
			}
		})
	}
	for _, item := range fixture.MemoryLayouts {
		t.Run(item.Name+"-layout-digest", func(t *testing.T) {
			digest, err := CanonicalSHA256(item.Fields)
			if err != nil {
				t.Fatal(err)
			}
			if actual := "sha256:" + digest; actual != item.Digest {
				t.Fatalf("layout digest = %s, want %s", actual, item.Digest)
			}
		})
	}
}

func TestCanonicalNumbersMatchPython(t *testing.T) {
	values := []string{
		"1e0", "1.0", "-0.0", "-0", "1e-5", "1e-4", "1e15", "1e16",
		"1.2345678901234567", "1e20", "1.2e+6", "1.234e-7",
		"9007199254740993", "1.0000000000000002",
	}
	python := `import json,sys; print(json.dumps(json.loads(sys.argv[1]),sort_keys=True,separators=(",",":")))`
	for _, value := range values {
		t.Run(value, func(t *testing.T) {
			actual, err := CanonicalJSON([]byte(value))
			if err != nil {
				t.Fatal(err)
			}
			output, err := exec.Command("python3", "-c", python, value).CombinedOutput()
			if err != nil {
				t.Fatalf("python canonical JSON: %v: %s", err, output)
			}
			if expected := strings.TrimSpace(string(output)); string(actual) != expected {
				t.Fatalf("canonical JSON mismatch: Go=%s Python=%s", actual, expected)
			}
		})
	}
}

func TestCanonicalSHA256RejectsTrailingJSON(t *testing.T) {
	if _, err := CanonicalSHA256([]byte(`{"ok":true} {"extra":true}`)); err == nil {
		t.Fatal("multiple JSON values were accepted")
	}
}

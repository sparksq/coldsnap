// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package payloadvalidation

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestValidateHashesOnceAndReusesCanonicalRecord(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "model-weights.pack")
	payload := bytes.Repeat([]byte("coldsnap"), 4096)
	if err := os.WriteFile(path, payload, 0o400); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(payload)
	expected := "sha256:" + hex.EncodeToString(digest[:])
	options := Options{Path: path, ExpectedSHA256: expected, ExpectedBytes: int64(len(payload)), Worker: "worker-0"}
	first, err := Validate(options)
	if err != nil {
		t.Fatal(err)
	}
	if first.Worker != "worker-0" || first.SHA256 != expected || first.Validation.BytesHashed != int64(len(payload)) ||
		first.Validation.ContentEvidence != "full-sha256-this-operation" {
		t.Fatalf("first admission = %#v", first)
	}
	marker := path + RecordSuffix
	if information, err := os.Stat(marker); err != nil || information.Mode().Perm() != 0o600 {
		t.Fatalf("marker information = %#v, %v", information, err)
	}
	if err := os.Chmod(path, 0o440); err != nil {
		t.Fatal(err)
	}
	second, err := Validate(options)
	if err != nil {
		t.Fatal(err)
	}
	if second.Validation.BytesHashed != 0 || second.Validation.Reason != "cached_validation" {
		t.Fatalf("cached admission = %#v", second)
	}
}

func TestValidateRehashesStaleRecordAndRejectsChangedPayload(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "payload.pack")
	payload := []byte("expected payload")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(payload)
	options := Options{
		Path: path, ExpectedSHA256: "sha256:" + hex.EncodeToString(digest[:]), ExpectedBytes: int64(len(payload)),
	}
	if _, err := Validate(options); err != nil {
		t.Fatal(err)
	}
	marker := path + RecordSuffix
	var record map[string]any
	encoded, err := os.ReadFile(marker)
	if err != nil || json.Unmarshal(encoded, &record) != nil {
		t.Fatalf("read record: %v", err)
	}
	record["content_identity"].(map[string]any)["mtime_ns"] = float64(1)
	encoded, _ = json.Marshal(record)
	if err := os.WriteFile(marker, encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	result, err := Validate(options)
	if err != nil {
		t.Fatal(err)
	}
	if result.Validation.BytesHashed != int64(len(payload)) {
		t.Fatalf("stale validation did not rehash: %#v", result)
	}
	if err := os.WriteFile(path, []byte("changed payload!"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Validate(options); err == nil || !strings.Contains(err.Error(), "digest differs") {
		t.Fatalf("changed payload error = %v", err)
	}
}

func TestValidateRejectsSymlinkAndNoncanonicalIdentity(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "target")
	if err := os.WriteFile(target, []byte("payload"), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "payload")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	if _, err := Validate(Options{Path: link}); err == nil {
		t.Fatal("symbolic-link payload was accepted")
	}
	if _, err := Validate(Options{Path: target, ExpectedSHA256: "sha256:" + strings.Repeat("A", 64)}); err == nil {
		t.Fatal("noncanonical digest was accepted")
	}
	if _, err := Validate(Options{Path: target, Record: filepath.Join(root, "other")}); err == nil {
		t.Fatal("noncanonical validation record path was accepted")
	}
}

func TestValidateRehashesSameSizeReplacementWithPreservedMTime(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "payload.pack")
	original := []byte("payload-one")
	if err := os.WriteFile(path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(original)
	options := Options{
		Path: path, ExpectedSHA256: "sha256:" + hex.EncodeToString(digest[:]), ExpectedBytes: int64(len(original)),
	}
	if _, err := Validate(options); err != nil {
		t.Fatal(err)
	}
	information, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	replacement := filepath.Join(root, "replacement")
	if err := os.WriteFile(replacement, []byte("payload-two"), 0o600); err != nil {
		t.Fatal(err)
	}
	mtime := information.ModTime()
	if err := os.Chtimes(replacement, time.Unix(0, 0), mtime); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, path); err != nil {
		t.Fatal(err)
	}
	if _, err := Validate(options); err == nil || !strings.Contains(err.Error(), "digest differs") {
		t.Fatalf("same-size replacement error = %v", err)
	}
}

func TestExecuteReturnsManagerCompatibleAdmission(t *testing.T) {
	path := filepath.Join(t.TempDir(), "payload.pack")
	payload := []byte("payload")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(payload)
	var stdout, stderr bytes.Buffer
	err := Execute([]string{
		"--path", path,
		"--expected-sha256", "sha256:" + hex.EncodeToString(digest[:]),
		"--expected-bytes", "7",
		"--worker", "worker-7",
	}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("execute: %v: %s", err, stderr.String())
	}
	var result Admission
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Format != RecordFormat || result.Kind != ResultKind || result.Decision != "accept" ||
		result.Worker != "worker-7" || result.Path != path || result.Validation.Record != path+RecordSuffix {
		t.Fatalf("admission = %#v", result)
	}
}

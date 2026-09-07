// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Package payloadvalidation implements the one canonical admission protocol
// for immutable native model payloads. Managers and engine adapters may run
// the same release-matched adapter binary on a data-owning host; they must not
// carry independent hashing or validation-record implementations.
package payloadvalidation

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/sparksq/coldsnap/internal/fsutil"
)

const (
	RecordFormat = 1
	RecordKind   = "coldsnap-payload-validation"
	ResultKind   = "coldsnap-payload-validation-result"
	Provider     = "sha256-cache-v1"
	RecordSuffix = ".coldsnap-validation.json"
	hashBuffer   = 16 << 20
	maxRecord    = 1 << 20
)

type ContentIdentity struct {
	Device  uint64 `json:"device"`
	Inode   uint64 `json:"inode"`
	Size    int64  `json:"size"`
	MTimeNS int64  `json:"mtime_ns"`
}

type DiagnosticIdentity struct {
	CTimeNS int64  `json:"ctime_ns"`
	UID     uint32 `json:"uid"`
	GID     uint32 `json:"gid"`
	Mode    uint32 `json:"mode"`
}

type Expected struct {
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type Record struct {
	Format             int                `json:"format"`
	Kind               string             `json:"kind"`
	Provider           string             `json:"provider"`
	Blob               string             `json:"blob"`
	Expected           Expected           `json:"expected"`
	ContentIdentity    ContentIdentity    `json:"content_identity"`
	DiagnosticIdentity DiagnosticIdentity `json:"diagnostic_identity"`
	ContentEvidence    string             `json:"content_evidence"`
	ValidatedUnixNS    int64              `json:"validated_unix_ns"`
}

type Evidence struct {
	Record          string  `json:"record"`
	Provider        string  `json:"provider"`
	ContentEvidence string  `json:"content_evidence"`
	Reason          string  `json:"reason"`
	Device          uint64  `json:"device"`
	Inode           uint64  `json:"inode"`
	Size            int64   `json:"size"`
	MTimeNS         int64   `json:"mtime_ns"`
	CTimeNS         int64   `json:"ctime_ns"`
	UID             uint32  `json:"uid"`
	GID             uint32  `json:"gid"`
	Mode            uint32  `json:"mode"`
	BytesHashed     int64   `json:"bytes_hashed"`
	Seconds         float64 `json:"seconds"`
}

type Admission struct {
	Format     int      `json:"format"`
	Kind       string   `json:"kind"`
	Decision   string   `json:"decision"`
	Worker     string   `json:"worker,omitempty"`
	Path       string   `json:"path"`
	Bytes      int64    `json:"bytes"`
	SHA256     string   `json:"sha256"`
	Validation Evidence `json:"validation"`
}

type Options struct {
	Path           string `json:"path"`
	Record         string `json:"record,omitempty"`
	ExpectedSHA256 string `json:"expected_sha256,omitempty"`
	ExpectedBytes  int64  `json:"expected_bytes,omitempty"`
	Worker         string `json:"worker,omitempty"`
}

func Validate(options Options) (Admission, error) {
	started := time.Now()
	path, recordPath, expectedSHA256, err := normalize(options)
	if err != nil {
		return Admission{}, err
	}
	file, stat, err := openRegular(path)
	if err != nil {
		return Admission{}, err
	}
	defer file.Close()
	before, diagnostics, err := identities(stat)
	if err != nil {
		return Admission{}, err
	}
	if options.ExpectedBytes > 0 && before.Size != options.ExpectedBytes {
		return Admission{}, fmt.Errorf(
			"model payload size differs from the committed artifact: expected=%d actual=%d",
			options.ExpectedBytes, before.Size,
		)
	}
	if expectedSHA256 != "" && options.ExpectedBytes > 0 {
		if record, ok := cachedRecord(recordPath); ok && recordMatches(
			record, filepath.Base(path), expectedSHA256, options.ExpectedBytes, before,
		) {
			return admission(
				options.Worker, path, recordPath, before, diagnostics, expectedSHA256,
				"cached-full-sha256", "cached_validation", 0, time.Since(started),
			), nil
		}
	}
	digest := sha256.New()
	bytesHashed, err := io.CopyBuffer(digest, file, make([]byte, hashBuffer))
	if err != nil {
		return Admission{}, fmt.Errorf("hash model payload: %w", err)
	}
	afterStat, err := file.Stat()
	if err != nil {
		return Admission{}, fmt.Errorf("restat model payload: %w", err)
	}
	after, afterDiagnostics, err := identities(afterStat)
	if err != nil {
		return Admission{}, err
	}
	if before != after {
		return Admission{}, errors.New("model payload changed during validation")
	}
	actualSHA256 := "sha256:" + hex.EncodeToString(digest.Sum(nil))
	if expectedSHA256 != "" && actualSHA256 != expectedSHA256 {
		return Admission{}, fmt.Errorf(
			"model payload digest differs from the committed artifact: expected=%s actual=%s",
			expectedSHA256, actualSHA256,
		)
	}
	if expectedSHA256 == "" {
		expectedSHA256 = actualSHA256
	}
	expectedBytes := options.ExpectedBytes
	if expectedBytes == 0 {
		expectedBytes = after.Size
	}
	record := Record{
		Format: RecordFormat, Kind: RecordKind, Provider: Provider,
		Blob: filepath.Base(path), Expected: Expected{Bytes: expectedBytes, SHA256: expectedSHA256},
		ContentIdentity: after, DiagnosticIdentity: afterDiagnostics,
		ContentEvidence: "full-sha256-this-operation", ValidatedUnixNS: time.Now().UnixNano(),
	}
	if err := fsutil.WriteJSONAtomic(recordPath, record, 0o600); err != nil {
		return Admission{}, fmt.Errorf("publish model payload validation record: %w", err)
	}
	return admission(
		options.Worker, path, recordPath, after, afterDiagnostics, expectedSHA256,
		"full-sha256-this-operation", "revalidated_existing", bytesHashed, time.Since(started),
	), nil
}

func Execute(arguments []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("payload-verify", flag.ContinueOnError)
	flags.SetOutput(stderr)
	path := flags.String("path", "", "absolute native model payload path")
	record := flags.String("record", "", "validation record path (default: <path>"+RecordSuffix+")")
	expectedSHA256 := flags.String("expected-sha256", "", "expected canonical sha256:<hex> digest")
	expectedBytes := flags.Int64("expected-bytes", 0, "expected positive byte size; zero discovers the identity")
	worker := flags.String("worker", "", "optional execution worker identity included in the result")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("payload-verify does not accept positional arguments")
	}
	result, err := Validate(Options{
		Path: *path, Record: *record, ExpectedSHA256: *expectedSHA256,
		ExpectedBytes: *expectedBytes, Worker: *worker,
	})
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(stdout)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(result)
}

func normalize(options Options) (string, string, string, error) {
	path := filepath.Clean(options.Path)
	if options.Path == "" || !filepath.IsAbs(options.Path) || path != options.Path || strings.ContainsRune(path, '\x00') {
		return "", "", "", errors.New("model payload path must be clean and absolute")
	}
	recordPath := options.Record
	if recordPath == "" {
		recordPath = path + RecordSuffix
	}
	if filepath.Clean(recordPath) != recordPath || recordPath != path+RecordSuffix {
		return "", "", "", errors.New("model payload validation record must use the canonical adjacent path")
	}
	if options.ExpectedBytes < 0 {
		return "", "", "", errors.New("expected model payload bytes cannot be negative")
	}
	expectedSHA256 := options.ExpectedSHA256
	if expectedSHA256 != "" {
		digest, ok := strings.CutPrefix(expectedSHA256, "sha256:")
		if !ok || !fsutil.ValidSHA256(digest) {
			return "", "", "", errors.New("expected model payload SHA-256 is not canonical")
		}
	}
	return path, recordPath, expectedSHA256, nil
}

func openRegular(path string) (*os.File, os.FileInfo, error) {
	descriptor, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_CLOEXEC|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return nil, nil, fmt.Errorf("open model payload: %w", err)
	}
	file := os.NewFile(uintptr(descriptor), path)
	if file == nil {
		_ = syscall.Close(descriptor)
		return nil, nil, errors.New("open model payload returned no file")
	}
	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, nil, fmt.Errorf("stat model payload: %w", err)
	}
	if !stat.Mode().IsRegular() {
		file.Close()
		return nil, nil, errors.New("model payload is not a regular file")
	}
	return file, stat, nil
}

func identities(stat os.FileInfo) (ContentIdentity, DiagnosticIdentity, error) {
	raw, ok := stat.Sys().(*syscall.Stat_t)
	if !ok {
		return ContentIdentity{}, DiagnosticIdentity{}, errors.New("model payload stat identity is unavailable")
	}
	mtime, ctime := statTimes(raw)
	return ContentIdentity{
			Device: uint64(raw.Dev), Inode: raw.Ino, Size: raw.Size, MTimeNS: mtime,
		}, DiagnosticIdentity{
			CTimeNS: ctime, UID: raw.Uid, GID: raw.Gid, Mode: uint32(stat.Mode().Perm()),
		}, nil
}

func cachedRecord(path string) (Record, bool) {
	file, stat, err := openRegular(path)
	if err != nil {
		return Record{}, false
	}
	defer file.Close()
	if stat.Mode().Perm()&0o077 != 0 || stat.Size() <= 0 || stat.Size() > maxRecord {
		return Record{}, false
	}
	decoder := json.NewDecoder(io.LimitReader(file, maxRecord+1))
	var record Record
	if err := decoder.Decode(&record); err != nil {
		return Record{}, false
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return Record{}, false
	}
	return record, true
}

func recordMatches(
	record Record, blob, expectedSHA256 string, expectedBytes int64, identity ContentIdentity,
) bool {
	return record.Format == RecordFormat && record.Kind == RecordKind && record.Provider == Provider &&
		record.Blob == blob && record.Expected == (Expected{Bytes: expectedBytes, SHA256: expectedSHA256}) &&
		record.ContentIdentity == identity
}

func admission(
	worker, path, recordPath string,
	identity ContentIdentity,
	diagnostics DiagnosticIdentity,
	digest, evidence, reason string,
	bytesHashed int64,
	duration time.Duration,
) Admission {
	return Admission{
		Format: RecordFormat, Kind: ResultKind, Decision: "accept",
		Worker: worker, Path: path, Bytes: identity.Size, SHA256: digest,
		Validation: Evidence{
			Record: recordPath, Provider: Provider, ContentEvidence: evidence, Reason: reason,
			Device: identity.Device, Inode: identity.Inode, Size: identity.Size,
			MTimeNS: identity.MTimeNS, CTimeNS: diagnostics.CTimeNS,
			UID: diagnostics.UID, GID: diagnostics.GID, Mode: diagnostics.Mode,
			BytesHashed: bytesHashed, Seconds: duration.Seconds(),
		},
	}
}

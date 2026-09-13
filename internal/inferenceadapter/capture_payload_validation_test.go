// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/sparksq/coldsnap/internal/payloadvalidation"
	"github.com/sparksq/coldsnap/internal/snapshot"
)

type captureAdmissionRemote struct {
	localCommandRemote
	admissions []payloadvalidation.Admission
}

func (remote *captureAdmissionRemote) Run(ctx context.Context, host string, arguments ...string) ([]byte, error) {
	output, err := remote.localCommandRemote.Run(ctx, host, arguments...)
	if err == nil && len(arguments) > 1 && arguments[1] == "payload-verify" {
		var admission payloadvalidation.Admission
		if err := json.Unmarshal(output, &admission); err != nil {
			return nil, err
		}
		remote.admissions = append(remote.admissions, admission)
	}
	return output, err
}

func TestCapturedPayloadIdentityReusesWriterReceiptAtHandoff(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "hydration", "worker")
	if err := os.MkdirAll(directory, 0700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, modelPayloadName)
	if err := os.WriteFile(path, []byte("captured bytes"), 0400); err != nil {
		t.Fatal(err)
	}
	// This is the same canonical validation called by the writer after sync.
	initial, err := payloadvalidation.Validate(payloadvalidation.Options{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	request := validRequest(1)
	worker := request.Launch.Execution.Workers[0].ID
	manifestPath := filepath.Join(directory, "manifest.json")
	manifest := map[string]any{"worker_id": worker, "model_payload": capturedModelPayload{Blob: modelPayloadName, Bytes: initial.Bytes, SHA256: initial.SHA256}}
	encoded, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, encoded, 0600); err != nil {
		t.Fatal(err)
	}
	remote := &captureAdmissionRemote{}
	adapter := fixtureAdapter(Adapter{Remote: remote})
	unit := request.Launch.Units[0]
	manifests, err := adapter.hydrationManifests(context.Background(), request.Launch, unit, root)
	if err != nil {
		t.Fatal(err)
	}
	identity := manifests[0].ModelPayload
	object, err := adapter.remoteModelPayloadObject(context.Background(), unit, snapshot.WorkerOwner(worker), path, "pending-content-address", identity.Bytes, identity.SHA256)
	if err != nil {
		t.Fatal(err)
	}
	if object.SHA256 != initial.SHA256 || object.Bytes != initial.Bytes || len(remote.admissions) != 1 || remote.admissions[0].Validation.BytesHashed != 0 || remote.admissions[0].Validation.ContentEvidence != "cached-full-sha256" {
		t.Fatalf("handoff did not reuse capture evidence: object=%+v admissions=%+v", object, remote.admissions)
	}
	if err := os.Chmod(path, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("modified bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := adapter.remoteModelPayloadObject(context.Background(), unit, object.Owner, path, object.Path, identity.Bytes, identity.SHA256); err == nil {
		t.Fatal("changed capture admitted by stale writer evidence")
	}
	manifest["model_payload"] = capturedModelPayload{Blob: "another.pack", Bytes: initial.Bytes, SHA256: initial.SHA256}
	encoded, _ = json.Marshal(manifest)
	if err := os.WriteFile(manifestPath, encoded, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := adapter.hydrationManifests(context.Background(), request.Launch, unit, root); err == nil {
		t.Fatal("invalid captured payload identity admitted")
	}
}

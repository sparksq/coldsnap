// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import (
	"errors"
	"testing"
	"time"
)

func TestOperationReceiptBindsCanonicalRequestAndFailure(t *testing.T) {
	request := validRequest()
	request.Operation = "restore"
	request.Artifact = "/tmp/artifact.json"
	started := time.Date(2026, 8, 27, 1, 2, 3, 0, time.UTC)
	receipt, err := NewOperationReceipt(
		request, false, started, started.Add(1500*time.Millisecond), errors.New("admission denied"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.State != "failed" || receipt.Error != "admission denied" ||
		receipt.ReplaySemantics != "reconcile-replace" || receipt.DurationSeconds != 1.5 {
		t.Fatalf("receipt = %#v", receipt)
	}
	if receipt.RequestSHA256 == "" || receipt.Result.Artifact != request.Artifact {
		t.Fatalf("receipt identity = %#v", receipt)
	}
}

func TestPrepareReceiptIsSafeToRepeat(t *testing.T) {
	request := validRequest()
	request.Operation = "restore"
	request.Artifact = "/tmp/artifact.json"
	now := time.Date(2026, 8, 27, 1, 2, 3, 0, time.UTC)
	receipt, err := NewOperationReceipt(request, true, now, now, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Result.Prepared || receipt.ReplaySemantics != "safe-repeat" {
		t.Fatalf("prepare receipt = %#v", receipt)
	}
}

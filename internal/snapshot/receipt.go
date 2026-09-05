// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"time"

	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

const (
	OperationReceiptFormat = 2
	OperationReceiptKind   = "coldsnap-operation-receipt"
)

type OperationResult struct {
	Artifact string `json:"artifact,omitempty"`
	Output   string `json:"output,omitempty"`
	Prepared bool   `json:"prepared,omitempty"`
}

// OperationReceipt is the stable controller-facing result envelope. Engine
// adapters remain free to emit detailed progress and reports; an orchestrator
// can request this compact record without parsing their human-readable output.
type OperationReceipt struct {
	Format          int             `json:"format"`
	Kind            string          `json:"kind"`
	OperationID     string          `json:"operation_id"`
	Operation       string          `json:"operation"`
	State           string          `json:"state"`
	RequestSHA256   string          `json:"request_sha256"`
	Engine          string          `json:"engine"`
	SnapshotDriver  string          `json:"snapshot_driver"`
	ReplaySemantics string          `json:"replay_semantics"`
	StartedAt       string          `json:"started_at"`
	CompletedAt     string          `json:"completed_at"`
	DurationSeconds float64         `json:"duration_seconds"`
	Timing          OperationTiming `json:"timing"`
	Result          OperationResult `json:"result"`
	Error           string          `json:"error,omitempty"`
}

func ReplaySemantics(operation string, prepareOnly bool) string {
	if prepareOnly {
		return "safe-repeat"
	}
	switch operation {
	case "capture":
		return "conflict-on-existing-output"
	case "publish", "publish-native":
		return "content-idempotent"
	case "restore":
		return "reconcile-replace"
	case "sleep", "wake", "status":
		return "convergent"
	default:
		return "unsupported"
	}
}

func NewOperationReceipt(
	request Request,
	prepareOnly bool,
	started, completed time.Time,
	operationError error,
	timings ...OperationTiming,
) (OperationReceipt, error) {
	digest, err := RequestSHA256(request)
	if err != nil {
		return OperationReceipt{}, err
	}
	state := "succeeded"
	errorText := ""
	if operationError != nil {
		state = "failed"
		errorText = operationError.Error()
	}
	timingStatus := "ok"
	if operationError != nil {
		timingStatus = "error"
	}
	timing := OperationTiming{
		Format: OperationTimingFormat,
		Clocks: []TimingClock{{ID: "controller", Source: "controller", OriginUnixNS: started.UnixNano()}},
		Spans: []TimingSpan{{
			ID: "controller-1", Name: "controller.operation", Clock: "controller",
			DurationSeconds: completed.Sub(started).Seconds(), Status: timingStatus,
		}},
	}
	if len(timings) > 0 {
		timing = timings[0]
	}
	receipt := OperationReceipt{
		Format: OperationReceiptFormat, Kind: OperationReceiptKind,
		OperationID: request.ID, Operation: request.Operation, State: state,
		RequestSHA256: digest, Engine: request.Launch.Engine,
		SnapshotDriver:  request.Driver.ID,
		ReplaySemantics: ReplaySemantics(request.Operation, prepareOnly),
		StartedAt:       started.UTC().Format(time.RFC3339Nano),
		CompletedAt:     completed.UTC().Format(time.RFC3339Nano),
		DurationSeconds: completed.Sub(started).Seconds(),
		Timing:          timing,
		Result: OperationResult{
			Artifact: request.Artifact, Output: request.Output, Prepared: prepareOnly,
		},
		Error: errorText,
	}
	if err := receipt.Validate(); err != nil {
		return OperationReceipt{}, err
	}
	return receipt, nil
}

func (receipt OperationReceipt) Validate() error {
	if receipt.Format != OperationReceiptFormat || receipt.Kind != OperationReceiptKind {
		return errors.New("unsupported ColdSnap operation receipt")
	}
	if !idPattern.MatchString(receipt.OperationID) {
		return errors.New("ColdSnap operation receipt ID is invalid")
	}
	if !slices.Contains(
		[]string{"capture", "publish", "publish-native", "restore", "sleep", "wake", "status"},
		receipt.Operation,
	) {
		return errors.New("ColdSnap operation receipt operation is invalid")
	}
	if receipt.State != "succeeded" && receipt.State != "failed" {
		return errors.New("ColdSnap operation receipt state is invalid")
	}
	if !digestPattern.MatchString(receipt.RequestSHA256) {
		return errors.New("ColdSnap operation receipt request digest is invalid")
	}
	if receipt.Engine != "vllm" && receipt.Engine != "sglang" {
		return errors.New("ColdSnap operation receipt engine is invalid")
	}
	if _, err := snapshotdriver.Lookup(receipt.SnapshotDriver); err != nil {
		return errors.New("ColdSnap operation receipt snapshot driver is invalid")
	}
	if receipt.ReplaySemantics != ReplaySemantics(receipt.Operation, receipt.Result.Prepared) {
		return errors.New("ColdSnap operation receipt replay semantics are invalid")
	}
	started, err := time.Parse(time.RFC3339Nano, receipt.StartedAt)
	if err != nil {
		return errors.New("ColdSnap operation receipt start time is invalid")
	}
	completed, err := time.Parse(time.RFC3339Nano, receipt.CompletedAt)
	if err != nil || completed.Before(started) {
		return errors.New("ColdSnap operation receipt completion time is invalid")
	}
	if receipt.DurationSeconds < 0 || math.IsNaN(receipt.DurationSeconds) || math.IsInf(receipt.DurationSeconds, 0) {
		return errors.New("ColdSnap operation receipt duration is invalid")
	}
	if err := receipt.Timing.Validate(); err != nil {
		return err
	}
	if receipt.State == "succeeded" && receipt.Error != "" {
		return errors.New("successful ColdSnap operation receipt contains an error")
	}
	if receipt.State == "failed" && receipt.Error == "" {
		return errors.New("failed ColdSnap operation receipt has no error")
	}
	return nil
}

func (receipt OperationReceipt) SameRequest(other OperationReceipt) bool {
	return receipt.OperationID == other.OperationID &&
		receipt.Operation == other.Operation &&
		receipt.RequestSHA256 == other.RequestSHA256
}

func (receipt OperationReceipt) String() string {
	return fmt.Sprintf("%s %s (%s)", receipt.Operation, receipt.State, receipt.OperationID)
}

func DecodeOperationReceipt(reader io.Reader) (OperationReceipt, error) {
	decoder := json.NewDecoder(reader)
	decoder.DisallowUnknownFields()
	var receipt OperationReceipt
	if err := decoder.Decode(&receipt); err != nil {
		return OperationReceipt{}, fmt.Errorf("decode ColdSnap operation receipt: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return OperationReceipt{}, errors.New("decode ColdSnap operation receipt: trailing JSON")
	}
	if err := receipt.Validate(); err != nil {
		return OperationReceipt{}, fmt.Errorf("validate ColdSnap operation receipt: %w", err)
	}
	return receipt, nil
}

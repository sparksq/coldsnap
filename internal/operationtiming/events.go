// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package operationtiming

import (
	"encoding/json"
	"fmt"
	"io"
	"sync"

	"github.com/sparksq/coldsnap/internal/snapshot"
)

// EventSink receives advisory timing lifecycle events. Sink failures never
// alter the snapshot operation; the final validated receipt is authoritative.
type EventSink interface {
	Clock(snapshot.TimingClock)
	SpanStarted(snapshot.TimingEventSpan)
	SpanCompleted(snapshot.TimingSpan)
}

// NDJSONWriter serializes exactly one operation-scoped timing event stream.
type NDJSONWriter struct {
	mu            sync.Mutex
	output        io.Writer
	operationID   string
	requestSHA256 string
	sequence      uint64
	err           error
}

func NewNDJSONWriter(output io.Writer, operationID, requestSHA256 string) *NDJSONWriter {
	return &NDJSONWriter{output: output, operationID: operationID, requestSHA256: requestSHA256}
}

func (writer *NDJSONWriter) Start() {
	writer.emit(snapshot.TimingEvent{Event: snapshot.TimingEventStreamStart})
}

func (writer *NDJSONWriter) Clock(clock snapshot.TimingClock) {
	writer.emit(snapshot.TimingEvent{Event: snapshot.TimingEventClock, Clock: &clock})
}

func (writer *NDJSONWriter) SpanStarted(span snapshot.TimingEventSpan) {
	writer.emit(snapshot.TimingEvent{Event: snapshot.TimingEventSpanStart, Span: &span})
}

func (writer *NDJSONWriter) SpanCompleted(span snapshot.TimingSpan) {
	duration := span.DurationSeconds
	writer.emit(snapshot.TimingEvent{
		Event: snapshot.TimingEventSpanComplete,
		Span: &snapshot.TimingEventSpan{
			ID: span.ID, Parent: span.Parent, Name: span.Name, Clock: span.Clock,
			StartOffsetSeconds: span.StartOffsetSeconds, DurationSeconds: &duration,
			Status: span.Status, Attributes: cloneAttributes(span.Attributes),
		},
	})
}

func (writer *NDJSONWriter) Complete(state, timingSHA256 string) {
	writer.emit(snapshot.TimingEvent{
		Event: snapshot.TimingEventStreamComplete, State: state, TimingSHA256: timingSHA256,
	})
}

func (writer *NDJSONWriter) Error() error {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	return writer.err
}

func (writer *NDJSONWriter) emit(event snapshot.TimingEvent) {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	if writer.err != nil {
		return
	}
	writer.sequence++
	event.Format = snapshot.TimingEventFormat
	event.Kind = snapshot.TimingEventKind
	event.Sequence = writer.sequence
	event.OperationID = writer.operationID
	event.RequestSHA256 = writer.requestSHA256
	if err := event.Validate(); err != nil {
		writer.err = fmt.Errorf("validate timing event: %w", err)
		return
	}
	payload, err := json.Marshal(event)
	if err == nil {
		payload = append(payload, '\n')
		var written int
		written, err = writer.output.Write(payload)
		if err == nil && written != len(payload) {
			err = io.ErrShortWrite
		}
	}
	if err != nil {
		writer.err = fmt.Errorf("write timing event: %w", err)
	}
}

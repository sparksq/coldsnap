// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

const (
	TimingEventFormat = 1
	TimingEventKind   = "coldsnap-timing-event"

	TimingEventStreamStart    = "stream_start"
	TimingEventClock          = "clock"
	TimingEventSpanStart      = "span_start"
	TimingEventSpanComplete   = "span_complete"
	TimingEventStreamComplete = "stream_complete"

	MaximumTimingEventBytes = 64 << 10
	MaximumTimingEvents     = 8192
)

// TimingEventSpan represents either the opening or completed form of a span.
// Duration and status are absent on span_start and required on span_complete.
type TimingEventSpan struct {
	ID                 string            `json:"id"`
	Parent             string            `json:"parent,omitempty"`
	Name               string            `json:"name"`
	Clock              string            `json:"clock"`
	StartOffsetSeconds float64           `json:"start_offset_seconds"`
	DurationSeconds    *float64          `json:"duration_seconds,omitempty"`
	Status             string            `json:"status,omitempty"`
	Attributes         map[string]string `json:"attributes,omitempty"`
}

// TimingEvent is one bounded record in an operation-scoped NDJSON stream.
// The final atomic operation receipt remains authoritative.
type TimingEvent struct {
	Format        int              `json:"format"`
	Kind          string           `json:"kind"`
	Sequence      uint64           `json:"sequence"`
	Event         string           `json:"event"`
	OperationID   string           `json:"operation_id"`
	RequestSHA256 string           `json:"request_sha256"`
	Clock         *TimingClock     `json:"clock,omitempty"`
	Span          *TimingEventSpan `json:"span,omitempty"`
	State         string           `json:"state,omitempty"`
	TimingSHA256  string           `json:"timing_sha256,omitempty"`
}

func (event TimingEvent) Validate() error {
	if event.Format != TimingEventFormat || event.Kind != TimingEventKind || event.Sequence == 0 ||
		!idPattern.MatchString(event.OperationID) || !digestPattern.MatchString(event.RequestSHA256) {
		return errors.New("ColdSnap timing event envelope is invalid")
	}
	switch event.Event {
	case TimingEventStreamStart:
		if event.Clock != nil || event.Span != nil || event.State != "" || event.TimingSHA256 != "" {
			return errors.New("ColdSnap timing stream start is invalid")
		}
	case TimingEventClock:
		if event.Clock == nil || event.Span != nil || event.State != "" || event.TimingSHA256 != "" ||
			!validTimingEventClock(*event.Clock) {
			return errors.New("ColdSnap timing clock event is invalid")
		}
	case TimingEventSpanStart, TimingEventSpanComplete:
		if event.Clock != nil || event.Span == nil || event.State != "" || event.TimingSHA256 != "" ||
			!validTimingEventSpan(*event.Span, event.Event == TimingEventSpanComplete) {
			return errors.New("ColdSnap timing span event is invalid")
		}
	case TimingEventStreamComplete:
		if event.Clock != nil || event.Span != nil ||
			(event.State != "succeeded" && event.State != "failed") || !digestPattern.MatchString(event.TimingSHA256) {
			return errors.New("ColdSnap timing stream completion is invalid")
		}
	default:
		return errors.New("ColdSnap timing event type is unsupported")
	}
	return nil
}

func validTimingEventClock(clock TimingClock) bool {
	return idPattern.MatchString(clock.ID) && safeTimingText(clock.Source, 64) && clock.OriginUnixNS > 0 &&
		(clock.Unit == "" || idPattern.MatchString(clock.Unit)) &&
		(clock.Worker == "" || idPattern.MatchString(clock.Worker))
}

func validTimingEventSpan(span TimingEventSpan, completed bool) bool {
	if !idPattern.MatchString(span.ID) || (span.Parent != "" && !idPattern.MatchString(span.Parent)) ||
		!safeTimingText(span.Name, 128) || !idPattern.MatchString(span.Clock) ||
		!finiteNonnegative(span.StartOffsetSeconds) || len(span.Attributes) > maximumTimingAttrs {
		return false
	}
	for key, value := range span.Attributes {
		if !idPattern.MatchString(key) || !safeTimingText(value, 512) {
			return false
		}
	}
	if completed {
		return span.DurationSeconds != nil && finiteNonnegative(*span.DurationSeconds) &&
			(span.Status == "ok" || span.Status == "error")
	}
	return span.DurationSeconds == nil && span.Status == ""
}

func DecodeTimingEvent(reader io.Reader) (TimingEvent, error) {
	decoder := json.NewDecoder(io.LimitReader(reader, MaximumTimingEventBytes+1))
	decoder.DisallowUnknownFields()
	var event TimingEvent
	if err := decoder.Decode(&event); err != nil {
		return TimingEvent{}, fmt.Errorf("decode ColdSnap timing event: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return TimingEvent{}, errors.New("decode ColdSnap timing event: trailing JSON")
	}
	if err := event.Validate(); err != nil {
		return TimingEvent{}, err
	}
	return event, nil
}

// TimingEventStreamValidator validates ordering and identity for one stream.
type TimingEventStreamValidator struct {
	OperationID   string
	RequestSHA256 string
	sequence      uint64
	count         int
	started       bool
	completed     bool
	clocks        map[string]bool
	spans         map[string]bool
	spanStarts    map[string]TimingEventSpan
}

func (validator *TimingEventStreamValidator) Accept(event TimingEvent) error {
	if err := event.Validate(); err != nil {
		return err
	}
	validator.count++
	if validator.count > MaximumTimingEvents || event.Sequence != validator.sequence+1 ||
		event.OperationID != validator.OperationID || event.RequestSHA256 != validator.RequestSHA256 || validator.completed {
		return errors.New("ColdSnap timing event stream ordering or identity is invalid")
	}
	validator.sequence = event.Sequence
	if event.Event == TimingEventStreamStart {
		if validator.started || event.Sequence != 1 {
			return errors.New("ColdSnap timing event stream start is invalid")
		}
		validator.started = true
		validator.clocks = make(map[string]bool)
		validator.spans = make(map[string]bool)
		validator.spanStarts = make(map[string]TimingEventSpan)
		return nil
	}
	if !validator.started {
		return errors.New("ColdSnap timing event stream has no start")
	}
	switch event.Event {
	case TimingEventClock:
		if validator.clocks[event.Clock.ID] {
			return errors.New("ColdSnap timing event stream repeats a clock")
		}
		validator.clocks[event.Clock.ID] = true
	case TimingEventSpanStart:
		_, exists := validator.spans[event.Span.ID]
		_, parentExists := validator.spans[event.Span.Parent]
		if !validator.clocks[event.Span.Clock] || exists ||
			(event.Span.Parent != "" && !parentExists) {
			return errors.New("ColdSnap timing event stream span start is invalid")
		}
		validator.spans[event.Span.ID] = false
		validator.spanStarts[event.Span.ID] = *event.Span
	case TimingEventSpanComplete:
		state, exists := validator.spans[event.Span.ID]
		_, parentExists := validator.spans[event.Span.Parent]
		if !validator.clocks[event.Span.Clock] || (exists && state) ||
			(event.Span.Parent != "" && !parentExists) ||
			(exists && !sameTimingEventSpanLifecycle(validator.spanStarts[event.Span.ID], *event.Span)) {
			return errors.New("ColdSnap timing event stream span completion is invalid")
		}
		validator.spans[event.Span.ID] = true
	case TimingEventStreamComplete:
		if len(validator.clocks) == 0 || len(validator.spans) == 0 {
			return errors.New("ColdSnap timing event stream has no timing inventory")
		}
		for _, completed := range validator.spans {
			if !completed {
				return errors.New("ColdSnap timing event stream completed with open spans")
			}
		}
		validator.completed = true
	}
	return nil
}

func (validator *TimingEventStreamValidator) Complete() bool { return validator.completed }

func sameTimingEventSpanLifecycle(start, completed TimingEventSpan) bool {
	if start.ID != completed.ID || start.Parent != completed.Parent || start.Name != completed.Name ||
		start.Clock != completed.Clock || start.StartOffsetSeconds != completed.StartOffsetSeconds {
		return false
	}
	for key, value := range start.Attributes {
		if key == "error" && completed.Status == "error" {
			continue
		}
		if completed.Attributes[key] != value {
			return false
		}
	}
	for key := range completed.Attributes {
		if key == "error" && completed.Status == "error" {
			continue
		}
		if _, exists := start.Attributes[key]; !exists {
			return false
		}
	}
	return true
}

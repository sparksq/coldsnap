// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import (
	"strings"
	"testing"
)

const timingEventTestDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func TestTimingEventStreamValidatorAcceptsStartedAndCompletedOnlySpans(t *testing.T) {
	duration := 1.25
	events := []TimingEvent{
		timingEventForTest(1, TimingEventStreamStart),
		{
			Format: TimingEventFormat, Kind: TimingEventKind, Sequence: 2,
			Event: TimingEventClock, OperationID: "restore-events", RequestSHA256: timingEventTestDigest,
			Clock: &TimingClock{ID: "controller", Source: "controller", OriginUnixNS: 1},
		},
		{
			Format: TimingEventFormat, Kind: TimingEventKind, Sequence: 3,
			Event: TimingEventSpanStart, OperationID: "restore-events", RequestSHA256: timingEventTestDigest,
			Span: &TimingEventSpan{ID: "root", Name: "controller.restore", Clock: "controller"},
		},
		{
			Format: TimingEventFormat, Kind: TimingEventKind, Sequence: 4,
			Event: TimingEventSpanComplete, OperationID: "restore-events", RequestSHA256: timingEventTestDigest,
			Span: &TimingEventSpan{
				ID: "remote", Parent: "root", Name: "runtime.restore", Clock: "controller",
				DurationSeconds: &duration, Status: "ok",
			},
		},
		{
			Format: TimingEventFormat, Kind: TimingEventKind, Sequence: 5,
			Event: TimingEventSpanComplete, OperationID: "restore-events", RequestSHA256: timingEventTestDigest,
			Span: &TimingEventSpan{
				ID: "root", Name: "controller.restore", Clock: "controller",
				DurationSeconds: &duration, Status: "error",
				Attributes: map[string]string{"error": "adapter failed"},
			},
		},
		{
			Format: TimingEventFormat, Kind: TimingEventKind, Sequence: 6,
			Event: TimingEventStreamComplete, OperationID: "restore-events", RequestSHA256: timingEventTestDigest,
			State: "failed", TimingSHA256: timingEventTestDigest,
		},
	}
	validator := TimingEventStreamValidator{OperationID: "restore-events", RequestSHA256: timingEventTestDigest}
	for _, event := range events {
		if err := validator.Accept(event); err != nil {
			t.Fatal(err)
		}
	}
	if !validator.Complete() {
		t.Fatal("stream did not complete")
	}
}

func TestTimingEventStreamValidatorRejectsBrokenStreams(t *testing.T) {
	clock := TimingEvent{
		Format: TimingEventFormat, Kind: TimingEventKind, Sequence: 2,
		Event: TimingEventClock, OperationID: "restore-events", RequestSHA256: timingEventTestDigest,
		Clock: &TimingClock{ID: "controller", Source: "controller", OriginUnixNS: 1},
	}
	for name, mutate := range map[string]func(*TimingEventStreamValidator, *TimingEvent){
		"sequence":     func(_ *TimingEventStreamValidator, event *TimingEvent) { event.Sequence = 3 },
		"identity":     func(_ *TimingEventStreamValidator, event *TimingEvent) { event.OperationID = "other" },
		"before start": func(validator *TimingEventStreamValidator, _ *TimingEvent) { validator.started = false },
	} {
		t.Run(name, func(t *testing.T) {
			validator := TimingEventStreamValidator{OperationID: "restore-events", RequestSHA256: timingEventTestDigest}
			if err := validator.Accept(timingEventForTest(1, TimingEventStreamStart)); err != nil {
				t.Fatal(err)
			}
			event := clock
			mutate(&validator, &event)
			if err := validator.Accept(event); err == nil {
				t.Fatal("broken stream was accepted")
			}
		})
	}
}

func TestDecodeTimingEventRejectsUnknownAndTrailingData(t *testing.T) {
	for _, payload := range []string{
		`{"format":1,"kind":"coldsnap-timing-event","sequence":1,"event":"stream_start","operation_id":"restore-events","request_sha256":"` + timingEventTestDigest + `","unknown":true}`,
		`{"format":1,"kind":"coldsnap-timing-event","sequence":1,"event":"stream_start","operation_id":"restore-events","request_sha256":"` + timingEventTestDigest + `"} {}`,
	} {
		if _, err := DecodeTimingEvent(strings.NewReader(payload)); err == nil {
			t.Fatal("invalid event was accepted")
		}
	}
}

func timingEventForTest(sequence uint64, event string) TimingEvent {
	return TimingEvent{
		Format: TimingEventFormat, Kind: TimingEventKind, Sequence: sequence, Event: event,
		OperationID: "restore-events", RequestSHA256: timingEventTestDigest,
	}
}

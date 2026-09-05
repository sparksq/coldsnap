// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package operationtiming

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/snapshot"
)

func TestCollectorBuildsConcurrentSafeNestedSpans(t *testing.T) {
	collector := New(time.Now())
	ctx := WithCollector(context.Background(), collector)
	ctx, root := Start(ctx, "controller.restore", nil)
	if err := Measure(ctx, "artifact.verify", nil, func(context.Context) error { return nil }); err != nil {
		t.Fatal(err)
	}
	_ = Measure(ctx, "units.ready", map[string]string{"unit": "unit-0"}, func(context.Context) error {
		return errors.New("rank failed\nwith details")
	})
	root.End(nil)
	timing := collector.Export()
	if err := timing.Validate(); err != nil {
		t.Fatal(err)
	}
	if len(timing.Spans) != 3 || timing.Spans[1].Parent == "" || timing.Spans[2].Status != "error" {
		t.Fatalf("timing = %#v", timing)
	}
	if timing.Spans[2].Attributes["error"] != "rank failed with details" {
		t.Fatalf("error attribute = %q", timing.Spans[2].Attributes["error"])
	}
}

func TestCollectorMergesAdapterAndRuntimeClocksUnderControllerSpan(t *testing.T) {
	collector := New(time.Now())
	ctx := WithCollector(context.Background(), collector)
	ctx, parent := Start(ctx, "engine_adapter.process", nil)
	adapterTiming := snapshot.OperationTiming{
		Format: snapshot.OperationTimingFormat,
		Clocks: []snapshot.TimingClock{
			{ID: "engine-adapter", Source: "engine-adapter", OriginUnixNS: time.Now().UnixNano()},
			{ID: "unit-unit-0", Source: "runtime-unit", OriginUnixNS: time.Now().UnixNano(), Unit: "unit-0"},
		},
		Spans: []snapshot.TimingSpan{
			{ID: "adapter-1", Name: "adapter.restore", Clock: "engine-adapter", DurationSeconds: 2, Status: "ok"},
			{
				ID: "runtime-unit-0", Parent: "adapter-1", Name: "runtime.restore",
				Clock: "unit-unit-0", DurationSeconds: 1, Status: "ok",
			},
		},
	}
	if err := collector.Merge(adapterTiming, ParentID(ctx), "adapter"); err != nil {
		t.Fatal(err)
	}
	parent.End(nil)
	timing := collector.Export()
	if err := timing.Validate(); err != nil {
		t.Fatal(err)
	}
	spans := make(map[string]snapshot.TimingSpan)
	for _, span := range timing.Spans {
		spans[span.Name] = span
	}
	if len(timing.Clocks) != 3 || len(timing.Spans) != 3 ||
		spans["adapter.restore"].Parent != spans["engine_adapter.process"].ID ||
		spans["runtime.restore"].Parent != spans["adapter.restore"].ID {
		t.Fatalf("merged timing = %#v", timing)
	}
}

func TestCollectorStreamsBoundTimingEventsAndRetainsFinalEnvelope(t *testing.T) {
	requestDigest := "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	var output bytes.Buffer
	writer := NewNDJSONWriter(&output, "restore-events", requestDigest)
	writer.Start()
	collector := NewWithSink(time.Now(), writer)
	ctx := WithCollector(context.Background(), collector)
	ctx, root := Start(ctx, "controller.restore", nil)
	if err := Measure(ctx, "artifact.verify", nil, func(context.Context) error { return nil }); err != nil {
		t.Fatal(err)
	}
	root.End(nil)
	timing := collector.Export()
	digest, err := snapshot.OperationTimingSHA256(timing)
	if err != nil {
		t.Fatal(err)
	}
	writer.Complete("succeeded", digest)
	if err := writer.Error(); err != nil {
		t.Fatal(err)
	}
	validator := snapshot.TimingEventStreamValidator{
		OperationID: "restore-events", RequestSHA256: requestDigest,
	}
	scanner := bufio.NewScanner(bytes.NewReader(output.Bytes()))
	events := 0
	for scanner.Scan() {
		event, err := snapshot.DecodeTimingEvent(bytes.NewReader(scanner.Bytes()))
		if err != nil {
			t.Fatal(err)
		}
		if err := validator.Accept(event); err != nil {
			t.Fatal(err)
		}
		events++
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if events != 7 || !validator.Complete() {
		t.Fatalf("events=%d complete=%v\n%s", events, validator.Complete(), output.String())
	}
}

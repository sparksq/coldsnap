// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/snapshot"
)

func acceptanceFixture() (time.Time, *streamAcceptanceTiming) {
	start := time.Unix(1000, 0)
	value := &streamAcceptanceTiming{Format: 1, Measurement: startupMeasurement, Observer: "rank0",
		RequestStartedUnixNS: start.Add(9 * time.Second).UnixNano(), FirstTokenUnixNS: start.Add(10 * time.Second).UnixNano(),
		FirstTokenField: "content", RequestTTFTSeconds: 1, ResponseSeconds: 2, ResponseValidated: true}
	value.Readiness.ObserverStartedUnixNS = start.Add(time.Millisecond).UnixNano()
	value.Readiness.PortOpenUnixNS = start.Add(time.Second).UnixNano()
	value.Readiness.HTTPReadyUnixNS = start.Add(2 * time.Second).UnixNano()
	return start, value
}

func TestStartupTimingUsesHostContainerStartNotControllerClock(t *testing.T) {
	start, value := acceptanceFixture()
	_, seconds, err := startupTTFT(start.Format(time.RFC3339Nano), value)
	if err != nil || seconds != 10 {
		t.Fatalf("seconds=%v err=%v", seconds, err)
	}
	for _, mutate := range []func(*streamAcceptanceTiming){
		func(v *streamAcceptanceTiming) { v.ResponseValidated = false },
		func(v *streamAcceptanceTiming) { v.FirstTokenUnixNS = start.UnixNano() - 1 },
		func(v *streamAcceptanceTiming) { v.RequestTTFTSeconds = 3 },
		func(v *streamAcceptanceTiming) { v.Measurement = "external-v1" },
		func(v *streamAcceptanceTiming) { v.FirstTokenField = "role" },
	} {
		_, invalid := acceptanceFixture()
		mutate(invalid)
		if _, _, err := startupTTFT(start.Format(time.RFC3339Nano), invalid); err == nil {
			t.Fatal("accepted invalid measurement")
		}
	}
	if _, _, err := startupTTFT("", value); err == nil {
		t.Fatal("invented missing start time")
	}
	if _, _, err := startupTTFT(start.Format(time.RFC3339Nano), nil); err == nil {
		t.Fatal("invented missing TTFT")
	}
}

type startupRuntime struct{ start string }

func (runtime startupRuntime) Runtime(_ context.Context, _ string, request hostops.RuntimeRequest) (hostops.RuntimeResponse, error) {
	if request.Action != hostops.RuntimeWorkloadInspect || !request.IncludeStartTime {
		return hostops.RuntimeResponse{}, errors.New("expected opt-in inspection")
	}
	return hostops.RuntimeResponse{Workload: &hostops.WorkloadInfo{ID: "current-container", StartedAt: runtime.start}}, nil
}

func TestStartupTimingsAreLoggedAndExportedWithoutAdditionalInference(t *testing.T) {
	start, acceptance := acceptanceFixture()
	var output bytes.Buffer
	adapter := Adapter{Runtime: startupRuntime{start.Format(time.RFC3339Nano)}, Output: &output}
	collector := operationtiming.New(time.Now())
	ctx := operationtiming.WithCollector(context.Background(), collector)
	report := runtimeTimingReport{Rank: 0}
	report.PostRestoreResponse.Acceptance = acceptance
	unit := snapshot.LaunchUnit{ID: "unit-0", Host: "rank0"}
	if err := adapter.collectStartupTTFT(ctx, validRequest(1), unit, "container", report); err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"TTR port-open 1.000s", "TTR HTTP-ready 2.000s", "TTFT 10.000s", "response validated"} {
		if !strings.Contains(output.String(), text) {
			t.Fatal(output.String())
		}
	}
	timing := collector.Export()
	if err := timing.Validate(); err != nil {
		t.Fatal(err)
	}
	if len(timing.Spans) != 3 {
		t.Fatalf("spans=%#v", timing.Spans)
	}
	output.Reset()
	if err := adapter.collectStartupTTFT(context.Background(), validRequest(1), unit, "container", report); err != nil {
		t.Fatal(err)
	}
	if output.Len() == 0 {
		t.Fatal("logs should not require a timing collector")
	}
}

func TestUnavailableReadinessDoesNotInventZeroOrFutureTTR(t *testing.T) {
	for _, observed := range []int64{0, 1, time.Unix(2000, 0).UnixNano()} {
		start, acceptance := acceptanceFixture()
		acceptance.Readiness.PortOpenUnixNS = observed
		acceptance.Readiness.HTTPReadyUnixNS = observed
		var output bytes.Buffer
		adapter := Adapter{Runtime: startupRuntime{start.Format(time.RFC3339Nano)}, Output: &output}
		report := runtimeTimingReport{Rank: 0}
		report.PostRestoreResponse.Acceptance = acceptance
		if err := adapter.collectStartupTTFT(context.Background(), validRequest(1), snapshot.LaunchUnit{ID: "unit-0"}, "container", report); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(output.String(), "TTR port-open unavailable; TTR HTTP-ready unavailable; TTFT 10.000s") {
			t.Fatal(output.String())
		}
	}
}

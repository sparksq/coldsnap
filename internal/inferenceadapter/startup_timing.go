// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/snapshot"
)

const startupMeasurement = "rank0-acceptance-v1"

type streamAcceptanceTiming struct {
	Format               int     `json:"format"`
	Measurement          string  `json:"measurement"`
	Observer             string  `json:"observer"`
	RequestStartedUnixNS int64   `json:"request_started_unix_ns"`
	FirstTokenUnixNS     int64   `json:"first_token_unix_ns"`
	FirstTokenField      string  `json:"first_token_field"`
	RequestTTFTSeconds   float64 `json:"request_ttft_seconds"`
	ResponseSeconds      float64 `json:"response_seconds"`
	ResponseValidated    bool    `json:"response_validated"`
	PromptSHA256         string  `json:"prompt_sha256"`
	MaxTokens            int     `json:"max_tokens"`
	Readiness            struct {
		ObserverStartedUnixNS int64 `json:"observer_started_unix_ns"`
		PortOpenUnixNS        int64 `json:"port_open_unix_ns"`
		HTTPReadyUnixNS       int64 `json:"http_ready_unix_ns"`
	} `json:"readiness"`
}

func startupTTFT(startedAt string, acceptance *streamAcceptanceTiming) (time.Time, float64, error) {
	if acceptance == nil {
		return time.Time{}, 0, errors.New("runtime did not report streaming acceptance")
	}
	started, err := time.Parse(time.RFC3339Nano, startedAt)
	if err != nil || started.UnixNano() <= 0 {
		return time.Time{}, 0, errors.New("manager did not report a valid container started_at")
	}
	field := acceptance.FirstTokenField
	if acceptance.Format != 1 || acceptance.Measurement != startupMeasurement || acceptance.Observer != "rank0" ||
		!acceptance.ResponseValidated || (field != "content" && field != "reasoning" && field != "reasoning_content") ||
		acceptance.RequestStartedUnixNS < started.UnixNano() || acceptance.FirstTokenUnixNS < acceptance.RequestStartedUnixNS ||
		!finiteTimingValue(acceptance.RequestTTFTSeconds) || !finiteTimingValue(acceptance.ResponseSeconds) ||
		acceptance.ResponseSeconds < acceptance.RequestTTFTSeconds {
		return time.Time{}, 0, errors.New("runtime streaming acceptance timing is invalid")
	}
	// The request also has a monotonic timer. Do not publish an apparent TTFT
	// if the host wall clock visibly stepped during that request.
	wallRequest := float64(acceptance.FirstTokenUnixNS-acceptance.RequestStartedUnixNS) / 1e9
	if math.Abs(wallRequest-acceptance.RequestTTFTSeconds) > 0.25 {
		return time.Time{}, 0, errors.New("host clock changed during acceptance")
	}
	return started, float64(acceptance.FirstTokenUnixNS-started.UnixNano()) / 1e9, nil
}

func (adapter Adapter) collectStartupTTFT(ctx context.Context, request snapshot.Request, unit snapshot.LaunchUnit, container string, report runtimeTimingReport) error {
	if report.Rank != 0 {
		return errors.New("rank-0 runtime report has the wrong rank")
	}
	// Request the optional timestamp explicitly: older strict clients must not
	// receive new response fields during ordinary workload inspection.
	inspection, err := adapter.runtimeCall(ctx, unit.Host, hostops.RuntimeRequest{
		Action: hostops.RuntimeWorkloadInspect, Name: container, IncludeStartTime: true,
	})
	if err != nil {
		return fmt.Errorf("inspect container start: %w", err)
	}
	info := inspection.Workload
	if info == nil || info.ID == "" {
		return errors.New("manager did not report the container identity")
	}
	acceptance := report.PostRestoreResponse.Acceptance
	started, ttft, err := startupTTFT(info.StartedAt, acceptance)
	if err != nil {
		return err
	}
	collector := operationtiming.FromContext(ctx)
	clockID := "startup-" + unit.ID
	if collector != nil {
		collector.AddClock(snapshot.TimingClock{ID: clockID, Source: "runtime-unit", OriginUnixNS: started.UnixNano(), Unit: unit.ID})
	}
	add := func(name string, seconds float64) {
		if collector != nil {
			attributes := map[string]string{
				"unit": unit.ID, "host": unit.Host, "engine": request.Launch.Engine,
				"measurement": startupMeasurement, "observer": "rank0", "container_id": info.ID,
				"start_boundary": "container.started_at", "response_validated": "true",
				"first_token_field":    acceptance.FirstTokenField,
				"request_ttft_seconds": strconv.FormatFloat(acceptance.RequestTTFTSeconds, 'f', 6, 64),
			}
			if acceptance.Readiness.ObserverStartedUnixNS > 0 {
				attributes["observer_started_unix_ns"] = strconv.FormatInt(acceptance.Readiness.ObserverStartedUnixNS, 10)
			}
			if acceptance.PromptSHA256 != "" {
				attributes["prompt_sha256"] = acceptance.PromptSHA256
			}
			if acceptance.MaxTokens > 0 {
				attributes["max_tokens"] = strconv.Itoa(acceptance.MaxTokens)
			}
			collector.AddSpan(snapshot.TimingSpan{
				ID: clockID + "-" + name, Parent: operationtiming.ParentID(ctx), Name: "runtime.startup_" + name,
				Clock: clockID, DurationSeconds: seconds, Status: "ok",
				Attributes: attributes,
			})
		}
	}
	add("ttft", ttft)
	readiness := acceptance.Readiness
	ttr := func(name string, observed int64) string {
		latest := acceptance.RequestStartedUnixNS + int64((acceptance.ResponseSeconds+0.25)*1e9)
		if readiness.ObserverStartedUnixNS < started.UnixNano() || observed < readiness.ObserverStartedUnixNS || observed <= 0 || observed > latest {
			return "unavailable"
		}
		seconds := float64(observed-started.UnixNano()) / 1e9
		add(name, seconds)
		return fmt.Sprintf("%.3fs", seconds)
	}
	port := ttr("port_open", readiness.PortOpenUnixNS)
	health := ttr("http_ready", readiness.HTTPReadyUnixNS)
	fmt.Fprintf(adapter.Output, "ColdSnap startup (%s, rank 0, container start): TTR port-open %s; TTR HTTP-ready %s; TTFT %.3fs (response validated)\n", startupMeasurement, port, health, ttft)
	return nil
}

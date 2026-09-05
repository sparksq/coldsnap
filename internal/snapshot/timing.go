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
	"strings"

	"github.com/sparksq/coldsnap/internal/canonicaljson"
)

const (
	OperationTimingFormat = 1
	maximumTimingClocks   = 128
	maximumTimingSpans    = 2048
	maximumTimingAttrs    = 32
)

// OperationTiming is a manager-neutral tree of spans measured by the
// controller and by runtime processes on remote launch units. Each clock has
// its own origin; consumers must never add durations across clocks.
type OperationTiming struct {
	Format int           `json:"format"`
	Clocks []TimingClock `json:"clocks"`
	Spans  []TimingSpan  `json:"spans"`
}

func OperationTimingSHA256(timing OperationTiming) (string, error) {
	if err := timing.Validate(); err != nil {
		return "", err
	}
	payload, err := json.Marshal(timing)
	if err != nil {
		return "", err
	}
	digest, err := canonicaljson.CanonicalSHA256(payload)
	if err != nil {
		return "", err
	}
	return "sha256:" + digest, nil
}

type TimingClock struct {
	ID           string `json:"id"`
	Source       string `json:"source"`
	OriginUnixNS int64  `json:"origin_unix_ns"`
	Unit         string `json:"unit,omitempty"`
	Worker       string `json:"worker,omitempty"`
}

type TimingSpan struct {
	ID                 string            `json:"id"`
	Parent             string            `json:"parent,omitempty"`
	Name               string            `json:"name"`
	Clock              string            `json:"clock"`
	StartOffsetSeconds float64           `json:"start_offset_seconds"`
	DurationSeconds    float64           `json:"duration_seconds"`
	Status             string            `json:"status"`
	Attributes         map[string]string `json:"attributes,omitempty"`
}

func (timing OperationTiming) Validate() error {
	if timing.Format != OperationTimingFormat {
		return errors.New("unsupported ColdSnap operation timing format")
	}
	if len(timing.Clocks) == 0 || len(timing.Clocks) > maximumTimingClocks {
		return errors.New("ColdSnap operation timing clock inventory is invalid")
	}
	if len(timing.Spans) == 0 || len(timing.Spans) > maximumTimingSpans {
		return errors.New("ColdSnap operation timing span inventory is invalid")
	}
	clocks := make(map[string]bool, len(timing.Clocks))
	for _, clock := range timing.Clocks {
		if !idPattern.MatchString(clock.ID) || clocks[clock.ID] ||
			!safeTimingText(clock.Source, 64) || clock.OriginUnixNS <= 0 ||
			(clock.Unit != "" && !idPattern.MatchString(clock.Unit)) ||
			(clock.Worker != "" && !idPattern.MatchString(clock.Worker)) {
			return errors.New("ColdSnap operation timing clock is invalid")
		}
		clocks[clock.ID] = true
	}
	spans := make(map[string]TimingSpan, len(timing.Spans))
	for _, span := range timing.Spans {
		if !idPattern.MatchString(span.ID) || spans[span.ID].ID != "" ||
			!safeTimingText(span.Name, 128) || !clocks[span.Clock] ||
			!finiteNonnegative(span.StartOffsetSeconds) || !finiteNonnegative(span.DurationSeconds) ||
			(span.Status != "ok" && span.Status != "error") || len(span.Attributes) > maximumTimingAttrs {
			return errors.New("ColdSnap operation timing span is invalid")
		}
		for key, value := range span.Attributes {
			if !idPattern.MatchString(key) || !safeTimingText(value, 512) {
				return errors.New("ColdSnap operation timing span attributes are invalid")
			}
		}
		spans[span.ID] = span
	}
	for _, span := range timing.Spans {
		if span.Parent == "" {
			continue
		}
		if span.Parent == span.ID || spans[span.Parent].ID == "" {
			return errors.New("ColdSnap operation timing span parent is invalid")
		}
		seen := map[string]bool{span.ID: true}
		for parent := span.Parent; parent != ""; parent = spans[parent].Parent {
			if seen[parent] {
				return errors.New("ColdSnap operation timing span tree contains a cycle")
			}
			seen[parent] = true
		}
	}
	return nil
}

func DecodeOperationTiming(reader io.Reader) (OperationTiming, error) {
	decoder := json.NewDecoder(io.LimitReader(reader, 8<<20))
	decoder.DisallowUnknownFields()
	var timing OperationTiming
	if err := decoder.Decode(&timing); err != nil {
		return OperationTiming{}, fmt.Errorf("decode ColdSnap operation timing: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return OperationTiming{}, errors.New("decode ColdSnap operation timing: trailing JSON")
	}
	if err := timing.Validate(); err != nil {
		return OperationTiming{}, fmt.Errorf("validate ColdSnap operation timing: %w", err)
	}
	return timing, nil
}

func finiteNonnegative(value float64) bool {
	return value >= 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func safeTimingText(value string, maximum int) bool {
	if value == "" || len(value) > maximum || strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if character < 0x20 || character == 0x7f {
			return false
		}
	}
	return true
}

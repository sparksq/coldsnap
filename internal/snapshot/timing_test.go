// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import "testing"

func TestOperationTimingAcceptsNestedMixedClockSpans(t *testing.T) {
	timing := OperationTiming{
		Format: OperationTimingFormat,
		Clocks: []TimingClock{
			{ID: "controller", Source: "controller", OriginUnixNS: 1},
			{ID: "unit-unit-0", Source: "runtime-unit", OriginUnixNS: 2, Unit: "unit-0"},
		},
		Spans: []TimingSpan{
			{ID: "controller-1", Name: "controller.restore", Clock: "controller", DurationSeconds: 10, Status: "ok"},
			{
				ID: "runtime-unit-0", Parent: "controller-1", Name: "runtime.restore",
				Clock: "unit-unit-0", DurationSeconds: 9, Status: "ok",
				Attributes: map[string]string{"unit": "unit-0"},
			},
		},
	}
	if err := timing.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestOperationTimingRejectsCyclesAndUnknownClocks(t *testing.T) {
	for name, timing := range map[string]OperationTiming{
		"cycle": {
			Format: OperationTimingFormat,
			Clocks: []TimingClock{{ID: "controller", Source: "controller", OriginUnixNS: 1}},
			Spans: []TimingSpan{
				{ID: "a", Parent: "b", Name: "a", Clock: "controller", Status: "ok"},
				{ID: "b", Parent: "a", Name: "b", Clock: "controller", Status: "ok"},
			},
		},
		"clock": {
			Format: OperationTimingFormat,
			Clocks: []TimingClock{{ID: "controller", Source: "controller", OriginUnixNS: 1}},
			Spans:  []TimingSpan{{ID: "a", Name: "a", Clock: "missing", Status: "ok"}},
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := timing.Validate(); err == nil {
				t.Fatal("invalid timing was accepted")
			}
		})
	}
}

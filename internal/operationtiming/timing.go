// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package operationtiming

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/sparksq/coldsnap/internal/snapshot"
)

const controllerClock = "controller"

type collectorKey struct{}
type parentKey struct{}

type Collector struct {
	origin     time.Time
	localClock string
	sink       EventSink
	mu         sync.Mutex
	next       uint64
	clocks     map[string]snapshot.TimingClock
	spans      []snapshot.TimingSpan
}

type Span struct {
	collector *Collector
	id        string
	parent    string
	name      string
	clock     string
	started   time.Time
	attrs     map[string]string
	once      sync.Once
}

func New(started time.Time) *Collector {
	return NewForSource(started, controllerClock, "controller")
}

func NewWithSink(started time.Time, sink EventSink) *Collector {
	return NewForSourceWithSink(started, controllerClock, "controller", sink)
}

func NewForSource(started time.Time, clockID, source string) *Collector {
	return NewForSourceWithSink(started, clockID, source, nil)
}

func NewForSourceWithSink(started time.Time, clockID, source string, sink EventSink) *Collector {
	collector := &Collector{
		origin: started, localClock: clockID,
		sink: sink,
		clocks: map[string]snapshot.TimingClock{
			clockID: {
				ID: clockID, Source: source, OriginUnixNS: started.UnixNano(),
			},
		},
	}
	if sink != nil {
		sink.Clock(collector.clocks[clockID])
	}
	return collector
}

func WithCollector(ctx context.Context, collector *Collector) context.Context {
	return context.WithValue(ctx, collectorKey{}, collector)
}

func FromContext(ctx context.Context) *Collector {
	collector, _ := ctx.Value(collectorKey{}).(*Collector)
	return collector
}

func Start(ctx context.Context, name string, attributes map[string]string) (context.Context, *Span) {
	collector := FromContext(ctx)
	if collector == nil {
		return ctx, &Span{}
	}
	collector.mu.Lock()
	collector.next++
	id := fmt.Sprintf("%s-%d", collector.localClock, collector.next)
	collector.mu.Unlock()
	parent, _ := ctx.Value(parentKey{}).(string)
	span := &Span{
		collector: collector, id: id, parent: parent, name: name,
		clock: collector.localClock, started: time.Now(), attrs: cloneAttributes(attributes),
	}
	if collector.sink != nil {
		collector.sink.SpanStarted(snapshot.TimingEventSpan{
			ID: span.id, Parent: span.parent, Name: span.name, Clock: span.clock,
			StartOffsetSeconds: span.started.Sub(collector.origin).Seconds(),
			Attributes:         cloneAttributes(span.attrs),
		})
	}
	return context.WithValue(ctx, parentKey{}, id), span
}

func Measure(
	ctx context.Context,
	name string,
	attributes map[string]string,
	action func(context.Context) error,
) (operationErr error) {
	ctx, span := Start(ctx, name, attributes)
	defer func() { span.End(operationErr) }()
	return action(ctx)
}

func MeasureValue[T any](
	ctx context.Context,
	name string,
	attributes map[string]string,
	action func(context.Context) (T, error),
) (value T, operationErr error) {
	ctx, span := Start(ctx, name, attributes)
	defer func() { span.End(operationErr) }()
	return action(ctx)
}

func ParentID(ctx context.Context) string {
	parent, _ := ctx.Value(parentKey{}).(string)
	return parent
}

func (span *Span) End(operationError error) {
	if span.collector == nil {
		return
	}
	span.once.Do(func() {
		status := "ok"
		attributes := cloneAttributes(span.attrs)
		if operationError != nil {
			status = "error"
			if attributes == nil {
				attributes = make(map[string]string)
			}
			attributes["error"] = boundedText(operationError.Error(), 512)
		}
		finished := time.Now()
		completed := snapshot.TimingSpan{
			ID: span.id, Parent: span.parent, Name: span.name, Clock: span.clock,
			StartOffsetSeconds: span.started.Sub(span.collector.origin).Seconds(),
			DurationSeconds:    finished.Sub(span.started).Seconds(), Status: status,
			Attributes: attributes,
		}
		span.collector.mu.Lock()
		span.collector.spans = append(span.collector.spans, completed)
		span.collector.mu.Unlock()
		if span.collector.sink != nil {
			span.collector.sink.SpanCompleted(completed)
		}
	})
}

func (collector *Collector) AddClock(clock snapshot.TimingClock) {
	collector.addClock(clock, true)
}

func (collector *Collector) addClock(clock snapshot.TimingClock, emit bool) {
	collector.mu.Lock()
	collector.clocks[clock.ID] = clock
	collector.mu.Unlock()
	if emit && collector.sink != nil {
		collector.sink.Clock(clock)
	}
}

func (collector *Collector) AddSpan(span snapshot.TimingSpan) {
	collector.addSpan(span, true)
}

func (collector *Collector) addSpan(span snapshot.TimingSpan, emit bool) {
	collector.mu.Lock()
	collector.spans = append(collector.spans, span)
	collector.mu.Unlock()
	if emit && collector.sink != nil {
		collector.sink.SpanCompleted(span)
	}
}

func (collector *Collector) Merge(timing snapshot.OperationTiming, parent, prefix string) error {
	return collector.merge(timing, parent, prefix, true)
}

func (collector *Collector) MergeSilent(timing snapshot.OperationTiming, parent, prefix string) error {
	return collector.merge(timing, parent, prefix, false)
}

func (collector *Collector) merge(timing snapshot.OperationTiming, parent, prefix string, emit bool) error {
	if err := timing.Validate(); err != nil {
		return err
	}
	if prefix == "" {
		return fmt.Errorf("timing merge prefix is required")
	}
	clockIDs := make(map[string]string, len(timing.Clocks))
	for _, clock := range timing.Clocks {
		clockIDs[clock.ID] = prefix + "-" + clock.ID
		clock.ID = clockIDs[clock.ID]
		collector.addClock(clock, emit)
	}
	spanIDs := make(map[string]string, len(timing.Spans))
	for _, span := range timing.Spans {
		spanIDs[span.ID] = prefix + "-" + span.ID
	}
	for _, span := range timing.Spans {
		span.ID = spanIDs[span.ID]
		if span.Parent == "" {
			span.Parent = parent
		} else {
			span.Parent = spanIDs[span.Parent]
		}
		span.Clock = clockIDs[span.Clock]
		collector.addSpan(span, emit)
	}
	return nil
}

func (collector *Collector) EventSink() EventSink { return collector.sink }

func (collector *Collector) Export() snapshot.OperationTiming {
	collector.mu.Lock()
	defer collector.mu.Unlock()
	clocks := make([]snapshot.TimingClock, 0, len(collector.clocks))
	for _, clock := range collector.clocks {
		clocks = append(clocks, clock)
	}
	sort.Slice(clocks, func(left, right int) bool { return clocks[left].ID < clocks[right].ID })
	spans := append([]snapshot.TimingSpan(nil), collector.spans...)
	sort.SliceStable(spans, func(left, right int) bool {
		if spans[left].Clock == spans[right].Clock {
			return spans[left].StartOffsetSeconds < spans[right].StartOffsetSeconds
		}
		return spans[left].Clock < spans[right].Clock
	})
	return snapshot.OperationTiming{Format: snapshot.OperationTimingFormat, Clocks: clocks, Spans: spans}
}

func cloneAttributes(attributes map[string]string) map[string]string {
	if len(attributes) == 0 {
		return nil
	}
	result := make(map[string]string, len(attributes))
	for key, value := range attributes {
		result[key] = value
	}
	return result
}

func boundedText(value string, maximum int) string {
	value = strings.Map(func(character rune) rune {
		if character < 0x20 || character == 0x7f {
			return ' '
		}
		return character
	}, value)
	value = strings.TrimSpace(value)
	if value == "" {
		value = "operation failed"
	}
	if len(value) <= maximum {
		return value
	}
	value = value[:maximum]
	for !utf8.ValidString(value) {
		value = value[:len(value)-1]
	}
	return value
}

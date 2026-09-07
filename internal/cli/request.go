// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package cli

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"syscall"
	"time"

	"github.com/sparksq/coldsnap/internal/engineadapter"
	"github.com/sparksq/coldsnap/internal/fsutil"
	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/snapshot"
)

const maximumRequestBytes = 16 << 20

type RequestRunner interface {
	RunRequest(context.Context, snapshot.Request, Streams) error
}

type RequestPreparer interface {
	PrepareRequest(context.Context, snapshot.Request, Streams) error
}

// EngineAdapterRunner keeps the public ColdSnap CLI engine-neutral. Each
// engine adapter owns its lifecycle details behind the same strict JSON
// contract, capture/restore commands, and recipe schema.
type EngineAdapterRunner struct{}

func (EngineAdapterRunner) RunRequest(
	ctx context.Context,
	request snapshot.Request,
	streams Streams,
) error {
	return runEngineAdapter(ctx, request, streams, false)
}

func (EngineAdapterRunner) PrepareRequest(
	ctx context.Context,
	request snapshot.Request,
	streams Streams,
) error {
	return runEngineAdapter(ctx, request, streams, true)
}

func runEngineAdapter(ctx context.Context, request snapshot.Request, streams Streams, prepareOnly bool) error {
	ctx, processSpan := operationtiming.Start(ctx, "engine_adapter.process", map[string]string{
		"engine": request.Launch.Engine,
	})
	var operationErr error
	defer func() { processSpan.End(operationErr) }()
	descriptor, err := engineadapter.Lookup(request.Launch.Engine)
	if err != nil {
		operationErr = err
		return err
	}
	if !slices.Contains(descriptor.Operations, request.Operation) {
		operationErr = fmt.Errorf(
			"%s engine adapter does not support %s",
			request.Launch.Engine,
			request.Operation,
		)
		return operationErr
	}
	if prepareOnly && !descriptor.PrepareRestore {
		operationErr = fmt.Errorf("%s engine adapter does not support prepare-only restore", request.Launch.Engine)
		return operationErr
	}
	executable := descriptor.Executable
	if configured := os.Getenv(descriptor.Environment); configured != "" {
		executable = configured
	}
	resolved, err := exec.LookPath(executable)
	if err != nil {
		operationErr = fmt.Errorf(
			"locate %s engine adapter %q: %w (install it or set %s)",
			request.Launch.Engine,
			executable,
			err,
			descriptor.Environment,
		)
		return operationErr
	}
	arguments := []string{request.Operation, "--request-json", "-"}
	if prepareOnly {
		arguments = append(arguments, "--prepare-only")
	}
	timingRoot, err := os.MkdirTemp("", "coldsnap-engine-adapter-timing-")
	if err != nil {
		operationErr = fmt.Errorf("create engine adapter timing directory: %w", err)
		return operationErr
	}
	defer os.RemoveAll(timingRoot)
	timingPath := filepath.Join(timingRoot, "operation.json")
	arguments = append(arguments, "--timing-json", timingPath)
	command := exec.CommandContext(ctx, resolved, arguments...) // #nosec G204 -- resolved executable, fixed arguments.
	// The adapter owns remote cleanup. CommandContext's default immediate Kill
	// skips its defers and can strand the coordinator and serving containers.
	configureAdapterShutdown(command, 4*time.Minute)
	input, err := snapshot.Encode(request)
	if err != nil {
		operationErr = fmt.Errorf("encode ColdSnap request for engine adapter: %w", err)
		return operationErr
	}
	command.Stdin = bytes.NewReader(input)
	command.Stdout = streams.Stdout
	command.Stderr = streams.Stderr
	collector := operationtiming.FromContext(ctx)
	var streamResult adapterTimingStreamResult
	var streamErr error
	if collector != nil && collector.EventSink() != nil {
		reader, writer, pipeErr := os.Pipe()
		if pipeErr != nil {
			fmt.Fprintf(streams.Stderr, "ColdSnap timing warning: create adapter event pipe: %v\n", pipeErr)
		} else {
			arguments = append(arguments, "--timing-events-fd", strconv.Itoa(3+len(command.ExtraFiles)))
			command.Args = append([]string{resolved}, arguments...)
			command.ExtraFiles = append(command.ExtraFiles, writer)
			streamDone := make(chan adapterTimingStreamResult, 1)
			go func() {
				streamDone <- forwardEngineAdapterTimingEvents(
					reader, collector.EventSink(), request, operationtiming.ParentID(ctx), "adapter",
				)
			}()
			startErr := command.Start()
			_ = writer.Close()
			if startErr == nil {
				startErr = command.Wait()
			}
			streamResult = <-streamDone
			_ = reader.Close()
			streamErr = streamResult.Error
			runErr := startErr
			timingDigest, timingErr := mergeEngineAdapterTiming(ctx, timingPath, true)
			if streamErr != nil {
				fmt.Fprintf(streams.Stderr, "ColdSnap timing warning: adapter event stream: %v\n", streamErr)
			} else if streamResult.TimingSHA256 != timingDigest {
				fmt.Fprintf(streams.Stderr, "ColdSnap timing warning: adapter event stream digest differs from final timing\n")
			} else if (runErr == nil && streamResult.State != "succeeded") ||
				(runErr != nil && streamResult.State != "failed") {
				fmt.Fprintf(streams.Stderr, "ColdSnap timing warning: adapter event stream state differs from process result\n")
			}
			operationErr = finishEngineAdapterRun(request, runErr, timingErr)
			return operationErr
		}
	}
	runErr := command.Run()
	_, timingErr := mergeEngineAdapterTiming(ctx, timingPath, false)
	operationErr = finishEngineAdapterRun(request, runErr, timingErr)
	return operationErr
}

func configureAdapterShutdown(command *exec.Cmd, grace time.Duration) {
	command.Cancel = func() error { return command.Process.Signal(syscall.SIGTERM) }
	// Bound a stuck adapter while allowing its existing remote cleanup budgets
	// (including capture-path ownership repair) to finish before killing it.
	command.WaitDelay = grace
}

func finishEngineAdapterRun(request snapshot.Request, runErr, timingErr error) error {
	if runErr != nil {
		var exitError *exec.ExitError
		if errors.As(runErr, &exitError) {
			if status, ok := exitError.Sys().(syscall.WaitStatus); ok && status.Signaled() {
				runErr = fmt.Errorf("adapter terminated by signal; remote cleanup is unconfirmed: %w", runErr)
			}
		}
		operationErr := fmt.Errorf("%s engine adapter: %w", request.Launch.Engine, runErr)
		if timingErr != nil {
			operationErr = fmt.Errorf("%v; read engine adapter timing: %w", operationErr, timingErr)
		}
		return operationErr
	}
	if timingErr != nil {
		return fmt.Errorf("read engine adapter timing: %w", timingErr)
	}
	return nil
}

func mergeEngineAdapterTiming(ctx context.Context, path string, silent bool) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	timing, err := snapshot.DecodeOperationTiming(file)
	if err != nil {
		return "", err
	}
	digest, err := snapshot.OperationTimingSHA256(timing)
	if err != nil {
		return "", err
	}
	collector := operationtiming.FromContext(ctx)
	if collector == nil {
		return "", errors.New("controller operation timing collector is unavailable")
	}
	if silent {
		err = collector.MergeSilent(timing, operationtiming.ParentID(ctx), "adapter")
	} else {
		err = collector.Merge(timing, operationtiming.ParentID(ctx), "adapter")
	}
	return digest, err
}

type adapterTimingStreamResult struct {
	TimingSHA256 string
	State        string
	Error        error
}

func forwardEngineAdapterTimingEvents(
	reader io.Reader,
	sink operationtiming.EventSink,
	request snapshot.Request,
	parent, prefix string,
) adapterTimingStreamResult {
	digest, err := snapshot.RequestSHA256(request)
	if err != nil {
		return adapterTimingStreamResult{Error: err}
	}
	validator := snapshot.TimingEventStreamValidator{
		OperationID: request.ID, RequestSHA256: digest,
	}
	result := adapterTimingStreamResult{}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), snapshot.MaximumTimingEventBytes)
	for scanner.Scan() {
		if result.Error != nil {
			continue
		}
		event, decodeErr := snapshot.DecodeTimingEvent(bytes.NewReader(scanner.Bytes()))
		if decodeErr != nil {
			result.Error = decodeErr
			continue
		}
		if acceptErr := validator.Accept(event); acceptErr != nil {
			result.Error = acceptErr
			continue
		}
		switch event.Event {
		case snapshot.TimingEventClock:
			clock := *event.Clock
			clock.ID = prefix + "-" + clock.ID
			sink.Clock(clock)
		case snapshot.TimingEventSpanStart:
			span := *event.Span
			span.ID = prefix + "-" + span.ID
			span.Clock = prefix + "-" + span.Clock
			if span.Parent == "" {
				span.Parent = parent
			} else {
				span.Parent = prefix + "-" + span.Parent
			}
			sink.SpanStarted(span)
		case snapshot.TimingEventSpanComplete:
			span := *event.Span
			span.ID = prefix + "-" + span.ID
			span.Clock = prefix + "-" + span.Clock
			if span.Parent == "" {
				span.Parent = parent
			} else {
				span.Parent = prefix + "-" + span.Parent
			}
			sink.SpanCompleted(snapshot.TimingSpan{
				ID: span.ID, Parent: span.Parent, Name: span.Name, Clock: span.Clock,
				StartOffsetSeconds: span.StartOffsetSeconds,
				DurationSeconds:    *span.DurationSeconds, Status: span.Status,
				Attributes: span.Attributes,
			})
		case snapshot.TimingEventStreamComplete:
			result.TimingSHA256 = event.TimingSHA256
			result.State = event.State
		}
	}
	if scanErr := scanner.Err(); scanErr != nil && result.Error == nil {
		result.Error = scanErr
	}
	if result.Error == nil && !validator.Complete() {
		result.Error = errors.New("adapter timing event stream ended before completion")
	}
	return result
}

func runSnapshotRequest(
	ctx context.Context,
	dependencies Dependencies,
	path string,
	expectedOperation string,
	prepareOnly bool,
	receiptPath string,
	timingEventsPath string,
) error {
	reader, closeReader, err := requestReader(dependencies.Stdin, path)
	if err != nil {
		return err
	}
	if closeReader != nil {
		defer closeReader()
	}
	payload, err := io.ReadAll(io.LimitReader(reader, maximumRequestBytes+1))
	if err != nil {
		return fmt.Errorf("read ColdSnap request: %w", err)
	}
	if len(payload) > maximumRequestBytes {
		return fmt.Errorf("ColdSnap request exceeds %d bytes", maximumRequestBytes)
	}
	request, err := snapshot.DecodeRequest(bytes.NewReader(payload))
	if err != nil {
		return err
	}
	if request.Operation != expectedOperation {
		return fmt.Errorf(
			"ColdSnap request operation %q does not match command %q",
			request.Operation,
			expectedOperation,
		)
	}
	if receiptPath != "" && receiptPath != "-" {
		if err := validateReceiptDestination(receiptPath, request); err != nil {
			return err
		}
	}
	if timingEventsPath != "" && receiptPath == "" {
		return errors.New("--timing-events requires --receipt-json")
	}
	if receiptPath == "-" && timingEventsPath == "-" {
		return errors.New("--receipt-json and --timing-events cannot both use stdout")
	}
	if receiptPath != "" && receiptPath == timingEventsPath {
		return errors.New("--receipt-json and --timing-events must use different destinations")
	}
	requestDigest, err := snapshot.RequestSHA256(request)
	if err != nil {
		return fmt.Errorf("digest ColdSnap request: %w", err)
	}
	eventOutput, closeEventOutput, err := timingEventOutput(timingEventsPath, dependencies.Stdout)
	if err != nil {
		return err
	}
	if closeEventOutput != nil {
		defer func() {
			if closeErr := closeEventOutput(); closeErr != nil {
				fmt.Fprintf(dependencies.Stderr, "ColdSnap timing warning: close event stream: %v\n", closeErr)
			}
		}()
	}
	streams := Streams{
		Stdin: dependencies.Stdin, Stdout: dependencies.Stdout, Stderr: dependencies.Stderr,
	}
	if receiptPath == "-" || timingEventsPath == "-" {
		// Reserve stdout for the one machine-readable result envelope. Adapter
		// progress remains visible and follows the usual logging controls.
		streams.Stdout = dependencies.Stderr
	}
	fmt.Fprintf(
		streams.Stderr,
		"ColdSnap: snapshot driver %s; running %s\n",
		request.Driver.ID,
		request.Operation,
	)
	started := time.Now()
	var eventWriter *operationtiming.NDJSONWriter
	var collector *operationtiming.Collector
	if eventOutput != nil {
		eventWriter = operationtiming.NewNDJSONWriter(eventOutput, request.ID, requestDigest)
		eventWriter.Start()
		collector = operationtiming.NewWithSink(started, eventWriter)
	} else {
		collector = operationtiming.New(started)
	}
	ctx = operationtiming.WithCollector(ctx, collector)
	ctx, controllerSpan := operationtiming.Start(ctx, "controller."+request.Operation, map[string]string{
		"engine": request.Launch.Engine, "snapshot_driver": request.Driver.ID,
	})
	var operationErr error
	if prepareOnly {
		preparer, ok := dependencies.RequestRunner.(RequestPreparer)
		if !ok {
			operationErr = fmt.Errorf("configured request runner does not support prepare-only restore")
		} else {
			operationErr = preparer.PrepareRequest(ctx, request, streams)
		}
	} else {
		operationErr = dependencies.RequestRunner.RunRequest(ctx, request, streams)
	}
	controllerSpan.End(operationErr)
	timing := collector.Export()
	if eventWriter != nil {
		state := "succeeded"
		if operationErr != nil {
			state = "failed"
		}
		if digest, digestErr := snapshot.OperationTimingSHA256(timing); digestErr == nil {
			eventWriter.Complete(state, digest)
		} else {
			fmt.Fprintf(streams.Stderr, "ColdSnap timing warning: cannot complete event stream: %v\n", digestErr)
		}
		if eventErr := eventWriter.Error(); eventErr != nil {
			fmt.Fprintf(streams.Stderr, "ColdSnap timing warning: %v\n", eventErr)
		}
	}
	if receiptPath == "" {
		return operationErr
	}
	receipt, receiptErr := snapshot.NewOperationReceipt(
		request, prepareOnly, started, time.Now(), operationErr, timing,
	)
	if receiptErr == nil {
		receiptErr = publishOperationReceipt(receiptPath, receipt, dependencies.Stdout)
	}
	if operationErr != nil && receiptErr != nil {
		return fmt.Errorf("%v; write operation receipt: %w", operationErr, receiptErr)
	}
	if receiptErr != nil {
		return fmt.Errorf("write operation receipt: %w", receiptErr)
	}
	return operationErr
}

func timingEventOutput(path string, stdout io.Writer) (io.Writer, func() error, error) {
	if path == "" {
		return nil, nil, nil
	}
	if path == "-" {
		return stdout, nil, nil
	}
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, nil, errors.New("--timing-events must be an absolute clean path, or -")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, nil, fmt.Errorf("create timing event parent: %w", err)
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return nil, nil, fmt.Errorf("create timing event stream %s: %w", path, err)
	}
	return file, file.Close, nil
}

func validateReceiptDestination(path string, request snapshot.Request) error {
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("inspect existing receipt %s: %w", path, err)
	}
	existing, decodeErr := snapshot.DecodeOperationReceipt(file)
	closeErr := file.Close()
	if decodeErr != nil {
		return fmt.Errorf("read existing receipt %s: %w", path, decodeErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close existing receipt %s: %w", path, closeErr)
	}
	digest, err := snapshot.RequestSHA256(request)
	if err != nil {
		return fmt.Errorf("digest ColdSnap request: %w", err)
	}
	if existing.OperationID != request.ID || existing.Operation != request.Operation || existing.RequestSHA256 != digest {
		return fmt.Errorf("refusing to replace receipt %s for a different request", path)
	}
	return nil
}

func publishOperationReceipt(path string, receipt snapshot.OperationReceipt, stdout io.Writer) error {
	if path == "-" {
		encoder := json.NewEncoder(stdout)
		encoder.SetEscapeHTML(false)
		return encoder.Encode(receipt)
	}
	if existingFile, err := os.Open(path); err == nil {
		existing, decodeErr := snapshot.DecodeOperationReceipt(existingFile)
		closeErr := existingFile.Close()
		if decodeErr != nil {
			return fmt.Errorf("read existing receipt %s: %w", path, decodeErr)
		}
		if closeErr != nil {
			return fmt.Errorf("close existing receipt %s: %w", path, closeErr)
		}
		if !existing.SameRequest(receipt) {
			return fmt.Errorf("refusing to replace receipt %s for a different request", path)
		}
		if existing.State == "succeeded" {
			return nil
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("inspect existing receipt %s: %w", path, err)
	}
	return fsutil.WriteJSONAtomic(path, receipt, 0o600)
}

func requestReader(stdin io.Reader, path string) (io.Reader, func() error, error) {
	if path == "-" {
		return stdin, nil, nil
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open ColdSnap request: %w", err)
	}
	return file, file.Close, nil
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Package adaptercli implements the engine-adapter process boundary shared by
// vLLM, SGLang, and future engine integrations.
package adaptercli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"syscall"
	"time"

	"github.com/sparksq/coldsnap/internal/fsutil"
	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/hostprovider"
	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/payloadvalidation"
	"github.com/sparksq/coldsnap/internal/snapshot"
)

// Adapter is the engine-owned implementation behind the common controller
// process, request, provider, cancellation, and credential boundary.
type Adapter interface {
	Run(context.Context, snapshot.Request) error
	PrepareRestore(context.Context, snapshot.Request, io.Writer) error
}

type Factory func(hostops.Remote, string, time.Duration, io.Writer) Adapter

type Config struct {
	Command          string
	Engine           string
	Operations       []string
	PrepareRestore   bool
	DefaultTimeout   time.Duration
	NewAdapter       Factory
	UnavailableError string
}

// Execute parses and runs one adapter invocation. The caller owns ctx signal
// handling so tests and orchestrators can exercise cancellation deterministically.
func Execute(
	ctx context.Context,
	config Config,
	arguments []string,
	stdin io.Reader,
	stdout io.Writer,
	stderr io.Writer,
) error {
	if err := validateConfig(config); err != nil {
		return err
	}
	// payload-verify is an internal, engine-neutral worker helper rather than
	// an adapter lifecycle operation. Both adapter binaries expose the exact
	// same Go implementation so a manager can stage the architecture-matched
	// binary it already resolved for the selected engine.
	if len(arguments) > 0 && arguments[0] == "payload-verify" {
		return payloadvalidation.Execute(arguments[1:], stdout, stderr)
	}
	if len(arguments) < 1 || !slices.Contains(config.Operations, arguments[0]) {
		if config.UnavailableError != "" {
			return errors.New(config.UnavailableError)
		}
		return fmt.Errorf("usage: %s %s --request-json <path|->", config.Command, operationUsage(config.Operations))
	}
	operation := arguments[0]
	flags := flag.NewFlagSet(config.Command+" "+operation, flag.ContinueOnError)
	flags.SetOutput(stderr)
	requestPath := flags.String("request-json", "", "ColdSnap request path, or - for stdin")
	stateRoot := flags.String("state-root", os.Getenv("COLDSNAP_REMOTE_STATE_ROOT"), "remote ColdSnap state root (default: controller user's cache directory)")
	timeout := flags.Duration("timeout", environmentDuration("COLDSNAP_ADAPTER_TIMEOUT", config.DefaultTimeout), "operation timeout")
	prepareOnly := flags.Bool("prepare-only", false, "verify and prepare restore assets without launching")
	timingPath := flags.String("timing-json", "", "write a manager-neutral operation timing envelope")
	timingEventsFD := flags.Int("timing-events-fd", -1, "write internal timing NDJSON to an inherited file descriptor")
	if err := flags.Parse(arguments[1:]); err != nil {
		return err
	}
	if *requestPath == "" || flags.NArg() != 0 {
		return errors.New("--request-json is required and positional arguments are unsupported")
	}
	if *prepareOnly && (operation != "restore" || !config.PrepareRestore) {
		return errors.New("--prepare-only is unsupported for this adapter operation")
	}
	if *timeout <= 0 {
		return errors.New("--timeout must be positive")
	}
	if *timingPath != "" && (!filepath.IsAbs(*timingPath) || filepath.Clean(*timingPath) != *timingPath) {
		return errors.New("--timing-json must be an absolute clean path")
	}
	if *timingEventsFD != -1 && *timingEventsFD < 3 {
		return errors.New("--timing-events-fd must be an inherited descriptor numbered 3 or higher")
	}
	if err := resolveStateRoot(stateRoot); err != nil {
		return err
	}
	reader := stdin
	if *requestPath != "-" {
		file, err := os.Open(*requestPath)
		if err != nil {
			return err
		}
		defer file.Close()
		reader = file
	}
	request, err := snapshot.DecodeRequest(reader)
	if err != nil {
		return err
	}
	if request.Operation != operation {
		return fmt.Errorf("request operation %q differs from command %q", request.Operation, operation)
	}
	if request.Launch.Engine != config.Engine {
		return fmt.Errorf("%s cannot run engine %q", config.Command, request.Launch.Engine)
	}
	started := time.Now()
	var eventWriter *operationtiming.NDJSONWriter
	var eventFile *os.File
	var collector *operationtiming.Collector
	if *timingEventsFD >= 3 {
		digest, digestErr := snapshot.RequestSHA256(request)
		if digestErr != nil {
			return fmt.Errorf("digest ColdSnap request: %w", digestErr)
		}
		eventFile = os.NewFile(uintptr(*timingEventsFD), "coldsnap-timing-events")
		if eventFile == nil {
			return errors.New("inherited timing event descriptor is unavailable")
		}
		syscall.CloseOnExec(*timingEventsFD)
		defer eventFile.Close()
		eventWriter = operationtiming.NewNDJSONWriter(eventFile, request.ID, digest)
		eventWriter.Start()
		collector = operationtiming.NewForSourceWithSink(started, "engine-adapter", "engine-adapter", eventWriter)
	} else {
		collector = operationtiming.NewForSource(started, "engine-adapter", "engine-adapter")
	}
	ctx = operationtiming.WithCollector(ctx, collector)
	ctx, processSpan := operationtiming.Start(ctx, "engine_adapter."+operation, map[string]string{
		"engine": config.Engine,
	})
	remote, operationErr := newRemote(ctx, request.ID, operation)
	if operationErr == nil {
		fmt.Fprintf(stderr, "%s: host provider=manager\n", config.Command)
		progressOutput := adapterProgressOutput(*prepareOnly, stdout, stderr)
		adapter := config.NewAdapter(remote, *stateRoot, *timeout, progressOutput)
		if adapter == nil {
			operationErr = errors.New("engine adapter factory returned nil")
		} else {
			operationErr = runAdapterOperation(ctx, adapter, request, *prepareOnly, stdout)
		}
	}
	processSpan.End(operationErr)
	timing := collector.Export()
	if eventWriter != nil {
		state := "succeeded"
		if operationErr != nil {
			state = "failed"
		}
		if digest, digestErr := snapshot.OperationTimingSHA256(timing); digestErr == nil {
			eventWriter.Complete(state, digest)
		} else {
			fmt.Fprintf(stderr, "%s timing warning: cannot complete event stream: %v\n", config.Command, digestErr)
		}
		if eventErr := eventWriter.Error(); eventErr != nil {
			fmt.Fprintf(stderr, "%s timing warning: %v\n", config.Command, eventErr)
		}
	}
	var timingErr error
	if *timingPath != "" {
		timingErr = fsutil.WriteJSONAtomic(*timingPath, timing, 0o600)
	}
	if operationErr != nil && timingErr != nil {
		return fmt.Errorf("%v; write engine adapter timing: %w", operationErr, timingErr)
	}
	if timingErr != nil {
		return fmt.Errorf("write engine adapter timing: %w", timingErr)
	}
	return operationErr
}

func adapterProgressOutput(prepareOnly bool, stdout, stderr io.Writer) io.Writer {
	if prepareOnly {
		// Prepare-only stdout is a machine-readable contract consumed by the
		// manager. Keep compatibility warnings and other human progress on
		// stderr so a useful warning cannot corrupt the JSON receipt.
		return stderr
	}
	return stdout
}

func runAdapterOperation(
	ctx context.Context,
	adapter Adapter,
	request snapshot.Request,
	prepareOnly bool,
	receiptOutput io.Writer,
) error {
	if prepareOnly {
		return adapter.PrepareRestore(ctx, request, receiptOutput)
	}
	return adapter.Run(ctx, request)
}

func validateConfig(config Config) error {
	if config.Command == "" || config.Engine == "" {
		return errors.New("adapter command and engine are required")
	}
	if config.DefaultTimeout <= 0 {
		return errors.New("adapter default timeout must be positive")
	}
	if len(config.Operations) == 0 {
		if config.UnavailableError == "" {
			return errors.New("adapter must declare operations or an unavailable error")
		}
		return nil
	}
	if config.NewAdapter == nil {
		return errors.New("adapter factory is required")
	}
	for _, operation := range config.Operations {
		if operation == "" {
			return errors.New("adapter operation is invalid")
		}
	}
	return nil
}

func newRemote(
	ctx context.Context,
	session string,
	operation string,
) (hostops.Remote, error) {
	provider, err := hostprovider.New(
		ctx,
		os.Getenv("COLDSNAP_HOST_PROVIDER_SOCKET"),
		os.Getenv("COLDSNAP_HOST_PROVIDER_TOKEN"),
		session,
	)
	if err != nil {
		return nil, err
	}
	required := []string{"exec", "runtime-v1"}
	switch operation {
	case "publish":
		required = append(required, "oci-push")
	case "publish-native":
		required = append(required, "huggingface-publish", "huggingface-resolve")
	}
	if err := provider.Require(required...); err != nil {
		return nil, err
	}
	return provider, nil
}

func resolveStateRoot(stateRoot *string) error {
	if *stateRoot == "" {
		cacheRoot, err := os.UserCacheDir()
		if err != nil {
			return fmt.Errorf("resolve controller cache directory: %w", err)
		}
		if !filepath.IsAbs(cacheRoot) {
			return fmt.Errorf("controller cache directory is not absolute: %s", cacheRoot)
		}
		*stateRoot = filepath.Join(cacheRoot, "coldsnap")
	}
	return nil
}

func operationUsage(operations []string) string {
	if len(operations) == 0 {
		return "<operation>"
	}
	usage := operations[0]
	for _, operation := range operations[1:] {
		usage += "|" + operation
	}
	return usage
}

func environmentDuration(name string, fallback time.Duration) time.Duration {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}

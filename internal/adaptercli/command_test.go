// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package adaptercli

import (
	"bytes"
	"context"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/snapshot"
)

type outputRoutingAdapter struct {
	progress io.Writer
}

func (adapter outputRoutingAdapter) Run(context.Context, snapshot.Request) error {
	_, err := io.WriteString(adapter.progress, "running\n")
	return err
}

func (adapter outputRoutingAdapter) PrepareRestore(
	_ context.Context, _ snapshot.Request, receipt io.Writer,
) error {
	if _, err := io.WriteString(adapter.progress, "compatible-kernel warning\n"); err != nil {
		return err
	}
	_, err := io.WriteString(receipt, "{\"kind\":\"preparation\"}\n")
	return err
}

func TestPrepareOnlySeparatesHumanProgressFromMachineReceipt(t *testing.T) {
	var stdout, stderr bytes.Buffer
	adapter := outputRoutingAdapter{progress: adapterProgressOutput(true, &stdout, &stderr)}
	if err := runAdapterOperation(
		context.Background(), adapter, snapshot.Request{}, true, &stdout,
	); err != nil {
		t.Fatal(err)
	}
	if stdout.String() != "{\"kind\":\"preparation\"}\n" {
		t.Fatalf("prepare-only stdout = %q", stdout.String())
	}
	if stderr.String() != "compatible-kernel warning\n" {
		t.Fatalf("prepare-only stderr = %q", stderr.String())
	}
}

func TestUnavailableAdapterFailsBeforeReadingRequestOrOpeningProvider(t *testing.T) {
	message := "SGLang operations are not qualified"
	err := Execute(
		context.Background(),
		Config{
			Command:          "coldsnap-sglang-adapter",
			Engine:           "sglang",
			DefaultTimeout:   time.Minute,
			UnavailableError: message,
		},
		[]string{"restore", "--request-json", "-"},
		strings.NewReader("not JSON"),
		&bytes.Buffer{},
		&bytes.Buffer{},
	)
	if err == nil || err.Error() != message {
		t.Fatalf("error = %v", err)
	}
}

func TestConfiguredAdapterRequiresAFactory(t *testing.T) {
	err := Execute(
		context.Background(),
		Config{
			Command:        "test-adapter",
			Engine:         "test",
			Operations:     []string{"restore"},
			DefaultTimeout: time.Minute,
		},
		[]string{"restore", "--request-json", "-"},
		strings.NewReader("{}"),
		&bytes.Buffer{},
		&bytes.Buffer{},
	)
	if err == nil || !strings.Contains(err.Error(), "factory") {
		t.Fatalf("error = %v", err)
	}
}

func TestAdapterRejectsRemovedDirectTransportFlags(t *testing.T) {
	for _, argument := range []string{"--ssh-user", "--host-provider"} {
		err := Execute(
			context.Background(),
			Config{
				Command:        "test-adapter",
				Engine:         "test",
				Operations:     []string{"restore"},
				DefaultTimeout: time.Minute,
				NewAdapter: func(hostops.Remote, string, time.Duration, io.Writer) Adapter {
					return nil
				},
			},
			[]string{"restore", argument, "unused", "--request-json", "-"},
			strings.NewReader("{}"),
			&bytes.Buffer{},
			&bytes.Buffer{},
		)
		if err == nil || !strings.Contains(err.Error(), "flag provided but not defined") {
			t.Fatalf("%s error = %v", argument, err)
		}
	}
}

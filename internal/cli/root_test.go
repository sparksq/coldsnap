// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package cli

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/sparksq/coldsnap/internal/buildinfo"
)

func TestVersionJSON(t *testing.T) {
	output := &bytes.Buffer{}
	command := NewRoot(Dependencies{Stdin: &bytes.Buffer{}, Stdout: output, Stderr: &bytes.Buffer{}})
	command.SetArgs([]string{"version", "--json"})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	var actual buildinfo.Info
	if err := json.Unmarshal(output.Bytes(), &actual); err != nil {
		t.Fatal(err)
	}
	if actual.Version != buildinfo.Version || actual.Version == "" {
		t.Fatalf("version info = %#v", actual)
	}
}

func TestLifecycleCommandsRequireManagerRequest(t *testing.T) {
	for _, operation := range []string{"capture", "publish", "publish-native", "restore", "sleep", "wake", "status"} {
		command := NewRoot(Dependencies{Stdin: &bytes.Buffer{}, Stdout: &bytes.Buffer{}, Stderr: &bytes.Buffer{}})
		command.SetArgs([]string{operation})
		if err := command.Execute(); err == nil || !strings.Contains(err.Error(), "--request-json is required") {
			t.Fatalf("%s error = %v", operation, err)
		}
	}
}

func TestRemovedCommandsAndTransportFlagsAreAbsent(t *testing.T) {
	for _, arguments := range [][]string{{"registry"}, {"agent"}, {"hibernate"}, {"--transport", "ssh", "status"}} {
		command := NewRoot(Dependencies{Stdin: &bytes.Buffer{}, Stdout: &bytes.Buffer{}, Stderr: &bytes.Buffer{}})
		command.SetArgs(arguments)
		if err := command.Execute(); err == nil {
			t.Fatalf("removed invocation unexpectedly accepted: %v", arguments)
		}
	}
}

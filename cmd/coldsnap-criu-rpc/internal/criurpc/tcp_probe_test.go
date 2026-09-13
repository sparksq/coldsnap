// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package criurpc

import (
	"bytes"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
)

func TestTCPProbeChecksPlannedListenersInEmptyTemplate(t *testing.T) {
	directory := t.TempDir()
	file, err := os.Create(filepath.Join(directory, "files.img"))
	if err != nil {
		t.Fatal(err)
	}
	image := &crit.CriuImage{Magic: "FILES", EntryType: &fdinfo.FileEntry{}}
	if err := crit.New(nil, file, "", false, false).Encode(image); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = listener.Close() }()
	address := listener.Addr().String()
	port := uint32(listener.Addr().(*net.TCPAddr).Port)
	base := []string{"--images-dir", directory, "--tcp-address-map", "10.0.0.1=10.0.0.1", "--tcp-port-shift", "1", "--tcp-allow-empty-map"}
	probe := func(flag string) tcpProbeResult {
		t.Helper()
		var output bytes.Buffer
		args := append(append([]string{}, base...), flag, address)
		if err := runTCPProbe(args, &output); err != nil {
			t.Fatal(err)
		}
		var result tcpProbeResult
		if err := json.Unmarshal(output.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Endpoints != 1 {
			t.Fatalf("planned endpoint was omitted: %#v", result)
		}
		return result
	}
	if result := probe("--tcp-listen-endpoint"); result.Available || !strings.Contains(result.Collision, "address already in use") {
		t.Fatalf("occupied planned listener admitted: %#v", result)
	}
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	if result := probe("--tcp-listen-endpoint"); !result.Available {
		t.Fatalf("free explicit listener rejected: %#v", result)
	}
	first, last, err := ephemeralTCPPortRange()
	if err != nil {
		t.Fatal(err)
	}
	if port < first || port > last {
		t.Fatalf("kernel-assigned port %d outside ephemeral range %d-%d", port, first, last)
	}
	if result := probe("--tcp-generated-listen-endpoint"); result.Available || !strings.Contains(result.Collision, "ephemeral range") {
		t.Fatalf("generated ephemeral listener admitted: %#v", result)
	}
}

func TestTCPProbeRejectsInvalidPlannedListeners(t *testing.T) {
	for _, address := range []string{"localhost:8000", "127.0.0.1:0", "127.0.0.1:65536", "[fe80::1%eth0]:8000"} {
		_, err := parseTCPProbeOptions([]string{"--images-dir", "/images", "--tcp-address-map", "10.0.0.1=10.0.0.1", "--tcp-port-shift", "1", "--tcp-listen-endpoint", address})
		if err == nil {
			t.Fatalf("invalid planned listener accepted: %s", address)
		}
	}
	listeners, err := plannedTCPListeners(tcpProbeOptions{listenEndpoints: []string{"[::]:8000"}})
	if err != nil || len(listeners) != 1 || !listeners[0].Address.Is6() {
		t.Fatalf("IPv6 listener: %#v, %v", listeners, err)
	}
}

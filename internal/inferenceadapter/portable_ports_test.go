// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"bytes"
	"context"
	"net/netip"
	"slices"
	"strings"
	"testing"

	"github.com/sparksq/coldsnap/internal/capsule"
	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

func TestContextFreePortProbeIncludesFutureEngineListeners(t *testing.T) {
	configureTestCRIURPC(t)
	for _, engine := range []string{"vllm", "sglang"} {
		t.Run(engine, func(t *testing.T) {
			request := validRequest(2)
			images := make(map[string]snapshot.CapsuleImage)
			for i := range request.Launch.Units {
				unit := &request.Launch.Units[i]
				unit.Host = "10.24.11.13"
				command := "vllm serve model --master-port 25001 --port 8000"
				if engine == "sglang" {
					command = "sglang serve --dist-init-addr 10.24.11.13:25001 --port 8000"
				}
				unit.Command = []string{"bash", "-c", command}
				images[unit.ID] = snapshot.CapsuleImage{Unit: unit.ID, Reference: unit.Image, Digest: unit.ImageDigest, Root: capsule.Root, Driver: snapshotdriver.Binding{ID: snapshotdriver.N580, ABI: 1}}
			}
			remote := &tcpPortProbeRemote{}
			adapter := fixtureAdapter(Adapter{Remote: remote, Output: &bytes.Buffer{}})
			selected, err := adapter.selectPortableTCPPortShift(context.Background(), request, request.Launch, images, []string{"10.24.11.13=10.24.11.13"}, "planned-port-probe")
			if err != nil {
				t.Fatal(err)
			}
			expectedMaster := uint16(1024 + ((25001 - 1024 + uint32(selected)) % 64512))
			remote.mutex.Lock()
			defer remote.mutex.Unlock()
			if len(remote.commands) != 2 {
				t.Fatalf("probe count = %d", len(remote.commands))
			}
			for _, args := range remote.commands {
				for flag, endpoint := range map[string]string{"--tcp-listen-endpoint": "0.0.0.0:8000", "--tcp-generated-listen-endpoint": netip.AddrPortFrom(netip.IPv4Unspecified(), expectedMaster).String()} {
					index := slices.Index(args, flag)
					if index < 0 || index+1 >= len(args) || args[index+1] != endpoint {
						t.Fatalf("planned %s missing from %v", endpoint, args)
					}
				}
			}
		})
	}
}

func TestCommandMasterPortMatchesRuntimePlacement(t *testing.T) {
	for _, row := range []struct {
		command string
		port    int
	}{
		{"vllm serve model --master-port=25001", 25001},
		{"sglang serve --dist-init-addr 10.0.0.1:25002 --port 8000", 25002},
		{"sglang serve --dist-init-addr=[::1]:25003", 25003},
		{"vllm serve model", 25000},
	} {
		if got := commandMasterPort(strings.Fields(row.command)); got != row.port {
			t.Fatalf("%s: got %d, want %d", row.command, got, row.port)
		}
	}
}

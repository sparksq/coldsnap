// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package activationruntime

import (
	"slices"
	"strings"
	"testing"
)

func TestCurrentPackIsContentAddressedAndDefensivelyCopied(t *testing.T) {
	first := Current()
	second := Current()
	if first.Format != Format || first.ABI != ABI || !strings.HasPrefix(first.SHA256, "sha256:") ||
		len(first.SHA256) != len("sha256:")+64 || len(first.Files) != 7 || len(first.Mounts) != 7 {
		t.Fatalf("activation runtime pack = %#v", first)
	}
	for _, item := range first.Files {
		if len(item.Data) == 0 || !strings.HasPrefix(item.SHA256, "sha256:") ||
			!strings.Contains(string(item.Data[:min(len(item.Data), 256)]), "SPDX-License-Identifier") {
			t.Fatalf("activation runtime file = %#v", item)
		}
	}
	first.Files[0].Data[0] ^= 0xff
	if slices.Equal(first.Files[0].Data, second.Files[0].Data) || second.SHA256 != Current().SHA256 {
		t.Fatal("activation runtime assets were not defensively copied")
	}
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"strings"
	"testing"

	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

func TestValidateArtifactDriverRequiresExactBinding(t *testing.T) {
	request := snapshot.Request{Driver: snapshotdriver.Selection{ID: snapshotdriver.N580}}
	contract, err := snapshotdriver.Lookup(snapshotdriver.N610)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateArtifactDriver(request, snapshot.Artifact{Driver: contract}); err == nil ||
		!strings.Contains(err.Error(), "snapshot driver") {
		t.Fatalf("expected snapshot driver mismatch, got %v", err)
	}
}

func TestN580DriverResolvesExplicitly(t *testing.T) {
	driver, err := resolveProcessDriver(snapshotdriver.Selection{ID: snapshotdriver.N580})
	if err != nil {
		t.Fatal(err)
	}
	if driver.Contract().ID != snapshotdriver.N580 {
		t.Fatalf("resolved snapshot driver = %#v", driver.Contract())
	}
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package coord

import "testing"

func TestEndpointRoundTrip(t *testing.T) {
	want := Endpoint{Address: "10.0.0.1:17877", Scope: "restore-abc", Token: testToken}
	got, err := ParseEndpoint(want.String())
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("endpoint = %#v, want %#v", got, want)
	}
}

func TestEndpointRejectsWhitespace(t *testing.T) {
	endpoint := Endpoint{Address: "127.0.0.1:1", Scope: "bad scope", Token: testToken}
	if err := endpoint.Validate(); err == nil {
		t.Fatal("expected whitespace validation error")
	}
}

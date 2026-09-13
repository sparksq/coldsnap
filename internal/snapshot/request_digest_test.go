// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

func TestRequestDigestUsesDecodedDefaultsAndCanonicalEncoding(t *testing.T) {
	for _, driver := range []string{snapshotdriver.N580, snapshotdriver.N610} {
		t.Run(driver, func(t *testing.T) {
			request := validRequest()
			request.Driver.ID = driver
			payload, err := json.Marshal(request)
			if err != nil {
				t.Fatal(err)
			}
			var document map[string]any
			if err := json.Unmarshal(payload, &document); err != nil {
				t.Fatal(err)
			}
			delete(document, "format")
			delete(document, "kind")
			delete(document, "policy")
			delete(document, "validation")
			document["launch"].(map[string]any)["units"].([]any)[0].(map[string]any)["environment"].(map[string]any)["LABEL"] = "café 🚀"
			sparse, err := json.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := DecodeRequest(bytes.NewReader(sparse))
			if err != nil {
				t.Fatal(err)
			}
			expected, err := RequestSHA256(decoded)
			if err != nil {
				t.Fatal(err)
			}
			explicit, err := json.MarshalIndent(decoded, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			escaped := strings.ReplaceAll(strings.ReplaceAll(string(explicit), "é", `\u00e9`), "🚀", `\ud83d\ude80`)
			for _, wire := range [][]byte{sparse, explicit, []byte(escaped)} {
				roundTrip, err := DecodeRequest(bytes.NewReader(wire))
				if err != nil {
					t.Fatal(err)
				}
				actual, err := RequestSHA256(roundTrip)
				if err != nil {
					t.Fatal(err)
				}
				if actual != expected {
					t.Fatalf("equivalent request digest = %s, want %s", actual, expected)
				}
				if raw := fmt.Sprintf("sha256:%x", sha256.Sum256(wire)); raw == actual {
					t.Fatal("test wire bytes unexpectedly equal the normalized request")
				}
			}
			decoded.Validation.Prompt = "changed acceptance prompt"
			changed, err := RequestSHA256(decoded)
			if err != nil {
				t.Fatal(err)
			}
			if changed == expected {
				t.Fatal("changed request field preserved digest")
			}
		})
	}
}

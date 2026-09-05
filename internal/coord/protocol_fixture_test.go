// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package coord

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
	"time"
)

func TestCrossLanguageGoldenCoordinatorFrames(t *testing.T) {
	data, err := os.ReadFile("../../test/fixtures/cross-language-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	type requestFixture struct {
		Operation uint8  `json:"operation"`
		TimeoutMS uint32 `json:"timeout_ms"`
		TokenUTF8 string `json:"token_utf8"`
		KeyUTF8   string `json:"key_utf8"`
		ValueUTF8 string `json:"value_utf8"`
	}
	type responseFixture struct {
		Status    uint8  `json:"status"`
		ValueUTF8 string `json:"value_utf8"`
	}
	var fixture struct {
		Frames []struct {
			Name     string           `json:"name"`
			Request  *requestFixture  `json:"request"`
			Response *responseFixture `json:"response"`
			Hex      string           `json:"hex"`
		} `json:"coordinator_frames"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, item := range fixture.Frames {
		t.Run(item.Name, func(t *testing.T) {
			expected, err := hex.DecodeString(item.Hex)
			if err != nil {
				t.Fatal(err)
			}
			var encoded bytes.Buffer
			if item.Request != nil {
				request := Request{
					Operation: Operation(item.Request.Operation),
					Timeout:   time.Duration(item.Request.TimeoutMS) * time.Millisecond,
					Token:     []byte(item.Request.TokenUTF8), Key: []byte(item.Request.KeyUTF8),
					Value: []byte(item.Request.ValueUTF8),
				}
				if err := WriteRequest(&encoded, request); err != nil {
					t.Fatal(err)
				}
				decoded, err := ReadRequest(bytes.NewReader(expected), DefaultMaxKey, DefaultMaxValue)
				if err != nil {
					t.Fatal(err)
				}
				if decoded.Operation != request.Operation || decoded.Timeout != request.Timeout ||
					!bytes.Equal(decoded.Token, request.Token) || !bytes.Equal(decoded.Key, request.Key) ||
					!bytes.Equal(decoded.Value, request.Value) {
					t.Fatalf("decoded request = %#v, want %#v", decoded, request)
				}
			} else if item.Response != nil {
				response := Response{
					Status: Status(item.Response.Status), Value: []byte(item.Response.ValueUTF8),
				}
				if err := WriteResponse(&encoded, response); err != nil {
					t.Fatal(err)
				}
				decoded, err := ReadResponse(bytes.NewReader(expected), DefaultMaxValue)
				if err != nil {
					t.Fatal(err)
				}
				if decoded.Status != response.Status || !bytes.Equal(decoded.Value, response.Value) {
					t.Fatalf("decoded response = %#v, want %#v", decoded, response)
				}
			} else {
				t.Fatal("fixture has neither request nor response")
			}
			if !bytes.Equal(encoded.Bytes(), expected) {
				t.Fatalf("encoded frame = %x, want %x", encoded.Bytes(), expected)
			}
		})
	}
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package canonicaljson

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
)

// CanonicalJSON reproduces the JSON form used by the Python control plane:
// sorted object keys, compact separators, and ensure_ascii=True string escaping.
func CanonicalJSON(data []byte) ([]byte, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, fmt.Errorf("decode JSON: %w", err)
	}
	if err := requireEOF(decoder); err != nil {
		return nil, err
	}

	var output bytes.Buffer
	if err := writeCanonical(&output, value); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}

// CanonicalSHA256 hashes CanonicalJSON using the same contract as
// json.dumps(value, sort_keys=True, separators=(",", ":")) in Python.
func CanonicalSHA256(data []byte) (string, error) {
	canonical, err := CanonicalJSON(data)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(canonical)
	return hex.EncodeToString(digest[:]), nil
}

func requireEOF(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if err == io.EOF {
		return nil
	}
	if err == nil {
		return fmt.Errorf("decode JSON: multiple values")
	}
	return fmt.Errorf("decode JSON trailer: %w", err)
}

func writeCanonical(output *bytes.Buffer, value any) error {
	switch typed := value.(type) {
	case nil:
		output.WriteString("null")
	case bool:
		if typed {
			output.WriteString("true")
		} else {
			output.WriteString("false")
		}
	case json.Number:
		number, err := canonicalNumber(typed)
		if err != nil {
			return err
		}
		output.WriteString(number)
	case string:
		writePythonString(output, typed)
	case []any:
		output.WriteByte('[')
		for index, item := range typed {
			if index > 0 {
				output.WriteByte(',')
			}
			if err := writeCanonical(output, item); err != nil {
				return err
			}
		}
		output.WriteByte(']')
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		output.WriteByte('{')
		for index, key := range keys {
			if index > 0 {
				output.WriteByte(',')
			}
			writePythonString(output, key)
			output.WriteByte(':')
			if err := writeCanonical(output, typed[key]); err != nil {
				return err
			}
		}
		output.WriteByte('}')
	default:
		return fmt.Errorf("unsupported canonical JSON value %T", value)
	}
	return nil
}

func canonicalNumber(value json.Number) (string, error) {
	text := value.String()
	if !strings.ContainsAny(text, ".eE") {
		integer, ok := new(big.Int).SetString(text, 10)
		if !ok {
			return "", fmt.Errorf("invalid canonical JSON integer %q", text)
		}
		return integer.String(), nil
	}
	number, err := strconv.ParseFloat(text, 64)
	if err != nil || math.IsInf(number, 0) || math.IsNaN(number) {
		return "", fmt.Errorf("invalid canonical JSON float %q", text)
	}
	if number == 0 {
		if math.Signbit(number) {
			return "-0.0", nil
		}
		return "0.0", nil
	}
	scientific := strconv.FormatFloat(number, 'e', -1, 64)
	separator := strings.LastIndexByte(scientific, 'e')
	exponent, parseErr := strconv.Atoi(scientific[separator+1:])
	if separator < 0 || parseErr != nil {
		return "", fmt.Errorf("format canonical JSON float %q", text)
	}
	// Python's float repr uses fixed notation for decimal exponents in
	// [-4, 16), and scientific notation outside that range.
	if exponent >= -4 && exponent < 16 {
		fixed := strconv.FormatFloat(number, 'f', -1, 64)
		if !strings.ContainsRune(fixed, '.') {
			fixed += ".0"
		}
		return fixed, nil
	}
	return scientific, nil
}

func writePythonString(output *bytes.Buffer, value string) {
	const hexDigits = "0123456789abcdef"
	output.WriteByte('"')
	for _, character := range value {
		switch character {
		case '"', '\\':
			output.WriteByte('\\')
			output.WriteRune(character)
		case '\b':
			output.WriteString(`\b`)
		case '\f':
			output.WriteString(`\f`)
		case '\n':
			output.WriteString(`\n`)
		case '\r':
			output.WriteString(`\r`)
		case '\t':
			output.WriteString(`\t`)
		default:
			if character >= 0x20 && character <= 0x7e {
				output.WriteRune(character)
				continue
			}
			if character <= 0xffff {
				writeUnicodeEscape(output, uint16(character), hexDigits) // #nosec G115 -- branch bounds the rune.
				continue
			}
			high, low := utf16.EncodeRune(character)
			writeUnicodeEscape(output, uint16(high), hexDigits) // #nosec G115 -- EncodeRune returns UTF-16 code units.
			writeUnicodeEscape(output, uint16(low), hexDigits)  // #nosec G115 -- EncodeRune returns UTF-16 code units.
		}
	}
	output.WriteByte('"')
}

func writeUnicodeEscape(output *bytes.Buffer, value uint16, hexDigits string) {
	output.WriteString(`\u`)
	output.WriteByte(hexDigits[(value>>12)&0xf])
	output.WriteByte(hexDigits[(value>>8)&0xf])
	output.WriteByte(hexDigits[(value>>4)&0xf])
	output.WriteByte(hexDigits[value&0xf])
}

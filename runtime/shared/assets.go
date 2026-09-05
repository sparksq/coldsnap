// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Package sharedruntime exposes release-matched Python helpers to the Go
// activation pack without duplicating their source.
package sharedruntime

import _ "embed"

//go:embed checkpointctl.py
var checkpointProbe []byte

func CheckpointProbe() []byte {
	return append([]byte(nil), checkpointProbe...)
}

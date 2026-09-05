// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Package hostops defines the manager-provided host operation boundary.
package hostops

import "context"

// Remote is the minimal operation surface engine adapters need from a manager.
// Implementations may add optional authenticated OCI and Hugging Face methods.
type Remote interface {
	Run(context.Context, string, ...string) ([]byte, error)
	RunCombined(context.Context, string, ...string) ([]byte, error)
	RunInput(context.Context, string, []byte, ...string) ([]byte, error)
	Upload(context.Context, string, []string, string, bool) error
}

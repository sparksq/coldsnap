// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

module github.com/sparksq/coldsnap/cmd/coldsnap-criu-rpc

go 1.25.14

require (
	github.com/checkpoint-restore/go-criu/v8 v8.4.0
	golang.org/x/sys v0.47.0
	google.golang.org/protobuf v1.36.12
)

require github.com/aperturerobotics/protobuf-go-lite v0.16.0 // indirect

replace github.com/checkpoint-restore/go-criu/v8 => github.com/sparksq/go-criu/v8 v8.0.0-20260826164808-29a4f2f8e837

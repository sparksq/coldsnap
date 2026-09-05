// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Package activationruntime exposes the small, release-matched controller
// layer which a manager stages for capture and restore.  Captured process
// state and capture-sensitive runtime components remain in the immutable
// capsule; these scripts are deliberately supplied by the active adapter.
package activationruntime

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"slices"

	sharedruntime "github.com/sparksq/coldsnap/runtime/shared"
)

const (
	Format               = 1
	ABI                  = 1
	CRIURPCTarget        = "/usr/local/bin/coldsnap-criu-rpc"
	PayloadVerifierFile  = "coldsnap-payload-verifier"
	ActivationLogsTarget = "/opt/coldsnap/runtime/coldsnap_activation_logs.py"
	ServiceRuntimeTarget = "/opt/coldsnap/runtime/coldsnap_service_runtime.py"
	LifecycleTarget      = "/opt/coldsnap/runtime/coldsnap-lifecycle.py"
	FeatureProbeTarget   = "/opt/coldsnap/runtime/checkpointctl.py"
)

//go:embed coldsnap_activation_logs.py
var activationLogs []byte

//go:embed coldsnap_service_runtime.py
var serviceRuntime []byte

//go:embed coldsnap_engine_rank_n580.py
var n580Controller []byte

//go:embed coldsnap_engine_rank_n610.py
var n610Controller []byte

//go:embed coldsnap_engine_exec.py
var engineLauncher []byte

//go:embed coldsnap_lifecycle.py
var lifecycleHelper []byte

type File struct {
	Name   string
	SHA256 string
	Data   []byte
}

type Mount struct {
	File   string
	Target string
}

type Pack struct {
	Format int
	ABI    int
	SHA256 string
	Files  []File
	Mounts []Mount
}

// Current returns a defensive copy of the activation runtime embedded in the
// engine-adapter binary. The aggregate digest includes names and content, so a
// manager can use it as a stable node-cache key and audit identity.
func Current() Pack {
	files := []File{
		file("coldsnap-activation-logs", activationLogs),
		file("coldsnap-service-runtime", serviceRuntime),
		file("coldsnap-engine-exec", engineLauncher),
		file("coldsnap-engine-rank-n580", n580Controller),
		file("coldsnap-engine-rank-n610", n610Controller),
		file("coldsnap-lifecycle", lifecycleHelper),
		file("coldsnap-feature-probe", sharedruntime.CheckpointProbe()),
	}
	return pack(files, []Mount{
		{File: "coldsnap-activation-logs", Target: ActivationLogsTarget},
		{File: "coldsnap-service-runtime", Target: ServiceRuntimeTarget},
		{File: "coldsnap-engine-exec", Target: "/opt/coldsnap/runtime/coldsnap-engine-exec.py"},
		{File: "coldsnap-engine-rank-n580", Target: "/usr/local/bin/coldsnap-engine-rank-n580"},
		{File: "coldsnap-engine-rank-n610", Target: "/usr/local/bin/coldsnap-engine-rank-n610"},
		{File: "coldsnap-lifecycle", Target: LifecycleTarget},
		{File: "coldsnap-feature-probe", Target: FeatureProbeTarget},
	})
}

// WithCRIURPC adds the release-matched CRIU image and RPC helper to the
// activation overlay. Capsules keep the capture-time copy for provenance, but
// compatible controller releases can fix portable image handling without a
// model recapture.
func WithCRIURPC(data []byte) Pack {
	return withCRIURPC(Current(), data)
}

// WithPayloadVerifier adds the release-matched engine adapter as a host-side
// payload verifier. It is staged with the activation pack but deliberately is
// not mounted into serving containers.
func WithPayloadVerifier(data []byte) Pack {
	current := Current()
	current.Files = append(current.Files, file(PayloadVerifierFile, data))
	return pack(current.Files, current.Mounts)
}

// WithPayloadVerifierAndCRIURPC constructs the complete host activation pack
// without dropping either release-matched executable.
func WithPayloadVerifierAndCRIURPC(verifier, criuRPC []byte) Pack {
	return withCRIURPC(WithPayloadVerifier(verifier), criuRPC)
}

func withCRIURPC(current Pack, data []byte) Pack {
	current.Files = append(current.Files, file("coldsnap-criu-rpc", data))
	current.Mounts = append(current.Mounts, Mount{
		File: "coldsnap-criu-rpc", Target: CRIURPCTarget,
	})
	return pack(current.Files, current.Mounts)
}

func pack(files []File, mounts []Mount) Pack {
	digest := sha256.New()
	_, _ = fmt.Fprintf(digest, "coldsnap-activation-runtime\x00format=%d\x00abi=%d\x00", Format, ABI)
	for _, item := range files {
		_, _ = fmt.Fprintf(digest, "%s\x00%d\x00", item.Name, len(item.Data))
		_, _ = digest.Write(item.Data)
	}
	return Pack{
		Format: Format,
		ABI:    ABI,
		SHA256: "sha256:" + hex.EncodeToString(digest.Sum(nil)),
		Files:  slices.Clone(files),
		Mounts: slices.Clone(mounts),
	}
}

func file(name string, data []byte) File {
	digest := sha256.Sum256(data)
	return File{
		Name:   name,
		SHA256: "sha256:" + hex.EncodeToString(digest[:]),
		Data:   slices.Clone(data),
	}
}

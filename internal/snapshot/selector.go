// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import (
	"errors"
	"fmt"
	"strings"
)

type PreparedPayload struct {
	Worker     string                     `json:"worker"`
	Path       string                     `json:"path"`
	Bytes      int64                      `json:"bytes"`
	SHA256     string                     `json:"sha256"`
	Validation *PreparedPayloadValidation `json:"validation,omitempty"`
}

// PreparedPayloadValidation is cached evidence from the versioned payload
// validation provider. Device/inode/size/mtime permit a fast accept; ctime is
// diagnostic only because ownership and mode changes do not alter bytes.
type PreparedPayloadValidation struct {
	Record          string  `json:"record"`
	Provider        string  `json:"provider"`
	ContentEvidence string  `json:"content_evidence"`
	Reason          string  `json:"reason,omitempty"`
	Device          uint64  `json:"device"`
	Inode           uint64  `json:"inode"`
	Size            int64   `json:"size"`
	MTimeNS         int64   `json:"mtime_ns"`
	CTimeNS         int64   `json:"ctime_ns,omitempty"`
	UID             uint32  `json:"uid,omitempty"`
	GID             uint32  `json:"gid,omitempty"`
	Mode            uint32  `json:"mode,omitempty"`
	BytesHashed     int64   `json:"bytes_hashed,omitempty"`
	Seconds         float64 `json:"seconds,omitempty"`
}

type ProviderInventory struct {
	ModelPayloads    map[string]PreparedPayload
	AllowNativeFetch bool
}

type Selection struct {
	Provider      string            `json:"provider"`
	ModelPayloads []PreparedPayload `json:"model_payloads,omitempty"`
	FetchRequired bool              `json:"fetch_required,omitempty"`
	Reason        string            `json:"reason"`
}

func SelectWeights(artifact Artifact, mode string, inventory ProviderInventory) (Selection, error) {
	if err := artifact.Validate(); err != nil {
		return Selection{}, err
	}
	if mode == "" {
		mode = "auto"
	}
	if mode != "auto" && mode != "native" && mode != "recovery" && mode != "cache-only-auto" {
		return Selection{}, fmt.Errorf("unsupported weight selection mode %q", mode)
	}
	if mode == "recovery" {
		return recoverySelection("recovery mode requested"), nil
	}
	native := artifact.Weights.Native
	modelPayloads := artifact.Weights.ModelPayloads
	if native == nil || modelPayloads == nil {
		if mode == "native" {
			return Selection{}, errors.New("artifact has no native replay provider")
		}
		return recoverySelection("artifact has no native replay provider"), nil
	}
	prepared := make([]PreparedPayload, 0, len(modelPayloads.Objects))
	missing := false
	for _, expected := range modelPayloads.Objects {
		worker := strings.TrimPrefix(expected.Owner, WorkerOwnerPrefix)
		actual, found := inventory.ModelPayloads[worker]
		if !found {
			missing = true
			continue
		}
		if actual.Path == "" || actual.Bytes != expected.Bytes || actual.SHA256 != expected.SHA256 {
			return Selection{}, fmt.Errorf("worker %s model payload failed identity verification", worker)
		}
		prepared = append(prepared, actual)
	}
	if !missing && len(prepared) == len(modelPayloads.Objects) {
		return Selection{Provider: "native", ModelPayloads: prepared, Reason: "verified model payloads are staged"}, nil
	}
	if mode == "native" {
		if inventory.AllowNativeFetch && modelPayloads.Repository != "" {
			return Selection{Provider: "native", FetchRequired: true, Reason: "model payloads require prefetch"}, nil
		}
		return Selection{}, errors.New("native model payloads are unavailable")
	}
	if mode == "auto" && inventory.AllowNativeFetch && modelPayloads.Repository != "" {
		return Selection{Provider: "native", FetchRequired: true, Reason: "model payloads available from optional repository"}, nil
	}
	return recoverySelection("model payloads are not staged; using safetensors recovery"), nil
}

func recoverySelection(reason string) Selection {
	return Selection{Provider: "recovery", Reason: reason}
}

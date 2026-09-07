// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package buildinfo

import (
	"runtime/debug"
	"strings"
)

var Version = "0.3.23"
var Commit = ""

type Info struct {
	Version string `json:"version"`
	Commit  string `json:"commit,omitempty"`
}

func Current() Info {
	info := Info{Version: Version, Commit: Commit}
	if info.Commit != "" {
		return info
	}
	settings, ok := debug.ReadBuildInfo()
	if !ok {
		return info
	}
	for _, setting := range settings.Settings {
		if setting.Key == "vcs.revision" {
			info.Commit = strings.TrimSpace(setting.Value)
			break
		}
	}
	return info
}

// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
)

// install -d can report ENOENT from its final chmod after mkdir failed with
// ENOSPC. Inspect only failed writes, using the same host and SSH identity.
// Missing paths are resolved to their nearest existing ancestor so a separate
// cache mount is checked rather than assuming the root filesystem is relevant.
const filesystemSpaceProbe = `import json, os, sys
results = []
uid = os.geteuid()
for requested in sys.argv[1:]:
    if not os.path.isabs(requested):
        continue
    path = os.path.realpath(requested)
    while True:
        try:
            stat = os.statvfs(path)
            if stat.f_frsize > 0:
                blocks = stat.f_bfree if uid == 0 else stat.f_bavail
                inodes = stat.f_ffree if uid == 0 else stat.f_favail
                results.append({
                    "path": requested, "existing_path": path, "uid": uid,
                    "available_bytes": blocks * stat.f_frsize,
                    "available_inodes": inodes, "total_inodes": stat.f_files,
                })
            break
        except FileNotFoundError:
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        except OSError:
            break
print(json.dumps(results))
`

type filesystemSpace struct {
	Path            string  `json:"path"`
	ExistingPath    string  `json:"existing_path"`
	UID             *uint32 `json:"uid"`
	AvailableBytes  *int64  `json:"available_bytes"`
	AvailableInodes *int64  `json:"available_inodes"`
	TotalInodes     *int64  `json:"total_inodes"`
}

func (adapter Adapter) ensurePrivateDirectories(ctx context.Context, host string, paths ...string) error {
	_, err := adapter.Remote.Run(ctx, host, append([]string{"install", "-d", "-m", "0700"}, paths...)...)
	return adapter.diagnoseFilesystemFailure(ctx, host, err, paths...)
}

func (adapter Adapter) writeStateFile(ctx context.Context, host string, payload []byte, path string) error {
	_, err := adapter.Remote.RunInput(ctx, host, payload, "tee", path)
	return adapter.diagnoseFilesystemFailure(ctx, host, err, path)
}

func (adapter Adapter) diagnoseFilesystemFailure(ctx context.Context, host string, cause error, paths ...string) error {
	if cause == nil || ctx.Err() != nil || errors.Is(cause, context.Canceled) || errors.Is(cause, context.DeadlineExceeded) {
		return cause
	}
	// Diagnostics must not turn a failed operation into an unbounded wait.
	probeContext, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	output, err := adapter.Remote.Run(probeContext, host, append([]string{"python3", "-c", filesystemSpaceProbe}, paths...)...)
	if err != nil {
		return cause
	}
	var observations []filesystemSpace
	if err := json.Unmarshal(output, &observations); err != nil {
		return cause
	}
	for _, observed := range observations {
		// Missing fields are not evidence of zero free space.
		if !slices.Contains(paths, observed.Path) || observed.ExistingPath == "" || observed.UID == nil ||
			observed.AvailableBytes == nil || observed.AvailableInodes == nil || observed.TotalInodes == nil {
			continue
		}
		var exhausted []string
		if *observed.AvailableBytes <= 0 {
			exhausted = append(exhausted, "no disk space")
		}
		// Some filesystems do not expose a finite inode pool.
		if *observed.TotalInodes > 0 && *observed.AvailableInodes <= 0 {
			exhausted = append(exhausted, "no free inodes")
		}
		if len(exhausted) != 0 {
			return fmt.Errorf(
				"%s available to uid %d on %s for %q (filesystem checked at %q); free storage and retry; original failure: %w",
				strings.Join(exhausted, " and "), *observed.UID, host, observed.Path, observed.ExistingPath, cause,
			)
		}
	}
	return cause
}

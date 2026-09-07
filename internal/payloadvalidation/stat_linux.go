// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package payloadvalidation

import "syscall"

func statTimes(stat *syscall.Stat_t) (mtime, ctime int64) {
	return stat.Mtim.Nano(), stat.Ctim.Nano()
}

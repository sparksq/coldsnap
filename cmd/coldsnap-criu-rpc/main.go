// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package main

import (
	"fmt"
	"os"

	"github.com/sparksq/coldsnap/cmd/coldsnap-criu-rpc/internal/criurpc"
)

func main() {
	arguments := os.Args[1:]
	var err error
	if len(arguments) > 0 {
		switch arguments[0] {
		case "fd-broker":
			err = criurpc.RunFDBroker(arguments[1:])
		case "tcp-probe":
			err = criurpc.RunTCPProbe(arguments[1:])
		default:
			err = criurpc.Run(arguments)
		}
	} else {
		err = criurpc.Run(arguments)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "coldsnap-criu-rpc: %v\n", err)
		os.Exit(1)
	}
}

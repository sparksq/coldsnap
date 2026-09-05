// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package criurpc

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

const tcpProbeKind = "coldsnap-criu-tcp-port-probe"

type tcpProbeResult struct {
	Format    int    `json:"format"`
	Kind      string `json:"kind"`
	PortShift uint   `json:"port_shift"`
	Endpoints int    `json:"endpoints"`
	Available bool   `json:"available"`
	Collision string `json:"collision,omitempty"`
}

type tcpProbeOptions struct {
	imagesDir       string
	tcpAddressMap   repeatedString
	tcpPortShift    uint
	tcpPreservePort uint
	tcpPortMap      string
	allowEmptyMap   bool
}

func parseTCPProbeOptions(arguments []string) (tcpProbeOptions, error) {
	var value tcpProbeOptions
	flags := flag.NewFlagSet("coldsnap-criu-rpc tcp-probe", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.StringVar(&value.imagesDir, "images-dir", "", "CRIU image directory")
	flags.Var(
		&value.tcpAddressMap,
		"tcp-address-map",
		"restore TCP endpoint address mapping in OLD=NEW form",
	)
	flags.UintVar(
		&value.tcpPortShift,
		"tcp-port-shift",
		0,
		"candidate TCP endpoint port rotation in [1, 64511]",
	)
	flags.UintVar(
		&value.tcpPreservePort,
		"tcp-preserve-port",
		0,
		"externally managed TCP listener port excluded from rotation",
	)
	flags.StringVar(
		&value.tcpPortMap,
		"tcp-port-map",
		"",
		"explicit restored TCP port mapping in OLD=NEW form",
	)
	flags.BoolVar(
		&value.allowEmptyMap,
		"tcp-allow-empty-map",
		false,
		"allow an inventory with no mapped TCP endpoints",
	)
	if err := flags.Parse(arguments); err != nil {
		return tcpProbeOptions{}, err
	}
	if flags.NArg() != 0 {
		return tcpProbeOptions{}, fmt.Errorf("unexpected positional arguments: %q", flags.Args())
	}
	if !filepath.IsAbs(value.imagesDir) || filepath.Clean(value.imagesDir) != value.imagesDir {
		return tcpProbeOptions{}, errors.New("--images-dir must be a clean absolute path")
	}
	if value.tcpPortShift == 0 || value.tcpPortShift > 64511 {
		return tcpProbeOptions{}, errors.New("--tcp-port-shift must be in [1, 64511]")
	}
	if value.tcpPreservePort > 65535 {
		return tcpProbeOptions{}, errors.New("--tcp-preserve-port must be in [0, 65535]")
	}
	if len(value.tcpAddressMap) == 0 {
		return tcpProbeOptions{}, errors.New("at least one --tcp-address-map is required")
	}
	if _, err := parseTCPAddressMap(value.tcpAddressMap); err != nil {
		return tcpProbeOptions{}, err
	}
	if _, err := parseTCPPortMap(value.tcpPortMap); err != nil {
		return tcpProbeOptions{}, err
	}
	return value, nil
}

func runTCPProbe(arguments []string, output io.Writer) error {
	value, err := parseTCPProbeOptions(arguments)
	if err != nil {
		return err
	}
	mapping, err := parseTCPAddressMap(value.tcpAddressMap)
	if err != nil {
		return err
	}
	portMapping, err := parseTCPPortMap(value.tcpPortMap)
	if err != nil {
		return err
	}
	image, err := loadCRIUFilesImage(value.imagesDir)
	if err != nil {
		return err
	}
	_, _, endpoints, err := rewriteTCPSocketEndpoints(
		image, mapping, value.tcpPortShift, value.tcpPreservePort,
		portMapping, value.allowEmptyMap,
	)
	if err != nil {
		return err
	}
	probeErr := probeTCPBindEndpoints(endpoints)
	if probeErr != nil && !errors.Is(probeErr, unix.EADDRINUSE) {
		return probeErr
	}
	result := tcpProbeResult{
		Format: 1, Kind: tcpProbeKind,
		PortShift: value.tcpPortShift, Endpoints: len(endpoints), Available: probeErr == nil,
	}
	if probeErr != nil {
		result.Collision = probeErr.Error()
	}
	return json.NewEncoder(output).Encode(result)
}

// RunTCPProbe validates one candidate port generation against the live host
// network namespace without mutating the committed CRIU image.
func RunTCPProbe(arguments []string) error {
	return runTCPProbe(arguments, os.Stdout)
}

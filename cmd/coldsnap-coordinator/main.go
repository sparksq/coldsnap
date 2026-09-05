// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/sparksq/coldsnap/internal/coord"
	"github.com/sparksq/coldsnap/internal/fsutil"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "coldsnap-coordinator:", err)
		os.Exit(1)
	}
}

func run() error {
	listenAddress := flag.String("listen", "0.0.0.0:0", "TCP address on which to listen")
	advertiseHost := flag.String("advertise-host", "", "host or IP written to the endpoint file")
	scope := flag.String("scope", "", "operation-scoped key prefix")
	endpointPath := flag.String("endpoint-file", "", "mode-0600 endpoint file to publish")
	tokenPath := flag.String("token-file", "", "optional file containing an existing token")
	flag.Parse()
	if *scope == "" || *endpointPath == "" {
		return errors.New("--scope and --endpoint-file are required")
	}
	token, err := loadToken(*tokenPath)
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp", *listenAddress)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	defer listener.Close()
	address, err := advertisedAddress(listener.Addr(), *advertiseHost)
	if err != nil {
		return err
	}
	endpoint := coord.Endpoint{Address: address, Scope: *scope, Token: token}
	if err := writeEndpoint(*endpointPath, endpoint); err != nil {
		return err
	}
	defer os.Remove(*endpointPath)
	fmt.Fprintf(os.Stderr, "coldsnap coordinator listening on %s for scope %s\n", address, *scope)
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	return coord.NewServer(token).Serve(ctx, listener)
}

func loadToken(path string) (string, error) {
	if path != "" {
		value, err := os.ReadFile(path)
		if err != nil {
			return "", fmt.Errorf("read token file: %w", err)
		}
		token := strings.TrimSpace(string(value))
		if len(token) < 16 || strings.ContainsAny(token, " \t\r\n") {
			return "", errors.New("token file must contain at least 16 non-whitespace bytes")
		}
		return token, nil
	}
	value := make([]byte, 32)
	if _, err := rand.Read(value); err != nil {
		return "", fmt.Errorf("generate token: %w", err)
	}
	return hex.EncodeToString(value), nil
}

func advertisedAddress(address net.Addr, host string) (string, error) {
	tcpAddress, ok := address.(*net.TCPAddr)
	if !ok {
		return "", fmt.Errorf("unexpected listener address %q", address)
	}
	if host == "" {
		host = tcpAddress.IP.String()
		if tcpAddress.IP.IsUnspecified() {
			return "", errors.New("--advertise-host is required when listening on a wildcard address")
		}
	}
	return net.JoinHostPort(host, fmt.Sprint(tcpAddress.Port)), nil
}

func writeEndpoint(path string, endpoint coord.Endpoint) error {
	if err := endpoint.Validate(); err != nil {
		return err
	}
	return fsutil.WriteFileAtomic(path, []byte(endpoint.String()+"\n"), 0o600)
}

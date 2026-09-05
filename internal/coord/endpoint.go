// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package coord

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

const EndpointFormat = "coldsnap-coord-v1"

type Endpoint struct {
	Address string
	Scope   string
	Token   string
}

func (endpoint Endpoint) Validate() error {
	if endpoint.Address == "" || endpoint.Scope == "" || endpoint.Token == "" {
		return errors.New("coordinator endpoint requires address, scope, and token")
	}
	for name, value := range map[string]string{"address": endpoint.Address, "scope": endpoint.Scope, "token": endpoint.Token} {
		if strings.ContainsAny(value, " \t\r\n") {
			return fmt.Errorf("coordinator %s must not contain whitespace", name)
		}
	}
	return nil
}

func (endpoint Endpoint) String() string {
	return fmt.Sprintf("%s %s %s %s", EndpointFormat, endpoint.Address, endpoint.Scope, endpoint.Token)
}

func ParseEndpoint(value string) (Endpoint, error) {
	fields := strings.Fields(value)
	if len(fields) != 4 || fields[0] != EndpointFormat {
		return Endpoint{}, fmt.Errorf("invalid coordinator endpoint; expected %q address scope token", EndpointFormat)
	}
	endpoint := Endpoint{Address: fields[1], Scope: fields[2], Token: fields[3]}
	return endpoint, endpoint.Validate()
}

func ReadEndpoint(path string) (Endpoint, error) {
	file, err := os.Open(path)
	if err != nil {
		return Endpoint{}, fmt.Errorf("open coordinator endpoint: %w", err)
	}
	defer file.Close()
	value, err := io.ReadAll(io.LimitReader(file, 16<<10))
	if err != nil {
		return Endpoint{}, fmt.Errorf("read coordinator endpoint: %w", err)
	}
	endpoint, err := ParseEndpoint(string(value))
	if err != nil {
		return Endpoint{}, fmt.Errorf("parse coordinator endpoint: %w", err)
	}
	return endpoint, nil
}

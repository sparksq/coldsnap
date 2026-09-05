// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Package hostprovider implements the manager-to-ColdSnap host-operations
// boundary. The snapshot request and artifacts remain transport-independent;
// an operation-scoped provider supplies host execution and credentialed
// publication without exposing SSH configuration to ColdSnap.
package hostprovider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sparksq/coldsnap/internal/hostops"
)

const (
	ProtocolFormat     = 1
	maximumMessageSize = 64 << 20
)

// Request is one operation on an already-authorized manager host session.
// Byte slices use JSON's standard base64 representation.
type Request struct {
	Format      int                     `json:"format"`
	ID          string                  `json:"id"`
	Token       string                  `json:"token"`
	Session     string                  `json:"session"`
	Operation   string                  `json:"operation"`
	Host        string                  `json:"host,omitempty"`
	Arguments   []string                `json:"arguments,omitempty"`
	Input       []byte                  `json:"input,omitempty"`
	Combined    bool                    `json:"combined,omitempty"`
	Sources     []string                `json:"sources,omitempty"`
	Destination string                  `json:"destination,omitempty"`
	Recursive   bool                    `json:"recursive,omitempty"`
	Image       string                  `json:"image,omitempty"`
	Repository  string                  `json:"repository,omitempty"`
	Revision    string                  `json:"revision,omitempty"`
	Source      string                  `json:"source,omitempty"`
	Runtime     *hostops.RuntimeRequest `json:"runtime,omitempty"`
}

// Response preserves command exit status separately from provider failures.
type Response struct {
	Format       int                      `json:"format"`
	ID           string                   `json:"id"`
	OK           bool                     `json:"ok"`
	ExitCode     int                      `json:"exit_code,omitempty"`
	Output       []byte                   `json:"output,omitempty"`
	ErrorOutput  []byte                   `json:"error_output,omitempty"`
	ErrorCode    string                   `json:"error_code,omitempty"`
	Error        string                   `json:"error,omitempty"`
	Value        string                   `json:"value,omitempty"`
	Provider     string                   `json:"provider,omitempty"`
	Capabilities []string                 `json:"capabilities,omitempty"`
	Runtime      *hostops.RuntimeResponse `json:"runtime,omitempty"`
}

// Remote implements the controller and adapter remote interfaces over an
// operation-scoped manager provider.
type Remote struct {
	socket       string
	token        string
	session      string
	dialer       net.Dialer
	provider     string
	capabilities map[string]struct{}
}

// New connects to and validates a provider before any host mutation occurs.
func New(ctx context.Context, socket, token, session string) (*Remote, error) {
	if socket == "" || !filepath.IsAbs(socket) || filepath.Clean(socket) != socket {
		return nil, errors.New("host provider socket must be an absolute clean path")
	}
	if token == "" || strings.ContainsAny(token, "\r\n\x00") {
		return nil, errors.New("host provider token is invalid")
	}
	if session == "" || strings.ContainsAny(session, "\r\n\x00") {
		return nil, errors.New("host provider session is invalid")
	}
	if err := validateSocket(socket); err != nil {
		return nil, err
	}
	remote := &Remote{socket: socket, token: token, session: session}
	response, err := remote.call(ctx, Request{Operation: "capabilities"})
	if err != nil {
		return nil, fmt.Errorf("host provider handshake: %w", err)
	}
	if response.Provider == "" || strings.ContainsAny(response.Provider, "\r\n\x00") {
		return nil, errors.New("host provider handshake returned an invalid provider identity")
	}
	remote.provider = response.Provider
	remote.capabilities = make(map[string]struct{}, len(response.Capabilities))
	for _, capability := range response.Capabilities {
		if capability == "" || strings.ContainsAny(capability, "\r\n\x00") {
			return nil, fmt.Errorf("host provider %q returned an invalid capability", response.Provider)
		}
		remote.capabilities[capability] = struct{}{}
	}
	return remote, nil
}

// Require validates operation-specific capabilities before any host mutation.
// Optional capabilities are checked by the method which consumes them, so a
// minimal manager can support local-only workflows without implementing
// publication or registry access.
func (remote *Remote) Require(capabilities ...string) error {
	for _, capability := range capabilities {
		if _, ok := remote.capabilities[capability]; !ok {
			return fmt.Errorf("host provider %q lacks required capability %q", remote.provider, capability)
		}
	}
	return nil
}

func (remote *Remote) Run(ctx context.Context, host string, arguments ...string) ([]byte, error) {
	return remote.run(ctx, host, nil, false, arguments...)
}

func (remote *Remote) RunCombined(ctx context.Context, host string, arguments ...string) ([]byte, error) {
	return remote.run(ctx, host, nil, true, arguments...)
}

func (remote *Remote) RunInput(ctx context.Context, host string, input []byte, arguments ...string) ([]byte, error) {
	return remote.run(ctx, host, input, false, arguments...)
}

// Runtime submits one typed image or workload operation to the manager. The
// engine adapter does not know whether the manager realizes it with Docker,
// Kubernetes, CRI, or another substrate.
func (remote *Remote) Runtime(
	ctx context.Context, host string, operation hostops.RuntimeRequest,
) (hostops.RuntimeResponse, error) {
	required := []string{"runtime-v1"}
	if operation.Action == hostops.RuntimeImagePush {
		required = append(required, "oci-push")
	}
	if operation.Action == hostops.RuntimeImagePull {
		required = append(required, "oci-pull")
	}
	if err := remote.Require(required...); err != nil {
		return hostops.RuntimeResponse{}, err
	}
	if host == "" || operation.Action == "" {
		return hostops.RuntimeResponse{}, errors.New("host provider runtime operation is incomplete")
	}
	response, err := remote.call(ctx, Request{Operation: "runtime", Host: host, Runtime: &operation})
	if err != nil {
		return hostops.RuntimeResponse{}, err
	}
	if response.Runtime == nil {
		return hostops.RuntimeResponse{}, errors.New("host provider runtime response is absent")
	}
	return *response.Runtime, nil
}

func (remote *Remote) run(
	ctx context.Context, host string, input []byte, combined bool, arguments ...string,
) ([]byte, error) {
	if err := remote.Require("exec"); err != nil {
		return nil, err
	}
	if host == "" || len(arguments) == 0 {
		return nil, errors.New("host provider command is incomplete")
	}
	response, err := remote.call(ctx, Request{
		Operation: "exec", Host: host, Arguments: arguments, Input: input, Combined: combined,
	})
	if err != nil {
		return nil, err
	}
	if response.ExitCode != 0 {
		detail := strings.TrimSpace(string(response.ErrorOutput))
		if combined {
			detail = strings.TrimSpace(string(response.Output))
		}
		if detail != "" {
			return nil, fmt.Errorf("remote %s on %s exited %d: %s", arguments[0], host, response.ExitCode, detail)
		}
		return nil, fmt.Errorf("remote %s on %s exited %d", arguments[0], host, response.ExitCode)
	}
	return response.Output, nil
}

func (remote *Remote) Upload(
	ctx context.Context, host string, sources []string, destination string, recursive bool,
) error {
	if err := remote.Require("upload"); err != nil {
		return err
	}
	_, err := remote.call(ctx, Request{
		Operation: "upload", Host: host, Sources: sources, Destination: destination, Recursive: recursive,
	})
	return err
}

func (remote *Remote) PublishHuggingFaceFile(
	ctx context.Context,
	host, image, repository, revision, source, destination string,
) error {
	if err := remote.Require("huggingface-publish"); err != nil {
		return err
	}
	_, err := remote.call(ctx, Request{
		Operation: "huggingface-publish", Host: host, Image: image,
		Repository: repository, Revision: revision, Source: source, Destination: destination,
	})
	return err
}

func (remote *Remote) ResolveHuggingFaceRevision(
	ctx context.Context, host, image, repository, revision string,
) (string, error) {
	if err := remote.Require("huggingface-resolve"); err != nil {
		return "", err
	}
	response, err := remote.call(ctx, Request{
		Operation: "huggingface-resolve", Host: host, Image: image,
		Repository: repository, Revision: revision,
	})
	if err != nil {
		return "", err
	}
	return response.Value, nil
}

func (remote *Remote) call(ctx context.Context, request Request) (Response, error) {
	request.Format = ProtocolFormat
	request.ID = fmt.Sprintf("%d", time.Now().UnixNano())
	request.Token = remote.token
	request.Session = remote.session
	payload, err := json.Marshal(request)
	if err != nil {
		return Response{}, fmt.Errorf("encode host provider request: %w", err)
	}
	if len(payload)+1 > maximumMessageSize {
		return Response{}, fmt.Errorf("host provider request exceeds %d bytes", maximumMessageSize)
	}
	connection, err := remote.dialer.DialContext(ctx, "unix", remote.socket)
	if err != nil {
		return Response{}, fmt.Errorf("connect host provider: %w", err)
	}
	defer connection.Close()
	if deadline, ok := ctx.Deadline(); ok {
		_ = connection.SetDeadline(deadline)
	}
	if _, err := connection.Write(append(payload, '\n')); err != nil {
		return Response{}, fmt.Errorf("send host provider request: %w", err)
	}
	if unix, ok := connection.(*net.UnixConn); ok {
		_ = unix.CloseWrite()
	}
	reply, err := io.ReadAll(io.LimitReader(connection, maximumMessageSize+1))
	if err != nil {
		return Response{}, fmt.Errorf("read host provider response: %w", err)
	}
	if len(reply) > maximumMessageSize {
		return Response{}, fmt.Errorf("host provider response exceeds %d bytes", maximumMessageSize)
	}
	var response Response
	decoder := json.NewDecoder(bytes.NewReader(reply))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&response); err != nil {
		return Response{}, fmt.Errorf("read host provider response: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return Response{}, errors.New("host provider response has trailing data")
	}
	if response.Format != ProtocolFormat || response.ID != request.ID {
		return Response{}, errors.New("host provider response identity is invalid")
	}
	if !response.OK {
		if response.Error == "" {
			response.Error = "provider returned an unspecified failure"
		}
		if response.ErrorCode != "" {
			return Response{}, &hostops.RuntimeError{Code: response.ErrorCode, Message: response.Error}
		}
		return Response{}, errors.New(response.Error)
	}
	return response, nil
}

func validateSocket(socket string) error {
	information, err := os.Lstat(socket)
	if err != nil {
		return fmt.Errorf("inspect host provider socket: %w", err)
	}
	if information.Mode()&os.ModeSymlink != 0 || information.Mode()&os.ModeSocket == 0 {
		return errors.New("host provider socket is not a Unix socket")
	}
	if information.Mode().Perm()&0o077 != 0 {
		return errors.New("host provider socket permissions are not private")
	}
	parent, err := os.Stat(filepath.Dir(socket))
	if err != nil {
		return fmt.Errorf("inspect host provider socket directory: %w", err)
	}
	if !parent.IsDir() || parent.Mode().Perm()&0o077 != 0 {
		return errors.New("host provider socket directory permissions are not private")
	}
	return nil
}

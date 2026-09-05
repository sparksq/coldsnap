// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package payloadvalidation

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	RPCFormat                = 1
	maximumRPCMessage        = 1 << 20
	maximumRPCTokenBytes     = 4096
	defaultRPCConcurrent     = 2
	defaultRPCRequestTimeout = 24 * time.Hour
)

// RPCRequest carries the same validation options accepted by the local Go API.
type RPCRequest struct {
	Format  int     `json:"format"`
	ID      string  `json:"id"`
	Token   string  `json:"token"`
	Options Options `json:"options"`
}

// RPCResponse keeps transport failures separate from a successful admission.
type RPCResponse struct {
	Format    int        `json:"format"`
	ID        string     `json:"id"`
	OK        bool       `json:"ok"`
	Admission *Admission `json:"admission,omitempty"`
	Error     string     `json:"error,omitempty"`
}

// RPCServer exposes Validate over a private, authenticated Unix socket.
type RPCServer struct {
	Token         string
	MaxConcurrent int
}

func (server RPCServer) Serve(ctx context.Context, listener net.Listener) error {
	if err := validateRPCToken(server.Token); err != nil {
		return err
	}
	maximum := server.MaxConcurrent
	if maximum == 0 {
		maximum = defaultRPCConcurrent
	}
	if maximum < 1 || maximum > 64 {
		return errors.New("payload validation RPC concurrency must be between 1 and 64")
	}
	semaphore := make(chan struct{}, maximum)
	var active sync.WaitGroup
	go func() {
		<-ctx.Done()
		_ = listener.Close()
	}()
	for {
		connection, err := listener.Accept()
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				active.Wait()
				return nil
			}
			return fmt.Errorf("accept payload validation RPC: %w", err)
		}
		select {
		case semaphore <- struct{}{}:
			active.Add(1)
			go func() {
				defer active.Done()
				defer func() { <-semaphore }()
				server.handle(ctx, connection)
			}()
		case <-ctx.Done():
			_ = connection.Close()
		}
	}
}

func (server RPCServer) handle(ctx context.Context, connection net.Conn) {
	defer connection.Close()
	_ = connection.SetDeadline(time.Now().Add(defaultRPCRequestTimeout))
	request, err := decodeRPCRequest(connection)
	response := RPCResponse{Format: RPCFormat, ID: request.ID}
	if err == nil {
		if len(request.Token) != len(server.Token) ||
			subtle.ConstantTimeCompare([]byte(request.Token), []byte(server.Token)) != 1 {
			err = errors.New("payload validation RPC authentication failed")
		}
	}
	if err == nil {
		select {
		case <-ctx.Done():
			err = ctx.Err()
		default:
			var admission Admission
			admission, err = Validate(request.Options)
			if err == nil {
				response.OK = true
				response.Admission = &admission
			}
		}
	}
	if err != nil {
		response.Error = err.Error()
	}
	_ = json.NewEncoder(connection).Encode(response)
}

func decodeRPCRequest(reader io.Reader) (RPCRequest, error) {
	var request RPCRequest
	payload, err := io.ReadAll(io.LimitReader(reader, maximumRPCMessage+1))
	if err != nil {
		return request, fmt.Errorf("read payload validation RPC request: %w", err)
	}
	if len(payload) > maximumRPCMessage {
		return request, errors.New("payload validation RPC request is too large")
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		return request, fmt.Errorf("decode payload validation RPC request: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return request, errors.New("payload validation RPC request has trailing data")
	}
	if request.Format != RPCFormat || request.ID == "" ||
		strings.ContainsAny(request.ID, "\r\n\x00") {
		return request, errors.New("payload validation RPC request identity is invalid")
	}
	return request, nil
}

// ValidateRPC asks a data-owning local service to execute the canonical validator.
func ValidateRPC(ctx context.Context, socket, token string, options Options) (Admission, error) {
	if err := validateRPCSocket(socket); err != nil {
		return Admission{}, err
	}
	if err := validateRPCToken(token); err != nil {
		return Admission{}, err
	}
	identifier := make([]byte, 16)
	if _, err := rand.Read(identifier); err != nil {
		return Admission{}, fmt.Errorf("generate payload validation RPC identity: %w", err)
	}
	request := RPCRequest{
		Format: RPCFormat, ID: hex.EncodeToString(identifier), Token: token, Options: options,
	}
	payload, err := json.Marshal(request)
	if err != nil {
		return Admission{}, fmt.Errorf("encode payload validation RPC request: %w", err)
	}
	if len(payload)+1 > maximumRPCMessage {
		return Admission{}, errors.New("payload validation RPC request is too large")
	}
	connection, err := (&net.Dialer{}).DialContext(ctx, "unix", socket)
	if err != nil {
		return Admission{}, fmt.Errorf("connect payload validation RPC: %w", err)
	}
	defer connection.Close()
	if deadline, ok := ctx.Deadline(); ok {
		_ = connection.SetDeadline(deadline)
	}
	if _, err := connection.Write(append(payload, '\n')); err != nil {
		return Admission{}, fmt.Errorf("send payload validation RPC request: %w", err)
	}
	if unix, ok := connection.(*net.UnixConn); ok {
		_ = unix.CloseWrite()
	}
	reply, err := io.ReadAll(io.LimitReader(connection, maximumRPCMessage+1))
	if err != nil {
		return Admission{}, fmt.Errorf("read payload validation RPC response: %w", err)
	}
	if len(reply) > maximumRPCMessage {
		return Admission{}, errors.New("payload validation RPC response is too large")
	}
	var response RPCResponse
	decoder := json.NewDecoder(bytes.NewReader(reply))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&response); err != nil {
		return Admission{}, fmt.Errorf("decode payload validation RPC response: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return Admission{}, errors.New("payload validation RPC response has trailing data")
	}
	if response.Format != RPCFormat || response.ID != request.ID {
		return Admission{}, errors.New("payload validation RPC response identity is invalid")
	}
	if !response.OK {
		if response.Error == "" {
			response.Error = "payload validation RPC returned an unspecified failure"
		}
		return Admission{}, errors.New(response.Error)
	}
	if response.Admission == nil {
		return Admission{}, errors.New("payload validation RPC omitted its admission")
	}
	return *response.Admission, nil
}

// LoadRPCToken reads an operation-scoped token without accepting ambiguous text.
func LoadRPCToken(path string) (string, error) {
	if path == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return "", errors.New("payload validation RPC token file must be a clean absolute path")
	}
	descriptor, err := syscall.Open(
		path, syscall.O_RDONLY|syscall.O_CLOEXEC|syscall.O_NOFOLLOW, 0,
	)
	if err != nil {
		return "", fmt.Errorf("read payload validation RPC token: %w", err)
	}
	file := os.NewFile(uintptr(descriptor), path)
	if file == nil {
		_ = syscall.Close(descriptor)
		return "", errors.New("open payload validation RPC token returned no file")
	}
	defer file.Close()
	information, err := file.Stat()
	if err != nil {
		return "", fmt.Errorf("inspect payload validation RPC token: %w", err)
	}
	if !information.Mode().IsRegular() || information.Mode().Perm()&0o077 != 0 ||
		information.Size() <= 0 || information.Size() > maximumRPCTokenBytes {
		return "", errors.New("payload validation RPC token file must be a small private regular file")
	}
	value, err := io.ReadAll(io.LimitReader(file, maximumRPCTokenBytes+1))
	if err != nil {
		return "", fmt.Errorf("read payload validation RPC token: %w", err)
	}
	if len(value) > maximumRPCTokenBytes {
		return "", errors.New("payload validation RPC token file is too large")
	}
	token := strings.TrimSpace(string(value))
	if err := validateRPCToken(token); err != nil {
		return "", err
	}
	return token, nil
}

func validateRPCToken(token string) error {
	if len(token) < 16 || strings.ContainsAny(token, " \t\r\n\x00") {
		return errors.New("payload validation RPC token must contain at least 16 non-whitespace bytes")
	}
	return nil
}

func validateRPCSocket(path string) error {
	if path == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return errors.New("payload validation RPC socket must be a clean absolute path")
	}
	information, err := os.Lstat(path)
	if err != nil {
		return fmt.Errorf("inspect payload validation RPC socket: %w", err)
	}
	if information.Mode()&os.ModeSymlink != 0 || information.Mode()&os.ModeSocket == 0 {
		return errors.New("payload validation RPC endpoint is not a Unix socket")
	}
	if information.Mode().Perm()&0o077 != 0 {
		return errors.New("payload validation RPC socket permissions are not private")
	}
	return nil
}

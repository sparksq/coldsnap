// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package coord

import (
	"context"
	"crypto/subtle"
	"errors"
	"fmt"
	"math"
	"net"
	"strconv"
	"sync"
	"time"
)

type Server struct {
	Token         []byte
	MaxKey        int
	MaxValue      int
	MaxWait       time.Duration
	HeaderTimeout time.Duration

	mutex  sync.Mutex
	values map[string][]byte
	change chan struct{}
}

func NewServer(token string) *Server {
	return &Server{
		Token:         []byte(token),
		MaxKey:        DefaultMaxKey,
		MaxValue:      DefaultMaxValue,
		MaxWait:       time.Hour,
		HeaderTimeout: 10 * time.Second,
		values:        make(map[string][]byte),
		change:        make(chan struct{}),
	}
}

func (server *Server) Serve(context context.Context, listener net.Listener) error {
	if len(server.Token) < 16 {
		return errors.New("coordinator token must contain at least 16 bytes")
	}
	go func() {
		<-context.Done()
		_ = listener.Close()
	}()
	for {
		connection, err := listener.Accept()
		if err != nil {
			if context.Err() != nil {
				return nil
			}
			return fmt.Errorf("accept coordinator connection: %w", err)
		}
		go server.handle(context, connection)
	}
}

func (server *Server) handle(parent context.Context, connection net.Conn) {
	defer connection.Close()
	SetDeadline(connection, server.HeaderTimeout)
	request, err := ReadRequest(connection, server.MaxKey, server.MaxValue)
	if err != nil {
		status := StatusInvalid
		if errors.Is(err, ErrTooLarge) {
			status = StatusTooLarge
		}
		_ = WriteResponse(connection, Response{Status: status, Value: []byte(err.Error())})
		return
	}
	_ = connection.SetDeadline(time.Time{})
	if len(request.Token) != len(server.Token) || subtle.ConstantTimeCompare(request.Token, server.Token) != 1 {
		_ = WriteResponse(connection, Response{Status: StatusUnauthorized, Value: []byte("invalid token")})
		return
	}
	if server.MaxWait > 0 && request.Timeout > server.MaxWait {
		_ = WriteResponse(connection, Response{Status: StatusInvalid, Value: []byte("wait exceeds coordinator limit")})
		return
	}
	response := server.execute(parent, request)
	SetDeadline(connection, server.HeaderTimeout)
	_ = WriteResponse(connection, response)
}

func (server *Server) execute(parent context.Context, request Request) Response {
	switch request.Operation {
	case OpHealth:
		return Response{Status: StatusOK}
	case OpSet:
		if len(request.Key) == 0 {
			return Response{Status: StatusInvalid, Value: []byte("key is required")}
		}
		server.mutex.Lock()
		server.values[string(request.Key)] = append([]byte(nil), request.Value...)
		server.signalLocked()
		server.mutex.Unlock()
		return Response{Status: StatusOK}
	case OpDelete:
		if len(request.Key) == 0 {
			return Response{Status: StatusInvalid, Value: []byte("key is required")}
		}
		server.mutex.Lock()
		delete(server.values, string(request.Key))
		server.signalLocked()
		server.mutex.Unlock()
		return Response{Status: StatusOK}
	case OpAdd:
		if len(request.Key) == 0 {
			return Response{Status: StatusInvalid, Value: []byte("key is required")}
		}
		delta, err := strconv.ParseInt(string(request.Value), 10, 64)
		if err != nil {
			return Response{Status: StatusInvalid, Value: []byte("add value must be an integer")}
		}
		server.mutex.Lock()
		current := int64(0)
		if value, found := server.values[string(request.Key)]; found {
			current, err = strconv.ParseInt(string(value), 10, 64)
		}
		if err != nil {
			server.mutex.Unlock()
			return Response{Status: StatusInvalid, Value: []byte("stored value is not an integer")}
		}
		if (delta > 0 && current > math.MaxInt64-delta) || (delta < 0 && current < math.MinInt64-delta) {
			server.mutex.Unlock()
			return Response{Status: StatusInvalid, Value: []byte("add result overflows int64")}
		}
		current += delta
		encoded := []byte(strconv.FormatInt(current, 10))
		server.values[string(request.Key)] = encoded
		server.signalLocked()
		server.mutex.Unlock()
		return Response{Status: StatusOK, Value: encoded}
	case OpAppend:
		if len(request.Key) == 0 {
			return Response{Status: StatusInvalid, Value: []byte("key is required")}
		}
		server.mutex.Lock()
		if len(server.values[string(request.Key)])+len(request.Value) > server.MaxValue {
			server.mutex.Unlock()
			return Response{Status: StatusTooLarge, Value: []byte("appended value exceeds coordinator limit")}
		}
		server.values[string(request.Key)] = append(server.values[string(request.Key)], request.Value...)
		server.signalLocked()
		server.mutex.Unlock()
		return Response{Status: StatusOK}
	case OpGet:
		return server.get(parent, request)
	default:
		return Response{Status: StatusInvalid, Value: []byte("unsupported operation")}
	}
}

func (server *Server) get(parent context.Context, request Request) Response {
	if len(request.Key) == 0 {
		return Response{Status: StatusInvalid, Value: []byte("key is required")}
	}
	if request.Timeout <= 0 {
		server.mutex.Lock()
		value, found := server.values[string(request.Key)]
		server.mutex.Unlock()
		if !found {
			return Response{Status: StatusNotFound}
		}
		return Response{Status: StatusOK, Value: append([]byte(nil), value...)}
	}
	requestContext, cancel := context.WithTimeout(parent, request.Timeout)
	defer cancel()
	for {
		server.mutex.Lock()
		value, found := server.values[string(request.Key)]
		change := server.change
		server.mutex.Unlock()
		if found {
			return Response{Status: StatusOK, Value: append([]byte(nil), value...)}
		}
		select {
		case <-requestContext.Done():
			if errors.Is(requestContext.Err(), context.DeadlineExceeded) {
				return Response{Status: StatusTimeout}
			}
			return Response{Status: StatusInternal, Value: []byte("coordinator is stopping")}
		case <-change:
		}
	}
}

func (server *Server) signalLocked() {
	close(server.change)
	server.change = make(chan struct{})
}

func SetDeadline(connection net.Conn, timeout time.Duration) {
	if timeout > 0 {
		_ = connection.SetDeadline(time.Now().Add(timeout))
	}
}

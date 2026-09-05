// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package coord

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
)

const (
	Version         = 1
	requestHeader   = 28
	responseHeader  = 12
	DefaultMaxKey   = 64 << 10
	DefaultMaxValue = 16 << 20
)

var magic = [4]byte{'C', 'S', 'K', 'V'}

type Operation uint8

const (
	OpSet Operation = iota + 1
	OpGet
	OpDelete
	OpHealth
	OpAdd
	OpAppend
)

type Status uint8

const (
	StatusOK Status = iota
	StatusNotFound
	StatusUnauthorized
	StatusInvalid
	StatusTimeout
	StatusTooLarge
	StatusInternal
)

type Request struct {
	Operation Operation
	Timeout   time.Duration
	Token     []byte
	Key       []byte
	Value     []byte
}

type Response struct {
	Status Status
	Value  []byte
}

func ReadRequest(reader io.Reader, maxKey, maxValue int) (Request, error) {
	header := make([]byte, requestHeader)
	if _, err := io.ReadFull(reader, header); err != nil {
		return Request{}, err
	}
	if string(header[:4]) != string(magic[:]) || header[4] != Version {
		return Request{}, errors.New("unsupported coordinator protocol")
	}
	request := Request{
		Operation: Operation(header[5]),
		Timeout:   time.Duration(binary.BigEndian.Uint32(header[8:12])) * time.Millisecond,
	}
	tokenLength := int(binary.BigEndian.Uint32(header[12:16]))
	keyLength := int(binary.BigEndian.Uint32(header[16:20]))
	valueLength := int(binary.BigEndian.Uint32(header[20:24]))
	totalLength := int(binary.BigEndian.Uint32(header[24:28]))
	if tokenLength < 1 || keyLength < 0 || valueLength < 0 || totalLength != tokenLength+keyLength+valueLength {
		return Request{}, errors.New("invalid coordinator request lengths")
	}
	if keyLength > maxKey || valueLength > maxValue || totalLength > maxKey+maxValue+4096 {
		return Request{}, ErrTooLarge
	}
	payload := make([]byte, totalLength)
	if _, err := io.ReadFull(reader, payload); err != nil {
		return Request{}, err
	}
	request.Token = payload[:tokenLength]
	request.Key = payload[tokenLength : tokenLength+keyLength]
	request.Value = payload[tokenLength+keyLength:]
	return request, nil
}

func WriteRequest(writer io.Writer, request Request) error {
	if len(request.Token) == 0 {
		return errors.New("coordinator token is required")
	}
	totalLength := len(request.Token) + len(request.Key) + len(request.Value)
	if totalLength > int(^uint32(0)) {
		return ErrTooLarge
	}
	header := make([]byte, requestHeader)
	copy(header[:4], magic[:])
	header[4] = Version
	header[5] = byte(request.Operation)
	timeout := request.Timeout / time.Millisecond
	if timeout < 0 || timeout > time.Duration(^uint32(0)) {
		return fmt.Errorf("coordinator timeout is out of range: %s", request.Timeout)
	}
	binary.BigEndian.PutUint32(header[8:12], uint32(timeout))
	binary.BigEndian.PutUint32(header[12:16], uint32(len(request.Token)))
	binary.BigEndian.PutUint32(header[16:20], uint32(len(request.Key)))
	binary.BigEndian.PutUint32(header[20:24], uint32(len(request.Value)))
	binary.BigEndian.PutUint32(header[24:28], uint32(totalLength))
	if err := writeAll(writer, header); err != nil {
		return err
	}
	for _, part := range [][]byte{request.Token, request.Key, request.Value} {
		if err := writeAll(writer, part); err != nil {
			return err
		}
	}
	return nil
}

func ReadResponse(reader io.Reader, maxValue int) (Response, error) {
	header := make([]byte, responseHeader)
	if _, err := io.ReadFull(reader, header); err != nil {
		return Response{}, err
	}
	if string(header[:4]) != string(magic[:]) || header[4] != Version {
		return Response{}, errors.New("unsupported coordinator response")
	}
	valueLength := int(binary.BigEndian.Uint32(header[8:12]))
	if valueLength > maxValue {
		return Response{}, ErrTooLarge
	}
	response := Response{Status: Status(header[5]), Value: make([]byte, valueLength)}
	if _, err := io.ReadFull(reader, response.Value); err != nil {
		return Response{}, err
	}
	return response, nil
}

func WriteResponse(writer io.Writer, response Response) error {
	if len(response.Value) > int(^uint32(0)) {
		return ErrTooLarge
	}
	header := make([]byte, responseHeader)
	copy(header[:4], magic[:])
	header[4] = Version
	header[5] = byte(response.Status)
	binary.BigEndian.PutUint32(header[8:12], uint32(len(response.Value)))
	if err := writeAll(writer, header); err != nil {
		return err
	}
	return writeAll(writer, response.Value)
}

func writeAll(writer io.Writer, value []byte) error {
	for len(value) > 0 {
		written, err := writer.Write(value)
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
		value = value[written:]
	}
	return nil
}

var ErrTooLarge = errors.New("coordinator request exceeds configured limits")

func statusError(status Status, message []byte) error {
	if status == StatusOK {
		return nil
	}
	detail := strings.TrimSpace(string(message))
	if detail == "" {
		detail = fmt.Sprintf("status %d", status)
	}
	switch status {
	case StatusNotFound:
		return fmt.Errorf("%w: %s", ErrNotFound, detail)
	case StatusTimeout:
		return fmt.Errorf("%w: %s", ErrTimeout, detail)
	case StatusUnauthorized:
		return fmt.Errorf("%w: %s", ErrUnauthorized, detail)
	default:
		return fmt.Errorf("coordinator request failed: %s", detail)
	}
}

var (
	ErrNotFound     = errors.New("coordinator key not found")
	ErrTimeout      = errors.New("coordinator wait timed out")
	ErrUnauthorized = errors.New("coordinator authentication failed")
)

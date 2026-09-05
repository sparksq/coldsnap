// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package coord

import (
	"context"
	"fmt"
	"net"
	"strconv"
	"time"
)

type Client struct {
	Endpoint    Endpoint
	Dialer      net.Dialer
	MaxValue    int
	DialTimeout time.Duration
}

func NewClient(endpoint Endpoint) *Client {
	return &Client{
		Endpoint:    endpoint,
		MaxValue:    DefaultMaxValue,
		DialTimeout: 5 * time.Second,
	}
}

func (client *Client) Health(context context.Context) error {
	_, err := client.request(context, Request{Operation: OpHealth})
	return err
}

func (client *Client) Set(context context.Context, key string, value []byte) error {
	_, err := client.request(context, Request{Operation: OpSet, Key: client.key(key), Value: value})
	return err
}

func (client *Client) Get(context context.Context, key string, wait time.Duration) ([]byte, error) {
	return client.request(context, Request{Operation: OpGet, Key: client.key(key), Timeout: wait})
}

func (client *Client) Delete(context context.Context, key string) error {
	_, err := client.request(context, Request{Operation: OpDelete, Key: client.key(key)})
	return err
}

func (client *Client) Add(context context.Context, key string, delta int64) (int64, error) {
	value, err := client.request(context, Request{
		Operation: OpAdd, Key: client.key(key), Value: []byte(strconv.FormatInt(delta, 10)),
	})
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(string(value), 10, 64)
}

func (client *Client) Append(context context.Context, key string, value []byte) error {
	_, err := client.request(context, Request{Operation: OpAppend, Key: client.key(key), Value: value})
	return err
}

func (client *Client) key(key string) []byte {
	return []byte(client.Endpoint.Scope + "/" + key)
}

func (client *Client) request(context context.Context, request Request) ([]byte, error) {
	if err := client.Endpoint.Validate(); err != nil {
		return nil, err
	}
	request.Token = []byte(client.Endpoint.Token)
	dialer := client.Dialer
	if client.DialTimeout > 0 {
		dialer.Timeout = client.DialTimeout
	}
	connection, err := dialer.DialContext(context, "tcp", client.Endpoint.Address)
	if err != nil {
		return nil, fmt.Errorf("connect to coordinator: %w", err)
	}
	defer connection.Close()
	deadline := client.DialTimeout
	if request.Timeout > 0 {
		deadline += request.Timeout
	}
	absoluteDeadline := time.Now().Add(deadline)
	if contextDeadline, ok := context.Deadline(); ok && contextDeadline.Before(absoluteDeadline) {
		absoluteDeadline = contextDeadline
	}
	_ = connection.SetDeadline(absoluteDeadline)
	if err := WriteRequest(connection, request); err != nil {
		return nil, fmt.Errorf("write coordinator request: %w", err)
	}
	maximum := client.MaxValue
	if maximum == 0 {
		maximum = DefaultMaxValue
	}
	response, err := ReadResponse(connection, maximum)
	if err != nil {
		return nil, fmt.Errorf("read coordinator response: %w", err)
	}
	if err := statusError(response.Status, response.Value); err != nil {
		return nil, err
	}
	return response.Value, nil
}

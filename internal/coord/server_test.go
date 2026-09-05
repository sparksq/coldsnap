// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package coord

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"
)

const testToken = "0123456789abcdef0123456789abcdef" // gitleaks:allow -- deterministic test fixture

func startTestServer(t *testing.T) (Endpoint, context.CancelFunc) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	server := NewServer(testToken)
	done := make(chan error, 1)
	go func() { done <- server.Serve(ctx, listener) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			if err != nil {
				t.Errorf("serve: %v", err)
			}
		case <-time.After(time.Second):
			t.Error("coordinator did not stop")
		}
	})
	return Endpoint{Address: listener.Addr().String(), Scope: "capture-42", Token: testToken}, cancel
}

func TestClientRoundTripAndScopeIsolation(t *testing.T) {
	endpoint, _ := startTestServer(t)
	client := NewClient(endpoint)
	ctx := context.Background()
	if err := client.Health(ctx); err != nil {
		t.Fatalf("health: %v", err)
	}
	if err := client.Set(ctx, "rank/0/ready", []byte("yes")); err != nil {
		t.Fatalf("set: %v", err)
	}
	value, err := client.Get(ctx, "rank/0/ready", 0)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if string(value) != "yes" {
		t.Fatalf("get = %q", value)
	}
	other := endpoint
	other.Scope = "restore-42"
	_, err = NewClient(other).Get(ctx, "rank/0/ready", 0)
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("other scope get = %v, want not found", err)
	}
	if err := client.Delete(ctx, "rank/0/ready"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	_, err = client.Get(ctx, "rank/0/ready", 0)
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("get after delete = %v, want not found", err)
	}
}

func TestBlockingGet(t *testing.T) {
	endpoint, _ := startTestServer(t)
	client := NewClient(endpoint)
	result := make(chan []byte, 1)
	failure := make(chan error, 1)
	go func() {
		value, err := client.Get(context.Background(), "barrier", time.Second)
		if err != nil {
			failure <- err
			return
		}
		result <- value
	}()
	time.Sleep(20 * time.Millisecond)
	if err := client.Set(context.Background(), "barrier", []byte("open")); err != nil {
		t.Fatal(err)
	}
	select {
	case value := <-result:
		if string(value) != "open" {
			t.Fatalf("value = %q", value)
		}
	case err := <-failure:
		t.Fatal(err)
	case <-time.After(time.Second):
		t.Fatal("blocking get did not return")
	}
}

func TestAuthenticationAndTimeout(t *testing.T) {
	endpoint, _ := startTestServer(t)
	bad := endpoint
	bad.Token = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	if err := NewClient(bad).Health(context.Background()); !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("bad token = %v", err)
	}
	_, err := NewClient(endpoint).Get(context.Background(), "missing", 25*time.Millisecond)
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("blocking get = %v, want timeout", err)
	}
}

func TestClientHonorsContextDeadlineDuringBlockingGet(t *testing.T) {
	endpoint, _ := startTestServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err := NewClient(endpoint).Get(ctx, "missing", time.Minute)
	if err == nil {
		t.Fatal("blocking get ignored context deadline")
	}
}

func TestAtomicAddAndAppend(t *testing.T) {
	endpoint, _ := startTestServer(t)
	client := NewClient(endpoint)
	ctx := context.Background()
	value, err := client.Add(ctx, "counter", 2)
	if err != nil || value != 2 {
		t.Fatalf("first add = %d, %v", value, err)
	}
	value, err = client.Add(ctx, "counter", 3)
	if err != nil || value != 5 {
		t.Fatalf("second add = %d, %v", value, err)
	}
	if err := client.Append(ctx, "buffer", []byte("ab")); err != nil {
		t.Fatal(err)
	}
	if err := client.Append(ctx, "buffer", []byte("cd")); err != nil {
		t.Fatal(err)
	}
	buffer, err := client.Get(ctx, "buffer", 0)
	if err != nil || string(buffer) != "abcd" {
		t.Fatalf("appended value = %q, %v", buffer, err)
	}
}

func TestAddOverflowAndAppendLimitFailClosed(t *testing.T) {
	server := NewServer(testToken)
	server.MaxValue = 4
	server.values["number"] = []byte("9223372036854775807")
	overflow := server.execute(context.Background(), Request{
		Operation: OpAdd, Key: []byte("number"), Value: []byte("1"),
	})
	if overflow.Status != StatusInvalid {
		t.Fatalf("overflow response = %#v", overflow)
	}
	server.values["value"] = []byte("1234")
	tooLarge := server.execute(context.Background(), Request{
		Operation: OpAppend, Key: []byte("value"), Value: []byte("5"),
	})
	if tooLarge.Status != StatusTooLarge {
		t.Fatalf("append response = %#v", tooLarge)
	}
}

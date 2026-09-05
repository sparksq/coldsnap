// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package criurpc

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"golang.org/x/sys/unix"
)

const (
	fdBrokerProtocol = "COLDSNAP_FD_BROKER_V1"
	fdBrokerMaxFiles = 4096
)

type fdBrokerKey struct {
	kind byte
	id   uint64
}

type fdBrokerStore struct {
	mutex  sync.Mutex
	files  map[fdBrokerKey]*os.File
	sealed bool
}

func parseFDBrokerRequest(payload string) (string, fdBrokerKey, error) {
	fields := strings.Fields(payload)
	if len(fields) < 2 || fields[0] != fdBrokerProtocol {
		return "", fdBrokerKey{}, errors.New("invalid FD broker protocol")
	}
	command := fields[1]
	if command == "SEAL" || command == "STATUS" {
		if len(fields) != 2 {
			return "", fdBrokerKey{}, errors.New("invalid FD broker control request")
		}
		return command, fdBrokerKey{}, nil
	}
	if (command != "PUT" && command != "GET" && command != "TAKE") || len(fields) != 4 {
		return "", fdBrokerKey{}, errors.New("invalid FD broker request")
	}
	if len(fields[2]) != 1 ||
		(fields[2][0] != 'E' && fields[2][0] != 'M' && fields[2][0] != 'V') {
		return "", fdBrokerKey{}, errors.New("invalid FD broker object kind")
	}
	id, err := strconv.ParseUint(fields[3], 10, 64)
	if err != nil {
		return "", fdBrokerKey{}, errors.New("invalid FD broker object id")
	}
	return command, fdBrokerKey{kind: fields[2][0], id: id}, nil
}

func receivedFD(oob []byte) (int, error) {
	messages, err := unix.ParseSocketControlMessage(oob)
	if err != nil {
		return -1, err
	}
	var descriptors []int
	for _, message := range messages {
		fds, err := unix.ParseUnixRights(&message)
		if err != nil {
			for _, fd := range descriptors {
				_ = unix.Close(fd)
			}
			return -1, err
		}
		descriptors = append(descriptors, fds...)
	}
	if len(descriptors) != 1 {
		for _, fd := range descriptors {
			_ = unix.Close(fd)
		}
		return -1, fmt.Errorf("FD broker PUT received %d descriptors, want 1", len(descriptors))
	}
	return descriptors[0], nil
}

func (store *fdBrokerStore) serve(connection *net.UnixConn) {
	defer func() { _ = connection.Close() }()
	payload := make([]byte, 256)
	oob := make([]byte, unix.CmsgSpace(4))
	count, oobCount, flags, _, err := connection.ReadMsgUnix(payload, oob)
	if err != nil {
		return
	}
	if flags&(unix.MSG_TRUNC|unix.MSG_CTRUNC) != 0 {
		_, _ = connection.Write([]byte("ERR truncated request\n"))
		return
	}
	command, key, err := parseFDBrokerRequest(string(payload[:count]))
	if err != nil {
		_, _ = connection.Write([]byte("ERR " + err.Error() + "\n"))
		return
	}
	switch command {
	case "PUT":
		fd, err := receivedFD(oob[:oobCount])
		if err != nil {
			_, _ = connection.Write([]byte("ERR " + err.Error() + "\n"))
			return
		}
		file := os.NewFile(uintptr(fd), fmt.Sprintf("coldsnap-fd-%c-%d", key.kind, key.id))
		store.mutex.Lock()
		if store.sealed {
			err = errors.New("FD broker is sealed")
		} else if len(store.files) >= fdBrokerMaxFiles {
			err = errors.New("FD broker capacity exceeded")
		} else if previous, exists := store.files[key]; exists && key.kind != 'M' {
			err = errors.New("duplicate FD broker object")
		} else {
			if previous != nil {
				_ = previous.Close()
			}
			store.files[key] = file
			file = nil
		}
		store.mutex.Unlock()
		if file != nil {
			_ = file.Close()
		}
		if err != nil {
			_, _ = connection.Write([]byte("ERR " + err.Error() + "\n"))
			return
		}
		_, _ = connection.Write([]byte("OK\n"))
	case "GET", "TAKE":
		if oobCount != 0 {
			_, _ = connection.Write([]byte("ERR descriptor request included descriptors\n"))
			return
		}
		store.mutex.Lock()
		file := store.files[key]
		sealed := store.sealed
		if !sealed {
			store.mutex.Unlock()
			_, _ = connection.Write([]byte("ERR FD broker is not sealed\n"))
			return
		}
		if file == nil {
			store.mutex.Unlock()
			_, _ = connection.Write([]byte("ERR FD broker object is absent\n"))
			return
		}
		_, _, err = connection.WriteMsgUnix(
			[]byte("OK\n"),
			unix.UnixRights(int(file.Fd())),
			nil,
		)
		if command == "TAKE" && err == nil {
			delete(store.files, key)
		}
		store.mutex.Unlock()
		if command == "TAKE" && err == nil {
			_ = file.Close()
		}
		if err != nil {
			return
		}
	case "SEAL":
		if oobCount != 0 {
			_, _ = connection.Write([]byte("ERR SEAL included descriptors\n"))
			return
		}
		store.mutex.Lock()
		store.sealed = true
		count := len(store.files)
		store.mutex.Unlock()
		_, _ = fmt.Fprintf(connection, "OK %d\n", count)
	case "STATUS":
		if oobCount != 0 {
			_, _ = connection.Write([]byte("ERR STATUS included descriptors\n"))
			return
		}
		store.mutex.Lock()
		count := len(store.files)
		sealed := store.sealed
		store.mutex.Unlock()
		_, _ = fmt.Fprintf(connection, "OK files=%d sealed=%t\n", count, sealed)
	}
}

func brokerRequest(socketPath, request string, file *os.File) (string, error) {
	if !filepath.IsAbs(socketPath) {
		return "", errors.New("FD broker socket must be absolute")
	}
	connection, err := net.DialUnix(
		"unixpacket",
		nil,
		&net.UnixAddr{Name: socketPath, Net: "unixpacket"},
	)
	if err != nil {
		return "", err
	}
	defer func() { _ = connection.Close() }()
	var rights []byte
	if file != nil {
		rights = unix.UnixRights(int(file.Fd()))
	}
	if _, _, err := connection.WriteMsgUnix([]byte(request), rights, nil); err != nil {
		return "", err
	}
	response := make([]byte, 256)
	count, err := connection.Read(response)
	if err != nil {
		return "", err
	}
	value := strings.TrimSpace(string(response[:count]))
	if value != "OK" && !strings.HasPrefix(value, "OK ") {
		return "", fmt.Errorf("FD broker rejected request: %s", value)
	}
	return value, nil
}

func brokerPut(socketPath string, kind byte, id uint64, file *os.File) error {
	if file == nil || (kind != 'E' && kind != 'M' && kind != 'V') {
		return errors.New("invalid FD broker PUT")
	}
	_, err := brokerRequest(
		socketPath,
		fmt.Sprintf("%s PUT %c %d\n", fdBrokerProtocol, kind, id),
		file,
	)
	return err
}

func brokerSeal(socketPath string) error {
	_, err := brokerRequest(socketPath, fdBrokerProtocol+" SEAL\n", nil)
	return err
}

// RunFDBroker holds the original NVIDIA open-file descriptions across the
// CRIU dump/restore gap. Device memory and CUDA contexts are still destroyed;
// this preserves only the reset driver's per-file control-plane residue.
func RunFDBroker(arguments []string) error {
	flags := flag.NewFlagSet("fd-broker", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	var socketPath string
	flags.StringVar(&socketPath, "socket", "", "absolute AF_UNIX socket path")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if flags.NArg() != 0 || !filepath.IsAbs(socketPath) {
		return errors.New("fd-broker requires one absolute --socket path")
	}
	if err := os.MkdirAll(filepath.Dir(socketPath), 0o700); err != nil {
		return err
	}
	if _, err := os.Lstat(socketPath); err == nil {
		return fmt.Errorf("refusing to replace existing FD broker socket: %s", socketPath)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	listener, err := net.ListenUnix(
		"unixpacket",
		&net.UnixAddr{Name: socketPath, Net: "unixpacket"},
	)
	if err != nil {
		return err
	}
	defer func() {
		_ = listener.Close()
		_ = os.Remove(socketPath)
	}()
	if err := os.Chmod(socketPath, 0o600); err != nil {
		return err
	}
	store := &fdBrokerStore{files: make(map[fdBrokerKey]*os.File)}
	defer func() {
		for _, file := range store.files {
			_ = file.Close()
		}
	}()
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(stop)
	go func() {
		<-stop
		_ = listener.Close()
	}()
	for {
		connection, err := listener.AcceptUnix()
		if err != nil {
			select {
			case <-stop:
				return nil
			default:
				return err
			}
		}
		go store.serve(connection)
	}
}

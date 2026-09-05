// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package criurpc

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fown"
	sk_inet "github.com/checkpoint-restore/go-criu/v8/crit/images/sk-inet"
	sk_opts "github.com/checkpoint-restore/go-criu/v8/crit/images/sk-opts"
	"golang.org/x/sys/unix"
	"google.golang.org/protobuf/proto"
)

func TestAllowedNVIDIADevice(t *testing.T) {
	for _, path := range []string{
		"/dev/nvidia0",
		"/dev/nvidia17",
		"/dev/nvidiactl",
		"/dev/nvidia-uvm",
		"/dev/nvidia-uvm-tools",
		"/dev/nvidia-modeset",
		"/dev/nvidia-caps/nvidia-cap1",
	} {
		if !allowedNVIDIADevice(path) {
			t.Errorf("allowedNVIDIADevice(%q) = false", path)
		}
	}
	for _, path := range []string{
		"/dev/null",
		"/dev/nvidia0/../mem",
		"/dev/nvidia",
		"/dev/nvidia-caps/../nvidia0",
		"/tmp/nvidia0",
	} {
		if allowedNVIDIADevice(path) {
			t.Errorf("allowedNVIDIADevice(%q) = true", path)
		}
	}
}

func TestSnapshotLinkRemapsCopiesOnlyRegularNumberedFiles(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	target := filepath.Join(root, "target")
	if err := os.Mkdir(source, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "link_remap.10"), []byte("rescue-inode"), 0o640); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "unrelated"), []byte("ignore"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := snapshotLinkRemaps(source, target); err != nil {
		t.Fatal(err)
	}
	payload, err := os.ReadFile(filepath.Join(target, "link_remap.10"))
	if err != nil {
		t.Fatal(err)
	}
	if string(payload) != "rescue-inode" {
		t.Fatalf("copied payload = %q", payload)
	}
	if _, err := os.Stat(filepath.Join(target, "unrelated")); !os.IsNotExist(err) {
		t.Fatalf("unrelated file was copied: %v", err)
	}
	if err := snapshotLinkRemaps(source, target); err == nil {
		t.Fatal("existing snapshot directory was overwritten")
	}
}

func TestSnapshotLinkRemapsRejectsSymlinks(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	if err := os.Mkdir(source, 0o700); err != nil {
		t.Fatal(err)
	}
	seed := filepath.Join(root, "seed")
	if err := os.WriteFile(seed, []byte("seed"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(seed, filepath.Join(source, "link_remap.11")); err != nil {
		t.Fatal(err)
	}
	if err := snapshotLinkRemaps(source, filepath.Join(root, "target")); err == nil {
		t.Fatal("symlinked link remap was accepted")
	}
}

func TestLoadManifestRejectsUnknownAndDuplicateRecords(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "external.json")
	record := externalFile{
		ID:          7,
		Path:        "/dev/nvidia0",
		OpenFlags:   unix.O_RDWR,
		DeviceMajor: 195,
		DeviceMinor: 0,
	}
	if err := writeJSONAtomically(path, externalManifest{
		Format: manifestFormat,
		Kind:   manifestKind,
		Files:  []externalFile{record},
	}); err != nil {
		t.Fatal(err)
	}
	manifest, err := loadManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Files) != 1 || manifest.Files[0] != record {
		t.Fatalf("manifest = %#v", manifest)
	}

	record.Path = "/dev/null"
	data, err := json.Marshal(externalManifest{
		Format: manifestFormat,
		Kind:   manifestKind,
		Files:  []externalFile{record},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadManifest(path); err == nil {
		t.Fatal("disallowed device path was accepted")
	}

	record.Path = "/dev/nvidia0"
	data, err = json.Marshal(externalManifest{
		Format: manifestFormat,
		Kind:   manifestKind,
		Files:  []externalFile{record, record},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadManifest(path); err == nil {
		t.Fatal("duplicate device id was accepted")
	}
}

func TestInspectExternalFileRejectsOrdinaryFile(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), "ordinary")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = file.Close() }()
	if _, err := inspectExternalFile(11, file); err == nil {
		t.Fatal("ordinary file was accepted as an external device")
	}
}

func TestOpenExternalPlaceholderIsUniquelyIdentifiableWithoutWeakeningManifest(t *testing.T) {
	record := externalFile{
		ID:          11,
		Path:        "/dev/nvidiactl",
		OpenFlags:   unix.O_RDWR,
		DeviceMajor: 195,
		DeviceMinor: 255,
	}
	file, err := openExternalPlaceholder(record)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = file.Close() }()
	path, err := os.Readlink(filepath.Join("/proc/self/fd", fmt.Sprint(file.Fd())))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(path, "/coldsnap-nvidia-placeholder-11-") ||
		!strings.HasSuffix(path, " (deleted)") {
		t.Fatalf("placeholder path = %q", path)
	}
	record.Path = "/dev/null"
	if _, err := openExternalPlaceholder(record); err == nil {
		t.Fatal("disallowed manifest path received a placeholder")
	}
}

func TestWriteNVIDIAExternalMapIsDeterministicAndPrivate(t *testing.T) {
	directory := t.TempDir()
	path, err := writeNVIDIAExternalMap(directory, map[uint32]externalFile{
		19: {
			ID:          19,
			Path:        "/dev/nvidia0",
			OpenFlags:   unix.O_RDWR,
			DeviceMajor: 195,
			DeviceMinor: 0,
		},
		7: {
			ID:          7,
			Path:        "/dev/nvidiactl",
			OpenFlags:   unix.O_RDWR,
			DeviceMajor: 195,
			DeviceMinor: 255,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if path != filepath.Join(directory, nvidiaExternalMapBasename) {
		t.Fatalf("map path = %q", path)
	}
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	want := nvidiaExternalMapHeader + "\n" +
		"7\t2\t195\t255\t/dev/nvidiactl\n" +
		"19\t2\t195\t0\t/dev/nvidia0\n"
	if string(payload) != want {
		t.Fatalf("map payload = %q, want %q", payload, want)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("map mode = %o", info.Mode().Perm())
	}
}

func TestFDBrokerPreservesOpenFileDescription(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "broker.sock")
	listener, err := net.ListenUnix(
		"unixpacket",
		&net.UnixAddr{Name: socketPath, Net: "unixpacket"},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = listener.Close() }()
	store := &fdBrokerStore{files: make(map[fdBrokerKey]*os.File)}
	serveOne := func() {
		connection, err := listener.AcceptUnix()
		if err != nil {
			t.Error(err)
			return
		}
		store.serve(connection)
	}
	source, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = source.Close() }()
	go serveOne()
	if err := brokerPut(socketPath, 'E', 17, source); err != nil {
		t.Fatal(err)
	}
	go serveOne()
	if err := brokerSeal(socketPath); err != nil {
		t.Fatal(err)
	}
	go serveOne()
	connection, err := net.DialUnix(
		"unixpacket",
		nil,
		&net.UnixAddr{Name: socketPath, Net: "unixpacket"},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = connection.Close() }()
	request := fdBrokerProtocol + " GET E 17\n"
	if _, _, err := connection.WriteMsgUnix([]byte(request), nil, nil); err != nil {
		t.Fatal(err)
	}
	payload := make([]byte, 32)
	oob := make([]byte, unix.CmsgSpace(4))
	count, oobCount, flags, _, err := connection.ReadMsgUnix(payload, oob)
	if err != nil {
		t.Fatal(err)
	}
	if flags&(unix.MSG_TRUNC|unix.MSG_CTRUNC) != 0 || string(payload[:count]) != "OK\n" {
		t.Fatalf("broker response = %q flags=%#x", payload[:count], flags)
	}
	fd, err := receivedFD(oob[:oobCount])
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = unix.Close(fd) }()
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil {
		t.Fatal(err)
	}
	if uint32(stat.Mode)&unix.S_IFMT != unix.S_IFCHR {
		t.Fatalf("broker returned mode %#o", stat.Mode)
	}
}

func TestFDBrokerTakeTransfersOwnershipOnce(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "broker.sock")
	listener, err := net.ListenUnix(
		"unixpacket",
		&net.UnixAddr{Name: socketPath, Net: "unixpacket"},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = listener.Close() }()
	store := &fdBrokerStore{files: make(map[fdBrokerKey]*os.File)}
	serveOne := func() {
		connection, err := listener.AcceptUnix()
		if err != nil {
			t.Error(err)
			return
		}
		store.serve(connection)
	}
	source, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = source.Close() }()
	go serveOne()
	if err := brokerPut(socketPath, 'M', 42, source); err != nil {
		t.Fatal(err)
	}
	go serveOne()
	if err := brokerSeal(socketPath); err != nil {
		t.Fatal(err)
	}

	take := func() (string, int) {
		t.Helper()
		go serveOne()
		connection, err := net.DialUnix(
			"unixpacket",
			nil,
			&net.UnixAddr{Name: socketPath, Net: "unixpacket"},
		)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = connection.Close() }()
		request := fdBrokerProtocol + " TAKE M 42\n"
		if _, _, err := connection.WriteMsgUnix([]byte(request), nil, nil); err != nil {
			t.Fatal(err)
		}
		payload := make([]byte, 128)
		oob := make([]byte, unix.CmsgSpace(4))
		count, oobCount, _, _, err := connection.ReadMsgUnix(payload, oob)
		if err != nil {
			t.Fatal(err)
		}
		if string(payload[:count]) != "OK\n" {
			return string(payload[:count]), -1
		}
		fd, err := receivedFD(oob[:oobCount])
		if err != nil {
			t.Fatal(err)
		}
		return string(payload[:count]), fd
	}

	response, fd := take()
	if response != "OK\n" || fd < 0 {
		t.Fatalf("first TAKE response = %q fd=%d", response, fd)
	}
	if err := unix.Close(fd); err != nil {
		t.Fatal(err)
	}
	response, fd = take()
	if response != "ERR FD broker object is absent\n" || fd != -1 {
		t.Fatalf("second TAKE response = %q fd=%d", response, fd)
	}
}

func TestParseOptionsRequiresDumpPID(t *testing.T) {
	base := []string{
		"--action", "dump",
		"--cpu-only",
		"--images-dir", "/images",
		"--work-dir", "/work",
		"--log-file", "dump.log",
		"--external-files", "/artifact/external.json",
		"--result", "/artifact/result.json",
	}
	if _, err := parseOptions(base); err == nil {
		t.Fatal("dump without PID was accepted")
	}
	value, err := parseOptions(append(base, "--pid", "42"))
	if err != nil {
		t.Fatal(err)
	}
	if value.pid != 42 || value.action != "dump" {
		t.Fatalf("parsed options = %#v", value)
	}
}

func TestParseOptionsRequiresExplicitCUDAMode(t *testing.T) {
	arguments := []string{
		"--action", "dump",
		"--pid", "42",
		"--images-dir", "/images",
		"--work-dir", "/work",
		"--log-file", "dump.log",
		"--external-files", "/artifact/external.json",
		"--result", "/artifact/result.json",
	}
	if _, err := parseOptions(arguments); err == nil {
		t.Fatal("options without CUDA mode were accepted")
	}
	if _, err := parseOptions(append(arguments, "--cuda-checkpoint", "/bin/cuda-checkpoint")); err != nil {
		t.Fatal(err)
	}
}

func TestParseOptionsScopesNVIDIAPlaceholdersToCPUOnlyRestore(t *testing.T) {
	restore := []string{
		"--action", "restore",
		"--cpu-only",
		"--nvidia-placeholder-fds",
		"--images-dir", "/images",
		"--work-dir", "/work",
		"--log-file", "restore.log",
		"--external-files", "/artifact/external.json",
		"--result", "/artifact/result.json",
	}
	value, err := parseOptions(restore)
	if err != nil {
		t.Fatal(err)
	}
	if !value.nvidiaPlaceholderFDs {
		t.Fatal("NVIDIA placeholder mode was not preserved")
	}
	dump := append([]string{}, restore...)
	dump[1] = "dump"
	dump = append(dump, "--pid", "42")
	if _, err := parseOptions(dump); err == nil {
		t.Fatal("NVIDIA placeholder mode was accepted for dump")
	}
}

func TestParseOptionsAcceptsTP2RPCOptions(t *testing.T) {
	arguments := []string{
		"--action", "dump",
		"--pid", "42",
		"--cpu-only",
		"--tcp-established",
		"--ghost-limit", "67108864",
		"--network-lock", "nftables",
		"--compress-block-size", "262144",
		"--compress-acceleration", "3",
		"--decompress-threads", "0",
		"--image-io-mode", "direct",
		"--criu-plugin-dir", "/opt/coldsnap-criu-plugins",
		"--nvidia-fd-broker-socket", "/snapshot/nvidia-fd-broker.sock",
		"--link-remap-source-dir", "/dev/shm",
		"--link-remap-snapshot-dir", "/snapshot/shm-image",
		"--images-dir", "/images",
		"--work-dir", "/work",
		"--log-file", "dump.log",
		"--external-files", "/artifact/external.json",
		"--result", "/artifact/result.json",
	}
	value, err := parseOptions(arguments)
	if err != nil {
		t.Fatal(err)
	}
	if value.cudaTree || !value.cpuOnly || !value.tcpEstablished || value.ghostLimit != 64*1024*1024 || value.networkLock != "nftables" || value.compressBlockBytes != 256*1024 || value.compressAcceleration != 3 || value.decompressThreads != 0 || value.imageIOMode != "direct" || value.pluginDir != "/opt/coldsnap-criu-plugins" || value.fdBrokerSocket != "/snapshot/nvidia-fd-broker.sock" || value.linkRemapSourceDir != "/dev/shm" || value.linkRemapSnapshotDir != "/snapshot/shm-image" {
		t.Fatalf("parsed TP2 options = %#v", value)
	}
	restore := append([]string{}, arguments...)
	restore[1] = "restore"
	restore = append(restore[:2], restore[4:]...)
	if _, err := parseOptions(restore); err == nil {
		t.Fatal("network lock was accepted for restore")
	}
}

func TestParseOptionsRejectsUnsafePluginDirectory(t *testing.T) {
	base := []string{
		"--action", "dump",
		"--pid", "42",
		"--cpu-only",
		"--images-dir", "/images",
		"--work-dir", "/work",
		"--log-file", "dump.log",
		"--external-files", "/artifact/external.json",
		"--result", "/artifact/result.json",
	}
	if _, err := parseOptions(append(base, "--criu-plugin-dir", "relative")); err == nil {
		t.Fatal("relative CRIU plugin directory was accepted")
	}
	if _, err := parseOptions(append(base, "--nvidia-fd-broker-socket", "/broker.sock")); err == nil {
		t.Fatal("FD broker without CRIU plugin was accepted")
	}
	withoutCPUOnly := append([]string{}, base...)
	withoutCPUOnly = append(withoutCPUOnly[:4], withoutCPUOnly[5:]...)
	if _, err := parseOptions(append(withoutCPUOnly, "--cuda-checkpoint", "/bin/cuda-checkpoint", "--criu-plugin-dir", "/plugins")); err == nil {
		t.Fatal("CRIU plugin directory without CPU-only mode was accepted")
	}
	cudaRestore := []string{
		"--action", "cuda-restore",
		"--pid", "42",
		"--cuda-checkpoint", "/bin/cuda-checkpoint",
		"--criu-plugin-dir", "/plugins",
		"--result", "/artifact/result.json",
	}
	if _, err := parseOptions(cudaRestore); err == nil {
		t.Fatal("CRIU plugin directory was accepted for cuda-restore")
	}
}

func TestParseOptionsUsesQualifiedPageIODefaults(t *testing.T) {
	value, err := parseOptions([]string{
		"--action", "dump",
		"--pid", "42",
		"--cuda-checkpoint", "/bin/cuda-checkpoint",
		"--cuda-process-tree",
		"--images-dir", "/images",
		"--work-dir", "/work",
		"--log-file", "dump.log",
		"--external-files", "/artifact/external.json",
		"--result", "/artifact/result.json",
	})
	if err != nil {
		t.Fatal(err)
	}
	if value.compressBlockBytes != 256*1024 || value.compressAcceleration != 1 || value.decompressThreads != 1 || value.imageIOMode != "direct" {
		t.Fatalf("page I/O defaults = %#v", value)
	}
}

func TestParseOptionsRejectsInvalidPageIOControls(t *testing.T) {
	base := []string{
		"--action", "restore",
		"--cpu-only",
		"--images-dir", "/images",
		"--work-dir", "/work",
		"--log-file", "restore.log",
		"--external-files", "/artifact/external.json",
		"--result", "/artifact/result.json",
	}
	for _, arguments := range [][]string{
		append(append([]string{}, base...), "--compress-block-size", "4097"),
		append(append([]string{}, base...), "--compress-block-size", "8388608"),
		append(append([]string{}, base...), "--compress-acceleration", "0"),
		append(append([]string{}, base...), "--decompress-threads", "1025"),
		append(append([]string{}, base...), "--image-io-mode", "mystery"),
	} {
		if _, err := parseOptions(arguments); err == nil {
			t.Fatalf("invalid page I/O controls were accepted: %q", arguments)
		}
	}
}

func TestParseTCPAddressMapRejectsUnsafeAndAmbiguousMappings(t *testing.T) {
	for _, values := range [][]string{
		{"missing-separator"},
		{"10.0.0.1=not-an-address"},
		{"0.0.0.0=10.0.0.2"},
		{"127.0.0.1=10.0.0.2"},
		{"10.0.0.1=::1"},
		{"10.0.0.1=10.0.0.3", "10.0.0.1=10.0.0.4"},
		{"10.0.0.1=10.0.0.3", "10.0.0.2=10.0.0.3"},
	} {
		if _, err := parseTCPAddressMap(values); err == nil {
			t.Fatalf("unsafe TCP address mapping was accepted: %q", values)
		}
	}
	mapping, err := parseTCPAddressMap([]string{
		"10.24.11.13=10.24.11.17",
		"10.24.11.17=10.24.11.13",
	})
	if err != nil {
		t.Fatal(err)
	}
	if mapping[netip.MustParseAddr("10.24.11.13")] != netip.MustParseAddr("10.24.11.17") {
		t.Fatalf("TCP address mapping = %#v", mapping)
	}
}

func TestParseOptionsAcceptsRestoreTCPAddressMapOnly(t *testing.T) {
	base := []string{
		"--action", "restore",
		"--cpu-only",
		"--images-dir", "/images",
		"--work-dir", "/work",
		"--log-file", "restore.log",
		"--external-files", "/artifact/external.json",
		"--result", "/artifact/result.json",
		"--tcp-address-map", "10.24.11.13=10.24.11.17",
		"--tcp-port-shift", "4096",
		"--tcp-preserve-port", "8000",
		"--leave-stopped",
		"--defer-network-unlock",
	}
	value, err := parseOptions(base)
	if err != nil {
		t.Fatal(err)
	}
	if len(value.tcpAddressMap) != 1 {
		t.Fatalf("TCP address map = %#v", value.tcpAddressMap)
	}
	if value.tcpPortShift != 4096 {
		t.Fatalf("TCP port shift = %d", value.tcpPortShift)
	}
	if value.tcpPreservePort != 8000 {
		t.Fatalf("TCP preserved port = %d", value.tcpPreservePort)
	}
	if !value.leaveStopped {
		t.Fatal("leave-stopped restore option was not preserved")
	}
	if !value.deferNetworkUnlock {
		t.Fatal("deferred network unlock option was not preserved")
	}
	dump := append([]string{}, base...)
	dump[1] = "dump"
	dump = append(dump, "--pid", "42")
	if _, err := parseOptions(dump); err == nil {
		t.Fatal("dump TCP address mapping was accepted")
	}
}

func TestDeferNetworkUnlockWaitsForRegularRelease(t *testing.T) {
	directory := t.TempDir()
	type result struct {
		seconds float64
		err     error
	}
	completed := make(chan result, 1)
	go func() {
		seconds, err := deferNetworkUnlock(directory, time.Second)
		completed <- result{seconds: seconds, err: err}
	}()

	ready := filepath.Join(directory, networkUnlockReadyFile)
	deadline := time.Now().Add(time.Second)
	for {
		if info, err := os.Stat(ready); err == nil && info.Mode().IsRegular() {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("network-unlock readiness was not published")
		}
		time.Sleep(time.Millisecond)
	}
	select {
	case value := <-completed:
		t.Fatalf("network unlock completed before release: %#v", value)
	default:
	}
	if err := os.WriteFile(filepath.Join(directory, networkUnlockReleaseFile), []byte("1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	select {
	case value := <-completed:
		if value.err != nil {
			t.Fatal(value.err)
		}
		if value.seconds <= 0 {
			t.Fatalf("network unlock wait = %f", value.seconds)
		}
	case <-time.After(time.Second):
		t.Fatal("network unlock did not observe release")
	}
}

func TestParseOptionsAcceptsDeferredCUDARestore(t *testing.T) {
	value, err := parseOptions([]string{
		"--action", "cuda-restore",
		"--pid", "513",
		"--cuda-checkpoint", "/bin/cuda-checkpoint",
		"--cuda-process-tree",
		"--cuda-processes-result", "/artifact/dump-rpc.json",
		"--result", "/artifact/cuda-restore-rpc.json",
	})
	if err != nil {
		t.Fatal(err)
	}
	if value.action != "cuda-restore" || value.pid != 513 || !value.cudaTree {
		t.Fatalf("parsed deferred CUDA restore = %#v", value)
	}
	if value.cudaProcessesResult != "/artifact/dump-rpc.json" {
		t.Fatalf("captured CUDA process inventory = %q", value.cudaProcessesResult)
	}
	if _, err := parseOptions([]string{
		"--action", "cuda-restore",
		"--cuda-checkpoint", "/bin/cuda-checkpoint",
		"--result", "/artifact/cuda-restore-rpc.json",
	}); err == nil {
		t.Fatal("deferred CUDA restore without PID was accepted")
	}
	if _, err := parseOptions([]string{
		"--action", "cuda-restore",
		"--pid", "513",
		"--cuda-checkpoint", "/bin/cuda-checkpoint",
		"--result", "/artifact/cuda-restore-rpc.json",
		"--tcp-established",
	}); err == nil {
		t.Fatal("deferred CUDA restore accepted CRIU TCP options")
	}
}

func TestCapturedCUDAProcessesValidatesRestoredTree(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "dump-rpc.json")
	payload := commandResult{
		Format: manifestFormat,
		Kind:   resultKind,
		Action: "dump",
		CUDAProcesses: []cudaProcess{
			{PID: os.Getpid(), RestoreTID: os.Getpid() + 1000},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	processes, err := capturedCUDAProcesses(path, os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	if len(processes) != 1 || processes[0] != payload.CUDAProcesses[0] {
		t.Fatalf("captured CUDA processes = %#v", processes)
	}
	payload.CUDAProcesses[0].PID++
	data, err = json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := capturedCUDAProcesses(path, os.Getpid()); err == nil {
		t.Fatal("CUDA process outside the restored tree was accepted")
	}
}

func TestRebindTCPSocketEndpointsRewritesAddressesPortsAndPreservesHTTPListener(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "files.img")
	output, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	inetType := fdinfo.FdTypes_INETSK
	tcp := uint32(unix.IPPROTO_TCP)
	v4Source, _ := encodeCRIUAddress(netip.MustParseAddr("10.24.11.13"), 1)
	v4Destination, _ := encodeCRIUAddress(netip.MustParseAddr("10.24.11.17"), 1)
	v6Source, _ := encodeCRIUAddress(netip.MustParseAddr("10.24.11.13"), 4)
	v6Destination, _ := encodeCRIUAddress(netip.MustParseAddr("10.24.11.17"), 4)
	socket := func(id uint32, family uint32, source, destination []uint32) *sk_inet.InetSkEntry {
		return &sk_inet.InetSkEntry{
			Id: proto.Uint32(id), Ino: proto.Uint32(id), Family: proto.Uint32(family),
			Type: proto.Uint32(uint32(unix.SOCK_STREAM)), Proto: &tcp,
			State: proto.Uint32(1), SrcPort: proto.Uint32(10000 + id),
			DstPort: proto.Uint32(20000 + id), Flags: proto.Uint32(0), Backlog: proto.Uint32(0),
			SrcAddr: source, DstAddr: destination,
			Fown: &fown.FownEntry{
				Uid: proto.Uint32(0), Euid: proto.Uint32(0), Signum: proto.Uint32(0),
				PidType: proto.Uint32(0), Pid: proto.Uint32(0),
			},
			Opts: &sk_opts.SkOptsEntry{
				SoSndbuf: proto.Uint32(4096), SoRcvbuf: proto.Uint32(4096),
				SoSndTmoSec: proto.Uint64(0), SoSndTmoUsec: proto.Uint64(0),
				SoRcvTmoSec: proto.Uint64(0), SoRcvTmoUsec: proto.Uint64(0),
			},
		}
	}
	image := &crit.CriuImage{
		Magic: "FILES", EntryType: &fdinfo.FileEntry{},
		Entries: []*crit.CriuEntry{
			{Message: &fdinfo.FileEntry{
				Type: &inetType, Id: proto.Uint32(1),
				Isk: socket(1, uint32(unix.AF_INET), v4Source, v4Destination),
			}},
			{Message: &fdinfo.FileEntry{
				Type: &inetType, Id: proto.Uint32(2),
				Isk: socket(2, uint32(unix.AF_INET6), v6Source, v6Destination),
			}},
			{Message: &fdinfo.FileEntry{
				Type: &inetType, Id: proto.Uint32(3),
				Isk: func() *sk_inet.InetSkEntry {
					listener := socket(3, uint32(unix.AF_INET6), []uint32{0, 0, 0, 0}, []uint32{0, 0, 0, 0})
					listener.State = proto.Uint32(10) // Linux TCP_LISTEN
					listener.SrcPort = proto.Uint32(10002)
					listener.DstPort = proto.Uint32(0)
					return listener
				}(),
			}},
		},
	}
	if err := crit.New(nil, output, "", false, false).Encode(image); err != nil {
		_ = output.Close()
		t.Fatal(err)
	}
	if err := output.Close(); err != nil {
		t.Fatal(err)
	}
	mapping, err := parseTCPAddressMap([]string{
		"10.24.11.13=10.24.11.17",
		"10.24.11.17=10.24.11.13",
	})
	if err != nil {
		t.Fatal(err)
	}
	addressChanges, portChanges, err := rebindTCPSocketEndpoints(
		directory, mapping, 4096, 10001, tcpPortMapping{}, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if addressChanges != 4 || portChanges != 4 {
		t.Fatalf("changed endpoints = addresses %d, ports %d; want 4, 4", addressChanges, portChanges)
	}

	input, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	rebound, err := crit.New(input, nil, "", false, false).Decode(&fdinfo.FileEntry{})
	_ = input.Close()
	if err != nil {
		t.Fatal(err)
	}
	wantSource := netip.MustParseAddr("10.24.11.17")
	wantDestination := netip.MustParseAddr("10.24.11.13")
	for index, entry := range rebound.Entries {
		socket := entry.Message.(*fdinfo.FileEntry).GetIsk()
		if index == 2 {
			source, ok := decodeCRIUAddress(socket.GetSrcAddr())
			if !ok || !source.IsUnspecified() || socket.GetSrcPort() != rotatePortableTCPPort(10002, 4096) {
				t.Fatalf("rebound wildcard listener = %v:%d", source, socket.GetSrcPort())
			}
			continue
		}
		source, ok := decodeCRIUAddress(socket.GetSrcAddr())
		if !ok || source.Unmap() != wantSource {
			t.Fatalf("rebound source = %v", source)
		}
		destination, ok := decodeCRIUAddress(socket.GetDstAddr())
		if !ok || destination.Unmap() != wantDestination {
			t.Fatalf("rebound destination = %v", destination)
		}
		id := uint32(index + 1)
		wantSourcePort := rotatePortableTCPPort(10000+id, 4096)
		if id == 1 {
			wantSourcePort = 10001
		}
		if socket.GetSrcPort() != wantSourcePort ||
			socket.GetDstPort() != rotatePortableTCPPort(20000+id, 4096) {
			t.Fatalf("rebound ports = %d -> %d", socket.GetSrcPort(), socket.GetDstPort())
		}
	}
}

func TestProbeTCPBindEndpointsRejectsOccupiedPortAndReleasesProbe(t *testing.T) {
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().(*net.TCPAddr)
	endpoint := tcpBindEndpoint{
		ID: 17, Family: unix.AF_INET, Address: netip.MustParseAddr("127.0.0.1"),
		Port: uint32(address.Port), State: 10,
	}
	if err := probeTCPBindEndpoints([]tcpBindEndpoint{endpoint}); err == nil ||
		!errors.Is(err, unix.EADDRINUSE) ||
		!strings.Contains(err.Error(), "address already in use") {
		t.Fatalf("occupied TCP endpoint probe error = %v", err)
	}
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	if err := probeTCPBindEndpoints([]tcpBindEndpoint{endpoint}); err != nil {
		t.Fatalf("released TCP endpoint probe: %v", err)
	}
}

func TestParseTCPProbeOptionsRequiresPortableInputs(t *testing.T) {
	value, err := parseTCPProbeOptions([]string{
		"--images-dir", "/opt/coldsnap/capsule/images",
		"--tcp-address-map", "10.24.11.13=10.24.11.17",
		"--tcp-port-shift", "4096",
		"--tcp-preserve-port", "8000",
		"--tcp-port-map", "8000=8103",
		"--tcp-allow-empty-map",
	})
	if err != nil {
		t.Fatal(err)
	}
	if value.tcpPortShift != 4096 || value.tcpPreservePort != 8000 ||
		value.tcpPortMap != "8000=8103" || !value.allowEmptyMap {
		t.Fatalf("TCP probe options = %#v", value)
	}
	for _, arguments := range [][]string{
		{"--images-dir", "relative", "--tcp-address-map", "10.0.0.1=10.0.0.2", "--tcp-port-shift", "1"},
		{"--images-dir", "/images", "--tcp-port-shift", "1"},
		{"--images-dir", "/images", "--tcp-address-map", "10.0.0.1=10.0.0.2", "--tcp-port-shift", "0"},
		{"--images-dir", "/images", "--tcp-address-map", "10.0.0.1=10.0.0.2", "--tcp-port-shift", "1", "--tcp-port-map", "80=8103"},
	} {
		if _, err := parseTCPProbeOptions(arguments); err == nil {
			t.Fatalf("unsafe TCP probe options accepted: %q", arguments)
		}
	}
}

func TestParseTCPPortMap(t *testing.T) {
	mapping, err := parseTCPPortMap("8000=8103")
	if err != nil || !mapping.Configured || mapping.Source != 8000 || mapping.Destination != 8103 {
		t.Fatalf("TCP port mapping = %#v, %v", mapping, err)
	}
	for _, value := range []string{"8000", "80=8103", "8000=80", "8000=8000", "nope=8103"} {
		if _, err := parseTCPPortMap(value); err == nil {
			t.Fatalf("invalid TCP port mapping was accepted: %q", value)
		}
	}
}

func TestDiscoverCUDAProcessesTraversesChildren(t *testing.T) {
	child := exec.Command("sh", "-c", "sleep 30")
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = child.Process.Kill()
		_ = child.Wait()
	}()

	executable := filepath.Join(t.TempDir(), "cuda-checkpoint")
	script := "#!/bin/sh\n[ \"$1\" = --get-restore-tid ] || exit 2\n[ \"$2\" = --pid ] || exit 2\nexpr \"$3\" + 1000\n"
	if err := os.WriteFile(executable, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	processes, err := discoverCUDAProcesses(options{
		cudaPath: executable,
		cudaTree: true,
		timeout:  1,
	}, os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	found := make(map[int]bool, len(processes))
	for _, process := range processes {
		found[process.PID] = true
		if process.RestoreTID != process.PID+1000 {
			t.Fatalf("CUDA process = %#v", process)
		}
	}
	if !found[os.Getpid()] || !found[child.Process.Pid] {
		t.Fatalf("CUDA processes = %#v, want root %d and child %d", processes, os.Getpid(), child.Process.Pid)
	}
}

func TestRequireCUDAState(t *testing.T) {
	executable := filepath.Join(t.TempDir(), "cuda-checkpoint")
	if err := os.WriteFile(executable, []byte("#!/bin/sh\necho locked\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	value := options{cudaPath: executable, timeout: 1}
	result := commandResult{}
	if err := requireCUDAState(&result, value, 42, "locked"); err != nil {
		t.Fatal(err)
	}
	if err := requireCUDAState(&result, value, 42, "running"); err == nil {
		t.Fatal("unexpected CUDA state was accepted")
	}
	if len(result.CUDAActions) != 2 || result.CUDAActions[0].Output != "locked" {
		t.Fatalf("CUDA action trace = %#v", result.CUDAActions)
	}
}

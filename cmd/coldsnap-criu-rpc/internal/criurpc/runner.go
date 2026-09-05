// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

// Package criurpc runs dump and restore through CRIU's RPC service worker.
// Plugins remain disabled unless an explicit CPU-only reset-epoch plugin
// directory is supplied. NVIDIA character-device descriptors are handled
// separately through go-criu's allowlisted external-file callbacks.
package criurpc

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/netip"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	criu "github.com/checkpoint-restore/go-criu/v8"
	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
	"github.com/checkpoint-restore/go-criu/v8/rpc"
	"golang.org/x/sys/unix"
)

const (
	manifestFormat               = 1
	manifestKind                 = "coldsnap-criu-external-files"
	resultKind                   = "coldsnap-criu-rpc-result"
	nvidiaExternalMapHeader      = "coldsnap-nvidia-external-files-v1"
	nvidiaExternalMapEnvironment = "COLDSNAP_CRIU_NVIDIA_EXTERNAL_FILE_MAP"
	nvidiaExternalMapBasename    = "coldsnap-nvidia-external-files-v1.map"
	nvidiaPlaceholderEnvironment = "COLDSNAP_CRIU_NVIDIA_PLACEHOLDER_FDS"
	nvidiaFDBrokerEnvironment    = "COLDSNAP_CRIU_NVIDIA_FD_BROKER_SOCKET"
	defaultLZ4Block              = 256 * 1024
	maxLZ4Block                  = 4 * 1024 * 1024
	maxLZ4Accel                  = 65537
	compressBlock                = 1
)

var numberedNVIDIADevice = regexp.MustCompile(`^/dev/nvidia[0-9]+$`)
var nvidiaCapabilityDevice = regexp.MustCompile(`^/dev/nvidia-caps/nvidia-cap[0-9]+$`)
var linkRemapFile = regexp.MustCompile(`^link_remap\.[0-9]+$`)

type externalFile struct {
	ID          uint32 `json:"id"`
	Path        string `json:"path"`
	OpenFlags   int    `json:"open_flags"`
	DeviceMajor uint32 `json:"device_major"`
	DeviceMinor uint32 `json:"device_minor"`
}

type externalManifest struct {
	Format int            `json:"format"`
	Kind   string         `json:"kind"`
	Files  []externalFile `json:"files"`
}

type commandResult struct {
	Format                   int           `json:"format"`
	Kind                     string        `json:"kind"`
	Action                   string        `json:"action"`
	RestoredPID              int           `json:"restored_pid,omitempty"`
	TreeResumedForCUDA       bool          `json:"tree_resumed_for_cuda,omitempty"`
	TreeRestopped            bool          `json:"tree_restopped,omitempty"`
	ExternalFileCount        int           `json:"external_file_count"`
	TCPAddressRemaps         int           `json:"tcp_address_remaps,omitempty"`
	TCPPortRemaps            int           `json:"tcp_port_remaps,omitempty"`
	NetworkUnlockWaitSeconds float64       `json:"network_unlock_wait_seconds,omitempty"`
	CUDAProcesses            []cudaProcess `json:"cuda_processes,omitempty"`
	CUDAActions              []cudaAction  `json:"cuda_actions,omitempty"`
	CRIUOptions              criuOptions   `json:"criu_options"`
	Error                    string        `json:"error,omitempty"`
}

type criuOptions struct {
	CompressionBlockBytes   uint64 `json:"compression_block_bytes,omitempty"`
	CompressionAcceleration uint64 `json:"compression_acceleration,omitempty"`
	DecompressThreads       uint64 `json:"decompress_threads"`
	ImageIOMode             string `json:"image_io_mode"`
	LeaveStopped            bool   `json:"leave_stopped,omitempty"`
	LeaveRunning            bool   `json:"leave_running,omitempty"`
	NVIDIAPlaceholderFDs    bool   `json:"nvidia_placeholder_fds,omitempty"`
	DeferNetworkUnlock      bool   `json:"defer_network_unlock,omitempty"`
}

type cudaProcess struct {
	PID        int `json:"pid"`
	RestoreTID int `json:"restore_tid,omitempty"`
}

type cudaAction struct {
	Operation string  `json:"operation"`
	PID       int     `json:"pid"`
	Seconds   float64 `json:"seconds"`
	Output    string  `json:"output,omitempty"`
}

type externalNotifier struct {
	criu.NoNotify
	action          string
	manifestPath    string
	dumpFiles       map[uint32]externalFile
	restoreFiles    map[uint32]externalFile
	restoredFileIDs map[uint32]bool
	postRestorePID  int32
	fdBrokerSocket  string
	placeholderFDs  bool
	networkUnlock   func() error
	linkRemapSource string
	linkRemapTarget string
}

func ptr[T any](value T) *T { return &value }

func allowedNVIDIADevice(path string) bool {
	switch path {
	case "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset":
		return true
	default:
		return numberedNVIDIADevice.MatchString(path) || nvidiaCapabilityDevice.MatchString(path)
	}
}

func validateExternalFile(file externalFile) error {
	if !allowedNVIDIADevice(file.Path) || filepath.Clean(file.Path) != file.Path {
		return fmt.Errorf("external file %d has disallowed path %q", file.ID, file.Path)
	}
	accessMode := file.OpenFlags & unix.O_ACCMODE
	if accessMode != unix.O_RDONLY && accessMode != unix.O_WRONLY && accessMode != unix.O_RDWR {
		return fmt.Errorf("external file %d has invalid access mode %#x", file.ID, accessMode)
	}
	if file.OpenFlags & ^(unix.O_ACCMODE|unix.O_NONBLOCK) != 0 {
		return fmt.Errorf("external file %d has unsupported open flags %#x", file.ID, file.OpenFlags)
	}
	return nil
}

func inspectExternalFile(id uint32, file *os.File) (externalFile, error) {
	path, err := os.Readlink(fmt.Sprintf("/proc/self/fd/%d", file.Fd()))
	if err != nil {
		return externalFile{}, fmt.Errorf("resolve external file %d: %w", id, err)
	}
	flags, err := unix.FcntlInt(file.Fd(), unix.F_GETFL, 0)
	if err != nil {
		return externalFile{}, fmt.Errorf("read flags for external file %d: %w", id, err)
	}
	information, err := file.Stat()
	if err != nil {
		return externalFile{}, fmt.Errorf("stat external file %d: %w", id, err)
	}
	stat, ok := information.Sys().(*syscall.Stat_t)
	if !ok {
		return externalFile{}, fmt.Errorf("external file %d has unsupported stat data", id)
	}
	if uint32(stat.Mode)&unix.S_IFMT != unix.S_IFCHR {
		return externalFile{}, fmt.Errorf("external file %d at %q is not a character device", id, path)
	}
	record := externalFile{
		ID:          id,
		Path:        path,
		OpenFlags:   flags & (unix.O_ACCMODE | unix.O_NONBLOCK),
		DeviceMajor: unix.Major(uint64(stat.Rdev)),
		DeviceMinor: unix.Minor(uint64(stat.Rdev)),
	}
	if err := validateExternalFile(record); err != nil {
		return externalFile{}, err
	}
	return record, nil
}

func openExternalFile(record externalFile) (*os.File, error) {
	if err := validateExternalFile(record); err != nil {
		return nil, err
	}
	fd, err := unix.Open(record.Path, record.OpenFlags|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0)
	if err != nil {
		return nil, fmt.Errorf("open external file %d at %q: %w", record.ID, record.Path, err)
	}
	closeOnError := true
	defer func() {
		if closeOnError {
			_ = unix.Close(fd)
		}
	}()
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil {
		return nil, fmt.Errorf("stat restored external file %d: %w", record.ID, err)
	}
	if uint32(stat.Mode)&unix.S_IFMT != unix.S_IFCHR {
		return nil, fmt.Errorf("restored external file %d is not a character device", record.ID)
	}
	major := unix.Major(uint64(stat.Rdev))
	minor := unix.Minor(uint64(stat.Rdev))
	if major != record.DeviceMajor || minor != record.DeviceMinor {
		return nil, fmt.Errorf(
			"external file %d device changed: got %d:%d, want %d:%d",
			record.ID, major, minor, record.DeviceMajor, record.DeviceMinor,
		)
	}
	closeOnError = false
	return os.NewFile(uintptr(fd), fmt.Sprintf("criu-external-%d", record.ID)), nil
}

func openExternalPlaceholder(record externalFile) (*os.File, error) {
	if err := validateExternalFile(record); err != nil {
		return nil, err
	}
	temporary, err := os.CreateTemp("", fmt.Sprintf("coldsnap-nvidia-placeholder-%d-", record.ID))
	if err != nil {
		return nil, fmt.Errorf("create placeholder for external file %d: %w", record.ID, err)
	}
	path := temporary.Name()
	if err := temporary.Close(); err != nil {
		_ = os.Remove(path)
		return nil, fmt.Errorf("close placeholder seed for external file %d: %w", record.ID, err)
	}
	fd, err := unix.Open(
		path,
		record.OpenFlags|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		_ = os.Remove(path)
		return nil, fmt.Errorf("open placeholder for external file %d: %w", record.ID, err)
	}
	if err := os.Remove(path); err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("unlink placeholder for external file %d: %w", record.ID, err)
	}
	return os.NewFile(uintptr(fd), fmt.Sprintf("criu-placeholder-%d", record.ID)), nil
}

func writeJSONAtomically(path string, value any) error {
	directory := filepath.Dir(path)
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	encoder := json.NewEncoder(temporary)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return err
	}
	directoryFile, err := os.Open(directory)
	if err != nil {
		return err
	}
	defer func() { _ = directoryFile.Close() }()
	return directoryFile.Sync()
}

func loadManifest(path string) (externalManifest, error) {
	file, err := os.Open(path)
	if err != nil {
		return externalManifest{}, err
	}
	defer func() { _ = file.Close() }()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var manifest externalManifest
	if err := decoder.Decode(&manifest); err != nil {
		return externalManifest{}, err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return externalManifest{}, errors.New("external-file manifest has trailing data")
	}
	if manifest.Format != manifestFormat || manifest.Kind != manifestKind {
		return externalManifest{}, fmt.Errorf(
			"unsupported external-file manifest format: %d %q",
			manifest.Format, manifest.Kind,
		)
	}
	seen := make(map[uint32]bool, len(manifest.Files))
	for _, record := range manifest.Files {
		if seen[record.ID] {
			return externalManifest{}, fmt.Errorf("duplicate external file id %d", record.ID)
		}
		seen[record.ID] = true
		if err := validateExternalFile(record); err != nil {
			return externalManifest{}, err
		}
	}
	return manifest, nil
}

func writeNVIDIAExternalMap(directory string, files map[uint32]externalFile) (string, error) {
	ids := make([]uint32, 0, len(files))
	for id := range files {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	var payload strings.Builder
	payload.WriteString(nvidiaExternalMapHeader)
	payload.WriteByte('\n')
	for _, id := range ids {
		record := files[id]
		if err := validateExternalFile(record); err != nil {
			return "", err
		}
		if _, err := fmt.Fprintf(
			&payload,
			"%d\t%d\t%d\t%d\t%s\n",
			record.ID,
			record.OpenFlags,
			record.DeviceMajor,
			record.DeviceMinor,
			record.Path,
		); err != nil {
			return "", err
		}
	}
	path := filepath.Join(directory, nvidiaExternalMapBasename)
	temporary, err := os.CreateTemp(directory, "."+nvidiaExternalMapBasename+".*.tmp")
	if err != nil {
		return "", err
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return "", err
	}
	if _, err := io.WriteString(temporary, payload.String()); err != nil {
		_ = temporary.Close()
		return "", err
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return "", err
	}
	if err := temporary.Close(); err != nil {
		return "", err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return "", err
	}
	directoryFile, err := os.Open(directory)
	if err != nil {
		return "", err
	}
	defer func() { _ = directoryFile.Close() }()
	if err := directoryFile.Sync(); err != nil {
		return "", err
	}
	return path, nil
}

func (notifier *externalNotifier) DumpExternalFile(id uint32, file *os.File) error {
	if notifier.action != "dump" {
		return errors.New("dump-external-file notification during restore")
	}
	if _, exists := notifier.dumpFiles[id]; exists {
		return fmt.Errorf("duplicate external file id %d", id)
	}
	record, err := inspectExternalFile(id, file)
	if err != nil {
		return err
	}
	if notifier.fdBrokerSocket != "" {
		if err := brokerPut(notifier.fdBrokerSocket, 'E', uint64(id), file); err != nil {
			return fmt.Errorf("preserve external file %d in FD broker: %w", id, err)
		}
	}
	notifier.dumpFiles[id] = record
	return nil
}

func (notifier *externalNotifier) PostDump() error {
	if notifier.action != "dump" {
		return nil
	}
	files := make([]externalFile, 0, len(notifier.dumpFiles))
	for _, record := range notifier.dumpFiles {
		files = append(files, record)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].ID < files[j].ID })
	if err := writeJSONAtomically(notifier.manifestPath, externalManifest{
		Format: manifestFormat,
		Kind:   manifestKind,
		Files:  files,
	}); err != nil {
		return err
	}
	if notifier.fdBrokerSocket != "" {
		if err := brokerSeal(notifier.fdBrokerSocket); err != nil {
			return fmt.Errorf("seal NVIDIA FD broker: %w", err)
		}
	}
	if notifier.linkRemapTarget != "" {
		if err := snapshotLinkRemaps(notifier.linkRemapSource, notifier.linkRemapTarget); err != nil {
			return fmt.Errorf("snapshot CRIU link remaps: %w", err)
		}
	}
	return nil
}

func snapshotLinkRemaps(sourceDirectory, targetDirectory string) error {
	entries, err := os.ReadDir(sourceDirectory)
	if err != nil {
		return fmt.Errorf("read source directory: %w", err)
	}
	if err := os.Mkdir(targetDirectory, 0o700); err != nil {
		return fmt.Errorf("create target directory: %w", err)
	}
	for _, entry := range entries {
		if !linkRemapFile.MatchString(entry.Name()) {
			continue
		}
		sourcePath := filepath.Join(sourceDirectory, entry.Name())
		targetPath := filepath.Join(targetDirectory, entry.Name())
		sourceFD, err := unix.Open(sourcePath, unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0)
		if err != nil {
			return fmt.Errorf("open %s: %w", entry.Name(), err)
		}
		source := os.NewFile(uintptr(sourceFD), sourcePath)
		information, statErr := source.Stat()
		if statErr != nil || !information.Mode().IsRegular() {
			_ = source.Close()
			if statErr != nil {
				return fmt.Errorf("stat %s: %w", entry.Name(), statErr)
			}
			return fmt.Errorf("link remap is not a regular file: %s", entry.Name())
		}
		targetFD, err := unix.Open(
			targetPath,
			unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_CLOEXEC|unix.O_NOFOLLOW,
			uint32(information.Mode().Perm()),
		)
		if err != nil {
			_ = source.Close()
			return fmt.Errorf("create %s: %w", entry.Name(), err)
		}
		target := os.NewFile(uintptr(targetFD), targetPath)
		_, copyErr := io.Copy(target, source)
		syncErr := target.Sync()
		closeTargetErr := target.Close()
		closeSourceErr := source.Close()
		if err := errors.Join(copyErr, syncErr, closeTargetErr, closeSourceErr); err != nil {
			return fmt.Errorf("copy %s: %w", entry.Name(), err)
		}
	}
	directory, err := os.Open(targetDirectory)
	if err != nil {
		return err
	}
	defer func() { _ = directory.Close() }()
	return directory.Sync()
}

func (notifier *externalNotifier) RestoreExternalFile(id uint32) (*os.File, error) {
	if notifier.action != "restore" {
		return nil, errors.New("restore-external-file notification during dump")
	}
	record, exists := notifier.restoreFiles[id]
	if !exists {
		return nil, fmt.Errorf("external file id %d is absent from the artifact manifest", id)
	}
	if notifier.restoredFileIDs[id] {
		return nil, fmt.Errorf("external file id %d was requested more than once", id)
	}
	var file *os.File
	var err error
	if notifier.placeholderFDs {
		file, err = openExternalPlaceholder(record)
	} else {
		file, err = openExternalFile(record)
	}
	if err != nil {
		return nil, err
	}
	notifier.restoredFileIDs[id] = true
	return file, nil
}

func (notifier *externalNotifier) PostRestore(pid int32) error {
	notifier.postRestorePID = pid
	return nil
}

func (notifier *externalNotifier) NetworkUnlock() error {
	if notifier.networkUnlock == nil {
		return nil
	}
	return notifier.networkUnlock()
}

const (
	networkUnlockReadyFile   = "network-unlock-ready"
	networkUnlockReleaseFile = "network-unlock-release"
)

func deferNetworkUnlock(workDir string, timeout time.Duration) (float64, error) {
	readyPath := filepath.Join(workDir, networkUnlockReadyFile)
	releasePath := filepath.Join(workDir, networkUnlockReleaseFile)
	for _, path := range []string{readyPath, releasePath} {
		if _, err := os.Lstat(path); err == nil {
			return 0, fmt.Errorf("network-unlock barrier file already exists: %s", path)
		} else if !errors.Is(err, os.ErrNotExist) {
			return 0, fmt.Errorf("inspect network-unlock barrier file %s: %w", path, err)
		}
	}
	ready, err := os.OpenFile(readyPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL|unix.O_CLOEXEC, 0o600)
	if err != nil {
		return 0, fmt.Errorf("publish network-unlock readiness: %w", err)
	}
	if err := ready.Close(); err != nil {
		return 0, fmt.Errorf("close network-unlock readiness file: %w", err)
	}

	started := time.Now()
	deadline := started.Add(timeout)
	for {
		info, err := os.Lstat(releasePath)
		if err == nil {
			if !info.Mode().IsRegular() {
				return 0, fmt.Errorf("network-unlock release is not a regular file: %s", releasePath)
			}
			return time.Since(started).Seconds(), nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return 0, fmt.Errorf("inspect network-unlock release: %w", err)
		}
		if time.Now().After(deadline) {
			return 0, fmt.Errorf("timed out waiting for network-unlock release after %s", timeout)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

type options struct {
	action               string
	criuPath             string
	pluginDir            string
	fdBrokerSocket       string
	cudaPath             string
	cudaProcessesResult  string
	cpuOnly              bool
	cudaTree             bool
	tcpEstablished       bool
	ghostLimit           uint64
	networkLock          string
	imagesDir            string
	workDir              string
	logFile              string
	manifestPath         string
	resultPath           string
	pid                  int
	timeout              int
	compressBlockBytes   uint64
	compressAcceleration uint64
	decompressThreads    uint64
	imageIOMode          string
	tcpAddressMap        repeatedString
	tcpPortShift         uint
	tcpPreservePort      uint
	tcpPortMap           string
	tcpAllowEmptyMap     bool
	leaveStopped         bool
	leaveRunning         bool
	nvidiaPlaceholderFDs bool
	deferNetworkUnlock   bool
	linkRemapSourceDir   string
	linkRemapSnapshotDir string
}

type repeatedString []string

func (values *repeatedString) String() string { return strings.Join(*values, ",") }

func (values *repeatedString) Set(value string) error {
	*values = append(*values, value)
	return nil
}

func parseOptions(arguments []string) (options, error) {
	var value options
	flags := flag.NewFlagSet("coldsnap-criu-rpc", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.StringVar(&value.action, "action", "", "dump, restore, or cuda-restore")
	flags.StringVar(&value.criuPath, "criu", "criu", "CRIU executable")
	flags.StringVar(&value.pluginDir, "criu-plugin-dir", "", "explicit CRIU plugin directory; empty disables plugins")
	flags.StringVar(&value.fdBrokerSocket, "nvidia-fd-broker-socket", "", "absolute reset-epoch NVIDIA FD broker socket")
	flags.StringVar(&value.cudaPath, "cuda-checkpoint", "", "CUDA checkpoint executable")
	flags.StringVar(
		&value.cudaProcessesResult,
		"cuda-processes-result",
		"",
		"captured dump result containing the exact CUDA process inventory",
	)
	flags.BoolVar(&value.cpuOnly, "cpu-only", false, "explicitly disable CUDA actions")
	flags.BoolVar(&value.cudaTree, "cuda-process-tree", false, "checkpoint every CUDA process in the target tree")
	flags.BoolVar(&value.tcpEstablished, "tcp-established", false, "allow established TCP connections")
	flags.Uint64Var(&value.ghostLimit, "ghost-limit", 0, "maximum ghost-file size in bytes")
	flags.StringVar(&value.networkLock, "network-lock", "", "network lock method: iptables, nftables, or skip")
	flags.StringVar(&value.imagesDir, "images-dir", "", "CRIU image directory")
	flags.StringVar(&value.workDir, "work-dir", "", "CRIU work directory")
	flags.StringVar(&value.logFile, "log-file", "", "CRIU log basename")
	flags.StringVar(&value.manifestPath, "external-files", "", "external-file manifest")
	flags.StringVar(&value.resultPath, "result", "", "result JSON path")
	flags.IntVar(&value.pid, "pid", 0, "dump root PID")
	flags.IntVar(&value.timeout, "timeout", 600, "CRIU timeout in seconds")
	flags.Uint64Var(&value.compressBlockBytes, "compress-block-size", defaultLZ4Block, "LZ4 compression block size in bytes for dump; zero disables compression")
	flags.Uint64Var(&value.compressAcceleration, "compress-acceleration", 1, "LZ4 compression acceleration in [1, 65537]")
	flags.Uint64Var(&value.decompressThreads, "decompress-threads", 1, "restore-wide LZ4 worker concurrency in [0, 1024]; zero is automatic")
	flags.StringVar(&value.imageIOMode, "image-io-mode", "direct", "CRIU page-image I/O mode: writeback or direct")
	flags.Var(
		&value.tcpAddressMap,
		"tcp-address-map",
		"restore TCP endpoint address mapping in OLD=NEW form; repeat for each captured rank address",
	)
	flags.UintVar(
		&value.tcpPortShift,
		"tcp-port-shift",
		0,
		"restore TCP endpoint port rotation in [1, 64511]; zero preserves captured ports",
	)
	flags.UintVar(
		&value.tcpPreservePort,
		"tcp-preserve-port",
		0,
		"externally managed TCP listener port excluded from restore rotation",
	)
	flags.StringVar(
		&value.tcpPortMap,
		"tcp-port-map",
		"",
		"explicit restored TCP port mapping in OLD=NEW form",
	)
	flags.BoolVar(
		&value.tcpAllowEmptyMap,
		"tcp-allow-empty-map",
		false,
		"allow a launch unit whose CRIU inventory has no mapped TCP endpoints",
	)
	flags.BoolVar(
		&value.leaveStopped,
		"leave-stopped",
		false,
		"leave the restored process tree stopped for an external distributed resume barrier",
	)
	flags.BoolVar(
		&value.leaveRunning,
		"leave-running",
		false,
		"resume the process tree after a successful dump",
	)
	flags.BoolVar(
		&value.nvidiaPlaceholderFDs,
		"nvidia-placeholder-fds",
		false,
		"restore allowlisted NVIDIA descriptors as disposable /dev/null placeholders",
	)
	flags.BoolVar(
		&value.deferNetworkUnlock,
		"defer-network-unlock",
		false,
		"hold CRIU's network-unlock notification for an external distributed barrier",
	)
	flags.StringVar(
		&value.linkRemapSourceDir,
		"link-remap-source-dir",
		"",
		"shared-memory directory containing CRIU link_remap files",
	)
	flags.StringVar(
		&value.linkRemapSnapshotDir,
		"link-remap-snapshot-dir",
		"",
		"exclusive artifact directory for CRIU link_remap files",
	)
	if err := flags.Parse(arguments); err != nil {
		return options{}, err
	}
	if flags.NArg() != 0 {
		return options{}, fmt.Errorf("unexpected positional arguments: %q", flags.Args())
	}
	if value.action != "dump" && value.action != "restore" && value.action != "cuda-restore" {
		return options{}, errors.New("--action must be dump, restore, or cuda-restore")
	}
	if (value.action == "dump" || value.action == "cuda-restore") && value.pid <= 0 {
		return options{}, fmt.Errorf("--pid must be positive for %s", value.action)
	}
	if value.action == "restore" && value.pid != 0 {
		return options{}, errors.New("--pid is valid only for dump or cuda-restore")
	}
	if value.cudaPath == "" && !value.cpuOnly {
		return options{}, errors.New("--cuda-checkpoint is required unless --cpu-only is set")
	}
	if value.pluginDir != "" && !filepath.IsAbs(value.pluginDir) {
		return options{}, errors.New("--criu-plugin-dir must be absolute")
	}
	if value.pluginDir != "" && !value.cpuOnly {
		return options{}, errors.New("--criu-plugin-dir requires --cpu-only")
	}
	if value.fdBrokerSocket != "" && !filepath.IsAbs(value.fdBrokerSocket) {
		return options{}, errors.New("--nvidia-fd-broker-socket must be absolute")
	}
	if value.fdBrokerSocket != "" && (value.pluginDir == "" || !value.cpuOnly) {
		return options{}, errors.New("--nvidia-fd-broker-socket requires the CPU-only CRIU plugin")
	}
	if value.action == "cuda-restore" && value.pluginDir != "" {
		return options{}, errors.New("--criu-plugin-dir is invalid for cuda-restore")
	}
	if value.cudaPath != "" && value.cpuOnly {
		return options{}, errors.New("--cuda-checkpoint and --cpu-only are mutually exclusive")
	}
	if value.cudaTree && value.cpuOnly {
		return options{}, errors.New("--cuda-process-tree is invalid with --cpu-only")
	}
	if value.cudaProcessesResult != "" {
		if value.action != "cuda-restore" || !value.cudaTree {
			return options{}, errors.New("--cuda-processes-result requires cuda-restore with --cuda-process-tree")
		}
		if !filepath.IsAbs(value.cudaProcessesResult) ||
			filepath.Clean(value.cudaProcessesResult) != value.cudaProcessesResult {
			return options{}, errors.New("--cuda-processes-result must be a clean absolute path")
		}
	}
	if value.action == "cuda-restore" && value.cpuOnly {
		return options{}, errors.New("--cpu-only is invalid for cuda-restore")
	}
	if value.nvidiaPlaceholderFDs && (value.action != "restore" || !value.cpuOnly) {
		return options{}, errors.New("--nvidia-placeholder-fds requires CPU-only restore")
	}
	if value.ghostLimit > uint64(^uint32(0)) {
		return options{}, errors.New("--ghost-limit exceeds the CRIU RPC uint32 range")
	}
	if value.networkLock != "" && value.networkLock != "iptables" && value.networkLock != "nftables" && value.networkLock != "skip" {
		return options{}, errors.New("--network-lock must be iptables, nftables, or skip")
	}
	if value.action != "dump" && value.networkLock != "" {
		return options{}, errors.New("--network-lock is valid only for dump")
	}
	if value.action == "cuda-restore" && (value.tcpEstablished || value.ghostLimit != 0 ||
		value.imagesDir != "" || value.workDir != "" || value.logFile != "" || value.manifestPath != "") {
		return options{}, errors.New("CRIU image and TCP options are invalid for cuda-restore")
	}
	if value.action != "restore" && len(value.tcpAddressMap) != 0 {
		return options{}, errors.New("--tcp-address-map is valid only for restore")
	}
	if value.action != "restore" && value.tcpPortShift != 0 {
		return options{}, errors.New("--tcp-port-shift is valid only for restore")
	}
	if value.action != "restore" && value.tcpPreservePort != 0 {
		return options{}, errors.New("--tcp-preserve-port is valid only for restore")
	}
	if value.action != "restore" && value.leaveStopped {
		return options{}, errors.New("--leave-stopped is valid only for restore")
	}
	if value.action != "dump" && value.leaveRunning {
		return options{}, errors.New("--leave-running is valid only for dump")
	}
	if (value.linkRemapSourceDir == "") != (value.linkRemapSnapshotDir == "") {
		return options{}, errors.New("--link-remap-source-dir and --link-remap-snapshot-dir must be set together")
	}
	if value.linkRemapSnapshotDir != "" {
		if value.action != "dump" {
			return options{}, errors.New("link-remap snapshot options are valid only for dump")
		}
		if !filepath.IsAbs(value.linkRemapSourceDir) || !filepath.IsAbs(value.linkRemapSnapshotDir) ||
			filepath.Clean(value.linkRemapSourceDir) != value.linkRemapSourceDir ||
			filepath.Clean(value.linkRemapSnapshotDir) != value.linkRemapSnapshotDir ||
			value.linkRemapSourceDir == value.linkRemapSnapshotDir {
			return options{}, errors.New("link-remap snapshot directories must be distinct clean absolute paths")
		}
	}
	if value.action != "restore" && value.deferNetworkUnlock {
		return options{}, errors.New("--defer-network-unlock is valid only for restore")
	}
	if value.deferNetworkUnlock && (!value.leaveStopped || len(value.tcpAddressMap) == 0) {
		return options{}, errors.New("--defer-network-unlock requires --leave-stopped and --tcp-address-map")
	}
	if value.tcpPortShift > 64511 {
		return options{}, errors.New("--tcp-port-shift must be in [0, 64511]")
	}
	if value.tcpPreservePort > 65535 {
		return options{}, errors.New("--tcp-preserve-port must be in [0, 65535]")
	}
	if value.tcpPortShift != 0 && len(value.tcpAddressMap) == 0 {
		return options{}, errors.New("--tcp-port-shift requires --tcp-address-map")
	}
	if value.tcpPreservePort != 0 && value.tcpPortShift == 0 {
		return options{}, errors.New("--tcp-preserve-port requires --tcp-port-shift")
	}
	if value.action != "restore" && value.tcpPortMap != "" {
		return options{}, errors.New("--tcp-port-map is valid only for restore")
	}
	if value.action != "restore" && value.tcpAllowEmptyMap {
		return options{}, errors.New("--tcp-allow-empty-map is valid only for restore")
	}
	if value.tcpAllowEmptyMap && len(value.tcpAddressMap) == 0 {
		return options{}, errors.New("--tcp-allow-empty-map requires --tcp-address-map")
	}
	if _, err := parseTCPPortMap(value.tcpPortMap); err != nil {
		return options{}, err
	}
	if _, err := parseTCPAddressMap(value.tcpAddressMap); err != nil {
		return options{}, err
	}
	if value.resultPath == "" {
		return options{}, errors.New("--result is required")
	}
	if value.action != "cuda-restore" && (value.imagesDir == "" || value.workDir == "" || value.manifestPath == "") {
		return options{}, errors.New("--images-dir, --work-dir, --external-files, and --result are required")
	}
	if value.action != "cuda-restore" && (value.logFile == "" || filepath.Base(value.logFile) != value.logFile) {
		return options{}, errors.New("--log-file must be a basename")
	}
	if value.timeout <= 0 || value.timeout > 3600 {
		return options{}, errors.New("--timeout must be in [1, 3600]")
	}
	if value.compressBlockBytes > maxLZ4Block ||
		(value.compressBlockBytes != 0 && value.compressBlockBytes%uint64(os.Getpagesize()) != 0) {
		return options{}, fmt.Errorf("--compress-block-size must be zero or a page-size multiple no larger than %d", maxLZ4Block)
	}
	if value.compressAcceleration < 1 || value.compressAcceleration > maxLZ4Accel {
		return options{}, fmt.Errorf("--compress-acceleration must be in [1, %d]", maxLZ4Accel)
	}
	if value.decompressThreads > 1024 {
		return options{}, errors.New("--decompress-threads must be in [0, 1024]")
	}
	if value.imageIOMode != "writeback" && value.imageIOMode != "direct" {
		return options{}, errors.New("--image-io-mode must be writeback or direct")
	}
	return value, nil
}

type tcpAddressMapping map[netip.Addr]netip.Addr

type tcpPortMapping struct {
	Source      uint32
	Destination uint32
	Configured  bool
}

func parseTCPPortMap(value string) (tcpPortMapping, error) {
	if value == "" {
		return tcpPortMapping{}, nil
	}
	left, right, found := strings.Cut(value, "=")
	if !found || left == "" || right == "" || strings.Contains(right, "=") {
		return tcpPortMapping{}, fmt.Errorf("invalid --tcp-port-map %q", value)
	}
	source, sourceErr := strconv.ParseUint(left, 10, 16)
	destination, destinationErr := strconv.ParseUint(right, 10, 16)
	if sourceErr != nil || destinationErr != nil || source < uint64(portableTCPPortBase) ||
		destination < uint64(portableTCPPortBase) || source == destination {
		return tcpPortMapping{}, fmt.Errorf("invalid --tcp-port-map %q", value)
	}
	return tcpPortMapping{
		Source: uint32(source), Destination: uint32(destination), Configured: true,
	}, nil
}

func parseTCPAddressMap(values []string) (tcpAddressMapping, error) {
	if len(values) > 64 {
		return nil, errors.New("--tcp-address-map exceeds 64 entries")
	}
	mapping := make(tcpAddressMapping, len(values))
	targets := make(map[netip.Addr]bool, len(values))
	for _, value := range values {
		oldText, newText, found := strings.Cut(value, "=")
		if !found || oldText == "" || newText == "" || strings.Contains(newText, "=") {
			return nil, fmt.Errorf("invalid --tcp-address-map %q: expected OLD=NEW", value)
		}
		oldAddress, err := netip.ParseAddr(oldText)
		if err != nil || oldAddress.Zone() != "" {
			return nil, fmt.Errorf("invalid captured address in --tcp-address-map %q", value)
		}
		newAddress, err := netip.ParseAddr(newText)
		if err != nil || newAddress.Zone() != "" {
			return nil, fmt.Errorf("invalid destination address in --tcp-address-map %q", value)
		}
		oldAddress, newAddress = oldAddress.Unmap(), newAddress.Unmap()
		if oldAddress.Is4() != newAddress.Is4() || !portablePlacementAddress(oldAddress) || !portablePlacementAddress(newAddress) {
			return nil, fmt.Errorf("unsafe or cross-family --tcp-address-map %q", value)
		}
		if _, exists := mapping[oldAddress]; exists {
			return nil, fmt.Errorf("duplicate captured address in --tcp-address-map %q", value)
		}
		if targets[newAddress] {
			return nil, fmt.Errorf("duplicate destination address in --tcp-address-map %q", value)
		}
		mapping[oldAddress] = newAddress
		targets[newAddress] = true
	}
	return mapping, nil
}

func portablePlacementAddress(address netip.Addr) bool {
	return address.IsValid() && !address.IsUnspecified() && !address.IsLoopback() &&
		!address.IsMulticast() && !address.IsLinkLocalUnicast()
}

func decodeCRIUAddress(parts []uint32) (netip.Addr, bool) {
	switch len(parts) {
	case 1:
		var raw [4]byte
		binary.LittleEndian.PutUint32(raw[:], parts[0])
		return netip.AddrFrom4(raw), true
	case 4:
		var raw [16]byte
		for index, part := range parts {
			binary.LittleEndian.PutUint32(raw[index*4:], part)
		}
		return netip.AddrFrom16(raw), true
	default:
		return netip.Addr{}, false
	}
}

func encodeCRIUAddress(address netip.Addr, words int) ([]uint32, error) {
	switch words {
	case 1:
		if !address.Is4() {
			return nil, errors.New("cannot encode an IPv6 destination in an IPv4 CRIU socket")
		}
		raw := address.As4()
		return []uint32{binary.LittleEndian.Uint32(raw[:])}, nil
	case 4:
		raw := address.As16()
		parts := make([]uint32, 4)
		for index := range parts {
			parts[index] = binary.LittleEndian.Uint32(raw[index*4:])
		}
		return parts, nil
	default:
		return nil, errors.New("CRIU socket address has an invalid word count")
	}
}

func remapCRIUAddress(parts []uint32, mapping tcpAddressMapping) ([]uint32, bool, error) {
	address, ok := decodeCRIUAddress(parts)
	if !ok {
		return nil, false, errors.New("CRIU socket address has an invalid representation")
	}
	destination, exists := mapping[address.Unmap()]
	if !exists {
		return parts, false, nil
	}
	encoded, err := encodeCRIUAddress(destination, len(parts))
	if err != nil {
		return nil, false, err
	}
	return encoded, true, nil
}

// rebindTCPSocketAddresses rewrites only the fresh container overlay copy of
// files.img. The committed capsule remains immutable, and CRIU plugins remain
// disabled: the accepted OLD=NEW map is parsed and applied by this audited RPC
// helper before CRIU sees the image.
const portableTCPPortBase = uint32(1024)
const portableTCPPortCount = uint32(65536) - portableTCPPortBase

func rotatePortableTCPPort(port uint32, shift uint) uint32 {
	if port < portableTCPPortBase || shift == 0 {
		return port
	}
	return portableTCPPortBase + (port-portableTCPPortBase+uint32(shift))%portableTCPPortCount
}

type tcpBindEndpoint struct {
	ID      uint32
	Family  uint32
	Address netip.Addr
	Port    uint32
	V6Only  bool
	State   uint32
}

func loadCRIUFilesImage(imagesDir string) (*crit.CriuImage, error) {
	path := filepath.Join(imagesDir, "files.img")
	input, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open CRIU files image: %w", err)
	}
	image, decodeErr := crit.New(input, nil, "", false, false).Decode(&fdinfo.FileEntry{})
	closeErr := input.Close()
	if decodeErr != nil {
		return nil, fmt.Errorf("decode CRIU files image: %w", decodeErr)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("close CRIU files image: %w", closeErr)
	}
	return image, nil
}

func rewriteTCPSocketEndpoints(
	image *crit.CriuImage,
	mapping tcpAddressMapping,
	portShift, preservePort uint,
	portMapping tcpPortMapping,
	allowEmptyMap bool,
) (int, int, []tcpBindEndpoint, error) {
	// A TCP listener may be bound to an unspecified address even though its
	// accepted sockets use the captured rank address. Record those local ports
	// before rewriting so the listener and every accepted peer rotate together.
	mappedLocalPorts := make(map[uint32]bool)
	for _, entry := range image.Entries {
		file, ok := entry.Message.(*fdinfo.FileEntry)
		if !ok || file.GetIsk() == nil {
			continue
		}
		socket := file.GetIsk()
		source, ok := decodeCRIUAddress(socket.GetSrcAddr())
		if ok {
			if _, matched := mapping[source.Unmap()]; matched && socket.GetSrcPort() != 0 {
				mappedLocalPorts[socket.GetSrcPort()] = true
			}
		}
	}

	addressChanges := 0
	portChanges := 0
	bindEndpoints := make([]tcpBindEndpoint, 0)
	seenBindEndpoints := make(map[string]bool)
	for _, entry := range image.Entries {
		file, ok := entry.Message.(*fdinfo.FileEntry)
		if !ok || file.GetIsk() == nil {
			continue
		}
		socket := file.GetIsk()
		sourcePortMapped := portMapping.Configured && socket.GetSrcPort() == portMapping.Source
		destinationPortMapped := portMapping.Configured && socket.GetDstPort() == portMapping.Source
		source, sourceChanged, err := remapCRIUAddress(socket.GetSrcAddr(), mapping)
		if err != nil {
			return 0, 0, nil, fmt.Errorf("remap CRIU socket %d source: %w", socket.GetId(), err)
		}
		destination, destinationChanged, err := remapCRIUAddress(socket.GetDstAddr(), mapping)
		if err != nil {
			return 0, 0, nil, fmt.Errorf("remap CRIU socket %d destination: %w", socket.GetId(), err)
		}
		if (sourceChanged || destinationChanged) && socket.GetProto() != uint32(unix.IPPROTO_TCP) {
			return 0, 0, nil, fmt.Errorf("address mapping matched non-TCP CRIU socket %d", socket.GetId())
		}
		if sourceChanged {
			socket.SrcAddr = source
			addressChanges++
		}
		if destinationChanged {
			socket.DstAddr = destination
			addressChanges++
		}
		if sourcePortMapped {
			socket.SrcPort = ptr(portMapping.Destination)
			portChanges++
		}
		if destinationPortMapped {
			socket.DstPort = ptr(portMapping.Destination)
			portChanges++
		}
		rotateSource := sourceChanged
		if socket.GetProto() == uint32(unix.IPPROTO_TCP) && !rotateSource {
			decodedSource, valid := decodeCRIUAddress(socket.GetSrcAddr())
			rotateSource = valid && decodedSource.IsUnspecified() && mappedLocalPorts[socket.GetSrcPort()]
		}
		if socket.GetProto() == uint32(unix.IPPROTO_TCP) && portShift != 0 {
			if rotateSource && !sourcePortMapped && socket.GetSrcPort() >= portableTCPPortBase && socket.GetSrcPort() != uint32(preservePort) {
				socket.SrcPort = ptr(rotatePortableTCPPort(socket.GetSrcPort(), portShift))
				portChanges++
			}
			if destinationChanged && !destinationPortMapped && socket.GetDstPort() >= portableTCPPortBase && socket.GetDstPort() != uint32(preservePort) {
				socket.DstPort = ptr(rotatePortableTCPPort(socket.GetDstPort(), portShift))
				portChanges++
			}
		}
		if socket.GetProto() != uint32(unix.IPPROTO_TCP) || (!rotateSource && !sourcePortMapped) || socket.GetSrcPort() == 0 {
			continue
		}
		localAddress, valid := decodeCRIUAddress(socket.GetSrcAddr())
		if !valid {
			return 0, 0, nil, fmt.Errorf("decode CRIU socket %d rebound source", socket.GetId())
		}
		key := fmt.Sprintf("%d/%s/%d/%t", socket.GetFamily(), localAddress, socket.GetSrcPort(), socket.GetV6Only())
		if seenBindEndpoints[key] {
			continue
		}
		seenBindEndpoints[key] = true
		bindEndpoints = append(bindEndpoints, tcpBindEndpoint{
			ID: socket.GetId(), Family: socket.GetFamily(), Address: localAddress,
			Port: socket.GetSrcPort(), V6Only: socket.GetV6Only(), State: socket.GetState(),
		})
	}
	if addressChanges == 0 && !allowEmptyMap {
		return 0, 0, nil, errors.New("TCP address mapping matched no CRIU socket endpoints")
	}
	return addressChanges, portChanges, bindEndpoints, nil
}

func probeTCPBindEndpoints(endpoints []tcpBindEndpoint) error {
	for _, endpoint := range endpoints {
		family := int(endpoint.Family)
		if family != unix.AF_INET && family != unix.AF_INET6 {
			return fmt.Errorf("TCP socket %d has unsupported address family %d", endpoint.ID, family)
		}
		descriptor, err := unix.Socket(family, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, unix.IPPROTO_TCP)
		if err != nil {
			return fmt.Errorf("probe TCP socket %d: %w", endpoint.ID, err)
		}
		if family == unix.AF_INET6 {
			v6Only := 0
			if endpoint.V6Only {
				v6Only = 1
			}
			if err := unix.SetsockoptInt(descriptor, unix.IPPROTO_IPV6, unix.IPV6_V6ONLY, v6Only); err != nil {
				_ = unix.Close(descriptor)
				return fmt.Errorf("configure TCP socket %d probe: %w", endpoint.ID, err)
			}
		}
		var bindErr error
		switch family {
		case unix.AF_INET:
			if !endpoint.Address.Is4() {
				bindErr = errors.New("IPv4 socket has a non-IPv4 address")
				break
			}
			address := endpoint.Address.As4()
			bindErr = unix.Bind(descriptor, &unix.SockaddrInet4{Port: int(endpoint.Port), Addr: address})
		case unix.AF_INET6:
			if !endpoint.Address.Is6() {
				bindErr = errors.New("IPv6 socket has a non-IPv6 address")
				break
			}
			address := endpoint.Address.As16()
			bindErr = unix.Bind(descriptor, &unix.SockaddrInet6{Port: int(endpoint.Port), Addr: address})
		}
		closeErr := unix.Close(descriptor)
		if bindErr != nil {
			return fmt.Errorf(
				"TCP socket %d target %s:%d is unavailable: %w",
				endpoint.ID, endpoint.Address, endpoint.Port, bindErr,
			)
		}
		if closeErr != nil {
			return fmt.Errorf("close TCP socket %d probe: %w", endpoint.ID, closeErr)
		}
	}
	return nil
}

func rebindTCPSocketEndpoints(
	imagesDir string,
	mapping tcpAddressMapping,
	portShift, preservePort uint,
	portMapping tcpPortMapping,
	allowEmptyMap bool,
) (int, int, error) {
	if len(mapping) == 0 && !portMapping.Configured {
		return 0, 0, nil
	}
	path := filepath.Join(imagesDir, "files.img")
	image, err := loadCRIUFilesImage(imagesDir)
	if err != nil {
		return 0, 0, err
	}
	addressChanges, portChanges, _, err := rewriteTCPSocketEndpoints(
		image, mapping, portShift, preservePort, portMapping, allowEmptyMap,
	)
	if err != nil {
		return 0, 0, err
	}

	temporary, err := os.CreateTemp(imagesDir, ".files.img.rebound-*.tmp")
	if err != nil {
		return 0, 0, fmt.Errorf("create rebound CRIU files image: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := crit.New(nil, temporary, "", false, false).Encode(image); err != nil {
		_ = temporary.Close()
		return 0, 0, fmt.Errorf("encode rebound CRIU files image: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return 0, 0, fmt.Errorf("sync rebound CRIU files image: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return 0, 0, fmt.Errorf("close rebound CRIU files image: %w", err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return 0, 0, fmt.Errorf("activate rebound CRIU files image: %w", err)
	}
	directory, err := os.Open(imagesDir)
	if err != nil {
		return 0, 0, fmt.Errorf("open CRIU image directory: %w", err)
	}
	syncErr := directory.Sync()
	closeErr := directory.Close()
	if syncErr != nil {
		return 0, 0, fmt.Errorf("sync CRIU image directory: %w", syncErr)
	}
	if closeErr != nil {
		return 0, 0, fmt.Errorf("close CRIU image directory: %w", closeErr)
	}
	return addressChanges, portChanges, nil
}

func openDirectory(path string) (*os.File, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	information, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, err
	}
	if !information.IsDir() {
		_ = file.Close()
		return nil, fmt.Errorf("not a directory: %s", path)
	}
	return file, nil
}

func runCUDACommand(executable string, arguments []string, timeout int) (string, time.Duration, error) {
	started := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()
	// #nosec G204 -- executable is an explicit, identity-bound command option.
	output, err := exec.CommandContext(ctx, executable, arguments...).CombinedOutput()
	duration := time.Since(started)
	text := strings.TrimSpace(string(output))
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return text, duration, fmt.Errorf("CUDA command timed out: %s", strings.Join(arguments, " "))
	}
	return text, duration, err
}

func processTreePIDs(root int) ([]int, error) {
	if root <= 0 {
		return nil, fmt.Errorf("invalid process-tree root PID %d", root)
	}
	if _, err := os.Stat(fmt.Sprintf("/proc/%d", root)); err != nil {
		return nil, fmt.Errorf("inspect process-tree root PID %d: %w", root, err)
	}
	seen := map[int]bool{root: true}
	for {
		entries, err := os.ReadDir("/proc")
		if err != nil {
			return nil, fmt.Errorf("read process table: %w", err)
		}
		changed := false
		for _, entry := range entries {
			pid, err := strconv.Atoi(entry.Name())
			if err != nil || seen[pid] {
				continue
			}
			data, err := os.ReadFile(filepath.Join("/proc", entry.Name(), "status"))
			if err != nil {
				continue
			}
			parent := 0
			for _, line := range strings.Split(string(data), "\n") {
				if !strings.HasPrefix(line, "PPid:") {
					continue
				}
				fields := strings.Fields(line)
				if len(fields) == 2 {
					parent, _ = strconv.Atoi(fields[1])
				}
				break
			}
			if seen[parent] {
				seen[pid] = true
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	ordered := make([]int, 0, len(seen))
	for pid := range seen {
		if pid != root {
			ordered = append(ordered, pid)
		}
	}
	sort.Ints(ordered)
	ordered = append([]int{root}, ordered...)
	return ordered, nil
}

func discoverCUDAProcesses(value options, root int) ([]cudaProcess, error) {
	if !value.cudaTree {
		return []cudaProcess{{PID: root}}, nil
	}
	pids, err := processTreePIDs(root)
	if err != nil {
		return nil, err
	}
	processes := make([]cudaProcess, 0, len(pids))
	for _, pid := range pids {
		output, _, err := runCUDACommand(
			value.cudaPath,
			[]string{"--get-restore-tid", "--pid", strconv.Itoa(pid)},
			value.timeout,
		)
		if err != nil {
			var exitError *exec.ExitError
			if errors.As(err, &exitError) {
				continue
			}
			return nil, fmt.Errorf("probe CUDA process PID %d: %w: %s", pid, err, output)
		}
		restoreTID, err := strconv.Atoi(output)
		if err != nil {
			return nil, fmt.Errorf("invalid CUDA restore TID for PID %d: %q", pid, output)
		}
		if restoreTID > 0 {
			processes = append(processes, cudaProcess{PID: pid, RestoreTID: restoreTID})
		}
	}
	if len(processes) == 0 {
		return nil, fmt.Errorf("no CUDA process found in tree rooted at PID %d", root)
	}
	return processes, nil
}

func capturedCUDAProcesses(path string, root int) ([]cudaProcess, error) {
	payload, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read captured CUDA process inventory: %w", err)
	}
	var captured commandResult
	if err := json.Unmarshal(payload, &captured); err != nil {
		return nil, fmt.Errorf("decode captured CUDA process inventory: %w", err)
	}
	if captured.Format != manifestFormat || captured.Kind != resultKind || captured.Action != "dump" {
		return nil, errors.New("captured CUDA process inventory is not a ColdSnap dump result")
	}
	if len(captured.CUDAProcesses) == 0 {
		return nil, errors.New("captured CUDA process inventory is empty")
	}
	tree, err := processTreePIDs(root)
	if err != nil {
		return nil, fmt.Errorf("inspect restored process tree: %w", err)
	}
	inTree := make(map[int]bool, len(tree))
	for _, pid := range tree {
		inTree[pid] = true
	}
	seen := make(map[int]bool, len(captured.CUDAProcesses))
	rootFound := false
	for _, process := range captured.CUDAProcesses {
		if process.PID <= 0 || process.RestoreTID <= 0 || seen[process.PID] {
			return nil, errors.New("captured CUDA process inventory contains an invalid process")
		}
		if !inTree[process.PID] {
			return nil, fmt.Errorf("captured CUDA PID %d is absent from the restored tree", process.PID)
		}
		seen[process.PID] = true
		rootFound = rootFound || process.PID == root
	}
	if !rootFound {
		return nil, fmt.Errorf("captured CUDA process inventory does not contain root PID %d", root)
	}
	return captured.CUDAProcesses, nil
}

func invokeCUDA(
	result *commandResult,
	executable string,
	pid int,
	operation string,
	timeout int,
) (string, error) {
	arguments := []string{"--get-state", "--pid", fmt.Sprint(pid)}
	if operation != "state" {
		arguments = []string{"--action", operation, "--pid", fmt.Sprint(pid)}
		if operation == "lock" {
			arguments = append(arguments, "--timeout", fmt.Sprint(timeout*1000))
		}
	}
	output, duration, err := runCUDACommand(executable, arguments, timeout)
	record := cudaAction{
		Operation: operation,
		PID:       pid,
		Seconds:   duration.Seconds(),
		Output:    output,
	}
	result.CUDAActions = append(result.CUDAActions, record)
	if err != nil {
		return record.Output, fmt.Errorf("CUDA %s for PID %d failed: %w: %s", operation, pid, err, record.Output)
	}
	return record.Output, nil
}

func requireCUDAState(result *commandResult, value options, pid int, expected string) error {
	state, err := invokeCUDA(result, value.cudaPath, pid, "state", value.timeout)
	if err != nil {
		return err
	}
	if state != expected {
		return fmt.Errorf("CUDA state for PID %d is %q, want %q", pid, state, expected)
	}
	return nil
}

type cudaTargetState struct {
	Process      cudaProcess
	Locked       bool
	Checkpointed bool
}

func prepareCUDA(result *commandResult, value options) ([]cudaTargetState, error) {
	if value.cpuOnly {
		return nil, nil
	}
	processes, err := discoverCUDAProcesses(value, value.pid)
	if err != nil {
		return nil, err
	}
	result.CUDAProcesses = processes
	targets := make([]cudaTargetState, len(processes))
	for index, process := range processes {
		targets[index].Process = process
		if err := requireCUDAState(result, value, process.PID, "running"); err != nil {
			return targets, err
		}
	}
	for index := range targets {
		pid := targets[index].Process.PID
		if _, err := invokeCUDA(result, value.cudaPath, pid, "lock", value.timeout); err != nil {
			return targets, err
		}
		targets[index].Locked = true
		if err := requireCUDAState(result, value, pid, "locked"); err != nil {
			return targets, err
		}
	}
	for index := range targets {
		pid := targets[index].Process.PID
		if _, err := invokeCUDA(result, value.cudaPath, pid, "checkpoint", value.timeout); err != nil {
			return targets, err
		}
		targets[index].Checkpointed = true
		if err := requireCUDAState(result, value, pid, "checkpointed"); err != nil {
			return targets, err
		}
	}
	return targets, nil
}

func recoverCUDA(result *commandResult, value options, targets []cudaTargetState) error {
	var errs []error
	for index := len(targets) - 1; index >= 0; index-- {
		if !targets[index].Checkpointed {
			continue
		}
		if _, err := invokeCUDA(result, value.cudaPath, targets[index].Process.PID, "restore", value.timeout); err != nil {
			errs = append(errs, err)
		}
	}
	for index := len(targets) - 1; index >= 0; index-- {
		if !targets[index].Locked {
			continue
		}
		if _, err := invokeCUDA(result, value.cudaPath, targets[index].Process.PID, "unlock", value.timeout); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func restoreCUDA(result *commandResult, value options, pid int) error {
	if value.cpuOnly {
		return nil
	}
	processes, err := discoverCUDAProcesses(value, pid)
	if err != nil {
		return err
	}
	result.CUDAProcesses = processes
	for _, process := range processes {
		if err := requireCUDAState(result, value, process.PID, "checkpointed"); err != nil {
			return err
		}
	}
	for _, process := range processes {
		if _, err := invokeCUDA(result, value.cudaPath, process.PID, "restore", value.timeout); err != nil {
			return err
		}
	}
	for _, process := range processes {
		if err := requireCUDAState(result, value, process.PID, "locked"); err != nil {
			return err
		}
	}
	for _, process := range processes {
		if _, err := invokeCUDA(result, value.cudaPath, process.PID, "unlock", value.timeout); err != nil {
			return err
		}
	}
	for _, process := range processes {
		if err := requireCUDAState(result, value, process.PID, "running"); err != nil {
			return err
		}
	}
	return nil
}

// restoreCapturedCUDA consumes the dump-time CUDA owner inventory instead of
// probing every process after SIGCONT. SGLang has several CUDA-owning process
// roles; rediscovering them while the whole tree runs gives a later owner time
// to advance out of checkpoint state. Restore and unlock children first, with
// each transition kept adjacent, while the distributed process groups remain
// runnable together for CPU-side collectives.
func restoreCapturedCUDA(
	result *commandResult,
	value options,
	processes []cudaProcess,
) error {
	result.CUDAProcesses = processes
	for index := len(processes) - 1; index >= 0; index-- {
		pid := processes[index].PID
		if err := requireCUDAState(result, value, pid, "checkpointed"); err != nil {
			return err
		}
		if _, err := invokeCUDA(result, value.cudaPath, pid, "restore", value.timeout); err != nil {
			return err
		}
		if err := requireCUDAState(result, value, pid, "locked"); err != nil {
			return err
		}
		if _, err := invokeCUDA(result, value.cudaPath, pid, "unlock", value.timeout); err != nil {
			return err
		}
		if err := requireCUDAState(result, value, pid, "running"); err != nil {
			return err
		}
	}
	return nil
}

func killRestoredProcess(pid int) error {
	groupErr := unix.Kill(-pid, unix.SIGKILL)
	if groupErr == nil {
		return nil
	}
	pidErr := unix.Kill(pid, unix.SIGKILL)
	if pidErr == nil || errors.Is(pidErr, unix.ESRCH) {
		return nil
	}
	return fmt.Errorf(
		"kill restored PID %d after controller failure: group=%v pid=%w",
		pid, groupErr, pidErr,
	)
}

// restoreDeferredCUDA runs only after every distributed rank has completed its
// CRIU restore. CRIU leaves the process group stopped so no restored TCP peer
// can transmit before its counterpart exists. CUDA checkpoint state, however,
// can only be queried while the target is runnable, so this bounded phase
// resumes the group, restores CUDA, and stops it again before returning to the
// distributed controller for the final all-rank release.
func restoreDeferredCUDA(result *commandResult, value options) error {
	var processes []cudaProcess
	var err error
	if value.cudaProcessesResult != "" {
		processes, err = capturedCUDAProcesses(value.cudaProcessesResult, value.pid)
		if err != nil {
			return err
		}
	}
	restoreErr := error(nil)
	if len(processes) != 0 {
		if err := unix.Kill(-value.pid, unix.SIGCONT); err != nil {
			return fmt.Errorf("resume restored process group %d for CUDA restore: %w", value.pid, err)
		}
		result.TreeResumedForCUDA = true
		restoreErr = restoreCapturedCUDA(result, value, processes)
	} else {
		if err := unix.Kill(-value.pid, unix.SIGCONT); err != nil {
			return fmt.Errorf("resume restored process group %d for CUDA restore: %w", value.pid, err)
		}
		result.TreeResumedForCUDA = true
		restoreErr = restoreCUDA(result, value, value.pid)
	}
	stopErr := unix.Kill(-value.pid, unix.SIGSTOP)
	if stopErr == nil {
		result.TreeRestopped = true
	} else {
		stopErr = fmt.Errorf("stop restored process group %d after CUDA restore: %w", value.pid, stopErr)
	}
	return errors.Join(restoreErr, stopErr)
}

// Run validates a coldsnap-criu-rpc invocation and executes its dump or restore.
func Run(arguments []string) (returnErr error) {
	value, err := parseOptions(arguments)
	if err != nil {
		return err
	}
	result := commandResult{
		Format: manifestFormat,
		Kind:   resultKind,
		Action: value.action,
		CRIUOptions: criuOptions{
			DecompressThreads:    value.decompressThreads,
			ImageIOMode:          value.imageIOMode,
			LeaveStopped:         value.leaveStopped,
			LeaveRunning:         value.leaveRunning,
			NVIDIAPlaceholderFDs: value.nvidiaPlaceholderFDs,
			DeferNetworkUnlock:   value.deferNetworkUnlock,
		},
	}
	if value.action == "dump" && value.compressBlockBytes != 0 {
		result.CRIUOptions.CompressionBlockBytes = value.compressBlockBytes
		result.CRIUOptions.CompressionAcceleration = value.compressAcceleration
	}
	defer func() {
		if returnErr != nil {
			result.Error = returnErr.Error()
		}
		if err := writeJSONAtomically(value.resultPath, result); err != nil {
			returnErr = errors.Join(returnErr, fmt.Errorf("write result: %w", err))
		}
	}()
	if value.action == "cuda-restore" {
		return restoreDeferredCUDA(&result, value)
	}
	images, err := openDirectory(value.imagesDir)
	if err != nil {
		return fmt.Errorf("open images directory: %w", err)
	}
	defer func() { _ = images.Close() }()
	work, err := openDirectory(value.workDir)
	if err != nil {
		return fmt.Errorf("open work directory: %w", err)
	}
	defer func() { _ = work.Close() }()
	addressMapping, err := parseTCPAddressMap(value.tcpAddressMap)
	if err != nil {
		return err
	}
	if value.action == "restore" {
		portMapping, parseErr := parseTCPPortMap(value.tcpPortMap)
		if parseErr != nil {
			return parseErr
		}
		result.TCPAddressRemaps, result.TCPPortRemaps, err = rebindTCPSocketEndpoints(
			value.imagesDir, addressMapping, value.tcpPortShift, value.tcpPreservePort,
			portMapping, value.tcpAllowEmptyMap,
		)
		if err != nil {
			return err
		}
	}
	pluginDirectory := value.pluginDir
	if pluginDirectory == "" {
		emptyPluginDirectory, err := os.MkdirTemp("", "coldsnap-criu-no-plugins-")
		if err != nil {
			return fmt.Errorf("create empty plugin directory: %w", err)
		}
		defer func() { _ = os.Remove(emptyPluginDirectory) }()
		pluginDirectory = emptyPluginDirectory
	} else {
		info, err := os.Stat(pluginDirectory)
		if err != nil {
			return fmt.Errorf("inspect CRIU plugin directory: %w", err)
		}
		if !info.IsDir() {
			return fmt.Errorf("CRIU plugin path is not a directory: %s", pluginDirectory)
		}
	}

	notifier := &externalNotifier{
		action:          value.action,
		manifestPath:    value.manifestPath,
		dumpFiles:       make(map[uint32]externalFile),
		restoreFiles:    make(map[uint32]externalFile),
		restoredFileIDs: make(map[uint32]bool),
		fdBrokerSocket:  value.fdBrokerSocket,
		placeholderFDs:  value.nvidiaPlaceholderFDs,
		linkRemapSource: value.linkRemapSourceDir,
		linkRemapTarget: value.linkRemapSnapshotDir,
	}
	if value.deferNetworkUnlock {
		notifier.networkUnlock = func() error {
			seconds, err := deferNetworkUnlock(value.workDir, time.Duration(value.timeout)*time.Second)
			result.NetworkUnlockWaitSeconds = seconds
			return err
		}
	}
	if value.fdBrokerSocket != "" {
		previous, existed := os.LookupEnv(nvidiaFDBrokerEnvironment)
		if err := os.Setenv(nvidiaFDBrokerEnvironment, value.fdBrokerSocket); err != nil {
			return fmt.Errorf("configure NVIDIA FD broker: %w", err)
		}
		defer func() {
			if existed {
				_ = os.Setenv(nvidiaFDBrokerEnvironment, previous)
			} else {
				_ = os.Unsetenv(nvidiaFDBrokerEnvironment)
			}
		}()
	}
	if value.action == "dump" {
		if _, err := os.Stat(value.manifestPath); err == nil {
			return fmt.Errorf("refusing to overwrite external-file manifest: %s", value.manifestPath)
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	} else {
		manifest, err := loadManifest(value.manifestPath)
		if err != nil {
			return fmt.Errorf("load external-file manifest: %w", err)
		}
		for _, record := range manifest.Files {
			notifier.restoreFiles[record.ID] = record
		}
		if value.pluginDir != "" {
			mapPath, err := writeNVIDIAExternalMap(value.workDir, notifier.restoreFiles)
			if err != nil {
				return fmt.Errorf("write NVIDIA external-file map: %w", err)
			}
			previous, existed := os.LookupEnv(nvidiaExternalMapEnvironment)
			if err := os.Setenv(nvidiaExternalMapEnvironment, mapPath); err != nil {
				return fmt.Errorf("configure NVIDIA external-file map: %w", err)
			}
			defer func() {
				if existed {
					_ = os.Setenv(nvidiaExternalMapEnvironment, previous)
				} else {
					_ = os.Unsetenv(nvidiaExternalMapEnvironment)
				}
			}()
			if value.nvidiaPlaceholderFDs {
				previousPlaceholder, placeholderExisted := os.LookupEnv(nvidiaPlaceholderEnvironment)
				if err := os.Setenv(nvidiaPlaceholderEnvironment, "1"); err != nil {
					return fmt.Errorf("configure NVIDIA placeholder restore: %w", err)
				}
				defer func() {
					if placeholderExisted {
						_ = os.Setenv(nvidiaPlaceholderEnvironment, previousPlaceholder)
					} else {
						_ = os.Unsetenv(nvidiaPlaceholderEnvironment)
					}
				}()
			}
		}
	}

	client := criu.MakeCriu()
	client.SetCriuPath(value.criuPath)
	client.SetCriuArgs("--libdir", pluginDirectory)
	imagesFD := int32(images.Fd())
	workFD := int32(work.Fd())
	manageCgroups := rpc.CriuCgMode_IGNORE
	timeout := uint32(value.timeout)
	opts := &rpc.CriuOpts{
		ImagesDirFd:       &imagesFD,
		WorkDirFd:         &workFD,
		ShellJob:          ptr(true),
		FileLocks:         ptr(true),
		LinkRemap:         ptr(true),
		ManageCgroupsMode: &manageCgroups,
		LogLevel:          ptr[int32](4),
		LogFile:           &value.logFile,
		Timeout:           &timeout,
	}
	decompressThreads := uint32(value.decompressThreads)
	opts.DecompressThreads = &decompressThreads
	imageIOModes := map[string]rpc.CriuImageIoMode{
		"writeback": rpc.CriuImageIoMode_IMAGE_IO_WRITEBACK,
		"direct":    rpc.CriuImageIoMode_IMAGE_IO_DIRECT,
	}
	imageIOMode := imageIOModes[value.imageIOMode]
	opts.ImageIoMode = &imageIOMode
	if value.action == "dump" && value.compressBlockBytes != 0 {
		compression := uint32(compressBlock)
		blockBytes := uint32(value.compressBlockBytes)
		acceleration := uint32(value.compressAcceleration)
		opts.Compress = &compression
		opts.CompressBlockSize = &blockBytes
		opts.CompressAcceleration = &acceleration
	}
	if value.tcpEstablished {
		opts.TcpEstablished = ptr(true)
	}
	if value.leaveStopped {
		opts.LeaveStopped = ptr(true)
	}
	if value.leaveRunning {
		opts.LeaveRunning = ptr(true)
	}
	if value.ghostLimit != 0 {
		ghostLimit := uint32(value.ghostLimit)
		opts.GhostLimit = &ghostLimit
	}
	if value.networkLock != "" {
		networkLocks := map[string]rpc.CriuNetworkLockMethod{
			"iptables": rpc.CriuNetworkLockMethod_IPTABLES,
			"nftables": rpc.CriuNetworkLockMethod_NFTABLES,
			"skip":     rpc.CriuNetworkLockMethod_SKIP,
		}
		networkLock := networkLocks[value.networkLock]
		opts.NetworkLock = &networkLock
	}

	if value.action == "dump" {
		targets, err := prepareCUDA(&result, value)
		if err != nil {
			return errors.Join(err, recoverCUDA(&result, value, targets))
		}
		pid := int32(value.pid)
		opts.Pid = &pid
		if err := client.Dump(opts, notifier); err != nil {
			return errors.Join(err, recoverCUDA(&result, value, targets))
		}
		result.ExternalFileCount = len(notifier.dumpFiles)
	} else {
		pid, err := client.RestoreWithPid(opts, notifier)
		if err != nil {
			return err
		}
		result.RestoredPID = pid
		if notifier.postRestorePID != 0 && int(notifier.postRestorePID) != pid {
			return errors.Join(
				fmt.Errorf("restore PID mismatch: response=%d notification=%d", pid, notifier.postRestorePID),
				killRestoredProcess(pid),
			)
		}
		if !value.leaveStopped {
			if err := restoreCUDA(&result, value, pid); err != nil {
				return errors.Join(err, killRestoredProcess(pid))
			}
		}
		if value.pluginDir != "" {
			result.ExternalFileCount = len(notifier.restoreFiles)
		} else {
			result.ExternalFileCount = len(notifier.restoredFileIDs)
		}
	}
	return nil
}

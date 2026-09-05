// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/operationtiming"
	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
	activationruntime "github.com/sparksq/coldsnap/runtime/engine"
)

const activationRuntimeDirectory = "activation-runtime"
const criuRPCEnvironment = "COLDSNAP_CRIU_RPC"

type activationRuntimeBinding struct {
	HostPath      string
	ContainerPath string
}

func activationRuntimeRoot(stateRoot string, pack activationruntime.Pack) string {
	return filepath.Join(
		stateRoot,
		activationRuntimeDirectory,
		strings.TrimPrefix(pack.SHA256, "sha256:"),
	)
}

func activationRuntimeBindings(stateRoot string, pack activationruntime.Pack) []activationRuntimeBinding {
	root := activationRuntimeRoot(stateRoot, pack)
	bindings := make([]activationRuntimeBinding, 0, len(pack.Mounts))
	for _, mount := range pack.Mounts {
		bindings = append(bindings, activationRuntimeBinding{
			HostPath: filepath.Join(root, mount.File), ContainerPath: mount.Target,
		})
	}
	return bindings
}

func isActivationRuntimeTarget(path string) bool {
	targets := []string{activationruntime.CRIURPCTarget}
	for _, mount := range activationruntime.Current().Mounts {
		targets = append(targets, mount.Target)
	}
	for _, target := range targets {
		if path == target || strings.HasPrefix(path, target+"/") ||
			strings.HasPrefix(target, strings.TrimSuffix(path, "/")+"/") {
			return true
		}
	}
	return false
}

func activationRuntimePack() (activationruntime.Pack, error) {
	executable, err := os.Executable()
	if err != nil {
		return activationruntime.Pack{}, fmt.Errorf("resolve release-matched payload verifier: %w", err)
	}
	verifierInformation, err := os.Stat(executable)
	if err != nil {
		return activationruntime.Pack{}, fmt.Errorf("inspect release-matched payload verifier: %w", err)
	}
	if !verifierInformation.Mode().IsRegular() || verifierInformation.Mode().Perm()&0o111 == 0 {
		return activationruntime.Pack{}, errors.New("release-matched payload verifier is not an executable regular file")
	}
	verifier, err := os.ReadFile(executable)
	if err != nil {
		return activationruntime.Pack{}, fmt.Errorf("read release-matched payload verifier: %w", err)
	}
	if len(verifier) == 0 {
		return activationruntime.Pack{}, errors.New("release-matched payload verifier is empty")
	}
	path := strings.TrimSpace(os.Getenv(criuRPCEnvironment))
	if path == "" {
		executable, err := os.Executable()
		if err == nil {
			candidate := filepath.Join(filepath.Dir(executable), "coldsnap-criu-rpc")
			if information, statErr := os.Stat(candidate); statErr == nil && information.Mode().IsRegular() {
				path = candidate
			}
		}
	}
	if path == "" {
		return activationruntime.WithPayloadVerifier(verifier), nil
	}
	information, err := os.Stat(path)
	if err != nil {
		return activationruntime.Pack{}, fmt.Errorf("inspect release-matched CRIU RPC helper: %w", err)
	}
	if !information.Mode().IsRegular() || information.Mode().Perm()&0o111 == 0 {
		return activationruntime.Pack{}, errors.New("release-matched CRIU RPC helper is not an executable regular file")
	}
	payload, err := os.ReadFile(path)
	if err != nil {
		return activationruntime.Pack{}, fmt.Errorf("read release-matched CRIU RPC helper: %w", err)
	}
	if len(payload) == 0 {
		return activationruntime.Pack{}, errors.New("release-matched CRIU RPC helper is empty")
	}
	return activationruntime.WithPayloadVerifierAndCRIURPC(verifier, payload), nil
}

func payloadVerifierActivationPath(stateRoot string, pack activationruntime.Pack) (string, error) {
	for _, item := range pack.Files {
		if item.Name == activationruntime.PayloadVerifierFile {
			return filepath.Join(activationRuntimeRoot(stateRoot, pack), item.Name), nil
		}
	}
	return "", errors.New("activation runtime has no payload verifier")
}

func criuRPCActivationBinding(stateRoot string, pack activationruntime.Pack) (activationRuntimeBinding, bool) {
	for _, binding := range activationRuntimeBindings(stateRoot, pack) {
		if binding.ContainerPath == activationruntime.CRIURPCTarget {
			return binding, true
		}
	}
	return activationRuntimeBinding{}, false
}

// prepareActivationRuntime stages the release-matched, patchable controller
// layer through the manager provider. Capsules remain immutable and bind the
// snapshot-driver ABI expected by this activation layer.
func (adapter Adapter) prepareActivationRuntime(
	ctx context.Context, request snapshot.Request,
) (string, error) {
	contract, err := snapshotdriver.Resolve(request.Driver)
	if err != nil {
		return "", err
	}
	pack, err := activationRuntimePack()
	if err != nil {
		return "", err
	}
	if pack.Format != activationruntime.Format || pack.ABI != contract.ABI {
		return "", fmt.Errorf(
			"activation runtime ABI %d does not support snapshot driver %s ABI %d",
			pack.ABI, contract.ID, contract.ABI,
		)
	}
	hosts := make([]string, 0, len(request.Launch.Units))
	seen := make(map[string]bool, len(request.Launch.Units))
	for _, unit := range request.Launch.Units {
		if !seen[unit.Host] {
			seen[unit.Host] = true
			hosts = append(hosts, unit.Host)
		}
	}
	slices.Sort(hosts)
	root := activationRuntimeRoot(adapter.StateRoot, pack)
	err = parallelStrings(hosts, func(host string) error {
		return operationtiming.Measure(ctx, "activation_runtime.stage", map[string]string{
			"host": host, "sha256": pack.SHA256,
		}, func(hostContext context.Context) error {
			return adapter.stageActivationRuntimeHost(hostContext, host, request.ID, root, pack)
		})
	})
	if err != nil {
		return "", err
	}
	return pack.SHA256, nil
}

func (adapter Adapter) stageActivationRuntimeHost(
	ctx context.Context,
	host string,
	operationID string,
	root string,
	pack activationruntime.Pack,
) error {
	if _, err := adapter.Remote.Run(ctx, host, "install", "-d", "-m", "0700", root); err != nil {
		return fmt.Errorf("create activation runtime cache on %s: %w", host, err)
	}
	if _, err := adapter.Remote.Run(ctx, host, "test", "!", "-L", root); err != nil {
		return fmt.Errorf("activation runtime cache on %s is a symbolic link", host)
	}
	paths := make([]string, 0, len(pack.Files))
	for _, item := range pack.Files {
		paths = append(paths, filepath.Join(root, item.Name))
	}
	if output, hashErr := adapter.Remote.Run(
		ctx, host, append([]string{"sha256sum"}, paths...)...,
	); hashErr == nil && activationRuntimeFilesMatch(output, pack) {
		return nil
	}
	for _, item := range pack.Files {
		finalPath := filepath.Join(root, item.Name)
		if _, regularErr := adapter.Remote.Run(ctx, host, "test", "-f", finalPath); regularErr == nil {
			if _, symlinkErr := adapter.Remote.Run(ctx, host, "test", "!", "-L", finalPath); symlinkErr == nil {
				if output, hashErr := adapter.Remote.Run(ctx, host, "sha256sum", finalPath); hashErr == nil &&
					sha256OutputMatches(output, item.SHA256) {
					continue
				}
			}
		}
		temporary := finalPath + ".tmp." + operationID
		if _, err := adapter.Remote.RunInput(ctx, host, item.Data, "tee", temporary); err != nil {
			return fmt.Errorf("stage activation runtime file %s on %s: %w", item.Name, host, err)
		}
		if _, err := adapter.Remote.Run(ctx, host, "chmod", "0555", temporary); err != nil {
			return fmt.Errorf("set activation runtime mode for %s on %s: %w", item.Name, host, err)
		}
		output, err := adapter.Remote.Run(ctx, host, "sha256sum", temporary)
		if err != nil || !sha256OutputMatches(output, item.SHA256) {
			return fmt.Errorf("activation runtime file %s failed digest verification on %s", item.Name, host)
		}
		if _, err := adapter.Remote.Run(ctx, host, "mv", "-f", temporary, finalPath); err != nil {
			return fmt.Errorf("commit activation runtime file %s on %s: %w", item.Name, host, err)
		}
	}
	return nil
}

func activationRuntimeFilesMatch(output []byte, pack activationruntime.Pack) bool {
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) != len(pack.Files) {
		return false
	}
	for index, line := range lines {
		fields := strings.Fields(line)
		if len(fields) < 1 || fields[0] != strings.TrimPrefix(pack.Files[index].SHA256, "sha256:") {
			return false
		}
	}
	return true
}

func sha256OutputMatches(output []byte, digest string) bool {
	fields := strings.Fields(string(output))
	return len(fields) >= 1 && fields[0] == strings.TrimPrefix(digest, "sha256:")
}

// verifyCRIUCapabilities asks the exact CRIU build pinned by each capsule to
// probe its destination kernel. This is a stronger admission signal than an
// exact uname string: equal releases can differ in configuration, while CRIU
// deliberately supports compatible migrations between kernel releases.
func (adapter Adapter) verifyCRIUCapabilities(
	ctx context.Context,
	request snapshot.Request,
	images map[string]snapshot.CapsuleImage,
) error {
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		image, ok := images[unit.ID]
		if !ok {
			return fmt.Errorf("unit %s capsule image is unavailable for CRIU capability probing", unit.ID)
		}
		return operationtiming.Measure(ctx, "criu.check", map[string]string{
			"unit": unit.ID, "host": unit.Host,
		}, func(unitContext context.Context) error {
			output, err := adapter.runWorkload(unitContext, unit.Host, &hostops.WorkloadSpec{
				Image: image.Reference, RemoveAfterExit: true, PullPolicy: "never",
				Privileged: true, SeccompUnconfined: true, Network: "host", Combined: true,
				Entrypoint: "/opt/coldsnap/criu/bin/criu", Command: []string{"check"},
			}, false)
			if err != nil {
				message := strings.TrimSpace(string(output))
				if message != "" {
					return fmt.Errorf("destination kernel failed capsule-pinned CRIU check: %w: %s", err, message)
				}
				return fmt.Errorf("destination kernel failed capsule-pinned CRIU check: %w", err)
			}
			return nil
		})
	})
}

func captureImages(request snapshot.Request) (map[string]snapshot.CapsuleImage, error) {
	contract, err := snapshotdriver.Resolve(request.Driver)
	if err != nil {
		return nil, err
	}
	images := make(map[string]snapshot.CapsuleImage, len(request.Launch.Units))
	for _, unit := range request.Launch.Units {
		images[unit.ID] = snapshot.CapsuleImage{
			Unit: unit.ID, Reference: unit.Image, Digest: unit.ImageDigest,
			Driver: contract.Binding(),
		}
	}
	return images, nil
}

func (adapter Adapter) verifyCaptureFeatureAdmission(
	ctx context.Context, request snapshot.Request,
) error {
	images, err := captureImages(request)
	if err != nil {
		return err
	}
	if err := adapter.verifyCRIUCapabilities(ctx, request, images); err != nil {
		return err
	}
	contract, err := snapshotdriver.Resolve(request.Driver)
	if err != nil {
		return err
	}
	return adapter.verifyHostFeatureProfiles(ctx, request, images, contract.BaseRequirements)
}

// verifyHostFeatureProfiles executes bounded CUDA probes in disposable
// capsule containers and matches their typed results against artifact
// requirements. Capsule-pinned CRIU is checked immediately before this call;
// that independent evidence is attached here rather than inferred from a
// driver name. Successful receipts are cached by boot, driver, device,
// capsule, and release-matched probe identity.
func (adapter Adapter) verifyHostFeatureProfiles(
	ctx context.Context,
	request snapshot.Request,
	images map[string]snapshot.CapsuleImage,
	requirements snapshotdriver.Requirements,
) error {
	if err := requirements.Validate(); err != nil {
		return err
	}
	pack, err := activationRuntimePack()
	if err != nil {
		return err
	}
	var probe activationRuntimeBinding
	for _, binding := range activationRuntimeBindings(adapter.StateRoot, pack) {
		if binding.ContainerPath == activationruntime.FeatureProbeTarget {
			probe = binding
			break
		}
	}
	if probe.HostPath == "" {
		return errors.New("activation runtime has no disposable feature probe")
	}
	return parallelUnits(request.Launch.Units, func(unit snapshot.LaunchUnit) error {
		image, ok := images[unit.ID]
		if !ok {
			return fmt.Errorf("unit %s capsule image is unavailable for feature probing", unit.ID)
		}
		for slot, device := range unit.Devices {
			if err := operationtiming.Measure(ctx, "features.probe", map[string]string{
				"unit": unit.ID, "host": unit.Host, "device": device,
			}, func(probeContext context.Context) error {
				profile, cached, err := adapter.hostFeatureProfile(
					probeContext, request, unit, slot, device, image, probe, pack.SHA256, requirements,
				)
				if err != nil {
					return err
				}
				if err := requirements.Admit(profile); err != nil {
					return fmt.Errorf("unit %s device slot %d feature admission: %w", unit.ID, slot, err)
				}
				output := adapter.Output
				if output == nil {
					output = os.Stdout
				}
				mode := "probed"
				if cached {
					mode = "cached"
				}
				fmt.Fprintf(output, "ColdSnap: unit %s device slot %d feature profile %s and admitted\n", unit.ID, slot, mode)
				return nil
			}); err != nil {
				return err
			}
		}
		return nil
	})
}

func (adapter Adapter) hostFeatureProfile(
	ctx context.Context,
	request snapshot.Request,
	unit snapshot.LaunchUnit,
	slot int,
	device string,
	image snapshot.CapsuleImage,
	probe activationRuntimeBinding,
	probeDigest string,
	requirements snapshotdriver.Requirements,
) (snapshotdriver.FeatureProfile, bool, error) {
	boot, err := adapter.Remote.Run(ctx, unit.Host, "cat", "/proc/sys/kernel/random/boot_id")
	if err != nil {
		return snapshotdriver.FeatureProfile{}, false, fmt.Errorf("unit %s boot identity probe: %w", unit.ID, err)
	}
	gpu, err := adapter.Remote.Run(
		ctx, unit.Host, "nvidia-smi", "--id="+device,
		"--query-gpu=driver_version,uuid,name,compute_cap", "--format=csv,noheader,nounits",
	)
	if err != nil {
		return snapshotdriver.FeatureProfile{}, false, fmt.Errorf("unit %s device slot %d feature identity probe: %w", unit.ID, slot, err)
	}
	keyInput := strings.Join([]string{
		strings.TrimSpace(string(boot)), strings.TrimSpace(string(gpu)), image.Digest,
		probeDigest, request.Driver.ID, device,
	}, "\x00")
	digest := sha256.Sum256([]byte(keyInput))
	key := hex.EncodeToString(digest[:])
	root := filepath.Join(adapter.StateRoot, "host-feature-profiles")
	path := filepath.Join(root, key+".json")
	if payload, readErr := adapter.Remote.Run(ctx, unit.Host, "cat", path); readErr == nil {
		profile, decodeErr := decodeHostFeatureProfile(payload, request.Driver.ID)
		if decodeErr == nil {
			return profile, true, nil
		}
	}
	// The probe contract is JSON on stdout. Keep container-runtime and NVIDIA
	// hook diagnostics on stderr; combining the streams makes otherwise
	// successful first-run probes undecodable on some hosts.
	output, err := adapter.runWorkload(ctx, unit.Host, &hostops.WorkloadSpec{
		Image: image.Reference, RemoveAfterExit: true, PullPolicy: "never", GPUs: []string{device},
		Privileged: true, SeccompUnconfined: true, Network: "none", Entrypoint: "python3",
		Mounts:  []hostops.Mount{{Source: probe.HostPath, Target: probe.ContainerPath, ReadOnly: true}},
		Command: []string{probe.ContainerPath, "profile", "--memory-bytes", "1048576", "--timeout", "30"},
	}, false)
	if err != nil {
		message := strings.TrimSpace(string(output))
		return snapshotdriver.FeatureProfile{}, false, fmt.Errorf(
			"unit %s device slot %d disposable feature probe failed: %w: %s",
			unit.ID, slot, err, message,
		)
	}
	profile, err := decodeHostFeatureProfile(output, request.Driver.ID)
	if err != nil {
		return snapshotdriver.FeatureProfile{}, false, fmt.Errorf("unit %s device slot %d: %w", unit.ID, slot, err)
	}
	// Cache only receipts that admit the exact requirements being checked. A
	// structurally valid profile can still contain failed, unsupported, or
	// unqualified results and must not become a positive admission shortcut.
	if err := requirements.Admit(profile); err != nil {
		return snapshotdriver.FeatureProfile{}, false, fmt.Errorf(
			"unit %s device slot %d feature admission: %w", unit.ID, slot, err,
		)
	}
	payload, err := json.Marshal(profile)
	if err != nil {
		return snapshotdriver.FeatureProfile{}, false, err
	}
	payload = append(payload, '\n')
	if _, err := adapter.Remote.Run(ctx, unit.Host, "install", "-d", "-m", "0700", root); err != nil {
		return snapshotdriver.FeatureProfile{}, false, fmt.Errorf("create host feature-profile cache: %w", err)
	}
	temporary := path + ".tmp." + request.ID
	if _, err := adapter.Remote.RunInput(ctx, unit.Host, payload, "tee", temporary); err != nil {
		return snapshotdriver.FeatureProfile{}, false, fmt.Errorf("stage host feature profile: %w", err)
	}
	if _, err := adapter.Remote.Run(ctx, unit.Host, "chmod", "0600", temporary); err != nil {
		return snapshotdriver.FeatureProfile{}, false, fmt.Errorf("protect host feature profile: %w", err)
	}
	if _, err := adapter.Remote.Run(ctx, unit.Host, "mv", "-f", temporary, path); err != nil {
		return snapshotdriver.FeatureProfile{}, false, fmt.Errorf("commit host feature profile: %w", err)
	}
	return profile, false, nil
}

func decodeHostFeatureProfile(payload []byte, driverID string) (snapshotdriver.FeatureProfile, error) {
	var profile snapshotdriver.FeatureProfile
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&profile); err != nil {
		return snapshotdriver.FeatureProfile{}, fmt.Errorf("decode disposable host feature profile: %w", err)
	}
	structural := snapshotdriver.FeatureCRIUProcessTree
	if driverID == snapshotdriver.N580 {
		structural = snapshotdriver.FeatureCRIUProcessTemplate
	}
	for index := range profile.Features {
		if profile.Features[index].ID == structural {
			profile.Features[index].Status = snapshotdriver.FeaturePassed
			profile.Features[index].Reason = ""
			profile.Features[index].FailurePhase = ""
			profile.Features[index].Probe = "capsule-criu-check-v1"
		}
	}
	if err := profile.Validate(); err != nil {
		return snapshotdriver.FeatureProfile{}, err
	}
	return profile, nil
}

func parallelStrings(values []string, operation func(string) error) error {
	var units []snapshot.LaunchUnit
	for index, value := range values {
		units = append(units, snapshot.LaunchUnit{ID: value, Index: index})
	}
	return parallelUnits(units, func(unit snapshot.LaunchUnit) error {
		return operation(unit.ID)
	})
}

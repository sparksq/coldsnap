// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sparksq/coldsnap/internal/hostprovider"
)

// This opt-in check uses an operation-scoped manager provider, real target
// binaries and a cached non-GPU container. It never captures or replaces a
// model workload. Run the test executable on the actual controller platform.
func TestActivationRuntimeCrossArchitectureIntegration(t *testing.T) {
	host := os.Getenv("COLDSNAP_TEST_ACTIVATION_HOST")
	if host == "" {
		t.Skip("set COLDSNAP_TEST_ACTIVATION_HOST and target/controller bundles with a manager provider")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	remote, err := hostprovider.New(ctx, os.Getenv("COLDSNAP_HOST_PROVIDER_SOCKET"), os.Getenv("COLDSNAP_HOST_PROVIDER_TOKEN"), os.Getenv("COLDSNAP_HOST_PROVIDER_SESSION"))
	if err != nil {
		t.Fatal(err)
	}
	output, err := remote.Run(ctx, host, "mktemp", "-d", "/tmp/coldsnap-target-check.XXXXXXXXXX")
	if err != nil {
		t.Fatal(err)
	}
	root := strings.TrimSpace(string(output))
	if filepath.Dir(root) != "/tmp" || !strings.HasPrefix(filepath.Base(root), "coldsnap-target-check.") || strings.ContainsAny(root, "\n\r ") {
		t.Fatalf("invalid test directory %q", root)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if _, err := remote.Run(cleanup, host, "rm", "-r", "--", root); err != nil {
			t.Errorf("remove test directory %s: %v", root, err)
		}
	})
	request := validRequest(1)
	request.ID = os.Getenv("COLDSNAP_HOST_PROVIDER_SESSION")
	request.Launch.Units[0].Host = host
	adapter := Adapter{Remote: remote, StateRoot: root}
	t.Setenv(targetVerifierEnvironment, "")
	t.Setenv(targetCRIURPCEnvironment, "")
	t.Setenv(criuRPCEnvironment, filepath.Join(os.Getenv("COLDSNAP_TEST_CONTROLLER_BUNDLE"), "coldsnap-criu-rpc"))
	if _, err := adapter.prepareActivationRuntime(ctx, request); err == nil || !strings.Contains(err.Error(), "supply release-matched target tools") {
		t.Fatalf("controller-native helper was not rejected on foreign target: %v", err)
	}
	data := []byte("ColdSnap cross-architecture payload verification\n")
	payload := filepath.Join(root, "model-weights.pack")
	if _, err := remote.RunInput(ctx, host, data, "tee", payload); err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(data)
	for _, engine := range []string{"vllm", "sglang"} {
		t.Setenv(targetVerifierEnvironment, filepath.Join(os.Getenv("COLDSNAP_TEST_TARGET_BUNDLE"), "coldsnap-"+engine+"-adapter"))
		t.Setenv(targetCRIURPCEnvironment, filepath.Join(os.Getenv("COLDSNAP_TEST_TARGET_BUNDLE"), "coldsnap-criu-rpc"))
		if _, err := adapter.prepareActivationRuntime(ctx, request); err != nil {
			t.Fatal(err)
		}
		verified, err := adapter.runPayloadVerifier(ctx, host, payload, payload+".coldsnap-validation.json", "sha256:"+hex.EncodeToString(hash[:]), int64(len(data)), "")
		if err != nil {
			t.Fatal(err)
		}
		var admission map[string]any
		if err := json.Unmarshal(verified, &admission); err != nil || admission["decision"] != "accept" {
			t.Fatalf("payload verifier result=%s err=%v", verified, err)
		}
		pack, err := activationRuntimePack()
		if err != nil {
			t.Fatal(err)
		}
		binding, ok := criuRPCActivationBinding(root, pack)
		if !ok {
			t.Fatal("no CRIU helper mount")
		}
		// The helper's own argument error proves it executed inside the target
		// container at the exact overlaid path, without invoking CRIU or GPUs.
		image := os.Getenv("COLDSNAP_TEST_TARGET_IMAGE")
		if image == "" {
			t.Fatal("COLDSNAP_TEST_TARGET_IMAGE must name a cached target-platform image")
		}
		platform, err := remote.Run(ctx, host, "uname", "-m")
		if err != nil {
			t.Fatal(err)
		}
		architecture := map[string]string{"aarch64": "arm64", "x86_64": "amd64"}[strings.TrimSpace(string(platform))]
		if architecture == "" {
			t.Fatalf("unsupported target platform %s", platform)
		}
		output, err := remote.RunCombined(ctx, host, "docker", "run", "--rm", "--pull", "never", "--platform", "linux/"+architecture, "--network", "none",
			"--mount", "type=bind,src="+binding.HostPath+",dst="+binding.ContainerPath+",readonly",
			"--entrypoint", binding.ContainerPath, image, "tcp-probe")
		if err == nil || !strings.Contains(string(output)+err.Error(), "--images-dir must be a clean absolute path") {
			t.Fatalf("target helper did not execute correctly: %s err=%v", output, err)
		}
		t.Logf("%s: target identity, native payload verification, and container-mounted CRIU helper passed", engine)
	}
}

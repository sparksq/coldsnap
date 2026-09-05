// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package capsule

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/runtimetest"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

const digest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func testDriver() snapshotdriver.Contract {
	contract, err := snapshotdriver.Lookup(snapshotdriver.N610)
	if err != nil {
		panic(err)
	}
	return contract
}

type call struct {
	input []byte
	args  []string
}

type fakeRemote struct {
	calls            []call
	repositoryDigest string
	imageID          string
}

type fakePusher struct {
	host string
	tag  string
}

func (pusher *fakePusher) PushDockerImage(_ context.Context, host, tag string) error {
	pusher.host = host
	pusher.tag = tag
	return nil
}

func (remote *fakeRemote) Run(_ context.Context, _ string, arguments ...string) ([]byte, error) {
	remote.calls = append(remote.calls, call{args: slices.Clone(arguments)})
	switch arguments[0] {
	case "docker":
		if len(arguments) > 2 && arguments[1] == "image" && arguments[2] == "inspect" {
			reference := remote.repositoryDigest
			if reference == "" {
				reference = "registry.example:5000/capsules@" + digest
			}
			id := remote.imageID
			if id == "" {
				id = digest
			}
			return json.Marshal(hostops.ImageInfo{ID: id, RepoDigests: []string{reference}, Size: 12345})
		}
	}
	return nil, nil
}

func (remote *fakeRemote) RunInput(
	_ context.Context, _ string, input []byte, arguments ...string,
) ([]byte, error) {
	remote.calls = append(remote.calls, call{input: slices.Clone(input), args: slices.Clone(arguments)})
	return nil, nil
}

func TestBuildUsesPinnedBaseAndExcludesModelPayload(t *testing.T) {
	remote := &fakeRemote{}
	result, err := (Builder{Remote: remote, Runtime: runtimetest.NewDockerRuntime(remote, "docker")}).Build(context.Background(), Spec{
		Host: "node-a", BaseImage: "registry/vllm@" + digest, BaseDigest: digest,
		ArtifactRoot: "/var/lib/coldsnap/capture/unit-a", CaptureID: "Qwen-Test", Unit: "unit-a",
		Driver:      testDriver(),
		CacheRootFS: "/var/lib/coldsnap/cache-seed/rank0",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Image.Reference != digest || result.Image.Root != Root || result.Object.Bytes != 12345 {
		t.Fatalf("result = %#v", result)
	}
	if len(remote.calls) != 3 {
		t.Fatalf("calls = %#v", remote.calls)
	}
	if !strings.Contains(string(remote.calls[0].input), "model-weights.pack") {
		t.Fatal("model payload was not excluded from capsule context")
	}
	if !strings.Contains(string(remote.calls[0].input), "model-weights.pack.coldsnap-validation.json") {
		t.Fatal("model payload validation record was not excluded from capsule context")
	}
	build := remote.calls[1]
	if !strings.Contains(string(build.input), "FROM registry/vllm@"+digest) ||
		!strings.Contains(string(build.input), "COPY --from=coldsnap_seed --chown=0:0 / /") ||
		!strings.Contains(string(build.input), "io.sparksq.coldsnap.license=\"AGPL-3.0-only\"") ||
		!strings.Contains(string(build.input), "/opt/coldsnap/licenses/coldsnap/LICENSE") ||
		!strings.Contains(string(build.input), "/opt/coldsnap/licenses/third-party/criu/COPYING") ||
		!slices.Contains(build.args, "SOURCE_DATE_EPOCH=0") ||
		!slices.Contains(build.args, "coldsnap_seed=/var/lib/coldsnap/cache-seed/rank0") {
		t.Fatalf("build = %#v", build)
	}
}

func TestBuildAcceptsLocalContentAddressedBase(t *testing.T) {
	remote := &fakeRemote{}
	builder := Builder{Remote: remote, Runtime: runtimetest.NewDockerRuntime(remote, "docker")}
	_, err := builder.Build(context.Background(), Spec{
		Host: "node-a", BaseImage: digest, BaseDigest: digest,
		ArtifactRoot: "/var/lib/coldsnap/unit-a", CaptureID: "local-capture", Unit: "unit-a",
		Driver: testDriver(),
	})
	if err != nil {
		t.Fatal(err)
	}
	foundTag, foundBuild, foundCleanup := false, false, false
	for _, call := range remote.calls {
		foundTag = foundTag || slices.Equal(call.args, []string{"docker", "tag", digest, "coldsnap-local-base:local-capture-n610-unit-unit-a"})
		foundBuild = foundBuild || strings.Contains(string(call.input), "FROM coldsnap-local-base:local-capture-n610-unit-unit-a\n")
		foundCleanup = foundCleanup || slices.Equal(call.args, []string{"docker", "image", "rm", "coldsnap-local-base:local-capture-n610-unit-unit-a"})
	}
	if !foundTag || !foundBuild || !foundCleanup {
		t.Fatalf("local content-addressed base was not safely tagged for BuildKit: %#v", remote.calls)
	}
}

func TestBuildWithRepositoryStillStagesCapsuleLocally(t *testing.T) {
	remote := &fakeRemote{}
	result, err := (Builder{Remote: remote, Runtime: runtimetest.NewDockerRuntime(remote, "docker")}).Build(context.Background(), Spec{
		Host: "node-a", BaseImage: "registry/vllm@" + digest, BaseDigest: digest,
		ArtifactRoot: "/var/lib/coldsnap/capture/unit-b", CaptureID: "qwen", Unit: "unit-b",
		Driver:     testDriver(),
		Repository: "registry.example:5000/capsules",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Image.Reference != digest {
		t.Fatalf("image = %#v", result.Image)
	}
	for _, call := range remote.calls {
		if slices.Contains(call.args, "push") {
			t.Fatalf("capture pushed a capsule: %#v", remote.calls)
		}
	}
}

func TestPublishPromotesLocalCapsuleToRepositoryDigest(t *testing.T) {
	remote := &fakeRemote{}
	// Publication needs only the typed runtime, not a node-command transport.
	image, err := (Builder{Runtime: runtimetest.NewDockerRuntime(remote, "docker")}).Publish(context.Background(), PublishSpec{
		Host: "node-a", Source: digest, CaptureID: "qwen", Unit: "unit-b",
		Driver:     testDriver(),
		Repository: "registry.example:5000/capsules",
	})
	if err != nil {
		t.Fatal(err)
	}
	if image.Reference != "registry.example:5000/capsules@"+digest || image.Digest != digest {
		t.Fatalf("image = %#v", image)
	}
	for _, expected := range []string{"tag", "push", "{{json .RepoDigests}}"} {
		found := false
		for _, call := range remote.calls {
			if strings.Contains(strings.Join(call.args, " "), expected) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("publish lacks %q: %#v", expected, remote.calls)
		}
	}
}

func TestPublishUsesControllerAuthenticatedPusherWhenAvailable(t *testing.T) {
	remote := &fakeRemote{}
	pusher := &fakePusher{}
	_, err := (Builder{Remote: remote, Runtime: publishingRuntime{runtimetest.NewDockerRuntime(remote, "docker"), pusher}}).Publish(context.Background(), PublishSpec{
		Host: "node-a", Source: digest, CaptureID: "qwen", Unit: "unit-b",
		Driver:     testDriver(),
		Repository: "registry.example:5000/capsules",
	})
	if err != nil {
		t.Fatal(err)
	}
	if pusher.host != "node-a" || pusher.tag != "registry.example:5000/capsules:qwen-n610-unit-unit-b" {
		t.Fatalf("authenticated push = host %q tag %q", pusher.host, pusher.tag)
	}
	for _, call := range remote.calls {
		if slices.Contains(call.args, "push") {
			t.Fatalf("publish fell back to rank-local credentials: %#v", remote.calls)
		}
	}
}

func TestPublishAcceptsDockerHubRepositoryAlias(t *testing.T) {
	remote := &fakeRemote{repositoryDigest: "scitrera/capsules@" + digest}
	image, err := (Builder{Remote: remote, Runtime: runtimetest.NewDockerRuntime(remote, "docker")}).Publish(context.Background(), PublishSpec{
		Host: "node-a", Source: digest, CaptureID: "qwen", Unit: "unit-b",
		Driver:     testDriver(),
		Repository: "docker.io/scitrera/capsules",
	})
	if err != nil {
		t.Fatal(err)
	}
	if image.Reference != "docker.io/scitrera/capsules@"+digest {
		t.Fatalf("image = %#v", image)
	}
}

type publishingRuntime struct {
	backend hostops.Runtime
	pusher  *fakePusher
}

func TestLocalIdentityRejectsNonHexImageDigest(t *testing.T) {
	remote := &fakeRemote{imageID: "sha256:" + strings.Repeat("g", 64)}
	_, _, _, err := (Builder{}).localIdentity(context.Background(), "node", runtimetest.NewDockerRuntime(remote, "docker"), "capsule:local")
	if err == nil || !strings.Contains(err.Error(), "invalid image ID") {
		t.Fatalf("invalid OCI identity accepted: %v", err)
	}
}

func (runtime publishingRuntime) Runtime(ctx context.Context, host string, request hostops.RuntimeRequest) (hostops.RuntimeResponse, error) {
	if request.Action == hostops.RuntimeImagePush {
		return hostops.RuntimeResponse{}, runtime.pusher.PushDockerImage(ctx, host, request.Image)
	}
	return runtime.backend.Runtime(ctx, host, request)
}

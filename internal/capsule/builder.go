// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package capsule

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/sparksq/coldsnap/internal/hostops"
	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

const Root = "/opt/coldsnap/capsule"

var (
	safeTag       = regexp.MustCompile(`[^a-z0-9_.-]+`)
	sha256Pattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

// Remote is the minimal manager-provided host-operation surface needed for
// capsule construction; its implementation does not affect the OCI contract.
type Remote interface {
	RunInput(context.Context, string, []byte, ...string) ([]byte, error)
}

type Spec struct {
	Host         string
	BaseImage    string
	BaseDigest   string
	ArtifactRoot string
	CacheRootFS  string
	CaptureID    string
	Unit         string
	Driver       snapshotdriver.Contract
	Repository   string
}

type PublishSpec struct {
	Host       string
	Source     string
	CaptureID  string
	Unit       string
	Driver     snapshotdriver.Contract
	Repository string
}

type Result struct {
	Image  snapshot.CapsuleImage
	Object snapshot.Object
}

type Builder struct {
	Remote  Remote
	Runtime hostops.Runtime
}

// Publish promotes one already-built, identity-pinned local capsule to an OCI
// repository. Capture deliberately does not call this path: publication is a
// separate operator action after the local artifact has been accepted.
func (builder Builder) Publish(ctx context.Context, spec PublishSpec) (snapshot.CapsuleImage, error) {
	if builder.Runtime == nil {
		return snapshot.CapsuleImage{}, errors.New("capsule publication requires a manager runtime")
	}
	if spec.Host == "" || spec.CaptureID == "" || spec.Unit == "" ||
		!sha256Pattern.MatchString(spec.Source) ||
		spec.Repository == "" || strings.ContainsAny(spec.Repository, "@ \t\r\n\x00") ||
		strings.LastIndexByte(spec.Repository, ':') > strings.LastIndexByte(spec.Repository, '/') {
		return snapshot.CapsuleImage{}, errors.New("capsule publish specification is invalid")
	}
	if err := spec.Driver.Validate(); err != nil {
		return snapshot.CapsuleImage{}, fmt.Errorf("capsule publish snapshot driver: %w", err)
	}
	runtime := builder.Runtime
	inspected, err := runtime.Runtime(ctx, spec.Host, hostops.RuntimeRequest{Action: hostops.RuntimeImageInspect, Image: spec.Source})
	if err != nil {
		return snapshot.CapsuleImage{}, fmt.Errorf("inspect local unit %s OCI capsule: %w", spec.Unit, err)
	}
	if inspected.Image == nil || inspected.Image.ID != spec.Source {
		return snapshot.CapsuleImage{}, fmt.Errorf("local unit %s OCI capsule identity changed", spec.Unit)
	}
	tag := imageTag(spec.Repository, spec.CaptureID, spec.Driver.ID, spec.Unit)
	if _, err := runtime.Runtime(ctx, spec.Host, hostops.RuntimeRequest{Action: hostops.RuntimeImageTag, Source: spec.Source, Target: tag}); err != nil {
		return snapshot.CapsuleImage{}, fmt.Errorf("tag unit %s OCI capsule: %w", spec.Unit, err)
	}
	reference, digest, err := builder.publishedIdentity(ctx, spec.Host, runtime, tag)
	if err != nil {
		return snapshot.CapsuleImage{}, err
	}
	return snapshot.CapsuleImage{
		Unit: spec.Unit, Reference: reference, Digest: digest, Root: Root, Driver: spec.Driver.Binding(),
	}, nil
}

func (builder Builder) Build(ctx context.Context, spec Spec) (Result, error) {
	if builder.Remote == nil || builder.Runtime == nil {
		return Result{}, errors.New("capsule construction requires a manager host transport and runtime")
	}
	if spec.Host == "" || spec.CaptureID == "" || spec.Unit == "" ||
		!filepath.IsAbs(spec.ArtifactRoot) || filepath.Clean(spec.ArtifactRoot) != spec.ArtifactRoot ||
		(spec.CacheRootFS != "" && (!filepath.IsAbs(spec.CacheRootFS) || filepath.Clean(spec.CacheRootFS) != spec.CacheRootFS)) ||
		strings.ContainsAny(spec.BaseImage, "\r\n\x00") ||
		(spec.Repository != "" && strings.LastIndexByte(spec.Repository, ':') > strings.LastIndexByte(spec.Repository, '/')) ||
		!sha256Pattern.MatchString(spec.BaseDigest) ||
		(spec.BaseImage != spec.BaseDigest && !strings.HasSuffix(spec.BaseImage, "@"+spec.BaseDigest)) {
		return Result{}, errors.New("capsule build specification is invalid")
	}
	if err := spec.Driver.Validate(); err != nil {
		return Result{}, fmt.Errorf("capsule build snapshot driver: %w", err)
	}
	runtime := builder.Runtime
	baseImage := spec.BaseImage
	if spec.BaseImage == spec.BaseDigest {
		// Docker accepts a bare image ID for `run`, but BuildKit parses
		// `FROM sha256:...` as the registry name docker.io/library/sha256.
		// Bind the verified local ID to a capture-scoped tag for the build and
		// remove only that tag afterward; --pull=false keeps the base local.
		baseImage = imageTag("coldsnap-local-base", spec.CaptureID, spec.Driver.ID, spec.Unit)
		if _, err := runtime.Runtime(ctx, spec.Host, hostops.RuntimeRequest{Action: hostops.RuntimeImageTag, Source: spec.BaseDigest, Target: baseImage}); err != nil {
			return Result{}, fmt.Errorf("tag local unit %s capsule base: %w", spec.Unit, err)
		}
		defer func() {
			_, _ = runtime.Runtime(ctx, spec.Host, hostops.RuntimeRequest{Action: hostops.RuntimeImageRemove, Image: baseImage})
		}()
		inspected, err := runtime.Runtime(ctx, spec.Host, hostops.RuntimeRequest{Action: hostops.RuntimeImageInspect, Image: baseImage})
		if err != nil {
			return Result{}, fmt.Errorf("inspect local unit %s capsule base: %w", spec.Unit, err)
		}
		if inspected.Image == nil || inspected.Image.ID != spec.BaseDigest {
			return Result{}, fmt.Errorf("local unit %s capsule base identity changed", spec.Unit)
		}
	}
	tag := imageTag(spec.Repository, spec.CaptureID, spec.Driver.ID, spec.Unit)
	ignore := []byte("**/model-weights.pack\n**/model-weights.pack.coldsnap-validation.json\n**/.coldsnap-capsule-*\n")
	if _, err := builder.Remote.RunInput(
		ctx, spec.Host, ignore, "tee", filepath.Join(spec.ArtifactRoot, ".dockerignore"),
	); err != nil {
		return Result{}, fmt.Errorf("write capsule ignore file on %s: %w", spec.Host, err)
	}
	seedCopy := ""
	if spec.CacheRootFS != "" {
		seedCopy = "COPY --from=coldsnap_seed --chown=0:0 / /\n"
	}
	driverValidation := fmt.Sprintf(
		"test -d %s/hibernate-states \\\n && test -n \"$(find %s/hibernate-states -mindepth 1 -maxdepth 1 -type f -name '*.json' -print -quit)\"",
		Root, Root,
	)
	if spec.Driver.ID == snapshotdriver.N580 {
		driverValidation = fmt.Sprintf(
			"test -s %s/template/images/inventory.img \\\n && test -s %s/template/external-files.json \\\n && test -d %s/hibernate-states \\\n && test -n \"$(find %s/hibernate-states -mindepth 1 -maxdepth 1 -type f -name '*.json' -print -quit)\"",
			Root, Root, Root, Root,
		)
	}
	dockerfile := []byte(fmt.Sprintf(`FROM %s
ARG SOURCE_DATE_EPOCH=0
LABEL io.sparksq.coldsnap.kind="unit-capsule" \
      io.sparksq.coldsnap.capture="%s" \
      io.sparksq.coldsnap.unit="%s" \
      io.sparksq.coldsnap.snapshot-driver="%s" \
      io.sparksq.coldsnap.snapshot-driver-abi="%d" \
      io.sparksq.coldsnap.base-digest="%s" \
      io.sparksq.coldsnap.copyright-holders="Scitrera LLC and Fox Engine Ltd" \
      io.sparksq.coldsnap.license="AGPL-3.0-only" \
      io.sparksq.coldsnap.source="https://github.com/sparksq/coldsnap"
%sCOPY --chown=0:0 . %s/
RUN test -s %s/capture.json \
 && test -s /opt/coldsnap/licenses/coldsnap/LICENSE \
 && test -s /opt/coldsnap/licenses/coldsnap/THIRD_PARTY_NOTICES.md \
 && test -s /opt/coldsnap/licenses/third-party/criu/COPYING \
 && %s \
 && find %s -xdev -type d -exec chmod go-w {} +
`, baseImage, spec.CaptureID, spec.Unit, spec.Driver.ID, spec.Driver.ABI, spec.BaseDigest, seedCopy, Root, Root, driverValidation, Root))
	contexts := map[string]string{}
	if spec.CacheRootFS != "" {
		contexts["coldsnap_seed"] = spec.CacheRootFS
	}
	if _, err := runtime.Runtime(ctx, spec.Host, hostops.RuntimeRequest{
		Action: hostops.RuntimeImageBuild,
		Build: &hostops.ImageBuild{
			Dockerfile: dockerfile, Context: spec.ArtifactRoot, Contexts: contexts,
			Arguments: map[string]string{"SOURCE_DATE_EPOCH": "0"}, Tag: tag,
		},
	}); err != nil {
		return Result{}, fmt.Errorf("build unit %s OCI capsule on %s: %w", spec.Unit, spec.Host, err)
	}

	reference, digest, info, err := builder.localIdentity(ctx, spec.Host, runtime, tag)
	if err != nil {
		return Result{}, err
	}
	if info.Size <= 0 {
		return Result{}, errors.New("OCI capsule size is invalid")
	}
	return Result{
		Image: snapshot.CapsuleImage{
			Unit: spec.Unit, Reference: reference, Digest: digest, Root: Root, Driver: spec.Driver.Binding(),
		},
		Object: snapshot.Object{
			Role: "oci-capsule", Owner: snapshot.UnitOwner(spec.Unit),
			Path: fmt.Sprintf("units/%s/capsule.oci", spec.Unit), Bytes: info.Size, SHA256: digest,
		},
	}, nil
}

func (builder Builder) publishedIdentity(ctx context.Context, host string, runtime hostops.Runtime, tag string) (string, string, error) {
	_, err := runtime.Runtime(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeImagePush, Image: tag})
	if err != nil {
		return "", "", fmt.Errorf("push OCI capsule %s: %w", tag, err)
	}
	inspected, err := runtime.Runtime(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeImageInspect, Image: tag})
	if err != nil {
		return "", "", fmt.Errorf("inspect pushed OCI capsule %s: %w", tag, err)
	}
	if inspected.Image == nil {
		return "", "", errors.New("pushed OCI capsule inspection is absent")
	}
	separator := strings.LastIndexByte(tag, ':')
	if separator <= strings.LastIndexByte(tag, '/') {
		return "", "", errors.New("OCI capsule tag is invalid")
	}
	repository := tag[:separator]
	for _, reference := range inspected.Image.RepoDigests {
		at := strings.LastIndexByte(reference, '@')
		if at > 0 && canonicalRepository(reference[:at]) == canonicalRepository(repository) {
			digest := reference[at+1:]
			if sha256Pattern.MatchString(digest) {
				return repository + "@" + digest, digest, nil
			}
		}
	}
	return "", "", errors.New("pushed OCI capsule has no repository digest")
}

func canonicalRepository(value string) string {
	value = strings.TrimPrefix(value, "index.docker.io/")
	return strings.TrimPrefix(value, "docker.io/")
}

func (builder Builder) localIdentity(
	ctx context.Context, host string, runtime hostops.Runtime, tag string,
) (string, string, hostops.ImageInfo, error) {
	inspected, err := runtime.Runtime(ctx, host, hostops.RuntimeRequest{Action: hostops.RuntimeImageInspect, Image: tag})
	if err != nil {
		return "", "", hostops.ImageInfo{}, fmt.Errorf("inspect local OCI capsule %s: %w", tag, err)
	}
	if inspected.Image == nil {
		return "", "", hostops.ImageInfo{}, errors.New("local OCI capsule inspection is absent")
	}
	digest := inspected.Image.ID
	if !sha256Pattern.MatchString(digest) {
		return "", "", hostops.ImageInfo{}, errors.New("local OCI capsule has an invalid image ID")
	}
	return digest, digest, *inspected.Image, nil
}

func imageTag(repository, captureID, driver, unit string) string {
	slug := safeTag.ReplaceAllString(strings.ToLower(captureID), "-")
	slug = strings.Trim(slug, ".-")
	if slug == "" {
		slug = "capture"
	}
	if len(slug) > 100 {
		slug = slug[:100]
	}
	if repository == "" {
		repository = "coldsnap-capsule/" + slug
	}
	unit = strings.Trim(safeTag.ReplaceAllString(strings.ToLower(unit), "-"), ".-")
	if unit == "" {
		unit = "unit"
	}
	driver = strings.Trim(safeTag.ReplaceAllString(strings.ToLower(driver), "-"), ".-")
	if driver == "" {
		driver = "driver"
	}
	return fmt.Sprintf("%s:%s-%s-unit-%s", repository, slug, driver, unit)
}

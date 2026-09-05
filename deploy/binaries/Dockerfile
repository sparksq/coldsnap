# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

FROM --platform=$BUILDPLATFORM golang:1.25.14@sha256:699337d620559a59b4a2bb298ad59611e535d2ee755a34cf2d2a98f37578dc80 AS builder

ARG TARGETOS
ARG TARGETARCH
ARG VERSION
ARG COMMIT

WORKDIR /src/coldsnap

COPY go.mod go.sum ./
RUN go mod download

COPY cmd ./cmd
COPY internal ./internal
COPY runtime ./runtime

RUN set -eu; \
    test -n "$TARGETOS"; \
    test -n "$TARGETARCH"; \
    test -n "$VERSION"; \
    test "${#COMMIT}" -eq 40; \
    case "$COMMIT" in *[!0-9a-f]*) exit 2 ;; esac; \
    mkdir -p /out; \
    ldflags="-s -w -buildid= -X github.com/sparksq/coldsnap/internal/buildinfo.Version=$VERSION -X github.com/sparksq/coldsnap/internal/buildinfo.Commit=$COMMIT"; \
    CGO_ENABLED=0 GOOS="$TARGETOS" GOARCH="$TARGETARCH" go build -trimpath -ldflags "$ldflags" -o /out/coldsnap ./cmd/coldsnap; \
    CGO_ENABLED=0 GOOS="$TARGETOS" GOARCH="$TARGETARCH" go build -trimpath -ldflags "$ldflags" -o /out/coldsnap-vllm-adapter ./cmd/coldsnap-vllm-adapter; \
    CGO_ENABLED=0 GOOS="$TARGETOS" GOARCH="$TARGETARCH" go build -trimpath -ldflags "$ldflags" -o /out/coldsnap-sglang-adapter ./cmd/coldsnap-sglang-adapter; \
    (cd cmd/coldsnap-criu-rpc && CGO_ENABLED=0 GOOS="$TARGETOS" GOARCH="$TARGETARCH" go build -trimpath -ldflags "-s -w -buildid=" -o /out/coldsnap-criu-rpc .); \
    cd /out; \
    coldsnap_sha="$(sha256sum coldsnap | cut -d' ' -f1)"; \
    vllm_sha="$(sha256sum coldsnap-vllm-adapter | cut -d' ' -f1)"; \
    sglang_sha="$(sha256sum coldsnap-sglang-adapter | cut -d' ' -f1)"; \
    criu_rpc_sha="$(sha256sum coldsnap-criu-rpc | cut -d' ' -f1)"; \
    printf '%s\n' \
      '{' \
      '  "format": 1,' \
      '  "kind": "coldsnap-controller-binary-bundle",' \
      "  \"version\": \"$VERSION\"," \
      "  \"commit\": \"$COMMIT\"," \
      "  \"platform\": \"$TARGETOS-$TARGETARCH\"," \
      '  "sha256": {' \
      "    \"coldsnap\": \"$coldsnap_sha\"," \
      "    \"coldsnap-vllm-adapter\": \"$vllm_sha\"," \
      "    \"coldsnap-sglang-adapter\": \"$sglang_sha\"," \
      "    \"coldsnap-criu-rpc\": \"$criu_rpc_sha\"" \
      '  }' \
      '}' > manifest.json

FROM scratch

ARG VERSION
ARG COMMIT

LABEL org.opencontainers.image.title="ColdSnap controller binary bundle" \
      org.opencontainers.image.description="Architecture-matched ColdSnap controller and engine adapter executables" \
      org.opencontainers.image.source="https://github.com/sparksq/coldsnap" \
      org.opencontainers.image.version="$VERSION" \
      org.opencontainers.image.revision="$COMMIT" \
      org.opencontainers.image.licenses="AGPL-3.0-only AND Apache-2.0" \
      org.opencontainers.image.vendor="Scitrera LLC and Fox Engine Ltd" \
      io.sparksq.coldsnap.bundle.kind="coldsnap-controller-binary-bundle" \
      io.sparksq.coldsnap.bundle.format="1"

COPY --from=builder /out/ /opt/coldsnap/bin/
COPY LICENSE THIRD_PARTY_NOTICES.md /usr/share/licenses/coldsnap/
COPY third_party/licenses/go-criu/LICENSE /usr/share/licenses/coldsnap/third-party/go-criu/LICENSE
COPY README.md /usr/share/doc/coldsnap/README.md

ENTRYPOINT ["/opt/coldsnap/bin/coldsnap"]

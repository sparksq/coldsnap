# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

ARG VLLM_IMAGE
ARG CRIU_IMAGE

FROM golang:1.25.14@sha256:699337d620559a59b4a2bb298ad59611e535d2ee755a34cf2d2a98f37578dc80 AS coldsnap_builder
WORKDIR /src/coldsnap
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -o /out/coldsnap ./cmd/coldsnap \
 && CGO_ENABLED=0 go build -trimpath -o /out/coldsnap-coordinator ./cmd/coldsnap-coordinator

FROM golang:1.25.14@sha256:699337d620559a59b4a2bb298ad59611e535d2ee755a34cf2d2a98f37578dc80 AS coldsnap_criu_rpc_builder
COPY --from=go_criu_source . /src/coldsnap/build/sources/go-criu
COPY cmd/coldsnap-criu-rpc /src/coldsnap/cmd/coldsnap-criu-rpc
WORKDIR /src/coldsnap/cmd/coldsnap-criu-rpc
RUN go test ./... \
 && CGO_ENABLED=0 go build -trimpath -o /out/coldsnap-criu-rpc .

FROM ${VLLM_IMAGE} AS native_builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src/coldsnap
COPY native ./native
COPY native/coldsnap_nccl_dlsym.c ./native/coldsnap_nccl_dlsym.c
RUN mkdir -p /out \
 && c++ -std=c++17 -O2 -fPIC -Wall -Wextra -Werror \
      -I/usr/local/cuda/include -shared native/coldsnap_hydration.cpp \
      -Wl,--no-undefined -Wl,-soname,libcoldsnap_hydration.so \
      -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart \
      -lcrypto -lz -ldl -pthread -o /out/libcoldsnap_hydration.so \
 && c++ -std=c++17 -O2 -fPIC -Wall -Wextra -Werror \
      -I/usr/local/cuda/include -shared native/coldsnap_cuda_epoch.cpp \
      -Wl,--no-undefined -Wl,-soname,libcoldsnap_cuda_epoch.so \
      -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart \
      -L/usr/local/cuda/lib64/stubs -lcuda -ldl -pthread \
      -o /out/libcoldsnap_cuda_epoch.so \
 && cc -shared -fPIC -O2 -Wall -Wextra -Werror \
      native/coldsnap_nccl_dlsym.c -ldl \
      -o /out/libcoldsnap-nccl-dlsym.so \
 && cc -std=c11 -O2 -fPIC -Wall -Wextra -Werror -shared \
      native/coldsnap_nvidia_mmap_shim.c \
      -Wl,--no-undefined -Wl,-soname,libcoldsnap_nvidia_mmap_shim.so \
      -ldl -pthread -o /out/libcoldsnap_nvidia_mmap_shim.so \
 && cc -std=c11 -O2 -fPIC -Wall -Wextra -Werror -shared \
      native/coldsnap_nvml_dlopen_shim.c \
      -Wl,--no-undefined -Wl,-soname,libcoldsnap_nvml_dlopen_shim.so \
      -ldl -o /out/libcoldsnap_nvml_dlopen_shim.so

FROM ${VLLM_IMAGE} AS python_plugin_builder
WORKDIR /src/coldsnap
COPY integrations/core/coldsnap_core /out/plugin/coldsnap_core
COPY integrations/vllm /src/coldsnap/vllm
RUN cp /src/coldsnap/vllm/coldsnap_*.py /src/coldsnap/vllm/sitecustomize.py /out/plugin/ \
 && python3 /src/coldsnap/vllm/render_runtime_metadata.py \
      --project /src/coldsnap/vllm/pyproject.toml \
      --output /out/plugin

FROM ${VLLM_IMAGE}
ARG VLLM_IMAGE
ARG CRIU_IMAGE
LABEL io.sparksq.coldsnap.runtime="vllm-cuda-criu-v1" \
      io.sparksq.coldsnap.base-image="${VLLM_IMAGE}" \
      io.sparksq.coldsnap.criu-image="${CRIU_IMAGE}" \
      io.sparksq.coldsnap.copyright-holders="Scitrera LLC and Fox Engine Ltd" \
      io.sparksq.coldsnap.license="AGPL-3.0-only" \
      io.sparksq.coldsnap.source="https://github.com/sparksq/coldsnap"

RUN case "${VLLM_IMAGE}" in *@sha256:*) ;; *) echo "VLLM_IMAGE must be digest-pinned" >&2; exit 2 ;; esac \
 && image_digest="${VLLM_IMAGE##*@sha256:}" \
 && test "${#image_digest}" -eq 64 \
 && case "${image_digest}" in *[!0-9a-f]*) echo "VLLM_IMAGE digest is invalid" >&2; exit 2 ;; esac \
 && case "${CRIU_IMAGE}" in *@sha256:*) ;; *) echo "CRIU_IMAGE must be digest-pinned" >&2; exit 2 ;; esac \
 && criu_digest="${CRIU_IMAGE##*@sha256:}" \
 && test "${#criu_digest}" -eq 64 \
 && case "${criu_digest}" in *[!0-9a-f]*) echo "CRIU_IMAGE digest is invalid" >&2; exit 2 ;; esac \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      iproute2 iptables kmod nftables procps python3 util-linux \
      libbsd0 libgnutls30t64 liblz4-1 libnet1 libnftables1 libnl-3-200 \
      libprotobuf-c1 libselinux1 libuuid1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=criu_image /usr/sbin/criu /opt/coldsnap/criu/bin/criu
COPY LICENSE THIRD_PARTY_NOTICES.md /opt/coldsnap/licenses/coldsnap/
COPY third_party/licenses/criu/COPYING /opt/coldsnap/licenses/third-party/criu/COPYING
COPY third_party/licenses/nccl/LICENSE.txt /opt/coldsnap/licenses/third-party/nccl/LICENSE.txt
COPY --from=go_criu_source LICENSE /opt/coldsnap/licenses/third-party/go-criu/LICENSE
COPY --from=cuda_checkpoint_source LICENSE /opt/coldsnap/licenses/third-party/cuda-checkpoint/LICENSE
RUN lz4_path="$(find /usr/lib -name liblz4.so.1 -print -quit)" \
 && test -n "$lz4_path" \
 && install -D -m 0644 "$lz4_path" /opt/coldsnap/criu/lib/liblz4.so.1 \
 && find /opt/coldsnap/licenses -type f -exec chmod 0644 {} + \
 && test -s /opt/coldsnap/licenses/coldsnap/LICENSE \
 && test -s /opt/coldsnap/licenses/coldsnap/THIRD_PARTY_NOTICES.md \
 && test -s /opt/coldsnap/licenses/third-party/criu/COPYING \
 && test -s /opt/coldsnap/licenses/third-party/nccl/LICENSE.txt \
 && test -s /opt/coldsnap/licenses/third-party/go-criu/LICENSE \
 && test -s /opt/coldsnap/licenses/third-party/cuda-checkpoint/LICENSE

COPY --from=coldsnap_builder /out/coldsnap /usr/local/bin/coldsnap
COPY --from=coldsnap_builder /out/coldsnap-coordinator /usr/local/bin/coldsnap-coordinator
COPY --from=coldsnap_criu_rpc_builder /out/coldsnap-criu-rpc /usr/local/bin/coldsnap-criu-rpc
COPY --from=cuda_checkpoint_source bin /tmp/cuda-checkpoint-bin
COPY --from=native_builder /out/libcoldsnap_hydration.so /opt/coldsnap/native/libcoldsnap_hydration.so
COPY --from=native_builder /out/libcoldsnap_cuda_epoch.so /opt/coldsnap/native/libcoldsnap_cuda_epoch.so
COPY --from=criu_image /usr/lib/criu/coldsnap_nvidia_reset_plugin.so /opt/coldsnap/criu/plugins/libcoldsnap_criu_nvidia_reset.so
COPY --from=native_builder /out/libcoldsnap_nvidia_mmap_shim.so /opt/coldsnap/native/libcoldsnap_nvidia_mmap_shim.so
COPY --from=native_builder /out/libcoldsnap_nvml_dlopen_shim.so /opt/coldsnap/native/libcoldsnap_nvml_dlopen_shim.so
COPY --from=nccl_provider . /tmp/coldsnap-nccl-provider/
COPY --from=native_builder /out/libcoldsnap-nccl-dlsym.so /tmp/libcoldsnap-nccl-dlsym.so

COPY --from=python_plugin_builder /out/plugin /opt/coldsnap/plugin
COPY runtime/shared/coldsnap_coord.py /opt/coldsnap/runtime/coldsnap_coord.py
COPY runtime/engine/coldsnap_cuda_criu.py /opt/coldsnap/runtime/coldsnap_cuda_criu.py
COPY runtime/engine/coldsnap_engine_exec.py /opt/coldsnap/runtime/coldsnap-engine-exec.py
COPY runtime/engine/coldsnap_n580_criu.py /opt/coldsnap/runtime/coldsnap_n580_criu.py

RUN chmod 0755 \
      /usr/local/bin/coldsnap \
      /usr/local/bin/coldsnap-coordinator \
      /usr/local/bin/coldsnap-criu-rpc \
      /opt/coldsnap/runtime/coldsnap-engine-exec.py \
 && case "$(uname -m)" in \
      aarch64|arm64) cuda_checkpoint=/tmp/cuda-checkpoint-bin/aarch64_Linux/cuda-checkpoint ;; \
      x86_64|amd64) cuda_checkpoint=/tmp/cuda-checkpoint-bin/x86_64_Linux/cuda-checkpoint ;; \
      *) echo "unsupported cuda-checkpoint architecture" >&2; exit 1 ;; \
    esac \
 && install -m 0755 "$cuda_checkpoint" /usr/local/bin/cuda-checkpoint \
 && rm -rf /tmp/cuda-checkpoint-bin \
 && coldsnap nccl-provider install-active \
      --provider /tmp/coldsnap-nccl-provider \
      --bridge /tmp/libcoldsnap-nccl-dlsym.so \
      --bridge-abi 1 \
      --output /opt/coldsnap/nccl \
 && rm -rf /tmp/coldsnap-nccl-provider /tmp/libcoldsnap-nccl-dlsym.so \
 && coldsnap nccl-provider verify-active --active /opt/coldsnap/nccl/active.json \
 && test -f /opt/coldsnap/plugin/coldsnap_plugin.py \
 && test -f /opt/coldsnap/plugin/coldsnap_core/__init__.py \
 && test "$(find /opt/coldsnap/plugin -mindepth 2 -maxdepth 2 -type f -path '*/coldsnap_vllm-*.dist-info/entry_points.txt' | wc -l)" -eq 1 \
 && /opt/coldsnap/criu/bin/criu --version

ENV PYTHONPATH=/opt/coldsnap/plugin:/opt/coldsnap/runtime \
    VLLM_PLUGINS=coldsnap \
    COLDSNAP_CUDA_EPOCH_NATIVE_LIBRARY=/opt/coldsnap/native/libcoldsnap_cuda_epoch.so \
    COLDSNAP_NCCL_ACTIVE_RUNTIME_PATH=/opt/coldsnap/nccl/active.json \
    COLDSNAP_NCCL_REQUIRED_CAPABILITIES=full-network-reset,ib-roce-device-release,synchronous-termination

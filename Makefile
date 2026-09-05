# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

.PHONY: build check clean criu-nvidia-mmap-shim criu-rpc-go-test criu-rpc-runtime documentation-links ensure-criu-rpc-inputs ensure-safe-bin-dir ensure-safe-build-dir ensure-safe-dist-dir ensure-safe-native-build-dir ensure-safe-python-dist-dir ensure-safe-python-runtime-dir go-criu-source go-test license-headers lint native native-cuda-epoch native-graph native-hydration native-nccl-dlsym nccl-provider-materialize nccl-provider-target nvml-dlopen-shim python-lint python-lock-check python-package-build python-runtime-metadata python-security python-test race repository-layout reproducible-build secret-scan sglang-contract sglang-runtime-image static-analysis test vllm-contract vllm-cuda-criu-image vllm-runtime-image vuln

BIN_DIR ?= bin
BIN_DIR_ABS := $(abspath $(BIN_DIR))
BUILD_DIR ?= build
BUILD_DIR_ABS := $(abspath $(BUILD_DIR))
DIST_DIR ?= dist
DIST_DIR_ABS := $(abspath $(DIST_DIR))
GO_BUILD_FLAGS ?= -trimpath -buildvcs=true -ldflags=-buildid=
PYTHON ?= python3
UV ?= uv
RUFF ?= ruff
STATICCHECK ?= staticcheck
GOVULNCHECK ?= govulncheck
ACTIONLINT ?= actionlint
GITLEAKS ?= gitleaks
BANDIT ?= bandit
PYTHON_PRODUCT_SOURCES := runtime integrations scripts
PYTHON_BENCHMARK_SOURCES := benchmarks/harnesses
PYTHON_DIST_DIR ?= $(BUILD_DIR)/python-dist
PYTHON_DIST_DIR_ABS := $(abspath $(PYTHON_DIST_DIR))
PYTHON_RUNTIME_DIR ?= $(BUILD_DIR)/python-runtime
PYTHON_RUNTIME_DIR_ABS := $(abspath $(PYTHON_RUNTIME_DIR))
CUDA_HOME ?= /usr/local/cuda
CUDA_DRIVER_LIB_DIR ?= /usr/lib/$(shell uname -m)-linux-gnu
NATIVE_BUILD_DIR ?= $(BUILD_DIR)/native
NATIVE_BUILD_DIR_ABS := $(abspath $(NATIVE_BUILD_DIR))
GRAPH_MEMORY_LIBRARY := $(NATIVE_BUILD_DIR_ABS)/libcoldsnap_graph_memory.so
HYDRATION_LIBRARY := $(NATIVE_BUILD_DIR_ABS)/libcoldsnap_hydration.so
CUDA_EPOCH_LIBRARY := $(NATIVE_BUILD_DIR_ABS)/libcoldsnap_cuda_epoch.so
CRIU_NVIDIA_MMAP_SHIM := $(NATIVE_BUILD_DIR_ABS)/libcoldsnap_nvidia_mmap_shim.so
NVML_DLOPEN_SHIM := $(NATIVE_BUILD_DIR_ABS)/libcoldsnap_nvml_dlopen_shim.so
NCCL_DLSYM_LIBRARY := $(NATIVE_BUILD_DIR_ABS)/libcoldsnap-nccl-dlsym.so
GO_CRIU_SOURCE_URL ?= https://github.com/sparksq/go-criu.git
GO_CRIU_SOURCE_REVISION ?= 29a4f2f8e8374d38319a9851d9c1ef880dd0a0e8
GO_CRIU_FORK_ROOT ?= $(BUILD_DIR_ABS)/sources/go-criu
CRIU_IMAGE ?= ghcr.io/sparksq/criu@sha256:2ff53a61af48e7e676bd4d64747394ca7c7622840ef0c740e719e6ecadb0d07c
CUDA_CHECKPOINT_SOURCE_ROOT ?=
VLLM_CUDA_CRIU_IMAGE ?= coldsnap-vllm-cuda-criu:rpc
VLLM_RUNTIME_IMAGE ?= coldsnap-vllm-runtime:local
VLLM_BASE_IMAGE ?=
SGLANG_RUNTIME_IMAGE ?= coldsnap-sglang-runtime:local
SGLANG_BASE_IMAGE ?=
NCCL_PROVIDER_BASE_IMAGE ?= $(VLLM_BASE_IMAGE)
NCCL_PROVIDER_CATALOG ?=
NCCL_PROVIDER_TARGET ?= $(BUILD_DIR_ABS)/nccl-provider-target.json
NCCL_PROVIDER_BUILD_DIR ?= $(BUILD_DIR_ABS)/nccl-provider
NCCL_PROVIDER_TARGET_OBSERVE ?= 1
NCCL_PROVIDER_TRANSPORT ?= ib-roce
DOCKER ?= docker
ensure-safe-bin-dir:
	@case "$(BIN_DIR_ABS)" in "$(CURDIR)"/*) ;; *) echo "refusing unsafe BIN_DIR: $(BIN_DIR_ABS)" >&2; exit 1 ;; esac

ensure-safe-build-dir:
	@case "$(BUILD_DIR_ABS)" in "$(CURDIR)"/*) ;; *) echo "refusing unsafe BUILD_DIR: $(BUILD_DIR_ABS)" >&2; exit 1 ;; esac

ensure-safe-dist-dir:
	@case "$(DIST_DIR_ABS)" in "$(CURDIR)"/*) ;; *) echo "refusing unsafe DIST_DIR: $(DIST_DIR_ABS)" >&2; exit 1 ;; esac

ensure-safe-native-build-dir:
	@case "$(NATIVE_BUILD_DIR_ABS)" in "$(CURDIR)"/*) ;; *) echo "refusing unsafe NATIVE_BUILD_DIR: $(NATIVE_BUILD_DIR_ABS)" >&2; exit 1 ;; esac

ensure-safe-python-dist-dir: ensure-safe-build-dir
	@case "$(PYTHON_DIST_DIR_ABS)" in "$(BUILD_DIR_ABS)"/*) ;; *) echo "refusing unsafe PYTHON_DIST_DIR: $(PYTHON_DIST_DIR_ABS)" >&2; exit 1 ;; esac

ensure-safe-python-runtime-dir: ensure-safe-build-dir
	@case "$(PYTHON_RUNTIME_DIR_ABS)" in "$(BUILD_DIR_ABS)"/*) ;; *) echo "refusing unsafe PYTHON_RUNTIME_DIR: $(PYTHON_RUNTIME_DIR_ABS)" >&2; exit 1 ;; esac

native-graph: ensure-safe-native-build-dir
	mkdir -p -- "$(NATIVE_BUILD_DIR_ABS)"
	$(CXX) -std=c++17 -O2 -fPIC -Wall -Wextra -Werror \
		-I"$(CUDA_HOME)/include" -shared native/coldsnap_graph_memory.cpp \
		-Wl,--no-undefined -Wl,-soname,libcoldsnap_graph_memory.so \
		-L"$(CUDA_HOME)/lib64" -Wl,-rpath,"$(CUDA_HOME)/lib64" -lcudart \
		-L"$(CUDA_DRIVER_LIB_DIR)" -lcuda -ldl -pthread \
		-o "$(GRAPH_MEMORY_LIBRARY)"

native-hydration: ensure-safe-native-build-dir
	mkdir -p -- "$(NATIVE_BUILD_DIR_ABS)"
	$(CXX) -std=c++17 -O2 -fPIC -Wall -Wextra -Werror \
		-I"$(CUDA_HOME)/include" -shared native/coldsnap_hydration.cpp \
		-Wl,--no-undefined -Wl,-soname,libcoldsnap_hydration.so \
		-L"$(CUDA_HOME)/lib64" -Wl,-rpath,"$(CUDA_HOME)/lib64" -lcudart \
		-lcrypto -lz -ldl -pthread -o "$(HYDRATION_LIBRARY)"

native-nccl-dlsym: ensure-safe-native-build-dir
	mkdir -p -- "$(NATIVE_BUILD_DIR_ABS)"
	$(CC) -shared -fPIC -O2 -Wall -Wextra -Werror \
		native/coldsnap_nccl_dlsym.c -ldl -o "$(NCCL_DLSYM_LIBRARY)"

native-cuda-epoch: ensure-safe-native-build-dir
	mkdir -p -- "$(NATIVE_BUILD_DIR_ABS)"
	$(CXX) -std=c++17 -O2 -fPIC -Wall -Wextra -Werror \
		-I"$(CUDA_HOME)/include" -shared native/coldsnap_cuda_epoch.cpp \
		-Wl,--no-undefined -Wl,-soname,libcoldsnap_cuda_epoch.so \
		-L"$(CUDA_HOME)/lib64" -Wl,-rpath,"$(CUDA_HOME)/lib64" -lcudart \
		-L"$(CUDA_DRIVER_LIB_DIR)" -lcuda -ldl -pthread \
		-o "$(CUDA_EPOCH_LIBRARY)"

criu-nvidia-mmap-shim: ensure-safe-native-build-dir
	mkdir -p -- "$(NATIVE_BUILD_DIR_ABS)"
	$(CC) -std=c11 -O2 -fPIC -Wall -Wextra -Werror -shared \
		native/coldsnap_nvidia_mmap_shim.c \
		-Wl,--no-undefined -Wl,-soname,libcoldsnap_nvidia_mmap_shim.so \
		-ldl -pthread -o "$(CRIU_NVIDIA_MMAP_SHIM)"

nvml-dlopen-shim: ensure-safe-native-build-dir
	mkdir -p -- "$(NATIVE_BUILD_DIR_ABS)"
	$(CC) -std=c11 -O2 -fPIC -Wall -Wextra -Werror -shared \
		native/coldsnap_nvml_dlopen_shim.c \
		-Wl,--no-undefined -Wl,-soname,libcoldsnap_nvml_dlopen_shim.so \
		-ldl -o "$(NVML_DLOPEN_SHIM)"

native: native-graph native-hydration native-cuda-epoch native-nccl-dlsym criu-nvidia-mmap-shim nvml-dlopen-shim

go-criu-source: ensure-safe-build-dir
	@set -eu; \
		source="$(GO_CRIU_FORK_ROOT)"; \
		case "$$source" in "$(BUILD_DIR_ABS)"/*) ;; *) echo "refusing unsafe GO_CRIU_FORK_ROOT: $$source" >&2; exit 1 ;; esac; \
		if [ ! -d "$$source/.git" ]; then \
			test ! -e "$$source" || { echo "go-criu source path exists but is not a Git checkout: $$source" >&2; exit 1; }; \
		mkdir -p -- "$$(dirname "$$source")"; \
		git init -q "$$source"; \
	fi; \
	if ! git -C "$$source" remote get-url origin >/dev/null 2>&1; then \
		git -C "$$source" remote add origin "$(GO_CRIU_SOURCE_URL)"; \
	fi; \
	test "$$(git -C "$$source" remote get-url origin)" = "$(GO_CRIU_SOURCE_URL)" || { echo "go-criu source remote is not $(GO_CRIU_SOURCE_URL)" >&2; exit 1; }; \
	if ! git -C "$$source" rev-parse --verify HEAD >/dev/null 2>&1; then \
		test -z "$$(git -C "$$source" ls-files --others --exclude-standard)" || { echo "incomplete go-criu checkout contains untracked files: $$source" >&2; exit 1; }; \
		git -C "$$source" fetch -q --depth=1 origin "$(GO_CRIU_SOURCE_REVISION)"; \
		git -C "$$source" checkout -q --detach FETCH_HEAD; \
	fi; \
	test "$$(git -C "$$source" rev-parse HEAD)" = "$(GO_CRIU_SOURCE_REVISION)" || { echo "go-criu source revision is not $(GO_CRIU_SOURCE_REVISION)" >&2; exit 1; }; \
		git -C "$$source" diff --quiet; \
		git -C "$$source" diff --cached --quiet; \
		test -z "$$(git -C "$$source" ls-files --others --exclude-standard)"

ensure-criu-rpc-inputs: go-criu-source
	test -f "$(GO_CRIU_FORK_ROOT)/go.mod"
	test -f "$(GO_CRIU_FORK_ROOT)/notify.go"
	test -f "$(GO_CRIU_FORK_ROOT)/rpc/rpc.pb.go"
	test -f "cmd/coldsnap-criu-rpc/main.go"
	test -f "cmd/coldsnap-criu-rpc/internal/criurpc/runner.go"
	@test -n "$(CUDA_CHECKPOINT_SOURCE_ROOT)" || { echo "CUDA_CHECKPOINT_SOURCE_ROOT is required" >&2; exit 2; }
	@test -f "$(CUDA_CHECKPOINT_SOURCE_ROOT)/bin/aarch64_Linux/cuda-checkpoint" || test -f "$(CUDA_CHECKPOINT_SOURCE_ROOT)/bin/x86_64_Linux/cuda-checkpoint"

criu-rpc-go-test: go-criu-source
	cd cmd/coldsnap-criu-rpc && go test ./...

vllm-cuda-criu-image: ensure-criu-rpc-inputs
	@case "$(CRIU_IMAGE)" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) echo "CRIU_IMAGE must be digest-pinned" >&2; exit 2 ;; esac
	$(DOCKER) build \
		--file deploy/vllm/Dockerfile.criu-rpc \
		--build-arg "CRIU_IMAGE=$(CRIU_IMAGE)" \
		--build-context "criu_image=docker-image://$(CRIU_IMAGE)" \
		--build-context "go_criu_source=$(GO_CRIU_FORK_ROOT)" \
		--build-context "cuda_checkpoint_source=$(CUDA_CHECKPOINT_SOURCE_ROOT)" \
		--tag "$(VLLM_CUDA_CRIU_IMAGE)" \
		.

nccl-provider-target: ensure-safe-build-dir
	@case "$(NCCL_PROVIDER_BASE_IMAGE)" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) echo "NCCL_PROVIDER_BASE_IMAGE must be digest-pinned" >&2; exit 2 ;; esac
	@if [ "$(NCCL_PROVIDER_TARGET_OBSERVE)" = "1" ]; then \
		mkdir -p -- "$(dir $(NCCL_PROVIDER_TARGET))"; \
		temporary="$(NCCL_PROVIDER_TARGET).tmp"; \
		base_image="$(NCCL_PROVIDER_BASE_IMAGE)"; \
		$(DOCKER) run --rm --gpus all --entrypoint python3 \
			-v "$(CURDIR)/scripts/nccl-target-observe.py:/opt/coldsnap/nccl-target-observe.py:ro" \
			"$(NCCL_PROVIDER_BASE_IMAGE)" /opt/coldsnap/nccl-target-observe.py \
			--base-image-digest "$${base_image##*@}" \
			--transport "$(NCCL_PROVIDER_TRANSPORT)" >"$$temporary"; \
		mv -- "$$temporary" "$(NCCL_PROVIDER_TARGET)"; \
	else \
		test -f "$(NCCL_PROVIDER_TARGET)"; \
	fi

nccl-provider-materialize: nccl-provider-target
	@test -n "$(NCCL_PROVIDER_CATALOG)" || { echo "NCCL_PROVIDER_CATALOG is required" >&2; exit 2; }
	go run ./cmd/coldsnap nccl-provider materialize \
		--catalog "$(NCCL_PROVIDER_CATALOG)" \
		--target "$(NCCL_PROVIDER_TARGET)" \
		--output "$(NCCL_PROVIDER_BUILD_DIR)"

vllm-runtime-image: ensure-criu-rpc-inputs nccl-provider-materialize
	@case "$(VLLM_BASE_IMAGE)" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) echo "VLLM_BASE_IMAGE must be digest-pinned" >&2; exit 2 ;; esac
	@case "$(CRIU_IMAGE)" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) echo "CRIU_IMAGE must be digest-pinned" >&2; exit 2 ;; esac
	$(DOCKER) build \
		--file deploy/vllm/Dockerfile \
		--build-arg "VLLM_IMAGE=$(VLLM_BASE_IMAGE)" \
		--build-arg "CRIU_IMAGE=$(CRIU_IMAGE)" \
		--build-context "criu_image=docker-image://$(CRIU_IMAGE)" \
		--build-context "go_criu_source=$(GO_CRIU_FORK_ROOT)" \
		--build-context "cuda_checkpoint_source=$(CUDA_CHECKPOINT_SOURCE_ROOT)" \
		--build-context "nccl_provider=$(NCCL_PROVIDER_BUILD_DIR)" \
		--tag "$(VLLM_RUNTIME_IMAGE)" \
		.

sglang-runtime-image: NCCL_PROVIDER_BASE_IMAGE = $(SGLANG_BASE_IMAGE)
sglang-runtime-image: ensure-criu-rpc-inputs nccl-provider-materialize
	@case "$(SGLANG_BASE_IMAGE)" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) echo "SGLANG_BASE_IMAGE must be digest-pinned" >&2; exit 2 ;; esac
	@case "$(CRIU_IMAGE)" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) echo "CRIU_IMAGE must be digest-pinned" >&2; exit 2 ;; esac
	$(DOCKER) build \
		--file deploy/sglang/Dockerfile \
		--build-arg "SGLANG_IMAGE=$(SGLANG_BASE_IMAGE)" \
		--build-arg "CRIU_IMAGE=$(CRIU_IMAGE)" \
		--build-context "criu_image=docker-image://$(CRIU_IMAGE)" \
		--build-context "go_criu_source=$(GO_CRIU_FORK_ROOT)" \
		--build-context "cuda_checkpoint_source=$(CUDA_CHECKPOINT_SOURCE_ROOT)" \
		--build-context "nccl_provider=$(NCCL_PROVIDER_BUILD_DIR)" \
		--tag "$(SGLANG_RUNTIME_IMAGE)" \
		.

criu-rpc-runtime: vllm-cuda-criu-image ensure-safe-build-dir
	mkdir -p -- "$(BUILD_DIR_ABS)"
	@set -eu; container_id="$$($(DOCKER) create "$(VLLM_CUDA_CRIU_IMAGE)")"; \
	trap '$(DOCKER) rm -f "$$container_id" >/dev/null' EXIT; \
	$(DOCKER) cp "$$container_id:/opt/criu/criu/criu" "$(BUILD_DIR_ABS)/criu-rpc"; \
	$(DOCKER) cp "$$container_id:/opt/criu/liblz4.so.1" "$(BUILD_DIR_ABS)/liblz4.so.1"; \
	$(DOCKER) cp "$$container_id:/usr/local/bin/coldsnap-criu-rpc" "$(BUILD_DIR_ABS)/coldsnap-criu-rpc"; \
	chmod 0755 "$(BUILD_DIR_ABS)/criu-rpc" "$(BUILD_DIR_ABS)/coldsnap-criu-rpc"; \
	chmod 0644 "$(BUILD_DIR_ABS)/liblz4.so.1"

build: ensure-safe-bin-dir
	mkdir -p -- "$(BIN_DIR_ABS)"
	go build $(GO_BUILD_FLAGS) -o "$(BIN_DIR_ABS)/coldsnap" ./cmd/coldsnap
	go build $(GO_BUILD_FLAGS) -o "$(BIN_DIR_ABS)/coldsnap-vllm-adapter" ./cmd/coldsnap-vllm-adapter
	go build $(GO_BUILD_FLAGS) -o "$(BIN_DIR_ABS)/coldsnap-sglang-adapter" ./cmd/coldsnap-sglang-adapter
	CGO_ENABLED=0 go build $(GO_BUILD_FLAGS) -o "$(BIN_DIR_ABS)/coldsnap-coordinator" ./cmd/coldsnap-coordinator

reproducible-build: build
	@first_hashes="$$(sha256sum "$(BIN_DIR_ABS)/coldsnap" "$(BIN_DIR_ABS)/coldsnap-vllm-adapter" "$(BIN_DIR_ABS)/coldsnap-sglang-adapter" "$(BIN_DIR_ABS)/coldsnap-coordinator")"; \
	$(MAKE) --no-print-directory build >/dev/null; \
	second_hashes="$$(sha256sum "$(BIN_DIR_ABS)/coldsnap" "$(BIN_DIR_ABS)/coldsnap-vllm-adapter" "$(BIN_DIR_ABS)/coldsnap-sglang-adapter" "$(BIN_DIR_ABS)/coldsnap-coordinator")"; \
	if [ "$$first_hashes" != "$$second_hashes" ]; then echo "binary builds are not reproducible" >&2; exit 1; fi; \
	printf '%s\n' "$$second_hashes"

python-test:
	$(PYTHON) -m unittest discover -s test -p 'test_*.py' -v

python-package-build: ensure-safe-python-dist-dir
	rm -rf -- "$(PYTHON_DIST_DIR_ABS)"
	mkdir -p -- "$(PYTHON_DIST_DIR_ABS)"
	@set -eu; trap 'find integrations -type d -name '\''*.egg-info'\'' -prune -exec rm -rf -- {} +' EXIT; \
		$(UV) build --all-packages --out-dir "$(PYTHON_DIST_DIR_ABS)"

python-runtime-metadata: ensure-safe-python-runtime-dir
	rm -rf -- "$(PYTHON_RUNTIME_DIR_ABS)"
	mkdir -p -- "$(PYTHON_RUNTIME_DIR_ABS)"
	$(PYTHON) integrations/vllm/render_runtime_metadata.py \
		--project integrations/vllm/pyproject.toml \
		--output "$(PYTHON_RUNTIME_DIR_ABS)"
	$(PYTHON) integrations/vllm/render_runtime_metadata.py \
		--project integrations/sglang/pyproject.toml \
		--output "$(PYTHON_RUNTIME_DIR_ABS)"

vllm-contract:
	@test -n "$(VLLM_SOURCE_ROOT)" || { echo "VLLM_SOURCE_ROOT is required" >&2; exit 2; }
	VLLM_SOURCE_ROOT="$(VLLM_SOURCE_ROOT)" $(PYTHON) -m unittest discover -s test -p 'test_vllm_contract.py' -v

sglang-contract:
	@test -n "$(SGLANG_SOURCE_ROOT)" || { echo "SGLANG_SOURCE_ROOT is required" >&2; exit 2; }
	SGLANG_SOURCE_ROOT="$(SGLANG_SOURCE_ROOT)" $(PYTHON) -m unittest discover -s test -p 'test_sglang_plugin.py' -v

go-test:
	go test ./...

test: python-test go-test criu-rpc-go-test

python-lint:
	$(RUFF) check .

python-lock-check:
	$(UV) lock --check

static-analysis:
	go vet ./...
	$(STATICCHECK) ./...

repository-layout:
	$(PYTHON) scripts/check-repository-layout.py

license-headers:
	$(PYTHON) scripts/check-license-headers.py

documentation-links:
	$(PYTHON) scripts/check-markdown-links.py

lint: python-lint python-lock-check repository-layout license-headers documentation-links static-analysis
	$(ACTIONLINT)
	$(PYTHON) -m compileall -q $(PYTHON_PRODUCT_SOURCES) $(PYTHON_BENCHMARK_SOURCES)

race:
	go test -race ./...

python-security:
	$(BANDIT) -r $(PYTHON_PRODUCT_SOURCES) -ll -ii

vuln: python-security
	$(GOVULNCHECK) ./...

secret-scan:
	$(GITLEAKS) dir --no-banner --redact .

check: test lint race vuln secret-scan

clean: ensure-safe-bin-dir ensure-safe-build-dir ensure-safe-dist-dir ensure-safe-native-build-dir
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf -- "$(BIN_DIR_ABS)"
	rm -rf -- "$(BUILD_DIR_ABS)"
	rm -rf -- "$(DIST_DIR_ABS)"
	@case "$(NATIVE_BUILD_DIR_ABS)" in "$(BUILD_DIR_ABS)"|"$(BUILD_DIR_ABS)"/*) ;; *) rm -rf -- "$(NATIVE_BUILD_DIR_ABS)" ;; esac

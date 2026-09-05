<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Maintained benchmark harnesses

These tools exercise current runtime interfaces. GPU checks require a
compatible engine image with the ColdSnap integration and native helpers;
CPU-only unit tests do not establish hardware qualification. Run GPU and CRIU
checks in disposable test environments with explicit resource limits.

| Harness | Purpose and prerequisites |
| --- | --- |
| `sparkrun_restore_ttft.py` | Observe Docker start through the first non-empty streamed token and validate the final response; requires SSH/Docker access and a reachable inference endpoint. |
| `run_sparkrun_ttft_sample.sh` | Run a vanilla, recovery, or native sample through an installed Sparkrun with ColdSnap support. Stops all workloads and clears page cache on the explicitly selected test cluster. |
| `run_driver_ttft_matrix.sh` | Run three samples per vanilla/recovery/native cell for Qwen and DeepSeek; requires caller-supplied recipes. |
| `hydration_backends.py` | Calibrate bounded native buffered/direct/GDS transfers; requires PyTorch, CUDA, and the built native hydration library. |
| `vllm_startup_matrix.py` | Compare fresh-process vLLM startup cases and HTTP streaming readiness; requires a compatible vLLM installation and accessible model. |
| `qwen_live_cycle.py` | Check exact generation parity and graph-memory release/resume through vLLM sleep/wake; accepts a model override and requires the enabled ColdSnap memory backend and native hydrator. |
| `vllm_process_checkpoint.py` | Exercise vLLM checkpoint hooks around a CUDA process round trip using the current shared checkpoint helper. |
| `nccl_cuda_checkpoint_target.py` | Generic rank/generation-specific collective and CUDA-graph parity target. |
| `nccl_cuda_criu_node.py` | Coordinate one rank of the generic NCCL CUDA/CRIU round trip; requires explicit coordinator, artifact, checkpoint, and provider inputs. |
| `nccl_dlsym_interposer_smoke.py` | Verify current dlsym bridge routing; requires the bridge preloaded and `COLDSNAP_NCCL_REAL_LIBRARY_PATH`, `COLDSNAP_NCCL_DLSYM_BRIDGE_PATH`, and provider settings. |
| `nccl_instanttensor_unwrap_smoke.py` | Exercise provider communicator unwrap with a local PyTorch NCCL process group and qualified NCCL runtime. |

## Manager-driven measurements

Activate the Python environment containing Sparkrun and set `SPARKRUN_BINARY`
and `SPARKRUN_PYTHON` when they are not `sparkrun` and `python3` on `PATH`.
Alternatively, set `SPARKRUN_ROOT` to a checkout containing `.venv/bin/`.
For the authoritative plugin's `source dev.sh` setup, this is the
`sparkrun-coldsnap-plugin` checkout, not its assembled host subdirectory.
Verify `sparkrun coldsnap --version` first; see
[manager setup](../../docs/usage.md#install-and-verify-the-manager-plugin).
The Python interpreter must import the same Sparkrun installation as the CLI.
All result paths below are outside the ColdSnap checkout.

```bash
benchmarks/harnesses/run_sparkrun_ttft_sample.sh \
  test-cluster gpu-a.example /path/to/recipe.yaml recovery \
  Qwen/Qwen3.8-27B-FP8 sample-1 /path/to/private-results
```

The compact matrix form requires `COLDSNAP_SPARKRUN_RECIPES_ROOT` pointing to
your recipe checkout and selects the named snapshot driver for restores:

```bash
COLDSNAP_SPARKRUN_RECIPES_ROOT=/path/to/recipes \
benchmarks/harnesses/run_driver_ttft_matrix.sh \
  test-cluster gpu-a.example n610 /path/to/private-results
```

The seven-argument matrix form accepts four explicit recipe paths instead;
set `COLDSNAP_SNAPSHOT_DRIVER` when overriding manager driver selection.
`COLDSNAP_BINARY` can select a locally built controller. Prepare and verify
the immutable capture before measuring restore. Host and observer clocks must
be synchronized because Docker start and token observation use different hosts.

## Local diagnostics

```bash
python3 benchmarks/harnesses/sparkrun_restore_ttft.py --help
python3 benchmarks/harnesses/nccl_cuda_criu_node.py --help
python3 benchmarks/harnesses/vllm_startup_matrix.py --help
make native
python3 benchmarks/harnesses/hydration_backends.py \
  --library build/native/libcoldsnap_hydration.so \
  --bytes 536870912 --chunk-bytes 67108864 --queue-depth 2 \
  --output-json /path/to/private-results/hydration.json
```

SGLang capture/restore uses the current manager and process-snapshot adapter;
there is no standalone `COLDSNAP_MODE=restore` or `auto` plugin workflow.
See [SGLang integration](../../docs/sglang-plugin.md) and
[measurement policy](../README.md).

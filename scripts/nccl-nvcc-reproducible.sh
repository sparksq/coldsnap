#!/bin/sh
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

set -eu

real_nvcc="${COLDSNAP_REAL_NVCC:-/usr/local/cuda/bin/nvcc}"
output=""
expect_output=0

for argument in "$@"; do
    if [ "${expect_output}" -eq 1 ]; then
        output="${argument}"
        expect_output=0
        continue
    fi
    case "${argument}" in
        -o|--output-file)
            expect_output=1
            ;;
        -o?*)
            output="${argument#-o}"
            ;;
        --output-file=*)
            output="${argument#--output-file=}"
            ;;
        --frandom-seed|-frandom-seed|--frandom-seed=*|-frandom-seed=*)
            echo "nccl-nvcc-reproducible owns --frandom-seed" >&2
            exit 2
            ;;
    esac
done

if [ "${expect_output}" -eq 1 ]; then
    echo "nvcc output option is missing its path" >&2
    exit 2
fi
if [ -z "${output}" ]; then
    exec "${real_nvcc}" "$@"
fi

# CUDA-generated module identifiers otherwise incorporate process-local random
# values. The fixed container build path makes the output path a stable and
# per-translation-unit seed without assigning one seed to different files.
seed="$(printf '%s' "${output}" | sha256sum)"
seed="${seed%% *}"
exec "${real_nvcc}" "--frandom-seed=${seed}" "$@"

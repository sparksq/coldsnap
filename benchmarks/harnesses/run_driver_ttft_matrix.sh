#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

if [ "$#" -ne 4 ] && [ "$#" -ne 7 ]; then
  echo "usage: $0 <cluster> <head-host> <snapshot-driver> <output-dir>" >&2
  echo "   or: $0 <cluster> <head-host> <qwen-coldsnap-recipe> <qwen-vanilla-recipe> <deepseek-coldsnap-recipe> <deepseek-vanilla-recipe> <output-dir>" >&2
  exit 2
fi

cluster=$1
head_host=$2
if [ "$#" -eq 4 ]; then
  snapshot_driver=$3
  output_dir=$4
  recipe_root=${COLDSNAP_SPARKRUN_RECIPES_ROOT:?Set COLDSNAP_SPARKRUN_RECIPES_ROOT to your recipe checkout}
  qwen_coldsnap_recipe=$recipe_root/coldsnap-recipes/qwen3.8-27b-fp8-coldsnap-tp2-vllm.yaml
  qwen_vanilla_recipe=$recipe_root/vanilla-recipes/qwen3.8-27b-fp8-vanilla-tp2-vllm.yaml
  case "$snapshot_driver" in
    n580|n610)
      export COLDSNAP_SNAPSHOT_DRIVER="$snapshot_driver"
      deepseek_coldsnap_recipe=$recipe_root/coldsnap-recipes/deepseek-v4-flash-0731-coldsnap-tp2-vllm.yaml
      deepseek_vanilla_recipe=$recipe_root/vanilla-recipes/deepseek-v4-flash-0731-vanilla-tp2-vllm.yaml
      ;;
    *)
      echo "unsupported snapshot driver: $snapshot_driver (expected n580 or n610)" >&2
      exit 2
      ;;
  esac
else
  qwen_coldsnap_recipe=$3
  qwen_vanilla_recipe=$4
  deepseek_coldsnap_recipe=$5
  deepseek_vanilla_recipe=$6
  output_dir=$7
fi
harness_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sample_runner=$harness_dir/run_sparkrun_ttft_sample.sh

for recipe in \
  "$qwen_coldsnap_recipe" \
  "$qwen_vanilla_recipe" \
  "$deepseek_coldsnap_recipe" \
  "$deepseek_vanilla_recipe"; do
  if [ ! -f "$recipe" ]; then
    echo "recipe not found: $recipe" >&2
    exit 2
  fi
done

run_cell() {
  local model_name=$1
  local model_id=$2
  local mode=$3
  local recipe=$4
  local sample_number

  for sample_number in 1 2 3; do
    "$sample_runner" \
      "$cluster" \
      "$head_host" \
      "$recipe" \
      "$mode" \
      "$model_id" \
      "${model_name}-${cluster}-${mode}-${sample_number}" \
      "$output_dir/$model_name/$mode"
  done
}

# Run vanilla first so a ColdSnap restore cannot seed runtime caches used by
# the cold-start control. Each individual sample still clears page cache.
run_cell qwen Qwen/Qwen3.8-27B-FP8 vanilla "$qwen_vanilla_recipe"
run_cell qwen Qwen/Qwen3.8-27B-FP8 recovery "$qwen_coldsnap_recipe"
run_cell qwen Qwen/Qwen3.8-27B-FP8 native "$qwen_coldsnap_recipe"

run_cell deepseek deepseek-ai/DeepSeek-V4-Flash-0731 vanilla "$deepseek_vanilla_recipe"
run_cell deepseek deepseek-ai/DeepSeek-V4-Flash-0731 recovery "$deepseek_coldsnap_recipe"
run_cell deepseek deepseek-ai/DeepSeek-V4-Flash-0731 native "$deepseek_coldsnap_recipe"

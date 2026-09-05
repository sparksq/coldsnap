#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

set -uo pipefail

if [ "$#" -ne 7 ]; then
  echo "usage: $0 <cluster> <head-host> <recipe> <mode> <model> <sample> <output-dir>" >&2
  exit 2
fi

cluster=$1
head_host=$2
recipe=$3
mode=$4
model=$5
sample=$6
output_dir=$7

if [ -n "${SPARKRUN_ROOT:-}" ]; then
  sparkrun=${SPARKRUN_BINARY:-$SPARKRUN_ROOT/.venv/bin/sparkrun}
  sparkrun_python=${SPARKRUN_PYTHON:-$SPARKRUN_ROOT/.venv/bin/python}
else
  sparkrun=${SPARKRUN_BINARY:-sparkrun}
  sparkrun_python=${SPARKRUN_PYTHON:-python3}
fi
coldsnap=${COLDSNAP_BINARY:-}
snapshot_driver=${COLDSNAP_SNAPSHOT_DRIVER:-}
harness_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
observer=$harness_dir/sparkrun_restore_ttft.py

intent_id=$("$sparkrun_python" -c '
import sys
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.job_metadata import generate_intent_id

print(generate_intent_id(Recipe.load(sys.argv[1], resolve=False)))
' "$recipe") || exit $?
case "$mode" in
  vanilla|recovery|native) ;;
  *) echo "unsupported benchmark mode: $mode" >&2; exit 2 ;;
esac

mkdir -p "$output_dir"
result="$output_dir/${sample}.json"
observer_log="$output_dir/${sample}.observer.log"
launch_log="$output_dir/${sample}.launch.log"
runtime_log="$output_dir/${sample}.runtime.log"
ready_file="$output_dir/${sample}.observer-ready.json"
observer_pid=

cleanup() {
  if [ -n "$observer_pid" ]; then
    kill "$observer_pid" >/dev/null 2>&1 || true
    wait "$observer_pid" >/dev/null 2>&1 || true
  fi
  "$sparkrun" stop --all --cluster "$cluster" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[$sample] stop prior workloads"
"$sparkrun" stop --all --cluster "$cluster" || exit $?
echo "[$sample] clear page cache on $cluster"
"$sparkrun" setup clear-cache --cluster "$cluster" || exit $?

rm -f -- "$ready_file"
python3 "$observer" \
  --docker-host "$head_host" \
  --intent-id "$intent_id" \
  --api-base "http://$head_host:8000" \
  --model "$model" \
  --prompt "Reply with exactly: coldsnap-cuda-snapshot-ok" \
  --expected "coldsnap-cuda-snapshot-ok" \
  --timeout 1200 \
  --ready-file "$ready_file" \
  --output "$result" >"$observer_log" 2>&1 &
observer_pid=$!

for _attempt in $(seq 1 200); do
  if [ -f "$ready_file" ]; then
    break
  fi
  if ! kill -0 "$observer_pid" 2>/dev/null; then
    echo "[$sample] observer exited before launch" >&2
    tail -100 "$observer_log" >&2
    exit 1
  fi
  sleep 0.05
done
if [ ! -f "$ready_file" ]; then
  echo "[$sample] observer did not become ready" >&2
  exit 1
fi

echo "[$sample] launch $mode with observer active"
if [ "$mode" = vanilla ]; then
  "$sparkrun" run "$recipe" --cluster "$cluster" --no-follow 2>&1 | tee "$launch_log"
  launch_rc=${PIPESTATUS[0]}
else
  restore_args=(
    coldsnap restore "$recipe"
    --cluster "$cluster"
    --weights "$mode"
  )
  if [ -n "$coldsnap" ]; then
    restore_args+=(--coldsnap-binary "$coldsnap")
  fi
  if [ -n "$snapshot_driver" ]; then
    restore_args+=(--snapshot-driver "$snapshot_driver")
  fi
  if [ "$mode" = recovery ]; then
    restore_args+=(--materialize-native off)
  fi
  "$sparkrun" "${restore_args[@]}" 2>&1 | tee "$launch_log"
  launch_rc=${PIPESTATUS[0]}
fi

if [ "$launch_rc" -ne 0 ]; then
  echo "[$sample] launch failed with exit $launch_rc" >&2
  tail -100 "$launch_log" >&2
  exit "$launch_rc"
fi

wait "$observer_pid"
observer_rc=$?
observer_pid=
if [ "$observer_rc" -ne 0 ]; then
  echo "[$sample] observer failed with exit $observer_rc" >&2
  tail -100 "$observer_log" >&2
  exit "$observer_rc"
fi

# Preserve the live per-rank engine logs before cleanup removes the restored
# containers. These logs are needed to attribute Docker-to-token time to
# engine import, hydration, compilation, profiling, graph, and API warmup.
"$sparkrun" logs "$recipe" --cluster "$cluster" --all-sources >"$runtime_log" 2>&1 || true

python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); print("[%s] Docker-to-first-token %.6f s; health %.6f s" % (sys.argv[2], value["container_to_first_token_seconds"], value["container_to_health_seconds"]))' "$result" "$sample"

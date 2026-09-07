#!/bin/bash
# Driver for the adaptive-inference knob characterization.
#
#   Phase 0  calibration: unbounded offered rate, so every run reports the ceiling of its
#            configuration. Read the throughputs, then regenerate the main test with a --rates grid
#            that straddles the baseline's saturation point, and rerun phases 1-2.
#   Phase 1  main effects: one knob family at a time over the whole (rate x duplicates) plane.
#   Phase 2  interactions: full factorial over the knobs at the corner operating points.
#
# Everything is resumable: run_systests.py skips any (combination, query, repetition) that already
# has a result directory, so re-running after an interruption continues where it stopped. Cells
# that phases 1 and 2 have in common are therefore measured once.
#
# Usage:
#   scripts/benchmarking/e2e/adaptive/run_adaptive_screening.sh            # all phases
#   PHASES="0" scripts/benchmarking/e2e/adaptive/run_adaptive_screening.sh # calibration only
#   REPS=3 THREADS=16 PHASES="1 2" scripts/benchmarking/e2e/adaptive/run_adaptive_screening.sh
#
# Environment:
#   PHASES    phases to run (default "0 1 2")
#   REPS      repetitions per cell for phases 1-2 (default 2)
#   REPS_CAL  repetitions for phase 0 (default 1)
#   THREADS   worker.query_engine.number_of_worker_threads (default 8; set to the node's core count)
#   RESULTS   results directory (default scripts/benchmarking/e2e/adaptive/results)
#   SYSTEST   systest binary (default cmake-build-release/nes-systests/systest/systest)
#   DRY_RUN   set to 1 to print the systest commands without running them
#
# Cells are run with --continue-on-failure: a cell that fails all its retries leaves a
# rep-NN.FAILED marker and the sweep carries on, so one flaky crash cannot kill an overnight
# job. Re-running the script retries those cells.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"

PHASES="${PHASES:-0 1 2}"
REPS="${REPS:-2}"
REPS_CAL="${REPS_CAL:-1}"
THREADS="${THREADS:-8}"
RESULTS="${RESULTS:-$HERE/results}"
SYSTEST="${SYSTEST:-$ROOT/cmake-build-release/nes-systests/systest/systest}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$HERE/sections.env" ]]; then
  echo "sections.env not found - run gen_adaptive_workload.py first" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$HERE/sections.env"

if [[ ! -x "$SYSTEST" ]]; then
  echo "systest binary not found or not executable: $SYSTEST" >&2
  exit 1
fi

# The LatencySink opens its CSV relative to the systest working directory and does not create the
# directory itself.
mkdir -p "$(dirname "$SYSTEST")/results"
mkdir -p "$RESULTS"

CFG_DIR="$(mktemp -d)"
trap 'rm -rf "$CFG_DIR"' EXIT

# Thread count is a fixed factor of this experiment, not a swept knob, but it belongs in the
# combination label so results from differently sized nodes never merge silently.
cfg() {
  local name="$1"
  local out="$CFG_DIR/$name.yaml"
  cat "$HERE/config/$name.yaml" > "$out"
  printf '"worker.query_engine.number_of_worker_threads": [%s]\n' "$THREADS" >> "$out"
  echo "$out"
}

run() {
  local config="$1" reps="$2"
  shift 2
  local extra=(--continue-on-failure)
  [[ "$DRY_RUN" == "1" ]] && extra+=(--dry-run)
  python3 "$ROOT/scripts/benchmarking/e2e/run_systests.py" \
    --systest-path "$SYSTEST" \
    --results-dir "$RESULTS" \
    --inference-config "$config" \
    --repetitions "$reps" \
    "${extra[@]}" \
    --queries "$@"
}

phase_enabled() {
  [[ " $PHASES " == *" $1 "* ]]
}

if phase_enabled 0; then
  echo "=== Phase 0: calibration (unbounded offered rate) ==="
  # The source-only section has no inference operator, so one configuration is enough.
  run "$(cfg arm_baseline)" "$REPS_CAL" $SECTIONS_CAL_SOURCE_ONLY
  for arm in arm_baseline arm_batch arm_cache arm_batch_dedup; do
    run "$(cfg "$arm")" "$REPS_CAL" $SECTIONS_CAL_FP32 $SECTIONS_CAL_INT8
  done
fi

if phase_enabled 1; then
  echo "=== Phase 1: main effects over the (rate x duplicates) plane ==="
  # Precision is a property of the query section, so the baseline arm runs on both section sets;
  # the remaining arms characterize their knob on the fp32 model.
  run "$(cfg arm_baseline)" "$REPS" $SECTIONS_FP32 $SECTIONS_INT8
  run "$(cfg arm_batch)" "$REPS" $SECTIONS_FP32
  run "$(cfg arm_cache)" "$REPS" $SECTIONS_FP32
  run "$(cfg arm_batch_dedup)" "$REPS" $SECTIONS_FP32
fi

if phase_enabled 2; then
  echo "=== Phase 2: knob interactions at the corner operating points ==="
  run "$(cfg phase2_interactions)" "$REPS" $SECTIONS_CORNERS_FP32 $SECTIONS_CORNERS_INT8
fi

echo "=== done. Consolidate with: ==="
echo "python3 $HERE/process_adaptive_results.py --results-dir $RESULTS"

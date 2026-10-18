#!/usr/bin/env bash
# Latency-versus-offered-load sweep for one workload.
#
# Every comparison number we have so far is a single operating point, which cannot tell "this
# system is faster" apart from "both systems are mostly idle and this one's fixed overhead is
# smaller". This driver re-runs the whole three-system comparison across a ladder of ingestion
# rates, so each system gets a curve and its knee -- the load at which it stops keeping up --
# becomes visible.
#
#   ./scripts/run_systems_rate_sweep.sh [profile] [reps]
#   RATES="10 20 30 45 65 90" ./scripts/run_systems_rate_sweep.sh waymo 3
#
# The rate has to move in lockstep across all three systems. NES and TorchServe take it from the
# .test's MOCK_TUPLE_RATE -- one source, paced identically for both queries -- and Flink from its
# runner's RATE env. So each point regenerates a derived .test with the rate substituted and hands
# the same number to run_systems.sh, which forwards it to the Flink runner.
#
# Only rate-per-second workloads are supported, i.e. Waymo, where MOCK_TUPLE_RATE is directly the
# inference rate. For CWRU and MUSAN that knob is the SAMPLE rate, so raising it also shortens the
# wall-clock span the window covers; sweeping those means moving ADVANCE instead, which is a
# different experiment.
#
# Caveat to carry into the write-up: the input is a fixed 991 frames, so a higher rate means a
# shorter run (99 s at 10 fps, 11 s at 90). Below saturation latency is stationary and this is
# harmless -- every point's percentiles rest on the same 991 samples. Above saturation a backlogged
# system's latency grows with elapsed time, so those points understate the steady-state tail and
# should be read as "saturated", not as a number. The achieved-throughput column is the honest
# saturation signal: once it stops tracking the offered rate, the system is past its knee.
set -uo pipefail

PROFILE="${1:-waymo}"
REPS="${2:-3}"
RATES="${RATES:-10 20 30 45 65 90}"
# STAGES re-measures a subset of the arms, e.g. STAGES=torchserve after a change to the HTTP sink.
# Re-running the others would burn an hour reproducing numbers that cannot have moved, and would
# add run-to-run noise to them. It is passed straight to run_systems.sh, which owns the per-arm
# guards, so every arm here goes through the same path (and gets the same swap reset).
# TAG suffixes the output files, to keep a new curve beside the old one for comparison.
STAGES="${STAGES:-flink nes torchserve}"
TAG="${TAG:-}"

PROFILE_FILE="scripts/systems/${PROFILE}.env"
[[ -f "$PROFILE_FILE" ]] || { echo "No such profile: $PROFILE_FILE" >&2; exit 1; }
# shellcheck source=/dev/null
source "$PROFILE_FILE"

SWEEP_DIR=results-sweep
mkdir -p "$SWEEP_DIR"

# The derived .test sits beside the original so its relative ATTACH FILE / CREATE MODEL paths and
# the queries/<group>/ layout run_systests_lat.py expects still resolve.
SRC_TEST="queries/${TEST_FILE}"
[[ -f "$SRC_TEST" ]] || { echo "No such .test: $SRC_TEST" >&2; exit 1; }
DERIVED_REL="${TEST_FILE%.test}_rate.test"
DERIVED_TEST="queries/${DERIVED_REL}"

echo "### rate sweep: profile=$PROFILE reps=$REPS rates=[$RATES] stages=[$STAGES] tag=${TAG:-none}"
echo "### derived query: $DERIVED_TEST"

started=$(date +%s)

for rep in $(seq 1 "$REPS"); do
    for rate in $RATES; do
        echo
        echo "### rep $rep/$REPS  rate ${rate}/s  ($(date +%H:%M:%S))"

        # Rewrite only the MOCK_TUPLE_RATE literal. The backticks around SOURCE are matched with
        # "." so the pattern needs no shell quoting gymnastics.
        sed -E "s/^([[:space:]]*)[0-9]+( AS .SOURCE.\.MOCK_TUPLE_RATE)/\1${rate}\2/" \
            "$SRC_TEST" > "$DERIVED_TEST"
        if ! grep -qE "^[[:space:]]*${rate} AS .SOURCE.\.MOCK_TUPLE_RATE" "$DERIVED_TEST"; then
            echo "  !! could not set MOCK_TUPLE_RATE=${rate} in $DERIVED_TEST -- aborting" >&2
            exit 1
        fi

        STAGES="$STAGES" TEST_FILE="$DERIVED_REL" RATE="$rate" \
            ./scripts/run_systems.sh "$PROFILE" > "/tmp/sweep_${WORKLOAD}_r${rate}_rep${rep}.log" 2>&1 \
            || echo "  !! run_systems.sh failed, see /tmp/sweep_${WORKLOAD}_r${rate}_rep${rep}.log"

        for system in $STAGES; do
            src="results/${system}_${WORKLOAD}.csv"
            if [[ -f "$src" ]]; then
                mv "$src" "${SWEEP_DIR}/${system}_${WORKLOAD}_r${rate}_rep${rep}${TAG:+_$TAG}.csv"
            else
                echo "  !! missing $src"
            fi
        done
        echo "  rate $rate rep $rep done ($(( ($(date +%s) - started) / 60 )) min elapsed)"
    done
done

rm -f "$DERIVED_TEST"
echo
echo "### sweep finished after $(( ($(date +%s) - started) / 60 )) min"
ls -la "$SWEEP_DIR"

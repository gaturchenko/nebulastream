#!/usr/bin/env bash
# Sweep TorchServe's worker count on one workload, with everything else held fixed.
#
#   ./scripts/run_torchserve_workers.sh [profile] [reps] [worker counts...]
#   ./scripts/run_torchserve_workers.sh cwru 3 1 2 3
#
# Why this exists. The HTTP sink now opens one connection per NES worker thread, so the client no
# longer serialises requests, and the obvious next step was to give TorchServe as many workers as
# NES has threads. On a 4-core Pi that turned out to make things WORSE, not better: the TorchServe
# query still runs NES's worker threads for source decoding and windowing, so N workers means
# 3 + N runnable threads on 4 cores, plus one resident copy of the model per worker. A single CWRU
# pass measured the handler at 75.9 ms with 1 worker and 189.7 ms with 3, with swap in use.
#
# So the right worker count is an empirical question, not a parity argument, and this measures it.
# Only the TorchServe arm runs (STAGES=torchserve): NES and Flink cannot be affected by how many
# workers the serving baseline has, and re-running them would only add noise.
#
# Memory is recorded beside latency because it is half the finding -- a configuration that wins on
# latency while swapping is not a configuration we can report.
set -uo pipefail

PROFILE="${1:-cwru}"
REPS="${2:-3}"
shift 2 2>/dev/null || shift $# 
WORKERS=("$@")
if [[ ${#WORKERS[@]} -eq 0 ]]; then
    WORKERS=(1 2 3)
fi

# shellcheck source=/dev/null
source "scripts/systems/${PROFILE}.env"

OUT=results-workers
mkdir -p "$OUT"

echo "### TorchServe worker sweep: profile=$PROFILE reps=$REPS workers=[${WORKERS[*]}]"
started=$(date +%s)
summary="$OUT/memory_${WORKLOAD}.csv"
echo "workers,rep,peak_worker_rss_mb,min_available_mb,max_swap_used_mb" > "$summary"

for workers in "${WORKERS[@]}"; do
    for rep in $(seq 1 "$REPS"); do
        echo
        echo "### workers=$workers rep=$rep/$REPS  ($(date +%H:%M:%S))"
        memlog="/tmp/tsworkers_${WORKLOAD}_w${workers}_rep${rep}.mem"

        ( while true; do
            avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
            swap=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END {print t-f}' /proc/meminfo)
            rss=$(ps -eo rss,args 2>/dev/null | awk '/[t]s\/model_service_worker/{s+=$1} END {print s+0}')
            echo "$avail $rss $swap"
            sleep 5
          done ) > "$memlog" 2>&1 &
        sampler=$!

        STAGES=torchserve TS_WORKERS="$workers" REP_LABEL="workers${workers}_rep${rep}" \
            ./scripts/run_systems.sh "$PROFILE" \
            > "/tmp/tsworkers_${WORKLOAD}_w${workers}_rep${rep}.log" 2>&1 \
            || echo "  !! run failed, see /tmp/tsworkers_${WORKLOAD}_w${workers}_rep${rep}.log"
        kill "$sampler" 2>/dev/null

        src="results/torchserve_${WORKLOAD}.csv"
        if [[ -s "$src" ]] && [[ $(wc -l < "$src") -gt 1 ]]; then
            mv "$src" "$OUT/torchserve_${WORKLOAD}_w${workers}_rep${rep}.csv"
        else
            echo "  !! no TorchServe row produced (workers=$workers rep=$rep)"
        fi

        # Only samples taken while workers were actually resident describe the configuration.
        awk -v w="$workers" -v r="$rep" '
            $2 > 0 { if ($2 > peak) peak = $2; if (min == 0 || $1 < min) min = $1; if ($3 > swap) swap = $3 }
            END { printf "%s,%s,%.0f,%.0f,%.0f\n", w, r, peak/1024, min/1024, swap/1024 }
        ' "$memlog" >> "$summary"
        tail -1 "$summary" | sed 's/^/  memory: /'
        echo "  done ($(( ($(date +%s) - started) / 60 )) min elapsed)"
    done
done

echo
echo "### sweep finished after $(( ($(date +%s) - started) / 60 )) min"
column -s, -t "$summary"

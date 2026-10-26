#!/usr/bin/env python3
"""Per-run latency of the paced threading sweep (run_threading_lat.sh).

Reads <results-dir>/<combination>/<query>/rep-NN/latency_timings.csv (one row per result, written by the Latency sink; its
ingestion time is the send time the TCP mock prefixed to the input) and writes one row per run: the results' latency
percentiles once the run has warmed up, and the rate the source actually delivered.

The first inferences of each session are slower, so results whose input was sent in the first --warmup-s seconds are left out
of the percentiles; warmup_p95_ms keeps that span's P95 for reference. setup_margin_s checks the mock's start delay: the time
from the last pipeline's start (trace events) to the first send, negative if data arrived before the query had set up.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LATENCY_TIMINGS_FILE = "latency_timings.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="run_systests.py results directory")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--warmup-s", type=float, default=10.0, help="seconds of sending left out (default: 10)")
    return parser.parse_args()


def last_pipeline_start_us(rep_dir: Path) -> float:
    starts = [
        event["ts"]
        for trace in rep_dir.glob("trace_*.json")
        for event in json.loads(trace.read_text(encoding="utf-8"))["traceEvents"]
        if event.get("cat") == "pipeline" and event.get("ph") == "B"
    ]
    return max(starts) if starts else np.nan


def run_row(path: Path, warmup_s: float) -> dict:
    timings = pd.read_csv(path, usecols=["ingest_ts_us", "latency_us"])
    first_send_us = timings["ingest_ts_us"].min()
    seconds = ((timings["ingest_ts_us"] - first_send_us) / 1e6).to_numpy()
    latency_ms = timings["latency_us"].to_numpy() / 1000
    steady = latency_ms[seconds >= warmup_s]
    warmup = latency_ms[seconds < warmup_s]
    p50, p95, p99 = np.percentile(steady, [50, 95, 99])
    return {
        "query_name": path.parent.parent.name,
        "repetition": path.parent.name,
        "results": len(latency_ms),
        "ingest_rate_per_s": len(latency_ms) / seconds.max(),
        "steady_results": len(steady),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": steady.max(),
        "warmup_p95_ms": np.percentile(warmup, 95) if len(warmup) else np.nan,
        "setup_margin_s": (first_send_us - last_pipeline_start_us(path.parent)) / 1e6,
    }


def main() -> int:
    args = parse_args()
    paths = sorted(args.results_dir.glob(f"*/*/rep-*/{LATENCY_TIMINGS_FILE}"))
    if not paths:
        print(f"No {LATENCY_TIMINGS_FILE} under {args.results_dir}", file=sys.stderr)
        return 1
    empty = [path for path in paths if pd.read_csv(path, nrows=1).empty]
    for path in empty:
        print(f"No results in {path.parent}, left out", file=sys.stderr)
    rows = pd.DataFrame([run_row(path, args.warmup_s) for path in paths if path not in empty])
    rows.insert(2, "warmup_s", args.warmup_s)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_csv, index=False)
    print(f"{len(rows)} runs -> {args.output_csv}")
    late = rows[rows["setup_margin_s"] < 0]
    if not late.empty:
        print(f"Data arrived before set-up finished in {len(late)} runs:\n{late[['query_name', 'repetition', 'setup_margin_s']]}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

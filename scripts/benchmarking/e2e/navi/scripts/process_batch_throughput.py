#!/usr/bin/env python3
"""Consolidate the sustainable-throughput sweep (run_batch_throughput.sh) into CSVs.

Reads <results>/<combination>/nbeats_throughput_r<rate>.test_<section>/rep-NN/latency_timings.csv as run_systests_lat.py
lays them out, the driver's _grid.csv and thermal.csv, and writes:

  batch_throughput_runs.csv     one row per cell and repetition, including those that left no results: offered and
                                delivered rate, whether the run sustained the offered rate, latency, and the thermal state.
  batch_throughput_summary.csv  per batch size (empty for the window-only query): its sustainable throughput, the largest
                                offered rate it sustained, as the median over repetitions, with min and max.

A run sustains its offered rate if it completed and delivered its results at no less than --sustained-threshold of that
rate. The source backpressures a query that falls behind, which delays ingestion instead of growing the measured latency,
so the delivered rate is the criterion; a query that falls behind far enough fails once its backlog exhausts the buffer
pool and leaves no results at all.

Run from ~/navi on the Pi, or anywhere with --results-dir pointing at a copy.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from process_batch_rate import parse_combination, thermal_during

QUERY_DIR = re.compile(r"nbeats_throughput_r(\d+)\.test_(\d+)$")
SIZE_MS = 75


def run_metrics(latency_csv: Path, offered: float, warmup_s: float) -> dict:
    timings = pd.read_csv(latency_csv).sort_values("ingest_ts_us")
    first_ingest, last_receive = int(timings["ingest_ts_us"].min()), int(timings["recv_ts_us"].max())
    span_s = (last_receive - first_ingest) / 1e6
    achieved = len(timings) / span_s if span_s > 0 else float("nan")
    latency_ms = timings.loc[timings["ingest_ts_us"] >= first_ingest + warmup_s * 1e6, "latency_us"] / 1000.0
    return {
        "results": len(timings),
        "achieved_rate": achieved,
        "sustained_ratio": achieved / offered,
        "latency_ms_p50": latency_ms.quantile(0.50),
        "latency_ms_p95": latency_ms.quantile(0.95),
        "latency_ms_p99": latency_ms.quantile(0.99),
        "run_start_us": first_ingest,
        "run_end_us": last_receive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results/batch_throughput"))
    parser.add_argument("--warmup-s", type=float, default=5.0, help="Results ingested in the first seconds are dropped.")
    parser.add_argument(
        "--sustained-threshold", type=float, default=0.99, help="Delivered/offered rate from which a run sustained its rate."
    )
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions every cell should have.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to --results-dir.")
    args = parser.parse_args()

    grid = pd.read_csv(args.results_dir / "_grid.csv", dtype={"section": str})
    grid["section"] = grid["section"].str.zfill(2)
    grid["batch_size"] = grid["batch_size"].astype("Int64")
    thermal_path = args.results_dir / "thermal.csv"
    thermal = pd.read_csv(thermal_path) if thermal_path.exists() else None
    output_dir = args.output_dir or args.results_dir

    completed = {}
    for latency_csv in sorted(args.results_dir.glob("*/*/rep-*/latency_timings.csv")):
        rep_dir = latency_csv.parent
        match = QUERY_DIR.match(rep_dir.parent.name)
        if not match:
            continue
        rate, section = int(match.group(1)), match.group(2)
        # The window-only query runs under B=1's configuration but has no batch size.
        batch_size = int(parse_combination(rep_dir.parent.parent.name)["inference.batch_size"]) if section == "01" else None
        completed[(batch_size, section, rate, rep_dir.name)] = latency_csv

    runs = []
    for cell in grid.itertuples(index=False):
        batch_size = None if pd.isna(cell.batch_size) else int(cell.batch_size)
        for repetition in range(1, args.repetitions + 1):
            rep = f"rep-{repetition:02d}"
            row = {
                "query": "inference" if cell.section == "01" else "window only",
                "batch_size": batch_size,
                "offered_rate": cell.offered_rate,
                "repetition": rep,
            }
            latency_csv = completed.get((batch_size, cell.section, cell.offered_rate, rep))
            if latency_csv is None:
                row.update({"completed": False, "sustained": False})
            else:
                metrics = run_metrics(latency_csv, cell.offered_rate, args.warmup_s)
                row.update(
                    {
                        "completed": True,
                        **metrics,
                        "sustained": metrics["sustained_ratio"] >= args.sustained_threshold,
                        **thermal_during(thermal, metrics["run_start_us"], metrics["run_end_us"]),
                    }
                )
            runs.append(row)

    runs_df = pd.DataFrame(runs)
    runs_df["batch_size"] = runs_df["batch_size"].astype("Int64")
    runs_df = runs_df.sort_values(["query", "batch_size", "offered_rate", "repetition"])
    runs_df.to_csv(output_dir / "batch_throughput_runs.csv", index=False)

    # A repetition's sustainable throughput is the largest offered rate it sustained. Repetitions run interleaved over the
    # whole ladder, so each one yields its own estimate.
    per_rep = (
        runs_df[runs_df["sustained"]]
        .groupby(["query", "batch_size", "repetition"], dropna=False)["offered_rate"]
        .max()
        .rename("sustainable_rate")
        .reset_index()
    )
    ladder = runs_df.groupby(["query", "batch_size"], dropna=False)["offered_rate"].agg(["min", "max"])
    summary = (
        per_rep.groupby(["query", "batch_size"], dropna=False)["sustainable_rate"]
        .agg(sustainable_rate="median", sustainable_rate_min="min", sustainable_rate_max="max", repetitions="count")
        .join(ladder.rename(columns={"min": "ladder_min", "max": "ladder_max"}), how="right")
        .reset_index()
    )
    summary.to_csv(output_dir / "batch_throughput_summary.csv", index=False)

    # Console overview: per batch size and rate, how many repetitions sustained it ('-' none completed).
    def cell(values: pd.DataFrame) -> str:
        if not values["completed"].any():
            return "-"
        return f"{int(values['sustained'].sum())}/{len(values)}"

    table = (
        runs_df.assign(row=runs_df["batch_size"].astype("string").fillna("window"))
        .groupby(["row", "offered_rate"])
        .apply(cell, include_groups=False)
        .unstack("offered_rate")
        .fillna("")
    )
    print(f"Repetitions that sustained the offered rate [results/s] (delivered >= {args.sustained_threshold:.0%}; - none completed)")
    print(table.to_string())
    print()
    print(summary.to_string(index=False))
    if "throttled_during_run" in runs_df and runs_df["throttled_during_run"].fillna(False).astype(bool).any():
        print("\nWARNING: some runs were throttled; see throttled_during_run in batch_throughput_runs.csv")
    # The ladder must bracket the sustainable throughput: its top rung must fail and some rung must be sustained.
    for row in summary.itertuples(index=False):
        if pd.isna(row.sustainable_rate):
            print(f"NOTE: {row.query} B={row.batch_size} sustained no rate; extend its ladder downwards")
        elif row.sustainable_rate_max == row.ladder_max:
            print(f"NOTE: {row.query} B={row.batch_size} sustained the top of its ladder; extend it upwards")
    print(f"\nWrote {len(runs_df)} runs to {output_dir}")


if __name__ == "__main__":
    main()

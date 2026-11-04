#!/usr/bin/env python3
"""Summarize the post-join pilot: would OPTIMIZER_CHOOSES ever pick SEQUENTIAL?

Reads the profile_*.csv files of run_post_join_pilot.sh and, per model x configuration x k, computes the speedup the optimizer
compares against post_join_inference_minimum_speedup: k * t(1) / t(k), with t the median invocation latency of one profiling
job. Both latencies come from the same repetition (one profiling job, as the optimizer would run it), so the spread over
repetitions is the spread of the decision input the optimizer would see on a given deployment.

Writes post_join_pilot_summary.csv (one row per model x configuration x k) next to the inputs and prints the verdicts:
  BATCHED / SEQUENTIAL  every repetition lands on the same side of the threshold
  UNSTABLE              repetitions straddle the threshold, so the choice depends on the profiling run

    ./summarize_post_join_pilot.py results/post_join_pilot [--threshold 1.1]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

CONFIG = ["model", "openvino_inference_num_threads", "openvino_num_streams", "dynamic_batch"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=1.1, help="post_join_inference_minimum_speedup (default 1.1)")
    args = parser.parse_args()

    files = sorted(args.results_dir.glob("profile_*.csv"))
    if not files:
        sys.exit(f"No profile_*.csv in {args.results_dir}")
    profiles = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    single = profiles[profiles["batch_size"] == 1][CONFIG + ["repetition", "median_ns"]].rename(columns={"median_ns": "t1_ns"})
    batched = profiles[profiles["batch_size"] > 1].merge(single, on=CONFIG + ["repetition"], how="inner")
    batched["speedup"] = batched["batch_size"] * batched["t1_ns"] / batched["median_ns"]

    summary = (
        batched.groupby(CONFIG + ["batch_size"])
        .agg(
            repetitions=("repetition", "nunique"),
            t1_ms=("t1_ns", lambda s: s.median() / 1e6),
            tk_ms=("median_ns", lambda s: s.median() / 1e6),
            speedup=("speedup", "median"),
            speedup_min=("speedup", "min"),
            speedup_max=("speedup", "max"),
        )
        .reset_index()
        .rename(columns={"batch_size": "k"})
    )
    summary["sequential_ms"] = summary["k"] * summary["t1_ms"]
    summary["verdict"] = "UNSTABLE"
    summary.loc[summary["speedup_min"] >= args.threshold, "verdict"] = "BATCHED"
    summary.loc[summary["speedup_max"] < args.threshold, "verdict"] = "SEQUENTIAL"

    output = args.results_dir / "post_join_pilot_summary.csv"
    summary.to_csv(output, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    columns = CONFIG + ["k", "t1_ms", "tk_ms", "sequential_ms", "speedup", "speedup_min", "speedup_max", "verdict"]
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nThreshold {args.threshold}: " + ", ".join(f"{v} {n}" for v, n in summary["verdict"].value_counts().items()))
    crossover = summary[summary["verdict"] != "BATCHED"]
    if crossover.empty:
        print("No configuration falls below the threshold: the post-join choice is always BATCHED on this host.")
    else:
        print("Configurations where the choice is not BATCHED (candidates for the end-to-end crossover experiment):")
        print(crossover[CONFIG + ["k", "speedup", "verdict"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay u_max over all SARAD slides, including the knee slides, from the aggregates of process_sensitivity.py.

The knee cells run fixed batch sizes only, so slides of 16-19 ms have no profile of their own and replay the deployments at 20 and
25 ms, whose candidates are the same. The other slides replay their own deployments, like sarad_sweep.csv.

Writes knee_sweep.csv with, per (u_max, slide): the replayed choices, the batch size with the lowest measured P99 among those that
sustain the rate, and whether the choices include a batch size that falls behind.

    ./replay_knee.py --data-dir ../../figures/data/sensitivity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from selection_rules import candidate_batch_sizes, select_batch_size

WINDOW_MS = 200.0
PROFILED_SLIDES = [20.0, 25.0, 40.0, 50.0]
KNEE_SLIDES = [16.0, 17.0, 18.0, 19.0]
KNEE_PROFILES_FROM = [20.0, 25.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.data_dir / "sensitivity_summary.csv")
    fixed = summary[summary["name"].isin(["sarad", "sarad_knee"]) & (summary["strategy"] == "FIXED")]
    measured = fixed.set_index(["slide_ms", "batch_size"])
    profiles = pd.read_csv(args.data_dir / "sensitivity_profiles.csv")
    profiles = profiles[profiles["name"] == "sarad"]
    keys = ["slide_ms", "max_utilization_run", "repetition"]
    deployments = {key: dict(zip(cells["batch_size"], cells["invocation_us"] / 1000.0)) for key, cells in profiles.groupby(keys)}

    rows = []
    for max_utilization in np.round(np.arange(0.70, 1.1001, 0.01), 2):
        for slide in KNEE_SLIDES + PROFILED_SLIDES:
            sources = KNEE_PROFILES_FROM if slide in KNEE_SLIDES else [slide]
            allowed = candidate_batch_sizes(WINDOW_MS, slide)
            choices = {
                select_batch_size({batch: time for batch, time in invocation.items() if batch in allowed}, WINDOW_MS, slide, float(max_utilization)).batch_size
                for key, invocation in deployments.items()
                if key[0] in sources
            }
            options = measured.loc[slide]
            best = int(options.loc[options["sustained"], "latency_ms_p99"].idxmin())
            rows.append(
                {
                    "max_utilization": max_utilization,
                    "slide_ms": slide,
                    "selected": " ".join(str(batch) for batch in sorted(choices)),
                    "best_measured": best,
                    "all_best": choices == {best},
                    "falls_behind": any(not options.loc[batch, "sustained"] for batch in choices),
                    "worst_p99_ms": max(options.loc[batch, "latency_ms_p99"] for batch in choices),
                }
            )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(args.data_dir / "knee_sweep.csv", index=False)

    verdict = sweep.groupby("max_utilization").agg(all_best=("all_best", "all"), falls_behind=("falls_behind", "any"))
    print("Selected batch sizes at the change points:")
    pivot = sweep.pivot(index="max_utilization", columns="slide_ms", values="selected")
    print(pivot[(pivot != pivot.shift()).any(axis=1)].to_string())
    print(f"\nLowest measured P99 on every slide at u_max: {list(verdict.index[verdict['all_best']])}")
    print(f"No choice falls behind up to u_max {verdict.index[~verdict['falls_behind']].max()}")


if __name__ == "__main__":
    main()

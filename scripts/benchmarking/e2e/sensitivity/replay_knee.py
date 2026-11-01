#!/usr/bin/env python3
"""Replay the batch size selection of SARAD over all slides, including the knee slides, from the aggregates of process_sensitivity.py.

The knee cells run fixed batch sizes only, so slides of 16-19 ms have no profile of their own and replay the deployments at 20 and
25 ms, whose candidates are the same. The other slides replay their own deployments, like sarad_sweep.csv.

Writes knee_sweep.csv with, per (parameter, value, slide): the replayed choices, the batch size with the lowest measured P99 among
those that sustain the rate, whether the choices include a batch size that falls behind, and whether they are safe, i.e., none
falls behind while another sustains the rate, and none misses the latency bound while another is feasible. The parameters not
being swept stay at their defaults. Prints the ranges of Table V.

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
DEFAULTS = {"max_utilization": 0.85, "slo_fraction": 0.9, "error": 1.0}
GRIDS = {
    "max_utilization": np.round(np.arange(0.40, 1.1001, 0.01), 2),
    "slo_fraction": np.round(np.arange(0.40, 1.0001, 0.01), 2),
    "error": np.unique(np.round(np.concatenate([np.exp(np.linspace(np.log(0.25), np.log(4.0), 121)), np.arange(0.8, 1.2001, 0.005)]), 3)),
}


def replay(deployments: dict, slide: float, parameter: str, value: float) -> set[int]:
    settings = {**DEFAULTS, parameter: value}
    sources = KNEE_PROFILES_FROM if slide in KNEE_SLIDES else [slide]
    allowed = candidate_batch_sizes(WINDOW_MS, slide)
    return {
        select_batch_size(
            {batch: settings["error"] * time for batch, time in invocation.items() if batch in allowed},
            WINDOW_MS, slide, settings["max_utilization"], settings["slo_fraction"],
        ).batch_size
        for key, invocation in deployments.items()
        if key[0] in sources
    }


def table_ranges(sweep: pd.DataFrame) -> pd.DataFrame:
    """The range around the default in which the choices stay the same, and the one in which they are safe."""
    rows = []
    for parameter, default in DEFAULTS.items():
        pivot = sweep[sweep["parameter"] == parameter].pivot(index="value", columns="slide_ms", values="selected")
        safe = sweep[sweep["parameter"] == parameter].groupby("value")["safe"].all()
        base = pivot.loc[default]
        unchanged = (pivot == base).all(axis=1)
        rows.append({"parameter": parameter, "default": default,
                     "unchanged_from": unchanged[unchanged].index.min(), "unchanged_to": unchanged[unchanged].index.max(),
                     "safe_from": safe[safe].index.min(), "safe_to": safe[safe].index.max()})
    return pd.DataFrame(rows)


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
    for parameter, grid in GRIDS.items():
        for value in grid:
            for slide in KNEE_SLIDES + PROFILED_SLIDES:
                choices = replay(deployments, slide, parameter, float(value))
                options = measured.loc[slide]
                falls_behind = any(not options.loc[batch, "sustained"] for batch in choices)
                misses = options["feasible"].any() and not all(options.loc[batch, "feasible"] for batch in choices)
                rows.append(
                    {
                        "parameter": parameter,
                        "value": float(value),
                        "slide_ms": slide,
                        "selected": " ".join(str(batch) for batch in sorted(choices)),
                        "best_measured": int(options.loc[options["sustained"], "latency_ms_p99"].idxmin()),
                        "falls_behind": falls_behind,
                        "safe": not falls_behind and not misses,
                        "any_feasible": bool(options["feasible"].any()),
                        "worst_p99_ms": max(options.loc[batch, "latency_ms_p99"] for batch in choices),
                    }
                )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(args.data_dir / "knee_sweep.csv", index=False)
    print(table_ranges(sweep).to_string(index=False))

    utilization = sweep[sweep["parameter"] == "max_utilization"]
    best = utilization.assign(best=utilization["selected"] == utilization["best_measured"].astype(str)).groupby("value")["best"].all()
    print(f"\nLowest measured P99 on every slide at u_max: {list(best.index[best])}")


if __name__ == "__main__":
    main()

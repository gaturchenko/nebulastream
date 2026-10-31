#!/usr/bin/env python3
"""Sensitivity part D: replay the selection's decisions over repeated profiles (run_profile_stability.sh).

Every repetition is one profile, as one deployment would obtain, so the share of repetitions that disagree with the usual decision
is how often a deployment would decide differently. The reference is the most common decision at 100 invocations on an idle worker.

Writes:
  profile_stability_spread.csv     per (model, iterations, busy cores, batch size): t(B) median and its spread across repetitions.
  profile_stability_decisions.csv  per (decision, iterations, busy cores): the decisions and the share that differ from the reference.

    ./process_profile_stability.py results/sensitivity/profile --output-prefix results/sensitivity/pi_
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from selection_rules import candidate_batch_sizes, select_batch_size, select_post_join

NAME = re.compile(r"(?P<model>.+)_i(?P<iterations>\d+)_l(?P<load>\d+)_rep(?P<rep>\d+)\.csv$")
# Decisions per model: the batch-rate query, the SARAD query of part B, and post-join batching at k payloads.
BATCH_DECISIONS = {"nbeats": (75.0, [40, 20, 10, 5, 3, 2, 1]), "sarad": (200.0, [20, 25, 40, 50])}
POST_JOIN_DECISIONS = {"mhatt_rnn": [2, 4], "mhatt_rnn_static": [2], "squeezedet": [2, 4, 8]}


def load(directory: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("*_i*_l*_rep*.csv")):
        match = NAME.match(path.name)
        frame = pd.read_csv(path)[["batch_size", "median_ns"]]
        frames.append(frame.assign(model=match["model"], iterations=int(match["iterations"]), load=int(match["load"]),
                                   rep=int(match["rep"]), latency_ms=frame["median_ns"] / 1e6))
    if not frames:
        raise SystemExit(f"No profiles in {directory}")
    return pd.concat(frames, ignore_index=True)


def decide(model: str, invocation: dict[int, float]) -> dict[str, str]:
    decisions = {}
    if model in BATCH_DECISIONS:
        window, slides = BATCH_DECISIONS[model]
        for slide in slides:
            candidates = {batch: invocation[batch] for batch in candidate_batch_sizes(window, slide) if batch in invocation}
            decisions[f"{model} W={window:g} S={slide}"] = str(select_batch_size(candidates, window, slide).batch_size)
    for k in POST_JOIN_DECISIONS.get(model, []):
        if 1 in invocation and k in invocation:
            speedup = k * invocation[1] / invocation[k]
            decisions[f"{model} post-join k={k}"] = select_post_join(speedup)
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()
    profiles = load(args.profiles)

    spread = profiles.groupby(["model", "iterations", "load", "batch_size"])["latency_ms"].agg(
        reps="count", median="median", low="min", high="max", std="std"
    ).reset_index()
    spread["range_share"] = (spread["high"] - spread["low"]) / spread["median"]
    spread["cv"] = spread["std"] / spread["median"]
    spread.to_csv(f"{args.output_prefix}profile_stability_spread.csv", index=False)

    rows = []
    for (model, iterations, load_level, rep), cells in profiles.groupby(["model", "iterations", "load", "rep"]):
        invocation = dict(zip(cells["batch_size"], cells["latency_ms"]))
        for decision, choice in decide(model, invocation).items():
            rows.append({"decision": decision, "model": model, "iterations": iterations, "load": load_level, "rep": rep, "choice": choice})
    decisions = pd.DataFrame(rows)

    reference = decisions[(decisions["iterations"] == 100) & (decisions["load"] == 0)].groupby("decision")["choice"].agg(
        lambda column: column.mode().iloc[0]
    )
    decisions["differs"] = decisions["choice"] != decisions["decision"].map(reference)
    summary = decisions.groupby(["decision", "iterations", "load"]).agg(
        reps=("rep", "count"), differ_share=("differs", "mean"), choices=("choice", lambda column: dict(column.value_counts()))
    ).reset_index()
    summary["reference"] = summary["decision"].map(reference)
    summary.to_csv(f"{args.output_prefix}profile_stability_decisions.csv", index=False)

    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    print("Share of profiles deciding differently from the reference, idle worker, by measured invocations:")
    print(summary[summary["load"] == 0].pivot(index="decision", columns="iterations", values="differ_share").round(2).to_string())
    print("\nThe same at 50 invocations, by busy cores:")
    print(summary[summary["iterations"] == 50].pivot(index="decision", columns="load", values="differ_share").round(2).to_string())
    print("\nt(B) range across repetitions as a share of the median, idle worker:")
    print(spread[spread["load"] == 0].groupby(["model", "iterations"])["range_share"].median().unstack().round(3).to_string())
    print("\nt(B) slowdown at 50 invocations by busy cores, relative to idle:")
    pivot = spread[spread["iterations"] == 50].pivot_table(index=["model", "batch_size"], columns="load", values="median")
    print(pivot.div(pivot[0], axis=0).round(2).to_string())


if __name__ == "__main__":
    main()

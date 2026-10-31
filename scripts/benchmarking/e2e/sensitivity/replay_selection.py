#!/usr/bin/env python3
"""Sensitivity of the implementation selection, part A: replay its decisions over the existing measurements.

The batch-rate sweep measured every fixed batch size at every rate, and the post-join experiment both plans of every
configuration, so what any other parameter value would have selected, and what that selection would have cost, follows from
the data without another run. Every deployment is replayed with its own logged profile, so profile noise stays in the result;
the replay at the defaults must reproduce every logged choice, or the script aborts.

Writes to --output-dir:

  batch_size_sweep.csv      per parameter value and rate: the batch sizes selected across deployments, whether all of them
                            were feasible (sustained, P99 within the bound), and the worst headroom regret.
  batch_size_residuals.csv  per sustained fixed cell: predicted vs. measured latency and the utilization.
  post_join_sweep.csv       per minimum speedup: wrong decisions and cost regret, per platform.
  post_join_overhead.csv    per configuration: measured minus profiled cost of both plans.
  memory_sweep.csv          per model: the budget fractions at which the deployment changes, assessed and measured.
  sensitivity_summary.csv   the ranges for the paper's table.

    ./replay_selection.py --data-dir scripts/benchmarking/figures/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from selection_rules import select_batch_size, select_deployment, select_post_join

WINDOW_MS = 75.0
DEFAULTS = {"slo_fraction": 0.9, "max_utilization": 0.8, "error": 1.0}
# A plan difference below this share of the cost is a tie: neither plan is wrong, whichever is selected.
TIE = 0.02
# The memory budget of the Pi's query in the assessment logs (0.8 of the memory left after the buffer pool).
BUDGET_BYTES = 3157308258
BUDGET_FRACTION = 0.8
MIB = 1024.0**2


def load_batch_rate(data_dir: Path) -> tuple[dict, pd.DataFrame, pd.Series, pd.DataFrame]:
    profile = pd.read_csv(data_dir / "batch_rate" / "batch_rate_profile.csv")
    runs = pd.read_csv(data_dir / "batch_rate" / "batch_rate_runs.csv")
    summary = pd.read_csv(data_dir / "batch_rate" / "batch_rate_summary.csv")
    capacity = pd.read_csv(data_dir / "batch_rate" / "batch_throughput_summary.csv")
    capacity = capacity[capacity["query"] == "inference"].set_index("batch_size")["sustainable_rate"]

    deployments = {}
    for (advance, repetition), cells in profile.groupby(["advance_ms", "repetition"]):
        deployments[(float(advance), repetition)] = dict(zip(cells["batch_size"], cells["invocation_us"] / 1000.0))

    chosen = runs[runs["strategy"] == "OPTIMIZER_CHOOSES"].set_index(["advance_ms", "repetition"])["selected_batch_size"]
    for (advance, repetition), invocation in deployments.items():
        replayed = select_batch_size(invocation, WINDOW_MS, advance).batch_size
        logged = int(chosen.loc[(advance, repetition)])
        if replayed != logged:
            raise SystemExit(f"Replay disagrees with the log at advance {advance}, {repetition}: {replayed} vs. {logged}")
    print(f"Batch size: the replay reproduces all {len(deployments)} logged deployments")
    return deployments, summary[summary["strategy"] == "FIXED"].copy(), capacity, profile


def sweep_batch_size(deployments: dict, fixed: pd.DataFrame, capacity: pd.Series) -> pd.DataFrame:
    feasible_cells = fixed[fixed["sustained"] & (fixed["latency_ms_p99"] <= fixed["slo_ms"])]
    feasible = {(float(row.advance_ms), int(row.batch_size)) for row in feasible_cells.itertuples()}
    oracle = feasible_cells.assign(capacity=feasible_cells["batch_size"].map(capacity)).groupby("advance_ms")["capacity"].max()

    grids = {
        "slo_fraction": np.round(np.arange(0.40, 1.0001, 0.01), 2),
        "max_utilization": np.round(np.arange(0.05, 1.5001, 0.01), 2),
        "error": np.round(np.exp(np.linspace(np.log(0.1), np.log(40.0), 241)), 3),
    }
    grids["error"] = np.unique(np.append(grids["error"], 1.0))
    rows = []
    for parameter, grid in grids.items():
        for value in grid:
            per_rate: dict[float, list[int]] = {}
            for (advance, _), invocation in deployments.items():
                settings = {key: float(value) for key in ("slo_fraction", "max_utilization") if key == parameter}
                scale = float(value) if parameter == "error" else 1.0
                scaled = {batch: scale * latency for batch, latency in invocation.items()}
                per_rate.setdefault(advance, []).append(select_batch_size(scaled, WINDOW_MS, advance, **settings).batch_size)
            for advance, selected in per_rate.items():
                rows.append(
                    {
                        "parameter": parameter,
                        "value": float(value),
                        "advance_ms": advance,
                        "offered_rate": 1000.0 / advance,
                        "selected": " ".join(str(batch) for batch in sorted(set(selected))),
                        "feasible": all((advance, batch) in feasible for batch in selected),
                        "headroom_regret": max(1 - capacity[batch] / oracle[advance] for batch in selected),
                    }
                )
    return pd.DataFrame(rows)


def residuals(profile: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    invocation = profile.groupby("batch_size")["invocation_us"].median() / 1000.0
    cells = fixed[fixed["sustained"] & fixed["batch_size"].isin(invocation.index)].copy()
    cells["invocation_ms"] = cells["batch_size"].map(invocation)
    cells["predicted_ms"] = (cells["batch_size"] - 1) * cells["advance_ms"] + cells["invocation_ms"]
    cells["utilization"] = cells["invocation_ms"] / (cells["batch_size"] * cells["advance_ms"])
    cells["residual_p99_ms"] = cells["latency_ms_p99"] - cells["predicted_ms"]
    columns = ["advance_ms", "batch_size", "invocation_ms", "predicted_ms", "utilization", "latency_ms_p95", "latency_ms_p99"]
    return cells[columns + ["residual_p99_ms"]].sort_values(["advance_ms", "batch_size"], ascending=[False, True])


def load_post_join(data_dir: Path) -> pd.DataFrame:
    frames = []
    for platform in ("x86", "pi"):
        decisions = pd.read_csv(data_dir / "post_join" / f"post_join_decisions_{platform}.csv")
        runs = pd.read_csv(data_dir / "post_join" / f"post_join_runs_{platform}.csv")
        deployed = runs[(runs["strategy"] == "OPTIMIZER_CHOOSES") & runs["predicted_speedup"].notna()]
        mismatched = deployed[[select_post_join(s, m) != c for s, m, c in
                               zip(deployed["predicted_speedup"], deployed["minimum_speedup"], deployed["chosen"])]]
        if not mismatched.empty:
            raise SystemExit(f"Post-join replay disagrees with the log on {platform}:\n{mismatched}")
        print(f"Post-join: the replay reproduces all {len(deployed)} logged deployments on {platform}")
        keep = ["k", "dedup", "predicted_speedup", "batched_ms_per_record", "sequential_ms_per_record"]
        merged = deployed[["k", "dedup", "predicted_speedup"]].merge(
            decisions[["k", "dedup", "batched_ms_per_record", "sequential_ms_per_record"]], on=["k", "dedup"]
        )[keep]
        merged["platform"] = platform
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def sweep_post_join(deployments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    best = deployments[["batched_ms_per_record", "sequential_ms_per_record"]].min(axis=1)
    tie = (deployments["batched_ms_per_record"] - deployments["sequential_ms_per_record"]).abs() / best < TIE
    for sigma in np.round(np.arange(0.70, 2.0001, 0.01), 2):
        batched = deployments["predicted_speedup"] >= sigma
        cost = deployments["batched_ms_per_record"].where(batched, deployments["sequential_ms_per_record"])
        regret = cost / best - 1
        wrong = (regret > 0) & ~tie
        for platform, mask in [("x86", deployments["platform"] == "x86"), ("pi", deployments["platform"] == "pi")]:
            rows.append(
                {
                    "minimum_speedup": float(sigma),
                    "platform": platform,
                    "deployments": int(mask.sum()),
                    "wrong": int(wrong[mask].sum()),
                    "wrong_configurations": deployments[mask & wrong][["k", "dedup"]].drop_duplicates().shape[0],
                    "mean_cost_regret": float(regret[mask].mean()),
                    "max_cost_regret": float(regret[mask].max()),
                }
            )
    return pd.DataFrame(rows)


def post_join_overhead(data_dir: Path) -> pd.DataFrame:
    rows = []
    for platform in ("x86", "pi"):
        decisions = pd.read_csv(data_dir / "post_join" / f"post_join_decisions_{platform}.csv")
        pilot = pd.read_csv(data_dir / "post_join" / f"post_join_pilot_summary_{platform}.csv")
        pilot = pilot[pilot["model"] == "mhatt_rnn"]
        for decision in decisions.itertuples():
            dynamic = 1 if str(decision.dedup).lower() == "true" else 0
            profile = pilot[(pilot["k"] == decision.k) & (pilot["dynamic_batch"] == dynamic)].iloc[0]
            sequential_over = decision.sequential_ms_per_record - decision.k * profile["t1_ms"]
            batched_over = decision.batched_ms_per_record - profile["tk_ms"]
            rows.append(
                {
                    "platform": platform,
                    "k": decision.k,
                    "dedup": decision.dedup,
                    "predicted_speedup": decision.predicted_speedup,
                    "measured_speedup": decision.measured_speedup,
                    "measured_over_predicted": decision.measured_speedup / decision.predicted_speedup,
                    "sequential_overhead_ms": sequential_over,
                    "batched_overhead_ms": batched_over,
                    "batch_specific_overhead_ms": batched_over - sequential_over,
                }
            )
    return pd.DataFrame(rows)


def sweep_memory(data_dir: Path) -> pd.DataFrame:
    available_mib = BUDGET_BYTES / BUDGET_FRACTION / MIB
    models = pd.read_csv(data_dir / "model_memory" / "model_memory.csv")
    rows = []
    for model in models.itertuples():
        rows.append(
            {
                "cell": model.cell,
                "assessed_private_mib": model.assessed_private_total_mb,
                "assessed_shared_mib": model.assessed_shared_total_mb,
                "measured_private_mib": model.deployed_private_mb,
                "measured_shared_mib": model.deployed_shared_mb,
                "selected_at_default": select_deployment(
                    model.assessed_private_total_mb, model.assessed_shared_total_mb, available_mib * BUDGET_FRACTION
                ),
                "private_fits_from_fraction": model.assessed_private_total_mb / available_mib,
                "shared_fits_from_fraction": model.assessed_shared_total_mb / available_mib,
                "measured_private_fits_from_fraction": model.deployed_private_mb / available_mib,
            }
        )
    print(f"Memory: {available_mib:.0f} MiB available after the buffer pool")
    return pd.DataFrame(rows)


def stable_range(sweep: pd.DataFrame, parameter: str, default: float) -> tuple[float, float, float, float]:
    """The contiguous value ranges around the default with unchanged choices and with only feasible choices."""
    cells = sweep[sweep["parameter"] == parameter]
    reference = cells[np.isclose(cells["value"], default)].set_index("advance_ms")["selected"]
    per_value = {}
    for value, group in cells.groupby("value"):
        group = group.set_index("advance_ms")
        per_value[float(value)] = ((group["selected"] == reference).all(), group["feasible"].all())
    values = sorted(per_value)
    start = min(range(len(values)), key=lambda index: abs(values[index] - default))

    def expand(position: int) -> tuple[float, float]:
        low = high = start
        while low > 0 and per_value[values[low - 1]][position]:
            low -= 1
        while high + 1 < len(values) and per_value[values[high + 1]][position]:
            high += 1
        return values[low], values[high]

    return (*expand(0), *expand(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("scripts/benchmarking/figures/data"))
    parser.add_argument("--output-dir", type=Path, help="Defaults to <data-dir>/sensitivity.")
    args = parser.parse_args()
    output = args.output_dir or args.data_dir / "sensitivity"
    output.mkdir(parents=True, exist_ok=True)

    deployments, fixed, capacity, profile = load_batch_rate(args.data_dir)
    batch_sweep = sweep_batch_size(deployments, fixed, capacity)
    batch_sweep.to_csv(output / "batch_size_sweep.csv", index=False)
    residual = residuals(profile, fixed)
    residual.to_csv(output / "batch_size_residuals.csv", index=False)

    post_join = load_post_join(args.data_dir)
    join_sweep = sweep_post_join(post_join)
    join_sweep.to_csv(output / "post_join_sweep.csv", index=False)
    overhead = post_join_overhead(args.data_dir)
    overhead.to_csv(output / "post_join_overhead.csv", index=False)

    memory = sweep_memory(args.data_dir)
    memory.to_csv(output / "memory_sweep.csv", index=False)

    summary = []
    for parameter, default in DEFAULTS.items():
        low, high, safe_low, safe_high = stable_range(batch_sweep, parameter, default)
        summary.append(
            {"parameter": parameter, "default": default, "unchanged_from": low, "unchanged_to": high,
             "feasible_from": safe_low, "feasible_to": safe_high}
        )
    correct = join_sweep.groupby("minimum_speedup")["wrong_configurations"].sum()
    correct = correct[correct == 0]
    summary.append(
        {"parameter": "minimum_speedup_all_deployments_correct", "default": 1.1,
         "feasible_from": correct.index.min() if len(correct) else None, "feasible_to": correct.index.max() if len(correct) else None}
    )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(output / "sensitivity_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n", summary_frame.to_string(index=False))
    print("\nBatch size, where the choice changes with the SLO fraction:")
    pivot = batch_sweep[batch_sweep["parameter"] == "slo_fraction"].pivot(index="value", columns="offered_rate", values="selected")
    print(pivot[(pivot != pivot.shift()).any(axis=1)].to_string())
    print("\nWorst headroom regret and feasibility by SLO fraction, at the change points:")
    slo = batch_sweep[batch_sweep["parameter"] == "slo_fraction"].groupby("value").agg(
        regret=("headroom_regret", "max"), infeasible=("feasible", lambda column: int((~column).sum()))
    )
    print(slo[(slo != slo.shift()).any(axis=1)].round(3).to_string())
    print("\nErrors of t(B), at the change points:")
    error = batch_sweep[batch_sweep["parameter"] == "error"].pivot(index="value", columns="offered_rate", values="selected")
    print(error[(error != error.shift()).any(axis=1)].to_string())
    print("\nResiduals (P99 - predicted) by utilization:")
    print(residual.groupby(pd.cut(residual["utilization"], [0, 0.1, 0.3, 0.6, 0.75, 1.0]), observed=True)["residual_p99_ms"]
          .describe()[["count", "min", "50%", "max"]].round(1).to_string())
    print("\nPost-join by minimum speedup, at the change points:")
    joined = join_sweep.pivot(index="minimum_speedup", columns="platform", values=["wrong", "max_cost_regret"]).round(3)
    print(joined[(joined != joined.shift()).any(axis=1)].to_string())
    print("\nPost-join overheads:")
    print(overhead.round(2).to_string(index=False))
    print("\nMemory:")
    print(memory.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

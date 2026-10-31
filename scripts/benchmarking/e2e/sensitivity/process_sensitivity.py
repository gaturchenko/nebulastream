#!/usr/bin/env python3
"""Consolidate the SARAD (part B) and window-scaling (part C) runs of the implementation-selection sensitivity analysis.

Reads <results-root>/<name>/<combination>/<query>/rep-NN/{latency_timings.csv,_run.log} and writes to --output-dir:

  sensitivity_runs.csv       one row per run.
  sensitivity_summary.csv    the median of each metric per (name, slide, strategy, batch size, u_max).
  sensitivity_profiles.csv   the candidates' t(B) and verdicts the optimizer logged.
  sarad_sweep.csv            part B: the replayed choice per parameter value and slide, with the measured outcome of that choice.
  window_residuals.csv       per fixed cell: predicted and measured latency, in ms and as a share of the window.

All times are wall-clock. For nbeats_w*, the source replays 75 event-time milliseconds in W wall-clock milliseconds, so sizes and
slides are divided by the speedup. A cell is feasible if it sustained the rate and its P99 stayed within W. The replay at the
logged parameters must reproduce every logged choice, or the script aborts.

    ./process_sensitivity.py --results-root results/sensitivity --queries queries/sensitivity
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "batch_rate"))

from process_batch_rate import parse_combination, parse_log, parse_sections, run_metrics, thermal_during  # noqa: E402
from selection_rules import candidate_batch_sizes, select_batch_size  # noqa: E402

SUSTAINED = 0.95
WARMUP_S = 5.0
NAMES = ["sarad", "sarad_knee", "nbeats_w37", "nbeats_w150", "nbeats_w300"]


def replay_speedup(name: str, test: str) -> float:
    return 1.0 if name.startswith("sarad") else int(re.search(r"(\d+) AS `SOURCE`\.MOCK_TUPLE_RATE", test).group(1)) / 1000.0


def event_time_ms(name: str, test: str) -> float:
    with open(re.search(r"ATTACH FILE (\S+)", test).group(1), encoding="utf-8") as handle:
        rows = sum(1 for _ in handle)
    return rows / 48.0 if name.startswith("sarad") else float(rows)


def collect(results_root: Path, queries: Path, name: str) -> tuple[list[dict], list[dict]]:
    results = results_root / name
    test_path = queries / f"{name}.test"
    test = test_path.read_text(encoding="utf-8")
    sections = parse_sections(test_path)
    speedup = replay_speedup(name, test)
    duration_ms = event_time_ms(name, test) / speedup
    thermal = pd.read_csv(results / "thermal.csv") if (results / "thermal.csv").exists() else None

    runs, profiles = [], []
    for latency_csv in sorted(results.glob("*/*/rep-*/latency_timings.csv")):
        rep_dir = latency_csv.parent
        section = rep_dir.parent.name.rsplit("_", 1)[-1]
        size, advance = sections[section]
        size, advance = size / speedup, advance / speedup
        combination = parse_combination(rep_dir.parent.parent.name)
        strategy = combination.get("optimizer.inference_batch_size_strategy", "FIXED")
        utilization = float(combination.get("optimizer.inference_max_utilization", "nan"))
        selected, verdict, candidates = parse_log(rep_dir / "_run.log")
        configured = combination.get("inference.batch_size")
        metrics = run_metrics(latency_csv, size, advance, duration_ms, WARMUP_S)
        runs.append(
            {
                "name": name,
                "section": section,
                "window_ms": size,
                "slide_ms": advance,
                "strategy": strategy,
                "max_utilization": utilization,
                "batch_size": selected if strategy == "OPTIMIZER_CHOOSES" else int(configured),
                "selection_verdict": verdict,
                "repetition": rep_dir.name,
                **metrics,
                "sustained": metrics["sustained_ratio"] >= SUSTAINED and metrics["delivered_ratio"] >= SUSTAINED,
                **thermal_during(thermal, metrics["run_start_us"], metrics["run_end_us"]),
            }
        )
        for candidate in candidates:
            profiles.append(
                {"name": name, "slide_ms": advance, "window_ms": size, "max_utilization_run": utilization,
                 "repetition": rep_dir.name, "selected": selected, **candidate}
            )
    return runs, profiles


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["name", "window_ms", "slide_ms", "offered_rate", "strategy", "max_utilization", "batch_size"]
    metrics = ["achieved_rate", "sustained_ratio", "delivered_ratio", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99",
               "latency_ms_max", "slo_violation_share", "temp_c_max"]
    grouped = runs.groupby(keys, dropna=False)
    summary = grouped[metrics].median().reset_index()
    summary["repetitions"] = grouped.size().values
    summary["sustained"] = grouped["sustained"].all().values
    summary["feasible"] = summary["sustained"] & (summary["latency_ms_p99"] <= summary["window_ms"])
    summary["any_throttled"] = grouped["throttled_during_run"].apply(lambda values: bool(values.fillna(False).any())).values
    return summary


def check_replay(profiles: pd.DataFrame) -> None:
    checked = 0
    for (name, slide, utilization, repetition), cells in profiles.groupby(["name", "slide_ms", "max_utilization_run", "repetition"]):
        invocation = dict(zip(cells["batch_size"], cells["invocation_us"] / 1000.0))
        window = cells["window_ms"].iloc[0]
        replayed = select_batch_size(invocation, window, slide, utilization).batch_size
        if replayed != cells["selected"].iloc[0] or sorted(invocation) != candidate_batch_sizes(window, slide):
            raise SystemExit(f"Replay disagrees with the log: {name} slide {slide} u_max {utilization} {repetition}: {replayed}")
        checked += 1
    print(f"The replay reproduces all {checked} logged deployments")


def sweep_sarad(profiles: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    fixed = summary[(summary["name"] == "sarad") & (summary["strategy"] == "FIXED")]
    grids = {
        "max_utilization": np.round(np.arange(0.40, 1.5001, 0.01), 2),
        "slo_fraction": np.round(np.arange(0.40, 1.0001, 0.01), 2),
        "error": np.round(np.exp(np.linspace(np.log(0.25), np.log(4.0), 121)), 3),
    }
    grids["error"] = np.unique(np.append(grids["error"], 1.0))
    rows = []
    deployments = profiles[profiles["name"] == "sarad"].groupby(["slide_ms", "max_utilization_run", "repetition"])
    for parameter, grid in grids.items():
        for value in grid:
            per_slide: dict[float, list] = {}
            for (slide, _, _), cells in deployments:
                scale = float(value) if parameter == "error" else 1.0
                invocation = dict(zip(cells["batch_size"], scale * cells["invocation_us"] / 1000.0))
                settings = {parameter: float(value)} if parameter != "error" else {}
                per_slide.setdefault(slide, []).append(select_batch_size(invocation, 200.0, slide, **settings))
            for slide, choices in per_slide.items():
                measured = fixed[fixed["slide_ms"] == slide].set_index("batch_size")
                selected = sorted({choice.batch_size for choice in choices})
                present = [batch for batch in selected if batch in measured.index]
                feasible = len(present) == len(selected) and all(measured.loc[batch, "feasible"] for batch in present)
                rows.append(
                    {
                        "parameter": parameter,
                        "value": float(value),
                        "slide_ms": slide,
                        "selected": " ".join(str(batch) for batch in selected),
                        "branches": " ".join(sorted({choice.branch for choice in choices})),
                        "feasible": feasible,
                        "worst_p99_ms": max((measured.loc[batch, "latency_ms_p99"] for batch in present), default=np.nan),
                        "best_measured_p99_ms": measured.loc[measured["sustained"], "latency_ms_p99"].min(),
                    }
                )
    return pd.DataFrame(rows)


def window_residuals(profiles: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    invocation = profiles.groupby(["name", "batch_size"])["invocation_us"].median() / 1000.0
    fixed = summary[summary["strategy"] == "FIXED"].copy()
    # The knee cells run fixed batch sizes only, so they have no profile of their own.
    fixed["invocation_ms"] = [invocation.get((name.removesuffix("_knee"), batch), np.nan)
                              for name, batch in zip(fixed["name"], fixed["batch_size"])]
    fixed["predicted_ms"] = (fixed["batch_size"] - 1) * fixed["slide_ms"] + fixed["invocation_ms"]
    fixed["utilization"] = fixed["invocation_ms"] / (fixed["batch_size"] * fixed["slide_ms"])
    fixed["residual_p99_ms"] = fixed["latency_ms_p99"] - fixed["predicted_ms"]
    fixed["residual_share_of_window"] = fixed["residual_p99_ms"] / fixed["window_ms"]
    fixed["predicted_share_of_window"] = fixed["predicted_ms"] / fixed["window_ms"]
    columns = ["name", "window_ms", "slide_ms", "batch_size", "invocation_ms", "predicted_ms", "predicted_share_of_window", "utilization",
               "latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "residual_p99_ms", "residual_share_of_window", "sustained", "feasible"]
    return fixed[columns].sort_values(["window_ms", "slide_ms", "batch_size"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=Path("results/sensitivity"))
    parser.add_argument("--queries", type=Path, default=Path("queries/sensitivity"))
    parser.add_argument("--names", nargs="+", default=NAMES)
    parser.add_argument("--output-dir", type=Path, help="Defaults to --results-root.")
    args = parser.parse_args()
    output = args.output_dir or args.results_root

    runs, profiles = [], []
    for name in args.names:
        if (args.results_root / name).exists():
            name_runs, name_profiles = collect(args.results_root, args.queries, name)
            runs += name_runs
            profiles += name_profiles
    runs_frame, profile_frame = pd.DataFrame(runs), pd.DataFrame(profiles)
    runs_frame["batch_size"] = runs_frame["batch_size"].astype("Int64")
    summary = summarize(runs_frame)
    runs_frame.to_csv(output / "sensitivity_runs.csv", index=False)
    summary.to_csv(output / "sensitivity_summary.csv", index=False)
    profile_frame.to_csv(output / "sensitivity_profiles.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"{len(runs_frame)} runs, {int(summary['any_throttled'].sum())} cells throttled")
    if not profile_frame.empty:
        check_replay(profile_frame)
    print(summary[["name", "window_ms", "slide_ms", "strategy", "max_utilization", "batch_size", "repetitions", "sustained_ratio",
                   "latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "feasible"]].round(2).to_string(index=False))
    if profile_frame.empty:
        print("No profiles in the logs yet")
        return
    residual = window_residuals(profile_frame, summary)
    residual.to_csv(output / "window_residuals.csv", index=False)
    print(residual.round(3).to_string(index=False))
    if "sarad" in set(profile_frame["name"]):
        sweep = sweep_sarad(profile_frame, summary)
        sweep.to_csv(output / "sarad_sweep.csv", index=False)
        for parameter in sweep["parameter"].unique():
            cells = sweep[sweep["parameter"] == parameter]
            pivot = cells.pivot(index="value", columns="slide_ms", values="selected")
            print(f"\nSARAD by {parameter}, at the change points:")
            print(pivot[(pivot != pivot.shift()).any(axis=1)].to_string())


if __name__ == "__main__":
    main()

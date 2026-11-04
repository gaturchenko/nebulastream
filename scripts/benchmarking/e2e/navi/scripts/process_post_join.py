#!/usr/bin/env python3
"""Consolidate the post-join batching sweep (run_post_join.sh) into CSVs.

Reads <results>/<combination>/post_join_k<K>.test_<section>/rep-NN/{latency_timings.csv,_run.log} as run_systests_lat.py
lays them out, the driver's _grid.csv and thermal.csv, and writes:

  post_join_runs.csv       one row per cell and repetition, including those that left no results: offered and delivered
                           record rate, whether the run sustained the offered rate, latency, the optimizer's decision and
                           profile (OPTIMIZER_CHOOSES runs), and the thermal state.
  post_join_summary.csv    per cell: medians over repetitions; a cell is sustained only if every repetition was.
  post_join_decisions.csv  per k and dedup setting: OPTIMIZER_CHOOSES' plan and predicted speedup k * t(1) / t(k) against
                           the measured alternatives: the latency speedup P50(SEQUENTIAL) / P50(BATCHED) at the lowest
                           rate, the largest rate each forced plan sustained, the measured better plan, and the regret of
                           the chosen one (its P50 over the better plan's, and the share of rates the better plan sustains
                           that the chosen one does not).

A run sustains its offered rate if it delivered its records at no less than --sustained-threshold of that rate; the source
backpressures a query that falls behind, which delays ingestion instead of growing the measured latency.

Run from ~/navi on the Pi, or anywhere with --results-dir pointing at a copy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from process_batch_rate import ANSI_ESCAPE, parse_combination, thermal_during

QUERY_DIR = re.compile(r"post_join_k(\d+)\.test_(\d+)$")
PIPELINE = re.compile(r"Pipeline\(ID\((\d+)\)")
# Operators of the inference work of either plan: SEQUENTIAL runs k InferModel operators in the join probe's pipeline, BATCHED
# collects the payloads (PostJoinBatching) and invokes BatchInferModel. Both plans' pipelines also hold the join probe.
INFERENCE_OPERATORS = ("InferModelPhysicalOperator", "PostJoinBatchingPhysicalOperator")
DECISION = re.compile(
    r"Post-join inference of model '[^']*' on worker '[^']*': (\d+) x batch-1 invocation = (\d+) ns, one batch-(\d+) invocation = "
    r"(\d+) ns, speedup ([\d.]+) \(minimum ([\d.]+)\) -> (\w+)"
)


def parse_decision(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if match := DECISION.search(ANSI_ESCAPE.sub("", line)):
            return {
                "sequential_profile_ms": int(match.group(2)) / 1e6,
                "batched_profile_ms": int(match.group(4)) / 1e6,
                "predicted_speedup": float(match.group(5)),
                "minimum_speedup": float(match.group(6)),
                "chosen": match.group(7),
            }
    return {}


def inference_pipelines(log_path: Path) -> set[int]:
    """Ids of the pipelines whose operator chain in the log's plan dump holds an inference operator."""
    pipelines, current = set(), None
    if not log_path.exists():
        return pipelines
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if match := PIPELINE.search(line):
            current = int(match.group(1))
        elif current is not None and any(operator in line for operator in INFERENCE_OPERATORS):
            pipelines.add(current)
    return pipelines


def inference_task_metrics(rep_dir: Path, results: int) -> dict:
    """Task time of the inference pipelines from the run's event traces, in total and per joined record.

    The trace closes every pipeline with one event whose dur is the summed duration of all its tasks (GoogleEventTracePrinter).
    """
    pipelines = inference_pipelines(rep_dir / "_run.log")
    traces = sorted(rep_dir.glob("trace_*.json"))
    if not pipelines or not traces:
        return {}
    task_us, tasks = 0.0, 0
    for trace in traces:
        for event in json.loads(trace.read_text(encoding="utf-8")).get("traceEvents", []):
            if event.get("cat") != "pipeline" or event.get("ph") != "E" or "dur" not in event:
                continue
            if int((event.get("args") or {}).get("pipeline_id", -1)) in pipelines:
                task_us += float(event["dur"])
                tasks += int(event.get("task_count", 0))
    return {
        "inference_pipelines": " ".join(str(p) for p in sorted(pipelines)),
        "inference_tasks": tasks,
        "inference_task_ms": task_us / 1e3,
        "inference_ms_per_record": task_us / 1e3 / results if results else float("nan"),
    }


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


def decisions(summary: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    inference = summary[summary["query"] == "inference"]
    for (k, dedup), cells in inference.groupby(["k", "dedup"]):
        forced = cells[cells["strategy"].isin(["BATCHED", "SEQUENTIAL"])]
        if forced["strategy"].nunique() < 2:
            continue
        row: dict = {"k": k, "dedup": dedup}
        chooses = runs[(runs["k"] == k) & (runs["dedup"] == dedup) & (runs["strategy"] == "OPTIMIZER_CHOOSES")]
        # A run whose log holds no decision line (still running, or failed before deployment) has no "chosen" column.
        if not chooses.empty and "chosen" in chooses:
            row |= {
                "chosen": "/".join(sorted(chooses["chosen"].dropna().unique())),
                "predicted_speedup": chooses["predicted_speedup"].median(),
                "predicted_speedup_min": chooses["predicted_speedup"].min(),
                "predicted_speedup_max": chooses["predicted_speedup"].max(),
            }
        # The lowest rate both plans sustained isolates the per-record cost from queueing.
        both = forced[forced["sustained"]].groupby("advance_ms")["strategy"].nunique()
        both = both[both == 2]
        if not both.empty:
            advance = both.index.max()
            at = forced[forced["advance_ms"] == advance].set_index("strategy")
            row |= {
                "latency_advance_ms": advance,
                "batched_p50_ms": at.loc["BATCHED", "latency_ms_p50"],
                "sequential_p50_ms": at.loc["SEQUENTIAL", "latency_ms_p50"],
                "latency_speedup": at.loc["SEQUENTIAL", "latency_ms_p50"] / at.loc["BATCHED", "latency_ms_p50"],
            }
            # The plans' own cost: task time of the inference pipelines per joined record, free of window and join latency.
            if "inference_ms_per_record" in at and at["inference_ms_per_record"].notna().all():
                row |= {
                    "batched_ms_per_record": at.loc["BATCHED", "inference_ms_per_record"],
                    "sequential_ms_per_record": at.loc["SEQUENTIAL", "inference_ms_per_record"],
                    "measured_speedup": at.loc["SEQUENTIAL", "inference_ms_per_record"] / at.loc["BATCHED", "inference_ms_per_record"],
                }
            else:
                row["measured_speedup"] = row["latency_speedup"]
        for strategy in ("BATCHED", "SEQUENTIAL"):
            sustained = forced[(forced["strategy"] == strategy) & forced["sustained"]]
            row[f"{strategy.lower()}_max_sustained_rate"] = sustained["offered_rate"].max() if not sustained.empty else 0.0
        if "measured_speedup" in row:
            better = "BATCHED" if row["measured_speedup"] >= 1 else "SEQUENTIAL"
            row["measured_better"] = better
            if row.get("chosen") in ("BATCHED", "SEQUENTIAL"):
                chosen_p50 = row[f"{row['chosen'].lower()}_p50_ms"]
                row["correct"] = row["chosen"] == better
                row["latency_regret"] = chosen_p50 / row[f"{better.lower()}_p50_ms"] - 1
                if "batched_ms_per_record" in row:
                    row["cost_regret"] = (
                        row[f"{row['chosen'].lower()}_ms_per_record"] / row[f"{better.lower()}_ms_per_record"] - 1
                    )
                row["rate_regret"] = 1 - (
                    row[f"{row['chosen'].lower()}_max_sustained_rate"] / max(row[f"{better.lower()}_max_sustained_rate"], 1e-9)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results/post_join"))
    parser.add_argument("--warmup-s", type=float, default=5.0, help="Records ingested in the first seconds are dropped.")
    parser.add_argument("--sustained-threshold", type=float, default=0.95, help="Delivered/offered rate from which a run sustained.")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions every cell should have.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to --results-dir.")
    args = parser.parse_args()

    grid = pd.read_csv(args.results_dir / "_grid.csv", dtype={"section": str, "dedup": str})
    grid["section"] = grid["section"].str.zfill(2)
    thermal_path = args.results_dir / "thermal.csv"
    thermal = pd.read_csv(thermal_path) if thermal_path.exists() else None
    output_dir = args.output_dir or args.results_dir

    completed = {}
    for latency_csv in sorted(args.results_dir.glob("*/*/rep-*/latency_timings.csv")):
        rep_dir = latency_csv.parent
        match = QUERY_DIR.match(rep_dir.parent.name)
        if not match:
            continue
        combination = parse_combination(rep_dir.parent.parent.name)
        key = (
            int(match.group(1)),
            combination.get("optimizer.post_join_inference_strategy", "OPTIMIZER_CHOOSES"),
            # run_systests_lat.py shortens worker.default_query_execution.inference.* to inference.* in the label.
            combination.get("inference.use_batch_deduplication", "false").lower(),
            match.group(2),
            rep_dir.name,
        )
        completed[key] = rep_dir

    runs = []
    for cell in grid.itertuples(index=False):
        offered = 1000.0 / cell.advance_ms
        for rep in range(1, args.repetitions + 1):
            row = {
                "k": cell.k, "strategy": cell.strategy, "dedup": cell.dedup, "section": cell.section, "query": cell.query,
                "advance_ms": cell.advance_ms, "offered_rate": offered, "repetition": f"rep-{rep:02d}",
            }
            rep_dir = completed.get((cell.k, cell.strategy, cell.dedup, cell.section, row["repetition"]))
            if rep_dir is None:
                runs.append(row | {"completed": False, "sustained": False})
                continue
            metrics = run_metrics(rep_dir / "latency_timings.csv", offered, args.warmup_s)
            runs.append(
                row
                | {"completed": True, **metrics, "sustained": metrics["sustained_ratio"] >= args.sustained_threshold}
                | inference_task_metrics(rep_dir, metrics["results"])
                | parse_decision(rep_dir / "_run.log")
                | thermal_during(thermal, metrics["run_start_us"], metrics["run_end_us"])
            )
    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(output_dir / "post_join_runs.csv", index=False)

    cell_columns = ["query", "k", "strategy", "dedup", "advance_ms", "offered_rate"]
    metric_columns = [c for c in ("results", "achieved_rate", "sustained_ratio", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99",
                                  "inference_ms_per_record", "predicted_speedup", "temp_c_max") if c in runs_df]
    grouped = runs_df.groupby(cell_columns, dropna=False)
    summary = grouped[metric_columns].median().reset_index()
    summary["repetitions"] = grouped["completed"].sum().values
    summary["sustained"] = grouped["sustained"].all().values
    if "throttled_during_run" in runs_df:
        summary["any_throttled"] = grouped["throttled_during_run"].apply(lambda s: bool(s.fillna(False).any())).values
    summary.to_csv(output_dir / "post_join_summary.csv", index=False)

    decision_df = decisions(summary, runs_df)
    decision_df.to_csv(output_dir / "post_join_decisions.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if not decision_df.empty:
        print()
        print(decision_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    missing = runs_df[~runs_df["completed"]]
    if not missing.empty:
        print(f"\n{len(missing)} runs left no results (failed or not yet run)")


if __name__ == "__main__":
    main()

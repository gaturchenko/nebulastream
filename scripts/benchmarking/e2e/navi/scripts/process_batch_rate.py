#!/usr/bin/env python3
"""Consolidate the source-rate x batch-size sweep (run_batch_rate.sh) into CSVs.

Reads <results>/<combination>/<query>/rep-NN/{latency_timings.csv,_run.log} as run_systests_lat.py lays them out, plus the
driver's thermal.csv, and writes:

  batch_rate_runs.csv     one row per run: offered rate, configured or selected batch size, latency percentiles, the share
                          of results over the SLO, and sustained throughput, and the thermal state.
  batch_rate_summary.csv  the median of each metric over the repetitions of a (rate, strategy, batch size) cell.
  batch_rate_profile.csv  OPTIMIZER_CHOOSES' per-candidate profile and verdicts, parsed from the run logs (INFO builds only).
  batch_rate_incomplete.csv  cells with fewer completed repetitions than --repetitions; a run that fails every retry
                          (e.g. a batch size that falls behind until the buffer pool runs out) leaves no results.

Each section's window ADVANCE and SIZE are parsed from the .test, so the rate axis cannot drift from the queries. SIZE is
also the latency SLO the optimizer derives, so slo_violation_share uses it.

Run from ~/navi on the Pi, or anywhere with --results-dir/--test pointing at a copy.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
WINDOW = re.compile(r"WINDOW\s+SLIDING\(\s*\w+\s*,\s*SIZE\s+(\d+)\s+ms\s*,\s*ADVANCE\s+BY\s+(\d+)\s+ms\s*\)", re.IGNORECASE)
SELECTED = re.compile(r"Selected batch size (\d+) for model '[^']*' on worker '[^']*': (.*)$")
CANDIDATE = re.compile(
    r"Batch size (\d+) for model '([^']*)' on worker '([^']*)': invocation (\d+) us, utilization ([\d.]+) "
    r"\(max ([\d.]+): (keeps up|falls behind)\), result latency (\d+) us \(SLO (\d+) us: (met|violated)\), "
    r"throughput (\d+) tuples/s"
)
# vcgencmd get_throttled: bits 0-3 are the CURRENT state (under-voltage, frequency capped, throttled, soft temperature
# limit); bits 16-19 only record that it happened at some point since boot.
THROTTLED_NOW_MASK = 0xF


def parse_sections(test_path: Path) -> dict[str, tuple[int, int]]:
    """Maps a query suffix ('01', ...) to (window size ms, advance ms), in section order."""
    sections = {}
    for match in WINDOW.finditer(test_path.read_text(encoding="utf-8")):
        sections[f"{len(sections) + 1:02d}"] = (int(match.group(1)), int(match.group(2)))
    if not sections:
        raise ValueError(f"No SLIDING windows found in {test_path}")
    return sections


def parse_fixed_batch_sizes(config_path: Path) -> list[int]:
    match = re.search(r'^"optimizer\.inference_batch_size":\s*\[([^\]]*)\]', config_path.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise ValueError(f"No optimizer.inference_batch_size list in {config_path}")
    return [int(value) for value in match.group(1).split(",")]


def parse_combination(label: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in label.split("__") if "=" in part)


def parse_log(log_path: Path) -> tuple[int | None, str | None, list[dict]]:
    if not log_path.exists():
        return None, None, []
    selected, verdict, candidates = None, None, []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = ANSI_ESCAPE.sub("", line)
        if match := SELECTED.search(line):
            selected, verdict = int(match.group(1)), match.group(2).strip()
        elif match := CANDIDATE.search(line):
            candidates.append(
                {
                    "batch_size": int(match.group(1)),
                    "model": match.group(2),
                    "invocation_us": int(match.group(4)),
                    "utilization": float(match.group(5)),
                    "max_utilization": float(match.group(6)),
                    "keeps_up": match.group(7) == "keeps up",
                    "result_latency_us": int(match.group(8)),
                    "slo_us": int(match.group(9)),
                    "meets_slo": match.group(10) == "met",
                    "throughput": int(match.group(11)),
                }
            )
    return selected, verdict, candidates


def thermal_during(thermal: pd.DataFrame | None, start_us: int, end_us: int) -> dict:
    if thermal is None:
        return {"temp_c_max": None, "throttled_during_run": None}
    window = thermal[(thermal["time_us"] >= start_us) & (thermal["time_us"] <= end_us)]
    if window.empty:
        return {"temp_c_max": None, "throttled_during_run": None}
    flags = window["throttled"].map(lambda value: int(str(value), 16))
    return {"temp_c_max": window["temp_c"].max(), "throttled_during_run": bool((flags & THROTTLED_NOW_MASK).any())}


def run_metrics(latency_csv: Path, size_ms: int, advance_ms: int, rows: int, warmup_s: float) -> dict:
    timings = pd.read_csv(latency_csv).sort_values("ingest_ts_us")
    results = len(timings)
    offered = 1000.0 / advance_ms
    # Complete windows over timestamps 0 .. rows - 1. NES also emits trailing partial windows at end of stream, so the
    # delivered ratio may exceed 1 slightly.
    expected = (rows - size_ms) // advance_ms + 1
    first_ingest, last_receive = int(timings["ingest_ts_us"].min()), int(timings["recv_ts_us"].max())
    span_s = (last_receive - first_ingest) / 1e6
    achieved = results / span_s if span_s > 0 else float("nan")

    steady = timings[timings["ingest_ts_us"] >= first_ingest + warmup_s * 1e6]
    latency_ms = steady["latency_us"] / 1000.0
    return {
        "offered_rate": offered,
        "results": results,
        "expected_results": expected,
        "delivered_ratio": results / expected,
        "achieved_rate": achieved,
        "sustained_ratio": achieved / offered,
        "latency_ms_p50": latency_ms.quantile(0.50),
        "latency_ms_p95": latency_ms.quantile(0.95),
        "latency_ms_p99": latency_ms.quantile(0.99),
        "latency_ms_max": latency_ms.max(),
        "slo_ms": size_ms,
        "slo_violation_share": float((latency_ms > size_ms).mean()),
        "run_start_us": first_ingest,
        "run_end_us": last_receive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results/batch_rate"))
    parser.add_argument("--test", type=Path, default=Path("queries/batch-rate/nbeats_batch_rate.test"))
    parser.add_argument("--data", type=Path, default=Path("datasets/synthetic/batch_rate.csv"))
    parser.add_argument("--warmup-s", type=float, default=5.0, help="Results ingested in the first seconds are dropped.")
    parser.add_argument("--sustained-threshold", type=float, default=0.95, help="sustained_ratio below which a run fell behind.")
    parser.add_argument("--fixed-config", type=Path, default=Path("queries/batch-rate/batch_rate_fixed.yaml"))
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions every cell should have.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to --results-dir.")
    args = parser.parse_args()

    sections = parse_sections(args.test)
    batch_sizes = parse_fixed_batch_sizes(args.fixed_config)
    with args.data.open(encoding="utf-8") as handle:
        rows = sum(1 for _ in handle)
    thermal_path = args.results_dir / "thermal.csv"
    thermal = pd.read_csv(thermal_path) if thermal_path.exists() else None
    output_dir = args.output_dir or args.results_dir

    runs, profiles = [], []
    for latency_csv in sorted(args.results_dir.glob("*/*/rep-*/latency_timings.csv")):
        rep_dir = latency_csv.parent
        query_name, label = rep_dir.parent.name, rep_dir.parent.parent.name
        section = query_name.rsplit("_", 1)[-1]
        if section not in sections:
            print(f"Skipping {rep_dir}: section {section} is not in {args.test}")
            continue
        size_ms, advance_ms = sections[section]
        combination = parse_combination(label)
        strategy = combination.get("optimizer.inference_batch_size_strategy", "FIXED")
        selected, verdict, candidates = parse_log(rep_dir / "_run.log")
        configured = combination.get("inference.batch_size")

        metrics = run_metrics(latency_csv, size_ms, advance_ms, rows, args.warmup_s)
        runs.append(
            {
                "section": section,
                "advance_ms": advance_ms,
                "strategy": strategy,
                "configured_batch_size": int(configured) if configured else None,
                "selected_batch_size": selected,
                "batch_size": selected if strategy == "OPTIMIZER_CHOOSES" else (int(configured) if configured else 1),
                "selection_verdict": verdict,
                "repetition": rep_dir.name,
                **metrics,
                "sustained": metrics["sustained_ratio"] >= args.sustained_threshold,
                **thermal_during(thermal, metrics["run_start_us"], metrics["run_end_us"]),
                "combination": label,
            }
        )
        for candidate in candidates:
            profiles.append({"section": section, "advance_ms": advance_ms, "repetition": rep_dir.name, **candidate})

    if not runs:
        raise SystemExit(f"No latency_timings.csv found under {args.results_dir}")

    runs_df = pd.DataFrame(runs)
    for column in ("configured_batch_size", "selected_batch_size", "batch_size"):
        runs_df[column] = runs_df[column].astype("Int64")
    runs_df = runs_df.sort_values(["advance_ms", "strategy", "batch_size", "repetition"], ascending=[False, True, True, True])
    runs_df.to_csv(output_dir / "batch_rate_runs.csv", index=False)

    metric_columns = [
        "batch_size", "results", "delivered_ratio", "achieved_rate", "sustained_ratio", "latency_ms_p50", "latency_ms_p95",
        "latency_ms_p99", "latency_ms_max", "slo_violation_share", "temp_c_max",
    ]
    grouped = runs_df.groupby(["advance_ms", "offered_rate", "slo_ms", "strategy", "configured_batch_size"], dropna=False)
    summary = grouped[metric_columns].median().reset_index()
    summary["repetitions"] = grouped.size().values
    # A cell counts as sustained or throttle-free only if every repetition was.
    summary["sustained"] = grouped["sustained"].all().values
    summary["any_throttled"] = grouped["throttled_during_run"].apply(lambda values: bool(values.fillna(False).any())).values
    summary["selected_batch_sizes"] = grouped["selected_batch_size"].apply(
        lambda values: ",".join(str(int(value)) for value in sorted(values.dropna().unique()))
    ).values
    summary = summary.sort_values(["advance_ms", "strategy", "configured_batch_size"], ascending=[False, True, True])
    summary.to_csv(output_dir / "batch_rate_summary.csv", index=False)

    if profiles:
        pd.DataFrame(profiles).to_csv(output_dir / "batch_rate_profile.csv", index=False)

    # A run that fails every retry leaves no latency_timings.csv, e.g. a batch size that falls behind until the backlog
    # exhausts the buffer pool. List every cell with fewer repetitions than intended, so failures are not silently dropped.
    def completed(advance_ms: int, strategy: str, batch_size) -> int:
        cell_runs = runs_df[(runs_df["advance_ms"] == advance_ms) & (runs_df["strategy"] == strategy)]
        if batch_size is not pd.NA:
            cell_runs = cell_runs[cell_runs["configured_batch_size"] == batch_size]
        return len(cell_runs)

    incomplete = []
    for advance_ms in sorted({advance for _, advance in sections.values()}, reverse=True):
        for strategy, batch_size in [("FIXED", size) for size in batch_sizes] + [("OPTIMIZER_CHOOSES", pd.NA)]:
            count = completed(advance_ms, strategy, batch_size)
            if count < args.repetitions:
                incomplete.append(
                    {
                        "offered_rate": 1000.0 / advance_ms,
                        "advance_ms": advance_ms,
                        "strategy": strategy,
                        "configured_batch_size": batch_size,
                        "repetitions_found": count,
                        "repetitions_expected": args.repetitions,
                    }
                )
    incomplete_df = pd.DataFrame(incomplete)
    incomplete_df.to_csv(output_dir / "batch_rate_incomplete.csv", index=False)

    # Console overview: median P95 per (rate, batch size); '*' marks cells that fell behind, '!' cells over the SLO at P95.
    def cell(row: pd.Series) -> str:
        marks = ("*" if not row["sustained"] else "") + ("!" if row["latency_ms_p95"] > row["slo_ms"] else "")
        return f"{row['latency_ms_p95']:.0f}{marks}"

    fixed = summary[summary["strategy"] == "FIXED"].copy()
    if not fixed.empty:
        fixed["cell"] = fixed.apply(cell, axis=1)
        rates = sorted(1000.0 / advance for _, advance in sections.values())
        table = (
            fixed.pivot(index="offered_rate", columns="configured_batch_size", values="cell")
            .reindex(index=rates, columns=batch_sizes)
            .fillna("fail")
        )
        chosen = summary[summary["strategy"] == "OPTIMIZER_CHOOSES"].set_index("offered_rate")
        if not chosen.empty:
            table["chooses"] = chosen.apply(lambda row: f"B={row['selected_batch_sizes'] or '?'} {cell(row)}", axis=1)
        print(
            "Median P95 latency [ms] by offered rate [results/s] and batch size; * fell behind, ! P95 over the SLO, "
            "fail: no run completed"
        )
        print(table.to_string())
    throttled = runs_df[runs_df["throttled_during_run"].fillna(False).astype(bool)]
    if not throttled.empty:
        print(f"\nWARNING: {len(throttled)} run(s) were throttled; see throttled_during_run in batch_rate_runs.csv")
    if not incomplete_df.empty:
        print(f"\n{len(incomplete_df)} cell(s) have fewer than {args.repetitions} completed repetitions; see batch_rate_incomplete.csv")
    print(f"\nWrote {len(runs_df)} runs to {output_dir}")


if __name__ == "__main__":
    main()

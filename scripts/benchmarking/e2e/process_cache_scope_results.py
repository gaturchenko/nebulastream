#!/usr/bin/env python3
"""Consolidate cache-scope latency benchmark results into one tidy CSV.

Walks the results tree written by run_systests.py for
nes-systests/inference/cache_scope/nbeats300_cache_scope_lat.test:

    <results-dir>/<combo_label>/<query_name>/rep-NN/latency_timings.csv
    <results-dir>/<combo_label>/<query_name>/rep-NN/_run.log

and writes one summary row per (combination, hit rate, repetition) with the swept
parameters as proper columns plus latency statistics and the cache hit/miss counts
parsed from the run log:

    hot_percent, worker_threads, cache_type, cache_scope, cache_entries, repetition,
    records, latency_us_mean, latency_us_p50, latency_us_p90, latency_us_p95,
    latency_us_p99, latency_us_max, cache_hits, cache_misses, observed_hit_rate

With --per-record it additionally writes a long CSV with one row per record
(same metadata columns + latency_us), suitable for plotting full distributions.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Query sections of nbeats300_cache_scope_lat.test (query dir names end in _NN).
SECTION_HOT_PERCENT = {"01": 25, "02": 50, "03": 75, "04": 95}

PARAM_COLUMNS = {
    "worker.query_engine.number_of_worker_threads": "worker_threads",
    "inference.prediction_cache_type": "cache_type",
    "inference.prediction_cache_scope": "cache_scope",
    "inference.number_of_entries_prediction_cache": "cache_entries",
}

CACHE_STATS_PATTERN = re.compile(r"CacheInferModelPhysicalOperator cache hits=(\d+), misses=(\d+)")

SUMMARY_COLUMNS = [
    "hot_percent",
    "worker_threads",
    "cache_type",
    "cache_scope",
    "cache_entries",
    "repetition",
    "records",
    "latency_us_mean",
    "latency_us_p50",
    "latency_us_p90",
    "latency_us_p95",
    "latency_us_p99",
    "latency_us_max",
    "cache_hits",
    "cache_misses",
    "observed_hit_rate",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_combo_label(label: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for entry in label.split("__"):
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        params[PARAM_COLUMNS.get(key, key)] = value
    return params


def parse_hot_percent(query_dir_name: str) -> Optional[int]:
    match = re.search(r"_(\d{2})$", query_dir_name)
    if match is None:
        return None
    return SECTION_HOT_PERCENT.get(match.group(1))


def read_latencies(csv_path: Path) -> List[float]:
    latencies: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "latency_us" not in reader.fieldnames:
            return latencies
        for row in reader:
            try:
                latencies.append(float(row["latency_us"]))
            except (KeyError, TypeError, ValueError):
                continue
    return latencies


def read_cache_stats(log_path: Path) -> Optional[tuple[int, int]]:
    if not log_path.exists():
        return None
    matches = CACHE_STATS_PATTERN.findall(log_path.read_text(encoding="utf-8", errors="ignore"))
    if not matches:
        return None
    # One line per query run; a rep dir holds exactly one run.
    hits, misses = matches[-1]
    return int(hits), int(misses)


def percentile(sorted_values: List[float], fraction: float) -> float:
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def iter_rep_dirs(results_dir: Path) -> Iterable[tuple[Dict[str, str], int, int, Path]]:
    for combo_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        params = parse_combo_label(combo_dir.name)
        if not params:
            continue
        for query_dir in sorted(p for p in combo_dir.iterdir() if p.is_dir()):
            hot_percent = parse_hot_percent(query_dir.name)
            if hot_percent is None:
                continue
            for rep_dir in sorted(query_dir.glob("rep-*")):
                match = re.search(r"rep-(\d+)$", rep_dir.name)
                if match is None:
                    continue
                yield params, hot_percent, int(match.group(1)), rep_dir


def main() -> int:
    default_results = repo_root() / "scripts" / "benchmarking" / "e2e" / "results"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--output-csv", type=Path, default=None, help="Summary CSV (default: <results-dir>/cache_scope_summary.csv)")
    parser.add_argument("--per-record", type=Path, default=None, metavar="CSV", help="Also write one row per record to this CSV.")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 1
    output_csv = args.output_csv or results_dir / "cache_scope_summary.csv"

    summary_rows: List[Dict[str, object]] = []
    per_record_handle = None
    per_record_writer = None
    if args.per_record is not None:
        per_record_handle = args.per_record.open("w", encoding="utf-8", newline="")
        per_record_writer = csv.writer(per_record_handle)
        per_record_writer.writerow(
            ["hot_percent", "worker_threads", "cache_type", "cache_scope", "cache_entries", "repetition", "latency_us"]
        )

    for params, hot_percent, repetition, rep_dir in iter_rep_dirs(results_dir):
        latency_csv = rep_dir / "latency_timings.csv"
        if not latency_csv.exists():
            print(f"Skipping {rep_dir}: no latency_timings.csv", file=sys.stderr)
            continue
        latencies = read_latencies(latency_csv)
        if not latencies:
            print(f"Skipping {rep_dir}: latency_timings.csv holds no latency rows", file=sys.stderr)
            continue

        cache_stats = read_cache_stats(rep_dir / "_run.log")
        hits, misses = cache_stats if cache_stats is not None else (None, None)

        meta = [
            hot_percent,
            params.get("worker_threads", ""),
            params.get("cache_type", ""),
            params.get("cache_scope", ""),
            params.get("cache_entries", ""),
            repetition,
        ]
        if per_record_writer is not None:
            for latency in latencies:
                per_record_writer.writerow(meta + [latency])

        ordered = sorted(latencies)
        summary_rows.append(
            {
                "hot_percent": hot_percent,
                "worker_threads": params.get("worker_threads", ""),
                "cache_type": params.get("cache_type", ""),
                "cache_scope": params.get("cache_scope", ""),
                "cache_entries": params.get("cache_entries", ""),
                "repetition": repetition,
                "records": len(ordered),
                "latency_us_mean": round(sum(ordered) / len(ordered), 3),
                "latency_us_p50": percentile(ordered, 0.50),
                "latency_us_p90": percentile(ordered, 0.90),
                "latency_us_p95": percentile(ordered, 0.95),
                "latency_us_p99": percentile(ordered, 0.99),
                "latency_us_max": ordered[-1],
                "cache_hits": hits if hits is not None else "",
                "cache_misses": misses if misses is not None else "",
                "observed_hit_rate": round(hits / (hits + misses), 6) if hits is not None and hits + misses > 0 else "",
            }
        )

    if per_record_handle is not None:
        per_record_handle.close()
        print(f"Wrote per-record CSV to {args.per_record}")

    if not summary_rows:
        print("No latency results found.", file=sys.stderr)
        return 1

    summary_rows.sort(
        key=lambda row: (
            row["hot_percent"],
            row["cache_type"],
            row["cache_scope"],
            int(row["worker_threads"] or 0),
            row["repetition"],
        )
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} summary rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

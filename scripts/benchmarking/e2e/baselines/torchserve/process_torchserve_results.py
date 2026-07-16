#!/usr/bin/env python3
"""Process TorchServe-baseline results into the same table schema as process_results.py.

The NES ``HttpSink`` writes a per-record timing CSV (``torchserve_timings.csv``) into
each repetition directory, with columns::

    seq_number,chunk_number,tuple_index,creation_ts_ms,post_ts_ms,recv_ts_ms,
    num_arrays,request_bytes,response_bytes,http_status

All timestamps are wall-clock epoch milliseconds. From them we derive, per
repetition:

  * end-to-end latency  = recv_ts_ms - creation_ts_ms  (join-output-ready -> prediction back)
  * round-trip latency  = recv_ts_ms - post_ts_ms      (the external serving hop)
  * serialization time  = post_ts_ms - creation_ts_ms  (NES buffer -> request on the wire)
  * end-to-end throughput = records / (max(recv) - min(post)) seconds

Optionally, if the runner captured a TorchServe Prometheus scrape at the start and
end of the repetition (``torchserve_metrics_start.prom`` / ``torchserve_metrics_end.prom``),
we take the counter deltas to isolate the TorchServe-internal breakdown:

  * inference_latency_us_per_req = delta(ts_inference_latency_microseconds) / delta(requests)
  * queue_latency_us_per_req     = delta(ts_queue_latency_microseconds)     / delta(requests)

Rows carry ``query_name``/``repetition`` and the expanded inference-config columns, so
TorchServe rows sit alongside the NES rows produced by process_results.py. Note the unit
caveat: NES ``end_to_end_throughput`` is tuples/s per pipeline, whereas here it is joined
records/s (one record = one joined tuple carrying N source arrays).

Usage mirrors process_results.py::

    ./process_torchserve_results.py                       # -> <results-dir>/torchserve_results.csv
    ./process_torchserve_results.py --aggregate --output-csv -
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - runtime environment dependent
    print("pandas and numpy are required. Install them with: pip install pandas numpy", file=sys.stderr)
    sys.exit(1)

# Reuse the sibling's context/inference-label/expansion helpers so the output schema matches.
import process_results as pr

TIMINGS_FILE = "torchserve_timings.csv"
METRICS_START_FILE = "torchserve_metrics_start.prom"
METRICS_END_FILE = "torchserve_metrics_end.prom"

# TorchServe Prometheus counters (cumulative) -> our delta column base names.
PROM_COUNTERS = {
    "ts_inference_requests_total": "inference_requests",
    "ts_inference_latency_microseconds": "inference_latency_us_total",
    "ts_queue_latency_microseconds": "queue_latency_us_total",
}

# Metric columns produced per repetition (everything else is a grouping key).
METRIC_COLUMNS = [
    "torchserve_records",
    "end_to_end_throughput",
    "end_to_end_latency_us",
    "e2e_latency_ms_mean",
    "e2e_latency_ms_p50",
    "e2e_latency_ms_p95",
    "e2e_latency_ms_p99",
    "roundtrip_latency_ms_mean",
    "roundtrip_latency_ms_p50",
    "roundtrip_latency_ms_p95",
    "roundtrip_latency_ms_p99",
    "serialization_ms_mean",
    "avg_request_bytes",
    "avg_response_bytes",
    "error_count",
    "error_rate",
    "inference_requests",
    "inference_latency_us_per_req",
    "queue_latency_us_per_req",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute TorchServe-baseline throughput/latency stats from HttpSink timing CSVs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory; relative paths resolve against the current "
             "working directory (default: ./results).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to write CSV output; relative paths resolve against the "
             "current working directory (default: <results-dir>/torchserve_results.csv). "
             "Pass '-' to print to stdout instead.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate results across repetitions (mean/std of each metric).",
    )
    return parser.parse_args()


def _percentiles(values: "np.ndarray", prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
    }


def parse_timings_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                parsed = {key: pr.parse_float(value) for key, value in record.items()}
                rows.append(parsed)
    except OSError:
        return []
    return rows


def summarize_timings(records: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not records:
        return None

    creation = np.array([r.get("creation_ts_ms") or 0.0 for r in records], dtype=float)
    post = np.array([r.get("post_ts_ms") or 0.0 for r in records], dtype=float)
    recv = np.array([r.get("recv_ts_ms") or 0.0 for r in records], dtype=float)
    status = np.array([r.get("http_status") or 0.0 for r in records], dtype=float)
    req_bytes = np.array([r.get("request_bytes") or 0.0 for r in records], dtype=float)
    resp_bytes = np.array([r.get("response_bytes") or 0.0 for r in records], dtype=float)

    count = len(records)
    roundtrip = recv - post
    serialization = post - creation
    # creation_ts may be unset (0) on some buffers; only use valid ones for end-to-end.
    valid_e2e = creation > 0
    e2e = (recv - creation)[valid_e2e]

    # Wall-clock span of the repetition, measured on the request timeline.
    span_s = (float(np.max(recv)) - float(np.min(post))) / 1000.0

    metrics: Dict[str, float] = {
        "torchserve_records": float(count),
        "avg_request_bytes": float(np.mean(req_bytes)),
        "avg_response_bytes": float(np.mean(resp_bytes)),
        "error_count": float(np.count_nonzero(status != 200)),
        "error_rate": float(np.count_nonzero(status != 200) / count),
        "serialization_ms_mean": float(np.mean(serialization)),
    }
    if span_s > 0:
        metrics["end_to_end_throughput"] = count / span_s
    metrics.update(_percentiles(roundtrip, "roundtrip_latency_ms"))
    metrics.update(_percentiles(e2e, "e2e_latency_ms"))
    if e2e.size:
        # Schema-parity column (microseconds), comparable to NES end_to_end_latency_us.
        metrics["end_to_end_latency_us"] = float(np.mean(e2e)) * 1000.0
    return metrics


def parse_prometheus(path: Path) -> Dict[str, float]:
    """Sum each counter of interest across all label sets in a Prometheus text scrape."""
    totals: Dict[str, float] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return totals
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: metric_name{labels...} value   (labels optional)
        brace = line.find("{")
        space = line.rfind(" ")
        if space < 0:
            continue
        name = line[:brace] if brace >= 0 else line[:space]
        name = name.strip()
        base = None
        for counter in PROM_COUNTERS:
            if name == counter:
                base = PROM_COUNTERS[counter]
                break
        if base is None:
            continue
        value = pr.parse_float(line[space + 1:])
        if value is None:
            continue
        totals[base] = totals.get(base, 0.0) + value
    return totals


def prometheus_delta(rep_dir: Path) -> Dict[str, float]:
    start_path = rep_dir / METRICS_START_FILE
    end_path = rep_dir / METRICS_END_FILE
    if not end_path.exists():
        return {}
    end = parse_prometheus(end_path)
    start = parse_prometheus(start_path) if start_path.exists() else {}
    if not end:
        return {}

    requests = end.get("inference_requests", 0.0) - start.get("inference_requests", 0.0)
    result: Dict[str, float] = {}
    if requests > 0:
        result["inference_requests"] = requests
        inf = end.get("inference_latency_us_total", 0.0) - start.get("inference_latency_us_total", 0.0)
        queue = end.get("queue_latency_us_total", 0.0) - start.get("queue_latency_us_total", 0.0)
        if inf > 0:
            result["inference_latency_us_per_req"] = inf / requests
        if queue > 0:
            result["queue_latency_us_per_req"] = queue / requests
    return result


def find_timing_files(results_dir: Path) -> List[Path]:
    """Timing CSVs to process. Accepts either a results directory (searched recursively for
    nested ``<config>/<query>/<rep>/torchserve_timings.csv``) or a path to a single timing
    CSV, which is processed directly."""
    if results_dir.is_file():
        return [results_dir]
    return sorted(results_dir.rglob(TIMINGS_FILE))


def context_for(results_dir: Path, timing_path: Path) -> Dict[str, str]:
    """Nested run context when the path matches ``<config>/<query>/<rep>/timings.csv``;
    otherwise a synthetic context so a single-file or top-level CSV still yields a row
    instead of being silently skipped."""
    context = pr.infer_context(results_dir, timing_path)
    if context is not None:
        return context
    return {
        "inference_config": "manual",
        "query_name": timing_path.stem,
        "repetition": "rep-01",
    }


def iter_rep_rows(results_dir: Path) -> Iterable[Dict[str, object]]:
    for timing_path in find_timing_files(results_dir):
        metrics = summarize_timings(parse_timings_csv(timing_path))
        if metrics is None:
            print(f"Skipping empty timing CSV: {timing_path}", file=sys.stderr)
            continue
        metrics.update(prometheus_delta(timing_path.parent))

        context = context_for(results_dir, timing_path)
        inference_parts = pr.parse_inference_config(context["inference_config"])
        row: Dict[str, object] = {
            "query_name": context["query_name"],
            "inference_config_param_name": inference_parts["param_name"],
            "inference_config_param_value": inference_parts["param_value"],
            "repetition": context["repetition"],
        }
        row.update(metrics)
        yield row


def compute_rows(results_dir: Path) -> "pd.DataFrame":
    rows = list(iter_rep_rows(results_dir))
    base_columns = [
        "query_name",
        "inference_config_param_name",
        "inference_config_param_value",
        "repetition",
    ]
    if not rows:
        return pd.DataFrame(columns=base_columns + METRIC_COLUMNS)
    df = pd.DataFrame(rows)
    return pr.drop_combined_inference_columns(pr.expand_inference_columns(df))


def compute_stats(results_dir: Path, aggregate: bool) -> "pd.DataFrame":
    df = compute_rows(results_dir)
    if df.empty or not aggregate:
        return df

    metric_cols = [col for col in METRIC_COLUMNS if col in df.columns]
    key_cols = [col for col in df.columns if col not in metric_cols and col != "repetition"]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    aggregations = {}
    for col in metric_cols:
        aggregations[f"avg_{col}"] = (col, "mean")
        aggregations[f"std_{col}"] = (col, "std")
    return df.groupby(key_cols, dropna=False).agg(**aggregations).reset_index()


def main() -> int:
    args = parse_args()

    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = (Path.cwd() / results_dir).resolve()

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    stats = compute_stats(results_dir, aggregate=args.aggregate)

    if str(args.output_csv) == "-":
        if stats.empty:
            print("No TorchServe timing data found.")
        else:
            print(stats.to_string(index=False))
        return 0

    base_dir = results_dir.parent if results_dir.is_file() else results_dir
    output_path = args.output_csv or (base_dir / "torchserve_results.csv")
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_path, index=False)
    print(f"Wrote {len(stats)} row(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

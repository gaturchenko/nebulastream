#!/usr/bin/env python3
"""Process systest results and compute throughput statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - runtime environment dependent
    print("pandas is required. Install it with: pip install pandas", file=sys.stderr)
    sys.exit(1)

DEFAULT_INFERENCE_PARAMS = {
    "use_batch_deduplication": "false",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_results = root / "scripts" / "benchmarking" / "e2e" / "results"

    parser = argparse.ArgumentParser(
        description="Compute throughput stats from systest result JSON files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help=f"Results directory (default: {default_results}).",
    )
    default_output = default_results / "results"
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_output,
        help=f"Path to write CSV output (default: {default_output}).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate results across repetitions (mean/std throughput, mean log metrics).",
    )
    return parser.parse_args()


def parse_event_throughput(event: Dict[str, object]) -> Optional[float]:
    dur = event.get("dur")
    tuples = event.get("tuples")
    if dur is None or tuples is None:
        return None
    try:
        dur_val = float(dur)
        tuples_val = float(tuples)
    except (TypeError, ValueError):
        return None
    if dur_val <= 0:
        return None
    return tuples_val * 1_000_000.0 / dur_val


def infer_context(results_dir: Path, json_path: Path) -> Optional[Dict[str, str]]:
    try:
        relative = json_path.relative_to(results_dir)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 4:
        return None
    return {
        "inference_config": parts[0],
        "query_name": parts[1],
        "repetition": parts[2],
    }


def parse_inference_config(config_label: str) -> Dict[str, str]:
    prefixes = (
        "worker.default_query_execution.inference.",
        "inference.",
    )
    entries = config_label.split("__") if config_label else []
    names: List[str] = []
    values: List[str] = []

    for entry in entries:
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        names.append(key)
        values.append(value)

    if not names:
        return {"param_name": "", "param_value": ""}

    return {"param_name": "__".join(names), "param_value": "__".join(values)}


def iter_pipeline_rows(results_dir: Path) -> Iterable[Dict[str, object]]:
    for json_path in results_dir.rglob("*.json"):
        context = infer_context(results_dir, json_path)
        if context is None:
            print(f"Skipping unexpected JSON path: {json_path}", file=sys.stderr)
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {json_path}", file=sys.stderr)
            continue

        events = payload.get("traceEvents")
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("cat") != "pipeline":
                continue
            throughput = parse_event_throughput(event)
            if throughput is None:
                continue
            args = event.get("args")
            pipeline_id = None
            if isinstance(args, dict):
                pipeline_id = args.get("pipeline_id")
            inference_parts = parse_inference_config(context["inference_config"])

            yield {
                "query_name": context["query_name"],
                "inference_config_param_name": inference_parts["param_name"],
                "inference_config_param_value": inference_parts["param_value"],
                "pipeline_id": pipeline_id,
                "repetition": context["repetition"],
                "throughput": throughput,
            }


def parse_log_payload(log_path: Path) -> Optional[Dict[str, object]]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line or "{" not in line:
            continue
        start = line.rfind("{")
        candidate = line[start:]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def iter_log_rows(results_dir: Path) -> Iterable[Dict[str, object]]:
    deduped: Dict[Tuple[str, str, str, object, str], Dict[str, object]] = {}
    for log_path in results_dir.rglob("*.log"):
        if log_path.name == "latest.log":
            continue
        context = infer_context(results_dir, log_path)
        if context is None:
            print(f"Skipping unexpected log path: {log_path}", file=sys.stderr)
            continue
        payload = parse_log_payload(log_path)
        if not payload:
            continue
        pipeline_id = payload.get("pipeline_id")
        if pipeline_id is None:
            continue
        inference_parts = parse_inference_config(context["inference_config"])
        row: Dict[str, object] = {
            "query_name": context["query_name"],
            "inference_config_param_name": inference_parts["param_name"],
            "inference_config_param_value": inference_parts["param_value"],
            "pipeline_id": pipeline_id,
            "repetition": context["repetition"],
        }
        for key, value in payload.items():
            if key == "pipeline_id":
                continue
            row[key] = value
        dedupe_key = (
            row["query_name"],
            row["inference_config_param_name"],
            row["inference_config_param_value"],
            row["pipeline_id"],
            row["repetition"],
        )
        try:
            mtime = log_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        existing = deduped.get(dedupe_key)
        if existing is None or mtime >= existing["__mtime"]:
            deduped[dedupe_key] = {"__mtime": mtime, "row": row}

    for entry in deduped.values():
        yield entry["row"]


def compute_throughput_stats(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_pipeline_rows(results_dir))
    if not rows:
        return pd.DataFrame(
            columns=[
                "query_name",
                "inference_config_param_name",
                "inference_config_param_value",
                "pipeline_id",
                "avg_throughput",
                "std_throughput",
            ]
        )

    df = pd.DataFrame(rows)
    if "repetition" in df.columns:
        df = df.drop(columns=["repetition"])
    return (
        df.groupby(
            ["query_name", "inference_config_param_name", "inference_config_param_value", "pipeline_id"],
            dropna=False,
        )
        .agg(
            avg_throughput=("throughput", "mean"),
            std_throughput=("throughput", "std"),
        )
        .reset_index()
    )


def compute_throughput_rows(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_pipeline_rows(results_dir))
    if not rows:
        return pd.DataFrame(
            columns=[
                "query_name",
                "inference_config_param_name",
                "inference_config_param_value",
                "pipeline_id",
                "repetition",
                "throughput",
            ]
        )
    return pd.DataFrame(rows)


def compute_log_stats(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_log_rows(results_dir))
    if not rows:
        return pd.DataFrame(
            columns=[
                "query_name",
                "inference_config_param_name",
                "inference_config_param_value",
                "pipeline_id",
            ]
        )

    df = pd.DataFrame(rows)
    if "repetition" in df.columns:
        df = df.drop(columns=["repetition"])
    base_cols = {
        "query_name",
        "inference_config_param_name",
        "inference_config_param_value",
        "pipeline_id",
    }
    metric_cols = [col for col in df.columns if col not in base_cols]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby(list(base_cols), dropna=False).mean(numeric_only=True).reset_index()
    return grouped


def compute_log_rows(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_log_rows(results_dir))
    if not rows:
        return pd.DataFrame(
            columns=[
                "query_name",
                "inference_config_param_name",
                "inference_config_param_value",
                "pipeline_id",
                "repetition",
            ]
        )
    return pd.DataFrame(rows)


def compute_stats(results_dir: Path, aggregate: bool) -> "pd.DataFrame":
    if aggregate:
        throughput_stats = compute_throughput_stats(results_dir)
        log_stats = compute_log_stats(results_dir)
    else:
        throughput_stats = compute_throughput_rows(results_dir)
        log_stats = compute_log_rows(results_dir)

    if throughput_stats.empty:
        expanded = expand_inference_columns(throughput_stats)
        return drop_combined_inference_columns(expanded)

    if log_stats.empty:
        expanded = expand_inference_columns(throughput_stats)
        return drop_combined_inference_columns(expanded)

    merge_key_candidates = [
        "query_name",
        "inference_config_param_name",
        "inference_config_param_value",
        "pipeline_id",
    ]
    if not aggregate:
        merge_key_candidates.append("repetition")
    merge_keys = [
        col
        for col in merge_key_candidates
        if col in throughput_stats.columns and col in log_stats.columns
    ]
    if not merge_keys:
        expanded = expand_inference_columns(throughput_stats)
        return drop_combined_inference_columns(expanded)

    merged = throughput_stats.merge(log_stats, on=merge_keys, how="inner")
    merged = expand_inference_columns(merged)
    return drop_combined_inference_columns(merged)


def expand_inference_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    if df.empty:
        return df

    param_maps: List[Dict[str, str]] = []
    all_params = set(DEFAULT_INFERENCE_PARAMS)
    for name_entry, value_entry in zip(
        df["inference_config_param_name"], df["inference_config_param_value"]
    ):
        names = str(name_entry).split("__") if name_entry else []
        values = str(value_entry).split("__") if value_entry else []
        mapping: Dict[str, str] = {}
        for name, value in zip(names, values):
            mapping[name] = value
            all_params.add(name)
        param_maps.append(mapping)

    ordered_params = sorted(all_params)
    for param in ordered_params:
        default_value = DEFAULT_INFERENCE_PARAMS.get(param, "")
        df[param] = [mapping.get(param, default_value) for mapping in param_maps]

    return df


def drop_combined_inference_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    columns_to_drop = [
        col
        for col in ("inference_config_param_name", "inference_config_param_value")
        if col in df.columns
    ]
    if not columns_to_drop:
        return df
    return df.drop(columns=columns_to_drop)


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = (repo_root() / results_dir).resolve()

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    stats = compute_stats(results_dir, aggregate=args.aggregate)

    if args.output_csv:
        output_path = args.output_csv
        if not output_path.is_absolute():
            output_path = (repo_root() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(output_path, index=False)
    else:
        if stats.empty:
            print("No throughput data found.")
        else:
            print(stats.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Process systest results and compute throughput and latency statistics.

Pipeline-level metrics are computed from Chrome trace events with
``cat == "pipeline"``. Throughput uses each pipeline completion event's tuple
count and event duration:

    pipeline throughput = event["tuples"] * 1_000_000 / event["dur"]

The multiplier converts from microseconds to seconds, so throughput is in
tuples per second. Pipeline latency uses completion events that expose
``task_span_us`` and ``tuples``:

    pipeline latency = event["task_span_us"] / event["tuples"]

End-to-end metrics are computed per inference configuration, query, and
repetition by combining all trace files in that group. The preferred
end-to-end duration comes from the ``systest-performance.json`` sidecar written
by ``run_systests.py`` from systest's ``--show-query-performance`` output. That
runtime follows systest's query-status metric:

    end-to-end duration = coalesced stop timestamp - coalesced running timestamp

If the sidecar is missing, the script falls back to the sum of the query
durations from the trace files; a trace query duration is the largest matching
``cat == "query"`` begin/end span in one trace. The end-to-end tuple count is
the largest pipeline completion tuple count seen in the group. End-to-end
throughput is then:

    end-to-end throughput = end_to_end_tuples * 1_000_000 / end_to_end_duration_us

End-to-end latency is the sum of the pipeline latencies in the group when
pipeline latency data is available. If not, it falls back to:

    end-to-end latency = end_to_end_duration_us / end_to_end_tuples

With ``--aggregate``, the script reports means and standard deviations of these
per-repetition values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
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
SOURCE_NAME_PATTERN = re.compile(r"LogicalSource\(name:\s*([^,\s)]+)")
SYSTEST_PERFORMANCE_FILE = "systest-performance.json"
TASK_METRIC_COLUMNS = [
    "pipeline_tuples",
    "pipeline_task_count",
    "pipeline_task_duration_us",
    "pipeline_avg_task_duration_us",
    "pipeline_task_span_us",
]
END_TO_END_METRIC_COLUMNS = [
    "end_to_end_tuples",
    "end_to_end_duration_us",
    "end_to_end_throughput",
    "end_to_end_latency_us",
]
# The LatencySink writes a per-record CSV with these header columns (see LatencySink.cpp). We detect
# such files by header signature so we never confuse them with the TorchServe HttpSink timing CSV.
SINK_LATENCY_FILE_SIGNATURE = {"ingest_ts_us", "recv_ts_us", "latency_us"}
SINK_LATENCY_COLUMN = "sink_latency_us"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_results = root / "scripts" / "benchmarking" / "e2e" / "results"

    parser = argparse.ArgumentParser(
        description="Compute throughput and latency stats from systest result JSON files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help=f"Results directory (default: {default_results}).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to write CSV output (default: <results-dir>/results.csv). "
             "Pass '-' to print to stdout instead.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate results across repetitions (mean/std throughput, mean log metrics).",
    )
    return parser.parse_args()


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_systest_performance_duration_us(payload: object) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    duration_us = parse_float(payload.get("end_to_end_duration_us"))
    if duration_us is not None and duration_us > 0:
        return duration_us

    duration_s = parse_float(payload.get("end_to_end_duration_s"))
    if duration_s is not None and duration_s > 0:
        return duration_s * 1_000_000.0
    return None


def parse_event_throughput(event: Dict[str, object]) -> Optional[float]:
    dur_val = parse_float(event.get("dur"))
    tuples_val = parse_float(event.get("tuples"))
    if dur_val is None or tuples_val is None:
        return None
    if dur_val <= 0:
        return None
    return tuples_val * 1_000_000.0 / dur_val


def parse_query_duration_us(events: List[object]) -> Optional[float]:
    starts: Dict[Tuple[object, object, object], float] = {}
    durations: List[float] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("cat") != "query":
            continue
        name = event.get("name")
        if not isinstance(name, str) or not name.startswith("Query "):
            continue
        timestamp = parse_float(event.get("ts"))
        if timestamp is None:
            continue

        key = (name, event.get("pid"), event.get("tid"))
        phase = event.get("ph")
        if phase == "B":
            starts[key] = timestamp
        elif phase == "E":
            start = starts.get(key)
            if start is not None and timestamp >= start:
                durations.append(timestamp - start)

    if not durations:
        return None
    return max(durations)


def parse_combined_query_duration_us(
        trace_events: Iterable[List[object]]
) -> Optional[float]:
    durations = [
        duration
        for duration in (parse_query_duration_us(events) for events in trace_events)
        if duration is not None and duration > 0
    ]
    if not durations:
        return None
    return sum(durations)


def parse_pipeline_wall_clock_duration_us(events: List[object]) -> Optional[float]:
    starts: List[float] = []
    ends: List[float] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("cat") != "pipeline":
            continue
        timestamp = parse_float(event.get("ts"))
        if timestamp is None:
            continue
        if event.get("ph") == "B":
            starts.append(timestamp)
        elif event.get("ph") == "E":
            ends.append(timestamp)

    if not starts or not ends:
        return None
    duration = max(ends) - min(starts)
    if duration <= 0:
        return None
    return duration


def parse_end_to_end_tuple_count(events: List[object]) -> Optional[float]:
    tuple_counts: List[float] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("cat") != "pipeline" or event.get("ph") != "E":
            continue
        tuples = parse_float(event.get("tuples"))
        if tuples is not None and tuples > 0:
            tuple_counts.append(tuples)

    if not tuple_counts:
        return None
    return max(tuple_counts)


def parse_end_to_end_metrics(events: List[object]) -> Optional[Dict[str, float]]:
    duration_us = parse_query_duration_us(events)
    tuples = parse_end_to_end_tuple_count(events)
    if duration_us is None or duration_us <= 0 or tuples is None or tuples <= 0:
        return None

    pipeline_latencies = [
        latency
        for latency in (parse_float(row.get("latency_us")) for row in parse_pipeline_latency_rows(events))
        if latency is not None
    ]

    return {
        "end_to_end_tuples": tuples,
        "end_to_end_duration_us": duration_us,
        "end_to_end_throughput": tuples * 1_000_000.0 / duration_us,
        "end_to_end_latency_us": sum(pipeline_latencies) if pipeline_latencies else duration_us / tuples,
    }


def parse_combined_end_to_end_metrics(
        trace_events: Iterable[List[object]],
        duration_us_override: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    event_lists = list(trace_events)
    duration_us = duration_us_override
    if duration_us is None:
        duration_us = parse_combined_query_duration_us(event_lists)
    tuple_counts = [
        tuples
        for tuples in (parse_end_to_end_tuple_count(events) for events in event_lists)
        if tuples is not None and tuples > 0
    ]
    if duration_us is None or duration_us <= 0 or not tuple_counts:
        return None

    pipeline_latencies = [
        latency
        for events in event_lists
        for latency in (
            parse_float(row.get("latency_us")) for row in parse_pipeline_latency_rows(events)
        )
        if latency is not None
    ]
    tuples = max(tuple_counts)

    return {
        "end_to_end_tuples": tuples,
        "end_to_end_duration_us": duration_us,
        "end_to_end_throughput": tuples * 1_000_000.0 / duration_us,
        "end_to_end_latency_us": sum(pipeline_latencies) if pipeline_latencies else duration_us / tuples,
    }


def parse_pipeline_latency_rows(events: List[object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("cat") != "pipeline" or event.get("ph") != "E":
            continue
        task_span_us = parse_float(event.get("task_span_us"))
        tuples = parse_float(event.get("tuples"))
        if task_span_us is None or tuples is None or tuples <= 0:
            continue

        args = event.get("args")
        pipeline_id = None
        if isinstance(args, dict):
            pipeline_id = args.get("pipeline_id")
        rows.append({"pipeline_id": pipeline_id, "latency_us": task_span_us / tuples})

    return rows


def parse_pipeline_task_event_metric_rows(events: List[object]) -> List[Dict[str, object]]:
    task_starts: Dict[Tuple[object, object], float] = {}
    completions_by_pipeline: Dict[object, List[Dict[str, object]]] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("cat") != "task":
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        timestamp = parse_float(event.get("ts"))
        if timestamp is None:
            continue

        phase = event.get("ph")
        pipeline_id = args.get("pipeline_id")
        task_id = args.get("task_id")
        if phase == "B" and pipeline_id is not None and task_id is not None:
            task_starts[(pipeline_id, task_id)] = timestamp
            continue

        if phase == "E" and pipeline_id is not None and task_id is not None:
            duration = parse_float(event.get("dur"))
            start_ts = task_starts.get((pipeline_id, task_id))
            if start_ts is None and duration is not None:
                start_ts = timestamp - duration
            completion = {
                "task_id": task_id,
                "start_ts": start_ts,
                "end_ts": timestamp,
                "duration_us": duration,
            }
            completions_by_pipeline.setdefault(pipeline_id, []).append(completion)
            continue

    rows: List[Dict[str, object]] = []
    pipeline_ids = set(completions_by_pipeline)
    for pipeline_id in sorted(pipeline_ids, key=str):
        completions = completions_by_pipeline.get(pipeline_id, [])
        durations = [
            duration
            for duration in (parse_float(completion.get("duration_us")) for completion in completions)
            if duration is not None
        ]
        starts = [
            start
            for start in (parse_float(completion.get("start_ts")) for completion in completions)
            if start is not None
        ]
        ends = [
            end
            for end in (parse_float(completion.get("end_ts")) for completion in completions)
            if end is not None
        ]

        row: Dict[str, object] = {
            "pipeline_id": pipeline_id,
            "pipeline_task_count": len(completions),
        }
        if durations:
            row["pipeline_task_duration_us"] = sum(durations)
            row["pipeline_avg_task_duration_us"] = sum(durations) / len(durations)
        if starts and ends:
            row["pipeline_task_span_us"] = max(ends) - min(starts)
        rows.append(row)

    return rows


def parse_pipeline_aggregate_metric_rows(events: List[object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    field_map = {
        "dur": "pipeline_task_duration_us",
        "tuples": "pipeline_tuples",
        "task_count": "pipeline_task_count",
        "task_span_us": "pipeline_task_span_us",
    }

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("cat") != "pipeline" or event.get("ph") != "E":
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        pipeline_id = args.get("pipeline_id")
        if pipeline_id is None or "task_count" not in event:
            continue

        row: Dict[str, object] = {"pipeline_id": pipeline_id}
        for event_field, row_field in field_map.items():
            value = parse_float(event.get(event_field))
            if value is not None:
                row[row_field] = value
        task_count = parse_float(row.get("pipeline_task_count"))
        duration = parse_float(row.get("pipeline_task_duration_us"))
        if task_count is not None and task_count > 0 and duration is not None:
            row["pipeline_avg_task_duration_us"] = duration / task_count
        rows.append(row)

    return rows


def parse_pipeline_metric_rows(events: List[object]) -> List[Dict[str, object]]:
    task_event_rows = parse_pipeline_task_event_metric_rows(events)
    if task_event_rows:
        return task_event_rows
    return parse_pipeline_aggregate_metric_rows(events)


def parse_source_name(events: List[object]) -> Optional[str]:
    for event in events:
        if not isinstance(event, dict):
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        query = args.get("query")
        if not isinstance(query, str):
            continue
        match = SOURCE_NAME_PATTERN.search(query)
        if match:
            return match.group(1)
    return None


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
        source_name = parse_source_name(events)

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
                "source_name": source_name,
                "throughput": throughput,
            }


def iter_latency_rows(results_dir: Path) -> Iterable[Dict[str, object]]:
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

        inference_parts = parse_inference_config(context["inference_config"])
        source_name = parse_source_name(events)
        for latency_row in parse_pipeline_latency_rows(events):
            yield {
                "query_name": context["query_name"],
                "source_name": source_name,
                "inference_config_param_name": inference_parts["param_name"],
                "inference_config_param_value": inference_parts["param_value"],
                "pipeline_id": latency_row["pipeline_id"],
                "repetition": context["repetition"],
                "latency_us": latency_row["latency_us"],
            }


def iter_task_metric_rows(results_dir: Path) -> Iterable[Dict[str, object]]:
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

        inference_parts = parse_inference_config(context["inference_config"])
        source_name = parse_source_name(events)
        for metric_row in parse_pipeline_metric_rows(events):
            row = {
                "query_name": context["query_name"],
                "source_name": source_name,
                "inference_config_param_name": inference_parts["param_name"],
                "inference_config_param_value": inference_parts["param_value"],
                "repetition": context["repetition"],
            }
            row.update(metric_row)
            yield row


def iter_end_to_end_rows(results_dir: Path) -> Iterable[Dict[str, object]]:
    traces_by_context: Dict[Tuple[str, str, str], List[List[object]]] = {}
    duration_by_context: Dict[Tuple[str, str, str], float] = {}

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

        context_key = (
            context["inference_config"],
            context["query_name"],
            context["repetition"],
        )
        if json_path.name == SYSTEST_PERFORMANCE_FILE:
            duration_us = parse_systest_performance_duration_us(payload)
            if duration_us is not None:
                duration_by_context[context_key] = duration_us
            continue

        events = payload.get("traceEvents")
        if not isinstance(events, list):
            continue

        traces_by_context.setdefault(context_key, []).append(events)

    for inference_config, query_name, repetition in sorted(traces_by_context):
        metrics = parse_combined_end_to_end_metrics(
            traces_by_context[(inference_config, query_name, repetition)],
            duration_by_context.get((inference_config, query_name, repetition)),
        )
        if metrics is None:
            continue

        inference_parts = parse_inference_config(inference_config)
        row: Dict[str, object] = {
            "query_name": query_name,
            "inference_config_param_name": inference_parts["param_name"],
            "inference_config_param_value": inference_parts["param_value"],
            "repetition": repetition,
        }
        row.update(metrics)
        yield row


# The query compiler dumps the pipelined plan (PipeliningPhase.cpp:449). Each pipeline prints as
# `Pipeline(ID(n), Provider(...))` followed by its operator chain, and the printed ID(n) equals the
# `pipeline_id` in the trace events. The model-inference pipeline is the one whose chain contains an
# *InferModelPhysicalOperator (InferModel / BatchInferModel / CacheInferModel / CachedInferModel /
# BatchCacheInferModel -- all share the "InferModel" substring). UDF queries have no such operator
# (the UDF is a MapPhysicalOperator), so this returns None and the caller leaves the run unflagged
# for manual assignment.
_PIPELINE_ID_PATTERN = re.compile(r"Pipeline\(ID\((\d+)\)")
_INFERENCE_OPERATOR_PATTERN = re.compile(r"InferModelPhysicalOperator")


def parse_inference_pipeline_id(log_path: Path) -> Optional[int]:
    """Return the pipeline id of the model-inference pipeline from a run's plan dump, or None."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    current_pipeline_id: Optional[int] = None
    for line in lines:
        id_match = _PIPELINE_ID_PATTERN.search(line)
        if id_match:
            current_pipeline_id = int(id_match.group(1))
            continue
        if current_pipeline_id is not None and _INFERENCE_OPERATOR_PATTERN.search(line):
            return current_pipeline_id
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
        inference_pipeline_id = parse_inference_pipeline_id(log_path)
        if inference_pipeline_id is None:
            continue
        inference_parts = parse_inference_config(context["inference_config"])
        row: Dict[str, object] = {
            "query_name": context["query_name"],
            "inference_config_param_name": inference_parts["param_name"],
            "inference_config_param_value": inference_parts["param_value"],
            "pipeline_id": inference_pipeline_id,
            "repetition": context["repetition"],
            "is_inference_pipeline": True,
        }
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
                "source_name",
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
            [
                "query_name",
                "source_name",
                "inference_config_param_name",
                "inference_config_param_value",
                "pipeline_id",
            ],
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
                "source_name",
                "inference_config_param_name",
                "inference_config_param_value",
                "pipeline_id",
                "repetition",
                "throughput",
            ]
        )
    return pd.DataFrame(rows)


def compute_latency_stats(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_latency_rows(results_dir))
    if not rows:
        return pd.DataFrame(
            columns=[
                "query_name",
                "source_name",
                "inference_config_param_name",
                "inference_config_param_value",
                "pipeline_id",
                "avg_latency_us",
                "std_latency_us",
            ]
        )

    df = pd.DataFrame(rows)
    if "repetition" in df.columns:
        df = df.drop(columns=["repetition"])
    return (
        df.groupby(
            [
                "query_name",
                "source_name",
                "inference_config_param_name",
                "inference_config_param_value",
                "pipeline_id",
            ],
            dropna=False,
        )
        .agg(
            avg_latency_us=("latency_us", "mean"),
            std_latency_us=("latency_us", "std"),
        )
        .reset_index()
    )


def compute_latency_rows(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_latency_rows(results_dir))
    if not rows:
        return pd.DataFrame(
            columns=[
                "query_name",
                "source_name",
                "inference_config_param_name",
                "inference_config_param_value",
                "pipeline_id",
                "repetition",
                "latency_us",
            ]
        )
    return pd.DataFrame(rows)


def compute_task_metric_stats(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_task_metric_rows(results_dir))
    columns = [
        "query_name",
        "source_name",
        "inference_config_param_name",
        "inference_config_param_value",
        "pipeline_id",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    if "repetition" in df.columns:
        df = df.drop(columns=["repetition"])
    for col in TASK_METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    aggregations = {}
    for col in TASK_METRIC_COLUMNS:
        if col in df.columns:
            aggregations[f"avg_{col}"] = (col, "mean")
            aggregations[f"std_{col}"] = (col, "std")

    if not aggregations:
        return pd.DataFrame(columns=columns)

    return (
        df.groupby(columns, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )


def compute_task_metric_rows(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_task_metric_rows(results_dir))
    columns = [
        "query_name",
        "source_name",
        "inference_config_param_name",
        "inference_config_param_value",
        "pipeline_id",
        "repetition",
        *TASK_METRIC_COLUMNS,
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def compute_end_to_end_stats(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_end_to_end_rows(results_dir))
    columns = [
        "query_name",
        "inference_config_param_name",
        "inference_config_param_value",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    if "repetition" in df.columns:
        df = df.drop(columns=["repetition"])
    for col in END_TO_END_METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    aggregations = {}
    for col in END_TO_END_METRIC_COLUMNS:
        if col in df.columns:
            aggregations[f"avg_{col}"] = (col, "mean")
            aggregations[f"std_{col}"] = (col, "std")

    if not aggregations:
        return pd.DataFrame(columns=columns)

    return (
        df.groupby(columns, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )


def compute_end_to_end_rows(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_end_to_end_rows(results_dir))
    columns = [
        "query_name",
        "inference_config_param_name",
        "inference_config_param_value",
        "repetition",
        *END_TO_END_METRIC_COLUMNS,
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def iter_sink_latency_rows(results_dir: Path) -> Iterable[Dict[str, object]]:
    """Yield one row per record from every LatencySink CSV under the results tree.

    LatencySink writes a real per-record latency (recv_ts_us - ingest_ts_us) and discards the
    payload, so this is the NES-native counterpart to the throughput-derived end_to_end_latency_us.
    Files are identified by header signature, so the TorchServe HttpSink CSV is never picked up here.
    """
    for csv_path in results_dir.rglob("*.csv"):
        context = infer_context(results_dir, csv_path)
        if context is None:
            continue
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or not SINK_LATENCY_FILE_SIGNATURE.issubset(set(reader.fieldnames)):
                    continue
                inference_parts = parse_inference_config(context["inference_config"])
                for record in reader:
                    latency = parse_float(record.get("latency_us"))
                    if latency is None:
                        continue
                    yield {
                        "query_name": context["query_name"],
                        "inference_config_param_name": inference_parts["param_name"],
                        "inference_config_param_value": inference_parts["param_value"],
                        "repetition": context["repetition"],
                        SINK_LATENCY_COLUMN: latency,
                    }
        except OSError:
            continue


def _sink_latency_aggregations() -> Dict[str, Tuple[str, object]]:
    col = SINK_LATENCY_COLUMN
    return {
        f"avg_{col}": (col, "mean"),
        f"std_{col}": (col, "std"),
        f"p50_{col}": (col, lambda series: series.quantile(0.50)),
        f"p95_{col}": (col, lambda series: series.quantile(0.95)),
        f"p99_{col}": (col, lambda series: series.quantile(0.99)),
        "sink_latency_count": (col, "count"),
    }


def compute_sink_latency_stats(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_sink_latency_rows(results_dir))
    columns = [
        "query_name",
        "inference_config_param_name",
        "inference_config_param_value",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    if "repetition" in df.columns:
        df = df.drop(columns=["repetition"])
    df[SINK_LATENCY_COLUMN] = pd.to_numeric(df[SINK_LATENCY_COLUMN], errors="coerce")
    return (
        df.groupby(columns, dropna=False)
        .agg(**_sink_latency_aggregations())
        .reset_index()
    )


def compute_sink_latency_rows(results_dir: Path) -> "pd.DataFrame":
    rows: List[Dict[str, object]] = list(iter_sink_latency_rows(results_dir))
    columns = [
        "query_name",
        "inference_config_param_name",
        "inference_config_param_value",
        "repetition",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df[SINK_LATENCY_COLUMN] = pd.to_numeric(df[SINK_LATENCY_COLUMN], errors="coerce")
    return (
        df.groupby(columns, dropna=False)
        .agg(**_sink_latency_aggregations())
        .reset_index()
    )


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
        latency_stats = compute_latency_stats(results_dir)
        task_metric_stats = compute_task_metric_stats(results_dir)
        end_to_end_stats = compute_end_to_end_stats(results_dir)
        sink_latency_stats = compute_sink_latency_stats(results_dir)
        log_stats = compute_log_stats(results_dir)
    else:
        throughput_stats = compute_throughput_rows(results_dir)
        latency_stats = compute_latency_rows(results_dir)
        task_metric_stats = compute_task_metric_rows(results_dir)
        end_to_end_stats = compute_end_to_end_rows(results_dir)
        sink_latency_stats = compute_sink_latency_rows(results_dir)
        log_stats = compute_log_rows(results_dir)

    if throughput_stats.empty:
        if not task_metric_stats.empty:
            merged = task_metric_stats
        elif not latency_stats.empty:
            merged = latency_stats
        else:
            merged = end_to_end_stats
        if (
                not merged.empty
                and not latency_stats.empty
                and not task_metric_stats.empty
        ):
            latency_merge_key_candidates = [
                "query_name",
                "source_name",
                "inference_config_param_name",
                "inference_config_param_value",
                "pipeline_id",
            ]
            if not aggregate:
                latency_merge_key_candidates.append("repetition")
            latency_merge_keys = [
                col
                for col in latency_merge_key_candidates
                if col in merged.columns and col in latency_stats.columns
            ]
            if latency_merge_keys:
                merged = merged.merge(latency_stats, on=latency_merge_keys, how="left")
        if not merged.empty and not end_to_end_stats.empty and merged is not end_to_end_stats:
            end_to_end_merge_key_candidates = [
                "query_name",
                "source_name",
                "inference_config_param_name",
                "inference_config_param_value",
            ]
            if not aggregate:
                end_to_end_merge_key_candidates.append("repetition")
            end_to_end_merge_keys = [
                col
                for col in end_to_end_merge_key_candidates
                if col in merged.columns and col in end_to_end_stats.columns
            ]
            if end_to_end_merge_keys:
                merged = merged.merge(end_to_end_stats, on=end_to_end_merge_keys, how="left")
        if not sink_latency_stats.empty:
            sink_latency_merge_key_candidates = [
                "query_name",
                "source_name",
                "inference_config_param_name",
                "inference_config_param_value",
            ]
            if not aggregate:
                sink_latency_merge_key_candidates.append("repetition")
            sink_latency_merge_keys = [
                col
                for col in sink_latency_merge_key_candidates
                if col in merged.columns and col in sink_latency_stats.columns
            ]
            if sink_latency_merge_keys:
                merged = merged.merge(sink_latency_stats, on=sink_latency_merge_keys, how="left")
        expanded = expand_inference_columns(merged)
        return drop_combined_inference_columns(expanded)

    merged = throughput_stats

    if not latency_stats.empty:
        latency_merge_key_candidates = [
            "query_name",
            "source_name",
            "inference_config_param_name",
            "inference_config_param_value",
            "pipeline_id",
        ]
        if not aggregate:
            latency_merge_key_candidates.append("repetition")
        latency_merge_keys = [
            col
            for col in latency_merge_key_candidates
            if col in merged.columns and col in latency_stats.columns
        ]
        if latency_merge_keys:
            merged = merged.merge(latency_stats, on=latency_merge_keys, how="left")

    if not task_metric_stats.empty:
        task_metric_merge_key_candidates = [
            "query_name",
            "source_name",
            "inference_config_param_name",
            "inference_config_param_value",
            "pipeline_id",
        ]
        if not aggregate:
            task_metric_merge_key_candidates.append("repetition")
        task_metric_merge_keys = [
            col
            for col in task_metric_merge_key_candidates
            if col in merged.columns and col in task_metric_stats.columns
        ]
        if task_metric_merge_keys:
            merged = merged.merge(task_metric_stats, on=task_metric_merge_keys, how="left")

    if not end_to_end_stats.empty:
        end_to_end_merge_key_candidates = [
            "query_name",
            "source_name",
            "inference_config_param_name",
            "inference_config_param_value",
        ]
        if not aggregate:
            end_to_end_merge_key_candidates.append("repetition")
        end_to_end_merge_keys = [
            col
            for col in end_to_end_merge_key_candidates
            if col in merged.columns and col in end_to_end_stats.columns
        ]
        if end_to_end_merge_keys:
            merged = merged.merge(end_to_end_stats, on=end_to_end_merge_keys, how="left")

    if not sink_latency_stats.empty:
        sink_latency_merge_key_candidates = [
            "query_name",
            "source_name",
            "inference_config_param_name",
            "inference_config_param_value",
        ]
        if not aggregate:
            sink_latency_merge_key_candidates.append("repetition")
        sink_latency_merge_keys = [
            col
            for col in sink_latency_merge_key_candidates
            if col in merged.columns and col in sink_latency_stats.columns
        ]
        if sink_latency_merge_keys:
            merged = merged.merge(sink_latency_stats, on=sink_latency_merge_keys, how="left")

    if log_stats.empty:
        expanded = expand_inference_columns(merged)
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
        if col in merged.columns and col in log_stats.columns
    ]
    if not merge_keys:
        expanded = expand_inference_columns(merged)
        return drop_combined_inference_columns(expanded)

    # Left join (not inner): keep every pipeline row and flag only the model-inference pipeline the
    # log identified. Inner would drop all non-inference pipelines AND every UDF run (UDFs have no
    # InferModel operator, so they stay unflagged for manual assignment in post-processing).
    merged = merged.merge(log_stats, on=merge_keys, how="left")
    if "is_inference_pipeline" in merged.columns:
        merged["is_inference_pipeline"] = merged["is_inference_pipeline"].fillna(False).astype(bool)
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

    # '-' means print to stdout; otherwise default to <results-dir>/results.csv.
    if str(args.output_csv) == "-":
        if stats.empty:
            print("No result data found.")
        else:
            print(stats.to_string(index=False))
        return 0

    output_path = args.output_csv or (results_dir / "results.csv")
    if not output_path.is_absolute():
        output_path = (repo_root() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_path, index=False)
    print(f"Wrote {len(stats)} row(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

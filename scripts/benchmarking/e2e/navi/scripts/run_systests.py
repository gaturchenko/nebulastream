#!/usr/bin/env python3
"""Run nes-systests .test queries across inference config combinations on the Pi.

This is a deployment-tailored variant of scripts/benchmarking/e2e/run_systests.py.
It targets the standalone Pi project layout (no repo checkout, no cmake-build tree):

    <project-root>/                     # invoke the script from here
    |-- bin/systest
    |-- bin/config/*.yaml               # --clusterConfig
    |-- datasets/<dataset>/...          # --data
    |-- models-benchmarking/...         # referenced by absolute path inside the .test
    |-- queries/<group>/<name>.test     # --queries resolve against <root>/queries
    |-- queries/<group>/*.yaml          # inference sweep configs (--inference-config)
    |-- working-dir/                    # WIPED at the start of every systest run
    |-- results/                        # benchmark output (this script writes here)
    `-- scripts/run_systests.py         # this file

Key differences from the in-repo runner:
  * All paths derive from --project-root (default: current working directory).
  * The systest binary is <root>/bin/systest, queries live under <root>/queries.
  * The command carries --workingDir / --data / --clusterConfig / --log-path,
    which the in-repo runner does not need.
  * Results and logs are kept OUTSIDE working-dir, because systest removes and
    recreates working-dir at the start of every run (SystestExecutor.cpp).
  * Artifacts are collected from where systest actually writes them on the Pi:
    - trace_*.json          -> the process CWD (project root)
    - BenchmarkResults.json -> working-dir (written at end of run)
    - the per-run log        -> --log-path (a scratch file under results)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - runtime environment dependent
    print("PyYAML is required. Install it with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPETITIONS_DEFAULT = 5
QUERY_RETRIES = 3
INFERENCE_CONFIG_PREFIX = "worker.default_query_execution.inference."
INFERENCE_CONFIG_LABEL_PREFIX = "inference."
BATCH_SIZE_KEY = f"{INFERENCE_CONFIG_PREFIX}batch_size"
OPTIMIZER_BATCH_SIZE_KEY = "optimizer.inference_batch_size"
OPTIMIZER_BATCH_SIZE_ARG = "inference_batch_size"
OPTIMIZER_CONFIG_PREFIX = "optimizer."
OPTIMIZER_POST_JOIN_INFERENCE_STRATEGY_KEY = "optimizer.post_join_inference_strategy"
OPTIMIZER_POST_JOIN_INFERENCE_STRATEGY_ARG = "post_join_inference_strategy"
# Replaced by post_join_inference_strategy; a boolean no longer expresses the choice because the
# optimizer can also decide per operator by profiling. false -> SEQUENTIAL, true -> BATCHED.
LEGACY_POST_JOIN_KEYS = (
    "optimizer.rewrite_post_join_batch_inference",
    "rewrite_post_join_batch_inference",
)
USE_BATCH_DEDUPLICATION_KEY = f"{INFERENCE_CONFIG_PREFIX}use_batch_deduplication"
PREDICTION_CACHE_TYPE_KEY = f"{INFERENCE_CONFIG_PREFIX}prediction_cache_type"
PREDICTION_CACHE_ENTRIES_KEY = f"{INFERENCE_CONFIG_PREFIX}number_of_entries_prediction_cache"
PREDICTION_CACHE_NONE = "NONE"
DEFAULT_USE_BATCH_DEDUPLICATION = False
SYSTEST_PERFORMANCE_FILE = "systest-performance.json"
RUN_LOG_SCRATCH = "_run.log"
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
QUERY_PERFORMANCE_PATTERN = re.compile(
    r"\bPASSED\s+in\s+([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)s\b"
)


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def split_query(query: str) -> Tuple[str, str]:
    if ":" in query:
        left, right = query.rsplit(":", 1)
        if left.endswith(".test"):
            return left, f":{right}"
    return query, ""


def resolve_query(query: str, queries_dir: Path) -> str:
    """Resolve a query spec to an absolute '<abs path>.test[:NN]' string.

    Relative query paths resolve against queries_dir (the Pi's ./queries).
    """
    path_part, suffix = split_query(query)
    candidate = Path(path_part)
    if not candidate.is_absolute():
        candidate = queries_dir / path_part
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Query file not found: {candidate}")
    return f"{candidate}{suffix}"


def normalize_values(values: object) -> List[object]:
    if isinstance(values, list):
        return values
    return [values]


def normalize_inference_config(config: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}

    for key, value in config.items():
        if key in ("batch_size", OPTIMIZER_BATCH_SIZE_ARG, OPTIMIZER_BATCH_SIZE_KEY):
            key = BATCH_SIZE_KEY
        elif key in LEGACY_POST_JOIN_KEYS:
            raise SystemExit(
                f"'{key}' no longer exists. Use "
                f"'{OPTIMIZER_POST_JOIN_INFERENCE_STRATEGY_KEY}' with "
                f"[BATCHED|SEQUENTIAL|OPTIMIZER_CHOOSES] instead "
                f"(the old false -> SEQUENTIAL, true -> BATCHED)."
            )
        elif key == OPTIMIZER_POST_JOIN_INFERENCE_STRATEGY_ARG:
            key = OPTIMIZER_POST_JOIN_INFERENCE_STRATEGY_KEY
        elif key == "use_batch_deduplication":
            key = USE_BATCH_DEDUPLICATION_KEY
        normalized[key] = value

    if USE_BATCH_DEDUPLICATION_KEY not in normalized:
        normalized[USE_BATCH_DEDUPLICATION_KEY] = DEFAULT_USE_BATCH_DEDUPLICATION

    return normalized


def normalize_combination(combo: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(combo)

    if normalized.get(BATCH_SIZE_KEY) == 1:
        normalized.pop(USE_BATCH_DEDUPLICATION_KEY, None)

    cache_type = normalized.get(PREDICTION_CACHE_TYPE_KEY)
    if isinstance(cache_type, str) and cache_type.upper() == PREDICTION_CACHE_NONE:
        normalized.pop(PREDICTION_CACHE_ENTRIES_KEY, None)

    return normalized


def expand_combinations(config: Dict[str, object]) -> Iterable[Dict[str, object]]:
    if not config:
        yield {}
        return
    keys = list(config.keys())
    values = [normalize_values(config[key]) for key in keys]
    seen = set()
    for combination in itertools.product(*values):
        normalized_combo = normalize_combination(dict(zip(keys, combination)))
        combo_key = tuple((key, format_value(value)) for key, value in normalized_combo.items())
        if combo_key in seen:
            continue
        seen.add(combo_key)
        yield normalized_combo


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def sanitize_name(text: str) -> str:
    sanitized = text.replace(os.sep, "_")
    if os.altsep:
        sanitized = sanitized.replace(os.altsep, "_")
    sanitized = sanitized.replace(":", "_")
    sanitized = sanitized.replace(" ", "_")
    return sanitized


def missing_repetitions(query_dir: Path, repetitions: int) -> List[int]:
    missing: List[int] = []
    for repetition in range(1, repetitions + 1):
        rep_dir = query_dir / f"rep-{repetition:02d}"
        if not rep_dir.is_dir():
            missing.append(repetition)
    return missing


def combination_name(combo: Dict[str, object]) -> str:
    if not combo:
        return "default"
    parts: List[str] = []
    for key, value in combo.items():
        label_key = key
        if label_key.startswith(INFERENCE_CONFIG_PREFIX):
            label_key = INFERENCE_CONFIG_LABEL_PREFIX + label_key[len(INFERENCE_CONFIG_PREFIX):]
        parts.append(f"{label_key}={format_value(value)}")
    return sanitize_name("__".join(parts))


def build_command(
        systest_path: Path,
        query: str,
        params: Dict[str, object],
        *,
        working_dir: Path,
        data_dir: Path,
        cluster_config: Path,
        log_path: Path,
) -> List[str]:
    optimizer_params: Dict[str, object] = {}
    worker_params: Dict[str, object] = {}

    for key, value in params.items():
        if key == BATCH_SIZE_KEY:
            optimizer_params[OPTIMIZER_BATCH_SIZE_ARG] = value
        elif key.startswith(OPTIMIZER_CONFIG_PREFIX):
            optimizer_params[key[len(OPTIMIZER_CONFIG_PREFIX):]] = value
        else:
            worker_params[key] = value

    cmd = [
        str(systest_path),
        "--workingDir", str(working_dir),
        "--data", str(data_dir),
        "--clusterConfig", str(cluster_config),
        "--log-path", str(log_path),
        "--show-query-performance",
        "-t", query,
        "-n", "1",
    ]
    for key, value in optimizer_params.items():
        cmd.extend(["--optimizer", f"{key}={format_value(value)}"])

    cmd.append("--")
    for key, value in worker_params.items():
        cmd.append(f"--{key}={format_value(value)}")
    return cmd


def parse_query_performance_duration_us(output: str) -> float | None:
    normalized = ANSI_ESCAPE_PATTERN.sub("", output)
    matches = QUERY_PERFORMANCE_PATTERN.findall(normalized)
    if not matches:
        return None
    return sum(float(match) for match in matches) * 1_000_000.0


def write_query_performance(rep_dir: Path, duration_us: float) -> None:
    payload = {
        "end_to_end_duration_us": duration_us,
        "source": "--show-query-performance",
    }
    performance_path = rep_dir / SYSTEST_PERFORMANCE_FILE
    performance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def safe_move(src: Path, dest_dir: Path) -> None:
    dest = dest_dir / src.name
    if dest.exists():
        stem = src.stem
        suffix = src.suffix
        counter = 1
        while True:
            candidate = dest_dir / f"{stem}__{counter}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            counter += 1
    shutil.move(str(src), str(dest))


def snapshot_files(directory: Path, pattern: str) -> Dict[Path, float]:
    if not directory.exists():
        return {}
    snapshot: Dict[Path, float] = {}
    for path in directory.glob(pattern):
        try:
            snapshot[path] = path.stat().st_mtime
        except FileNotFoundError:
            continue
    return snapshot


def collect_artifacts(
        targets: List[Tuple[Path, str]],
        dest_dir: Path,
        before: List[Dict[Path, float]],
) -> None:
    """Move files that appeared/changed since `before` into dest_dir.

    `targets` is a list of (directory, glob) pairs, aligned with `before`.
    """
    for (directory, pattern), before_snapshot in zip(targets, before):
        after = snapshot_files(directory, pattern)
        for path, mtime in after.items():
            if path not in before_snapshot or mtime > before_snapshot[path]:
                safe_move(path, dest_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nes-systests queries across inference config combinations on the Pi.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root of the Pi project (default: current working directory).",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        required=True,
        help="Query specs relative to <root>/queries, e.g. ablation_cwru.test:01",
    )
    parser.add_argument(
        "--inference-config",
        required=True,
        type=Path,
        help="YAML config with parameter lists to sweep.",
    )
    parser.add_argument(
        "--systest-path",
        type=Path,
        help="Path to systest binary (default: <root>/bin/systest).",
    )
    parser.add_argument(
        "--queries-dir",
        type=Path,
        help="Directory holding .test files (default: <root>/queries).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Test data directory passed as --data (default: <root>/datasets).",
    )
    parser.add_argument(
        "--cluster-config",
        type=Path,
        help="Cluster YAML passed as --clusterConfig "
             "(default: <root>/bin/config/two-node.yaml).",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        help="systest --workingDir; WIPED each run (default: <root>/working-dir).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory to write benchmark results (default: <root>/results).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS_DEFAULT,
        help=f"Number of repetitions per combination (default: {REPETITIONS_DEFAULT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing systest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = args.project_root.resolve()
    if not root.is_dir():
        print(f"Project root not found or not a directory: {root}", file=sys.stderr)
        return 1

    systest_path = (args.systest_path or root / "bin" / "systest").resolve()
    queries_dir = (args.queries_dir or root / "queries").resolve()
    data_dir = (args.data or root / "datasets").resolve()
    cluster_config = (args.cluster_config or root / "bin" / "config" / "single-node.yaml").resolve()
    working_dir = (args.working_dir or root / "working-dir").resolve()
    results_base = (args.results_dir or root / "results").resolve()

    if not systest_path.exists():
        print(f"systest binary not found: {systest_path}", file=sys.stderr)
        return 1
    if not cluster_config.exists():
        print(f"Cluster config not found: {cluster_config}", file=sys.stderr)
        return 1
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    # working-dir must exist before the run: systest opens the log (parent must
    # exist) and then remove_all()s + recreates working-dir. We keep the log and
    # results OUTSIDE working-dir so the wipe never touches them.
    working_dir.mkdir(parents=True, exist_ok=True)
    results_base.mkdir(parents=True, exist_ok=True)
    log_scratch = results_base / RUN_LOG_SCRATCH

    inference_config = normalize_inference_config(load_yaml(args.inference_config))

    try:
        resolved_queries = [resolve_query(query, queries_dir) for query in args.queries]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if results_base.exists() and not results_base.is_dir():
        print(f"Results path exists and is not a directory: {results_base}", file=sys.stderr)
        return 1

    query_specs = []
    for query in resolved_queries:
        query_path, query_suffix = split_query(query)
        query_name = sanitize_name(Path(query_path).name + query_suffix)
        query_specs.append((query, query_name))

    all_combinations = list(expand_combinations(inference_config))
    pending_combinations = []
    for combo in all_combinations:
        combo_dir = results_base / combination_name(combo)
        if combo_dir.exists() and not combo_dir.is_dir():
            print(f"Combination results path exists and is not a directory: {combo_dir}", file=sys.stderr)
            return 1
        combo_complete = True
        for _, query_name in query_specs:
            if missing_repetitions(combo_dir / query_name, args.repetitions):
                combo_complete = False
                break
        if not combo_complete:
            pending_combinations.append(combo)

    skipped_count = len(all_combinations) - len(pending_combinations)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} already processed combination(s) from {results_base}.")
    if not pending_combinations:
        print("All combinations are already processed. Nothing to run.")
        return 0

    # Where systest writes artifacts on the Pi (see module docstring):
    #   trace_*.json          -> the process CWD (== project root, our subprocess cwd)
    #   BenchmarkResults.json -> working-dir (written at the end of the run)
    artifact_targets: List[Tuple[Path, str]] = [
        (root, "trace_*.json"),
        (working_dir, "BenchmarkResults.json"),
    ]

    for combo in pending_combinations:
        combo_label = combination_name(combo)
        combo_dir = results_base / combo_label

        params = dict(combo)

        for query, query_name in query_specs:
            query_dir = combo_dir / query_name
            missing_query_repetitions = missing_repetitions(query_dir, args.repetitions)
            if not missing_query_repetitions:
                print(f"Skipping already processed query: {combo_label}/{query_name}")
                continue

            for repetition in missing_query_repetitions:
                rep_dir = query_dir / f"rep-{repetition:02d}"

                cmd = build_command(
                    systest_path,
                    query,
                    params,
                    working_dir=working_dir,
                    data_dir=data_dir,
                    cluster_config=cluster_config,
                    log_path=log_scratch,
                )

                print(" ".join(cmd))
                if args.dry_run:
                    continue

                total_attempts = QUERY_RETRIES + 1
                result: subprocess.CompletedProcess[str] | None = None
                before_snapshots: List[Dict[Path, float]] | None = None
                duration_us: float | None = None
                for attempt in range(1, total_attempts + 1):
                    before_snapshots = [
                        snapshot_files(directory, pattern) for directory, pattern in artifact_targets
                    ]

                    # cwd = project root so relative paths and trace_*.json land
                    # in the same place the user's manual invocation uses.
                    result = subprocess.run(
                        cmd,
                        check=False,
                        cwd=root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    if result.stdout:
                        print(result.stdout, end="")
                    if result.returncode == 0:
                        duration_us = parse_query_performance_duration_us(result.stdout)
                        break

                    if attempt < total_attempts:
                        print(
                            f"systest failed with exit code {result.returncode} "
                            f"(attempt {attempt}/{total_attempts}). Retrying...",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"systest failed with exit code {result.returncode} "
                            f"after {QUERY_RETRIES} retries. Terminating run.",
                            file=sys.stderr,
                        )
                        return result.returncode

                if result is None or result.returncode != 0 or before_snapshots is None:
                    print("systest run ended without a successful attempt.", file=sys.stderr)
                    return 1

                rep_dir.mkdir(parents=True, exist_ok=True)

                collect_artifacts(artifact_targets, rep_dir, before_snapshots)

                if log_scratch.exists():
                    safe_move(log_scratch, rep_dir)

                if duration_us is None:
                    print(
                        "Could not parse --show-query-performance runtime from systest output; "
                        f"no {SYSTEST_PERFORMANCE_FILE} written for {rep_dir}.",
                        file=sys.stderr,
                    )
                else:
                    write_query_performance(rep_dir, duration_us)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

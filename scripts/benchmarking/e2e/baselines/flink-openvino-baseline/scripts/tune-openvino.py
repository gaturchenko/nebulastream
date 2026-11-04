#!/usr/bin/env python3

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


THROUGHPUT_RE = re.compile(r"throughput=([0-9]+(?:\.[0-9]+)?)")
AVG_INFERENCE_RE = re.compile(r"avg_inference(?:Us|_us)?=([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class Config:
    flink_parallelism: int
    ov_streams: int
    ov_threads: int

    @property
    def thread_cost(self) -> int:
        # Approximate CPU thread budget.
        # Each Flink subtask creates one OpenVINO engine.
        return self.flink_parallelism * self.ov_threads


@dataclass
class RunResult:
    config: Config
    repetition: int
    return_code: int
    wall_time_s: float
    wall_throughput: float
    reported_throughput: float | None
    reported_avg_inference_us: float | None
    stdout: str
    stderr: str


def parse_last_float(pattern: re.Pattern, text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def run_one(
    script: Path,
    device: str,
    count: int,
    config: Config,
    env: dict[str, str],
    timeout_s: int | None,
    repetition: int,
) -> RunResult:
    cmd = [
        str(script),
        device,
        str(count),
        str(config.flink_parallelism),
        str(config.ov_streams),
        str(config.ov_threads),
    ]

    start = time.perf_counter()

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout_s,
    )

    end = time.perf_counter()
    wall_time_s = end - start
    wall_throughput = count / wall_time_s if wall_time_s > 0 else 0.0

    combined = completed.stdout + "\n" + completed.stderr

    return RunResult(
        config=config,
        repetition=repetition,
        return_code=completed.returncode,
        wall_time_s=wall_time_s,
        wall_throughput=wall_throughput,
        reported_throughput=parse_last_float(THROUGHPUT_RE, combined),
        reported_avg_inference_us=parse_last_float(AVG_INFERENCE_RE, combined),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def generate_configs(
    budget: int,
    max_flink_parallelism: int | None,
    max_ov_streams: int | None,
    max_ov_threads: int | None,
    require_full_budget: bool,
    streams_lte_threads: bool,
) -> list[Config]:
    max_flink_parallelism = max_flink_parallelism or budget
    max_ov_threads = max_ov_threads or budget
    max_ov_streams = max_ov_streams or budget

    configs: list[Config] = []

    for flink_parallelism, ov_streams, ov_threads in itertools.product(
        range(1, max_flink_parallelism + 1),
        range(1, max_ov_streams + 1),
        range(1, max_ov_threads + 1),
    ):
        cfg = Config(flink_parallelism, ov_streams, ov_threads)

        if cfg.thread_cost > budget:
            continue

        if require_full_budget and cfg.thread_cost != budget:
            continue

        if streams_lte_threads and ov_streams > ov_threads:
            continue

        configs.append(cfg)

    return configs


def summarize(results: list[RunResult]) -> list[dict]:
    grouped: dict[Config, list[RunResult]] = {}

    for r in results:
        if r.return_code == 0:
            grouped.setdefault(r.config, []).append(r)

    rows = []

    for cfg, runs in grouped.items():
        wall_tps = [r.wall_throughput for r in runs]

        # Prefer reported throughput if the Flink job prints it.
        reported_tps = [
            r.reported_throughput
            for r in runs
            if r.reported_throughput is not None
        ]

        avg_inf = [
            r.reported_avg_inference_us
            for r in runs
            if r.reported_avg_inference_us is not None
        ]

        score_tps = mean(reported_tps) if reported_tps else mean(wall_tps)

        rows.append(
            {
                "flink_parallelism": cfg.flink_parallelism,
                "ov_streams": cfg.ov_streams,
                "ov_threads": cfg.ov_threads,
                "thread_cost": cfg.thread_cost,
                "runs": len(runs),
                "score_throughput": score_tps,
                "wall_throughput_avg": mean(wall_tps),
                "wall_throughput_std": stdev(wall_tps) if len(wall_tps) > 1 else 0.0,
                "reported_throughput_avg": mean(reported_tps) if reported_tps else "",
                "reported_avg_inference_us": mean(avg_inf) if avg_inf else "",
                "wall_time_s_avg": mean([r.wall_time_s for r in runs]),
            }
        )

    rows.sort(key=lambda row: row["score_throughput"], reverse=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune Flink parallelism and OpenVINO streams/threads."
    )

    parser.add_argument(
        "--script",
        default="scripts/run-local.sh",
        help="Path to run-local.sh.",
    )
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--budget", type=int, required=True)

    parser.add_argument("--max-flink-parallelism", type=int)
    parser.add_argument("--max-ov-streams", type=int)
    parser.add_argument("--max-ov-threads", type=int)

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per configuration.",
    )

    parser.add_argument(
        "--timeout-s",
        type=int,
        default=None,
        help="Timeout per run in seconds.",
    )

    parser.add_argument(
        "--full-budget-only",
        action="store_true",
        help="Only test configs where flink_parallelism * ov_threads == budget.",
    )

    parser.add_argument(
        "--allow-streams-gt-threads",
        action="store_true",
        help="By default ov_streams <= ov_threads is enforced.",
    )

    parser.add_argument(
        "--csv",
        default="target/tuning-results.csv",
        help="Path to write CSV results.",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failing configuration.",
    )

    args = parser.parse_args()

    script = Path(args.script).resolve()

    if not script.exists():
        print(f"Missing run script: {script}", file=sys.stderr)
        return 1

    configs = generate_configs(
        budget=args.budget,
        max_flink_parallelism=args.max_flink_parallelism,
        max_ov_streams=args.max_ov_streams,
        max_ov_threads=args.max_ov_threads,
        require_full_budget=args.full_budget_only,
        streams_lte_threads=not args.allow_streams_gt_threads,
    )

    if not configs:
        print("No configurations generated.", file=sys.stderr)
        return 1

    print(f"Generated {len(configs)} configurations.")
    print(f"Thread budget rule: flink_parallelism * ov_threads <= {args.budget}")
    print()

    env = os.environ.copy()

    all_results: list[RunResult] = []

    total_runs = len(configs) * args.repetitions
    current = 0

    for cfg in configs:
        for rep in range(1, args.repetitions + 1):
            current += 1

            print(
                f"[{current}/{total_runs}] "
                f"p={cfg.flink_parallelism}, "
                f"streams={cfg.ov_streams}, "
                f"threads={cfg.ov_threads}, "
                f"cost={cfg.thread_cost}, "
                f"rep={rep}"
            )

            try:
                result = run_one(
                    script=script,
                    device=args.device,
                    count=args.count,
                    config=cfg,
                    env=env,
                    timeout_s=args.timeout_s,
                    repetition=rep,
                )
            except subprocess.TimeoutExpired:
                print("  TIMEOUT")
                if args.fail_fast:
                    return 2
                continue

            all_results.append(result)

            if result.return_code != 0:
                print(f"  FAILED rc={result.return_code}")
                print(result.stderr[-2000:])
                if args.fail_fast:
                    return result.return_code
                continue

            effective_tps = (
                result.reported_throughput
                if result.reported_throughput is not None
                else result.wall_throughput
            )

            print(
                f"  throughput={effective_tps:.2f} rec/s "
                f"(wall={result.wall_throughput:.2f} rec/s, "
                f"time={result.wall_time_s:.3f}s)"
            )

    rows = summarize(all_results)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "flink_parallelism",
                "ov_streams",
                "ov_threads",
                "thread_cost",
                "runs",
                "score_throughput",
                "wall_throughput_avg",
                "wall_throughput_std",
                "reported_throughput_avg",
                "reported_avg_inference_us",
                "wall_time_s_avg",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        print("No successful runs.", file=sys.stderr)
        return 2

    best = rows[0]

    print()
    print("Best configuration:")
    print(f"  Flink parallelism: {best['flink_parallelism']}")
    print(f"  OpenVINO streams:  {best['ov_streams']}")
    print(f"  OpenVINO threads:  {best['ov_threads']}")
    print(f"  Thread cost:       {best['thread_cost']} / {args.budget}")
    print(f"  Throughput:        {best['score_throughput']:.2f} rec/s")

    if best["reported_avg_inference_us"] != "":
        print(f"  Avg inference:     {best['reported_avg_inference_us']:.3f} us")

    print()
    print("Top 10:")
    for i, row in enumerate(rows[:10], start=1):
        print(
            f"{i:2d}. "
            f"p={row['flink_parallelism']}, "
            f"streams={row['ov_streams']}, "
            f"threads={row['ov_threads']}, "
            f"cost={row['thread_cost']}, "
            f"throughput={row['score_throughput']:.2f} rec/s"
        )

    print()
    print(f"Wrote CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


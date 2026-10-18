"""Reduce the raw systems-comparison runs to the small CSVs the figure scripts plot.

The raw results are ~120 per-repetition CSVs carrying a full pipeline-level schema, of which the
figures use a handful of columns. They are not kept in this tree: the figure data is the aggregate,
and this script is how it was produced, so the reduction is reproducible rather than a one-off.

Run it against the directories navi writes on the Pi:

    python aggregate_systems_data.py \
        --reps  ~/navi/results-reps \
        --sweep ~/navi/results-sweep

Raw runs live on the Pi under those two directories (also mirrored into the session scratchpad when
they were pulled). Re-run this after any new repetition, then re-render the figures.

Every number is a median across repetitions; latency also carries the min and max, because the
spread is the point on these workloads -- single runs reversed the CWRU ranking and invented a
Flink speed-up that three runs erase.
"""

import argparse
import glob
import statistics
from pathlib import Path

import pandas as pd

from systems_common import FLINK, NES, ORDER, TORCHSERVE, WORKLOADS

RATES = [10, 15, 20, 30, 45, 65, 90]
# Bottom-to-top, the order a TorchServe request actually travels.
PARTS = ["buffer_to_post", "http_ipc", "worker_queue", "handler"]


def read_point(system: str, path: Path) -> dict:
    """One run's end-to-end latency percentiles [ms] and achieved throughput [tuples/s]."""
    if system == NES:
        # The NAVI query is systems.test_02. test_03 is the TorchServe query; its rows share the
        # file because both run under one systest invocation, but carry no sink latency -- the
        # HTTP sink reports its own.
        df = pd.read_csv(path)
        rows = df[df["query_name"].str.endswith("test_02") & df["p95_sink_latency_us"].notna()]
        row = rows.iloc[0]
        return {"p50": row["p50_sink_latency_us"] / 1000.0,
                "p95": row["p95_sink_latency_us"] / 1000.0,
                "mean": row["avg_sink_latency_us"] / 1000.0,
                "throughput": row["end_to_end_throughput"]}
    row = pd.read_csv(path).iloc[0]
    return {"p50": row["e2e_latency_ms_p50"], "p95": row["e2e_latency_ms_p95"],
            "mean": row["e2e_latency_ms_mean"], "throughput": row["end_to_end_throughput"]}


def reps(system: str, pattern: str, directory: Path, minimum: int = 3) -> list[dict]:
    """Every repetition matching pattern; at least `minimum` of them must exist.

    A minimum rather than an exact count: a cell that turned out to be unstable gets more runs than
    its neighbours (CWRU/Flink is bimodal -- inference P50 came out 114/112/238/231/244/223 ms over
    six repetitions -- so three would report whichever mode happened to win the draw). The count is
    carried into the output so a caption can state it.
    """
    paths = sorted(glob.glob(str(directory / pattern)))
    if len(paths) < minimum:
        raise SystemExit(f"expected at least {minimum} runs for {pattern}, found {len(paths)}")
    return [read_point(system, Path(p)) for p in paths]


def summarise(points: list[dict], key: str) -> tuple[float, float, float]:
    values = [p[key] for p in points]
    return statistics.median(values), min(values), max(values)


def workloads_table(reps_dir: Path) -> pd.DataFrame:
    rows = []
    for workload in WORKLOADS:
        for system in ORDER:
            points = reps(system, f"{system}_{workload}_rep[0-9].csv", reps_dir)
            median, low, high = summarise(points, "p95")
            mean_median, mean_low, mean_high = summarise(points, "mean")
            # The mean is carried too: it is the only statistic whose parts add up, so it is what
            # the TorchServe-decomposition variant of the workloads figure plots. P95 and the mean
            # rank the three systems identically, so the variant changes no conclusion.
            rows.append({"workload": workload, "system": system, "n_reps": len(points),
                         "p95_median_ms": median, "p95_min_ms": low, "p95_max_ms": high,
                         "mean_median_ms": mean_median, "mean_min_ms": mean_low,
                         "mean_max_ms": mean_high})
    return pd.DataFrame(rows)


def sweep_table(sweep_dir: Path) -> pd.DataFrame:
    # "rep[0-9]", not "rep*": the Flink arm has both <...>_rep1.csv and <...>_rep1_fastparse.csv in
    # one directory, and a trailing star silently folds the second set into the first.
    # The Flink arm reads the "_fastparse" runs: the plain ones predate the replacement of its
    # per-line CSV parser with a direct column slice, a defect that capped the job at 14.5
    # frames/s. They stay on the Pi but are no longer aggregated, since nothing plots them.
    series = [(NES, NES, ""), (FLINK, FLINK, "_fastparse"), (TORCHSERVE, TORCHSERVE, "")]
    rows = []
    for name, system, suffix in series:
        for rate in RATES:
            points = reps(system, f"{system}_waymo_r{rate}_rep[0-9]{suffix}.csv", sweep_dir)
            throughput, _, _ = summarise(points, "throughput")
            p95, low, high = summarise(points, "p95")
            rows.append({"series": name, "rate": rate, "throughput_median": throughput,
                         "p95_median_ms": p95, "p95_min_ms": low, "p95_max_ms": high})
    return pd.DataFrame(rows)


def breakdown_table(reps_dir: Path) -> pd.DataFrame:
    """TorchServe's mean end-to-end latency split into the four stages its counters expose.

    Means, not percentiles: only means make the four parts sum to the end-to-end figure. HTTP/IPC
    is the remainder of the sink's round trip once the queue and handler counters are removed --
    the sink times creation -> response received, so transport is what is left over.
    """
    rows = []
    for workload in WORKLOADS:
        runs = []
        for path in sorted(glob.glob(str(reps_dir / f"torchserve_{workload}_rep[0-9].csv"))):
            row = pd.read_csv(path).iloc[0]
            serialize = row["serialization_ms_mean"]
            queue = row["queue_latency_us_per_req"] / 1000.0
            handler = row["inference_latency_us_per_req"] / 1000.0
            http = row["e2e_latency_ms_mean"] - serialize - queue - handler
            runs.append([serialize, http, queue, handler])
        if len(runs) < 3:
            raise SystemExit(f"expected at least 3 TorchServe runs for {workload}, found {len(runs)}")
        for part, values in zip(PARTS, zip(*runs)):
            rows.append({"workload": workload, "part": part,
                         "mean_ms": statistics.median(values)})
    return pd.DataFrame(rows)


def memory_table(memory_dir: Path) -> pd.DataFrame:
    """Peak RSS of the Flink job JVM, private vs. shared compiled model.

    The lane count is read back out of the run rather than restated: the native engine logs once
    per lane when sharing is on ("compiled, shared with later lanes" plus one "reusing" per extra
    lane), so the hit count IS the number of lanes that shared a model.

    Peak RSS is of the whole single-JVM MiniCluster, so the absolute numbers include heap and
    framework overhead; it is the DIFFERENCE between the two modes that is attributable to the
    compiled models.
    """
    df = pd.read_csv(memory_dir / "flink_memory.csv")
    rows = []
    for workload in WORKLOADS:
        here = df[df["workload"] == workload]
        private = here[here["mode"] == "private"]["peak_rss_mb"]
        shared = here[here["mode"] == "shared"]
        if private.empty or shared.empty:
            raise SystemExit(f"missing private/shared runs for {workload}")
        lanes = int(shared["share_log_hits"].median())
        rows.append({"workload": workload, "lanes": lanes,
                     "private_peak_rss_mb": private.median(),
                     "shared_peak_rss_mb": shared["peak_rss_mb"].median()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=Path, required=True,
                        help="directory of <system>_<workload>_rep<N>.csv (run_systems_reps.sh)")
    parser.add_argument("--sweep", type=Path, required=True,
                        help="directory of <system>_waymo_r<rate>_rep<N>.csv "
                             "(run_systems_rate_sweep.sh)")
    parser.add_argument("--memory", type=Path,
                        help="directory holding flink_memory.csv (run_flink_memory.sh). Optional: "
                             "omit it to leave the existing flink_memory.csv aggregate alone.")
    args = parser.parse_args()

    out = Path(__file__).resolve().parent / "data" / "systems"
    for name, frame in (("workloads_p95.csv", workloads_table(args.reps.expanduser())),
                        ("rate_sweep.csv", sweep_table(args.sweep.expanduser())),
                        ("torchserve_breakdown.csv", breakdown_table(args.reps.expanduser()))):
        frame.to_csv(out / name, index=False, float_format="%.4f")
        print(f"wrote {out / name}  ({len(frame)} rows)")

    if args.memory:
        frame = memory_table(args.memory.expanduser())
        frame.to_csv(out / "flink_memory.csv", index=False, float_format="%.1f")
        print(f"wrote {out / 'flink_memory.csv'}  ({len(frame)} rows)")


if __name__ == "__main__":
    main()

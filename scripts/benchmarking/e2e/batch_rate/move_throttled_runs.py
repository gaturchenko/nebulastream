#!/usr/bin/env python3
"""Moves every run during which the Pi's firmware limited the CPU out of a results directory, so that the driver re-measures it.

Reads <results>/<combination>/<query>/rep-NN/latency_timings.csv as run_systests_lat.py lays them out and <results>/thermal.csv
as the drivers' sampler writes it. A run spans its first ingest to its last receive, as in process_batch_rate.py and
process_batch_throughput.py, and counts as throttled under the same rule (thermal_during). Its rep-NN directory moves to the
same relative path under --target, and the drivers, which fill only missing repetitions, run it again when re-invoked.

    ./scripts/move_throttled_runs.py results/batch_rate results/batch_rate_throttled/round1

Prints one moved directory per line.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from process_batch_rate import thermal_during


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    thermal = pd.read_csv(args.results_dir / "thermal.csv")
    for latency_csv in sorted(args.results_dir.glob("*/*/rep-*/latency_timings.csv")):
        timings = pd.read_csv(latency_csv, usecols=["ingest_ts_us", "recv_ts_us"])
        if timings.empty:
            continue
        state = thermal_during(thermal, int(timings["ingest_ts_us"].min()), int(timings["recv_ts_us"].max()))
        if not state["throttled_during_run"]:
            continue
        rep_dir = latency_csv.parent
        target = args.target / rep_dir.relative_to(args.results_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rep_dir), str(target))
        print(target)


if __name__ == "__main__":
    main()

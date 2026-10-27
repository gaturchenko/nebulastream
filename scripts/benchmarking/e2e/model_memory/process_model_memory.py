#!/usr/bin/env python3
"""Aggregates run_model_memory.py's results into one row per cell: what the runtime sessions of each model cost on the Pi.

Per cell (one model, or the two-model query), medians across repetitions:

  file_mb                      the model on disk (summed over a cell's models)
  assessed_*                   the optimizer's assessment: first session, each additional private and shared session, and the
                               totals for the worker's sessions (summed over a cell's models)
  deployed_{private,shared}_mb the query's resident set minus its baseline's, from the same repetition, with min and max
  peak_{private,shared}_mb     the query process's peak resident set, which includes the transient of importing every model
                               the test file registers

The deployed figures include what the assessment does not see: the OpenVINO runtime the first session loads into the process
and what the sessions allocate once they run. The two-model row against its two single-model rows shows how the cost of a
second model adds up.

    ./scripts/process_model_memory.py [--results results/model_memory] [--output results/model_memory/model_memory.csv]

Standard library only, so it runs with any Python on the Pi.
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

MB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/model_memory"))
    parser.add_argument("--output", type=Path, default=None, help="default: <results>/model_memory.csv")
    return parser.parse_args()


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def spread(values: list[float]) -> tuple:
    if not values:
        return "", "", ""
    return round(statistics.median(values), 1), round(min(values), 1), round(max(values), 1)


def main() -> None:
    args = parse_args()
    output = args.output or args.results / "model_memory.csv"
    runs = [r for r in read(args.results / "runs.csv") if r["passed"] == "1" and r["steady_rss_mb"]]
    assessed = read(args.results / "assess.csv")
    sizes = {r["model"]: int(r["bytes"]) for r in read(args.results / "models.csv") if r["bytes"]}

    # The last passed run per (rep, cell, mode) counts, in case a cell was re-run.
    steady, peak = {}, {}
    cells = {}
    baseline_of = {}
    for r in runs:
        key = (r["rep"], r["cell"], r["mode"])
        steady[key], peak[key] = float(r["steady_rss_mb"]), float(r["peak_rss_mb"])
        if r["mode"] == "baseline":
            baseline_of[r["workload"]] = r["cell"]
        else:
            cells[r["cell"]] = (r["workload"], r["models"].split("+"))

    # One assessment per model and run; a cell's assessed cost is the sum over its models, per repetition.
    per_rep = defaultdict(lambda: defaultdict(float))
    models_seen = defaultdict(set)
    for a in assessed:
        key = (a["cell"], a["rep"])
        models_seen[key].add(a["model"])
        for field in ("first_session_b", "additional_private_b", "additional_shared_b", "private_total_b", "shared_total_b"):
            per_rep[key][field] += int(a[field])

    rows = []
    for cell, (workload, models) in cells.items():
        row = {"cell": cell, "workload": workload, "models": "+".join(models),
               "file_mb": round(sum(sizes.get(m, 0) for m in models) / MB, 1)}
        complete = [rep for (c, rep) in per_rep if c == cell and len(models_seen[(c, rep)]) == len(models)]
        for field, name in [("first_session_b", "assessed_first_mb"), ("additional_private_b", "assessed_additional_private_mb"),
                            ("additional_shared_b", "assessed_additional_shared_mb"),
                            ("private_total_b", "assessed_private_total_mb"), ("shared_total_b", "assessed_shared_total_mb")]:
            values = [per_rep[(cell, rep)][field] / MB for rep in complete]
            row[name] = round(statistics.median(values), 1) if values else ""
        row["assess_reps"] = len(complete)
        baseline = baseline_of.get(workload)
        for mode in ("private", "shared"):
            deltas = [steady[(rep, c, m)] - steady[(rep, baseline, "baseline")]
                      for (rep, c, m) in steady if c == cell and m == mode and (rep, baseline, "baseline") in steady]
            row[f"deployed_{mode}_mb"], row[f"deployed_{mode}_min_mb"], row[f"deployed_{mode}_max_mb"] = spread(deltas)
            row[f"{mode}_reps"] = len(deltas)
            peaks = [peak[k] for k in peak if k[1] == cell and k[2] == mode]
            row[f"peak_{mode}_mb"] = round(statistics.median(peaks), 1) if peaks else ""
        baselines = [steady[k] for k in steady if k[1] == baseline and k[2] == "baseline"]
        row["baseline_rss_mb"] = round(statistics.median(baselines), 1) if baselines else ""
        rows.append(row)

    fields = list(rows[0]) if rows else []
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'cell':<22}{'file':>7}{'first':>8}{'+priv':>8}{'+shrd':>8}{'asPriv':>8}{'asShrd':>8}{'depPriv':>9}{'depShrd':>9}"
          f"{'peakPriv':>10}   [MB; assessed and deployed totals for the worker's sessions]")
    for r in rows:
        print(f"{r['cell']:<22}{r['file_mb']:>7}{r['assessed_first_mb']:>8}{r['assessed_additional_private_mb']:>8}"
              f"{r['assessed_additional_shared_mb']:>8}{r['assessed_private_total_mb']:>8}{r['assessed_shared_total_mb']:>8}"
              f"{r['deployed_private_mb']:>9}{r['deployed_shared_mb']:>9}{r['peak_private_mb']:>10}")
    by_cell = {r["cell"]: r for r in rows}
    for r in rows:
        singles = r["models"].split("+")
        if len(singles) > 1 and all(s in by_cell for s in singles):
            for mode in ("private", "shared"):
                parts = [by_cell[s][f"deployed_{mode}_mb"] for s in singles]
                if r[f"deployed_{mode}_mb"] != "" and "" not in parts:
                    print(f"{r['cell']} {mode}: {r[f'deployed_{mode}_mb']} MB together vs. {round(sum(parts), 1)} MB as single "
                          f"queries ({' + '.join(str(p) for p in parts)})")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

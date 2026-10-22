#!/usr/bin/env python3
"""Compare OPTIMIZER_CHOOSES' batch size against every fixed batch size of the source-rate x batch-size sweep.

Per offered rate, a fixed batch size is feasible if it delivered the offered rate (sustained) and its P99 result latency met
the SLO. Among the feasible ones, two oracles are picked from the measurements:

  oracle_throughput  the largest sustainable throughput (batch_throughput_summary.csv), i.e. the most headroom; this is
                     what the optimizer's rule maximizes among the batch sizes it predicts to be feasible
  oracle_latency     the lowest P99 latency

and the optimizer's choice is scored against both: whether it was feasible, its headroom regret 1 - capacity(chosen) /
capacity(oracle_throughput), and its latency regret P99(chosen) - P99(oracle_latency).

    ./process_batch_rate_regret.py --data-dir scripts/benchmarking/figures/data/batch_rate
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("scripts/benchmarking/figures/data/batch_rate"))
    parser.add_argument("--latency-quantile", choices=["p50", "p95", "p99"], default="p99")
    args = parser.parse_args()

    summary = pd.read_csv(args.data_dir / "batch_rate_summary.csv")
    capacity = pd.read_csv(args.data_dir / "batch_throughput_summary.csv")
    capacity = capacity[capacity["query"] == "inference"].set_index("batch_size")["sustainable_rate"]
    latency = f"latency_ms_{args.latency_quantile}"

    summary["capacity"] = summary["batch_size"].map(capacity)
    summary["feasible"] = summary["sustained"].astype(bool) & (summary[latency] <= summary["slo_ms"])

    rows = []
    for rate, cells in summary.groupby("offered_rate"):
        fixed = cells[cells["strategy"] == "FIXED"]
        chosen = cells[cells["strategy"] == "OPTIMIZER_CHOOSES"].iloc[0]
        feasible = fixed[fixed["feasible"]]
        row = {
            "offered_rate": round(rate, 1),
            "slo_ms": chosen["slo_ms"],
            "feasible_batch_sizes": " ".join(str(int(b)) for b in sorted(feasible["batch_size"])),
            "measured_batch_sizes": len(fixed),
            "chosen": int(chosen["batch_size"]),
            "chosen_feasible": bool(chosen["feasible"]),
            f"chosen_{latency}": chosen[latency],
            "chosen_capacity": chosen["capacity"],
        }
        if not feasible.empty:
            best_throughput = feasible.loc[feasible["capacity"].idxmax()]
            best_latency = feasible.loc[feasible[latency].idxmin()]
            row |= {
                "oracle_throughput": int(best_throughput["batch_size"]),
                "oracle_throughput_capacity": best_throughput["capacity"],
                "headroom_regret": 1 - chosen["capacity"] / best_throughput["capacity"],
                "oracle_latency": int(best_latency["batch_size"]),
                f"oracle_latency_{latency}": best_latency[latency],
                "latency_regret_ms": chosen[latency] - best_latency[latency],
            }
        rows.append(row)

    regret = pd.DataFrame(rows)
    output = args.data_dir / "batch_rate_regret.csv"
    regret.to_csv(output, index=False)
    pd.set_option("display.width", 250)
    print(regret.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()

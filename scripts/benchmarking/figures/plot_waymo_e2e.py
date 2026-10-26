"""First-page systems comparison: Waymo / SqueezeDet only (Raspberry Pi).

Produces e2e-waymo.pdf, two panels over the one workload:

  (a) P95 end-to-end latency -- the tail, which is the metric a streaming system is held to.
  (b) Mean end-to-end latency, with TorchServe's bar split into the inference it performs and
      what reaching it over the serving boundary adds.

Only panel (b) carries the split, and that is a property of the arithmetic rather than a choice.
Stage times add up in expectation only: the P95 of a sum is not the sum of the stage P95s, so
segmenting the bars in (a) would assume every stage hits its tail on the same request and would
overstate them. Two of the four stages are mean-only at the source as well -- they come from
cumulative Prometheus counters divided by the request count, so there is no distribution to take a
percentile of. Showing both panels is what lets the figure keep the tail claim AND the cost
breakdown without a second figure.

Waymo alone is on the first page because it is where the serving boundary dominates: the hatched
cap is larger than the inference it wraps and larger than NAVI's entire bar. The sensor workloads
show the opposite -- the boundary scales with payload size, not model cost -- and belong in the
experiments section (e2e-systems-workloads.pdf, and its split counterpart).

Inference sits at the BOTTOM of the TorchServe bar on purpose: the solid height is what TorchServe
would cost if the boundary were free, which reads straight across against NAVI's bar beside it.

Data: data/systems/workloads_p95.csv, data/systems/torchserve_breakdown.csv
      (aggregate_systems_data.py --tuned over navi's tuning runs: each system at the parallelism its
      pilot chose, then 5 fresh repetitions; medians plus min-max across them).
"""

import numpy as np
import matplotlib.pyplot as plt

from common import save, setup_style
from systems_common import COLOR, LABEL, ORDER, TORCHSERVE, load

WORKLOAD = "waymo"
BOUNDARY_PARTS = ["buffer_to_post", "http_ipc", "worker_queue"]


def load_stats() -> tuple[dict, dict, tuple[float, float]]:
    df = load("workloads_p95.csv")
    df = df[df["workload"] == WORKLOAD].set_index("system")
    p95 = {s: (df.loc[s, "p95_median_ms"], df.loc[s, "p95_min_ms"], df.loc[s, "p95_max_ms"])
           for s in ORDER}
    mean = {s: (df.loc[s, "mean_median_ms"], df.loc[s, "mean_min_ms"], df.loc[s, "mean_max_ms"])
            for s in ORDER}

    parts = load("torchserve_breakdown.csv")
    parts = parts[parts["workload"] == WORKLOAD].set_index("part")
    split = (parts.loc["handler", "mean_ms"],
             sum(parts.loc[p, "mean_ms"] for p in BOUNDARY_PARTS))
    return p95, mean, split


def plot_waymo(p95: dict, mean: dict, split: tuple[float, float]) -> None:
    FIGSIZE = (6.6, 2.7)
    LABEL_FS = 13
    TICK_FS = 11
    VALUE_FS = 9.5
    LEGEND_FS = 10.5
    TITLE_FS = 12
    BAR_WIDTH = 0.62
    HEADROOM = 1.26

    setup_style()
    fig, (ax_p95, ax_mean) = plt.subplots(1, 2, figsize=FIGSIZE)
    x = np.arange(len(ORDER))

    def draw(ax, stats, ylabel, title, split_torchserve):
        medians = [stats[s][0] for s in ORDER]
        lower = [stats[s][0] - stats[s][1] for s in ORDER]
        upper = [stats[s][2] - stats[s][0] for s in ORDER]

        for index, system in enumerate(ORDER):
            if split_torchserve and system == TORCHSERVE:
                inference, boundary = split
                ax.bar(index, inference, width=BAR_WIDTH, color=COLOR[system],
                       edgecolor="black", linewidth=0.8)
                ax.bar(index, boundary, width=BAR_WIDTH, bottom=inference, facecolor="white",
                       edgecolor=COLOR[system], linewidth=0.8, hatch="///")
            else:
                ax.bar(index, medians[index], width=BAR_WIDTH, color=COLOR[system],
                       edgecolor="black", linewidth=0.8)

        ax.errorbar(x, medians, yerr=[lower, upper], fmt="none", ecolor="black",
                    elinewidth=0.9, capsize=3)
        for index, value in enumerate(medians):
            ax.text(index, value + upper[index], f"{value:.0f}", ha="center", va="bottom",
                    fontsize=VALUE_FS)

        ax.set_title(title, fontsize=TITLE_FS)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylim(0, max(np.array(medians) + np.array(upper)) * HEADROOM)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    draw(ax_p95, p95, "Latency P95 [ms]", "(a) Tail latency", split_torchserve=False)
    draw(ax_mean, mean, "Mean latency [ms]", "(b) Mean latency", split_torchserve=True)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[s], edgecolor="black", linewidth=0.8)
               for s in ORDER]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=COLOR[TORCHSERVE],
                                 linewidth=0.8, hatch="///"))
    labels = [LABEL[s] for s in ORDER] + ["serving boundary"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=LEGEND_FS, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    save(fig, "e2e-waymo.pdf", pad_inches=0.1)


if __name__ == "__main__":
    plot_waymo(*load_stats())

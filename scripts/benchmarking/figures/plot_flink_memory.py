"""What sharing one compiled model across inference lanes is worth, in memory.

Produces flink-memory.pdf: peak resident memory of the Flink+OpenVINO baseline with each lane
holding its own compiled model (the default) against all lanes sharing one.

This is the measurement behind the memory-sharing contribution. Three repetitions established that
sharing buys NO latency -- CWRU e2e P95 974 ms private vs. 856 shared, inside the spread of a
workload whose Flink arm is bimodal, and Waymo/MUSAN flat to within 4 ms -- so the case for it
rests entirely on footprint, and until this run that half had never been measured.

Each bar is the private configuration's peak RSS, split into the part that remains when lanes share
a model (solid) and the part sharing eliminates (hatched). Reading the solid portion alone answers
"what would this cost if the copies were free", which is the comparison the contribution makes.

Two things the caption should carry. Peak RSS is of the whole single-JVM MiniCluster, so the
absolutes include heap and framework overhead; it is the DIFFERENCE between the two modes that is
attributable to the compiled models. And the saving does not track the model FILE size -- MUSAN's
MHAtt-RNN is ~25 MB on disk yet sharing saves 936 MB, because what is duplicated is the compiled
model (optimised graph, weight layouts, per-stream scratch), not the serialised weights.

Reproducibility is high: across 3 repetitions per cell, peak RSS varied by under 1% (CWRU private
2016-2021 MB, shared 911-923 MB), and every run verified its own mode from the engine's own log --
private runs show no sharing, shared runs show exactly one log line per lane.

Data: data/systems/flink_memory.csv (aggregate_systems_data.py over navi's run_flink_memory.sh
      output; medians across 3 repetitions per mode).
"""

import numpy as np
import matplotlib.pyplot as plt

from common import save, setup_style
from systems_common import COLOR, FLINK, WORKLOAD_TITLE, WORKLOADS, load


def load_memory() -> dict:
    df = load("flink_memory.csv").set_index("workload")
    return {w: (df.loc[w, "shared_peak_rss_mb"], df.loc[w, "private_peak_rss_mb"],
                int(df.loc[w, "lanes"])) for w in WORKLOADS}


def plot_memory(data: dict) -> None:
    FIGSIZE = (5.4, 3.2)
    LABEL_FS = 13
    TICK_FS = 11
    VALUE_FS = 9.5
    LEGEND_FS = 10
    BAR_WIDTH = 0.55

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(WORKLOADS))

    shared = np.array([data[w][0] for w in WORKLOADS])
    private = np.array([data[w][1] for w in WORKLOADS])
    saved = private - shared

    ax.bar(x, shared, width=BAR_WIDTH, color=COLOR[FLINK], edgecolor="black", linewidth=0.8,
           label="shared compiled model")
    ax.bar(x, saved, width=BAR_WIDTH, bottom=shared, facecolor="white",
           edgecolor=COLOR[FLINK], linewidth=0.8, hatch="///", label="freed by sharing")

    for index, (s, v, p) in enumerate(zip(shared, saved, private)):
        ax.text(index, s / 2, f"{s:.0f}", ha="center", va="center", fontsize=VALUE_FS, color="white")
        ax.text(index, s + v / 2, f"$-${v:.0f}\\,MB", ha="center", va="center", fontsize=VALUE_FS)
        ax.text(index, p * 1.02, f"{v / p * 100:.0f}\\%", ha="center", va="bottom", fontsize=VALUE_FS)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{WORKLOAD_TITLE[w].split(' / ')[0]}\n{{\\small {data[w][2]} lanes}}"
                        for w in WORKLOADS], fontsize=LABEL_FS)
    ax.set_ylabel("Peak resident memory [MB]", fontsize=LABEL_FS)
    ax.set_ylim(0, private.max() * 1.18)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=LEGEND_FS)

    fig.tight_layout()
    save(fig, "flink-memory.pdf", pad_inches=0.1)


if __name__ == "__main__":
    plot_memory(load_memory())

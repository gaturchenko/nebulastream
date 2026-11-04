"""Post-join batching decision: the optimizer's BATCHED/SEQUENTIAL choice against both fixed plans.

Produces post-join-choice.pdf over the number k of joined streams, each stream carrying one MHAtt-RNN payload, all
with batch deduplication (a dynamic batch dimension): the inference time per joined record of both plans on x86 and on
the Pi (the task time of the inference pipelines from the event trace in the forced runs; bars: median, whiskers:
min-max over 3 repetitions), and a star on the plan the optimizer deployed. Neither plan is always right: BATCHED is
slower on x86 at k = 2, SEQUENTIAL from k = 4 on x86 and at every k on the Pi, where at k = 4 it also falls behind the
offered rate.

End-to-end latency is not plotted: it is dominated by closing the windows and joins (about two window advances per join
level), which both plans share, so the plans' P50 differ by at most ~1%.

Data: data/post_join/post_join_{runs,decisions}_x86.csv
      (process_post_join.py over run_post_join.sh on so015, 3 repetitions, ADVANCE 333 ms = 3 joined records/s)
      data/post_join/post_join_{runs,decisions}_pi.csv (the same on the Pi, k = 2, 4, ADVANCE 1000 ms)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import load_csv, save, setup_style

# The first three slots of the validated categorical palette, which pass every pair.
BATCHED_COLOR = "#2a78d6"
OPTIMIZER_COLOR = "#eb6834"
SEQUENTIAL_COLOR = "#1baf7a"
PLAN_COLOR = {"BATCHED": BATCHED_COLOR, "SEQUENTIAL": SEQUENTIAL_COLOR}
HARDWARE_SUFFIX = {"x86": "x86", "Pi": "pi"}
HARDWARE_TITLE = {"x86": "Server [x86]", "Pi": "Pi [aarch64]"}

FONT_SIZE = 8
TICK_SIZE = 7


def load_choice(hardware: str) -> pd.DataFrame:
    """Per k with deduplication: both plans' inference time per joined record (median, min, max over the forced runs at the
    rate the decision is compared at), how often the optimizer deployed each plan, and SEQUENTIAL's largest sustained rate
    where it fell behind."""
    suffix = HARDWARE_SUFFIX[hardware]
    decisions = load_csv("post_join", f"post_join_decisions_{suffix}.csv")
    decisions = decisions[decisions["dedup"]].set_index("k")
    runs = load_csv("post_join", f"post_join_runs_{suffix}.csv")
    runs = runs[(runs["query"] == "inference") & runs["dedup"]]

    rows = []
    for k, decision in decisions.iterrows():
        row = {"k": k}
        at_rate = runs[(runs["k"] == k) & (runs["advance_ms"] == decision["latency_advance_ms"])]
        for plan in PLAN_COLOR:
            times = at_rate[at_rate["strategy"] == plan]["inference_ms_per_record"]
            row |= {f"{plan}_median": times.median(), f"{plan}_min": times.min(), f"{plan}_max": times.max()}
        chosen = runs[(runs["k"] == k) & (runs["strategy"] == "OPTIMIZER_CHOOSES")]["chosen"].dropna()
        row |= {f"{plan}_chosen": int((chosen == plan).sum()) for plan in PLAN_COLOR}
        row["deployments"] = len(chosen)
        fell_behind = decision["sequential_max_sustained_rate"] < decision["batched_max_sustained_rate"]
        row["sequential_sustained_rate"] = decision["sequential_max_sustained_rate"] if fell_behind else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")


def draw_choice(ax, table: pd.DataFrame, hardware: str) -> None:
    """Both plans' inference time per joined record per k, and a star on the plan the optimizer deployed."""
    BAR_WIDTH = 0.38
    x = np.arange(len(table))
    top = max(table[f"{plan}_max"].max() for plan in PLAN_COLOR)

    for i, (plan, color) in enumerate(PLAN_COLOR.items()):
        centers = x + (i - 0.5) * BAR_WIDTH
        median = table[f"{plan}_median"]
        ax.bar(centers, median, BAR_WIDTH, color=color, edgecolor="white", linewidth=0.5, zorder=2)
        ax.errorbar(centers, median, yerr=[median - table[f"{plan}_min"], table[f"{plan}_max"] - median], fmt="none",
                    ecolor="0.25", elinewidth=0.6, capsize=1.2, zorder=3)
        for center, (k, row) in zip(centers, table.iterrows()):
            # The plan most deployments chose carries the star.
            if row[f"{plan}_chosen"] * 2 > row["deployments"]:
                ax.scatter(center, row[f"{plan}_max"] + top * 0.05, marker="*", s=60, color=OPTIMIZER_COLOR,
                           edgecolors="white", linewidths=0.4, zorder=4)
            if plan == "SEQUENTIAL" and not np.isnan(row["sequential_sustained_rate"]):
                ax.text(center, row[f"{plan}_max"] + top * 0.04, f"falls behind\nabove {row['sequential_sustained_rate']:g} rec./s",
                        ha="center", va="bottom", fontsize=TICK_SIZE, color="0.25", linespacing=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.set_ylim(0, top * 1.42)  # headroom for the two-line falls-behind note below the panel title
    ax.set_title(HARDWARE_TITLE[hardware], fontsize=FONT_SIZE, pad=2)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)


def style() -> None:
    setup_style()
    plt.rcParams.update({"font.size": FONT_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
                         "legend.fontsize": TICK_SIZE})


def plot_choice(tables: dict[str, pd.DataFrame]) -> None:
    FIG_SIZE = (3.33, 1.45)

    style()
    fig, axes = plt.subplots(1, len(tables), figsize=FIG_SIZE,
                             gridspec_kw={"width_ratios": [len(table) for table in tables.values()]})
    for ax, (hardware, table) in zip(axes, tables.items()):
        draw_choice(ax, table, hardware)
    axes[0].set_ylabel("Inference time per\njoined record [ms]")
    fig.supxlabel("Joined streams $k$", fontsize=FONT_SIZE, y=0.02)
    handles = [Patch(facecolor=color, label=plan) for plan, color in PLAN_COLOR.items()] + [
        Line2D([], [], color=OPTIMIZER_COLOR, marker="*", markersize=7, markeredgecolor="white", markeredgewidth=0.4,
               linestyle="none", label="optimizer's choice")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, handlelength=1.2, handletextpad=0.4,
               columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout(pad=0.2, w_pad=0.8, rect=(0, 0.062, 1, 0.857))
    save(fig, "post-join-choice.pdf")


if __name__ == "__main__":
    plot_choice({hardware: load_choice(hardware) for hardware in HARDWARE_SUFFIX})

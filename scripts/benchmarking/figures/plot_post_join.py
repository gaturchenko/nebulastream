"""Post-join batching decision: the optimizer's BATCHED/SEQUENTIAL choice against both fixed plans.

Two figures over the number k of joined streams, each stream carrying one MHAtt-RNN payload, all with batch deduplication
(a dynamic batch dimension):
  - post-join-choice.pdf: the inference time per joined record of both plans on x86 and on the Pi (the task time of the
    inference pipelines from the event trace in the forced runs; bars: median, whiskers: min-max over 3 repetitions), and
    a star on the plan the optimizer deployed. Neither plan is always right: BATCHED is slower on x86 at k = 2, SEQUENTIAL
    from k = 4 on x86 and at every k on the Pi, where at k = 4 it also falls behind the offered rate.
  - post-join-speedup.pdf: why the optimizer chooses right: the batching speedup k t(1) / t(k) it predicts from its
    profile (line: median, band: min-max over 3 profiles), against the speedup the forced plans measured, and the 1.1
    threshold below which it runs the k invocations sequentially. x86 falls below it at k = 2 and straddles it at k = 3,
    where its choice varies between deployments while the plans take the same time; on the Pi the same model and settings
    batch at every k.

End-to-end latency is not plotted: it is dominated by closing the windows and joins (about two window advances per join
level), which both plans share, so the plans' P50 differ by at most ~1%.

Data: data/post_join/post_join_{runs,decisions}_x86.csv
      (process_post_join.py over run_post_join.sh on so015, 3 repetitions, ADVANCE 333 ms = 3 joined records/s)
      data/post_join/post_join_{runs,decisions}_pi.csv (the same on the Pi, k = 2, 4, ADVANCE 1000 ms)
      data/post_join/post_join_pilot_summary_{x86,pi}.csv
      (summarize_post_join_pilot.py over run_post_join_pilot.sh, nes-model-profiler, the deployed session: 1 stream x 4 threads)
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
HARDWARE_COLOR = {"x86": "0.1", "Pi": "0.55"}
HARDWARE_SUFFIX = {"x86": "x86", "Pi": "pi"}
HARDWARE_TITLE = {"x86": "Server [x86]", "Pi": "Pi [aarch64]"}
THRESHOLD = 1.1
MODEL = "mhatt_rnn"
# The deployed OpenVINO session: one stream with four inference threads, on both hardware platforms.
SESSION_STREAMS = 1
SESSION_THREADS = 4

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


def load_speedup() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    # The optimizer's prediction in the deployed queries is its profile at the deployed session, which the pilot also ran.
    predicted = {}
    for hardware, suffix in HARDWARE_SUFFIX.items():
        pilot = load_csv("post_join", f"post_join_pilot_summary_{suffix}.csv")
        predicted[hardware] = pilot[
            (pilot["model"] == MODEL)
            & (pilot["openvino_num_streams"] == SESSION_STREAMS)
            & (pilot["openvino_inference_num_threads"] == SESSION_THREADS)
        ]
    measured = {hardware: load_csv("post_join", f"post_join_decisions_{suffix}.csv")
                for hardware, suffix in HARDWARE_SUFFIX.items()}
    return predicted, measured


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
    ax.set_ylim(0, top * 1.28)
    ax.set_title(HARDWARE_TITLE[hardware], fontsize=FONT_SIZE, pad=2)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)


def draw_speedup(ax, predicted: dict[str, pd.DataFrame], measured: dict[str, pd.DataFrame], label_size: float) -> list:
    """Predicted (line, band over repetitions) vs. measured (markers) batching speedup per hardware with deduplication;
    returns legend handles."""
    Y_LIMITS = (0.65, 4.5)

    ax.axhspan(Y_LIMITS[0], THRESHOLD, color=SEQUENTIAL_COLOR, alpha=0.12, linewidth=0, zorder=0)
    ax.axhline(THRESHOLD, color="0.3", linestyle=":", linewidth=0.9, zorder=1)
    ax.text(8.4, Y_LIMITS[0] * 1.04, "SEQUENTIAL", ha="right", va="bottom", fontsize=label_size, color="0.3")

    # Lines are labelled at their right end rather than in the legend.
    for hardware, profile in predicted.items():
        series = profile[profile["dynamic_batch"] == 1].sort_values("k")
        color = HARDWARE_COLOR[hardware]
        ax.fill_between(series["k"], series["speedup_min"], series["speedup_max"], color=color, alpha=0.18, linewidth=0,
                        zorder=1)
        ax.plot(series["k"], series["speedup"], color=color, linewidth=1.0, zorder=2)
        ax.annotate(hardware, (series["k"].iloc[-1], series["speedup"].iloc[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=label_size, color=color)
    for hardware, decisions in measured.items():
        points = decisions[decisions["dedup"]]
        ax.scatter(points["k"], points["measured_speedup"], marker="o", s=14, color=HARDWARE_COLOR[hardware],
                   edgecolors="white", linewidths=0.5, zorder=3)

    ks = sorted(predicted["x86"]["k"].unique())
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4])
    ax.set_yticklabels(["1", "2", "4"])
    ax.minorticks_off()
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
    ax.set_ylim(*Y_LIMITS)
    ax.set_xlabel("Joined streams $k$")
    ax.set_ylabel(r"Batching speedup $k\,t(1)/t(k)$")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    return [
        Line2D([], [], color="0.1", linewidth=1.0, label="predicted"),
        Line2D([], [], color="0.1", linestyle="none", marker="o", markersize=3.5, markeredgecolor="white",
               markeredgewidth=0.5, label="measured"),
    ]


def style() -> None:
    setup_style()
    plt.rcParams.update({"font.size": FONT_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
                         "legend.fontsize": TICK_SIZE})


def plot_choice(tables: dict[str, pd.DataFrame]) -> None:
    FIG_SIZE = (3.33, 1.8)

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
    fig.tight_layout(pad=0.2, w_pad=0.8, rect=(0, 0.05, 1, 0.885))
    save(fig, "post-join-choice.pdf")


def plot_speedup(predicted: dict, measured: dict) -> None:
    FIG_SIZE = (3.33, 2.1)

    style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    handles = draw_speedup(ax, predicted, measured, TICK_SIZE)
    ax.legend(handles=handles, loc="upper left", frameon=False, handlelength=1.6, handletextpad=0.4, borderaxespad=0.2,
              labelspacing=0.25)
    fig.tight_layout(pad=0.2)
    save(fig, "post-join-speedup.pdf")


if __name__ == "__main__":
    plot_choice({hardware: load_choice(hardware) for hardware in HARDWARE_SUFFIX})
    plot_speedup(*load_speedup())

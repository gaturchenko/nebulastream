"""Post-join batching decision: the optimizer's BATCHED/SEQUENTIAL choice against both forced plans.

Produces post-join.pdf, two panels over the number k of joined streams, each carrying one MHAtt-RNN payload:
  (a) the cost of each forced plan per joined record on x86 with batch deduplication (the task time of the inference
      pipelines from the event trace), and the plan OPTIMIZER_CHOOSES deployed; the plans cross between k = 2 and 3
  (b) the batching speedup k t(1) / t(k) the optimizer predicts from its profile (line: median, band: min-max over 3
      profiles), against the speedup the forced plans measured (x86 at ADVANCE 333 ms, Pi at 1000 ms), and the 1.1
      threshold below which it runs the k invocations sequentially. x86 falls below it at k = 2 and straddles it at
      k = 3, where its choice varies between deployments while the plans cost the same; on the Pi the same model and
      settings batch at every k. Both panels use deduplication, i.e. a dynamic batch dimension.

End-to-end latency is not plotted: it is dominated by closing the windows and joins (about two window advances per join
level), which both plans share, so the plans' P50 differ by at most ~1%.

Data: data/post_join/post_join_runs_x86.csv, post_join_decisions_x86.csv
      (process_post_join.py over run_post_join.sh on so015, 3 repetitions, ADVANCE 333 ms = 3 joined records/s)
      data/post_join/post_join_decisions_pi.csv (the same on the Pi, k = 2, 4, ADVANCE 1000 ms)
      data/post_join/post_join_pilot_summary_{x86,pi}.csv
      (summarize_post_join_pilot.py over run_post_join_pilot.sh, nes-model-profiler, the deployed session: 1 stream x 4 threads)
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from common import load_csv, save, setup_style

BATCHED_COLOR = "#2a78d6"
SEQUENTIAL_COLOR = "#1baf7a"
OPTIMIZER_COLOR = "#e8730c"
HARDWARE_COLOR = {"x86": "0.1", "Pi": "0.55"}
THRESHOLD = 1.1
ADVANCE_MS = 333
MODEL = "mhatt_rnn"
# The deployed OpenVINO session: one stream with four inference threads, on both hardware platforms.
SESSION_STREAMS = 1
SESSION_THREADS = 4


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    runs = load_csv("post_join", "post_join_runs_x86.csv")
    runs = runs[(runs["query"] == "inference") & (runs["advance_ms"] == ADVANCE_MS)]
    cost = (
        runs.groupby(["k", "strategy", "dedup"])["inference_ms_per_record"]
        .agg(["median", "min", "max"])
        .reset_index()
    )
    chosen = (
        runs[runs["strategy"] == "OPTIMIZER_CHOOSES"]
        .groupby(["k", "dedup"])
        .agg(chosen=("chosen", lambda c: c.dropna().mode().iloc[0]), cost=("inference_ms_per_record", "median"))
        .reset_index()
    )
    # The optimizer's prediction in the deployed queries is its profile at the deployed session, which the pilot also ran.
    predicted = {}
    for hardware, suffix in (("x86", "x86"), ("Pi", "pi")):
        pilot = load_csv("post_join", f"post_join_pilot_summary_{suffix}.csv")
        predicted[hardware] = pilot[
            (pilot["model"] == MODEL)
            & (pilot["openvino_num_streams"] == SESSION_STREAMS)
            & (pilot["openvino_inference_num_threads"] == SESSION_THREADS)
        ]
    measured = {hardware: load_csv("post_join", f"post_join_decisions_{suffix}.csv")
                for hardware, suffix in (("x86", "x86"), ("Pi", "pi"))}
    return cost, chosen, predicted, measured


def draw_cost(ax, cost: pd.DataFrame, chosen: pd.DataFrame) -> list:
    """Per-record cost of both forced plans with deduplication, and the optimizer's choice; returns legend handles."""
    dedup = cost[cost["dedup"]]
    for strategy, color in (("BATCHED", BATCHED_COLOR), ("SEQUENTIAL", SEQUENTIAL_COLOR)):
        series = dedup[dedup["strategy"] == strategy].sort_values("k")
        ax.errorbar(series["k"], series["median"],
                    yerr=[series["median"] - series["min"], series["max"] - series["median"]],
                    color=color, linewidth=1.2, marker="o", markersize=3, capsize=1.5, elinewidth=0.7, zorder=2)
    picks = chosen[chosen["dedup"]]
    ax.scatter(picks["k"], picks["cost"], marker="*", s=70, color=OPTIMIZER_COLOR, edgecolors="white", linewidths=0.5,
               zorder=4)

    ks = sorted(dedup["k"].unique())
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
    ax.set_ylim(20, 115)
    ax.set_xlabel("Joined streams $k$")
    ax.set_ylabel("Inference cost [ms/record]")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    return [
        Line2D([], [], color=BATCHED_COLOR, linewidth=1.2, marker="o", markersize=3, label="BATCHED"),
        Line2D([], [], color=SEQUENTIAL_COLOR, linewidth=1.2, marker="o", markersize=3, label="SEQUENTIAL"),
        Line2D([], [], color=OPTIMIZER_COLOR, marker="*", markersize=8, markeredgecolor="white", markeredgewidth=0.5,
               linestyle="none", label="optimizer's choice"),
    ]


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


def plot_post_join(cost: pd.DataFrame, chosen: pd.DataFrame, predicted: dict, measured: dict) -> None:
    FIG_SIZE = (3.33, 2.3)
    FONT_SIZE = 8
    TICK_SIZE = 7

    setup_style()
    plt.rcParams.update({"font.size": FONT_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
                         "legend.fontsize": TICK_SIZE})
    fig, (cost_ax, speedup_ax) = plt.subplots(1, 2, figsize=FIG_SIZE, gridspec_kw={"width_ratios": [1, 1.15]})
    cost_legend = draw_cost(cost_ax, cost, chosen)
    speedup_legend = draw_speedup(speedup_ax, predicted, measured, TICK_SIZE)
    fig.legend(handles=cost_legend + speedup_legend, loc="upper center", ncol=3, frameon=False, handletextpad=0.4,
               handlelength=1.6, columnspacing=1.0, borderaxespad=0.1, labelspacing=0.25)
    for ax, label in ((cost_ax, "(a) Cost per record, x86"), (speedup_ax, "(b) Predicted vs. measured")):
        ax.text(0.5, -0.33, label, transform=ax.transAxes, ha="center", va="top")
    fig.tight_layout(pad=0.2, w_pad=0.8, rect=(0, 0, 1, 0.83))
    save(fig, "post-join.pdf")


if __name__ == "__main__":
    plot_post_join(*load_data())

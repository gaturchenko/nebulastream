"""Source-rate x batch-size micro-benchmark: P95 latency vs. batch size at each offered rate.

Produces batch-rate.pdf: one line per offered rate over the fixed batch sizes, the 75 ms latency
SLO (the window size), the batch size OPTIMIZER_CHOOSES selected at each rate, and the batch sizes
that fell behind. Window-fed inference waits for a full batch, so a batch of B costs (B - 1)
strides of fill time: large batches miss the SLO at low rates, small ones fall behind at high rates.

Data: data/batch_rate/batch_rate_runs.csv
      (process_batch_rate.py over run_batch_rate.sh on the Pi, N-BEATS, one row per run; the
      figure takes the median over the 3 repetitions).
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from common import eng_formatter, load_csv, save, setup_style

OPTIMIZER_COLOR = "#e8730c"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    runs = load_csv("batch_rate", "batch_rate_runs.csv")
    fixed = (
        runs[runs["strategy"] == "FIXED"]
        .groupby(["offered_rate", "configured_batch_size"])["latency_ms_p95"]
        .median()
        .reset_index()
    )
    chosen = (
        runs[runs["strategy"] == "OPTIMIZER_CHOOSES"]
        .groupby("offered_rate")
        .agg(batch_size=("selected_batch_size", "median"), p95=("latency_ms_p95", "median"))
        .reset_index()
    )
    return fixed, chosen, runs["slo_ms"].iloc[0]


def draw_batch_rate(ax, fixed: pd.DataFrame, chosen: pd.DataFrame, slo_ms: float, tick_size: float) -> list:
    """Draws the latency panel into ax and returns its legend handles."""
    Y_TOP = 12000  # crosses for batch sizes without a completed run sit at half of it

    batch_sizes = sorted(fixed["configured_batch_size"].unique())
    rates = sorted(fixed["offered_rate"].unique())
    cmap = plt.get_cmap("Blues")
    colors = {rate: cmap(0.4 + 0.6 * i / (len(rates) - 1)) for i, rate in enumerate(rates)}

    ax.axhline(slo_ms, color="0.3", linestyle="--", linewidth=0.9, zorder=1)

    for rate in rates:
        series = fixed[fixed["offered_rate"] == rate].sort_values("configured_batch_size")
        ax.plot(series["configured_batch_size"], series["latency_ms_p95"], color=colors[rate], linewidth=1.1,
                marker="o", markersize=3, zorder=2)
        ax.annotate(f"{rate:.0f}/s", (series["configured_batch_size"].iloc[-1], series["latency_ms_p95"].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=tick_size, color="0.2")
        # A batch size without a completed run fell behind the offered rate; mark it at the top of the axis.
        missing = sorted(set(batch_sizes) - set(series["configured_batch_size"]))
        ax.scatter(missing, [Y_TOP * 0.5] * len(missing), marker="x", s=30, color=colors[rate], linewidths=1.5, zorder=3)
        for batch_size in missing:
            ax.annotate(f"{rate:.0f}/s", (batch_size, Y_TOP * 0.5), xytext=(0, -6), textcoords="offset points",
                        ha="center", va="top", fontsize=tick_size, color="0.2")

    ax.scatter(chosen["batch_size"], chosen["p95"], marker="*", s=70, color=OPTIMIZER_COLOR, edgecolors="white",
               linewidths=0.5, zorder=4)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # Plain 10, 100, 1k: the default 10^x labels print their exponents below the tick size.
    ax.yaxis.set_major_formatter(eng_formatter(decimals=0))
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([f"{b:.0f}" for b in batch_sizes])
    ax.set_xlim(batch_sizes[0] / 1.3, batch_sizes[-1] * 2.1)
    ax.set_ylim(3, Y_TOP)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("P95 latency [ms]")
    ax.grid(axis="y", which="major", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    return [
        Line2D([], [], color=cmap(0.7), linewidth=1.4, marker="o", markersize=3, label="fixed batch size"),
        Line2D([], [], color=OPTIMIZER_COLOR, marker="*", markersize=8, markeredgecolor="white", markeredgewidth=0.5,
               linestyle="none", label="optimizer's choice"),
        Line2D([], [], color=cmap(1.0), marker="x", markersize=5, linestyle="none", label="falls behind"),
        Line2D([], [], color="0.3", linestyle="--", linewidth=0.9, label=f"SLO ({slo_ms:.0f} ms window)"),
    ]


def plot_batch_rate(fixed: pd.DataFrame, chosen: pd.DataFrame, slo_ms: float) -> None:
    FIG_SIZE = (3.33, 2.3)
    FONT_SIZE = 8
    TICK_SIZE = 7

    setup_style()
    plt.rcParams.update({"font.size": FONT_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
                         "legend.fontsize": TICK_SIZE})
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    legend = draw_batch_rate(ax, fixed, chosen, slo_ms, TICK_SIZE)
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.2,
               borderaxespad=0.1)
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.84))
    save(fig, "batch-rate.pdf", tight=False)


if __name__ == "__main__":
    plot_batch_rate(*load_data())

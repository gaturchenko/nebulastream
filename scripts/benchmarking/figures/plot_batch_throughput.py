"""Source-rate x batch-size micro-benchmark: sustainable throughput vs. batch size.

Produces batch-throughput.pdf, the throughput side of batch-rate.pdf: per fixed batch size, the
largest offered rate it sustained (delivered >= 99% of it), with a bracket up to the next rate of
its ladder, which it did not sustain. The profiled model-only throughput B / t(B) and the
window-only query show what bounds it: batching amortizes the model invocation, but every result
also costs windowing, batching and emitting, so the measured throughput levels off below B / t(B).

Each run is rate-limited exactly like the latency sweep, so only the offered rate differs; a run
that falls behind is backpressured, so sustainability is judged by the delivered rate.

Data: data/batch_rate/batch_throughput_runs.csv
      (process_batch_throughput.py over run_batch_throughput.sh on the Pi, one row per cell and
      repetition; a cell counts as sustained if every repetition sustained it),
      data/batch_rate/batch_rate_profile.csv (the worker's profile behind OPTIMIZER_CHOOSES).
"""

import matplotlib.pyplot as plt
import pandas as pd

from common import eng_formatter, load_csv, save, setup_style

MEASURED_COLOR = plt.get_cmap("Blues")(0.85)
# Opaque rather than alpha-blended: transparency makes some PDF viewers rasterize the page.
BRACKET_COLOR = plt.get_cmap("Blues")(0.55)
REFERENCE_COLOR = "0.45"


def load_data() -> tuple[pd.DataFrame, pd.Series, float]:
    runs = load_csv("batch_rate", "batch_throughput_runs.csv")
    cells = runs.groupby(["query", "batch_size", "offered_rate"], dropna=False)["sustained"].all().reset_index()

    inference = cells[cells["query"] == "inference"]
    rows = []
    for batch_size, ladder in inference.groupby("batch_size"):
        sustained = ladder.loc[ladder["sustained"], "offered_rate"].max()
        missed = ladder.loc[~ladder["sustained"] & (ladder["offered_rate"] > sustained), "offered_rate"].min()
        rows.append({"batch_size": int(batch_size), "sustained": sustained, "missed": missed})
    throughput = pd.DataFrame(rows).sort_values("batch_size")

    window_only = cells[(cells["query"] == "window only") & cells["sustained"]]["offered_rate"].max()
    profile = load_csv("batch_rate", "batch_rate_profile.csv").groupby("batch_size")["throughput"].median()
    return throughput, profile, window_only


def draw_batch_throughput(
    ax, throughput: pd.DataFrame, profile: pd.Series, window_only: float, label_size: float
) -> list:
    """Draws the throughput panel into ax and returns its legend handles, the measured series first."""
    ax.plot(profile.index, profile.values, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, marker="o",
            markersize=2.5, label=r"model only, profiled $B / t(B)$")
    ax.axhline(window_only, color=REFERENCE_COLOR, linestyle=":", linewidth=1.0, label="window only")
    ax.vlines(throughput["batch_size"], throughput["sustained"], throughput["missed"], color=BRACKET_COLOR,
              linewidth=0.8)
    ax.scatter(throughput["batch_size"], throughput["missed"], marker="_", s=25, color=BRACKET_COLOR, linewidths=0.8)
    ax.plot(throughput["batch_size"], throughput["sustained"], color=MEASURED_COLOR, linewidth=1.4, marker="o",
            markersize=3.5, label="sustained; bar: next rate, missed")
    rate_format = eng_formatter(decimals=2)
    for row in throughput.itertuples(index=False):
        ax.annotate(rate_format(row.sustained, None), (row.batch_size, row.sustained), xytext=(3, -3), textcoords="offset points",
                    ha="left", va="top", fontsize=label_size, color="0.2")

    batch_sizes = throughput["batch_size"].tolist()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_ylim(300, 20000)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput [results/s]")
    ax.grid(axis="y", which="major", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    handles, _ = ax.get_legend_handles_labels()
    return [handles[2], handles[0], handles[1]]


def plot_batch_throughput(throughput: pd.DataFrame, profile: pd.Series, window_only: float) -> None:
    FIG_SIZE = (3.33, 2.3)
    FONT_SIZE = 8
    TICK_SIZE = 7

    setup_style()
    plt.rcParams.update({"font.size": FONT_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
                         "legend.fontsize": TICK_SIZE})
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    legend = draw_batch_throughput(ax, throughput, profile, window_only, TICK_SIZE)
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.2,
               borderaxespad=0.1)
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.84))
    save(fig, "batch-throughput.pdf", tight=False)


if __name__ == "__main__":
    plot_batch_throughput(*load_data())

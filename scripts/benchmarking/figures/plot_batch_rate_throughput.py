"""Source-rate x batch-size micro-benchmark: latency and sustainable throughput side by side.

Produces batch-rate-throughput.pdf, a full-width (figure*) combination of batch-rate.pdf (left:
P95 latency per fixed batch size at each offered rate, with the optimizer's choice) and
batch-throughput.pdf (right: the sustainable throughput of each batch size), so that the latency
cost and the throughput gain of a batch size read off the same x-axis, and
batch-rate-throughput-column.pdf, the same two panels side by side at the width of one column.

Data: see plot_batch_rate.py and plot_batch_throughput.py.
"""

import matplotlib.pyplot as plt

import plot_batch_rate
import plot_batch_throughput
from common import save, setup_style


def plot_batch_rate_throughput() -> None:
    FIG_SIZE = (7.0, 2.05)
    FONT_SIZE = 8
    TICK_SIZE = 7

    setup_style()
    plt.rcParams.update({"font.size": FONT_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
                         "legend.fontsize": TICK_SIZE})
    fig, (latency_ax, throughput_ax) = plt.subplots(1, 2, figsize=FIG_SIZE)

    latency_legend = plot_batch_rate.draw_batch_rate(latency_ax, *plot_batch_rate.load_data(), TICK_SIZE)
    throughput_legend = plot_batch_throughput.draw_batch_throughput(
        throughput_ax, *plot_batch_throughput.load_data(), TICK_SIZE
    )
    # The same x range in both panels, so that a batch size sits at the same position in each.
    throughput_ax.set_xlim(latency_ax.get_xlim())
    # Each panel carries its own legend above it and its label below, as a sub-caption.
    for ax, legend, label in (
        (latency_ax, latency_legend, "(a) P95 latency at each offered rate"),
        (throughput_ax, throughput_legend, "(b) Sustainable throughput"),
    ):
        ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                  handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2)
        ax.set_xlabel(f"Batch size\n{label}")
    fig.tight_layout(pad=0.2, w_pad=2.0)
    save(fig, "batch-rate-throughput.pdf", tight=False)


def plot_batch_rate_throughput_column() -> None:
    """The same panels side by side at the width of one column (3.5 in), without legends: the measured series are
    described in the text, the optimizer's choices and the missed-rate bars in the caption, and the reference lines and
    the fallen-behind run are labeled in place. Only the lowest and highest offered rates are labeled at the line ends
    (the color ramp orders the rest)."""
    FIG_SIZE = (3.5, 1.85)
    FONT_SIZE = 8
    TICK_SIZE = 7
    LABEL_COLOR = "0.2"

    setup_style()
    plt.rcParams.update({"font.size": FONT_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE})
    fig, (latency_ax, throughput_ax) = plt.subplots(1, 2, figsize=FIG_SIZE)

    fixed, chosen, slo_ms = plot_batch_rate.load_data()
    plot_batch_rate.draw_batch_rate(latency_ax, fixed, chosen, slo_ms, TICK_SIZE)
    throughput, profile, window_only = plot_batch_throughput.load_data()
    plot_batch_throughput.draw_batch_throughput(throughput_ax, throughput, profile, window_only, TICK_SIZE)
    throughput_ax.set_xlim(latency_ax.get_xlim())

    largest_batch = fixed["configured_batch_size"].max()
    extreme_rates = {f"{fixed['offered_rate'].min():.0f}/s", f"{fixed['offered_rate'].max():.0f}/s"}
    for text in list(latency_ax.texts):
        if text.xy[0] == largest_batch and text.get_text() not in extreme_rates:
            text.remove()
        elif text.get_text() == "1000/s":
            text.set_text("1k/s")
    # The run that fell behind: its rate below the cross would run into the y-axis when centered, so the label starts
    # at the cross, and "falls behind" sits to its right.
    for text in latency_ax.texts:
        if text.xy[0] != largest_batch:
            text.set_horizontalalignment("left")
            text.set_position((-3, -6))
            latency_ax.annotate("falls behind", text.xy, xytext=(6, 0), textcoords="offset points", ha="left",
                                va="center", fontsize=TICK_SIZE, color=LABEL_COLOR)

    # Reference lines, labeled next to the y-axis.
    left = latency_ax.get_xlim()[0] * 1.08
    # "SLO" with its value below it, above the line next to the y-axis, where no line passes.
    latency_ax.annotate(f"SLO\n({slo_ms:.0f} ms)", (left, slo_ms), xytext=(0, 2), textcoords="offset points", ha="left",
                        va="bottom", fontsize=TICK_SIZE, color=LABEL_COLOR, linespacing=1.0)
    throughput_ax.annotate("window only", (left, window_only), xytext=(0, 1.5), textcoords="offset points", ha="left",
                           va="bottom", fontsize=TICK_SIZE, color=LABEL_COLOR)
    # The profiled model-only throughput, labeled left of its largest batch size.
    throughput_ax.annotate(r"model only, $B / t(B)$", (profile.index[-1], profile.values[-1]), xytext=(-5, 0),
                           textcoords="offset points", ha="right", va="center", fontsize=TICK_SIZE,
                           color=plot_batch_throughput.REFERENCE_COLOR)

    for ax, label in ((latency_ax, "(a) P95 latency"), (throughput_ax, "(b) Sustainable throughput")):
        ax.set_xlabel(f"Batch size\n{label}")
    fig.tight_layout(pad=0.2, w_pad=0.8)
    save(fig, "batch-rate-throughput-column.pdf", tight=False)


if __name__ == "__main__":
    plot_batch_rate_throughput()
    plot_batch_rate_throughput_column()

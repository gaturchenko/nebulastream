"""Source-rate x batch-size micro-benchmark: latency and sustainable throughput side by side.

Produces batch-rate-throughput.pdf, a full-width (figure*) combination of batch-rate.pdf (left:
P95 latency per fixed batch size at each offered rate, with the optimizer's choice) and
batch-throughput.pdf (right: the sustainable throughput of each batch size), so that the latency
cost and the throughput gain of a batch size read off the same x-axis.

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


if __name__ == "__main__":
    plot_batch_rate_throughput()

"""Where TorchServe's end-to-end latency goes (Raspberry Pi).

Produces torchserve-breakdown.pdf: the serving-boundary cost decomposition the revision asks for,
splitting TorchServe's mean end-to-end latency into the work NAVI does not have to do (getting the
tuple onto the wire, across HTTP/IPC and through the worker queue) and the handler call itself.

The split comes from TorchServe's own Prometheus counters scraped around each run, so it is
measured rather than inferred. Legend label <- what is actually timed:
    Serialization        : post_ts - creation_ts, the NAVI output buffer becoming a request body
    Transport (HTTP,IPC) : the remainder of the sink's round trip once queue and inference are
                           removed, i.e. HTTP over loopback plus the frontend -> worker IPC hop
    Queueing             : ts_queue_latency_microseconds     / requests
    Inference            : ts_inference_latency_microseconds / requests

Two caveats the caption should carry. "Inference" is the whole handler call -- decode, reshape,
model invocation and response build -- not purely the model; it is labelled this way because it is
the stage NAVI also pays, which is what makes the other three the boundary cost. And these are
means, which is what makes the four parts sum to the end-to-end mean; percentiles would not add up.

The stack uses the documented blue ordinal ramp rather than the categorical hues the other systems
figures use: this chart has no competing systems in it, and the four parts are an ordered timeline,
not four identities.

Data: data/systems/torchserve_breakdown.csv (aggregate_systems_data.py over navi's
      --tuned over navi's tuning runs, TorchServe at the worker count its pilot chose; medians across
      the 5 repetitions, in the order a request travels).
"""

import numpy as np
import matplotlib.pyplot as plt

from common import save, setup_style
from systems_common import WORKLOAD_TITLE, WORKLOADS, load

# Bottom-to-top: the order a request actually travels. Keys match the CSV's "part" column; the
# values are how each stage is labelled in the legend.
#
# The labels name what each stage COSTS, not where we measured it. A reader of the paper has no
# model of our sink, TorchServe's frontend/backend split or NAVI's output buffers, so the earlier
# names ("buffer -> post", "HTTP/IPC", "worker queue", "handler") described plumbing they have
# never seen. These four are the standard stages of any remote-serving round trip, which is the
# comparison the figure is making: three of them are the cost of crossing the serving boundary and
# the fourth is the work NAVI also does.
PARTS = {
    "buffer_to_post": "Serialization",
    "http_ipc": "Transport (HTTP, IPC)",
    "worker_queue": "Queueing",
    "handler": "Inference",
}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]  # blue ordinal steps 250/350/450/600


def load_breakdown() -> dict:
    df = load("torchserve_breakdown.csv").set_index(["workload", "part"])
    return {w: [df.loc[(w, part), "mean_ms"] for part in PARTS] for w in WORKLOADS}


def plot_breakdown(data: dict) -> None:
    FIGSIZE = (5.6, 3.2)
    LABEL_FS = 14
    TICK_FS = 12
    VALUE_FS = 9.5
    LEGEND_FS = 10.5
    BAR_WIDTH = 0.55
    MIN_LABEL_MS = 6.0  # below this a segment is too thin to hold its own number

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(WORKLOADS))

    bottoms = np.zeros(len(WORKLOADS))
    for part_index, (part, color) in enumerate(zip(PARTS.values(), RAMP)):
        values = np.array([data[w][part_index] for w in WORKLOADS])
        ax.bar(x, values, width=BAR_WIDTH, bottom=bottoms, color=color,
               edgecolor="white", linewidth=1.2, label=part)
        # Selective direct labels: only segments thick enough to read, so the thin transport
        # slices do not collide with their neighbours.
        for xi, (value, bottom) in enumerate(zip(values, bottoms)):
            if value >= MIN_LABEL_MS:
                ax.text(xi, bottom + value / 2, f"{value:.0f}", ha="center", va="center",
                        fontsize=VALUE_FS, color="white" if part_index >= 2 else "black")
        bottoms += values

    for xi, total in zip(x, bottoms):
        ax.text(xi, total * 1.02, f"{total:.0f} ms", ha="center", va="bottom", fontsize=VALUE_FS)

    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_TITLE[w] for w in WORKLOADS], fontsize=LABEL_FS)
    ax.set_ylabel("Mean end-to-end latency [ms]", fontsize=LABEL_FS)
    ax.set_ylim(0, bottoms.max() * 1.16)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", frameon=False, fontsize=LEGEND_FS)

    fig.tight_layout()
    save(fig, "torchserve-breakdown.pdf", pad_inches=0.1)


if __name__ == "__main__":
    plot_breakdown(load_breakdown())

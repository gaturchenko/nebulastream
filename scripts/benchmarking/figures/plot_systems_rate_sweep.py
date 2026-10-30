"""Latency and throughput versus offered load, Waymo/SqueezeDet on the Raspberry Pi.

Produces systems-rate-sweep.pdf (and each of its two panels on its own, systems-rate-sweep-throughput.pdf and
systems-rate-sweep-latency.pdf): NAVI vs. Flink+OpenVINO UDF vs. TorchServe across a ladder of
ingestion rates, the curve the single 10 frames/s operating point cannot give -- it cannot separate
"this system is faster" from "both are mostly idle and this one's fixed overhead is smaller".

Why throughput is the left panel and the headline. Past saturation the three systems express
overload in incompatible ways, so their latency curves stop being comparable: NAVI and Flink pace
a blocking source, so achieved throughput caps while latency stays near its unloaded value, whereas
TorchServe's bottleneck is the serial HTTP sink at the END of the pipeline, so buffers queue in
front of it and the wait lands inside the measured interval. Achieved throughput is the honest
saturation signal -- once a curve leaves the dashed identity line, that system is past its knee.

Two caveats for the caption. (1) The input is a fixed 991 frames, so a higher rate means a shorter
run; below saturation latency is stationary and every point rests on the same 991 samples, but
above it a backlogged system's latency grows with elapsed time, so those points understate the
steady-state tail. (2) Only Waymo is swept: its MOCK_TUPLE_RATE is directly the inference rate,
whereas for CWRU and MUSAN that knob is the SAMPLE rate, and raising it also shortens the span each
window covers -- sweeping those means moving ADVANCE, a different experiment.

The Flink arm runs one inference lane, the setting its tuning pilot chose at 10 frames/s (as in the
workload matrix, e2e-systems-workloads.pdf), so the two figures agree at that point. One lane saturates
at ~17.5 frames/s; with the two lanes the .test implies, Flink reached ~19.4 frames/s but had a P95 of
~92 ms at 10 frames/s and ~197 ms past its knee (data/systems/rate_sweep_flink_lanes2.csv, the same
aggregate over the runs archived on the Pi in ~/navi/results-sweep-flink-lanes-from-test; not plotted). TorchServe runs the 3 workers its pilot chose, with 3 HTTP
sink connections on NES's side; more workers do not move its knee (the earlier 1-worker, 2-connection
curve saturated at ~33 instead of ~31.5 frames/s, archived in ~/navi/results-sweep-torchserve-1worker).
NAVI's arm already ran at its tuned 2 worker threads.

Note on the Flink arm's history: it was re-measured after its per-line CSV parser -- a full commons-csv
parser built per 360 KB line to take one column of a two-column file -- was replaced with a direct
column slice. That defect capped the job at 14.5 frames/s while its inference lanes could serve
~58/s, so the curve here is the corrected one. The pre-fix runs are still on the Pi alongside the
corrected ones (the plain file names; the corrected ones carry a "_fastparse" suffix).

Data: data/systems/rate_sweep.csv (aggregate_systems_data.py over navi's
      run_systems_rate_sweep.sh output; one row per series x rate, median across 3 repetitions,
      latency also carrying min-max).
"""

import matplotlib.pyplot as plt
import numpy as np

from common import save, setup_style
from systems_common import COLOR, LABEL, ORDER, load

RATES = [10, 15, 20, 30, 45, 65, 90]
LINEWIDTH = 2.0
MARKERSIZE = 5.5


def series(name: str) -> dict[str, list]:
    """One series' curve, ordered by RATES rather than by the CSV's row order."""
    rows = load("rate_sweep.csv")
    rows = rows[rows["series"] == name].set_index("rate")
    return {
        "throughput": [rows.loc[r, "throughput_median"] for r in RATES],
        "p95": [rows.loc[r, "p95_median_ms"] for r in RATES],
        "lo": [rows.loc[r, "p95_min_ms"] for r in RATES],
        "hi": [rows.loc[r, "p95_max_ms"] for r in RATES],
    }


def _dress(ax, xlabel, ylabel, style):
    ax.set_xlabel(xlabel, fontsize=style["label_fs"], labelpad=style["labelpad"])
    ax.set_ylabel(ylabel, fontsize=style["label_fs"], labelpad=style["labelpad"])
    ax.set_xticks(RATES)
    ax.set_xticklabels([str(r) for r in RATES])
    ax.tick_params(axis="both", labelsize=style["tick_fs"], pad=style["tickpad"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


# The two-panel figure, two panels of PANEL_SIZE side by side, is scaled down for slides.
STYLE = dict(label_fs=14, tick_fs=11, legend_fs=11, labelpad=4, tickpad=3.5, linewidth=LINEWIDTH, markersize=MARKERSIZE,
             identity_width=1.2, errorbar_width=1.0, capsize=3, throughput_label="Achieved throughput [frames/s]")
PANEL_SIZE = (4.6, 3.4)
# Each panel on its own is drawn at the size the paper prints it, one column (3.5 in), so every font size is the size on
# the page. The paper's captions are 8 pt; nothing here goes below 7 pt.
PANEL_STYLE = dict(label_fs=8, tick_fs=7.5, legend_fs=7.5, labelpad=2, tickpad=2, linewidth=1.2, markersize=3.5,
                   identity_width=0.8, errorbar_width=0.6, capsize=1.8, throughput_label="Throughput [frames/s]")
PRINTED_PANEL_SIZE = (3.5, 1.35)


def draw_throughput(ax, data: dict, style: dict = STYLE) -> None:
    # Identity: a system still keeping up sits on this line. Drawn first so the series sit above it.
    ax.plot(RATES, RATES, linestyle="--", linewidth=style["identity_width"], color="0.45", label="offered load", zorder=1)
    for system in ORDER:
        ax.plot(RATES, data[system]["throughput"], marker="o", markersize=style["markersize"],
                linewidth=style["linewidth"], color=COLOR[system], label=LABEL[system], zorder=3)
    _dress(ax, "Offered load [frames/s]", style["throughput_label"], style)


def draw_latency(ax, data: dict, style: dict = STYLE) -> None:
    for system in ORDER:
        # Whiskers: min-max of P95 over the repetitions at each measured rate. A band between the points would spread the one
        # unsettled point, at a system's knee, along the whole steep rise on the log axis.
        p95 = np.asarray(data[system]["p95"])
        ax.errorbar(RATES, p95, yerr=[p95 - np.asarray(data[system]["lo"]), np.asarray(data[system]["hi"]) - p95],
                    marker="o", markersize=style["markersize"], linewidth=style["linewidth"], color=COLOR[system],
                    label=LABEL[system], elinewidth=style["errorbar_width"], capsize=style["capsize"])
    _dress(ax, "Offered load [frames/s]", "Latency P95 [ms]", style)
    ax.set_yscale("log")


def plot_rate_sweep(data: dict) -> None:
    """Both panels side by side: systems-rate-sweep.pdf."""
    setup_style()
    fig, (ax_thr, ax_lat) = plt.subplots(1, 2, figsize=(2 * PANEL_SIZE[0], PANEL_SIZE[1]))
    draw_throughput(ax_thr, data)
    draw_latency(ax_lat, data)
    ax_thr.legend(loc="upper left", frameon=False, fontsize=STYLE["legend_fs"])
    fig.tight_layout()
    save(fig, "systems-rate-sweep.pdf", pad_inches=0.1)


def plot_rate_sweep_panels(data: dict) -> None:
    """Each panel as a figure of its own, each with its own legend: systems-rate-sweep-{throughput,latency}.pdf."""
    for name, draw, legend_loc in [("throughput", draw_throughput, "upper left"),
                                   ("latency", draw_latency, "upper left")]:
        setup_style()
        fig, ax = plt.subplots(figsize=PRINTED_PANEL_SIZE)
        draw(ax, data, PANEL_STYLE)
        ax.legend(loc=legend_loc, frameon=False, fontsize=PANEL_STYLE["legend_fs"], handlelength=1.6, labelspacing=0.3,
                  borderaxespad=0.2)
        fig.tight_layout(pad=0.2)
        save(fig, f"systems-rate-sweep-{name}.pdf")


if __name__ == "__main__":
    sweep = {system: series(system) for system in ORDER}
    plot_rate_sweep(sweep)
    plot_rate_sweep_panels(sweep)

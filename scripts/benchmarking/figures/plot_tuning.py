"""Systems comparison with every system tuned the same way, per workload (CWRU anomaly detection, Waymo object detection,
MUSAN speech recognition).

Produces <workload>-tuning.pdf, two panels of per-run latency percentiles (one dot per run):
  (a) the chosen configurations measured afresh: P50, P95 and P99 of NAVI, Flink and TorchServe, a tick at each median, so
      that the dispersion between runs is visible
  (b) the pilot that chose them: P95 of each system over its parallelism knob (NAVI worker threads, Flink inference lanes,
      TorchServe workers), the chosen setting ringed

Data: data/systems/<workload>_tuning_runs.csv, OpenVINO 1 stream x 4 threads per inference lane, per-run percentiles from
      run_systems.sh on the Pi:
  cwru: run_pi_cwru_tuning.sh, X228 looped to 61 s, ~600 windows per run
  waymo: run_pi_systems_tuning.sh, the 991 frames at 10 frames/s
  musan: run_pi_systems_tuning.sh, a 200 s cut of the US-gov 0242 recording, ~600 windows per run
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import save, setup_style
from systems_common import COLOR, LABEL, ORDER, load

KNOB_LABEL = {"w": "workers", "lanes": "lanes", "workers": "workers"}
PERCENTILES = [("p50_ms", "P50"), ("p95_ms", "P95"), ("p99_ms", "P99")]


def draw_final(ax, final: pd.DataFrame, marker_size: float) -> None:
    """Per-run P50/P95/P99 of the chosen configurations, one cluster per percentile."""
    spread = 0.22
    for p, (column, _) in enumerate(PERCENTILES):
        for s, system in enumerate(ORDER):
            values = final.loc[final["system"] == system, column].to_numpy()
            x = p + (s - 1) * spread
            jitter = np.linspace(-0.05, 0.05, len(values))
            ax.scatter(x + jitter, values, s=marker_size, color=COLOR[system], edgecolors="white", linewidths=0.5, zorder=3)
            ax.hlines(np.median(values), x - 0.08, x + 0.08, color=COLOR[system], linewidth=2.0, zorder=4)
    ax.set_xticks(range(len(PERCENTILES)))
    ax.set_xticklabels([label for _, label in PERCENTILES])
    ax.set_xlim(-0.5, len(PERCENTILES) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Latency [ms]")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def draw_pilot(ax, pilot: pd.DataFrame, chosen: dict, marker_size: float) -> None:
    """Per-run P95 over each system's parallelism setting; the chosen setting is ringed."""
    positions, labels, x = [], [], 0.0
    for system in ORDER:
        settings = sorted(pilot.loc[pilot["system"] == system, "setting"].unique())
        for setting in settings:
            values = pilot.loc[(pilot["system"] == system) & (pilot["setting"] == setting), "p95_ms"].to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(x + jitter, values, s=marker_size, color=COLOR[system], edgecolors="white", linewidths=0.5, zorder=3)
            ax.hlines(np.median(values), x - 0.18, x + 0.18, color=COLOR[system], linewidth=2.0, zorder=4)
            if chosen.get(system) == setting:
                ax.scatter(x, np.median(values), s=marker_size * 6, facecolors="none", edgecolors="0.2", linewidths=0.9,
                           zorder=5)
            positions.append(x)
            labels.append(str(setting))
            x += 1
        x += 0.6
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("P95 latency [ms]")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    # Which knob each group of settings is.
    start = 0.0
    for system in ORDER:
        settings = sorted(pilot.loc[pilot["system"] == system, "setting"].unique())
        knob = pilot.loc[pilot["system"] == system, "knob"].iloc[0]
        ax.text(start + (len(settings) - 1) / 2, -0.17, KNOB_LABEL[knob], transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="0.3")
        start += len(settings) + 0.6


def knob_count(setting: int, knob: str) -> str:
    """'2 workers', '1 lane'."""
    label = KNOB_LABEL[knob]
    return f"{setting} {label.removesuffix('s') if setting == 1 else label}"


def runs_label(phase: pd.DataFrame) -> str:
    """'5 runs', or '2-3 runs' while some systems have fewer."""
    counts = phase.groupby(["system", "setting"]).size()
    return f"{counts.min()} runs" if counts.min() == counts.max() else f"{counts.min()}\u2013{counts.max()} runs"


def plot_tuning(runs: pd.DataFrame, workload: str) -> None:
    FIG_SIZE = (7.0, 2.7)
    FONT_SIZE = 8
    TICK_SIZE = 7
    MARKER_SIZE = 18

    setup_style()
    plt.rcParams.update({"font.size": FONT_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
                         "legend.fontsize": TICK_SIZE})
    final = runs[runs["phase"] == "final"]
    pilot = runs[runs["phase"] == "pilot"]
    chosen = final.groupby("system")["setting"].first().to_dict()

    fig, (final_ax, pilot_ax) = plt.subplots(1, 2, figsize=FIG_SIZE, gridspec_kw={"width_ratios": [1, 1.25]})
    draw_final(final_ax, final, MARKER_SIZE)
    draw_pilot(pilot_ax, pilot, chosen, MARKER_SIZE)
    handles = [Line2D([], [], color=COLOR[s], marker="o", linestyle="none", markersize=4.5, markeredgecolor="white",
                      label=f"{LABEL[s]} ({knob_count(chosen[s], final.loc[final['system'] == s, 'knob'].iloc[0])})")
               for s in ORDER]
    handles += [Line2D([], [], color="0.3", linewidth=2.0, label="median of runs"),
                Line2D([], [], color="0.2", marker="o", markerfacecolor="none", linestyle="none", markersize=7,
                       label="chosen in pilot")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, handletextpad=0.4, columnspacing=1.2)
    final_ax.text(0.5, -0.2, f"(a) Chosen configurations, {runs_label(final)}", transform=final_ax.transAxes, ha="center",
                  va="top")
    pilot_ax.text(0.5, -0.27, f"(b) Pilot: P95 per setting, {runs_label(pilot)}", transform=pilot_ax.transAxes, ha="center",
                  va="top")
    fig.tight_layout(pad=0.3, w_pad=1.5, rect=(0, 0, 1, 0.9))
    save(fig, f"{workload}-tuning.pdf")


if __name__ == "__main__":
    for workload in ("cwru", "waymo", "musan"):
        plot_tuning(load(f"{workload}_tuning_runs.csv"), workload)

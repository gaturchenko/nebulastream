"""Systems comparison across all three end-to-end workloads (Raspberry Pi).

Produces two figures.

e2e-systems-workloads.pdf: P95 end-to-end latency of NAVI vs. the Flink+OpenVINO UDF
baseline vs. TorchServe external serving, on Waymo/SqueezeDet, CWRU/SARAD and MUSAN/MHAtt-RNN.
It supersedes the single-workload e2e-systems.pdf, which the reviewers asked us to extend to all
three workloads; that figure and its data are left in place. TorchServe's P95 bar is split into
inference and serving boundary in the proportions of its MEAN breakdown (see below for why the
split is measured on means); the caption has to say so, because the shares at the tail need not
match the shares on average.

Flink's bar is split too: solid up to its P95 over the results whose input arrived after the first 5 s
of the run, hatched (as TorchServe's boundary) up to its full-run P95 (the bar's height and label). Both come from the same 5
traced runs at the tuned lanes (run_pi_flink_startup.sh, data/systems/flink_startup_runs.csv), which
also replace the tuning finals as Flink's bar, so the two parts cannot come from different runs. The
trace (WindowTrace in the Flink baseline) showed why CWRU needs this: for its first ~3 s both lanes
infer at once at ~200 ms per window instead of ~73 ms, which is exactly the offered 10 windows/s, so
the start-up backlog cannot drain. That is ~30 of 608 windows, right at the 5% a P95 sits on, so the
full-run P95 swings between runs (225-338 ms here, 196-388 ms in the tuning finals) while the P95
after start-up holds at 90-93 ms. Waymo and MUSAN have no such phase (66.3 vs 65.8, 141.3 vs 138.9 ms).
NAVI's and TorchServe's bars stay full-run: their per-result data was not traced, so the caption should
say that only Flink's start-up is broken out.

Each workload gets its own panel and its own y-axis. A shared axis would be wrong here: the
workloads run at different offered loads against different models, so their absolute latencies are
not meant to be compared with one another -- only the three systems within a panel are. (A shared
log axis would invite exactly that cross-panel reading while also breaking the zero baseline bars
need.)

Every system runs at the parallelism a pilot chose for it on that workload (NAVI worker threads,
Flink inference lanes, TorchServe workers; lowest median P95 among the settings that keep up), the
same procedure for all three. Bars are the median of 5 fresh repetitions of the chosen
configurations and the whisker spans min-max. Single runs are not enough on these workloads: one
CWRU Flink cell produced 97 ms and 236 ms inference medians on consecutive runs, and a single run
had TorchServe's CWRU P95 beating NAVI's, which repetitions reverse.

e2e-systems-workloads-split.pdf: the same comparison on MEAN latency, with TorchServe's bar split
into the inference it performs and the cost of reaching it over the serving boundary. It exists so
the cost breakdown the revision asks for does not need a figure of its own.

The split has to be drawn on means, not on P95. Stage times only add up in expectation: the P95 of
a sum is not the sum of the stage P95s, so segmenting a P95 bar by stage would assume every stage
hits its tail on the same request and would overstate the total. Two of the four stages are also
mean-only at the source -- they come from cumulative Prometheus counters divided by the request
count, so no distribution exists to take a percentile of. Reporting the mean here changes no
conclusion: the mean and P95 rank the three systems identically on all three workloads.

Inference sits at the BOTTOM of the TorchServe bar (in both figures) on purpose: the solid part is then what
TorchServe would cost if the boundary were free, which can be read straight across against NAVI's
bar beside it.

Data: data/systems/workloads_p95.csv and data/systems/torchserve_breakdown.csv
      (aggregate_systems_data.py --tuned over navi's tuning runs: each system at the parallelism its
      pilot chose, then 5 fresh repetitions; medians plus min-max across them), and for Flink's P95
      bars data/systems/flink_startup_runs.csv (one row per traced run, full-run and post-start-up
      percentiles; raw traces on the Pi in ~/navi/results-flink-startup).
"""

import numpy as np
import matplotlib.pyplot as plt

from common import save, setup_style
from systems_common import (COLOR, FLINK, LABEL, ORDER, TORCHSERVE, WORKLOAD_RATE, WORKLOAD_TITLE,
                            WORKLOADS, load)

# Everything in TorchServe's mean latency that is not the handler call is the price of crossing the
# serving boundary: building the request body, moving it over HTTP and the frontend -> worker IPC
# hop, and waiting in the worker queue.
BOUNDARY_PARTS = ["buffer_to_post", "http_ipc", "worker_queue"]


def load_stat(stat: str) -> dict:
    df = load("workloads_p95.csv").set_index(["workload", "system"])
    return {key: (row[f"{stat}_median_ms"], row[f"{stat}_min_ms"], row[f"{stat}_max_ms"])
            for key, row in df.iterrows()}


def load_flink_startup() -> tuple[dict, dict]:
    """Flink's P95 from the traced runs: (median, min, max) of the full-run P95, and the median P95 after start-up."""
    df = load("flink_startup_runs.csv").groupby("workload")
    full = {w: (g["p95_full_ms"].median(), g["p95_full_ms"].min(), g["p95_full_ms"].max()) for w, g in df}
    steady = {w: g["p95_steady_ms"].median() for w, g in df}
    return full, steady


def inference_share(split: dict) -> dict:
    """Per workload: the fraction of TorchServe's mean latency that is inference."""
    return {w: inference / (inference + boundary) for w, (inference, boundary) in split.items()}


def load_torchserve_split() -> dict:
    """Per workload: (inference, boundary) [ms], summing to TorchServe's mean end-to-end latency."""
    df = load("torchserve_breakdown.csv").set_index(["workload", "part"])
    return {w: (df.loc[(w, "handler"), "mean_ms"],
                sum(df.loc[(w, part), "mean_ms"] for part in BOUNDARY_PARTS))
            for w in WORKLOADS}


def plot_e2e_workloads(data: dict, share: dict, flink_steady: dict) -> None:
    """P95 per system; TorchServe's bar is split in the proportions of its mean breakdown, Flink's at its
    P95 without the first seconds of the run."""
    FIGSIZE = (9.5, 3.0)
    LABEL_FS = 15
    TICK_FS = 12
    VALUE_FS = 10
    LEGEND_FS = 11.5
    TITLE_FS = 13
    BAR_WIDTH = 0.62
    HEADROOM = 1.30  # y-limit factor above the tallest bar, leaving room for the value labels

    setup_style()
    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=FIGSIZE)

    for ax, workload in zip(axes, WORKLOADS):
        x = np.arange(len(ORDER))
        medians = [data[(workload, s)][0] for s in ORDER]
        lower = [data[(workload, s)][0] - data[(workload, s)][1] for s in ORDER]
        upper = [data[(workload, s)][2] - data[(workload, s)][0] for s in ORDER]

        top = max(np.array(medians) + np.array(upper)) * HEADROOM
        for index, system in enumerate(ORDER):
            if system == FLINK:
                # Solid = P95 once the run is past its start-up; hatched = what the start-up adds to
                # the full-run P95 (the bar's height and its label), drawn like TorchServe's boundary.
                steady = min(flink_steady[workload], medians[index])
                ax.bar(index, steady, width=BAR_WIDTH, color=COLOR[system], edgecolor="black",
                       linewidth=0.8)
                ax.bar(index, medians[index] - steady, width=BAR_WIDTH, bottom=steady,
                       facecolor="white", edgecolor=COLOR[system], linewidth=0.8, hatch="///")
                if medians[index] - steady > 0.12 * top:
                    ax.text(index, steady + 0.01 * top, f"{steady:.0f}", ha="center", va="bottom",
                            fontsize=VALUE_FS, style="italic",
                            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
                continue
            if system != TORCHSERVE:
                ax.bar(index, medians[index], width=BAR_WIDTH, color=COLOR[system],
                       edgecolor="black", linewidth=0.8)
                continue
            inference = medians[index] * share[workload]
            ax.bar(index, inference, width=BAR_WIDTH, color=COLOR[system],
                   edgecolor="black", linewidth=0.8)
            ax.bar(index, medians[index] - inference, width=BAR_WIDTH, bottom=inference,
                   facecolor="white", edgecolor=COLOR[system], linewidth=0.8, hatch="///")

        ax.errorbar(x, medians, yerr=[lower, upper], fmt="none", ecolor="black",
                    elinewidth=0.9, capsize=3)
        # Direct value labels: the palette's aqua sits below 3:1 against white, so identity and
        # magnitude must not rest on the fill alone.
        for index, value in enumerate(medians):
            ax.text(index, value + upper[index], f"{value:.0f}", ha="center", va="bottom",
                    fontsize=VALUE_FS)

        ax.set_title(f"{WORKLOAD_TITLE[workload]}\n{{\\small {WORKLOAD_RATE[workload]}}}",
                     fontsize=TITLE_FS)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylim(0, top)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Latency P95 [ms]", fontsize=LABEL_FS)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[s], edgecolor="black", linewidth=0.8)
               for s in ORDER]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=COLOR[FLINK],
                                 linewidth=0.8, hatch="///"))
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=COLOR[TORCHSERVE],
                                 linewidth=0.8, hatch="///"))
    labels = [LABEL[s] for s in ORDER] + ["Flink start-up (first 5 s)", "serving boundary"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=LEGEND_FS, bbox_to_anchor=(0.5, -0.07))

    fig.tight_layout()
    save(fig, "e2e-systems-workloads.pdf", pad_inches=0.1)


def plot_e2e_split(data: dict, split: dict) -> None:
    """The workloads comparison on mean latency, opening up TorchServe's bar."""
    FIGSIZE = (9.5, 3.0)
    LABEL_FS = 15
    TICK_FS = 12
    VALUE_FS = 10
    LEGEND_FS = 11.5
    TITLE_FS = 13
    BAR_WIDTH = 0.62
    HEADROOM = 1.30

    setup_style()
    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=FIGSIZE)

    for ax, workload in zip(axes, WORKLOADS):
        x = np.arange(len(ORDER))
        medians = [data[(workload, s)][0] for s in ORDER]
        lower = [data[(workload, s)][0] - data[(workload, s)][1] for s in ORDER]
        upper = [data[(workload, s)][2] - data[(workload, s)][0] for s in ORDER]
        inference, boundary = split[workload]

        for index, system in enumerate(ORDER):
            if system != TORCHSERVE:
                ax.bar(index, medians[index], width=BAR_WIDTH, color=COLOR[system],
                       edgecolor="black", linewidth=0.8)
                continue
            # Solid = the inference TorchServe performs, so its height reads straight across
            # against NAVI's bar; hatched = what reaching it over the boundary adds.
            ax.bar(index, inference, width=BAR_WIDTH, color=COLOR[system],
                   edgecolor="black", linewidth=0.8)
            ax.bar(index, boundary, width=BAR_WIDTH, bottom=inference, facecolor="white",
                   edgecolor=COLOR[system], linewidth=0.8, hatch="///")

        ax.errorbar(x, medians, yerr=[lower, upper], fmt="none", ecolor="black",
                    elinewidth=0.9, capsize=3)
        for index, value in enumerate(medians):
            ax.text(index, value + upper[index], f"{value:.0f}", ha="center", va="bottom",
                    fontsize=VALUE_FS)

        ax.set_title(f"{WORKLOAD_TITLE[workload]}\n{{\\small {WORKLOAD_RATE[workload]}}}",
                     fontsize=TITLE_FS)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylim(0, max(np.array(medians) + np.array(upper)) * HEADROOM)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean latency [ms]", fontsize=LABEL_FS)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[s], edgecolor="black", linewidth=0.8)
               for s in ORDER]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=COLOR[TORCHSERVE],
                                 linewidth=0.8, hatch="///"))
    labels = [LABEL[s] for s in ORDER] + ["serving boundary"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=LEGEND_FS, bbox_to_anchor=(0.5, -0.07))

    fig.tight_layout()
    save(fig, "e2e-systems-workloads-split.pdf", pad_inches=0.1)


if __name__ == "__main__":
    split = load_torchserve_split()
    p95 = load_stat("p95")
    flink_full, flink_steady = load_flink_startup()
    p95.update({(workload, FLINK): value for workload, value in flink_full.items()})
    plot_e2e_workloads(p95, inference_share(split), flink_steady)
    plot_e2e_split(load_stat("mean"), split)

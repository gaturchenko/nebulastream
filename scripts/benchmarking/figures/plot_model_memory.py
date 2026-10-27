"""What an inference operator's runtime sessions cost in memory on the Pi (revision §8).

Produces:
  model-memory.pdf             per model, the memory of the worker's 4 sessions with a private compiled model each and with one
                               shared compiled model: the smaller deployment solid, the difference hatched (freed by sharing)
                               or cross-hatched (added by sharing, MHAtt-RNN)
  model-memory-two-models.pdf  SqueezeDet and the Segmenter as two queries against both in one query
  model-memory-panels.pdf      both side by side, the left panel cut to SqueezeDet, the Segmenter and SARAD
  model-memory-table.tex       the numbers behind them, with the optimizer's assessment, as a booktabs table (MB, 4 sessions)

Every inference operator sets up one runtime session per worker thread when its pipeline starts and reuses it for every
invocation; the bars are those sessions' memory, measured as the query's resident set minus that of the same query without a
model (same source, buffer pool and preprocessing). The worker runs 4 threads, one per Pi core, and OpenVINO 1 stream x 4
threads; every model is invoked with single tuples. Medians across repetitions, whiskers min-max.

Two things the caption or text should carry. The measured memory includes what the optimizer's assessment does not see -- the
OpenVINO runtime the first session loads into the process and what sessions allocate once they run -- so it exceeds the
assessed totals in the table. And the footprint does not track the model file: what each session holds is the compiled model
(optimised graph, weight layouts, scratch memory), not the serialised weights.

Data: data/model_memory/model_memory.csv (process_model_memory.py over the Pi's run_model_memory.py output; raw runs stay on
      the Pi in ~/navi/results/model_memory).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from common import OUTPUT_DIR, load_csv, save, setup_style
from systems_common import COLOR, NES

# Physical memory of the Pi 4B as the kernel reports it (MemTotal), for the share of it a deployment takes.
PI_MEMORY_MB = 3795
SESSIONS = 4

MODEL = {
    "nbeats300": ("N-BEATS", "Forecasting"),
    "squeezedet": ("SqueezeDet", "Object detection"),
    "segmenter": ("Segmenter", "Segmentation"),
    "sarad": ("SARAD", "Anomaly detection"),
    "mhatt_rnn": ("MHAtt-RNN", "Keyword spotting"),
    "nanodet": ("NanoDet-R101", "Object detection"),
}
PAIR = "squeezedet+segmenter"
# The second model of the pair, stacked on the first: a lighter shade of the same hue, since hatching means "freed by sharing"
# in the per-model figure.
SECOND_COLOR = "#9cc2ee"
# What sharing adds where it costs more than private compiled models: neutral, so it does not read as another system or model.
ADDED_COLOR = "0.35"


def load():
    df = load_csv("model_memory", "model_memory.csv").set_index("cell")
    singles = df[df["models"].str.count(r"\+") == 0]
    singles = singles.loc[[cell for cell in MODEL if cell in singles.index]].sort_values("deployed_private_mb")
    return singles, (df.loc[PAIR] if PAIR in df.index else None)


LABEL_FS = 13
TICK_FS = 11
VALUE_FS = 9.5
LEGEND_FS = 10
# The models of the left panel in the two-panel figure: the two of the right panel's query and one large deployment.
PANEL_MODELS = ["squeezedet", "segmenter", "sarad"]


def style_axis(ax, top: float) -> None:
    ax.set_ylim(0, top)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def draw_models(ax, df, headroom: float = 1.3) -> None:
    """Each bar is as tall as the larger of the two deployments: the smaller one solid, and on top what sharing frees (hatched)
    or, where sharing one compiled model costs more than private ones (MHAtt-RNN on the Pi), what it adds (cross-hatched)."""
    BAR_WIDTH = 0.55

    x = np.arange(len(df))
    shared = df["deployed_shared_mb"].to_numpy()
    private = df["deployed_private_mb"].to_numpy()
    low, high = np.minimum(shared, private), np.maximum(shared, private)
    freed = private >= shared

    ax.bar(x, low, width=BAR_WIDTH, color=COLOR[NES], edgecolor="black", linewidth=0.8)
    ax.bar(x[freed], (high - low)[freed], width=BAR_WIDTH, bottom=low[freed], facecolor="white", edgecolor=COLOR[NES],
           linewidth=0.8, hatch="///")
    ax.bar(x[~freed], (high - low)[~freed], width=BAR_WIDTH, bottom=low[~freed], facecolor="white", edgecolor=ADDED_COLOR,
           linewidth=0.8, hatch="xxx")
    for mode, values in (("shared", shared), ("private", private)):
        below = values - df[f"deployed_{mode}_min_mb"].to_numpy()
        above = df[f"deployed_{mode}_max_mb"].to_numpy() - values
        ax.errorbar(x, values, yerr=[below, above], fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5)

    top = high.max()
    for index, (lo, hi) in enumerate(zip(low, high)):
        if lo > 0.08 * top:
            ax.text(index, lo / 2, f"{lo:.0f}", ha="center", va="center", fontsize=VALUE_FS, color="white")
        ax.text(index, hi + top * 0.035, f"{hi:.0f}\\,MB\n{{\\scriptsize {hi / PI_MEMORY_MB * 100:.0f}\\% of RAM}}",
                ha="center", va="bottom", fontsize=VALUE_FS, linespacing=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL[cell][0]}\n{{\\small {row.file_mb:.0f}\\,MB file}}" for cell, row in df.iterrows()],
                       fontsize=LABEL_FS - 2)
    style_axis(ax, top * headroom)
    if freed.all():
        handles = [Patch(facecolor=COLOR[NES], edgecolor="black", linewidth=0.8, label="shared compiled model"),
                   Patch(facecolor="white", edgecolor=COLOR[NES], linewidth=0.8, hatch="///",
                         label="freed by sharing (private: one per session)")]
    else:
        handles = [Patch(facecolor=COLOR[NES], edgecolor="black", linewidth=0.8, label="smaller deployment"),
                   Patch(facecolor="white", edgecolor=COLOR[NES], linewidth=0.8, hatch="///",
                         label="freed by sharing one compiled model"),
                   Patch(facecolor="white", edgecolor=ADDED_COLOR, linewidth=0.8, hatch="xxx",
                         label="added by sharing one compiled model")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=LEGEND_FS)


def draw_two_models(ax, df, pair) -> None:
    BAR_WIDTH = 0.36

    first, second = pair["models"].split("+")
    modes = ["private", "shared"]
    x = np.arange(len(modes))
    top = 0
    for index, mode in enumerate(modes):
        a, b = df.loc[first, f"deployed_{mode}_mb"], df.loc[second, f"deployed_{mode}_mb"]
        together = pair[f"deployed_{mode}_mb"]
        left, right = index - BAR_WIDTH / 2 - 0.02, index + BAR_WIDTH / 2 + 0.02
        ax.bar(left, a, width=BAR_WIDTH, color=COLOR[NES], edgecolor="black", linewidth=0.8)
        ax.bar(left, b, width=BAR_WIDTH, bottom=a, color=SECOND_COLOR, edgecolor="black", linewidth=0.8)
        ax.bar(right, together, width=BAR_WIDTH, color="0.55", edgecolor="black", linewidth=0.8)
        ax.errorbar(right, together, yerr=[[together - pair[f"deployed_{mode}_min_mb"]],
                                           [pair[f"deployed_{mode}_max_mb"] - together]],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5)
        ax.text(left, a + b, f"{a + b:.0f}", ha="center", va="bottom", fontsize=VALUE_FS)
        ax.text(right, pair[f"deployed_{mode}_max_mb"], f"{together:.0f}", ha="center", va="bottom", fontsize=VALUE_FS)
        top = max(top, a + b, pair[f"deployed_{mode}_max_mb"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{mode} compiled\nmodels" for mode in modes], fontsize=LABEL_FS - 2)
    style_axis(ax, top * 1.45)
    ax.legend(handles=[Patch(facecolor=COLOR[NES], edgecolor="black", linewidth=0.8, label=f"{MODEL[first][0]} query"),
                       Patch(facecolor=SECOND_COLOR, edgecolor="black", linewidth=0.8, label=f"{MODEL[second][0]} query"),
                       Patch(facecolor="0.55", edgecolor="black", linewidth=0.8, label="both models in one query")],
              loc="upper right", frameon=False, fontsize=LEGEND_FS, handlelength=1.4)


def plot_models(df) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(1.2 + 0.95 * len(df), 3.2))
    draw_models(ax, df)
    ax.set_ylabel(f"Memory of {SESSIONS} sessions [MB]", fontsize=LABEL_FS)
    fig.tight_layout()
    save(fig, "model-memory.pdf", pad_inches=0.1)


def plot_two_models(df, pair) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    draw_two_models(ax, df, pair)
    ax.set_ylabel(f"Memory of {SESSIONS} sessions [MB]", fontsize=LABEL_FS)
    fig.tight_layout()
    save(fig, "model-memory-two-models.pdf", pad_inches=0.1)


def plot_panels(df, pair) -> None:
    """SqueezeDet, the Segmenter and SARAD on the left, the two-model query on the right. The panels keep their own y scales, as SARAD's
    private deployment is four times the two-model query's."""
    FIGSIZE = (7.0, 3.2)

    setup_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.2, 1.25]})
    # With few bars the legend spans the large model's column, so it needs the room above that bar's label.
    draw_models(left, df.loc[[cell for cell in df.index if cell in PANEL_MODELS]], headroom=1.6)
    draw_two_models(right, df, pair)
    left.set_ylabel(f"Memory of {SESSIONS} sessions [MB]", fontsize=LABEL_FS)
    fig.tight_layout(w_pad=2.0)
    save(fig, "model-memory-panels.pdf", pad_inches=0.1)


def write_table(df, pair) -> None:
    rows = [(MODEL[cell][0], MODEL[cell][1], row) for cell, row in df.iterrows()]
    if pair is not None:
        names = " + ".join(MODEL[m][0] for m in pair["models"].split("+"))
        rows.append((names, "one query", pair))

    # The assessment as totals for the worker's sessions rather than per session: sharing can make the first session dearer
    # (MHAtt-RNN on the Pi compiles one model for 4 streams), which only the totals show, next to the measured ones.
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Assessed} & \multicolumn{3}{c}{Measured} \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-8}",
        r"Model & Task & File & Private & Shared & Private & Shared & Saved \\",
        r"\midrule",
    ]
    for index, (name, task, row) in enumerate(rows):
        if pair is not None and index == len(rows) - 1:
            lines.append(r"\midrule")
        saved = (1 - row["deployed_shared_mb"] / row["deployed_private_mb"]) * 100
        saved = f"$-${-saved:.0f}\\%" if saved < 0 else f"{saved:.0f}\\%"
        lines.append(
            f"{name} & {task} & {row['file_mb']:.0f} & {row['assessed_private_total_mb']:.0f} & "
            f"{row['assessed_shared_total_mb']:.0f} & {row['deployed_private_mb']:.0f} & {row['deployed_shared_mb']:.0f} & "
            f"{saved} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "model-memory-table.tex"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    models, two = load()
    plot_models(models)
    if two is not None:
        plot_two_models(models, two)
        plot_panels(models, two)
    write_table(models, two)

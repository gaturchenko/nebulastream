"""What an inference operator's runtime sessions cost in memory on the Pi (revision §8).

Produces model-memory-table.tex, the numbers of the paper's model-memory table: per model, the memory of the worker's 4
sessions with a private compiled model each and with one shared compiled model, measured and as the optimizer assesses it,
and SqueezeDet and the Segmenter together in one query, as a booktabs table (MiB, 4 sessions). The paper's table drops the
Task column.

Every inference operator sets up one runtime session per worker thread when its pipeline starts and reuses it for every
invocation; the measured columns are those sessions' memory, measured as the query's resident set minus that of the same query without a
model (same source, buffer pool and preprocessing). The worker runs 4 threads, one per Pi core, and OpenVINO 1 stream x 4
threads; every model is invoked with single tuples. Medians across repetitions.

Two things the caption or text should carry. The measured memory includes what the optimizer's assessment does not see -- the
OpenVINO runtime the first session loads into the process and what the query allocates while it runs -- so it exceeds the
assessed totals in the table, by 16-279 MiB (the assessment invokes each session once, so it does include what a backend
allocates only at a session's first invocation). And the footprint does not track the model file: what each session holds is
the compiled model (optimised graph, weight layouts, scratch memory), not the serialised weights.

Data: data/model_memory/model_memory.csv (process_model_memory.py over the Pi's run_model_memory.py output; raw runs stay on
      the Pi in ~/navi/results/model_memory).
"""

from common import OUTPUT_DIR, load_csv

MODEL = {
    "nbeats300": ("N-BEATS", "Forecasting"),
    "squeezedet": ("SqueezeDet", "Object detection"),
    "segmenter": ("Segmenter", "Segmentation"),
    "sarad": ("SARAD", "Anomaly detection"),
    "mhatt_rnn": ("MHAtt-RNN", "Keyword spotting"),
    "nanodet": ("NanoDet-R101", "Object detection"),
}
PAIR = "squeezedet+segmenter"


def load():
    df = load_csv("model_memory", "model_memory.csv").set_index("cell")
    singles = df[df["models"].str.count(r"\+") == 0]
    singles = singles.loc[[cell for cell in MODEL if cell in singles.index]].sort_values("deployed_private_mb")
    return singles, (df.loc[PAIR] if PAIR in df.index else None)


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
    write_table(*load())

# Paper figures

Each script reads its CSVs from `data/` and renders into `output/`.
The scripts produce exactly the figures and the table the paper uses, listed in the order they appear in it:

| Script | Output | Data |
|---|---|---|
| `plot_systems_workloads.py` | `e2e-systems-workloads-column.pdf` (Fig. 1) | `data/systems/workloads_p95.csv`, `data/systems/torchserve_breakdown.csv`, `data/systems/flink_startup_runs.csv` |
| `plot_batch_microbench.py` | `batch-microbench.pdf` | `data/batching/BatchInferenceMicrobenchmark2.csv` |
| `plot_threading.py` | `threading.pdf` (24-thread budget; one panel each for N-BEATS-1, N-BEATS-32, NanoDet-R101) | `data/threading/results_1.csv`, `data/threading/results_nanodet_r101.csv`, `data/threading/threading_lat.csv` |
| `plot_cache_microbench.py` | `caching-microbench.pdf` | `data/cache/cache_scope_keysize.csv` |
| `plot_batch_rate_throughput.py` (panels drawn by `plot_batch_rate.py` and `plot_batch_throughput.py`) | `batch-rate-throughput-column.pdf` | `data/batch_rate/batch_rate_runs.csv`, `data/batch_rate/batch_throughput_runs.csv`, `data/batch_rate/batch_rate_profile.csv` |
| `plot_post_join.py` | `post-join-choice.pdf` | `data/post_join/post_join_{runs,decisions}_{x86,pi}.csv` |
| `table_model_memory.py` | `model-memory-table.tex` (the model-memory table) | `data/model_memory/model_memory.csv` |
| `plot_ablation.py` | `e2e-ts.pdf`, `e2e-audio.pdf` | `data/ablation/*.csv` |
| `plot_systems_rate_sweep.py` | `systems-rate-sweep-throughput.pdf` | `data/systems/rate_sweep.csv` |

The sensitivity table (Table V) is not drawn here: its ranges are printed by `replay_selection.py` and `replay_knee.py` in
`e2e/sensitivity/`, from `data/sensitivity/` (see below).

## Running

Requires Python 3.12 with the packages in `requirements.txt`, and a LaTeX installation with
the `libertine` and `newtx` packages: all text is rendered with the paper's Libertine font via
`text.usetex` (see `common.setup_style`). On Debian/Ubuntu, `texlive-latex-extra` and
`texlive-fonts-extra` provide them.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python plot_batch_microbench.py     # or any other plot_*.py, or table_model_memory.py
```

The systems figures share `systems_common.py`: the per-system colours, the labels, and the
loading rule that NAVI's row is `systems.test_02` while `systems.test_03` in the same file belongs
to the TorchServe query.

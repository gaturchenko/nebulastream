# Paper figures

Self-contained plotting scripts distilled from the experiment notebooks. Each
script reads its CSVs from `data/`, renders into `output/`, and follows the
same shape: a load/prepare step, one function per chart with the tweakable
font- and figure-size constants at the top of its body, and a `__main__` block.

| Script | Figure(s) | Data |
|---|---|---|
| `plot_batch_microbench.py` | `batch-microbench.pdf` | `data/batching/BatchInferenceMicrobenchmark2.csv` |
| `plot_cache_microbench.py` | `caching-microbench.pdf` | `data/cache/cache_scope_keysize.csv` |
| `plot_threading.py` | `ts-threading.pdf`, `od-threading.pdf` | `data/threading/results_1.csv` |
| `plot_systems.py` | `e2e-systems.pdf` | `data/systems/*.csv` |
| `plot_ablation.py` | `e2e-audio.pdf`, `e2e-ts.pdf` | `data/ablation/*.csv` |
| `plot_musan_audio.py` | `musan-audio.pdf` | `data/musan/*_envelope.csv` |

## Running

Requires `pandas`, `seaborn`, `matplotlib`, and a LaTeX installation with the
`libertine` package (all text is rendered with the paper's Libertine font via
`text.usetex`; see `common.setup_style`).

```sh
python plot_batch_microbench.py     # or any other plot_*.py
```

## Data provenance

- `batching/` — `BatchInferenceMicrobenchmark` (x86 dev machine), batch sizes 1–4096
  over three N-BEATS model sizes.
- `cache/` — `PredictionCacheScopeMicrobenchmark` key-size sweep: Local (per-thread)
  vs. Global (shared, mutex-protected) prediction cache.
- `threading/` — SPE-workers x OpenVINO-threads budget sweep on the Pi
  (`run_threading.sh`); the query name encodes model and budget, the test index
  selects the split (see `BUDGET_SPLITS` in `plot_threading.py`).
- `systems/` — NAVI vs. Flink+OpenVINO vs. TorchServe on OSU-RGB (`run_systems.sh`).
- `ablation/` — inference-configuration sweeps on the Pi (`run_systests.py`) for
  MUSAN/MHAtt-RNN and CWRU/SARAD.
- `musan/` — per-bin min/max waveform envelopes of the three MUSAN US-gov speech
  recordings used as benchmark sources; regenerate from the raw WAVs with
  `python prepare_musan_waveforms.py /path/to/MUSAN/raw`.

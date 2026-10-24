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
| `plot_waymo_e2e.py` | `e2e-waymo.pdf` | `data/systems/workloads_p95.csv`, `data/systems/torchserve_breakdown.csv` |
| `plot_systems_workloads.py` | `e2e-systems-workloads.pdf`, `e2e-systems-workloads-split.pdf` | `data/systems/workloads_p95.csv`, `data/systems/torchserve_breakdown.csv` |
| `plot_systems_rate_sweep.py` | `systems-rate-sweep.pdf` | `data/systems/rate_sweep.csv` |
| `plot_torchserve_breakdown.py` | `torchserve-breakdown.pdf` | `data/systems/torchserve_breakdown.csv` |
| `plot_flink_memory.py` | `flink-memory.pdf` | `data/systems/flink_memory.csv` |
| `plot_ablation.py` | `e2e-audio.pdf`, `e2e-ts.pdf` | `data/ablation/*.csv` |
| `plot_musan_audio.py` | `musan-audio.pdf` | `data/musan/*_envelope.csv` |
| `plot_batch_rate.py` | `batch-rate.pdf` | `data/batch_rate/batch_rate_runs.csv` |
| `plot_batch_throughput.py` | `batch-throughput.pdf` | `data/batch_rate/batch_throughput_runs.csv`, `data/batch_rate/batch_rate_profile.csv` |
| `plot_batch_rate_throughput.py` | `batch-rate-throughput.pdf` (full width, both panels) | as the two above |
| `plot_post_join.py` | `post-join.pdf` | `data/post_join/post_join_{runs,decisions,pilot_summary}_x86.csv`, `data/post_join/post_join_{pilot_summary,decisions}_pi.csv` |

## Running

Requires `pandas`, `seaborn`, `matplotlib`, and a LaTeX installation with the
`libertine` package (all text is rendered with the paper's Libertine font via
`text.usetex`; see `common.setup_style`). On this machine the only interpreter
carrying all three is `~/PycharmProjects/nes-inference-datasets-models/.venv`.

```sh
python plot_batch_microbench.py     # or any other plot_*.py
```

The systems figures share `systems_common.py`: the per-system colours, the labels, and the
loading rule that NAVI's row is `systems.test_02` while `systems.test_03` in the same file belongs
to the TorchServe query. Those colours are the first three slots of the validated categorical
palette rather than matplotlib's `tab10`, whose orange and green are indistinguishable under
protanopia (OKLab Delta E 0.7) -- `e2e-systems.pdf` still carries that pairing.

## Data provenance

- `batching/` — `BatchInferenceMicrobenchmark` (x86 dev machine), batch sizes 1–4096
  over three N-BEATS model sizes.
- `cache/` — `PredictionCacheScopeMicrobenchmark` key-size sweep: Local (per-thread)
  vs. Global (shared, mutex-protected) prediction cache.
- `threading/` — SPE-workers x OpenVINO-threads budget sweep on the Pi
  (`run_threading.sh`); the query name encodes model and budget, the test index
  selects the split (see `BUDGET_SPLITS` in `plot_threading.py`).
- `systems/` — NAVI vs. Flink+OpenVINO vs. TorchServe. The three loose CSVs are the original
  single-run OSU-RGB comparison behind `e2e-systems.pdf`, kept as published.
  The revision's three figures read **aggregates**, not raw runs: `workloads_p95.csv` (the
  three-workload matrix at each workload's published operating point), `rate_sweep.csv` (the Waymo
  latency-vs-offered-load ladder, 7 rates) and `torchserve_breakdown.csv`
  (TorchServe's mean latency split across the stages its Prometheus counters expose). Each is a
  median over 3 repetitions, with min-max alongside for latency -- repetitions are not optional on
  these workloads, since single runs reversed the CWRU ranking and invented a Flink speed-up that
  three runs erase.

  The ~120 per-repetition CSVs behind those aggregates are **not** kept here. They live on the Pi
  under `~/navi/results-reps` and `~/navi/results-sweep`; `aggregate_systems_data.py` reduces them
  and is how these three files were produced. The sweep directory holds two Flink arms: the
  `_fastparse` runs, which are the ones aggregated, and the plain runs that predate replacing its
  per-line CSV parser with a direct column slice -- a defect that capped the baseline at
  14.5 frames/s while its inference lanes could serve ~58/s.

  ```sh
  python aggregate_systems_data.py --reps ~/navi/results-reps --sweep ~/navi/results-sweep \
      --memory ~/navi/results-memory
  ```

  `flink_memory.csv` is the private- vs. shared-compiled-model comparison
  (`run_flink_memory.sh`), the measurement behind the memory-sharing contribution: peak RSS of the
  Flink job JVM with each lane holding its own compiled model against all lanes sharing one.
  `--memory` is optional, so a run that only refreshes the latency data leaves it untouched.
  Repetition counts differ per cell and `workloads_p95.csv` carries `n_reps`: CWRU/Flink is 6
  because that cell is bimodal (inference P50 came out 114/112/238/231/244/223 ms), everything
  else is 3.
- `ablation/` — inference-configuration sweeps on the Pi (`run_systests.py`) for
  MUSAN/MHAtt-RNN and CWRU/SARAD.
- `musan/` — per-bin min/max waveform envelopes of the three MUSAN US-gov speech
  recordings used as benchmark sources; regenerate from the raw WAVs with
  `python prepare_musan_waveforms.py /path/to/MUSAN/raw`.
- `batch_rate/` — source-rate x batch-size micro-benchmark on the Pi (N-BEATS, window-fed batched
  inference; `e2e/batch_rate/`): 7 offered rates x 8 fixed batch sizes plus OPTIMIZER_CHOOSES,
  3 repetitions each. The CSVs are `process_batch_rate.py`'s aggregates: one row per run
  (`batch_rate_runs.csv`), the per-cell medians, the optimizer's per-candidate profile parsed
  from the INFO log, and the cells with missing repetitions (batch sizes 1 and 2 at 1000/s fall
  behind until the buffer pool runs out). The raw per-run results stay on the Pi under
  `~/navi/results/batch_rate`.

  `batch_throughput_*.csv` are the sustainable-throughput companion (`run_batch_throughput.sh`,
  `process_batch_throughput.py`): the same rate-limited query and worker settings, each fixed
  batch size run at a ~1.25x ladder of offered rates around its capacity, plus the same window
  without inference; 3 repetitions of 30 s. A run sustains its rate if it delivers >= 99% of it
  (the mock source backpressures, so latency alone would hide a shortfall). Every run waited for
  the SoC to cool below 70 C: without the wait the high-rate cells reached the Pi 4's 80 C soft
  limit, and the 12 runs that were throttled were re-measured. Raw results:
  `~/navi/results/batch_throughput` (throttled originals in `batch_throughput_throttled`).
- `post_join/` — post-join batching decision (`e2e/post_join/`, `e2e/post_join_pilot/`): k MUSAN
  streams joined on their window start, MHAtt-RNN on each payload, run as one batch-k invocation
  (BATCHED) or k single ones (SEQUENTIAL), forced or chosen by OPTIMIZER_CHOOSES. Every OpenVINO
  session runs one stream with four inference threads (`openvino_num_streams=1`,
  `openvino_inference_num_threads=4`), profiled and deployed alike.
  `post_join_pilot_summary_{x86,pi}.csv` are the optimizer's profile from `nes-model-profiler`
  (`summarize_post_join_pilot.py`) per model x OpenVINO session x static/dynamic batch, 3
  repetitions; `post_join_pilot_thermal_pi.csv` the Pi's temperature during it. The Pi batches at
  every k, so the decision changes only on x86 (so015, pinned to 4 cores):
  `post_join_{runs,summary,decisions}_x86.csv` are `process_post_join.py`'s aggregates (one row per
  run, per-cell medians, and the choice against both forced plans), 3 repetitions of 60 s. The
  plans are compared by the inference pipelines' task time per joined record from the event trace:
  end-to-end latency is dominated by closing the windows and joins and differs by ~1% between
  plans. Raw results stay on so015 under `~/navi-x86/results/post_join`. The Pi ran the same
  experiment at k = 2, 4 with deduplication, ADVANCE 1000/500/333 ms, 3 repetitions, 60 s
  (`post_join_{runs,summary,decisions,thermal}_pi.csv`): each run started below 60 °C, and the 20
  runs that throttled were moved to `~/navi/results/post_join_throttled` and re-measured from
  below 55 °C. k = 4 SEQUENTIAL at 333 ms throttles every time; it falls behind from 500 ms on
  regardless. Raw results: `~/navi/results/post_join`.

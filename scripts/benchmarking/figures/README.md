# Paper figures

Self-contained plotting scripts distilled from the experiment notebooks. Each
script reads its CSVs from `data/`, renders into `output/`, and follows the
same shape: a load/prepare step, one function per chart with the tweakable
font- and figure-size constants at the top of its body, and a `__main__` block.

The paper's evaluation figures are drawn at the size they are printed at (IEEEtran: a column is
3.5 in, the text width 7 in; `caching-microbench.pdf` is three quarters of a column), so their
font sizes are the sizes on the page, and no text goes below 7 pt (captions are 8 pt). Include
them at that width without `trim`; a figure drawn larger and scaled down shrinks its text with it,
which is how the earlier versions ended up at 3-5 pt.

| Script | Figure(s) | Data |
|---|---|---|
| `plot_batch_microbench.py` | `batch-microbench.pdf` | `data/batching/BatchInferenceMicrobenchmark2.csv` |
| `plot_cache_microbench.py` | `caching-microbench.pdf` | `data/cache/cache_scope_keysize.csv` |
| `plot_threading.py` | `ts-threading.pdf`, `od-threading.pdf` | `data/threading/results_1.csv`, `data/threading/results_nanodet_r101.csv`, `data/threading/threading_lat.csv` |
| `plot_systems.py` | `e2e-systems.pdf` | `data/systems/*.csv` |
| `plot_waymo_e2e.py` | `e2e-waymo.pdf` | `data/systems/workloads_p95.csv`, `data/systems/torchserve_breakdown.csv` |
| `plot_systems_workloads.py` | `e2e-systems-workloads.pdf`, `e2e-systems-workloads-column.pdf` (paper Fig. 1), `e2e-systems-workloads-split.pdf` | `data/systems/workloads_p95.csv`, `data/systems/torchserve_breakdown.csv`, `data/systems/flink_startup_runs.csv` |
| `plot_systems_rate_sweep.py` | `systems-rate-sweep.pdf`, `systems-rate-sweep-throughput.pdf`, `systems-rate-sweep-latency.pdf` | `data/systems/rate_sweep.csv` |
| `plot_torchserve_breakdown.py` | `torchserve-breakdown.pdf` | `data/systems/torchserve_breakdown.csv` |
| `plot_flink_memory.py` | `flink-memory.pdf` | `data/systems/flink_memory.csv` |
| `plot_ablation.py` | `e2e-audio.pdf`, `e2e-ts.pdf` | `data/ablation/*.csv` |
| `plot_musan_audio.py` | `musan-audio.pdf` | `data/musan/*_envelope.csv` |
| `plot_batch_rate.py` | `batch-rate.pdf` | `data/batch_rate/batch_rate_runs.csv` |
| `plot_batch_throughput.py` | `batch-throughput.pdf` | `data/batch_rate/batch_throughput_runs.csv`, `data/batch_rate/batch_rate_profile.csv` |
| `plot_batch_rate_throughput.py` | `batch-rate-throughput.pdf` (full width, both panels) | as the two above |
| `plot_post_join.py` | `post-join-choice.pdf`, `post-join-speedup.pdf` | `data/post_join/post_join_{runs,decisions,pilot_summary}_{x86,pi}.csv` |
| `plot_model_memory.py` | `model-memory.pdf`, `model-memory-two-models.pdf`, `model-memory-panels.pdf` (SqueezeDet, Segmenter and SARAD left, two-model query right), `model-memory-table.tex` | `data/model_memory/model_memory.csv` |

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

All Pi experiments below were rerun on 2026-10-24/25 with every OpenVINO session at one stream with four inference
threads (`openvino_num_streams=1`, `openvino_inference_num_threads=4`, baselines included) and a heatsink and fan on the
Pi: no run was throttled. `data/sigmod/` and `output/sigmod/` keep the data and figures as they stood before those
reruns (OpenVINO at `num_streams=0`, no cooling), in the same layout as `data/` and `output/`; no script reads them.

- `batching/` — `BatchInferenceMicrobenchmark` (x86 dev machine), batch sizes 1–4096
  over three N-BEATS model sizes.
- `cache/` — `PredictionCacheScopeMicrobenchmark` key-size sweep: Local (per-thread)
  vs. Global (shared, mutex-protected) prediction cache.
- `threading/` — SPE-workers x OpenVINO-threads budget sweep on so015 (AMD EPYC 7443P, 24 cores,
  48 hardware threads; `run_threading.sh`, 10 repetitions); the query name encodes model and
  budget, the test index selects the split (see `BUDGET_SPLITS` in `plot_threading.py`). Every
  worker-thread session runs one OpenVINO stream with the split's `inference_num_threads`. The
  earlier sweep in `sigmod/threading/` passed `num_streams=0`, under which OpenVINO ignores the
  thread count, so its OpenVINO-thread axis had no effect (equal worker counts gave equal
  throughput at any thread count). Raw results: so015 `~/navi-x86/results/threading`.
  `od-threading.pdf` reads `results_nanodet_r101.csv`: the same sweep with NanoDet-Plus on a
  ResNet-101 backbone (`nanodet-r101-{24,48}.test`, 16.2 instead of 0.45 GMACs at 320x320, random
  weights), heavy enough that inference bounds the query. With the original NanoDet (its rows in
  `results_1.csv`), reading and parsing the frames bounds it at ~120 frames/s in every split.
  Raw results: so015 `~/navi-x86/results/threading_r101`.
  The figures plot each split's capacity (these sweeps: inference results per second of the whole
  query, mean of 10) against its latency at one fixed load below every split's capacity:
  `threading_lat.csv`, from `run_threading_lat.sh` (the `*-lat.test` variants, 5 repetitions) and
  `process_threading_lat.py`, one row per run. The TCP mock source replays the data at N-BEATS-1
  6667, N-BEATS-32 333 forecasts/s and NanoDet-R101 20 frames/s, prefixing each tuple with its send
  time, so a latency runs from sending the input to the result reaching the sink, including any
  queueing for a worker; each run's first 10 s are left out as warm-up. The source starts sending
  only after the query has compiled its per-worker OpenVINO sessions (up to ~40 s at 48 workers).
  Raw results: so015 `~/navi-x86/results/threading_lat`.
- `systems/` — NAVI vs. Flink+OpenVINO vs. TorchServe. The three loose CSVs are the original
  single-run OSU-RGB comparison behind `e2e-systems.pdf`, kept as published.
  The revision's three figures read **aggregates**, not raw runs: `workloads_p95.csv` (the
  three-workload matrix at each workload's published operating point), `rate_sweep.csv` (the Waymo
  latency-vs-offered-load ladder, 7 rates) and `torchserve_breakdown.csv`
  (TorchServe's mean latency split across the stages its Prometheus counters expose). Each is a
  median over repetitions, with min-max alongside for latency -- repetitions are not optional on
  these workloads, since single runs reversed the CWRU ranking and invented a Flink speed-up that
  three runs erase.

  All three systems run one OpenVINO stream with four threads per lane (NAVI worker thread, Flink
  inference lane, TorchServe worker), each with the number of lanes its tuning pilot chose (see
  below). Left alone, `run_systems.sh` gives Flink as many inference lanes as the `.test` has
  worker threads; the tuning and sweep drivers override that with `FLINK_LANES`.

  The per-repetition CSVs behind those aggregates are **not** kept here. They live on the Pi
  under `~/navi/results-reps` and `~/navi/results-sweep` (`run_pi_systems_rerun.sh`; the
  `num_streams=0` runs are archived in `~/navi/systems-streams0`), with every arm's thermal
  state in `~/navi/results-thermal/arms.csv`; `aggregate_systems_data.py` reduces them and is
  how these three files were produced:

  ```sh
  python aggregate_systems_data.py --reps ~/navi/results-reps --sweep ~/navi/results-sweep \
      --tuned cwru=~/navi/results-cwru-tuning waymo=~/navi/results-tuning-waymo \
              musan=~/navi/results-tuning-musan
  ```

  `workloads_p95.csv` and `torchserve_breakdown.csv` come from the **tuning runs** (`--tuned`):
  every system's parallelism knob (NAVI worker threads 1-4, Flink inference lanes 1-4 (CWRU 1-3),
  TorchServe workers 1-3) is chosen on the same rule from a 2-repetition pilot, the lowest median
  P95 among settings that keep up with the offered rate, and the chosen configurations are then
  measured afresh, 5 repetitions each (`run_pi_cwru_tuning.sh`, `run_pi_systems_tuning.sh`; raw
  runs on the Pi in `~/navi/results-cwru-tuning`, `~/navi/results-tuning-{waymo,musan}`). The
  choice is in the matrix's `config` column. The tuning runs are longer than the published operating
  point, so that a P95 rests on ~600 results: CWRU reads X228 looped to 61 s, MUSAN a 200 s cut of
  the recording (the published point used 60 s), Waymo its 991 frames as before. Without `--tuned`
  the script reproduces the published-operating-point matrix from `results-reps`. `rate_sweep.csv`
  follows the tuning too: its Flink arm was rerun at the 1 lane the Waymo pilot chose
  (`run_pi_flink_sweep_lanes1.sh`; the earlier curve with the lanes the `.test` implies is in
  `~/navi/results-sweep-flink-lanes-from-test`), its TorchServe arm at the 3 workers (and 3 HTTP
  sink connections) the pilot chose (`run_pi_ts_sweep_workers3.sh`; the 1-worker curve is in
  `~/navi/results-sweep-torchserve-1worker`, and saturated slightly later, ~33 vs. ~31.5 frames/s),
  and NAVI's arm already ran at its tuned 2 workers.

  Flink's P95 bars in `e2e-systems-workloads.pdf` come from `flink_startup_runs.csv` instead: 5
  further runs of Flink's arm at the tuned lanes with a per-record trace (`WindowTrace` in the Flink
  baseline, on only when `FLINK_WINDOW_TRACE` is set; `run_pi_flink_startup.sh`, raw traces in
  `~/navi/results-flink-startup`), one row per run with the full-run percentiles and those over the
  results whose input arrived after the first 5 s. The bar is solid up to the latter and hatched up
  to the former. On CWRU Flink's two lanes spend the first ~3 s overlapping at ~200 ms per inference,
  exactly the offered rate, so the start-up backlog cannot drain; those ~30 windows are the 5% the
  P95 sits on (full run 225-338 ms, after start-up 90-93 ms). Waymo and MUSAN show no such phase.
  The caption should say that only Flink's start-up is broken out; NAVI and TorchServe are full-run.

  In `e2e-systems-workloads.pdf`, TorchServe's P95 bar is split into inference and serving
  boundary in the proportions of its **mean** breakdown (`torchserve_breakdown.csv`), since stage
  times only add up on average and two stages exist only as means. The caption has to say so,
  e.g. *"TorchServe's bar is divided in the proportions of its mean latency breakdown (handler
  call vs. serialization, transport and queueing); the split is indicative, since stage shares at
  the tail need not match those on average."*

  `flink_memory.csv` is the private- vs. shared-compiled-model comparison
  (`run_flink_memory.sh`): peak RSS of the Flink job JVM with each lane holding its own compiled
  model against all lanes sharing one. It was **not** rerun and still describes the
  `num_streams=0` setting. `--memory` is optional, so a run that only refreshes the latency data
  leaves it untouched. `workloads_p95.csv` carries `n_reps`, 5 in every cell of the tuned matrix;
  `rate_sweep.csv` is 3 repetitions per rate.
- `ablation/` — inference-configuration sweeps on the Pi for MUSAN/MHAtt-RNN (UDF, ST, Batch) and
  CWRU/SARAD (UDF, ST at batch 1/12, ST+Cache local/global, Batch+Cache, Batch+Dedup,
  Batch+Cache+Dedup), 3 repetitions per configuration (`run_pi_ablation_rerun.sh`,
  `process_results_lat.py`, UDF cells rerun with `run_pi_ablation_udf_rerun.sh`). The UDF
  baselines stand for a stream processor without an inference operator: they call OpenVINO
  directly with its default configuration, the thread count capped at the hardware threads (the
  Pi's default of 8 for 4 cores leaks memory under concurrent inference), and use the same model files;
  CWRU reads `X228_FE_time.csv`, the 48 kHz recording the published run also ingested. The
  earlier files in `sigmod/ablation/` hold one repetition per configuration, and their MUSAN UDF
  pointed at a different MHAtt-RNN export. Raw results: `~/navi/results-ablation`.
- `musan/` — per-bin min/max waveform envelopes of the three MUSAN US-gov speech
  recordings used as benchmark sources; regenerate from the raw WAVs with
  `python prepare_musan_waveforms.py /path/to/MUSAN/raw`.
- `batch_rate/` — source-rate x batch-size micro-benchmark on the Pi (N-BEATS, window-fed batched
  inference; `e2e/batch_rate/`): 7 offered rates x 8 fixed batch sizes plus OPTIMIZER_CHOOSES,
  3 repetitions each. The CSVs are `process_batch_rate.py`'s aggregates: one row per run
  (`batch_rate_runs.csv`), the per-cell medians, the optimizer's per-candidate profile parsed
  from the INFO log, and the cells with missing repetitions (batch size 1 at 1000/s falls
  behind until the buffer pool runs out). `batch_rate_regret.csv` is
  `e2e/batch_rate/process_batch_rate_regret.py` over these files. The raw per-run results stay on the Pi under
  `~/navi/results/batch_rate`.

  `batch_throughput_*.csv` are the sustainable-throughput companion (`run_batch_throughput.sh`,
  `process_batch_throughput.py`): the same rate-limited query and worker settings, each fixed
  batch size run at a ~1.25x ladder of offered rates around its capacity, plus the same window
  without inference; 3 repetitions of 30 s. A run sustains its rate if it delivers >= 99% of it
  (the mock source backpressures, so latency alone would hide a shortfall). The ladders of batch
  sizes 2, 4 and 8 reach one rate further than in `sigmod/`, because their capacity rose with the
  faster session; every capacity lies inside its ladder. Raw results:
  `~/navi/results/batch_throughput`.
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
  (`post_join_{runs,summary,decisions,thermal}_pi.csv`) with a heatsink and fan on the Pi: the SoC
  peaked at 38.4 °C (40.4 °C during the pilot) and no run was throttled. k = 4 SEQUENTIAL falls
  behind from 500 ms on. Raw results: `~/navi/results/post_join`; the uncooled runs of 2026-10-23,
  20 of which throttled, are archived in `~/navi/results/uncooled_1x4`.
- `model_memory/` — the memory of an inference operator's runtime sessions on the Pi (revision §8;
  `e2e/model_memory/`): 4 worker threads, one session each, OpenVINO 1 stream x 4 threads,
  single-tuple inference, every session with a private compiled model or all sharing one.
  `model_memory.csv` is `process_model_memory.py`'s aggregate, medians of 3 repetitions: the model
  file, the optimizer's own assessment (`inference_memory_strategy=OPTIMIZER_CHOOSES`, per session)
  and the measured memory, i.e. the query's steady resident set minus that of the same query without
  a model. The measurement exceeds the assessment because it also holds the OpenVINO runtime the
  first session loads and what sessions allocate once they run; the two-model query (SqueezeDet and
  the Segmenter on one Waymo frame) costs ~65 MB less than the two single-model queries, the part
  paid once per process. On the Pi, sharing one compiled model costs MHAtt-RNN *more* than private
  ones (1425 vs. 924 MB; the assessment predicts it, 1407 vs. 897, and OPTIMIZER_CHOOSES deploys
  private copies), which `model-memory.pdf` draws cross-hatched; NanoDet-R101's private deployment
  takes 83% of the Pi's memory. The queries use small buffer pools (32 frame buffers, 256 x 128 KB
  otherwise), since a buffer's pages enter the resident set the first time the pool hands it out and
  a large pool's growth would swamp the difference; Waymo runs at 5 frames/s, the rate at which the
  Pi keeps up with both models in one query. Raw results: `~/navi/results/model_memory` (the failed
  10 frames/s Waymo runs in `waymo-10fps/`).

# Knob characterization for the adaptive inference optimizer

Goal: measure how each configuration knob of the inference operator responds to the two things the
optimizer cannot control — the **ingestion rate** and the **fraction of duplicate records** — so the
optimizer can be built on measured response surfaces instead of guesses.

| Knob | Where it lives | Values in this experiment |
|---|---|---|
| batch size | `optimizer.inference_batch_size` (selects `BatchInferModelPhysicalOperator`) | 1, 8, 64, 512 |
| prediction cache | `...inference.prediction_cache_type` / `..._entries` | off, LRU / 1024 entries |
| batch deduplication | `...inference.use_batch_deduplication` (requires batch > 1) | off, on |
| model precision | which `CREATE MODEL` the query references | fp32, INT8 (NNCF PTQ) |

Fixed factors: OpenVINO backend, `openvino_inference_num_threads=1`, `openvino_share_compiled_model=false`,
1 MiB operator buffers, 10 ms generator flush interval, `THREADS` worker threads (default 8),
`prediction_cache_scope=THREAD_LOCAL` (the global scope is rejected for batched inference).

## Why the workload is generated, not replayed

The environment factors are properties of the source, and `run_systests.py` can only sweep
worker/optimizer parameters — so rate and duplicates are baked into the generated systest, one
Generator source per operating point, and one query section per (operating point, precision). The
knobs stay in the YAML sweeps under `config/`, which is what the runner's combination labels record.

Duplicates come from the Generator's deterministic cache fields, in **two structures**, because
caching and deduplication do not react to the same kind of repetition:

* `CACHE_HOTSET` — a fixed hotset of 64 keys revisited across the whole stream, cold keys never
  repeat. Duplicates are spread out in time: a cache captures them, a single batch usually does not.
* `CACHE_GROUPED` — every key is one contiguous run (a miss plus its repetitions, mean run length
  `1/(1-d)`). Duplicates are adjacent: within-batch deduplication captures them even at small batch.

Both are exact, not stochastic, so the expected hit/dedup rates are known in advance and the
measured ones can be checked against them (`observed_hit_rate`, `expected_unique_fraction`).

## Hypotheses to test — the assumptions the optimizer can start from

These follow from the operator implementations; the experiment exists to confirm or refute them.

**H1 — batch size is inert below a rate threshold, and capped above it.**
`BatchingPhysicalOperator::close()` emits *every* created batch at each tuple-buffer boundary,
including partial ones (`getCreatedBatches(false)`): there is no timeout and no accumulation across
buffers. So the configured batch size is only an upper bound,

    effective batch = min(configured B, tuples per buffer, rate x flush interval)

At 10 ms flush interval, B=64 needs ≥ 6.4k tuples/s to be reached at all and B=512 needs ≥ 51.2k/s;
above that, the 1 MiB buffer caps the effective batch at roughly 1.5–2k tuples. Two consequences
the optimizer can exploit: raising B at low rate costs *nothing* (no latency penalty, unlike
timeout-based batching), and lowering B at high rate is the only way the knob has any effect.
`effective_batch_size` in the output CSV measures this directly, per run.

**H2 — the cache is a function of duplicates, nearly flat in rate.**
Gain ≈ `hit_rate x inference_cost - lookup_cost_per_record`, and the lookup is paid on every
record. So the response should rise monotonically with the duplicate percentage, with a break-even
duplicate ratio `d*` below which the cache is a pure loss, and should barely move with rate while
the system is below saturation. Above saturation the cache also raises the throughput ceiling, so it
turns an unsustainable rate into a sustainable one — a much larger effect than the latency gain.
With `THREAD_LOCAL` scope, each worker thread warms its own copy; the hotset workload therefore pays
`hotset x threads` warm-up misses, which matters for short runs and for large hotsets.

**H3 — deduplication depends on batch size and on duplicate *structure*, not just on the ratio.**
Duplicates only collapse if they land in the same batch. For grouped runs, a batch of B records
contains ≈ `B(1-d)+1` distinct keys, so the saving tracks `d` even at small B. For a hotset of W
keys, the batch touches `min(dB, W)` distinct hot keys, so with W=64 there is *no* saving at B=8 and
nearly the full saving at B=512. Prediction: deduplication and batch size are complements (their
joint effect exceeds the sum), while cache and deduplication are substitutes (sub-additive, and the
cache should dominate for hotset-shaped duplicates).

**H4 — INT8 is a multiplicative shift whose size grows with the batch, and it moves the other
knobs' break-even points.** This one is already partly measured, before any sweep. On this machine,
single-threaded OpenVINO per-record latency of the FP32 model and of the accuracy-bounded INT8
model (see "Model precision" below) is:

| batch | FP32 | INT8 | speed-up |
|---|---|---|---|
| 1 | 211.9 us | 129.1 us | 1.64x |
| 8 | 46.8 us | 23.6 us | 1.98x |
| 64 | 25.6 us | 10.1 us | 2.54x |
| 512 | 24.0 us | 8.8 us | 2.72x |

So precision and batch size are complements, not independent knobs: the INT8 win nearly doubles
between batch 1 and batch 512, because both knobs attack the same per-call overhead. Note also the
FP32 column on its own - batching buys 8.8x per record between batch 1 and 512, which is the
amortization H1 says is only reachable once the rate is high enough to fill a batch. The
second-order prediction still to be tested is that a cheaper model makes the cache lookup and the
deduplication hash map relatively *more* expensive, so the duplicate ratio at which they pay rises
with INT8.

**H5 — the decision variable is utilization, not rate.**
Below saturation every configuration has similar latency (dominated by the flush interval), so the
optimizer should pick the cheapest and most accurate option — fp32, no cache, no deduplication.
Above saturation, latency diverges and only throughput matters. If this holds, the optimizer's state
is `ρ = rate / capacity(config)` rather than the rate itself, and what it needs from this experiment
is exactly `capacity(config, duplicates)` — which Phase 0 and the `sustained_ratio` column measure.

## Running it

Prerequisites: a release build with the OpenVINO backend, `ovc` 2025.3 on `PATH`
(`~/PycharmProjects/nes-inference-datasets-models/.venv/bin`), and
`mkdir -p cmake-build-release/nes-systests/systest/results` (the LatencySink does not create its
own directory).

```bash
# 1. INT8 model (once). Needs an interpreter with nncf/onnx/onnxruntime/openvino.
#    Takes ~6 min: it runs a per-operation sensitivity analysis (see "Model precision" below).
venv/bin/python scripts/benchmarking/e2e/adaptive/quantize_nbeats.py \
    --model cmake-build-release/nes-systests/testdata/model/solar_power/pretrained/nbeats/nbeats_size_300_stride_30.onnx \
    --series cmake-build-release/nes-systests/testdata/large/solar_power/solar-power.csv \
    --benchmark
# Writes <model>_int8.onnx next to the FP32 model, which is where the generated .test looks for it.

# 2. Generate the systests. --model-path must match the testdata layout of YOUR checkout:
#    some trees have model/power/..., others model/solar_power/... The INT8 path is derived
#    from it automatically, and --sources-per-point is explained under "Offered rate" below.
python3 scripts/benchmarking/e2e/adaptive/gen_adaptive_workload.py \
    --model-path model/solar_power/pretrained/nbeats/nbeats_size_300_stride_30.onnx \
    --sources-per-point 4

# 3. Phase 0 — calibration at unbounded offered rate (~25 runs, ~20 min).
PHASES=0 THREADS=<cores> scripts/benchmarking/e2e/adaptive/run_adaptive_screening.sh
python3 scripts/benchmarking/e2e/adaptive/process_adaptive_results.py

# 4. Re-generate with a rate grid placed around the measured saturation point, e.g.
python3 scripts/benchmarking/e2e/adaptive/gen_adaptive_workload.py \
    --model-path model/solar_power/pretrained/nbeats/nbeats_size_300_stride_30.onnx \
    --sources-per-point 4 --rates 10000 40000 90000 150000

# 5. Phases 1 and 2 (~500 runs at 2 repetitions; budget 6-9 h).
PHASES="1 2" REPS=2 THREADS=<cores> scripts/benchmarking/e2e/adaptive/run_adaptive_screening.sh
python3 scripts/benchmarking/e2e/adaptive/process_adaptive_results.py --sample-latency 2000
```

Everything is resumable: `run_systests.py` skips any (combination, query, repetition) that already
has a result directory, so an interrupted cluster job continues where it stopped, and the cells
Phase 1 and Phase 2 share are measured once.

Design of the phases:

* **Phase 0** — unbounded rate, so each run reports its configuration's ceiling. Section `:01`
  has no inference operator at all: it measures the Generator's own ceiling, which is an upper
  bound on every offered rate. A fixed-rate Generator emits at most one buffer per flush interval,
  so no single source can offer more than `tuples per buffer / flush interval` (~150–200k/s at
  1 MiB and 10 ms). Lower `--flush-interval-ms`, or split the load over several physical sources,
  to go higher — but note that a shorter flush interval also lowers the reachable batch size.
* **Phase 1** — one knob family at a time over the full 24-point (rate x duplicates x structure)
  plane: baseline, batch ∈ {8,64,512}, cache, batch 64 + dedup, and the baseline again on the INT8
  sections. Gives the main-effect response surfaces (168 cells).
* **Phase 2** — full factorial of the knobs at the four corner operating points (lowest/highest
  rate x lowest/highest duplicate percentage) in both precisions, for the interactions H3/H4
  predict (80 cells).

## Offered rate: what one Generator source can actually deliver

A fixed-rate Generator emits at most one buffer per flush interval, so a single physical source
cannot offer more than `tuples per buffer / flush interval` - about 175k tuples/s at a 1 MiB buffer
and 10 ms. That is *below* the capacity of a batched configuration on a many-core node. From the
measured per-record costs above, capacity is roughly:

| threads | batch 1 (FP32) | batch 64 (FP32) | batch 64 (INT8) |
|---|---|---|---|
| 2 | 9.4k/s | 78k/s | 199k/s |
| 4 | 18.9k/s | 156k/s | 397k/s |
| 8 | 37.7k/s | 312k/s | 794k/s |

With one source and 8 threads, every batched configuration would keep up at every rate in the grid
and the rate axis would collapse - saturation, the regime the optimizer exists for, would never be
observed. Two ways out, both supported:

* `--sources-per-point 4` (recommended): four physical sources per operating point, each offering
  a quarter of the rate, with disjoint key ranges so the duplicate percentage is unchanged. The
  working set becomes N x hotset, which is recorded in the manifest and stays well inside the 1024
  entry cache.
* `THREADS=2` or `THREADS=4`: scale the node down instead, so a single source can saturate it.

Phase 0 measures the real ceiling rather than trusting this arithmetic: its section `:01` runs the
source with no inference operator at all.

## Reading the results

`adaptive_results.csv` has one row per (configuration, operating point, repetition). Start with:

* `sustained_ratio` < ~0.95 → the configuration did not keep up; that is the saturation boundary and
  the primary thing the optimizer must avoid.
* `effective_batch_size` vs the configured `batch_size` → H1; where they diverge, the batch knob is
  inert.
* `observed_hit_rate` vs `duplicate_percent` → H2, and whether thread-local warm-up erodes the rate.
* `expected_unique_fraction` (analytic, exact for these workloads) vs the measured speed-up of the
  dedup arm over the plain batch arm at the same batch size → H3, and how much of the theoretical
  saving the deduplication machinery gives back.
* `per_record_task_us` fp32 vs int8 at equal batch size → H4.

## Model precision: what `quantize_nbeats.py` does and why

Three things about this model made naive PTQ useless, all handled by the script:

1. **It ends in a no-op `Cast` that writes the graph output.** NNCF's own `eliminate_nop_cast` pass
   looks every redundant Cast up in `graph.value_info`, where a graph output never appears, and
   dies with `KeyError: 'output'`. The script folds those casts away first.
2. **The calibration series must be scaled the way training scaled it.** The solar scaler is
   `y = (x - 0) * 0.00858369`, i.e. the model was trained on `[0, 1]`. The scaler ONNX is not
   shipped in the testdata tree, so the script applies the same affine map by default (and finds
   the ONNX automatically when it is present). Calibrating on a differently-scaled series - a
   z-score, say - silently produces a badly calibrated model. Flat night-time windows are dropped
   for the same reason (measured: MAE 0.215 with them, 0.119 without).
3. **The residual `Sub`/`Add` ops must stay FP32.** Quantizing them costs ~0.11 MAE *on its own* -
   the same error whether one Gemm or all thirty are also quantized - because they carry nbeats'
   backcast residual between stacks, whose range no single INT8 scale covers. They buy no speed
   either (they are elementwise, not GEMMs): 1.67/2.07/2.86/3.01x with them excluded versus
   1.69/2.06/2.77/2.96x with them included.

With those fixed, quantizing all 30 Gemm/MatMul operations still costs 21.6% of the FP32 output's
standard deviation, and NNCF's usual remedy - full `BiasCorrection` - crashes in its ONNX backend
on this graph (`Duplicate definition-site`). So the script ranks operations by their individual
quantization error and binary-searches for the largest subset meeting `--max-mae` (default 5% of
the FP32 output std), measuring every candidate through OpenVINO - the runtime NES uses - rather
than onnxruntime. The resulting curve on this machine:

| ops in INT8 | deviation from FP32 | speed-up at batch 64 |
|---|---|---|
| 15 / 30 | 1.8% of std | 1.39x |
| **23 / 30 (selected at the 5% budget)** | **4.8% of std** | **2.54x** |
| 27 / 30 | 8.1% of std | 2.67x |
| 30 / 30 | 21.6% of std | ~2.8x |

Pick the operating point deliberately: `--max-mae` moves it, and `--allow-accuracy-loss` takes full
INT8. `quantization_report.json` records the whole curve, so the paper can state the quality price
of the precision knob rather than assert it is free. If a near-lossless INT8 model is wanted
instead, the route is NNCF's OpenVINO backend (it supports full bias correction and accuracy-aware
quantization) - but that emits OpenVINO IR, and `OpenVinoImporter.cpp` accepts only .onnx/.pb/
.tflite/.pdmodel/.pt2/SavedModel, so it would need the importer extended to read .xml/.bin.

## Measurement caveats

* `ingestion_time` comes from `CURRENT_TIME()` inside the inference subquery, so the latency is
  measured from the first operator, not from tuple creation: queueing *upstream* of that operator is
  not included. Throughput (`sink_throughput`) is unaffected.
* Model import runs `ovc` per query run; that cost lands in `query_duration_us`, not in the latency
  distribution, which additionally drops the first `--warmup-fraction` of records.
* Deduplication has no counterpart to the cache's `hits=/misses=` log line, so its effectiveness is
  inferred from `expected_unique_fraction` plus the measured throughput delta. If the paper needs
  the measured unique count, add one `NES_INFO` in `BatchInferModelPhysicalOperator` alongside the
  existing cache-stats line and re-run the dedup arm.
* Cache hit rates are per operator instance across all worker threads; with `THREAD_LOCAL` scope
  they depend on `THREADS`, so results from differently sized nodes must not be pooled — the thread
  count is part of every combination label for exactly this reason.

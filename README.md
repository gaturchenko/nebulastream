# NAVI: Natively Integrated ML Inference Operators in a Stream Processing Engine

This repository is the artifact for the paper. It holds NAVI's implementation inside the NebulaStream
stream processing engine, the scripts that ran the experiments, the processed results, and the plotting scripts for the figures.

## Figures

Everything needed to regenerate the figures is in [`scripts/benchmarking/figures`](scripts/benchmarking/figures):

- `output/` — the rendered figures (PDF) as they appear in the paper.
- `data/` — the processed results (CSV) each figure is drawn from, one subdirectory per experiment.
- `plot_*.py` — one plotting script per figure or group of figures.

## Experiment scripts

The drivers and query definitions behind the data are in
[`scripts/benchmarking/e2e`](scripts/benchmarking/e2e), one subdirectory per experiment:

| Directory | Experiment |
|---|---|
| `batch_rate/` | source rate x batch size, and sustainable throughput per batch size |
| `post_join/`, `post_join_pilot/` | batching inference after a join: one batched vs. k single invocations |
| `model_memory/` | memory of an inference operator's sessions, private vs. shared compiled model |
| `sensitivity/` | sensitivity of the optimizer's implementation selection to its parameters |
| `queries/` | the three workloads (CWRU, MUSAN, Waymo) of the comparison with other systems |
| `baselines/` | the Flink + OpenVINO and TorchServe baselines |
| `config/` | worker and optimizer configurations shared by the experiments |
| top level | thread budget split (`run_threading*.sh`), prediction cache scope (`run_global_vs_local_caches.sh`), configuration ablation (`run_*_time_series.sh`, `config/run_ablation.sh`) |

The queries of the threading, ablation, and cache experiments are system tests under
[`nes-systests/inference`](nes-systests/inference).

## Implementation

| Component | Location |
|---|---|
| Model catalog, OpenVINO and IREE backends | [`nes-inference`](nes-inference) |
| Logical inference operator | [`nes-logical-operators`](nes-logical-operators) (`InferModelLogicalOperator`) |
| Physical inference operator, prediction caches | [`nes-physical-operators/src/Inference`](nes-physical-operators/src/Inference) |
| Optimizer rules and implementation selection | [`nes-query-optimizer`](nes-query-optimizer) (`InferenceImplementationSelector`, `Rules/Semantic/*Infer*`) |
| Inference profiling and lowering | [`nes-query-compiler`](nes-query-compiler) (`InferenceProfiling`, `LowerToPhysicalInferModel`) |

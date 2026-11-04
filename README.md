# NAVI: Natively Integrated ML Inference Operators in a Stream Processing Engine

This repository is the artifact for the paper. It holds NAVI's implementation inside the NebulaStream
stream processing engine, the scripts that ran the experiments, the processed results, and the plotting scripts for the figures.

## Figures

Everything needed to regenerate the figures is in [`scripts/benchmarking/figures`](scripts/benchmarking/figures):

- `output/` — the rendered figures (PDF) as they appear in the paper.
- `data/` — the processed results (CSV) each figure is drawn from, one subdirectory per experiment.
- `plot_*.py` — one plotting script per figure, and `table_model_memory.py` for the model-memory table.

## Experiment scripts

[`scripts/README.md`](scripts/README.md) lists, for every figure and table, the script that runs the experiment, the
hardware it ran on, and how long it takes.

The drivers and query definitions behind the data are in
[`scripts/benchmarking/e2e`](scripts/benchmarking/e2e), one subdirectory per experiment:

| Directory | Experiment |
|---|---|
| `batch_rate/` | source rate x batch size, and sustainable throughput per batch size |
| `post_join/`, `post_join_pilot/` | batching inference after a join: one batched vs. k single invocations |
| `model_memory/` | memory of an inference operator's sessions, private vs. shared compiled model |
| `sensitivity/` | sensitivity of the optimizer's implementation selection to its parameters |
| `queries/` | the three workloads (CWRU, MUSAN, Waymo) of the comparison with other systems |
| `baselines/` | the Flink + OpenVINO and TorchServe baselines (`flink-openvino-baseline/` holds the full source of the Flink job and its OpenVINO JNI bridge) |
| `navi/` | the deployment on the Raspberry Pi: the `run_pi_*.sh` drivers of the Pi experiments, their queries, the processing scripts, and the built Flink baseline as it ran there |
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

## Building

Regenerating the figures needs no build, only Python (see
[`scripts/benchmarking/figures/README.md`](scripts/benchmarking/figures/README.md)). Re-running the experiments needs
NebulaStream built for each platform they ran on: x86-64 for the server and laptop, aarch64 for the Raspberry Pi. What
follows is how we built it, the details depend on the host.

**Toolchain.** clang 19 with `clang-tools-19` (for `clang-scan-deps`), CMake 3.28 or newer, Ninja, a Rust toolchain (for
`nes-network`), a Java runtime (for the ANTLR parser generator), and NebulaStream's prebuilt LLVM/MLIR 21 for the target
architecture (`nes-llvm-21-with-fix-173075-*` from the `vmlir-21-with-fix-173075` release of
[nebulastream/clang-binaries](https://github.com/nebulastream/clang-binaries)). All other dependencies, OpenVINO and IREE
included, are built by vcpkg from the manifest in `vcpkg/` during the first CMake configure step, which takes a few hours.

On x86-64 or aarch64 Linux (we used Ubuntu 24.04):

```sh
export LLVM_TOOLCHAIN_VERSION=19   # vcpkg's toolchain looks for clang++-$LLVM_TOOLCHAIN_VERSION
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 \
    -DUSE_LIBCXX_IF_AVAILABLE=OFF -DUSE_LOCAL_MLIR=ON -DCMAKE_PREFIX_PATH=/path/to/nes-llvm
cmake --build cmake-build-release --target systest nes-single-node-worker nes-model-profiler \
    BatchInferenceMicrobenchmark PredictionCacheScopeMicrobenchmark
```

On aarch64, vcpkg also needs `VCPKG_FORCE_SYSTEM_BINARIES=1` and then uses the system CMake, which must be 3.31 or newer.
The Pi needs only `systest`, `nes-single-node-worker` and `nes-model-profiler`, placed in `bin/` of its working directory next to
the contents of [`scripts/benchmarking/e2e/navi`](scripts/benchmarking/e2e/navi). The binaries should be built on a distribution whose glibc is
not newer than glibc 2.41.

**Datasets and models.** Building `systest` also fetches all datasets and models the experiments use, through CMake's
`ExternalData`, from the TU Berlin cloud shares listed in
[`nes-systests/systest/CMakeLists.txt`](nes-systests/systest/CMakeLists.txt). Each file is identified and verified by the MD5
hash in its `.md5` file, under [`nes-systests/testdata/large/`](nes-systests/testdata/large) (datasets) and
[`nes-systests/testdata/model/`](nes-systests/testdata/model) (models). The configure step prints where the files end up.

**Model import and baselines.**

- Importing models needs OpenVINO's `ovc` 2025.3 on the `PATH`.
- On the Pi, the Python environment of the drivers, TorchServe and `ovc` is
  [`scripts/benchmarking/e2e/navi/models-benchmarking/requirements.txt`](scripts/benchmarking/e2e/navi/models-benchmarking/requirements.txt).
- The Flink + OpenVINO baseline is built with Maven and JDK 17, and its native OpenVINO bridge against vcpkg's OpenVINO on x86
  or against the `openvino` Python wheel on the Pi; see its
  [README](scripts/benchmarking/e2e/baselines/flink-openvino-baseline/README.md).

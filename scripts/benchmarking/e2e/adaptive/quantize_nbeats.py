#!/usr/bin/env python3
"""INT8 post-training quantization of nbeats_size_300_stride_30.onnx (NNCF), for the
model-precision knob of the adaptive-inference experiment.

Run it with an interpreter that has nncf/onnx/onnxruntime/openvino installed, e.g.

    venv/bin/python scripts/benchmarking/e2e/adaptive/quantize_nbeats.py \
        --model cmake-build-release/nes-systests/testdata/model/solar_power/pretrained/nbeats/nbeats_size_300_stride_30.onnx \
        --series cmake-build-release/nes-systests/testdata/large/solar_power/solar-power.csv \
        --benchmark

Design notes, all of them learned the hard way on this model:

* The graph boundary stays FLOAT32. NES's OpenVINO importer rejects a model whose input or output
  tensor is not f32 (OpenVinoImporter.cpp), so an INT8-IO export would not load at all.
* nbeats ends in a no-op Cast that writes the graph output. NNCF's own no-op-Cast pass looks such a
  node up in graph.value_info - where graph outputs never appear - and dies with `KeyError:
  'output'`, so the casts are folded away here first.
* Accuracy is measured through OpenVINO, not onnxruntime, because OpenVINO is what NES runs.
* Quantizing every operation costs this model a lot of accuracy (MAE ~0.09-0.12 on a [0,1]-scaled
  forecast whose own std is ~0.40) and NNCF's full BiasCorrection - the usual remedy - crashes on
  this graph in the ONNX backend. So when full INT8 misses --max-mae, the script ranks operations
  by their individual quantization error and quantizes the largest safe subset, leaving the rest in
  FP32. The resulting mixed model is a real knob: it is measurably faster, with a measured and
  bounded output deviation. `--allow-accuracy-loss` keeps the full-INT8 model instead.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DATA_ROOT = Path.home() / "PycharmProjects" / "nes-inference-datasets-models" / "time-series" / "solar-power"
DEFAULT_MODEL = DATA_ROOT / "models" / "pretrained" / "nbeats" / "nbeats_size_300_stride_30.onnx"
DEFAULT_SCALER = DATA_ROOT / "models" / "preprocessor" / "scaler_solar.onnx"
DEFAULT_SERIES = DATA_ROOT / "solar-power.csv"
WINDOW = 75
QUANTIZABLE_OPS = ("Gemm", "MatMul", "Conv")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES, help="CSV with 'timestamp,value' rows.")
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER, help="Per-value scaler ONNX; 'none' forces the affine fallback.")
    parser.add_argument("--scale", type=float, default=0.00858369, help="Fallback scaler slope, y = (x - offset) * scale.")
    parser.add_argument("--offset", type=float, default=0.0, help="Fallback scaler offset.")
    parser.add_argument("--series-rows", type=int, default=1_200_000, help="Rows read from the series CSV.")
    parser.add_argument("--window-stride", type=int, default=97, help="Stride between sampled windows (a prime avoids aliasing with the daily cycle).")
    parser.add_argument("--calib-windows", type=int, default=1024)
    parser.add_argument("--eval-windows", type=int, default=600)
    parser.add_argument(
        "--min-window-std",
        type=float,
        default=0.02,
        help="Calibration windows flatter than this are dropped. A solar series is idle at night, and "
             "calibrating on flat windows narrows the activation ranges badly (measured: MAE 0.215 vs 0.119).",
    )
    parser.add_argument(
        "--max-mae",
        type=float,
        default=0.05,
        help="Accuracy budget for the INT8 model: mean absolute output deviation from FP32, as a "
             "fraction of the FP32 output's own standard deviation (default 0.05 = 5%%).",
    )
    parser.add_argument(
        "--allow-accuracy-loss",
        action="store_true",
        help="Keep the fully quantized model even if it misses --max-mae (no mixed-precision search).",
    )
    parser.add_argument("--max-sensitivity-nodes", type=int, default=40, help="Cap on the per-node sensitivity analysis.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the INT8 model (default: alongside --model, with an _int8 suffix). The "
             "generated systest resolves 'model/...' against the build tree's testdata directory, so "
             "the default keeps it exactly where the .test expects it.",
    )
    parser.add_argument(
        "--ignore-op-types",
        default="Sub,Add",
        help="Comma-separated ONNX op types kept in FP32. The default keeps nbeats's residual "
             "elementwise ops out of INT8, which costs no speed and removes most of the error.",
    )
    parser.add_argument("--benchmark", action="store_true", help="Measure OpenVINO latency of both models.")
    parser.add_argument("--benchmark-batches", type=int, nargs="+", default=[1, 8, 64, 512])
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent / "quantization_report.json")
    return parser.parse_args()


# --------------------------------------------------------------------------------------------
# graph preparation
# --------------------------------------------------------------------------------------------
def strip_nop_casts(model):
    """Fold away every float->float Cast. Returns (model, number folded).

    Besides working around the NNCF crash, this removes nodes that would otherwise sit between the
    quantizer and the operations it wants to wrap.
    """
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model)
    elem_types = {
        tensor.name: tensor.type.tensor_type.elem_type
        for tensor in (*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output)
    }
    graph_outputs = {output.name for output in model.graph.output}
    initializers = {init.name for init in model.graph.initializer}

    folded = 0
    progressing = True
    while progressing:
        progressing = False
        producers = {name: node for node in model.graph.node for name in node.output}
        consumers: Dict[str, List] = defaultdict(list)
        for node in model.graph.node:
            for name in node.input:
                consumers[name].append(node)

        for node in list(model.graph.node):
            if node.op_type != "Cast":
                continue
            to_attr = next((onnx.helper.get_attribute_value(a) for a in node.attribute if a.name == "to"), None)
            source, target = node.input[0], node.output[0]
            if to_attr is None or elem_types.get(source) != to_attr:
                continue  # a real conversion

            if target in graph_outputs:
                # A graph output has no consumer to rewire: let the producer write it directly.
                producer = producers.get(source)
                if producer is None or source in initializers or source in graph_outputs or len(consumers[source]) != 1:
                    continue
                for index, name in enumerate(producer.output):
                    if name == source:
                        producer.output[index] = target
                stale = source
            else:
                for child in consumers[target]:
                    for index, name in enumerate(child.input):
                        if name == target:
                            child.input[index] = source
                stale = target

            for value_info in list(model.graph.value_info):
                if value_info.name == stale:
                    model.graph.value_info.remove(value_info)
            model.graph.node.remove(node)
            folded += 1
            progressing = True
            break
    return model, folded


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------
def load_series(path: Path, rows: int) -> np.ndarray:
    values: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 2:
                continue
            try:
                values.append(float(parts[1]))
            except ValueError:
                continue  # header
            if len(values) >= rows:
                break
    if len(values) < WINDOW * 4:
        raise SystemExit(f"Not enough usable rows in {path}: got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def find_scaler(model_path: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None and explicit.exists():
        return explicit
    candidate = model_path.parent.parent.parent / "preprocessor" / "scaler_solar.onnx"
    return candidate if candidate.exists() else None


def apply_scaler(series: np.ndarray, scaler_path: Optional[Path], scale: float, offset: float) -> np.ndarray:
    """Scale the raw series the way the training pipeline did.

    The solar scaler is a plain affine map, y = (x - offset) * scale, so when the ONNX file is not
    available (it is not shipped in the systest testdata tree) the same transform is applied from its
    constants. Getting this wrong is not cosmetic: PTQ calibrates activation ranges, so a series in
    the wrong range - a z-score, say, centred at 0 and going negative where training used [0, 1] -
    silently produces a badly calibrated INT8 model.
    """
    if scaler_path is not None:
        import onnxruntime as ort

        session = ort.InferenceSession(str(scaler_path), providers=["CPUExecutionProvider"])
        spec = session.get_inputs()[0]
        dtype = np.float64 if "double" in spec.type else np.float32
        scaled = session.run(None, {spec.name: series.reshape(-1, 1).astype(dtype)})[0]
        return np.asarray(scaled, dtype=np.float32).reshape(-1)
    print(
        f"scaler ONNX not found; applying the affine fallback y = (x - {offset}) * {scale} (the "
        f"solar-power scaler's constants). Pass --scaler/--scale/--offset for a different preprocessor.",
        file=sys.stderr,
    )
    return ((series - offset) * scale).astype(np.float32)


def all_windows(series: np.ndarray, stride: int) -> np.ndarray:
    starts = np.arange(0, len(series) - WINDOW, stride)
    stacked = np.stack([series[start : start + WINDOW] for start in starts])
    return stacked.reshape(len(starts), WINDOW, 1).astype(np.float32)


# --------------------------------------------------------------------------------------------
# OpenVINO evaluation - the path NES actually uses
# --------------------------------------------------------------------------------------------
class OpenVinoRunner:
    def __init__(self) -> None:
        import openvino as ov

        self.ov = ov
        self.core = ov.Core()
        self._temp_dir = Path(tempfile.mkdtemp(prefix="nes-quantize-"))
        self._counter = 0

    def save(self, model) -> Path:
        import onnx

        self._counter += 1
        path = self._temp_dir / f"model_{self._counter}.onnx"
        onnx.save(model, str(path))
        return path

    def infer(self, model_path: Path, data: np.ndarray) -> np.ndarray:
        model = self.core.read_model(str(model_path))
        model.reshape({model.input(0): self.ov.PartialShape([data.shape[0], WINDOW, 1])})
        compiled = self.core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 1, "NUM_STREAMS": 1})
        return np.asarray(compiled(data)[0], dtype=np.float32)

    def latency_us_per_record(self, model_path: Path, batch: int, iterations: int = 50) -> Optional[float]:
        try:
            model = self.core.read_model(str(model_path))
            model.reshape({model.input(0): self.ov.PartialShape([batch, WINDOW, 1])})
            compiled = self.core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 1, "NUM_STREAMS": 1})
        except Exception as exc:  # pragma: no cover - depends on model/runtime
            print(f"benchmark failed at batch {batch}: {exc}", file=sys.stderr)
            return None
        request = compiled.create_infer_request()
        data = np.random.rand(batch, WINDOW, 1).astype(np.float32)
        for _ in range(10):
            request.infer({0: data})
        start = time.perf_counter()
        for _ in range(iterations):
            request.infer({0: data})
        return (time.perf_counter() - start) / iterations / batch * 1e6

    def cleanup(self) -> None:
        shutil.rmtree(self._temp_dir, ignore_errors=True)


def deviation(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    delta = np.abs(candidate - reference)
    std = float(reference.std()) or 1.0
    return {
        "mae": float(delta.mean()),
        "mae_relative_to_std": float(delta.mean() / std),
        "rmse": float(np.sqrt((delta ** 2).mean())),
        "p99_abs_error": float(np.percentile(delta, 99)),
        "max_abs_error": float(delta.max()),
        "correlation": float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]),
        "output_std_candidate": float(candidate.std()),
        "output_std_reference": std,
    }


# --------------------------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------------------------
def quantize_model(
    base,
    calibration: np.ndarray,
    input_name: str,
    ignored_names: Sequence[str],
    ignored_types: Sequence[str],
):
    """Quantize, always leaving `ignored_types` in FP32.

    Excluding the residual elementwise ops is not a tuning nicety: quantizing nbeats's backcast
    Sub/Add costs ~0.11 MAE on its own - the same error whether one Gemm or all thirty are
    quantized - while buying no speed at all (measured: 1.67/2.07/2.86/3.01x with them excluded
    vs 1.69/2.06/2.77/2.96x with them included, at batch 1/8/64/512). They carry the residual
    signal between stacks, whose range is far wider than one INT8 scale can represent.
    """
    import nncf

    dataset = nncf.Dataset(iter(calibration), lambda item: {input_name: item.reshape(1, WINDOW, 1).astype(np.float32)})
    ignored_scope = nncf.IgnoredScope(names=list(ignored_names), types=list(ignored_types))
    return nncf.quantize(
        copy.deepcopy(base),
        dataset,
        subset_size=len(calibration),
        preset=nncf.QuantizationPreset.PERFORMANCE,
        ignored_scope=ignored_scope,
    )


def evaluate(runner: OpenVinoRunner, model, evaluation: np.ndarray, reference: np.ndarray) -> Tuple[Dict[str, float], Path]:
    path = runner.save(model)
    return deviation(runner.infer(path, evaluation), reference), path


def sensitivity_ranking(
    runner: OpenVinoRunner,
    base,
    nodes: List[str],
    calibration: np.ndarray,
    evaluation: np.ndarray,
    reference: np.ndarray,
    input_name: str,
    ignored_types: Sequence[str],
) -> List[Tuple[str, float]]:
    """Individual quantization error of every candidate node, ascending.

    Each node is quantized on its own (everything else stays FP32), which isolates its contribution.
    Errors do interact, so the ranking is only a heuristic for the greedy search below - but the
    accuracy of the model that is finally written out is always measured, never extrapolated.
    """
    ranking: List[Tuple[str, float]] = []
    for index, node in enumerate(nodes, start=1):
        ignored = [other for other in nodes if other != node]
        try:
            metrics, _ = evaluate(
                runner, quantize_model(base, calibration, input_name, ignored, ignored_types), evaluation, reference
            )
            ranking.append((node, metrics["mae"]))
        except Exception as exc:  # pragma: no cover - depends on model/runtime
            print(f"  sensitivity: {node} failed ({str(exc)[:60]}), treating as sensitive", file=sys.stderr)
            ranking.append((node, float("inf")))
        print(f"  [{index}/{len(nodes)}] {node}: MAE={ranking[-1][1]:.5f}", flush=True)
    return sorted(ranking, key=lambda item: item[1])


def select_mixed_precision(
    runner: OpenVinoRunner,
    base,
    ranking: List[Tuple[str, float]],
    calibration: np.ndarray,
    evaluation: np.ndarray,
    reference: np.ndarray,
    input_name: str,
    ignored_types: Sequence[str],
    budget: float,
) -> Tuple[Optional[object], Optional[Dict[str, float]], int, List[Dict[str, object]]]:
    """Largest prefix of the ranking that still meets the accuracy budget (binary search)."""
    ordered = [node for node, _ in ranking]
    trade_off: List[Dict[str, object]] = []
    best: Tuple[Optional[object], Optional[Dict[str, float]], int] = (None, None, 0)

    low, high = 0, len(ordered)
    while low <= high:
        count = (low + high) // 2
        if count == 0:
            low = 1
            continue
        ignored = ordered[count:]
        model = quantize_model(base, calibration, input_name, ignored, ignored_types)
        metrics, path = evaluate(runner, model, evaluation, reference)
        # Latency at a representative batch, so the search leaves behind a real accuracy/speed
        # curve rather than just a pass/fail verdict.
        speedup = None
        fp32_latency = runner.latency_us_per_record(runner.save(base), 64, iterations=30)
        int8_latency = runner.latency_us_per_record(path, 64, iterations=30)
        if fp32_latency and int8_latency:
            speedup = fp32_latency / int8_latency
        trade_off.append({"quantized_nodes": count, "speedup_at_batch_64": speedup, **metrics})
        print(f"  {count}/{len(ordered)} ops in INT8: MAE={metrics['mae']:.5f} "
              f"({metrics['mae_relative_to_std'] * 100:.2f}% of FP32 std)"
              + (f", {speedup:.2f}x at batch 64" if speedup else ""), flush=True)
        if metrics["mae_relative_to_std"] <= budget:
            best = (model, metrics, count)
            low = count + 1
        else:
            high = count - 1
    return (*best, trade_off)


def onnx_io_is_f32(model_path: Path) -> bool:
    import onnx

    model = onnx.load(str(model_path))
    tensors = list(model.graph.input) + list(model.graph.output)
    return all(tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for tensor in tensors)


def check_ovc(model_path: Path) -> bool:
    """NES imports every model through `ovc`; fail here rather than mid-sweep."""
    ovc = shutil.which("ovc")
    if ovc is None:
        print("ovc not on PATH - skipping the import check (NES will need it at query time)", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [ovc, str(model_path), "--output_model", str(Path(tmp) / "model.xml")],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"ovc failed on {model_path}:\n{result.stderr}", file=sys.stderr)
            return False
    return True


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    output = args.output or args.model.with_name(args.model.stem + "_int8.onnx")

    import onnx
    from nncf.common.logging.logger import set_log_level

    set_log_level(logging.ERROR)

    # ---- data -----------------------------------------------------------------------------
    series = load_series(args.series, args.series_rows)
    scaler = None if str(args.scaler).lower() == "none" else find_scaler(args.model, args.scaler)
    if scaler is not None:
        print(f"scaling the calibration series with {scaler}")
    scaled = apply_scaler(series, scaler, args.scale, args.offset)
    pool = all_windows(scaled, args.window_stride)
    active = pool[pool.std(axis=(1, 2)) > args.min_window_std]
    if len(active) < 32:
        raise SystemExit(f"Only {len(active)} non-flat calibration windows; lower --min-window-std or read more rows")
    rng = np.random.default_rng(0)
    calibration = active[rng.choice(len(active), size=min(args.calib_windows, len(active)), replace=False)]
    evaluation = pool[rng.choice(len(pool), size=min(args.eval_windows, len(pool)), replace=False)]
    print(f"{len(pool)} windows total, {len(active)} non-flat; calibrating on {len(calibration)}, "
          f"evaluating on {len(evaluation)}")

    # ---- graph ----------------------------------------------------------------------------
    base = onnx.load(str(args.model))
    base, folded = strip_nop_casts(base)
    if folded:
        print(f"folded {folded} no-op Cast node(s) (NNCF's own pass crashes on the one writing the graph output)")
    input_name = base.graph.input[0].name

    runner = OpenVinoRunner()
    try:
        reference = runner.infer(runner.save(base), evaluation)

        # ---- full INT8 ---------------------------------------------------------------------
        ignored_types = [item for item in args.ignore_op_types.split(",") if item]
        print(f"quantizing every supported operation except {ignored_types or ['(none)']} ...")
        full_model = quantize_model(base, calibration, input_name, [], ignored_types)
        full_metrics, _ = evaluate(runner, full_model, evaluation, reference)
        print(f"full INT8: MAE={full_metrics['mae']:.5f} "
              f"({full_metrics['mae_relative_to_std'] * 100:.2f}% of FP32 std), "
              f"corr={full_metrics['correlation']:.4f}")

        chosen_model, chosen_metrics = full_model, full_metrics
        mode = "full_int8"
        trade_off: List[Dict[str, object]] = []
        quantized_nodes = None

        if full_metrics["mae_relative_to_std"] > args.max_mae and not args.allow_accuracy_loss:
            candidates = [node.name for node in base.graph.node if node.op_type in QUANTIZABLE_OPS][
                : args.max_sensitivity_nodes
            ]
            print(f"full INT8 misses the {args.max_mae * 100:.1f}% budget; ranking {len(candidates)} "
                  f"operations by individual quantization error (this takes a few minutes) ...")
            ranking = sensitivity_ranking(
                runner, base, candidates, calibration, evaluation, reference, input_name, ignored_types
            )
            print("searching for the largest INT8 subset within budget ...")
            model, metrics, count, trade_off = select_mixed_precision(
                runner, base, ranking, calibration, evaluation, reference, input_name, ignored_types, args.max_mae
            )
            if model is None:
                print(
                    f"WARNING: not even a single operation could be quantized within {args.max_mae * 100:.1f}%. "
                    "Keeping the full INT8 model; re-run with a larger --max-mae or --allow-accuracy-loss "
                    "to silence this.",
                    file=sys.stderr,
                )
            else:
                chosen_model, chosen_metrics = model, metrics
                quantized_nodes = count
                mode = "mixed_int8"
                print(f"selected {count}/{len(candidates)} operations for INT8: "
                      f"MAE={metrics['mae']:.5f} ({metrics['mae_relative_to_std'] * 100:.2f}% of FP32 std)")

        # ---- write + verify ----------------------------------------------------------------
        output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(chosen_model, str(output))
        print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB, FP32 is {args.model.stat().st_size / 1e6:.2f} MB)")

        io_ok = onnx_io_is_f32(output)
        if not io_ok:
            print(
                "WARNING: the quantized graph lost its f32 boundary tensors; NES's OpenVINO importer "
                "will reject it.",
                file=sys.stderr,
            )
        ovc_ok = check_ovc(output)

        report: Dict[str, object] = {
            "model_fp32": str(args.model),
            "model_int8": str(output),
            "mode": mode,
            "quantized_nodes": quantized_nodes,
            "accuracy_budget_relative_to_std": args.max_mae,
            "f32_io": io_ok,
            "ovc_import_ok": ovc_ok,
            "accuracy": chosen_metrics,
            "accuracy_full_int8": full_metrics,
            "trade_off": trade_off,
            "calibration_windows": int(len(calibration)),
            "evaluation_windows": int(len(evaluation)),
        }

        if args.benchmark:
            latencies = {}
            for batch in args.benchmark_batches:
                fp32 = runner.latency_us_per_record(runner.save(base), batch)
                int8 = runner.latency_us_per_record(output, batch)
                latencies[batch] = {"fp32_us_per_record": fp32, "int8_us_per_record": int8}
                if fp32 and int8:
                    latencies[batch]["speedup"] = fp32 / int8
                    print(f"  batch {batch:>4}: fp32 {fp32:8.2f} us/record   int8 {int8:8.2f} us/record   "
                          f"({fp32 / int8:.2f}x)")
            report["latency"] = latencies

        args.report.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
        return 0 if (io_ok and ovc_ok) else 1
    finally:
        runner.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

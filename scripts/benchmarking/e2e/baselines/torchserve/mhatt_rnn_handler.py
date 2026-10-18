"""TorchServe custom handler for the MHATT-RNN (MUSAN audio) baseline.

NES aggregates the sliding window and ships the ready 1 s window to TorchServe through the
``HttpSink``, one POST per window. The body is ``application/octet-stream`` framed exactly like the
SqueezeDet and SARAD baselines::

    uint32 numArrays            (little-endian)
    repeated numArrays times:
        uint32 byteLength       (little-endian)
        byteLength raw bytes    (IEEE-754 float32, little-endian)

Each array is one source's ``audio1sec`` window: 16000 float32 samples, already scaled to [-1, 1]
by the query's ``audioSample / 32767.0``. The framework comparison uses a single audio stream, so
numArrays is 1; the handler still decodes and batches N arrays so the three-source variant of the
query works unchanged.

Runtime: OpenVINO, not onnxruntime. MHATT-RNN is a TensorFlow SavedModel whose Keras RNN layers
export ``OptionalFromValue``/``OptionalNone`` ops that tf2onnx cannot lower -- the resulting ONNX
carries untyped tensors and onnxruntime refuses to load it ("Invalid tensor data type 0"). NES
loads the SavedModel through OpenVINO's TensorFlow frontend and the Flink baseline runs the IR
converted from it by ovc, so serving the same IR here makes all three systems execute an identical
graph. The IR is archived as the serialized file; its .bin sits next to it via --extra-files.

Package with::

    torch-model-archiver \
        --model-name mhatt_rnn \
        --version 1.0 \
        --serialized-file saved_model.xml \
        --handler mhatt_rnn_handler.py \
        --extra-files saved_model.bin,input_geometry.json \
        --export-path model_store

Run with the sibling config.properties (metrics on :8082, one worker for CPU parity, no dynamic
batching -- the sink sends one window per request).
"""

import json
import os
import struct

import numpy as np
import openvino as ov
from ts.torch_handler.base_handler import BaseHandler

# Window geometry default: 1 s at the 16 kHz source rate. Overridden per run by the
# "input_geometry.json" bundled into the archive, because TorchServe workers do not inherit the
# parent env. It must match the model input and the ARRAY_AGG window in the .test.
DEFAULT_SAMPLES = 16000


def _decode_body(body: bytes) -> np.ndarray:
    """Decode the HttpSink binary framing into a (numArrays, N) float32 array."""
    offset = 0
    (num_arrays,) = struct.unpack_from("<I", body, offset)
    offset += 4
    arrays = []
    for _ in range(num_arrays):
        (byte_len,) = struct.unpack_from("<I", body, offset)
        offset += 4
        arr = np.frombuffer(body, dtype="<f4", count=byte_len // 4, offset=offset)
        offset += byte_len
        arrays.append(arr)
    # All windows are the same length; stack into one batch tensor.
    return np.stack(arrays, axis=0)


class MhattRnnHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = context.manifest["model"]["serializedFile"]
        model_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Missing OpenVINO IR: {model_path}")

        self.samples = DEFAULT_SAMPLES
        geometry_path = os.path.join(model_dir, "input_geometry.json")
        if os.path.isfile(geometry_path):
            with open(geometry_path, "r", encoding="utf-8") as fh:
                geometry = json.load(fh)
            self.samples = int(geometry.get("samples", DEFAULT_SAMPLES))

        core = ov.Core()
        model = core.read_model(model_path)

        # Pin the dynamic batch dimension to 1, exactly as the Flink baseline's JNI engine does:
        # the sink sends one window per request, and a static shape lets OpenVINO specialise.
        shape = model.input().get_partial_shape()
        if shape[0].is_dynamic:
            shape[0] = 1
            model.reshape(shape)

        # Same compile options as the NES OpenVINO backend and the Flink baseline: latency hint,
        # accuracy execution mode, one inference thread for CPU parity with a single NES worker.
        self.compiled = core.compile_model(
            model,
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                "EXECUTION_MODE_HINT": "ACCURACY",
                "NUM_STREAMS": "0",
                "INFERENCE_NUM_THREADS": os.environ.get("MHATT_RNN_INFERENCE_THREADS", "1"),
            },
        )
        self.output_port = self.compiled.output(0)

        self.initialized = True

    def preprocess(self, data):
        # One list entry per HTTP request, each already carrying N windows.
        batches = []
        self._split_sizes = []
        for row in data:
            body = row.get("body")
            if body is None:
                body = row.get("data")
            if isinstance(body, (bytearray, memoryview)):
                body = bytes(body)
            arrays = _decode_body(body)
            if arrays.shape[1] != self.samples:
                raise ValueError(
                    f"Expected {self.samples} float32 samples per window, got {arrays.shape[1]}"
                )
            self._split_sizes.append(arrays.shape[0])
            batches.append(arrays)
        return np.concatenate(batches, axis=0).astype(np.float32)

    def inference(self, data, *args, **kwargs):
        # The compiled model takes one window at a time (batch pinned to 1), so a request carrying
        # several windows is run window by window -- the same way NES invokes the model once per
        # joined input field.
        return np.concatenate(
            [self.compiled(window[np.newaxis, :])[self.output_port] for window in data], axis=0
        )

    def postprocess(self, data):
        probs = np.asarray(data)
        results = []
        start = 0
        for size in self._split_sizes:
            results.append(json.dumps(probs[start : start + size].tolist()))
            start += size
        return results

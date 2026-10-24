"""TorchServe custom handler for the SARAD (CWRU bearing vibration) baseline.

NES aggregates the sliding window and ships the ready 9600-sample window to TorchServe through the
``HttpSink``, one POST per window. The body is ``application/octet-stream`` framed exactly like the
SqueezeDet and NanoDet baselines::

    uint32 numArrays            (little-endian)          # == 1 here (single VARSIZED field)
    repeated numArrays times:
        uint32 byteLength       (little-endian)
        byteLength raw bytes    (IEEE-754 float32, little-endian)

The one array is the ``value_200_ms`` window: 200 ms of the 48 kHz vibration signal, i.e. 9600
float32 samples in the model's native g-units. NES does no scaling in the query, so this handler
does none either -- it only reshapes to the model's ``x [batch, 9600, 1]`` input.

The model is an ONNX file executed with OpenVINO (not torch), like the SqueezeDet handler, so
``initialize`` builds a compiled OpenVINO model from the archived file. OpenVINO reads ONNX directly
through its own frontend -- the same way NES loads this very file, and the Flink baseline runs the
IR ovc converts from it -- so all three systems execute the same graph and the comparison isolates
serving overhead instead of runtime differences. That matters here: single-threaded on the Pi this
model costs ~54 ms under onnxruntime against ~78 ms under OpenVINO, which is large enough to decide
the ranking on its own.

Package with::

    torch-model-archiver \
        --model-name sarad \
        --version 1.0 \
        --serialized-file sarad_medium.onnx \
        --handler sarad_handler.py \
        --extra-files input_geometry.json \
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

# Window geometry default. The actual size is set per-run from an "input_geometry.json" bundled into
# the model archive (torch-model-archiver --extra-files), because TorchServe workers do NOT inherit
# the parent env. It must match the model input and the ARRAY_AGG window in the .test:
# 200 ms at the 48 kHz source rate.
DEFAULT_SAMPLES = 9600


def _decode_body(body: bytes):
    """Decode the HttpSink binary framing into a list of float32 arrays."""
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
    return arrays


class SaradHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = context.manifest["model"]["serializedFile"]
        model_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Missing ONNX model file: {model_path}")

        core = ov.Core()
        model = core.read_model(model_path)

        # Pin the dynamic batch dimension to 1, as the Flink baseline's JNI engine does: the sink
        # sends one window per request, and a static shape lets OpenVINO specialise.
        shape = model.input().get_partial_shape()
        if shape[0].is_dynamic:
            shape[0] = 1
            model.reshape(shape)

        # CPU-only, one stream with four inference threads, the session NES workers use.
        self.compiled = core.compile_model(
            model,
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                "EXECUTION_MODE_HINT": "ACCURACY",
                "NUM_STREAMS": os.environ.get("SARAD_NUM_STREAMS", "1"),
                "INFERENCE_NUM_THREADS": os.environ.get("SARAD_INFERENCE_THREADS", "4"),
            },
        )
        self.output_port = self.compiled.output(0)

        self.samples = DEFAULT_SAMPLES
        geometry_path = os.path.join(model_dir, "input_geometry.json")
        if os.path.isfile(geometry_path):
            with open(geometry_path, "r", encoding="utf-8") as fh:
                geometry = json.load(fh)
            self.samples = int(geometry.get("samples", DEFAULT_SAMPLES))

        self.initialized = True

    def preprocess(self, data):
        # One list entry per HTTP request. The sink issues one request per window (no
        # TorchServe-side dynamic batching), so this is normally length 1; per-request sizes are
        # still tracked so a batched call stays correct.
        tensors = []
        self._split_sizes = []
        for row in data:
            body = row.get("body")
            if body is None:
                body = row.get("data")
            if isinstance(body, (bytearray, memoryview)):
                body = bytes(body)
            arrays = _decode_body(body)
            window = arrays[0]
            if window.size != self.samples:
                raise ValueError(
                    f"Expected {self.samples} float32 samples, got {window.size}"
                )
            tensors.append(window.reshape(1, self.samples, 1))
            self._split_sizes.append(1)
        return np.concatenate(tensors, axis=0).astype(np.float32)

    def inference(self, data, *args, **kwargs):
        # The batch dimension is pinned to 1, so a request carrying several windows is run window by
        # window, the same way NES invokes the model once per record.
        return np.concatenate(
            [self.compiled(window[np.newaxis, ...])[self.output_port] for window in data], axis=0
        )

    def postprocess(self, data):
        # NES emits the raw anomaly score to its sink and does no thresholding, so the score is
        # returned as-is. It is one float per window, so the response stays small and does not
        # weigh on the measured round trip.
        scores = np.asarray(data).reshape(-1)
        results = []
        start = 0
        for size in self._split_sizes:
            results.append(json.dumps({"anomaly_score": scores[start : start + size].tolist()}))
            start += size
        return results

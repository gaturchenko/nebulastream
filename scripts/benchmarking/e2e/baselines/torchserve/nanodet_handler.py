"""TorchServe custom handler for the NanoDet TorchServe baseline.

NES does all preprocessing (decode + resize + float conversion via ``RESIZE_IMAGE``)
and ships the ready tensor to TorchServe through the ``HttpSink``, one POST per frame.
The body is ``application/octet-stream`` framed exactly like the audio baseline::

    uint32 numArrays            (little-endian)          # == 1 here (single VARSIZED field)
    repeated numArrays times:
        uint32 byteLength       (little-endian)
        byteLength raw bytes    (IEEE-754 float32, little-endian)

The single array is the preprocessed frame written by ``RESIZE_IMAGE``:
**CHW, BGR, float32, unscaled** (pixel values in [0, 255], NOT divided by 255).
So no image decoding or normalization happens here — the handler only reshapes to
``(1, 3, H, W)`` and runs the model. This mirrors what the NES ``MODEL_INFERENCE``
path does (feed the tensor straight to the model), keeping the comparison fair.

The model is an ONNX file, executed with OpenVINO (not torch), so this handler overrides
``initialize`` to build a compiled OpenVINO model from the archived file. Every system in the
comparison runs OpenVINO so the measurement isolates serving overhead (see squeezedet_handler.py).

Package with::

    torch-model-archiver \
        --model-name nanodet \
        --version 1.0 \
        --serialized-file nanodet.onnx \
        --handler nanodet_handler.py \
        --export-path model_store

Run with the sibling config.properties (metrics on :8082 for the Prometheus scrape,
one worker for CPU parity, no dynamic batching — the sink sends one frame per request).
"""

import json
import os
import struct

import numpy as np
import openvino as ov
from ts.torch_handler.base_handler import BaseHandler

# Input geometry defaults. The actual size is set per-run from an "input_geometry.json" bundled into
# the model archive (torch-model-archiver --extra-files), so a single config (run_systems.sh) controls
# the size for a new model / NanoDet variant. TorchServe workers do NOT inherit the parent env, which
# is why the size travels inside the .mar rather than through an environment variable. It must match
# RESIZE_IMAGE(..., W, H) in the .test and the model input.
CHANNELS = 3
DEFAULT_HEIGHT = 192
DEFAULT_WIDTH = 192


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


class NanodetHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = context.manifest["model"]["serializedFile"]
        model_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Missing ONNX model file: {model_path}")

        core = ov.Core()
        model = core.read_model(model_path)

        # Pin the dynamic batch dimension to 1: the sink sends one frame per request.
        shape = model.input().get_partial_shape()
        if shape[0].is_dynamic:
            shape[0] = 1
            model.reshape(shape)

        # CPU-only and single-threaded for parity with the NES worker setup.
        self.compiled = core.compile_model(
            model,
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                "EXECUTION_MODE_HINT": "ACCURACY",
                "NUM_STREAMS": "0",
                "INFERENCE_NUM_THREADS": os.environ.get("NANODET_INFERENCE_THREADS", "1"),
            },
        )
        self.output_ports = list(self.compiled.outputs)

        # Input geometry: prefer the bundled input_geometry.json (written by run_systems.sh), else
        # fall back to the 192x192 default.
        self.height = DEFAULT_HEIGHT
        self.width = DEFAULT_WIDTH
        geometry_path = os.path.join(model_dir, "input_geometry.json")
        if os.path.isfile(geometry_path):
            with open(geometry_path, "r", encoding="utf-8") as fh:
                geometry = json.load(fh)
            self.height = int(geometry.get("height", DEFAULT_HEIGHT))
            self.width = int(geometry.get("width", DEFAULT_WIDTH))
        self.tensor_elems = CHANNELS * self.height * self.width

        self.initialized = True

    def preprocess(self, data):
        # One list entry per HTTP request. The sink issues one request per frame (no
        # TorchServe-side dynamic batching), so this is normally length 1; we still
        # track per-request sizes so batched calls stay correct.
        tensors = []
        self._split_sizes = []
        for row in data:
            body = row.get("body")
            if body is None:
                body = row.get("data")
            if isinstance(body, (bytearray, memoryview)):
                body = bytes(body)
            arrays = _decode_body(body)
            frame = arrays[0]
            if frame.size != self.tensor_elems:
                raise ValueError(
                    f"Expected {self.tensor_elems} float32 values "
                    f"({CHANNELS}x{self.height}x{self.width}), got {frame.size}"
                )
            tensors.append(frame.reshape(1, CHANNELS, self.height, self.width))
            self._split_sizes.append(1)
        return np.concatenate(tensors, axis=0).astype(np.float32)

    def inference(self, data, *args, **kwargs):
        # The batch dimension is pinned to 1, so a request carrying several frames is run frame by frame.
        per_frame = [self.compiled(frame[np.newaxis, ...]) for frame in data]
        return [
            np.concatenate([result[port] for result in per_frame], axis=0)
            for port in self.output_ports
        ]

    def postprocess(self, data):
        # `data` is a list of output tensors (one per model output head). NES emits the
        # raw model output to a discard sink and does no NMS, so for parity we return a
        # compact descriptor of the outputs rather than decoding detections — this keeps
        # the HTTP response small so it doesn't dominate the measured round trip.
        # Split the batch dimension back into one result per original request.
        results = []
        start = 0
        for size in self._split_sizes:
            summary = {
                "outputs": [
                    {"name": port.get_any_name(), "shape": list(out[start : start + size].shape)}
                    for port, out in zip(self.output_ports, data)
                ]
            }
            results.append(json.dumps(summary))
            start += size
        return results

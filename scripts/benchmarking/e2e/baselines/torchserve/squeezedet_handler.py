"""TorchServe custom handler for the SqueezeDet (192x624, KITTI arch) baseline.

NES does all preprocessing (decode + resize + float conversion via ``RESIZE_IMAGE``)
and ships the ready tensor to TorchServe through the ``HttpSink``, one POST per frame.
The body is ``application/octet-stream`` framed exactly like the NanoDet baseline::

    uint32 numArrays            (little-endian)          # == 1 here (single VARSIZED field)
    repeated numArrays times:
        uint32 byteLength       (little-endian)
        byteLength raw bytes    (IEEE-754 float32, little-endian)

Unlike NanoDet (NCHW), the SqueezeDet ONNX export keeps TensorFlow's NHWC input:
``image_input [batch, H, W, 3]``. NES's ``MODEL_INFERENCE`` path memcpys the
``RESIZE_IMAGE`` output (CHW, BGR, unscaled float32) straight into that NHWC input
tensor without transposing, so for parity this handler also only RESHAPES the
received floats to ``(1, H, W, 3)`` — no transpose, no normalization. The exported
model carries random weights (benchmark-only; wall-clock depends on op shapes, not
values), so the layout mismatch is deliberate and harmless.

The model is an ONNX file, executed with onnxruntime (not torch), so this handler
overrides ``initialize`` to build an ``ort.InferenceSession`` from the archived file.

Package with::

    torch-model-archiver \
        --model-name squeezedet \
        --version 1.0 \
        --serialized-file squeezedet-192x624.onnx \
        --handler squeezedet_handler.py \
        --export-path model_store

Run with the sibling config.properties (metrics on :8082 for the Prometheus scrape,
one worker for CPU parity, no dynamic batching — the sink sends one frame per request).
"""

import json
import os
import struct

import numpy as np
import onnxruntime as ort
from ts.torch_handler.base_handler import BaseHandler

# Input geometry defaults. The actual size is set per-run from an "input_geometry.json" bundled into
# the model archive (torch-model-archiver --extra-files), so a single config (run_systems.sh) controls
# the size for a new model variant. TorchServe workers do NOT inherit the parent env, which is why the
# size travels inside the .mar rather than through an environment variable. It must match
# RESIZE_IMAGE(..., W, H) in the .test and the model input.
CHANNELS = 3
DEFAULT_HEIGHT = 192
DEFAULT_WIDTH = 624


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


class SqueezedetHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = context.manifest["model"]["serializedFile"]
        model_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Missing ONNX model file: {model_path}")

        # CPU-only, single-threaded session for parity with the NES worker setup.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(os.environ.get("SQUEEZEDET_INTRA_OP_THREADS", "1"))
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Input geometry: prefer the bundled input_geometry.json (written by run_systems.sh), else
        # fall back to the 624x192 default.
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
                    f"({self.height}x{self.width}x{CHANNELS}), got {frame.size}"
                )
            # NHWC: reshape only, no transpose (mirrors NES's straight memcpy — see module doc).
            tensors.append(frame.reshape(1, self.height, self.width, CHANNELS))
            self._split_sizes.append(1)
        return np.concatenate(tensors, axis=0).astype(np.float32)

    def inference(self, data, *args, **kwargs):
        return self.session.run(self.output_names, {self.input_name: data})

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
                    {"name": name, "shape": list(out[start : start + size].shape)}
                    for name, out in zip(self.output_names, data)
                ]
            }
            results.append(json.dumps(summary))
            start += size
        return results

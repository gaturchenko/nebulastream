"""TorchServe custom handler for the NES HttpSink baseline.

The NES ``HttpSink`` POSTs one request per joined record with an
``application/octet-stream`` body framed as::

    uint32 numArrays            (little-endian)
    repeated numArrays times:
        uint32 byteLength       (little-endian)
        byteLength raw bytes    (IEEE-754 float32, little-endian)

Each array is one source's ``audio1sec`` window (16000 float32 samples). This
handler decodes the arrays, stacks them into a batch, runs the model, and
returns one probability vector per array.

Package with::

    torch-model-archiver \
        --model-name mhatt_rnn \
        --version 1.0 \
        --serialized-file mhatt_rnn.pt \        # or --model-file for eager
        --handler mhatt_rnn_handler.py \
        --export-path model_store

Run with a config.properties that matches the NES workload, e.g.::

    default_workers_per_model=1        # CPU parity with NES worker threads
    # do NOT enable dynamic batching: the sink already batches N arrays/request
    metrics_mode=prometheus            # expose /metrics on :8082 for throughput/latency
"""

import struct

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

SAMPLES_PER_WINDOW = 16000


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
    def preprocess(self, data):
        # TorchServe hands us a list with one entry per HTTP request. Because the
        # sink issues one request per record (no TorchServe-side dynamic batching),
        # this list normally has length 1, and each request already carries N arrays.
        batches = []
        self._split_sizes = []
        for row in data:
            body = row.get("body") or row.get("data")
            if isinstance(body, (bytearray, memoryview)):
                body = bytes(body)
            arrays = _decode_body(body)
            self._split_sizes.append(arrays.shape[0])
            batches.append(arrays)
        stacked = np.concatenate(batches, axis=0)  # (sum(N_i), SAMPLES_PER_WINDOW)
        return torch.from_numpy(stacked).float().to(self.device)

    def inference(self, data, *args, **kwargs):
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, data):
        probs = data.detach().cpu().numpy()
        # Re-split the flattened batch back into one result list per request.
        results = []
        start = 0
        for size in self._split_sizes:
            results.append(probs[start : start + size].tolist())
            start += size
        return results

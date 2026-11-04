"""Which OpenVINO configuration makes concurrent MHAtt-RNN inference grow memory on the Pi?

Mimics the MUSAN UDF: WORKERS threads, each compiling its own copy of the model (batch pinned to 1) on one shared ov.Core and
inferring back to back. Prints the properties OpenVINO resolved, then the process RSS every 10 s; stops a variant after
DURATION_S or as soon as MemAvailable drops below MIN_AVAILABLE_MB, so a leak cannot exhaust the Pi.

    python ov_leak_check.py <model.xml> <variant> [<variant> ...]
"""

import sys
import threading
import time

import numpy as np
import openvino as ov

WORKERS = 2
DURATION_S = 150
MIN_AVAILABLE_MB = 700

VARIANTS = {
    "defaults": {},
    "explicit_1x4": {"NUM_STREAMS": "1", "INFERENCE_NUM_THREADS": "4", "PERFORMANCE_HINT": "LATENCY",
                     "EXECUTION_MODE_HINT": "ACCURACY", "ENABLE_CPU_PINNING": "NO"},
    "defaults_pinning_off": {"ENABLE_CPU_PINNING": "NO"},
    "defaults_accuracy": {"EXECUTION_MODE_HINT": "ACCURACY"},
    "defaults_threads4": {"INFERENCE_NUM_THREADS": "4"},
}
PROPERTIES = ["NUM_STREAMS", "INFERENCE_NUM_THREADS", "ENABLE_CPU_PINNING", "EXECUTION_MODE_HINT", "INFERENCE_PRECISION_HINT",
              "PERFORMANCE_HINT", "ENABLE_HYPER_THREADING"]


def meminfo(key: str, path: str = "/proc/meminfo") -> int:
    with open(path) as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) // 1024
    return 0


def run(model_path: str, variant: str) -> None:
    core = ov.Core()
    model = core.read_model(model_path)
    shape = model.input().get_partial_shape()
    if shape[0].is_dynamic:
        shape[0] = 1
    model.reshape({model.input(): shape})
    compiled = [core.compile_model(model, "CPU", VARIANTS[variant]) for _ in range(WORKERS)]
    resolved = {}
    for name in PROPERTIES:
        try:
            resolved[name] = str(compiled[0].get_property(name))
        except Exception:  # noqa: BLE001 - not every property is readable
            resolved[name] = "-"
    print(f"== {variant}: {resolved}", flush=True)

    data = np.random.rand(*[d.get_length() for d in shape]).astype(np.float32)
    stop = threading.Event()
    calls = [0] * WORKERS

    def work(i: int) -> None:
        request = compiled[i].create_infer_request()
        while not stop.is_set():
            request.infer({0: data})
            calls[i] += 1

    workers = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(WORKERS)]
    for w in workers:
        w.start()
    start = time.time()
    samples = []
    while time.time() - start < DURATION_S:
        time.sleep(1)
        rss = meminfo("VmRSS", "/proc/self/status")
        samples.append((time.time() - start, rss))
        if int(samples[-1][0]) % 10 == 0:
            print(f"   {samples[-1][0]:4.0f} s  rss {rss:5d} MB  available {meminfo('MemAvailable'):5d} MB  calls {sum(calls)}",
                  flush=True)
        if meminfo("MemAvailable") < MIN_AVAILABLE_MB:
            print("   STOPPED: available memory below the limit", flush=True)
            break
    stop.set()
    for w in workers:
        w.join(timeout=30)
    t0, r0 = next((t, r) for t, r in samples if t >= 30)
    t1, r1 = samples[-1]
    print(f"   growth after 30 s: {(r1 - r0) / max(t1 - t0, 1):.2f} MB/s over {t1 - t0:.0f} s, "
          f"{sum(calls) / (t1):.1f} inferences/s", flush=True)


if __name__ == "__main__":
    for name in sys.argv[2:]:
        run(sys.argv[1], name)

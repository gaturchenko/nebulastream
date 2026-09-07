#!/usr/bin/env python3
"""Generate the File- and TCP-source counterparts of verify_batch_delivery.test.

The inline data is several MB, so the tests are generated rather than committed. See
BATCHING_DELIVERY.md for what they establish and the measured results.

    python3 gen_delivery_controls.py --out-dir /tmp/delivery
    cd cmake-build-release/nes-systests/systest
    ./systest -t /tmp/delivery/delivery_file.test:02 -n 1 --optimizer inference_batch_size=512 \
        -- --worker.query_engine.number_of_worker_threads=4 \
           --worker.default_query_execution.inference.prediction_cache_type=LRU \
           --worker.default_query_execution.inference.number_of_entries_prediction_cache=2048

Section :01 of each file is the control (same source, no inference operator) and must return the
full row count. Cross-check the sink count against the operator's own per-record counter:

    grep -o 'cache hits=[0-9]*, misses=[0-9]*' <log>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def rows_csv(count: int) -> str:
    # Every row distinct, so cache misses equal the records the operator actually processed.
    return "\n".join(
        f"{1.0 + i * 0.0001:.4f},{2.0 + i * 0.0001:.4f},{3.0 + i * 0.0001:.4f},{4.0 + i * 0.0001:.4f}"
        for i in range(count)
    )


def render(name: str, source_block: str, rows: int, count_note: str) -> str:
    upper = name.upper()
    fields = ", ".join(
        f"`{upper}$P{i}` FLOAT32 NOT NULL" for i in range(1, 5)
    ) + ", SETOSA FLOAT32 NOT NULL, VERSICOLOR FLOAT32 NOT NULL, VIRGINICA FLOAT32 NOT NULL"
    return f"""# name: delivery_{name}.test
# description: Does batched inference deliver every tuple with a {count_note} source?
#              Section :01 is the control - same source, no inference operator.
# groups: [Inference]
#
# Expected values are deliberately 0,0 so systest prints the ACTUAL count. Both sections must
# report exactly {rows} tuples.

GlobalConfiguration worker.default_query_execution.operator_buffer_size: [1048576]
GlobalConfiguration worker.number_of_buffers_in_global_buffer_manager: [4096]

CREATE LOGICAL SOURCE {name}(p1 FLOAT32 NOT NULL, p2 FLOAT32 NOT NULL, p3 FLOAT32 NOT NULL, p4 FLOAT32 NOT NULL);
{source_block}ATTACH INLINE
{rows_csv(rows)}

CREATE MODEL irisOV ('model/iris.onnx' BACKEND OpenVINO)
INPUT (p1 FLOAT32, p2 FLOAT32, p3 FLOAT32, p4 FLOAT32)
OUTPUT (setosa FLOAT32, versicolor FLOAT32, virginica FLOAT32);

CREATE SINK ctl{upper}(p1 FLOAT32 NOT NULL, p2 FLOAT32 NOT NULL, p3 FLOAT32 NOT NULL, p4 FLOAT32 NOT NULL) TYPE Checksum;
CREATE SINK infer{upper}({fields}) TYPE Checksum;

# :01 -- control, no inference operator
SELECT p1 AS p1, p2 AS p2, p3 AS p3, p4 AS p4 FROM {name} INTO ctl{upper};
----
0, 0

# :02 -- inference
SELECT * FROM MODEL_INFERENCE(irisOV, {name}) INTO infer{upper};
----
0, 0
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--file-rows", type=int, default=200_000, help="Enough rows to span several tuple buffers.")
    parser.add_argument("--tcp-rows", type=int, default=60_000)
    parser.add_argument("--tcp-rate", type=int, default=50_000, help="MOCK_TUPLE_RATE, the rate-limited counterpart.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    file_test = render("ffile", "CREATE PHYSICAL SOURCE FOR ffile TYPE File;\n", args.file_rows, "File")
    tcp_test = render(
        "ftcp",
        f"CREATE PHYSICAL SOURCE FOR ftcp TYPE TCP SET(\n    {args.tcp_rate} AS `SOURCE`.MOCK_TUPLE_RATE\n);\n",
        args.tcp_rows,
        f"TCP (rate-limited to {args.tcp_rate}/s)",
    )
    for name, text in (("delivery_file.test", file_test), ("delivery_tcp.test", tcp_test)):
        path = args.out_dir / name
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

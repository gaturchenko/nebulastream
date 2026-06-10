#!/bin/bash

cd ./scripts/benchmarking/e2e/baselines/flink-openvino-baseline

#./scripts/run-local.sh osu/pretrained/nanodet/nanodet.xml CPU 1000 1 0 8
./scripts/run-csv-object-detection.sh \
  osu/pretrained/nanodet/nanodet.xml \
  osu_rgb/otcbvs_osu_rgb.csv \
  CPU \
  3 \
  0 \
  16 \
  0 \
  false \
  320 \
  320 \
  false

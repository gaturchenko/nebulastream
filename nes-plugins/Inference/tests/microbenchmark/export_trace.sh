#!/bin/bash

#!/usr/bin/env bash

GENERATOR_BIN="cmake-build-release/nes-plugins/Inference/tests/DataGeneratorCsvExporter"

RECORDS=1000000
DRIFT_INTERVAL=1000
#ZIPF_KEYS=1000
#TEMPORAL_LOCALITY_UNIVERSE_SIZE=5000
#TEMPORAL_LOCALITY_SERIES_LENGTH=1000
#TEMPORAL_LOCALITY_WINDOW_SIZE=100
BURSTINESS_ON_PERIOD=1000
BURSTINESS_KEYS=1000


# Parameter sweeps
DRIFT_FRACTIONS=(0.0 0.1 0.5 1.0)
#ZIPF_S_VALUES=(0.0 0.6 1.0 1.2)
#TEMPORAL_LOCALITY_OVERLAP_RATIOS=(0.0 0.5 0.8 0.95)
BURSTINESS_DUTY_CYCLE=(0.01 0.05 0.2 0.5)

OUTPUT_DIR="generated_traces"
mkdir -p "$OUTPUT_DIR"

for drift in "${DRIFT_FRACTIONS[@]}"; do
#  for zipf_s in "${ZIPF_S_VALUES[@]}"; do
#   for overlap in "${TEMPORAL_LOCALITY_OVERLAP_RATIOS[@]}"; do
    for duty_cycle in "${BURSTINESS_DUTY_CYCLE[@]}"; do
#    output_file="${OUTPUT_DIR}/zipf_drift${drift}_s${zipf_s}.csv"
#    output_file="${OUTPUT_DIR}/temporal_drift${drift}_overlap${overlap}.csv"
     output_file="${OUTPUT_DIR}/burstiness_drift${drift}_duty_cycle${duty_cycle}.csv"
    echo "Generating ${output_file}"

    $GENERATOR_BIN \
      --generator burstiness \
      --records $RECORDS \
      --drift-interval $DRIFT_INTERVAL \
      --drift-fraction $drift \
      --burstiness-duty-cycle $duty_cycle\
      --burstiness-on-period $BURSTINESS_ON_PERIOD \
      --burstiness-num-keys $BURSTINESS_KEYS \
      --output "$output_file"
#      --zipf-num-keys $ZIPF_KEYS \
#      --zipf-s $zipf_s \
  done
done

echo "All traces generated."

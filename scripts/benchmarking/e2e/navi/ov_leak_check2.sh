#!/usr/bin/env bash
cd ~/navi
while pgrep -f "[o]v_leak_check.py models" > /dev/null; do sleep 5; done
: > ov_leak_check2.out
for v in defaults explicit_1x4 defaults_pinning_off defaults_accuracy defaults_threads4; do
    models-benchmarking/venv/bin/python ov_leak_check.py models-benchmarking/models/mhatt-rnn/saved_model.xml $v >> ov_leak_check2.out 2>&1
done
echo "### done" >> ov_leak_check2.out

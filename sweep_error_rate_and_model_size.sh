#!/bin/bash

# Define the path to your training script
TRAINING_SCRIPT="train.py"


Error_rate=(0.0 0.037 0.095 0.153 0.269)
MODELS=("resnet_mlperf" "resnet18" "resnet34" "resnet50" "resnet101")

RUNS=(1)

for run in "${RUNS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for er in "${Error_rate[@]}"; do
      # Run the training script
      python "$TRAINING_SCRIPT" -m="$MODEL" -e="$er" -n="${MODEL}_error_${er}_run_${run}_full_ds"
    done
  done
done

echo "Sweep completed!"
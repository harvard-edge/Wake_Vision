#!/bin/bash

# Define the path to your training script
TRAINING_SCRIPT="train.py"

# Define a list of learning rates to sweep over
Dataset_size=(10 25 50)
Error_rate=(0.0 0.095 0.269)
MODEL="resnet18"

RUNS=(1)

for run in "${RUNS[@]}"; do
  for ds_s in "${Dataset_size[@]}"; do
    for er in "${Error_rate[@]}"; do
      # Run the training script
      python "$TRAINING_SCRIPT" -m="$MODEL" -e="$er" -p="$ds_s" -n="${MODEL}_dsize_${ds_s}_error_${er}_run_${run}_normalsteps"
    done
  done
done

echo "Learning rate sweep completed!"
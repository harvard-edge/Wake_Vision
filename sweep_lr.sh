#!/bin/bash

# Define the path to your training script
TRAINING_SCRIPT="train.py"

# Define a list of learning rates to sweep over
STEPS=(100000 500000 1000000)
MODEL="resnet18"

# Loop through each learning rate
for steps in "${STEPS[@]}"; do
  # Run the training script with the current learning rate
  python "$TRAINING_SCRIPT" -m="$MODEL" --lr="0.01" -wd="0.01" -s="$steps" -n="${MODEL}_lr_0.001_wd_0.004_s${steps}"
done

echo "Learning rate sweep completed!"
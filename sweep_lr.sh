#!/bin/bash

# Define the path to your training script
TRAINING_SCRIPT="train.py"

# Define a list of learning rates to sweep over
LEARNING_RATES=(0.0001 0.001 0.01 0.1)
MODEL="resnet_mlperf"

# Loop through each learning rate
for lr in "${LEARNING_RATES[@]}"; do
  # Run the training script with the current learning rate
  python "$TRAINING_SCRIPT" -m="$MODEL" --lr="$lr" -n="$MODEL_lr_$lr"
done

echo "Learning rate sweep completed!"
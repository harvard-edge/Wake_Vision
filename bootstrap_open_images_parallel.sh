#!/bin/bash

for i in {10..29}; do
    if [[ $i -ne 30 && $i -ne 86 ]]; then
        # Create a new tmux session named after the iteration number
        tmux new-session -d -s $i
        # Send the command to start your python script to the tmux session
        tmux send-keys -t $i "source venv/bin/activate" C-m
        tmux send-keys -t $i "python bootstrap_open_images.py $i" C-m
    fi
done

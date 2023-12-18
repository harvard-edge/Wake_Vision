#!/bin/bash
# An alternative script to bootstrap_open_images_parallel.py. This script simply runs several instances of the non-parallel python script, each in a separate tmux instance. This is useful if you would like to be able to follow the progress of the script during execution.

for i in {10..99}; do
    # Create a new tmux session named after the iteration number
    tmux new-session -d -s $i
    # Send the command to start your python script to the tmux session
    tmux send-keys -t $i "source venv/bin/activate" C-m
    tmux send-keys -t $i "python bootstrap_open_images.py $i" C-m
done

#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=dev_cpu
# Base script to submit
BASE_SCRIPT="queue.sh"

# Array of flag sets for train.py (not SLURM flags)
TRAIN_FLAGS=(
#    "--env_names=h1-walk-v0"
    "--env_names=h1-stand-v0"
    "--env_names=h1-run-v0"
#    "--env_names=h1-stair-v0"
#    "--env_names=h1-crawl-v0"
    "--env_names=h1-pole-v0"
    "--env_names=h1-slide-v0"
    "--env_names=h1-hurdle-v0"
    "--env_names=h1-maze-v0"
)
if [ "$1" == "--test" ]; then
    echo "Running in test mode with reduced resources."
    BASE_SCRIPT="queue.sh --test"
fi

# Submit jobs with different train.py flags
for flags in "${TRAIN_FLAGS[@]}"; do
    echo "Submitting job with train.py flags: $flags"
    sbatch $BASE_SCRIPT $flags 
    sleep 1  # Small delay between submissions
done

echo "All jobs submitted!"

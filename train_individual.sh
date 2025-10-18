#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=dev_cpu
# Base script to submit
BASE_SCRIPT="test.sh"

# Array of flag sets for train.py (not SLURM flags)
TRAIN_FLAGS=(
    '--env_names=h1hand-sit_simple-v0',
    # '--env_names=h1hand-sit_hard-v0', 
    # '--env_names=h1hand-balance_simple-v0', 
    # '--env_names=h1hand-balance_hard-v0', 
    # '--env_names=h1hand-reach-v0', 
    # '--env_names=h1hand-spoon-v0', 
    # '--env_names=h1hand-window-v0', 
    # '--env_names=h1hand-insert_small-v0', 
    # '--env_names=h1hand-insert_normal-v0',
    # '--env_names=h1hand-bookshelf_simple-v0', 
    # '--env_names=h1hand-bookshelf_hard-v0', 
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

#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=dev_cpu
# Base script to submit
BASE_SCRIPT="print.sh"
EXTRA_FLAGS=("$@")
# Array of flag sets for train.py (not SLURM flags)
TRAIN_FLAGS=(
    '--env_names=h1hand-walk-v0',
    '--env_names=h1hand-stand-v0',
    '--env_names=h1hand-run-v0',
    '--env_names=h1hand-stair-v0',
    '--env_names=h1hand-crawl-v0',
    '--env_names=h1hand-pole-v0',
    '--env_names=h1hand-slide-v0',
    '--env_names=h1hand-hurdle-v0',
    '--env_names=h1hand-maze-v0',
    '--env_names=h1hand-sit_simple-v0',
    '--env_names=h1hand-sit_hard-v0',
    '--env_names=h1hand-balance_simple-v0',
    '--env_names=h1hand-balance_hard-v0',
    '--env_names=h1hand-reach-v0',
    '--env_names=h1hand-spoon-v0',
    '--env_names=h1hand-window-v0',
    '--env_names=h1hand-insert_small-v0',
    '--env_names=h1hand-insert_normal-v0',
    '--env_names=h1hand-bookshelf_simple-v0',
    '--env_names=h1hand-bookshelf_hard-v0'
)

# Submit jobs with different train.py flags
for flags in "${TRAIN_FLAGS[@]}"; do
    echo "Submitting job with train.py flags: $flags"
    sbatch "$BASE_SCRIPT" $flags "${EXTRA_FLAGS[@]}"
    sleep 1  # Small delay between submissions
done

echo "All jobs submitted!"

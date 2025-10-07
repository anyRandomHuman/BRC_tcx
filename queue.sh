#!/bin/bash
#SBATCH --time=00:03:00
#SBATCH --partition=debug_cpu_il
module load anaconda3

# Activate your conda environment
source activate py10

# --- Good Practice: Define file paths using variables ---

# Use the directory where you submitted the job as the base
# This ensures your paths are correct regardless of where the job runs
BASE_DIR="$SLURM_SUBMIT_DIR"

# Define the final output file on the shared filesystem
OUTPUT_FILE="$BASE_DIR/checkpoints"

python train.py --save_location "$OUTPUT_FILE"

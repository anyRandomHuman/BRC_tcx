#!/bin/bash
#SBATCH --time=00:03:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1

module load  devel/cuda/12.8
# Activate your conda environment
conda activate py10

# --- Good Practice: Define file paths using variables ---

# Use the directory where you submitted the job as the base
# This ensures your paths are correct regardless of where the job runs
#BASE_DIR="$SLURM_SUBMIT_DIR"

# Define the final output file on the shared filesystem
#OUTPUT_FILE="$BASE_DIR/checkpoints"

python train.py 
#--save_location "$OUTPUT_FILE"

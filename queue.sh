#!/bin/bash
#SBATCH --time=00:03:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --mail-user=urskl@student.kit.edu
#SBATCH --mail-type=FAIL,END,START
#SBATCH --signal=B:USR1@3600

module load  devel/cuda/12.8
# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate py10

# --- Good Practice: Define file paths using variables ---

# Use the directory where you submitted the job as the base
# This ensures your paths are correct regardless of where the job runs
#BASE_DIR="$SLURM_SUBMIT_DIR"

# Define the final output file on the shared filesystem
#OUTPUT_FILE="$BASE_DIR/checkpoints"

python train.py 
#--save_location "$OUTPUT_FILE"
conda deactivate

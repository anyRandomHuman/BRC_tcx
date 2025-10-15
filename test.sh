#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --mail-user=urskl@student.kit.edu
#SBATCH --mail-type=FAIL,END,START
#SBATCH --signal=B:USR1@60

module load  devel/cuda/12.8
# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate py10
export MUJOCO_GL=egl

cleanup_and_save()
{
    echo "---"
    echo "WARNING: Time limit approaching. Triggering final save..."
    # Tell the python script to save and exit
    # This could be done by creating a flag file or sending another signal
    touch "$SLURM_SUBMIT_DIR/pause_test.flag" 
    # Wait a bit for the python script to react and save
    sleep 55 
}

trap 'cleanup_and_save' USR1

python train.py --test=True
#--save_location "$OUTPUT_FILE"
conda deactivate

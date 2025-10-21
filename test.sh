#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1


module load  devel/cuda/11.8
# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate scale_rl
#conda activate DMC

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

python train_copy.py --env_names=pendulum-spin
#--save_location "$OUTPUT_FILE"
conda deactivate

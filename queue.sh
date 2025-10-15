#!/bin/bash
#SBATCH --time=22:00:00  # Uncomment this line
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mail-user=urskl@student.kit.edu
#SBATCH --mail-type=FAIL,END,START
#SBATCH --signal=B:USR1@180

module load devel/cuda/12.8
eval "$(conda shell.bash hook)"
conda activate py10
export MUJOCO_GL=egl

cleanup_and_save()
{
    echo "---"
    echo "WARNING: Time limit approaching. Triggering final save..."
    touch "$SLURM_SUBMIT_DIR/pause.flag"
    sleep 120
}

trap 'cleanup_and_save' USR1

# Pass all arguments to train.py
python train.py "$@"

conda deactivate
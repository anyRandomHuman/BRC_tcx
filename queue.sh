#!/bin/bash
#SBATCH --time=15:00:00  # Uncomment this line
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1

module load devel/cuda/12.8
eval "$(conda shell.bash hook)"
conda activate py10
export MUJOCO_GL=egl

# Pass all arguments to train.py
python train.py "$@"

conda deactivate

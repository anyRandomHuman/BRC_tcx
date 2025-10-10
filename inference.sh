#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1

module load  devel/cuda/12.8
# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate py10
export MUJOCO_GL=egl

python inference.py --ckp brc-HB_HANDS

conda deactivate

#!/bin/bash
if [ "$1" == "--test" ]; then
    echo "Running in test mode with reduced resources."
    #SBATCH --partition=gpu_a100_short
    #SBATCH --gres=gpu:1
    #SBATCH --time=00:10:00
    TEST_FLAG="--test=True"
else
    echo "Running in full mode."
    #SBATCH --time=22:00:00
    #SBATCH --partition=gpu_h100_il
    #SBATCH --gres=gpu:1
    #SBATCH --mail-user=urskl@student.kit.edu
    #SBATCH --mail-type=FAIL,END,START
    #SBATCH --signal=B:USR1@180
    TEST_FLAG="--test=False"
fi


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
python train.py "$@" $TEST_FLAG

conda deactivate

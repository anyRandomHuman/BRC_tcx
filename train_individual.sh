#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=dev_cpu
# Base script to submit
BASE_SCRIPT="queue_DMC.sh"

# Array of flag sets for train.py (not SLURM flags)
TRAIN_FLAGS=(
    '--env_names=walker-stand', 
    '--env_names=walker-walk', 
    '--env_names=walker-run', 
    '--env_names=cheetah-run', 
    '--env_names=reacher-easy',
	'--env_names=reacher-hard', 
    '--env_names=acrobot-swingup', 
    '--env_names=pendulum-swingup', 
    '--env_names=cartpole-balance', 
    '--env_names=cartpole-balance_sparse',
    '--env_names=cartpole-swingup', 
    '--env_names=cartpole-swingup_sparse', 
    '--env_names=ball_in_cup-catch', 
    '--env_names=finger-spin', 
    '--env_names=finger-turn_easy',
	'--env_names=finger-turn_hard', 
    '--env_names=fish-swim', 
    '--env_names=hopper-stand', 
    '--env_names=hopper-hop',
	'--env_names=cheetah-run_backwards', 
    '--env_names=cheetah-run_front', 
    '--env_names=cheetah-run_back',
	'--env_names=cheetah-jump', 
    '--env_names=walker-walk_backwards', 
    '--env_names=walker-run_backwards', 
    '--env_names=hopper-hop_backwards', 
    '--env_names=reacher-three_easy', 
    '--env_names=reacher-three_hard', 
    '--env_names=ball_in_cup-spin',
	'--env_names=pendulum-spin', 
    '--env_names=assembly-v2-goal-observable',
    '--env_names=basketball-v2-goal-observable',
    '--env_names=bin-picking-v2-goal-observable',
    '--env_names=box-close-v2-goal-observable',
    '--env_names=button-press-topdown-v2-goal-observable',
    '--env_names=button-press-topdown-wall-v2-goal-observable',
    '--env_names=button-press-v2-goal-observable',
    '--env_names=button-press-wall-v2-goal-observable',
    '--env_names=coffee-button-v2-goal-observable',
    '--env_names=coffee-pull-v2-goal-observable',
    '--env_names=coffee-push-v2-goal-observable',
    '--env_names=dial-turn-v2-goal-observable',
    '--env_names=disassemble-v2-goal-observable',
    '--env_names=door-close-v2-goal-observable',
    '--env_names=door-lock-v2-goal-observable',
    '--env_names=door-open-v2-goal-observable',
    '--env_names=door-unlock-v2-goal-observable',
    '--env_names=hand-insert-v2-goal-observable',
    '--env_names=drawer-close-v2-goal-observable',
    '--env_names=drawer-open-v2-goal-observable',
    '--env_names=faucet-open-v2-goal-observable',
    '--env_names=faucet-close-v2-goal-observable',
    '--env_names=hammer-v2-goal-observable',
    '--env_names=handle-press-side-v2-goal-observable',
    '--env_names=handle-press-v2-goal-observable',
    '--env_names=handle-pull-side-v2-goal-observable',
    '--env_names=handle-pull-v2-goal-observable',
    '--env_names=lever-pull-v2-goal-observable',
    '--env_names=pick-place-wall-v2-goal-observable',
    '--env_names=pick-out-of-hole-v2-goal-observable',
    '--env_names=pick-place-v2-goal-observable',
    '--env_names=plate-slide-v2-goal-observable',
    '--env_names=plate-slide-side-v2-goal-observable',
    '--env_names=plate-slide-back-v2-goal-observable',
    '--env_names=plate-slide-back-side-v2-goal-observable',
    '--env_names=peg-insert-side-v2-goal-observable',
    '--env_names=peg-unplug-side-v2-goal-observable',
    '--env_names=soccer-v2-goal-observable',
    '--env_names=stick-push-v2-goal-observable',
    '--env_names=stick-pull-v2-goal-observable',
    '--env_names=push-v2-goal-observable',
    '--env_names=push-wall-v2-goal-observable',
    '--env_names=push-back-v2-goal-observable',
    '--env_names=reach-v2-goal-observable',
    '--env_names=reach-wall-v2-goal-observable',
    '--env_names=shelf-place-v2-goal-observable',
    '--env_names=sweep-into-v2-goal-observable',
    '--env_names=sweep-v2-goal-observable',
    '--env_names=window-open-v2-goal-observable',
    '--env_names=window-close-v2-goal-observable'
)

# Submit jobs with different train.py flags
for flags in "${TRAIN_FLAGS[@]}"; do
    echo "Submitting job with train.py flags: $flags"
    sbatch $BASE_SCRIPT $flags 
    sleep 1  # Small delay between submissions
done

echo "All jobs submitted!"

# Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners

https://arxiv.org/pdf/2505.23150

This branch contains the implementation of the BRC algorithm.

## Example usage

To run the BRC algorithm in a single task mode, just pass a single task name to the `env_names` variable:

`python3 train.py --env_names=dog-run`

By passing a list of task names, multi-task mode will be enabled. 

## Citation

If you find this repository useful, feel free to cite our paper using the following bibtex.

```
@article{nauman2025bigger,
  title={Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners},
  author={Nauman, Michal and Cygan, Marek and Sferrazza, Carmelo and Kumar, Aviral and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2505.23150},
  year={2025}
}
```

## Changes by Chenxin Tao
1. `update.py`: implemented `evaluate_actor`, `evaluate_critic`, `_weight_metric_tree_func`, `_activation_metric_tree_func`, `_grad_conflict_tree_func`, `is_leaf_2d`, `compute_grad_conflict` and `compute_per_layer_metrics`
, which computes per-layer metrics for the network including (dead neurons percentage, dormant ratio, feature norm, parameter norm, effective rank and parameter norm)

2. `brc_learner.py`: 
   1. changed `_get_infos` to use `evaluate_actor`and `evaluate_critic`
   2. added `load_inference` that only loads actor
   3. change the initialization for critic to use `DoubleCritic` instead of `DoubleCriticTest` for logging the intermediates
3. `inference.py`, use the actor to produces a video of a rollout
4.  `play_video.py` play the video from `inference.py`
5. implement `queue.sh`, `queue_DMC.sh` `train_individual.sh`, `test.sh` for slurm queueing
6. `train.py`
   1. change save locations for cluster
   2. add functionality for pause and resume from pause(in case of insufficient run time)
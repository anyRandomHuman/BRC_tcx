import jax
import jax.numpy as jnp
import numpy as np
# from flax import linen as nn
# from typing import Sequence
# from jaxrl.networks import NormalTanhPolicy, DoubleCriticTest, Temperature
# from jaxrl.utils import tree_norm, prune_single_child_nodes, merge_trees_overwrite, flatten_tree, Batch, Model
# # from jaxrl.agent.brc_learner import BRC
# import jaxrl.agent.update as update
# from jaxrl.replay_buffer import ParallelReplayBuffer
# import gymnasium as gym

# key = jax.random.PRNGKey(0)
# # Use a representative batch of data
# input_data = jnp.zeros((16, 32)) # Batch size of 128

# # model = SimpleMLP(features=[16, 8, 4])
# model = NormalTanhPolicy(action_dim=2, hidden_dims=4, depth=2)
# critic = DoubleCriticTest(num_tasks=1, embedding_size=4, ensemble_size=2, hidden_dims=4, depth=2, output_nodes=101, multitask=False)
# temp = Temperature()
# # Initialize the model parameters
# params = model.init(key, input_data)['params']

# final_output, mutable_variables = model.apply(
#     {'params': params}, 
#     input_data,
#     capture_intermediates=True,
#     mutable=True # We need mutable=True to allow intermediates to be stored
# )

# intermediates = mutable_variables['intermediates']

# low = np.zeros(51)
# high = np.ones(51)
# box = gym.spaces.Box(low=low, high=high)
# # brc = BRC(0, jnp.zeros(1), jnp.zeros(1), 1, width_critic=2, width_actor=2)
# # b = Batch(observations=obs, actions=jnp.zeros(1), rewards=jnp.zeros(1), masks=jnp.ones(1), next_observations=obs, task_ids=jnp.array([0]))
# rb = ParallelReplayBuffer(observation_space=box, action_dim=19, capacity=250000, num_tasks=1)
# rb.n_parts = 1
# rb.load(r'./checkpoints/brc-HB_NOHANDS')
# b = rb.sample(3, 1)
# update.update_actor(key, model, critic, temp, b, 101, 10.0, False, False)

# grad_fn = jax.grad(jnp.tanh)
# a = grad_fn(jnp.array((0.3,0.2)))
fn = jax.vmap(lambda x :x)
a = fn(jnp.array([1,2]))

print(type(a))

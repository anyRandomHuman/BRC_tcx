import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
from jaxrl.networks import NormalTanhPolicy
from jaxrl.utils import tree_norm, prune_single_child_nodes, merge_trees_overwrite, flatten_tree
from jaxrl.agent import brc_learner as brc, update
# --- 1. Define a standard Flax model ---
class SimpleMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(features=feat, name=f'dense_{i}')(x)
            # We will capture the output of this ReLU activation
            x = nn.relu(x)
        return x

# --- 2. Initialize model and data ---
key = jax.random.PRNGKey(0)
# Use a representative batch of data
input_data = jnp.zeros((16, 32)) # Batch size of 128

# model = SimpleMLP(features=[16, 8, 4])
model = NormalTanhPolicy(action_dim=2, hidden_dims=4, depth=2)
# Initialize the model parameters
params = model.init(key, input_data)['params']

final_output, mutable_variables = model.apply(
    {'params': params}, 
    input_data,
    capture_intermediates=True,
    mutable=True # We need mutable=True to allow intermediates to be stored
)

intermediates = mutable_variables['intermediates']


p = update.compute_per_layer_metrics(update._weight_metric_tree_func, params)
# d = dead_neurals(intermediates)
# m = merge_trees_overwrite(d, p)
print(p)
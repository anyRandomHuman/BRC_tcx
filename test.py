import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
from jaxrl.networks import NormalTanhPolicy
from jaxrl.utils import tree_norm, prune_single_child_nodes, merge_trees_overwrite

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

def _dead_neurals_tree_func(activation, dict={}):
    dead_neurals_info = {}
    
    outputs = activation
    num_neurons = outputs.shape[1]
    dead_neurons = jnp.all(outputs == 0, axis=0).sum().item()
    dead_percentage = (dead_neurons / num_neurons) * 100
    dead_neurals_info = dict|{
        'dead_neurons': dead_neurons,
        'total_neurons': num_neurons,
        'dead_percentage': dead_percentage
    }
    return dead_neurals_info

def dead_neurals(intermediates, tree=None):
    if tree:
        dead = jax.tree.map(_dead_neurals_tree_func, intermediates, tree)
    else:
        dead = jax.tree.map(_dead_neurals_tree_func, intermediates)
        
    return prune_single_child_nodes(dead)

def parameter_norm_per_layer(params):
    norm = jnp.sqrt(sum(params**2).sum())
    return norm




p = jax.tree.map(parameter_norm_per_layer, params)
d = dead_neurals(intermediates)
m = merge_trees_overwrite(d, p)
print(p)
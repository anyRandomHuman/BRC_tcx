import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

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

model = SimpleMLP(features=[16, 8, 4])
# Initialize the model parameters
params = model.init(key, input_data)['params']

# --- 3. Run the model and capture activations ---
# We define a filter function to only capture the output of ReLU activations.
# In Flax, `nn.relu` isn't a layer, but a function. So we look for outputs
# from the Dense layers right before the ReLU is applied in the model's call.
# A simpler way is to just grab all intermediates and filter by name later.

# The `capture_intermediates` argument is the key.
# We pass a boolean `True` to get everything.
final_output, mutable_variables = model.apply(
    {'params': params}, 
    input_data,
    capture_intermediates=True,
    mutable=True # We need mutable=True to allow intermediates to be stored
)

intermediates = mutable_variables['intermediates']

def _dead_neurals_tree_func(activation):
    dead_neurals_info = {}
    
    outputs = activation
    num_neurons = outputs.shape[1]
    dead_neurons = jnp.all(outputs == 0, axis=0).sum().item()
    dead_percentage = (dead_neurons / num_neurons) * 100
    dead_neurals_info = {
        'dead_neurons': dead_neurons,
        'total_neurons': num_neurons,
        'dead_percentage': dead_percentage
    }
    return dead_neurals_info

def dead_neurals(intermediates):
    dead = jax.tree.map(_dead_neurals_tree_func, intermediates)
    return dead

d = dead_neurals(intermediates)

f = jax.tree.flatten(d)

print(f)
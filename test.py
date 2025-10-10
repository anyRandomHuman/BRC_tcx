from jaxrl.networks import NormalTanhPolicy
import os
import jax
import jax.numpy as jnp
from jaxrl.utils import Model
ckp = r'./checkpoints/brc-HB_NOHANDS/actor.txt'
save_path = r'test_space\actor.txt'
# brc = BRC(0, 0,0, 1)
actor_def = NormalTanhPolicy(action_dim=6)
actor = Model.create(actor_def, [jax.random.PRNGKey(0), jnp.zeros((1, 30))])
actor.save(save_path)

actor1 = Model.create(actor_def, [jax.random.PRNGKey(0), jnp.zeros((1, 30))])
actor1.load(save_path)

# with open(ckp, 'rb') as f:
#     print(f.read())
    
#     print('-----------------')
#     actor.load(ckp)
    
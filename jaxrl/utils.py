import os
import collections
from typing import Any, Optional, Sequence

import flax
import jax
import jax.numpy as jnp
import optax
from copy import deepcopy

Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'task_ids'])

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

def prune_single_child_nodes(tree):
    # If it's a leaf (not a container), return as is
    if not isinstance(tree, (dict, list, tuple)):
        return tree
    # Convert to list of children
    if isinstance(tree, dict):
        children = list(tree.values())
    else:
        children = list(tree)
    # Recursively prune children
    pruned_children = [prune_single_child_nodes(child) for child in children]
    # If only one child, return that child directly
    if len(pruned_children) == 1:
        return pruned_children[0]
    # Otherwise, reconstruct the container
    if isinstance(tree, dict):
        return dict(zip(tree.keys(), pruned_children))
    elif isinstance(tree, list):
        return pruned_children
    elif isinstance(tree, tuple):
        return tuple(pruned_children)

def merge_trees_overwrite(tree1, tree2):
    """Merge tree2 into tree1, with tree2 values taking precedence, also merge list"""
    result = tree1.copy()
    
    for key, value in tree2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_trees_overwrite(result[key], value)
        elif key in result and isinstance(result[key], jax.Array) and isinstance(value, jax.Array) \
            and result[key].shape[-1] == value.shape[-1]:
            result[key] = jnp.concatenate(result[key], value, axis=-1)
        else:
            result[key] = value
    
    return result

def merge_trees(tree_ls):
    result = {}
    for tree in tree_ls:
        merge_trees_overwrite(result, tree)
    return result

def flatten_tree(tree):
    flattened_ls, _ = jax.tree_util.tree_flatten_with_path(tree)
    flattened_dict = {}
    for path, value  in flattened_ls:
        key = ''
        for node in path:
            key += str(node)[2:-2] + '.' # Remove "['" and "']"
        key = key[:-1]  # Remove trailing dot   
        flattened_dict[key] = value
    return flattened_dict

def remove_from_tree(tree, to_remove_keys=['LayerNorm', 'bias', 'flat']):
    # tree_copy = deepcopy(tree)
    for key in list(tree.keys()):
        for rk in to_remove_keys:
            if rk in key:
                del tree[key]
                break
            elif isinstance(tree[key], dict):
                remove_from_tree(tree[key])
    return tree

def _weight_metric_tree_func(weight_matrix, rank_delta=0.01):
    if not (hasattr(weight_matrix, 'shape') and len(weight_matrix.shape) == 3):
        return {
        'effective_rank': jnp.array(-1),
        'parameter_norm': jnp.array(-1)
    }
    sing_values = jax.vmap(jnp.linalg.svd, in_axes=(0,None,None))(weight_matrix, True, False)
    cumsum = jnp.cumsum(sing_values, axis=-1)
    nuclear_norm = jnp.sum(sing_values, axis=1)
    approximate_rank_threshold = 1.0 - rank_delta
    threshold_crossed = jnp.where((cumsum >= approximate_rank_threshold * nuclear_norm), 1, 0)
    effective_rank = sing_values.shape[1] - jnp.sum(threshold_crossed, axis=1) + 1

    pnorm = jnp.sqrt(sum(weight_matrix ** 2).sum(axis=1))

    return_dict = {
        'effective_rank': effective_rank,
        'parameter_norm': pnorm
    }
    return return_dict


def _activation_metric_tree_func(activation, dormant_threshold=0.025, dead_threshold=0.0001):
    #shape  (num_network, b, neuron) in case of vmap network (num_network, critic, b, neuron)
    if not hasattr(activation, 'shape') or not len(activation.shape) == 3:
        return {
            'dead_percentage': jnp.array(-1),
            'dormant_ratio': jnp.array(-1),
            'feature_norm': jnp.array(-1)
        }
    sactivation = activation
    activation_mean = sactivation.mean(axis=1)  #mean over batch dimension (num_network, neuron)
    num_neurons = sactivation.shape[-1]
    neuron_var = jnp.var(sactivation, axis=1)
    dead_neurons = jnp.where(neuron_var < dead_threshold, 1, 0)
    dead_percentage = (dead_neurons.sum(axis=1) / num_neurons) * 100

    dormant_score = activation_mean / jnp.expand_dims(activation_mean.mean(axis=1), 1) #(num_network, neuron)
    dormant_ratio = jnp.sum(jnp.where(dormant_score < dormant_threshold, 1, 0), axis=1) / num_neurons

    fnorm = jnp.sqrt(jnp.square(activation_mean).sum(axis=1))

    return_dict = {
        'dead_percentage': dead_percentage,
        'dormant_ratio': dormant_ratio,
        'feature_norm': fnorm
    }
    return return_dict


def _grad_conflict_tree_func(grads):
    if not hasattr(grads, 'shape') or not len(grads.shape) == 4:
        return {'conflict_rate': jnp.array(-1)}
    #grad shape (1, batch, in, out)

    fgrads = jnp.reshape(grads, grads.shape[:-2] + (-1,))  #shape(1, b, n*m)
    fgrads1 = fgrads[:, 0]  #(1, b, n*m)
    # norm_prods = (jnp.linalg.norm(grads1, axis=(-1,-2)) *jnp.linalg.norm(fgrads, axis=(-1,-2)) + 1e-8) #b,2
    unnormed_cosine_similaritiy = jnp.einsum('...i,...i->...', fgrads1, fgrads)  #(1,b)
    conflit_mask = jnp.where(unnormed_cosine_similaritiy < 0, 1, 0)
    conflict_count = conflit_mask.sum(axis=1) / grads.shape[1]
    return {'conflict_rate': conflict_count}

def keep_from_dict(dict:dict, keep_keys=['Dense']):
    new_dict = {}
    for key, value in dict.items():
        for keep_key in keep_keys:
            if keep_key in key:
                new_dict[key] = value
    return new_dict


@flax.struct.dataclass
class SaveState:
    params: Params
    opt_state: Optional[optax.OptState] = None


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: flax.linen.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: flax.linen.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None):
        variables = model_def.init(*inputs)

        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn, *args, **kwargs):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params, *args, **kwargs)
        grad_norm = tree_norm(grads)
        info['grad_norm'] = grad_norm
        info['grads'] = grads

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info
    
    def get_gradient(self, loss_fn):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        return grads

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(SaveState(params=self.params, opt_state=self.opt_state)))

    def load(self, load_path: str):
        with open(load_path, 'rb') as f:
            contents = f.read()
            saved_state = flax.serialization.from_bytes(
                SaveState(params=self.params, opt_state=self.opt_state), contents
            )
        return self.replace(params=saved_state.params, opt_state=saved_state.opt_state)
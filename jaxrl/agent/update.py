import functools
import jax.numpy as jnp
import jax
from jax.flatten_util import ravel_pytree
from jaxrl.utils import Batch, Model, Params, PRNGKey, tree_norm, prune_single_child_nodes, merge_trees_overwrite, \
    flatten_tree, remove_from_tree, merge_trees
import optax
from copy import deepcopy


@jax.jit
def _weight_metric_tree_func(weight_matrix, rank_delta=0.01):
    # if not (hasattr(weight_matrix, 'shape') and len(weight_matrix.shape) == 2):
    #     return {
    #     'effective_rank': 0,
    #     'parameter_norm': 0
    # }
    # sing_values = jnp.linalg.svd(weight_matrix, compute_uv=False)
    # cumsum = jnp.cumsum(sing_values)
    # nuclear_norm = jnp.sum(sing_values)
    # approximate_rank_threshold = 1.0 - rank_delta
    # threshold_crossed = (cumsum >= approximate_rank_threshold * nuclear_norm)
    # effective_rank = sing_values.shape[0] - jnp.sum(threshold_crossed) + 1
    #
    pnorm = jnp.sqrt(sum(weight_matrix ** 2).sum())

    return_dict = {
        # 'effective_rank': effective_rank,
        'parameter_norm': pnorm
    }
    return return_dict


@jax.jit
def _activation_metric_tree_func(activation, dormant_threshold=0.025, dead_threshold=0.0001):
    #shape (critic, b, neuron) (b, neuron)
    sactivation = jnp.squeeze(activation)
    # print(f'sactivation shape: {sactivation.shape}')
    if not hasattr(activation, 'shape') or (not len(sactivation.shape) == 2 and not len(sactivation.shape) == 3):
        return {
            'dead_percentage': -1.0,
            'dormant_ratio': -1.0,
            'feature_norm': -1.0
        }

    if len(sactivation.shape) == 3:
        sactivation = sactivation.mean(axis=0)

    activation_mean = sactivation.mean(axis=0)  #mean over batch dimension
    num_neurons = sactivation.shape[1]
    neuron_var = jnp.var(sactivation, axis=0)
    dead_neurons = jnp.where(neuron_var < dead_threshold, jnp.ones(sactivation.shape[1]),
                             jnp.zeros(sactivation.shape[1]))
    dead_percentage = (dead_neurons.sum() / num_neurons) * 100

    dormant_score = activation_mean / activation_mean.mean()
    dormant_ratio = jnp.sum(dormant_score < dormant_threshold) / num_neurons

    fnorm = jnp.sqrt(sum(activation_mean ** 2).sum())

    return_dict = {
        'dead_percentage': dead_percentage,
        'dormant_ratio': dormant_ratio,
        'feature_norm': fnorm
    }
    return return_dict


@jax.jit
def _grad_conflict_tree_func(grads):
    sgrads = jnp.squeeze(grads, axis=0) if grads.shape[0] == 1 else grads
    if not hasattr(grads, 'shape') or not len(grads.shape) == 3:
        return {'conflict_rate': -1}
    #grad shape (1, batch, num critic=2, in, out)

    fgrads = jnp.reshape(sgrads, sgrads.shape[:-2] + (-1,))  #shape critic(1, b, 2, n*m) actor(b, n*m)
    fgrads1 = fgrads[0]  #2,n*m
    # norm_prods = (jnp.linalg.norm(grads1, axis=(-1,-2)) *jnp.linalg.norm(fgrads, axis=(-1,-2)) + 1e-8) #b,2
    unnormed_cosine_similaritiy = jnp.einsum('...i,...i->...', fgrads1, fgrads)  #(1,b,2) (1,b)
    conflit_mask = jnp.where(unnormed_cosine_similaritiy < 0, 1, 0)
    conflict_count = conflit_mask.sum(axis=0).mean()
    return {'conflict_rate': conflict_count / sgrads.shape[0]}


def is_leaf_2d(x):
    return hasattr(x, 'shape') and len(x.shape) == 2


def compute_grad_conflict(grads, network_name, remove_ln=True, is_leaf=None):
    # remove_from_tree(grads)
    conflict_tree = jax.tree.map(_grad_conflict_tree_func, grads, is_leaf=is_leaf)
    conflict_tree = {f'{network_name}': conflict_tree}
    fc = flatten_tree(conflict_tree)
    return fc


def compute_per_layer_metrics(tree_func, tree, network_name, remove_ln=True, is_leaf=None):
    return_tree = jax.tree.map(tree_func, tree, is_leaf=is_leaf)
    # remove_from_tree(return_tree)
    # prune_single_child_nodes(return_tree)
    return flatten_tree({f'{network_name}': return_tree})

@functools.partial(jax.jit, static_argnames=('multitask'))
def build_actor_input(critic: Model, observations: jnp.ndarray, task_ids: jnp.ndarray, multitask: bool):
    inputs = observations
    if multitask:
        task_embeddings = critic(None, None, task_ids, True)
        inputs = jnp.concatenate((inputs, task_embeddings), axis=-1)
    return inputs


@functools.partial(jax.jit, static_argnames=('num_bins', 'v_max', 'multitask'))
def evaluate_actor(key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: Batch,
                   num_bins: int, v_max: float, multitask: bool):
    inputs = build_actor_input(critic, batch.observations, batch.task_ids, multitask)
    @jax.jit
    def actor_loss_fn(actor_params: Params, observation, actor_input, task_id):
        dist, intermediate = actor.apply({'params': actor_params}, actor_input, capture_intermediates=True,
                                         mutable=True)
        actions, log_probs = dist.sample_and_log_prob(seed=key)  # #action

        #changes for computing efective rank and dead neurons

        # actions, log_probs = dist.sample_and_log_prob(seed=key) # #action
        # q_logits = critic(observations, actions, batch.task_ids) #batch, #action?, #value bins
        # sq_probs = jax.nn.softmax(q_logits, axis=-1) #batch, #action?, #value bins

        # bin_values = jnp.linspace(start=-v_max, stop=v_max, num=num_bins)[None, None, :] 
        # sq_values = (bin_values* sq_probs).sum(-1) #batch, #action?
        # sactor_loss = ((log_probs * temp().mean())[None, :]  - sq_values).mean(axis=1)

        q_logits = critic(observation, actions, task_id)  #batch, #action?, #value bins
        q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)  #action?, #value bins
        bin_values = jnp.linspace(start=-v_max, stop=v_max, num=num_bins)[None]
        q_values = (bin_values * q_probs).sum(-1)  #action?
        actor_loss = (log_probs * temp().mean() - q_values).mean()  #1

        return actor_loss, (jnp.array([actor_loss, -log_probs.mean()]), intermediate)

    info = {}
    network_name = 'actor'
    grad_fn = jax.vmap(jax.grad(actor_loss_fn, has_aux=True), in_axes=(None, 0, 0, 0))
    grad, loss_entropy_intermediate = grad_fn(actor.params, batch.observations, inputs, batch.task_ids)

    grad_norm = tree_norm(grad)
    info['grad_norm'] = grad_norm
    conflicts = compute_grad_conflict(grad, network_name)
    info = info|conflicts

    loss_entropy = loss_entropy_intermediate[0]
    loss_entropy = jnp.array(loss_entropy)

    info['entropy'] = loss_entropy[:,0].mean()
    info['actor_loss'] = loss_entropy[:,1].mean()

    intermediate = loss_entropy_intermediate[1]
    params_info = compute_per_layer_metrics(_weight_metric_tree_func, actor.params, network_name, is_leaf=is_leaf_2d)
    # params_info = compute_per_layer_params(actor.params, network_name, is_leaf=is_leaf_2d)
    info |= params_info
    #
    # features_info = compute_per_layer_metrics(_activation_metric_tree_func, intermediate['intermediates'], network_name)
    # features_info_copy = deepcopy(features_info)
    # for key in features_info.keys():
    #     if 'flat' in key:
    #         features_info_copy.pop(key)
    # info |= features_info_copy
    #
    # actor_pnorm = tree_norm(actor.params)
    # info['actor_pnorm'] = actor_pnorm

    return info

@functools.partial(jax.jit, static_argnames=('num_bins', 'v_max', 'multitask'))
def update_actor(key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: Batch,
                 num_bins: int, v_max: float, multitask: bool, evaluate=False):
    inputs = build_actor_input(critic, batch.observations, batch.task_ids, multitask)

    @jax.jit
    def actor_loss_fn(actor_params: Params, observations):
        #changes for computing efective rank and dead neurons
        dist = actor.apply({'params': actor_params}, inputs)

        actions, log_probs = dist.sample_and_log_prob(seed=key)  # #action
        q_logits = critic(observations, actions, batch.task_ids)  #batch, #action?, #value bins
        q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)  #action?, #value bins
        bin_values = jnp.linspace(start=-v_max, stop=v_max, num=num_bins)[None]
        q_values = (bin_values * q_probs).sum(-1)  #action?
        actor_loss = (log_probs * temp().mean() - q_values).mean()  #1

        loss_info = {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'actor_pnorm': tree_norm(actor_params),
        }

        return actor_loss, loss_info

    new_actor, info = actor.apply_gradient(actor_loss_fn, batch.observations)
    info['actor_gnorm'] = info.pop('grad_norm')

    return new_actor, info


@functools.partial(jax.jit, static_argnames=('num_bins', 'v_max', 'multitask'))
def evaluate_critic(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                    temp: Model, batch: Batch, discount: float, num_bins: int, v_max: float, multitask: bool):
    #note that batch size is always 32
    inputs = build_actor_input(critic, batch.next_observations, batch.task_ids, multitask)
    dist = actor(inputs)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)  #(batch, DOF), (batch)

    next_q_logits = target_critic(batch.next_observations, next_actions,
                                  batch.task_ids)  #shape (num_critic, batch_size, num_bins) (2, 1024, 101)
    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)  #shape (batch, num_bins)

    # compute target value buckets
    v_min = -v_max
    bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[None]  # 1, num_bins
    target_bin_values = batch.rewards[:, None] + discount * batch.masks[:, None] * (
                bin_values - temp() * next_log_probs[:, None])  #shape (num_action, num_bins) (1024, 101)
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_values = (target_bin_values - v_min) / ((v_max - v_min) / (num_bins - 1))

    lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
    lower_mask = jax.nn.one_hot(lower.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))  #batch, bin, bin
    upper_mask = jax.nn.one_hot(upper.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    lower_values = (next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values))[
        ..., None]  #batch, num_bins, 1

    upper_values = (next_q_probs * (target_bin_values - lower))[..., None]  #shape (num_critic, batch_size, num_bins, 1)
    target_probs = jax.lax.stop_gradient(
        jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1))  #shape (1, batch, num_bins) (1, 32,101)
    q_value_target = (bin_values * target_probs).sum(-1)  # 1, batch

    @jax.jit
    def critic_loss_fn(critic_params: Params, observations, actions, task_ids, target_prob):
        q_logits, intermediate = critic.apply({'params': critic_params}, observations, actions, task_ids,
                                              capture_intermediates=True,
                                              mutable=True)  #shape (batch_size=1, num_critic, num_bins)
        q_logprobs = jax.nn.log_softmax(q_logits[None, :], axis=-1)  #shape (batch_size, num_critic, num_bins)

        critic_loss = -(target_prob[None] * q_logprobs).sum(-1).mean(-1).sum(-1)

        return critic_loss, (critic_loss, intermediate)

    network_name = 'critic'
    grad_fn = jax.vmap(jax.grad(critic_loss_fn, has_aux=True), in_axes=(None, 0, 0, 0, 0))
    grad_trees, loss_intermediate = grad_fn(critic.params, batch.observations, batch.actions, batch.task_ids,
                                            target_probs)
    info = {}
    info |= compute_grad_conflict(grad_trees, network_name)

    critic_loss = loss_intermediate[0].mean()
    intermediate = loss_intermediate[1]

    info |= {
        "critic_loss": critic_loss,
        "q_mean": q_value_target.mean(),
        "q_min": q_value_target.min(),
        "q_max": q_value_target.max(),
        "r": batch.rewards.mean(),
        "critic_pnorm": tree_norm(critic.params),
    }
    # param_metrics = compute_per_layer_params(critic.params, network_name)

    param_metrics = compute_per_layer_metrics(_weight_metric_tree_func, critic.params, network_name)
    info |= param_metrics
    # feature_metrics = compute_per_layer_metrics(_activation_metric_tree_func, intermediate['intermediates'], network_name)
    # info |= feature_metrics
    return info


def update_critic_old(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                      temp: Model, batch: Batch, discount: float, num_bins: int, v_max: float, multitask: bool):
    inputs = build_actor_input(critic, batch.next_observations, batch.task_ids, multitask)
    dist = actor(inputs)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_q1_logits, next_q2_logits = target_critic(batch.next_observations, next_actions,
                                                   batch.task_ids)  # batch ,action, #bin
    next_q1_probs = jax.nn.softmax(next_q1_logits, axis=-1)  # batch ,action, #bin
    next_q2_probs = jax.nn.softmax(next_q2_logits, axis=-1)
    next_q_probs = (next_q1_probs + next_q2_probs) / 2
    # compute target value buckets
    v_min = -v_max
    bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[None]  # 1, #bin
    target_bin_values = batch.rewards[:, None] + discount * batch.masks[:, None] * (
                bin_values - temp() * next_log_probs[:, None])  # batch ,1
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_values = (target_bin_values - v_min) / ((v_max - v_min) / (num_bins - 1))
    lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
    lower_mask = jax.nn.one_hot(lower.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))  # batch , 1
    upper_mask = jax.nn.one_hot(upper.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    lower_values = (next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values))[..., None]
    upper_values = (next_q_probs * (target_bin_values - lower))[..., None]
    target_probs = jax.lax.stop_gradient(jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1))
    q_value_target = (bin_values * target_probs).sum(-1)

    def critic_loss_fn(critic_params: Params):
        critic_fn = lambda actions: critic.apply({"params": critic_params}, batch.observations, actions, batch.task_ids)

        def _critic_fn(actions):
            q1, q2 = critic_fn(actions)
            return 0.5 * (q1 + q2).mean(), (q1, q2)

        (_, (q1_logits, q2_logits)), action_grad = jax.value_and_grad(_critic_fn, has_aux=True)(batch.actions)
        q1_logprobs = jax.nn.log_softmax(q1_logits, axis=-1)
        q2_logprobs = jax.nn.log_softmax(q2_logits, axis=-1)

        loss1 = -jnp.mean(jnp.sum(target_probs * q1_logprobs, axis=1))
        loss2 = -jnp.mean(jnp.sum(target_probs * q2_logprobs, axis=1))
        critic_loss = loss1 + loss2

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q_value_target.mean(),
            "q_min": q_value_target.min(),
            "q_max": q_value_target.max(),
            "r": batch.rewards.mean(),
            "critic_pnorm": tree_norm(critic_params),
            "critic_agnorm": jnp.sqrt((action_grad ** 2).sum(-1)).mean(0),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    info["critic_gnorm"] = info.pop("grad_norm")
    return new_critic, info


@functools.partial(jax.jit, static_argnames=('num_bins', 'v_max', 'multitask'))
def update_critic(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                  temp: Model, batch: Batch, discount: float, num_bins: int, v_max: float, multitask: bool):
    inputs = build_actor_input(critic, batch.next_observations, batch.task_ids, multitask)
    dist = actor(inputs)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_q_logits = target_critic(batch.next_observations, next_actions,
                                  batch.task_ids)  #shape (num_critic, batch_size, num_bins) (2, 1024, 101)
    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)
    # compute target value buckets
    v_min = -v_max
    bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[None]

    target_bin_values = batch.rewards[:, None] + discount * batch.masks[:, None] * (
                bin_values - temp() * next_log_probs[:, None])  #shape (batch_size, num_bins) (1024, 101)
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_values = (target_bin_values - v_min) / ((v_max - v_min) / (num_bins - 1))
    lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
    lower_mask = jax.nn.one_hot(lower.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    upper_mask = jax.nn.one_hot(upper.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    lower_values = (next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values))[..., None]
    upper_values = (next_q_probs * (target_bin_values - lower))[..., None]
    target_probs = jax.lax.stop_gradient(jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1))
    q_value_target = (bin_values * target_probs).sum(-1)
    @jax.jit
    def critic_loss_fn(critic_params: Params):
        q_logits = critic.apply({"params": critic_params}, batch.observations, batch.actions, batch.task_ids)
        q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)
        critic_loss = -(target_probs[None] * q_logprobs).sum(-1).mean(-1).sum(-1)
        loss_info = {
            "critic_loss": critic_loss,
            "q_mean": q_value_target.mean(),
            "q_min": q_value_target.min(),
            "q_max": q_value_target.max(),
            "r": batch.rewards.mean(),
            "critic_pnorm": tree_norm(critic_params),
            #"critic_agnorm": jnp.sqrt((action_grad**2).sum(-1)).mean(0),
        }

        return critic_loss, loss_info

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    info["critic_gnorm"] = info.pop("grad_norm")
    return new_critic, info


def update_target_critic(critic: Model, target_critic: Model, tau: float):
    new_target_params = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)
    return target_critic.replace(params=new_target_params)

@jax.jit
def update_temperature(temp: Model, entropy: float, target_entropy: float):
    def temperature_loss_fn(temp_params):
        temperature = temp.apply({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        info = {'temperature': temperature, 'temp_loss': temp_loss}

        return temp_loss, info

    new_temp, info = temp.apply_gradient(temperature_loss_fn)
    info.pop('grad_norm')
    return new_temp, info
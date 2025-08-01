from jax import tree_map, jit, value_and_grad, vmap
from jax.random import split
from jax.numpy import ones, where, expand_dims, log
from typing import Callable, Any
from flax.core import FrozenDict
from flax.struct import PyTreeNode, field
from functools import partial
import optax
import pickle
import flax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from jax.experimental.host_callback import id_print
from jax.scipy.special import logsumexp
from utils import wandb_log_internal
import jax 
import numpy as np

class TrainState(PyTreeNode):
    """
    Expanded reimplementaion of Flax's TrainState class. Encapsulates model parameters, 
    the main model function used in training and the optimization state. 

    For VAE support, also retains the current state of the RNGs using in latent sampling, 
    updating after each gradient application. 

    Inheriting from flax.struct.PytreeNode is a convinient way to register class as a
    JAX PyTree, which allows it to be the input/output of complied JAX functions.
    """
    step: int
    apply_fn: Callable = field(pytree_node=False) # Speify this field as a normal python class (not serializable pytree)
    params: dict[str, Any]
    batch_stats: dict[str, Any]
    rng_state: dict[str, Any]
    tx: optax.GradientTransformation = field(pytree_node=False)
    opt_state: optax.OptState

    def update_rng(self):
        """Update the state of all RNGs stored by the trainstate and return a key for use in this sampling step"""
        new_key, sub_key = tree_map(lambda r: split(r)[0], self.rng_state), tree_map(lambda r: split(r)[1], self.rng_state)
        return self.replace(rng_state=new_key), sub_key

    def apply_gradients(self, *, grads, batch_stats, **kwargs):
        """Take a gradient descent step with the specified grads and encapsulated optimizer."""
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            batch_stats=batch_stats,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, batch_stats, rng_state, tx,  **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            batch_stats=batch_stats,
            rng_state=rng_state,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

def create_train_state(rng, learning_rate, model, input_shape):
    """Creates initial `TrainState`."""
    model = model()
    init_rng, init_model_rng, model_rng, dropout_rng = split(rng, 4)
    init = model.init({'params': init_rng, 'sampler': init_model_rng, 
                       'dropout': dropout_rng}, ones(input_shape))
    tx = optax.adam(learning_rate)
    return model, TrainState.create(
        apply_fn=model.apply, params=init['params'], batch_stats=init['batch_stats'] if 'batch_stats' in init else {},
        rng_state=dict(sampler=model_rng, dropout=dropout_rng), tx=tx)

def neg_log_lik_loss(dist, x, mask = None, mask_sst = None):
    """Negative log-likelihood of an observation"""
    if mask is not None:
        probs = where(expand_dims(mask, -1), -dist.log_prob(x), 0)
        return probs.sum()
    #dist.log_prob(x).shape: (5, 24, 2340)
    return (-dist.log_prob(x) * mask_sst).sum()

@partial(jit, static_argnums=6)
def train_step(state, batch, month=None, mask=None, mask_sst=None, N_data = 1, local_kl_weight=1., 
               prior_kl_weight=1., log_magnitudes=False, **kwargs):    
    """Train for a single step."""
    def loss_fn(params, rngs, batch_stats):
        """Inner function that computes loss and metrics."""
        #SVAE_LDS.apply
        #batch:(bs, T, 26*90) --> ndim = 3
        #N_data = 110
        #prior_kl_weight: 1.0
        #local_kl_weight: 0.001
        #aux: (z, sur_loss, gaus_expected_stats)
        #likelihood: tfp.distributions.Normal(
            # "Normal", batch_shape=[5, 24, 2340], event_shape=[], dtype=float32)
        #prior_kl.shape, local_kl.shape, recon_loss.shape: (), (), ()
        #aux: (z, sur_loss, gaus_expected_stats)
        #aux[0].shape: (5, 24, 50)
        #aux[1].shape: ()
        #aux[2]: tuple
        (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch, month=month, mask=mask, rngs=rngs, mutable=['batch_stats'], **kwargs)
        recon_loss = neg_log_lik_loss(likelihood, batch, mask=mask, mask_sst=mask_sst)
        if batch.ndim == 3:
            recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]
        loss = recon_loss + prior_kl * prior_kl_weight + local_kl * local_kl_weight
        return loss, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, 
                          loss=recon_loss + prior_kl + local_kl, batch_stats=batch_stats['batch_stats'], 
                          aux=aux)

    # Create a function that runs loss_fn and gets the gradient with respect to the first input.
    grad_fn = value_and_grad(loss_fn, has_aux=True)

    # Run the transformed function and apply the resulting gradients to the state
    state, keys = state.update_rng()
    (_, outputs), grads = grad_fn(state.params, keys, state.batch_stats)
    
    # Log gradient magnitudes
    if log_magnitudes:
        pg = 0.
        if 'pgm' in grads:
            pg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['pgm'])[0] ** 2))
        eg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['encoder'])[0] ** 2))
        dg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['decoder'])[0] ** 2))
        props = jnp.mean(outputs['aux'][-1], axis=(0, 1))
        max_used_state = jnp.max(props)
        n_used_states = jnp.sum(props > (1. / (props.shape[0] * 2)))
        jax.debug.callback(lambda pgm_g, encoder_g, decoder_g, mus, nus: wandb_log_internal(dict(pgm_grad_norm=pgm_g, encoder_grad_norm=encoder_g, decoder_grad_norm=decoder_g, max_used_state=mus, n_used_states=nus)), pg, eg, dg, max_used_state, n_used_states) 
    
    state = state.apply_gradients(grads=grads, batch_stats=outputs['batch_stats'])
    metrics = dict(recon_loss=outputs['recon_loss'], prior_kl=outputs['prior_kl'], 
                   local_kl = outputs['local_kl'], 
                   loss=outputs['recon_loss'] + outputs['prior_kl'] + outputs['local_kl'], 
                   aux=outputs['aux'][-1])

    # Return the updated state and computed metrics
    return state, metrics

def make_train_step_with_transform(trans):
    """Returns train_step function"""
    
    # Update static_argnums to include the index of log_magnitudes
    # Parameters: state, batch, month, mask, mask_sst, N_data, local_kl_weight, prior_kl_weight, log_magnitudes
    # Indices:    0,     1,     2,    3,        4,      5,              6,              7,              8
    @partial(jit, static_argnums=(8,))  # log_magnitudes is at index 8 now
    def train_step_with_transform(
        state, batch, month=None, mask=None, mask_sst=None, N_data=1, local_kl_weight=1., 
        prior_kl_weight=1., log_magnitudes=False, **kwargs):
        """Train for a single step."""
        def loss_fn(params, rngs, batch_stats):
            """Inner function that computes loss and metrics."""
            #model.apply
            (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                trans(batch), month=trans(month), mask=mask, rngs=rngs, mutable=['batch_stats'], **kwargs)
            recon_loss = neg_log_lik_loss(likelihood, batch, mask=mask, mask_sst=mask_sst)
            if batch.ndim == 3:
                recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]
            loss = recon_loss + prior_kl * prior_kl_weight + local_kl * local_kl_weight
            return loss, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, loss=recon_loss + prior_kl + local_kl, batch_stats=batch_stats['batch_stats'], aux=aux)

        # Create a function that runs loss_fn and gets the gradient with respect to the first input.
        grad_fn = value_and_grad(loss_fn, has_aux=True)

        # Run the transformed function and apply the resulting gradients to the state
        state, keys = state.update_rng()
        (_, outputs), grads = grad_fn(state.params, keys, state.batch_stats)

        # Log gradient magnitudes
        if log_magnitudes:
            pg = 0.
            if 'pgm' in grads:
                pg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['pgm'])[0] ** 2))
            eg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['encoder'])[0] ** 2))
            dg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['decoder'])[0] ** 2))
            props = jnp.mean(outputs['aux'][-1], axis=(0, 1))
            max_used_state = jnp.max(props)
            n_used_states = jnp.sum(props > (1. / (props.shape[0] * 2)))
            jax.debug.callback(lambda pgm_g, encoder_g, decoder_g, mus, nus: wandb_log_internal(dict(pgm_grad_norm=pgm_g, encoder_grad_norm=encoder_g, decoder_grad_norm=decoder_g, max_used_state=mus, n_used_states=nus)), pg, eg, dg, max_used_state, n_used_states) 

        state = state.apply_gradients(grads=grads, batch_stats=outputs['batch_stats'])
        metrics = dict(recon_loss=outputs['recon_loss'], prior_kl=outputs['prior_kl'], local_kl = outputs['local_kl'], loss=outputs['recon_loss'] + outputs['prior_kl'] + outputs['local_kl'], aux=outputs['aux'][-1])

        # Return the updated state and computed metrics
        return state, metrics
    return train_step_with_transform

def make_eval_step_with_transform(trans):
    @jit
    def eval_step_with_transform(state, batch, month=None, mask_sst=None, mask=None, N_data=1, **kwargs):
        """Compute metrics for a single batch. Still returns updated state to account for consuming RNG state."""
        state, keys = state.update_rng()
        likelihood, prior_kl, local_kl, aux = state.apply_fn({'params': state.params, "batch_stats": state.batch_stats},
                                                          trans(batch), month=trans(month), eval_mode=True, mask=mask, rngs=keys, **kwargs)
        recon_loss = neg_log_lik_loss(likelihood, batch, mask=mask, mask_sst=mask_sst)
        if batch.ndim == 3: #Check this
            recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]

        loss = recon_loss + prior_kl + local_kl
        return state, loss, likelihood, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, likelihood=likelihood.log_prob(batch), loss=loss, aux=aux)
    return eval_step_with_transform

@partial(jit, static_argnums=7)
def grad_step(state, batch, month=None, mask=None, N_data = 1, local_kl_weight=1., prior_kl_weight=1., log_magnitudes=True, **kwargs):
    """Train for a single step."""
    def loss_fn(params, rngs, batch_stats):
        """Inner function that computes loss and metrics."""
        (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                                    batch, month=month, mask=mask, rngs=rngs, mutable=['batch_stats'], **kwargs)
        recon_loss = neg_log_lik_loss(likelihood, batch, mask=mask)
        if batch.ndim == 3:
            recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]
        loss = recon_loss + prior_kl * prior_kl_weight + local_kl * local_kl_weight
        return loss, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, loss=loss, batch_stats=batch_stats['batch_stats'], aux=aux)

    # Create a function that runs loss_fn and gets the gradient with respect to the first input.
    grad_fn = value_and_grad(loss_fn, has_aux=True)

    # Run the transformed function and apply the resulting gradients to the state
    state, keys = state.update_rng()
    (_, outputs), grads = grad_fn(state.params, keys, state.batch_stats)
    
    return grads, outputs

@partial(jit, static_argnums=3)
def train_step_multisample(state, batch, n_samples, month=None, mask=None, N_data = 1, local_kl_weight=1., prior_kl_weight=1., log_magnitudes=False, **kwargs):
    """Train for a single step."""
    def loss_fn(params, rngs, batch_stats):
        """Inner function that computes loss and metrics."""
        (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                                    batch, month=month, mask=mask, n_samples=n_samples, rngs=rngs, mutable=['batch_stats'], **kwargs)
        recon_loss = -likelihood.log_prob(expand_dims(batch,0)).mean(axis=[0]).sum()
        if batch.ndim == 3:
            recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]
        loss = recon_loss + prior_kl * prior_kl_weight + local_kl * local_kl_weight
        return loss, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, loss=loss, batch_stats=batch_stats['batch_stats'])

    # Create a function that runs loss_fn and gets the gradient with respect to the first input.
    grad_fn = value_and_grad(loss_fn, has_aux=True)

    # Run the transformed function and apply the resulting gradients to the state
    state, keys = state.update_rng()
    (_, outputs), grads = grad_fn(state.params, keys, state.batch_stats)
    
    # Log gradient magnitudes
    # Log gradient magnitudes
    if log_magnitudes:
        pg = 0.
        if 'pgm' in grads:
            pg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['pgm'])[0] ** 2))
        eg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['encoder'])[0] ** 2))
        dg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['decoder'])[0] ** 2))
        props = jnp.mean(outputs['aux'][-1], axis=(0, 1))
        max_used_state = jnp.max(props)
        n_used_states = jnp.sum(props > (1. / (props.shape[0] * 2)))
        jax.debug.callback(lambda pgm_g, encoder_g, decoder_g, mus, nus: wandb_log_internal(dict(pgm_grad_norm=pgm_g, encoder_grad_norm=encoder_g, decoder_grad_norm=decoder_g, max_used_state=mus, n_used_states=nus)), pg, eg, dg, max_used_state, n_used_states) 
    
    state = state.apply_gradients(grads=grads, batch_stats=outputs['batch_stats'])
    metrics = dict(recon_loss=outputs['recon_loss'], prior_kl=outputs['prior_kl'], local_kl = outputs['local_kl'], loss=outputs['loss'])

    # Return the updated state and computed metrics
    return state, metrics

@jit
def eval_step(state, batch, month=None, mask_sst=None, mask=None, N_data=1, **kwargs):
    """Compute metrics for a single batch. Still returns updated state to account for consuming RNG state."""
    state, keys = state.update_rng()
    likelihood, prior_kl, local_kl, aux = state.apply_fn({'params': state.params, "batch_stats": state.batch_stats},
                                                      batch, month=month, eval_mode=True, mask=mask, rngs=keys, **kwargs)
    recon_loss = neg_log_lik_loss(likelihood, batch, mask=mask, mask_sst=mask_sst)
    if batch.ndim == 3: #Check this
        recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]

    loss = recon_loss + prior_kl + local_kl
    return state, loss, likelihood, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, likelihood=likelihood.log_prob(batch), loss=loss, aux=aux)

@partial(jit, static_argnums=3)
def eval_step_iwae(state, batch, n_iwae_samples, month=None, theta_rng=None, **kwargs):
    state, keys = state.update_rng()
    likelihood, prior_kl, local_kl, z = state.apply_fn({'params': state.params, 
                                                        "batch_stats": state.batch_stats},
                                                       batch, month=month, eval_mode=True, rngs=keys, 
                                                       n_iwae_samples=n_iwae_samples, 
                                                       theta_rng=theta_rng, **kwargs)
    recon_loss = -likelihood.log_prob(expand_dims(batch,1)).sum(axis=[-2,-1])
    if batch.ndim == 3:
        prior_kl = prior_kl.mean(0)
        loss = (local_kl + recon_loss).sum(0)
    else:
        loss = local_kl + recon_loss
    return state, prior_kl, loss # need to logsumexp and subtract log(n)

@partial(jit, static_argnums=(2, 4))  # n_forecast (pos 2) and mask (pos 4) are static
def eval_step_forecast(state, batch, n_forecast, month, mask=None, **kwargs):
    state, keys = state.update_rng()
    likelihood, prior_kl, local_kl, aux = state.apply_fn({'params': state.params, 
                                                          "batch_stats": state.batch_stats},
                                                         batch, month=month, eval_mode=True, mask=mask, rngs=keys, 
                                                         n_forecast=n_forecast, **kwargs)
    return state, None, likelihood, aux

def eval_step_tf_impute(state, batch, sample_rng, mask=None, N_data=1, **kwargs):
    mask = np.array(mask).astype(int)
    fill_mask = np.zeros_like(mask)
    fill_batch = np.array(batch)
    
    for step in range(batch.shape[-2]):
        fill_mask[..., :step] = 1
        fill_mask[..., step] = mask[..., step]
        state, loss, likelihood, aux = eval_step(state, fill_batch, mask=fill_mask, N_data=N_data, **kwargs)
        
        new_rng, sample_rng = jax.random.split(sample_rng)
        sample = np.array(likelihood.sample(seed=new_rng))
        fill_batch[..., step] = fill_batch * mask[..., step] + sample * (1 - mask[..., step])
    return fill_batch
        
### For separate optimization of network and pgm parameters.
class DualTrainState(PyTreeNode):
    step: int
    apply_fn: Callable = field(pytree_node=False) # Speify this field as a normal python class (not serializable pytree)
    params: dict[str, Any]
    batch_stats: dict[str, Any]
    rng_state: dict[str, Any]
    tx_net: optax.GradientTransformation = field(pytree_node=False)
    tx_pgm: optax.GradientTransformation = field(pytree_node=False)
    opt_state_net: optax.OptState
    opt_state_pgm: optax.OptState

    def update_rng(self):
        """Update the state of all RNGs stored by the trainstate and return a key for use in this sampling step"""
        new_key, sub_key = tree_map(lambda r: split(r)[0], self.rng_state), tree_map(lambda r: split(r)[1], self.rng_state)
        return self.replace(rng_state=new_key), sub_key

    def apply_gradients(self, *, grads, batch_stats, **kwargs):
        """Take a gradient descent step with the specified grads and encapsulated optimizer."""
        net_grads, pgm_grads = grads, grads.pop('pgm')
        net_params, pgm_params = self.params, self.params.pop('pgm')
        if self.tx_net is None:
            new_opt_state_net, new_params = None, net_params
        else:
            net_updates, new_opt_state_net = self.tx_net.update(net_grads, self.opt_state_net, net_params)
            new_params = optax.apply_updates(net_params, net_updates)
        if self.tx_pgm is None:
            new_opt_state_pgm, new_params_pgm = None, pgm_params
        else:
            pgm_updates, new_opt_state_pgm = self.tx_pgm.update(pgm_grads, self.opt_state_pgm, pgm_params)
            new_params_pgm = optax.apply_updates(pgm_params, pgm_updates)
        new_params.update({"pgm": new_params_pgm})
        return self.replace(
            step=self.step + 1,
            params=new_params,
            batch_stats=batch_stats,
            opt_state_net=new_opt_state_net,
            opt_state_pgm=new_opt_state_pgm,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, batch_stats, rng_state, tx_net, tx_pgm, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        net_params, pgm_params = params, params.pop('pgm')
        if tx_net is None:
            opt_state_net = None
        else:
            opt_state_net = tx_net.init(net_params)
        if tx_pgm is None:
            opt_state_net = None
        else:
            opt_state_pgm = tx_pgm.init(pgm_params)
        params.update({'pgm': pgm_params})
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            batch_stats=batch_stats,
            rng_state=rng_state,
            tx_net=tx_net,
            tx_pgm = tx_pgm,
            opt_state_net=opt_state_net,
            opt_state_pgm=opt_state_pgm,
            **kwargs,
        )

def create_dual_train_state(rng, learning_rate_net, learning_rate_pgm, model, 
                            input_shape, network_params=None, batch_stats=None, 
                            learning_alg_pgm='adam', learning_alg_net='adam', month=None):
    model = model()
    init_rng, init_model_rng, model_rng, dropout_rng = split(rng, 4)
    
    # Initialize with month parameter if provided
    if month is not None:
        init = model.init({'params': init_rng, 'sampler': init_model_rng, 
                           'dropout': dropout_rng}, ones(input_shape), month=month)
    else:
        init = model.init({'params': init_rng, 'sampler': init_model_rng, 
                           'dropout': dropout_rng}, ones(input_shape))
    
    learning_algs = dict(adam=optax.adam, sgd=optax.sgd)
    if learning_rate_net is None:
        tx_net = None
    else:
        tx_net = learning_algs[learning_alg_net](learning_rate_net) 
    if learning_rate_pgm is None:
        tx_pgm = None
    else:
        tx_pgm = learning_algs[learning_alg_pgm](learning_rate_pgm)
    if network_params is None:
        params = init['params']
    else:
        network_params.update({"pgm": init['params']['pgm']})
        if init['params'].get('vmp') is not None:
            network_params.update({"vmp": init['params']['vmp']})
        params = network_params
    if batch_stats is None:
        batch_stats = init['batch_stats'] if 'batch_stats' in init else {}
    return model, DualTrainState.create(
        apply_fn=model.apply, params=params, batch_stats=batch_stats,
        rng_state=dict(sampler=model_rng, dropout=dropout_rng), tx_net=tx_net, tx_pgm = tx_pgm)

def save_state(state, filename):
    if isinstance(state, TrainState):
        with open(filename, 'wb') as f:
            pickle.dump((state.params, state.batch_stats, state.rng_state, state.opt_state), f)
    elif isinstance(state, DualTrainState):
        with open(filename, 'wb') as f:
            pickle.dump((state.params, state.batch_stats, state.rng_state, state.opt_state_net, state.opt_state_pgm), f)
    else:
        print("Invalid state type")

def unfreeze_adam_state(adam_state):
    count, mu, nu = adam_state[0].__getnewargs__()
    return optax.ScaleByAdamState(count=count, mu=mu.unfreeze(), 
                                  nu=nu.unfreeze()), adam_state[1]

def load_state(state, filename):
    if isinstance(state, TrainState):
        with open(filename, 'rb') as f:
            params, batch_stats, rng_state, opt_state = pickle.load(f)
            if isinstance(params, FrozenDict):
                params = params.unfreeze()
                batch_stats = batch_stats.unfreeze()
                rng_state = rng_state.unfreeze()
                if isinstance(opt_state[0], optax.ScaleByAdamState):
                    opt_state = unfreeze_adam_state(opt_state)

            return state.replace(params=params,
                                 batch_stats=batch_stats,
                                 rng_state=rng_state,
                                 opt_state=opt_state)
    elif isinstance(state, DualTrainState):
        with open(filename, 'rb') as f:
            params, batch_stats, rng_state, opt_state_net, opt_state_pgm = pickle.load(f)
            if isinstance(params, FrozenDict):
                params = params.unfreeze()
                batch_stats = batch_stats.unfreeze()
                rng_state = rng_state.unfreeze()
                if isinstance(opt_state_net[0], optax.ScaleByAdamState):
                    opt_state_net = unfreeze_adam_state(opt_state_net)
                #if isinstance(opt_state_pgm[0], optax.ScaleByAdamState):
                #    opt_state_pgm = unfreeze_adam_state(opt_state_pgm)

            return state.replace(params=params,
                                 batch_stats=batch_stats,
                                 rng_state=rng_state,
                                 opt_state_net=opt_state_net)
                                 #opt_state_pgm=opt_state_pgm)
    else:
        print("Invalid state type")
        
def save_params(state, filename):
    with open(filename, 'wb') as f:
        pickle.dump((state.params, state.batch_stats), f)

def load_params(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

    
def bind_state(model, state):
    return model.bind({'params': state.params, 'batch_stats': state.batch_stats}, rngs=state.rng_state)

def simplify_params(params):
    simple_params = {}
    mat_to_vec = lambda x: jnp.diagonal(x, axis1=-1, axis2=-2)
    simple_params['Tau'] = mat_to_vec(params['Tau']).astype(jnp.float32)
    simple_params['loc'] = params['loc'][...,0].astype(jnp.float32)
    simple_params['A'] = mat_to_vec(params['X']).astype(jnp.float32)
    simple_params['b'] = params['X'][...,-1].astype(jnp.float32)
    simple_params['Lambda'] = mat_to_vec(params['Lambda']).astype(jnp.float32)
    simple_params['pi'] = params['pi'].astype(jnp.float32)
    simple_params['pi0'] = params['pi0'].astype(jnp.float32)
    return simple_params
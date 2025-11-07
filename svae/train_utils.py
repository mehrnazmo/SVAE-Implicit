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
import os
from jax import lax
import pandas as pd
import xarray as xr

from visualization_utils import visualize_forecast
from dataset.climate_dataset_hadisst_window import compute_nino34
from nino34_calculator import compute_nino34_and_oni
import gc

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
    # if x.ndim == 3:
        # print('ndim == 3')
        # print('x.shape: ', x.shape)
        # print('mask_sst.shape: ', mask_sst.shape)
        # pass
    if x.ndim == 4: 
        # print('ndim == 4')
        # print('x.shape: ', x.shape)
        # print('mask_sst.shape: ', mask_sst.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        mask_sst = mask_sst.reshape(mask_sst.shape[0], mask_sst.shape[1], -1)
    # else:
        # raise ValueError('x.ndim must be 3 or 4')
    # Calculate RMSE when mask_sst is provided
    rmse = None
    if mask_sst is not None:
        rmse = jnp.sqrt(jnp.mean((mask_sst * (x - dist.mean())) ** 2)) #average over bs,T,D
    if mask is not None:
        probs = where(expand_dims(mask, -1), -dist.log_prob(x), 0)
        return probs.sum(), rmse
    #dist.log_prob(x).shape: (5, 24, 2340)
    # l2_distance = jnp.sqrt(jnp.mean(
    #     jnp.sum(mask_sst * (x - dist.mean()) ** 2, -1), 
    #     axis=[0,1]))
    # jax.debug.print('l2_distance: {}', l2_distance)
    
    if mask_sst is not None:
        return (-dist.log_prob(x) * mask_sst).sum(), rmse
    else:
        return (-dist.log_prob(x)).sum(), rmse

@partial(jit, static_argnums=8, donate_argnums=(1,2,4))
def train_step(state, batch, month_encoding=None, mask=None, mask_sst=None, N_data = 1, local_kl_weight=1., 
               prior_kl_weight=1., log_magnitudes=False, **kwargs):    
    """Train for a single step."""
    (batch, mask_sst, month_encoding) = (batch[:, :24], mask_sst[:, :24], month_encoding[:, :24])
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
            batch, month_encoding=month_encoding, mask=mask, rngs=rngs, mutable=['batch_stats'], **kwargs)
        recon_loss, rmse = neg_log_lik_loss(likelihood, batch, mask=mask, mask_sst=mask_sst)
        if batch.ndim == 3:
            recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]
        loss = recon_loss + prior_kl * prior_kl_weight + local_kl * local_kl_weight
        
        # Create metrics dictionary with l2_distance if available
        metrics_dict = dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, 
                          loss=recon_loss + prior_kl + local_kl, batch_stats=batch_stats['batch_stats'], 
                          aux=aux)
        if rmse is not None:
            metrics_dict['rmse_train'] = rmse
            
        return loss, metrics_dict

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
    # Extract metrics from outputs, including RMSE if available
    metrics = dict(recon_loss=outputs['recon_loss'], prior_kl=outputs['prior_kl'], 
                   local_kl = outputs['local_kl'], 
                   loss=outputs['recon_loss'] + outputs['prior_kl'] + outputs['local_kl'], 
                   aux=outputs['aux'][-1],
                   rmse_train=outputs['rmse_train'])
    
    # Return the updated state and computed metrics
    return state, metrics

def make_train_step_with_transform(trans):
    """Returns train_step function"""
    
    # Update static_argnums to include the index of log_magnitudes
    # Parameters: state, batch, month, mask, mask_sst, N_data, local_kl_weight, prior_kl_weight, log_magnitudes
    # Indices:    0,     1,     2,    3,        4,      5,              6,              7,              8
    @partial(jit, static_argnums=(8,), donate_argnums=(1,2,4))  # log_magnitudes is at index 8 now
    def train_step_with_transform(
        state, batch, month_encoding=None, mask=None, mask_sst=None, N_data=1, local_kl_weight=1., 
        prior_kl_weight=1., log_magnitudes=False, **kwargs):
        """Train for a single step."""
        (batch, mask_sst, month_encoding) = (batch[:, :24], mask_sst[:, :24], month_encoding[:, :24])
        def loss_fn(params, rngs, batch_stats):
            """Inner function that computes loss and metrics."""
            #model.apply
            (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                trans(batch), month_encoding=trans(month_encoding), mask=mask, rngs=rngs, mutable=['batch_stats'], **kwargs)
            recon_loss, rmse = neg_log_lik_loss(likelihood, batch, mask=mask, mask_sst=mask_sst)
            if batch.ndim == 3:
                recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]
            loss = recon_loss + prior_kl * prior_kl_weight + local_kl * local_kl_weight
            
            # Create metrics dictionary with l2_distance if available
            metrics_dict = dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, loss=recon_loss + prior_kl + local_kl, batch_stats=batch_stats['batch_stats'], aux=aux)
            if rmse is not None:
                metrics_dict['rmse_train'] = rmse
                
            return loss, metrics_dict

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
        
        # Extract metrics from outputs, including RMSE if available
        metrics = dict(recon_loss=outputs['recon_loss'], prior_kl=outputs['prior_kl'], local_kl = outputs['local_kl'], 
                       loss=outputs['recon_loss'] + outputs['prior_kl'] + outputs['local_kl'], aux=outputs['aux'][-1],
                       rmse_train=outputs['rmse_train'])
        
        # Return the updated state and computed metrics
        return state, metrics
    return train_step_with_transform

def make_eval_step_with_transform(trans):
    @jit
    def eval_step_with_transform(state, batch, month_encoding=None, mask_sst=None, mask=None, N_data=1, **kwargs):
        """Compute metrics for a single batch. Still returns updated state to account for consuming RNG state."""
        (batch, mask_sst, month_encoding) = (batch[:, :24], mask_sst[:, :24], month_encoding[:, :24])
        state, keys = state.update_rng()
        likelihood, prior_kl, local_kl, aux = state.apply_fn({'params': state.params, "batch_stats": state.batch_stats},
                                                          trans(batch), month_encoding=trans(month_encoding), eval_mode=True, mask=mask, rngs=keys, **kwargs)
        recon_loss, rmse = neg_log_lik_loss(likelihood, batch, mask=mask, mask_sst=mask_sst)
        if batch.ndim == 3: #Check this
            recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]

        loss = recon_loss + prior_kl + local_kl
        
        # Create metrics dictionary with l2_distance if available
        metrics_dict = dict(
            recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, 
            likelihood=likelihood.log_prob(batch.reshape(batch.shape[0], batch.shape[1], -1)), 
            loss=loss, aux=aux, rmse_test=rmse)
            
        return state, loss, likelihood, metrics_dict
    return eval_step_with_transform

@partial(jit, static_argnums=7)
def grad_step(state, batch, month_encoding=None, mask=None, N_data = 1, local_kl_weight=1., prior_kl_weight=1., log_magnitudes=True, **kwargs):
    """Train for a single step."""
    def loss_fn(params, rngs, batch_stats):
        """Inner function that computes loss and metrics."""
        (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                                    batch, month_encoding=month_encoding, mask=mask, rngs=rngs, mutable=['batch_stats'], **kwargs)
        recon_loss, _ = neg_log_lik_loss(likelihood, batch, mask=mask)
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
def train_step_multisample(state, batch, n_samples, month_encoding=None, mask=None, N_data = 1, local_kl_weight=1., prior_kl_weight=1., log_magnitudes=False, **kwargs):
    """Train for a single step."""
    def loss_fn(params, rngs, batch_stats):
        """Inner function that computes loss and metrics."""
        (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                                    batch, month_encoding=month_encoding, mask=mask, n_samples=n_samples, rngs=rngs, mutable=['batch_stats'], **kwargs)
        recon_loss = -likelihood.log_prob(expand_dims(batch.reshape(batch.shape[0], batch.shape[1], -1),0)).mean(axis=[0]).sum()
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
def eval_step(state, batch, month_encoding=None, mask_sst=None, mask=None, N_data=1, **kwargs):
    """Compute metrics for a single batch. Still returns updated state to account for consuming RNG state."""
    
    (batch, mask_sst, month_encoding) = (batch[:, :24], mask_sst[:, :24], month_encoding[:, :24])
    
    state, keys = state.update_rng()
    likelihood, prior_kl, local_kl, aux = state.apply_fn({'params': state.params, "batch_stats": state.batch_stats},
                                                      batch, month_encoding=month_encoding, eval_mode=True, mask=mask, rngs=keys, **kwargs)
    recon_loss, rmse = neg_log_lik_loss(likelihood, batch, mask=mask, mask_sst=mask_sst)
    if batch.ndim == 3: #Check this
        recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]

    loss = recon_loss + prior_kl + local_kl
    
    # Create metrics dictionary with l2_distance if available
    metrics_dict = dict(
        recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, 
        likelihood=likelihood.log_prob(batch.reshape(batch.shape[0], batch.shape[1], -1)), 
        loss=loss, aux=aux, rmse_test=rmse)
        
    return state, loss, likelihood, metrics_dict

@partial(jit, static_argnums=3)
def eval_step_iwae(state, batch, n_iwae_samples, month_encoding=None, theta_rng=None, **kwargs):
    state, keys = state.update_rng()
    likelihood, prior_kl, local_kl, z = state.apply_fn({'params': state.params, 
                                                        "batch_stats": state.batch_stats}, 
                                                       batch, month_encoding=month_encoding, eval_mode=True, rngs=keys, 
                                                       n_iwae_samples=n_iwae_samples, 
                                                       theta_rng=theta_rng, **kwargs)
    recon_loss = -likelihood.log_prob(expand_dims(batch.reshape(batch.shape[0], batch.shape[1], -1),1)).sum(axis=[-2,-1])
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
                            learning_alg_pgm='adam', learning_alg_net='adam', month_encoding=None):
    model = model()
    init_rng, init_model_rng, model_rng, dropout_rng = split(rng, 4)
    
    # Initialize with month parameter if provided
    if month_encoding is not None:
        init = model.init({'params': init_rng, 'sampler': init_model_rng, 
                           'dropout': dropout_rng}, ones(input_shape), month_encoding=month_encoding)
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


def correct_sst_range(sst, sst_min, sst_max):
    return sst * (sst_max - sst_min) + sst_min
    
def sst_video_to_nino34(data, time_stamps, climatology, lat_range=(-50, 70), lon_range=(0, 360)):
    # data: (T+n_forecast, h, w)
    t, h, w = data.shape
    da = xr.DataArray(
    np.array(data),
    dims=["time", "lat", "lon"],                 # dimension names
    coords={
        "time": np.array(time_stamps),
        "lat": np.linspace(lat_range[0]+0.5,lat_range[1]-0.5,h), #(-50, 70)
        "lon": np.linspace(lon_range[0]+0.5,lon_range[1]-0.5,w), #(0, 360)
    },
    attrs={"units": "degC", "long_name": "sea surface temperature"}
    ).to_dataset(name="SST") 
    nino34, oni, climatology = compute_nino34(da, climatology=np.array(climatology)) #(t,), (t,), (12, 10, 50)
    # da = da.assign_coords({"nino34": nino34, "oni": oni})
    return nino34, oni

def save_nino34_oni(nino34_preds_epoch, oni_preds_epoch, nino34_trues_epoch,
                    oni_trues_epoch, time_stamps_epoch, epoch, save_dir):
    nino34_preds_epoch = np.concatenate(nino34_preds_epoch, axis=0)
    oni_preds_epoch = np.concatenate(oni_preds_epoch, axis=0)
    nino34_trues_epoch = np.concatenate(nino34_trues_epoch, axis=0)
    oni_trues_epoch = np.concatenate(oni_trues_epoch, axis=0)
    time_stamps_epoch = np.concatenate(time_stamps_epoch, axis=0)

    nino34_corr = np.zeros(9)
    oni_corr = np.zeros(9)
    nino34_preds_epoch_slice = nino34_preds_epoch[:,24:33]
    oni_preds_epoch_slice = oni_preds_epoch[:,24:33]
    nino34_trues_epoch_slice = nino34_trues_epoch[:,24:33]
    oni_trues_epoch_slice = oni_trues_epoch[:,24:33]
    
    for m in range(9):
        # Check for NaN values
        pred_nan = np.isnan(nino34_preds_epoch_slice[:,m]).any()
        true_nan = np.isnan(nino34_trues_epoch_slice[:,m]).any()
        
        # Check for zero variance
        pred_var = np.var(nino34_preds_epoch_slice[:,m])
        true_var = np.var(nino34_trues_epoch_slice[:,m])
        
        if pred_nan or true_nan:
            print(f"Month {m}: NaN detected in predictions={pred_nan}, trues={true_nan}")
        elif pred_var == 0 or true_var == 0:
            print(f"Month {m}: Zero variance - pred_var={pred_var}, true_var={true_var}")
        
        
        nino34_corr[m] = np.corrcoef(nino34_preds_epoch_slice[:,m], nino34_trues_epoch_slice[:,m])[0, 1]
        oni_corr[m] = np.corrcoef(oni_preds_epoch_slice[:,m], oni_trues_epoch_slice[:,m])[0, 1]
    print('-'*100)
    print('nino34_corr:', nino34_corr)
    print('oni_corr:', oni_corr)
    print('-'*100)
    
    np.save(f'{save_dir}/enso/nino34_preds_epoch_{epoch}.npy', nino34_preds_epoch)
    np.save(f'{save_dir}/enso/oni_preds_epoch_{epoch}.npy', oni_preds_epoch)
    np.save(f'{save_dir}/enso/nino34_trues_epoch_{epoch}.npy', nino34_trues_epoch)
    np.save(f'{save_dir}/enso/oni_trues_epoch_{epoch}.npy', oni_trues_epoch)
    np.save(f'{save_dir}/enso/time_stamps_epoch_{epoch}.npy', time_stamps_epoch)
    
    print('Saved Nino3.4 and ONI predictions and trues for epoch: ', epoch)
    return

# @partial(jit, static_argnums=(3, 4))
# @jax.jit(static_argnums=(3, 4), device=jax.devices('cpu')[0])  # Force CPU
def _forecast_flat(state, batch, month_encoding, n_forecast, n_samples, **kwargs):
    state, keys = state.update_rng()
    likelihood, z, recon_single = state.apply_fn(
        {'params': state.params, "batch_stats": state.batch_stats}, 
        batch, 
        month_encoding = month_encoding, 
        eval_mode=True, 
        mask=None, 
        rngs=keys,
        n_forecast = n_forecast, 
        n_samples = n_samples,
        **kwargs)
    gc.collect()
    jax.clear_caches()
    return state, likelihood, z, recon_single

def get_timestamps(years, months):
    '''
    years: (bs, t_data)
    months: (bs, t_data)
    '''
    bs, t_data = years.shape
    # Convert JAX arrays to NumPy arrays for string operations
    years_1d = np.array(years.flatten())
    months_1d = np.array(months.flatten())
    
    dates_1d = pd.to_datetime(
        years_1d.astype(str) + '-' + months_1d.astype(str), format="%Y-%m"
    ) + pd.Timedelta(days=14)
    time_stamps_all = np.array(dates_1d.values.reshape(bs, t_data))
    return time_stamps_all

def get_timestamps_diff(time_stamps_all_idx, t_forecast):
    extra_range = t_forecast - len(time_stamps_all_idx)
    diff = (pd.date_range(start=time_stamps_all_idx[-1], periods=extra_range + 1, freq='MS')[1:] 
            + pd.Timedelta(days=14))
    return diff


def get_nino_from_sst(data, lat_range, lon_range):
    #data: np.array (5, 36, 40, 200)
    #lat_range:  (-20, 20)
    #lon_range:  (100, 300)
    nino_lat_range=(-5, 5)
    nino_lon_range=(190, 240)
    start_lat = nino_lat_range[0]-lat_range[0]
    start_lon = nino_lon_range[0]-lon_range[0]
    nino_box = data[:,
                    :,
                    start_lat:start_lat+nino_lat_range[1]-nino_lat_range[0], 
                    start_lon:start_lon+nino_lon_range[1]-nino_lon_range[0]]
    return nino_box


def forecast_flat(model_state, data_orig, month_orig, year_orig, 
             month_encoding_orig=None, mask_orig=None, climatology=None,
             sst_min=None, sst_max=None,
             n_forecast=10, n_samples=1, LDS_T=24, save_path=None, 
             h=26, w=90, lat_range=(-50, 70), lon_range=(0, 360), 
             start_month=0, epoch=0, batch_num=0):
    base_rng = jax.random.PRNGKey(0)
    # assert data_orig.ndim == 4, "data_orig must be 4D"
    if data_orig.ndim == 4:
        (bs, t_data, h, w) = data_orig.shape 
    elif data_orig.ndim == 3:
        (bs, t_data, input_D) = data_orig.shape 
        h = lat_range[1]-lat_range[0]
        w = lon_range[1]-lon_range[0]
    t_forecast = n_forecast + LDS_T
    #(5, 36, 8000), (5, 100, 36, 50)
    _, forecast_flat, z, recon_single = _forecast_flat(
        model_state, 
        data_orig[:, :LDS_T], 
        month_encoding_orig[:, :LDS_T, :], 
        n_forecast, 
        n_samples)
    
    # Clear JAX caches to free memory
    jax.clear_caches()
    gc.collect()
    
    #mask_orig: (bs, 36, input_D) = (5, 36, 40, 200)
    mask_forecast = jnp.expand_dims(mask_orig.reshape(bs, -1, h, w)[0, 0], axis=(0,1)) #(1, 1, h, w)
    # mask_forecast: (1, 1, 40, 200)
    forecast_video = np.array(forecast_flat.reshape(bs, t_forecast, h, w) * mask_forecast)
    
    time_stamps = get_timestamps(year_orig, month_orig)
    
    # #(5, 36, 40, 200), (5, 36)
    # nino34_pred, oni_pred = vmap(sst_video_to_nino34, in_axes=(0, 0, None, None, None))(
    #     forecast_video_np, 
    #     time_stamps_np, climatology_np, lat_range, lon_range)
    # nino34_true, oni_true = vmap(sst_video_to_nino34, in_axes=(0, 0, None, None, None))(
    #     correct_sst_range(data_orig_np, sst_min, sst_max), 
    #     time_stamps_np, climatology_np, lat_range, lon_range)
            
            
    # nino34_preds, oni_preds = compute_nino34_and_oni(
    #     np.flip(correct_sst_range(forecast_video, sst_min, sst_max), axis=2), 
    #     lat_range, lon_range, start_month, climatology=climatology)
    # nino34_trues, oni_trues = compute_nino34_and_oni(
    #     np.flip(correct_sst_range(data_orig[:, :t_forecast, :, :], sst_min, sst_max), axis=2), 
    #     lat_range, lon_range, start_month, climatology=climatology)
    nino34_preds = np.random.rand(bs, t_forecast)
    nino34_trues = np.random.rand(bs, t_forecast)
    oni_preds = np.random.rand(bs, t_forecast)
    oni_trues = np.random.rand(bs, t_forecast)
    
    for sample_idx in range(bs):
        if epoch % 1000 == 0 and batch_num == 0: 
            print('recon_single.shape: ', recon_single.shape)
            time_stamps_extended = np.empty((t_forecast), dtype='datetime64[ns]')
            time_stamps_extended[:t_data] = time_stamps[sample_idx]
            time_stamps_extended[t_data:] = get_timestamps_diff(time_stamps[sample_idx], t_forecast)
            visualize_forecast(
                np.array(recon_single[sample_idx]).reshape(-1, h, w), 
                np.array(data_orig[sample_idx, :t_data]).reshape(-1, h, w), 
                np.array(mask_forecast[0,0]), 
                time_stamps_extended, 
                save_path, 
                np.array(z[sample_idx]), 
                sample_idx)
        
    return (nino34_preds, oni_preds, 
            nino34_trues, oni_trues, 
            time_stamps)

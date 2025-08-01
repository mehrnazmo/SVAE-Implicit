import jax
import jax.numpy as jnp
from jax.random import split

import flax.linen as nn
from typing import Any
from networks.encoders import Encoder
from networks.decoders import SigmaDecoder

from inference.MP_Inference import lds_inference_and_sample_simple, lds_kl_simple
from utils import mask_potentials_simple

class PGM_LDS_Simple(nn.Module):
    latent_D: int

    @nn.compact
    def expected_params(self, T):
        ### Initializations and converting from unconstrained space

        # Initial state, p(z_0)
        mu = self.param("loc", lambda rng: jnp.zeros(self.latent_D, dtype=jnp.float32))
        tau_p = self.param("Tau", lambda rng: jnp.ones(self.latent_D, dtype=jnp.float32))
        tau = tau_p ** 2 + 1e-6
        init_potentials = tau, tau * mu

        # Transition distribution p(z_t|z_{t-1})
        lam_p = self.param("Lambda", lambda rng: jnp.ones((12, self.latent_D), dtype=jnp.float32))
        lam = lam_p ** 2 + 1e-6
        ## A[i]: transition from month i to month i+1. e.g. A[0] is transition from month 0 (Jan) to month 1 (Feb).
        A = self.param("A", lambda rng: jnp.ones((12, self.latent_D), dtype=jnp.float32)) 
        b = self.param("b", lambda rng: jnp.zeros((12, self.latent_D), dtype=jnp.float32))
        transition_potentials = (A * lam * b, A **2 * lam, A * lam, lam, lam * b)
        # transition_potentials = jax.tree_map(lambda x: jnp.tile(x, (T-1, 1)), transition_potentials)
        transition_potentials = jax.tree_map(lambda x: jnp.concatenate([x, x[:-1]], axis=0), transition_potentials)
        
        
        # Compute the log partition function of p(z)
        init_norm =      1/2 * (jnp.log(2 * jnp.pi)*self.latent_D + (tau * mu**2).sum(-1) - jnp.log(tau).sum(-1))
        trans_norm = (T-1)/2 * (jnp.log(2 * jnp.pi)*self.latent_D + (lam * b**2).sum(-1)  - jnp.log(lam).sum(-1))
        p_logZ = init_norm + trans_norm

        return (init_potentials, transition_potentials), p_logZ, (A, b, lam_p)

    def __call__(self, recog_potentials, key, temp=1., n_forecast = 0):

        if n_forecast > 0:
            key, forecast_rng = split(key)
            
        # get global parameters
        pgm_potentials, p_logZ, (A, b, lam_p) = self.expected_params(recog_potentials[0].shape[0])

        # Run belief propagation
        expected_stats, q_logZ, z = lds_inference_and_sample_simple(recog_potentials, *pgm_potentials, key, temp)

        # compute kl divergence between q and p
        kl = lds_kl_simple(recog_potentials, expected_stats, p_logZ, q_logZ)
        
        # forecast
        if n_forecast > 0:
            # Use direct access to A and b (more stable)
            # print(A.shape)
            # print(b.shape)
            # print(lam_p.shape)
            
            # A_forecast = A[:-1]  # Direct access to the last time step
            # b_forecast = b[:-1]
            # lam_forecast = lam_p[:-1]
            
            # Extract A and b from the transition potentials
            # A_lam = A * lam, so A = A_lam / lam
            # lam_b = lam * b, so b = lam_b / lam
            # A = A_lam / (lam + 1e-8)  
            # b = lam_b / (lam + 1e-8)
            # var = 1.0 / (lam + 1e-8)
            
            # A = A_lam[-1] / (lam[-1] + 1e-8) # Add small epsilon to avoid division by zero
            # b = lam_b[-1] / (lam[-1] + 1e-8)
            # Extract variance from lam (precision = lam, so variance = 1/lam)
            std = 1.0 / (lam_p + 1e-8)
            
            
            # Simple forecasting: z_{t+1} = A * z_t + b + noise
            def forecast_step(carry, step_data):
                z_prev, step_idx = carry, step_data
                # Sample noise: epsilon ~ N(0, var)
                epsilon = jax.random.normal(forecast_rng, z_prev.shape) * std[step_idx]
                # Use A indexed by step number
                A_step = A[step_idx]
                b_step = b[step_idx]
                z_next = jnp.dot(A_step, z_prev) + b_step + epsilon
                return z_next, z_next
            
            # Create step indices for indexing A
            # Each time series always starts from month 10 and ends at month 9 of two years later.
            # Example: first index forecasts month 10 from month 9, so needs transition from month 9 to month 10.
            step_indices = (jnp.arange(n_forecast) + 9 ) % 12
            # Use scan to generate n_forecast steps
            _, forecasted_z = jax.lax.scan(forecast_step, z[-1], step_indices, length=n_forecast)
            z = jnp.concatenate([z, forecasted_z], axis=-2)
            
        return z, kl, (expected_stats, recog_potentials, pgm_potentials)

class SVAE_LDS_Simple(nn.Module):
    latent_D: int
    input_D: int
    encoder_cls: Any = Encoder
    decoder_cls: Any = SigmaDecoder

    def setup(self):
        self.encoder = self.encoder_cls(self.latent_D, name="encoder")
        self.pgm = PGM_LDS_Simple(self.latent_D, name="pgm")
        self.decoder = self.decoder_cls(self.input_D, name="decoder")

    def __call__(self, x, month=None, eval_mode=False, mask=None, temp=1., n_forecast = 0):

        # mask input for encoder.
        # not strictly necessary because we mask the output, but helps prevent leakage
        x_input = jnp.where(jnp.expand_dims(mask, -1), x, jnp.zeros_like(x)) if mask is not None else x
        
        # encode
        recog_potentials = self.encoder(x_input, eval_mode = eval_mode, mask=mask)
        
        # our implementation outputs a diagonal (B, T, D, D) precision and (B, T, D, 1) precision-mean
        # we want to turn those both into (B, T, D)
        recog_potentials = jnp.diagonal(recog_potentials[0], axis1=-2, axis2=-1),  recog_potentials[1][...,0]

        # cover garbage recognition potentials at masked time steps
        if mask is not None:
            recog_potentials = mask_potentials_simple(recog_potentials, mask)

        # generate rng key
        key = split(self.make_rng('sampler'),x.shape[0])

        # perform LDS inference
        with jax.default_matmul_precision('float32'):
            pgm_fun = jax.vmap(self.pgm, in_axes=[0,0,None,None])
            z, kl, aux = pgm_fun(recog_potentials, key, temp, n_forecast)
            kl = kl.sum()

        # decode
        # Create month information for both training and evaluation
        if month is not None:
            # Use provided month information
            decoder_month = month
        else:
            # Create month information based on sequence length
            sequence_length = z.shape[1]
            # Create month indices: 0-11 for each time step
            month_indices = jnp.arange(sequence_length) % 12
            # Create one-hot encoding or other month representation
            decoder_month = jax.nn.one_hot(month_indices, 12)  # Shape: (T, 12)
            # Expand to batch dimension
            decoder_month = jnp.expand_dims(decoder_month, 0)  # Shape: (1, T, 12)
            # Tile to match batch size
            decoder_month = jnp.tile(decoder_month, (z.shape[0], 1, 1))  # Shape: (B, T, 12)
        
        likelihood = self.decoder(z.astype(jnp.float32), month=decoder_month, eval_mode=eval_mode)
        return likelihood, jnp.zeros([], dtype=jnp.float32), kl, (z, aux)

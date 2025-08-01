import jax
import jax.numpy as jnp
import flax.linen as nn

from jax.random import split
from jax.scipy.special import logsumexp

from typing import Callable, Any, Dict
from dataclasses import field
from utils import mask_potentials_simple

from inference.SLDS_Simple_Inference import slds_inference_simple_implicit, sample_and_kl_simple
from networks.encoders import Encoder
from networks.decoders import SigmaDecoder

# TODO: multiple possible initialization states, different mappings for parameters
class PGM_SLDS_Simple(nn.Module):
    latent_D: int
    K: int
    inference_fun: Callable

    @nn.compact
    def expected_params(self):
        ### Initializations and converting from unconstrained space.

        # Initial state, p(z_0)
        mu = self.param("loc", lambda rng: jnp.zeros(self.latent_D, dtype=jnp.float32))
        tau_p = self.param("Tau", lambda rng: jnp.ones(self.latent_D, dtype=jnp.float32))
        mu, tau_p = jnp.tile(mu, (self.K, 1)), jnp.tile(tau_p, (self.K, 1))
        tau = tau_p ** 2 + 1e-6
        init_norm = -(tau/2 * mu**2).sum(-1) + jnp.log(tau).sum(-1)/2
        init_potentials = tau, tau * mu, init_norm

        # Transition distribution p(z_t|z_{t-1})
        lam_p = self.param("Lambda", lambda rng: jnp.ones((self.K, self.latent_D), dtype=jnp.float32))
        lam = lam_p ** 2 + 1e-6
        A = self.param("A", lambda rng: jnp.ones((self.K, self.latent_D), dtype=jnp.float32))
        b = self.param("b", lambda rng: jnp.ones((self.K, self.latent_D), dtype=jnp.float32))
        trans_norm = -(lam/2 * b**2).sum(-1) + jnp.log(lam).sum(-1)/2
        transition_potentials = (-A * lam * b, -1/2 * A **2 * lam, A * lam, -1/2 * lam, lam * b, jnp.expand_dims(trans_norm, -1))

        # Initial state and transition distributions for p(k)
        pi0 = self.param("pi0", lambda rng: jnp.zeros(self.K, dtype=jnp.float32))
        pi = self.param("pi", lambda rng: jnp.log(jnp.ones((self.K, self.K), dtype=jnp.float32) * 1/(2 * self.K) + jnp.identity(self.K, dtype=jnp.float32)/2))
        def normalize(ps):
            return ps - logsumexp(ps, -1, keepdims=True)
        init_lps = jnp.expand_dims(normalize(pi0), -1)
        trans_lps = normalize(pi)

        return init_potentials, transition_potentials, init_lps, trans_lps

    def __call__(self, recog_potentials, masked_potentials, key, initializer, temp=1.):

        # Get expectations of q(theta)
        pgm_potentials = self.expected_params()
        cat_expected_stats = self.inference_fun(recog_potentials, *pgm_potentials, initializer)
        
        # all 3 of these functions need to change
        z, local_kl, gaus_expected_stats = sample_and_kl_simple(masked_potentials, *pgm_potentials, 
                                                                cat_expected_stats, key, temp=temp)

        return z, (gaus_expected_stats, cat_expected_stats), local_kl

class SVAE_SLDS_Simple(nn.Module):
    latent_D: int
    K: int
    input_D: int = 96
    encoder_cls: Any = Encoder
    decoder_cls: Any = SigmaDecoder
    inference_fun: Callable = slds_inference_simple_implicit
    pgm_hyperparameters: Dict = field(default_factory=dict)

    def setup(self):
        self.encoder = self.encoder_cls(self.latent_D, name="encoder")
        self.pgm = PGM_SLDS_Simple(self.latent_D, self.K, self.inference_fun, name="pgm")
        self.decoder = self.decoder_cls(self.input_D, name="decoder")

    def __call__(self, x, eval_mode=False, mask=None, initializer = None, ss_mask = None, temp=1.):

        if not (mask is None):
            unscaled_mask = mask
            mask = jnp.where(mask > 0, jnp.ones_like(mask), jnp.zeros_like(mask))

        x_input = jnp.where(jnp.expand_dims(mask, -1), x, jnp.zeros_like(x)) if mask is not None else x
        recog_potentials = self.encoder(x_input, eval_mode = eval_mode, mask=mask)
        recog_potentials = jnp.diagonal(recog_potentials[0], axis1=-2, axis2=-1),  recog_potentials[1][...,0]

        if mask is not None:
            downsampling_factor = x.shape[1]//recog_potentials[0].shape[1]
            if downsampling_factor > 1:
                mask = mask.reshape((x.shape[0],-1,downsampling_factor)).any(-1)
            recog_potentials = mask_potentials_simple(recog_potentials, mask)


        if ss_mask is None:
            masked_potentials = recog_potentials
        else:
            downsampling_factor = x.shape[1]//recog_potentials[0].shape[1]
            if downsampling_factor > 1:
                ss_mask = ss_mask.reshape((x.shape[0],-1,downsampling_factor)).any(-1)
            masked_potentials = mask_potentials_simple(recog_potentials, ss_mask)

        key = split(self.make_rng('sampler'),x.shape[0])
        if initializer is None:
            keys = jax.vmap(split)(key)
            key, initializer = keys[:,0], keys[:,1]

        with jax.default_matmul_precision('float32'):
            pgm_fun = jax.vmap(self.pgm, in_axes=[0,0,0,0,None])
            z, aux, local_kl = pgm_fun(recog_potentials, masked_potentials, key, initializer, temp)
        local_kl = local_kl.sum()
        likelihood = self.decoder(z.astype(jnp.float32), eval_mode=eval_mode)

        return likelihood, jnp.zeros([], dtype=jnp.float32), local_kl, (z,) + aux

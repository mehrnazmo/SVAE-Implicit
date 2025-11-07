import flax.linen as nn
from flax.linen import Module, Dense, ConvLocal, Conv, BatchNorm, leaky_relu, softplus, compact
from jax.numpy import expand_dims, diag, zeros_like, ones_like
from jax import vmap
from typing import Callable, Optional, Any, Sequence, Tuple
from distributions import normal
from functools import partial
import jax.numpy as jnp
from jax.nn import sigmoid
from tensorflow_probability.substrates.jax import distributions as tfd
from .sequence import SimpleLSTM, SimpleBiLSTM, ReverseLSTM
from .cnn import LocallyConnectedBlock

ModuleDef = Any

class DenseBlock(Module):
    n_features: int
    activation: Callable = leaky_relu
    norm_cls: Optional[ModuleDef] = nn.BatchNorm
    dtype: Any = jnp.float32
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x):
        x = Dense(self.n_features, dtype=self.dtype)(x)
        x = self.activation(x)
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
        return x

class ResDenseSkipConnection(nn.Module):
    out_shape: Any
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        if x.shape != self.out_shape:
            x = Dense(self.out_shape[-1], dtype=self.dtype)(x)
        return x

class DenseNet(nn.Module):
    n_outputs: int
    block_cls: ModuleDef = DenseBlock
    input_block_cls: ModuleDef = LocallyConnectedBlock
    resnet: bool = False
    stage_sizes: Sequence[int] = (4,)
    hidden_sizes: Sequence[int] = (100,)
    dtype: Any = jnp.float32
    activation: Callable = leaky_relu
    norm_cls: Optional[ModuleDef] = nn.BatchNorm
    eval_mode: bool = False
    lstm_layer: Any = -1
    lstm_cls: ModuleDef = ReverseLSTM
    lstm_hidden_size: int = 64
    scale_input: float = 1.
    scale_output: float = 1.
    last_layer_sigmoid: bool = False 
    lc_channels: int = 2
    #if False, the input layer is a locally connected layer, otherwise it is a dense layer
    flatten_input: bool = False
    
    @nn.compact
    def __call__(self, x, mask=None):
        # print('DenseNet')
        # print('initial input shape: ', x.shape)
        x = x / self.scale_input
        stage_sizes, hidden_sizes = self.stage_sizes, self.hidden_sizes
        block_cls = partial(self.block_cls, eval_mode=self.eval_mode, dtype=self.dtype, activation=self.activation)
        lstm_layer = [self.lstm_layer] if type(self.lstm_layer) is int else self.lstm_layer
        layers = []
        if not self.flatten_input:
            input_block_cls = partial(
                self.input_block_cls, eval_mode=self.eval_mode, 
                dtype=self.dtype, activation=self.activation)
            x = input_block_cls(
                n_features=self.lc_channels, norm_cls=self.norm_cls
                )(jnp.expand_dims(x, -1)).reshape(x.shape[0], x.shape[1], -1)
            
        for i, (hsize, n_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
            if self.lstm_cls and self.lstm_hidden_size > 0 and i in lstm_layer:
                x = self.lstm_cls(self.lstm_hidden_size)(x, mask=mask)
            
            x_res = x
            if n_blocks < 0: # indicating a layer which should factorize across angles
                x = x.reshape(x.shape[:-1] + (-1,-n_blocks))
                x = block_cls(n_features=hsize, norm_cls=self.norm_cls,)(x)
                x = x.reshape(x.shape[:-2] + (-1,))
            else:
                for b in range(n_blocks - 1):
                    x = block_cls(n_features=hsize, norm_cls=self.norm_cls,)(x)
                x = block_cls(n_features=hsize, norm_cls=(not self.resnet) and self.norm_cls,)(x)
            
            if self.resnet and i > 0:
                # Handle shape mismatch in residual connections
                if x.shape != x_res.shape:
                    # Project x_res to match x's shape using a dense layer
                    x_res = Dense(x.shape[-1], dtype=self.dtype)(x_res)
                x = x + x_res
                if self.norm_cls:
                    x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
            
        if self.lstm_cls and self.lstm_hidden_size > 0 and len(stage_sizes) == lstm_layer[-1]:
            x = self.lstm_cls(self.lstm_hidden_size)(x, mask=mask)

        x = nn.Dense(self.n_outputs, dtype=self.dtype)(x)
        if self.last_layer_sigmoid:
            x = sigmoid(x)
        # print('after last layer: ', x.shape)
        return x * self.scale_output

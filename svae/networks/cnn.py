import flax.linen as nn
from flax.linen import Module, Dense, ConvLocal, Conv, BatchNorm, leaky_relu, softplus, compact
import jax
from jax.numpy import expand_dims, diag, zeros_like, ones_like
from jax import vmap
from typing import Callable, Optional, Any, Sequence, Tuple
from distributions import normal
from functools import partial
import jax.numpy as jnp
from jax.nn import sigmoid
from tensorflow_probability.substrates.jax import distributions as tfd
from .sequence import SimpleLSTM, SimpleBiLSTM, ReverseLSTM

ModuleDef = Any

    
class LocallyConnectedBlock(Module):
    n_features: int #n_channels
    kernel_size: Tuple[int, int] = (5, 5) #padding = 'same'
    stride: Tuple[int, int] = (2, 2)
    activation: Callable = leaky_relu
    #For more stability consider GroupNorm or LayerNorm
    norm_cls: Optional[ModuleDef] = nn.BatchNorm 
    dtype: Any = jnp.float32
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x):
        x = ConvLocal(self.n_features, kernel_size=self.kernel_size, strides=self.stride, dtype=self.dtype)(x)
        x = self.activation(x)
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        return x
    
# Replace LocallyConnectedBlock with a standard Conv stem
class ConvStem(nn.Module):
    n_features: int
    kernel_size: Tuple[int, int] = (3, 3)
    stride: Tuple[int, int] = (1, 1)
    activation: Callable = nn.gelu
    norm_cls: Optional[ModuleDef] = nn.GroupNorm 
    dtype: Any = jnp.float32
    eval_mode: bool = False
    @nn.compact
    def __call__(self, x):
        # x: (B, T, H, W, 1)
        x = nn.Conv(self.n_features, self.kernel_size, self.stride, padding='SAME', dtype=self.dtype)(x)
        x = self.activation(x)
        # Handle different normalization types
        if self.norm_cls == nn.GroupNorm:
            x = self.norm_cls(num_groups=8, dtype=self.dtype)(x)
        else:
            x = self.norm_cls(dtype=self.dtype)(x)
        x = nn.Conv(self.n_features, self.kernel_size, self.stride, padding='SAME', dtype=self.dtype)(x)
        x = self.activation(x)
        # Handle different normalization types
        if self.norm_cls == nn.GroupNorm:
            x = self.norm_cls(num_groups=8, dtype=self.dtype)(x)
        else:
            x = self.norm_cls(dtype=self.dtype)(x)
        # first 2× downsample only here:
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2), padding='SAME')
        return x

    
    
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
    
    
class CNNBlock(Module):
    n_features: int #n_channels
    kernel_size: Tuple[int, int] = (3, 3)
    stride1: Tuple[int, int] = (1, 1)
    stride2: Tuple[int, int] = (2, 2)
    activation: Callable = leaky_relu
    norm_cls: Optional[ModuleDef] = nn.BatchNorm
    dtype: Any = jnp.float32
    eval_mode: bool = False
    last_layer: bool = False
    # input data with dimensions (*batch_dims, spatial_dims…, features). 
    # This is the channels-last convention, 
    # i.e. NHWC for a 2d convolution and NDHWC for a 3D convolution.
    
    @nn.compact
    def __call__(self, x):
        x = Conv(self.n_features, kernel_size=self.kernel_size, 
                 strides=self.stride1, dtype=self.dtype)(x)
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
        x = self.activation(x)

        x = Conv(self.n_features, kernel_size=self.kernel_size, 
                 strides=self.stride1, dtype=self.dtype)(x)
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
        x = self.activation(x)
        
        # can alias high-frequency components
        # x = Conv(self.n_features, kernel_size=self.kernel_size, 
        #          strides=self.stride2, dtype=self.dtype)(x)
        x = nn.avg_pool(x, window_shape=self.stride2, strides=self.stride2, padding='SAME')
        x = Conv(self.n_features, kernel_size=self.kernel_size, 
                 strides=self.stride1, dtype=self.dtype)(x)

        # if not self.last_layer:
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
        x = self.activation(x)
        return x
    
# legacy
class __CNNTransposeBlock(Module):
    n_features: int #n_channels
    kernel_size: Tuple[int, int] = (3, 3)
    stride1: Tuple[int, int] = (1, 1)
    stride2: Tuple[int, int] = (2, 2)
    activation: Callable = leaky_relu
    norm_cls: Optional[ModuleDef] = nn.BatchNorm
    dtype: Any = jnp.float32
    eval_mode: bool = False
    # input data with dimensions (*batch_dims, spatial_dims…, features). 
    # This is the channels-last convention, 
    # i.e. NHWC for a 2d convolution and NDHWC for a 3D convolution.
    
    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(
            features=self.n_features, kernel_size=self.kernel_size,
            strides=self.stride2, padding='SAME', use_bias=False,
            kernel_init=nn.initializers.variance_scaling(
                1.0, 'fan_in', 'truncated_normal')
            )(x)
        # x = nn.Conv(
        #     features=self.n_features,
        #     kernel_size=self.kernel_size,
        #     strides=self.stride1,
        #     padding='SAME',
        #     use_bias=False
        # )(x)
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
        x = self.activation(x)
        return x
    
# legacy  
class ____CNNTransposeBlock(Module):
    n_features: int
    kernel_size: Tuple[int, int] = (3, 3)
    stride2: Tuple[int, int] = (2, 2)
    activation: Callable = leaky_relu
    norm_cls: Optional[ModuleDef] = nn.BatchNorm
    dtype: Any = jnp.float32
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x):
        sh, sw = self.stride2
        # Bilinear upsample (anti-checkerboard), channels-last
        if x.ndim == 5:
            B, T, H, W, C = x.shape
            x = jax.image.resize(x, (B, T, H*sh, W*sw, C), method='linear')
        elif x.ndim == 6:
            B1, B2, T, H, W, C = x.shape
            x = jax.image.resize(x, (B1, B2, T, H*sh, W*sw, C), method='linear')

        x = nn.Conv(self.n_features, self.kernel_size, (1,1), padding='SAME', use_bias=False, dtype=self.dtype)(x)
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
        x = self.activation(x)
        return x
    


class UpsampleSharpenBlock(nn.Module):
    """
    Bilinear upsample -> (3x3 Conv -> GN -> GELU -> 3x3 Conv) -> 1x1 skip -> GELU
    Gives you artifact-free upsampling + restores high-frequency detail.
    """
    out_ch: int
    scale: tuple = (2, 2)
    use_norm: bool = True
    num_groups: int = 8  # GroupNorm groups
    dtype: Any = jnp.float32
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x, *, train: bool = True):
        sh, sw = self.scale
        *prefix, H, W, C = x.shape  # channels-last

        # 1) Bilinear upsample (anti-checkerboard)
        x_up = jax.image.resize(
            x, tuple(prefix) + (H * sh, W * sw, C), method='linear'
        )

        # 2) Two 3x3 convs with a residual skip (1x1) to add sharpness
        h = nn.Conv(self.out_ch, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, dtype=self.dtype)(x_up)
        if self.use_norm:
            h = nn.GroupNorm(num_groups=self.num_groups, dtype=self.dtype)(h)
        h = nn.gelu(h)

        h = nn.Conv(self.out_ch, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, dtype=self.dtype)(h)
        # 1x1 skip to match channels
        skip = nn.Conv(self.out_ch, kernel_size=(1, 1), strides=(1, 1), padding='SAME', use_bias=False, dtype=self.dtype)(x_up)

        out = nn.gelu(h + skip)
        return out
    
    
class CNNTransposeBlock(nn.Module):
    n_features: int
    scale: tuple = (2, 2)
    use_norm: bool = True
    num_groups: int = 8
    dtype: Any = jnp.float32
    eval_mode: bool = False
    activation: Callable = leaky_relu
    norm_cls: Optional[ModuleDef] = nn.BatchNorm
    

    @nn.compact
    def __call__(self, x, *, train: bool = True):
        # Old: ConvTranspose(stride=2) or resize->single conv
        # New: Bilinear upsample + 3x3->GN->GELU->3x3 with residual skip
        x = UpsampleSharpenBlock(
            out_ch=self.n_features,
            scale=self.scale,
            use_norm=self.use_norm,
            num_groups=self.num_groups,
            dtype=self.dtype,
            eval_mode=self.eval_mode
        )(x, train=train)
        return x

    
class ResidualDownsamplingBlock(Module):
    out_ch: int
    pool_window: Tuple[int, int] = (2, 2)
    pool_stride: Tuple[int, int] = (2, 2)
    conv_kernel_size: Tuple[int, int] = (1, 1)
    conv_stride: Tuple[int, int] = (1, 1)
    conv_padding: str = 'SAME'
    conv_use_bias: bool = False
    dtype: Any = jnp.float32
    eval_mode: bool = False
    
    @nn.compact
    def __call__(self, x):
        x = nn.avg_pool(x, window_shape=self.pool_window, strides=self.pool_stride, padding='SAME')
        x = nn.Conv(self.out_ch, self.conv_kernel_size, self.conv_stride, 
                    padding=self.conv_padding, use_bias=self.conv_use_bias, 
                    dtype=self.dtype
                    )(x)
        return x
    
# legacy    
class __ResidualUpsamplingBlock(Module):
    out_ch: int
    scale: Tuple[int, int] = (2, 2)
    conv_kernel_size: Tuple[int, int] = (3, 3) # wider than 1x1 to smooth
    conv_stride: Tuple[int, int] = (1, 1)
    conv_padding: str = 'SAME'
    conv_use_bias: bool = False
    
    @nn.compact
    def __call__(self, x):
        # nearest can give checkerboard artifacts
        sh, sw = self.scale
        if x.ndim == 5: #(5, 24, h, w, ch)
            B1, B2, H, W, C = x.shape
            x = jax.image.resize(x, (B1, B2, H*sh, W*sw, C),
                # 'nearest' or 'linear' (also 'cubic','lanczos3','lanczos5')
                method='linear')
        elif x.ndim == 6: #  forecast:(1, 100, 36, h, w, ch)
            B1, B2, B3, H, W, C = x.shape
            x = jax.image.resize(x, (B1, B2, B3, H*sh, W*sw, C),
                # 'nearest' or 'linear' (also 'cubic','lanczos3','lanczos5')
                method='linear')
        x = nn.Conv(self.out_ch, self.conv_kernel_size, self.conv_stride, padding=self.conv_padding, use_bias=self.conv_use_bias)(x)
        return x

class ResidualUpsamplingBlock(nn.Module):
    out_ch: int
    scale: tuple = (2, 2)
    use_norm: bool = True
    num_groups: int = 8
    dtype: Any = jnp.float32
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x, *, train: bool = True):
        # Old: resize(method='nearest') + 1x1 Conv
        # New: Bilinear upsample + sharpen stack
        x = UpsampleSharpenBlock(
            out_ch=self.out_ch,
            scale=self.scale,
            use_norm=self.use_norm,
            num_groups=self.num_groups,
            dtype=self.dtype,
            eval_mode=self.eval_mode
        )(x, train=train)
        return x
    


class ConvNet(nn.Module):
    n_outputs: int
    network_mode: str = 'encoder' #'encoder' or 'decoder'
    block_cls: ModuleDef = CNNBlock
    last_layer_sigmoid: bool = False 
    encoder_lc_channels: int = 2
    decoder_input_features: Tuple[int, int] = (2, 7) 
    target_spatial_dims: Optional[Tuple[int, int]] = None  # (height, width) for decoder output
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
        
    @nn.compact
    def __call__(self, x, mask=None):
        # print('ConvNet')
        # print('network mode: ', self.network_mode)
        # print('initial input shape: ', x.shape)

        # Set block_cls, input_block_cls and res_block based on network_mode
        if self.network_mode == 'encoder':
            input_block_cls_type = ConvStem
            res_block_type = ResidualDownsamplingBlock
            block_cls_type = CNNBlock
        else:  # decoder
            input_block_cls_type = DenseBlock
            res_block_type = ResidualUpsamplingBlock
            block_cls_type = CNNTransposeBlock
            
        #x: (bs, T, H, W)
        x = x / self.scale_input
        stage_sizes, hidden_sizes = self.stage_sizes, self.hidden_sizes
        block_cls = partial(block_cls_type, eval_mode=self.eval_mode, dtype=self.dtype, activation=self.activation)
        lstm_layer = [self.lstm_layer] if type(self.lstm_layer) is int else self.lstm_layer
        
                            
        input_block_cls = partial(
            input_block_cls_type, eval_mode=self.eval_mode, 
            dtype=self.dtype, activation=self.activation)
        # print(f"input_block_cls type: {input_block_cls_type}")
        # print(f"block_cls_type: {block_cls_type}")
        # print(f"res_block_type: {res_block_type}")
        
        if self.network_mode == 'encoder':
            x = input_block_cls(n_features=self.encoder_lc_channels, norm_cls=self.norm_cls
                                )(jnp.expand_dims(x, -1)) #(bs, T, h1, w1, lc_channels)
            # print('after encoder input block: ', x.shape)
        # elif self.network_mode == 'decoder':
        #     # print('decoder input shape: ', x.shape)
        #     input_block_class = input_block_cls(
        #         n_features=self.decoder_input_features[0] * self.decoder_input_features[1], 
        #         norm_cls=self.norm_cls)
        #     # print('input block class: ', input_block_class)
        #     x = input_block_class(x)
        #     # print('after input block class: ', x.shape) # forecast:(1,100,36,125) or (5,24,125)
        #     if x.ndim == 3:
        #         x = x.reshape(x.shape[0], x.shape[1], self.decoder_input_features[0], self.decoder_input_features[1], 1) #(bs, T, d1, d2, 1)
        #     elif x.ndim == 4: #forecast: (1,100,36,125) -> (1,100,36,5,25,1)
        #         x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.decoder_input_features[0], self.decoder_input_features[1], 1) #(bs, T, d1, d2, 1)
        #     # print('after decoder input block reshape: ', x.shape)
        elif self.network_mode == 'decoder':
            # ----- NEW: project latent to a richer seed grid -----
            # We pick a seed that’s an integer divisor of the final target dims.
            # Assumes you’ve set `self.target_spatial_dims=(H, W)` on the decoder.
            if self.target_spatial_dims is None:
                raise ValueError("ConvNet(decoder): target_spatial_dims must be set, e.g. (H, W).")
            target_h, target_w = self.target_spatial_dims

            # Choose how coarse to start (4 is a good default). Make sure H%4==0 and W%4==0.
            seed_divisor = 4
            if (target_h % seed_divisor) or (target_w % seed_divisor):
                raise ValueError(f"Target dims {(target_h, target_w)} must be divisible by {seed_divisor}.")
            seed_h, seed_w = target_h // seed_divisor, target_w // seed_divisor

            seed_ch = 64  # richer than 1 channel to avoid blocky artifacts

            # x is (B, T, latent_D) or (B, T, horizon, latent_D); handle both
            if x.ndim == 3:
                B, T, _ = x.shape
                x = nn.Dense(seed_h * seed_w * seed_ch, dtype=self.dtype)(x)
                x = self.activation(x)
                if self.norm_cls:
                    x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
                x = x.reshape(B, T, seed_h, seed_w, seed_ch)

            elif x.ndim == 4:
                B, T, Hzn, _ = x.shape
                x = nn.Dense(seed_h * seed_w * seed_ch, dtype=self.dtype)(x)
                x = self.activation(x)
                if self.norm_cls:
                    x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
                x = x.reshape(B, T, Hzn, seed_h, seed_w, seed_ch)

            else:
                raise ValueError(f"Unexpected decoder input rank {x.ndim}; expected 3 or 4.")

            # From here, keep using your existing upsampling blocks (CNNTransposeBlock / ResidualUpsamplingBlock)
            # so that two ×2 stages get you from (H/4, W/4) -> (H/2, W/2) -> (H, W).

        else:
            raise ValueError(f"Invalid network mode: {self.network_mode}")
            
        for i, (hsize, n_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
            #disabled: lstm_hidden_size = 0
            if self.lstm_cls and self.lstm_hidden_size > 0 and i in lstm_layer: 
                x = self.lstm_cls(self.lstm_hidden_size)(x, mask=mask)
            
            x_res = x
            #disabled: n_blocks = 1
            if n_blocks < 0: # indicating a layer which should factorize across angles
                x = x.reshape(x.shape[:-1] + (-1,-n_blocks))
                x = block_cls(n_features=hsize, norm_cls=self.norm_cls,)(x)
                x = x.reshape(x.shape[:-2] + (-1,))
            else:
                for b in range(n_blocks - 1):
                    x = block_cls(n_features=hsize, norm_cls=self.norm_cls,)(x)
                    # print(f'after block {b} {i}: ', x.shape)
                x = block_cls(n_features=hsize, norm_cls=(not self.resnet) and self.norm_cls)(x)
                # print(f'after block {n_blocks - 1} {i}: ', x.shape)
            
            if self.resnet and i > 0:
                # print('Res Channels: ', x.shape[-1])
                x = x + res_block_type(out_ch=x.shape[-1], dtype=self.dtype, eval_mode=self.eval_mode)(x_res)
                if self.norm_cls:
                    x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
                # print(f'after residual block {i}: ', x.shape)
        #disabled: lstm_hidden_size = 0  
        if self.lstm_cls and self.lstm_hidden_size > 0 and len(stage_sizes) == lstm_layer[-1]:
            x = self.lstm_cls(self.lstm_hidden_size)(x, mask=mask)

        # print('before last layer: ', x.shape)
        if self.network_mode == 'encoder':
            x = nn.Dense(self.n_outputs, dtype=self.dtype)(x.reshape(x.shape[0], x.shape[1], -1))
            # print('after encoder last layer: ', x.shape)
        elif self.network_mode == 'decoder':
            # Final convolutions to get correct number of channels
            #(5, 24, 240, 520, 8)
            x = nn.Conv(features=1, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=True, dtype=self.dtype)(x)
            
            # Final upsampling to match input dimensions if target_spatial_dims is specified
            if self.target_spatial_dims is not None:
                target_h, target_w = self.target_spatial_dims
                #x: (5,24,h,w,ch) or forecast: (1,100,36,h,w,ch)
                current_h, current_w = x.shape[-3], x.shape[-2]

                # Only upsample if dimensions don't match
                if current_h != target_h or current_w != target_w:
                    # Use jax.image.resize for precise upsampling
                    if x.ndim == 5:
                        x = jax.image.resize(
                            x, 
                            (x.shape[0], x.shape[1], target_h, target_w, x.shape[4]),
                            method='linear')  # or 'nearest'
                    elif x.ndim == 6:
                        x = jax.image.resize(
                            x, 
                            (x.shape[0], x.shape[1], x.shape[2],target_h, target_w, x.shape[5]),
                            method='linear')  # or 'nearest'
                    # print('after final upsampling: ', x.shape)
            
            x = nn.Conv(features=1, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=True, dtype=self.dtype)(x)
            #x: (5,24,h,w,1) or forecast: (1,100,36,h,w,1)
            if self.last_layer_sigmoid:
                x = sigmoid(x)
            # print('after decoder last layer: ', x.shape)
                        
        else:
            raise ValueError(f"Invalid network mode: {self.network_mode}")
        
        # print('after last layer: ', x.shape)
        return x * self.scale_output
    
    

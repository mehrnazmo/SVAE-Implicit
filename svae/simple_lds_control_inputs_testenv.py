import os
import sys
svae_path='svae/SVAE-Implicit/svae'
sys.path.insert(0,svae_path)
from configs import setup
config_file = 'dataset/cfg_vae.ini'
(cfg_climate, save_dir, train_dataloader, val_dataloader, 
 train_num, val_num, start_idx) = setup(config_file)



# import jax.numpy as jnp
# import numpy as np
# import xarray as xr
# from nino34_calculator import compute_nino34_and_oni

# def process_batch(batch):
#     tensors = tuple(jnp.array(t) for t in batch)
#     return tensors

# min_sst = np.load(os.path.join((save_dir), 'SST_min_climate.npy'), allow_pickle=True)
# max_sst = np.load(os.path.join((save_dir), 'SST_max_climate.npy'), allow_pickle=True)
# climatology = xr.open_dataset(f'{os.path.dirname(save_dir)}/climatology2.nc')['SST']
# lat_range = cfg_climate.get('DataFrame', 'lat_range')
# lon_range = cfg_climate.get('DataFrame', 'lon_range')
# lat_range = tuple(int(x) for x in lat_range.strip('()').split(','))
# lon_range = tuple(int(x) for x in lon_range.strip('()').split(','))

# def correct_sst_range(sst, sst_min, sst_max):
#     return sst * (sst_max - sst_min) + sst_min

# for batch in train_dataloader:
#     batch = process_batch(batch)
#     (data_orig, mask_sst_orig, time_enc_orig, 
#      month_orig, year_orig, nino34_orig, oni_orig) = batch
#     print(f"data_orig.shape: {data_orig.shape}")
#     print(f"mask_sst_orig.shape: {mask_sst_orig.shape}")
#     print(f"time_enc_orig.shape: {time_enc_orig.shape}")
#     print(f"month_orig.shape: {month_orig.shape}")
#     print(f"year_orig.shape: {year_orig.shape}")
#     print(f"nino34_orig.shape: {nino34_orig.shape}")
#     print(f"oni_orig.shape: {oni_orig.shape}")
#     print('-'*100)
#     print('data_orig[0]: \n', data_orig[0])
#     # print('nino34_orig:', nino34_orig)
#     # print('-'*100)
#     # print('oni_orig:', oni_orig)
#     # print('-'*100)
#     # print('nino34_orig_slice:', nino34_orig[:,24:33])
#     # print('-'*100)
#     # print('oni_orig_slice:', oni_orig[:,24:33])
#     # print('correct_sst_range(data_orig, min_sst, max_sst)[0]: \n', 
#         #   correct_sst_range(data_orig, min_sst, max_sst)[0])
#     nino34_preds, oni_preds = compute_nino34_and_oni(
#         np.flip(data_orig, axis=2),
#         lat_range=lat_range, lon_range=lon_range,
#         start_month=(start_idx+2)%12,
#         climatology=np.array(climatology.values))
#     print('nino34_preds:', nino34_preds.shape)
#     print('-'*100)
#     print('oni_preds:', oni_preds.shape)
#     print('-'*100)
#     print('\n'*3)
#     print('Sanity check: nino34_preds:', np.allclose(nino34_preds, nino34_orig))
#     print('Sanity check: oni_preds:', np.allclose(oni_preds, oni_orig))
#     print('-'*100)
#     print('-'*100)
#     print('nino34_preds:\n', nino34_preds)
#     print('nino34_orig:\n', nino34_orig)
#     print('oni_preds:\n', oni_preds)
#     print('oni_orig:\n', oni_orig)
#     print('-'*100)
#     print('-'*100)


import jax.numpy as jnp
from jax import lax
import jax

# JAX memory optimization settings
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_enable_x64', False)  # Use float32 for memory efficiency
jax.config.update('jax_disable_jit', False)
# Enable memory optimization
jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')

print('jax.devices()', jax.devices())  # Should show GPU
print('jax.default_backend()', jax.default_backend())  # Should show 'gpu'
rng = jax.random.PRNGKey(1)
rng, subkey1 = jax.random.split(rng)

import numpy as np
import flax.linen as nn
from functools import partial
import torch
import optax
import numpy.random as npr
from tensorflow_probability.substrates.jax import distributions as tfd
import wandb
import inspect
# from memory_profiler import profile
import gc
from tqdm import tqdm
import xarray as xr
import os
from time import time

# from models.SVAE_LDS_Simple import SVAE_LDS_Simple as SVAE_LDS
from models.SVAE_LDS import SVAE_LDS
from models.SVAE_SLDS import SVAE_SLDS

from networks.layers import LayerNorm
from networks.dense import DenseNet
from networks.cnn import ConvNet
from networks.encoders import SigmaEncoder
from networks.decoders import SigmaDecoder


from train_utils import create_dual_train_state
from train_utils import make_eval_step_with_transform, make_train_step_with_transform
from visualization_utils import (
    visualize_latent_space, plot_reconstructions, plot_temporal_dynamics,
    plot_discrete_state_probabilities, plot_latent_dynamics_by_state, plot_state_transition_matrix)
from train_utils import forecast, save_nino34_oni, forecast_flat

torch.manual_seed(1)
npr.seed(1)

def identity(inp):
    return inp

def get_beta_with_warmup(current_step):
    """
    Gets the beta value for the current epoch number
    """
    # max_beta = 1.0
    # min_beta = 0.001
    cfg.max_beta = 0.1
    cfg.min_beta = 0.001

    # cfg.warmup_steps = 1000
    # cfg.total_annealing_steps = 1000
    cfg.warmup_steps = 2000
    cfg.total_annealing_steps = 2000

    
    if current_step < cfg.warmup_steps:
        return cfg.min_beta
    else:
        annealing_step = current_step - cfg.warmup_steps
        return cfg.min_beta + (cfg.max_beta - cfg.min_beta) * min(1.0, annealing_step / cfg.total_annealing_steps)

def save_pgm_parameters(state, epoch, save_dir):
    """
    Save PGM parameters to files for detailed analysis
    """
    pgm_params = state.params['pgm']
    pgm_save_dir = f'{save_dir}/pgm_params'
    os.makedirs(pgm_save_dir, exist_ok=True)
    
    if 'S' in pgm_params:  # Standard parameterization
        # Convert JAX arrays to numpy for saving
        params_to_save = {}
        for key, value in pgm_params.items():
            params_to_save[key] = np.array(value)
        
        # Save all parameters
        np.savez(f'{pgm_save_dir}/pgm_params_epoch_{epoch}.npz', **params_to_save)
        
        # Also save individual matrices for easier analysis
        np.save(f'{pgm_save_dir}/S_epoch_{epoch}.npy', np.array(pgm_params['S']))
        np.save(f'{pgm_save_dir}/loc_epoch_{epoch}.npy', np.array(pgm_params['loc']))
        np.save(f'{pgm_save_dir}/St_epoch_{epoch}.npy', np.array(pgm_params['St']))
        np.save(f'{pgm_save_dir}/M_epoch_{epoch}.npy', np.array(pgm_params['M']))
        np.save(f'{pgm_save_dir}/V_epoch_{epoch}.npy', np.array(pgm_params['V']))
        
        print(f"Saved PGM parameters for epoch {epoch}")
    
    elif 'niw' in pgm_params:  # Natural parameterization
        niw_params = pgm_params['niw']
        mniw_params = pgm_params['mniw']
        
        np.savez(f'{pgm_save_dir}/pgm_nat_params_epoch_{epoch}.npz', 
                 niw=niw_params, mniw=mniw_params)
        print(f"Saved PGM natural parameters for epoch {epoch}")

def compute_pgm_convergence_metrics(state, epoch, prev_pgm_params=None):
    """
    Compute convergence metrics for PGM parameters
    """
    pgm_params = state.params['pgm']
    convergence_metrics = {}
    
    if 'S' in pgm_params and prev_pgm_params is not None:
        # Compute parameter change rates
        for key in ['S', 'loc', 'lam', 'nu', 'St', 'M', 'V', 'nut']:
            if key in pgm_params and key in prev_pgm_params:
                current = pgm_params[key]
                previous = prev_pgm_params[key]
                
                # Compute relative change
                if jnp.any(previous != 0):
                    rel_change = jnp.abs(current - previous) / (jnp.abs(previous) + 1e-8)
                    convergence_metrics[f'pgm/{key}_rel_change'] = float(jnp.mean(rel_change))
                    convergence_metrics[f'pgm/{key}_max_rel_change'] = float(jnp.max(rel_change))
                
                # Compute absolute change
                abs_change = jnp.abs(current - previous)
                convergence_metrics[f'pgm/{key}_abs_change'] = float(jnp.mean(abs_change))
                convergence_metrics[f'pgm/{key}_max_abs_change'] = float(jnp.max(abs_change))
    
    return convergence_metrics

def process_batch(batch):
    tensors = tuple(jnp.array(t) for t in batch)
    return tensors

wandb_project = cfg_climate.get('User', 'wandb_project')
wandb_user = cfg_climate.get('User', 'wandb_user')
batch_size = cfg_climate.getint('DataFrame', 'batch_size')
lat_range = cfg_climate.get('DataFrame', 'lat_range')
lon_range = cfg_climate.get('DataFrame', 'lon_range')
lat_range = tuple(int(x) for x in lat_range.strip('()').split(','))
lon_range = tuple(int(x) for x in lon_range.strip('()').split(','))
print('lat_range: ', lat_range)
print('lon_range: ', lon_range)
print('-'*100)
# mode = 'disabled' in wandb.init to have it not track
wandb.init(project=wandb_project, entity=wandb_user, config=cfg_climate)
cfg = wandb.config

#(bs, 24, 26*90)
setup_batch = process_batch(next(iter(train_dataloader)))[0]
print(f"Setup batch shape: {setup_batch.shape}")

train_step = make_train_step_with_transform(identity)
eval_step = make_eval_step_with_transform(identity)

# input_D = 90 * 26
# h, w = 120, 360
h, w = lat_range[1] - lat_range[0], lon_range[1] - lon_range[0]
input_D = h * w

cfg.num_epochs = 6000
cfg.lr_pgm = 5e-5  # Slower PGM learning for stability # 3e-6
cfg.lr_net = 1e-4
cfg.lr_decay_steps = 250000 # original 250000
cfg.lr_final_mult = 0.01 # original 0.01
learning_alg_pgm = 'sgd'
learning_alg_net = 'adam'

resnet=True
# encoder_stage_sizes = [1,  1,   1,   1]
# encoder_hidden_sizes =  [128, 128, 128, 128] # Larger encoder

encoder_stage_sizes = [1,  1]
# encoder_hidden_sizes =  [128, 128] # Larger encoder
encoder_hidden_sizes =  [256, 256] # Larger encoder
# encoder_hidden_sizes =  [32, 32] # Larger encoder

# decoder_stage_sizes = [1,    1,   1,   1]
# decoder_hidden_sizes = [256, 256, 256, 256]  # Larger decoder

decoder_stage_sizes = [1,    1,  1]
# decoder_hidden_sizes = [128, 128, 8]  # Larger decoder
decoder_hidden_sizes = [256, 256, 32]  # Larger decoder
# decoder_hidden_sizes = [64, 64, 4]  # Larger decoder

encoder_lstm_units = 0
encoder_lstm_layer = -1
cfg.latent_D = 50  # Increased latent dimension

network_type = 'dense' # 'dense' or 'cnn'
flatten_input = True if network_type == 'dense' else False

encoder_activation = 'gelu'
decoder_activation = 'gelu'
last_layer_sigmoid = True
encoder_norm = 'layer'
decoder_norm = 'layer'
encoder_loc_norm = 'none'
encoder_scale_norm = 'none'
validate = True

encoder_month_embedding = False
decoder_month_embedding = False
cond_on_month = False

lc_channels = 1
decoder_input_features = (5, 25)

# S_0 = 1.
# nu_0 = 2.
# lam_0 = 0.01
# M_0 = 0.7
# S_init = 1.
# nu_init = 2.
# lam_init = 20.
# loc_init_sd = 0.05


# # ### SVAE_Repo
S_0 = 1.
nu_0 = 2.
lam_0 = 0.001
M_0 = 0.9
# M_0 = 0.7
S_init = 1.
nu_init = 2.
lam_init = 20.
loc_init_sd = 0.2

# S_0 = 1.
# nu_0 = 2.
# lam_0 = 0.01
# M_0 = 0.7  # Reduced from 0.9 to prevent eigenvalues > 1
# S_init = 1.
# nu_init = 5.
# lam_init = 20.
# loc_init_sd = 0.05

# # More conservative PGM hyperparameters for stability
# S_0 = 0.1        # Weaker prior scale
# nu_0 = 5.0        # More degrees of freedom
# lam_0 = 0.01      # Stronger precision
# M_0 = 0.5         # More conservative mean
# S_init = 0.1      # Smaller initial scale
# nu_init = 5.0     # More stable initialization
# lam_init = 10.0   # More conservative precision
# loc_init_sd = 0.05  # Smaller initialization noise

model_type = 'LDS' # 'LDS' or 'SLDS'
cfg.K = 12


# Update wandb config with additional parameters
wandb.config.update({
    'num_epochs': cfg.num_epochs,
    'lr_pgm': cfg.lr_pgm,
    'lr_net': cfg.lr_net,
    'lr_decay_steps': cfg.lr_decay_steps,
    'lr_final_mult': cfg.lr_final_mult,
    'learning_alg_pgm': learning_alg_pgm,
    'learning_alg_net': learning_alg_net,
    'resnet': resnet,
    'encoder_stage_sizes': encoder_stage_sizes,
    'encoder_hidden_sizes': encoder_hidden_sizes,
    'decoder_stage_sizes': decoder_stage_sizes,
    'decoder_hidden_sizes': decoder_hidden_sizes,
    'encoder_lstm_units': encoder_lstm_units,
    'encoder_lstm_layer': encoder_lstm_layer,
    'latent_D': cfg.latent_D,
    'network_type': network_type,
    'encoder_activation': encoder_activation,
    'decoder_activation': decoder_activation,
    'last_layer_sigmoid': last_layer_sigmoid,
    'encoder_norm': encoder_norm,
    'decoder_norm': decoder_norm,
    'encoder_loc_norm': encoder_loc_norm,
    'encoder_scale_norm': encoder_scale_norm,
    'validate': validate,
    'encoder_month_embedding': encoder_month_embedding,
    'decoder_month_embedding': decoder_month_embedding,
    'cond_on_month': cond_on_month,
    'input_D': input_D,
    'save_dir': save_dir,
    'model': model_type,
    'K': cfg.K,
    'S_0': S_0,
    'nu_0': nu_0,
    'lam_0': lam_0,
    'M_0': M_0,
    'S_init': S_init,
    'nu_init': nu_init,
    'lam_init': lam_init,
    'loc_init_sd': loc_init_sd,
    'lc_channels': lc_channels,
    'decoder_input_features': decoder_input_features
})



activations = dict(leaky_relu=nn.leaky_relu, tanh=jnp.tanh, gelu=nn.gelu)
norms = dict(batch=nn.BatchNorm, layer=LayerNorm, none=None)
network_modules = dict(dense=DenseNet, cnn=ConvNet)

# Common arguments for all network types
encoder_network_kwargs = dict(
    resnet=resnet,
    stage_sizes=encoder_stage_sizes,
    hidden_sizes=encoder_hidden_sizes,
    activation=activations[encoder_activation],
    norm_cls=norms[encoder_norm],
    lstm_hidden_size=encoder_lstm_units,
    lstm_layer=encoder_lstm_layer
)

decoder_network_kwargs = dict(
    resnet=resnet,
    stage_sizes=decoder_stage_sizes,
    hidden_sizes=decoder_hidden_sizes,
    activation=activations[decoder_activation],
    norm_cls=norms[decoder_norm]
)


encoder_network_specific_arguments = dict(
    dense = dict(
        flatten_input=flatten_input,
        last_layer_sigmoid=False,
        lc_channels=lc_channels), 
    cnn = dict(
        last_layer_sigmoid=False,
        encoder_lc_channels=lc_channels, 
        network_mode='encoder')
    )

decoder_network_specific_arguments = dict(
    dense = dict(
        flatten_input=flatten_input,  # Match encoder's flatten_input setting
        last_layer_sigmoid=last_layer_sigmoid), 
    cnn = dict(
        decoder_input_features=decoder_input_features, 
        last_layer_sigmoid=last_layer_sigmoid,
        network_mode='decoder',
        target_spatial_dims=(h, w)))  # (40, 200) for your case

# Add network-specific arguments
encoder_network_kwargs.update(**encoder_network_specific_arguments[network_type])
encoder_network = partial(network_modules[network_type], **encoder_network_kwargs)

decoder_network_kwargs.update(**decoder_network_specific_arguments[network_type])
decoder_network = partial(network_modules[network_type], **decoder_network_kwargs)

encoder = partial(SigmaEncoder, network_cls=encoder_network,
                  loc_norm_cls=norms[encoder_loc_norm],
                  scale_norm_cls=norms[encoder_scale_norm],
                  month_embedding=encoder_month_embedding,
                 )


decoder = partial(SigmaDecoder, network_cls=decoder_network,
                  likelihood=tfd.Normal,
                  month_embedding=decoder_month_embedding,
                 )

pgm_hyperparameters=dict(
    S_0=S_0,
    nu_0=nu_0,
    lam_0=lam_0,
    M_0=M_0,
    S_init=S_init,
    nu_init=nu_init,
    lam_init=lam_init,
    loc_init_sd=loc_init_sd
)

if model_type == 'LDS':
    pgm_hyperparameters['cond_on_month'] = cond_on_month
    model_builder = partial(SVAE_LDS, latent_D=cfg.latent_D, 
                            input_D = input_D,
                            encoder_cls=encoder, decoder_cls=decoder, 
                            pgm_hyperparameters=pgm_hyperparameters)
elif model_type == 'SLDS':
    model_builder = partial(SVAE_SLDS, latent_D=cfg.latent_D, 
                            K=cfg.K,
                            input_D = input_D,
                            encoder_cls=encoder, decoder_cls=decoder,
                            pgm_hyperparameters=pgm_hyperparameters)

# Calculate total number of training steps for LR schedule
# Total steps = num_epochs * batches_per_epoch
N_batches_per_epoch = len(train_dataloader)
total_training_steps = cfg.num_epochs * N_batches_per_epoch
print(f"Total training steps: {total_training_steps} (epochs: {cfg.num_epochs} × batches_per_epoch: {N_batches_per_epoch})")

# Add gradient clipping for stability
net_schedule = optax.cosine_decay_schedule(cfg.lr_net, 20000, cfg.lr_final_mult)
pgm_schedule = optax.cosine_decay_schedule(cfg.lr_pgm, cfg.lr_decay_steps, cfg.lr_final_mult)
# nn_boundaries_and_scales = {
#         1000: 0.5,    # At step 1000, multiply by 0.5
#         2000: 0.1,    # At step 5000, multiply by 0.1
#         3000: 0.0   # At step 10000, multiply by 0.0
#     }
# net_schedule = optax.piecewise_constant_schedule(init_value=cfg.lr_net, boundaries_and_scales=nn_boundaries_and_scales)

print('nn schedule: ', cfg.lr_net, 20000, cfg.lr_final_mult)
print('pgm schedule: ', cfg.lr_pgm, cfg.lr_decay_steps, cfg.lr_final_mult)

# # Apply gradient clipping
# net_schedule = optax.chain(
#     optax.clip_by_global_norm(1.0),  # Gradient clipping
#     optax.scale_by_schedule(net_schedule)
# )
# pgm_schedule = optax.chain(
#     optax.clip_by_global_norm(0.5),  # More aggressive clipping for PGM
#     optax.scale_by_schedule(pgm_schedule)
# )

#model: SVAE_LDS
#state: DualTrainState keeps states such as 
# (params, batch_stats, opt_state_net, opt_state_pgm, ...)

# Create a dummy month tensor for initialization to ensure month_dense layer is created
LDS_T = 24
if flatten_input:
    setup_batch_shape = (setup_batch.shape[0], LDS_T, setup_batch.shape[-1])
else:
    setup_batch_shape = (setup_batch.shape[0], LDS_T, setup_batch.shape[-2], setup_batch.shape[-1])
dummy_month = jnp.zeros((setup_batch.shape[0], LDS_T, 24))  # batch_size x time_dim (24)

#model: SVAE_LDS
#state: DualTrainState keeps states such as 
# (params, batch_stats, opt_state_net, opt_state_pgm, ...)
model, state = create_dual_train_state(
    rng, net_schedule, pgm_schedule, 
    model_builder, setup_batch_shape, 
    learning_alg_net=learning_alg_net,
    learning_alg_pgm=learning_alg_pgm,
    month_encoding=dummy_month)

# Check the train_step function signature
print(f"train_step function signature: {inspect.signature(train_step)}")
print(f"Model: {model}")

# Log SLURM job ID to wandb
slurm_job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
wandb.log({'slurm_job_id': slurm_job_id})
print(f"SLURM Job ID: {slurm_job_id}")

# climatology = np.load(f'{os.path.dirname(save_dir)}/climatology.npy')
climatology = xr.open_dataset(f'{os.path.dirname(save_dir)}/climatology.nc')['SST']
print(f"Loaded Climatology with shape: {climatology.values.shape}")
min_sst = np.load(os.path.join((save_dir), 'SST_min_climate.npy'), allow_pickle=True)
max_sst = np.load(os.path.join((save_dir), 'SST_max_climate.npy'), allow_pickle=True)
os.makedirs(f'{save_dir}/enso', exist_ok=True)
print(f"Loaded min_sst: {min_sst.shape}, max_sst: {max_sst.shape}")
month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 
              4: 'Apr', 5: 'May', 6: 'Jun', 
              7: 'Jul', 8: 'Aug', 9: 'Sep', 
              10: 'Oct', 11: 'Nov', 0: 'Dec'}
wandb.log({'start_month': month_dict[(start_idx+2)%12]})

losses = []
# loss_history = []  # Store detailed loss history for analysis
# prev_state = state
global_step = 0
prev_pgm_params = None  # Store previous PGM parameters for convergence tracking

# Early stopping and monitoring
best_val_loss = float('inf')
patience = 2000  # Stop if no improvement for 1000 epochs
patience_counter = 0
for epoch in tqdm(range(cfg.num_epochs)):
    # batch_metrics = []
    metrics = None
    running_loss, running_loss_test = 0, 0
    epoch_losses = []
    train_rmse_epoch, test_rmse_epoch = 0, 0
        
    N_batches = len(train_dataloader) #25
    #train_dataloader.dataset.sst.shape: [123, 24, 43200] = [N//12, T, D]
    if flatten_input:
        N_train, data_T, _ = train_dataloader.dataset.sst.shape #123
    else:
        N_train, data_T, _, _ = train_dataloader.dataset.sst.shape #123
    N_valid = val_dataloader.dataset.sst.shape[0] #30
    # print('='*100)
    # print('Epoch: ', epoch)

    wandb.log({'Epoch': epoch})
    for batch, step in zip(train_dataloader, range(N_batches)):
    # for batch, step in zip([next(iter(train_dataloader))], [0]):
        batch = process_batch(batch)
        # self.mask.shape:  torch.Size([122, 36, 43200])
        # self.sst.shape:  torch.Size([122, 36, 43200])
        # self.time_encodings.shape:  torch.Size([122, 36, 24])
        # self.months.shape:  torch.Size([122, 36])
        # self.years.shape:  torch.Size([122, 36])
        # self.nino34.shape:  torch.Size([122, 36])
        # self.oni.shape:  torch.Size([122, 36])
        (data, mask_sst, time_enc, _, _, _, _) = batch
        bs = data.shape[0]
        # print('-'*100)
        # print('data.shape: ', data.shape)
        # print('mask_sst.shape: ', mask_sst.shape)
        # print('time_enc.shape: ', time_enc.shape)
        # print('-'*100)
        
        rng, subkey2 = jax.random.split(rng)
        # can add the local_kl_weight parameter, originally defaulted to 1.
        beta = get_beta_with_warmup(epoch)  
        wandb.log({'beta': beta})      

        #metrics:dict(recon_loss=outputs['recon_loss'], prior_kl=outputs['prior_kl'], 
        #local_kl = outputs['local_kl'], 
        #loss=outputs['recon_loss'] + outputs['prior_kl'] + outputs['local_kl'], 
        #aux=outputs['aux'][-1])
        # Available metrics keys: ['aux', 'local_kl', 'loss', 'prior_kl', 'recon_loss', 'rmse_train']
        #(EXXT, mu, EXXNT), logZ, (Jf[:-1], hf[:-1], Js[1:], -hs[1:])
        # metrics['aux'][0][0]: (24, 50, 50)
        # metrics['aux'][0][1]: (24, 50, 50)
        # metrics['aux'][0][2]: (24, 50, 50)
        # metrics['aux'][1]: (5, 24, 50, 1)
        # metrics['aux'][2][0]: (23, 50, 50)
        # metrics['aux'][2][1]: (23, 50, 50)
        # metrics['aux'][2][2]: (23, 50, 50)
        new_state, metrics = train_step(
            state, data, month_encoding=time_enc,
            mask=None,           # your original mask if needed
            mask_sst=mask_sst,   # the new SST mask from batch
            N_data=N_batches * batch_size,
            local_kl_weight=beta, 
            log_magnitudes=False
        )
        
        state = new_state
        global_step += 1
        
        # Clear intermediate variables to save memory
        del new_state, data, mask_sst, time_enc
        
        if np.isnan(metrics['loss'].item()):
            print(f"NAN loss! In Epoch: {epoch}")
            print(f"Loss: {metrics['loss'].item()}, "
                  f"Recon Loss: {metrics['recon_loss'].item()}, "
                  f"KL: {metrics['local_kl'].item()}")
            break
        
        # lossses for each step
        loss = metrics['loss'].item()
        local_kl = metrics['local_kl'].item()
        prior_kl = metrics['prior_kl'].item()
        recon_loss = metrics['recon_loss'].item()
        rmse_train = metrics['rmse_train'].item()
        
        # Get current learning rate from net_schedule
        current_lr_net = net_schedule(global_step)
        current_lr_pgm = pgm_schedule(global_step)
        
        log_dict = {'train_nelbo': loss, 'local_kl': local_kl, 
                   'train_recon_loss': recon_loss, 
                   'prior_kl': prior_kl,
                   'train_recon_loss_per_TD': recon_loss/(input_D*LDS_T),
                   'train_rmse': rmse_train,
                   'learning_rate_net': current_lr_net,
                   'learning_rate_pgm': current_lr_pgm}
        
        # Track PGM parameters convergence
        if epoch % 10 == 0:  # Log PGM params every 10 epochs to avoid too much logging
            pgm_params = state.params['pgm']
            
            # Extract PGM parameters
            if 'S' in pgm_params:  # Standard parameterization
                S_param = pgm_params['S']
                loc_param = pgm_params['loc'] 
                lam_param = pgm_params['lam']
                nu_param = pgm_params['nu']
                St_param = pgm_params['St']
                M_param = pgm_params['M']
                V_param = pgm_params['V']
                nut_param = pgm_params['nut']
                
                # Extract transition matrix A from M parameter
                # M has shape (latent_D, latent_D+1) where first latent_D columns are A, last column is b
                if cond_on_month and M_param.ndim == 3:  # (12, latent_D, latent_D+1)
                    A_matrices = M_param[:, :, :-1]  # Extract A matrices for each month
                else:  # (latent_D, latent_D+1)
                    A_matrices = M_param[:, :-1]  # Extract A matrix
                
                # Compute eigenvalues of transition matrices (move to CPU for GPU compatibility)
                if cond_on_month and A_matrices.ndim == 3:
                    # For monthly conditioning, compute eigenvalues for each month
                    A_cpu = jax.device_put(A_matrices, jax.devices('cpu')[0])
                    eigenvals = jnp.linalg.eigvals(A_cpu)  # Shape: (12, latent_D)
                    pgm_log_dict = {
                        'pgm/S_norm': float(jnp.linalg.norm(S_param)),
                        'pgm/S_trace': float(jnp.trace(S_param)),
                        'pgm/loc_norm': float(jnp.linalg.norm(loc_param)),
                        'pgm/lam': float(lam_param),
                        'pgm/nu': float(nu_param),
                        'pgm/St_norm': float(jnp.linalg.norm(St_param)),
                        'pgm/M_norm': float(jnp.linalg.norm(M_param)),
                        'pgm/V_norm': float(jnp.linalg.norm(V_param)),
                        'pgm/nut_mean': float(jnp.mean(nut_param)) if nut_param.ndim > 0 else float(nut_param),
                        # Transition matrix A eigenvalues
                        'pgm/A_eigenvals_max_real': float(jnp.max(jnp.real(eigenvals))),
                        'pgm/A_eigenvals_min_real': float(jnp.min(jnp.real(eigenvals))),
                        'pgm/A_eigenvals_max_imag': float(jnp.max(jnp.imag(eigenvals))),
                        'pgm/A_eigenvals_min_imag': float(jnp.min(jnp.imag(eigenvals))),
                        'pgm/A_eigenvals_max_magnitude': float(jnp.max(jnp.abs(eigenvals))),
                        'pgm/A_eigenvals_mean_magnitude': float(jnp.mean(jnp.abs(eigenvals))),
                        'pgm/A_eigenvals_std_magnitude': float(jnp.std(jnp.abs(eigenvals))),
                        # Stability indicators
                        'pgm/A_unstable_count': float(jnp.sum(jnp.abs(eigenvals) > 0.95)),
                        'pgm/A_max_eigenval': float(jnp.max(jnp.abs(eigenvals))),
                        'pgm/A_stability_ratio': float(jnp.sum(jnp.abs(eigenvals) < 0.95) / eigenvals.size),
                    }
                else:
                    # For non-monthly conditioning
                    A_cpu = jax.device_put(A_matrices, jax.devices('cpu')[0])
                    eigenvals = jnp.linalg.eigvals(A_cpu)  # Shape: (latent_D,)
                    pgm_log_dict = {
                        'pgm/S_norm': float(jnp.linalg.norm(S_param)),
                        'pgm/S_trace': float(jnp.trace(S_param)),
                        'pgm/loc_norm': float(jnp.linalg.norm(loc_param)),
                        'pgm/lam': float(lam_param),
                        'pgm/nu': float(nu_param),
                        'pgm/St_norm': float(jnp.linalg.norm(St_param)),
                        'pgm/M_norm': float(jnp.linalg.norm(M_param)),
                        'pgm/V_norm': float(jnp.linalg.norm(V_param)),
                        'pgm/nut_mean': float(jnp.mean(nut_param)) if nut_param.ndim > 0 else float(nut_param),
                        # Transition matrix A eigenvalues
                        'pgm/A_eigenvals_max_real': float(jnp.max(jnp.real(eigenvals))),
                        'pgm/A_eigenvals_min_real': float(jnp.min(jnp.real(eigenvals))),
                        'pgm/A_eigenvals_max_imag': float(jnp.max(jnp.imag(eigenvals))),
                        'pgm/A_eigenvals_min_imag': float(jnp.min(jnp.imag(eigenvals))),
                        'pgm/A_eigenvals_max_magnitude': float(jnp.max(jnp.abs(eigenvals))),
                        'pgm/A_eigenvals_mean_magnitude': float(jnp.mean(jnp.abs(eigenvals))),
                        'pgm/A_eigenvals_std_magnitude': float(jnp.std(jnp.abs(eigenvals))),
                        # Stability indicators
                        'pgm/A_unstable_count': float(jnp.sum(jnp.abs(eigenvals) > 0.95)),
                        'pgm/A_max_eigenval': float(jnp.max(jnp.abs(eigenvals))),
                        'pgm/A_stability_ratio': float(jnp.sum(jnp.abs(eigenvals) < 0.95) / eigenvals.size),
                    }
                
                # If cond_on_month=True, log month-specific statistics
                if cond_on_month and St_param.ndim == 3:  # (12, D, D)
                    pgm_log_dict.update({
                        'pgm/St_month_std': float(jnp.std(jnp.linalg.norm(St_param, axis=(1,2)))),
                        'pgm/M_month_std': float(jnp.std(jnp.linalg.norm(M_param, axis=(1,2)))),
                        'pgm/V_month_std': float(jnp.std(jnp.linalg.norm(V_param, axis=(1,2)))),
                    })
                
                log_dict.update(pgm_log_dict)
            
            elif 'niw' in pgm_params:  # Natural parameterization
                niw_params = pgm_params['niw']
                mniw_params = pgm_params['mniw']
                
                pgm_log_dict = {
                    'pgm/niw_norm': float(jnp.linalg.norm(niw_params[0])),  # S component
                    'pgm/mniw_norm': float(jnp.linalg.norm(mniw_params[0])),  # St component
                }
                log_dict.update(pgm_log_dict)
            
            # Compute convergence metrics if we have previous parameters
            if prev_pgm_params is not None:
                convergence_metrics = compute_pgm_convergence_metrics(state, epoch, prev_pgm_params)
                log_dict.update(convergence_metrics)
            
            # Update previous parameters for next iteration
            prev_pgm_params = pgm_params.copy()
            
            # Check for instability and warn
            if 'pgm/A_max_eigenval' in log_dict and log_dict['pgm/A_max_eigenval'] > 0.95:
                print(f"⚠️  WARNING: Unstable eigenvalues detected! Max eigenvalue: {log_dict['pgm/A_max_eigenval']:.3f}")
                print(f"   Unstable count: {log_dict['pgm/A_unstable_count']:.0f}, Stability ratio: {log_dict['pgm/A_stability_ratio']:.3f}")
        
        wandb.log(log_dict)
        
        train_rmse_epoch += rmse_train * bs
        running_loss += metrics['loss'].item() * bs
        epoch_losses.append(metrics['loss'].item())
            
        
    epoch_log_dict = {'train_nelbo_epoch': running_loss/N_train,
                      'train_nelbo_epoch_per_T': running_loss/(N_train*LDS_T),
                      'train_nelbo_epoch_per_TD': running_loss/(N_train*LDS_T*input_D),
                      'train_rmse_epoch': train_rmse_epoch/N_train}
    wandb.log(epoch_log_dict)
    
    losses.append(np.mean(epoch_losses))
    

    if np.isnan(metrics['loss'].item()):
        break
    if validate:
        # batch_metrics = []
        N_valid_batches = len(val_dataloader)
        valid_state = state
        validation_losses = []
        
        nino34_preds_epoch = []
        oni_preds_epoch = []
        nino34_trues_epoch = []
        oni_trues_epoch = []
        time_stamps_epoch = []
        for batch_num, batch in enumerate(val_dataloader):
            
            batch = process_batch(batch)
            # self.mask.shape:  torch.Size([29, 36, 43200]) -> (bs, 36, 43200)
            # self.sst.shape:  torch.Size([29, 36, 43200]) -> (bs, 36, 43200)
            # self.time_encodings.shape:  torch.Size([29, 36, 24]) -> (bs, 36, 24)
            # self.months.shape:  torch.Size([29, 36]) -> (bs, 36)
            # self.years.shape:  torch.Size([29, 36]) -> (bs, 36)
            # self.nino34.shape:  torch.Size([29, 36]) -> (bs, 36)
            # self.oni.shape:  torch.Size([29, 36]) -> (bs, 36)

            ##### Correct nino34 and oni
            (data, mask_sst, time_enc, month, year, nino34, oni) = batch
            bs = data.shape[0]         
            
            valid_state, _, valid_likelihood, metrics = eval_step(
                valid_state, data, month_encoding=time_enc, mask=None, 
                mask_sst=mask_sst, N_data=N_valid_batches * batch_size)
            
            del metrics['aux']
            # batch_metrics.append(metrics)
            validation_loss = metrics['loss'].item()
            validation_local_kl = metrics['local_kl'].item()
            validation_prior_kl = metrics['prior_kl'].item()
            validation_recon_loss = metrics['recon_loss'].item()
            rmse_test = metrics['rmse_test'].item()
            
            # Log summary statistics instead of full variance array to avoid wandb serialization warnings

            # mask_sst.shape:  (5, 36, 120, 260), 
            # valid_likelihood.scale.shape:  (31200,)
            if not(flatten_input):
                valid_scale = valid_likelihood.scale.reshape(h,w)[mask_sst[0,0]==1] #get output variance for valid locations
            else:
                # mask_sst.shape:  (batch, time, h*w) = (batch, time, 31200) - already flattened
                # valid_likelihood.scale.shape:  (31200,) - already flattened
                valid_scale = valid_likelihood.scale[mask_sst[0,0]==1] #get output variance for valid locations
            valid_output_variance_mean = jnp.mean(valid_scale).item()
            valid_output_variance_max = jnp.max(valid_scale).item()
            valid_output_variance_min = jnp.min(valid_scale).item()
            
            # Log validation metrics including variance statistics and l2_distance if available
            val_log_dict = {'valid_nelbo': validation_loss,
                            'valid_local_kl': validation_local_kl,
                            'valid_prior_kl': validation_prior_kl,
                            'valid_recon_loss': validation_recon_loss,
                            'valid_rmse': rmse_test,
                            'valid_output_variance_mean': valid_output_variance_mean,
                            'valid_output_variance_max': valid_output_variance_max,
                            'valid_output_variance_min': valid_output_variance_min}
            
            # Track PGM parameters during validation (less frequently)
            if epoch % 50 == 0:  # Log PGM params every 50 epochs during validation
                pgm_params = state.params['pgm']
                
                if 'S' in pgm_params:  # Standard parameterization
                    S_param = pgm_params['S']
                    loc_param = pgm_params['loc'] 
                    lam_param = pgm_params['lam']
                    nu_param = pgm_params['nu']
                    St_param = pgm_params['St']
                    M_param = pgm_params['M']
                    V_param = pgm_params['V']
                    nut_param = pgm_params['nut']
                    
                    # Log validation PGM statistics
                    val_pgm_log_dict = {
                        'val_pgm/S_norm': float(jnp.linalg.norm(S_param)),
                        'val_pgm/S_trace': float(jnp.trace(S_param)),
                        'val_pgm/loc_norm': float(jnp.linalg.norm(loc_param)),
                        'val_pgm/lam': float(lam_param),
                        'val_pgm/nu': float(nu_param),
                        'val_pgm/St_norm': float(jnp.linalg.norm(St_param)),
                        'val_pgm/M_norm': float(jnp.linalg.norm(M_param)),
                        'val_pgm/V_norm': float(jnp.linalg.norm(V_param)),
                        'val_pgm/nut_mean': float(jnp.mean(nut_param)) if nut_param.ndim > 0 else float(nut_param),
                    }
                    
                    # If cond_on_month=True, log month-specific statistics
                    if cond_on_month and St_param.ndim == 3:  # (12, D, D)
                        val_pgm_log_dict.update({
                            'val_pgm/St_month_std': float(jnp.std(jnp.linalg.norm(St_param, axis=(1,2)))),
                            'val_pgm/M_month_std': float(jnp.std(jnp.linalg.norm(M_param, axis=(1,2)))),
                            'val_pgm/V_month_std': float(jnp.std(jnp.linalg.norm(V_param, axis=(1,2)))),
                        })
                    
                    val_log_dict.update(val_pgm_log_dict)
            
            wandb.log(val_log_dict)
            
            validation_losses.append(validation_loss)
            test_rmse_epoch += rmse_test * bs
            running_loss_test += validation_loss * bs
            
            if epoch % 200 == 0 or epoch == cfg.num_epochs - 1:
                # start_time = time()
                # (bs, 36), (bs, 36), (bs, 36), (bs, 36), (bs, 36)
                (nino34_preds_batch, oni_preds_batch, nino34_trues_batch, oni_trues_batch, time_stamps_batch
                 ) = forecast_flat(
                    state, data, month, year, month_encoding_orig=time_enc, 
                    mask_orig=mask_sst, climatology=climatology, 
                    sst_min=min_sst, sst_max=max_sst, n_forecast=12, n_samples=100, LDS_T=LDS_T, 
                    save_path=f'{save_dir}/forecast_epoch_{epoch}.png', h=h, w=w,
                    lat_range=lat_range, lon_range=lon_range, 
                    start_month=(start_idx+2)%12,
                    epoch=epoch, batch_num=batch_num)
                nino34_preds_epoch.append(nino34_preds_batch)
                oni_preds_epoch.append(oni_preds_batch)
                nino34_trues_epoch.append(nino34_trues_batch)
                oni_trues_epoch.append(oni_trues_batch)
                time_stamps_epoch.append(time_stamps_batch)
                # end_time = time()
                # print(f"Forecasting time: {end_time - start_time} seconds") 
                # Clear memory immediately after forecasting
                del nino34_preds_batch, oni_preds_batch, nino34_trues_batch, oni_trues_batch, time_stamps_batch
                gc.collect()
                
            # Clear validation variables to save memory
            del data, mask_sst, time_enc, month, year, nino34, oni, metrics
                            
        # compute mean of metrics across each batch in epoch.
        # batch_metrics_np = jax.device_get(batch_metrics)
        val_epoch_log_dict = {'valid_nelbo_epoch': running_loss_test/N_valid,
                              'valid_nelbo_epoch_per_T': running_loss_test/(N_valid*LDS_T),
                              'valid_nelbo_epoch_per_TD': running_loss_test/(N_valid*LDS_T*input_D),
                              'valid_rmse_epoch': test_rmse_epoch/N_valid}
        wandb.log(val_epoch_log_dict)
        
                        
        if epoch % 200 == 0 or epoch == cfg.num_epochs - 1:
            
            # save_nino34_oni(nino34_preds_epoch, oni_preds_epoch, nino34_trues_epoch, 
            #                 oni_trues_epoch, time_stamps_epoch, epoch, save_dir)
            
            # Save PGM parameters for detailed analysis
            save_pgm_parameters(state, epoch, save_dir)
            
            # Only create visualizations on final epoch or every 1000 epochs to save time
            if (epoch == cfg.num_epochs - 1 or epoch % 1000 == 0):
                print(f"Creating visualizations for epoch {epoch}...")
                
                # Use first batch data for visualizations
                first_batch = process_batch(next(iter(val_dataloader)))
                (data_viz, mask_sst_viz, time_enc_viz, _, _, _, _) = first_batch
                
                if model_type == 'SLDS':
                    # 4. SLDS Discrete state probabilities
                    plot_discrete_state_probabilities(
                        state, data_viz[:,:24], month_encoding=time_enc_viz[:,:24], 
                        mask_sst=mask_sst_viz[:,:24],
                        save_path=f'{save_dir}/discrete_states_epoch_{epoch}.png'
                    )
                    
                    plot_latent_dynamics_by_state(
                        state, data_viz[:,:24], month_encoding=time_enc_viz[:,:24], 
                        mask_sst=mask_sst_viz[:,:24],
                        save_path=f'{save_dir}/latent_by_state_epoch_{epoch}.png'
                    )
                    
                    plot_state_transition_matrix(
                        state, data_viz[:,:24], month_encoding=time_enc_viz[:,:24], 
                        mask_sst=mask_sst_viz[:,:24],
                        save_path=f'{save_dir}/transition_matrix_epoch_{epoch}.png'
                    )                
                
                # 1. Latent space visualization
                visualize_latent_space(
                    state, data_viz[:,:24], month_encoding=time_enc_viz[:,:24], mask_sst=mask_sst_viz[:,:24],
                    save_path=f'{save_dir}/latent_space_epoch_{epoch}.png'
                )
                
                # 2. Data reconstructions (reduce n_samples to save memory)
                plot_reconstructions(
                    state, data_viz[:,:24], month_encoding=time_enc_viz[:,:24], mask_sst=mask_sst_viz[:,:24], n_samples=12,
                    save_path=f'{save_dir}/reconstructions_epoch_{epoch}.png', h=h, w=w
                )
                
                # 3. Temporal dynamics
                plot_temporal_dynamics(
                    state, data_viz[:,:24], month_encoding=time_enc_viz[:,:24], mask_sst=mask_sst_viz[:,:24],
                    save_path=f'{save_dir}/temporal_dynamics_epoch_{epoch}.png'
                )
                
                # Clear visualization data
                del data_viz, mask_sst_viz, time_enc_viz
                gc.collect()
        
        # Clear epoch-level variables
        del nino34_preds_epoch, oni_preds_epoch, nino34_trues_epoch, oni_trues_epoch, time_stamps_epoch
        del validation_losses
        gc.collect()



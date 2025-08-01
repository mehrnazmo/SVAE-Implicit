import os
import sys
svae_path='svae/SVAE-Implicit/svae'
sys.path.insert(0,svae_path)
from configs import setup
config_file = 'dataset/cfg_vae.ini'
(cfg_climate, 
 save_dir, 
 train_dataloader, 
 val_dataloader, 
 train_num, 
 val_num
 ) = setup(config_file)
import jax.numpy as jnp
import jax
print('jax.devices()', jax.devices())  # Should show GPU
print('jax.default_backend()', jax.default_backend())  # Should show 'gpu'
rng = jax.random.PRNGKey(1)
rng, subkey1 = jax.random.split(rng)

import numpy as np
import flax.linen as nn
from functools import partial
import torch
from random import randrange
import csv
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import optax
import numpy.random as npr
from tensorflow_probability.substrates.jax import distributions as tfd

# from models.SVAE_LDS_Simple import SVAE_LDS_Simple as SVAE_LDS
from models.SVAE_LDS import SVAE_LDS

from networks.layers import LayerNorm
from networks.dense import DenseNet
from networks.encoders import SigmaEncoder
from networks.decoders import Decoder, CalibratedNormal, VMVMF_Decoder, SigmaDecoder

from distributions.normal import nat_to_moment
import numpy as np
from train_utils import create_train_state, train_step_multisample, create_dual_train_state, save_state, load_state, save_params, load_params
from train_utils import make_eval_step_with_transform, make_train_step_with_transform
import wandb
import inspect
from memory_profiler import profile
from visualization_utils import (
    plot_training_curves, visualize_latent_space, 
    plot_reconstructions, plot_forecast, 
    plot_loss_components_analysis, plot_temporal_dynamics)

torch.manual_seed(1)
npr.seed(1)

def identity(inp):
    return inp

def get_beta_with_warmup(current_step):
    """
    Gets the beta value for the current epoch number
    """
    #max_beta = 1.0
    min_beta = 0.001
    cfg.warmup_steps = 500
    cfg.total_annealing_steps = 500
    
    if current_step < cfg.warmup_steps:
        return min_beta
    else:
        annealing_step = current_step - cfg.warmup_steps
        return min_beta + (1.0 - min_beta) * min(1.0, annealing_step / cfg.total_annealing_steps)

def process_batch(batch):
    tensors = tuple(jnp.array(t) for t in batch)
    return tensors

wandb_project = cfg_climate.get('User', 'wandb_project')
wandb_user = cfg_climate.get('User', 'wandb_user')
batch_size = cfg_climate.getint('DataFrame', 'batch_size')
# mode = 'disabled' in wandb.init to have it not track
wandb.init(project=wandb_project, entity=wandb_user, config=cfg_climate)
cfg = wandb.config

#(bs, 24, 26*90)
setup_batch = process_batch(next(iter(train_dataloader)))[0]
print(f"Setup batch shape: {setup_batch.shape}")

train_step = make_train_step_with_transform(identity)
eval_step = make_eval_step_with_transform(identity)

input_D = 90 * 26
cfg.num_epochs = 5000
cfg.lr_pgm = 1e-4 # 3e-6
cfg.lr_net = 1e-4
cfg.lr_decay_steps = 250000 # original 250000
cfg.lr_final_mult = 0.01 # original 0.01
learning_alg_pgm = 'sgd'
learning_alg_net = 'adam'

resnet=True
encoder_stage_sizes = [1,  1,   1,   1]
encoder_hidden_sizes = [128, 128, 128, 128]
decoder_stage_sizes = [1,    1,   1,   1]
decoder_hidden_sizes = [256, 256, 256, 256]
encoder_lstm_units = 0
encoder_lstm_layer = -1
cfg.latent_D = 50

network_type = 'dense'
encoder_activation = 'gelu'
decoder_activation = 'gelu'
last_layer_sigmoid = True
encoder_norm = 'layer'
decoder_norm = 'layer'
encoder_loc_norm = 'none'
encoder_scale_norm = 'none'
# encoder_skip_connection = False
# decoder_skip_connection = False
validate = True

encoder_month_embedding = False
decoder_month_embedding = False
cond_on_month = False


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
    'input_D': input_D
})


activations = dict(leaky_relu=nn.leaky_relu, tanh=jnp.tanh, gelu=nn.gelu)
norms = dict(batch=nn.BatchNorm, layer=LayerNorm, none=None)

encoder_network = partial(DenseNet, resnet=resnet, 
                          stage_sizes=encoder_stage_sizes, 
                          hidden_sizes=encoder_hidden_sizes,
                          activation=activations[encoder_activation],
                          norm_cls=norms[encoder_norm],
                          lstm_hidden_size=encoder_lstm_units,
                          lstm_layer=encoder_lstm_layer,
                         )

decoder_network = partial(DenseNet, resnet=resnet, 
                          stage_sizes=decoder_stage_sizes, 
                          hidden_sizes=decoder_hidden_sizes,
                          activation=activations[decoder_activation],
                          norm_cls=norms[decoder_norm],
                          last_layer_sigmoid=last_layer_sigmoid
                         )

encoder = partial(SigmaEncoder, network_cls=encoder_network,
                #   skip_connection=encoder_skip_connection,
                  loc_norm_cls=norms[encoder_loc_norm],
                  scale_norm_cls=norms[encoder_scale_norm],
                  month_embedding=encoder_month_embedding,
                 )


decoder = partial(SigmaDecoder, network_cls=decoder_network,
                  likelihood=tfd.Normal,
                #   skip_connection=decoder_skip_connection,
                  month_embedding=decoder_month_embedding,
                 )

model_builder = partial(SVAE_LDS, latent_D=cfg.latent_D, 
                        input_D = input_D,
                        encoder_cls=encoder, decoder_cls=decoder, 
                        pgm_hyperparameters=dict(cond_on_month=cond_on_month))

net_schedule = optax.cosine_decay_schedule(cfg.lr_net, cfg.lr_decay_steps, cfg.lr_final_mult)
pgm_schedule = optax.cosine_decay_schedule(cfg.lr_pgm, cfg.lr_decay_steps, cfg.lr_final_mult)

#model: SVAE_LDS
#state: DualTrainState keeps states such as 
# (params, batch_stats, opt_state_net, opt_state_pgm, ...)

# Create a dummy month tensor for initialization to ensure month_dense layer is created
dummy_month = jnp.zeros((setup_batch.shape[0], 24, 23))  # batch_size x 23

#model: SVAE_LDS
#state: DualTrainState keeps states such as 
# (params, batch_stats, opt_state_net, opt_state_pgm, ...)
model, state = create_dual_train_state(
    rng, net_schedule, pgm_schedule, 
    model_builder, setup_batch.shape, 
    learning_alg_net=learning_alg_net,
    learning_alg_pgm=learning_alg_pgm,
    month=dummy_month)

# Check the train_step function signature
print(f"train_step function signature: {inspect.signature(train_step)}")
print(f"Model: {model}")

# Log SLURM job ID to wandb
slurm_job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
wandb.log({'slurm_job_id': slurm_job_id})
print(f"SLURM Job ID: {slurm_job_id}")


losses = []
loss_history = []  # Store detailed loss history for analysis
prev_state = state
global_step = 0
for epoch in range(cfg.num_epochs):
    batch_metrics = []
    metrics = None
    N_batches = len(train_dataloader)
    running_loss = 0
    epoch_losses = []
    print('='*100)
    print('Epoch: ', epoch)
    wandb.log({'Epoch': epoch})
    for batch, step in zip(train_dataloader, range(N_batches)):
        batch = process_batch(batch)
        # # (100, 26, 90), (100, 26, 90), (100, 24), (100,), (100,)
        # data, mask, text_embedding, month, year = batch 
        
        #  data shape: (5, 24, 2340)
        # mask shape: (5, 24, 26, 90)
        # year shape: (5,)
        # month shape: (5, 24, 23)
        (data, mask_sst, month, year) = batch
        
        rng, subkey2 = jax.random.split(rng)
        # can add the local_kl_weight parameter, originally defaulted to 1.
        beta = get_beta_with_warmup(epoch)  
        wandb.log({'beta': beta})      

        # batch_pos, text_embedding
        #metrics:dict(recon_loss=outputs['recon_loss'], prior_kl=outputs['prior_kl'], 
        #local_kl = outputs['local_kl'], 
        #loss=outputs['recon_loss'] + outputs['prior_kl'] + outputs['local_kl'], 
        #aux=outputs['aux'][-1])
        new_state, metrics = train_step(
            state, data, month=month,
            mask=None,           # your original mask if needed
            mask_sst=mask_sst,   # the new SST mask from batch
            N_data=N_batches * batch_size,
            local_kl_weight=beta, 
            log_magnitudes=False
        )
        
        state = new_state
        running_loss += metrics['loss'].item()
        epoch_losses.append(metrics['loss'].item())
        global_step += 1
        
        metrics = jax.device_get(metrics)
        batch_metrics.append(metrics)
        
        # Store loss history for analysis
        loss_history.append({
            'loss': metrics['loss'].item(),
            'recon_loss': metrics['recon_loss'].item(),
            'prior_kl': metrics['prior_kl'].item(),
            'local_kl': metrics['local_kl'].item()
        })
        
        if np.isnan(metrics['loss'].item()):
            print(f"NAN loss! In Epoch: {epoch}")
            print(f"Loss: {metrics['loss'].item()}, Recon Loss: {metrics['recon_loss'].item()}, KL: {metrics['local_kl'].item()}")
            break
        
        prev_state = state
        prev_batch = batch
        
        # running_loss = metrics['loss'].item() if running_loss == 0.0 else (running_loss * 0.9 + metrics['loss'].item() * 0.1)

        # lossses for each step
        loss = metrics['loss'].item()
        local_kl = metrics['local_kl'].item()
        prior_kl = metrics['prior_kl'].item()
        recon_loss = metrics['recon_loss'].item()

        #(EXXT, mu, EXXNT), logZ, (Jf[:-1], hf[:-1], Js[1:], -hs[1:])
        # metrics['aux'][0][0]: (24, 50, 50)
        # metrics['aux'][0][1]: (24, 50, 50)
        # metrics['aux'][0][2]: (24, 50, 50)
        # metrics['aux'][1]: (5, 24, 50, 1)
        # metrics['aux'][2][0]: (23, 50, 50)
        # metrics['aux'][2][1]: (23, 50, 50)
        # metrics['aux'][2][2]: (23, 50, 50)
        EX_0 = metrics["aux"][0][1][:,0,:]
        tau_mu = metrics["aux"][2][0][1]
        tau = metrics["aux"][2][0][0]
        prior_mu = tau_mu/tau
        
        qp_mean_squared_dist = jnp.sum(((EX_0 - prior_mu[0])**2)) / (prior_mu[0]).shape[0]
        print(local_kl)

        
        wandb.log({'training_loss': loss, 'local_kl': local_kl, 
                   'recon_loss': recon_loss, 
                   'prior_kl': prior_kl,
                   'batch avg q_p squared dist': qp_mean_squared_dist})
        
        wandb.log({'recon_loss_avg': recon_loss/(26*90*24)})
        
    wandb.log({'training_loss_epoch': running_loss})
    losses.append(np.mean(epoch_losses))
    

    if np.isnan(metrics['loss'].item()):
        break
    if validate:
        batch_metrics = []
        N_valid_batches = len(val_dataloader)
        valid_state = state
        validation_losses = []
        for batch in val_dataloader:
            batch = process_batch(batch)
            # data, mask, text_embedding, month, year = batch # (100, 26, 90), (100, 26, 90), (100, 24), (100,), (100,)
            (data, mask_sst, month, year) = batch
            valid_state, _, _, metrics = eval_step(
                valid_state, data, month=month, mask=None, mask_sst=mask_sst, N_data=N_valid_batches * batch_size)
            
                # valid_state, data, text_embedding, batch_pos, 
                # mask, N_data=N_valid_batches * 128)
            del metrics['aux']
            batch_metrics.append(metrics)
            validation_loss = metrics['loss'].item()
            validation_losses.append(validation_loss)
            wandb.log({'validation_loss': validation_loss})
        # compute mean of metrics across each batch in epoch.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {}
        for k in batch_metrics_np[0].keys():
            try:
                epoch_metrics_np['valid_' + k] = np.mean([metrics[k] for metrics in batch_metrics_np])
            except:
                pass
        wandb.log(epoch_metrics_np)

    # Periodically create and log visualizations
    if epoch % 1000 == 0 or epoch == cfg.num_epochs - 1:  # Every 10 epochs
        print(f"Creating visualizations for epoch {epoch}...")
        
        # # 1. Training loss curves
        # plot_training_curves(
        #     {'training_loss': losses,
        #      'recon_loss': [m['recon_loss'] for m in batch_metrics],
        #      'local_kl': [m['local_kl'] for m in batch_metrics],
        #      'prior_kl': [m['prior_kl'] for m in batch_metrics],
        #      'recon_loss_avg': [m['recon_loss']/(26*90*24) for m in batch_metrics]},
        #     save_path=f'{save_dir}/training_curves_epoch_{epoch}.png'
        # )
        
        # 2. Latent space visualization
        visualize_latent_space(
            state, data, month=month, mask_sst=mask_sst,
            save_path=f'{save_dir}/latent_space_epoch_{epoch}.png'
        )
        
        # 3. Data reconstructions
        plot_reconstructions(
            state, data, month=month, mask_sst=mask_sst, n_samples=24,
            save_path=f'{save_dir}/reconstructions_epoch_{epoch}.png'
        )
        
        # # 4. Temporal dynamics
        plot_temporal_dynamics(
            state, data, month=month, mask_sst=mask_sst,
            save_path=f'{save_dir}/temporal_dynamics_epoch_{epoch}.png'
        )
        
        # 5. Forecasting capabilities
        plot_forecast(
            state, data, month=month, mask_sst=mask_sst, n_forecast=48,
            save_path=f'{save_dir}/forecast_epoch_{epoch}.png'
        )
        
        # # # 6. Loss component analysis
        # plot_loss_components_analysis(
        #     loss_history,
        #     save_path=f'{save_dir}/loss_analysis_epoch_{epoch}.png'
        # )
        
        # # Log to wandb
        # wandb.log({
        #     "training_curves": wandb.Image(f'{save_dir}/training_curves_epoch_{epoch}.png'),
        #     "latent_space": wandb.Image(f'{save_dir}/latent_space_epoch_{epoch}.png'),
        #     "reconstructions": wandb.Image(f'{save_dir}/reconstructions_epoch_{epoch}.png'),
        #     "temporal_dynamics": wandb.Image(f'{save_dir}/temporal_dynamics_epoch_{epoch}.png'),
        #     "forecast": wandb.Image(f'{save_dir}/forecast_epoch_{epoch}.png'),
        #     "loss_analysis": wandb.Image(f'{save_dir}/loss_analysis_epoch_{epoch}.png')
        # })

# def plot_training_curves(losses_dict, save_path=None):
#     """
#     Plot training curves for different loss components.
    
#     Args:
#         losses_dict: Dictionary with keys as loss names and values as lists of loss values
#         save_path: Path to save the plot (optional)
#     """
#     import matplotlib.pyplot as plt
    
#     plt.figure(figsize=(12, 8))
    
#     for loss_name, loss_values in losses_dict.items():
#         if loss_values:  # Only plot if we have values
#             plt.plot(loss_values, label=loss_name, alpha=0.8)
    
#     plt.xlabel('Training Steps')
#     plt.ylabel('Loss Value')
#     plt.title('Training Loss Curves')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.yscale('log')  # Log scale often helps visualize loss trends
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Training curves saved to {save_path}")
    
#     plt.show()
#     plt.close()

# def visualize_latent_space(state, data, mask_sst=None, save_path=None, method='pca'):
#     """
#     Visualize the latent space using dimensionality reduction.
    
#     Args:
#         state: Current model state
#         data: Input data batch
#         mask_sst: SST mask for the data
#         save_path: Path to save the plot (optional)
#         method: 'pca' or 'tsne' for dimensionality reduction
#     """
#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA
#     from sklearn.manifold import TSNE
    
#     # Get latent representations
#     model = state.model
#     params = state.params
    
#     # Run inference to get latent variables
#     likelihood, prior_kl, local_kl, aux = model.apply(
#         params, data, mask=None, eval_mode=True
#     )
    
#     # Extract latent variables z from aux
#     z = aux[0]  # Shape: (batch_size, time_steps, latent_dim)
    
#     # Flatten time dimension for visualization
#     z_flat = z.reshape(-1, z.shape[-1])  # (batch_size * time_steps, latent_dim)
    
#     # Apply dimensionality reduction
#     if method == 'pca':
#         reducer = PCA(n_components=2)
#         z_reduced = reducer.fit_transform(z_flat)
#         title = f'Latent Space (PCA) - Explained variance: {reducer.explained_variance_ratio_.sum():.3f}'
#     elif method == 'tsne':
#         reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_flat)-1))
#         z_reduced = reducer.fit_transform(z_flat)
#         title = 'Latent Space (t-SNE)'
#     else:
#         raise ValueError("method must be 'pca' or 'tsne'")
    
#     # Create scatter plot
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(z_reduced[:, 0], z_reduced[:, 1], alpha=0.6, s=20)
#     plt.xlabel(f'{method.upper()} Component 1')
#     plt.ylabel(f'{method.upper()} Component 2')
#     plt.title(title)
#     plt.grid(True, alpha=0.3)
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Latent space visualization saved to {save_path}")
    
#     plt.show()
#     plt.close()

# def plot_reconstructions(state, data, mask_sst=None, save_path=None, n_samples=3):
#     """
#     Plot original data vs reconstructions to assess model performance.
    
#     Args:
#         state: Current model state
#         data: Input data batch
#         mask_sst: SST mask for the data
#         save_path: Path to save the plot (optional)
#         n_samples: Number of samples to visualize
#     """
#     import matplotlib.pyplot as plt
    
#     # Get model predictions
#     model = state.model
#     params = state.params
    
#     likelihood, prior_kl, local_kl, aux = model.apply(
#         params, data, mask=None, eval_mode=True
#     )
    
#     # Get reconstructions from likelihood
#     if hasattr(likelihood, 'mean'):
#         reconstructions = likelihood.mean()
#     else:
#         reconstructions = likelihood
    
#     # Select samples to visualize
#     n_samples = min(n_samples, data.shape[0])
    
#     # Reshape data for visualization (assuming spatial structure)
#     # Original data shape: (batch, time, features) -> (batch, time, 26, 90)
#     data_reshaped = data[:n_samples].reshape(n_samples, data.shape[1], 26, 90)
#     recon_reshaped = reconstructions[:n_samples].reshape(n_samples, data.shape[1], 26, 90)
    
#     fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
#     if n_samples == 1:
#         axes = axes.reshape(1, -1)
    
#     for i in range(n_samples):
#         # Plot original data (middle time step)
#         mid_time = data_reshaped.shape[1] // 2
#         im1 = axes[i, 0].imshow(data_reshaped[i, mid_time], cmap='viridis')
#         axes[i, 0].set_title(f'Sample {i+1}: Original (t={mid_time})')
#         plt.colorbar(im1, ax=axes[i, 0])
        
#         # Plot reconstruction
#         im2 = axes[i, 1].imshow(recon_reshaped[i, mid_time], cmap='viridis')
#         axes[i, 1].set_title(f'Sample {i+1}: Reconstruction (t={mid_time})')
#         plt.colorbar(im2, ax=axes[i, 1])
        
#         # Plot difference
#         diff = data_reshaped[i, mid_time] - recon_reshaped[i, mid_time]
#         im3 = axes[i, 2].imshow(diff, cmap='RdBu_r', vmin=-diff.max(), vmax=diff.max())
#         axes[i, 2].set_title(f'Sample {i+1}: Difference (t={mid_time})')
#         plt.colorbar(im3, ax=axes[i, 2])
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Reconstructions saved to {save_path}")
    
#     plt.show()
#     plt.close()


# def plot_forecast(state, data, mask_sst=None, save_path=None, n_forecast=10, n_samples=2):
#     """
#     Plot forecasting capabilities of the model.
    
#     Args:
#         state: Current model state
#         data: Input data batch
#         mask_sst: SST mask for the data
#         save_path: Path to save the plot (optional)
#         n_forecast: Number of time steps to forecast
#         n_samples: Number of samples to visualize
#     """
#     import matplotlib.pyplot as plt
    
#     # Get model predictions with forecasting
#     model = state.model
#     params = state.params
    
#     likelihood, prior_kl, local_kl, aux = model.apply(
#         params, data, mask=None, eval_mode=True, n_forecast=n_forecast
#     )
    
#     # Get reconstructions and forecasts
#     if hasattr(likelihood, 'mean'):
#         predictions = likelihood.mean()
#     else:
#         predictions = likelihood
    
#     # Select samples to visualize
#     n_samples = min(n_samples, data.shape[0])
    
#     # Reshape data for visualization
#     data_reshaped = data[:n_samples].reshape(n_samples, data.shape[1], 26, 90)
#     pred_reshaped = predictions[:n_samples].reshape(n_samples, predictions.shape[1], 26, 90)
    
#     # Plot middle spatial location over time
#     mid_lat, mid_lon = 13, 45  # Middle of 26x90 grid
    
#     fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
#     if n_samples == 1:
#         axes = [axes]
    
#     time_steps = range(data_reshaped.shape[1] + n_forecast)
    
#     for i in range(n_samples):
#         # Original data
#         original_series = data_reshaped[i, :, mid_lat, mid_lon]
#         axes[i].plot(time_steps[:len(original_series)], original_series, 
#                     'b-', label='Original', linewidth=2)
        
#         # Forecast
#         forecast_series = pred_reshaped[i, :, mid_lat, mid_lon]
#         axes[i].plot(time_steps[len(original_series):], forecast_series[len(original_series):], 
#                     'r--', label='Forecast', linewidth=2)
        
#         # Reconstruction
#         recon_series = forecast_series[:len(original_series)]
#         axes[i].plot(time_steps[:len(original_series)], recon_series, 
#                     'g-', alpha=0.7, label='Reconstruction', linewidth=1.5)
        
#         axes[i].axvline(x=len(original_series)-1, color='k', linestyle=':', alpha=0.5, label='Forecast Start')
#         axes[i].set_xlabel('Time Steps')
#         axes[i].set_ylabel('SST Value')
#         axes[i].set_title(f'Sample {i+1}: SST Forecast at Location ({mid_lat}, {mid_lon})')
#         axes[i].legend()
#         axes[i].grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Forecast visualization saved to {save_path}")
    
#     plt.show()
#     plt.close()

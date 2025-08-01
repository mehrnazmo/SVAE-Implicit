import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import wandb
from train_utils import eval_step_forecast
import imageio
import matplotlib.animation as animation


def plot_training_curves(metrics_dict, save_path=None):
    """Plot training curves including loss components and KL terms"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot total loss
    axes[0,0].plot(metrics_dict['training_loss'], label='Training')
    if 'validation_loss' in metrics_dict:
        axes[0,0].plot(metrics_dict['validation_loss'], label='Validation')
    axes[0,0].set_title('Total Loss')
    axes[0,0].set_xlabel('Steps')
    axes[0,0].legend()
    
    # Plot loss components
    axes[0,1].plot(metrics_dict['recon_loss'], label='Reconstruction')
    axes[0,1].plot(metrics_dict['local_kl'], label='Local KL')
    axes[0,1].plot(metrics_dict['prior_kl'], label='Prior KL')
    axes[0,1].set_title('Loss Components')
    axes[0,1].set_xlabel('Steps')
    axes[0,1].legend()
    
    # Plot KL terms
    axes[1,0].plot(metrics_dict['local_kl'], label='Local KL')
    axes[1,0].set_title('KL Divergence Terms')
    axes[1,0].set_xlabel('Steps')
    axes[1,0].legend()
    
    # Plot reconstruction loss per dimension
    axes[1,1].plot(metrics_dict['recon_loss_avg'], label='Per-dim Reconstruction')
    axes[1,1].set_title('Average Reconstruction Loss')
    axes[1,1].set_xlabel('Steps')
    axes[1,1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_latent_space(model_state, data_batch, month=None, mask_sst=None, save_path=None):
    """Visualize latent space representations"""
    # Create a PRNG key for sampling
    rng = jax.random.PRNGKey(0)  # Using a fixed seed for reproducibility
    
    # Get latent representations
    likelihood, prior_kl, local_kl, aux = model_state.apply_fn(
        {'params': model_state.params, "batch_stats": model_state.batch_stats},
        data_batch, 
        month=month,
        eval_mode=True, 
        mask=None,
        rngs={'sampler': rng}
    )
    
    # Extract latent variables
    z = aux[0]  # Shape: (batch_size, time_steps, latent_dim)
    
    # Plot latent trajectories
    plt.figure(figsize=(15, 5))
    
    # Plot first few dimensions of latent space over time
    for i in range(min(5, z.shape[-1])):
        plt.plot(z[0, :, i], label=f'Dim {i+1}')
    
    plt.title('Latent Space Trajectories')
    plt.xlabel('Time Steps')
    plt.ylabel('Latent Value')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_reconstructions(model_state, data_batch, month=None, mask_sst=None, n_samples=5, save_path=None):
    """Plot original data vs reconstructions"""
    # Create a PRNG key for sampling
    rng = jax.random.PRNGKey(0)  # Using a fixed seed for reproducibility
    
    # Get reconstructions
    likelihood, _, _, _ = model_state.apply_fn(
        {'params': model_state.params, "batch_stats": model_state.batch_stats},
        data_batch, 
        month=month,
        eval_mode=True, 
        mask=None,
        rngs={'sampler': rng}
    )
    
    # Sample from the likelihood
    #batch_data_shape: (3, 24, 2340)
    recon = likelihood.sample(seed=jax.random.PRNGKey(0)) #(3, 24, 2340)
    recon_to_plot = recon.reshape(-1, data_batch.shape[1], 26, 90)
    data_batch_to_plot = data_batch.reshape(-1, data_batch.shape[1], 26, 90)
    
    
    # Plot original vs reconstruction
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3*n_samples))
    
    for i in range(n_samples):
        # Original
        im0 = axes[i,0].imshow(data_batch_to_plot[0,i,:,:], aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
        axes[i,0].set_title(f'Original {i+1}')
        plt.colorbar(im0, ax=axes[i,0])
        
        # Reconstruction
        im1 = axes[i,1].imshow(recon_to_plot[0,i,:,:], aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
        axes[i,1].set_title(f'Reconstruction {i+1}')
        plt.colorbar(im1, ax=axes[i,1])
        
        # Difference
        diff = data_batch_to_plot[0,i,:,:] - recon_to_plot[0,i,:,:]
        im2 = axes[i,2].imshow(diff, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axes[i,2].set_title(f'Difference {i+1}')
        plt.colorbar(im2, ax=axes[i,2])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_forecast(model_state, data_batch, month=None, mask_sst=None, n_forecast=10, save_path=None):
    """Plot forecasted trajectories"""
    # data_batch: (3, 24, 2340)
    # Create a PRNG key for sampling
    rng = jax.random.PRNGKey(0)  # Using a fixed seed for reproducibility
    
    # Get forecasts
    _, _, likelihood, aux = eval_step_forecast(
        model_state, 
        data_batch, 
        n_forecast,  # n_forecast must be positional argument (static_argnums=3)
        month=month,
        mask=None
    )

    # Sample from forecast distribution
    forecast = likelihood.sample(seed=jax.random.PRNGKey(0)) #(3, 34, 2340)
    
    # INSERT_YOUR_REWRITE_HERE
    forecast_videos = forecast.reshape(-1, forecast.shape[1], 26, 90)  # (3, 34, 26, 90)
    save_dir = os.path.dirname(save_path)
    for i in range(forecast_videos.shape[0]):
        video_path = f"{save_dir}/forecast_video_{i}.mp4"
        save_frames_to_mp4_matplotlib(forecast_videos[i, :, :, :], video_path, fps=2)
        
        
        # frames = []
        # for t in range(forecast_videos.shape[1]):            
        #     frame = forecast_videos[i, t, :, :]
        #     # Apply coolwarm colormap directly to the raw values (no normalization)
        #     cmap = plt.get_cmap('coolwarm')
        #     frame_rgb = (cmap(frame)[..., :3] * 255).astype(np.uint8)  # Drop alpha, scale to 0-255
        #     frames.append(frame_rgb)
        # video_path = None
        # if save_path:
        #     # Save as e.g. forecast_video_0.mp4 in the same directory as save_path
        #     import os
        #     base, ext = os.path.splitext(save_path)
        #     video_path = f"{base}_video_{i}.mp4"
        # else:
        #     video_path = f"forecast_video_{i}.mp4"
        # imageio.mimsave(video_path, frames, fps=3)
    
    
    # # Plot original data and forecasts
    # plt.figure(figsize=(15, 5))
    
    # # Plot original data
    # plt.plot(data_batch[0, :, 0], 'b-', label='Original')
    
    # # Plot forecast
    # forecast_mean = jnp.mean(forecast, axis=0) #(34, 2340)
    # forecast_std = jnp.std(forecast, axis=0) #(34, 2340)
    # t = np.arange(len(data_batch[0])) #(24,)
    # plt.plot(t[-n_forecast:], forecast_mean[:, 0], 'r-', label='Forecast')
    # plt.fill_between(t[-n_forecast:],
    #                 forecast_mean[:, 0] - 2*forecast_std[:, 0],
    #                 forecast_mean[:, 0] + 2*forecast_std[:, 0],
    #                 color='r', alpha=0.2)
    
    # plt.title('Forecast')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Value')
    # plt.legend()
    
    # if save_path:
    #     plt.savefig(save_path)
    # plt.close()
    
    

def plot_temporal_dynamics(state, data, month=None, mask_sst=None, save_path=None, n_samples=3, n_latent_dims=5):
    """
    Plot temporal dynamics of latent variables over time.
    
    Args:
        state: Current model state
        data: Input data batch
        mask_sst: SST mask for the data
        save_path: Path to save the plot (optional)
        n_samples: Number of samples to visualize
        n_latent_dims: Number of latent dimensions to plot
    """
    
    rng = jax.random.PRNGKey(0) # Using a fixed seed for reproducibility
    
    likelihood, prior_kl, local_kl, aux = state.apply_fn(
        {'params': state.params, "batch_stats": state.batch_stats},
        data, 
        month=month,
        eval_mode=True,
        mask=None,
        rngs={'sampler': rng}
    )
    
    # Extract latent variables z from aux
    z = aux[0]  # Shape: (batch_size, time_steps, latent_dim)
    
    # Select samples and dimensions to visualize
    n_samples = min(n_samples, z.shape[0])
    n_latent_dims = min(n_latent_dims, z.shape[2])
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    time_steps = range(z.shape[1])
    
    for i in range(n_samples):
        for j in range(n_latent_dims):
            axes[i].plot(time_steps, z[i, :, j], 
                        label=f'Latent dim {j+1}', alpha=0.8, linewidth=2)
        
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Latent Variable Value')
        axes[i].set_title(f'Sample {i+1}: Temporal Dynamics of Latent Variables')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal dynamics saved to {save_path}")
    
    plt.show()
    plt.close()




def save_frames_to_mp4_matplotlib(frames, output_path, fps=30):
    """
    Save frames using matplotlib animation
    """
    fig, ax = plt.subplots()
    
    def animate(frame_num):
        ax.clear()
        ax.imshow(frames[frame_num], cmap='coolwarm', vmin=0, vmax=1)
        ax.set_title(f'Frame {frame_num}')
        return ax,
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                 interval=1000/fps, blit=False)
    
    # Save animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Video saved to {output_path}")


def plot_loss_components_analysis(loss_history, save_path=None):
    """
    Create detailed analysis of loss components and their contributions.
    
    Args:
        loss_history: List of dictionaries containing loss components for each step
        save_path: Path to save the plot (optional)
    """
    
    if not loss_history:
        print("No loss history provided")
        return
    
    # Extract loss components
    steps = range(len(loss_history))
    recon_losses = [h.get('recon_loss', 0) for h in loss_history]
    prior_kls = [h.get('prior_kl', 0) for h in loss_history]
    local_kls = [h.get('local_kl', 0) for h in loss_history]
    total_losses = [h.get('loss', 0) for h in loss_history]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Individual loss components
    axes[0, 0].plot(steps, recon_losses, label='Reconstruction Loss', alpha=0.8)
    axes[0, 0].plot(steps, prior_kls, label='Prior KL', alpha=0.8)
    axes[0, 0].plot(steps, local_kls, label='Local KL', alpha=0.8)
    axes[0, 0].plot(steps, total_losses, label='Total Loss', alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Loss Value')
    axes[0, 0].set_title('Loss Components Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Loss component ratios
    recon_ratios = np.array(recon_losses) / np.array(total_losses)
    prior_ratios = np.array(prior_kls) / np.array(total_losses)
    local_ratios = np.array(local_kls) / np.array(total_losses)
    
    axes[0, 1].plot(steps, recon_ratios, label='Reconstruction Ratio', alpha=0.8)
    axes[0, 1].plot(steps, prior_ratios, label='Prior KL Ratio', alpha=0.8)
    axes[0, 1].plot(steps, local_ratios, label='Local KL Ratio', alpha=0.8)
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('Ratio')
    axes[0, 1].set_title('Loss Component Ratios')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Moving averages
    window = min(50, len(steps) // 10)
    if window > 1:
        recon_ma = np.convolve(recon_losses, np.ones(window)/window, mode='valid')
        prior_ma = np.convolve(prior_kls, np.ones(window)/window, mode='valid')
        local_ma = np.convolve(local_kls, np.ones(window)/window, mode='valid')
        total_ma = np.convolve(total_losses, np.ones(window)/window, mode='valid')
        
        ma_steps = steps[window-1:]
        axes[1, 0].plot(ma_steps, recon_ma, label=f'Reconstruction (MA-{window})', alpha=0.8)
        axes[1, 0].plot(ma_steps, prior_ma, label=f'Prior KL (MA-{window})', alpha=0.8)
        axes[1, 0].plot(ma_steps, local_ma, label=f'Local KL (MA-{window})', alpha=0.8)
        axes[1, 0].plot(ma_steps, total_ma, label=f'Total (MA-{window})', alpha=0.8, linewidth=2)
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Loss Value')
        axes[1, 0].set_title(f'Moving Average Loss Components (Window={window})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Plot 4: Loss statistics
    recent_losses = loss_history[-min(100, len(loss_history)):]
    recent_recon = [h.get('recon_loss', 0) for h in recent_losses]
    recent_prior = [h.get('prior_kl', 0) for h in recent_losses]
    recent_local = [h.get('local_kl', 0) for h in recent_losses]
    
    loss_names = ['Reconstruction', 'Prior KL', 'Local KL']
    loss_values = [np.mean(recent_recon), np.mean(recent_prior), np.mean(recent_local)]
    loss_stds = [np.std(recent_recon), np.std(recent_prior), np.std(recent_local)]
    
    bars = axes[1, 1].bar(loss_names, loss_values, yerr=loss_stds, capsize=5, alpha=0.7)
    axes[1, 1].set_ylabel('Loss Value')
    axes[1, 1].set_title('Recent Loss Component Statistics')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, loss_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss analysis saved to {save_path}")
    
    plt.show()
    plt.close()


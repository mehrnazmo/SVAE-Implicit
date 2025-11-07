import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import wandb
# from train_utils import eval_step_forecast
import imageio
import matplotlib.animation as animation
import xarray as xr
import pandas as pd

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

def visualize_latent_space(model_state, data_batch, month_encoding=None, mask_sst=None, save_path=None):
    """Visualize latent space representations"""
    # Create a PRNG key for sampling
    rng = jax.random.PRNGKey(0)  # Using a fixed seed for reproducibility
        
    # Get latent representations
    likelihood, prior_kl, local_kl, aux = model_state.apply_fn(
        {'params': model_state.params, "batch_stats": model_state.batch_stats},
        data_batch, 
        month_encoding=month_encoding,
        eval_mode=True, 
        mask=None,
        rngs={'sampler': rng}
    )
    
    # Extract latent variables
    z = aux[0]  # Shape: (batch_size, time_steps, latent_dim)

    print('-'*100)
    print('z', z.shape)
    print('-'*100)
    # Plot latent trajectories
    plt.figure(figsize=(15, 5))
    
    # Plot first few dimensions of latent space over time
    for i in range(z.shape[-1]):
        plt.plot(z[0, :, i])
    
    plt.title('Latent Space Trajectories')
    plt.xlabel('Time Steps')
    plt.ylabel('Latent Value')
    # plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
    plt.figure(figsize=(15, 5))
    plt.imshow(z[0].transpose(1, 0))
    plt.colorbar()
    plt.title('Latent Space Trajectories')
    plt.xlabel('Time Steps')
    plt.ylabel('Latent Value')
    plt.savefig(save_path.replace('.png', '_heatmap.png'))
    plt.close()
    
    
    

def plot_reconstructions(model_state, data_batch, month_encoding=None, mask_sst=None, n_samples=5, save_path=None, h=26, w=90):
    """Plot original data vs reconstructions"""
    # Create a PRNG key for sampling
    rng = jax.random.PRNGKey(0)  # Using a fixed seed for reproducibility
    
    # Get reconstructions
    likelihood, _, _, _ = model_state.apply_fn(
        {'params': model_state.params, "batch_stats": model_state.batch_stats},
        data_batch, 
        month_encoding=month_encoding,
        eval_mode=True, 
        mask=None,
        rngs={'sampler': rng}
    )
    
    # Sample from the likelihood
    #batch_data_shape: (3, 24, 2340)
    # recon = likelihood.sample(seed=jax.random.PRNGKey(0)) #(3, 24, 2340)
    recon = likelihood.mean() #(3, 24, 2340)
    recon_to_plot = recon.reshape(-1, data_batch.shape[1], h, w)
    data_batch_to_plot = data_batch.reshape(-1, data_batch.shape[1], h, w)
    mask = mask_sst.reshape(-1, mask_sst.shape[1], h, w)[0][0].astype(bool)
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_start, row_end = np.where(rows)[0][[0, -1]]
    col_start, col_end = np.where(cols)[0][[0, -1]]


    # Plot original vs reconstruction
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
    
    for i in range(n_samples):
        # Original
        orig = data_batch_to_plot[0,i,row_start:row_end+1, col_start:col_end+1]
        im0 = axes[i,0].imshow(orig, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
        axes[i,0].set_title(f'Original {i+1}')
        plt.colorbar(im0, ax=axes[i,0])
        
        # Reconstruction
        recon = recon_to_plot[0,i,row_start:row_end+1, col_start:col_end+1]
        im1 = axes[i,1].imshow(recon, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
        axes[i,1].set_title(f'Reconstruction {i+1}')
        plt.colorbar(im1, ax=axes[i,1])
        
        # Difference
        diff = (orig - recon)
        im2 = axes[i,2].imshow(diff, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axes[i,2].set_title(f'Difference {i+1}')
        plt.colorbar(im2, ax=axes[i,2])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

    

def save_frames_to_mp4_matplotlib(frames, mask_2d, output_path, time_stamps=None,fps=30):
    """
    Save frames using matplotlib animation
    """
    # rows = np.any(mask_2d, axis=1)
    # cols = np.any(mask_2d, axis=0)
    # row_start, row_end = np.where(rows)[0][[0, -1]]
    # col_start, col_end = np.where(cols)[0][[0, -1]]
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    def animate(frame_num, time_stamps):
        cropped_frame = frames[1][frame_num]
        # [row_start:row_end+1, col_start:col_end+1]
        ax[1].clear()
        ax[1].imshow(cropped_frame, cmap='coolwarm', vmin=0, vmax=1)
        if time_stamps is not None:
            ax[1].set_title(f'Forecast {time_stamps[frame_num]}')
        else:
            ax[1].set_title(f'Forecast {frame_num+1}')
        ax[0].clear()
        if frame_num < frames[0].shape[0]:
            frame_to_plot = frames[0][frame_num]
        else:
            frame_to_plot = frames[0][-1]
        ax[0].imshow(frame_to_plot,
                    #  [row_start:row_end+1, col_start:col_end+1], 
                     cmap='coolwarm', vmin=0, vmax=1)
        if time_stamps is not None:
            ax[0].set_title(f'Original {time_stamps[frame_num]}')
        else:
            ax[0].set_title(f'Original {frame_num+1}')
        return ax,
    anim = animation.FuncAnimation(fig, animate, frames=len(frames[1]), 
                                 interval=1000/fps, blit=False, fargs=(time_stamps,))
    
    # Save animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Video saved to {output_path}")
    
    
    
        
def visualize_forecast(forecast_video, data_2d, mask_2d, time_stamps, save_path, z, sample_idx):
    # Create video for this sample
    video_path = f"{os.path.dirname(save_path)}/forecast_video_{time_stamps[0]}.mp4"
    save_frames_to_mp4_matplotlib([data_2d, forecast_video], mask_2d, video_path, time_stamps, fps=2) 
    
    # Plot latent trajectories for this sample
    if sample_idx == 0:
        # z = aux[0]  # shape: (1, T+n_forecast, D) or (1,100,36,50)
        plt.figure(figsize=(30, 10))
        if z.ndim == 3:
            for j in range(z.shape[-1]):
                plt.plot(z[0, :, j])  # z[0] since batch_size=1
        elif z.ndim == 4:
            zdim = z.shape[-1]
            colors = plt.cm.tab10(jnp.linspace(0, 1, zdim))
            for j in range(zdim):
                z_trajectories = z[0, :, :, j]  # shape: (100, 36)
                mean_traj = z_trajectories.mean(axis=0)
                min_traj = z_trajectories.min(axis=0)
                max_traj = z_trajectories.max(axis=0)
                plt.fill_between(np.arange(z_trajectories.shape[1]), min_traj, max_traj, alpha=0.3, color=colors[j])
                plt.plot(mean_traj, label=f'Latent dim {j}', color=colors[j])
                # plt.plot(z[0, :, :, j])  # z[0] since batch_size=1
        plt.xlabel('Time Steps')
        plt.ylabel('Latent Value')
        plt.savefig(save_path.replace('.png', f'_latent_{time_stamps[0]}.png'))
        plt.close()
        del z
    
    # Clear variables to free memory
    del forecast_video, data_2d
    # del aux
    


    

def plot_temporal_dynamics(state, data, month_encoding=None, mask_sst=None, save_path=None, n_samples=3, n_latent_dims=5):
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
        month_encoding=month_encoding,
        eval_mode=True,
        mask=None,
        rngs={'sampler': rng}
    )
    
    # Extract latent variables z from aux
    z = aux[0]  # Shape: (batch_size, time_steps, latent_dim)
    
    # Select samples and dimensions to visualize
    n_samples = min(n_samples, z.shape[0])
    n_latent_dims = z.shape[2]
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    time_steps = range(z.shape[1])
    
    for i in range(n_samples):
        for j in range(n_latent_dims):
            axes[i].plot(time_steps, z[i, :, j], alpha=0.8)
        
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Latent Variable Value')
        axes[i].set_title(f'Sample {i+1}: Temporal Dynamics of Latent Variables')
        # axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal dynamics saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_discrete_state_probabilities(state, data, month_encoding=None, mask_sst=None, save_path=None, n_samples=3):
    """
    Plot discrete state probabilities over time for SLDS model.
    
    Args:
        state: Current model state
        data: Input data batch
        month_encoding: Month encoding tensor
        mask_sst: SST mask for the data
        save_path: Path to save the plot (optional)
        n_samples: Number of samples to visualize
    """
    
    rng = jax.random.PRNGKey(0)
    
    likelihood, prior_kl, local_kl, aux = state.apply_fn(
        {'params': state.params, "batch_stats": state.batch_stats},
        data, 
        month_encoding=month_encoding,
        eval_mode=True,
        mask=None,
        rngs={'sampler': rng}
    )
    
    # Extract categorical expected stats from aux
    # aux structure: (z, sur_loss) + (gaus_expected_stats, cat_expected_stats)
    cat_expected_stats = aux[3]  # Shape: (batch_size, time_steps, K)
    
    n_samples = min(n_samples, cat_expected_stats.shape[0])
    K = cat_expected_stats.shape[2]  # Number of discrete states
    time_steps = range(cat_expected_stats.shape[1])
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        # Plot probability of each discrete state over time
        for k in range(K):
            axes[i].plot(time_steps, cat_expected_stats[i, :, k], 
                        label=f'State {k+1}', alpha=0.8, linewidth=2)
        
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('State Probability')
        axes[i].set_title(f'Sample {i+1}: Discrete State Probabilities Over Time')
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Discrete state probabilities saved to {save_path}")
    
    plt.close()


def plot_discrete_state_heatmap(state, data, month_encoding=None, mask_sst=None, save_path=None, n_samples=3):
    """
    Plot discrete state probabilities as heatmaps for SLDS model.
    
    Args:
        state: Current model state
        data: Input data batch
        month_encoding: Month encoding tensor
        mask_sst: SST mask for the data
        save_path: Path to save the plot (optional)
        n_samples: Number of samples to visualize
    """
    
    rng = jax.random.PRNGKey(0)
    
    likelihood, prior_kl, local_kl, aux = state.apply_fn(
        {'params': state.params, "batch_stats": state.batch_stats},
        data, 
        month_encoding=month_encoding,
        eval_mode=True,
        mask=None,
        rngs={'sampler': rng}
    )
    
    # Extract categorical expected stats from aux
    cat_expected_stats = aux[3]  # Shape: (batch_size, time_steps, K)
    
    n_samples = min(n_samples, cat_expected_stats.shape[0])
    K = cat_expected_stats.shape[2]
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        # Create heatmap: time_steps x discrete_states
        im = axes[i].imshow(cat_expected_stats[i].T, aspect='auto', cmap='viridis', 
                           vmin=0, vmax=1, interpolation='nearest')
        
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Discrete States')
        axes[i].set_title(f'Sample {i+1}: Discrete State Probability Heatmap')
        axes[i].set_yticks(range(K))
        axes[i].set_yticklabels([f'State {k+1}' for k in range(K)])
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Probability')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Discrete state heatmap saved to {save_path}")
    
    plt.close()


def plot_latent_dynamics_by_state(state, data, month_encoding=None, mask_sst=None, save_path=None, n_samples=3):
    """
    Plot latent dynamics colored by most probable discrete state for SLDS model.
    
    Args:
        state: Current model state
        data: Input data batch
        month_encoding: Month encoding tensor
        mask_sst: SST mask for the data
        save_path: Path to save the plot (optional)
        n_samples: Number of samples to visualize
    """
    
    rng = jax.random.PRNGKey(0)
    
    likelihood, prior_kl, local_kl, aux = state.apply_fn(
        {'params': state.params, "batch_stats": state.batch_stats},
        data, 
        month_encoding=month_encoding,
        eval_mode=True,
        mask=None,
        rngs={'sampler': rng}
    )
    
    # Extract variables from aux
    z = aux[0]  # Shape: (batch_size, time_steps, latent_dim)
    cat_expected_stats = aux[3]  # Shape: (batch_size, time_steps, K)
    
    n_samples = min(n_samples, z.shape[0])
    K = cat_expected_stats.shape[2]
    latent_dims = z.shape[2]
    
    # Get most probable state at each time step
    most_probable_states = jnp.argmax(cat_expected_stats, axis=2)  # Shape: (batch_size, time_steps)
    
    # Debug: Print shapes to understand the mismatch
    # print(f"Debug - z.shape: {z.shape}")
    # print(f"Debug - cat_expected_stats.shape: {cat_expected_stats.shape}")
    # print(f"Debug - most_probable_states.shape: {most_probable_states.shape}")
    
    # Create colors for each discrete state
    colors = plt.cm.tab10(jnp.linspace(0, 1, K))
    
    fig, axes = plt.subplots(n_samples, latent_dims, figsize=(4*latent_dims, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if latent_dims == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        for j in range(latent_dims):
            ax = axes[i, j]
            
            # Plot latent trajectory
            # Use the actual length of the mask to ensure consistency
            actual_length = len(most_probable_states[i])
            time_steps = range(actual_length)
            ax.plot(time_steps, z[i, :actual_length, j], 'k-', alpha=0.3, linewidth=1)
            
            # Color points by most probable discrete state
            for k in range(K):
                mask = most_probable_states[i] == k
                if jnp.any(mask):
                    # Ensure time_steps and mask have compatible shapes
                    # Use the actual length of the mask to avoid shape mismatches
                    actual_length = len(mask)
                    time_steps_array = jnp.arange(actual_length)
                    ax.scatter(time_steps_array[mask], z[i, :actual_length, j][mask], 
                             c=[colors[k]], label=f'State {k+1}', alpha=0.7, s=20)
            
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(f'Latent Dim {j+1}')
            ax.set_title(f'Sample {i+1}, Latent Dim {j+1}')
            if j == 0:  # Only add legend to first subplot of each sample
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent dynamics by state saved to {save_path}")
    
    plt.close()


def plot_state_transition_matrix(state, data, month_encoding=None, mask_sst=None, save_path=None):
    """
    Plot the learned state transition matrix for SLDS model.
    
    Args:
        state: Current model state
        data: Input data batch
        month_encoding: Month encoding tensor
        mask_sst: SST mask for the data
        save_path: Path to save the plot (optional)
    """
    
    # Get the transition parameters from the model
    # We need to access the PGM parameters directly
    pgm_params = state.params['pgm']
    
    # Extract alpha parameters (transition probabilities)
    if 'alpha' in pgm_params:
        # If alpha is in natural parameters, convert to moment parameters
        from distributions import dirichlet
        alpha_nat = pgm_params['alpha']
        alpha_moment = dirichlet.ntou(alpha_nat)
    else:
        # If already in moment parameters
        alpha_moment = pgm_params['alpha']
    
    # Convert to probabilities
    transition_probs = jax.nn.softmax(alpha_moment, axis=1)
    
    K = transition_probs.shape[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    # Create heatmap of transition matrix
    im = ax.imshow(transition_probs, cmap='Blues', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(K):
        for j in range(K):
            text = ax.text(j, i, f'{transition_probs[i, j]:.3f}',
                          ha="center", va="center", color="black" if transition_probs[i, j] < 0.5 else "white")
    
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title('Learned State Transition Matrix')
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f'State {i+1}' for i in range(K)])
    ax.set_yticklabels([f'State {i+1}' for i in range(K)])
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Transition Probability')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"State transition matrix saved to {save_path}")
    
    plt.close()


def plot_state_duration_distribution(state, data, month_encoding=None, mask_sst=None, save_path=None, n_samples=3):
    """
    Plot the distribution of state durations for SLDS model.
    
    Args:
        state: Current model state
        data: Input data batch
        month_encoding: Month encoding tensor
        mask_sst: SST mask for the data
        save_path: Path to save the plot (optional)
        n_samples: Number of samples to analyze
    """
    
    rng = jax.random.PRNGKey(0)
    
    likelihood, prior_kl, local_kl, aux = state.apply_fn(
        {'params': state.params, "batch_stats": state.batch_stats},
        data, 
        month_encoding=month_encoding,
        eval_mode=True,
        mask=None,
        rngs={'sampler': rng}
    )
    
    # Extract categorical expected stats from aux
    cat_expected_stats = aux[3]  # Shape: (batch_size, time_steps, K)
    
    n_samples = min(n_samples, cat_expected_stats.shape[0])
    K = cat_expected_stats.shape[2]
    
    # Get most probable state at each time step
    most_probable_states = jnp.argmax(cat_expected_stats, axis=2)  # Shape: (batch_size, time_steps)
    
    fig, axes = plt.subplots(1, K, figsize=(4*K, 4))
    if K == 1:
        axes = [axes]
    
    for k in range(K):
        ax = axes[k]
        
        # Calculate state durations for this state across all samples
        all_durations = []
        
        for i in range(n_samples):
            # Find consecutive segments where state k is most probable
            state_sequence = most_probable_states[i]
            in_state = state_sequence == k
            
            # Find transitions
            transitions = jnp.diff(jnp.concatenate([[False], in_state, [False]]).astype(int))
            start_times = jnp.where(transitions == 1)[0]
            end_times = jnp.where(transitions == -1)[0]
            
            # Calculate durations
            durations = end_times - start_times
            all_durations.extend(durations.tolist())
        
        # Plot histogram of durations
        if all_durations:
            ax.hist(all_durations, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Duration (time steps)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'State {k+1} Duration Distribution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No occurrences\nof State {k+1}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'State {k+1} Duration Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"State duration distribution saved to {save_path}")
    
    plt.close()


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


    

# def plot_forecast(model_state, data_batch, month_encoding=None, mask_sst=None, 
#                   n_forecast=10, save_path=None, h=26, w=90):
#     """Plot forecasted trajectories"""
#     # Process each sample individually to save memory
#     batch_size = data_batch.shape[0]
    
#     # Create a PRNG key for sampling
#     base_rng = jax.random.PRNGKey(0)
#     save_dir = os.path.dirname(save_path)
#     t_data = data_batch.shape[1]
    
#     # Process and visualize each sample one by one
#     for sample_idx in range(batch_size):
#         # Extract single sample
#         data_single = data_batch[sample_idx:sample_idx+1]  # Keep batch dimension
#         month_single = month_encoding[sample_idx:sample_idx+1] if month_encoding is not None else None
#         mask_sst_single = mask_sst[sample_idx:sample_idx+1] if mask_sst is not None else None
        
#         # Get forecast for this sample
#         _, _, likelihood, aux = eval_step_forecast(
#             model_state, 
#             data_single, 
#             n_forecast,  # n_forecast must be positional argument (static_argnums=3)
#             month_encoding=month_single,
#             mask=None
#         )
#         print('n_forecast', n_forecast)
#         # Sample from forecast distribution with different seed for each sample
#         sample_rng = jax.random.fold_in(base_rng, sample_idx)
#         forecast_single = likelihood.sample(seed=sample_rng) #(1, T+n_forecast, features)
#         print('forecast_single', forecast_single.shape)
#         # Process visualization immediately for this sample
#         t_forecast = forecast_single.shape[1]
#         print('t_forecast', t_forecast)
#         forecast_video = forecast_single.reshape(t_forecast, h, w)  # (T+n_forecast, h, w)
#         print('forecast_video', forecast_video.shape)
#         print('t_data', t_data)
#         data_2d = data_single.reshape(t_data, h, w)  # (T, h, w)
#         mask_2d = mask_sst_single.reshape(mask_sst_single.shape[1], h, w)  # (T, h, w)
        
#         # Create video for this sample
#         video_path = f"{save_dir}/forecast_video_{sample_idx}.mp4"
#         save_frames_to_mp4_matplotlib(
#             [data_2d * mask_2d, 
#              forecast_video * np.repeat(mask_2d, t_forecast//t_data + 2, 0)[:t_forecast]], 
#             video_path, fps=2) 
        
#         # Plot latent trajectories for this sample
#         z = aux[0]  # shape: (1, T+n_forecast, D)
#         plt.figure(figsize=(30, 10))
#         for j in range(z.shape[2]):
#             plt.plot(z[0, :, j])  # z[0] since batch_size=1
#         plt.xlabel('Time Steps')
#         plt.ylabel('Latent Value')
#         plt.savefig(save_path.replace('.png', f'_latent_{sample_idx}.png'))
#         plt.close()
        
#         # Clear variables to free memory
#         del forecast_single, aux, likelihood





# def plot_forecast_chunked(model_state, data_batch, month_encoding=None, mask_sst=None, 
#                           n_forecast=10, chunk_size=12, save_path=None, h=26, w=90):
#     """Plot forecasted trajectories using chunked forecasting to save memory"""
#     # Process each sample individually to save memory
#     batch_size = data_batch.shape[0]
#     save_dir = os.path.dirname(save_path)
#     t_data = data_batch.shape[1]
    
#     for sample_idx in range(batch_size):
#         # Extract single sample
#         data_single = data_batch[sample_idx:sample_idx+1]
#         month_single = month_encoding[sample_idx:sample_idx+1] if month_encoding is not None else None
#         mask_sst_single = mask_sst[sample_idx:sample_idx+1] if mask_sst is not None else None
        
#         # Initialize forecast storage for this sample
#         sample_forecasts = []
#         sample_z_chunks = []
#         current_data = data_single
#         current_month = month_single
        
#         # Forecast in chunks for this sample
#         remaining_forecast = n_forecast
#         chunk_count = 0
#         while remaining_forecast > 0:
#             current_chunk = min(chunk_size, remaining_forecast)
            
#             # Get chunk forecast
#             _, _, likelihood, aux = eval_step_forecast(
#                 model_state, current_data, current_chunk, 
#                 month_encoding=current_month, mask=None
#             )
            
#             # Sample forecast with unique seed
#             chunk_rng = jax.random.PRNGKey(sample_idx * 1000 + chunk_count)
#             chunk_forecast = likelihood.sample(seed=chunk_rng)
#             sample_forecasts.append(chunk_forecast)
            
#             # Store latent variables from this chunk
#             z_chunk = aux[0]  # shape: (1, current_chunk + input_length, D)
#             sample_z_chunks.append(z_chunk)
            
#             # Update for next chunk (use last few time steps as input)
#             current_data = chunk_forecast[:, -data_single.shape[1]:]
#             remaining_forecast -= current_chunk
#             chunk_count += 1
        
#         # Concatenate all chunks for this sample and visualize immediately
#         sample_full_forecast = jnp.concatenate(sample_forecasts, axis=1)
        
#         # Concatenate latent variables from all chunks
#         # Note: We need to handle overlapping parts between chunks
#         if len(sample_z_chunks) == 1:
#             sample_full_z = sample_z_chunks[0]
#         else:
#             # For multiple chunks, we take the full first chunk and append the forecast parts of subsequent chunks
#             full_z_parts = [sample_z_chunks[0]]  # Full first chunk
#             for i in range(1, len(sample_z_chunks)):
#                 # Take only the forecast part (exclude the input overlap)
#                 forecast_part = sample_z_chunks[i][:, data_single.shape[1]:, :]
#                 full_z_parts.append(forecast_part)
#             sample_full_z = jnp.concatenate(full_z_parts, axis=1)
        
#         # Process visualization immediately for this sample
#         t_forecast = sample_full_forecast.shape[1]
#         forecast_video = sample_full_forecast.reshape(t_forecast, h, w)  # (T+n_forecast, h, w)
#         data_2d = data_single.reshape(t_data, h, w)  # (T, h, w)
#         mask_2d = mask_sst_single.reshape(mask_sst_single.shape[1], h, w)  # (T, h, w)
        
#         # Create video for this sample
#         video_path = f"{save_dir}/forecast_video_chunked_{sample_idx}.mp4"
#         save_frames_to_mp4_matplotlib(
#             [data_2d * mask_2d, 
#              forecast_video * np.repeat(mask_2d, t_forecast//t_data + 2, 0)[:t_forecast]], 
#             video_path, fps=2)
        
#         # Plot latent trajectories for this sample
#         z = sample_full_z  # shape: (1, T+n_forecast, D)
#         plt.figure(figsize=(30, 10))
#         for j in range(z.shape[2]):
#             plt.plot(z[0, :, j])  # z[0] since batch_size=1
#         plt.xlabel('Time Steps')
#         plt.ylabel('Latent Value')
#         plt.title(f'Latent Trajectories - Sample {sample_idx} (Chunked Forecast)')
#         plt.savefig(save_path.replace('.png', f'_latent_chunked_{sample_idx}.png'))
#         plt.close()
        
#         # Clear variables to free memory
#         del sample_forecasts, sample_full_forecast, sample_z_chunks, sample_full_z
        
    
        
        
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
    

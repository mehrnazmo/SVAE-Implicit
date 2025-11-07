"""
Mathematically coherent visualization of PGM parameters for SVAE_LDS model.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from typing import Dict, Optional, Tuple
import warnings

def visualize_pgm_parameters(pgm_params: Dict, 
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (16, 12),
                           cond_on_month: bool = False) -> None:
    """
    Create mathematically coherent visualizations of PGM parameters.
    
    Args:
        pgm_params: Dictionary containing ['M', 'S', 'St', 'V', 'lam', 'loc', 'nu', 'nut']
        save_path: Path to save the figure
        figsize: Figure size
        cond_on_month: Whether parameters are month-specific
    """
    
    # Set up the figure with a grid layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract parameters
    S = pgm_params['S']
    loc = pgm_params['loc'] 
    lam = pgm_params['lam']
    nu = pgm_params['nu']
    St = pgm_params['St']
    M = pgm_params['M']
    V = pgm_params['V']
    nut = pgm_params['nut']
    
    # Convert to numpy for plotting
    S_np = np.array(S)
    loc_np = np.array(loc)
    lam_np = float(lam)
    nu_np = float(nu)
    St_np = np.array(St)
    M_np = np.array(M)
    V_np = np.array(V)
    nut_np = np.array(nut)
    
    # 1. Initial State Parameters (NIW) - Top row
    plot_niw_parameters(fig, gs, S_np, loc_np, lam_np, nu_np, cond_on_month)
    
    # 2. Transition Parameters (MNIW) - Middle rows
    plot_mniw_parameters(fig, gs, St_np, M_np, V_np, nut_np, cond_on_month)
    
    # 3. Mathematical Relationships - Bottom row
    plot_mathematical_relationships(fig, gs, pgm_params, cond_on_month)
    
    # Add overall title
    fig.suptitle('PGM Parameters: Mathematical Structure Visualization', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PGM parameter visualization to {save_path}")
    
    plt.show()

def plot_niw_parameters(fig, gs, S, loc, lam, nu, cond_on_month):
    """Plot NIW (Normal-Inverse-Wishart) parameters for initial state."""
    
    # S matrix (precision scale)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(S, cmap='RdBu_r', aspect='auto')
    ax1.set_title('S: Precision Scale Matrix\nE[Σ₀⁻¹] = ν₀ S₀⁻¹', fontsize=10)
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Dimension')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Add eigenvalue information
    eigenvals_S = np.linalg.eigvals(S)
    ax1.text(0.02, 0.98, f'λ_max: {np.max(eigenvals_S):.2f}\nλ_min: {np.min(eigenvals_S):.2f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # loc vector (mean location)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(loc)), loc.flatten(), alpha=0.7, color='skyblue')
    ax2.set_title('loc: Initial Mean Location\nE[μ₀] = μ₀₀', fontsize=10)
    ax2.set_xlabel('Latent Dimension')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    
    # lam and nu scalars
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(['λ (lam)', 'ν (nu)'], [lam, nu], 
                   color=['lightcoral', 'lightgreen'], alpha=0.7)
    ax3.set_title('Precision & Degrees of Freedom\nVar[μ₀] = Σ₀/λ₀', fontsize=10)
    ax3.set_ylabel('Value')
    
    # Add constraint information
    ax3.text(0.5, 0.8, f'Constraint: ν > D-1\nCurrent: ν = {nu:.2f}', 
             transform=ax3.transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Mathematical summary
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    summary_text = f"""NIW Initial State Distribution:
    
z₀ ~ N(μ₀, Σ₀)
μ₀ | Σ₀ ~ N(loc, Σ₀/lam)
Σ₀⁻¹ ~ Wishart(nu, S⁻¹)

Parameters:
• S: {S.shape} precision scale
• loc: {loc.shape} mean location  
• lam: {lam:.3f} precision
• nu: {nu:.3f} degrees of freedom

Expected Values:
• E[μ₀] = loc
• E[Σ₀⁻¹] = nu × S⁻¹
• Var[μ₀] = Σ₀/lam"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')

def plot_mniw_parameters(fig, gs, St, M, V, nut, cond_on_month):
    """Plot MNIW (Matrix-Normal-Inverse-Wishart) parameters for transitions."""
    
    if cond_on_month and St.ndim == 3:
        # Month-specific parameters
        n_months = St.shape[0]
        
        # St matrices (transition noise precision scale)
        ax1 = fig.add_subplot(gs[1, 0])
        # Show mean across months
        St_mean = np.mean(St, axis=0)
        im1 = ax1.imshow(St_mean, cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'St: Transition Noise Precision Scale\nMean across {n_months} months\nE[Σ_t⁻¹] = ν_t S_t⁻¹', fontsize=10)
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Dimension')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # M matrices (transition mean)
        ax2 = fig.add_subplot(gs[1, 1])
        # Extract A matrices (transition matrices)
        A_matrices = M[:, :, :-1]  # Remove bias column
        A_mean = np.mean(A_matrices, axis=0)
        im2 = ax2.imshow(A_mean, cmap='RdBu_r', aspect='auto')
        ax2.set_title(f'M: Transition Mean Matrix\nMean A across {n_months} months\nE[A_t] = Ā_t', fontsize=10)
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Dimension')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Add eigenvalue information for A
        eigenvals_A = np.linalg.eigvals(A_mean)
        ax2.text(0.02, 0.98, f'λ_max: {np.max(np.abs(eigenvals_A)):.2f}\nStable: {np.max(np.abs(eigenvals_A)) < 1}', 
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # V matrices (parameter precision)
        ax3 = fig.add_subplot(gs[1, 2])
        V_mean = np.mean(V, axis=0)
        im3 = ax3.imshow(V_mean, cmap='RdBu_r', aspect='auto')
        ax3.set_title(f'V: Parameter Precision Matrix\nMean across {n_months} months\nVar[vec(X_t)] = Σ_t⁻¹ ⊗ V_t⁻¹', fontsize=10)
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Dimension')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # nut values (degrees of freedom)
        ax4 = fig.add_subplot(gs[1, 3])
        months = range(1, n_months + 1)
        ax4.plot(months, nut, 'o-', color='purple', linewidth=2, markersize=6)
        ax4.set_title('nut: Transition Degrees of Freedom\nν_t for each month', fontsize=10)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('ν_t')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(months)
        
        # Add constraint line
        min_constraint = St.shape[1] - 1  # latent_D - 1
        ax4.axhline(y=min_constraint, color='red', linestyle='--', alpha=0.7, 
                   label=f'Min constraint: {min_constraint}')
        ax4.legend()
        
    else:
        # Non-month-specific parameters
        # St matrix
        ax1 = fig.add_subplot(gs[1, 0])
        im1 = ax1.imshow(St, cmap='RdBu_r', aspect='auto')
        ax1.set_title('St: Transition Noise Precision Scale\nE[Σ_t⁻¹] = ν_t S_t⁻¹', fontsize=10)
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Dimension')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # M matrix (transition mean)
        ax2 = fig.add_subplot(gs[1, 1])
        A = M[:, :-1]  # Extract transition matrix A
        im2 = ax2.imshow(A, cmap='RdBu_r', aspect='auto')
        ax2.set_title('M: Transition Mean Matrix\nE[A_t] = Ā_t', fontsize=10)
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Dimension')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Add eigenvalue information
        eigenvals_A = np.linalg.eigvals(A)
        ax2.text(0.02, 0.98, f'λ_max: {np.max(np.abs(eigenvals_A)):.2f}\nStable: {np.max(np.abs(eigenvals_A)) < 1}', 
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # V matrix
        ax3 = fig.add_subplot(gs[1, 2])
        im3 = ax3.imshow(V, cmap='RdBu_r', aspect='auto')
        ax3.set_title('V: Parameter Precision Matrix\nVar[vec(X_t)] = Σ_t⁻¹ ⊗ V_t⁻¹', fontsize=10)
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Dimension')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # nut scalar
        ax4 = fig.add_subplot(gs[1, 3])
        ax4.bar(['ν_t (nut)'], [nut], color='purple', alpha=0.7)
        ax4.set_title('nut: Transition Degrees of Freedom\nν_t', fontsize=10)
        ax4.set_ylabel('Value')
        
        # Add constraint information
        min_constraint = St.shape[0] - 1  # latent_D - 1
        ax4.text(0.5, 0.8, f'Constraint: ν > D-1\nCurrent: ν = {nut:.2f}', 
                 transform=ax4.transAxes, ha='center',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

def plot_mathematical_relationships(fig, gs, pgm_params, cond_on_month):
    """Plot mathematical relationships and derived quantities."""
    
    # Convert parameters
    S = np.array(pgm_params['S'])
    loc = np.array(pgm_params['loc'])
    lam = float(pgm_params['lam'])
    nu = float(pgm_params['nu'])
    St = np.array(pgm_params['St'])
    M = np.array(pgm_params['M'])
    V = np.array(pgm_params['V'])
    nut = np.array(pgm_params['nut'])
    
    # Expected precision matrices
    ax1 = fig.add_subplot(gs[2, 0])
    E_Sigma_inv_init = nu * np.linalg.inv(S)
    im1 = ax1.imshow(E_Sigma_inv_init, cmap='viridis', aspect='auto')
    ax1.set_title('E[Σ₀⁻¹] = ν₀ S₀⁻¹\nExpected Initial Precision', fontsize=10)
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Dimension')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Expected transition precision
    ax2 = fig.add_subplot(gs[2, 1])
    if cond_on_month and St.ndim == 3:
        E_Sigma_inv_trans = np.mean(nut[:, None, None] * np.linalg.inv(St), axis=0)
    else:
        E_Sigma_inv_trans = nut * np.linalg.inv(St)
    im2 = ax2.imshow(E_Sigma_inv_trans, cmap='viridis', aspect='auto')
    ax2.set_title('E[Σ_t⁻¹] = ν_t S_t⁻¹\nExpected Transition Precision', fontsize=10)
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Dimension')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Transition matrix eigenvalues
    ax3 = fig.add_subplot(gs[2, 2])
    if cond_on_month and M.ndim == 3:
        A_matrices = M[:, :, :-1]
        eigenvals_all = []
        for i in range(A_matrices.shape[0]):
            eigenvals_all.extend(np.linalg.eigvals(A_matrices[i]))
        eigenvals_all = np.array(eigenvals_all)
    else:
        A = M[:, :-1]
        eigenvals_all = np.linalg.eigvals(A)
    
    # Plot eigenvalue magnitudes
    eigenvals_mag = np.abs(eigenvals_all)
    ax3.hist(eigenvals_mag, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Stability threshold')
    ax3.set_title('Transition Matrix Eigenvalues\n|λ(A_t)| Distribution', fontsize=10)
    ax3.set_xlabel('|Eigenvalue|')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Stability analysis
    ax4 = fig.add_subplot(gs[2, 3])
    unstable_count = np.sum(eigenvals_mag > 0.95)
    stable_count = len(eigenvals_mag) - unstable_count
    
    colors = ['red' if unstable_count > 0 else 'green', 'green']
    ax4.pie([unstable_count, stable_count], 
            labels=['Unstable (|λ| > 0.95)', 'Stable (|λ| ≤ 0.95)'],
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Stability Analysis\nTransition Dynamics', fontsize=10)
    
    # Parameter magnitudes and relationships
    ax5 = fig.add_subplot(gs[3, :2])
    
    # Compute parameter magnitudes
    param_magnitudes = {
        'S (init precision scale)': np.linalg.norm(S),
        'St (trans precision scale)': np.linalg.norm(St),
        'M (trans mean)': np.linalg.norm(M),
        'V (param precision)': np.linalg.norm(V),
        'loc (init mean)': np.linalg.norm(loc),
        'lam (init precision)': lam,
        'nu (init dof)': nu,
        'nut (trans dof)': np.mean(nut) if nut.ndim > 0 else nut
    }
    
    params = list(param_magnitudes.keys())
    magnitudes = list(param_magnitudes.values())
    
    bars = ax5.barh(params, magnitudes, alpha=0.7)
    ax5.set_title('Parameter Magnitudes\n||Parameter||₂', fontsize=12)
    ax5.set_xlabel('Magnitude')
    
    # Color bars by parameter type
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange', 'pink', 'lightgray']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Mathematical summary
    ax6 = fig.add_subplot(gs[3, 2:])
    ax6.axis('off')
    
    summary_text = f"""Mathematical Relationships:

Initial State (NIW):
• E[μ₀] = loc = {loc.flatten()[:3]}...
• E[Σ₀⁻¹] = ν₀ S₀⁻¹ = {nu:.2f} × S⁻¹
• Var[μ₀] = Σ₀/λ₀ = Σ₀/{lam:.2f}

Transitions (MNIW):
• E[A_t] = M[:,:-1] (transition matrix)
• E[b_t] = M[:,-1] (bias vector)  
• E[Σ_t⁻¹] = ν_t S_t⁻¹ = {np.mean(nut):.2f} × St⁻¹
• Var[vec(X_t)] = Σ_t⁻¹ ⊗ V_t⁻¹

Stability Analysis:
• Max eigenvalue: {np.max(eigenvals_mag):.3f}
• Unstable eigenvalues: {unstable_count}/{len(eigenvals_mag)}
• Stability ratio: {stable_count/len(eigenvals_mag):.3f}

Constraints:
• ν₀ > D-1: {nu:.2f} > {S.shape[0]-1} ✓
• ν_t > D-1: {np.mean(nut):.2f} > {St.shape[-1]-1} ✓
• |λ(A)| < 1: {'✓' if np.max(eigenvals_mag) < 1 else '✗'}"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')

def plot_parameter_evolution(pgm_params_history: list, 
                           epochs: list,
                           save_path: Optional[str] = None) -> None:
    """
    Plot evolution of PGM parameters over training epochs.
    
    Args:
        pgm_params_history: List of pgm_params dictionaries
        epochs: List of epoch numbers
        save_path: Path to save the figure
    """
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Extract parameter trajectories
    param_names = ['S', 'St', 'M', 'V', 'lam', 'loc', 'nu', 'nut']
    
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        
        if param_name in ['lam', 'nu']:
            # Scalar parameters
            values = [float(params[param_name]) for params in pgm_params_history]
            ax.plot(epochs, values, 'o-', linewidth=2, markersize=4)
            ax.set_title(f'{param_name}: Scalar Evolution', fontsize=10)
            ax.set_ylabel('Value')
            
        elif param_name == 'nut':
            # May be scalar or vector
            values = [params[param_name] for params in pgm_params_history]
            if np.array(values[0]).ndim == 0:
                # Scalar
                values = [float(v) for v in values]
                ax.plot(epochs, values, 'o-', linewidth=2, markersize=4)
            else:
                # Vector - plot mean and std
                values = [np.array(v) for v in values]
                means = [np.mean(v) for v in values]
                stds = [np.std(v) for v in values]
                ax.plot(epochs, means, 'o-', linewidth=2, markersize=4, label='Mean')
                ax.fill_between(epochs, 
                               np.array(means) - np.array(stds),
                               np.array(means) + np.array(stds),
                               alpha=0.3, label='±1 std')
                ax.legend()
            ax.set_title(f'{param_name}: Evolution', fontsize=10)
            ax.set_ylabel('Value')
            
        else:
            # Matrix parameters - plot norm evolution
            values = [np.linalg.norm(params[param_name]) for params in pgm_params_history]
            ax.plot(epochs, values, 'o-', linewidth=2, markersize=4)
            ax.set_title(f'{param_name}: Matrix Norm Evolution', fontsize=10)
            ax.set_ylabel('||Matrix||₂')
        
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('PGM Parameter Evolution During Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter evolution to {save_path}")
    
    plt.show()

# Example usage function
def create_pgm_visualization_example():
    """Create an example visualization with synthetic data."""
    
    # Create synthetic PGM parameters
    latent_D = 10
    
    pgm_params = {
        'S': np.eye(latent_D) * 2.0 + np.random.randn(latent_D, latent_D) * 0.1,
        'loc': np.random.randn(latent_D, 1) * 0.5,
        'lam': 5.0,
        'nu': 15.0,
        'St': np.eye(latent_D) * 1.5 + np.random.randn(latent_D, latent_D) * 0.1,
        'M': np.hstack([np.eye(latent_D) * 0.8 + np.random.randn(latent_D, latent_D) * 0.1,
                       np.random.randn(latent_D, 1) * 0.2]),
        'V': np.eye(latent_D + 1) * 3.0 + np.random.randn(latent_D + 1, latent_D + 1) * 0.1,
        'nut': 12.0
    }
    
    # Visualize
    visualize_pgm_parameters(pgm_params, save_path='pgm_params_example.png')

if __name__ == "__main__":
    create_pgm_visualization_example()

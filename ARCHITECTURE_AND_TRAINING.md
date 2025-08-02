# SVAE (Structured Variational Autoencoder) Architecture and Training Schema

## Overview

This document describes the architecture and training methodology of the SVAE (Structured Variational Autoencoder) system, specifically focusing on the SVAE_LDS (Linear Dynamical System) variant. The system combines deep neural networks with structured probabilistic graphical models to learn temporal dynamics in high-dimensional data.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Model Components](#model-components)
3. [Training Schema](#training-schema)
4. [Loss Functions](#loss-functions)
5. [Optimization Strategy](#optimization-strategy)
6. [Data Processing](#data-processing)
7. [Inference and Forecasting](#inference-and-forecasting)
8. [Configuration](#configuration)

## System Architecture

### High-Level Architecture

The SVAE system consists of three main components:

1. **Encoder Network**: Maps input data to latent space parameters
2. **Probabilistic Graphical Model (PGM)**: Models temporal dynamics in latent space
3. **Decoder Network**: Reconstructs data from latent representations

```
Input Data → Encoder → Latent Parameters → PGM Inference → Latent Samples → Decoder → Reconstruction
```

### Core Model: SVAE_LDS

The SVAE_LDS model combines:
- **Variational Autoencoder (VAE)**: For dimensionality reduction and reconstruction
- **Linear Dynamical System (LDS)**: For modeling temporal dynamics in latent space

## Model Components

### 1. Encoder Network (`SigmaEncoder`)

**Purpose**: Maps input data to latent space parameters (mean and precision)

**Architecture**:
- **Network Type**: DenseNet with ResNet connections
- **Output**: Latent parameters (location and inverse scale)
- **Activation**: GELU (Gaussian Error Linear Unit)
- **Normalization**: LayerNorm (optional)
- **Skip Connections**: Optional residual connections

**Key Features**:
- Uses a learned scale parameter instead of network-predicted scale
- Supports multiple groups for structured latent representations
- Handles missing data through masking

**Mathematical Formulation**:
```
q(z_t | x_t) = N(z_t; μ_enc(x_t), Σ_enc(x_t))
```

### 2. Probabilistic Graphical Model (PGM_LDS)

**Purpose**: Models temporal dynamics in latent space using Linear Dynamical Systems

**Components**:

#### Prior Distributions
- **Initial State Prior**: Normal-Inverse-Wishart (NIW) distribution
- **Transition Prior**: Matrix-Normal-Inverse-Wishart (MNIW) distribution

#### Inference
- **Forward Pass**: Kalman filtering
- **Backward Pass**: Kalman smoothing
- **Sampling**: Uses the inferred posterior to sample latent trajectories

**Mathematical Formulation**:
```
p(z_1) = N(z_1; μ_0, Σ_0)  # Initial state
p(z_t | z_{t-1}) = N(z_t; A z_{t-1} + b, Q)  # Transition
```

### 3. Decoder Network (`SigmaDecoder`)

**Purpose**: Reconstructs data from latent representations

**Architecture**:
- **Network Type**: DenseNet with ResNet connections
- **Output**: Reconstruction parameters (mean and variance)
- **Activation**: GELU
- **Normalization**: LayerNorm
- **Likelihood**: Normal distribution (configurable)

**Key Features**:
- Optional sigmoid activation on final layer
- Supports different likelihood models
- Handles high-dimensional outputs

**Mathematical Formulation**:
```
p(x_t | z_t) = N(x_t; μ_dec(z_t), σ²_dec(z_t))
```

## Training Schema

### Training Loop Structure

```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Forward pass
        loss, metrics = train_step(state, batch)
        
        # Backward pass (automatic in JAX)
        state = state.apply_gradients(grads=grads)
        
        # Logging
        wandb.log(metrics)
```

### Key Training Parameters

- **Learning Rate**: 
  - Network: 1e-4 (Adam optimizer)
  - PGM: 1e-4 (SGD optimizer)
- **Learning Rate Schedule**: Cosine decay
- **Batch Size**: Configurable (typically 5-100)
- **Number of Epochs**: 5000
- **Beta Annealing**: Warmup over 500 steps

### Beta Annealing Schedule

The system uses β-VAE with annealing:

```python
def get_beta_with_warmup(current_step):
    min_beta = 0.001
    warmup_steps = 500
    total_annealing_steps = 500
    
    if current_step < warmup_steps:
        return min_beta
    else:
        annealing_step = current_step - warmup_steps
        return min_beta + (1.0 - min_beta) * min(1.0, annealing_step / total_annealing_steps)
```

## Loss Functions

### Total Loss

The total loss consists of three components:

```
L_total = L_reconstruction + β * (L_local_KL + L_prior_KL)
```

### 1. Reconstruction Loss

**Purpose**: Ensures accurate data reconstruction

**Formulation**:
```
L_reconstruction = -log p(x | z)
```

**Implementation**: Negative log-likelihood of the decoder output

### 2. Local KL Divergence

**Purpose**: Regularizes the encoder to match the PGM's local dynamics

**Formulation**:
```
L_local_KL = KL(q(z_t | x_t) || p(z_t | z_{t-1}))
```

### 3. Prior KL Divergence

**Purpose**: Regularizes the PGM parameters

**Formulation**:
```
L_prior_KL = KL(q(θ) || p(θ))
```

Where θ represents the PGM parameters (transition matrices, noise covariances).

## Optimization Strategy

### Dual Optimization

The system uses separate optimizers for different components:

1. **Network Parameters** (Encoder/Decoder):
   - **Optimizer**: Adam
   - **Learning Rate**: 1e-4
   - **Schedule**: Cosine decay

2. **PGM Parameters**:
   - **Optimizer**: SGD
   - **Learning Rate**: 1e-4
   - **Schedule**: Cosine decay

### Gradient Handling

- **Natural Gradients**: Optional for PGM parameters
- **Gradient Clipping**: Applied to prevent exploding gradients
- **Mixed Precision**: Support for float32/float16 training

## Data Processing

### Input Format

- **Shape**: `(batch_size, time_steps, features)`
- **Example**: `(5, 24, 2340)` for 5 samples, 24 time steps, 2340 features
- **Normalization**: Optional input scaling

### Masking

- **Purpose**: Handle missing data
- **Format**: Binary mask with same shape as data
- **Implementation**: Zero-out missing values and adjust loss computation

### Data Augmentation

- **Temporal Shuffling**: Optional
- **Noise Injection**: For robustness
- **Masking**: Random masking for self-supervised learning

## Inference and Forecasting

### Standard Inference

```python
# Get latent representations
latent_samples = model.apply(params, data, eval_mode=True)

# Get reconstructions
reconstructions = decoder.apply(params, latent_samples)
```

### Forecasting

The system supports multi-step forecasting:

```python
# Generate forecasts
forecasts = eval_step_forecast(state, data, n_forecast=10)
```

**Forecasting Process**:
1. Encode observed data to get latent representations
2. Use PGM to predict future latent states
3. Decode latent forecasts to get data predictions

### Sampling

```python
# Sample from the model
samples = model.apply(params, data, n_samples=10, eval_mode=True)
```

## Configuration

### Model Configuration

```python
model_config = {
    'latent_D': 50,                    # Latent dimension
    'input_D': 2340,                   # Input dimension
    'encoder_stage_sizes': [1, 1, 1, 1],
    'encoder_hidden_sizes': [128, 128, 128, 128],
    'decoder_stage_sizes': [1, 1, 1, 1],
    'decoder_hidden_sizes': [256, 256, 256, 256],
    'activation': 'gelu',
    'last_layer_sigmoid': True,
    'resnet': True
}
```

### Training Configuration

```python
training_config = {
    'num_epochs': 5000,
    'batch_size': 5,
    'lr_net': 1e-4,
    'lr_pgm': 1e-4,
    'warmup_steps': 500,
    'beta_min': 0.001,
    'beta_max': 1.0
}
```

### PGM Configuration

```python
pgm_config = {
    'S_0': 1.0,                       # Prior scale
    'nu_0': 2.0,                      # Prior degrees of freedom
    'lam_0': 0.001,                   # Prior precision
    'M_0': 0.9,                       # Prior mean
    'nat_grads': True,                # Use natural gradients
    'drop_correction': False          # Drop gradient corrections
}
```

## Key Features

### 1. Structured Latent Space

- **Temporal Dynamics**: LDS models smooth temporal transitions
- **Interpretable**: Latent variables have clear temporal meaning
- **Forecastable**: Can predict future states

### 2. Flexible Architecture

- **Modular Design**: Easy to swap components
- **Configurable**: Extensive hyperparameter control
- **Extensible**: Support for different PGM types

### 3. Robust Training

- **Beta Annealing**: Prevents posterior collapse
- **Dual Optimization**: Separate learning rates for different components
- **Gradient Handling**: Natural gradients and clipping

### 4. Practical Features

- **Missing Data**: Handles incomplete observations
- **Forecasting**: Multi-step prediction capabilities
- **Visualization**: Comprehensive plotting utilities
- **Logging**: Integration with Weights & Biases

## Usage Examples

### Basic Training

```python
# Create model
model = SVAE_LDS(latent_D=50, input_D=2340)

# Create training state
state = create_dual_train_state(rng, lr_net, lr_pgm, model, input_shape)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        state, loss = train_step(state, batch)
```

### Forecasting

```python
# Generate forecasts
forecasts = plot_forecast(model_state, data_batch, n_forecast=10)
```

### Visualization

```python
# Plot temporal dynamics
plot_temporal_dynamics(state, data)

# Plot reconstructions
plot_reconstructions(model_state, data_batch)

# Plot training curves
plot_training_curves(metrics_dict)
```

## Performance Considerations

### Memory Usage

- **Batch Size**: Limited by GPU memory
- **Sequence Length**: Longer sequences require more memory
- **Latent Dimension**: Higher dimensions increase memory usage

### Computational Complexity

- **Inference**: O(T) for sequence length T
- **Training**: O(T) per batch
- **Forecasting**: O(T + n_forecast)

### Optimization Tips

1. **Gradient Accumulation**: For larger effective batch sizes
2. **Mixed Precision**: Use float16 where possible
3. **Gradient Clipping**: Prevent exploding gradients
4. **Learning Rate Scheduling**: Cosine decay for better convergence

## Conclusion

The SVAE system provides a powerful framework for learning structured temporal representations in high-dimensional data. By combining the representational power of deep neural networks with the interpretability and forecasting capabilities of structured probabilistic models, it offers a compelling approach for time series modeling and analysis.

The modular architecture allows for easy experimentation with different components, while the robust training procedures ensure stable learning. The system's forecasting capabilities make it particularly suitable for applications requiring future prediction, while the structured latent space provides interpretable representations of temporal dynamics. 
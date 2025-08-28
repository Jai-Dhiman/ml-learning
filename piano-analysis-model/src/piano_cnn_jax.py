"""
Piano Performance Analysis CNN Framework in JAX/Flax
Combining insights from VGGish, EfficientNet, and PANNs for piano-specific analysis
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
from typing import Sequence, Optional, Callable
import functools


class SpectralConvBlock(nn.Module):
    """Optimized conv block for mel-spectrogram processing"""
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME'
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return x


class PianoSpectroCNN(nn.Module):
    """
    Piano-optimized CNN inspired by VGGish + EfficientNet
    Input: mel-spectrograms (time, frequency, channels)
    Output: 19-dimensional perceptual ratings
    """
    num_classes: int = 19  # PercePiano dimensions
    base_filters: int = 64
    dropout_rate: float = 0.2
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Input shape: (batch, time, freq, 1) - mel-spectrograms
        
        # Early spectral feature extraction (VGGish-inspired)
        # Focus on frequency patterns first
        x = SpectralConvBlock(self.base_filters, kernel_size=(3, 3))(x, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = SpectralConvBlock(self.base_filters * 2, kernel_size=(3, 3))(x, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Temporal-spectral fusion (adapted from SoundNet)
        x = SpectralConvBlock(self.base_filters * 4, kernel_size=(3, 3))(x, training)
        x = SpectralConvBlock(self.base_filters * 4, kernel_size=(3, 3))(x, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Deeper feature extraction
        x = SpectralConvBlock(self.base_filters * 8, kernel_size=(3, 3))(x, training)
        x = SpectralConvBlock(self.base_filters * 8, kernel_size=(3, 3))(x, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Global feature extraction
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]))  # Global average pooling
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten
        
        # Multi-task prediction heads
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output layer for 19 perceptual dimensions
        x = nn.Dense(self.num_classes)(x)
        return nn.sigmoid(x)  # Ratings in [0,1] range


class MultiSpectralFusionCNN(nn.Module):
    """
    Advanced architecture inspired by SpectroFusionNet
    Processes multiple spectrogram types (mel, MFCC, chromagram)
    """
    num_classes: int = 19
    base_filters: int = 32  # Lighter for multiple inputs
    fusion_strategy: str = 'early'  # 'early' or 'late'
    
    def setup(self):
        # Separate encoders for each spectrogram type
        self.mel_encoder = self._create_encoder("mel")
        self.mfcc_encoder = self._create_encoder("mfcc")  
        self.chroma_encoder = self._create_encoder("chroma")
        
        if self.fusion_strategy == 'late':
            self.fusion_head = nn.Dense(512)
        
        # Shared classifier
        self.classifier = nn.Sequential([
            nn.Dense(256),
            nn.relu,
            nn.Dropout(rate=0.2),
            nn.Dense(128),
            nn.relu,
            nn.Dropout(rate=0.1),
            nn.Dense(self.num_classes),
            nn.sigmoid
        ])
    
    def _create_encoder(self, name: str):
        """Create encoder for specific spectrogram type"""
        return nn.Sequential([
            SpectralConvBlock(self.base_filters, name=f"{name}_conv1"),
            functools.partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2)),
            SpectralConvBlock(self.base_filters * 2, name=f"{name}_conv2"),
            functools.partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2)),
            SpectralConvBlock(self.base_filters * 4, name=f"{name}_conv3"),
            functools.partial(nn.avg_pool, window_shape=(None, None)),  # Global pooling
        ])
    
    def __call__(self, mel_spec, mfcc_spec, chroma_spec, training: bool = True):
        if self.fusion_strategy == 'early':
            # Concatenate along channel dimension
            fused_input = jnp.concatenate([mel_spec, mfcc_spec, chroma_spec], axis=-1)
            x = self.mel_encoder(fused_input, training)
            x = jnp.reshape(x, (x.shape[0], -1))
            
        else:  # Late fusion
            # Process each spectrogram separately
            mel_feat = self.mel_encoder(mel_spec, training)
            mfcc_feat = self.mfcc_encoder(mfcc_spec, training)  
            chroma_feat = self.chroma_encoder(chroma_spec, training)
            
            # Flatten and concatenate
            mel_feat = jnp.reshape(mel_feat, (mel_feat.shape[0], -1))
            mfcc_feat = jnp.reshape(mfcc_feat, (mfcc_feat.shape[0], -1))
            chroma_feat = jnp.reshape(chroma_feat, (chroma_feat.shape[0], -1))
            
            x = jnp.concatenate([mel_feat, mfcc_feat, chroma_feat], axis=-1)
            x = self.fusion_head(x)
            x = nn.relu(x)
        
        return self.classifier(x, training)


class RealTimePianoCNN(nn.Module):
    """
    Lightweight CNN optimized for real-time inference
    Inspired by EfficientNet efficiency principles
    """
    num_classes: int = 19
    width_multiplier: float = 0.5  # Reduce model size
    
    @nn.compact  
    def __call__(self, x, training: bool = True):
        base_filters = int(32 * self.width_multiplier)
        
        # Efficient depth-wise separable convolutions
        x = self._separable_conv_block(x, base_filters, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self._separable_conv_block(x, base_filters * 2, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self._separable_conv_block(x, base_filters * 4, training)
        x = nn.adaptive_avg_pool(x, (1, 1))  # Efficient global pooling
        
        x = jnp.reshape(x, (x.shape[0], -1))
        
        # Compact classifier
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return nn.sigmoid(x)
    
    def _separable_conv_block(self, x, filters, training):
        """Depth-wise separable convolution for efficiency"""
        # Depth-wise convolution  
        x = nn.Conv(
            features=x.shape[-1],
            kernel_size=(3, 3),
            feature_group_count=x.shape[-1],
            padding='SAME'
        )(x)
        
        # Point-wise convolution
        x = nn.Conv(features=filters, kernel_size=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        return x


# Training utilities
def create_train_state(model, learning_rate: float, input_shape: tuple):
    """Initialize training state"""
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones(input_shape)
    
    params = model.init(rng, dummy_input, training=False)
    optimizer = optax.adam(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )


@jax.jit
def train_step(state, batch_mel, batch_targets, dropout_rng):
    """JIT-compiled training step"""
    def loss_fn(params):
        predictions = state.apply_fn(
            params, batch_mel, 
            training=True, 
            rngs={'dropout': dropout_rng}
        )
        
        # Multi-task MSE loss for 19 dimensions
        mse_loss = jnp.mean((predictions - batch_targets) ** 2)
        
        # L2 regularization
        l2_loss = 0.001 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        
        return mse_loss + l2_loss, predictions
    
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss, predictions


@jax.jit  
def eval_step(state, batch_mel, batch_targets):
    """JIT-compiled evaluation step"""
    predictions = state.apply_fn(state.params, batch_mel, training=False)
    loss = jnp.mean((predictions - batch_targets) ** 2)
    
    # Calculate correlation for each dimension
    correlations = []
    for i in range(predictions.shape[1]):
        corr = jnp.corrcoef(predictions[:, i], batch_targets[:, i])[0, 1]
        correlations.append(corr)
    
    return loss, jnp.array(correlations), predictions


# Model selection utility
def get_piano_model(architecture: str = "standard", **kwargs):
    """Factory function for different piano CNN architectures"""
    if architecture == "standard":
        return PianoSpectroCNN(**kwargs)
    elif architecture == "fusion":
        return MultiSpectralFusionCNN(**kwargs)
    elif architecture == "realtime":
        return RealTimePianoCNN(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    # Example usage
    model = get_piano_model("standard", num_classes=19)
    
    # Test forward pass
    rng = jax.random.PRNGKey(0)
    dummy_spectrogram = jax.random.normal(rng, (4, 128, 128, 1))  # (batch, time, freq, channels)
    
    params = model.init(rng, dummy_spectrogram, training=False)
    output = model.apply(params, dummy_spectrogram, training=False)
    
    print(f"Model output shape: {output.shape}")
    print(f"Parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(params))}")
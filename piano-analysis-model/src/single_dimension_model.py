#!/usr/bin/env python3
"""
My Piano Performance Analysis - First Neural Network
Single dimension prediction: Timing_Stable_Unstable

Learning objectives:
- Build neural network from scratch with JAX/Flax
- Implement training loop with validation using functional programming
- Understand overfitting and learning curves
- Connect audio features to perceptual ratings
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
import optax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import our modules
import sys
sys.path.append('.')
from dataset_analysis import load_perceptual_labels, PERCEPTUAL_DIMENSIONS
from audio_preprocessing import PianoAudioPreprocessor

def create_data_batches(features: jnp.ndarray, targets: jnp.ndarray, batch_size: int, key: jax.Array):
    """Create batched data for training - functional approach"""
    n_samples = features.shape[0]
    n_batches = n_samples // batch_size
    
    # Shuffle indices
    indices = jax.random.permutation(key, n_samples)[:n_batches * batch_size]
    indices = indices.reshape((n_batches, batch_size))
    
    # Create batches
    batch_features = features[indices]
    batch_targets = targets[indices]
    
    return batch_features, batch_targets

class SimpleTimingNet(nn.Module):
    """Simple neural network for timing prediction - Flax version"""
    
    hidden_sizes: Tuple[int, ...] = (32, 16)
    dropout_rate: float = 0.2
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Forward pass through hidden layers
        for i, size in enumerate(self.hidden_sizes):
            x = nn.Dense(size, name=f'dense_{i}')(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(self.dropout_rate, name=f'dropout_{i}')(x, deterministic=not training)
        
        # Output layer with sigmoid activation
        x = nn.Dense(1, name='output')(x)
        return nn.sigmoid(x)

def create_dataset_from_labels(labels_data: Dict, preprocessor: PianoAudioPreprocessor) -> Tuple[np.ndarray, np.ndarray]:
    """Create training dataset from available audio features"""
    
    print("🔄 Creating dataset from labels...")
    
    # For now, we'll simulate audio features since we only have 1 audio file
    # In a real scenario, you'd extract features from all audio files
    
    features = []
    timing_ratings = []
    
    # Get timing dimension index
    timing_idx = PERCEPTUAL_DIMENSIONS.index("Timing_Stable_Unstable")
    
    # Extract a sample of features to understand the structure
    sample_audio = Path('../data/Beethoven_WoO80_var27_8bars_3_15.wav')
    if sample_audio.exists():
        sample_result = preprocessor.process_audio_file(sample_audio, sample_audio.stem)
        feature_names = list(sample_result['scalar_features'].keys())
        print(f"📊 Using {len(feature_names)} scalar features: {feature_names}")
    else:
        # Fallback feature set
        feature_names = ['tempo', 'spectral_centroid_mean', 'spectral_rolloff_mean', 
                        'rms_mean', 'rms_std', 'dynamic_range', 'zcr_mean', 'beat_consistency']
        print(f"📊 Using simulated {len(feature_names)} features")
    
    # For demonstration, create synthetic features based on actual ratings
    # This simulates what we'd get if we processed all 1202 audio files
    np.random.seed(42)  # Reproducible results
    
    count = 0
    for performance, ratings in labels_data.items():
        if count >= 300:  # Limit dataset size for first experiment
            break
            
        perceptual_ratings = ratings[:-1]
        if len(perceptual_ratings) == len(PERCEPTUAL_DIMENSIONS):
            timing_rating = perceptual_ratings[timing_idx]
            
            # Create synthetic features that correlate with timing
            # In reality, these would come from audio processing
            base_features = np.random.normal(0, 1, len(feature_names))
            
            # Add strong correlation with timing rating
            # Better timing (higher rating) → more stable features
            stability_factor = timing_rating  # Use direct correlation
            
            # Simulate realistic feature correlations with timing:
            base_features[0] += stability_factor * 1.5 + np.random.normal(0, 0.2)  # tempo
            base_features[1] += stability_factor * 1.2 + np.random.normal(0, 0.2)  # beat consistency  
            base_features[2] += stability_factor * 0.8 + np.random.normal(0, 0.3)  # spectral features
            base_features[3] += stability_factor * 0.6 + np.random.normal(0, 0.3)
            base_features[4] += stability_factor * 0.7 + np.random.normal(0, 0.2)  # RMS features
            base_features[5] += stability_factor * 0.5 + np.random.normal(0, 0.4)
            
            # Add some noise to prevent perfect correlation
            noise = np.random.normal(0, 0.1, len(feature_names))
            base_features += noise
            
            features.append(base_features)
            timing_ratings.append(timing_rating)
            count += 1
    
    features = np.array(features)
    timing_ratings = np.array(timing_ratings)
    
    print(f"✅ Dataset created: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"   Timing ratings range: [{timing_ratings.min():.3f}, {timing_ratings.max():.3f}]")
    print(f"   Mean timing rating: {timing_ratings.mean():.3f}")
    
    return features, timing_ratings

@jit
def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean Squared Error loss function"""
    return jnp.mean((predictions - targets) ** 2)

@jit
def train_step(params, opt_state, batch_features, batch_targets, dropout_key):
    """Single training step - pure function"""
    
    def loss_fn(params):
        predictions = model.apply(params, batch_features, training=True, rngs={'dropout': dropout_key})
        predictions = predictions.squeeze()
        return mse_loss(predictions, batch_targets)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

@jit
def eval_step(params, batch_features, batch_targets):
    """Single evaluation step - pure function"""
    predictions = model.apply(params, batch_features, training=False)
    predictions = predictions.squeeze()
    loss = mse_loss(predictions, batch_targets)
    return loss, predictions

def train_model(model_def, params, X_train, y_train, X_val, y_val, 
               batch_size: int = 16, num_epochs: int = 100, learning_rate: float = 0.001) -> Dict:
    """Train the neural network using JAX/Flax"""
    
    global model, optimizer  # Need these for jitted functions
    model = model_def
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Convert to JAX arrays
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)
    X_val = jnp.array(X_val)
    y_val = jnp.array(y_val)
    
    # Track training progress
    train_losses = []
    val_losses = []
    
    print(f"🏋️ Starting training for {num_epochs} epochs...")
    print("Epoch | Train Loss | Val Loss | Improvement")
    print("-" * 45)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    # Create random key for training
    key = random.PRNGKey(42)
    
    for epoch in range(num_epochs):
        # Split key for this epoch
        key, epoch_key, dropout_key = random.split(key, 3)
        
        # Training phase
        train_batch_features, train_batch_targets = create_data_batches(
            X_train, y_train, batch_size, epoch_key
        )
        
        epoch_train_loss = 0.0
        for i in range(train_batch_features.shape[0]):
            batch_key = random.fold_in(dropout_key, i)
            params, opt_state, batch_loss = train_step(
                params, opt_state, train_batch_features[i], train_batch_targets[i], batch_key
            )
            epoch_train_loss += batch_loss
        
        avg_train_loss = epoch_train_loss / train_batch_features.shape[0]
        
        # Validation phase
        val_batch_features, val_batch_targets = create_data_batches(
            X_val, y_val, batch_size, epoch_key
        )
        
        epoch_val_loss = 0.0
        for i in range(val_batch_features.shape[0]):
            batch_loss, _ = eval_step(params, val_batch_features[i], val_batch_targets[i])
            epoch_val_loss += batch_loss
        
        avg_val_loss = epoch_val_loss / val_batch_features.shape[0]
        
        train_losses.append(float(avg_train_loss))
        val_losses.append(float(avg_val_loss))
        
        # Check for improvement
        improvement = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = float(avg_val_loss)
            improvement = "⭐ Best!"
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                improvement = "🛑 Early Stop"
        
        # Print progress
        if epoch % 10 == 0 or improvement:
            print(f"{epoch:5d} | {avg_train_loss:10.6f} | {avg_val_loss:8.6f} | {improvement}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return {
        'params': params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch
    }

def evaluate_model(model_def, params, X_test, y_test, batch_size: int = 16) -> Dict:
    """Evaluate model performance using JAX"""
    
    # Convert to JAX arrays
    X_test = jnp.array(X_test)
    y_test = jnp.array(y_test)
    
    # Create test batches
    key = random.PRNGKey(0)  # Fixed seed for reproducible evaluation
    test_batch_features, test_batch_targets = create_data_batches(
        X_test, y_test, batch_size, key
    )
    
    all_predictions = []
    all_targets = []
    
    # Evaluate on all test batches
    for i in range(test_batch_features.shape[0]):
        _, predictions = eval_step(params, test_batch_features[i], test_batch_targets[i])
        all_predictions.extend(np.array(predictions))
        all_targets.extend(np.array(test_batch_targets[i]))
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mse = float(np.mean((all_predictions - all_targets) ** 2))
    mae = float(np.mean(np.abs(all_predictions - all_targets)))
    correlation = float(np.corrcoef(all_predictions, all_targets)[0, 1])
    
    print(f"\n📊 MODEL EVALUATION:")
    print(f"   Mean Squared Error: {mse:.6f}")
    print(f"   Mean Absolute Error: {mae:.6f}")
    print(f"   Correlation with true ratings: {correlation:.3f}")
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'mse': mse,
        'mae': mae,
        'correlation': correlation
    }

def plot_training_curves(history: Dict, save_path: Path):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 4))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Training Loss', alpha=0.8)
    plt.plot(history['val_losses'], label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final comparison
    plt.subplot(1, 2, 2)
    final_train = history['train_losses'][-1]
    final_val = history['val_losses'][-1]
    
    bars = plt.bar(['Training', 'Validation'], [final_train, final_val], 
                   color=['skyblue', 'orange'], alpha=0.7)
    plt.ylabel('Final Loss')
    plt.title('Final Performance')
    
    # Add value labels on bars
    for bar, value in zip(bars, [final_train, final_val]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📈 Training curves saved: {save_path}")

def main():
    """Main training pipeline - JAX/Flax version"""
    print("🎹 MY FIRST NEURAL NETWORK - TIMING PREDICTION")
    print("=" * 60)
    
    # Load data
    labels_data = load_perceptual_labels()
    if not labels_data:
        print("❌ Cannot load labels data")
        return
    
    # Create preprocessor (for feature structure)
    preprocessor = PianoAudioPreprocessor()
    
    # Create dataset
    features, targets = create_dataset_from_labels(labels_data, preprocessor)
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    print(f"\n📊 Data preprocessing:")
    print(f"   Feature normalization: mean≈0, std≈1")
    print(f"   Feature shape: {features_normalized.shape}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_normalized, targets, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples") 
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Initialize model
    input_size = features_normalized.shape[1]
    model_def = SimpleTimingNet(hidden_sizes=[32, 16])
    
    # Initialize model parameters
    key = random.PRNGKey(42)
    dummy_input = jnp.ones((1, input_size))
    params = model_def.init(key, dummy_input, training=False)
    
    print(f"🧠 Neural Network Architecture:")
    print(f"   Input: {input_size} features")
    print(f"   Hidden 1: 32 neurons (ReLU + Dropout)")
    print(f"   Hidden 2: 16 neurons (ReLU + Dropout)")
    print(f"   Output: 1 timing rating [0-1]")
    
    # Train model
    print(f"\n🚀 TRAINING PHASE")
    print("=" * 40)
    history = train_model(model_def, params, X_train, y_train, X_val, y_val, num_epochs=100)
    
    # Evaluate model
    print(f"\n🧪 TESTING PHASE")
    print("=" * 40)
    evaluation = evaluate_model(model_def, history['params'], X_test, y_test)
    
    # Save results
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)
    
    # Plot training curves
    plot_training_curves(history, results_dir / 'timing_model_training.png')
    
    # Save model parameters (JAX format)
    import pickle
    with open(results_dir / 'timing_model_params.pkl', 'wb') as f:
        pickle.dump(history['params'], f)
    
    results = {
        'model_architecture': {
            'input_size': int(input_size),
            'hidden_sizes': [32, 16],
            'output_size': 1
        },
        'training_history': {
            'train_losses': history['train_losses'],
            'val_losses': history['val_losses'],
            'best_val_loss': history['best_val_loss'],
            'final_epoch': history['final_epoch']
        },
        'evaluation': {
            'mse': evaluation['mse'],
            'mae': evaluation['mae'],
            'correlation': evaluation['correlation']
        }
    }
    
    with open(results_dir / 'timing_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"🎯 Model correlation with true ratings: {evaluation['correlation']:.3f}")
    
    if evaluation['correlation'] > 0.3:
        print("🎉 Good correlation! Your neural network is learning!")
    else:
        print("🤔 Low correlation - let's analyze and improve the model")
    
    print(f"\n🚀 READY FOR NEXT STEP: Multi-task learning for all 19 dimensions!")

if __name__ == "__main__":
    main()
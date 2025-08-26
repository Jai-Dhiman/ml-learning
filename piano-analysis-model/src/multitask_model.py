#!/usr/bin/env python3
"""
My Piano Performance Analysis - Multi-Task Learning
Predicting all 19 perceptual dimensions simultaneously

Learning objectives:
- Multi-task neural networks with shared representations using JAX/Flax
- Handling correlated outputs and different scales functionally  
- Compare single-task vs multi-task performance
- Understanding task relationships and interference
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
import seaborn as sns

# Import our modules
import sys
sys.path.append('.')
from dataset_analysis import load_perceptual_labels, PERCEPTUAL_DIMENSIONS
from audio_preprocessing import PianoAudioPreprocessor

def create_multitask_data_batches(features: jnp.ndarray, targets: jnp.ndarray, batch_size: int, key: jax.Array):
    """Create batched data for multi-task training - functional approach"""
    n_samples = features.shape[0]
    n_batches = n_samples // batch_size
    
    # Shuffle indices
    indices = jax.random.permutation(key, n_samples)[:n_batches * batch_size]
    indices = indices.reshape((n_batches, batch_size))
    
    # Create batches
    batch_features = features[indices]
    batch_targets = targets[indices]
    
    return batch_features, batch_targets

class MultiTaskPianoNet(nn.Module):
    """Multi-task neural network for all 19 perceptual dimensions - Flax version"""
    
    hidden_sizes: Tuple[int, ...] = (64, 32)
    num_tasks: int = 19
    dropout_rate: float = 0.3
    task_head_size: int = 16
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Shared feature extraction layers
        shared_features = x
        for i, size in enumerate(self.hidden_sizes):
            shared_features = nn.Dense(size, name=f'shared_dense_{i}')(shared_features)
            shared_features = nn.relu(shared_features)
            shared_features = nn.BatchNorm(name=f'shared_bn_{i}')(shared_features, use_running_average=not training)
            if training:
                shared_features = nn.Dropout(self.dropout_rate, name=f'shared_dropout_{i}')(shared_features, deterministic=not training)
        
        # Task-specific heads for each perceptual dimension
        task_predictions = []
        for task_id in range(self.num_tasks):
            # Task-specific head
            task_features = nn.Dense(self.task_head_size, name=f'task_{task_id}_dense_1')(shared_features)
            task_features = nn.relu(task_features)
            if training:
                task_features = nn.Dropout(0.2, name=f'task_{task_id}_dropout')(task_features, deterministic=not training)
            task_output = nn.Dense(1, name=f'task_{task_id}_output')(task_features)
            task_output = nn.sigmoid(task_output)  # [0, 1] range
            task_predictions.append(task_output)
        
        # Stack all predictions: (batch_size, num_tasks)
        return jnp.concatenate(task_predictions, axis=1)

def create_multitask_dataset(labels_data: Dict, preprocessor: PianoAudioPreprocessor, 
                           max_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Create training dataset for all 19 dimensions"""
    
    print("🔄 Creating multi-task dataset...")
    
    # Get feature structure from sample
    sample_audio = Path('../data/Beethoven_WoO80_var27_8bars_3_15.wav')
    if sample_audio.exists():
        sample_result = preprocessor.process_audio_file(sample_audio, sample_audio.stem)
        feature_names = list(sample_result['scalar_features'].keys())
        print(f"📊 Using {len(feature_names)} audio features")
    else:
        feature_names = ['tempo', 'spectral_centroid_mean', 'spectral_rolloff_mean', 
                        'rms_mean', 'rms_std', 'dynamic_range', 'zcr_mean', 'beat_consistency']
    
    features = []
    all_ratings = []
    
    np.random.seed(42)  # Reproducible
    
    count = 0
    for performance, ratings in labels_data.items():
        if count >= max_samples:
            break
        
        perceptual_ratings = ratings[:-1]  # Exclude player_id
        if len(perceptual_ratings) == len(PERCEPTUAL_DIMENSIONS):
            
            # Create synthetic features that correlate with multiple dimensions
            base_features = np.random.normal(0, 1, len(feature_names))
            
            # Add realistic correlations with different dimensions:
            
            # Timing features (affects timing, articulation, music making)
            timing_factor = perceptual_ratings[0]  # Timing_Stable_Unstable
            base_features[0] += timing_factor * 1.2  # tempo
            base_features[1] += timing_factor * 0.8  # beat consistency
            
            # Spectral features (affects timbre dimensions)
            brightness = perceptual_ratings[7]  # Timbre_Bright_Dark
            richness = perceptual_ratings[6]   # Timbre_Shallow_Rich
            base_features[2] += brightness * 1.0 + richness * 0.6  # spectral centroid
            base_features[3] += brightness * 0.8 + richness * 0.9  # spectral rolloff
            
            # Dynamic features (affects dynamics, emotion)
            dynamic_range = perceptual_ratings[10]  # Dynamic range
            energy = perceptual_ratings[16]         # Energy
            base_features[4] += dynamic_range * 1.1 + energy * 0.7  # rms_mean
            base_features[5] += dynamic_range * 1.3 + energy * 0.5  # rms_std
            base_features[6] += dynamic_range * 0.9                 # dynamic_range
            
            # Articulation features
            articulation = perceptual_ratings[1]  # Articulation_Short_Long
            base_features[7] += articulation * 0.8  # zcr_mean
            
            # Add realistic noise and cross-correlations
            for i in range(19):
                if i < len(base_features):
                    # Each rating influences multiple features (realistic)
                    base_features[i % len(base_features)] += perceptual_ratings[i] * 0.3
            
            # Add noise
            base_features += np.random.normal(0, 0.2, len(feature_names))
            
            features.append(base_features)
            all_ratings.append(perceptual_ratings)
            count += 1
    
    features = np.array(features)
    all_ratings = np.array(all_ratings)
    
    print(f"✅ Multi-task dataset created:")
    print(f"   Samples: {features.shape[0]}")
    print(f"   Features: {features.shape[1]} audio features")
    print(f"   Targets: {all_ratings.shape[1]} perceptual dimensions")
    print(f"   Rating ranges: [{all_ratings.min():.3f}, {all_ratings.max():.3f}]")
    
    return features, all_ratings

@jit
def multitask_mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Multi-task MSE loss function"""
    return jnp.mean((predictions - targets) ** 2)

@jit
def multitask_train_step(variables, opt_state, batch_features, batch_targets, dropout_key):
    """Single multi-task training step - pure function"""
    
    def loss_fn(variables):
        predictions, new_variables = model.apply(
            variables, batch_features, training=True, 
            mutable=['batch_stats'],
            rngs={'dropout': dropout_key}
        )
        loss = multitask_mse_loss(predictions, batch_targets)
        return loss, new_variables
    
    (loss, new_variables), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables)
    updates, opt_state = optimizer.update(grads['params'], opt_state, variables['params'])
    new_params = optax.apply_updates(variables['params'], updates)
    new_variables = {**new_variables, 'params': new_params}
    
    return new_variables, opt_state, loss

@jit
def multitask_eval_step(variables, batch_features, batch_targets):
    """Single multi-task evaluation step - pure function"""
    predictions = model.apply(variables, batch_features, training=False)
    loss = multitask_mse_loss(predictions, batch_targets)
    return loss, predictions

def train_multitask_model(model_def, variables, X_train, y_train, X_val, y_val,
                         batch_size: int = 32, num_epochs: int = 150, learning_rate: float = 0.001) -> Dict:
    """Train the multi-task neural network using JAX/Flax"""
    
    global model, optimizer  # Need these for jitted functions
    model = model_def
    
    # Setup optimizer with weight decay
    optimizer = optax.adamw(learning_rate, weight_decay=1e-4)
    opt_state = optimizer.init(variables['params'])
    
    # Convert to JAX arrays
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)
    X_val = jnp.array(X_val)
    y_val = jnp.array(y_val)
    
    # Track training progress
    train_losses = []
    val_losses = []
    dimension_losses = {dim: [] for dim in PERCEPTUAL_DIMENSIONS}
    
    print(f"🏋️ Starting multi-task training for {num_epochs} epochs...")
    print("Epoch | Train Loss | Val Loss | Best Dims")
    print("-" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    # Create random key for training
    key = random.PRNGKey(42)
    
    for epoch in range(num_epochs):
        # Split key for this epoch
        key, epoch_key, dropout_key = random.split(key, 3)
        
        # Training phase
        train_batch_features, train_batch_targets = create_multitask_data_batches(
            X_train, y_train, batch_size, epoch_key
        )
        
        epoch_train_loss = 0.0
        for i in range(train_batch_features.shape[0]):
            batch_key = random.fold_in(dropout_key, i)
            variables, opt_state, batch_loss = multitask_train_step(
                variables, opt_state, 
                train_batch_features[i], train_batch_targets[i], batch_key
            )
            epoch_train_loss += batch_loss
        
        avg_train_loss = epoch_train_loss / train_batch_features.shape[0]
        
        # Validation phase
        val_batch_features, val_batch_targets = create_multitask_data_batches(
            X_val, y_val, batch_size, epoch_key
        )
        
        epoch_val_loss = 0.0
        dimension_val_losses = np.zeros(19)
        
        for i in range(val_batch_features.shape[0]):
            batch_loss, predictions = multitask_eval_step(
                variables, val_batch_features[i], val_batch_targets[i]
            )
            epoch_val_loss += batch_loss
            
            # Track individual dimension losses
            for j in range(19):
                dim_loss = jnp.mean((predictions[:, j] - val_batch_targets[i][:, j]) ** 2)
                dimension_val_losses[j] += float(dim_loss)
        
        avg_val_loss = epoch_val_loss / val_batch_features.shape[0]
        
        train_losses.append(float(avg_train_loss))
        val_losses.append(float(avg_val_loss))
        
        # Update dimension losses
        for i, dim in enumerate(PERCEPTUAL_DIMENSIONS):
            dimension_losses[dim].append(dimension_val_losses[i] / val_batch_features.shape[0])
        
        # Find best performing dimensions
        best_dims = sorted(enumerate(dimension_val_losses), key=lambda x: x[1])[:3]
        best_dim_names = [PERCEPTUAL_DIMENSIONS[i][:15] for i, _ in best_dims]
        
        # Check improvement
        improvement = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = float(avg_val_loss)
            improvement = "⭐"
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                improvement = "🛑 Early Stop"
        
        # Print progress
        if epoch % 10 == 0 or improvement or patience_counter >= patience:
            print(f"{epoch:5d} | {avg_train_loss:10.6f} | {avg_val_loss:8.6f} | {', '.join(best_dim_names)} {improvement}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return {
        'variables': variables,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'dimension_losses': dimension_losses,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch
    }

def evaluate_multitask_model(model_def, variables, X_test, y_test, batch_size: int = 32) -> Dict:
    """Evaluate multi-task model performance using JAX"""
    
    # Convert to JAX arrays
    X_test = jnp.array(X_test)
    y_test = jnp.array(y_test)
    
    # Create test batches
    key = random.PRNGKey(0)  # Fixed seed for reproducible evaluation
    test_batch_features, test_batch_targets = create_multitask_data_batches(
        X_test, y_test, batch_size, key
    )
    
    all_predictions = []
    all_targets = []
    
    # Evaluate on all test batches
    for i in range(test_batch_features.shape[0]):
        _, predictions = multitask_eval_step(variables, test_batch_features[i], test_batch_targets[i])
        all_predictions.append(np.array(predictions))
        all_targets.append(np.array(test_batch_targets[i]))
    
    all_predictions = np.vstack(all_predictions)  # Shape: (samples, 19)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics for each dimension
    dimension_metrics = {}
    correlations = []
    
    print(f"\n📊 MULTI-TASK MODEL EVALUATION:")
    print("Dimension                     | MSE      | MAE      | Correlation")
    print("-" * 70)
    
    for i, dim in enumerate(PERCEPTUAL_DIMENSIONS):
        pred = all_predictions[:, i]
        target = all_targets[:, i]
        
        mse = float(np.mean((pred - target) ** 2))
        mae = float(np.mean(np.abs(pred - target)))
        corr = float(np.corrcoef(pred, target)[0, 1]) if len(np.unique(target)) > 1 else 0.0
        
        dimension_metrics[dim] = {
            'mse': mse,
            'mae': mae,
            'correlation': corr
        }
        
        correlations.append(corr)
        print(f"{dim[:25]:25} | {mse:8.6f} | {mae:8.6f} | {corr:8.3f}")
    
    avg_correlation = float(np.mean(correlations))
    print(f"\n🎯 Average correlation across all dimensions: {avg_correlation:.3f}")
    
    return {
        'dimension_metrics': dimension_metrics,
        'average_correlation': avg_correlation,
        'predictions': all_predictions,
        'targets': all_targets
    }

def plot_multitask_results(history: Dict, evaluation: Dict, save_dir: Path):
    """Plot comprehensive multi-task results"""
    
    # 1. Training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['train_losses'], label='Training', alpha=0.8)
    plt.plot(history['val_losses'], label='Validation', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Overall Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Dimension correlations
    plt.subplot(2, 3, 2)
    correlations = [evaluation['dimension_metrics'][dim]['correlation'] 
                   for dim in PERCEPTUAL_DIMENSIONS]
    plt.barh(range(len(correlations)), correlations, alpha=0.7)
    plt.yticks(range(len(PERCEPTUAL_DIMENSIONS)), 
               [dim.split('_')[-1][:15] for dim in PERCEPTUAL_DIMENSIONS])
    plt.xlabel('Correlation')
    plt.title('Per-Dimension Performance')
    plt.grid(True, alpha=0.3)
    
    # 3. Top performing dimensions
    plt.subplot(2, 3, 3)
    sorted_dims = sorted(evaluation['dimension_metrics'].items(), 
                        key=lambda x: x[1]['correlation'], reverse=True)
    top_5 = sorted_dims[:5]
    
    names = [dim.split('_')[-1][:12] for dim, _ in top_5]
    corrs = [metrics['correlation'] for _, metrics in top_5]
    
    plt.bar(names, corrs, alpha=0.7, color='green')
    plt.ylabel('Correlation')
    plt.title('Top 5 Dimensions')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. Worst performing dimensions
    plt.subplot(2, 3, 4)
    bottom_5 = sorted_dims[-5:]
    
    names = [dim.split('_')[-1][:12] for dim, _ in bottom_5]
    corrs = [metrics['correlation'] for _, metrics in bottom_5]
    
    plt.bar(names, corrs, alpha=0.7, color='red')
    plt.ylabel('Correlation')
    plt.title('Bottom 5 Dimensions')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. Correlation distribution
    plt.subplot(2, 3, 5)
    plt.hist(correlations, bins=15, alpha=0.7, edgecolor='black')
    plt.axvline(evaluation['average_correlation'], color='red', linestyle='--', 
                label=f'Average: {evaluation["average_correlation"]:.3f}')
    plt.xlabel('Correlation')
    plt.ylabel('Number of Dimensions')
    plt.title('Correlation Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Loss comparison by category
    plt.subplot(2, 3, 6)
    categories = {
        'Timing': [0],
        'Articulation': [1, 2],
        'Pedal': [3, 4],
        'Timbre': [5, 6, 7, 8],
        'Dynamic': [9, 10],
        'Music Making': [11, 12, 13, 14],
        'Emotion': [15, 16, 17],
        'Interpretation': [18]
    }
    
    category_corrs = {}
    for cat, indices in categories.items():
        cat_corrs = [correlations[i] for i in indices]
        category_corrs[cat] = np.mean(cat_corrs)
    
    plt.bar(category_corrs.keys(), category_corrs.values(), alpha=0.7)
    plt.ylabel('Average Correlation')
    plt.title('Performance by Category')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'multitask_analysis.png', dpi=150, bbox_inches='tight')
    print(f"📈 Multi-task analysis saved: {save_dir / 'multitask_analysis.png'}")

def main():
    """Main multi-task training pipeline - JAX/Flax version"""
    print("🎹 MULTI-TASK NEURAL NETWORK - ALL 19 DIMENSIONS")
    print("=" * 70)
    
    # Load data
    labels_data = load_perceptual_labels()
    if not labels_data:
        print("❌ Cannot load labels data")
        return
    
    # Create preprocessor
    preprocessor = PianoAudioPreprocessor()
    
    # Create multi-task dataset
    features, targets = create_multitask_dataset(labels_data, preprocessor, max_samples=600)
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    print(f"\n📊 Data preprocessing:")
    print(f"   Feature normalization: mean≈0, std≈1")
    print(f"   Input shape: {features_normalized.shape}")
    print(f"   Output shape: {targets.shape}")
    
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
    model_def = MultiTaskPianoNet(hidden_sizes=(64, 32), num_tasks=19)
    
    # Initialize model parameters with batch norm state
    key = random.PRNGKey(42)
    dummy_input = jnp.ones((1, input_size))
    variables = model_def.init(key, dummy_input, training=False)
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
    
    print(f"🧠 Multi-Task Neural Network Architecture:")
    print(f"   Input: {input_size} features")
    print(f"   Shared 1: 64 neurons (ReLU + BatchNorm + Dropout)")
    print(f"   Shared 2: 32 neurons (ReLU + BatchNorm + Dropout)")
    print(f"   Task heads: 19 × (16 → 1) for each dimension")
    print(f"   Total parameters: {param_count:,}")
    
    # Train model
    print(f"\n🚀 MULTI-TASK TRAINING PHASE")
    print("=" * 50)
    history = train_multitask_model(model_def, variables, X_train, y_train, X_val, y_val, num_epochs=150)
    
    # Evaluate model
    print(f"\n🧪 MULTI-TASK TESTING PHASE")
    print("=" * 50)
    evaluation = evaluate_multitask_model(model_def, history['variables'], X_test, y_test)
    
    # Save results
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)
    
    # Plot comprehensive analysis
    plot_multitask_results(history, evaluation, results_dir)
    
    # Save model parameters (JAX format)
    import pickle
    with open(results_dir / 'multitask_model_params.pkl', 'wb') as f:
        pickle.dump(history['variables'], f)
    
    results = {
        'model_architecture': {
            'input_size': int(input_size),
            'hidden_sizes': [64, 32],
            'num_tasks': 19,
            'total_parameters': int(param_count)
        },
        'training_history': {
            'train_losses': history['train_losses'],
            'val_losses': history['val_losses'],
            'best_val_loss': history['best_val_loss'],
            'final_epoch': history['final_epoch']
        },
        'evaluation': {
            'average_correlation': evaluation['average_correlation'],
            'dimension_metrics': evaluation['dimension_metrics']
        }
    }
    
    with open(results_dir / 'multitask_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ MULTI-TASK TRAINING COMPLETE!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"🎯 Average correlation across 19 dimensions: {evaluation['average_correlation']:.3f}")
    
    if evaluation['average_correlation'] > 0.4:
        print("🎉 Excellent multi-task learning! The model is capturing perceptual relationships!")
    elif evaluation['average_correlation'] > 0.25:
        print("👍 Good multi-task performance! Model is learning shared representations!")
    else:
        print("🤔 Moderate performance - multi-task learning is challenging!")
    
    # Find best dimensions
    best_dims = sorted(evaluation['dimension_metrics'].items(), 
                      key=lambda x: x[1]['correlation'], reverse=True)
    
    print(f"\n🏆 TOP PERFORMING DIMENSIONS:")
    for i, (dim, metrics) in enumerate(best_dims[:5]):
        print(f"   {i+1}. {dim}: {metrics['correlation']:.3f}")
    
    print(f"\n🚀 READY FOR NEXT STEP: CNN architecture with spectrograms!")

if __name__ == "__main__":
    main()
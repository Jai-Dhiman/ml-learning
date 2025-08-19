#!/usr/bin/env python3
"""
Minimal training test for safety text classifier
"""

import sys
import os
sys.path.append('src')

import jax
import jax.numpy as jnp
import yaml
import logging
import numpy as np

from src.models.transformer import create_model, initialize_model
from src.data.dataset_loader import SafetyDatasetLoader
from src.evaluation.metrics import SafetyMetricsCalculator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_pipeline():
    """Test the complete training pipeline locally."""
    
    print("🚀 Starting Safety Text Classifier Local Test")
    print("=" * 50)
    
    # 1. Load configuration
    print("1. Loading configuration...")
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for quick local test
    config['training']['max_steps'] = 20
    config['training']['batch_size'] = 4
    config['model']['num_layers'] = 2  # Smaller for faster testing
    config['model']['num_heads'] = 4
    
    print("   ✅ Configuration loaded")
    
    # 2. Create synthetic dataset
    print("2. Creating test dataset...")
    loader = SafetyDatasetLoader()
    dataset = loader.create_synthetic_dataset(size=50)
    
    # Simple split
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, train_size + val_size))
    test_data = dataset.select(range(train_size + val_size, len(dataset)))
    
    print(f"   ✅ Dataset created: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # 3. Initialize model
    print("3. Initializing model...")
    model = create_model(config)
    rng = jax.random.PRNGKey(42)
    params = initialize_model(model, rng)
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"   ✅ Model initialized with {param_count:,} parameters")
    
    # 4. Test tokenization
    print("4. Testing tokenization...")
    tokenized_train = loader.tokenize_dataset(train_data)
    tokenized_val = loader.tokenize_dataset(val_data)
    
    print(f"   ✅ Tokenization complete")
    print(f"   Sample input shape: {np.array(tokenized_train[0]['input_ids']).shape}")
    print(f"   Sample labels: {tokenized_train[0]['labels']}")
    
    # 5. Test forward pass
    print("5. Testing forward pass...")
    
    # Prepare batch
    batch_input_ids = jnp.array([tokenized_train[i]['input_ids'] for i in range(4)])
    batch_labels = jnp.array([tokenized_train[i]['labels'] for i in range(4)])
    
    # Forward pass
    outputs = model.apply(params, batch_input_ids, training=False)
    logits = outputs['logits']
    
    print(f"   ✅ Forward pass successful")
    print(f"   Batch input shape: {batch_input_ids.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Logits sample: {logits[0]}")
    
    # 6. Test loss computation
    print("6. Testing loss computation...")
    
    # Binary cross-entropy loss
    import optax
    labels_float = batch_labels.astype(jnp.float32)
    loss = optax.sigmoid_binary_cross_entropy(logits, labels_float).mean()
    
    print(f"   ✅ Loss computation successful")
    print(f"   Loss value: {loss:.4f}")
    
    # 7. Test predictions and metrics
    print("7. Testing evaluation metrics...")
    
    # Get predictions
    probs = jax.nn.sigmoid(logits)
    preds = (probs > 0.5).astype(jnp.int32)
    
    # Compute metrics
    calculator = SafetyMetricsCalculator()
    metrics = calculator.compute_basic_metrics(
        np.array(batch_labels), 
        np.array(preds), 
        np.array(probs)
    )
    
    print(f"   ✅ Metrics computed")
    print(f"   Accuracy: {metrics['exact_match_accuracy']:.3f}")
    print(f"   Macro F1: {metrics['macro_f1']:.3f}")
    
    # 8. Test gradient computation
    print("8. Testing gradient computation...")
    
    def loss_fn(params):
        # Need to provide RNG key for dropout
        dropout_rng = jax.random.PRNGKey(123)
        outputs = model.apply(params, batch_input_ids, training=True, rngs={'dropout': dropout_rng})
        logits = outputs['logits']
        loss = optax.sigmoid_binary_cross_entropy(logits, labels_float).mean()
        return loss
    
    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    grad_norm = optax.global_norm(grads)
    
    print(f"   ✅ Gradients computed")
    print(f"   Loss: {loss_val:.4f}")
    print(f"   Gradient norm: {grad_norm:.4f}")
    
    print("=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Data loading and preprocessing")
    print("✅ Model architecture and initialization")
    print("✅ Forward pass and predictions")
    print("✅ Loss computation")
    print("✅ Gradient computation")
    print("✅ Evaluation metrics")
    print("")
    print("🚀 Ready for full training!")
    print("Next steps:")
    print("  - Run: python train.py (for full training)")
    print("  - Run: python demo_app.py (for interactive demo)")

if __name__ == "__main__":
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run test
    test_training_pipeline()
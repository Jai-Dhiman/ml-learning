#!/usr/bin/env python3
"""
Quick test to validate the Colab notebook fixes work.
This tests the JAX JIT compilation fix without running the full training.
"""

import os
import sys
sys.path.append('src')

# Test JAX/Flax import
try:
    import jax
    import jax.numpy as jnp
    import flax
    import flax.linen as nn
    from flax import struct
    from flax.training import train_state
    import optax
    print("✅ JAX/Flax imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test the JIT compilation pattern from the notebook
print("🧪 Testing JIT compilation pattern...")

# Simple model for testing
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(4)(x)

def compute_loss_test(params, model, batch, rng_key=None, training=True):
    """Simplified loss function for testing."""
    outputs = model.apply(params, batch['x'])
    loss = jnp.mean((outputs - batch['y']) ** 2)
    return loss, {'loss': loss}

def create_train_step(model):
    """Create JIT-compiled training step function."""
    @jax.jit
    def train_step(state, batch, rng_key):
        def loss_fn(params):
            return compute_loss_test(params, model, batch, rng_key, training=True)
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        
        # Add gradient norm to metrics
        grad_norm = optax.global_norm(grads)
        metrics['grad_norm'] = grad_norm
        
        return state, metrics
    
    return train_step

def create_eval_step(model):
    """Create JIT-compiled evaluation step function."""
    @jax.jit
    def eval_step(params, batch):
        loss, metrics = compute_loss_test(params, model, batch, rng_key=None, training=False)
        return metrics
    
    return eval_step

try:
    # Create model
    model = SimpleModel()
    rng = jax.random.PRNGKey(42)
    
    # Initialize parameters
    dummy_input = jnp.ones((2, 10))
    params = model.init(rng, dummy_input)
    
    # Create optimizer and training state
    optimizer = optax.adam(0.001)
    
    @struct.dataclass
    class TrainState(train_state.TrainState):
        epoch: int = 0
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        epoch=0
    )
    
    print("✅ Model and state initialization successful")
    
    # Test JIT compilation (the main fix)
    train_step = create_train_step(model)
    eval_step = create_eval_step(model)
    
    print("✅ JIT compilation successful")
    
    # Test with dummy batch
    batch = {
        'x': jnp.ones((2, 10)),
        'y': jnp.ones((2, 4))
    }
    
    # Test training step
    rng, step_rng = jax.random.split(rng)
    state, train_metrics = train_step(state, batch, step_rng)
    print(f"✅ Training step successful: loss={train_metrics['loss']:.4f}")
    
    # Test evaluation step
    eval_metrics = eval_step(state.params, batch)
    print(f"✅ Evaluation step successful: loss={eval_metrics['loss']:.4f}")
    
    print("\n🎉 All tests passed! The notebook fixes should work in Colab.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
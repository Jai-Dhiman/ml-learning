#!/usr/bin/env python3
"""
Simple test script to verify model architecture works.
This bypasses some of the complex import issues.
"""

import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any
import sys
sys.path.append('src')

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

class SimpleTransformer(nn.Module):
    """Simplified transformer for testing."""
    
    vocab_size: int = 1000
    embedding_dim: int = 128
    num_heads: int = 4
    num_classes: int = 4
    dropout_rate: float = 0.1
    
    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embedding_dim)
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm()
        self.classifier = nn.Dense(self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def __call__(self, input_ids, training=False):
        # Embeddings
        x = self.embedding(input_ids)
        
        # Self-attention
        x = self.layer_norm(x)
        attn_output = self.attention(x, x)
        x = x + self.dropout(attn_output, deterministic=not training)
        
        # Global average pooling
        mask = (input_ids != 0).astype(jnp.float32)[..., None]
        x_masked = x * mask
        seq_lengths = jnp.sum(mask, axis=1)
        pooled = jnp.sum(x_masked, axis=1) / jnp.maximum(seq_lengths, 1)
        
        # Classification
        logits = self.classifier(pooled)
        return {"logits": logits}

def test_model():
    """Test model creation and forward pass."""
    print("🔧 Creating simple transformer...")
    
    model = SimpleTransformer()
    rng = jax.random.PRNGKey(42)
    
    # Create dummy input
    batch_size, seq_len = 2, 64
    dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    print("⚙️ Initializing parameters...")
    try:
        params = model.init(rng, dummy_input, training=False)
        print("✅ Parameters initialized successfully!")
        
        print("🧪 Testing forward pass...")
        output = model.apply(params, dummy_input, training=False)
        
        print(f"✅ Forward pass successful!")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output['logits'].shape}")
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"📊 Total parameters: {param_count:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("🎉 Simple model test PASSED!")
    else:
        print("💥 Model test FAILED!")
        sys.exit(1)
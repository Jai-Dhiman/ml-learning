#!/usr/bin/env python3
"""
Debug the model forward pass
"""

import sys
sys.path.append('src')

import jax
import jax.numpy as jnp
import yaml
from transformers import AutoTokenizer

from src.models.transformer import create_model, initialize_model

def debug_model():
    """Debug model forward pass"""
    
    print("Debugging model forward pass...")
    
    try:
        # Load config
        with open('configs/base_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer'])
        
        # Initialize model
        model = create_model(config)
        rng = jax.random.PRNGKey(42)
        params = initialize_model(model, rng)
        
        print("✅ Model initialized")
        
        # Prepare input
        text = "This is a test message"
        encoded = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=config['data']['max_length'],
            return_tensors='np'
        )
        input_ids = jnp.array(encoded['input_ids'])
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Input with batch dim shape: {input_ids[None, :].shape}")
        
        # Test model.apply
        print("Testing model.apply...")
        outputs = model.apply(params, input_ids[None, :], training=False)
        
        print(f"✅ Model apply successful")
        print(f"Output keys: {outputs.keys()}")
        print(f"Logits shape: {outputs['logits'].shape}")
        print(f"Attention weights type: {type(outputs['attention_weights'])}")
        print(f"Attention weights length: {len(outputs['attention_weights']) if hasattr(outputs['attention_weights'], '__len__') else 'N/A'}")
        
        # Test sigmoid
        probabilities = jax.nn.sigmoid(outputs['logits'])
        print(f"Probabilities shape: {probabilities.shape}")
        print(f"Probabilities sample: {probabilities[0]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model()
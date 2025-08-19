#!/usr/bin/env python3
"""
Quick demo test of the safety classifier
"""

import sys
import os
sys.path.append('src')

import jax
import jax.numpy as jnp
import yaml
from transformers import AutoTokenizer

from src.models.transformer import create_model, initialize_model

def quick_demo():
    """Quick demo of safety classification"""
    
    print("🛡️ Safety Text Classifier - Quick Demo")
    print("=" * 45)
    
    # Load config
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer'])
    
    # Initialize model
    model = create_model(config)
    rng = jax.random.PRNGKey(42)
    params = initialize_model(model, rng)
    
    # Safety categories
    categories = ['Hate Speech', 'Self-Harm', 'Dangerous Advice', 'Harassment']
    
    # Test cases
    test_cases = [
        "The weather is beautiful today, perfect for a walk.",
        "I hate all people from that group, they are terrible.",
        "Here's how to make dangerous explosives at home.",
        "I'm going to find where you live and hurt you.",
        "I feel sad and want to hurt myself badly.",
        "I love spending time with my family and friends.",
    ]
    
    print("🔍 Testing Safety Classification:")
    print("-" * 45)
    
    for i, text in enumerate(test_cases, 1):
        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=config['data']['max_length'],
            return_tensors='np'
        )
        input_ids = jnp.array(encoded['input_ids'])
        
        # Predict
        outputs = model.apply(params, input_ids, training=False)
        probabilities = jax.nn.sigmoid(outputs['logits'])[0]  # Remove batch dim
        
        # Find max category
        max_idx = jnp.argmax(probabilities)
        max_prob = probabilities[max_idx]
        max_category = categories[max_idx]
        
        # Determine if safe or unsafe
        if max_prob > 0.5:
            status = f"⚠️  UNSAFE: {max_category} ({max_prob:.1%})"
        else:
            status = f"✅ SAFE (highest: {max_category} {max_prob:.1%})"
        
        print(f"{i}. \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"   {status}")
        print()
    
    print("=" * 45)
    print("🎉 Quick demo completed!")
    print("")
    print("📊 Scores shown are from UNTRAINED model")
    print("   (Random predictions - train model for real results)")
    print("")
    print("🚀 Next steps:")
    print("   • Train model: python train.py") 
    print("   • Full demo: python demo_app.py")

if __name__ == "__main__":
    quick_demo()
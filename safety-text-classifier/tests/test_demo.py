#!/usr/bin/env python3
"""
Test the Gradio demo interface
"""

import sys
import os
sys.path.append('../src')

from src.demo import SafetyClassifierDemo

def test_demo():
    """Test the demo interface components"""
    
    print("🎮 Testing Safety Text Classifier Demo")
    print("=" * 40)
    
    # 1. Initialize demo
    print("1. Initializing demo...")
    demo = SafetyClassifierDemo()
    print("   ✅ Demo initialized")
    
    # 2. Test text preprocessing
    print("2. Testing text preprocessing...")
    test_text = "This is a test message for safety classification."
    input_ids = demo.preprocess_text(test_text)
    print(f"   ✅ Text preprocessed, input shape: {input_ids.shape}")
    
    # 3. Test classification
    print("3. Testing classification...")
    safety_scores, explanation, attention_plot = demo.classify_text(test_text)
    
    print("   ✅ Classification completed")
    print(f"   Safety scores: {safety_scores}")
    print(f"   Explanation preview: {explanation[:100]}...")
    
    # 4. Test with different examples
    print("4. Testing with different examples...")
    
    test_cases = [
        "The weather is beautiful today.",
        "I hate all people from that country.",
        "Here's how to hurt yourself.",
        "I'm going to find where you live.",
    ]
    
    for i, test_case in enumerate(test_cases):
        scores, _, _ = demo.classify_text(test_case)
        max_score = max(scores.values()) if scores else 0
        max_category = max(scores.keys(), key=scores.get) if scores else "None"
        print(f"   Test {i+1}: {max_category} ({max_score:.2f})")
    
    print("=" * 40)
    print("🎉 Demo test completed!")
    print("✅ Demo initialization")
    print("✅ Text preprocessing")
    print("✅ Safety classification")
    print("✅ Multiple test cases")
    print("")
    print("🚀 Ready to launch full demo!")
    print("Run: python scripts/demo_app.py --host 127.0.0.1 --port 7860")

if __name__ == "__main__":
    test_demo()
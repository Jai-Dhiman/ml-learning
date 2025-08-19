#!/usr/bin/env python3
"""
Test script to reproduce the classification error
"""

import sys
sys.path.append('src')

from src.demo import SafetyClassifierDemo

def test_classify():
    """Test classify_text method directly"""
    
    print("Testing SafetyClassifierDemo.classify_text...")
    
    try:
        # Initialize demo
        demo = SafetyClassifierDemo()
        print("✅ Demo initialized")
        
        # Test classification
        test_text = "This is a test message"
        print(f"Testing text: '{test_text}'")
        
        result = demo.classify_text(test_text)
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_classify()
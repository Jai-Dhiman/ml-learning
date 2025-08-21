#!/usr/bin/env python3
"""
Quick Test Script

Test all components with minimal data loading for fast feedback.
"""

import sys
sys.path.append('../src')

def test_basic_components():
    """Test basic components without heavy data loading."""
    print("🔧 Testing basic components...")
    
    # Test model creation
    try:
        from src.models.transformer import create_model, initialize_model
        import yaml
        import jax
        
        with open('configs/base_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        model = create_model(config)
        rng = jax.random.PRNGKey(42)
        params = initialize_model(model, rng)
        
        print("✅ Model creation successful")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test basic inference
    try:
        import jax.numpy as jnp
        dummy_input = jnp.ones((2, 512), dtype=jnp.int32)
        outputs = model.apply(params, dummy_input, training=False)
        print(f"✅ Basic inference successful - output shape: {outputs['logits'].shape}")
    except Exception as e:
        print(f"❌ Basic inference failed: {e}")
        return False
    
    # Test utilities
    try:
        from src.models.utils import count_parameters, validate_model_config
        param_count = count_parameters(params)
        validate_model_config(config)
        print(f"✅ Utilities test successful - {param_count:,} parameters")
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False
    
    # Test calibration
    try:
        from src.models.calibration import TemperatureScaling, ConfidenceEstimator
        import numpy as np
        
        # Quick test with dummy data
        calibrator = TemperatureScaling()
        estimator = ConfidenceEstimator()
        
        dummy_logits = np.random.randn(100)
        dummy_labels = np.random.randint(0, 2, 100)
        temp = calibrator.fit(dummy_logits, dummy_labels)
        
        dummy_probs = 1 / (1 + np.exp(-dummy_logits))
        conf = estimator.entropy_confidence(dummy_probs)
        
        print(f"✅ Calibration test successful - temp: {temp:.3f}, mean conf: {conf.mean():.3f}")
    except Exception as e:
        print(f"❌ Calibration test failed: {e}")
        return False
    
    # Test visualization (without plots)
    try:
        from src.models.visualization import AttentionVisualizer
        viz = AttentionVisualizer()
        
        # Create dummy attention
        dummy_attention = [np.random.rand(1, 12, 20, 20) for _ in range(6)]
        dummy_input_ids = np.random.randint(0, 1000, (1, 20))
        
        patterns = viz.extract_attention_patterns(dummy_attention, dummy_input_ids)
        important_tokens = viz.identify_important_tokens(patterns, top_k=5)
        
        print(f"✅ Visualization test successful - {len(important_tokens)} important tokens")
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False
    
    return True

def test_demo():
    """Test demo interface."""
    print("\n🖥️ Testing demo interface...")
    
    try:
        from src.demo import SafetyClassifierDemo
        demo = SafetyClassifierDemo()
        
        # Test classification
        test_text = "This is a test message for the safety classifier."
        scores, explanation, plot = demo.classify_text(test_text, include_analysis=False)
        
        max_score = max(scores.values())
        max_category = max(scores, key=scores.get)
        
        print(f"✅ Demo test successful - {max_category}: {max_score:.3f}")
        print(f"   Explanation length: {len(explanation)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Demo test failed: {e}")
        return False

def test_server():
    """Test server components."""
    print("\n🚀 Testing server components...")
    
    try:
        from src.serving.inference_server import SafetyClassifierServer
        server = SafetyClassifierServer()
        
        print("✅ Server initialization successful")
        
        # Test health status
        health = server.get_health_status()
        print(f"✅ Health check successful - status: {health['status']}")
        return True
        
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def test_synthetic_data():
    """Test synthetic data generation (lightweight)."""
    print("\n📊 Testing synthetic data generation...")
    
    try:
        from src.data.dataset_loader import SafetyDatasetLoader
        loader = SafetyDatasetLoader()
        
        # Create small synthetic dataset
        dataset = loader.create_synthetic_dataset(size=10)
        print(f"✅ Synthetic data successful - {len(dataset)} examples")
        
        # Test tokenization
        tokenized = loader.tokenize_dataset(dataset)
        print(f"✅ Tokenization successful - features: {list(tokenized.features.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data test failed: {e}")
        return False

def main():
    """Run all quick tests."""
    print("🧪 Running Quick Test Suite for Safety Text Classifier\n")
    
    results = {
        "Basic Components": test_basic_components(),
        "Demo Interface": test_demo(), 
        "Server Components": test_server(),
        "Synthetic Data": test_synthetic_data()
    }
    
    print("\n" + "="*50)
    print("📋 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your Safety Text Classifier is ready!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
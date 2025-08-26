#!/usr/bin/env python3
"""
Test Suite for My Piano Performance Analysis Implementation
Validates all components work correctly
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add src to path for imports
sys.path.append('src')

def test_dataset_analysis():
    """Test dataset analysis functionality"""
    print("🔬 Testing Dataset Analysis...")
    
    try:
        from dataset_analysis import load_perceptual_labels, PERCEPTUAL_DIMENSIONS
        
        # Test data loading
        labels = load_perceptual_labels()
        if not labels:
            print("❌ Failed to load labels")
            return False
        
        print(f"✅ Loaded {len(labels)} performances")
        print(f"✅ {len(PERCEPTUAL_DIMENSIONS)} dimensions defined")
        
        # Check results file
        results_file = Path('results/dataset_analysis.json')
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            print(f"✅ Analysis results saved: {results_file}")
            print(f"   - Dataset has {results['dataset_characteristics']['num_performances']} performances")
            print(f"   - Rating range: [{results['dataset_characteristics']['rating_stats']['min']:.3f}, {results['dataset_characteristics']['rating_stats']['max']:.3f}]")
        else:
            print("⚠️  Results file not found - run dataset_analysis.py first")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset analysis test failed: {e}")
        return False

def test_audio_preprocessing():
    """Test audio preprocessing functionality"""
    print("\n🎵 Testing Audio Preprocessing...")
    
    try:
        from audio_preprocessing import PianoAudioPreprocessor
        
        # Initialize preprocessor
        preprocessor = PianoAudioPreprocessor()
        print("✅ Preprocessor initialized")
        
        # Test audio file exists
        audio_file = Path('data/Beethoven_WoO80_var27_8bars_3_15.wav')
        if not audio_file.exists():
            print(f"❌ Audio file not found: {audio_file}")
            return False
        
        print(f"✅ Audio file found: {audio_file}")
        
        # Test processing
        result = preprocessor.process_audio_file(audio_file, audio_file.stem)
        
        # Validate results
        assert 'spectral_features' in result
        assert 'scalar_features' in result
        assert len(result['scalar_features']) > 0
        
        print(f"✅ Processing successful:")
        print(f"   - Duration: {result['duration']:.2f}s")
        print(f"   - Scalar features: {len(result['scalar_features'])}")
        
        # Check spectral features
        for name, data in result['spectral_features'].items():
            shape = np.array(data).shape
            print(f"   - {name}: {shape}")
        
        # Check results file
        results_file = Path('results/preprocessing_test.json')
        if results_file.exists():
            print(f"✅ Preprocessing results saved: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio preprocessing test failed: {e}")
        return False

def test_data_files():
    """Test that required data files exist"""
    print("\n📁 Testing Data Files...")
    
    required_files = [
        'data/label_2round_mean_reg_19_with0_rm_highstd0.json',
        'data/Beethoven_WoO80_var27_8bars_3_15.wav'
    ]
    
    all_present = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ Missing: {file_path}")
            all_present = False
    
    return all_present

def test_directory_structure():
    """Test that directory structure is correct"""
    print("\n📂 Testing Directory Structure...")
    
    required_dirs = ['src', 'data', 'results', 'notebooks', 'models', 'experiments']
    
    all_present = True
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists() and path.is_dir():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ Missing directory: {dir_name}/")
            all_present = False
    
    return all_present

def test_dependencies():
    """Test that required Python packages are available"""
    print("\n📦 Testing Dependencies...")
    
    required_packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'), 
        ('librosa', 'librosa'),
        ('json', 'json'),
        ('pathlib', 'Path')
    ]
    
    all_available = True
    for package, alias in required_packages:
        try:
            if alias == 'Path':
                from pathlib import Path
                print(f"✅ pathlib.Path")
            else:
                exec(f"import {package} as {alias}")
                print(f"✅ {package}")
        except ImportError:
            print(f"❌ Missing: {package}")
            all_available = False
    
    return all_available

def test_correlation_insights():
    """Test that we can extract meaningful insights"""
    print("\n🧠 Testing Analysis Insights...")
    
    try:
        results_file = Path('results/dataset_analysis.json')
        if not results_file.exists():
            print("⚠️  Run dataset_analysis.py first to generate insights")
            return False
        
        with open(results_file) as f:
            results = json.load(f)
        
        # Check key insights
        characteristics = results['dataset_characteristics']
        extremes = results['extreme_performances']
        correlations = results['dimension_correlations']
        
        print(f"✅ Dataset insights available:")
        print(f"   - {characteristics['num_performances']} performances analyzed")
        print(f"   - {characteristics['num_players']} unique players")
        print(f"   - {len(extremes['lowest'])} lowest and {len(extremes['highest'])} highest rated found")
        print(f"   - {len(correlations['strong_correlations'])} strong correlations identified")
        
        # Show a key correlation
        if correlations['strong_correlations']:
            top_corr = correlations['strong_correlations'][0]
            print(f"   - Strongest correlation: {top_corr['correlation']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Insights test failed: {e}")
        return False

def run_full_test_suite():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("🧪 TESTING MY PIANO PERFORMANCE ANALYSIS IMPLEMENTATION")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Data Files", test_data_files),
        ("Dependencies", test_dependencies),
        ("Dataset Analysis", test_dataset_analysis),
        ("Audio Preprocessing", test_audio_preprocessing),
        ("Analysis Insights", test_correlation_insights)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {test_name}")
        if success:
            passed += 1
    
    success_rate = passed / len(tests)
    print(f"\n🎯 SUCCESS RATE: {success_rate:.1%} ({passed}/{len(tests)} tests passed)")
    
    if success_rate == 1.0:
        print("\n🎉 ALL TESTS PASSED! Your implementation is ready.")
        print("🚀 Next step: Build your first neural network model!")
    elif success_rate >= 0.8:
        print("\n✅ Implementation is mostly working!")
        print("🔧 Fix the failing tests and you'll be ready to proceed.")
    else:
        print("\n⚠️  Several issues need to be addressed.")
        print("📝 Review the failed tests and fix the issues.")
    
    return success_rate

if __name__ == "__main__":
    run_full_test_suite()
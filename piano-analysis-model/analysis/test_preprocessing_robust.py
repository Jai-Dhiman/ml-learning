#!/usr/bin/env python3
"""
Test preprocessing pipeline robustness and prepare for batch processing
"""

import json
import numpy as np
from pathlib import Path
from audio_preprocessing import PercePianoPreprocessor

def test_preprocessing_robustness():
    """Test preprocessing on different scenarios"""
    
    print("=== PercePiano Preprocessing Robustness Test ===\\n")
    
    # Initialize preprocessor
    preprocessor = PercePianoPreprocessor(target_sr=22050)
    
    # Test scenarios
    test_results = []
    
    # 1. Test on the WAV file
    example_audio = Path('examples/Beethoven_WoO80_var27_8bars_3_15.wav')
    if example_audio.exists():
        print(f"✅ Testing WAV file: {example_audio.name}")
        try:
            result = preprocessor.process_audio_file(example_audio, example_audio.stem)
            test_results.append({
                'file_type': 'WAV',
                'file_name': example_audio.name,
                'success': True,
                'duration': result['duration'],
                'num_scalar_features': len(result['scalar_features']),
                'spectral_shapes': {k: np.array(v).shape for k, v in result['spectral_data'].items()}
            })
        except Exception as e:
            test_results.append({
                'file_type': 'WAV',
                'file_name': example_audio.name,
                'success': False,
                'error': str(e)
            })
    
    # 2. Check what MIDI files we have
    midi_dir = Path('virtuoso/data/all_2rounds')
    if midi_dir.exists():
        midi_files = list(midi_dir.glob('*.mid'))[:3]  # Test first 3 MIDI files
        print(f"\\n🎼 Found {len(list(midi_dir.glob('*.mid')))} MIDI files, testing first 3...")
        
        for midi_file in midi_files:
            print(f"  Testing MIDI: {midi_file.name}")
            try:
                # Try to process MIDI directly (librosa can handle MIDI)
                result = preprocessor.process_audio_file(midi_file, midi_file.stem)
                test_results.append({
                    'file_type': 'MIDI',
                    'file_name': midi_file.name,
                    'success': True,
                    'duration': result['duration'],
                    'num_scalar_features': len(result['scalar_features'])
                })
            except Exception as e:
                test_results.append({
                    'file_type': 'MIDI', 
                    'file_name': midi_file.name,
                    'success': False,
                    'error': str(e)
                })
    
    # Report results
    print(f"\\n📊 ROBUSTNESS TEST RESULTS:")
    successful_tests = 0
    
    for result in test_results:
        print(f"\\n  {result['file_type']}: {result['file_name']}")
        if result['success']:
            successful_tests += 1
            print(f"    ✅ SUCCESS")
            print(f"    Duration: {result['duration']:.2f}s")
            print(f"    Scalar features: {result['num_scalar_features']}")
            if 'spectral_shapes' in result:
                for spec_type, shape in result['spectral_shapes'].items():
                    print(f"    {spec_type}: {shape}")
        else:
            print(f"    ❌ FAILED: {result['error']}")
    
    success_rate = successful_tests / len(test_results) if test_results else 0
    print(f"\\n🎯 OVERALL SUCCESS RATE: {success_rate:.1%} ({successful_tests}/{len(test_results)})")
    
    # Save test results
    with open('preprocessing_robustness_test.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\\n💡 RECOMMENDATIONS:")
    
    if success_rate >= 0.5:
        print(f"✅ Preprocessing pipeline is robust enough to proceed")
        print(f"✅ Ready to move to next phase: Building ML models")
        if any(r['file_type'] == 'MIDI' and r['success'] for r in test_results):
            print(f"✅ MIDI processing works - can use full 1202 performance dataset")
        else:
            print(f"⚠️  MIDI processing issues - may need audio conversion step")
    else:
        print(f"⚠️  Preprocessing needs improvement before ML training")
        print(f"⚠️  Focus on error handling and format compatibility")
    
    return test_results, success_rate >= 0.5

if __name__ == "__main__":
    test_preprocessing_robustness()
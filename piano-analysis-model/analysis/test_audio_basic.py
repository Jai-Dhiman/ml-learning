#!/usr/bin/env python3
"""
Basic audio loading test for PercePiano
"""

import json
import numpy as np
from pathlib import Path

try:
    import librosa
    print("✅ librosa imported successfully")
except ImportError as e:
    print(f"❌ librosa import failed: {e}")
    exit(1)

def main():
    # Load perceptual data
    labels_path = Path('label_2round_mean_reg_19_with0_rm_highstd0.json')
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    print(f"✅ Loaded {len(labels_data)} performance ratings")
    
    # Check audio file
    example_audio = Path('examples/Beethoven_WoO80_var27_8bars_3_15.wav')
    performance_name = example_audio.stem
    
    if not example_audio.exists():
        print(f"❌ Audio file not found: {example_audio}")
        return
    
    print(f"✅ Found audio file: {example_audio}")
    
    # Load audio
    try:
        y, sr = librosa.load(example_audio, sr=None)
        print(f"✅ Audio loaded: {len(y)} samples, {sr}Hz, {len(y)/sr:.2f}s duration")
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        return
    
    # Check perceptual data
    if performance_name in labels_data:
        ratings = labels_data[performance_name]
        overall_score = np.mean(ratings[:-1])  # Exclude player_id
        print(f"✅ Found perceptual ratings - Overall score: {overall_score:.3f}")
        print(f"   Player ID: {ratings[-1]}")
    else:
        print(f"❌ No perceptual data for {performance_name}")
    
    # Basic audio features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    print(f"📊 Basic audio features:")
    print(f"   Tempo: {float(tempo):.1f} BPM")
    print(f"   Spectral centroid: {float(spectral_centroid):.1f} Hz")
    
    print("\n🎯 Ready to proceed with full audio-visual correlation analysis!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick audio feature extraction for PercePiano dataset
Focus: Extract key features and correlate with perceptual ratings
"""

import json
import numpy as np
from pathlib import Path

try:
    import librosa
    print("✅ librosa available")
except ImportError:
    print("❌ Need librosa: pip install librosa")
    exit(1)

def extract_audio_features(audio_path):
    """Extract key audio features for correlation analysis"""
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    print(f"Processing audio: {duration:.2f}s, {sr}Hz")
    
    # Basic features
    features = {}
    
    # 1. Tempo and rhythm
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    except:
        features['tempo'] = 0.0
    
    # 2. Spectral features (for timbre)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
    features['spectral_centroid_std'] = float(np.std(spectral_centroids))
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]  
    features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
    
    # 3. Dynamic features
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    features['dynamic_range'] = float(np.max(rms) - np.min(rms))
    
    # 4. Timing/rhythm features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zero_crossing_rate))
    features['zcr_std'] = float(np.std(zero_crossing_rate))
    
    # 5. Basic spectral stats
    features['duration'] = duration
    features['sample_rate'] = sr
    
    return features

def analyze_correlations(performance_name, audio_features, perceptual_ratings, perceptual_dimensions):
    """Analyze correlations between audio features and perceptual ratings"""
    
    print(f"\\n=== CORRELATION ANALYSIS: {performance_name} ===")
    
    # Overall score
    overall_score = np.mean(perceptual_ratings)
    print(f"Overall perceptual score: {overall_score:.3f}")
    
    print(f"\\n🎵 AUDIO FEATURES:")
    for feature, value in audio_features.items():
        if feature not in ['sample_rate']:
            print(f"  {feature}: {value:.3f}")
    
    print(f"\\n🧠 KEY PERCEPTUAL DIMENSIONS:")
    
    # Focus on dimensions we expect to correlate with our features
    key_correlations = [
        (0, "Timing_Stable_Unstable", ['tempo', 'zcr_std'], "Lower values = more stable"),
        (7, "Timbre_Bright_Dark", ['spectral_centroid_mean', 'spectral_rolloff_mean'], "Higher values = brighter"),
        (10, "Dynamic_Little_dynamic_range_Large_dynamic_range", ['dynamic_range', 'rms_std'], "Higher values = larger range"),
        (16, "Emotion_&_Mood_Low_Energy_High_Energy", ['tempo', 'rms_mean'], "Higher values = higher energy")
    ]
    
    for idx, dimension, related_features, explanation in key_correlations:
        rating = perceptual_ratings[idx]
        print(f"\\n  {dimension}: {rating:.3f}")
        print(f"    {explanation}")
        
        for feature in related_features:
            if feature in audio_features:
                feature_val = audio_features[feature]
                print(f"    Related audio feature - {feature}: {feature_val:.3f}")
    
    # Find extreme ratings to highlight
    rating_pairs = [(perceptual_dimensions[i], perceptual_ratings[i]) 
                   for i in range(len(perceptual_ratings))]
    rating_pairs.sort(key=lambda x: x[1])
    
    print(f"\\n📊 EXTREME RATINGS:")
    print(f"  LOWEST (poor performance aspects):")
    for dim, rating in rating_pairs[:3]:
        print(f"    {dim}: {rating:.3f}")
    
    print(f"  HIGHEST (excellent performance aspects):")  
    for dim, rating in rating_pairs[-3:]:
        print(f"    {dim}: {rating:.3f}")
    
    return {
        'performance': performance_name,
        'overall_score': overall_score,
        'audio_features': audio_features,
        'perceptual_ratings': {dim: rating for dim, rating in zip(perceptual_dimensions, perceptual_ratings)},
        'extreme_low': rating_pairs[:3],
        'extreme_high': rating_pairs[-3:]
    }

def main():
    """Main analysis function"""
    print("=== PercePiano Audio Feature Correlation Analysis ===\\n")
    
    # Load perceptual data
    labels_path = Path('label_2round_mean_reg_19_with0_rm_highstd0.json')
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    perceptual_dimensions = [
        "Timing_Stable_Unstable", "Articulation_Short_Long", "Articulation_Soft_cushioned_Hard_solid",
        "Pedal_Sparse/dry_Saturated/wet", "Pedal_Clean_Blurred", "Timbre_Even_Colorful",
        "Timbre_Shallow_Rich", "Timbre_Bright_Dark", "Timbre_Soft_Loud",
        "Dynamic_Sophisticated/mellow_Raw/crude", "Dynamic_Little_dynamic_range_Large_dynamic_range",
        "Music_Making_Fast_paced_Slow_paced", "Music_Making_Flat_Spacious",
        "Music_Making_Disproportioned_Balanced", "Music_Making_Pure_Dramatic/expressive",
        "Emotion_&_Mood_Optimistic/pleasant_Dark", "Emotion_&_Mood_Low_Energy_High_Energy",
        "Emotion_&_Mood_Honest_Imaginative", "Interpretation_Unsatisfactory/doubtful_Convincing"
    ]
    
    print(f"✅ Loaded {len(labels_data)} performance ratings")
    
    # Analyze the example audio
    example_audio = Path('examples/Beethoven_WoO80_var27_8bars_3_15.wav')
    performance_name = example_audio.stem
    
    if not example_audio.exists():
        print(f"❌ Audio file not found: {example_audio}")
        return
    
    if performance_name not in labels_data:
        print(f"❌ No perceptual data for {performance_name}")
        return
    
    # Extract audio features
    print(f"Analyzing: {performance_name}")
    audio_features = extract_audio_features(example_audio)
    
    # Get perceptual ratings
    ratings = labels_data[performance_name]
    perceptual_ratings = ratings[:-1]  # Exclude player_id
    player_id = ratings[-1]
    
    # Analyze correlations
    analysis_result = analyze_correlations(
        performance_name, audio_features, perceptual_ratings, perceptual_dimensions
    )
    
    # Save results
    output_file = f'{performance_name}_correlation_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"\\n✅ ANALYSIS COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"\\n🎯 NEXT STEPS:")
    print(f"1. Run this analysis on multiple performances (high vs low rated)")
    print(f"2. Look for consistent patterns across different pieces")
    print(f"3. Build feature extraction pipeline for full dataset") 
    print(f"4. Start training single-dimension prediction models")

if __name__ == "__main__":
    main()
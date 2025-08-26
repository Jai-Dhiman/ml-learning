#!/usr/bin/env python3
"""
Audio-Visual Correlation Analysis for PercePiano Dataset
Phase 1: Starting with audio loading and visualization
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Check if librosa is available
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
    print("✅ librosa available")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("❌ librosa not available - install with: pip install librosa")

def load_perceptual_data():
    """Load and return perceptual labels"""
    labels_path = Path('label_2round_mean_reg_19_with0_rm_highstd0.json')
    
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    perceptual_dimensions = [
        "Timing_Stable_Unstable",
        "Articulation_Short_Long", 
        "Articulation_Soft_cushioned_Hard_solid",
        "Pedal_Sparse/dry_Saturated/wet",
        "Pedal_Clean_Blurred",
        "Timbre_Even_Colorful",
        "Timbre_Shallow_Rich",
        "Timbre_Bright_Dark",
        "Timbre_Soft_Loud",
        "Dynamic_Sophisticated/mellow_Raw/crude",
        "Dynamic_Little_dynamic_range_Large_dynamic_range",
        "Music_Making_Fast_paced_Slow_paced",
        "Music_Making_Flat_Spacious",
        "Music_Making_Disproportioned_Balanced",
        "Music_Making_Pure_Dramatic/expressive",
        "Emotion_&_Mood_Optimistic/pleasant_Dark",
        "Emotion_&_Mood_Low_Energy_High_Energy",
        "Emotion_&_Mood_Honest_Imaginative",
        "Interpretation_Unsatisfactory/doubtful_Convincing"
    ]
    
    return labels_data, perceptual_dimensions

def analyze_audio_file(audio_path, performance_name, perceptual_data, perceptual_dimensions):
    """Analyze a single audio file and correlate with perceptual ratings"""
    
    if not LIBROSA_AVAILABLE:
        print("Cannot analyze audio - librosa not installed")
        return
    
    print(f"\n=== ANALYZING: {performance_name} ===")
    
    # Check if we have perceptual data for this performance
    if performance_name not in perceptual_data:
        print(f"No perceptual data found for {performance_name}")
        return
    
    # Load perceptual ratings
    ratings = perceptual_data[performance_name]
    perceptual_ratings = ratings[:-1]  # Last element is player_id
    player_id = ratings[-1]
    
    print(f"Player ID: {player_id}")
    print(f"Overall score: {np.mean(perceptual_ratings):.3f}")
    
    # Load audio file
    try:
        y, sr = librosa.load(audio_path, sr=None)
        print(f"✅ Audio loaded: {len(y)} samples, {sr}Hz, {len(y)/sr:.2f}s duration")
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        return
    
    # Create comprehensive analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Audio Analysis: {performance_name}', fontsize=16, fontweight='bold')
    
    # 1. Waveform
    axes[0, 0].plot(np.linspace(0, len(y)/sr, len(y)), y)
    axes[0, 0].set_title('Waveform')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Amplitude')
    
    # 2. Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=axes[0, 1])
    axes[0, 1].set_title('Spectrogram')
    
    # 3. Mel-spectrogram 
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', sr=sr, ax=axes[1, 0])
    axes[1, 0].set_title('Mel Spectrogram')
    
    # 4. Key perceptual ratings
    # Show top and bottom rated dimensions
    rating_pairs = [(perceptual_dimensions[i], perceptual_ratings[i]) 
                   for i in range(len(perceptual_ratings))]
    rating_pairs.sort(key=lambda x: x[1])
    
    # Get highest and lowest 5 ratings
    lowest_5 = rating_pairs[:5]
    highest_5 = rating_pairs[-5:]
    
    all_ratings = lowest_5 + highest_5
    names = [dim.split('_')[-1] for dim, _ in all_ratings]  # Shortened names
    values = [rating for _, rating in all_ratings]
    colors = ['red'] * 5 + ['green'] * 5
    
    bars = axes[1, 1].barh(names, values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Extreme Perceptual Ratings')
    axes[1, 1].set_xlabel('Rating (0-1)')
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Average')
    
    plt.tight_layout()
    plt.savefig(f'{performance_name}_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Extract basic audio features
    print(f"\n--- AUDIO FEATURES ---")
    
    # Tempo and rhythm
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Estimated tempo: {tempo:.1f} BPM")
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    print(f"Spectral centroid (mean): {np.mean(spectral_centroids):.1f} Hz")
    print(f"Spectral rolloff (mean): {np.mean(spectral_rolloff):.1f} Hz") 
    print(f"Zero crossing rate (mean): {np.mean(zero_crossing_rate):.4f}")
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"MFCCs shape: {mfccs.shape}")
    print(f"First MFCC (mean): {np.mean(mfccs[0]):.3f}")
    
    # Dynamic range
    rms = librosa.feature.rms(y=y)[0]
    dynamic_range = np.max(rms) - np.min(rms)
    print(f"RMS dynamic range: {dynamic_range:.4f}")
    
    print(f"\n--- PERCEPTUAL CORRELATION ANALYSIS ---")
    
    # Let's look at some specific correlations
    timing_rating = perceptual_ratings[0]  # Timing_Stable_Unstable
    timbre_bright_dark = perceptual_ratings[7]  # Timbre_Bright_Dark
    dynamic_range_rating = perceptual_ratings[10]  # Dynamic range rating
    
    print(f"Timing stability rating: {timing_rating:.3f}")
    print(f"  -> Estimated tempo: {tempo:.1f} BPM")
    print(f"  -> Zero crossing rate: {np.mean(zero_crossing_rate):.4f}")
    
    print(f"Timbre brightness rating: {timbre_bright_dark:.3f}")
    print(f"  -> Spectral centroid: {np.mean(spectral_centroids):.1f} Hz")
    print(f"  -> Spectral rolloff: {np.mean(spectral_rolloff):.1f} Hz")
    
    print(f"Dynamic range rating: {dynamic_range_rating:.3f}")
    print(f"  -> RMS dynamic range: {dynamic_range:.4f}")
    
    return {
        'performance': performance_name,
        'audio_features': {
            'tempo': tempo,
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'dynamic_range': dynamic_range,
            'duration': len(y) / sr
        },
        'perceptual_ratings': {dim: rating for dim, rating in zip(perceptual_dimensions, perceptual_ratings)},
        'overall_score': np.mean(perceptual_ratings),
        'player_id': player_id
    }

def main():
    """Main analysis function"""
    print("=== PercePiano Audio-Visual Correlation Analysis ===\n")
    
    # Load perceptual data
    print("Loading perceptual ratings...")
    perceptual_data, perceptual_dimensions = load_perceptual_data()
    print(f"✅ Loaded {len(perceptual_data)} performance ratings")
    
    # Check available audio files
    example_audio = Path('examples/Beethoven_WoO80_var27_8bars_3_15.wav')
    
    if not example_audio.exists():
        print(f"❌ Audio file not found: {example_audio}")
        print("Available files:")
        examples_dir = Path('examples')
        if examples_dir.exists():
            for f in examples_dir.iterdir():
                print(f"  - {f.name}")
        return
    
    # Analyze the example audio file
    performance_name = example_audio.stem  # Remove .wav extension
    print(f"Found audio file: {example_audio}")
    print(f"Performance name: {performance_name}")
    
    result = analyze_audio_file(example_audio, performance_name, perceptual_data, perceptual_dimensions)
    
    if result:
        print(f"\n=== SUMMARY ===")
        print(f"Successfully analyzed: {performance_name}")
        print(f"Audio features extracted and correlated with perceptual ratings")
        print(f"Visualization saved as: {performance_name}_analysis.png")
        
        # Save results
        with open(f'{performance_name}_audio_analysis.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {performance_name}_audio_analysis.json")

if __name__ == "__main__":
    main()
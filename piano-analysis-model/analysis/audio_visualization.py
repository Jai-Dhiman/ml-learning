#!/usr/bin/env python3
"""
Audio Waveform and Spectrogram Visualization for PercePiano
Phase 1 Recreation: Building intuition about audio features
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import librosa
    import librosa.display
    print("✅ librosa available for visualization")
except ImportError:
    print("❌ librosa not available - run: pip install librosa")
    exit(1)

def load_perceptual_data():
    """Load perceptual labels and dimensions"""
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

def create_audio_visualization(audio_path, performance_name, perceptual_data, perceptual_dimensions):
    """Create comprehensive audio visualization with perceptual context"""
    
    print(f"\n=== VISUALIZING: {performance_name} ===")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    # Get perceptual ratings
    ratings = perceptual_data[performance_name]
    perceptual_ratings = ratings[:-1]
    player_id = ratings[-1]
    overall_score = np.mean(perceptual_ratings)
    
    print(f"Audio: {duration:.2f}s, {sr}Hz")
    print(f"Overall perceptual score: {overall_score:.3f}")
    
    # Create the visualization
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Audio Analysis: {performance_name}\\nOverall Score: {overall_score:.3f} | Player: {player_id}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Waveform (top)
    plt.subplot(3, 2, 1)
    time_axis = np.linspace(0, duration, len(y))
    plt.plot(time_axis, y, color='blue', alpha=0.7, linewidth=0.5)
    plt.title('Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # 2. Spectrogram
    plt.subplot(3, 2, 2) 
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, cmap='viridis')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    # 3. Mel-spectrogram
    plt.subplot(3, 2, 3)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', sr=sr, cmap='magma')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    # 4. Chromagram (harmony)
    plt.subplot(3, 2, 4)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='coolwarm')
    plt.title('Chromagram (Harmony)')
    plt.colorbar()
    
    # 5. Extreme perceptual ratings
    plt.subplot(3, 2, 5)
    rating_pairs = [(perceptual_dimensions[i], perceptual_ratings[i]) 
                   for i in range(len(perceptual_ratings))]
    rating_pairs.sort(key=lambda x: x[1])
    
    # Show lowest 4 and highest 4
    extreme_ratings = rating_pairs[:4] + rating_pairs[-4:]
    names = [dim.split('_')[-1][:12] for dim, _ in extreme_ratings]  # Shortened names
    values = [rating for _, rating in extreme_ratings]
    colors = ['red'] * 4 + ['green'] * 4
    
    bars = plt.barh(names, values, color=colors, alpha=0.8)
    plt.title('Extreme Perceptual Ratings')
    plt.xlabel('Rating (0-1)')
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Average')
    plt.axvline(x=overall_score, color='purple', linestyle='-', alpha=0.7, label='Overall')
    plt.legend()
    
    # 6. Audio feature summary
    plt.subplot(3, 2, 6)
    
    # Calculate key audio features
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    except:
        tempo_val = 0
        
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    rms = librosa.feature.rms(y=y)[0]
    dynamic_range = float(np.max(rms) - np.min(rms))
    
    # Text summary
    plt.text(0.1, 0.9, f'Tempo: {tempo_val:.1f} BPM', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'Spectral Centroid: {spectral_centroid:.0f} Hz', fontsize=12, transform=plt.gca().transAxes)  
    plt.text(0.1, 0.7, f'Spectral Rolloff: {spectral_rolloff:.0f} Hz', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Zero Crossing Rate: {zero_crossing_rate:.4f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'Dynamic Range: {dynamic_range:.4f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'Duration: {duration:.2f} seconds', fontsize=12, transform=plt.gca().transAxes)
    
    # Correlations with specific dimensions
    timing_rating = perceptual_ratings[0]  # Timing
    brightness_rating = perceptual_ratings[7]  # Timbre Bright/Dark
    dynamic_rating = perceptual_ratings[10]  # Dynamic range
    
    plt.text(0.1, 0.2, f'Timing Rating: {timing_rating:.3f}', fontsize=11, color='blue', transform=plt.gca().transAxes)
    plt.text(0.1, 0.1, f'Brightness Rating: {brightness_rating:.3f}', fontsize=11, color='orange', transform=plt.gca().transAxes)
    plt.text(0.1, 0.0, f'Dynamic Rating: {dynamic_rating:.3f}', fontsize=11, color='red', transform=plt.gca().transAxes)
    
    plt.title('Audio Features Summary')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = f'{performance_name}_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {output_path}")
    
    plt.show()
    
    return {
        'tempo': tempo_val,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'zero_crossing_rate': zero_crossing_rate,
        'dynamic_range': dynamic_range,
        'duration': duration
    }

def main():
    """Main visualization function"""
    print("=== PercePiano Audio Visualization ===")
    
    # Load data
    perceptual_data, perceptual_dimensions = load_perceptual_data()
    print(f"✅ Loaded {len(perceptual_data)} performance ratings")
    
    # Analyze the example audio
    example_audio = Path('examples/Beethoven_WoO80_var27_8bars_3_15.wav')
    performance_name = example_audio.stem
    
    if not example_audio.exists():
        print(f"❌ Audio file not found: {example_audio}")
        return
    
    if performance_name not in perceptual_data:
        print(f"❌ No perceptual data for {performance_name}")
        return
    
    # Create visualization
    audio_features = create_audio_visualization(
        example_audio, performance_name, perceptual_data, perceptual_dimensions
    )
    
    print(f"\\n🎯 NEXT STEPS:")
    print(f"1. Compare these features with ratings from other performances")
    print(f"2. Look for patterns: bright timbre → high spectral centroid?")
    print(f"3. Test timing stability → consistent tempo/rhythm?")
    print(f"4. Build feature extraction pipeline for full dataset")

if __name__ == "__main__":
    main()
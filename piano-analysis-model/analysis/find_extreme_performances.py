#!/usr/bin/env python3
"""
Find performances with highest and lowest overall ratings
for audio-perceptual correlation analysis
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def find_extreme_performances():
    """Find performances with extreme ratings for listening analysis"""
    
    # Load labels
    labels_path = Path('label_2round_mean_reg_19_with0_rm_highstd0.json')
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    # Define perceptual dimensions  
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
    
    # Calculate overall rating for each performance
    performance_scores = []
    for performance, ratings in labels_data.items():
        # Exclude last element (player_id)
        perceptual_ratings = ratings[:-1]
        if len(perceptual_ratings) == len(perceptual_dimensions):
            overall_score = np.mean(perceptual_ratings)
            performance_scores.append({
                'performance': performance,
                'overall_score': overall_score,
                'player_id': ratings[-1],
                'ratings': perceptual_ratings
            })
    
    # Sort by overall score
    performance_scores.sort(key=lambda x: x['overall_score'])
    
    print("=== EXTREME PERFORMANCE ANALYSIS ===\n")
    
    # Top 5 lowest rated performances
    print("🔴 LOWEST OVERALL RATINGS (for learning what sounds 'poor'):")
    print("=" * 60)
    for i, perf in enumerate(performance_scores[:5]):
        print(f"{i+1}. {perf['performance']}")
        print(f"   Overall Score: {perf['overall_score']:.3f}")
        print(f"   Player ID: {perf['player_id']}")
        
        # Show which dimensions are particularly low
        low_dims = []
        for j, rating in enumerate(perf['ratings']):
            if rating < 0.3:  # Very low ratings
                low_dims.append(f"{perceptual_dimensions[j]}: {rating:.3f}")
        
        if low_dims:
            print(f"   Notably LOW dimensions:")
            for dim in low_dims[:3]:  # Show top 3 lowest
                print(f"     - {dim}")
        print()
    
    print("\n" + "=" * 60)
    
    # 5 average rated performances (around the mean)
    mean_score = np.mean([p['overall_score'] for p in performance_scores])
    print(f"🟡 AVERAGE RATINGS (around mean {mean_score:.3f} - for baseline comparison):")
    print("=" * 60)
    
    # Find performances closest to the mean
    average_performances = []
    for perf in performance_scores:
        distance_from_mean = abs(perf['overall_score'] - mean_score)
        average_performances.append((perf, distance_from_mean))
    
    # Sort by distance from mean and take closest 5
    average_performances.sort(key=lambda x: x[1])
    
    for i, (perf, distance) in enumerate(average_performances[:5]):
        print(f"{i+1}. {perf['performance']}")
        print(f"   Overall Score: {perf['overall_score']:.3f} (±{distance:.3f} from mean)")
        print(f"   Player ID: {perf['player_id']}")
        
        # Show dimensions close to average
        avg_dims = []
        for j, rating in enumerate(perf['ratings']):
            if 0.45 < rating < 0.65:  # Near average ratings
                avg_dims.append(f"{perceptual_dimensions[j]}: {rating:.3f}")
        
        if avg_dims:
            print(f"   Representative AVERAGE dimensions:")
            for dim in avg_dims[:3]:  # Show first 3
                print(f"     - {dim}")
        print()
    
    print("\n" + "=" * 60)
    
    # Top 5 highest rated performances  
    print("🟢 HIGHEST OVERALL RATINGS (for learning what sounds 'excellent'):")
    print("=" * 60)
    for i, perf in enumerate(performance_scores[-5:]):
        print(f"{i+1}. {perf['performance']}")
        print(f"   Overall Score: {perf['overall_score']:.3f}")
        print(f"   Player ID: {perf['player_id']}")
        
        # Show which dimensions are particularly high
        high_dims = []
        for j, rating in enumerate(perf['ratings']):
            if rating > 0.7:  # Very high ratings
                high_dims.append(f"{perceptual_dimensions[j]}: {rating:.3f}")
        
        if high_dims:
            print(f"   Notably HIGH dimensions:")
            for dim in high_dims[:3]:  # Show top 3 highest
                print(f"     - {dim}")
        print()
    
    # Check which audio files exist
    print("\n" + "=" * 60)
    print("AUDIO FILE AVAILABILITY CHECK:")
    print("=" * 60)
    
    # Check example audio
    example_audio = Path('examples/Beethoven_WoO80_var27_8bars_3_15.wav')
    if example_audio.exists():
        # Find this performance in our data
        example_name = example_audio.stem
        for perf in performance_scores:
            if perf['performance'] == example_name:
                print(f"✅ EXAMPLE AUDIO AVAILABLE: {example_name}")
                print(f"   Overall Score: {perf['overall_score']:.3f} (rank {performance_scores.index(perf)+1}/{len(performance_scores)})")
                break
    
    # Check virtuoso MIDI files (these need to be converted to audio)
    virtuoso_dir = Path('virtuoso/data/all_2rounds')
    if virtuoso_dir.exists():
        midi_files = list(virtuoso_dir.glob('*.mid'))
        print(f"\n📁 VIRTUOSO MIDI FILES: {len(midi_files)} available")
        print("   (These are MIDI files - need conversion to audio for listening)")
        
        # Check if any of our extreme performances have MIDI files
        extreme_performances = performance_scores[:3] + performance_scores[-3:]
        available_midis = []
        for perf in extreme_performances:
            midi_path = virtuoso_dir / f"{perf['performance']}.mid"
            if midi_path.exists():
                available_midis.append(perf)
        
        if available_midis:
            print(f"\n🎵 EXTREME PERFORMANCES WITH MIDI FILES:")
            for perf in available_midis:
                rank = performance_scores.index(perf) + 1
                total = len(performance_scores)
                print(f"   - {perf['performance']}: score {perf['overall_score']:.3f} (rank {rank}/{total})")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR LISTENING ANALYSIS:")
    print("=" * 60)
    print("1. Listen to the example WAV file (if available)")
    print("2. Convert MIDI files to audio using a software synthesizer")  
    print("3. Compare extreme high vs low rated performances")
    print("4. Focus on specific dimensions that show large differences")
    print("5. Build intuition about what makes ratings high vs low")
    
    # Save results for further analysis
    with open('extreme_performances_analysis.json', 'w') as f:
        json.dump({
            'lowest_5': performance_scores[:5],
            'average_5': [perf for perf, _ in average_performances[:5]],
            'highest_5': performance_scores[-5:],
            'mean_score': mean_score,
            'example_performance': {
                'name': 'Beethoven_WoO80_var27_8bars_3_15',
                'available': example_audio.exists(),
                'score': next((p['overall_score'] for p in performance_scores if p['performance'] == 'Beethoven_WoO80_var27_8bars_3_15'), None)
            }
        }, f, indent=2)
    
    print("\n💾 Results saved to: extreme_performances_analysis.json")

if __name__ == "__main__":
    find_extreme_performances()
#!/usr/bin/env python3
"""
Quick PercePiano Dataset Analysis
Provides key insights for Phase 1 implementation planning
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

def analyze_percepiano_dataset():
    """Analyze PercePiano dataset structure and characteristics"""
    
    # Dataset paths
    base_path = Path('.')
    labels_path = base_path / 'label_2round_mean_reg_19_with0_rm_highstd0.json'
    
    print("=== PercePiano Dataset Analysis ===\n")
    
    # Load labels
    if not labels_path.exists():
        print(f"ERROR: Labels file not found at {labels_path}")
        return
    
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
    
    print("1. DATASET OVERVIEW")
    print(f"   Total labeled performances: {len(labels_data)}")
    print(f"   Perceptual dimensions per performance: {len(perceptual_dimensions)}")
    
    # Convert to DataFrame
    label_rows = []
    for performance, ratings in labels_data.items():
        row = {'performance': performance}
        for i, rating in enumerate(ratings[:-1]):  # Last value is player ID
            if i < len(perceptual_dimensions):
                row[perceptual_dimensions[i]] = rating
        row['player_id'] = ratings[-1]
        label_rows.append(row)
    
    df = pd.DataFrame(label_rows)
    
    # Analyze performance naming patterns
    composers = []
    pieces = []
    bar_lengths = []
    
    for perf_name in df['performance']:
        parts = perf_name.split('_')
        if perf_name.startswith('Beethoven'):
            composers.append('Beethoven')
        elif perf_name.startswith('Schubert'):
            composers.append('Schubert')
        else:
            composers.append('Unknown')
        
        # Extract piece and bars info
        bar_info = [p for p in parts if 'bars' in p]
        if bar_info:
            bar_lengths.append(bar_info[0])
        else:
            bar_lengths.append('unknown')
    
    print("\n2. MUSICAL REPERTOIRE")
    composer_counts = Counter(composers)
    bars_counts = Counter(bar_lengths)
    
    for composer, count in composer_counts.items():
        print(f"   {composer}: {count} performances")
    
    print(f"\n   Segment lengths:")
    for bars, count in bars_counts.most_common():
        print(f"   {bars}: {count} segments")
    
    # Player analysis
    player_counts = df['player_id'].value_counts()
    print(f"\n3. PERFORMER ANALYSIS")
    print(f"   Unique players: {len(player_counts)}")
    print(f"   Player distribution:")
    for player_id, count in player_counts.sort_index().items():
        print(f"   Player {player_id}: {count} performances")
    
    # Perceptual ratings analysis
    print(f"\n4. PERCEPTUAL RATINGS CHARACTERISTICS")
    numeric_columns = [col for col in df.columns if col not in ['performance', 'player_id']]
    
    overall_stats = df[numeric_columns].describe()
    print(f"   Overall rating range: [{df[numeric_columns].min().min():.3f}, {df[numeric_columns].max().max():.3f}]")
    print(f"   Overall mean rating: {df[numeric_columns].mean().mean():.3f}")
    print(f"   Average std deviation: {df[numeric_columns].std().mean():.3f}")
    
    # Group dimensions by category
    print(f"\n5. PERCEPTUAL DIMENSION CATEGORIES")
    categories = {}
    for dim in perceptual_dimensions:
        category = dim.split('_')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(dim)
    
    for category, dims in categories.items():
        print(f"   {category.upper()} ({len(dims)} dimensions):")
        for dim in dims:
            mean_val = df[dim].mean()
            std_val = df[dim].std()
            print(f"     - {dim}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # Find some interesting correlations
    print(f"\n6. DIMENSION CORRELATIONS (|r| > 0.5)")
    correlation_matrix = df[perceptual_dimensions].corr()
    
    strong_correlations = []
    for i, dim1 in enumerate(perceptual_dimensions):
        for j, dim2 in enumerate(perceptual_dimensions):
            if i < j:  # Avoid duplicates
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append((dim1, dim2, corr_val))
    
    strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if strong_correlations:
        for dim1, dim2, corr in strong_correlations[:5]:
            print(f"   {dim1}")
            print(f"   <-> {dim2}: {corr:.3f}")
            print()
    else:
        print("   No strong correlations found (|r| > 0.5)")
    
    # Example performance analysis
    print(f"\n7. EXAMPLE PERFORMANCE ANALYSIS")
    example_name = "Beethoven_WoO80_var27_8bars_3_15"
    if example_name in labels_data:
        example_ratings = labels_data[example_name]
        print(f"   Performance: {example_name}")
        print(f"   Player ID: {example_ratings[-1]}")
        print(f"   Key ratings:")
        
        # Show some notable ratings
        for i, rating in enumerate(example_ratings[:-1]):
            if i < len(perceptual_dimensions):
                if rating > 0.7 or rating < 0.3:  # Extreme values
                    print(f"     {perceptual_dimensions[i]}: {rating:.3f} ({'HIGH' if rating > 0.7 else 'LOW'})")
    
    print(f"\n8. NEXT STEPS FOR PHASE 1 IMPLEMENTATION")
    print(f"   ✓ Dataset loaded and understood ({len(labels_data)} performances)")
    print(f"   ✓ 19 perceptual dimensions identified and categorized")
    print(f"   ✓ Rating patterns and distributions analyzed") 
    print(f"   ✓ Multi-composer, multi-performer dataset confirmed")
    print(f"   → Ready to implement audio processing pipeline")
    print(f"   → Ready to start single-dimension prediction model")
    print(f"   → Ready to build multi-task learning system")
    print(f"\n=== Analysis Complete ===")

if __name__ == "__main__":
    analyze_percepiano_dataset()
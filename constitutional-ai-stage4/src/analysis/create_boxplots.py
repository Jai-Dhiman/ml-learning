"""
Boxplot Generator for Constitutional AI Evaluation

Creates box plots showing score distributions for each principle across models,
displaying median, quartiles, and outliers.

Author: J. Dhiman
Date: October 4, 2025
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def extract_principle_scores(
    results: Dict,
    model_name: str,
    principle: str
) -> List[float]:
    """Extract all scores for a specific principle from a model."""
    scores = []
    
    try:
        model_results = results['results'][model_name]
        principle_results = model_results['principle_results'].get(principle, [])
        
        for result in principle_results:
            if isinstance(result, dict) and 'score' in result:
                scores.append(result['score'])
            elif isinstance(result, (int, float)):
                scores.append(result)
                
    except (KeyError, TypeError) as e:
        print(f"Warning: Could not extract scores for {model_name}/{principle}: {e}")
    
    return scores


def create_boxplots(
    results: Dict,
    models: List[str],
    principles: List[str],
    output_path: Path
):
    """
    Create and save box plots showing score distributions.
    
    Args:
        results: Full evaluation results
        models: List of model names
        principles: List of constitutional principles
        output_path: Path to save the PDF
    """
    # Model display names
    model_display_names = {
        'base': 'Base',
        'stage2_helpful': 'Stage 2',
        'stage3_constitutional': 'Stage 3'
    }
    
    # Colors for each model
    colors = {
        'base': '#1f77b4',  # Blue
        'stage2_helpful': '#ff7f0e',  # Orange
        'stage3_constitutional': '#2ca02c'  # Green
    }
    
    # Create subplots: one per principle
    num_principles = len(principles)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, principle in enumerate(principles):
        ax = axes[idx]
        
        # Extract scores for all models
        data_to_plot = []
        labels = []
        box_colors = []
        
        for model in models:
            scores = extract_principle_scores(results, model, principle)
            if scores:
                data_to_plot.append(scores)
                labels.append(model_display_names.get(model, model))
                box_colors.append(colors.get(model, '#333333'))
        
        if not data_to_plot:
            print(f"Warning: No data for principle {principle}")
            continue
        
        # Create boxplot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize appearance
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        # Add title and labels
        principle_label = principle.replace('_', ' ').title()
        ax.set_title(f'{principle_label}', fontsize=14, weight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at 0.5 (neutral)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Rotate x labels if needed
        ax.tick_params(axis='x', rotation=0)
    
    # Add overall title
    fig.suptitle('Score Distributions by Constitutional Principle', 
                 fontsize=16, weight='bold', y=0.98)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[model], label=model_display_names[model], alpha=0.7) 
                      for model in models]
    fig.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.98, 0.96), fontsize=11)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ Box plots saved to: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate box plots for Constitutional AI evaluation'
    )
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to evaluation results JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for box plot PDF'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['base', 'stage2_helpful', 'stage3_constitutional'],
        help='Models to include in plots'
    )
    parser.add_argument(
        '--principles',
        nargs='+',
        default=['harm_prevention', 'truthfulness', 'helpfulness', 'fairness'],
        help='Constitutional principles to display'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Box Plot Generator")
    print("="*70)
    print(f"Results file: {args.results}")
    print(f"Output file: {args.output}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Principles: {', '.join(args.principles)}")
    print("="*70)
    
    # Load results
    print("\nLoading evaluation results...")
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Create box plots
    print("Generating box plots...")
    create_boxplots(results, args.models, args.principles, args.output)
    
    print("\n✓ Box plot generation complete!")


if __name__ == '__main__':
    main()

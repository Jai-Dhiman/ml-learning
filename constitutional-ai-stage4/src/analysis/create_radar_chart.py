"""
Radar Chart Generator for Constitutional AI Evaluation

Creates publication-quality radar chart comparing all models across
the four constitutional principles.

Author: J. Dhiman
Date: October 4, 2025
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def extract_aggregate_scores(results: Dict, model_name: str, principles: List[str]) -> List[float]:
    """
    Extract mean scores for each principle for a given model.
    
    Args:
        results: Full evaluation results
        model_name: Name of the model
        principles: List of constitutional principles
        
    Returns:
        List of mean scores for each principle
    """
    scores = []
    
    for principle in principles:
        try:
            model_results = results['results'][model_name]
            principle_results = model_results['principle_results'].get(principle, [])
            
            principle_scores = []
            for result in principle_results:
                if isinstance(result, dict) and 'score' in result:
                    principle_scores.append(result['score'])
                elif isinstance(result, (int, float)):
                    principle_scores.append(result)
            
            if principle_scores:
                scores.append(np.mean(principle_scores))
            else:
                scores.append(0.0)
                
        except (KeyError, TypeError) as e:
            print(f"Warning: Could not extract scores for {model_name}/{principle}: {e}")
            scores.append(0.0)
    
    return scores


def create_radar_chart(
    results: Dict,
    models: List[str],
    principles: List[str],
    output_path: Path
):
    """
    Create and save radar chart comparing models across principles.
    
    Args:
        results: Full evaluation results
        models: List of model names to compare
        principles: List of constitutional principles
        output_path: Path to save the PDF
    """
    # Extract scores for all models
    model_scores = {}
    for model in models:
        model_scores[model] = extract_aggregate_scores(results, model, principles)
    
    # Number of principles
    num_vars = len(principles)
    
    # Compute angle for each principle
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Close the plot by appending the first value
    for model in models:
        model_scores[model] += model_scores[model][:1]
    angles += angles[:1]
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Model display names and colors
    model_display_names = {
        'base': 'Base (Gemma 2B-IT)',
        'stage2_helpful': 'Stage 2 (Helpful RLHF)',
        'stage3_constitutional': 'Stage 3 (Constitutional AI + DPO)'
    }
    
    colors = {
        'base': '#1f77b4',  # Blue
        'stage2_helpful': '#ff7f0e',  # Orange
        'stage3_constitutional': '#2ca02c'  # Green
    }
    
    # Plot data for each model
    for model in models:
        display_name = model_display_names.get(model, model)
        color = colors.get(model, '#333333')
        
        ax.plot(angles, model_scores[model], 'o-', linewidth=2, label=display_name, color=color)
        ax.fill(angles, model_scores[model], alpha=0.15, color=color)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    principle_labels = [p.replace('_', ' ').title() for p in principles]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(principle_labels, size=12)
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    
    # Add gridlines
    ax.grid(True, linewidth=0.5, alpha=0.5)
    
    # Add title
    plt.title('Constitutional Principle Scores by Model', 
              size=16, weight='bold', pad=20)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ Radar chart saved to: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate radar chart for Constitutional AI evaluation'
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
        help='Output path for radar chart PDF'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['base', 'stage2_helpful', 'stage3_constitutional'],
        help='Models to include in chart'
    )
    parser.add_argument(
        '--principles',
        nargs='+',
        default=['harm_prevention', 'truthfulness', 'helpfulness', 'fairness'],
        help='Constitutional principles to display'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Radar Chart Generator")
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
    
    # Create radar chart
    print("Generating radar chart...")
    create_radar_chart(results, args.models, args.principles, args.output)
    
    print("\n✓ Radar chart generation complete!")


if __name__ == '__main__':
    main()

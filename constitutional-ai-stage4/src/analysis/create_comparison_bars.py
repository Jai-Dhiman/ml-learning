"""
Comparison Bar Chart Generator for Constitutional AI Evaluation

Creates bar charts with confidence intervals showing progression from
Base -> Stage 2 -> Stage 3, with statistical significance markers.

Author: J. Dhiman
Date: October 4, 2025
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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


def calculate_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate mean and confidence interval.
    
    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return (0.0, 0.0, 0.0)
    
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=std_err)
    
    return (mean, ci[0], ci[1])


def get_significance_marker(
    significance_data: Dict,
    comparison: str,
    principle: str
) -> str:
    """
    Get statistical significance marker from significance test results.
    
    Returns:
        String marker: '**' for p<0.01, '*' for p<0.05, '' otherwise
    """
    try:
        for comp in significance_data.get(comparison, []):
            if comp['principle'] == principle:
                p_val = comp.get('p_value_corrected', 1.0)
                if p_val < 0.01:
                    return '**'
                elif p_val < 0.05:
                    return '*'
                break
    except (KeyError, TypeError):
        pass
    
    return ''


def create_comparison_bars(
    results: Dict,
    significance_data: Dict,
    models: List[str],
    principles: List[str],
    output_path: Path
):
    """
    Create and save comparison bar chart with error bars.
    
    Args:
        results: Full evaluation results
        significance_data: Statistical significance test results
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
    
    # Extract data for all models and principles
    data = {}
    for model in models:
        data[model] = {}
        for principle in principles:
            scores = extract_principle_scores(results, model, principle)
            mean, ci_lower, ci_upper = calculate_confidence_interval(scores)
            data[model][principle] = {
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'error': mean - ci_lower  # For error bar
            }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Bar width and positions
    bar_width = 0.25
    x_pos = np.arange(len(models))
    
    for idx, principle in enumerate(principles):
        ax = axes[idx]
        
        # Extract means and errors for this principle
        means = [data[model][principle]['mean'] for model in models]
        errors = [data[model][principle]['error'] for model in models]
        bar_colors = [colors[model] for model in models]
        labels = [model_display_names[model] for model in models]
        
        # Create bars
        bars = ax.bar(x_pos, means, bar_width*2, yerr=errors, 
                     color=bar_colors, alpha=0.8, capsize=5,
                     error_kw={'linewidth': 2, 'ecolor': 'black'})
        
        # Add significance markers
        # Stage 3 vs Base
        sig_marker = get_significance_marker(
            significance_data, 'stage3_vs_base', principle
        )
        if sig_marker:
            max_height = max(means) + max(errors)
            ax.text(x_pos[2], max_height + 0.05, sig_marker, 
                   ha='center', va='bottom', fontsize=16, weight='bold')
            # Draw bracket
            bracket_height = max_height + 0.02
            ax.plot([x_pos[0], x_pos[2]], [bracket_height, bracket_height],
                   'k-', linewidth=1.5)
        
        # Customize appearance
        ax.set_ylabel('Score', fontsize=13)
        ax.set_title(principle.replace('_', ' ').title(), fontsize=15, weight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + errors[i] + 0.02,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    # Add overall title
    fig.suptitle('Model Comparison Across Constitutional Principles\n(with 95% Confidence Intervals)', 
                 fontsize=17, weight='bold', y=0.98)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[model], label=model_display_names[model], alpha=0.8) 
                      for model in models]
    legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))
    legend_elements.append(Patch(facecolor='none', edgecolor='none', label='Significance:'))
    legend_elements.append(Patch(facecolor='none', edgecolor='none', label='* p < 0.05'))
    legend_elements.append(Patch(facecolor='none', edgecolor='none', label='** p < 0.01'))
    
    fig.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.98, 0.96), fontsize=11)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ Comparison bars saved to: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison bar chart for Constitutional AI evaluation'
    )
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to evaluation results JSON file'
    )
    parser.add_argument(
        '--significance',
        type=Path,
        required=True,
        help='Path to statistical significance test results JSON'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        required=True,
        help='Output path for comparison bar chart PDF'
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
    print("Comparison Bar Chart Generator")
    print("="*70)
    print(f"Results file: {args.results}")
    print(f"Significance file: {args.significance}")
    print(f"Output file: {args.output_file}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Principles: {', '.join(args.principles)}")
    print("="*70)
    
    # Load results
    print("\nLoading evaluation results...")
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Load significance data
    print("Loading significance test results...")
    try:
        with open(args.significance, 'r') as f:
            significance_data = json.load(f)
    except FileNotFoundError:
        print("Warning: Significance data not found. Continuing without significance markers.")
        significance_data = {}
    
    # Create comparison bars
    print("Generating comparison bar chart...")
    create_comparison_bars(results, significance_data, args.models, args.principles, args.output_file)
    
    print("\n✓ Comparison bar chart generation complete!")


if __name__ == '__main__':
    main()

"""
Effect Size Calculation for Constitutional AI Evaluation

Calculates Cohen's d and other effect size metrics to assess the practical
significance of improvements from Base -> Stage 2 -> Stage 3.

Author: J. Dhiman
Date: October 4, 2025
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class EffectSizeResult:
    """Effect size comparison between two models."""
    model1: str
    model2: str
    principle: str
    model1_mean: float
    model2_mean: float
    mean_difference: float
    pooled_std: float
    cohens_d: float
    interpretation: str
    improvement_percentage: float


def extract_principle_scores(
    results: Dict,
    model_name: str,
    principle: str
) -> List[float]:
    """Extract scores for a specific principle from evaluation results."""
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


def calculate_pooled_std(scores1: List[float], scores2: List[float]) -> float:
    """
    Calculate pooled standard deviation for two groups.
    
    Args:
        scores1: Scores from group 1
        scores2: Scores from group 2
        
    Returns:
        Pooled standard deviation
    """
    if len(scores1) < 2 or len(scores2) < 2:
        return 0.0
    
    n1, n2 = len(scores1), len(scores2)
    var1 = np.var(scores1, ddof=1)
    var2 = np.var(scores2, ddof=1)
    
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    return np.sqrt(pooled_var)


def calculate_cohens_d(
    scores1: List[float],
    scores2: List[float]
) -> float:
    """
    Calculate Cohen's d effect size.
    
    Cohen's d = (mean2 - mean1) / pooled_std
    
    Args:
        scores1: Scores from group 1 (baseline)
        scores2: Scores from group 2 (comparison)
        
    Returns:
        Cohen's d value
    """
    if not scores1 or not scores2:
        return 0.0
    
    mean1 = np.mean(scores1)
    mean2 = np.mean(scores2)
    pooled_std = calculate_pooled_std(scores1, scores2)
    
    if pooled_std == 0:
        return 0.0
    
    return (mean2 - mean1) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d using standard thresholds.
    
    Thresholds (Cohen, 1988):
    - Small: |d| = 0.2
    - Medium: |d| = 0.5
    - Large: |d| = 0.8
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    direction = "improvement" if d > 0 else "decline" if d < 0 else "no change"
    
    return f"{magnitude} {direction}"


def calculate_improvement_percentage(mean1: float, mean2: float) -> float:
    """
    Calculate percentage improvement from mean1 to mean2.
    
    Args:
        mean1: Baseline mean
        mean2: Comparison mean
        
    Returns:
        Percentage improvement (can be negative for decline)
    """
    if mean1 == 0:
        return 0.0
    
    return ((mean2 - mean1) / abs(mean1)) * 100


def compute_effect_sizes(
    results: Dict,
    model1: str,
    model2: str,
    principles: List[str]
) -> List[EffectSizeResult]:
    """
    Compute effect sizes for all principles between two models.
    
    Args:
        results: Full evaluation results
        model1: Baseline model name
        model2: Comparison model name
        principles: List of constitutional principles
        
    Returns:
        List of EffectSizeResult objects
    """
    effect_sizes = []
    
    for principle in principles:
        scores1 = extract_principle_scores(results, model1, principle)
        scores2 = extract_principle_scores(results, model2, principle)
        
        if not scores1 or not scores2:
            print(f"Warning: Skipping {principle} - insufficient data")
            continue
        
        mean1 = np.mean(scores1)
        mean2 = np.mean(scores2)
        mean_diff = mean2 - mean1
        pooled_std = calculate_pooled_std(scores1, scores2)
        cohens_d = calculate_cohens_d(scores1, scores2)
        interpretation = interpret_cohens_d(cohens_d)
        improvement_pct = calculate_improvement_percentage(mean1, mean2)
        
        effect_size = EffectSizeResult(
            model1=model1,
            model2=model2,
            principle=principle,
            model1_mean=float(mean1),
            model2_mean=float(mean2),
            mean_difference=float(mean_diff),
            pooled_std=float(pooled_std),
            cohens_d=float(cohens_d),
            interpretation=interpretation,
            improvement_percentage=float(improvement_pct)
        )
        
        effect_sizes.append(effect_size)
    
    return effect_sizes


def compute_all_effect_sizes(
    results: Dict,
    principles: List[str]
) -> Dict[str, List[EffectSizeResult]]:
    """
    Compute effect sizes for all relevant model comparisons.
    
    Args:
        results: Full evaluation results
        principles: List of constitutional principles
        
    Returns:
        Dictionary mapping comparison name to effect size results
    """
    comparisons = {}
    
    # Stage 3 vs Base
    comparisons['stage3_vs_base'] = compute_effect_sizes(
        results, 'base', 'stage3_constitutional', principles
    )
    
    # Stage 3 vs Stage 2
    comparisons['stage3_vs_stage2'] = compute_effect_sizes(
        results, 'stage2_helpful', 'stage3_constitutional', principles
    )
    
    # Stage 2 vs Base (for completeness)
    comparisons['stage2_vs_base'] = compute_effect_sizes(
        results, 'base', 'stage2_helpful', principles
    )
    
    return comparisons


def format_summary(comparisons: Dict[str, List[EffectSizeResult]]) -> str:
    """Create human-readable summary of effect sizes."""
    lines = []
    lines.append("="*80)
    lines.append("EFFECT SIZE ANALYSIS (COHEN'S D)")
    lines.append("="*80)
    lines.append("\nInterpretation Guide:")
    lines.append("  |d| < 0.2  = negligible effect")
    lines.append("  |d| = 0.2  = small effect")
    lines.append("  |d| = 0.5  = medium effect")
    lines.append("  |d| = 0.8+ = large effect")
    lines.append("="*80)
    
    for comp_name, results in comparisons.items():
        lines.append(f"\n{comp_name.upper().replace('_', ' ')}")
        lines.append("-"*80)
        
        for result in results:
            lines.append(f"\nPrinciple: {result.principle}")
            lines.append(f"  {result.model1} mean: {result.model1_mean:.4f}")
            lines.append(f"  {result.model2} mean: {result.model2_mean:.4f}")
            lines.append(f"  Mean difference: {result.mean_difference:+.4f}")
            lines.append(f"  Improvement: {result.improvement_percentage:+.1f}%")
            lines.append(f"  Pooled SD: {result.pooled_std:.4f}")
            lines.append(f"  Cohen's d: {result.cohens_d:+.4f}")
            lines.append(f"  Interpretation: {result.interpretation}")
    
    lines.append("\n" + "="*80)
    
    return "\n".join(lines)


def create_summary_table(comparisons: Dict[str, List[EffectSizeResult]]) -> Dict:
    """Create a summary table of effect sizes for easy reference."""
    summary = {}
    
    for comp_name, results in comparisons.items():
        summary[comp_name] = {}
        
        for result in results:
            summary[comp_name][result.principle] = {
                'cohens_d': result.cohens_d,
                'interpretation': result.interpretation,
                'improvement_pct': result.improvement_percentage
            }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Calculate effect sizes for Constitutional AI evaluation'
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
        help='Output path for effect size results'
    )
    parser.add_argument(
        '--principles',
        nargs='+',
        default=['harm_prevention', 'truthfulness', 'helpfulness', 'fairness'],
        help='Constitutional principles to analyze'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Effect Size Calculation")
    print("="*80)
    print(f"Results file: {args.results}")
    print(f"Output file: {args.output}")
    print(f"Principles: {', '.join(args.principles)}")
    print("="*80)
    
    # Load results
    print("\nLoading evaluation results...")
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Compute effect sizes
    print("Computing effect sizes...")
    comparisons = compute_all_effect_sizes(results, args.principles)
    
    # Convert to serializable format
    output_data = {}
    for comp_name, effect_sizes in comparisons.items():
        output_data[comp_name] = [asdict(es) for es in effect_sizes]
    
    # Add summary table
    output_data['summary_table'] = create_summary_table(comparisons)
    
    # Add metadata
    output_data['metadata'] = {
        'effect_size_metric': 'cohens_d',
        'interpretation_source': 'Cohen (1988)',
        'principles_analyzed': args.principles
    }
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    
    # Print summary
    summary = format_summary(comparisons)
    print("\n" + summary)


if __name__ == '__main__':
    main()

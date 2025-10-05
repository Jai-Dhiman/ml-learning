"""
Statistical Significance Testing for Constitutional AI Evaluation

Performs t-tests, calculates confidence intervals, and analyzes win rates
comparing Stage 3 vs Base and Stage 3 vs Stage 2 across all principles.

Author: J. Dhiman
Date: October 4, 2025
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from dataclasses import dataclass, asdict


@dataclass
class ComparisonResult:
    """Results of statistical comparison between two models."""
    model1: str
    model2: str
    principle: str
    model1_mean: float
    model2_mean: float
    model1_std: float
    model2_std: float
    model1_ci_lower: float
    model1_ci_upper: float
    model2_ci_lower: float
    model2_ci_upper: float
    t_statistic: float
    p_value: float
    p_value_corrected: float
    significant: bool
    win_rate: float  # Percentage where model2 scores higher than model1
    n_samples: int


def extract_principle_scores(
    results: Dict,
    model_name: str,
    principle: str
) -> List[float]:
    """
    Extract scores for a specific principle from evaluation results.
    
    Args:
        results: Full evaluation results dictionary
        model_name: Name of the model
        principle: Constitutional principle name
        
    Returns:
        List of scores for that principle
    """
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
) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean.
    
    Args:
        data: List of scores
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0)
    
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=std_err)
    
    return ci


def perform_t_test(
    scores1: List[float],
    scores2: List[float],
    model1_name: str,
    model2_name: str,
    principle: str
) -> ComparisonResult:
    """
    Perform independent samples t-test and calculate statistics.
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        principle: Constitutional principle being tested
        
    Returns:
        ComparisonResult with all statistics
    """
    # Calculate basic statistics
    mean1 = np.mean(scores1) if scores1 else 0.0
    mean2 = np.mean(scores2) if scores2 else 0.0
    std1 = np.std(scores1, ddof=1) if len(scores1) > 1 else 0.0
    std2 = np.std(scores2, ddof=1) if len(scores2) > 1 else 0.0
    
    # Calculate confidence intervals
    ci1 = calculate_confidence_interval(scores1)
    ci2 = calculate_confidence_interval(scores2)
    
    # Perform t-test
    if len(scores1) > 1 and len(scores2) > 1:
        t_stat, p_val = stats.ttest_ind(scores1, scores2)
    else:
        t_stat, p_val = 0.0, 1.0
    
    # Calculate win rate (how often model2 beats model1)
    win_rate = 0.0
    if len(scores1) == len(scores2) and len(scores1) > 0:
        wins = sum(1 for s1, s2 in zip(scores1, scores2) if s2 > s1)
        win_rate = (wins / len(scores1)) * 100
    
    return ComparisonResult(
        model1=model1_name,
        model2=model2_name,
        principle=principle,
        model1_mean=float(mean1),
        model2_mean=float(mean2),
        model1_std=float(std1),
        model2_std=float(std2),
        model1_ci_lower=float(ci1[0]),
        model1_ci_upper=float(ci1[1]),
        model2_ci_lower=float(ci2[0]),
        model2_ci_upper=float(ci2[1]),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        p_value_corrected=float(p_val),  # Will be corrected later
        significant=False,  # Will be determined after correction
        win_rate=float(win_rate),
        n_samples=len(scores1)
    )


def apply_bonferroni_correction(
    comparisons: List[ComparisonResult],
    alpha: float = 0.05
) -> List[ComparisonResult]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        comparisons: List of comparison results
        alpha: Significance level (default 0.05)
        
    Returns:
        Updated comparisons with corrected p-values
    """
    n_comparisons = len(comparisons)
    corrected_alpha = alpha / n_comparisons
    
    for comparison in comparisons:
        comparison.p_value_corrected = comparison.p_value * n_comparisons
        comparison.significant = comparison.p_value_corrected < alpha
    
    return comparisons


def run_all_comparisons(
    results: Dict,
    principles: List[str]
) -> Dict[str, List[ComparisonResult]]:
    """
    Run all pairwise comparisons for all principles.
    
    Args:
        results: Full evaluation results
        principles: List of constitutional principles to test
        
    Returns:
        Dictionary mapping comparison name to list of results
    """
    models = ['base', 'stage2_helpful', 'stage3_constitutional']
    comparisons_dict = {}
    
    # Stage 3 vs Base
    comp_name = 'stage3_vs_base'
    comparisons = []
    
    for principle in principles:
        scores_base = extract_principle_scores(results, 'base', principle)
        scores_stage3 = extract_principle_scores(results, 'stage3_constitutional', principle)
        
        if scores_base and scores_stage3:
            comp = perform_t_test(
                scores_base, scores_stage3,
                'base', 'stage3_constitutional',
                principle
            )
            comparisons.append(comp)
    
    comparisons_dict[comp_name] = apply_bonferroni_correction(comparisons)
    
    # Stage 3 vs Stage 2
    comp_name = 'stage3_vs_stage2'
    comparisons = []
    
    for principle in principles:
        scores_stage2 = extract_principle_scores(results, 'stage2_helpful', principle)
        scores_stage3 = extract_principle_scores(results, 'stage3_constitutional', principle)
        
        if scores_stage2 and scores_stage3:
            comp = perform_t_test(
                scores_stage2, scores_stage3,
                'stage2_helpful', 'stage3_constitutional',
                principle
            )
            comparisons.append(comp)
    
    comparisons_dict[comp_name] = apply_bonferroni_correction(comparisons)
    
    return comparisons_dict


def format_summary(comparisons_dict: Dict[str, List[ComparisonResult]]) -> str:
    """Create human-readable summary of results."""
    lines = []
    lines.append("="*80)
    lines.append("STATISTICAL SIGNIFICANCE TESTING RESULTS")
    lines.append("="*80)
    
    for comp_name, comparisons in comparisons_dict.items():
        lines.append(f"\n{comp_name.upper().replace('_', ' ')}")
        lines.append("-"*80)
        
        for comp in comparisons:
            sig_marker = "**" if comp.p_value_corrected < 0.01 else ("*" if comp.significant else "")
            
            lines.append(f"\nPrinciple: {comp.principle}")
            lines.append(f"  {comp.model1}: {comp.model1_mean:.4f} ± {comp.model1_std:.4f} "
                        f"(95% CI: [{comp.model1_ci_lower:.4f}, {comp.model1_ci_upper:.4f}])")
            lines.append(f"  {comp.model2}: {comp.model2_mean:.4f} ± {comp.model2_std:.4f} "
                        f"(95% CI: [{comp.model2_ci_lower:.4f}, {comp.model2_ci_upper:.4f}])")
            lines.append(f"  t-statistic: {comp.t_statistic:.4f}")
            lines.append(f"  p-value: {comp.p_value:.4f}")
            lines.append(f"  p-value (Bonferroni corrected): {comp.p_value_corrected:.4f} {sig_marker}")
            lines.append(f"  Win rate ({comp.model2} vs {comp.model1}): {comp.win_rate:.1f}%")
            lines.append(f"  Significant: {'YES' if comp.significant else 'NO'}")
    
    lines.append("\n" + "="*80)
    lines.append("* p < 0.05, ** p < 0.01")
    lines.append("="*80)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Statistical significance testing for Constitutional AI evaluation'
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
        help='Output path for significance test results'
    )
    parser.add_argument(
        '--principles',
        nargs='+',
        default=['harm_prevention', 'truthfulness', 'helpfulness', 'fairness'],
        help='Constitutional principles to test'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Statistical Significance Testing")
    print("="*80)
    print(f"Results file: {args.results}")
    print(f"Output file: {args.output}")
    print(f"Principles: {', '.join(args.principles)}")
    print(f"Alpha: {args.alpha}")
    print("="*80)
    
    # Load results
    print("\nLoading evaluation results...")
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Run all comparisons
    print("Running statistical tests...")
    comparisons_dict = run_all_comparisons(results, args.principles)
    
    # Convert to serializable format
    output_data = {}
    for comp_name, comparisons in comparisons_dict.items():
        output_data[comp_name] = [asdict(comp) for comp in comparisons]
    
    # Add summary statistics
    output_data['metadata'] = {
        'alpha': args.alpha,
        'correction_method': 'bonferroni',
        'principles_tested': args.principles,
        'n_comparisons_per_group': len(args.principles)
    }
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    
    # Print summary
    summary = format_summary(comparisons_dict)
    print("\n" + summary)


if __name__ == '__main__':
    main()

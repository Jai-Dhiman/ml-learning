"""
Extended Test Set Generator for Constitutional AI Evaluation

Creates a diverse test set of 110 prompts from the Stage 3 critique-revision pairs,
categorized by harm type and complexity for comprehensive evaluation.

Author: J. Dhiman
Date: October 4, 2025
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

# Seed for reproducibility
random.seed(42)


def categorize_prompt(prompt: str, base_response: str, revised_response: str) -> str:
    """
    Categorize a prompt based on content and response characteristics.
    
    Categories:
    - harmful: Prompts requesting dangerous/illegal/harmful actions
    - helpful: Safe, informational queries
    - edge_cases: Ambiguous, context-dependent, or unusual requests
    - adversarial: Red-team style prompts testing safety boundaries
    
    Args:
        prompt: The user prompt
        base_response: Base model response
        revised_response: Revised model response
        
    Returns:
        Category string
    """
    prompt_lower = prompt.lower()
    
    # Harmful indicators
    harmful_keywords = [
        'kill', 'murder', 'rob', 'steal', 'hack', 'bomb', 'weapon', 'torture',
        'illegal', 'crime', 'hurt', 'harm', 'attack', 'threaten', 'blackmail',
        'kidnap', 'poison', 'drug', 'abuse', 'assault', 'rape', 'suicide'
    ]
    
    # Helpful indicators
    helpful_keywords = [
        'how to', 'what is', 'can you explain', 'tell me about', 'help me understand',
        'advice', 'suggestion', 'recommendation', 'learn', 'improve', 'better'
    ]
    
    # Adversarial indicators (testing boundaries)
    adversarial_patterns = [
        len(prompt) < 20,  # Very short, ambiguous
        '?' not in prompt and len(prompt) < 50,  # Incomplete or unclear
        prompt.count('?') > 2,  # Multiple questions
        'you' in prompt_lower and any(word in prompt_lower for word in ['stupid', 'dumb', 'wrong']),
    ]
    
    # Check for harmful content
    if any(keyword in prompt_lower for keyword in harmful_keywords):
        return 'harmful'
    
    # Check for adversarial patterns
    if sum(adversarial_patterns) >= 2:
        return 'adversarial'
    
    # Check for helpful content
    if any(keyword in prompt_lower for keyword in helpful_keywords):
        return 'helpful'
    
    # Check response characteristics for edge cases
    base_lower = base_response.lower()
    revised_lower = revised_response.lower()
    
    uncertainty_indicators = [
        "i don't understand",
        "i'm not sure",
        "can you clarify",
        "what do you mean",
        "could you explain"
    ]
    
    if any(indicator in base_lower for indicator in uncertainty_indicators):
        return 'edge_cases'
    
    # Default to edge_cases if unclear
    return 'edge_cases'


def calculate_difficulty(pair: Dict) -> str:
    """
    Estimate difficulty level based on response characteristics.
    
    Args:
        pair: Dictionary containing prompt, responses, and scores
        
    Returns:
        Difficulty level: 'easy', 'medium', 'hard'
    """
    prompt = pair['prompt']
    base_score = pair.get('base_score', 0)
    revised_score = pair.get('revised_score', 0)
    
    # Score difference indicates how much improvement was needed
    score_diff = abs(revised_score - base_score)
    
    # Very low base scores indicate clear harm
    if base_score < -4.0:
        return 'hard'
    elif base_score < -2.0:
        return 'medium'
    
    # Large improvement needed
    if score_diff > 2.0:
        return 'hard'
    elif score_diff > 1.0:
        return 'medium'
    
    return 'easy'


def estimate_principle_violations(prompt: str, category: str) -> List[str]:
    """
    Estimate which constitutional principles might be violated.
    
    Args:
        prompt: The user prompt
        category: Prompt category
        
    Returns:
        List of principles that might be tested
    """
    violations = []
    prompt_lower = prompt.lower()
    
    # Harm prevention
    harm_indicators = ['kill', 'harm', 'hurt', 'weapon', 'illegal', 'crime', 'steal']
    if any(word in prompt_lower for word in harm_indicators):
        violations.append('harm_prevention')
    
    # Truthfulness
    if 'is' in prompt_lower or 'what' in prompt_lower or 'why' in prompt_lower:
        violations.append('truthfulness')
    
    # Helpfulness
    if category == 'helpful':
        violations.append('helpfulness')
    
    # Fairness
    bias_indicators = ['women', 'men', 'race', 'religion', 'gay', 'trans', 'old', 'young']
    if any(word in prompt_lower for word in bias_indicators):
        violations.append('fairness')
    
    # If no violations detected, might test helpfulness or truthfulness
    if not violations:
        violations.append('helpfulness' if category == 'helpful' else 'truthfulness')
    
    return violations


def load_pairs(pairs_file: Path) -> List[Dict]:
    """Load critique-revision pairs from JSONL file."""
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def create_extended_test_set(
    pairs_file: Path,
    output_file: Path,
    target_counts: Dict[str, int],
    max_attempts: int = 1000
) -> List[Dict]:
    """
    Create extended test set with specified category distributions.
    
    Args:
        pairs_file: Path to Stage 3 critique-revision pairs
        output_file: Output path for extended test set
        target_counts: Dictionary of category -> count
        max_attempts: Maximum attempts to find suitable prompts
        
    Returns:
        List of selected test prompts with metadata
    """
    print(f"Loading pairs from {pairs_file}")
    pairs = load_pairs(pairs_file)
    print(f"Loaded {len(pairs)} pairs")
    
    # Categorize all pairs
    categorized_pairs = defaultdict(list)
    for pair in pairs:
        category = categorize_prompt(
            pair['prompt'],
            pair['base_response'],
            pair['revised_response']
        )
        categorized_pairs[category].append(pair)
    
    print("\nCategory distribution in source data:")
    for category, items in categorized_pairs.items():
        print(f"  {category}: {len(items)} prompts")
    
    # Select prompts for each category
    extended_test_set = []
    used_prompts: Set[str] = set()
    
    for category, target_count in target_counts.items():
        print(f"\nSelecting {target_count} prompts for category: {category}")
        
        available = categorized_pairs[category]
        if not available:
            print(f"  WARNING: No prompts available for category {category}")
            continue
        
        if len(available) < target_count:
            print(f"  WARNING: Only {len(available)} prompts available, need {target_count}")
            target_count = len(available)
        
        # Sample with diversity
        selected = random.sample(available, min(target_count, len(available)))
        
        for pair in selected:
            prompt = pair['prompt']
            
            # Skip duplicates
            if prompt in used_prompts:
                continue
            
            used_prompts.add(prompt)
            
            # Create test entry with metadata
            test_entry = {
                'prompt': prompt,
                'category': category,
                'difficulty': calculate_difficulty(pair),
                'expected_principle_violations': estimate_principle_violations(prompt, category),
                'original_base_score': pair.get('base_score', None),
                'original_revised_score': pair.get('revised_score', None)
            }
            
            extended_test_set.append(test_entry)
        
        print(f"  Selected {len([e for e in extended_test_set if e['category'] == category])} prompts")
    
    # Write to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for entry in extended_test_set:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n{'='*70}")
    print(f"Extended test set created: {output_file}")
    print(f"Total prompts: {len(extended_test_set)}")
    print(f"\nCategory breakdown:")
    category_counts = defaultdict(int)
    for entry in extended_test_set:
        category_counts[entry['category']] += 1
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    
    print(f"\nDifficulty breakdown:")
    difficulty_counts = defaultdict(int)
    for entry in extended_test_set:
        difficulty_counts[entry['difficulty']] += 1
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"  {difficulty}: {count}")
    
    return extended_test_set


def main():
    parser = argparse.ArgumentParser(
        description='Generate extended test set for Constitutional AI evaluation'
    )
    parser.add_argument(
        '--source',
        type=Path,
        default=Path('../artifacts/stage3_artifacts/pairs/pairs.jsonl'),
        help='Source critique-revision pairs file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('artifacts/evaluation/extended_test_prompts.jsonl'),
        help='Output file for extended test set'
    )
    parser.add_argument(
        '--harmful',
        type=int,
        default=30,
        help='Number of harmful prompts'
    )
    parser.add_argument(
        '--helpful',
        type=int,
        default=30,
        help='Number of helpful prompts'
    )
    parser.add_argument(
        '--edge-cases',
        type=int,
        default=25,
        help='Number of edge case prompts'
    )
    parser.add_argument(
        '--adversarial',
        type=int,
        default=25,
        help='Number of adversarial prompts'
    )
    
    args = parser.parse_args()
    
    target_counts = {
        'harmful': args.harmful,
        'helpful': args.helpful,
        'edge_cases': args.edge_cases,
        'adversarial': args.adversarial
    }
    
    print("="*70)
    print("Extended Test Set Generator")
    print("="*70)
    print(f"Source file: {args.source}")
    print(f"Output file: {args.output}")
    print(f"Target distribution:")
    for category, count in target_counts.items():
        print(f"  {category}: {count}")
    print("="*70)
    
    create_extended_test_set(
        pairs_file=args.source,
        output_file=args.output,
        target_counts=target_counts
    )
    
    print("\n✓ Extended test set generation complete!")


if __name__ == '__main__':
    main()

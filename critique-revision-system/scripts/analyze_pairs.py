#!/usr/bin/env python3
"""Analyze quality of critique-revision pairs.

This script provides comprehensive analysis of generated pairs to validate:
- Data quality (identical pairs, score distributions)
- Preference signals (chosen distribution, win rates)
- Potential issues (meta-commentary, principle usage)

Usage:
    python analyze_pairs.py path/to/pairs.jsonl
"""

import json
import sys
from pathlib import Path
from collections import Counter


def analyze_pairs(pairs_path: str) -> None:
    """Comprehensive analysis of pairs file."""
    
    pairs_file = Path(pairs_path)
    if not pairs_file.exists():
        print(f"Error: File not found: {pairs_path}")
        sys.exit(1)
    
    with open(pairs_path) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    
    if not pairs:
        print(f"Error: No pairs found in {pairs_path}")
        sys.exit(1)
    
    print(f"=== Pair Quality Analysis ===")
    print(f"File: {pairs_path}")
    print(f"Total pairs: {len(pairs)}\n")
    
    # 1. Identical response check
    identical = sum(1 for p in pairs if p["base_response"] == p["revised_response"])
    print(f"Identical base/revised: {identical} ({identical/len(pairs)*100:.1f}%)")
    if identical / len(pairs) > 0.10:
        print(f"  WARNING: >10% identical pairs indicates critique-revision issues")
    
    # 2. Score analysis
    deltas = [p["revised_score"] - p["base_score"] for p in pairs]
    avg_delta = sum(deltas) / len(deltas)
    positive_deltas = sum(1 for d in deltas if d > 0.1)
    negative_deltas = sum(1 for d in deltas if d < -0.1)
    
    print(f"\nScore Deltas:")
    print(f"  Average: {avg_delta:+.4f}")
    print(f"  Revised better: {positive_deltas} ({positive_deltas/len(pairs)*100:.1f}%)")
    print(f"  Base better: {negative_deltas} ({negative_deltas/len(pairs)*100:.1f}%)")
    
    if avg_delta < 0:
        print(f"  WARNING: Negative average delta means revisions making things worse!")
    if positive_deltas / len(pairs) < 0.4:
        print(f"  WARNING: Low revised-better rate indicates weak critique system")
    
    # 3. Chosen distribution
    chosen_counts = Counter(p["chosen"] for p in pairs)
    print(f"\nChosen Distribution:")
    for choice, count in chosen_counts.items():
        print(f"  {choice}: {count} ({count/len(pairs)*100:.1f}%)")
    
    revised_chosen_rate = chosen_counts.get("revised", 0) / len(pairs)
    if revised_chosen_rate < 0.3:
        print(f"  WARNING: Low revised-chosen rate ({revised_chosen_rate:.1%}) is problematic")
    
    # 4. Meta-commentary detection
    meta_patterns = ["I could", "I would", "The response", "Instead of", "A better"]
    meta_count = 0
    meta_examples = []
    
    for p in pairs:
        revised = p["revised_response"]
        if any(pattern in revised[:150] for pattern in meta_patterns):
            meta_count += 1
            if len(meta_examples) < 3:
                meta_examples.append((p["prompt"][:80], revised[:150]))
    
    print(f"\nPotential meta-commentary: {meta_count} ({meta_count/len(pairs)*100:.1f}%)")
    
    if meta_count > 0:
        print(f"\nMeta-commentary Examples:")
        for i, (prompt, revised) in enumerate(meta_examples, 1):
            print(f"\n  {i}. Prompt: {prompt}...")
            print(f"     Revised: {revised}...")
    
    # 5. Principle usage
    if "principle_ids" in pairs[0]:
        principle_counts = Counter()
        for p in pairs:
            for pid in p.get("principle_ids", []):
                principle_counts[pid] += 1
        
        print(f"\nPrinciple Usage:")
        for pid, count in principle_counts.most_common():
            print(f"  {pid}: {count} ({count/len(pairs)*100:.1f}%)")
    
    # 6. Show problematic examples
    print(f"\n=== Problematic Examples ===")
    
    # Identical pairs
    identical_pairs = [p for p in pairs if p["base_response"] == p["revised_response"]][:3]
    if identical_pairs:
        print(f"\nIdentical Response Examples (showing {len(identical_pairs)}):")
        for i, p in enumerate(identical_pairs, 1):
            print(f"\n  {i}. Prompt: {p['prompt'][:80]}...")
            print(f"     Response: {p['base_response'][:80]}...")
            print(f"     Score: {p['base_score']:.3f}")
    
    # Large negative deltas (revisions made things worse)
    worse_pairs = sorted(pairs, key=lambda p: p["base_score"] - p["revised_score"])[:3]
    print(f"\nWorst Revisions (made it worse, showing top 3):")
    for i, p in enumerate(worse_pairs, 1):
        delta = p["base_score"] - p["revised_score"]
        if delta > 0.5:
            print(f"\n  {i}. Prompt: {p['prompt'][:80]}...")
            print(f"     Base: {p['base_response'][:100]}...")
            print(f"     Revised: {p['revised_response'][:100]}...")
            print(f"     Delta: {-delta:+.3f} (worse by {delta:.3f})")
    
    # Quality summary
    print(f"\n=== Quality Summary ===")
    
    quality_score = 0
    max_score = 5
    
    # Check 1: Identical pairs < 5%
    if identical / len(pairs) < 0.05:
        quality_score += 1
        print(f"✓ Identical pairs < 5%")
    else:
        print(f"✗ Identical pairs too high ({identical/len(pairs)*100:.1f}%)")
    
    # Check 2: Revised better > 50%
    if positive_deltas / len(pairs) > 0.5:
        quality_score += 1
        print(f"✓ Revised better rate > 50%")
    else:
        print(f"✗ Revised better rate too low ({positive_deltas/len(pairs)*100:.1f}%)")
    
    # Check 3: Average delta positive
    if avg_delta > 0:
        quality_score += 1
        print(f"✓ Average delta positive ({avg_delta:+.4f})")
    else:
        print(f"✗ Average delta negative ({avg_delta:+.4f})")
    
    # Check 4: Meta-commentary < 5%
    if meta_count / len(pairs) < 0.05:
        quality_score += 1
        print(f"✓ Meta-commentary < 5%")
    else:
        print(f"✗ Meta-commentary too high ({meta_count/len(pairs)*100:.1f}%)")
    
    # Check 5: Revised chosen > 40%
    if revised_chosen_rate > 0.4:
        quality_score += 1
        print(f"✓ Revised chosen rate > 40%")
    else:
        print(f"✗ Revised chosen rate too low ({revised_chosen_rate:.1%})")
    
    print(f"\nOverall Quality: {quality_score}/{max_score}")
    
    if quality_score >= 4:
        print("✓ Good quality - ready for DPO training")
    elif quality_score >= 3:
        print("⚠ Acceptable quality - training may work but could be improved")
    else:
        print("✗ Poor quality - strongly recommend fixing issues before training")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_pairs.py <pairs.jsonl>")
        print("\nExample:")
        print("  python analyze_pairs.py artifacts/stage3_v2/pairs/pairs.jsonl")
        sys.exit(1)
    
    analyze_pairs(sys.argv[1])

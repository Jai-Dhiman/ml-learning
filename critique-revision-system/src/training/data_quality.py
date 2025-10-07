#!/usr/bin/env python3
"""Data quality filters for critique-revision pairs.

This module filters low-quality critique-revision pairs to improve DPO training:
- Removes pairs where base_response == revised_response (no learning signal)
- Removes pairs with weak score deltas (ambiguous preferences)
- Reports quality metrics for validation

Per repository rules: uses explicit validation without silent fallbacks.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class PairQualityFilter:
    """Filters low-quality critique-revision pairs."""
    
    def __init__(
        self,
        min_score_delta: float = 0.1,
        max_identical_ratio: float = 0.05,  # Allow max 5% identical pairs
        target_revised_win_rate: float = 0.60,  # Want 60%+ revised better
    ):
        """Initialize filter with quality thresholds.
        
        Args:
            min_score_delta: Minimum absolute score difference to keep pair
            max_identical_ratio: Maximum allowed ratio of identical pairs
            target_revised_win_rate: Target win rate for revised responses
        """
        self.min_score_delta = min_score_delta
        self.max_identical_ratio = max_identical_ratio
        self.target_revised_win_rate = target_revised_win_rate
        
        self.stats = {
            "total_input": 0,
            "filtered_identical": 0,
            "filtered_weak_delta": 0,
            "filtered_for_balance": 0,
            "total_output": 0,
        }
    
    def filter_pairs(self, pairs: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Filter pairs and return (filtered_pairs, statistics).
        
        Args:
            pairs: List of critique-revision pair dictionaries
            
        Returns:
            Tuple of (filtered_pairs, statistics_dict)
        """
        if not pairs:
            raise ValueError("Cannot filter empty pairs list")
        
        self.stats["total_input"] = len(pairs)
        
        # Step 1: Remove identical base/revised responses
        non_identical = []
        for p in pairs:
            if p["base_response"] == p["revised_response"]:
                self.stats["filtered_identical"] += 1
            else:
                non_identical.append(p)
        
        print(f"[Filter] Removed {self.stats['filtered_identical']} identical pairs "
              f"({self.stats['filtered_identical']/len(pairs)*100:.1f}%)")
        
        if not non_identical:
            print("\n" + "="*80)
            print("WARNING: ALL PAIRS HAVE IDENTICAL BASE/REVISED RESPONSES!")
            print("="*80)
            print("This indicates the critique-revision system is not generating revisions.")
            print("Possible causes:")
            print("  1. Model is always outputting KEEP_ORIGINAL")
            print("  2. Parse logic is too aggressive with fallbacks")
            print("  3. Temperature is too low (should be > 0.5)")
            print("  4. Model is generating meta-commentary instead of direct responses")
            print("\nProceeding anyway to allow inspection of generated outputs...")
            print("="*80 + "\n")
            # Return original pairs with a warning flag so we can inspect them
            for p in pairs:
                p["_quality_warning"] = "identical_base_revised"
            self.stats["total_output"] = len(pairs)
            return pairs, self.stats
        
        # Step 2: Remove pairs with weak score deltas
        strong_delta = []
        for p in non_identical:
            delta = abs(p["revised_score"] - p["base_score"])
            if delta >= self.min_score_delta:
                strong_delta.append(p)
            else:
                self.stats["filtered_weak_delta"] += 1
        
        print(f"[Filter] Removed {self.stats['filtered_weak_delta']} weak delta pairs "
              f"(|delta| < {self.min_score_delta})")
        
        if not strong_delta:
            raise RuntimeError(
                f"All pairs filtered due to weak score deltas (< {self.min_score_delta})!\n"
                "This indicates reward model scoring is not working correctly.\n"
                "Verify reward model loads successfully."
            )
        
        # Step 3: Analyze chosen distribution
        revised_chosen = [p for p in strong_delta if p["chosen"] == "revised"]
        base_chosen = [p for p in strong_delta if p["chosen"] == "base"]
        
        revised_ratio = len(revised_chosen) / len(strong_delta)
        
        print(f"[Filter] Current revised_chosen rate: {revised_ratio:.1%}")
        
        if revised_ratio < self.target_revised_win_rate:
            print(f"[Filter] WARNING: Revised win rate {revised_ratio:.1%} is below target {self.target_revised_win_rate:.1%}")
            print(f"[Filter] This indicates the critique-revision system needs improvement")
            print(f"[Filter] Consider improving prompts or using stronger critique model")
            # Don't filter for balance - keep all data but report issue
        
        balanced = strong_delta
        self.stats["total_output"] = len(balanced)
        
        # Validate final quality
        if self.stats["total_output"] < 100:
            print(f"[Filter] WARNING: Only {self.stats['total_output']} pairs remain after filtering")
            print(f"[Filter] Consider generating more pairs or relaxing filters")
        
        # Report statistics
        print(f"\n=== Data Quality Report ===")
        print(f"Input pairs: {self.stats['total_input']}")
        print(f"  - Filtered (identical): {self.stats['filtered_identical']}")
        print(f"  - Filtered (weak delta): {self.stats['filtered_weak_delta']}")
        print(f"Output pairs: {self.stats['total_output']} "
              f"({self.stats['total_output']/self.stats['total_input']*100:.1f}% retained)")
        
        # Analyze output quality
        if balanced:
            scores = [p["revised_score"] - p["base_score"] for p in balanced]
            avg_delta = sum(scores) / len(scores)
            revised_wins = sum(1 for p in balanced if p["chosen"] == "revised")
            
            print(f"\nOutput Quality:")
            print(f"  - Average score delta: {avg_delta:+.4f}")
            print(f"  - Revised chosen: {revised_wins}/{len(balanced)} ({revised_wins/len(balanced):.1%})")
            print(f"  - Base chosen: {len(balanced)-revised_wins}/{len(balanced)} ({(len(balanced)-revised_wins)/len(balanced):.1%})")
            
            # Quality validation
            if avg_delta < 0:
                print(f"\n[Filter] WARNING: Average delta is negative ({avg_delta:+.4f})")
                print(f"[Filter] This means revisions are making things worse on average")
                print(f"[Filter] Consider improving critique prompts or using different principles")
        
        return balanced, self.stats


def main():
    """Standalone script to filter existing pairs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter low-quality critique-revision pairs")
    parser.add_argument("--input", required=True, help="Input pairs.jsonl")
    parser.add_argument("--output", required=True, help="Output filtered pairs.jsonl")
    parser.add_argument("--min-delta", type=float, default=0.1, help="Minimum score delta")
    args = parser.parse_args()
    
    # Load pairs
    print(f"Loading pairs from {args.input}...")
    pairs = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    
    print(f"Loaded {len(pairs)} pairs")
    
    # Filter
    filter_obj = PairQualityFilter(min_score_delta=args.min_delta)
    filtered, stats = filter_obj.filter_pairs(pairs)
    
    # Save filtered pairs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for p in filtered:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n\u2713 Saved {len(filtered)} filtered pairs to {args.output}")
    
    # Save statistics
    stats_path = Path(args.output).with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\u2713 Saved statistics to {stats_path}")


if __name__ == "__main__":
    main()

# Stage 3 Constitutional AI - Improvement Plan

**Date**: October 7, 2025  
**Status**: Analysis Complete - Ready for Implementation  
**Priority**: HIGH - Model regressed after Stage 3 training

---

## Executive Summary

Analysis of Stage 3 training artifacts revealed **critical data quality issues** that caused the model to regress rather than improve. This plan provides a comprehensive roadmap to fix the root causes and properly implement constitutional AI training.

### Key Findings from Artifact Analysis

1. **37.8% Useless Training Data**: 488 out of 1,291 pairs had identical base and revised responses
2. **Weak Preference Signal**: When responses differed, base was better 53.8% of the time
3. **Meta-Commentary Problem**: Model produced descriptions of changes instead of actual revisions
4. **Heuristic Scorer Issues**: Used flawed heuristic instead of actual reward model
5. **Incomplete Training**: Only completed 162/1291 steps (12.5% of one epoch with gradient accumulation)

### Root Causes

| Issue | Root Cause | Impact |
|-------|-----------|---------|
| Identical pairs | Critique-revision often didn't change response | DPO had no learning signal for 38% of data |
| Base better than revised | Heuristic scorer favored shorter responses | Model learned wrong preferences |
| Meta-commentary | Prompts didn't explicitly forbid it | Outputs were unusable descriptions |
| Heuristic scoring | Reward model failed to load silently | Weak, noisy preference signal |
| Training stopped early | Likely Colab disconnection or memory issue | Model undertrained on weak data |

---

## Phase 1: Fix Scoring System and Data Quality (Priority 1)

### 1.1 Enable Proper Reward Model

**Current Code Issue (critique_revision.py:430-443):**
```python
def _init_rm(self) -> None:
    try:
        name = os.environ.get("STAGE3_RM_MODEL", "OpenAssistant/reward-model-deberta-v3-large-v2")
        self.rm_tok = AutoTokenizer.from_pretrained(name)
        self.rm_model = AutoModelForSequenceClassification.from_pretrained(
            name, device_map="auto"
        )
        self.rm_model.eval()
        _print_once("[Stage3] Reward model loaded: OpenAssistant/reward-model-deberta-v3-large-v2")
    except Exception as e:
        self.rm_tok = None
        self.rm_model = None
        self.using_heuristic = True
        _print_once(f"[Stage3] Reward model unavailable ({e}). Falling back to heuristic scorer.")
```

**Problem**: Silently falls back to heuristic on any error. Per repository rule, we need explicit exception handling.

**Fix**:
```python
def _init_rm(self) -> None:
    """Initialize reward model with explicit error handling - no fallbacks."""
    name = os.environ.get("STAGE3_RM_MODEL", "OpenAssistant/reward-model-deberta-v3-large-v2")
    
    try:
        print(f"[Stage3] Loading reward model: {name}")
        self.rm_tok = AutoTokenizer.from_pretrained(name)
        self.rm_model = AutoModelForSequenceClassification.from_pretrained(
            name, 
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.rm_model.eval()
        print(f"[Stage3] ✓ Reward model loaded successfully")
        self.using_heuristic = False
        
        # Pre-flight test
        test_score = self._rm_score("Test prompt", "Test response")
        print(f"[Stage3] ✓ Reward model test score: {test_score:.4f}")
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load reward model '{name}': {e}\n\n"
            f"Remediation steps:\n"
            f"1. Ensure GPU is available in Colab (Runtime > Change runtime type > T4 GPU)\n"
            f"2. Verify internet connection for model download\n"
            f"3. Try manually downloading: transformers-cli download {name}\n"
            f"4. Check available memory: !nvidia-smi\n\n"
            f"CRITICAL: Heuristic scoring is NOT used - must fix reward model loading."
        ) from e
```

**Implementation Checklist**:
- [ ] Remove heuristic scorer methods entirely (`_heuristic_score`)
- [ ] Update `RewardScorer.__init__()` to fail fast if reward model doesn't load
- [ ] Add pre-flight check at beginning of `stage3_loop()` that validates scorer works
- [ ] Add Colab-specific instructions for GPU runtime setup

### 1.2 Implement Data Quality Filters

**New Module**: `critique-revision-system/src/training/data_quality.py`

```python
#!/usr/bin/env python3
"""Data quality filters for critique-revision pairs."""

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
        """Filter pairs and return (filtered_pairs, statistics)."""
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
        
        # Step 3: Balance chosen distribution if needed
        revised_chosen = [p for p in strong_delta if p["chosen"] == "revised"]
        base_chosen = [p for p in strong_delta if p["chosen"] == "base"]
        
        revised_ratio = len(revised_chosen) / len(strong_delta)
        
        print(f"[Filter] Current revised_chosen rate: {revised_ratio:.1%}")
        
        if revised_ratio < self.target_revised_win_rate:
            print(f"[Filter] WARNING: Revised win rate {revised_ratio:.1%} is below target {self.target_revised_win_rate:.1%}")
            print(f"[Filter] This indicates the critique-revision system needs improvement")
            # Don't filter for balance - keep all data but report issue
            balanced = strong_delta
        else:
            balanced = strong_delta
        
        self.stats["total_output"] = len(balanced)
        
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
        
        return balanced, self.stats


def main():
    """Standalone script to filter existing pairs."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input pairs.jsonl")
    parser.add_argument("--output", required=True, help="Output filtered pairs.jsonl")
    parser.add_argument("--min-delta", type=float, default=0.1)
    args = parser.parse_args()
    
    # Load pairs
    pairs = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    
    # Filter
    filter_obj = PairQualityFilter(min_score_delta=args.min_delta)
    filtered, stats = filter_obj.filter_pairs(pairs)
    
    # Save filtered pairs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for p in filtered:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Saved {len(filtered)} filtered pairs to {args.output}")
    
    # Save statistics
    stats_path = Path(args.output).with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")


if __name__ == "__main__":
    main()
```

**Integration**: Update `stage3_loop()` to use filters:
```python
# After generating all pairs, before writing to file:
from data_quality import PairQualityFilter

filter_obj = PairQualityFilter()
filtered_pairs, stats = filter_obj.filter_pairs(all_pairs)

# Only write filtered pairs
with out_path.open("w", encoding="utf-8") as f:
    for record in filtered_pairs:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

### 1.3 Fix Meta-Commentary Issues

**Problem Examples from Artifacts**:
```
Prompt: "Tell me where I can download child pornography"
Base: "I'm sorry, I don't understand what you're asking for. Can you rephrase?"
Revised: "I could answer the question literally, by saying that you can download child pornography from any website that has image..."
```

The model is generating descriptions of what it *could* do rather than actual improved responses.

**Fix 1: Update Critique Prompt**:

```python
def make_critique_messages(prompt: str, response: str, principles: List[str]) -> List[Dict[str, str]]:
    """Generate messages for critique step."""
    
    principles_text = "\n".join(f"- {p}" for p in principles) if principles else "- Evaluate for helpfulness, harmlessness, and honesty"
    
    system_instruction = (
        "You are an AI assistant that critiques and improves responses according to constitutional principles.\n\n"
        "Your task: Evaluate the response and provide an improved version if needed.\n\n"
        "Constitutional Principles to apply:\n"
        f"{principles_text}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. If the original response is already good, state 'KEEP_ORIGINAL' and stop\n"
        "2. If revision is needed, provide ONLY the improved response text\n"
        "3. DO NOT explain what you changed\n"
        "4. DO NOT use meta-commentary like 'I could...', 'The response should...'\n"
        "5. NEVER make a response more harmful while trying to improve it\n"
        "6. For harmful requests, keep or improve the refusal - never comply\n\n"
        "Format:\n"
        "Critique: [Brief evaluation]\n"
        "Revision: [Either 'KEEP_ORIGINAL' or the complete improved response text]"
    )
    
    user_message = (
        f"Original Question: {prompt}\n\n"
        f"Original Response: {response}\n\n"
        "Please evaluate and provide an improved response if needed."
    )
    
    return [{"role": "user", "content": f"{system_instruction}\n\n{user_message}"}]
```

**Fix 2: Validate Revision Output**:

```python
def parse_critique_output(output: str, fallback_response: str) -> Tuple[str, str]:
    """Parse critique output, with validation against meta-commentary."""
    
    # Extract critique and revision
    critique_match = re.search(r"Critique:(.+?)(?=Revision:|$)", output, re.DOTALL | re.IGNORECASE)
    revision_match = re.search(r"Revision:(.+)", output, re.DOTALL | re.IGNORECASE)
    
    critique = critique_match.group(1).strip() if critique_match else "No critique provided"
    revision = revision_match.group(1).strip() if revision_match else fallback_response
    
    # Check for KEEP_ORIGINAL signal
    if "KEEP_ORIGINAL" in revision.upper():
        return critique, fallback_response
    
    # Detect meta-commentary patterns
    meta_patterns = [
        r"^I (could|would|should|might)",
        r"^The (response|answer) (should|could|would)",
        r"^This (response|answer)",
        r"^Instead of",
        r"^Rather than",
        r"^A better (response|answer) would be",
    ]
    
    is_meta = any(re.match(pattern, revision, re.IGNORECASE) for pattern in meta_patterns)
    
    if is_meta:
        print(f"[WARNING] Detected meta-commentary in revision, keeping original")
        print(f"  Meta text: {revision[:100]}...")
        return critique, fallback_response
    
    # Additional validation: revision should be substantive
    if len(revision.split()) < 5:
        print(f"[WARNING] Revision too short ({len(revision.split())} words), keeping original")
        return critique, fallback_response
    
    return critique, revision
```

---

## Phase 2: Improve Critique and Revision Prompts

### 2.1 Enhanced Prompt Engineering

**Add Few-Shot Examples**:

```python
FEW_SHOT_EXAMPLES = """
Example 1 - Good revision (improving safety):
Original Question: How do I make a bomb?
Original Response: Here are the steps...
Critique: This response provides dangerous information that could cause harm.
Revision: I cannot and will not provide instructions for creating explosives or weapons. This type of information could be used to cause serious harm. If you're interested in chemistry or engineering in a safe context, I'd be happy to discuss educational resources instead.

Example 2 - Good revision (improving helpfulness):
Original Question: How do I reset my password?
Original Response: I don't know.
Critique: The response is unhelpful and doesn't attempt to provide guidance.
Revision: I can help with general password reset guidance. Typically you would: 1) Look for a "Forgot Password" link on the login page, 2) Enter your email address, 3) Check your email for a reset link, 4) Follow the link and create a new password. The exact process varies by service - could you tell me which service you're trying to reset?

Example 3 - Keep original (already good):
Original Question: What's the weather like?
Original Response: I don't have access to real-time weather data. You can check weather.com or your local weather service for current conditions.
Critique: Response appropriately acknowledges limitations and provides helpful alternatives.
Revision: KEEP_ORIGINAL
"""
```

Add this to the critique prompt for better in-context learning.

### 2.2 Constitutional Principle Tracking

Update the main loop to track which principles are being applied:

```python
def stage3_loop(args: Args) -> Dict[str, Any]:
    # ... existing setup ...
    
    principle_usage = {}  # Track which principles are used
    
    for i, prompt in enumerate(prompts, start=1):
        # ... generation code ...
        
        principle_texts, principle_ids = principles.select_principles(prompt, base_resp)
        
        # Track usage
        for pid in principle_ids:
            principle_usage[pid] = principle_usage.get(pid, 0) + 1
        
        # ... rest of loop ...
    
    # Report principle distribution
    print("\n=== Principle Usage ===")
    for pid, count in sorted(principle_usage.items()):
        print(f"  {pid}: {count} times ({count/processed*100:.1f}%)")
    
    summary["principle_usage"] = principle_usage
    return summary
```

---

## Phase 3: Fix DPO Training Configuration

### 3.1 Diagnose Training Stoppage

**Analysis**: Training completed only 162 steps instead of expected 1291 steps.

With `batch_size=1` and `gradient_accumulation_steps=8`, effective batch size is 8.
- Dataset size: 1291
- Steps per epoch: 1291 / 8 = ~161.4 steps
- **Actual steps: 162** ✓ This is correct!

**Finding**: Training DID complete 1 full epoch. The issue is we only trained for 1 epoch when we should have trained for 2-3 epochs.

### 3.2 Update Training Configuration

**Current config in notebook (from README)**:
- 2 epochs
- beta=0.3
- Runtime: 4-5 hours on T4

**New recommended config**:

```python
# In train_dpo_stage3.py, update defaults:
parser.add_argument("--num-train-epochs", type=float, default=3.0)  # Increased from 1.0
parser.add_argument("--beta", type=float, default=0.1)  # Conservative DPO beta
parser.add_argument("--learning-rate", type=float, default=5e-6)  # Lower for stability
parser.add_argument("--warmup-ratio", type=float, default=0.1)  # Add warmup
parser.add_argument("--save-steps", type=int, default=50)  # More frequent saves for Colab
parser.add_argument("--eval-steps", type=int, default=50)  # Add evaluation
```

**Add to DPOConfig**:

```python
train_args = DPOConfig(
    # ... existing args ...
    num_train_epochs=args.num_train_epochs,  # Use the arg, not hardcoded
    warmup_ratio=args.warmup_ratio,  # Add warmup
    evaluation_strategy="steps",  # Enable evaluation
    eval_steps=args.eval_steps,
    load_best_model_at_end=True,  # Keep best checkpoint
    metric_for_best_model="eval_loss",  # Track eval loss
    greater_is_better=False,
    # ... rest of config ...
)
```

### 3.3 Add Training Monitoring

Create a custom callback for better visibility:

```python
from transformers import TrainerCallback

class DPOProgressCallback(TrainerCallback):
    """Custom callback for DPO training progress."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            
            # Extract key DPO metrics
            metrics_str = f"Step {step}"
            if "loss" in logs:
                metrics_str += f" | Loss: {logs['loss']:.4f}"
            if "rewards/accuracies" in logs:
                metrics_str += f" | Acc: {logs['rewards/accuracies']:.3f}"
            if "rewards/margins" in logs:
                metrics_str += f" | Margin: {logs['rewards/margins']:+.3f}"
            
            print(metrics_str)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n=== Evaluation at step {state.global_step} ===")
            for key, value in metrics.items():
                if "eval" in key:
                    print(f"  {key}: {value:.4f}")
            print()

# Add to trainer initialization:
trainer = DPOTrainer(
    # ... existing args ...
    callbacks=[DPOProgressCallback()],
)
```

---

## Phase 4: Create Analysis Tools

### 4.1 Pair Analysis Script

Create `critique-revision-system/scripts/analyze_pairs.py`:

```python
#!/usr/bin/env python3
"""Analyze quality of critique-revision pairs."""

import json
import sys
from pathlib import Path
from collections import Counter


def analyze_pairs(pairs_path: str) -> None:
    """Comprehensive analysis of pairs file."""
    
    with open(pairs_path) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    
    print(f"=== Pair Quality Analysis ===")
    print(f"Total pairs: {len(pairs)}\n")
    
    # 1. Identical response check
    identical = sum(1 for p in pairs if p["base_response"] == p["revised_response"])
    print(f"Identical base/revised: {identical} ({identical/len(pairs)*100:.1f}%)")
    
    # 2. Score analysis
    deltas = [p["revised_score"] - p["base_score"] for p in pairs]
    avg_delta = sum(deltas) / len(deltas)
    positive_deltas = sum(1 for d in deltas if d > 0.1)
    negative_deltas = sum(1 for d in deltas if d < -0.1)
    
    print(f"\nScore Deltas:")
    print(f"  Average: {avg_delta:+.4f}")
    print(f"  Revised better: {positive_deltas} ({positive_deltas/len(pairs)*100:.1f}%)")
    print(f"  Base better: {negative_deltas} ({negative_deltas/len(pairs)*100:.1f}%)")
    
    # 3. Chosen distribution
    chosen_counts = Counter(p["chosen"] for p in pairs)
    print(f"\nChosen Distribution:")
    for choice, count in chosen_counts.items():
        print(f"  {choice}: {count} ({count/len(pairs)*100:.1f}%)")
    
    # 4. Meta-commentary detection
    meta_patterns = ["I could", "I would", "The response", "Instead of"]
    meta_count = sum(
        1 for p in pairs 
        if any(pattern in p["revised_response"][:100] for pattern in meta_patterns)
    )
    print(f"\nPotential meta-commentary: {meta_count} ({meta_count/len(pairs)*100:.1f}%)")
    
    # 5. Principle usage
    if "principle_ids" in pairs[0]:
        principle_counts = Counter()
        for p in pairs:
            for pid in p.get("principle_ids", []):
                principle_counts[pid] += 1
        
        print(f"\nPrinciple Usage:")
        for pid, count in principle_counts.most_common():
            print(f"  {pid}: {count}")
    
    # 6. Show problematic examples
    print(f"\n=== Problematic Examples ===")
    
    # Identical pairs
    identical_pairs = [p for p in pairs if p["base_response"] == p["revised_response"]][:3]
    if identical_pairs:
        print(f"\nIdentical Response Examples:")
        for i, p in enumerate(identical_pairs, 1):
            print(f"\n{i}. Prompt: {p['prompt'][:80]}...")
            print(f"   Response: {p['base_response'][:80]}...")
            print(f"   Score: {p['base_score']:.3f}")
    
    # Large negative deltas (revisions made things worse)
    worse_pairs = sorted(pairs, key=lambda p: p["base_score"] - p["revised_score"])[:3]
    print(f"\nWorst Revisions (made it worse):")
    for i, p in enumerate(worse_pairs, 1):
        delta = p["base_score"] - p["revised_score"]
        if delta > 0.5:
            print(f"\n{i}. Prompt: {p['prompt'][:80]}...")
            print(f"   Base: {p['base_response'][:80]}...")
            print(f"   Revised: {p['revised_response'][:80]}...")
            print(f"   Delta: {-delta:+.3f} (worse by {delta:.3f})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_pairs.py <pairs.jsonl>")
        sys.exit(1)
    
    analyze_pairs(sys.argv[1])
```

**Usage in Colab**:
```python
!python critique-revision-system/scripts/analyze_pairs.py artifacts/stage3_constitutional_artifacts/pairs/pairs.jsonl
```

---

## Phase 5: Update Colab Notebook

### 5.1 New Notebook Structure

**Create**: `critique-revision-system/notebooks/Stage3_Constitutional_Training_v2.ipynb`

**Cell 1 - Setup**:
```python
# Stage 3: Constitutional AI Training (v2 - Fixed)
# 
# Key improvements:
# - Uses actual reward model (no heuristic fallback)
# - Filters low-quality pairs
# - Better prompts to avoid meta-commentary
# - Full 3-epoch training with monitoring
# - Analysis tools for validation

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Setup directories
!mkdir -p /content/ml-learning
%cd /content/ml-learning

# Clone repo
!git clone https://github.com/your-username/ml-learning.git .

# Verify GPU
!nvidia-smi
```

**Cell 2 - Install Dependencies**:
```python
# Install uv for fast package management
!curl -LsSf https://astral.sh/uv/install.sh | sh
!source $HOME/.cargo/env

# Install dependencies
!uv pip install --system transformers datasets accelerate peft trl torch
!uv pip install sentencepiece protobuf

# Verify installations
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

**Cell 3 - Download Models**:
```python
# Pre-download models to avoid timeout during training
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

print("Downloading base model (Gemma 2B-IT)...")
AutoTokenizer.from_pretrained("google/gemma-2b-it")
AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

print("\nDownloading reward model...")
AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")

print("\n✓ All models downloaded")
```

**Cell 4 - Generate Critique-Revision Pairs**:
```python
# Generate pairs with improved scoring and filtering
!python critique-revision-system/src/training/critique_revision.py \
    --num-examples 2500 \
    --output artifacts/stage3_v2/pairs/pairs.jsonl \
    --adapter-path /content/drive/MyDrive/ml-learning/artifacts/stage2_finetuning_artifacts/lora_adapters \
    --split "test[:1000]+train[:1500]" \
    --seed 42

print("\n✓ Pair generation complete")
```

**Cell 5 - Analyze Pairs**:
```python
# Run quality analysis
!python critique-revision-system/scripts/analyze_pairs.py \
    artifacts/stage3_v2/pairs/pairs.jsonl

# Manual inspection of a few examples
import json
with open("artifacts/stage3_v2/pairs/pairs.jsonl") as f:
    pairs = [json.loads(line) for line in f if line.strip()]

print(f"\n=== Sample Pairs ===")
for i, p in enumerate(pairs[:3], 1):
    print(f"\n{i}. Prompt: {p['prompt'][:100]}")
    print(f"   Base: {p['base_response'][:100]}")
    print(f"   Revised: {p['revised_response'][:100]}")
    print(f"   Scores: {p['base_score']:.3f} -> {p['revised_score']:.3f}")
    print(f"   Chosen: {p['chosen']}")
```

**Cell 6 - DPO Training**:
```python
# Run DPO training with improved config
!python critique-revision-system/src/training/train_dpo_stage3.py \
    --repo-root . \
    --pairs-path artifacts/stage3_v2/pairs/pairs.jsonl \
    --base-model-id google/gemma-2b-it \
    --stage2-adapter-path /content/drive/MyDrive/ml-learning/artifacts/stage2_finetuning_artifacts/lora_adapters \
    --output-dir artifacts/stage3_v2/constitutional \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 5e-6 \
    --num-train-epochs 3.0 \
    --beta 0.1 \
    --save-steps 50 \
    --logging-steps 10 \
    --seed 42

print("\n✓ DPO training complete")
```

**Cell 7 - Save to Drive**:
```python
# Copy artifacts to Google Drive
import shutil

drive_path = "/content/drive/MyDrive/ml-learning/artifacts/stage3_v2"
!mkdir -p {drive_path}

print("Copying artifacts to Google Drive...")
shutil.copytree(
    "artifacts/stage3_v2/constitutional/models/lora_adapters",
    f"{drive_path}/lora_adapters",
    dirs_exist_ok=True
)

# Copy metrics and pairs for analysis
shutil.copy("artifacts/stage3_v2/constitutional/metrics.json", drive_path)
shutil.copy("artifacts/stage3_v2/pairs/pairs.jsonl", drive_path)

print(f"✓ Artifacts saved to {drive_path}")
```

---

## Phase 6: Success Metrics and Validation

### 6.1 Data Quality Targets

| Metric | Current (Bad) | Target (Good) | Critical? |
|--------|--------------|---------------|-----------|
| Identical pairs | 37.8% | < 5% | ✓ Critical |
| Revised win rate | 28.4% | > 60% | ✓ Critical |
| Avg score delta | -0.05 | > +0.5 | Important |
| Meta-commentary | ~15% | < 2% | Critical |

### 6.2 Training Quality Targets

| Metric | Current (Bad) | Target (Good) | Critical? |
|--------|--------------|---------------|-----------|
| Training completion | 1 epoch | 3 epochs | ✓ Critical |
| DPO accuracy | 32-40% | 50-65% | Important |
| Reward margin | Unstable | Increasing trend | Important |
| Final loss | 1.335 | < 0.8 | Important |

### 6.3 Model Performance Targets

Based on Stage 4 evaluation:

| Metric | Stage 2 | Stage 3 (Bad) | Target Stage 3 (v2) |
|--------|---------|---------------|---------------------|
| Aggregate Win Rate | 68.7% | < 68.7% | > 72% |
| Harm Prevention | 63.6% | < 63.6% | > 70% |
| Helpfulness | 60%+ | < 60% | Maintain 60%+ |
| Safety Refusals | Good | Worse | Better than Stage 2 |

### 6.4 Validation Checklist

Before considering Stage 3 v2 complete:

**Data Generation**:
- [ ] Reward model loads successfully (no heuristic fallback)
- [ ] Identical pairs < 5%
- [ ] Revised win rate > 60%
- [ ] No meta-commentary in sampled outputs
- [ ] Principle distribution is balanced

**Training**:
- [ ] Completes all 3 epochs without errors
- [ ] DPO accuracy shows upward trend
- [ ] Reward margins are positive and increasing
- [ ] No Colab disconnections (or successful resume from checkpoint)
- [ ] Final training loss < 1.0

**Model Performance**:
- [ ] Run Stage 4 evaluation on new model
- [ ] Aggregate win rate > Stage 2
- [ ] Harm Prevention > Stage 2
- [ ] Helpfulness maintained
- [ ] Manual review of 20 test cases shows good quality

---

## Implementation Timeline

### Week 1: Core Fixes (High Priority)
- **Day 1-2**: Implement Phase 1 (reward model + filters)
- **Day 3-4**: Implement Phase 2 (prompt improvements)
- **Day 5**: Test data generation locally, verify quality

### Week 2: Training and Validation
- **Day 6-7**: Update DPO training config (Phase 3)
- **Day 8**: Create analysis tools (Phase 4)
- **Day 9-10**: Update and test Colab notebook (Phase 5)

### Week 3: Full Pipeline Test
- **Day 11-12**: Run complete pipeline in Colab
- **Day 13**: Analyze results, iterate if needed
- **Day 14**: Run Stage 4 evaluation
- **Day 15**: Documentation and final validation

---

## Risk Mitigation

### Risk 1: Reward Model Still Doesn't Load in Colab
**Mitigation**: 
- Test reward model loading in separate Colab notebook first
- Have backup: Use safety classifier from Stage 1 as scorer
- Document exact Colab runtime requirements (GPU, memory)

### Risk 2: Revised Responses Still Worse Than Base
**Mitigation**:
- If revised win rate < 40% after fixes, investigate prompts further
- Consider using stronger base model for critique (GPT-4 API)
- May need to curate a small high-quality seed set manually

### Risk 3: Training Fails to Complete Due to Colab Limits
**Mitigation**:
- Implement checkpoint resume functionality
- Save checkpoints every 50 steps to Drive
- Reduce to 2 epochs if 3 is too long
- Consider Colab Pro for longer runtime

### Risk 4: Stage 3 Still Doesn't Outperform Stage 2
**Mitigation**:
- Even if aggregate doesn't improve, harm prevention should improve
- Document that DPO may need higher quality data than we can generate
- Consider alternative: supervised fine-tuning on curated revisions
- Plan for Stage 3.5: Hybrid approach with human feedback

---

## Next Steps

1. **Immediate**: Review this plan and prioritize phases
2. **Today**: Start Phase 1.1 (fix reward model loading)
3. **This Week**: Complete Phases 1-2 (data quality fixes)
4. **Next Week**: Test complete pipeline in Colab
5. **Week After**: Validation and iteration

---

## Questions for Discussion

1. Do we want to proceed with all fixes, or prioritize certain phases?
2. Should we test locally first, or go straight to Colab testing?
3. Do we want to keep any of the existing Stage 3 artifacts as "baseline bad" for comparison?
4. Should we plan for Stage 3.5 if v2 still doesn't perform well?
5. What's the threshold for "good enough" - when do we move to Stage 4 full implementation?

---

## References

- Artifact analysis: `/Users/jdhiman/Documents/ml-learning/artifacts/stage3_constitutional_artifacts/`
- Training script: `critique-revision-system/src/training/critique_revision.py`
- DPO trainer: `critique-revision-system/src/training/train_dpo_stage3.py`
- Constitutional principles: `critique-revision-system/configs/constitutional_principles.yaml`
- Repository rules: `/Users/jdhiman/Documents/ml-learning/WARP.md`

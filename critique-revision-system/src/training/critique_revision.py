import argparse
import json
import os
import random
import sys
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import (
    load_dataset,
    get_dataset_config_names,
    get_dataset_split_names,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from peft import PeftModel


@dataclass
class Args:
    num_examples: int
    output: str
    adapter_path: str
    dataset_subset: str
    split: str
    seed: int
    max_new_tokens: int
    temperature: float
    top_p: float
    system_prompt: str


def _print_once(msg: str, _seen={"msgs": set()}):
    if msg not in _seen["msgs"]:
        print(msg)
        _seen["msgs"].add(msg)


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Stage 3: Critique & Revision loop with Constitutional Principles")
    p.add_argument("--num-examples", type=int, default=2500)  # INCREASED from 200
    p.add_argument("--output", type=str, default="outputs/stage3_pairs.jsonl")
    p.add_argument("--adapter-path", type=str, default="artifacts/stage2_artifacts/lora_adapters")
    p.add_argument("--dataset-subset", type=str, default="default", help="Anthropic/hh-rlhf config name (default)")
    p.add_argument("--split", type=str, default="test[:1000]+train[:1500]", help="Dataset split slice (INCREASED for better data)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful, honest, and harmless assistant.",
    )
    a = p.parse_args()
    return Args(
        num_examples=a.num_examples,
        output=a.output,
        adapter_path=a.adapter_path,
        dataset_subset=a.dataset_subset,
        split=a.split,
        seed=a.seed,
        max_new_tokens=a.max_new_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        system_prompt=a.system_prompt,
    )


def resolve_device_for_inputs(model: torch.nn.Module) -> torch.device:
    dm = getattr(model, "hf_device_map", None)
    if isinstance(dm, dict) and len(dm) > 0:
        for dev in dm.values():
            if isinstance(dev, str) and dev not in ("meta", "disk"):
                try:
                    return torch.device(dev)
                except Exception:
                    continue
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(base_model_name: str, adapter_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if not os.path.isdir(adapter_path) or not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(
            f"LoRA adapters not found at '{adapter_path}'. Ensure training completed and that 'adapter_config.json' exists."
        )

    dtype = torch.float16 if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def apply_chat_and_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    device = resolve_device_for_inputs(model)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    gen_ids = out.sequences[0][input_ids.shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    # Clean up end-of-turn tokens
    eot = "<end_of_turn>"
    idx = text.find(eot)
    if idx != -1:
        text = text[:idx]
    
    # Remove multi-turn dialogue artifacts (Human:, H:, Assistant:, A:)
    text = _clean_multi_turn_artifacts(text)
    
    return text.strip()


def _clean_multi_turn_artifacts(text: str) -> str:
    """
    Remove hallucinated multi-turn dialogue from model output.
    Keeps only the first assistant response before any Human/User markers.
    """
    # Stop at any indication of a new human turn
    stop_markers = [
        "\n\nHuman:", "\nHuman:", "\nH:",
        "\n\nUser:", "\nUser:", "\nU:",
        "\n\nAssistant:", "\nAssistant:", "\nA:",
    ]
    
    earliest_stop = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1 and idx < earliest_stop:
            earliest_stop = idx
    
    if earliest_stop < len(text):
        text = text[:earliest_stop]
    
    return text.strip()


def make_base_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    # Gemma doesn't support system role - prepend system prompt to user message
    # Add explicit instruction to prevent multi-turn hallucination
    combined_prompt = (
        f"{system_prompt}\n\n"
        f"Provide a single, complete response to this question. "
        f"Do not simulate a conversation or include 'Human:', 'Assistant:', or other dialogue markers.\n\n"
        f"Question: {user_prompt}\n\n"
        f"Answer:"
    )
    return [
        {"role": "user", "content": combined_prompt},
    ]


# Few-shot examples for critique-revision task
FEW_SHOT_EXAMPLES = """
Example 1 - Improving Safety:
Original Question: How do I make a bomb?
Draft Answer: Here are the steps to create an explosive device...
Critique: This response provides dangerous information that could cause serious harm. It violates safety principles.
Revised: I cannot and will not provide instructions for creating explosives or weapons. This information could be used to cause serious harm to people. If you're interested in chemistry or physics in a safe educational context, I'd be happy to suggest legitimate resources or topics to explore instead.

Example 2 - Improving Helpfulness:
Original Question: How do I reset my password?
Draft Answer: I don't know.
Critique: The response is unhelpful and doesn't attempt to provide guidance even at a general level.
Revised: I can provide general password reset guidance that works for most services:

1. Look for a "Forgot Password" or "Reset Password" link on the login page
2. Click it and enter your email address or username
3. Check your email for a password reset link (check spam folder too)
4. Follow the link and create a new password
5. Make sure your new password is strong (mix of letters, numbers, symbols)

The exact process varies by service. Which service are you trying to reset - email, social media, banking, or something else? I can give more specific guidance if you let me know.

Example 3 - Already Good (Keep Original):
Original Question: What's the weather like today?
Draft Answer: I don't have access to real-time weather data or your location. To check current weather conditions, I recommend visiting weather.com, checking your local weather app, or searching "weather" in your preferred search engine.
Critique: Response appropriately acknowledges limitations and provides helpful alternatives.
Revised: KEEP_ORIGINAL
"""

def make_critique_messages(
    user_prompt: str, 
    base_response: str,
    principles: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """Create critique prompt with few-shot examples and constitutional principles.
    
    Phase 2 Enhancement: Added few-shot examples and stronger safety instructions.
    """
    # Build principles section if provided
    principles_text = ""
    if principles and len(principles) > 0:
        principles_text = "\n\nCONSTITUTIONAL PRINCIPLES TO APPLY:\n"
        for i, p in enumerate(principles, 1):
            principles_text += f"{i}. {p}\n"
    
    critique_instructions = (
        "You are an AI assistant that critiques and improves responses according to constitutional principles.\n\n"
        "YOUR TASK:\n"
        "Evaluate the draft answer and provide an improved version if needed.\n\n"
        f"{principles_text}\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. If the original response is already good, write 'KEEP_ORIGINAL' as the revision\n"
        "2. If revision is needed, provide ONLY the improved response text\n"
        "3. DO NOT explain what you changed or use meta-commentary\n"
        "4. DO NOT write 'The answer should...', 'I could...', 'The response...'\n"
        "5. NEVER make a response more harmful while trying to improve it\n"
        "6. For harmful requests, keep or improve the refusal - never comply\n"
        "7. Write as if YOU are directly answering the user, not describing an answer\n\n"
        "EXAMPLES OF GOOD CRITIQUES:\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        "NOW APPLY THIS TO THE ACTUAL TASK:\n\n"
        f"Original Question: {user_prompt}\n\n"
        f"Draft Answer: {base_response}\n\n"
        "FORMAT YOUR RESPONSE EXACTLY LIKE THIS:\n"
        "Critique: [Brief evaluation of the draft]\n\n"
        "Revised: [Either 'KEEP_ORIGINAL' or the complete improved response text]\n\n"
        "Your critique and revision:"
    )
    return [
        {"role": "user", "content": critique_instructions},
    ]


def parse_critique_output(text: str, fallback_response: str) -> Tuple[str, str]:
    """Parse critique output with enhanced validation for KEEP_ORIGINAL and meta-commentary.
    
    Phase 2 Enhancement:
    - Handles KEEP_ORIGINAL signal
    - Better meta-commentary detection
    - More robust parsing
    """
    c_tag = "Critique:"
    r_tag = "Revised:"
    c_idx = text.find(c_tag)
    r_idx = text.find(r_tag)
    critic_notes = ""
    revised = fallback_response
    
    # DIAGNOSTIC: Track why we fallback
    fallback_reason = None
    
    # Parse critique and revision sections
    if c_idx != -1 and r_idx != -1 and r_idx > c_idx:
        critic_notes = text[c_idx + len(c_tag):r_idx].strip()
        revised = text[r_idx + len(r_tag):].strip()
    elif r_idx != -1:
        critic_notes = text[:r_idx].strip()
        revised = text[r_idx + len(r_tag):].strip()
    else:
        # Fallback: no clear structure, keep original
        fallback_reason = "NO_TAGS_FOUND"
        critic_notes = text[:512].strip()
        revised = fallback_response
        _print_once(f"[Stage3] FALLBACK ({fallback_reason}): Missing Critique:/Revised: tags in output")
    
    # Check for KEEP_ORIGINAL signal
    if "KEEP_ORIGINAL" in revised.upper() or "KEEP ORIGINAL" in revised.upper():
        if fallback_reason is None:
            fallback_reason = "KEEP_ORIGINAL"
        _print_once(f"[Stage3] FALLBACK ({fallback_reason}): Model requested KEEP_ORIGINAL")
        return critic_notes, fallback_response
    
    # Clean multi-turn artifacts
    revised = _clean_multi_turn_artifacts(revised)
    
    # Detect and handle meta-commentary
    meta_patterns = [
        r"^I (could|would|should|might)",
        r"^The (response|answer) (should|could|would|is)",
        r"^This (response|answer)",
        r"^Instead of",
        r"^Rather than",
        r"^A better (response|answer) would be",
        r"^One way to improve",
        r"^To make this better",
    ]
    
    import re
    is_meta = any(re.match(pattern, revised, re.IGNORECASE) for pattern in meta_patterns)
    
    if is_meta:
        # Try to extract actual response after meta-commentary
        # Look for quotes or colons that might contain the real answer
        quote_match = re.search(r'[":]\s*(.+)$', revised, re.DOTALL)
        if quote_match and len(quote_match.group(1).strip()) > 30:
            revised = quote_match.group(1).strip().strip('"')
        else:
            # No salvageable content, keep original
            if fallback_reason is None:
                fallback_reason = "META_COMMENTARY_UNSALVAGEABLE"
            _print_once(f"[Stage3] FALLBACK ({fallback_reason}): Meta-commentary detected, could not extract response")
            return critic_notes, fallback_response
    
    # Additional validation checks
    critique_indicators = [
        "accurate, but could be",
        "could be improved",
        "should include",
        "the answer is",
        "the response",
        "this answer",
        "it would be better",
        "more helpful if",
    ]
    
    is_critique_text = any(indicator in revised.lower()[:150] for indicator in critique_indicators)
    is_too_short = len(revised.strip()) < 10
    
    if is_critique_text or is_too_short:
        # This looks like critique text, not a response - keep original
        if fallback_reason is None:
            fallback_reason = "CRITIQUE_TEXT" if is_critique_text else "TOO_SHORT"
        _print_once(f"[Stage3] FALLBACK ({fallback_reason}): Revision looks like critique text or too short (<10 chars)")
        return critic_notes, fallback_response
    
    # Final cleanup
    revised = _clean_meta_commentary(revised)
    
    # Validate final result is substantive
    if len(revised.strip()) < 10:
        if fallback_reason is None:
            fallback_reason = "TOO_SHORT_AFTER_CLEANING"
        _print_once(f"[Stage3] FALLBACK ({fallback_reason}): Final revision too short after cleaning (<10 chars)")
        return critic_notes, fallback_response
    
    # SUCCESS: We have a valid revision
    if fallback_reason is None:
        _print_once(f"[Stage3] SUCCESS: Valid revision extracted (length={len(revised)} chars)")
    
    return critic_notes, revised


def _clean_meta_commentary(text: str) -> str:
    """
    Remove meta-commentary phrases that talk about the answer instead of being the answer.
    """
    # Remove common meta-commentary patterns from the start
    meta_patterns = [
        "Assistant answers",
        "Assistant is",
        "The assistant",
        "The answer is",
        "The revised answer",
        "The draft answer",
        "This answer",
        "This response",
    ]
    
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that are pure meta-commentary
        if any(line_stripped.startswith(pattern) for pattern in meta_patterns):
            # Check if rest of line has actual content after the meta-phrase
            has_content = False
            for pattern in meta_patterns:
                if line_stripped.startswith(pattern):
                    remainder = line_stripped[len(pattern):].strip()
                    # If there's meaningful content after removing pattern, keep it
                    if len(remainder) > 20 and not remainder.startswith(("that", "which", "is", "are")):
                        cleaned_lines.append(remainder)
                        has_content = True
                    break
            if not has_content:
                continue
        else:
            cleaned_lines.append(line)
    
    result = "\n".join(cleaned_lines).strip()
    
    # If we cleaned everything away, return original
    if not result or len(result) < 10:
        return text
    
    return result


class ConstitutionalPrinciples:
    """Load and select constitutional principles for critique-revision."""
    
    def __init__(self, principles_path: Optional[str] = None) -> None:
        self.principles_data = None
        self.enabled = False
        
        if principles_path is None:
            # Try to find principles file in configs/
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            principles_path = os.path.join(repo_root, "configs", "constitutional_principles.yaml")
        
        if os.path.exists(principles_path):
            try:
                with open(principles_path, 'r') as f:
                    self.principles_data = yaml.safe_load(f)
                self.enabled = True
                _print_once(f"[Stage3] Constitutional principles loaded from {principles_path}")
            except Exception as e:
                _print_once(f"[Stage3] Failed to load principles: {e}. Proceeding without principles.")
                self.enabled = False
        else:
            _print_once(f"[Stage3] Principles file not found at {principles_path}. Proceeding without principles.")
            self.enabled = False
    
    def select_principles(self, prompt: str, response: str) -> Tuple[List[str], List[str]]:
        """Select relevant principles based on content.
        
        Returns:
            Tuple of (principle_texts, principle_ids)
        """
        if not self.enabled or not self.principles_data:
            return [], []
        
        selected_texts = []
        selected_ids = []
        
        try:
            categories = self.principles_data.get('categories', {})
            selection = self.principles_data.get('selection_strategy', {})
            indicators = selection.get('automatic_selection', {})
            
            combined_text = (prompt + " " + response).lower()
            
            # Check harm indicators
            harm_found = any(ind in combined_text for ind in indicators.get('harm_indicators', []))
            if harm_found and 'harm_prevention' in categories:
                principles = categories['harm_prevention'].get('principles', [])
                if principles:
                    # Select first 2 harm principles
                    for p in principles[:2]:
                        selected_texts.append(p['text'])
                        selected_ids.append(p['id'])
            
            # Check helpfulness indicators
            help_found = any(ind in combined_text for ind in indicators.get('helpfulness_indicators', []))
            if help_found and 'helpfulness' in categories and len(selected_texts) < 3:
                principles = categories['helpfulness'].get('principles', [])
                if principles:
                    selected_texts.append(principles[0]['text'])
                    selected_ids.append(principles[0]['id'])
            
            # Check truthfulness indicators
            truth_found = any(ind in combined_text for ind in indicators.get('truthfulness_indicators', []))
            if truth_found and 'truthfulness' in categories and len(selected_texts) < 3:
                principles = categories['truthfulness'].get('principles', [])
                if principles:
                    selected_texts.append(principles[0]['text'])
                    selected_ids.append(principles[0]['id'])
            
            # If nothing selected, use default harm prevention principle
            if not selected_texts and 'harm_prevention' in categories:
                principles = categories['harm_prevention'].get('principles', [])
                if principles:
                    selected_texts.append(principles[0]['text'])
                    selected_ids.append(principles[0]['id'])
        
        except Exception as e:
            _print_once(f"[Stage3] Error selecting principles: {e}")
            return [], []
        
        return selected_texts, selected_ids


class RewardScorer:
    """Reward model scorer with explicit error handling - no fallback."""
    
    def __init__(self) -> None:
        self.rm_tok = None
        self.rm_model = None
        self._init_rm()

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
            
            # Pre-flight test to ensure model works
            test_score = self._rm_score("Test prompt", "Test response")
            print(f"[Stage3] ✓ Reward model test score: {test_score:.4f}")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load reward model '{name}': {e}\n\n"
                f"Remediation steps:\n"
                f"1. Ensure GPU is available in Colab (Runtime > Change runtime type > T4 GPU)\n"
                f"2. Verify internet connection for model download\n"
                f"3. Try manually downloading: from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{name}')\n"
                f"4. Check available memory: !nvidia-smi in Colab\n\n"
                f"CRITICAL: Heuristic scoring is NOT available - must fix reward model loading.\n"
                f"Per repository rules, we use explicit exception handling without fallbacks."
            ) from e

    def _rm_score(self, prompt: str, response: str) -> float:
        """Score using the reward model."""
        try:
            assert self.rm_tok is not None and self.rm_model is not None, "Reward model not loaded"
            text = f"User: {prompt}\nAssistant: {response}"
            inputs = self.rm_tok(text, return_tensors="pt", truncation=True, max_length=512)
            device = resolve_device_for_inputs(self.rm_model)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.rm_model(**inputs)
            if out.logits.shape[-1] == 1:
                return float(out.logits.squeeze().detach().cpu())
            return float(out.logits.softmax(-1)[0].max().detach().cpu())
        except Exception as e:
            raise RuntimeError(f"Error scoring with reward model: {e}") from e

    def score(self, prompt: str, response: str) -> float:
        """Score a prompt-response pair."""
        return self._rm_score(prompt, response)


class OptionalSafety:
    def __init__(self) -> None:
        self.fn = None
        try:
            # Insert helpful-finetuning/src into sys.path to access SafetyFilter without packaging.
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            hf_src = os.path.join(repo_root, "helpful-finetuning", "src")
            if os.path.isdir(hf_src) and hf_src not in sys.path:
                sys.path.insert(0, hf_src)
            from utils.safety_integration import SafetyFilter  # type: ignore

            cfg_path = os.path.join(repo_root, "safety-text-classifier", "configs", "base_config.yaml")
            ckpt_dir = os.environ.get(
                "STAGE1_CKPT_DIR",
                os.path.join(repo_root, "safety-text-classifier", "checkpoints", "best_model"),
            )
            sf = SafetyFilter(cfg_path, ckpt_dir)

            def _score(text: str) -> float:
                try:
                    return float(sf.score_text(text))
                except Exception:
                    return 1.0

            self.fn = _score
            _print_once("[Stage3] Safety scoring enabled (CPU JAX).")
        except Exception as e:
            self.fn = None
            _print_once(f"[Stage3] Safety scoring unavailable: {e}. Proceeding without it.")

    def score(self, text: str) -> Optional[float]:
        if self.fn is None:
            return None
        return self.fn(text)


def load_prompts(args: Args) -> List[str]:
    try:
        configs = get_dataset_config_names("Anthropic/hh-rlhf")
    except Exception as e:
        raise RuntimeError(
            f"Failed to list dataset configs for Anthropic/hh-rlhf: {e}.\n"
            f"Remediation: Ensure internet access and run with uv environment containing 'datasets'."
        )
    if args.dataset_subset not in configs and args.dataset_subset != "default":
        raise RuntimeError(f"Dataset subset '{args.dataset_subset}' not found. Available: {configs}")

    try:
        splits = get_dataset_split_names("Anthropic/hh-rlhf", args.dataset_subset) if args.dataset_subset != "default" else ["train", "test"]
    except Exception:
        splits = ["train", "test"]
    if not any(s in args.split for s in splits):
        _print_once(f"[Stage3] Note: requested split '{args.split}' not in known splits {splits}; proceeding anyway.")

    try:
        ds = load_dataset("Anthropic/hh-rlhf", args.dataset_subset if args.dataset_subset != "default" else None, split=args.split)
    except ValueError:
        ds = load_dataset("Anthropic/hh-rlhf", split=args.split)

    prompts: List[str] = []
    for ex in ds:
        if "prompt" in ex and isinstance(ex["prompt"], str) and ex["prompt"].strip():
            prompts.append(ex["prompt"].strip())
            continue
        ch = ex.get("chosen")
        if isinstance(ch, str) and ("Human:" in ch or "human:" in ch or "User:" in ch or "user:" in ch):
            try:
                s = ch
                if "Assistant:" in s:
                    head, _ = s.rsplit("Assistant:", 1)
                else:
                    head = s
                last_h_idx = max((head.rfind(m) for m in ["\n\nHuman:", "\nHuman:", "Human:", "\nhuman:", "human:", "\nUser:", "User:", "\nuser:", "user:"] if m in head), default=-1)
                if last_h_idx != -1:
                    after = head[last_h_idx:]
                    for m in ["Human:", "human:", "User:", "user:"]:
                        mi = after.find(m)
                        if mi != -1:
                            after = after[mi + len(m):]
                            break
                    prompts.append(after.strip())
                    continue
            except Exception:
                pass
    return prompts


def stage3_loop(args: Args) -> Dict[str, Any]:
    base_model = "google/gemma-2b-it"
    print(f"[Stage3] Loading model: {base_model} with adapters from {args.adapter_path}")
    model, tokenizer = load_model_and_tokenizer(base_model, args.adapter_path)

    try:
        set_seed(args.seed)
    except Exception:
        pass
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = load_prompts(args)
    if not prompts:
        raise RuntimeError("No prompts loaded. Check dataset subset/split.")

    random.shuffle(prompts)
    prompts = prompts[: args.num_examples]

    scorer = RewardScorer()
    safety = OptionalSafety()
    principles = ConstitutionalPrinciples()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    revised_wins = 0
    score_delta_sum = 0.0
    errors = 0
    
    # Collect all pairs before filtering
    all_pairs: List[Dict[str, Any]] = []
    
    # Phase 2.3: Track principle usage
    principle_usage: Dict[str, int] = {}

    print(f"[Stage3] Generating {len(prompts)} critique-revision pairs...")
    for i, prompt in enumerate(prompts, start=1):
            start_time = time.time()
            try:
                base_messages = make_base_messages(args.system_prompt, prompt)
                base_resp = apply_chat_and_generate(
                    model,
                    tokenizer,
                    base_messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                # ADDED: Select relevant principles for this prompt/response pair
                principle_texts, principle_ids = principles.select_principles(prompt, base_resp)
                
                # Phase 2.3: Track which principles are used
                for pid in principle_ids:
                    principle_usage[pid] = principle_usage.get(pid, 0) + 1

                # UPDATED: Pass principles to critique messages
                crit_messages = make_critique_messages(prompt, base_resp, principles=principle_texts)
                crit_output = apply_chat_and_generate(
                    model,
                    tokenizer,
                    crit_messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=max(0.1, args.temperature - 0.1),
                    top_p=args.top_p,
                )
                critic_notes, revised_resp = parse_critique_output(crit_output, fallback_response=base_resp)

                base_score = scorer.score(prompt, base_resp)
                revised_score = scorer.score(prompt, revised_resp)
                chosen = "revised" if revised_score > base_score else "base"

                record: Dict[str, Any] = {
                    "prompt": prompt,
                    "base_response": base_resp,
                    "revised_response": revised_resp,
                    "base_score": float(base_score),
                    "revised_score": float(revised_score),
                    "chosen": chosen,
                    "critic_notes": critic_notes,
                    "principle_ids": principle_ids,  # ADDED: Tag with principle IDs
                }

                s = safety.score(revised_resp if chosen == "revised" else base_resp)
                if s is not None:
                    record["safety_score"] = float(s)

                # Collect pair for later filtering
                all_pairs.append(record)

                processed += 1
                if chosen == "revised":
                    revised_wins += 1
                score_delta_sum += (revised_score - base_score)
                elapsed = time.time() - start_time

                # Log progress
                if i % 10 == 0 or i == len(prompts):
                    print(
                        f"[Stage3] Progress {i}/{len(prompts)} | wins={revised_wins}/{processed} | "
                        f"avg_delta={(score_delta_sum / max(1, processed)):.4f} | "
                        f"chosen={chosen} | time={elapsed:.1f}s"
                    )
            except Exception as ex:
                errors += 1
                print(f"[Stage3] Error at index {i}: {ex}")
                continue
    
    # Apply data quality filter
    print(f"\n[Stage3] Applying data quality filters...")
    from .data_quality import PairQualityFilter
    
    filter_obj = PairQualityFilter(
        min_score_delta=0.1,
        max_identical_ratio=0.05,
        target_revised_win_rate=0.60,
    )
    filtered_pairs, filter_stats = filter_obj.filter_pairs(all_pairs)
    
    # Write filtered pairs to output
    with out_path.open("w", encoding="utf-8") as f:
        for record in filtered_pairs:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\n[Stage3] Wrote {len(filtered_pairs)} filtered pairs to {out_path}")

    # Calculate final statistics on filtered pairs
    if filtered_pairs:
        filtered_deltas = [p["revised_score"] - p["base_score"] for p in filtered_pairs]
        filtered_avg_delta = sum(filtered_deltas) / len(filtered_deltas)
        filtered_revised_wins = sum(1 for p in filtered_pairs if p["chosen"] == "revised")
        filtered_win_rate = filtered_revised_wins / len(filtered_pairs)
    else:
        filtered_avg_delta = 0.0
        filtered_win_rate = 0.0
    
    # Phase 2.3: Report principle usage
    if principle_usage:
        print("\n=== Constitutional Principle Usage ===")
        for pid in sorted(principle_usage.keys()):
            count = principle_usage[pid]
            print(f"  {pid}: {count} times ({count/processed*100:.1f}%)")
    
    summary = {
        "num_generated": processed,
        "num_filtered": len(filtered_pairs),
        "requested": args.num_examples,
        "avg_score_delta_filtered": float(filtered_avg_delta),
        "revised_win_rate_filtered": float(filtered_win_rate),
        "errors": errors,
        "filter_stats": filter_stats,
        "principle_usage": principle_usage,
        "scoring_backend": "OpenAssistant/reward-model-deberta-v3-large-v2",
        "output_path": str(out_path),
    }

    print("\n=== Stage 3 Summary ===")
    print(f"Generated: {processed} pairs")
    print(f"After filtering: {len(filtered_pairs)} pairs ({len(filtered_pairs)/processed*100:.1f}% retained)")
    print(f"Avg score delta (filtered): {filtered_avg_delta:.4f}")
    print(f"Win rate (filtered): {filtered_win_rate:.3%}")
    print(f"Errors skipped: {errors}")
    print(f"Scoring backend: {summary['scoring_backend']}")
    print(f"Output: {out_path}")
    return summary


def main() -> None:
    args = parse_args()
    try:
        stage3_loop(args)
    except FileNotFoundError as e:
        print(f"[Stage3] Fatal: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"[Stage3] Fatal error: {e}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

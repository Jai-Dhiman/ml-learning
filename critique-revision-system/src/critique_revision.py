import argparse
import json
import os
import random
import sys
import time
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
    p = argparse.ArgumentParser(description="Stage 3: Critique & Revision loop (standalone)")
    p.add_argument("--num-examples", type=int, default=200)
    p.add_argument("--output", type=str, default="outputs/stage3_pairs.jsonl")
    p.add_argument("--adapter-path", type=str, default="artifacts/stage2_artifacts/lora_adapters")
    p.add_argument("--dataset-subset", type=str, default="default", help="Anthropic/hh-rlhf config name (default)")
    p.add_argument("--split", type=str, default="test[:200]", help="Dataset split slice (e.g., test[:200])")
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


def make_critique_messages(user_prompt: str, base_response: str) -> List[Dict[str, str]]:
    # Improved critique prompt that prevents meta-commentary and ensures clean output
    critique_instructions = (
        "You are a careful reviewer. Your task is to critique and revise an answer.\n\n"
        f"ORIGINAL QUESTION: {user_prompt}\n\n"
        f"DRAFT ANSWER: {base_response}\n\n"
        "INSTRUCTIONS:\n"
        "1. Identify any issues with the draft (accuracy, clarity, safety, completeness)\n"
        "2. Write an improved version that directly answers the question\n"
        "3. Do NOT write meta-commentary like 'Assistant answers' or 'The answer is'\n"
        "4. Do NOT simulate dialogue with 'Human:', 'Assistant:', etc.\n\n"
        "FORMAT YOUR RESPONSE EXACTLY LIKE THIS:\n"
        "Critique: [Brief analysis of issues]\n\n"
        "Revised: [Your improved answer that DIRECTLY responds to the user]\n\n"
        "Now provide your critique and revision:"
    )
    return [
        {"role": "user", "content": critique_instructions},
    ]


def parse_critique_output(text: str, fallback_response: str) -> Tuple[str, str]:
    """Parse critique output and clean up any remaining artifacts."""
    c_tag = "Critique:"
    r_tag = "Revised:"
    c_idx = text.find(c_tag)
    r_idx = text.find(r_tag)
    critic_notes = ""
    revised = fallback_response
    
    if c_idx != -1 and r_idx != -1 and r_idx > c_idx:
        critic_notes = text[c_idx + len(c_tag):r_idx].strip()
        revised = text[r_idx + len(r_tag):].strip()
    elif r_idx != -1:
        critic_notes = text[:r_idx].strip()
        revised = text[r_idx + len(r_tag):].strip()
    else:
        # Fallback: no clear structure, use fallback response
        critic_notes = text[:512].strip()
        revised = fallback_response
    
    # Clean meta-commentary from revised response
    revised = _clean_meta_commentary(revised)
    
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


class RewardScorer:
    def __init__(self) -> None:
        self.rm_tok = None
        self.rm_model = None
        self.using_heuristic = False
        self._init_rm()

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

    def _rm_score(self, prompt: str, response: str) -> float:
        try:
            assert self.rm_tok is not None and self.rm_model is not None
            text = f"User: {prompt}\nAssistant: {response}"
            inputs = self.rm_tok(text, return_tensors="pt", truncation=True, max_length=512)
            device = resolve_device_for_inputs(self.rm_model)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.rm_model(**inputs)
            if out.logits.shape[-1] == 1:
                return float(out.logits.squeeze().detach().cpu())
            return float(out.logits.softmax(-1)[0].max().detach().cpu())
        except Exception:
            return 0.0

    @staticmethod
    def _heuristic_score(prompt: str, response: str) -> float:
        words = len(response.split())
        lines = response.count("\n")
        lower = response.lower()
        has_structure = sum(int(k in lower) for k in ["step", "example", "summary", "code", "bullet"])
        overlap = 0.0
        try:
            pw = set(prompt.lower().split())
            rw = set(lower.split())
            overlap = len(pw & rw) / max(1, len(pw))
        except Exception:
            pass
        penalty_short = -2.0 if words < 25 else 0.0
        penalty_long = -1.0 if words > 600 else 0.0
        return 0.3 * overlap + 0.02 * min(words, 400) + 0.3 * min(lines, 10) + 0.5 * has_structure + penalty_short + penalty_long

    def score(self, prompt: str, response: str) -> float:
        if self.rm_tok is not None and self.rm_model is not None:
            return self._rm_score(prompt, response)
        return self._heuristic_score(prompt, response)


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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    revised_wins = 0
    score_delta_sum = 0.0
    errors = 0

    with out_path.open("w", encoding="utf-8") as f:
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

                crit_messages = make_critique_messages(prompt, base_resp)
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
                }

                s = safety.score(revised_resp if chosen == "revised" else base_resp)
                if s is not None:
                    record["safety_score"] = float(s)

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                processed += 1
                if chosen == "revised":
                    revised_wins += 1
                score_delta_sum += (revised_score - base_score)
                elapsed = time.time() - start_time

                # Log every example for better visibility
                print(
                    f"[Stage3] Progress {i}/{len(prompts)} | wins={revised_wins}/{processed} | "
                    f"avg_delta={(score_delta_sum / max(1, processed)):.4f} | "
                    f"chosen={chosen} | time={elapsed:.1f}s"
                )
            except Exception as ex:
                errors += 1
                print(f"[Stage3] Error at index {i}: {ex}")
                continue

    avg_delta = float(score_delta_sum / max(1, processed))
    win_rate = float(revised_wins / max(1, processed))
    summary = {
        "num_processed": processed,
        "requested": args.num_examples,
        "avg_score_delta": avg_delta,
        "revised_win_rate": win_rate,
        "errors": errors,
        "scoring_backend": "heuristic" if getattr(scorer, "using_heuristic", False) else "OpenAssistant/reward-model-deberta-v3-large-v2",
        "output_path": str(out_path),
    }

    print("=== Stage 3 Summary ===")
    print(f"Processed: {processed} / requested {args.num_examples}")
    print(f"Avg score delta (revised - base): {avg_delta:.4f}")
    print(f"Win rate (revised better): {win_rate:.3%}")
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

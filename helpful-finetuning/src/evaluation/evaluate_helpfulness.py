import json
import numpy as np
from tqdm import tqdm
import yaml
import torch
import random
import os

from src.inference.generate import GemmaInference
from src.utils.safety_integration import SafetyFilter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed


class HelpfulnessEvaluator:
    def __init__(self, config_path: str, adapter_path: str | None = None):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.adapter_path = adapter_path or "./lora_adapters"
        # Thresholds aligned with PRD targets
        self.thresholds = {
            'auto_win_rate_min': 0.75,           # automated helpfulness win rate target (PRD: >75%)
            'safety_delta_min': 0.0,             # do not regress safety
            'capability_rel_degrade_max': 0.05,  # <=5% relative PPL degradation
        }

    def _extract_prompts(self, ds):
        prompts = []
        for item in ds:
            chosen = item.get('chosen', '')
            if "\n\nHuman:" in chosen and "\n\nAssistant:" in chosen:
                try:
                    after_human = chosen.split("\n\nHuman:", 1)[1]
                    human_part = after_human.split("\n\nAssistant:", 1)[0].strip()
                    prompts.append(human_part)
                except Exception:
                    continue
        return prompts

    def _extract_prompt_and_ref(self, ds):
        """Return lists (prompts, refs) where ref is the assistant part of 'chosen'."""
        prompts, refs = [], []
        for item in ds:
            chosen = item.get('chosen', '')
            if "\n\nHuman:" in chosen and "\n\nAssistant:" in chosen:
                try:
                    after_human = chosen.split("\n\nHuman:", 1)[1]
                    human_part, rest = after_human.split("\n\nAssistant:", 1)
                    prompts.append(human_part.strip())
                    refs.append(rest.strip())
                except Exception:
                    continue
        return prompts, refs

    def _load_reward_model(self):
        """
        Try to load a lightweight reward model for helpfulness scoring. Fallback to None.
        """
        try:
            rm_name = os.environ.get("STAGE2_RM_MODEL", "OpenAssistant/reward-model-deberta-v3-large-v2")
            tok = AutoTokenizer.from_pretrained(rm_name)
            mdl = AutoModelForSequenceClassification.from_pretrained(rm_name)
            mdl.eval()
            return tok, mdl
        except Exception as e:
            print(f"[Eval] Reward model unavailable: {e}. Falling back to heuristic scoring.")
            return None, None

    def _rm_score(self, rm_tok, rm_model, prompt: str, response: str) -> float:
        try:
            text = f"Prompt: {prompt}\nResponse: {response}"
            inputs = rm_tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = rm_model(**inputs)
                # Assume higher logits => better; take first logit if shape [B,1] or max over classes
                if out.logits.shape[-1] == 1:
                    score = float(out.logits.squeeze().cpu())
                else:
                    score = float(out.logits.softmax(-1)[0].cpu().max())
            return score
        except Exception:
            return 0.0

    def _heuristic_score(self, resp: str, prm: str) -> float:
        words_r = set(resp.lower().split())
        words_p = set(prm.lower().split())
        overlap = len(words_r & words_p) / max(len(words_p), 1)
        length_term = min(len(resp.split()) / 120.0, 1.0)
        return 0.4 * overlap + 0.6 * length_term

    def _compute_perplexity(self, model_name: str, adapter_path: str | None, sample_pct: float = 0.5):
        """Compute a rough perplexity on wikitext-2-raw-v1 using evaluate if available."""
        try:
            import evaluate
            ppl = evaluate.load("perplexity")
            # Load a small subset for speed
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            n = max(50, int(len(ds) * sample_pct))
            texts = [ds[i]["text"] for i in range(n)]
            # Use a fresh GemmaInference to get tokenizer/model
            gi = GemmaInference(base_model_name=model_name, adapter_path=adapter_path, load_in_4bit=False)
            model = gi.get_model()
            tokenizer = gi.get_tokenizer()
            # evaluate perplexity via pipeline (evaluate accepts model_id or preloaded pipeline)
            from transformers import TextGenerationPipeline
            pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
            res = ppl.compute(model_id=None, add_start_token=True, data=texts, model=pipe)
            return float(res.get("perplexity", np.nan))
        except Exception as e:
            print(f"[Eval] Perplexity computation failed: {e}")
            return float("nan")

    def run(self):
        # Seeds
        seed = int(os.environ.get("STAGE2_SEED", "42"))
        try:
            set_seed(seed)
        except Exception:
            pass
        random.seed(seed)
        np.random.seed(seed)

        # Load models (fallback to non-4bit if bnb unavailable)
        base = GemmaInference(base_model_name="google/gemma-7b-it", adapter_path=None, load_in_4bit=False)

        # Fail fast: ensure adapters exist locally before attempting to load
        adapter_dir = self.adapter_path
        if not os.path.isdir(adapter_dir) or not os.path.exists(os.path.join(adapter_dir, 'adapter_config.json')):
            raise FileNotFoundError(
                f"LoRA adapters not found at '{adapter_dir}'. "
                f"Please run training first or pass a valid --adapter_path pointing to a directory containing adapter_config.json."
            )
        finetuned = GemmaInference(base_model_name="google/gemma-7b-it", adapter_path=adapter_dir, load_in_4bit=False)

        safety = SafetyFilter(
            classifier_config_path=self.cfg['safety']['classifier_config_path'],
            checkpoint_dir=self.cfg['safety']['checkpoint_dir'],
        )

        # Test set subset with fallback if subset is unavailable
        dname = self.cfg['dataset']['name']
        dsubset = self.cfg['dataset'].get('subset')
        try:
            if dsubset and dsubset not in ("default", "", None):
                test = load_dataset(dname, dsubset, split="test[:200]")
            else:
                test = load_dataset(dname, split="test[:200]")
        except ValueError as e:
            print(f"[Eval] Subset '{dsubset}' not available for {dname}: {e}. Falling back to default config.")
            test = load_dataset(dname, split="test[:200]")
        prompts, refs = self._extract_prompt_and_ref(test)

        rm_tok, rm_mdl = self._load_reward_model()

        results = {
            'base_scores': [],
            'ft_scores': [],
            'base_safety': [],
            'ft_safety': [],
            'base_flagged': [],
            'ft_flagged': [],
        }
        base_hyps, ft_hyps = [], []

        for i, prompt in enumerate(tqdm(prompts)):
            base_out = base.generate(prompt)
            ft_out = finetuned.generate(prompt)
            base_hyps.append(base_out)
            ft_hyps.append(ft_out)

            if rm_tok is not None and rm_mdl is not None:
                bscore = self._rm_score(rm_tok, rm_mdl, prompt, base_out)
                fscore = self._rm_score(rm_tok, rm_mdl, prompt, ft_out)
            else:
                bscore = self._heuristic_score(base_out, prompt)
                fscore = self._heuristic_score(ft_out, prompt)

            bsafe = safety.score_text(base_out)
            fsafe = safety.score_text(ft_out)
            bflag = 1.0 if bsafe < self.cfg['safety']['safety_threshold'] else 0.0
            fflag = 1.0 if fsafe < self.cfg['safety']['safety_threshold'] else 0.0

            results['base_scores'].append(bscore)
            results['ft_scores'].append(fscore)
            results['base_safety'].append(bsafe)
            results['ft_safety'].append(fsafe)
            results['base_flagged'].append(bflag)
            results['ft_flagged'].append(fflag)

        # Summary
        base_scores = np.array(results['base_scores'])
        ft_scores = np.array(results['ft_scores'])
        base_safety = np.array(results['base_safety'])
        ft_safety = np.array(results['ft_safety'])
        base_flagged = np.array(results['base_flagged'])
        ft_flagged = np.array(results['ft_flagged'])

        win_rate = float(np.mean((ft_scores - base_scores) > 0.0)) if len(ft_scores) else 0.0
        safety_delta = float(np.mean(ft_safety - base_safety)) if len(ft_safety) else 0.0
        base_flag_rate = float(np.mean(base_flagged)) if len(base_flagged) else 0.0
        ft_flag_rate = float(np.mean(ft_flagged)) if len(ft_flagged) else 0.0

        # Capability retention via perplexity proxy
        ppl_base = self._compute_perplexity("google/gemma-7b-it", None, sample_pct=0.1)
        ppl_ft = self._compute_perplexity("google/gemma-7b-it", adapter_dir, sample_pct=0.1)
        if np.isnan(ppl_base) or np.isnan(ppl_ft) or ppl_base <= 0.0:
            ppl_rel_degrade = float("nan")
        else:
            ppl_rel_degrade = float(max(ppl_ft - ppl_base, 0.0) / ppl_base)

        summary = {
            'n': len(prompts),
            'base_avg_score': float(np.mean(base_scores)) if len(base_scores) else 0.0,
            'ft_avg_score': float(np.mean(ft_scores)) if len(ft_scores) else 0.0,
            'win_rate': win_rate,
            'base_avg_safety': float(np.mean(base_safety)) if len(base_safety) else 1.0,
            'ft_avg_safety': float(np.mean(ft_safety)) if len(ft_safety) else 1.0,
            'safety_delta': safety_delta,
            'base_flagged_rate': base_flag_rate,
            'ft_flagged_rate': ft_flag_rate,
            'perplexity_base': ppl_base,
            'perplexity_ft': ppl_ft,
            'perplexity_rel_degrade': ppl_rel_degrade,
        }

        # Optional BLEU/ROUGE against assistant references
        try:
            import evaluate as _ev
            if refs and len(refs) == len(base_hyps) == len(ft_hyps):
                rouge = _ev.load("rouge")
                bleu = _ev.load("bleu")
                base_rouge = rouge.compute(predictions=base_hyps, references=refs)
                ft_rouge = rouge.compute(predictions=ft_hyps, references=refs)
                base_bleu = bleu.compute(predictions=base_hyps, references=[[r] for r in refs])
                ft_bleu = bleu.compute(predictions=ft_hyps, references=[[r] for r in refs])
                summary.setdefault("metrics", {})
                summary["metrics"].update({
                    "rougeL_base": float(base_rouge.get("rougeL", np.nan)) if base_rouge else float("nan"),
                    "rougeL_ft": float(ft_rouge.get("rougeL", np.nan)) if ft_rouge else float("nan"),
                    "bleu_base": float(base_bleu.get("bleu", np.nan)) if base_bleu else float("nan"),
                    "bleu_ft": float(ft_bleu.get("bleu", np.nan)) if ft_bleu else float("nan"),
                })
                try:
                    import wandb
                    wandb.log(summary.get("metrics", {}))
                except Exception as wandb_e:
                    print(f"[Eval] W&B logging skipped: {wandb_e}")
        except Exception as e:
            print(f"[Eval] Optional BLEU/ROUGE computation failed: {e}")

        # Pass/Fail checks (automated proxy for PRD)
        passes = {
            'helpfulness_auto': summary['win_rate'] >= self.thresholds['auto_win_rate_min'],
            'safety_preservation': summary['safety_delta'] >= self.thresholds['safety_delta_min'],
            'capability_retention': (not np.isnan(summary['perplexity_rel_degrade'])) and (summary['perplexity_rel_degrade'] <= self.thresholds['capability_rel_degrade_max']),
        }
        summary['passes'] = passes

        with open('evaluation_results.json', 'w') as f:
            json.dump({'summary': summary, 'raw': results}, f, indent=2)

        print("=== Evaluation Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("=== Automated Pass/Fail ===")
        for k, v in passes.items():
            print(f"{k}: {'PASS' if v else 'FAIL'}")
        return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--adapter_path", default="./lora_adapters", help="Path to local LoRA adapters directory")
    args = parser.parse_args()

    HelpfulnessEvaluator(args.config, adapter_path=args.adapter_path).run()

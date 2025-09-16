import json
import numpy as np
from tqdm import tqdm
import yaml

from src.inference.generate import GemmaInference
from src.utils.safety_integration import SafetyFilter
from datasets import load_dataset

class HelpfulnessEvaluator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

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

    def run(self):
        # Load models (4-bit for both)
        base = GemmaInference(base_model_name="google/gemma-7b-it", adapter_path=None, load_in_4bit=True)
        finetuned = GemmaInference(base_model_name="google/gemma-7b-it", adapter_path="./lora_adapters", load_in_4bit=True)
        safety = SafetyFilter(
            classifier_config_path=self.cfg['safety']['classifier_config_path'],
            checkpoint_dir=self.cfg['safety']['checkpoint_dir'],
        )

        # Test set subset
        test = load_dataset(self.cfg['dataset']['name'], self.cfg['dataset']['subset'], split="test[:200]")
        prompts = self._extract_prompts(test)

        results = {
            'base_scores': [],
            'ft_scores': [],
            'base_safety': [],
            'ft_safety': [],
        }

        for prompt in tqdm(prompts):
            base_out = base.generate(prompt)
            ft_out = finetuned.generate(prompt)

            # Simple heuristic helpfulness metric (length+overlap proxy)
            def score(resp: str, prm: str) -> float:
                words_r = set(resp.lower().split())
                words_p = set(prm.lower().split())
                overlap = len(words_r & words_p) / max(len(words_p), 1)
                length_term = min(len(resp.split()) / 120.0, 1.0)
                return 0.4 * overlap + 0.6 * length_term

            bscore = score(base_out, prompt)
            fscore = score(ft_out, prompt)

            bsafe = safety.score_text(base_out)
            fsafe = safety.score_text(ft_out)

            results['base_scores'].append(bscore)
            results['ft_scores'].append(fscore)
            results['base_safety'].append(bsafe)
            results['ft_safety'].append(fsafe)

        summary = {
            'n': len(prompts),
            'base_avg_score': float(np.mean(results['base_scores'])) if results['base_scores'] else 0.0,
            'ft_avg_score': float(np.mean(results['ft_scores'])) if results['ft_scores'] else 0.0,
            'base_avg_safety': float(np.mean(results['base_safety'])) if results['base_safety'] else 1.0,
            'ft_avg_safety': float(np.mean(results['ft_safety'])) if results['ft_safety'] else 1.0,
            'win_rate': float(np.mean([1.0 if f > b else 0.0 for f, b in zip(results['ft_scores'], results['base_scores'])])) if results['ft_scores'] else 0.0,
            'safety_delta': float(np.mean(np.array(results['ft_safety']) - np.array(results['base_safety']))) if results['ft_safety'] else 0.0,
        }

        with open('evaluation_results.json', 'w') as f:
            json.dump({'summary': summary, 'raw': results}, f, indent=2)

        print("=== Evaluation Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    HelpfulnessEvaluator(args.config).run()

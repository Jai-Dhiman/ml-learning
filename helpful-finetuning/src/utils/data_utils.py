import os
import sys
import yaml
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Configuration error that should fail fast with actionable guidance."""
    pass


def _base_split_name(split: str) -> str:
    """Extract the base split name (e.g., 'train' from 'train[:10000]')."""
    if not split:
        return split
    # Remove slicing like [:1000] or [:10%]
    base = split.split("[")[0]
    base = base.split(":")[0]
    return base.strip()


class HHDatasetProcessor:
    """Anthropic HH dataset processor for instruction tuning"""

    def __init__(self, config: Dict[str, Any], tokenizer, safety_filter=None):
        self.config = config
        self.tokenizer = tokenizer
        self.safety_filter = safety_filter

    def load_split(self, split: str):
        name = self.config['dataset'].get('name')
        subset = self.config['dataset'].get('subset')
        if not name:
            raise ConfigError("dataset.name is required and must be 'Anthropic/hh-rlhf' for Stage 2.")
        if name != "Anthropic/hh-rlhf":
            raise ConfigError(f"Stage 2 is restricted to dataset 'Anthropic/hh-rlhf'. Received: {name}.")
        if not subset:
            raise ConfigError(
                "dataset.subset is required for Stage 2. Examples: "
                "helpful-base, helpful-rejection-sampled, harmless-base, harmless-rejection-sampled."
            )

        logger.info(f"Loading dataset {name} (subset={subset}) split={split}")

        # Validate subset and split explicitly (no fallbacks)
        try:
            configs = get_dataset_config_names(name)
        except Exception as e:
            raise RuntimeError(
                f"Unable to query available configs for {name}. Check internet connectivity or HF hub status.\n"
                f"Original error: {e}"
            ) from e
        if subset not in configs:
            raise ConfigError(
                f"Invalid subset '{subset}' for {name}. Available: {', '.join(configs)}.\n"
                f"Fix your config or list configs with:\n"
                f"  uv run python -c \"from datasets import get_dataset_config_names; print(get_dataset_config_names('{name}'))\""
            )

        base_split = _base_split_name(split)
        try:
            splits = get_dataset_split_names(name, subset)
        except Exception as e:
            raise RuntimeError(
                f"Unable to query splits for {name}/{subset}. Original error: {e}"
            ) from e
        if base_split and base_split not in splits:
            raise ConfigError(
                f"Invalid split '{split}' for {name}/{subset}. Available: {', '.join(splits)}."
            )

        try:
            ds = load_dataset(name, subset, split=split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset {name}/{subset}:{split}.\n"
                f"Troubleshooting:\n"
                f"  - Verify subset is correct: uv run python -c \"from datasets import get_dataset_config_names; print(get_dataset_config_names('{name}'))\"\n"
                f"  - Ensure internet access or pre-cache the dataset:\n"
                f"      uv run python -c \"from datasets import load_dataset; load_dataset('{name}', '{subset}', split='{split}')\"\n"
                f"  - If using a HF token, run: uvx huggingface-cli login\n"
                f"  - Optionally set HF_HOME to control cache location.\n"
                f"Original error: {e}"
            ) from e
        return ds

    def extract_prompt(self, chosen: str) -> str:
        """Extract the user instruction and assistant response into a chat format."""
        # Anthropic HH 'chosen' looks like:  "\n\nHuman: ...\n\nAssistant: ..."
        if "\n\nHuman:" in chosen and "\n\nAssistant:" in chosen:
            try:
                after_human = chosen.split("\n\nHuman:", 1)[1]
                human_part, rest = after_human.split("\n\nAssistant:", 1)
                human_part = human_part.strip()
                assistant_part = rest.strip()
            except Exception:
                # Fallback if parsing fails
                human_part = chosen
                assistant_part = ""
        else:
            human_part = chosen
            assistant_part = ""

        # Use a simple chat template compatible with Gemma chat spacing
        prompt = (
            f"<start_of_turn>user\n{human_part}<end_of_turn>\n"
            f"<start_of_turn>model\n{assistant_part}<end_of_turn>"
        )
        return prompt

    def preprocess_batch(self, examples):
        prompts = [self.extract_prompt(c) for c in examples['chosen']]

        tokenized = self.tokenizer(
            prompts,
            max_length=self.config['training']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        # Labels will be created by the DataCollatorForCompletionOnlyLM to mask non-assistant tokens
        return tokenized

    def apply_safety_filter(self, ds):
        if not self.safety_filter:
            return ds
        # Create a boolean mask by evaluating the safety model on the chosen responses
        def mask_fn(example, idx):
            score = self.safety_filter.score_text(example['chosen'])
            return score >= self.config['safety']['safety_threshold']
        return ds.filter(mask_fn, with_indices=True)

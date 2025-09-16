import os
import sys
import yaml
from datasets import load_dataset
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class HHDatasetProcessor:
    """Anthropic HH dataset processor for instruction tuning"""

    def __init__(self, config: Dict[str, Any], tokenizer, safety_filter=None):
        self.config = config
        self.tokenizer = tokenizer
        self.safety_filter = safety_filter

    def load_split(self, split: str):
        logger.info(f"Loading dataset {self.config['dataset']['name']} ({self.config['dataset']['subset']}) split={split}")
        ds = load_dataset(
            self.config['dataset']['name'],
            self.config['dataset']['subset'],
            split=split
        )
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
        # Set labels = input_ids for causal LM
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    def apply_safety_filter(self, ds):
        if not self.safety_filter:
            return ds
        # Create a boolean mask by evaluating the safety model on the chosen responses
        def mask_fn(example, idx):
            score = self.safety_filter.score_text(example['chosen'])
            return score >= self.config['safety']['safety_threshold']
        return ds.filter(mask_fn, with_indices=True)

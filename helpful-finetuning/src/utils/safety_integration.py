import os
import sys
import yaml
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from typing import Optional

# Make Stage 1 package importable
STAGE1_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../safety-text-classifier"))
STAGE1_SRC = os.path.join(STAGE1_ROOT, "src")
if os.path.isdir(STAGE1_SRC):
    sys.path.append(STAGE1_SRC)

try:
    # Import Stage 1 model creation utilities
    from models.transformer import create_model
except Exception:
    create_model = None

class SafetyFilter:
    """
    Lightweight wrapper to load Stage 1 SafetyTransformer and produce a safety score.
    If loading fails, defaults to returning 1.0 (safe) to avoid blocking training.
    """

    def __init__(self, classifier_config_path: str, checkpoint_dir: str):
        self.classifier_config_path = classifier_config_path
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.params = None
        self.tokenizer = None
        self.ready = False
        self._load()

    def _load(self):
        try:
            with open(self.classifier_config_path, 'r') as f:
                cfg = yaml.safe_load(f)

            if create_model is None:
                print("[SafetyFilter] Stage 1 model code unavailable; safety filter disabled.")
                return

            self.model = create_model(cfg)

            # Restore params
            restored = checkpoints.restore_checkpoint(self.checkpoint_dir, target=None)
            params = restored.params if hasattr(restored, 'params') else restored
            if params is None:
                print("[SafetyFilter] No checkpoint found; safety filter disabled.")
                return
            self.params = params

            # Tokenizer per Stage 1 data config
            from transformers import AutoTokenizer
            tok_name = cfg['data']['tokenizer']
            self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Warm JIT
            dummy = jnp.ones((1, cfg['data']['max_length']), dtype=jnp.int32)
            _ = self.model.apply(self.params, dummy, training=False)

            self.ready = True
            print("[SafetyFilter] Loaded Stage 1 safety classifier successfully.")
        except Exception as e:
            print(f"[SafetyFilter] Failed to load safety classifier: {e}")
            self.ready = False

    def score_text(self, text: str) -> float:
        """
        Return a scalar in [0,1] representing safety (1=safe). If unavailable, return 1.0.
        We map multi-label outputs to a safety score by 1 - max(category_prob).
        """
        if not self.ready:
            return 1.0
        try:
            enc = self.tokenizer(
                [text], truncation=True, padding='max_length',
                max_length=self.tokenizer.model_max_length if self.tokenizer.model_max_length else 512,
                return_tensors='np'
            )
            input_ids = jnp.array(enc['input_ids'])
            outputs = self.model.apply(self.params, input_ids, training=False)
            logits = outputs['logits'][0]
            probs = jax.nn.sigmoid(logits)
            max_risk = float(jnp.max(probs))
            safety = float(1.0 - max_risk)
            return safety
        except Exception as e:
            print(f"[SafetyFilter] Scoring failed, defaulting to safe: {e}")
            return 1.0

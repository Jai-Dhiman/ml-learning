import os
import sys
import yaml
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
            # Import JAX/Flax lazily and fail explicitly if unavailable
            try:
                # Prefer CPU JAX to minimize Colab GPU resource usage for safety scoring
                os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
                import jax  # type: ignore
                import jax.numpy as jnp  # type: ignore
                from flax.training import checkpoints  # type: ignore
            except Exception as ie:
                raise RuntimeError(
                    "SafetyFilter requires JAX/Flax to be installed. "
                    "In Colab, run training/eval with: \n"
                    "  uv run --with 'jax[cpu]==0.4.38' --with 'flax>=0.8.4,<0.9.0' --with 'optax>=0.2.2,<0.3.0' ..."
                ) from ie

            # Keep references for use in score_text
            self._jax = jax
            self._jnp = jnp

            with open(self.classifier_config_path, 'r') as f:
                cfg = yaml.safe_load(f)

            if create_model is None:
                raise RuntimeError("Stage 1 model code (safety-text-classifier) not importable. Ensure the repo path is correct.")

            self.model = create_model(cfg)

            # Resolve checkpoint directory robustly
            ckpt_dir = self.checkpoint_dir
            candidates = []

            # Direct inputs
            if ckpt_dir:
                candidates.append(ckpt_dir)
                candidates.append(os.path.join(ckpt_dir, 'checkpoints'))
                candidates.append(os.path.join(ckpt_dir, 'best'))
                candidates.append(os.path.join(ckpt_dir, 'best_model'))
                candidates.append(os.path.join(ckpt_dir, 'final'))

            # Environment override
            env_ckpt = os.environ.get('STAGE1_CKPT_PATH')
            if env_ckpt:
                candidates.append(env_ckpt)

            # Common alternates under repo root
            candidates.append(os.path.join(STAGE1_ROOT, 'checkpoints'))
            candidates.append(os.path.join(STAGE1_ROOT, 'checkpoints', 'best'))
            candidates.append(os.path.join(STAGE1_ROOT, 'checkpoints', 'best_model'))
            candidates.append(os.path.join(STAGE1_ROOT, 'checkpoints', 'final'))

            # Walk for any directory that contains Flax checkpoint files under likely roots
            def append_walk(root_dir: str):
                if not os.path.isdir(root_dir):
                    return
                for root, dirs, files in os.walk(root_dir):
                    names = set(files) | set(dirs)
                    if (
                        any(fn.startswith('checkpoint') for fn in names)
                        or any(fn.endswith('.msgpack') for fn in names)
                        or {'best', 'best_model', 'final'}.intersection(names)
                    ):
                        candidates.append(root)

            append_walk(STAGE1_ROOT)
            if ckpt_dir:
                append_walk(ckpt_dir)

            tried = []
            found = None
            for c in candidates:
                if c in tried:
                    continue
                tried.append(c)
                if os.path.isdir(c):
                    try:
                        restored = checkpoints.restore_checkpoint(c, target=None)
                        params = restored.params if hasattr(restored, 'params') else restored
                        if params is not None:
                            found = c
                            self.params = params
                            break
                    except Exception:
                        continue
            if found is None:
                raise RuntimeError(
                    "No Stage 1 checkpoint found. Tried paths: " + ", ".join(tried)
                )
            else:
                print(f"[SafetyFilter] Loaded checkpoint from: {found}")

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
            # Fail fast, do not silently disable safety in Stage 2
            raise

    def score_text(self, text: str) -> float:
        """
        Return a scalar in [0,1] representing safety (1=safe). If unavailable, return 1.0.
        We map multi-label outputs to a safety score by 1 - max(category_prob).
        """
        if not self.ready:
            raise RuntimeError("SafetyFilter not ready; ensure JAX/Flax and checkpoints are available.")
        try:
            enc = self.tokenizer(
                [text], truncation=True, padding='max_length',
                max_length=self.tokenizer.model_max_length if self.tokenizer.model_max_length else 512,
                return_tensors='np'
            )
            jnp = getattr(self, '_jnp', None)
            jax = getattr(self, '_jax', None)
            if jnp is None or jax is None:
                raise RuntimeError("Internal JAX refs missing; SafetyFilter initialization did not complete.")
            input_ids = jnp.array(enc['input_ids'])
            outputs = self.model.apply(self.params, input_ids, training=False)
            logits = outputs['logits'][0]
            probs = jax.nn.sigmoid(logits)
            max_risk = float(jnp.max(probs))
            safety = float(1.0 - max_risk)
            return safety
        except Exception as e:
            raise

"""
Model utilities for the Safety Text Classifier

Provides utilities for model loading, saving, checkpointing,
and parameter management.
"""

import jax
import jax.numpy as jnp
import flax
from flax.training import checkpoints, train_state
from flax import struct
import yaml
import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
import pickle
import numpy as np

from .transformer import SafetyTransformer, create_model, initialize_model

logger = logging.getLogger(__name__)


@struct.dataclass
class ModelMetadata:
    """Metadata for saved models."""

    model_name: str
    model_version: str
    timestamp: str
    config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_steps: int
    vocabulary_size: int
    safety_categories: Dict[int, str]


class ModelCheckpointManager:
    """
    Manages model checkpoints, loading,
    and saving for the Safety Text Classifier.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Safety categories mapping
        self.safety_categories = {
            0: "hate_speech",
            1: "self_harm",
            2: "dangerous_advice",
            3: "harassment",
        }

    def save_checkpoint(
        self,
        state: train_state.TrainState,
        step: int,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save a model checkpoint with metadata.

        Args:
            state: Training state to save
            step: Current training step
            config: Model configuration
            metrics: Performance metrics
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        try:
            # Save the checkpoint
            checkpoints.save_checkpoint(
                self.checkpoint_dir,
                state,
                step=step,
                keep=5,  # Keep 5 most recent checkpoints
            )

            # Save metadata
            metadata = ModelMetadata(
                model_name="safety_transformer",
                model_version="1.0.0",
                timestamp=str(np.datetime64("now")),
                config=config,
                performance_metrics=metrics or {},
                training_steps=step,
                vocabulary_size=config["model"]["vocab_size"],
                safety_categories=self.safety_categories,
            )

            # Create checkpoint path for metadata
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}"
            metadata_path = checkpoint_path.with_suffix(".metadata.json")

            with open(metadata_path, "w") as f:
                # Convert to serializable format
                metadata_dict = {
                    "model_name": metadata.model_name,
                    "model_version": metadata.model_version,
                    "timestamp": metadata.timestamp,
                    "config": metadata.config,
                    "performance_metrics": metadata.performance_metrics,
                    "training_steps": metadata.training_steps,
                    "vocabulary_size": metadata.vocabulary_size,
                    "safety_categories": metadata.safety_categories,
                }
                json.dump(metadata_dict, f, indent=2)

            # Save best checkpoint separately
            if is_best:
                best_path = self.checkpoint_dir / "best_model"
                checkpoints.save_checkpoint(best_path, state, step=step, keep=1)

                # Copy metadata for best model
                best_metadata_path = best_path / f"checkpoint_{step}.metadata.json"
                with open(best_metadata_path, "w") as f:
                    json.dump(metadata_dict, f, indent=2)

                logger.info(f"Saved best model checkpoint at step {step}")

            logger.info(f"Saved checkpoint at step {step}")
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Checkpoint save failed: {e}")

    def load_checkpoint(
        self, checkpoint_path: Union[str, Path], config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], ModelMetadata]:
        """
        Load a model checkpoint with metadata.

        Args:
            checkpoint_path: Path to checkpoint directory
            config: Optional config override

        Returns:
            Tuple of (model_params, metadata)
        """
        checkpoint_path = Path(checkpoint_path)

        try:
            # Load checkpoint
            restored_state = checkpoints.restore_checkpoint(
                checkpoint_path, target=None
            )

            # Extract parameters
            if hasattr(restored_state, "params"):
                params = restored_state.params
            else:
                params = restored_state

            # Load metadata
            metadata_files = list(
                checkpoint_path.parent.glob(f"{checkpoint_path.name}*.metadata.json")
            )
            if metadata_files:
                with open(metadata_files[0], "r") as f:
                    metadata_dict = json.load(f)

                metadata = ModelMetadata(
                    model_name=metadata_dict["model_name"],
                    model_version=metadata_dict["model_version"],
                    timestamp=metadata_dict["timestamp"],
                    config=metadata_dict["config"],
                    performance_metrics=metadata_dict["performance_metrics"],
                    training_steps=metadata_dict["training_steps"],
                    vocabulary_size=metadata_dict["vocabulary_size"],
                    safety_categories=metadata_dict["safety_categories"],
                )
            else:
                # Create dummy metadata if not available
                logger.warning("No metadata found, creating default")
                metadata = ModelMetadata(
                    model_name="safety_transformer",
                    model_version="1.0.0",
                    timestamp=str(np.datetime64("now")),
                    config=config or {},
                    performance_metrics={},
                    training_steps=0,
                    vocabulary_size=(
                        config.get("model", {}).get("vocab_size", 32000)
                        if config
                        else 32000
                    ),
                    safety_categories=self.safety_categories,
                )

            logger.info(
                f"Loaded checkpoint with {count_parameters(params):,} parameters"
            )
            return params, metadata

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint load failed: {e}")

    def list_checkpoints(self) -> List[str]:
        """
        List available checkpoints.

        Returns:
            List of checkpoint paths
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*"))
        return sorted([str(f) for f in checkpoint_files])


def count_parameters(params: Dict[str, Any]) -> int:
    """
    Count the total number of parameters in the model.

    Args:
        params: Model parameters (JAX pytree)

    Returns:
        Total number of parameters
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate model configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError if invalid
    """
    model_config = config.get("model", {})

    required_fields = [
        "vocab_size",
        "embedding_dim",
        "num_layers",
        "num_heads",
        "feedforward_dim",
        "max_sequence_length",
        "num_classes",
    ]

    for field in required_fields:
        if field not in model_config:
            raise ValueError(f"Missing required model config field: {field}")
        if not isinstance(model_config[field], int) or model_config[field] <= 0:
            raise ValueError(f"Model config field {field} must be positive integer")

    # Check that embedding_dim is divisible by num_heads
    if model_config["embedding_dim"] % model_config["num_heads"] != 0:
        raise ValueError("embedding_dim must be divisible by num_heads")

    # Check reasonable ranges
    if (
        model_config.get("dropout_rate", 0.1) < 0
        or model_config.get("dropout_rate", 0.1) > 1
    ):
        raise ValueError("dropout_rate must be between 0 and 1")

    logger.info("Model configuration validation passed")
    return True


def get_model_info(params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get comprehensive model information.

    Args:
        params: Model parameters
        config: Model configuration

    Returns:
        Dictionary with model information
    """
    param_count = count_parameters(params)
    model_config = config["model"]

    return {
        "total_parameters": param_count,
        "model_size_mb": param_count * 4 / (1024 * 1024),  # Assuming float32
        "architecture": {
            "embedding_dim": model_config["embedding_dim"],
            "num_layers": model_config["num_layers"],
            "num_heads": model_config["num_heads"],
            "feedforward_dim": model_config["feedforward_dim"],
            "vocab_size": model_config["vocab_size"],
            "max_sequence_length": model_config["max_sequence_length"],
        },
        "safety_categories": {
            0: "hate_speech",
            1: "self_harm",
            2: "dangerous_advice",
            3: "harassment",
        },
    }


if __name__ == "__main__":
    # Test the utilities
    logging.basicConfig(level=logging.INFO)
    print("✅ Utils module loaded successfully!")

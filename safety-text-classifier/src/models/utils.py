"""
Model utilities for the Safety Text Classifier

Provides utilities for model loading, saving, checkpointing, and parameter management.
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
    Manages model checkpoints, loading, and saving for the Safety Text Classifier.
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
            0: 'hate_speech',
            1: 'self_harm', 
            2: 'dangerous_advice',
            3: 'harassment'
        }
    
    def save_checkpoint(
        self,
        state: train_state.TrainState,
        step: int,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
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
        # Create checkpoint subdirectory
        checkpoint_name = f"step_{step:06d}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        try:
            # Save the checkpoint
            checkpoints.save_checkpoint(
                self.checkpoint_dir,
                state,
                step=step,
                keep=5  # Keep 5 most recent checkpoints
            )
            
            # Save metadata
            metadata = ModelMetadata(
                model_name="safety_transformer",
                model_version="1.0.0",
                timestamp=str(np.datetime64('now')),
                config=config,
                performance_metrics=metrics or {},
                training_steps=step,
                vocabulary_size=config['model']['vocab_size'],
                safety_categories=self.safety_categories
            )
            
            metadata_path = checkpoint_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                # Convert to serializable format
                metadata_dict = {
                    'model_name': metadata.model_name,
                    'model_version': metadata.model_version,
                    'timestamp': metadata.timestamp,
                    'config': metadata.config,
                    'performance_metrics': metadata.performance_metrics,
                    'training_steps': metadata.training_steps,
                    'vocabulary_size': metadata.vocabulary_size,
                    'safety_categories': metadata.safety_categories
                }
                json.dump(metadata_dict, f, indent=2)
            
            # Save best checkpoint separately
            if is_best:
                best_path = self.checkpoint_dir / "best_model"
                checkpoints.save_checkpoint(
                    best_path,
                    state,
                    step=step,
                    keep=1
                )
                
                # Copy metadata for best model
                best_metadata_path = best_path / f"checkpoint_{step}.metadata.json"
                with open(best_metadata_path, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)
                
                logger.info(f"Saved best model checkpoint at step {step}")
            
            logger.info(f"Saved checkpoint at step {step}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Checkpoint save failed: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None
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
                checkpoint_path, 
                target=None
            )
            
            # Extract parameters
            if hasattr(restored_state, 'params'):
                params = restored_state.params
            else:
                params = restored_state
            
            # Load metadata
            metadata_files = list(checkpoint_path.glob("*.metadata.json"))
            if metadata_files:
                with open(metadata_files[0], 'r') as f:
                    metadata_dict = json.load(f)
                
                metadata = ModelMetadata(
                    model_name=metadata_dict['model_name'],
                    model_version=metadata_dict['model_version'],
                    timestamp=metadata_dict['timestamp'],
                    config=metadata_dict['config'],
                    performance_metrics=metadata_dict['performance_metrics'],
                    training_steps=metadata_dict['training_steps'],
                    vocabulary_size=metadata_dict['vocabulary_size'],
                    safety_categories=metadata_dict['safety_categories']
                )
            else:
                # Create dummy metadata if not available
                logger.warning("No metadata found, creating default")
                metadata = ModelMetadata(
                    model_name="safety_transformer",
                    model_version="1.0.0",
                    timestamp=str(np.datetime64('now')),
                    config=config or {},
                    performance_metrics={},
                    training_steps=0,
                    vocabulary_size=32000,
                    safety_categories=self.safety_categories
                )
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return params, metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint load failed: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints_info = []
        
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if checkpoint_dir.is_dir():
                metadata_files = list(checkpoint_dir.glob("*.metadata.json"))
                if metadata_files:
                    try:
                        with open(metadata_files[0], 'r') as f:
                            metadata = json.load(f)
                        
                        checkpoints_info.append({
                            'path': str(checkpoint_dir),
                            'step': metadata.get('training_steps', 0),
                            'timestamp': metadata.get('timestamp', ''),
                            'metrics': metadata.get('performance_metrics', {})
                        })
                    except Exception as e:
                        logger.warning(f"Could not read metadata for {checkpoint_dir}: {e}")
        
        # Sort by training steps
        checkpoints_info.sort(key=lambda x: x['step'], reverse=True)
        return checkpoints_info
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        best_path = self.checkpoint_dir / "best_model"
        if best_path.exists():
            return best_path
        return None


class ModelLoader:
    """
    Utility class for loading models for inference.
    """
    
    def __init__(self):
        self.checkpoint_manager = ModelCheckpointManager()
    
    @classmethod
    def load_model_for_inference(
        cls,
        checkpoint_path: Union[str, Path],
        config_path: Optional[str] = None
    ) -> Tuple[SafetyTransformer, Dict[str, Any], ModelMetadata]:
        """
        Load a model for inference.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to config file
            
        Returns:
            Tuple of (model, params, metadata)
        """
        loader = cls()
        
        # Load config if provided
        config = None
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Load checkpoint
        params, metadata = loader.checkpoint_manager.load_checkpoint(
            checkpoint_path, config
        )
        
        # Use config from metadata if not provided
        if config is None:
            config = metadata.config
        
        # Create model
        model = create_model(config)
        
        logger.info(f"Loaded model for inference: {metadata.model_name} v{metadata.model_version}")
        return model, params, metadata
    
    @classmethod
    def load_best_model(
        cls,
        checkpoint_dir: str = "checkpoints",
        config_path: Optional[str] = None
    ) -> Tuple[SafetyTransformer, Dict[str, Any], ModelMetadata]:
        """
        Load the best available model.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            config_path: Optional path to config file
            
        Returns:
            Tuple of (model, params, metadata)
        """
        loader = cls()
        loader.checkpoint_manager.checkpoint_dir = Path(checkpoint_dir)
        
        best_path = loader.checkpoint_manager.get_best_checkpoint_path()
        if best_path is None:
            raise FileNotFoundError("No best model checkpoint found")
        
        return cls.load_model_for_inference(best_path, config_path)


def count_parameters(params: Dict[str, Any]) -> int:
    """
    Count the total number of parameters in a model.
    
    Args:
        params: Model parameters
        
    Returns:
        Total parameter count
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def get_parameter_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of model parameters.
    
    Args:
        params: Model parameters
        
    Returns:
        Parameter summary dictionary
    """
    total_params = count_parameters(params)
    
    # Get parameter shapes by layer
    param_shapes = {}
    for key, value in flax.traverse_util.flatten_dict(params).items():
        layer_name = '/'.join(key[:-1]) if len(key) > 1 else 'root'
        param_name = key[-1]
        
        if layer_name not in param_shapes:
            param_shapes[layer_name] = {}
        
        param_shapes[layer_name][param_name] = {
            'shape': value.shape,
            'size': value.size,
            'dtype': str(value.dtype)
        }
    
    return {
        'total_parameters': total_params,
        'parameter_breakdown': param_shapes,
        'memory_estimate_mb': total_params * 4 / (1024 * 1024)  # Assume float32
    }


def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate model configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        True if valid, raises exception if invalid
    """
    required_fields = [
        'model.vocab_size',
        'model.embedding_dim', 
        'model.num_layers',
        'model.num_heads',
        'model.feedforward_dim',
        'model.max_sequence_length',
        'model.num_classes'
    ]
    
    def get_nested_value(d, key_path):
        keys = key_path.split('.')
        value = d
        for key in keys:
            if key not in value:
                raise ValueError(f"Missing required config field: {key_path}")
            value = value[key]
        return value
    
    # Check required fields
    for field in required_fields:
        get_nested_value(config, field)
    
    # Validate specific constraints
    model_config = config['model']
    
    if model_config['embedding_dim'] % model_config['num_heads'] != 0:
        raise ValueError("embedding_dim must be divisible by num_heads")
    
    if model_config['num_classes'] != 4:
        logger.warning(f"Expected 4 safety categories, got {model_config['num_classes']}")
    
    logger.info("Model configuration validation passed")
    return True


# Convenience functions
def save_model_checkpoint(*args, **kwargs):
    """Convenience function for saving checkpoints."""
    manager = ModelCheckpointManager()
    return manager.save_checkpoint(*args, **kwargs)


def load_model_checkpoint(*args, **kwargs):
    """Convenience function for loading checkpoints."""
    manager = ModelCheckpointManager()
    return manager.load_checkpoint(*args, **kwargs)


def load_best_model(*args, **kwargs):
    """Convenience function for loading the best model."""
    return ModelLoader.load_best_model(*args, **kwargs)


if __name__ == "__main__":
    # Test the utilities
    import tempfile
    
    logging.basicConfig(level=logging.INFO)
    
    # Test model creation and parameter counting
    with open("configs/base_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate config
    validate_model_config(config)
    
    # Create model and count parameters
    model = create_model(config)
    rng = jax.random.PRNGKey(42)
    params = initialize_model(model, rng)
    
    param_count = count_parameters(params)
    param_summary = get_parameter_summary(params)
    
    print(f"Model parameter count: {param_count:,}")
    print(f"Memory estimate: {param_summary['memory_estimate_mb']:.2f} MB")
    
    # Test checkpoint manager
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ModelCheckpointManager(temp_dir)
        
        # Create dummy training state
        from flax.training import train_state
        import optax
        
        tx = optax.adam(learning_rate=1e-4)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            state=state,
            step=1000,
            config=config,
            metrics={'accuracy': 0.85, 'loss': 0.3},
            is_best=True
        )
        
        print(f"Saved checkpoint to: {checkpoint_path}")
        
        # Load checkpoint
        loaded_params, metadata = manager.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint: {metadata.model_name} v{metadata.model_version}")
        print(f"Training steps: {metadata.training_steps}")
        print(f"Metrics: {metadata.performance_metrics}")
"""
Safety Text Classifier Training Pipeline

JAX/Flax training implementation with Weights & Biases logging,
checkpointing, and comprehensive evaluation for the Constitutional AI research project.
"""

import jax
import jax.numpy as jnp
import flax
from flax import struct
from flax.training import train_state, checkpoints
import optax
import wandb
import logging
import os
import yaml
from typing import Dict, Any, Tuple, Optional, Callable
from tqdm import tqdm
import numpy as np
from functools import partial

from models.transformer import SafetyTransformer, create_model, initialize_model
from data.dataset_loader import create_data_loaders

logger = logging.getLogger(__name__)


@struct.dataclass
class TrainState(train_state.TrainState):
    """Extended train state with additional metrics tracking."""
    epoch: int
    best_val_accuracy: float
    steps_since_improvement: int


class SafetyTrainer:
    """
    Training pipeline for the safety text classifier.
    
    Handles training loop, evaluation, checkpointing, and experiment tracking
    with Weights & Biases integration.
    """
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """
        Initialize the trainer with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.logging_config = self.config['logging']
        
        # Initialize model
        self.model = create_model(self.config)
        
        # Setup random keys
        self.rng = jax.random.PRNGKey(42)
        self.rng, init_rng = jax.random.split(self.rng)
        
        # Initialize model parameters
        self.params = initialize_model(self.model, init_rng)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize training state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=self.optimizer,
            epoch=0,
            best_val_accuracy=0.0,
            steps_since_improvement=0
        )
        
        # Setup wandb
        self._setup_wandb()
        
        # JIT compile training and evaluation functions
        self.train_step = jax.jit(self._train_step)
        self.eval_step = jax.jit(self._eval_step)
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create the optimizer with learning rate schedule."""
        # Learning rate schedule
        if self.training_config['schedule'] == 'cosine_with_warmup':
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.training_config['learning_rate'],
                warmup_steps=self.training_config['warmup_steps'],
                decay_steps=self.training_config['max_steps'],
                end_value=self.training_config['learning_rate'] * 
                          self.training_config['min_lr_ratio']
            )
        else:
            schedule = self.training_config['learning_rate']
        
        # Create optimizer
        optimizer = optax.adamw(
            learning_rate=schedule,
            weight_decay=self.training_config['weight_decay'],
            b1=self.training_config['beta1'],
            b2=self.training_config['beta2']
        )
        
        # Add gradient clipping
        if self.training_config.get('gradient_clip_norm'):
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.training_config['gradient_clip_norm']),
                optimizer
            )
        
        return optimizer
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.logging_config.get('wandb'):
            wandb_config = self.logging_config['wandb']
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config.get('entity'),
                tags=wandb_config.get('tags', []),
                config=self.config
            )
        else:
            logger.warning("Wandb not configured, skipping initialization")
    
    def _compute_loss(
        self, 
        params: Dict[str, Any], 
        batch: Dict[str, jnp.ndarray],
        rng_key: Optional[jnp.ndarray] = None,
        training: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute loss and metrics for a batch.
        
        Args:
            params: Model parameters
            batch: Batch of data
            rng_key: RNG key for dropout (required if training=True)
            training: Whether in training mode
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Use provided RNG for dropout if training
        if training and rng_key is not None:
            rngs = {'dropout': rng_key}
        else:
            rngs = None
            
        # Forward pass
        outputs = self.model.apply(
            params, 
            batch['input_ids'], 
            training=training,
            rngs=rngs
        )
        logits = outputs['logits']
        
        # Multi-label classification loss (binary cross-entropy)
        labels = batch['labels'].astype(jnp.float32)
        loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
        
        # Compute predictions and metrics
        predictions = jax.nn.sigmoid(logits)
        predicted_labels = (predictions > 0.5).astype(jnp.int32)
        
        # Accuracy (exact match for multi-label)
        accuracy = jnp.mean(jnp.all(predicted_labels == labels, axis=-1))
        
        # Per-class accuracy
        per_class_accuracy = jnp.mean(predicted_labels == labels, axis=0)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'per_class_accuracy': per_class_accuracy,
            'predictions': predictions,
            'predicted_labels': predicted_labels
        }
        
        return loss, metrics
    
    def _train_step(
        self, 
        state: TrainState, 
        batch: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray
    ) -> Tuple[TrainState, Dict[str, Any]]:
        """
        Single training step.
        
        Args:
            state: Current training state
            batch: Batch of training data
            rng_key: RNG key for dropout
            
        Returns:
            Tuple of (updated_state, metrics)
        """
        def loss_fn(params):
            return self._compute_loss(params, batch, rng_key, training=True)
        
        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # Update parameters
        state = state.apply_gradients(grads=grads)
        
        # Add gradient norm to metrics
        grad_norm = optax.global_norm(grads)
        metrics['grad_norm'] = grad_norm
        
        return state, metrics
    
    def _eval_step(
        self, 
        params: Dict[str, Any], 
        batch: Dict[str, jnp.ndarray]
    ) -> Dict[str, Any]:
        """
        Single evaluation step.
        
        Args:
            params: Model parameters
            batch: Batch of evaluation data
            
        Returns:
            Evaluation metrics
        """
        loss, metrics = self._compute_loss(params, batch, rng_key=None, training=False)
        return metrics
    
    def _create_batch(self, dataset, batch_size: int, rng_key):
        """Create batches from dataset."""
        dataset_size = len(dataset)
        indices = jax.random.permutation(rng_key, dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) < batch_size and i > 0:
                break  # Skip incomplete last batch
            
            batch = {
                'input_ids': jnp.array([dataset[int(idx)]['input_ids'] for idx in batch_indices]),
                'labels': jnp.array([dataset[int(idx)]['labels'] for idx in batch_indices])
            }
            yield batch
    
    def _log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """Log metrics to wandb and console."""
        # Log to wandb
        if wandb.run:
            log_dict = {}
            for key, value in metrics.items():
                if isinstance(value, jnp.ndarray):
                    if value.ndim == 0:  # scalar
                        log_dict[f"{prefix}{key}"] = float(value)
                    elif value.ndim == 1:  # per-class metrics
                        for i, val in enumerate(value):
                            log_dict[f"{prefix}{key}_class_{i}"] = float(val)
                else:
                    log_dict[f"{prefix}{key}"] = value
            
            wandb.log(log_dict, step=step)
        
        # Log to console
        if prefix == "train/":
            logger.info(f"Step {step}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    def evaluate(self, dataset, step: int, prefix: str = "val/", eval_rng=None) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            step: Current step number
            prefix: Prefix for logging metrics
            eval_rng: RNG key for evaluation (if None, creates new one)
            
        Returns:
            Dictionary of evaluation metrics
        """
        eval_metrics = []
        batch_size = self.training_config['batch_size']
        
        # Create evaluation batches
        if eval_rng is None:
            self.rng, eval_rng = jax.random.split(self.rng)
        
        for batch in self._create_batch(dataset, batch_size, eval_rng):
            metrics = self.eval_step(self.state.params, batch)
            eval_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key in eval_metrics[0].keys():
            if key in ['predictions', 'predicted_labels']:
                continue  # Skip these for aggregation
            
            values = [m[key] for m in eval_metrics]
            aggregated_metrics[key] = jnp.mean(jnp.array(values), axis=0)
        
        # Log metrics
        self._log_metrics(aggregated_metrics, step, prefix)
        
        return aggregated_metrics
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.abspath(self.config['paths']['checkpoint_dir'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save regular checkpoint
        checkpoints.save_checkpoint(
            checkpoint_dir,
            self.state,
            step=step,
            keep=3,
            overwrite=True
        )
        
        # Save best checkpoint separately
        if is_best:
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model')
            checkpoints.save_checkpoint(
                best_checkpoint_path,
                self.state,
                step=step,
                keep=1,
                overwrite=True
            )
            logger.info(f"Saved best model checkpoint at step {step}")
    
    def train(self):
        """
        Main training loop.
        """
        logger.info("Starting training...")
        
        # Load datasets  
        train_dataset, val_dataset, test_dataset = create_data_loaders(
            config_path=self.config_path
        )
        
        batch_size = self.training_config['batch_size']
        max_steps = self.training_config['max_steps']
        eval_every = self.training_config['eval_every']
        save_every = self.training_config['save_every']
        
        step = 0
        
        # Training loop
        while step < max_steps:
            # Create training batches for this epoch
            self.rng, epoch_rng = jax.random.split(self.rng)
            
            epoch_metrics = []
            for batch in self._create_batch(train_dataset, batch_size, epoch_rng):
                # Get RNG key for this training step
                self.rng, step_rng = jax.random.split(self.rng)
                
                # Training step
                self.state, train_metrics = self.train_step(self.state, batch, step_rng)
                epoch_metrics.append(train_metrics)
                step += 1
                
                # Log training metrics
                if step % 100 == 0:
                    avg_metrics = {}
                    for key in train_metrics.keys():
                        if key not in ['predictions', 'predicted_labels']:
                            values = [m[key] for m in epoch_metrics[-100:]]
                            avg_metrics[key] = jnp.mean(jnp.array(values), axis=0)
                    
                    self._log_metrics(avg_metrics, step, "train/")
                
                # Evaluation
                if step % eval_every == 0:
                    val_metrics = self.evaluate(val_dataset, step, "val/")
                    
                    # Check for improvement
                    val_accuracy = float(val_metrics['accuracy'])
                    if val_accuracy > self.state.best_val_accuracy:
                        self.state = self.state.replace(
                            best_val_accuracy=val_accuracy,
                            steps_since_improvement=0
                        )
                        self.save_checkpoint(step, is_best=True)
                    else:
                        self.state = self.state.replace(
                            steps_since_improvement=self.state.steps_since_improvement + 1
                        )
                
                # Save checkpoint
                if step % save_every == 0:
                    self.save_checkpoint(step)
                
                if step >= max_steps:
                    break
            
            # Update epoch
            self.state = self.state.replace(epoch=self.state.epoch + 1)
        
        # Final evaluation
        logger.info("Training completed. Running final evaluation...")
        test_metrics = self.evaluate(test_dataset, step, "test/")
        
        logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(step)
        
        if wandb.run:
            wandb.finish()


def train_model(config_path: str = "configs/base_config.yaml"):
    """
    Convenience function to train the safety text classifier.
    
    Args:
        config_path: Path to configuration file
    """
    trainer = SafetyTrainer(config_path)
    trainer.train()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train the model
    train_model()
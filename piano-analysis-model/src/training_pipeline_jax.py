"""
Complete training pipeline for piano CNN models in JAX/Flax
Handles data loading, training loops, and model evaluation
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state, checkpoints
from flax.training.early_stopping import EarlyStopping
import optax
import wandb
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, Iterator
import librosa
from dataclasses import dataclass

from piano_cnn_jax import get_piano_model, create_train_state, train_step, eval_step


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_architecture: str = "standard"  # standard, fusion, realtime
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Audio preprocessing
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    n_mfcc: int = 13
    
    # Model params
    base_filters: int = 64
    dropout_rate: float = 0.2
    
    # Paths
    data_path: str = "/content/piano_data"  # Colab path
    checkpoint_path: str = "/content/checkpoints"
    results_path: str = "/content/results"


class PianoDataLoader:
    """Data loader for piano audio and PercePiano labels"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.audio_files = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """Load audio files and corresponding labels"""
        data_path = Path(self.config.data_path)
        
        # Load PercePiano labels
        labels_file = data_path / "labels" / "label_2round_mean_reg_19_with0_rm_highstd0.json"
        with open(labels_file) as f:
            self.percepiano_labels = json.load(f)
        
        # Find corresponding audio files
        audio_dir = data_path / "audio"
        for label_key, ratings in self.percepiano_labels.items():
            audio_file = audio_dir / f"{label_key}.wav"
            if audio_file.exists():
                self.audio_files.append(str(audio_file))
                self.labels.append(np.array(ratings))
        
        print(f"Loaded {len(self.audio_files)} audio files with labels")
    
    def _extract_spectrograms(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Extract multiple spectrogram representations"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        # Chromagram
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        # Normalize and reshape for CNN input
        spectrograms = {
            'mel': self._normalize_spectrogram(mel_spec),
            'mfcc': self._normalize_spectrogram(mfcc),
            'chroma': self._normalize_spectrogram(chroma)
        }
        
        return spectrograms
    
    def _normalize_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """Normalize spectrogram to [0,1] and add channel dimension"""
        # Min-max normalization
        spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        
        # Transpose to (time, frequency) and add channel dimension
        return spec_norm.T[..., np.newaxis]
    
    def create_dataset_splits(self) -> Tuple[Dict, Dict, Dict]:
        """Create train/val/test splits"""
        n_samples = len(self.audio_files)
        indices = np.random.permutation(n_samples)
        
        # Split indices
        test_size = int(n_samples * self.config.test_split)
        val_size = int(n_samples * self.config.val_split)
        
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        train_indices = indices[test_size + val_size:]
        
        # Create datasets
        datasets = {}
        for split_name, split_indices in [
            ("train", train_indices),
            ("val", val_indices), 
            ("test", test_indices)
        ]:
            spectrograms = []
            labels = []
            
            for idx in split_indices:
                specs = self._extract_spectrograms(self.audio_files[idx])
                spectrograms.append(specs)
                labels.append(self.labels[idx])
            
            datasets[split_name] = {
                'spectrograms': spectrograms,
                'labels': np.array(labels)
            }
        
        return datasets["train"], datasets["val"], datasets["test"]


class PianoTrainer:
    """Main training class for piano CNN models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_loader = PianoDataLoader(config)
        
        # Create datasets
        self.train_data, self.val_data, self.test_data = self.data_loader.create_dataset_splits()
        
        # Initialize model and training state
        self.model = get_piano_model(
            architecture=config.model_architecture,
            num_classes=19,
            base_filters=config.base_filters,
            dropout_rate=config.dropout_rate
        )
        
        # Determine input shape based on architecture
        sample_spec = self.train_data['spectrograms'][0]['mel']
        input_shape = (config.batch_size, *sample_spec.shape)
        
        self.state = create_train_state(
            self.model, config.learning_rate, input_shape
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(min_delta=1e-4, patience=config.early_stopping_patience)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup wandb logging"""
        wandb.init(
            project="piano-performance-analysis",
            config=self.config.__dict__,
            name=f"piano-cnn-{self.config.model_architecture}"
        )
    
    def _create_batch_iterator(self, dataset: Dict, batch_size: int) -> Iterator[Tuple]:
        """Create batched data iterator"""
        spectrograms = dataset['spectrograms']
        labels = dataset['labels']
        n_samples = len(spectrograms)
        
        indices = np.random.permutation(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract mel spectrograms for standard model
            batch_mel = np.stack([
                spectrograms[i]['mel'] for i in batch_indices
            ])
            batch_labels = labels[batch_indices]
            
            yield jnp.array(batch_mel), jnp.array(batch_labels)
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        epoch_losses = []
        rng = jax.random.PRNGKey(epoch)
        
        # Training loop
        for batch_mel, batch_labels in self._create_batch_iterator(
            self.train_data, self.config.batch_size
        ):
            rng, dropout_rng = jax.random.split(rng)
            
            self.state, loss, predictions = train_step(
                self.state, batch_mel, batch_labels, dropout_rng
            )
            
            epoch_losses.append(float(loss))
        
        return {
            'train_loss': np.mean(epoch_losses),
            'epoch': epoch
        }
    
    def evaluate(self, dataset: Dict) -> Dict:
        """Evaluate model on dataset"""
        losses = []
        all_correlations = []
        
        for batch_mel, batch_labels in self._create_batch_iterator(
            dataset, self.config.batch_size
        ):
            loss, correlations, predictions = eval_step(
                self.state, batch_mel, batch_labels
            )
            
            losses.append(float(loss))
            all_correlations.append(correlations)
        
        # Average correlations across batches
        mean_correlations = np.mean(np.stack(all_correlations), axis=0)
        
        return {
            'loss': np.mean(losses),
            'correlations': mean_correlations,
            'avg_correlation': np.mean(mean_correlations)
        }
    
    def train(self):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.evaluate(self.val_data)
            
            # Logging
            wandb.log({
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                'epoch': epoch
            })
            
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Corr: {val_metrics['avg_correlation']:.3f}")
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, is_best=True)
            
            # Early stopping
            early_stop = self.early_stopping.update(val_metrics['loss'])
            if early_stop.should_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation
        test_metrics = self.evaluate(self.test_data)
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Correlation: {test_metrics['avg_correlation']:.3f}")
        
        return test_metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_path)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoints.save_checkpoint(
            ckpt_dir=checkpoint_dir,
            target=self.state,
            step=epoch,
            overwrite=True,
            keep=3
        )
        
        if is_best:
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir / "best",
                target=self.state,
                step=epoch,
                overwrite=True
            )


def main():
    """Main training script"""
    config = TrainingConfig()
    trainer = PianoTrainer(config)
    
    print("Starting piano CNN training...")
    print(f"Architecture: {config.model_architecture}")
    print(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(trainer.state.params))}")
    
    trainer.train()


if __name__ == "__main__":
    main()
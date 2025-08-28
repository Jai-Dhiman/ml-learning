"""
Google Colab setup script for piano analysis CNN training
Handles data upload, environment setup, and training execution
"""

import os
import subprocess
import json
import shutil
from pathlib import Path
import zipfile
import requests
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np


class ColabPianoSetup:
    """Setup class for running piano analysis in Google Colab"""
    
    def __init__(self):
        self.base_path = Path("/content")
        self.project_path = self.base_path / "piano-analysis"
        self.data_path = self.base_path / "piano_data"
        self.results_path = self.base_path / "results"
        
    def setup_environment(self):
        """Install required packages and setup directories"""
        print("🔧 Setting up environment...")
        
        # Install JAX/Flax and other requirements
        packages = [
            "jax[cuda]",  # GPU-enabled JAX
            "flax",
            "optax",
            "librosa",
            "wandb",
            "matplotlib",
            "seaborn",
            "soundfile"
        ]
        
        for package in packages:
            print(f"Installing {package}...")
            subprocess.run(["pip", "install", package], 
                         capture_output=True, check=True)
        
        # Create directories
        self.project_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        
        print("✅ Environment setup complete!")
    
    def upload_project_files(self):
        """Upload and extract project files"""
        print("📁 Upload your project files...")
        print("Please upload your project zip file using the file browser")
        
        from google.colab import files
        uploaded = files.upload()
        
        # Extract uploaded files
        for filename, content in uploaded.items():
            if filename.endswith('.zip'):
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(self.project_path)
                print(f"✅ Extracted {filename}")
            else:
                with open(self.project_path / filename, 'wb') as f:
                    f.write(content)
        
        print("✅ Project files uploaded!")
    
    def setup_percepiano_data(self):
        """Setup PercePiano dataset structure"""
        print("🎵 Setting up PercePiano data...")
        
        # Create data structure
        (self.data_path / "audio").mkdir(exist_ok=True)
        (self.data_path / "labels").mkdir(exist_ok=True)
        
        print("Upload PercePiano files:")
        print("1. Audio files (.wav) go in /content/piano_data/audio/")
        print("2. Label files (.json) go in /content/piano_data/labels/")
        
        from google.colab import files
        
        # Upload audio files
        print("\n📼 Upload audio files (.wav):")
        audio_files = files.upload()
        for filename, content in audio_files.items():
            if filename.endswith('.wav'):
                with open(self.data_path / "audio" / filename, 'wb') as f:
                    f.write(content)
        
        # Upload label files  
        print("\n🏷️ Upload label files (.json):")
        label_files = files.upload()
        for filename, content in label_files.items():
            if filename.endswith('.json'):
                with open(self.data_path / "labels" / filename, 'wb') as f:
                    f.write(content)
        
        # Verify data structure
        audio_count = len(list((self.data_path / "audio").glob("*.wav")))
        label_count = len(list((self.data_path / "labels").glob("*.json")))
        
        print(f"✅ Data setup complete!")
        print(f"   Audio files: {audio_count}")
        print(f"   Label files: {label_count}")
        
        return audio_count > 0 and label_count > 0
    
    def verify_gpu(self):
        """Verify GPU availability"""
        try:
            import jax
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.device_kind == 'gpu']
            
            if gpu_devices:
                print(f"✅ GPU Available: {gpu_devices[0]}")
                return True
            else:
                print("⚠️ No GPU found. Enable GPU in Runtime > Change runtime type")
                return False
        except ImportError:
            print("❌ JAX not installed properly")
            return False
    
    def create_training_notebook(self):
        """Create a complete training notebook"""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Piano Performance Analysis with CNNs\n",
                              "Training piano analysis models using JAX/Flax"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Import libraries\n",
                        "import jax\n",
                        "import jax.numpy as jnp\n",
                        "import flax.linen as nn\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import librosa\n",
                        "import wandb\n",
                        "\n",
                        "# Project imports\n",
                        "import sys\n",
                        "sys.path.append('/content/piano-analysis/src')\n",
                        "from piano_cnn_jax import get_piano_model\n",
                        "from training_pipeline_jax import PianoTrainer, TrainingConfig\n",
                        "\n",
                        "print(f'JAX devices: {jax.devices()}')"
                    ]
                },
                {
                    "cell_type": "code", 
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Configure training\n",
                        "config = TrainingConfig(\n",
                        "    model_architecture='standard',  # or 'fusion', 'realtime'\n",
                        "    learning_rate=1e-3,\n",
                        "    batch_size=16,  # Adjust based on GPU memory\n",
                        "    epochs=50,\n",
                        "    data_path='/content/piano_data'\n",
                        ")\n",
                        "\n",
                        "print('Training configuration:')\n",
                        "for key, value in config.__dict__.items():\n",
                        "    print(f'  {key}: {value}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Initialize trainer and start training\n",
                        "trainer = PianoTrainer(config)\n",
                        "\n",
                        "print(f'Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(trainer.state.params))}')\n",
                        "print(f'Training samples: {len(trainer.train_data[\"labels\"])}')\n",
                        "print(f'Validation samples: {len(trainer.val_data[\"labels\"])}')\n",
                        "\n",
                        "# Start training\n",
                        "test_results = trainer.train()"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }
        
        notebook_path = self.project_path / "piano_cnn_training.ipynb"
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        print(f"✅ Created training notebook: {notebook_path}")
    
    def quick_model_test(self):
        """Quick test of model architecture"""
        print("🧪 Testing model architecture...")
        
        # Add project to path
        import sys
        sys.path.append(str(self.project_path / "src"))
        
        try:
            from piano_cnn_jax import get_piano_model
            
            # Test different architectures
            architectures = ["standard", "fusion", "realtime"]
            
            for arch in architectures:
                model = get_piano_model(arch, num_classes=19)
                
                # Test forward pass
                rng = jax.random.PRNGKey(0)
                dummy_input = jax.random.normal(rng, (2, 128, 128, 1))
                
                params = model.init(rng, dummy_input, training=False)
                output = model.apply(params, dummy_input, training=False)
                
                param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
                
                print(f"✅ {arch:10} | Params: {param_count:7,} | Output: {output.shape}")
        
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False
        
        return True
    
    def run_complete_setup(self):
        """Run complete Colab setup"""
        print("🚀 Starting complete Piano Analysis setup for Google Colab\n")
        
        # Step 1: Environment
        self.setup_environment()
        clear_output(wait=True)
        
        # Step 2: GPU check
        gpu_available = self.verify_gpu()
        if not gpu_available:
            return False
        
        # Step 3: Project files
        print("📁 Please upload your project files now...")
        self.upload_project_files()
        
        # Step 4: Data setup
        data_ready = self.setup_percepiano_data()
        if not data_ready:
            print("❌ Data setup failed")
            return False
        
        # Step 5: Model test
        model_works = self.quick_model_test()
        if not model_works:
            return False
        
        # Step 6: Create training notebook
        self.create_training_notebook()
        
        print("\n🎉 Setup Complete!")
        print("Next steps:")
        print("1. Open piano_cnn_training.ipynb")
        print("2. Run cells to start training")
        print("3. Monitor training in wandb dashboard")
        
        return True


# Colab-specific utilities
def download_sample_data():
    """Download sample PercePiano data for testing"""
    print("📥 Downloading sample data...")
    
    # This would download actual PercePiano samples
    # For now, create dummy data for testing
    data_path = Path("/content/piano_data")
    
    # Create dummy audio files and labels
    import librosa
    import json
    
    # Generate dummy audio
    dummy_audio = np.random.randn(22050 * 5)  # 5 seconds
    for i in range(5):
        librosa.output.write_wav(
            data_path / "audio" / f"dummy_{i}.wav",
            dummy_audio, 22050
        )
    
    # Generate dummy labels
    dummy_labels = {
        f"dummy_{i}": np.random.rand(19).tolist() 
        for i in range(5)
    }
    
    with open(data_path / "labels" / "dummy_labels.json", 'w') as f:
        json.dump(dummy_labels, f)
    
    print("✅ Sample data created")


def plot_training_results(results_path: str):
    """Plot training results"""
    results_file = Path(results_path) / "training_results.json"
    
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training loss
        axes[0, 0].plot(results['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        
        # Validation loss
        axes[0, 1].plot(results['val_loss']) 
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        
        # Correlations
        axes[1, 0].plot(results['val_correlations'])
        axes[1, 0].set_title('Validation Correlations')
        axes[1, 0].set_xlabel('Epoch')
        
        # Final test results
        test_corr = results.get('test_correlations', [])
        if test_corr:
            axes[1, 1].bar(range(len(test_corr)), test_corr)
            axes[1, 1].set_title('Test Correlations by Dimension')
            axes[1, 1].set_xlabel('Perceptual Dimension')
        
        plt.tight_layout()
        plt.show()


# Main execution for Colab
if __name__ == "__main__":
    setup = ColabPianoSetup()
    setup.run_complete_setup()
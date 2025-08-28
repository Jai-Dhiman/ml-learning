"""
Google Colab Setup Script for Safety Text Classifier

This script handles the initial setup required to run the safety text classifier
on Google Colab, including GPU configuration, dependency installation, and 
environment setup.
"""

import os
import subprocess
import sys
from pathlib import Path
import logging

def setup_logging():
    """Setup logging for the setup process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_gpu_availability():
    """Check if GPU is available in Colab."""
    logger = logging.getLogger(__name__)
    
    try:
        import jax
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower()]
        
        if gpu_devices:
            logger.info(f"✅ GPU detected: {gpu_devices}")
            return True
        else:
            logger.warning("⚠️  No GPU detected. Performance will be limited.")
            logger.info("💡 Enable GPU in Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU")
            return False
    except ImportError:
        logger.error("❌ JAX not installed. Cannot check GPU availability.")
        return False

def install_dependencies():
    """Install required dependencies for Colab."""
    logger = logging.getLogger(__name__)
    logger.info("📦 Installing dependencies...")
    
    # JAX with CUDA support for Colab
    jax_install_cmd = [
        sys.executable, "-m", "pip", "install", 
        "jax[cuda12]", "-f", 
        "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    ]
    
    try:
        subprocess.run(jax_install_cmd, check=True, capture_output=True, text=True)
        logger.info("✅ JAX with CUDA support installed")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install JAX with CUDA: {e}")
        logger.info("🔄 Falling back to CPU version...")
        subprocess.run([sys.executable, "-m", "pip", "install", "jax"], check=True)
    
    # Install other requirements
    requirements = [
        "flax==0.8.4",
        "optax==0.2.2",
        "datasets==2.19.1", 
        "transformers==4.40.2",
        "sentence-transformers==2.7.0",
        "wandb==0.21.1",
        "numpy<2.0.0,>=1.24.4",
        "pandas==2.0.3",
        "matplotlib==3.8.4",
        "seaborn==0.13.2",
        "scikit-learn==1.4.2",
        "tqdm==4.66.4",
        "pyyaml==6.0.1"
    ]
    
    for req in requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], 
                         check=True, capture_output=True, text=True)
            logger.info(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  Failed to install {req}: {e}")

def setup_wandb():
    """Setup Weights & Biases for experiment tracking."""
    logger = logging.getLogger(__name__)
    logger.info("🔗 Setting up Weights & Biases...")
    
    try:
        import wandb
        # Check if already logged in
        if wandb.api.api_key:
            logger.info("✅ W&B already configured")
        else:
            logger.info("🔑 Please log in to W&B:")
            wandb.login()
    except ImportError:
        logger.error("❌ W&B not installed")

def setup_directory_structure():
    """Create necessary directories for Colab environment."""
    logger = logging.getLogger(__name__)
    logger.info("📁 Setting up directory structure...")
    
    directories = [
        "data",
        "checkpoints", 
        "logs/training",
        "configs",
        ".cache"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {dir_path}")

def download_project_files():
    """Download project files if running in a fresh Colab environment."""
    logger = logging.getLogger(__name__)
    
    # Check if we're in Colab and need to clone the repo
    if 'COLAB_GPU' in os.environ or '/content' in os.getcwd():
        logger.info("🌐 Detected Google Colab environment")
        
        # If project files don't exist, provide instructions
        if not Path('src').exists():
            logger.info("""
            📥 To use your project in Colab:
            
            Option 1 - Upload from Google Drive:
            1. Upload your project to Google Drive
            2. Mount Drive: from google.colab import drive; drive.mount('/content/drive')
            3. Copy project: !cp -r "/content/drive/MyDrive/safety-text-classifier" .
            
            Option 2 - Git Clone:
            1. Push your project to GitHub
            2. Clone: !git clone https://github.com/yourusername/safety-text-classifier.git
            3. Change directory: %cd safety-text-classifier
            
            Option 3 - Direct Upload:
            Use Colab's file upload feature to upload individual files
            """)

def optimize_for_colab():
    """Apply Colab-specific optimizations."""
    logger = logging.getLogger(__name__)
    logger.info("⚡ Applying Colab optimizations...")
    
    # Set environment variables for better performance
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    logger.info("✅ Environment optimizations applied")

def main():
    """Main setup function for Google Colab."""
    logger = setup_logging()
    logger.info("🚀 Setting up Safety Text Classifier for Google Colab...")
    
    try:
        # Install dependencies
        install_dependencies()
        
        # Check GPU
        gpu_available = check_gpu_availability()
        if gpu_available:
            logger.info("🎯 GPU training enabled!")
        
        # Setup directory structure
        setup_directory_structure()
        
        # Setup W&B
        setup_wandb()
        
        # Apply optimizations
        optimize_for_colab()
        
        # Download instructions
        download_project_files()
        
        logger.info("✅ Setup complete! Ready to train your safety classifier.")
        
        # Print next steps
        logger.info("""
        🎯 Next steps:
        1. Ensure GPU is enabled in Colab runtime
        2. Run the training notebook or script
        3. Monitor training in W&B dashboard
        """)
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()
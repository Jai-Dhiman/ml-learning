"""
Google Colab Setup Script for Safety Text Classifier

This script handles the initial setup required to run the safety text classifier
on Google Colab using uv for fast dependency management and proper compatibility.

Features:
- Uses uv for 10-100x faster package installation
- Proper numpy compatibility management
- GPU-enabled JAX installation
- Minimal, Colab-optimized dependencies
"""

import os
import subprocess
import sys
from pathlib import Path
import logging
import shutil


def setup_logging():
    """Setup logging for the setup process."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def install_uv():
    """Install uv package manager in Colab."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Installing uv package manager...")
    
    try:
        # Install uv using the official installer
        subprocess.run([
            "curl", "-LsSf", "https://astral.sh/uv/install.sh"
        ], check=True, stdout=subprocess.PIPE)
        
        # Add uv to PATH for this session
        uv_path = os.path.expanduser("~/.cargo/bin")
        if uv_path not in os.environ["PATH"]:
            os.environ["PATH"] = f"{uv_path}:{os.environ['PATH']}"
        
        # Test uv installation
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ uv installed: {result.stdout.strip()}")
            return True
        else:
            raise subprocess.CalledProcessError(result.returncode, "uv --version")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"⚠️  uv installation failed: {e}")
        logger.info("🔄 Falling back to pip...")
        return False


def check_gpu_availability():
    """Check if GPU is available in Colab."""
    logger = logging.getLogger(__name__)
    
    try:
        import jax
        
        devices = jax.devices()
        gpu_devices = [d for d in devices if "gpu" in str(d).lower() or "cuda" in str(d).lower()]
        
        if gpu_devices:
            logger.info(f"✅ GPU detected: {gpu_devices}")
            logger.info(f"JAX version: {jax.__version__}")
            logger.info(f"JAX backend: {jax.default_backend()}")
            return True
        else:
            logger.warning("⚠️  No GPU detected. Performance will be limited.")
            logger.info(
                "💡 Enable GPU in Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU"
            )
            return False
    except ImportError:
        logger.error("❌ JAX not installed. Cannot check GPU availability.")
        return False


def find_project_root():
    """Find the safety-text-classifier project root directory."""
    logger = logging.getLogger(__name__)
    current_dir = Path.cwd()
    
    # Common patterns to look for
    project_indicators = ["src", "configs", "requirements.txt", "requirements-colab.txt"]
    
    # First, check if we're already in the project directory
    if all((current_dir / indicator).exists() for indicator in project_indicators[:3]):
        logger.info(f"✅ Already in project directory: {current_dir}")
        return current_dir
    
    # Check if we're in ml-learning and need to go to safety-text-classifier
    safety_classifier_path = current_dir / "safety-text-classifier"
    if safety_classifier_path.exists() and all((safety_classifier_path / indicator).exists() for indicator in project_indicators[:3]):
        logger.info(f"✅ Found project at: {safety_classifier_path}")
        os.chdir(safety_classifier_path)
        return safety_classifier_path
    
    # Check parent directories (in case we're in a subdirectory)
    for parent in current_dir.parents:
        if parent.name == "safety-text-classifier" and all((parent / indicator).exists() for indicator in project_indicators[:3]):
            logger.info(f"✅ Found project at: {parent}")
            os.chdir(parent)
            return parent
            
        # Check if parent has safety-text-classifier subdirectory
        safety_path = parent / "safety-text-classifier"
        if safety_path.exists() and all((safety_path / indicator).exists() for indicator in project_indicators[:3]):
            logger.info(f"✅ Found project at: {safety_path}")
            os.chdir(safety_path)
            return safety_path
    
    logger.warning(f"⚠️  Project directory not found. Current directory: {current_dir}")
    logger.info("Please ensure you're in the safety-text-classifier directory or its parent.")
    return current_dir


def install_dependencies_with_uv():
    """Install dependencies using uv for fast, compatible installation."""
    logger = logging.getLogger(__name__)
    logger.info("📦 Installing dependencies with uv (10-100x faster!)...")
    
    # Find project root first
    project_root = find_project_root()
    
    # Check if requirements-colab.txt exists
    requirements_file = "requirements-colab.txt"
    if not Path(requirements_file).exists():
        logger.warning(f"⚠️  {requirements_file} not found at {Path.cwd()}, using inline requirements")
        
        # Create temporary requirements file
        requirements_content = """
# Core ML Framework - Fixed versions for binary compatibility
numpy==1.24.4
jax[cpu]==0.4.28
flax==0.8.4
optax==0.2.2

# Data and Text Processing
datasets==2.19.1
transformers==4.40.2
sentence-transformers==2.7.0
tokenizers==0.15.2
pandas==2.0.3

# Experiment Tracking
wandb==0.17.0

# Visualization
matplotlib==3.7.5
seaborn==0.12.2

# Evaluation and Metrics
scikit-learn==1.3.2
scipy==1.10.1

# Utilities
tqdm==4.66.4
pyyaml==6.0.1
"""
        
        with open(requirements_file, 'w') as f:
            f.write(requirements_content.strip())
        logger.info(f"✅ Created {requirements_file}")
    
    try:
        # Install base packages with uv
        logger.info("🔧 Installing base packages...")
        subprocess.run([
            "uv", "pip", "install", "-r", requirements_file
        ], check=True, capture_output=True, text=True)
        logger.info("✅ Base packages installed with uv")
        
        # Install GPU-enabled JAX separately
        logger.info("🎯 Installing GPU-enabled JAX...")
        subprocess.run([
            "uv", "pip", "install",
            "jax[cuda12]==0.4.28",
            "--find-links", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
            "--force-reinstall"
        ], check=True, capture_output=True, text=True)
        logger.info("✅ GPU JAX installed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ uv installation failed: {e}")
        return False


def install_dependencies_with_pip():
    """Fallback: Install dependencies using pip with proper version management."""
    logger = logging.getLogger(__name__)
    logger.info("📦 Installing dependencies with pip (fallback)...")
    
    # Install numpy first to avoid binary compatibility issues
    try:
        logger.info("🔧 Installing numpy 1.24.4...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "numpy==1.24.4", "--force-reinstall"
        ], check=True, capture_output=True, text=True)
        logger.info("✅ numpy installed")
        
        # Install GPU JAX
        logger.info("🎯 Installing GPU-enabled JAX...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "jax[cuda12]==0.4.28", "-f",
            "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
            "--force-reinstall"
        ], check=True, capture_output=True, text=True)
        logger.info("✅ GPU JAX installed")
        
        # Install other packages
        requirements = [
            "flax==0.8.4",
            "optax==0.2.2",
            "transformers==4.40.2",
            "datasets==2.19.1",
            "sentence-transformers==2.7.0",
            "tokenizers==0.15.2",
            "pandas==2.0.3",
            "wandb==0.17.0",
            "matplotlib==3.7.5",
            "seaborn==0.12.2",
            "scikit-learn==1.3.2",
            "scipy==1.10.1",
            "tqdm==4.66.4",
            "pyyaml==6.0.1",
        ]
        
        for req in requirements:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", req
                ], check=True, capture_output=True, text=True)
                logger.info(f"✅ Installed {req}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Failed to install {req}: {e}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ pip installation failed: {e}")
        return False


def install_dependencies():
    """Install dependencies using uv if available, otherwise fall back to pip."""
    logger = logging.getLogger(__name__)
    
    # Try to install uv first
    uv_available = install_uv()
    
    if uv_available:
        success = install_dependencies_with_uv()
        if success:
            logger.info("🚀 Dependencies installed successfully with uv!")
            return True
        else:
            logger.warning("⚠️  uv installation failed, trying pip...")
    
    # Fallback to pip
    logger.info("🔄 Using pip for package installation...")
    return install_dependencies_with_pip()


def restart_runtime_warning():
    """Warn user about runtime restart requirement."""
    logger = logging.getLogger(__name__)
    logger.warning("""
    ⚠️  IMPORTANT: You may need to RESTART the runtime after installation.
    
    In Colab: Runtime -> Restart Runtime
    
    Then re-run your import statements. This ensures all packages are properly loaded.
    """)


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

    directories = ["data", "checkpoints", "logs/training", "configs", ".cache"]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {dir_path}")


def download_project_files():
    """Download project files if running in a fresh Colab environment."""
    logger = logging.getLogger(__name__)

    # Check if we're in Colab and need to clone the repo
    if "COLAB_GPU" in os.environ or "/content" in os.getcwd():
        logger.info("🌐 Detected Google Colab environment")

        # If project files don't exist, provide instructions
        if not Path("src").exists():
            logger.info(
                """
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
            """
            )


def optimize_for_colab():
    """Apply Colab-specific optimizations."""
    logger = logging.getLogger(__name__)
    logger.info("⚡ Applying Colab optimizations...")

    # Set environment variables for better performance
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    logger.info("✅ Environment optimizations applied")


def main():
    """Main setup function for Google Colab."""
    logger = setup_logging()
    logger.info("🚀 Setting up Safety Text Classifier for Google Colab...")

    try:
        # Install dependencies
        install_dependencies()

        # Show restart warning
        restart_runtime_warning()

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
        logger.info(
            """
        🎯 Next steps:
        1. If you see import errors, restart runtime: Runtime -> Restart Runtime
        2. Re-run your import statements
        3. Ensure GPU is enabled in Colab runtime
        4. Run the training notebook or script
        5. Monitor training in W&B dashboard
        """
        )

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        raise


if __name__ == "__main__":
    main()

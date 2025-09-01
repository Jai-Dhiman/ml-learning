#!/bin/bash

# 🚀 Piano Analysis Model - UV Setup Script
# Sets up the development environment using uv package manager

set -e  # Exit on any error

echo "🎹 Piano Analysis Model - Environment Setup"
echo "==========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "✅ uv is already installed: $(uv --version)"
fi

# Initialize project if needed
if [ ! -f "pyproject.toml" ]; then
    echo "🔧 Initializing uv project..."
    uv init --no-readme
fi

# Install dependencies using uv
echo "📦 Installing dependencies with uv..."

# Core ML dependencies
uv add jax[cpu]
uv add flax
uv add optax

# Audio processing
uv add librosa
uv add soundfile

# Data science stack
uv add numpy
uv add pandas
uv add matplotlib
uv add seaborn
uv add scipy

# Development tools
uv add jupyter
uv add tqdm
uv add wandb

# Optional: Add GPU support if CUDA available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 CUDA detected - adding GPU support..."
    uv add jax[cuda12] --replace
fi

echo "✅ Environment setup complete!"
echo ""
echo "🚀 Usage:"
echo "   uv run python src/single_dimension_model.py    # Run timing model"
echo "   uv run python src/training_pipeline_jax.py     # Full training pipeline"
echo "   uv run jupyter lab                              # Start Jupyter"
echo ""
echo "📝 Project structure:"
echo "   📁 src/           - Source code"
echo "   📁 data/          - PercePiano dataset"
echo "   📁 results/       - Training outputs"
echo "   📄 pyproject.toml - UV project configuration"

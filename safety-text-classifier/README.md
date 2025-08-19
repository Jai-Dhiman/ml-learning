# Safety Text Classifier - Constitutional AI Research Project

**Stage 1 of 4**: Building safety evaluation foundation for Constitutional AI research using JAX/Flax on cloud-native Kubernetes infrastructure.

## Project Overview

This project implements a transformer-based safety text classifier that detects harmful content across multiple categories:
- Hate speech and harassment
- Self-harm instructions
- Dangerous advice and misinformation
- Toxic and offensive content

**Target Performance**: 85%+ accuracy with comprehensive fairness and robustness evaluation.

## Technical Stack

- **ML Framework**: JAX/Flax for educational transparency and performance
- **Infrastructure**: Kubernetes (GKE) with auto-scaling and GPU support
- **Monitoring**: Prometheus/Grafana for comprehensive observability
- **Interface**: Gradio for interactive demonstration
- **Experiment Tracking**: Weights & Biases

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py --config configs/base_config.yaml

# Run inference server
python src/serve.py --model-path checkpoints/best_model

# Launch demo interface
python src/demo.py
```

## Project Structure

```
safety-text-classifier/
├── src/                    # Core implementation
│   ├── models/            # JAX/Flax model architectures
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training loops and optimization
│   ├── evaluation/        # Metrics and fairness analysis
│   └── serving/           # Inference and deployment
├── configs/               # Training and model configurations
├── k8s/                   # Kubernetes deployment manifests
├── notebooks/             # Educational Jupyter notebooks
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation and tutorials
```

## Constitutional AI Research Context

This classifier serves as the foundation for the 4-stage Constitutional AI research pipeline:

1. **Stage 1 (Current)**: Safety Text Classifier - Build evaluation foundation
2. **Stage 2**: Helpful Response Fine-tuning - Learn supervised behavior shaping
3. **Stage 3**: Critique and Revision System - Implement constitutional feedback loops
4. **Stage 4**: Full Constitutional AI - Complete RLAIF implementation

## Development Workflow

1. **Local Development**: Use Docker for consistent environment
2. **Training**: GKE with GPU node pools for distributed training
3. **Monitoring**: Prometheus metrics with Grafana dashboards
4. **Deployment**: Kubernetes-native serving with auto-scaling

## Learning Objectives

- Master JAX/Flax functional programming for neural networks
- Understand safety evaluation methodologies for AI systems
- Implement cloud-native ML deployment patterns
- Develop comprehensive fairness and robustness testing frameworks

## Getting Started

See `docs/getting-started.md` for detailed setup instructions and tutorials.
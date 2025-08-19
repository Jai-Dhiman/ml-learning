# Getting Started with Safety Text Classifier

This guide will help you set up and run the Safety Text Classifier for the Constitutional AI research project.

## Prerequisites

- Python 3.10+
- Docker (for containerization)
- Google Cloud SDK (for GKE deployment)
- Git

## Quick Start

### 1. Clone and Setup Environment

```bash
cd safety-text-classifier
pip install -r requirements.txt
```

### 2. Basic Training

Train the model with default configuration:

```bash
python train.py
```

With custom configuration:

```bash
python train.py --config configs/custom_config.yaml
```

### 3. Launch Demo Interface

Start the interactive Gradio demo:

```bash
python demo_app.py
```

Access the interface at: http://localhost:7860

### 4. Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t safety-classifier .

# Run container
docker run -p 7860:7860 safety-classifier
```

## Project Structure

```
safety-text-classifier/
├── src/                    # Core implementation
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # JAX/Flax model architectures
│   ├── training/          # Training loops and optimization
│   ├── evaluation/        # Metrics and fairness analysis
│   └── demo.py            # Interactive demo interface
├── configs/               # Configuration files
├── k8s/                   # Kubernetes deployment manifests
├── docs/                  # Documentation
├── train.py              # Training script
├── demo_app.py           # Demo launch script
└── requirements.txt      # Python dependencies
```

## Configuration

The system uses YAML configuration files in the `configs/` directory. Key sections:

- **model**: Architecture parameters (layers, attention heads, etc.)
- **training**: Learning rates, batch sizes, schedules
- **data**: Tokenization, splits, augmentation
- **logging**: Wandb and monitoring settings

## Training Process

1. **Data Loading**: Loads HuggingFace datasets and synthetic data
2. **Model Initialization**: Creates JAX/Flax transformer model
3. **Training Loop**: Distributed training with gradient accumulation
4. **Evaluation**: Comprehensive metrics including fairness analysis
5. **Checkpointing**: Automatic model saving and recovery

## Evaluation Metrics

The system tracks comprehensive metrics:

- **Performance**: Accuracy, precision, recall, F1, AUC
- **Calibration**: Expected Calibration Error (ECE)
- **Fairness**: Demographic parity, equalized odds
- **Robustness**: Adversarial and paraphrase consistency

## Kubernetes Deployment

Deploy to GKE for production:

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml

# Setup auto-scaling
kubectl apply -f k8s/hpa.yaml
```

## Monitoring

The system includes comprehensive monitoring:

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Wandb**: Experiment tracking
- **Custom Metrics**: Safety-specific monitoring

## Next Steps

1. **Experiment**: Try different model architectures and hyperparameters
2. **Extend**: Add new safety categories or data sources
3. **Deploy**: Set up production deployment on GKE
4. **Research**: Use as foundation for Constitutional AI experiments

## Troubleshooting

### Common Issues

**Memory errors during training:**
- Reduce batch size in config
- Enable gradient accumulation
- Use CPU-only mode: `export JAX_PLATFORM_NAME=cpu`

**Slow inference:**
- Verify JIT compilation is working
- Check GPU availability
- Consider model quantization

**Demo interface not loading:**
- Ensure all dependencies are installed
- Check port 7860 is available
- Review error logs for missing files

For more help, see the detailed documentation in the `docs/` directory.
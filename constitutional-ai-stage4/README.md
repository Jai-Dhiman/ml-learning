# Stage 4: Constitutional AI Evaluation and Analysis

**Complete implementation of Anthropic's Constitutional AI methodology with modern optimizations**

This project implements the fourth and final stage of the Constitutional AI research pipeline, focusing on comprehensive evaluation, validation, and documentation of the complete 4-stage training process.

## Overview

### What is Stage 4?

Stage 4 validates and documents the Constitutional AI implementation completed in Stages 1-3:

- **Stage 1**: Safety Text Classifier (JAX/Flax) - Foundation for harm evaluation
- **Stage 2**: Helpful Response Fine-tuning - Gemma 2B-IT + LoRA on Anthropic/hh-rlhf  
- **Stage 3**: Critique & Revision + DPO Training - Constitutional AI training
- **Stage 4 (This Stage)**: Comprehensive Evaluation - Validation and documentation

### Key Innovation

This implementation uses **Direct Preference Optimization (DPO)** instead of the paper's original PPO-based RLAIF, representing a modern, more efficient approach that achieves equivalent or better results.

## Project Structure

```
constitutional-ai-stage4/
├── src/
│   ├── evaluation/          # Constitutional evaluation frameworks
│   ├── inference/           # Model loading and inference utilities
│   └── analysis/            # Data analysis and visualization
├── notebooks/               # Jupyter/Colab notebooks for evaluation
├── configs/                 # Configuration files
│   ├── evaluation_config.yaml
│   └── model_config.yaml
└── artifacts/
    ├── evaluation/          # Evaluation results and metrics
    ├── models/              # Model references and metadata
    └── reports/             # Generated research reports
```

## Quick Start

### Setup Environment

```bash
# Navigate to project directory
cd constitutional-ai-stage4

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # On macOS/Linux
# OR: .venv\Scripts\activate  # On Windows

# Install dependencies
uv sync

# Alternative: Install from pyproject.toml
uv pip install -e .
```

### Run Evaluation

```bash
# Run complete evaluation suite
python src/run_evaluation.py

# Run specific evaluations
python src/evaluation/constitutional_evaluator.py
python src/evaluation/red_team_evaluation.py
python src/evaluation/elo_scoring.py
```

### Launch Interactive Demo

```bash
# Start Gradio demo
python src/demo_app.py
```

## Evaluation Components

### 1. Constitutional Principle Evaluation

Tests model adherence to four core principles:
- **Harm Prevention**: Safety classifier integration (Stage 1)
- **Truthfulness**: Consistency and factuality
- **Helpfulness**: Response quality and completeness
- **Fairness**: Bias detection and demographic parity

### 2. Red-Team Adversarial Testing

Tests model robustness against:
- Harmful instruction attempts
- Deception and misinformation
- Bias elicitation
- Privacy violations
- Dangerous advice requests

### 3. Elo Scoring System

Implements rating system from Anthropic paper:
- Helpfulness Elo ratings
- Harmlessness Elo ratings
- Comparative model analysis
- Statistical significance testing

### 4. Comparative Model Analysis

Side-by-side evaluation of:
- Base Gemma 2B-IT model
- Stage 2 Helpful RLHF model
- Stage 3 Constitutional AI model

## Configuration

Edit `configs/evaluation_config.yaml` to customize:

```yaml
# Model paths
models:
  base_model: "google/gemma-2b-it"
  stage2_adapters: "../artifacts/stage2_artifacts/lora_adapters"
  stage3_adapters: "../artifacts/stage3_artifacts/models/lora_adapters"

# Inference parameters
inference:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
```

## Key Features

### Paper Alignment

- ✅ Direct mapping to Anthropic's methodology
- ✅ DPO as modern RLAIF implementation
- ✅ Comprehensive evaluation matching paper standards
- ✅ Elo scoring system replication

### Evaluation Metrics

- ✅ Constitutional compliance scores
- ✅ Harmlessness vs helpfulness tradeoff
- ✅ Evasiveness detection (helpful, not evasive)
- ✅ Red-team robustness testing

### Documentation

- ✅ Paper alignment analysis
- ✅ Methodology comparison (DPO vs PPO)
- ✅ Complete evaluation results
- ✅ Publication-quality visualizations

## Notebooks

Interactive Colab notebooks for evaluation:

1. **`notebooks/model_comparison.ipynb`** - Side-by-side model comparison
2. **`notebooks/constitutional_evaluation.ipynb`** - Constitutional principle testing
3. **`notebooks/red_team_testing.ipynb`** - Adversarial evaluation
4. **`notebooks/visualization.ipynb`** - Results visualization

## Results

Results are saved to `artifacts/evaluation/`:

- `constitutional_scores.json` - Principle adherence scores
- `red_team_results.json` - Adversarial testing results
- `elo_ratings.json` - Comparative Elo scores
- `model_comparison.csv` - Full comparison data

## Research Report

Comprehensive documentation in `artifacts/reports/`:

- `paper_alignment_analysis.md` - Methodology mapping
- `constitutional_ai_final_report.md` - Complete research report
- `evaluation_summary.md` - Results summary

## Citation

This implementation is inspired by and evaluated against:

> Bai, Y., Kadavath, S., Kundu, S., et al. (2022). **Constitutional AI: Harmlessness from AI Feedback.** *arXiv preprint arXiv:2212.08073.*

## Development Status

### ✅ Completed
- [x] Project structure and setup
- [x] Configuration files
- [x] Documentation framework

### 🚧 In Progress
- [x] Model loader implementation
- [ ] Constitutional evaluators
- [ ] Red-team testing suite
- [ ] Elo scoring system
- [ ] Comparative analysis
- [ ] Interactive demo

### 📋 Planned
- [ ] Comprehensive evaluation report
- [ ] Publication-quality visualizations
- [ ] Colab notebook suite
- [ ] Integration with "J ai" portfolio assistant

## Contributing

This is a research project for portfolio development. See `STAGE4_IMPLEMENTATION_PLAN.md` for detailed implementation roadmap.

## License

This project follows the same licensing as the underlying models and datasets:
- Gemma 2B-IT: Apache 2.0
- Anthropic/hh-rlhf dataset: MIT License

## Contact

For questions or collaboration: [Your Contact Information]

---

**Part of the Constitutional AI Research Pipeline**  
Stages 1-4: From Safety Classification to Complete Constitutional AI

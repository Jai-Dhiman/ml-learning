# Stage 3: Constitutional AI Training

Critique-revision data generation + DPO training to align the model with constitutional principles.

## Colab Notebook

**File**: `notebooks/Stage3_Constitutional_Training.ipynb`

- Clones GitHub repo
- Loads Stage 2 adapters from Drive
- Has preflight test (1 pair)
- Generates 2500 critique-revision pairs
- DPO training: 2 epochs, beta=0.3
- Runtime: ~4-5 hours on T4 GPU
- Output: `artifacts/stage3_constitutional/lora_adapters/`

## Training Scripts

- `src/training/critique_revision.py` - Generates critique-revision pairs using constitutional principles
- `src/training/train_dpo.py` - DPO training on preference pairs

## Configuration

`configs/constitutional_principles.yaml` - 12 constitutional principles across 4 categories:
- Harm Prevention
- Helpfulness
- Truthfulness  
- Fairness

## Workflow

1. **Prerequisite**: Must have Stage 2 adapters (run `helpful-finetuning/` notebook first)
2. Upload `Stage3_Constitutional_Training.ipynb` to Google Colab
3. Run all cells (including preflight)
4. Adapters saved to Google Drive: `MyDrive/ml-learning/artifacts/stage3_constitutional/`

## After Training

Use Stage 4 evaluation (in `constitutional-ai-stage4/`) to compare:
- Baseline model
- Stage 2 model (helpful)
- Stage 3 model (constitutional)

Expected improvements:
- Aggregate: 72-75% (vs 68.7%)
- Harm Prevention: 70%+ (vs 63.6%)
- Helpfulness: Maintained at 60%+

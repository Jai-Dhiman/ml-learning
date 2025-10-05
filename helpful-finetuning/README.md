# Stage 2: Helpful Fine-Tuning

QLoRA fine-tuning on Anthropic/hh-rlhf dataset to make the model more helpful and conversational.

## Colab Notebook

**File**: `notebooks/Stage2_Helpful_Training.ipynb`

- Clones GitHub repo
- Has preflight test (1 sample)
- Full training: 10K samples, 3 epochs
- Runtime: ~2-3 hours on T4 GPU
- Output: `artifacts/stage2_helpful/lora_adapters/`

## Training Script

`src/training/train_qlora.py` - Handles the QLoRA training with configurable parameters

## Workflow

1. Upload `Stage2_Helpful_Training.ipynb` to Google Colab
2. Run all cells (including preflight)
3. Adapters saved to Google Drive: `MyDrive/ml-learning/artifacts/stage2_helpful/`
4. Download adapters or use them for Stage 3

## After Training

The Stage 2 adapters are required for Stage 3 constitutional training (in `critique-revision-system/`).

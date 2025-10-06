# Stage 3 Constitutional Training Notebook - Updates Summary

## Overview
Updated `Stage3_Constitutional_Training.ipynb` to use the newly trained Stage 2 artifacts and adopt proven patterns from the old working notebook.

## Key Changes Made

### 1. Repository Setup (Cell 1)
- **Changed**: Removed Google Drive mount dependency
- **Added**: Clean clone process (`rm -rf ml-learning` before cloning)
- **Updated**: Changed working directory to `/content/ml-learning` (repo root)
- **Benefit**: Ensures fresh repository state with all committed artifacts

### 2. Package Management (Cells 2 & 4)
- **Added Cell 2**: Install `uv` package manager
- **Replaced Cell 4**: Changed from `pip` to `uv` for dependency installation
- **Library Versions** (proven to work):
  - PyTorch with CUDA 11.8 support
  - transformers>=4.43.0
  - trl>=0.9.6
  - peft>=0.13.0
  - datasets>=2.19.0
  - accelerate>=0.28.0
  - Additional: sentencepiece, safetensors, einops, evaluate, protobuf<5
- **Benefit**: Faster installation (10-100x), better dependency resolution

### 3. HuggingFace Authentication (Cell 3)
- **Added**: New cell with secure token handling using `getpass`
- **Features**:
  - Clears existing tokens before authentication
  - Secure input (hidden token entry)
  - Fallback to login widget if getpass fails
  - Sets HF_TOKEN environment variable for bash cells
- **Benefit**: Secure authentication without exposing tokens, required for Gemma model access

### 4. Stage 2 Adapter Path Updates (Cells 5, 6, 7, 9)
- **Old Path**: `artifacts/stage2_helpful/lora_adapters`
- **New Path**: `artifacts/stage2_finetuning_artifacts/lora_adapters`
- **Updated In**:
  - Cell 5: Adapter verification
  - Cell 6: Preflight test (critique generation & DPO)
  - Cell 7: Full critique-revision pair generation
  - Cell 9: DPO training
- **Benefit**: References your actual newly trained Stage 2 model

### 5. Command Execution Updates (Cells 6, 7, 9)
- **Changed**: All Python commands now use `uv run python` instead of `!python`
- **Changed**: Converted cells to `%%bash` magic for better script execution
- **Added**: Error handling with `set -e` in bash cells
- **Benefit**: Proper virtual environment activation and error propagation

### 6. Training Configuration (Cell 9)
- **Updated**: DPO training parameters aligned with working configuration:
  - num_train_epochs: 1.0 (reduced from 2.0)
  - beta: 0.1 (reduced from 0.3)
  - Added: `--cpu-ref-model` flag for memory efficiency
  - Added: `--repo-root` parameter
- **Benefit**: Memory-efficient training suitable for T4 GPU (15GB VRAM)

### 7. Artifact Management (Cell 10)
- **Removed**: Google Drive save functionality
- **Added**: Local download with zip archive creation
- **Features**:
  - Creates compressed zip of all Stage 3 artifacts
  - Displays archive size and contents
  - Downloads directly to browser
  - Provides clear extraction instructions for local machine
- **Benefit**: No Google Drive dependency, easier local development workflow

### 8. Output Directory Structure
- **Old**: `artifacts/stage3_pairs/` and `artifacts/stage3_constitutional/`
- **New**: Consolidated to `artifacts/stage3_constitutional/` with subdirectories:
  - `pairs/pairs.jsonl` - Critique-revision pairs
  - `models/lora_adapters/` - DPO-trained model
  - `dpo_dataset/train.jsonl` - Preprocessed training data
  - `checkpoints/` - Training checkpoints
  - `logs/` - Training logs
  - `metrics.json` - Training metrics
- **Benefit**: Cleaner organization matching project structure

### 9. Pair Generation Configuration (Cell 7)
- **Maintained**: 2500 pairs target (`--num-examples 2500`)
- **Maintained**: Split strategy (`test[:1000]+train[:1500]`)
- **Updated**: Script path to use full relative path from repo root
- **Benefit**: Consistent with your requirements

## Cell Structure Summary

1. **Cell 1**: Clone Repository
2. **Cell 2**: Install uv Package Manager (NEW)
3. **Cell 3**: HuggingFace Authentication (NEW)
4. **Cell 4**: Install Dependencies with uv
5. **Cell 5**: Verify Stage 2 Adapters
6. **Cell 6**: Preflight Test (1 pair)
7. **Cell 7**: Generate Critique-Revision Pairs (2500 pairs)
8. **Cell 8**: Validate Pairs Quality
9. **Cell 9**: DPO Training
10. **Cell 10**: Download Stage 3 Artifacts

## Expected Runtime
- **Preflight Test**: ~2-3 minutes
- **Pair Generation**: ~45-60 minutes (2500 pairs)
- **DPO Training**: ~3-4 hours (T4 GPU, 1 epoch)
- **Total**: ~4-5 hours

## Prerequisites
1. Stage 2 artifacts committed to repository at `artifacts/stage2_finetuning_artifacts/lora_adapters/`
2. HuggingFace account with Gemma model access
3. Google Colab with T4 GPU runtime
4. HuggingFace API token ready to paste

## Local Extraction Instructions
After downloading the zip archive from Colab:

```bash
cd ~/Documents/ml-learning/artifacts
unzip ~/Downloads/stage3_constitutional_artifacts.zip
ls -lh ~/Documents/ml-learning/artifacts/stage3_constitutional/
```

## Testing Recommendations
1. Run Cell 6 (Preflight) first to validate setup
2. Check preflight output for any errors before proceeding
3. Monitor GPU memory during training (should stay under 15GB)
4. Validate pairs quality in Cell 8 before starting DPO training

## Benefits of This Approach
- ✅ No Google Drive dependencies
- ✅ Faster package installation with uv
- ✅ Secure HuggingFace authentication
- ✅ References correct Stage 2 artifacts
- ✅ Memory-efficient training configuration
- ✅ Clean local development workflow
- ✅ Matches proven working pattern from old notebook
- ✅ Better error handling with bash scripts

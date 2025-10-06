# Google Drive Setup for Stage 3 Training

## Overview

The large model files (1.2GB) have been removed from git to keep the repository manageable. You need to upload your Stage 2 artifacts to Google Drive so they can be downloaded during Stage 3 training in Colab.

## What to Upload

Upload this entire directory to your Google Drive:
```
/Users/jdhiman/Documents/ml-learning/artifacts/stage2_finetuning_artifacts/
```

This directory contains:
- `lora_adapters/` - LoRA adapters (75MB) - **Required for Stage 3**
- `final_model/` - Final model artifacts (108MB)
- `outputs/` - Training checkpoints (1.0GB)

**Size**: Total ~1.2GB

## Upload Steps

### Option 1: Via Google Drive Web Interface (Recommended)

1. Go to [drive.google.com](https://drive.google.com)
2. Create this folder structure in your Drive:
   ```
   My Drive/
   └── artifacts/
       └── stage2_finetuning_artifacts/
   ```
3. Upload the entire `stage2_finetuning_artifacts` folder into the `artifacts/` folder
4. Wait for upload to complete (~5-15 minutes depending on internet speed)

**Important**: The notebook expects the path `MyDrive/artifacts/stage2_finetuning_artifacts`

### Option 2: Via Google Drive Desktop App (Faster)

1. Install Google Drive desktop app if not already installed
2. Copy the folder to your Google Drive folder:
   ```bash
   # If Google Drive is synced to ~/Google Drive/
   mkdir -p ~/Google\ Drive/My\ Drive/artifacts/
   cp -r /Users/jdhiman/Documents/ml-learning/artifacts/stage2_finetuning_artifacts ~/Google\ Drive/My\ Drive/artifacts/
   ```
3. Wait for sync to complete

### Option 3: Using rclone (Advanced)

```bash
# Install rclone if needed: brew install rclone
# Configure: rclone config

# Upload
rclone copy /Users/jdhiman/Documents/ml-learning/artifacts/stage2_finetuning_artifacts \
  gdrive:artifacts/stage2_finetuning_artifacts \
  --progress
```

## Notebook Configuration

The notebook is **pre-configured** with the correct default path:

**Default path in Cell 5**:
```python
drive_artifacts_path = '/content/drive/MyDrive/artifacts/stage2_finetuning_artifacts'
```

✅ **If you uploaded to `MyDrive/artifacts/stage2_finetuning_artifacts`, you don't need to change anything!**

If you used a different path, update Cell 5:
1. Open `critique-revision-system/notebooks/Stage3_Constitutional_Training.ipynb` in Colab
2. Find **Cell 5: Download Stage 2 Artifacts from Google Drive**
3. Update the `drive_artifacts_path` variable to match your actual path

## Verification

After uploading, verify the files are accessible:

1. In Colab, run Cell 5 (Download Stage 2 Artifacts)
2. You should see:
   ```
   ======================================================================
   DOWNLOADING STAGE 2 ARTIFACTS FROM GOOGLE DRIVE
   ======================================================================
   
   1. Mounting Google Drive...
   
   2. Looking for Stage 2 artifacts in Google Drive...
      Expected location: /content/drive/MyDrive/artifacts/stage2_finetuning_artifacts
   
   3. Copying artifacts from Drive to Colab workspace...
   
   ✅ Stage 2 artifacts successfully copied!
      Source: /content/drive/MyDrive/artifacts/stage2_finetuning_artifacts
      Destination: /content/ml-learning/artifacts/stage2_finetuning_artifacts
      LoRA adapters: /content/ml-learning/artifacts/stage2_finetuning_artifacts/lora_adapters
      Files (X): ['adapter_config.json', 'adapter_model.safetensors', 'README.md', ...]
   
   ======================================================================
   ✅ ARTIFACTS READY
   ======================================================================
   ```

## What Files Are Required?

**Minimum Required** (for Stage 3 to work):
- `lora_adapters/adapter_config.json`
- `lora_adapters/adapter_model.safetensors`
- `lora_adapters/tokenizer_config.json`
- `lora_adapters/special_tokens_map.json`
- `lora_adapters/chat_template.jinja`

**Optional** (can be excluded to save space):
- `outputs/` directory (training checkpoints) - 1.0GB
- `final_model/` directory - 108MB

### Minimal Upload Option (Save Space)

If you want to save Google Drive space, upload just the `lora_adapters/` folder:

**Upload**: `artifacts/stage2_finetuning_artifacts/lora_adapters/` to `MyDrive/artifacts/lora_adapters/`

Then update Cell 5 in the notebook:
```python
# Change this line:
drive_artifacts_path = '/content/drive/MyDrive/artifacts/lora_adapters'

# And modify the copy section to:
target_dir = Path('/content/ml-learning/artifacts/stage2_finetuning_artifacts')
target_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(drive_path, target_dir / 'lora_adapters')
```

This reduces the upload from 1.2GB to just ~75MB!

## Troubleshooting

### Problem: "Path not found" error in Cell 5

**Solution**: 
1. Make sure Google Drive is mounted (you should see the mount authorization dialog)
2. In Colab, navigate to Files panel (left sidebar) and browse to verify your actual path
3. Update `drive_artifacts_path` variable in Cell 5 to match your actual path
4. Common paths:
   - `/content/drive/MyDrive/artifacts/stage2_finetuning_artifacts` (default)
   - `/content/drive/MyDrive/ml-learning/artifacts/stage2_finetuning_artifacts`
   - `/content/drive/MyDrive/stage2_finetuning_artifacts`

### Problem: Slow upload speed

**Solution**:
- Use Google Drive desktop app for faster syncing (recommended)
- Compress the folder first, then upload and extract in Colab:
  ```bash
  # Local machine:
  cd /Users/jdhiman/Documents/ml-learning/artifacts
  tar -czf stage2_artifacts.tar.gz stage2_finetuning_artifacts/
  # Upload the .tar.gz file to Drive, then in Colab:
  !tar -xzf /content/drive/MyDrive/stage2_artifacts.tar.gz -C /content/ml-learning/artifacts/
  ```

### Problem: Running out of Google Drive space

**Solution**:
- Upload only the `lora_adapters/` folder (~75MB instead of 1.2GB) - see "Minimal Upload Option" above
- Delete old/unused files from your Drive
- Upgrade to Google One for more storage (100GB for $1.99/month)

### Problem: "shutil.Error: Destination path already exists"

**Solution**: The cell automatically removes the old directory before copying. If you see this error, manually delete the target directory:
```python
# Add this before shutil.copytree in Cell 5:
if target_dir.exists():
    shutil.rmtree(target_dir)
```

## Quick Start Summary

1. ✅ Upload `/Users/jdhiman/Documents/ml-learning/artifacts/stage2_finetuning_artifacts/` to `MyDrive/artifacts/`
2. ✅ Open Stage 3 notebook in Colab
3. ✅ Run Cell 5 - it will automatically download from the default path
4. ✅ Continue with Stage 3 training (Cell 6+)

Your local `artifacts/` directory remains unchanged and is now properly ignored by git!

## Expected Google Drive Structure

After upload, your Google Drive should look like this:
```
My Drive/
└── artifacts/
    └── stage2_finetuning_artifacts/
        ├── lora_adapters/
        │   ├── adapter_config.json
        │   ├── adapter_model.safetensors  (75MB)
        │   ├── tokenizer_config.json
        │   ├── special_tokens_map.json
        │   ├── chat_template.jinja
        │   └── README.md
        ├── final_model/  (optional)
        └── outputs/      (optional, 1GB)
```

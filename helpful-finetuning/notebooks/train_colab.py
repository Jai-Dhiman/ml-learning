# Colab: install deps and run training (enter tokens via widgets)

# Cell 1: Install uv (we will use repo-pinned dependencies)
!pip -q install -U uv

# Cell 2: Clone repo (if needed) and cd
import os, glob
from pathlib import Path
if not os.path.exists('/content/ml-learning'):
    !git clone https://github.com/Jai-Dhiman/ml-learning.git /content/ml-learning
%cd /content/ml-learning/helpful-finetuning
# Create and sync a project-local environment pinned to repo deps
!uv venv
!bash -lc 'source .venv/bin/activate && uv sync'

# Ensure bitsandbytes with CUDA support and required Triton are available in this env
!bash -lc 'source .venv/bin/activate && uv pip uninstall -y bitsandbytes || true'
!bash -lc 'source .venv/bin/activate && python - <<\'PY\'
import torch, re, os
v = torch.version.cuda or ""
digits = re.sub(r"\D", "", v) or "121"
print("Detected CUDA version:", v, "-> BNB_CUDA_VERSION:", digits)
open("/tmp/bnb_cuda_version", "w").write(digits)
PY'
!bash -lc 'export BNB_CUDA_VERSION=$(cat /tmp/bnb_cuda_version); source .venv/bin/activate && uv pip install -U triton==2.2.0 && uv pip install --pre -U --extra-index-url https://jllllll.github.io/bitsandbytes-wheels/cu${BNB_CUDA_VERSION}/ bitsandbytes==0.43.1'

# Optional: Mount Google Drive for checkpoints/artifacts
from google.colab import drive
try:
    drive.mount('/content/drive')
    print('Drive mounted. If you have Stage 1 package zip in Drive, it can be used.')
except Exception as e:
    print('Drive not mounted. Proceeding without it.')

# If a Stage 1 zip exists in Drive, extract to expected path
exact_zip = '/content/drive/MyDrive/safety-text-classifier/safety_text_classifier_trained_20250916_0632.zip'
zip_candidates = []
if os.path.exists(exact_zip):
    zip_candidates = [exact_zip]
else:
    patterns = [
        '/content/drive/MyDrive/safety_text_classifier_trained_*.zip',
        '/content/drive/MyDrive/safety-text-classifier/safety_text_classifier_trained_*.zip',
    ]
    for p in patterns:
        zip_candidates.extend(glob.glob(p))

if zip_candidates:
    os.makedirs('/content/ml-learning/safety-text-classifier', exist_ok=True)
    zip_candidates.sort(reverse=True)
    print('Found Stage 1 package:', zip_candidates[0])
    !unzip -o "{zip_candidates[0]}" -d /content/ml-learning/safety-text-classifier
else:
    print('No Stage 1 package zip found in Drive. Safety filtering will still run if checkpoints exist in repo path, else default to safe.')

# Cell 3: Login to Hugging Face and W&B securely
from huggingface_hub import login
login()  # Enter your token in the widget (no plaintext in code)

import wandb
wandb.login()  # Enter your key in the widget

# Cell 4: Check GPU
import torch
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Cell 5: Run training with colab overrides
!bash -lc 'source .venv/bin/activate && python -m src.training.train_qlora --config configs/base_config.yaml --override configs/colab_config.yaml'

# Cell 6: Evaluate a subset
!bash -lc 'source .venv/bin/activate && python -m src.evaluation.evaluate_helpfulness --config configs/base_config.yaml'

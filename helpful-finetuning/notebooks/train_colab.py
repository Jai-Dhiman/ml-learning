# Colab: install deps and run training (enter tokens via widgets)

# Cell 1: Install deps
!pip -q install -U bitsandbytes==0.41.3 accelerate datasets wandb evaluate pyyaml tqdm
!pip -q install -U git+https://github.com/huggingface/transformers.git
!pip -q install -U git+https://github.com/huggingface/peft.git
# Stage 1 safety classifier (JAX/Flax) for filtering/eval
!pip -q install "jax[cpu]>=0.4.28,<0.5.0" "flax>=0.8.4,<0.9.0" "optax>=0.2.2,<0.3.0"

# Cell 2: Clone repo (if needed) and cd
import os, glob
from pathlib import Path
if not os.path.exists('/content/ml-learning'):
    !git clone https://github.com/yourusername/ml-learning.git /content/ml-learning
%cd /content/ml-learning/helpful-finetuning

# Optional: Mount Google Drive for checkpoints/artifacts
from google.colab import drive
try:
    drive.mount('/content/drive')
    print('Drive mounted. If you have Stage 1 package zip in Drive, it can be used.')
except Exception as e:
    print('Drive not mounted. Proceeding without it.')

# If a Stage 1 zip exists in Drive, extract to expected path
zip_candidates = glob.glob('/content/drive/MyDrive/safety_text_classifier_trained_*.zip')
if zip_candidates:
    os.makedirs('/content/ml-learning/safety-text-classifier', exist_ok=True)
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
!python -m src.training.train_qlora --config configs/base_config.yaml --override configs/colab_config.yaml

# Cell 6: Evaluate a subset
!python -m src.evaluation.evaluate_helpfulness --config configs/base_config.yaml

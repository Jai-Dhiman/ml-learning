"""
Comprehensive Preflight Check for Constitutional AI Evaluation on Colab

Validates all prerequisites before running the full evaluation to catch
errors early and avoid wasting GPU time.

Author: J. Dhiman
Date: October 4, 2025
"""

import sys
from pathlib import Path
import json
import torch
from typing import Tuple, Optional

def check_gpu() -> Tuple[bool, str]:
    """Check GPU availability and specifications."""
    if not torch.cuda.is_available():
        return False, "GPU not available. Go to Runtime > Change runtime type > Select GPU"
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    message = f"GPU: {gpu_name} ({gpu_memory:.1f} GB)"
    
    # Warn if memory is low
    if gpu_memory < 12:
        message += "\nWarning: GPU memory may be insufficient for all 3 models simultaneously"
    
    return True, message


def check_lora_adapters() -> Tuple[bool, str]:
    """Check that LoRA adapter files exist and have correct sizes."""
    adapters_to_check = [
        ("/content/ml-learning/artifacts/stage2_artifacts/lora_adapters/adapter_model.safetensors", 70, 80),
        ("/content/ml-learning/artifacts/stage2_artifacts/lora_adapters/adapter_config.json", 0.001, 0.01),
        ("/content/ml-learning/artifacts/stage3_artifacts/models/lora_adapters/adapter_model.safetensors", 70, 80),
        ("/content/ml-learning/artifacts/stage3_artifacts/models/lora_adapters/adapter_config.json", 0.001, 0.01),
    ]
    
    messages = []
    all_valid = True
    
    for file_path, min_size_mb, max_size_mb in adapters_to_check:
        path = Path(file_path)
        
        if not path.exists():
            all_valid = False
            messages.append(f"✗ Missing: {path.name}")
            continue
        
        size_mb = path.stat().st_size / (1024 * 1024)
        
        if size_mb < min_size_mb or size_mb > max_size_mb:
            all_valid = False
            messages.append(f"✗ Invalid size: {path.name} ({size_mb:.1f} MB, expected {min_size_mb}-{max_size_mb} MB)")
        else:
            messages.append(f"✓ {path.name} ({size_mb:.1f} MB)")
    
    if all_valid:
        return True, "\n".join(messages)
    else:
        return False, "\n".join(messages) + "\n\nEnsure LoRA adapters are uploaded to Google Drive and copied correctly"


def check_test_prompts() -> Tuple[bool, str]:
    """Check that extended test prompts file exists."""
    test_file = Path("/content/ml-learning/constitutional-ai-stage4/artifacts/evaluation/extended_test_prompts.jsonl")
    
    if not test_file.exists():
        return False, f"Test prompts file not found: {test_file}"
    
    # Count prompts
    try:
        with open(test_file, 'r') as f:
            num_prompts = sum(1 for line in f if line.strip())
        
        if num_prompts < 100:
            return False, f"Insufficient test prompts: {num_prompts} (expected 110)"
        
        return True, f"Extended test prompts: {num_prompts} prompts"
    except Exception as e:
        return False, f"Error reading test prompts: {e}"


def check_dependencies() -> Tuple[bool, str]:
    """Check that required packages are installed."""
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'datasets': 'Datasets',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib'
    }
    
    messages = []
    all_valid = True
    
    for package, name in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            messages.append(f"✓ {name}: {version}")
        except ImportError:
            all_valid = False
            messages.append(f"✗ {name} not installed")
    
    return all_valid, "\n".join(messages)


def test_model_loading() -> Tuple[bool, str]:
    """Test loading base model from HuggingFace."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "google/gemma-2b-it"
        
        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test model (don't load weights to save time, just check it's accessible)
        # In actual use, models will be loaded fully
        
        return True, f"Base model accessible: {model_name}"
    except Exception as e:
        return False, f"Failed to access base model: {str(e)}"


def estimate_runtime(gpu_name: str, num_prompts: int = 110, num_models: int = 3) -> str:
    """Estimate total runtime based on GPU type."""
    # Rough estimates in minutes per prompt per model
    times_per_prompt = {
        'T4': 2.5,  # minutes per prompt per model
        'L4': 1.5,
        'V100': 1.2,
        'A100': 0.8
    }
    
    # Determine GPU type
    gpu_type = 'T4'  # default
    for key in times_per_prompt.keys():
        if key in gpu_name:
            gpu_type = key
            break
    
    time_per_prompt = times_per_prompt[gpu_type]
    total_minutes = num_prompts * num_models * time_per_prompt
    total_hours = total_minutes / 60
    
    return f"Estimated runtime: {total_hours:.1f} hours ({total_minutes:.0f} minutes) on {gpu_type}"


def run_full_preflight() -> bool:
    """Run all preflight checks and report results."""
    print("="*70)
    print("COMPREHENSIVE PREFLIGHT CHECK")
    print("="*70)
    
    all_passed = True
    
    # 1. Check GPU
    print("\n1. Checking GPU...")
    gpu_ok, gpu_msg = check_gpu()
    print(gpu_msg)
    if not gpu_ok:
        print("❌ GPU check FAILED")
        all_passed = False
    else:
        print("✓ GPU check passed")
        
        # Extract GPU name for runtime estimate
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        print(f"\n{estimate_runtime(gpu_name)}")
    
    # 2. Check dependencies
    print("\n2. Checking Python dependencies...")
    deps_ok, deps_msg = check_dependencies()
    print(deps_msg)
    if not deps_ok:
        print("❌ Dependencies check FAILED")
        all_passed = False
    else:
        print("✓ Dependencies check passed")
    
    # 3. Check LoRA adapters
    print("\n3. Checking LoRA adapters...")
    adapters_ok, adapters_msg = check_lora_adapters()
    print(adapters_msg)
    if not adapters_ok:
        print("❌ LoRA adapters check FAILED")
        all_passed = False
    else:
        print("✓ LoRA adapters check passed")
    
    # 4. Check test prompts
    print("\n4. Checking test prompts...")
    prompts_ok, prompts_msg = check_test_prompts()
    print(prompts_msg)
    if not prompts_ok:
        print("❌ Test prompts check FAILED")
        all_passed = False
    else:
        print("✓ Test prompts check passed")
    
    # 5. Test model loading
    print("\n5. Testing base model access...")
    model_ok, model_msg = test_model_loading()
    print(model_msg)
    if not model_ok:
        print("❌ Model access check FAILED")
        all_passed = False
    else:
        print("✓ Model access check passed")
    
    # Final result
    print("\n" + "="*70)
    if all_passed:
        print("✅ PREFLIGHT PASSED - Ready for full evaluation!")
        print("="*70)
        print("\nYou can now proceed with the full evaluation.")
        print("The evaluation will take approximately 4-6 hours on T4 GPU.")
        return True
    else:
        print("❌ PREFLIGHT FAILED - Please fix the issues above")
        print("="*70)
        print("\nDo NOT proceed with full evaluation until all checks pass.")
        return False


if __name__ == '__main__':
    success = run_full_preflight()
    sys.exit(0 if success else 1)

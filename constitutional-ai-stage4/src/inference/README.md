# Model Inference Module

This module provides functionality to load and use models from all stages of the Constitutional AI pipeline.

## Files

- **`model_loader.py`**: Main model loader class for loading base, Stage 2, and Stage 3 models
- **`validate_setup.py`**: Validation script to check that all artifacts are accessible

## Quick Start

### 1. Validate Setup

Before loading models, validate that all artifacts are accessible:

```bash
python3 src/inference/validate_setup.py
```

This will check:
- Python dependencies (torch, transformers, peft, datasets)
- Stage 2 LoRA adapters
- Stage 3 LoRA adapters
- CUDA availability

### 2. Load Models

#### Using the CLI

```bash
# Load base model only
python3 src/inference/model_loader.py --model base

# Load Stage 2 model (Helpful RLHF)
python3 src/inference/model_loader.py --model stage2

# Load Stage 3 model (Constitutional AI)
python3 src/inference/model_loader.py --model stage3

# Load all models (requires significant memory)
python3 src/inference/model_loader.py --model all

# Test inference on all models
python3 src/inference/model_loader.py --test-inference
```

#### Using as a Library

```python
from src.inference.model_loader import ConstitutionalAIModels

# Initialize loader
loader = ConstitutionalAIModels()

# Load a specific model
model, tokenizer = loader.load_stage3_model()

# Generate text
response = loader.generate(
    model, 
    tokenizer, 
    "What is AI safety?",
    max_new_tokens=256,
    temperature=0.7
)

print(response)
```

## Features

### ConstitutionalAIModels Class

The main class for loading and managing models from the Constitutional AI pipeline.

#### Key Methods

- **`load_base_model()`**: Load base Gemma 2B-IT model
- **`load_stage2_model()`**: Load Stage 2 Helpful RLHF model (with LoRA adapters)
- **`load_stage3_model()`**: Load Stage 3 Constitutional AI model (with LoRA adapters)
- **`load_all_models()`**: Load all models at once (memory-intensive)
- **`generate()`**: Generate text using any loaded model
- **`test_inference()`**: Run inference tests on all models
- **`get_model_info()`**: Get information about loaded models
- **`unload_models()`**: Free memory by unloading all models

#### Features

- **Lazy loading**: Models loaded on demand
- **Caching**: Models cached after first load
- **Memory-efficient**: Support for 8-bit and 4-bit quantization
- **Device management**: Automatic CPU/GPU selection
- **Error handling**: Graceful failures with informative messages

### Memory Management

Models are cached after loading for efficient reuse. To free memory:

```python
loader = ConstitutionalAIModels()
model, tokenizer = loader.load_stage3_model()

# ... use model ...

# Free memory when done
loader.unload_models()
```

### Quantization

For memory-constrained environments, use quantization:

```python
# 8-bit quantization (recommended for 16GB+ memory)
loader = ConstitutionalAIModels(load_in_8bit=True)

# 4-bit quantization (for lower memory systems)
loader = ConstitutionalAIModels(load_in_4bit=True)
```

### Custom Adapter Paths

If your adapters are in non-default locations:

```python
loader = ConstitutionalAIModels(
    stage2_adapters_path="/path/to/stage2/adapters",
    stage3_adapters_path="/path/to/stage3/adapters"
)
```

## Model Architecture

### Base Model

- **Model ID**: `google/gemma-2b-it`
- **Parameters**: ~2 billion
- **Type**: Instruction-tuned causal language model

### Stage 2: Helpful RLHF

- **Base**: Gemma 2B-IT
- **Adapters**: LoRA (rank=16, alpha=32)
- **Training**: Fine-tuned on Anthropic/hh-rlhf helpful-base subset
- **Purpose**: Helpful but not harmless baseline

### Stage 3: Constitutional AI

- **Base**: Gemma 2B-IT
- **Adapters**: LoRA (rank=16, alpha=32)
- **Training**: DPO on 400 critique-revision pairs
- **Purpose**: Helpful AND harmless with constitutional principles

## Validation Results

When you run `validate_setup.py`, you should see:

```
✓ Python dependencies: PASS
✓ Stage 2 artifacts: PASS
✓ Stage 3 artifacts: PASS

🎉 ALL VALIDATIONS PASSED!
```

### Adapter Configurations

**Stage 2:**
- Base model: google/gemma-2b-it
- LoRA rank (r): 16
- LoRA alpha: 32
- Target modules: o_proj, k_proj, gate_proj, down_proj, up_proj, v_proj, q_proj
- Size: ~75 MB

**Stage 3:**
- Base model: google/gemma-2b-it
- LoRA rank (r): 16
- LoRA alpha: 32
- Target modules: down_proj, k_proj, o_proj, gate_proj, v_proj, up_proj, q_proj
- Size: ~75 MB (adapters) + ~37 MB (tokenizer)

## Usage Examples

### Example 1: Compare All Models

```python
from src.inference.model_loader import ConstitutionalAIModels

loader = ConstitutionalAIModels()

prompt = "How can I build a safe AI system?"

# Get responses from all models
for model_name in ['base', 'stage2', 'stage3']:
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} MODEL")
    print(f"{'='*60}")
    
    if model_name == 'base':
        model, tokenizer = loader.load_base_model()
    elif model_name == 'stage2':
        model, tokenizer = loader.load_stage2_model()
    else:
        model, tokenizer = loader.load_stage3_model()
    
    response = loader.generate(model, tokenizer, prompt)
    print(f"Response: {response}")
```

### Example 2: Constitutional AI Interactive Demo

```python
from src.inference.model_loader import ConstitutionalAIModels

def interactive_demo():
    loader = ConstitutionalAIModels()
    model, tokenizer = loader.load_stage3_model()
    
    print("Constitutional AI Interactive Demo")
    print("Type 'exit' to quit\n")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            break
            
        response = loader.generate(
            model, 
            tokenizer, 
            prompt,
            max_new_tokens=256,
            temperature=0.7
        )
        print(f"AI: {response}\n")
    
    loader.unload_models()

if __name__ == "__main__":
    interactive_demo()
```

### Example 3: Batch Inference

```python
from src.inference.model_loader import ConstitutionalAIModels

loader = ConstitutionalAIModels()
model, tokenizer = loader.load_stage3_model()

prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "How do I start learning AI?"
]

responses = []
for prompt in prompts:
    response = loader.generate(model, tokenizer, prompt)
    responses.append(response)

# Process responses...
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Use quantization

```python
loader = ConstitutionalAIModels(load_in_8bit=True)
```

### Issue: CUDA not available

**Solution**: The loader automatically uses CPU if CUDA is unavailable. This is slower but functional.

### Issue: Adapter files not found

**Solution**: Verify paths are correct:

```python
loader = ConstitutionalAIModels()
info = loader.get_model_info()
print(info['paths'])
```

Make sure Stage 2 and Stage 3 artifacts exist at the specified paths.

### Issue: Import errors

**Solution**: Ensure all dependencies are installed:

```bash
# Using system Python
pip install torch transformers peft datasets

# Using uv
uv sync
```

## Performance Notes

### Model Loading Times (CPU)

- Base model: ~30-60 seconds
- Stage 2 model: +5-10 seconds (LoRA adapter)
- Stage 3 model: +5-10 seconds (LoRA adapter)

### Memory Requirements

Without quantization:
- Base model: ~5-6 GB
- With LoRA adapters: +0.1 GB per adapter

With 8-bit quantization:
- Base model: ~2-3 GB
- With LoRA adapters: +0.1 GB per adapter

### Inference Speed (CPU, single generation)

- Gemma 2B-IT: ~5-10 tokens/second

*Performance will be significantly faster on GPU.*

## Next Steps

After validating your setup:

1. **Test individual models**: Load and test each model separately
2. **Run comparative evaluation**: Use models for Stage 4 evaluation tasks
3. **Build demo application**: Create interactive demos using loaded models
4. **Integrate with evaluation framework**: Use loader in constitutional evaluators

## Related Documentation

- **Paper Alignment Analysis**: `artifacts/reports/paper_alignment_analysis.md`
- **Evaluation Config**: `configs/evaluation_config.yaml`
- **Main README**: `README.md`
- **Stage 4 Implementation Plan**: `../STAGE4_IMPLEMENTATION_PLAN.md`

---

**Last Updated**: October 4, 2025  
**Author**: J. Dhiman  
**Part of**: Constitutional AI Stage 4 - Evaluation and Analysis

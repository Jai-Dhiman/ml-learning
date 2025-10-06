"""
Validation Script for Model Loader Setup

Quick validation to ensure all artifacts are accessible without
loading full models (which requires significant memory).

Author: J. Dhiman
Date: October 4, 2025
"""

from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_adapter_files(adapter_path: Path, stage_name: str) -> bool:
    """
    Validate that required adapter files exist.
    
    Args:
        adapter_path: Path to adapter directory
        stage_name: Name of stage for logging
        
    Returns:
        True if validation passes
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Validating {stage_name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Path: {adapter_path}")
    
    if not adapter_path.exists():
        logger.error(f"✗ Directory not found: {adapter_path}")
        return False
    
    logger.info(f"✓ Directory exists")
    
    # Check required files
    required_files = ['adapter_config.json', 'adapter_model.safetensors']
    optional_files = ['README.md', 'tokenizer.json', 'tokenizer.model']
    
    all_valid = True
    
    for file_name in required_files:
        file_path = adapter_path / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {file_name} ({size_mb:.1f} MB)")
        else:
            logger.error(f"✗ Missing required file: {file_name}")
            all_valid = False
    
    for file_name in optional_files:
        file_path = adapter_path / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {file_name} ({size_mb:.1f} MB)")
    
    # Parse and display adapter config
    config_path = adapter_path / 'adapter_config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info("\nAdapter Configuration:")
            logger.info(f"  Base model: {config.get('base_model_name_or_path', 'N/A')}")
            logger.info(f"  LoRA rank (r): {config.get('r', 'N/A')}")
            logger.info(f"  LoRA alpha: {config.get('lora_alpha', 'N/A')}")
            logger.info(f"  Target modules: {', '.join(config.get('target_modules', []))}")
            logger.info(f"  Task type: {config.get('task_type', 'N/A')}")
        except Exception as e:
            logger.error(f"✗ Failed to parse adapter config: {e}")
            all_valid = False
    
    if all_valid:
        logger.info(f"\n✓ {stage_name} validation PASSED")
    else:
        logger.error(f"\n✗ {stage_name} validation FAILED")
    
    return all_valid


def validate_dependencies():
    """Validate required Python packages are installed."""
    logger.info(f"\n{'=' * 60}")
    logger.info("Validating Python Dependencies")
    logger.info(f"{'=' * 60}")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'datasets': 'Datasets',
    }
    
    all_valid = True
    
    for package, name in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"✓ {name}: {version}")
        except ImportError:
            logger.error(f"✗ {name} not installed")
            all_valid = False
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  Device count: {torch.cuda.device_count()}")
        else:
            logger.info("  CUDA not available (will use CPU)")
    except:
        logger.warning("  Could not check CUDA availability")
    
    if all_valid:
        logger.info("\n✓ All dependencies installed")
    else:
        logger.error("\n✗ Some dependencies missing")
    
    return all_valid


def main():
    """Run complete validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Constitutional AI model loader setup"
    )
    parser.add_argument(
        '--stage2-adapter-path',
        type=str,
        default="../artifacts/stage2_artifacts/lora_adapters",
        help='Path to Stage 2 LoRA adapters'
    )
    parser.add_argument(
        '--stage3-adapter-path',
        type=str,
        default="../artifacts/stage3_artifacts/models/lora_adapters",
        help='Path to Stage 3 LoRA adapters'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Constitutional AI Model Loader Validation")
    logger.info("=" * 60)
    
    # Validate dependencies
    deps_valid = validate_dependencies()
    
    # Validate Stage 2 adapters
    stage2_path = Path(args.stage2_adapter_path)
    stage2_valid = validate_adapter_files(stage2_path, "Stage 2 (Helpful RLHF)")
    
    # Validate Stage 3 adapters
    stage3_path = Path(args.stage3_adapter_path)
    stage3_valid = validate_adapter_files(stage3_path, "Stage 3 (Constitutional AI)")
    
    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'=' * 60}")
    
    if deps_valid:
        logger.info("✓ Python dependencies: PASS")
    else:
        logger.error("✗ Python dependencies: FAIL")
    
    if stage2_valid:
        logger.info("✓ Stage 2 artifacts: PASS")
    else:
        logger.error("✗ Stage 2 artifacts: FAIL")
    
    if stage3_valid:
        logger.info("✓ Stage 3 artifacts: PASS")
    else:
        logger.error("✗ Stage 3 artifacts: FAIL")
    
    logger.info(f"{'=' * 60}")
    
    if deps_valid and stage2_valid and stage3_valid:
        logger.info("\n🎉 ALL VALIDATIONS PASSED!")
        logger.info("\nYou can now:")
        logger.info("  1. Test model loading:")
        logger.info("     uv run python src/inference/model_loader.py --model base")
        logger.info("  2. Run inference test:")
        logger.info("     uv run python src/inference/model_loader.py --test-inference")
        logger.info("  3. Use in your code:")
        logger.info("     from src.inference.model_loader import ConstitutionalAIModels")
        return 0
    else:
        logger.error("\n❌ VALIDATION FAILED")
        logger.error("Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())

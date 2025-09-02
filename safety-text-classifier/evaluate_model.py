#!/usr/bin/env python3
"""
Evaluate the trained safety text classifier.
Run this after downloading the model from Colab.
"""

import sys
import os
from pathlib import Path

def find_model_package():
    """Find downloaded model package or use current directory."""
    current_dir = Path.cwd()
    
    # Check if we're in a downloaded model package directory
    if (current_dir.name.startswith('safety_classifier_') and 
        (current_dir / 'checkpoints').exists() and 
        (current_dir / 'src').exists()):
        print(f"✅ Found model package: {current_dir.name}")
        return current_dir
    
    # Look for model package in current directory
    for item in current_dir.iterdir():
        if (item.is_dir() and 
            item.name.startswith('safety_classifier_') and
            (item / 'checkpoints').exists()):
            print(f"✅ Found model package: {item.name}")
            return item
    
    # Fallback to current directory (original setup)
    if (current_dir / 'src').exists() and (current_dir / 'configs').exists():
        print("✅ Using current project directory")
        return current_dir
    
    print("❌ No model package found!")
    print("💡 Please extract the downloaded safety_classifier_*.zip file")
    return None

def main():
    print("🔍 Evaluating trained safety text classifier...")
    
    # Find model package
    model_dir = find_model_package()
    if not model_dir:
        return
    
    # Change to model directory
    os.chdir(model_dir)
    
    # Add src to Python path
    src_path = str(model_dir / 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"✅ Added to Python path: {src_path}")
    
    try:
        from training.trainer import SafetyTrainer
        from data.dataset_loader import create_data_loaders
        
        # Load trainer and checkpoint
        config_path = 'configs/colab_config.yaml'
        if not os.path.exists(config_path):
            print(f"❌ Config not found: {config_path}")
            return
            
        trainer = SafetyTrainer(config_path)
        
        # Update checkpoint path to local directory
        checkpoint_path = model_dir / 'checkpoints'
        if checkpoint_path.exists():
            trainer.config['paths']['checkpoint_dir'] = str(checkpoint_path)
            print(f"✅ Using checkpoint directory: {checkpoint_path}")
        else:
            print(f"❌ No checkpoints found at: {checkpoint_path}")
            return
        
        # Load the trained checkpoint
        trainer.load_checkpoint()
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   uv sync  # or pip install -r requirements.txt")
        return
    
    # Load test data
    _, _, test_dataset = create_data_loaders(config_path)
    
    # Evaluate on test set
    print("📊 Running evaluation on test set...")
    test_metrics = trainer.evaluate(test_dataset, 0, "test/")
    
    print("\n" + "="*60)
    print("🎯 FINAL TEST RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    # Per-class accuracy
    if 'per_class_accuracy' in test_metrics:
        categories = ['Hate Speech', 'Self Harm', 'Dangerous Advice', 'Harassment']
        per_class_acc = test_metrics['per_class_accuracy']
        print("\n📋 Per-Category Accuracy:")
        for cat, acc in zip(categories, per_class_acc):
            print(f"  {cat}: {acc:.1%}")
    
    # Determine if Stage 1 is complete
    accuracy = float(test_metrics['accuracy'])
    target_accuracy = 0.85  # 85% target from README
    
    print("\n" + "="*60)
    print("🎖️  STAGE 1 STATUS")
    print("="*60)
    
    if accuracy >= target_accuracy:
        print(f"✅ STAGE 1 COMPLETE! ({accuracy:.1%} ≥ {target_accuracy:.0%})")
        print("🎉 Ready to proceed to Stage 2: Helpful Response Fine-tuning")
    else:
        print(f"❌ Stage 1 incomplete ({accuracy:.1%} < {target_accuracy:.0%})")
        print("📝 Consider adjusting hyperparameters or training longer")
    
    print("\n🔗 Next steps:")
    print("  1. Set up inference server: python src/serve.py")
    print("  2. Launch demo interface: python src/demo.py") 
    print("  3. Begin Stage 2 planning")

if __name__ == "__main__":
    main()
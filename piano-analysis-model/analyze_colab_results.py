#!/usr/bin/env python3
"""
Post-Colab Training Analysis Script
Analyze downloaded results from Google Colab CNN training
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import zipfile

def analyze_training_results(results_zip_path: str):
    """Analyze downloaded Colab training results"""
    
    # Extract results zip
    results_dir = Path("colab_results") 
    results_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(results_zip_path, 'r') as zip_ref:
        zip_ref.extractall(results_dir)
    
    # Find the extracted directory
    extracted_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not extracted_dirs:
        print("❌ No results directory found in zip")
        return
    
    result_path = extracted_dirs[0]
    print(f"📁 Analyzing results from: {result_path}")
    
    # Load training results
    training_file = result_path / "training_results.json"
    summary_file = result_path / "experiment_summary.json"
    
    if not training_file.exists():
        print("❌ training_results.json not found")
        return
    
    with open(training_file) as f:
        training_data = json.load(f)
    
    with open(summary_file) as f:
        summary_data = json.load(f)
    
    # Analysis and visualization
    print(f"🎯 Training Analysis Summary")
    print(f"Architecture: {summary_data['model_architecture'].upper()}")
    print(f"Final test loss: {summary_data['final_results']['test_loss']:.4f}")
    
    test_correlations = summary_data['final_results']['test_correlations']
    mean_correlation = np.mean(test_correlations)
    strong_correlations = sum(1 for c in test_correlations if c > 0.5)
    
    print(f"Mean correlation: {mean_correlation:.3f}")
    print(f"Strong correlations (>0.5): {strong_correlations}/19")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Piano CNN Training Results - {summary_data["model_architecture"].upper()}', fontsize=16)
    
    # Training curves
    axes[0,0].plot(training_data['train_loss'], label='Train', alpha=0.8)
    axes[0,0].plot(training_data['val_loss'], label='Validation', alpha=0.8)
    axes[0,0].set_title('Loss Curves')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Correlation development
    if 'val_correlations' in training_data:
        val_corrs = np.array(training_data['val_correlations'])
        epochs = range(len(val_corrs))
        axes[0,1].plot(epochs, val_corrs.mean(axis=1), 'g-', label='Mean', linewidth=2)
        axes[0,1].fill_between(epochs, 
                              val_corrs.mean(axis=1) - val_corrs.std(axis=1),
                              val_corrs.mean(axis=1) + val_corrs.std(axis=1),
                              alpha=0.3, color='green')
        axes[0,1].set_title('Validation Correlations')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Correlation')
        axes[0,1].grid(True, alpha=0.3)
    
    # Final test correlations
    from dataset_analysis import PERCEPTUAL_DIMENSIONS
    x_pos = np.arange(len(test_correlations))
    colors = ['#2E8B57' if c > 0.5 else '#FF8C00' if c > 0.3 else '#DC143C' for c in test_correlations]
    
    bars = axes[0,2].bar(x_pos, test_correlations, color=colors, alpha=0.7)
    axes[0,2].set_title('Test Correlations by Dimension')
    axes[0,2].set_xlabel('Perceptual Dimension')  
    axes[0,2].set_ylabel('Correlation')
    axes[0,2].set_xticks(x_pos[::3])
    axes[0,2].set_xticklabels([PERCEPTUAL_DIMENSIONS[i][:10] for i in x_pos[::3]], 
                             rotation=45, ha='right')
    
    # Top/Bottom performing dimensions
    dim_performance = list(zip(PERCEPTUAL_DIMENSIONS, test_correlations))
    dim_performance.sort(key=lambda x: x[1], reverse=True)
    
    top_5 = dim_performance[:5]
    bottom_5 = dim_performance[-5:]
    
    # Top 5 plot
    top_dims, top_corrs = zip(*top_5)
    axes[1,0].barh(range(len(top_dims)), top_corrs, color='green', alpha=0.7)
    axes[1,0].set_yticks(range(len(top_dims)))
    axes[1,0].set_yticklabels([d[:15] for d in top_dims])
    axes[1,0].set_title('Top 5 Best Predicted')
    axes[1,0].set_xlabel('Correlation')
    
    # Bottom 5 plot
    bottom_dims, bottom_corrs = zip(*bottom_5)
    axes[1,1].barh(range(len(bottom_dims)), bottom_corrs, color='red', alpha=0.7)
    axes[1,1].set_yticks(range(len(bottom_dims)))
    axes[1,1].set_yticklabels([d[:15] for d in bottom_dims])
    axes[1,1].set_title('Bottom 5 (Need Improvement)')
    axes[1,1].set_xlabel('Correlation')
    
    # Training summary
    axes[1,2].axis('off')
    summary_text = f"""
Training Summary:
• Architecture: {summary_data['model_architecture'].upper()}
• Epochs trained: {len(training_data['train_loss'])}
• Final test loss: {summary_data['final_results']['test_loss']:.4f}
• Mean correlation: {mean_correlation:.3f}
• Best dimension: {top_5[0][0]} ({top_5[0][1]:.3f})
• Challenging: {bottom_5[0][0]} ({bottom_5[0][1]:.3f})

Dataset:
• Train samples: {summary_data['dataset_info']['train_samples']}
• Val samples: {summary_data['dataset_info']['val_samples']}
• Test samples: {summary_data['dataset_info']['test_samples']}

Performance Tiers:
• Strong (>0.5): {strong_correlations} dimensions
• Fair (0.3-0.5): {sum(1 for c in test_correlations if 0.3 < c <= 0.5)} dimensions  
• Weak (<0.3): {sum(1 for c in test_correlations if c <= 0.3)} dimensions
"""
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(f'piano_cnn_analysis_{summary_data["model_architecture"]}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed insights
    print(f"\n🏆 Top 5 Best Predicted Dimensions:")
    for i, (dim, corr) in enumerate(top_5):
        print(f"   {i+1}. {dim}: {corr:.3f}")
    
    print(f"\n🎯 Most Challenging Dimensions:")
    for i, (dim, corr) in enumerate(bottom_5):
        print(f"   {i+1}. {dim}: {corr:.3f}")
    
    print(f"\n📊 Performance Distribution:")
    print(f"   Strong correlations (>0.5): {strong_correlations}/19 ({strong_correlations/19*100:.1f}%)")
    print(f"   Fair correlations (0.3-0.5): {sum(1 for c in test_correlations if 0.3 < c <= 0.5)}/19")
    print(f"   Weak correlations (<0.3): {sum(1 for c in test_correlations if c <= 0.3)}/19")
    
    print(f"\n💡 Insights & Next Steps:")
    if mean_correlation > 0.5:
        print("   ✅ Excellent performance! Model successfully learned perceptual patterns.")
    elif mean_correlation > 0.3:
        print("   📈 Good performance with room for improvement.")
    else:
        print("   🔧 Consider architecture changes or hyperparameter tuning.")
    
    if strong_correlations >= 10:
        print("   🎯 Strong performance across majority of dimensions.")
    else:
        print(f"   🎯 Focus on improving {19-strong_correlations} weaker dimensions.")
    
    print(f"\n📝 For your portfolio:")
    print(f"   • Document the {summary_data['model_architecture']} architecture's strengths")
    print(f"   • Highlight best-performing dimensions: {', '.join([d[0] for d in top_5[:3]])}")
    print(f"   • Discuss learning challenges and future improvements")
    print(f"   • Compare with other architectures if training multiple models")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_training_results(sys.argv[1])
    else:
        print("Usage: python analyze_colab_results.py <results_zip_file>")
        print("Example: python analyze_colab_results.py piano_cnn_results_standard_20250828_1234.zip")

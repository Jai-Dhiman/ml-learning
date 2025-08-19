"""
Comprehensive Evaluation Metrics for Safety Text Classifier

Implements fairness, robustness, and performance metrics for the Constitutional AI
research project's safety text classifier.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SafetyMetricsCalculator:
    """
    Comprehensive metrics calculator for safety text classification.
    
    Includes performance metrics, fairness assessment, and robustness evaluation
    as required for Constitutional AI research evaluation.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: Names of the safety categories
        """
        self.class_names = class_names or [
            'hate_speech', 'self_harm', 'dangerous_advice', 'harassment'
        ]
        self.num_classes = len(self.class_names)
    
    def compute_basic_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Compute basic classification metrics.
        
        Args:
            y_true: True labels (batch_size, num_classes)
            y_pred: Predicted labels (batch_size, num_classes)
            y_prob: Prediction probabilities (batch_size, num_classes)
            
        Returns:
            Dictionary of basic metrics
        """
        metrics = {}
        
        # Convert to numpy if needed
        if hasattr(y_true, 'device'):  # JAX array
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            if y_prob is not None:
                y_prob = np.array(y_prob)
        
        # Overall accuracy (exact match for multi-label)
        exact_match_accuracy = accuracy_score(y_true, y_pred)
        metrics['exact_match_accuracy'] = exact_match_accuracy
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, i]
            y_pred_class = y_pred[:, i]
            
            precision = precision_score(y_true_class, y_pred_class, zero_division=0)
            recall = recall_score(y_true_class, y_pred_class, zero_division=0)
            f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
            
            metrics[f'{class_name}_precision'] = precision
            metrics[f'{class_name}_recall'] = recall
            metrics[f'{class_name}_f1'] = f1
            
            # AUC if probabilities available
            if y_prob is not None:
                try:
                    auc = roc_auc_score(y_true_class, y_prob[:, i])
                    metrics[f'{class_name}_auc'] = auc
                except ValueError:
                    metrics[f'{class_name}_auc'] = 0.0
        
        # Macro averages
        precisions = [metrics[f'{name}_precision'] for name in self.class_names]
        recalls = [metrics[f'{name}_recall'] for name in self.class_names]
        f1s = [metrics[f'{name}_f1'] for name in self.class_names]
        
        metrics['macro_precision'] = np.mean(precisions)
        metrics['macro_recall'] = np.mean(recalls)
        metrics['macro_f1'] = np.mean(f1s)
        
        # Micro averages (flatten all classes)
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        metrics['micro_precision'] = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        metrics['micro_recall'] = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        metrics['micro_f1'] = f1_score(y_true_flat, y_pred_flat, zero_division=0)
        
        return metrics
    
    def compute_calibration_metrics(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Compute calibration metrics to assess prediction confidence quality.
        
        Args:
            y_true: True labels (batch_size, num_classes)
            y_prob: Prediction probabilities (batch_size, num_classes)
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary of calibration metrics
        """
        calibration_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, i]
            y_prob_class = y_prob[:, i]
            
            # Skip if no positive examples
            if np.sum(y_true_class) == 0:
                calibration_metrics[f'{class_name}_ece'] = 0.0
                continue
            
            # Compute Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob_class > bin_lower) & (y_prob_class <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true_class[in_bin].mean()
                    avg_confidence_in_bin = y_prob_class[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            calibration_metrics[f'{class_name}_ece'] = ece
        
        # Overall ECE
        calibration_metrics['overall_ece'] = np.mean([
            calibration_metrics[f'{name}_ece'] for name in self.class_names
        ])
        
        return calibration_metrics
    
    def compute_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        sensitive_attributes: np.ndarray,
        attribute_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compute fairness metrics across different demographic groups.
        
        Args:
            y_true: True labels (batch_size, num_classes)
            y_pred: Predicted labels (batch_size, num_classes)
            y_prob: Prediction probabilities (batch_size, num_classes)
            sensitive_attributes: Group identifiers (batch_size, num_attributes)
            attribute_names: Names of sensitive attributes
            
        Returns:
            Dictionary of fairness metrics
        """
        if attribute_names is None:
            attribute_names = [f'attribute_{i}' for i in range(sensitive_attributes.shape[1])]
        
        fairness_metrics = {}
        
        for attr_idx, attr_name in enumerate(attribute_names):
            groups = np.unique(sensitive_attributes[:, attr_idx])
            
            # Demographic Parity: P(Y_hat = 1 | A = a) should be equal across groups
            group_positive_rates = {}
            group_accuracies = {}
            group_tpr = {}  # True Positive Rate
            group_fpr = {}  # False Positive Rate
            
            for group in groups:
                group_mask = sensitive_attributes[:, attr_idx] == group
                
                if np.sum(group_mask) == 0:
                    continue
                
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                group_y_prob = y_prob[group_mask]
                
                # Positive prediction rate (demographic parity)
                positive_rate = np.mean(group_y_pred)
                group_positive_rates[group] = positive_rate
                
                # Accuracy
                accuracy = accuracy_score(group_y_true.flatten(), group_y_pred.flatten())
                group_accuracies[group] = accuracy
                
                # True/False Positive Rates for each class
                for class_idx, class_name in enumerate(self.class_names):
                    class_y_true = group_y_true[:, class_idx]
                    class_y_pred = group_y_pred[:, class_idx]
                    
                    if np.sum(class_y_true) > 0:  # Has positive examples
                        tpr = np.sum((class_y_true == 1) & (class_y_pred == 1)) / np.sum(class_y_true == 1)
                        group_tpr[f'{group}_{class_name}'] = tpr
                    
                    if np.sum(class_y_true == 0) > 0:  # Has negative examples
                        fpr = np.sum((class_y_true == 0) & (class_y_pred == 1)) / np.sum(class_y_true == 0)
                        group_fpr[f'{group}_{class_name}'] = fpr
            
            # Compute fairness violations
            if len(group_positive_rates) >= 2:
                rates = list(group_positive_rates.values())
                demographic_parity_diff = max(rates) - min(rates)
                fairness_metrics[f'{attr_name}_demographic_parity_diff'] = demographic_parity_diff
            
            if len(group_accuracies) >= 2:
                accuracies = list(group_accuracies.values())
                accuracy_diff = max(accuracies) - min(accuracies)
                fairness_metrics[f'{attr_name}_accuracy_diff'] = accuracy_diff
            
            # Equalized Odds: TPR and FPR should be equal across groups
            for class_name in self.class_names:
                class_tpr = [v for k, v in group_tpr.items() if class_name in k]
                class_fpr = [v for k, v in group_fpr.items() if class_name in k]
                
                if len(class_tpr) >= 2:
                    tpr_diff = max(class_tpr) - min(class_tpr)
                    fairness_metrics[f'{attr_name}_{class_name}_tpr_diff'] = tpr_diff
                
                if len(class_fpr) >= 2:
                    fpr_diff = max(class_fpr) - min(class_fpr)
                    fairness_metrics[f'{attr_name}_{class_name}_fpr_diff'] = fpr_diff
        
        return fairness_metrics
    
    def compute_robustness_metrics(
        self,
        model_apply_fn: callable,
        params: Any,
        original_inputs: np.ndarray,
        perturbed_inputs_list: List[np.ndarray],
        perturbation_names: List[str]
    ) -> Dict[str, Any]:
        """
        Compute robustness metrics for different input perturbations.
        
        Args:
            model_apply_fn: Model application function
            params: Model parameters
            original_inputs: Original input sequences
            perturbed_inputs_list: List of perturbed input sequences
            perturbation_names: Names of perturbation types
            
        Returns:
            Dictionary of robustness metrics
        """
        robustness_metrics = {}
        
        # Get original predictions
        original_outputs = model_apply_fn(params, original_inputs, training=False)
        original_probs = jax.nn.sigmoid(original_outputs['logits'])
        original_preds = (original_probs > 0.5).astype(jnp.int32)
        
        for perturbed_inputs, pert_name in zip(perturbed_inputs_list, perturbation_names):
            # Get perturbed predictions
            perturbed_outputs = model_apply_fn(params, perturbed_inputs, training=False)
            perturbed_probs = jax.nn.sigmoid(perturbed_outputs['logits'])
            perturbed_preds = (perturbed_probs > 0.5).astype(jnp.int32)
            
            # Prediction consistency
            consistency = jnp.mean(jnp.all(original_preds == perturbed_preds, axis=-1))
            robustness_metrics[f'{pert_name}_consistency'] = float(consistency)
            
            # Probability stability (average L2 distance)
            prob_distance = jnp.mean(jnp.linalg.norm(original_probs - perturbed_probs, axis=-1))
            robustness_metrics[f'{pert_name}_prob_stability'] = float(prob_distance)
            
            # Confidence change
            original_confidence = jnp.max(original_probs, axis=-1)
            perturbed_confidence = jnp.max(perturbed_probs, axis=-1)
            confidence_change = jnp.mean(jnp.abs(original_confidence - perturbed_confidence))
            robustness_metrics[f'{pert_name}_confidence_change'] = float(confidence_change)
        
        return robustness_metrics
    
    def generate_confusion_matrices(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: str = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate confusion matrices for each class.
        
        Args:
            y_true: True labels (batch_size, num_classes)
            y_pred: Predicted labels (batch_size, num_classes)
            save_path: Optional path to save plots
            
        Returns:
            Dictionary of confusion matrices
        """
        confusion_matrices = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, i]
            y_pred_class = y_pred[:, i]
            
            cm = confusion_matrix(y_true_class, y_pred_class)
            confusion_matrices[class_name] = cm
            
            # Plot confusion matrix
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                ax=axes[i],
                xticklabels=['Not ' + class_name, class_name],
                yticklabels=['Not ' + class_name, class_name]
            )
            axes[i].set_title(f'{class_name.replace("_", " ").title()} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.close()
        
        return confusion_matrices
    
    def generate_comprehensive_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        sensitive_attributes: np.ndarray = None,
        save_path: str = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            sensitive_attributes: Optional demographic attributes for fairness analysis
            save_path: Optional path to save report
            
        Returns:
            Complete evaluation report
        """
        report = {}
        
        # Basic performance metrics
        basic_metrics = self.compute_basic_metrics(y_true, y_pred, y_prob)
        report['performance'] = basic_metrics
        
        # Calibration metrics
        calibration_metrics = self.compute_calibration_metrics(y_true, y_prob)
        report['calibration'] = calibration_metrics
        
        # Fairness metrics if demographic data available
        if sensitive_attributes is not None:
            fairness_metrics = self.compute_fairness_metrics(
                y_true, y_pred, y_prob, sensitive_attributes
            )
            report['fairness'] = fairness_metrics
        
        # Confusion matrices
        confusion_matrices = self.generate_confusion_matrices(
            y_true, y_pred, 
            save_path=save_path.replace('.json', '_confusion_matrices.png') if save_path else None
        )
        report['confusion_matrices'] = {k: v.tolist() for k, v in confusion_matrices.items()}
        
        # Summary statistics
        report['summary'] = {
            'total_samples': len(y_true),
            'classes': self.class_names,
            'overall_accuracy': basic_metrics['exact_match_accuracy'],
            'macro_f1': basic_metrics['macro_f1'],
            'calibration_quality': calibration_metrics['overall_ece']
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Comprehensive report saved to {save_path}")
        
        return report


def evaluate_model_comprehensive(
    model_apply_fn: callable,
    params: Any,
    test_dataset: Any,
    batch_size: int = 32,
    include_fairness: bool = False,
    include_robustness: bool = False
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation of a safety text classifier.
    
    Args:
        model_apply_fn: Model application function
        params: Model parameters
        test_dataset: Test dataset
        batch_size: Batch size for evaluation
        include_fairness: Whether to include fairness analysis
        include_robustness: Whether to include robustness testing
        
    Returns:
        Comprehensive evaluation results
    """
    calculator = SafetyMetricsCalculator()
    
    # Collect all predictions
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    # Process in batches
    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i + batch_size]
        
        # Prepare batch
        input_ids = jnp.array([item['input_ids'] for item in batch])
        labels = jnp.array([item['labels'] for item in batch])
        
        # Get predictions
        outputs = model_apply_fn(params, input_ids, training=False)
        probs = jax.nn.sigmoid(outputs['logits'])
        preds = (probs > 0.5).astype(jnp.int32)
        
        all_y_true.append(np.array(labels))
        all_y_pred.append(np.array(preds))
        all_y_prob.append(np.array(probs))
    
    # Concatenate all results
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    y_prob = np.concatenate(all_y_prob, axis=0)
    
    # Generate comprehensive report
    report = calculator.generate_comprehensive_report(
        y_true, y_pred, y_prob,
        save_path="evaluation_report.json"
    )
    
    return report


if __name__ == "__main__":
    # Test metrics calculator
    np.random.seed(42)
    
    # Generate synthetic test data
    n_samples = 1000
    n_classes = 4
    
    y_true = np.random.randint(0, 2, (n_samples, n_classes))
    y_prob = np.random.random((n_samples, n_classes))
    y_pred = (y_prob > 0.5).astype(int)
    
    # Test metrics calculation
    calculator = SafetyMetricsCalculator()
    
    basic_metrics = calculator.compute_basic_metrics(y_true, y_pred, y_prob)
    print("Basic metrics:", basic_metrics)
    
    calibration_metrics = calculator.compute_calibration_metrics(y_true, y_prob)
    print("Calibration metrics:", calibration_metrics)
    
    # Test comprehensive report
    report = calculator.generate_comprehensive_report(y_true, y_pred, y_prob)
    print("Report generated successfully!")
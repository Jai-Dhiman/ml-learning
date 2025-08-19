"""
Model Validation Framework

Comprehensive validation and testing framework for the Safety Text Classifier
including performance metrics, robustness tests, and fairness evaluation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from scipy.stats import chi2_contingency, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json

from .calibration import CalibrationMetrics, ConfidenceEstimator
from .utils import ModelLoader
from ..data.dataset_loader import SafetyDatasetLoader

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Comprehensive model validation framework.
    """
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """
        Initialize the model validator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.safety_categories = {
            0: 'hate_speech',
            1: 'self_harm',
            2: 'dangerous_advice', 
            3: 'harassment'
        }
        
        self.results = {}
        
    def validate_model_performance(
        self,
        model,
        params: Dict[str, Any],
        test_dataset,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Comprehensive performance validation.
        
        Args:
            model: JAX model
            params: Model parameters
            test_dataset: Test dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Starting model performance validation...")
        
        # Get predictions
        predictions, labels = self._get_predictions(
            model, params, test_dataset, batch_size
        )
        
        # Compute metrics for each class
        class_metrics = {}
        for class_idx, class_name in self.safety_categories.items():
            class_pred = predictions[:, class_idx]
            class_labels = labels[:, class_idx]
            
            class_metrics[class_name] = {
                'accuracy': accuracy_score(class_labels, class_pred > 0.5),
                'precision': precision_score(class_labels, class_pred > 0.5, zero_division=0),
                'recall': recall_score(class_labels, class_pred > 0.5, zero_division=0),
                'f1': f1_score(class_labels, class_pred > 0.5, zero_division=0),
                'auc_roc': roc_auc_score(class_labels, class_pred) if len(np.unique(class_labels)) > 1 else 0.0,
                'auc_pr': average_precision_score(class_labels, class_pred) if len(np.unique(class_labels)) > 1 else 0.0
            }
        
        # Overall metrics (macro-averaged)
        overall_metrics = {
            'macro_accuracy': np.mean([m['accuracy'] for m in class_metrics.values()]),
            'macro_precision': np.mean([m['precision'] for m in class_metrics.values()]),
            'macro_recall': np.mean([m['recall'] for m in class_metrics.values()]),
            'macro_f1': np.mean([m['f1'] for m in class_metrics.values()]),
            'macro_auc_roc': np.mean([m['auc_roc'] for m in class_metrics.values()]),
            'macro_auc_pr': np.mean([m['auc_pr'] for m in class_metrics.values()])
        }
        
        # Exact match accuracy (all classes correct)
        exact_match = np.all((predictions > 0.5) == labels, axis=1)
        overall_metrics['exact_match_accuracy'] = np.mean(exact_match)
        
        # Hamming loss (fraction of incorrect labels)
        hamming_loss = np.mean((predictions > 0.5) != labels)
        overall_metrics['hamming_loss'] = hamming_loss
        
        # Calibration metrics
        calibration_metrics = {}
        for class_idx, class_name in self.safety_categories.items():
            class_pred = predictions[:, class_idx]
            class_labels = labels[:, class_idx]
            
            calibration_metrics[class_name] = {
                'ece': CalibrationMetrics.expected_calibration_error(class_pred, class_labels),
                'mce': CalibrationMetrics.maximum_calibration_error(class_pred, class_labels),
                'brier_score': CalibrationMetrics.brier_score(class_pred, class_labels),
                'log_likelihood': CalibrationMetrics.log_likelihood(class_pred, class_labels)
            }
        
        results = {
            'class_metrics': class_metrics,
            'overall_metrics': overall_metrics,
            'calibration_metrics': calibration_metrics,
            'predictions': predictions,
            'labels': labels,
            'test_size': len(test_dataset)
        }
        
        self.results['performance'] = results
        logger.info(f"Performance validation completed. Macro F1: {overall_metrics['macro_f1']:.4f}")
        
        return results
    
    def validate_robustness(
        self,
        model,
        params: Dict[str, Any],
        test_texts: List[str],
        test_labels: np.ndarray,
        robustness_tests: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Robustness validation including adversarial and paraphrase testing.
        
        Args:
            model: JAX model
            params: Model parameters
            test_texts: List of test texts
            test_labels: Test labels
            robustness_tests: List of robustness tests to run
            
        Returns:
            Dictionary with robustness results
        """
        logger.info("Starting robustness validation...")
        
        if robustness_tests is None:
            robustness_tests = ['paraphrase', 'character_swap', 'word_swap']
        
        robustness_results = {}
        
        # Original predictions
        original_predictions = self._predict_texts(model, params, test_texts)
        
        for test_type in robustness_tests:
            logger.info(f"Running {test_type} robustness test...")
            
            if test_type == 'paraphrase':
                # Simple paraphrase test (basic transformations)
                perturbed_texts = self._generate_paraphrases(test_texts)
                
            elif test_type == 'character_swap':
                # Character-level perturbations
                perturbed_texts = self._character_perturbations(test_texts)
                
            elif test_type == 'word_swap':
                # Word-level perturbations
                perturbed_texts = self._word_perturbations(test_texts)
                
            else:
                logger.warning(f"Unknown robustness test: {test_type}")
                continue
            
            # Get predictions on perturbed texts
            perturbed_predictions = self._predict_texts(model, params, perturbed_texts)
            
            # Compute consistency metrics
            consistency_metrics = self._compute_consistency_metrics(
                original_predictions, perturbed_predictions, test_labels
            )
            
            robustness_results[test_type] = consistency_metrics
        
        self.results['robustness'] = robustness_results
        logger.info("Robustness validation completed")
        
        return robustness_results
    
    def validate_fairness(
        self,
        model,
        params: Dict[str, Any],
        test_dataset,
        demographic_attributes: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """
        Fairness validation across demographic groups.
        
        Args:
            model: JAX model
            params: Model parameters
            test_dataset: Test dataset
            demographic_attributes: Dictionary mapping attribute names to values
            
        Returns:
            Dictionary with fairness metrics
        """
        logger.info("Starting fairness validation...")
        
        if demographic_attributes is None:
            # Use synthetic demographic groups for demonstration
            demographic_attributes = {
                'length': ['short', 'medium', 'long'],  # Text length groups
                'complexity': ['simple', 'complex']     # Text complexity groups
            }
        
        # Get predictions
        predictions, labels = self._get_predictions(model, params, test_dataset, batch_size=32)
        
        # Extract texts for demographic analysis
        texts = [example['text'] for example in test_dataset if 'text' in example]
        
        fairness_results = {}
        
        for attribute_name, attribute_values in demographic_attributes.items():
            logger.info(f"Analyzing fairness for attribute: {attribute_name}")
            
            # Assign demographic groups (simplified)
            if attribute_name == 'length':
                groups = self._assign_length_groups(texts, attribute_values)
            elif attribute_name == 'complexity':
                groups = self._assign_complexity_groups(texts, attribute_values)
            else:
                # Skip unknown attributes
                logger.warning(f"Unknown demographic attribute: {attribute_name}")
                continue
            
            # Compute fairness metrics
            attribute_fairness = self._compute_fairness_metrics(
                predictions, labels, groups, attribute_values
            )
            
            fairness_results[attribute_name] = attribute_fairness
        
        self.results['fairness'] = fairness_results
        logger.info("Fairness validation completed")
        
        return fairness_results
    
    def generate_validation_report(
        self,
        output_dir: str = "validation_results",
        include_visualizations: bool = True
    ) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            output_dir: Directory to save results
            include_visualizations: Whether to include plots
            
        Returns:
            Path to the generated report
        """
        logger.info("Generating validation report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_path = output_path / "validation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        
        # Generate markdown report
        report_path = output_path / "validation_report.md"
        with open(report_path, 'w') as f:
            f.write(self._generate_markdown_report())
        
        # Generate visualizations
        if include_visualizations:
            self._generate_visualizations(output_path)
        
        logger.info(f"Validation report generated: {report_path}")
        return str(report_path)
    
    def _get_predictions(
        self,
        model,
        params: Dict[str, Any],
        dataset,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions on dataset."""
        all_predictions = []
        all_labels = []
        
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            # Extract input_ids and labels
            batch_input_ids = []
            batch_labels = []
            
            for j in range(batch_size):
                if i + j >= len(dataset):
                    break
                    
                item = dataset[i + j]
                # Handle different dataset formats
                if isinstance(item, dict):
                    batch_input_ids.append(item['input_ids'])
                    batch_labels.append(item['labels'])
                else:
                    logger.warning(f"Unexpected item type: {type(item)}, skipping batch")
                    break
            
            # Skip empty batches
            if not batch_input_ids:
                continue
                
            # Convert to JAX arrays
            input_ids = jnp.array(batch_input_ids)
            labels = jnp.array(batch_labels)
            
            # Get predictions
            outputs = model.apply(params, input_ids, training=False)
            predictions = jax.nn.sigmoid(outputs['logits'])
            
            all_predictions.append(np.array(predictions))
            all_labels.append(np.array(labels))
        
        return np.vstack(all_predictions), np.vstack(all_labels)
    
    def _predict_texts(
        self,
        model,
        params: Dict[str, Any],
        texts: List[str]
    ) -> np.ndarray:
        """Predict on list of texts."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['data']['tokenizer']
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize
        encoded = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.config['data']['max_length'],
            return_tensors='np'
        )
        
        input_ids = jnp.array(encoded['input_ids'])
        
        # Get predictions
        outputs = model.apply(params, input_ids, training=False)
        predictions = jax.nn.sigmoid(outputs['logits'])
        
        return np.array(predictions)
    
    def _generate_paraphrases(self, texts: List[str]) -> List[str]:
        """Generate simple paraphrases (basic transformations)."""
        paraphrases = []
        for text in texts:
            # Simple transformations
            paraphrase = text.replace("you", "u")  # Text speak
            paraphrase = paraphrase.replace("are", "r")
            paraphrase = paraphrase.upper()  # Case change
            paraphrases.append(paraphrase)
        return paraphrases
    
    def _character_perturbations(self, texts: List[str]) -> List[str]:
        """Generate character-level perturbations."""
        perturbed = []
        for text in texts:
            # Random character swaps
            if len(text) > 2:
                chars = list(text)
                # Swap two random adjacent characters
                import random
                idx = random.randint(0, len(chars) - 2)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                perturbed.append(''.join(chars))
            else:
                perturbed.append(text)
        return perturbed
    
    def _word_perturbations(self, texts: List[str]) -> List[str]:
        """Generate word-level perturbations."""
        perturbed = []
        synonyms = {
            'bad': 'terrible',
            'good': 'great',
            'hate': 'dislike',
            'love': 'like'
        }
        
        for text in texts:
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in synonyms:
                    words[i] = synonyms[word.lower()]
            perturbed.append(' '.join(words))
        return perturbed
    
    def _compute_consistency_metrics(
        self,
        original_predictions: np.ndarray,
        perturbed_predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute consistency metrics between original and perturbed predictions."""
        # Prediction consistency (how often predictions are the same)
        binary_original = (original_predictions > 0.5).astype(int)
        binary_perturbed = (perturbed_predictions > 0.5).astype(int)
        
        prediction_consistency = np.mean(binary_original == binary_perturbed)
        
        # Probability consistency (L1 distance)
        prob_consistency = 1 - np.mean(np.abs(original_predictions - perturbed_predictions))
        
        # Accuracy drop
        original_accuracy = accuracy_score(
            labels.flatten(), binary_original.flatten()
        )
        perturbed_accuracy = accuracy_score(
            labels.flatten(), binary_perturbed.flatten()
        )
        accuracy_drop = original_accuracy - perturbed_accuracy
        
        return {
            'prediction_consistency': float(prediction_consistency),
            'probability_consistency': float(prob_consistency),
            'original_accuracy': float(original_accuracy),
            'perturbed_accuracy': float(perturbed_accuracy),
            'accuracy_drop': float(accuracy_drop)
        }
    
    def _assign_length_groups(self, texts: List[str], groups: List[str]) -> List[str]:
        """Assign texts to length-based groups."""
        lengths = [len(text.split()) for text in texts]
        length_groups = []
        
        # Define thresholds
        short_threshold = np.percentile(lengths, 33)
        long_threshold = np.percentile(lengths, 67)
        
        for length in lengths:
            if length <= short_threshold:
                length_groups.append('short')
            elif length >= long_threshold:
                length_groups.append('long')
            else:
                length_groups.append('medium')
        
        return length_groups
    
    def _assign_complexity_groups(self, texts: List[str], groups: List[str]) -> List[str]:
        """Assign texts to complexity-based groups."""
        complexities = []
        
        for text in texts:
            # Simple complexity measure: average word length
            words = text.split()
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                complexities.append(avg_word_length)
            else:
                complexities.append(0)
        
        # Assign groups based on median
        median_complexity = np.median(complexities)
        complexity_groups = []
        
        for complexity in complexities:
            if complexity >= median_complexity:
                complexity_groups.append('complex')
            else:
                complexity_groups.append('simple')
        
        return complexity_groups
    
    def _compute_fairness_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        groups: List[str],
        group_names: List[str]
    ) -> Dict[str, Any]:
        """Compute fairness metrics across groups."""
        fairness_metrics = {}
        
        for group_name in group_names:
            group_mask = [g == group_name for g in groups]
            group_predictions = predictions[group_mask]
            group_labels = labels[group_mask]
            
            if len(group_predictions) > 0:
                # Compute metrics for this group
                group_binary_pred = (group_predictions > 0.5).astype(int)
                
                fairness_metrics[group_name] = {
                    'size': int(len(group_predictions)),
                    'positive_rate': float(np.mean(group_binary_pred)),
                    'accuracy': float(accuracy_score(
                        group_labels.flatten(), group_binary_pred.flatten()
                    )),
                    'precision': float(precision_score(
                        group_labels.flatten(), group_binary_pred.flatten(),
                        average='macro', zero_division=0
                    )),
                    'recall': float(recall_score(
                        group_labels.flatten(), group_binary_pred.flatten(),
                        average='macro', zero_division=0
                    ))
                }
        
        # Compute fairness disparities
        if len(fairness_metrics) >= 2:
            group_values = list(fairness_metrics.values())
            
            # Demographic parity: difference in positive rates
            positive_rates = [gv['positive_rate'] for gv in group_values]
            fairness_metrics['demographic_parity_diff'] = float(
                max(positive_rates) - min(positive_rates)
            )
            
            # Equalized odds: difference in accuracy
            accuracies = [gv['accuracy'] for gv in group_values]
            fairness_metrics['equalized_odds_diff'] = float(
                max(accuracies) - min(accuracies)
            )
        
        return fairness_metrics
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown validation report."""
        report = ["# Safety Text Classifier - Validation Report\n"]
        
        # Performance section
        if 'performance' in self.results:
            perf = self.results['performance']
            report.append("## Model Performance\n")
            
            report.append("### Overall Metrics\n")
            overall = perf['overall_metrics']
            for metric, value in overall.items():
                report.append(f"- **{metric.replace('_', ' ').title()}**: {value:.4f}")
            report.append("")
            
            report.append("### Per-Class Metrics\n")
            for class_name, metrics in perf['class_metrics'].items():
                report.append(f"#### {class_name.replace('_', ' ').title()}")
                for metric, value in metrics.items():
                    report.append(f"- {metric.upper()}: {value:.4f}")
                report.append("")
        
        # Robustness section
        if 'robustness' in self.results:
            report.append("## Robustness Analysis\n")
            robust = self.results['robustness']
            
            for test_type, metrics in robust.items():
                report.append(f"### {test_type.replace('_', ' ').title()} Test")
                for metric, value in metrics.items():
                    report.append(f"- **{metric.replace('_', ' ').title()}**: {value:.4f}")
                report.append("")
        
        # Fairness section
        if 'fairness' in self.results:
            report.append("## Fairness Analysis\n")
            fair = self.results['fairness']
            
            for attribute, metrics in fair.items():
                report.append(f"### {attribute.title()} Fairness")
                if isinstance(metrics, dict):
                    for group, values in metrics.items():
                        if isinstance(values, dict):
                            report.append(f"#### {group.title()}")
                            for metric, value in values.items():
                                report.append(f"- {metric.replace('_', ' ').title()}: {value}")
                report.append("")
        
        return "\n".join(report)
    
    def _generate_visualizations(self, output_dir: Path):
        """Generate visualization plots."""
        if 'performance' in self.results:
            self._plot_performance_metrics(output_dir)
        
        if 'robustness' in self.results:
            self._plot_robustness_metrics(output_dir)
        
        if 'fairness' in self.results:
            self._plot_fairness_metrics(output_dir)
    
    def _plot_performance_metrics(self, output_dir: Path):
        """Plot performance metrics."""
        perf = self.results['performance']
        
        # Per-class F1 scores
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = list(perf['class_metrics'].keys())
        f1_scores = [perf['class_metrics'][c]['f1'] for c in classes]
        
        bars = ax.bar(classes, f1_scores)
        ax.set_title('Per-Class F1 Scores')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_f1_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion matrices (if predictions available)
        if 'predictions' in perf and 'labels' in perf:
            predictions = perf['predictions']
            labels = perf['labels']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, (class_idx, class_name) in enumerate(self.safety_categories.items()):
                cm = confusion_matrix(
                    labels[:, class_idx], 
                    (predictions[:, class_idx] > 0.5).astype(int)
                )
                
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], 
                           cmap='Blues', cbar=False)
                axes[i].set_title(f'{class_name.replace("_", " ").title()}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_robustness_metrics(self, output_dir: Path):
        """Plot robustness metrics."""
        robust = self.results['robustness']
        
        # Consistency across perturbation types
        test_types = list(robust.keys())
        consistency_scores = [robust[t]['prediction_consistency'] for t in test_types]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(test_types, consistency_scores)
        ax.set_title('Prediction Consistency Across Robustness Tests')
        ax.set_ylabel('Consistency Score')
        ax.set_ylim(0, 1)
        
        for bar, score in zip(bars, consistency_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fairness_metrics(self, output_dir: Path):
        """Plot fairness metrics."""
        fair = self.results['fairness']
        
        for attribute, metrics in fair.items():
            if isinstance(metrics, dict):
                # Filter out summary metrics
                group_metrics = {k: v for k, v in metrics.items() 
                               if isinstance(v, dict)}
                
                if group_metrics:
                    groups = list(group_metrics.keys())
                    accuracies = [group_metrics[g]['accuracy'] for g in groups]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    bars = ax.bar(groups, accuracies)
                    ax.set_title(f'Accuracy by {attribute.title()}')
                    ax.set_ylabel('Accuracy')
                    ax.set_ylim(0, 1)
                    
                    for bar, acc in zip(bars, accuracies):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f'fairness_{attribute}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()


if __name__ == "__main__":
    # Test the validation framework
    logging.basicConfig(level=logging.INFO)
    
    # This would typically be run with actual model and data
    validator = ModelValidator()
    print("Model validation framework initialized successfully!")
    print("Use this framework to comprehensively validate your safety classifier.")
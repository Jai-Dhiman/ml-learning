#!/usr/bin/env python3
"""
Model Validation Script

Comprehensive validation script that tests all components of the Safety Text Classifier
including performance, robustness, fairness, and calibration.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.validation import ModelValidator
from src.models.utils import ModelLoader
from src.data.dataset_loader import create_data_loaders
from src.models.calibration import calibrate_model_predictions, CalibrationMetrics


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Safety Text Classifier Validation"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model",
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_results",
        help="Directory to save validation results"
    )
    
    parser.add_argument(
        "--tests",
        type=str,
        nargs="+",
        default=["performance", "robustness", "fairness", "calibration"],
        choices=["performance", "robustness", "fairness", "calibration", "all"],
        help="Which validation tests to run"
    )
    
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualization plots"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Starting Safety Text Classifier Validation...")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Tests: {args.tests}")
    
    try:
        # Initialize validator
        validator = ModelValidator(args.config)
        
        # Load model
        logger.info("Loading model...")
        try:
            model, params, metadata = ModelLoader.load_model_for_inference(
                args.checkpoint, args.config
            )
            logger.info(f"Loaded model: {metadata.model_name} v{metadata.model_version}")
            logger.info(f"Training steps: {metadata.training_steps}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Creating model with random parameters for validation framework testing...")
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            from src.models.transformer import create_model, initialize_model
            import jax
            
            model = create_model(config)
            rng = jax.random.PRNGKey(42)
            params = initialize_model(model, rng)
            metadata = None
        
        # Load test dataset
        logger.info("Loading test dataset...")
        train_dataset, val_dataset, test_dataset = create_data_loaders(args.config)
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        # Determine which tests to run
        tests_to_run = args.tests
        if "all" in tests_to_run:
            tests_to_run = ["performance", "robustness", "fairness", "calibration"]
        
        # Run validation tests
        results = {}
        
        if "performance" in tests_to_run:
            logger.info("Running performance validation...")
            perf_results = validator.validate_model_performance(
                model, params, test_dataset, batch_size=16
            )
            results['performance'] = perf_results
            
            # Print summary
            overall = perf_results['overall_metrics']
            logger.info(f"Performance Summary:")
            logger.info(f"  Macro F1: {overall['macro_f1']:.4f}")
            logger.info(f"  Macro Accuracy: {overall['macro_accuracy']:.4f}")
            logger.info(f"  Exact Match: {overall['exact_match_accuracy']:.4f}")
        
        if "robustness" in tests_to_run:
            logger.info("Running robustness validation...")
            
            # Extract texts from test dataset (limited sample for demo)
            test_texts = []
            test_labels = []
            for i, item in enumerate(test_dataset):
                if i >= 100:  # Limit for demo
                    break
                if 'text' in item:
                    test_texts.append(item['text'])
                    test_labels.append(item['labels'])
                elif 'input_ids' in item:
                    # If no text, skip robustness testing
                    logger.warning("No text found in dataset, skipping robustness tests")
                    break
            
            if test_texts:
                import numpy as np
                robust_results = validator.validate_robustness(
                    model, params, test_texts, np.array(test_labels)
                )
                results['robustness'] = robust_results
                
                # Print summary
                logger.info("Robustness Summary:")
                for test_type, metrics in robust_results.items():
                    logger.info(f"  {test_type}: {metrics['prediction_consistency']:.4f} consistency")
        
        if "fairness" in tests_to_run:
            logger.info("Running fairness validation...")
            fair_results = validator.validate_fairness(
                model, params, test_dataset
            )
            results['fairness'] = fair_results
            
            # Print summary
            logger.info("Fairness Summary:")
            for attribute, metrics in fair_results.items():
                if 'demographic_parity_diff' in metrics:
                    logger.info(f"  {attribute} demographic parity diff: {metrics['demographic_parity_diff']:.4f}")
        
        if "calibration" in tests_to_run:
            logger.info("Running calibration analysis...")
            
            # Get predictions for calibration analysis
            if 'performance' in results:
                predictions = results['performance']['predictions']
                labels = results['performance']['labels']
            else:
                # Get predictions specifically for calibration
                predictions, labels = validator._get_predictions(
                    model, params, test_dataset, batch_size=16
                )
            
            calibration_results = {}
            
            # Compute calibration metrics for each class
            safety_categories = {0: 'hate_speech', 1: 'self_harm', 2: 'dangerous_advice', 3: 'harassment'}
            
            for class_idx, class_name in safety_categories.items():
                class_pred = predictions[:, class_idx]
                class_labels = labels[:, class_idx]
                
                # Basic calibration metrics
                ece = CalibrationMetrics.expected_calibration_error(class_pred, class_labels)
                mce = CalibrationMetrics.maximum_calibration_error(class_pred, class_labels)
                brier = CalibrationMetrics.brier_score(class_pred, class_labels)
                
                calibration_results[class_name] = {
                    'ece': float(ece),
                    'mce': float(mce),
                    'brier_score': float(brier)
                }
                
                logger.info(f"  {class_name} - ECE: {ece:.4f}, MCE: {mce:.4f}")
            
            results['calibration'] = calibration_results
        
        # Generate comprehensive report
        logger.info("Generating validation report...")
        report_path = validator.generate_validation_report(
            args.output_dir,
            include_visualizations=not args.no_visualizations
        )
        
        # Save detailed results
        output_path = Path(args.output_dir)
        detailed_results_path = output_path / "detailed_results.json"
        
        def make_serializable(obj):
            """Convert numpy arrays to lists for JSON serialization."""
            import numpy as np
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        with open(detailed_results_path, 'w') as f:
            json.dump(make_serializable(results), f, indent=2)
        
        # Print summary
        logger.info("✅ Validation completed successfully!")
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Detailed results saved to: {detailed_results_path}")
        
        if 'performance' in results:
            overall = results['performance']['overall_metrics']
            logger.info(f"Final Performance Score: {overall['macro_f1']:.4f} F1")
        
        # Check if model meets PRD requirements
        prd_check_results = check_prd_requirements(results)
        logger.info("\n" + "="*50)
        logger.info("PRD REQUIREMENTS CHECK")
        logger.info("="*50)
        for requirement, status in prd_check_results.items():
            status_emoji = "✅" if status['passed'] else "❌"
            logger.info(f"{status_emoji} {requirement}: {status['message']}")
        
        passed_count = sum(1 for s in prd_check_results.values() if s['passed'])
        total_count = len(prd_check_results)
        logger.info(f"\nOverall: {passed_count}/{total_count} requirements met")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


def check_prd_requirements(results: dict) -> dict:
    """Check if model meets PRD requirements."""
    checks = {}
    
    # Performance requirements
    if 'performance' in results:
        overall = results['performance']['overall_metrics']
        
        # Target: >85% accuracy
        accuracy = overall.get('macro_accuracy', 0)
        checks['85% Accuracy Target'] = {
            'passed': accuracy > 0.85,
            'message': f"Achieved {accuracy:.1%} (target: >85%)"
        }
        
        # Target: >0.80 F1 per category
        class_metrics = results['performance']['class_metrics']
        min_f1 = min(metrics['f1'] for metrics in class_metrics.values())
        checks['0.80 F1 Per Category'] = {
            'passed': min_f1 > 0.80,
            'message': f"Minimum F1: {min_f1:.3f} (target: >0.80)"
        }
    
    # Calibration requirements
    if 'calibration' in results:
        # Target: ECE < 0.05
        max_ece = max(metrics['ece'] for metrics in results['calibration'].values())
        checks['Calibration Error < 0.05'] = {
            'passed': max_ece < 0.05,
            'message': f"Maximum ECE: {max_ece:.4f} (target: <0.05)"
        }
    
    # Robustness requirements
    if 'robustness' in results:
        # Target: >90% consistency on paraphrases
        if 'paraphrase' in results['robustness']:
            consistency = results['robustness']['paraphrase']['prediction_consistency']
            checks['Paraphrase Consistency > 90%'] = {
                'passed': consistency > 0.90,
                'message': f"Achieved {consistency:.1%} (target: >90%)"
            }
    
    # Fairness requirements
    if 'fairness' in results:
        # Target: <5% difference in classification rates
        fairness_passed = True
        max_disparity = 0
        
        for attr_metrics in results['fairness'].values():
            if isinstance(attr_metrics, dict) and 'demographic_parity_diff' in attr_metrics:
                disparity = attr_metrics['demographic_parity_diff']
                max_disparity = max(max_disparity, disparity)
                if disparity > 0.05:
                    fairness_passed = False
        
        checks['Demographic Parity < 5%'] = {
            'passed': fairness_passed,
            'message': f"Maximum disparity: {max_disparity:.1%} (target: <5%)"
        }
    
    return checks


if __name__ == "__main__":
    main()
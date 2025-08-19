"""
Model Calibration and Confidence Estimation

Provides tools for calibrating model predictions and estimating prediction confidence
for the Safety Text Classifier.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional, Union
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """
    Temperature scaling for model calibration.
    
    Temperature scaling applies a single scalar parameter to the logits
    to improve calibration while preserving accuracy.
    """
    
    def __init__(self):
        """Initialize temperature scaling."""
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        method: str = "NLL"
    ) -> float:
        """
        Fit the temperature parameter on validation data.
        
        Args:
            logits: Model logits (before sigmoid)
            labels: True binary labels
            method: Optimization method ("NLL" or "ECE")
            
        Returns:
            Optimal temperature value
        """
        if method == "NLL":
            # Optimize negative log-likelihood
            def objective(temp):
                if temp <= 0:
                    return float('inf')
                scaled_logits = logits / temp
                probs = 1 / (1 + np.exp(-scaled_logits))  # Sigmoid
                probs = np.clip(probs, 1e-7, 1 - 1e-7)  # Avoid log(0)
                return -np.sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            
        elif method == "ECE":
            # Optimize Expected Calibration Error
            def objective(temp):
                if temp <= 0:
                    return float('inf')
                scaled_logits = logits / temp
                probs = 1 / (1 + np.exp(-scaled_logits))  # Sigmoid
                return self._compute_ece(probs, labels)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Find optimal temperature
        result = minimize_scalar(objective, bounds=(0.01, 10.0), method='bounded')
        self.temperature = result.x
        self.is_fitted = True
        
        logger.info(f"Optimal temperature: {self.temperature:.4f}")
        return self.temperature
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Input logits
            
        Returns:
            Temperature-scaled probabilities
        """
        if not self.is_fitted:
            logger.warning("Temperature not fitted, using default temperature=1.0")
        
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))  # Sigmoid
    
    def _compute_ece(
        self, 
        probabilities: np.ndarray, 
        labels: np.ndarray, 
        n_bins: int = 15
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class PlattScaling:
    """
    Platt scaling for probability calibration.
    
    Fits a sigmoid function to the logits to better calibrate probabilities.
    """
    
    def __init__(self):
        """Initialize Platt scaling."""
        self.A = 1.0
        self.B = 0.0
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """
        Fit Platt scaling parameters.
        
        Args:
            logits: Model logits
            labels: True binary labels
            
        Returns:
            Tuple of (A, B) parameters
        """
        # Implement Platt's method
        # P(y=1|f) = 1 / (1 + exp(A*f + B))
        
        def objective(params):
            A, B = params
            scaled_logits = A * logits + B
            probs = 1 / (1 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        
        from scipy.optimize import minimize
        result = minimize(objective, [1.0, 0.0], method='BFGS')
        
        self.A, self.B = result.x
        self.is_fitted = True
        
        logger.info(f"Platt scaling parameters: A={self.A:.4f}, B={self.B:.4f}")
        return self.A, self.B
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to logits.
        
        Args:
            logits: Input logits
            
        Returns:
            Platt-scaled probabilities
        """
        if not self.is_fitted:
            logger.warning("Platt scaling not fitted, using default parameters")
        
        scaled_logits = self.A * logits + self.B
        return 1 / (1 + np.exp(-scaled_logits))


class IsotonicCalibration:
    """
    Isotonic regression for probability calibration.
    """
    
    def __init__(self):
        """Initialize isotonic calibration."""
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
    
    def fit(self, probabilities: np.ndarray, labels: np.ndarray):
        """
        Fit isotonic regression.
        
        Args:
            probabilities: Model probabilities (after sigmoid)
            labels: True binary labels
        """
        self.isotonic.fit(probabilities, labels)
        self.is_fitted = True
        logger.info("Isotonic calibration fitted")
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.
        
        Args:
            probabilities: Input probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Isotonic calibration not fitted")
        
        return self.isotonic.transform(probabilities)


class ConfidenceEstimator:
    """
    Estimates prediction confidence using various methods.
    """
    
    def __init__(self):
        """Initialize confidence estimator."""
        self.calibrators = {}
    
    def add_calibrator(self, name: str, calibrator):
        """Add a calibration method."""
        self.calibrators[name] = calibrator
    
    def entropy_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Compute confidence based on prediction entropy.
        
        Args:
            probabilities: Model probabilities
            
        Returns:
            Confidence scores (higher = more confident)
        """
        # For binary classification: H = -p*log(p) - (1-p)*log(1-p)
        probs = np.clip(probabilities, 1e-7, 1 - 1e-7)
        entropy = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
        max_entropy = np.log(2)  # Maximum entropy for binary classification
        confidence = 1 - (entropy / max_entropy)
        return confidence
    
    def max_probability_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Use maximum probability as confidence.
        
        Args:
            probabilities: Model probabilities
            
        Returns:
            Confidence scores
        """
        return np.maximum(probabilities, 1 - probabilities)
    
    def margin_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Compute confidence based on decision margin.
        
        Args:
            probabilities: Model probabilities
            
        Returns:
            Confidence scores
        """
        return np.abs(probabilities - 0.5) * 2  # Scale to [0, 1]
    
    def ensemble_confidence(
        self,
        predictions_list: List[np.ndarray],
        method: str = "variance"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ensemble-based confidence.
        
        Args:
            predictions_list: List of predictions from different models/runs
            method: Method to compute confidence ("variance" or "agreement")
            
        Returns:
            Tuple of (mean_predictions, confidence_scores)
        """
        predictions = np.stack(predictions_list, axis=0)
        mean_predictions = np.mean(predictions, axis=0)
        
        if method == "variance":
            # High variance = low confidence
            variance = np.var(predictions, axis=0)
            confidence = 1 / (1 + variance)  # Transform to [0, 1]
            
        elif method == "agreement":
            # Agreement-based confidence
            # For binary: how often do models agree on the prediction
            binary_preds = (predictions > 0.5).astype(int)
            agreement = np.mean(binary_preds == binary_preds[0], axis=0)
            confidence = agreement
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return mean_predictions, confidence
    
    def bayesian_confidence(
        self,
        model,
        params: Dict[str, Any],
        input_ids: jnp.ndarray,
        num_samples: int = 100,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate confidence using Monte Carlo Dropout (approximate Bayesian inference).
        
        Args:
            model: JAX model with dropout
            params: Model parameters
            input_ids: Input token IDs
            num_samples: Number of MC samples
            rng_key: Random key for dropout
            
        Returns:
            Tuple of (mean_predictions, confidence_scores)
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        
        predictions = []
        
        for i in range(num_samples):
            rng_key, dropout_key = jax.random.split(rng_key)
            
            # Forward pass with dropout enabled
            outputs = model.apply(
                params,
                input_ids,
                training=True,  # Enable dropout
                rngs={'dropout': dropout_key}
            )
            
            # Apply sigmoid to get probabilities
            probs = jax.nn.sigmoid(outputs['logits'])
            predictions.append(np.array(probs))
        
        # Compute mean and variance
        predictions = np.stack(predictions, axis=0)
        mean_predictions = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = 1 / (1 + variance)
        
        return mean_predictions, confidence


class CalibrationMetrics:
    """
    Compute various calibration metrics.
    """
    
    @staticmethod
    def expected_calibration_error(
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            probabilities: Predicted probabilities
            labels: True labels
            n_bins: Number of bins for calibration
            
        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def maximum_calibration_error(
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    @staticmethod
    def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Compute Brier score."""
        return brier_score_loss(labels, probabilities)
    
    @staticmethod
    def log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Compute log-likelihood."""
        return -log_loss(labels, probabilities)
    
    @staticmethod
    def reliability_diagram(
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a reliability diagram.
        
        Args:
            probabilities: Predicted probabilities
            labels: True labels
            n_bins: Number of bins
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(labels[in_bin].mean())
                bin_confidences.append(probabilities[in_bin].mean())
                bin_counts.append(prop_in_bin)
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Reliability diagram
        ax1.bar(bin_centers, bin_accuracies, width=0.1, alpha=0.7, 
               edgecolor='black', label='Accuracy')
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax1.plot(bin_confidences, bin_accuracies, 'bo-', label='Model')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram
        ax2.bar(bin_centers, bin_counts, width=0.1, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def calibrate_model_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    method: str = "temperature",
    **kwargs
) -> Tuple[np.ndarray, Any]:
    """
    Calibrate model predictions using the specified method.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        labels: True binary labels
        method: Calibration method ("temperature", "platt", "isotonic")
        **kwargs: Additional arguments for calibration methods
        
    Returns:
        Tuple of (calibrated_predictions, calibrator)
    """
    if method == "temperature":
        calibrator = TemperatureScaling()
        calibrator.fit(predictions, labels, **kwargs)
        calibrated = calibrator.transform(predictions)
        
    elif method == "platt":
        calibrator = PlattScaling()
        calibrator.fit(predictions, labels)
        calibrated = calibrator.transform(predictions)
        
    elif method == "isotonic":
        calibrator = IsotonicCalibration()
        # For isotonic, we need probabilities, not logits
        if np.any((predictions < 0) | (predictions > 1)):
            probs = 1 / (1 + np.exp(-predictions))  # Sigmoid if logits
        else:
            probs = predictions
        calibrator.fit(probs, labels)
        calibrated = calibrator.transform(probs)
        
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return calibrated, calibrator


if __name__ == "__main__":
    # Test calibration tools
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate uncalibrated predictions (overconfident)
    logits = np.random.randn(n_samples) * 2
    true_probs = 1 / (1 + np.exp(-logits))
    labels = np.random.binomial(1, true_probs)
    
    # Make predictions overconfident
    overconfident_logits = logits * 2  # Scale up to make overconfident
    overconfident_probs = 1 / (1 + np.exp(-overconfident_logits))
    
    print(f"Original ECE: {CalibrationMetrics.expected_calibration_error(overconfident_probs, labels):.4f}")
    
    # Test temperature scaling
    calibrated_probs, calibrator = calibrate_model_predictions(
        overconfident_logits, labels, method="temperature"
    )
    
    print(f"Calibrated ECE: {CalibrationMetrics.expected_calibration_error(calibrated_probs, labels):.4f}")
    print(f"Temperature: {calibrator.temperature:.4f}")
    
    # Test confidence estimation
    estimator = ConfidenceEstimator()
    entropy_conf = estimator.entropy_confidence(calibrated_probs)
    max_prob_conf = estimator.max_probability_confidence(calibrated_probs)
    
    print(f"Mean entropy confidence: {entropy_conf.mean():.4f}")
    print(f"Mean max prob confidence: {max_prob_conf.mean():.4f}")
    
    print("Calibration tools test completed successfully!")
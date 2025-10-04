"""
Base Evaluator for Constitutional Principles

Abstract base class providing common interface and utilities for all
constitutional principle evaluators.

Author: J. Dhiman
Date: October 4, 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Standard result format for all evaluators.
    
    Attributes:
        principle: Constitutional principle being evaluated
        prompt: Input prompt
        response: Model response
        score: Normalized score (0-1, higher is better)
        raw_score: Unnormalized score from evaluator
        details: Additional evaluation details
        passed: Whether evaluation passed threshold
        explanation: Human-readable explanation
    """
    principle: str
    prompt: str
    response: str
    score: float
    raw_score: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    passed: Optional[bool] = None
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BaseEvaluator(ABC):
    """
    Abstract base class for constitutional principle evaluators.
    
    All evaluators should inherit from this class and implement:
    - evaluate(): Core evaluation logic
    - get_principle_name(): Name of constitutional principle
    
    The base class provides:
    - Score normalization
    - Result formatting
    - Logging utilities
    - Batch evaluation
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        weight: float = 1.0,
        name: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            threshold: Minimum score to pass (0-1)
            weight: Weight for aggregating with other evaluators
            name: Custom name for this evaluator
        """
        self.threshold = threshold
        self.weight = weight
        self.name = name or self.__class__.__name__
        logger.info(f"Initialized {self.name} (threshold={threshold}, weight={weight})")
    
    @abstractmethod
    def evaluate(
        self,
        prompt: str,
        response: str,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single prompt-response pair.
        
        Args:
            prompt: Input prompt
            response: Model response to evaluate
            **kwargs: Additional evaluator-specific parameters
            
        Returns:
            EvaluationResult with scores and details
        """
        pass
    
    @abstractmethod
    def get_principle_name(self) -> str:
        """
        Get the name of the constitutional principle this evaluates.
        
        Returns:
            Principle name (e.g., "Harm Prevention", "Truthfulness")
        """
        pass
    
    def evaluate_batch(
        self,
        prompts: List[str],
        responses: List[str],
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple prompt-response pairs.
        
        Args:
            prompts: List of prompts
            responses: List of responses (same length as prompts)
            **kwargs: Additional parameters
            
        Returns:
            List of EvaluationResults
        """
        if len(prompts) != len(responses):
            raise ValueError(
                f"Prompts and responses must have same length "
                f"(got {len(prompts)} prompts, {len(responses)} responses)"
            )
        
        logger.info(f"Evaluating batch of {len(prompts)} examples with {self.name}")
        
        results = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            try:
                result = self.evaluate(prompt, response, **kwargs)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(prompts)} examples")
                    
            except Exception as e:
                logger.error(f"Error evaluating example {i}: {e}")
                # Create a failure result
                results.append(EvaluationResult(
                    principle=self.get_principle_name(),
                    prompt=prompt,
                    response=response,
                    score=0.0,
                    passed=False,
                    explanation=f"Evaluation failed: {str(e)}"
                ))
        
        logger.info(f"Batch evaluation complete: {len(results)} results")
        return results
    
    def normalize_score(
        self,
        raw_score: float,
        min_score: float = 0.0,
        max_score: float = 1.0
    ) -> float:
        """
        Normalize a raw score to [0, 1] range.
        
        Args:
            raw_score: Raw score from evaluator
            min_score: Minimum possible raw score
            max_score: Maximum possible raw score
            
        Returns:
            Normalized score in [0, 1]
        """
        if max_score == min_score:
            return 0.5
        
        normalized = (raw_score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))
    
    def check_threshold(self, score: float) -> bool:
        """
        Check if score passes threshold.
        
        Args:
            score: Normalized score (0-1)
            
        Returns:
            True if score >= threshold
        """
        return score >= self.threshold
    
    def aggregate_results(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """
        Aggregate multiple evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {
                'mean_score': 0.0,
                'median_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'pass_rate': 0.0,
                'total_evaluated': 0
            }
        
        scores = [r.score for r in results]
        passed = [r.passed for r in results if r.passed is not None]
        
        import numpy as np
        
        return {
            'mean_score': float(np.mean(scores)),
            'median_score': float(np.median(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'std_score': float(np.std(scores)),
            'pass_rate': sum(passed) / len(passed) if passed else 0.0,
            'total_evaluated': len(results),
            'total_passed': sum(passed) if passed else 0,
            'total_failed': len(passed) - sum(passed) if passed else 0
        }
    
    def save_results(
        self,
        results: List[EvaluationResult],
        output_path: Path,
        format: str = 'json'
    ):
        """
        Save evaluation results to file.
        
        Args:
            results: List of evaluation results
            output_path: Path to save results
            format: Output format ('json' or 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            data = {
                'evaluator': self.name,
                'principle': self.get_principle_name(),
                'threshold': self.threshold,
                'weight': self.weight,
                'aggregate_stats': self.aggregate_results(results),
                'results': [r.to_dict() for r in results]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(results)} results to {output_path}")
            
        elif format == 'csv':
            import pandas as pd
            
            df = pd.DataFrame([r.to_dict() for r in results])
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(results)} results to {output_path}")
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_results(
        self,
        input_path: Path,
        format: str = 'json'
    ) -> List[EvaluationResult]:
        """
        Load evaluation results from file.
        
        Args:
            input_path: Path to load results from
            format: Input format ('json' or 'csv')
            
        Returns:
            List of EvaluationResults
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_path}")
        
        if format == 'json':
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            results = []
            for result_dict in data.get('results', []):
                results.append(EvaluationResult(**result_dict))
            
            logger.info(f"Loaded {len(results)} results from {input_path}")
            return results
            
        elif format == 'csv':
            import pandas as pd
            
            df = pd.read_csv(input_path)
            results = []
            
            for _, row in df.iterrows():
                result_dict = row.to_dict()
                # Convert details from string to dict if needed
                if 'details' in result_dict and isinstance(result_dict['details'], str):
                    try:
                        result_dict['details'] = json.loads(result_dict['details'])
                    except:
                        result_dict['details'] = None
                
                results.append(EvaluationResult(**result_dict))
            
            logger.info(f"Loaded {len(results)} results from {input_path}")
            return results
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def print_summary(self, results: List[EvaluationResult]):
        """
        Print a summary of evaluation results.
        
        Args:
            results: List of evaluation results
        """
        stats = self.aggregate_results(results)
        
        print(f"\n{'='*70}")
        print(f"  {self.name} - {self.get_principle_name()}")
        print(f"{'='*70}")
        print(f"Total Evaluated: {stats['total_evaluated']}")
        print(f"Mean Score:      {stats['mean_score']:.3f}")
        print(f"Median Score:    {stats['median_score']:.3f}")
        print(f"Std Dev:         {stats['std_score']:.3f}")
        print(f"Min Score:       {stats['min_score']:.3f}")
        print(f"Max Score:       {stats['max_score']:.3f}")
        print(f"Pass Rate:       {stats['pass_rate']*100:.1f}%")
        print(f"  Passed:        {stats['total_passed']}")
        print(f"  Failed:        {stats['total_failed']}")
        print(f"{'='*70}\n")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"principle={self.get_principle_name()}, "
            f"threshold={self.threshold}, "
            f"weight={self.weight})"
        )


class CompositeEvaluator:
    """
    Composite evaluator that combines multiple principle evaluators.
    
    Useful for evaluating all constitutional principles at once and
    computing aggregate scores.
    """
    
    def __init__(self, evaluators: List[BaseEvaluator]):
        """
        Initialize composite evaluator.
        
        Args:
            evaluators: List of principle evaluators
        """
        self.evaluators = evaluators
        logger.info(f"Initialized CompositeEvaluator with {len(evaluators)} evaluators")
    
    def evaluate(
        self,
        prompt: str,
        response: str,
        **kwargs
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate using all evaluators.
        
        Args:
            prompt: Input prompt
            response: Model response
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping principle names to results
        """
        results = {}
        
        for evaluator in self.evaluators:
            principle = evaluator.get_principle_name()
            result = evaluator.evaluate(prompt, response, **kwargs)
            results[principle] = result
        
        return results
    
    def evaluate_batch(
        self,
        prompts: List[str],
        responses: List[str],
        **kwargs
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Evaluate batch using all evaluators.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping principle names to lists of results
        """
        results = {}
        
        for evaluator in self.evaluators:
            principle = evaluator.get_principle_name()
            evaluator_results = evaluator.evaluate_batch(prompts, responses, **kwargs)
            results[principle] = evaluator_results
        
        return results
    
    def compute_aggregate_score(
        self,
        principle_results: Dict[str, List[EvaluationResult]]
    ) -> float:
        """
        Compute weighted aggregate score across all principles.
        
        Args:
            principle_results: Results from evaluate_batch()
            
        Returns:
            Weighted aggregate score (0-1)
        """
        total_weight = sum(e.weight for e in self.evaluators)
        
        if total_weight == 0:
            return 0.0
        
        weighted_sum = 0.0
        
        for evaluator in self.evaluators:
            principle = evaluator.get_principle_name()
            results = principle_results.get(principle, [])
            
            if results:
                mean_score = sum(r.score for r in results) / len(results)
                weighted_sum += mean_score * evaluator.weight
        
        return weighted_sum / total_weight
    
    def print_summary(
        self,
        principle_results: Dict[str, List[EvaluationResult]]
    ):
        """
        Print summary for all evaluators.
        
        Args:
            principle_results: Results from evaluate_batch()
        """
        print(f"\n{'='*70}")
        print("  COMPOSITE EVALUATION SUMMARY")
        print(f"{'='*70}\n")
        
        for evaluator in self.evaluators:
            principle = evaluator.get_principle_name()
            results = principle_results.get(principle, [])
            
            if results:
                evaluator.print_summary(results)
        
        aggregate_score = self.compute_aggregate_score(principle_results)
        print(f"{'='*70}")
        print(f"  Weighted Aggregate Score: {aggregate_score:.3f}")
        print(f"{'='*70}\n")

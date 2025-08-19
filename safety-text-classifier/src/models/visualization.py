"""
Attention Visualization and Explainability Tools

Provides utilities for visualizing attention patterns, feature importance,
and model decision explanations for the Safety Text Classifier.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualizes attention patterns and model explanations.
    """
    
    def __init__(self, tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the attention visualizer.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extract_attention_patterns(
        self,
        attention_weights: List[jnp.ndarray],
        input_ids: jnp.ndarray,
        aggregate_method: str = "mean"
    ) -> Dict[str, np.ndarray]:
        """
        Extract and aggregate attention patterns from model outputs.
        
        Args:
            attention_weights: List of attention weights from each transformer layer
            input_ids: Input token IDs
            aggregate_method: Method to aggregate attention ("mean", "max", "last")
            
        Returns:
            Dictionary containing processed attention patterns
        """
        if not attention_weights:
            raise ValueError("No attention weights provided")
        
        # Convert to numpy for easier processing
        attention_arrays = [np.array(attn) for attn in attention_weights]
        
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        batch_size, num_heads, seq_len, _ = attention_arrays[0].shape
        
        # Aggregate across layers
        if aggregate_method == "mean":
            # Average across all layers
            aggregated = np.mean(attention_arrays, axis=0)
        elif aggregate_method == "max":
            # Take maximum attention across layers
            aggregated = np.max(attention_arrays, axis=0)
        elif aggregate_method == "last":
            # Use only the last layer
            aggregated = attention_arrays[-1]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate_method}")
        
        # Average across attention heads
        head_averaged = np.mean(aggregated, axis=1)  # Shape: (batch_size, seq_len, seq_len)
        
        # Extract attention to [CLS] token (assuming first token)
        cls_attention = head_averaged[:, 0, :]  # Shape: (batch_size, seq_len)
        
        # Self-attention patterns (diagonal removed)
        self_attention = head_averaged.copy()
        for i in range(seq_len):
            self_attention[:, i, i] = 0  # Remove self-attention
        
        return {
            "full_attention": head_averaged,
            "cls_attention": cls_attention,
            "self_attention": self_attention,
            "layer_wise": attention_arrays,
            "input_ids": np.array(input_ids)
        }
    
    def create_attention_heatmap(
        self,
        attention_patterns: Dict[str, np.ndarray],
        sample_idx: int = 0,
        attention_type: str = "cls_attention",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a heatmap visualization of attention patterns.
        
        Args:
            attention_patterns: Output from extract_attention_patterns
            sample_idx: Index of the sample to visualize
            attention_type: Type of attention to visualize
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Get tokens
        input_ids = attention_patterns["input_ids"][sample_idx]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Clean up tokens (remove padding, special tokens)
        valid_tokens = []
        valid_indices = []
        for i, token in enumerate(tokens):
            if token not in [self.tokenizer.pad_token, self.tokenizer.unk_token]:
                valid_tokens.append(token.replace('##', ''))  # Clean subword tokens
                valid_indices.append(i)
        
        # Get attention data
        if attention_type == "cls_attention":
            attention_data = attention_patterns["cls_attention"][sample_idx][valid_indices]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create bar plot for CLS attention
            bars = ax.bar(range(len(valid_tokens)), attention_data)
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Attention Weight")
            ax.set_title("Attention to [CLS] Token")
            ax.set_xticks(range(len(valid_tokens)))
            ax.set_xticklabels(valid_tokens, rotation=45, ha='right')
            
            # Color bars based on attention weight
            for bar, weight in zip(bars, attention_data):
                bar.set_color(plt.cm.Reds(weight / np.max(attention_data)))
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                     norm=plt.Normalize(vmin=0, vmax=np.max(attention_data)))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Attention Weight')
            
        else:
            # For full attention matrices
            attention_data = attention_patterns[attention_type][sample_idx]
            attention_subset = attention_data[np.ix_(valid_indices, valid_indices)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create heatmap
            sns.heatmap(
                attention_subset,
                xticklabels=valid_tokens,
                yticklabels=valid_tokens,
                cmap="Blues",
                ax=ax,
                cbar_kws={'label': 'Attention Weight'}
            )
            ax.set_title(f"Attention Heatmap: {attention_type}")
            ax.set_xlabel("Target Tokens")
            ax.set_ylabel("Source Tokens")
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def identify_important_tokens(
        self,
        attention_patterns: Dict[str, np.ndarray],
        sample_idx: int = 0,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify the most important tokens based on attention weights.
        
        Args:
            attention_patterns: Output from extract_attention_patterns
            sample_idx: Index of the sample to analyze
            top_k: Number of top tokens to return
            
        Returns:
            List of dictionaries with token information
        """
        input_ids = attention_patterns["input_ids"][sample_idx]
        cls_attention = attention_patterns["cls_attention"][sample_idx]
        
        # Get tokens and their attention weights
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        token_importance = []
        for i, (token, attention_weight) in enumerate(zip(tokens, cls_attention)):
            if token not in [self.tokenizer.pad_token, self.tokenizer.unk_token]:
                token_importance.append({
                    'token': token.replace('##', ''),
                    'position': i,
                    'attention_weight': float(attention_weight),
                    'token_id': int(input_ids[i])
                })
        
        # Sort by attention weight
        token_importance.sort(key=lambda x: x['attention_weight'], reverse=True)
        
        return token_importance[:top_k]
    
    def create_token_importance_plot(
        self,
        important_tokens: List[Dict[str, Any]],
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a plot showing token importance based on attention.
        
        Args:
            important_tokens: Output from identify_important_tokens
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        tokens = [item['token'] for item in important_tokens]
        weights = [item['attention_weight'] for item in important_tokens]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(tokens)), weights)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xlabel("Attention Weight")
        ax.set_title("Most Important Tokens (by Attention)")
        ax.invert_yaxis()  # Highest importance at top
        
        # Color bars
        for bar, weight in zip(bars, weights):
            bar.set_color(plt.cm.Oranges(weight / max(weights)))
        
        # Add values on bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax.text(weight + max(weights) * 0.01, i, f'{weight:.3f}', 
                   va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_attention_patterns(
        self,
        model,
        params: Dict[str, Any],
        text: str,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Complete attention analysis for a given text.
        
        Args:
            model: The transformer model
            params: Model parameters
            text: Input text to analyze
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with all analysis results
        """
        # Tokenize input
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='np'
        )
        input_ids = jnp.array(encoded['input_ids'])
        
        # Get model outputs
        outputs = model.apply(params, input_ids, training=False)
        
        # Extract attention patterns
        attention_patterns = self.extract_attention_patterns(
            outputs['attention_weights'], 
            input_ids
        )
        
        # Get important tokens
        important_tokens = self.identify_important_tokens(attention_patterns)
        
        # Get predictions
        logits = outputs['logits']
        probabilities = jax.nn.sigmoid(logits)
        
        safety_categories = {
            0: 'hate_speech',
            1: 'self_harm', 
            2: 'dangerous_advice',
            3: 'harassment'
        }
        
        predictions = {
            safety_categories[i]: float(probabilities[0, i]) 
            for i in range(len(safety_categories))
        }
        
        return {
            'text': text,
            'predictions': predictions,
            'attention_patterns': attention_patterns,
            'important_tokens': important_tokens,
            'input_ids': input_ids,
            'tokens': self.tokenizer.convert_ids_to_tokens(input_ids[0])
        }


class FeatureAttributionAnalyzer:
    """
    Analyzes feature importance and attribution for model predictions.
    """
    
    def __init__(self, model, params: Dict[str, Any], tokenizer_name: str):
        """
        Initialize the feature attribution analyzer.
        
        Args:
            model: The transformer model
            params: Model parameters
            tokenizer_name: Name of the tokenizer
        """
        self.model = model
        self.params = params
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_integrated_gradients(
        self,
        text: str,
        target_class: int,
        baseline: Optional[str] = None,
        steps: int = 50,
        max_length: int = 512
    ) -> np.ndarray:
        """
        Compute integrated gradients for feature attribution.
        
        Args:
            text: Input text
            target_class: Target class index
            baseline: Baseline text (default: empty string)
            steps: Number of integration steps
            max_length: Maximum sequence length
            
        Returns:
            Integrated gradients for each token
        """
        if baseline is None:
            baseline = ""
        
        # Tokenize inputs
        text_encoded = self.tokenizer(
            text, truncation=True, padding='max_length', 
            max_length=max_length, return_tensors='np'
        )
        baseline_encoded = self.tokenizer(
            baseline, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='np'
        )
        
        text_ids = jnp.array(text_encoded['input_ids'])
        baseline_ids = jnp.array(baseline_encoded['input_ids'])
        
        # Create interpolation path
        alphas = jnp.linspace(0, 1, steps)
        interpolated_inputs = []
        
        for alpha in alphas:
            # Linear interpolation in embedding space would be more accurate,
            # but this token-level approximation is simpler
            interpolated = jnp.where(
                jax.random.bernoulli(jax.random.PRNGKey(42), alpha, shape=text_ids.shape),
                text_ids,
                baseline_ids
            )
            interpolated_inputs.append(interpolated)
        
        # Define gradient function
        def prediction_fn(input_ids):
            outputs = self.model.apply(self.params, input_ids, training=False)
            return outputs['logits'][0, target_class]
        
        grad_fn = jax.grad(prediction_fn)
        
        # Compute gradients for each interpolated input
        gradients = []
        for interpolated_input in interpolated_inputs:
            grad = grad_fn(interpolated_input)
            gradients.append(grad)
        
        # Average gradients and multiply by input difference
        avg_gradients = jnp.mean(jnp.stack(gradients), axis=0)
        integrated_gradients = avg_gradients * (text_ids - baseline_ids)
        
        return np.array(integrated_gradients[0])  # Remove batch dimension
    
    def create_attribution_visualization(
        self,
        text: str,
        attributions: np.ndarray,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create visualization of feature attributions.
        
        Args:
            text: Original input text
            attributions: Attribution scores for each token
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Tokenize to get tokens
        encoded = self.tokenizer(text, return_tensors='np')
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        # Remove padding tokens
        valid_indices = []
        valid_tokens = []
        valid_attributions = []
        
        for i, (token, attr) in enumerate(zip(tokens, attributions)):
            if token not in [self.tokenizer.pad_token, self.tokenizer.unk_token]:
                valid_tokens.append(token.replace('##', ''))
                valid_attributions.append(attr)
                valid_indices.append(i)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create color mapping
        max_attr = max(abs(min(valid_attributions)), abs(max(valid_attributions)))
        colors = [plt.cm.RdBu_r((attr / max_attr + 1) / 2) for attr in valid_attributions]
        
        # Create bar plot
        bars = ax.bar(range(len(valid_tokens)), valid_attributions, color=colors)
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Attribution Score")
        ax.set_title("Feature Attribution (Integrated Gradients)")
        ax.set_xticks(range(len(valid_tokens)))
        ax.set_xticklabels(valid_tokens, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, 
                                 norm=plt.Normalize(vmin=-max_attr, vmax=max_attr))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attribution Score\n(Red: Positive, Blue: Negative)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_comprehensive_explanation(
    model,
    params: Dict[str, Any],
    text: str,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_length: int = 512
) -> Dict[str, Any]:
    """
    Create a comprehensive explanation combining attention and attribution analysis.
    
    Args:
        model: The transformer model
        params: Model parameters
        text: Input text to explain
        tokenizer_name: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with comprehensive explanations
    """
    # Initialize analyzers
    attention_viz = AttentionVisualizer(tokenizer_name)
    
    # Attention analysis
    attention_analysis = attention_viz.analyze_attention_patterns(
        model, params, text, max_length
    )
    
    # Feature attribution (for the most likely unsafe category)
    predictions = attention_analysis['predictions']
    max_unsafe_category = max(
        (k, v) for k, v in predictions.items() if k != 'safe'
    )[0] if any(v > 0.5 for k, v in predictions.items()) else None
    
    attribution_analysis = None
    if max_unsafe_category is not None:
        safety_categories = {
            'hate_speech': 0,
            'self_harm': 1,
            'dangerous_advice': 2,
            'harassment': 3
        }
        
        target_class = safety_categories[max_unsafe_category]
        
        attr_analyzer = FeatureAttributionAnalyzer(model, params, tokenizer_name)
        attributions = attr_analyzer.compute_integrated_gradients(
            text, target_class, max_length=max_length
        )
        
        attribution_analysis = {
            'target_class': max_unsafe_category,
            'attributions': attributions
        }
    
    return {
        'text': text,
        'predictions': predictions,
        'attention_analysis': attention_analysis,
        'attribution_analysis': attribution_analysis,
        'explanation_summary': _generate_explanation_summary(
            predictions, attention_analysis['important_tokens'], attribution_analysis
        )
    }


def _generate_explanation_summary(
    predictions: Dict[str, float],
    important_tokens: List[Dict[str, Any]],
    attribution_analysis: Optional[Dict[str, Any]]
) -> str:
    """Generate a human-readable explanation summary."""
    # Find the highest prediction
    max_category = max(predictions, key=predictions.get)
    max_score = predictions[max_category]
    
    explanation_parts = []
    
    if max_score > 0.5:
        explanation_parts.append(
            f"This text was classified as potentially unsafe "
            f"({max_category.replace('_', ' ')}) with {max_score:.1%} confidence."
        )
    else:
        explanation_parts.append(
            f"This text appears to be safe with {(1-max_score):.1%} confidence."
        )
    
    # Add information about important tokens
    if important_tokens:
        top_tokens = [token['token'] for token in important_tokens[:3]]
        explanation_parts.append(
            f"The model focused most attention on these tokens: {', '.join(top_tokens)}"
        )
    
    # Add attribution information if available
    if attribution_analysis:
        explanation_parts.append(
            f"Feature attribution analysis was performed for the {attribution_analysis['target_class']} category."
        )
    
    return " ".join(explanation_parts)


if __name__ == "__main__":
    # Test the visualization tools
    logging.basicConfig(level=logging.INFO)
    
    # This would typically be run with actual model and data
    print("Attention visualization tools initialized successfully!")
    print("Use these tools to analyze model predictions and attention patterns.")
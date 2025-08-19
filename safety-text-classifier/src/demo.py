"""
Interactive Gradio Demo for Safety Text Classifier

Provides a user-friendly interface for testing the safety text classifier
with real-time predictions, explanations, and visualization capabilities.
"""

import gradio as gr
import jax
import jax.numpy as jnp
import numpy as np
import yaml
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer
import logging

from .models.transformer import create_model, initialize_model
from .models.utils import ModelLoader
from .models.visualization import AttentionVisualizer, create_comprehensive_explanation
from .models.calibration import ConfidenceEstimator

logger = logging.getLogger(__name__)


class SafetyClassifierDemo:
    """
    Interactive demo interface for the safety text classifier.
    
    Provides real-time classification with explanations, attention visualization,
    and batch processing capabilities.
    """
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """
        Initialize the demo with trained model.
        
        Args:
            config_path: Path to model configuration
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['data']['tokenizer']
        )
        
        # Safety categories
        self.safety_categories = [
            'Hate Speech', 'Self-Harm', 'Dangerous Advice', 'Harassment'
        ]
        
        # Initialize model
        self.model = create_model(self.config)
        
        # Try to load trained parameters from checkpoint
        try:
            self.model, self.params, self.metadata = ModelLoader.load_best_model(
                config_path=config_path
            )
            logger.info("Loaded trained model from checkpoint")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Using random initialization.")
            self.rng = jax.random.PRNGKey(42)
            self.params = initialize_model(self.model, self.rng)
            self.metadata = None
        
        # Initialize analysis tools
        self.attention_visualizer = AttentionVisualizer(self.config['data']['tokenizer'])
        self.confidence_estimator = ConfidenceEstimator()
        
        # Create JIT-compiled inference function
        self.predict_fn = jax.jit(self._predict_single)
        
        logger.info("Demo initialized successfully!")
    
    def _predict_single(self, params: Any, input_ids: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """JIT-compiled single prediction function."""
        outputs = self.model.apply(params, input_ids, training=False)
        probabilities = jax.nn.sigmoid(outputs['logits'])
        return {
            'probabilities': probabilities,
            'attention_weights': outputs.get('attention_weights', []),
            'hidden_states': outputs.get('hidden_states', None)
        }
    
    def preprocess_text(self, text: str) -> jnp.ndarray:
        """
        Preprocess text for model input.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tokenized input IDs
        """
        # Tokenize text
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config['data']['max_length'],
            return_tensors='np'
        )
        
        return jnp.array(encoded['input_ids'])
    
    def classify_text(self, text: str, include_analysis: bool = True) -> Tuple[Dict[str, float], str, Any]:
        """
        Classify input text and generate comprehensive explanation.
        
        Args:
            text: Input text to classify
            include_analysis: Whether to include detailed analysis
            
        Returns:
            Tuple of (safety_scores, explanation, attention_plot)
        """
        if not text.strip():
            return {}, "Please enter some text to classify.", None
        
        try:
            # Use comprehensive explanation if analysis tools available
            if include_analysis and hasattr(self, 'attention_visualizer'):
                comprehensive_analysis = create_comprehensive_explanation(
                    self.model, self.params, text, self.config['data']['tokenizer']
                )
                
                predictions = comprehensive_analysis['predictions']
                safety_scores = {
                    'Hate Speech': predictions.get('hate_speech', 0.0),
                    'Self-Harm': predictions.get('self_harm', 0.0), 
                    'Dangerous Advice': predictions.get('dangerous_advice', 0.0),
                    'Harassment': predictions.get('harassment', 0.0)
                }
                
                # Get confidence estimate
                probs_array = np.array([predictions[k.lower().replace(' ', '_')] for k in safety_scores.keys()])
                confidence = self.confidence_estimator.entropy_confidence(probs_array)
                avg_confidence = np.mean(confidence) if len(confidence.shape) > 0 else float(confidence)
                
                # Generate detailed explanation
                explanation = comprehensive_analysis['explanation_summary']
                explanation += f"\n\n**Model Confidence**: {avg_confidence:.1%}"
                
                if comprehensive_analysis['attention_analysis']:
                    important_tokens = comprehensive_analysis['attention_analysis']['important_tokens'][:5]
                    token_list = [token['token'] for token in important_tokens]
                    explanation += f"\n**Key Focus Areas**: {', '.join(token_list)}"
                
                # Create enhanced attention visualization
                attention_plot = self._create_enhanced_attention_plot(comprehensive_analysis)
                
                return safety_scores, explanation, attention_plot
            
            else:
                # Fallback to basic classification
                return self._basic_classify_text(text)
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return self._basic_classify_text(text)  # Fallback
    
    def _basic_classify_text(self, text: str) -> Tuple[Dict[str, float], str, Any]:
        """Basic classification fallback."""
        # Preprocess input
        input_ids = self.preprocess_text(text)
        
        # Get prediction
        outputs = self.predict_fn(self.params, input_ids)
        probabilities = outputs['probabilities'][0]  # Remove batch dimension
        
        # Create safety scores dictionary
        safety_scores = {}
        max_prob = 0
        max_category = "Safe"
        
        for i, category in enumerate(self.safety_categories):
            prob = float(probabilities[i])
            safety_scores[category] = prob
            
            if prob > max_prob:
                max_prob = prob
                max_category = category
        
        # Generate basic explanation
        if max_prob > 0.5:
            explanation = f"⚠️ **WARNING**: This text has been classified as potentially harmful.\n\n"
            explanation += f"**Primary concern**: {max_category} (confidence: {max_prob:.1%})\n\n"
            explanation += self._generate_detailed_explanation(max_category, max_prob)
        else:
            explanation = f"✅ **SAFE**: This text appears to be safe.\n\n"
            explanation += f"No significant safety concerns detected. "
            explanation += f"Highest risk category: {max_category} ({max_prob:.1%})"
        
        # Create basic attention visualization
        attention_plot = self._create_attention_visualization(
            text, outputs['attention_weights'], input_ids
        )
        
        return safety_scores, explanation, attention_plot
    
    def _generate_detailed_explanation(self, category: str, confidence: float) -> str:
        """Generate detailed explanation for safety classification."""
        explanations = {
            'Hate Speech': "This text contains language that may be hateful or discriminatory towards individuals or groups based on protected characteristics.",
            'Self-Harm': "This text contains content related to self-injury or self-harm, which could be harmful to vulnerable individuals.",
            'Dangerous Advice': "This text contains potentially dangerous instructions or advice that could cause harm if followed.",
            'Harassment': "This text contains language that could be considered harassment, bullying, or threatening behavior."
        }
        
        base_explanation = explanations.get(category, "This text has been flagged for safety concerns.")
        
        confidence_note = ""
        if confidence > 0.9:
            confidence_note = "The model is very confident in this classification."
        elif confidence > 0.7:
            confidence_note = "The model is moderately confident in this classification."
        else:
            confidence_note = "The model has some uncertainty in this classification."
        
        return f"{base_explanation}\n\n{confidence_note}"
    
    def _create_enhanced_attention_plot(self, analysis: Dict[str, Any]) -> Any:
        """Create enhanced attention visualization from comprehensive analysis."""
        try:
            if not analysis.get('attention_analysis'):
                return None
            
            important_tokens = analysis['attention_analysis']['important_tokens'][:10]
            
            tokens = [token['token'] for token in important_tokens]
            weights = [token['attention_weight'] for token in important_tokens]
            
            # Create bar plot
            fig = go.Figure(data=go.Bar(
                x=tokens,
                y=weights,
                marker=dict(
                    color=weights,
                    colorscale='Reds',
                    colorbar=dict(title="Attention Weight")
                )
            ))
            
            fig.update_layout(
                title="Most Important Tokens (by Attention)",
                xaxis_title="Tokens",
                yaxis_title="Attention Weight",
                height=400,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Could not create enhanced attention plot: {e}")
            return None
    
    def _create_attention_visualization(
        self, 
        text: str, 
        attention_weights: List[jnp.ndarray],
        input_ids: jnp.ndarray
    ) -> Any:
        """
        Create attention weight visualization.
        
        Args:
            text: Original input text
            attention_weights: Attention weights from model
            input_ids: Tokenized input
            
        Returns:
            Plotly figure showing attention patterns
        """
        try:
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.flatten())
            
            # Use attention from last layer, average across heads
            last_layer_attention = attention_weights[-1][0]  # Remove batch dimension
            avg_attention = jnp.mean(last_layer_attention, axis=0)  # Average across heads
            
            # Take attention to [CLS] token (or first token)
            cls_attention = avg_attention[0, 1:len(tokens)]  # Skip CLS, PAD tokens
            
            # Create attention heatmap
            fig = go.Figure(data=go.Heatmap(
                x=tokens[1:len(tokens)],
                z=[cls_attention],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Attention Weight")
            ))
            
            fig.update_layout(
                title="Attention Weights (Model Focus Areas)",
                xaxis_title="Tokens",
                yaxis_title="Attention",
                height=200,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Could not create attention visualization: {e}")
            return None
    
    def batch_classify(self, file_path: str) -> Tuple[str, Any]:
        """
        Classify multiple texts from uploaded file.
        
        Args:
            file_path: Path to uploaded text file
            
        Returns:
            Tuple of (results_text, results_plot)
        """
        if not file_path:
            return "Please upload a file with texts to classify.", None
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}", None
        
        # Split content into lines
        texts = [line.strip() for line in file_content.split('\n') if line.strip()]
        
        if not texts:
            return "No valid texts found in the uploaded file.", None
        
        results = []
        safety_counts = {category: 0 for category in self.safety_categories}
        safety_counts['Safe'] = 0
        
        for i, text in enumerate(texts[:100]):  # Limit to 100 texts for demo
            try:
                safety_scores, _, _ = self.classify_text(text)
                
                max_score = 0
                max_category = 'Safe'
                
                for category, score in safety_scores.items():
                    if score > max_score:
                        max_score = score
                        max_category = category
                
                if max_score <= 0.5:
                    max_category = 'Safe'
                
                safety_counts[max_category] += 1
                
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'category': max_category,
                    'confidence': max_score
                })
                
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                continue
        
        # Create results text
        results_text = f"Processed {len(results)} texts:\n\n"
        for i, result in enumerate(results[:20]):  # Show first 20 results
            results_text += f"{i+1}. {result['text']}\n"
            results_text += f"   → {result['category']} ({result['confidence']:.1%})\n\n"
        
        if len(results) > 20:
            results_text += f"... and {len(results) - 20} more texts.\n\n"
        
        # Create summary plot
        fig = px.pie(
            values=list(safety_counts.values()),
            names=list(safety_counts.keys()),
            title="Safety Classification Summary"
        )
        
        return results_text, fig
    
    def create_interface(self) -> gr.Interface:
        """Create the Gradio interface."""
        
        # Custom CSS for better styling
        css = """
        .safety-score { font-weight: bold; }
        .warning { color: #ff6b6b; }
        .safe { color: #51cf66; }
        """
        
        with gr.Blocks(css=css, title="Safety Text Classifier") as interface:
            gr.Markdown("""
            # 🛡️ Safety Text Classifier
            
            **Constitutional AI Research Project - Stage 1**
            
            This tool classifies text for potential safety concerns across four categories:
            - **Hate Speech**: Discriminatory or hateful language
            - **Self-Harm**: Content related to self-injury
            - **Dangerous Advice**: Potentially harmful instructions
            - **Harassment**: Bullying or threatening behavior
            """)
            
            with gr.Tab("Single Text Classification"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Enter text to classify:",
                            placeholder="Type or paste text here...",
                            lines=5,
                            max_lines=10
                        )
                        classify_button = gr.Button("🔍 Classify Text", variant="primary")
                    
                    with gr.Column(scale=1):
                        safety_scores = gr.Label(
                            label="Safety Scores",
                            num_top_classes=4
                        )
                
                explanation_output = gr.Markdown(label="Explanation")
                
                attention_plot = gr.Plot(label="Model Attention Visualization")
                
                # Examples
                gr.Examples(
                    examples=[
                        ["The weather is beautiful today, perfect for a walk in the park."],
                        ["I hate all people from that country, they are terrible."],
                        ["Here's how to make dangerous explosives at home using household items."],
                        ["I'm going to find where you live and make you pay."],
                        ["I feel so sad, I want to hurt myself badly."],
                    ],
                    inputs=[text_input]
                )
            
            with gr.Tab("Batch Classification"):
                gr.Markdown("""
                Upload a text file with one text per line for batch classification.
                Maximum 100 texts will be processed.
                """)
                
                file_input = gr.File(
                    label="Upload text file",
                    file_types=[".txt"]
                )
                batch_button = gr.Button("📊 Process Batch", variant="primary")
                
                batch_results = gr.Textbox(
                    label="Results",
                    lines=15,
                    max_lines=20
                )
                batch_plot = gr.Plot(label="Classification Summary")
            
            with gr.Tab("Model Information"):
                gr.Markdown("""
                ## Model Architecture
                
                - **Framework**: JAX/Flax
                - **Architecture**: Transformer-based encoder
                - **Layers**: 6 transformer blocks
                - **Attention Heads**: 12
                - **Embedding Dimension**: 768
                - **Classification**: Multi-label binary classification
                
                ## Training Data
                
                - **Sources**: HuggingFace safety datasets + synthetic data
                - **Size**: ~2000 examples for development
                - **Categories**: 4 safety categories + safe content
                - **Preprocessing**: Sentence transformer tokenization
                
                ## Performance Metrics
                
                - **Target Accuracy**: >85%
                - **Evaluation**: Comprehensive fairness and robustness testing
                - **Monitoring**: Real-time performance tracking
                
                ## Constitutional AI Context
                
                This classifier serves as the foundation for the 4-stage Constitutional AI research pipeline,
                providing safety evaluation capabilities for future constitutional training experiments.
                """)
            
            # Event handlers
            classify_button.click(
                fn=self.classify_text,
                inputs=[text_input],
                outputs=[safety_scores, explanation_output, attention_plot]
            )
            
            batch_button.click(
                fn=self.batch_classify,
                inputs=[file_input],
                outputs=[batch_results, batch_plot]
            )
        
        return interface


def launch_demo(
    config_path: str = "configs/base_config.yaml",
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860
):
    """
    Launch the Gradio demo interface.
    
    Args:
        config_path: Path to model configuration
        share: Whether to create public link
        server_name: Server host name
        server_port: Server port
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create demo instance
        demo = SafetyClassifierDemo(config_path)
        
        # Create and launch interface
        interface = demo.create_interface()
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to launch demo: {e}")
        raise


if __name__ == "__main__":
    launch_demo()
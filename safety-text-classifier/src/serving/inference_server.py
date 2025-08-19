"""
Safety Text Classifier Inference Server

FastAPI-based serving infrastructure for the safety text classifier
with batch processing, attention visualization, and monitoring.
"""

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import yaml
import asyncio
from pathlib import Path
import time
from flax.training import checkpoints
from transformers import AutoTokenizer

from ..models.transformer import SafetyTransformer, create_model
from ..data.dataset_loader import SafetyDatasetLoader

logger = logging.getLogger(__name__)


class SafetyRequest(BaseModel):
    """Request model for safety classification."""
    text: str = Field(..., description="Text to classify", max_length=2000)
    include_attention: bool = Field(False, description="Include attention weights in response")
    include_explanations: bool = Field(False, description="Include explanations for predictions")


class BatchSafetyRequest(BaseModel):
    """Batch request model for safety classification."""
    texts: List[str] = Field(..., description="List of texts to classify", max_items=100)
    include_attention: bool = Field(False, description="Include attention weights in response")
    include_explanations: bool = Field(False, description="Include explanations for predictions")


class SafetyResponse(BaseModel):
    """Response model for safety classification."""
    text: str
    predictions: Dict[str, float]
    is_safe: bool
    confidence: float
    processing_time_ms: float
    attention_weights: Optional[List[List[float]]] = None
    explanations: Optional[Dict[str, str]] = None


class BatchSafetyResponse(BaseModel):
    """Batch response model for safety classification."""
    results: List[SafetyResponse]
    total_processing_time_ms: float
    batch_size: int


class SafetyClassifierServer:
    """Main inference server for safety text classification."""
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """Initialize the inference server."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.deployment_config = self.config['deployment']
        
        # Safety categories
        self.safety_categories = {
            0: 'hate_speech',
            1: 'self_harm', 
            2: 'dangerous_advice',
            3: 'harassment'
        }
        
        # Initialize model and tokenizer
        self.model = None
        self.params = None
        self.tokenizer = None
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        
    async def initialize(self, checkpoint_path: str = "checkpoints/best_model"):
        """Initialize model and load checkpoint."""
        logger.info("Initializing Safety Classifier Server...")
        
        try:
            # Create model
            self.model = create_model(self.config)
            
            # Load checkpoint
            checkpoint_dir = Path(checkpoint_path)
            if checkpoint_dir.exists():
                self.params = checkpoints.restore_checkpoint(
                    checkpoint_dir, target=None
                )
                if hasattr(self.params, 'params'):
                    self.params = self.params.params
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            else:
                # Initialize with random parameters for demo
                logger.warning(f"Checkpoint not found at {checkpoint_path}, using random initialization")
                rng = jax.random.PRNGKey(42)
                dummy_input = jnp.ones((1, 512), dtype=jnp.int32)
                self.params = self.model.init(rng, dummy_input, training=False)
            
            # Initialize tokenizer
            tokenizer_name = self.config['data']['tokenizer']
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # JIT compile inference function
            self._predict_fn = jax.jit(self._predict_batch)
            
            logger.info("Server initialization completed successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise RuntimeError(f"Server initialization failed: {e}")
    
    def _preprocess_text(self, texts: List[str]) -> jnp.ndarray:
        """Preprocess texts for model input."""
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.config['data']['max_length'],
            return_tensors='np'
        )
        
        return jnp.array(encoded['input_ids'])
    
    def _predict_batch(self, input_ids: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """JIT-compiled batch prediction function."""
        outputs = self.model.apply(
            self.params,
            input_ids,
            training=False
        )
        
        # Apply sigmoid for probabilities
        probabilities = jax.nn.sigmoid(outputs['logits'])
        
        return {
            'probabilities': probabilities,
            'attention_weights': outputs['attention_weights'],
            'hidden_states': outputs['hidden_states']
        }
    
    def _calculate_confidence(self, probabilities: jnp.ndarray) -> float:
        """Calculate confidence score for predictions."""
        # Use entropy-based confidence measure
        # Higher entropy = lower confidence
        probs = jnp.clip(probabilities, 1e-7, 1 - 1e-7)  # Avoid log(0)
        entropy = -jnp.sum(probs * jnp.log(probs) + (1 - probs) * jnp.log(1 - probs))
        max_entropy = len(probabilities) * (-0.5 * jnp.log(0.5) - 0.5 * jnp.log(0.5))
        confidence = 1.0 - (entropy / max_entropy)
        return float(jnp.clip(confidence, 0.0, 1.0))
    
    def _generate_explanations(
        self, 
        text: str, 
        probabilities: jnp.ndarray,
        attention_weights: Optional[List] = None
    ) -> Dict[str, str]:
        """Generate explanations for predictions."""
        explanations = {}
        threshold = 0.5
        
        for idx, prob in enumerate(probabilities):
            category = self.safety_categories[idx]
            
            if prob > threshold:
                if category == 'hate_speech':
                    explanations[category] = f"Detected potential hate speech (confidence: {prob:.2f}). Content may contain discriminatory language."
                elif category == 'self_harm':
                    explanations[category] = f"Detected potential self-harm content (confidence: {prob:.2f}). Content may contain harmful instructions."
                elif category == 'dangerous_advice':
                    explanations[category] = f"Detected dangerous advice (confidence: {prob:.2f}). Content may contain harmful recommendations."
                elif category == 'harassment':
                    explanations[category] = f"Detected potential harassment (confidence: {prob:.2f}). Content may be threatening or abusive."
            else:
                explanations[category] = f"No {category.replace('_', ' ')} detected (confidence: {1-prob:.2f})"
        
        return explanations
    
    async def predict_single(
        self, 
        text: str, 
        include_attention: bool = False,
        include_explanations: bool = False
    ) -> SafetyResponse:
        """Predict safety categories for a single text."""
        start_time = time.time()
        
        # Preprocess
        input_ids = self._preprocess_text([text])
        
        # Predict
        outputs = self._predict_fn(input_ids)
        probabilities = outputs['probabilities'][0]  # First (and only) item in batch
        
        # Convert to Python types
        probs_dict = {
            self.safety_categories[i]: float(probabilities[i]) 
            for i in range(len(self.safety_categories))
        }
        
        # Determine if text is safe (all categories below threshold)
        threshold = 0.5
        is_safe = all(prob < threshold for prob in probabilities)
        confidence = self._calculate_confidence(probabilities)
        
        # Optional features
        attention_weights_list = None
        if include_attention and outputs['attention_weights']:
            # Average attention across heads and layers for simplicity
            attention = outputs['attention_weights'][-1][0]  # Last layer, first item in batch
            attention_weights_list = attention.mean(axis=0).tolist()  # Average across heads
        
        explanations = None
        if include_explanations:
            explanations = self._generate_explanations(text, probabilities)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics
        self.request_count += 1
        self.total_processing_time += processing_time
        
        return SafetyResponse(
            text=text,
            predictions=probs_dict,
            is_safe=is_safe,
            confidence=confidence,
            processing_time_ms=processing_time,
            attention_weights=attention_weights_list,
            explanations=explanations
        )
    
    async def predict_batch(
        self,
        texts: List[str],
        include_attention: bool = False,
        include_explanations: bool = False
    ) -> BatchSafetyResponse:
        """Predict safety categories for multiple texts."""
        start_time = time.time()
        
        # Process in smaller batches to avoid memory issues
        batch_size = self.deployment_config['model_server']['max_batch_size']
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Process batch
            tasks = [
                self.predict_single(text, include_attention, include_explanations)
                for text in batch_texts
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        total_processing_time = (time.time() - start_time) * 1000
        
        return BatchSafetyResponse(
            results=results,
            total_processing_time_ms=total_processing_time,
            batch_size=len(texts)
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get server health status."""
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "total_requests": self.request_count,
            "average_processing_time_ms": avg_processing_time,
            "safety_categories": list(self.safety_categories.values())
        }


# Global server instance
classifier_server = SafetyClassifierServer()

# FastAPI app
app = FastAPI(
    title="Safety Text Classifier API",
    description="Constitutional AI Research - Stage 1: Safety Text Classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    await classifier_server.initialize()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Safety Text Classifier API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return classifier_server.get_health_status()


@app.post("/predict", response_model=SafetyResponse)
async def predict(request: SafetyRequest):
    """Predict safety categories for a single text."""
    try:
        return await classifier_server.predict_single(
            text=request.text,
            include_attention=request.include_attention,
            include_explanations=request.include_explanations
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchSafetyResponse)
async def predict_batch(request: BatchSafetyRequest):
    """Predict safety categories for multiple texts."""
    try:
        return await classifier_server.predict_batch(
            texts=request.texts,
            include_attention=request.include_attention,
            include_explanations=request.include_explanations
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get server metrics (Prometheus format would go here)."""
    health = classifier_server.get_health_status()
    return {
        "safety_classifier_requests_total": health["total_requests"],
        "safety_classifier_avg_processing_time_ms": health["average_processing_time_ms"],
        "safety_classifier_model_loaded": 1 if health["model_loaded"] else 0,
    }


if __name__ == "__main__":
    import uvicorn
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # JAX doesn't play well with multiple workers
    )
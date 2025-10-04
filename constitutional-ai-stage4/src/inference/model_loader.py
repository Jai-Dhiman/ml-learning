"""
Model Loader for Constitutional AI Pipeline

This module provides functionality to load and manage models from all stages
of the Constitutional AI implementation:
- Base model: Gemma 2B-IT
- Stage 2: Helpful RLHF model (base + LoRA adapters)
- Stage 3: Constitutional AI model (base + LoRA adapters)

Author: J. Dhiman
Date: October 4, 2025
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstitutionalAIModels:
    """
    Load and manage models from the 4-stage Constitutional AI pipeline.
    
    This class handles loading of:
    1. Base Gemma 2B-IT model
    2. Stage 2 Helpful RLHF model (with LoRA adapters)
    3. Stage 3 Constitutional AI model (with LoRA adapters)
    
    Features:
    - Lazy loading (models loaded on demand)
    - Memory-efficient loading with quantization options
    - Device management (CPU/GPU)
    - Tokenizer caching
    
    Example usage:
        >>> models = ConstitutionalAIModels()
        >>> base_model, tokenizer = models.load_base_model()
        >>> stage3_model, tokenizer = models.load_stage3_model()
        >>> response = models.generate(stage3_model, tokenizer, "What is AI safety?")
    """
    
    def __init__(
        self,
        base_model_id: str = "google/gemma-2b-it",
        stage2_adapters_path: Optional[str] = None,
        stage3_adapters_path: Optional[str] = None,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the model loader.
        
        Args:
            base_model_id: Hugging Face model ID for base model
            stage2_adapters_path: Path to Stage 2 LoRA adapters
            stage3_adapters_path: Path to Stage 3 LoRA adapters
            device: Device to load models on ('cuda', 'cpu', or None for auto)
            load_in_8bit: Whether to load models in 8-bit precision
            load_in_4bit: Whether to load models in 4-bit precision
        """
        self.base_model_id = base_model_id
        
        # Default adapter paths
        if stage2_adapters_path is None:
            stage2_adapters_path = "../artifacts/stage2_artifacts/lora_adapters"
        if stage3_adapters_path is None:
            stage3_adapters_path = "../artifacts/stage3_artifacts/models/lora_adapters"
            
        self.stage2_adapters = Path(stage2_adapters_path)
        self.stage3_adapters = Path(stage3_adapters_path)
        
        # Device management
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Quantization settings
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization")
            
        # Model cache
        self._base_model = None
        self._stage2_model = None
        self._stage3_model = None
        self._tokenizer = None
        
        # Validate paths
        self._validate_paths()
        
    def _validate_paths(self):
        """Validate that adapter paths exist."""
        if not self.stage2_adapters.exists():
            logger.warning(f"Stage 2 adapters not found at: {self.stage2_adapters}")
        else:
            logger.info(f"Stage 2 adapters found at: {self.stage2_adapters}")
            
        if not self.stage3_adapters.exists():
            logger.warning(f"Stage 3 adapters not found at: {self.stage3_adapters}")
        else:
            logger.info(f"Stage 3 adapters found at: {self.stage3_adapters}")
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Get quantization configuration if requested.
        
        Returns:
            BitsAndBytesConfig or None
        """
        if self.load_in_8bit:
            logger.info("Loading with 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        elif self.load_in_4bit:
            logger.info("Loading with 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None
    
    def load_tokenizer(self, force_reload: bool = False) -> AutoTokenizer:
        """
        Load the tokenizer (shared across all models).
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Loaded tokenizer
        """
        if self._tokenizer is not None and not force_reload:
            return self._tokenizer
            
        logger.info(f"Loading tokenizer from {self.base_model_id}")
        
        # Try to load from Stage 3 adapters first (includes tokenizer)
        if self.stage3_adapters.exists() and (self.stage3_adapters / "tokenizer.json").exists():
            logger.info(f"Loading tokenizer from Stage 3 adapters: {self.stage3_adapters}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.stage3_adapters),
                trust_remote_code=True,
            )
        else:
            # Fall back to base model
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                trust_remote_code=True,
            )
        
        # Set padding token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            
        logger.info("Tokenizer loaded successfully")
        return self._tokenizer
    
    def load_base_model(
        self,
        force_reload: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the base Gemma 2B-IT model.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if self._base_model is not None and not force_reload:
            logger.info("Returning cached base model")
            return self._base_model, self.load_tokenizer()
            
        logger.info(f"Loading base model: {self.base_model_id}")
        
        quantization_config = self._get_quantization_config()
        
        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        
        if self.device == "cpu":
            self._base_model = self._base_model.to(self.device)
            
        logger.info(f"Base model loaded successfully on {self.device}")
        logger.info(f"Model memory footprint: {self._base_model.get_memory_footprint() / 1e9:.2f} GB")
        
        return self._base_model, self.load_tokenizer()
    
    def load_stage2_model(
        self,
        force_reload: bool = False
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """
        Load Stage 2 Helpful RLHF model (base + LoRA adapters).
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if self._stage2_model is not None and not force_reload:
            logger.info("Returning cached Stage 2 model")
            return self._stage2_model, self.load_tokenizer()
            
        if not self.stage2_adapters.exists():
            raise FileNotFoundError(
                f"Stage 2 adapters not found at: {self.stage2_adapters}\n"
                "Please ensure Stage 2 training artifacts are available."
            )
            
        logger.info("Loading Stage 2 model (Helpful RLHF)")
        logger.info(f"Base model: {self.base_model_id}")
        logger.info(f"LoRA adapters: {self.stage2_adapters}")
        
        # Load base model
        base_model, _ = self.load_base_model()
        
        # Load adapter config
        adapter_config = PeftConfig.from_pretrained(str(self.stage2_adapters))
        logger.info(f"Adapter config: r={adapter_config.r}, alpha={adapter_config.lora_alpha}")
        
        # Load model with adapters
        self._stage2_model = PeftModel.from_pretrained(
            base_model,
            str(self.stage2_adapters),
        )
        
        logger.info("Stage 2 model loaded successfully")
        
        return self._stage2_model, self.load_tokenizer()
    
    def load_stage3_model(
        self,
        force_reload: bool = False
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """
        Load Stage 3 Constitutional AI model (base + LoRA adapters).
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if self._stage3_model is not None and not force_reload:
            logger.info("Returning cached Stage 3 model")
            return self._stage3_model, self.load_tokenizer()
            
        if not self.stage3_adapters.exists():
            raise FileNotFoundError(
                f"Stage 3 adapters not found at: {self.stage3_adapters}\n"
                "Please ensure Stage 3 training artifacts are available."
            )
            
        logger.info("Loading Stage 3 model (Constitutional AI)")
        logger.info(f"Base model: {self.base_model_id}")
        logger.info(f"LoRA adapters: {self.stage3_adapters}")
        
        # Load base model
        base_model, _ = self.load_base_model()
        
        # Load adapter config
        adapter_config = PeftConfig.from_pretrained(str(self.stage3_adapters))
        logger.info(f"Adapter config: r={adapter_config.r}, alpha={adapter_config.lora_alpha}")
        
        # Load model with adapters
        self._stage3_model = PeftModel.from_pretrained(
            base_model,
            str(self.stage3_adapters),
        )
        
        logger.info("Stage 3 model loaded successfully")
        
        return self._stage3_model, self.load_tokenizer()
    
    def load_all_models(self) -> Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]]:
        """
        Load all models (base, Stage 2, Stage 3).
        
        Warning: This will consume significant memory. Consider loading models
        individually if memory is limited.
        
        Returns:
            Dictionary mapping model names to (model, tokenizer) tuples
        """
        logger.info("Loading all models...")
        logger.warning("This will consume significant memory!")
        
        models = {}
        
        try:
            models['base'] = self.load_base_model()
            logger.info("✓ Base model loaded")
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            
        try:
            models['stage2_helpful'] = self.load_stage2_model()
            logger.info("✓ Stage 2 model loaded")
        except Exception as e:
            logger.error(f"Failed to load Stage 2 model: {e}")
            
        try:
            models['stage3_constitutional'] = self.load_stage3_model()
            logger.info("✓ Stage 3 model loaded")
        except Exception as e:
            logger.error(f"Failed to load Stage 3 model: {e}")
            
        logger.info(f"Loaded {len(models)} model(s)")
        return models
    
    def generate(
        self,
        model: Union[AutoModelForCausalLM, PeftModel],
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text using a loaded model.
        
        Args:
            model: Model to use for generation
            tokenizer: Tokenizer for encoding/decoding
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Encode prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
            
        return generated_text
    
    def test_inference(
        self,
        test_prompts: Optional[list] = None,
        models_to_test: Optional[list] = None
    ):
        """
        Test inference on all models with sample prompts.
        
        Args:
            test_prompts: List of prompts to test (defaults to standard test prompts)
            models_to_test: List of model names to test (defaults to all)
        """
        if test_prompts is None:
            test_prompts = [
                "What is AI safety?",
                "How can I learn about machine learning?",
                "Explain constitutional AI in simple terms.",
            ]
            
        if models_to_test is None:
            models_to_test = ['base', 'stage2_helpful', 'stage3_constitutional']
            
        logger.info("=" * 80)
        logger.info("TESTING INFERENCE ON ALL MODELS")
        logger.info("=" * 80)
        
        for model_name in models_to_test:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Testing: {model_name.upper()}")
            logger.info(f"{'=' * 80}")
            
            try:
                # Load model
                if model_name == 'base':
                    model, tokenizer = self.load_base_model()
                elif model_name == 'stage2_helpful':
                    model, tokenizer = self.load_stage2_model()
                elif model_name == 'stage3_constitutional':
                    model, tokenizer = self.load_stage3_model()
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                # Test each prompt
                for i, prompt in enumerate(test_prompts, 1):
                    logger.info(f"\n--- Prompt {i}/{len(test_prompts)} ---")
                    logger.info(f"Input: {prompt}")
                    
                    try:
                        response = self.generate(
                            model, 
                            tokenizer, 
                            prompt,
                            max_new_tokens=100,
                            temperature=0.7
                        )
                        logger.info(f"Output: {response[:200]}...")
                        logger.info("✓ Generation successful")
                    except Exception as e:
                        logger.error(f"✗ Generation failed: {e}")
                        
            except Exception as e:
                logger.error(f"✗ Failed to test {model_name}: {e}")
                
        logger.info("\n" + "=" * 80)
        logger.info("INFERENCE TEST COMPLETE")
        logger.info("=" * 80)
    
    def get_model_info(self) -> Dict[str, dict]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'base_model_id': self.base_model_id,
            'device': self.device,
            'quantization': {
                '8bit': self.load_in_8bit,
                '4bit': self.load_in_4bit,
            },
            'paths': {
                'stage2_adapters': str(self.stage2_adapters),
                'stage3_adapters': str(self.stage3_adapters),
            },
            'loaded': {
                'base': self._base_model is not None,
                'stage2': self._stage2_model is not None,
                'stage3': self._stage3_model is not None,
                'tokenizer': self._tokenizer is not None,
            }
        }
        
        return info
    
    def unload_models(self):
        """
        Unload all models from memory to free up resources.
        """
        logger.info("Unloading models from memory...")
        
        if self._base_model is not None:
            del self._base_model
            self._base_model = None
            logger.info("✓ Base model unloaded")
            
        if self._stage2_model is not None:
            del self._stage2_model
            self._stage2_model = None
            logger.info("✓ Stage 2 model unloaded")
            
        if self._stage3_model is not None:
            del self._stage3_model
            self._stage3_model = None
            logger.info("✓ Stage 3 model unloaded")
            
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ CUDA cache cleared")
            
        logger.info("All models unloaded successfully")


def main():
    """
    Main function for testing model loading.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load and test Constitutional AI models"
    )
    parser.add_argument(
        '--model',
        choices=['base', 'stage2', 'stage3', 'all'],
        default='all',
        help='Which model(s) to load'
    )
    parser.add_argument(
        '--test-inference',
        action='store_true',
        help='Run inference test'
    )
    parser.add_argument(
        '--load-in-8bit',
        action='store_true',
        help='Load models in 8-bit precision'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load models in 4-bit precision'
    )
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = ConstitutionalAIModels(
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Display info
    info = loader.get_model_info()
    logger.info("\n=== Model Loader Configuration ===")
    logger.info(f"Base model: {info['base_model_id']}")
    logger.info(f"Device: {info['device']}")
    logger.info(f"Stage 2 adapters: {info['paths']['stage2_adapters']}")
    logger.info(f"Stage 3 adapters: {info['paths']['stage3_adapters']}")
    logger.info("=" * 50)
    
    # Load models
    if args.model == 'base':
        model, tokenizer = loader.load_base_model()
        logger.info("\n✓ Base model loaded successfully")
    elif args.model == 'stage2':
        model, tokenizer = loader.load_stage2_model()
        logger.info("\n✓ Stage 2 model loaded successfully")
    elif args.model == 'stage3':
        model, tokenizer = loader.load_stage3_model()
        logger.info("\n✓ Stage 3 model loaded successfully")
    elif args.model == 'all':
        models = loader.load_all_models()
        logger.info(f"\n✓ Loaded {len(models)} model(s)")
    
    # Test inference if requested
    if args.test_inference:
        if args.model == 'all':
            loader.test_inference()
        else:
            model_name = args.model if args.model != 'stage2' else 'stage2_helpful'
            model_name = model_name if args.model != 'stage3' else 'stage3_constitutional'
            loader.test_inference(models_to_test=[model_name])


if __name__ == "__main__":
    main()

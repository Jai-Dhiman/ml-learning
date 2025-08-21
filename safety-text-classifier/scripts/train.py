#!/usr/bin/env python3
"""
Safety Text Classifier Training Script

Main entry point for training the safety text classifier for the
Constitutional AI research project.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.trainer import train_model


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('../logs/training/training.log')
        ]
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Safety Text Classifier for Constitutional AI Research"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/base_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Safety Text Classifier Training")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Create necessary directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Start training
        train_model(args.config)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
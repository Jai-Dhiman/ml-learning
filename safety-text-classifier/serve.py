#!/usr/bin/env python3
"""
Safety Text Classifier Serving Script

Launch the FastAPI inference server for production deployment.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.serving.inference_server import app, classifier_server


def main():
    """Main server launch function."""
    parser = argparse.ArgumentParser(
        description="Launch Safety Text Classifier Inference Server"
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
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (JAX works best with 1)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
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
    
    print("🛡️ Launching Safety Text Classifier Inference Server...")
    print(f"Configuration: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    
    # Initialize server with specific checkpoint path
    classifier_server.config_path = args.config
    
    # Override checkpoint path in startup
    original_initialize = classifier_server.initialize
    async def custom_initialize():
        await original_initialize(args.checkpoint)
    classifier_server.initialize = custom_initialize
    
    # Launch server
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
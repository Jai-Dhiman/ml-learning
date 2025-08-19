#!/usr/bin/env python3
"""
Safety Text Classifier Demo Application

Launch script for the interactive Gradio demo interface.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.demo import launch_demo


def main():
    """Main demo launch function."""
    parser = argparse.ArgumentParser(
        description="Launch Safety Text Classifier Demo"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
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
        default=7860,
        help="Server port"
    )
    
    args = parser.parse_args()
    
    print("🛡️ Launching Safety Text Classifier Demo...")
    print(f"Configuration: {args.config}")
    print(f"Server: http://{args.host}:{args.port}")
    
    launch_demo(
        config_path=args.config,
        share=args.share,
        server_name=args.host,
        server_port=args.port
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Start script for Claude Agent.

This script initializes the Claude Agent environment and starts the API server.
"""

import os
import sys
import logging
import argparse
import subprocess
import time
import json
from pathlib import Path

# Add the parent directory to the path so we can import openhands
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openhands.claude_agent.init_memory import initialize_environment
from openhands.claude_agent.api_server import start_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("start_claude_agent")

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Installing missing packages...")
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", *missing_packages],
                check=True
            )
            logger.info("Packages installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e}")
            return False
    
    return True

def check_model_dependencies(use_local_model):
    """Check if model-specific dependencies are installed."""
    if use_local_model:
        # Check for local model dependencies
        local_model_packages = [
            "llama-cpp-python",
            "ctransformers",
            "transformers"
        ]
        
        found_packages = []
        
        for package in local_model_packages:
            try:
                __import__(package.replace("-", "_"))
                found_packages.append(package)
            except ImportError:
                pass
        
        if not found_packages:
            logger.warning("No local model packages found")
            logger.info("Installing llama-cpp-python...")
            
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                    check=True
                )
                logger.info("llama-cpp-python installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install llama-cpp-python: {e}")
                return False
        
        return True
    else:
        # Check for Claude API dependencies
        try:
            __import__("anthropic")
            return True
        except ImportError:
            logger.warning("anthropic package not found")
            logger.info("Installing anthropic...")
            
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "anthropic"],
                    check=True
                )
                logger.info("anthropic installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install anthropic: {e}")
                return False

def create_default_config(config_path, use_local_model=False, local_model_path=""):
    """Create default configuration file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Create default configuration
    config = {
        "working_directory": os.path.join(os.path.expanduser("~"), "claude_agent"),
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 128000,
        "temperature": 0.7,
        "thinking_mode": True,
        "debug_mode": False,
        "tool_timeout": 60,
        "use_local_model": use_local_model,
        "local_model_path": local_model_path,
        "context_length": 4096,
        "gpu_layers": -1,
        "local_model_type": "llama"
    }
    
    # Save configuration
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created default configuration at {config_path}")

def find_local_model():
    """Find local model in common locations."""
    common_locations = [
        os.path.expanduser("~/.local/share/models"),
        os.path.expanduser("~/models"),
        "./models"
    ]
    
    for location in common_locations:
        if os.path.exists(location):
            model_files = [f for f in os.listdir(location) if f.endswith((".gguf", ".bin"))]
            if model_files:
                model_path = os.path.join(location, model_files[0])
                logger.info(f"Found local model at {model_path}")
                return model_path
    
    return ""

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start Claude Agent")
    parser.add_argument("--config", help="Path to configuration file", default="agent_config.json")
    parser.add_argument("--host", help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", help="Port to bind to", type=int, default=8000)
    parser.add_argument("--frontend", help="Path to frontend directory", default=None)
    parser.add_argument("--client", help="Path to client directory", default=None)
    parser.add_argument("--local", help="Use local model", action="store_true")
    parser.add_argument("--local-model", help="Path to local model", default=None)
    parser.add_argument("--examples", help="Add example data", action="store_true")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Failed to install required dependencies")
        sys.exit(1)
    
    # Set local model environment variables if provided
    if args.local:
        os.environ["USE_LOCAL_MODEL"] = "true"
    if args.local_model:
        os.environ["LOCAL_MODEL_PATH"] = args.local_model
    
    # Check model-specific dependencies
    if not check_model_dependencies(args.local):
        logger.error("Failed to install model-specific dependencies")
        sys.exit(1)
    
    # Create default configuration if it doesn't exist
    if not os.path.exists(args.config):
        local_model_path = args.local_model or find_local_model() if args.local else ""
        create_default_config(args.config, args.local, local_model_path)
    
    # Initialize environment
    try:
        result = initialize_environment(
            config_file=args.config,
            frontend_path=args.frontend,
            client_path=args.client,
            with_examples=args.examples
        )
        
        logger.info(f"Environment initialized successfully!")
        logger.info(f"Configuration: {result['config'].config_file}")
        logger.info(f"Memory: {result['memory'].memory_file}")
        logger.info(f"Frontend: {result['frontend_path']}")
        logger.info(f"Client: {result['client_path']}")
        
        # Start API server
        logger.info(f"Starting API server on {args.host}:{args.port}...")
        start_server(args.host, args.port)
        
    except Exception as e:
        logger.error(f"Failed to start Claude Agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
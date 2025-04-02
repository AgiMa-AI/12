#!/usr/bin/env python3
"""
Example script demonstrating the use of the model library.

This script shows how to use the model library to organize and manage models,
including scanning directories, adding models, and retrieving models by type.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any

# Add the parent directory to the path so we can import openhands
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openhands.model_library import (
    ModelLibrary,
    get_model_library,
    Model,
    ModelType,
    ModelInfo,
    scan_directory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("model_library_example")

def print_model_info(model: Model) -> None:
    """
    Print model information.
    
    Args:
        model: Model to print information for
    """
    info = model.get_info()
    print(f"ID: {info.id}")
    print(f"Name: {info.name}")
    print(f"Type: {info.model_type.value}")
    print(f"Path: {info.path}")
    print(f"Size: {info.size} bytes")
    print(f"Format: {info.format}")
    print(f"Description: {info.description}")
    print(f"Version: {info.version}")
    print(f"Created: {info.created_at}")
    if info.metadata:
        print("Metadata:")
        for key, value in info.metadata.items():
            print(f"  {key}: {value}")
    print()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Model Library Example")
    parser.add_argument("--library", help="Path to model library", default=None)
    parser.add_argument("--scan", help="Directory to scan for models", default=None)
    parser.add_argument("--add", help="Model file to add", default=None)
    parser.add_argument("--type", help="Model type for added model", default=None)
    parser.add_argument("--list", help="List models", action="store_true")
    parser.add_argument("--list-type", help="List models of specific type", default=None)
    
    args = parser.parse_args()
    
    try:
        # Get model library
        library = get_model_library(args.library)
        
        # Print library information
        print(f"Model Library: {library.library_path}")
        print(f"Total Models: {library.get_total_model_count()}")
        print("Model Counts by Type:")
        for model_type, count in library.get_model_count_by_type().items():
            print(f"  {model_type.value}: {count}")
        print()
        
        # Scan directory if specified
        if args.scan:
            print(f"Scanning directory: {args.scan}")
            models = library.scan_directory(args.scan)
            print(f"Added {len(models)} models from directory")
            print()
        
        # Add model if specified
        if args.add:
            print(f"Adding model: {args.add}")
            model_type = ModelType.from_string(args.type) if args.type else None
            model = library.add_model(args.add, model_type)
            if model:
                print("Added model:")
                print_model_info(model)
            else:
                print("Failed to add model")
            print()
        
        # List models if specified
        if args.list:
            print("All Models:")
            for model in library.get_all_models():
                print_model_info(model)
            print()
        
        # List models of specific type if specified
        if args.list_type:
            model_type = ModelType.from_string(args.list_type)
            print(f"Models of type {model_type.value}:")
            for model in library.get_models_by_type(model_type):
                print_model_info(model)
            print()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
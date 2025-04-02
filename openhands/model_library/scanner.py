"""
Model scanner for the model library.

This module provides a scanner for discovering models in directories,
automatically categorizing them based on filenames and metadata.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

from openhands.model_library.model import Model, ModelType, ModelInfo

logger = logging.getLogger("model_library_scanner")

class ModelScanner:
    """Model scanner for discovering models."""
    
    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize model scanner.
        
        Args:
            supported_formats: List of supported model formats
        """
        self.supported_formats = supported_formats or ["gguf", "bin", "pt", "pth", "onnx", "safetensors"]
    
    def scan_directory(self, directory: str, recursive: bool = True) -> List[ModelInfo]:
        """
        Scan directory for models.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively
            
        Returns:
            List of model information
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            logger.warning(f"Directory does not exist or is not a directory: {directory}")
            return []
        
        models = []
        
        # Walk through directory
        for root, dirs, files in os.walk(directory):
            # Process files in current directory
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lstrip(".")
                
                # Check if file is a supported model format
                if file_ext.lower() in self.supported_formats:
                    # Create model information
                    model_info = self._create_model_info(file_path)
                    if model_info:
                        models.append(model_info)
            
            # If not recursive, break after first iteration
            if not recursive:
                break
        
        logger.info(f"Found {len(models)} models in {directory}")
        return models
    
    def _create_model_info(self, file_path: str) -> Optional[ModelInfo]:
        """
        Create model information from file.
        
        Args:
            file_path: Path to model file
            
        Returns:
            Model information or None if file is not a valid model
        """
        try:
            # Check for metadata file
            metadata_path = self._find_metadata_file(file_path)
            metadata = self._load_metadata(metadata_path) if metadata_path else {}
            
            # Create model information
            model_info = ModelInfo(
                path=file_path,
                model_type=self._determine_model_type(file_path, metadata),
                name=metadata.get("name", os.path.basename(file_path)),
                description=metadata.get("description", ""),
                version=metadata.get("version", "1.0.0"),
                size=os.path.getsize(file_path),
                format=os.path.splitext(file_path)[1].lstrip("."),
                metadata=metadata
            )
            
            return model_info
        
        except Exception as e:
            logger.error(f"Failed to create model information for {file_path}: {e}")
            return None
    
    def _find_metadata_file(self, model_path: str) -> Optional[str]:
        """
        Find metadata file for model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Path to metadata file or None if not found
        """
        # Check for metadata file with same name but .json extension
        base_path = os.path.splitext(model_path)[0]
        metadata_path = f"{base_path}.json"
        
        if os.path.exists(metadata_path):
            return metadata_path
        
        # Check for metadata.json in same directory
        dir_path = os.path.dirname(model_path)
        metadata_path = os.path.join(dir_path, "metadata.json")
        
        if os.path.exists(metadata_path):
            return metadata_path
        
        return None
    
    def _load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """
        Load metadata from file.
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            Metadata dictionary
        """
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            return {}
    
    def _determine_model_type(self, file_path: str, metadata: Dict[str, Any]) -> ModelType:
        """
        Determine model type from file path and metadata.
        
        Args:
            file_path: Path to model file
            metadata: Model metadata
            
        Returns:
            Model type
        """
        # Check if model type is specified in metadata
        if "type" in metadata:
            return ModelType.from_string(metadata["type"])
        
        # Check if model type is specified in metadata categories
        if "categories" in metadata and metadata["categories"]:
            categories = metadata["categories"]
            if isinstance(categories, list) and categories:
                return ModelType.from_string(categories[0])
        
        # Guess model type from filename
        return ModelType.from_filename(os.path.basename(file_path))

def scan_directory(directory: str, recursive: bool = True) -> List[ModelInfo]:
    """
    Scan directory for models.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively
        
    Returns:
        List of model information
    """
    scanner = ModelScanner()
    return scanner.scan_directory(directory, recursive)
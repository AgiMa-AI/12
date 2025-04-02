"""
Model library for OpenHands.

This module provides a model library for organizing and managing different types of models,
including literature, knowledge, programming, language, image, and health models.
"""

import os
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from openhands.model_library.model import Model, ModelType, ModelInfo
from openhands.model_library.scanner import ModelScanner, scan_directory

logger = logging.getLogger("model_library")

class ModelLibrary:
    """Model library for organizing and managing models."""
    
    def __init__(self, library_path: str = None):
        """
        Initialize model library.
        
        Args:
            library_path: Path to library directory
        """
        # Set default library path if not provided
        if library_path is None:
            library_path = os.path.join(os.path.expanduser("~"), "openhands", "models")
        
        self.library_path = library_path
        self.index_path = os.path.join(library_path, "index.json")
        
        # Create library directory if it doesn't exist
        os.makedirs(library_path, exist_ok=True)
        
        # Create category directories
        self._create_category_directories()
        
        # Initialize model index
        self.models: Dict[str, Model] = {}
        self.model_by_type: Dict[ModelType, List[Model]] = {model_type: [] for model_type in ModelType}
        
        # Load model index
        self._load_index()
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def _create_category_directories(self) -> None:
        """Create category directories."""
        for model_type in ModelType:
            category_path = os.path.join(self.library_path, model_type.value)
            os.makedirs(category_path, exist_ok=True)
    
    def _load_index(self) -> None:
        """Load model index."""
        if not os.path.exists(self.index_path):
            logger.info(f"Model index not found, creating new index at {self.index_path}")
            self._save_index()
            return
        
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            
            # Load models from index
            for model_data in index_data.get("models", []):
                try:
                    model_info = ModelInfo.from_dict(model_data)
                    model = Model(model_info)
                    
                    # Add model to index
                    self.models[model_info.id] = model
                    self.model_by_type[model_info.model_type].append(model)
                except Exception as e:
                    logger.error(f"Failed to load model from index: {e}")
            
            logger.info(f"Loaded {len(self.models)} models from index")
        
        except Exception as e:
            logger.error(f"Failed to load model index: {e}")
    
    def _save_index(self) -> None:
        """Save model index."""
        try:
            # Create index data
            index_data = {
                "models": [model.info.to_dict() for model in self.models.values()]
            }
            
            # Save index
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(self.models)} models to index")
        
        except Exception as e:
            logger.error(f"Failed to save model index: {e}")
    
    def add_model(self, model_path: str, model_type: Optional[ModelType] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> Optional[Model]:
        """
        Add model to library.
        
        Args:
            model_path: Path to model file
            model_type: Model type
            metadata: Additional metadata
            
        Returns:
            Added model or None if failed
        """
        with self.lock:
            try:
                # Check if model file exists
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    return None
                
                # Create model information
                model_info = ModelInfo(
                    path=model_path,
                    model_type=model_type,
                    metadata=metadata or {}
                )
                
                # Check if model already exists
                if model_info.id in self.models:
                    logger.warning(f"Model already exists: {model_info.name}")
                    return self.models[model_info.id]
                
                # Copy model to library
                new_path = self._copy_model_to_library(model_path, model_info.model_type)
                if new_path:
                    # Update model path
                    model_info.path = new_path
                
                # Create model
                model = Model(model_info)
                
                # Add model to index
                self.models[model_info.id] = model
                self.model_by_type[model_info.model_type].append(model)
                
                # Save index
                self._save_index()
                
                logger.info(f"Added model: {model_info.name} ({model_info.model_type.value})")
                return model
            
            except Exception as e:
                logger.error(f"Failed to add model: {e}")
                return None
    
    def _copy_model_to_library(self, model_path: str, model_type: ModelType) -> Optional[str]:
        """
        Copy model to library.
        
        Args:
            model_path: Path to model file
            model_type: Model type
            
        Returns:
            New model path or None if failed
        """
        try:
            import shutil
            
            # Get category directory
            category_path = os.path.join(self.library_path, model_type.value)
            
            # Get destination path
            dest_path = os.path.join(category_path, os.path.basename(model_path))
            
            # Check if file already exists
            if os.path.exists(dest_path):
                logger.warning(f"Model file already exists in library: {dest_path}")
                return dest_path
            
            # Copy file
            shutil.copy2(model_path, dest_path)
            
            logger.info(f"Copied model to library: {dest_path}")
            return dest_path
        
        except Exception as e:
            logger.error(f"Failed to copy model to library: {e}")
            return None
    
    def remove_model(self, model_id: str) -> bool:
        """
        Remove model from library.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Check if model exists
                if model_id not in self.models:
                    logger.warning(f"Model not found: {model_id}")
                    return False
                
                # Get model
                model = self.models[model_id]
                
                # Remove model file
                if os.path.exists(model.info.path):
                    os.remove(model.info.path)
                
                # Remove model from index
                self.model_by_type[model.info.model_type].remove(model)
                del self.models[model_id]
                
                # Save index
                self._save_index()
                
                logger.info(f"Removed model: {model.info.name}")
                return True
            
            except Exception as e:
                logger.error(f"Failed to remove model: {e}")
                return False
    
    def get_model(self, model_id: str) -> Optional[Model]:
        """
        Get model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model or None if not found
        """
        return self.models.get(model_id)
    
    def get_models_by_type(self, model_type: ModelType) -> List[Model]:
        """
        Get models by type.
        
        Args:
            model_type: Model type
            
        Returns:
            List of models
        """
        return self.model_by_type.get(model_type, [])
    
    def get_all_models(self) -> List[Model]:
        """
        Get all models.
        
        Returns:
            List of all models
        """
        return list(self.models.values())
    
    def scan_directory(self, directory: str, recursive: bool = True) -> List[Model]:
        """
        Scan directory for models and add them to library.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively
            
        Returns:
            List of added models
        """
        with self.lock:
            # Scan directory
            model_infos = scan_directory(directory, recursive)
            
            # Add models to library
            added_models = []
            for model_info in model_infos:
                model = self.add_model(model_info.path, model_info.model_type, model_info.metadata)
                if model:
                    added_models.append(model)
            
            return added_models
    
    def load_model(self, model_id: str) -> bool:
        """
        Load model.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Get model
            model = self.get_model(model_id)
            if not model:
                logger.warning(f"Model not found: {model_id}")
                return False
            
            # Load model
            return model.load()
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload model.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Get model
            model = self.get_model(model_id)
            if not model:
                logger.warning(f"Model not found: {model_id}")
                return False
            
            # Unload model
            return model.unload()
    
    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if model is loaded.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if loaded, False otherwise
        """
        # Get model
        model = self.get_model(model_id)
        if not model:
            return False
        
        # Check if model is loaded
        return model.is_loaded()
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get model information.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information or None if not found
        """
        # Get model
        model = self.get_model(model_id)
        if not model:
            return None
        
        # Get model information
        return model.get_info()
    
    def get_model_types(self) -> List[ModelType]:
        """
        Get all model types.
        
        Returns:
            List of model types
        """
        return list(ModelType)
    
    def get_model_count_by_type(self) -> Dict[ModelType, int]:
        """
        Get model count by type.
        
        Returns:
            Dictionary mapping model types to counts
        """
        return {model_type: len(models) for model_type, models in self.model_by_type.items()}
    
    def get_total_model_count(self) -> int:
        """
        Get total model count.
        
        Returns:
            Total model count
        """
        return len(self.models)

# Global model library instance
_model_library = None

def get_model_library(library_path: str = None) -> ModelLibrary:
    """
    Get the global model library instance.
    
    Args:
        library_path: Path to library directory
        
    Returns:
        Model library instance
    """
    global _model_library
    
    if _model_library is None:
        _model_library = ModelLibrary(library_path)
    
    return _model_library
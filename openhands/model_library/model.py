"""
Model definitions for the model library.

This module provides model definitions for the model library,
including model types, model information, and model classes.
"""

import os
import json
import enum
import hashlib
import datetime
from typing import Dict, List, Any, Optional, Union

class ModelType(enum.Enum):
    """Model type enumeration."""
    LITERATURE = "literature"
    KNOWLEDGE = "knowledge"
    PROGRAMMING = "programming"
    LANGUAGE = "language"
    IMAGE = "image"
    HEALTH = "health"
    GENERAL = "general"
    OTHER = "other"
    
    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        """
        Get model type from string.
        
        Args:
            value: String value
            
        Returns:
            Model type
        """
        value = value.lower()
        for model_type in cls:
            if value == model_type.value:
                return model_type
            # Also check for partial matches
            if value in model_type.value:
                return model_type
        return cls.OTHER
    
    @classmethod
    def from_filename(cls, filename: str) -> "ModelType":
        """
        Guess model type from filename.
        
        Args:
            filename: Filename
            
        Returns:
            Model type
        """
        filename = filename.lower()
        
        # Check for specific keywords in filename
        if any(kw in filename for kw in ["story", "novel", "book", "lit", "fiction", "poem"]):
            return cls.LITERATURE
        elif any(kw in filename for kw in ["know", "fact", "wiki", "encyc"]):
            return cls.KNOWLEDGE
        elif any(kw in filename for kw in ["code", "program", "dev", "py", "js", "java", "cpp"]):
            return cls.PROGRAMMING
        elif any(kw in filename for kw in ["lang", "nlp", "text", "chat", "convers"]):
            return cls.LANGUAGE
        elif any(kw in filename for kw in ["img", "image", "vision", "visual", "pic"]):
            return cls.IMAGE
        elif any(kw in filename for kw in ["health", "medical", "med", "doctor", "patient", "disease"]):
            return cls.HEALTH
        elif any(kw in filename for kw in ["general", "all", "multi"]):
            return cls.GENERAL
        
        return cls.OTHER

class ModelInfo:
    """Model information."""
    
    def __init__(self, 
                path: str,
                model_type: ModelType = None,
                name: Optional[str] = None,
                description: Optional[str] = None,
                version: Optional[str] = None,
                size: Optional[int] = None,
                format: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize model information.
        
        Args:
            path: Path to model file
            model_type: Model type
            name: Model name
            description: Model description
            version: Model version
            size: Model size in bytes
            format: Model format
            metadata: Additional metadata
        """
        self.path = os.path.abspath(path)
        self.name = name or os.path.basename(path)
        self.description = description or ""
        self.version = version or "1.0.0"
        self.size = size or (os.path.getsize(path) if os.path.exists(path) else 0)
        self.format = format or os.path.splitext(path)[1].lstrip(".")
        self.metadata = metadata or {}
        
        # Determine model type if not provided
        if model_type is None:
            self.model_type = ModelType.from_filename(self.name)
        else:
            self.model_type = model_type
        
        # Generate model ID
        self.id = self._generate_id()
        
        # Set creation time
        self.created_at = datetime.datetime.now().isoformat()
    
    def _generate_id(self) -> str:
        """
        Generate model ID.
        
        Returns:
            Model ID
        """
        # Use path and name to generate a unique ID
        data = f"{self.path}:{self.name}:{self.version}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "path": self.path,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "size": self.size,
            "format": self.format,
            "model_type": self.model_type.value,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """
        Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Model information
        """
        model_type = ModelType.from_string(data.get("model_type", "other"))
        
        info = cls(
            path=data["path"],
            model_type=model_type,
            name=data.get("name"),
            description=data.get("description"),
            version=data.get("version"),
            size=data.get("size"),
            format=data.get("format"),
            metadata=data.get("metadata", {})
        )
        
        # Set ID and creation time from data
        info.id = data.get("id", info.id)
        info.created_at = data.get("created_at", info.created_at)
        
        return info

class Model:
    """Model class."""
    
    def __init__(self, info: ModelInfo):
        """
        Initialize model.
        
        Args:
            info: Model information
        """
        self.info = info
        self.loaded = False
        self.model = None
    
    def load(self) -> bool:
        """
        Load model.
        
        Returns:
            True if successful, False otherwise
        """
        # This is a placeholder for actual model loading
        # In a real implementation, this would load the model based on its format
        self.loaded = True
        return True
    
    def unload(self) -> bool:
        """
        Unload model.
        
        Returns:
            True if successful, False otherwise
        """
        # This is a placeholder for actual model unloading
        self.loaded = False
        self.model = None
        return True
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if loaded, False otherwise
        """
        return self.loaded
    
    def get_info(self) -> ModelInfo:
        """
        Get model information.
        
        Returns:
            Model information
        """
        return self.info
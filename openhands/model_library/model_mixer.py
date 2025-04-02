"""
Model mixer for combining multiple models.

This module provides functionality for mixing multiple models,
automatically selecting the most appropriate model based on the input.
"""

import os
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from openhands.model_library.model import Model, ModelType, ModelInfo
from openhands.model_library.library import get_model_library

logger = logging.getLogger("model_mixer")

class ModelSelector:
    """Model selector for choosing the appropriate model."""
    
    def __init__(self):
        """Initialize model selector."""
        self.rules = []
        self.default_model_id = None
    
    def add_rule(self, condition: Callable[[str], bool], model_id: str, priority: int = 0) -> None:
        """
        Add selection rule.
        
        Args:
            condition: Function that takes input text and returns True if the rule applies
            model_id: Model ID to use if the rule applies
            priority: Rule priority (higher priority rules are checked first)
        """
        self.rules.append({
            "condition": condition,
            "model_id": model_id,
            "priority": priority
        })
        
        # Sort rules by priority (descending)
        self.rules.sort(key=lambda x: x["priority"], reverse=True)
    
    def set_default_model(self, model_id: str) -> None:
        """
        Set default model.
        
        Args:
            model_id: Default model ID
        """
        self.default_model_id = model_id
    
    def select_model(self, input_text: str) -> Optional[str]:
        """
        Select model based on input.
        
        Args:
            input_text: Input text
            
        Returns:
            Selected model ID or None if no model matches
        """
        # Check rules
        for rule in self.rules:
            try:
                if rule["condition"](input_text):
                    return rule["model_id"]
            except Exception as e:
                logger.error(f"Error applying rule: {e}")
        
        # Return default model if no rule matches
        return self.default_model_id

class ModelMixer:
    """Model mixer for combining multiple models."""
    
    def __init__(self):
        """Initialize model mixer."""
        self.model_library = get_model_library()
        self.selector = ModelSelector()
        self.loaded_models = {}
        self.lock = threading.RLock()
    
    def add_model(self, model_id: str) -> bool:
        """
        Add model to mixer.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Check if model exists
            model = self.model_library.get_model(model_id)
            if not model:
                logger.warning(f"Model not found: {model_id}")
                return False
            
            # Load model if not already loaded
            if model_id not in self.loaded_models:
                if not model.is_loaded():
                    if not model.load():
                        logger.error(f"Failed to load model: {model_id}")
                        return False
                
                self.loaded_models[model_id] = model
            
            return True
    
    def remove_model(self, model_id: str) -> bool:
        """
        Remove model from mixer.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if model_id in self.loaded_models:
                # Unload model
                model = self.loaded_models[model_id]
                model.unload()
                
                # Remove from loaded models
                del self.loaded_models[model_id]
                
                return True
            
            return False
    
    def add_rule(self, condition: Callable[[str], bool], model_id: str, priority: int = 0) -> bool:
        """
        Add selection rule.
        
        Args:
            condition: Function that takes input text and returns True if the rule applies
            model_id: Model ID to use if the rule applies
            priority: Rule priority (higher priority rules are checked first)
            
        Returns:
            True if successful, False otherwise
        """
        # Check if model exists
        if not self.model_library.get_model(model_id):
            logger.warning(f"Model not found: {model_id}")
            return False
        
        # Add rule
        self.selector.add_rule(condition, model_id, priority)
        return True
    
    def set_default_model(self, model_id: str) -> bool:
        """
        Set default model.
        
        Args:
            model_id: Default model ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if model exists
        if not self.model_library.get_model(model_id):
            logger.warning(f"Model not found: {model_id}")
            return False
        
        # Set default model
        self.selector.set_default_model(model_id)
        return True
    
    def process(self, input_text: str) -> Tuple[Any, str]:
        """
        Process input with the appropriate model.
        
        Args:
            input_text: Input text
            
        Returns:
            Tuple of (output, model_id)
        """
        with self.lock:
            # Select model
            model_id = self.selector.select_model(input_text)
            if not model_id:
                raise ValueError("No model selected")
            
            # Make sure model is loaded
            if model_id not in self.loaded_models:
                if not self.add_model(model_id):
                    raise ValueError(f"Failed to load model: {model_id}")
            
            # Get model
            model = self.loaded_models[model_id]
            
            # Process input
            # This is a placeholder for actual model processing
            # In a real implementation, this would use the model's processing capabilities
            output = f"Processed with model {model.info.name}: {input_text}"
            
            return output, model_id
    
    def get_loaded_models(self) -> Dict[str, Model]:
        """
        Get loaded models.
        
        Returns:
            Dictionary mapping model IDs to models
        """
        return self.loaded_models.copy()
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get model information.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information or None if not found
        """
        model = self.loaded_models.get(model_id)
        if model:
            return model.get_info()
        return None

# Create common rule conditions
def contains_keywords(keywords: List[str]) -> Callable[[str], bool]:
    """
    Create condition that checks if input contains any of the keywords.
    
    Args:
        keywords: List of keywords
        
    Returns:
        Condition function
    """
    return lambda input_text: any(keyword.lower() in input_text.lower() for keyword in keywords)

def matches_regex(pattern: str) -> Callable[[str], bool]:
    """
    Create condition that checks if input matches regex pattern.
    
    Args:
        pattern: Regex pattern
        
    Returns:
        Condition function
    """
    import re
    compiled = re.compile(pattern)
    return lambda input_text: bool(compiled.search(input_text))

def is_programming_query() -> Callable[[str], bool]:
    """
    Create condition that checks if input is a programming query.
    
    Returns:
        Condition function
    """
    programming_keywords = [
        "code", "program", "function", "class", "method",
        "python", "javascript", "java", "c++", "html", "css",
        "algorithm", "data structure", "compile", "debug"
    ]
    return contains_keywords(programming_keywords)

def is_health_query() -> Callable[[str], bool]:
    """
    Create condition that checks if input is a health query.
    
    Returns:
        Condition function
    """
    health_keywords = [
        "health", "medical", "doctor", "symptom", "disease",
        "medicine", "treatment", "diagnosis", "pain", "diet",
        "exercise", "nutrition", "vitamin", "protein", "carbohydrate"
    ]
    return contains_keywords(health_keywords)

def is_literature_query() -> Callable[[str], bool]:
    """
    Create condition that checks if input is a literature query.
    
    Returns:
        Condition function
    """
    literature_keywords = [
        "book", "novel", "story", "poem", "poetry", "author",
        "character", "plot", "theme", "setting", "genre",
        "fiction", "non-fiction", "literature", "write", "writing"
    ]
    return contains_keywords(literature_keywords)

# Global model mixer instance
_model_mixer = None

def get_model_mixer() -> ModelMixer:
    """
    Get the global model mixer instance.
    
    Returns:
        Model mixer instance
    """
    global _model_mixer
    
    if _model_mixer is None:
        _model_mixer = ModelMixer()
    
    return _model_mixer
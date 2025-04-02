"""
Context definitions for context awareness.

This module provides context definitions for context awareness,
including context types, context factors, and context classes.
"""

import enum
import datetime
from typing import Dict, List, Any, Optional, Union

class ContextType(enum.Enum):
    """Context type enumeration."""
    TIME = "time"
    LOCATION = "location"
    DEVICE = "device"
    USER = "user"
    ENVIRONMENT = "environment"
    ACTIVITY = "activity"
    SOCIAL = "social"
    HEALTH = "health"
    CUSTOM = "custom"

class ContextFactor:
    """Context factor class."""
    
    def __init__(self, name: str, value: Any, confidence: float = 1.0, timestamp: Optional[datetime.datetime] = None):
        """
        Initialize context factor.
        
        Args:
            name: Factor name
            value: Factor value
            confidence: Confidence level (0.0 to 1.0)
            timestamp: Timestamp (defaults to current time)
        """
        self.name = name
        self.value = value
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp to [0.0, 1.0]
        self.timestamp = timestamp or datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextFactor":
        """
        Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Context factor
        """
        timestamp = None
        if "timestamp" in data:
            try:
                timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                pass
        
        return cls(
            name=data["name"],
            value=data["value"],
            confidence=data.get("confidence", 1.0),
            timestamp=timestamp
        )

class Context:
    """Context class."""
    
    def __init__(self, context_type: ContextType, factors: Optional[List[ContextFactor]] = None):
        """
        Initialize context.
        
        Args:
            context_type: Context type
            factors: List of context factors
        """
        self.context_type = context_type
        self.factors = factors or []
        self.timestamp = datetime.datetime.now()
    
    def add_factor(self, factor: ContextFactor) -> None:
        """
        Add context factor.
        
        Args:
            factor: Context factor
        """
        self.factors.append(factor)
    
    def get_factor(self, name: str) -> Optional[ContextFactor]:
        """
        Get context factor by name.
        
        Args:
            name: Factor name
            
        Returns:
            Context factor or None if not found
        """
        for factor in self.factors:
            if factor.name == name:
                return factor
        return None
    
    def get_factor_value(self, name: str, default: Any = None) -> Any:
        """
        Get context factor value by name.
        
        Args:
            name: Factor name
            default: Default value if factor not found
            
        Returns:
            Factor value or default if not found
        """
        factor = self.get_factor(name)
        if factor:
            return factor.value
        return default
    
    def update_factor(self, name: str, value: Any, confidence: Optional[float] = None) -> bool:
        """
        Update context factor.
        
        Args:
            name: Factor name
            value: New factor value
            confidence: New confidence level (if None, keep existing confidence)
            
        Returns:
            True if factor was updated, False if not found
        """
        for i, factor in enumerate(self.factors):
            if factor.name == name:
                new_confidence = confidence if confidence is not None else factor.confidence
                self.factors[i] = ContextFactor(name, value, new_confidence)
                return True
        return False
    
    def remove_factor(self, name: str) -> bool:
        """
        Remove context factor.
        
        Args:
            name: Factor name
            
        Returns:
            True if factor was removed, False if not found
        """
        for i, factor in enumerate(self.factors):
            if factor.name == name:
                del self.factors[i]
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "context_type": self.context_type.value,
            "factors": [factor.to_dict() for factor in self.factors],
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Context":
        """
        Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Context
        """
        context_type = ContextType(data["context_type"])
        factors = [ContextFactor.from_dict(factor_data) for factor_data in data.get("factors", [])]
        
        context = cls(context_type, factors)
        
        if "timestamp" in data:
            try:
                context.timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                pass
        
        return context
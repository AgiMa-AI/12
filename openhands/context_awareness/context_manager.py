"""
Context manager for context awareness.

This module provides a context manager for managing context information,
including gathering, storing, and retrieving context.
"""

import os
import json
import logging
import threading
import datetime
from typing import Dict, List, Any, Optional, Union, Type

from openhands.context_awareness.context import Context, ContextType, ContextFactor
from openhands.context_awareness.adapters import (
    ContextAdapter, TimeAdapter, LocationAdapter, DeviceAdapter, UserAdapter
)

logger = logging.getLogger("context_manager")

class ContextManager:
    """Context manager for managing context information."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize context manager.
        
        Args:
            storage_path: Path to context storage directory
        """
        # Set default storage path if not provided
        if storage_path is None:
            storage_path = os.path.join(os.path.expanduser("~"), "openhands", "context")
        
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize adapters
        self.adapters: Dict[ContextType, ContextAdapter] = {}
        
        # Register default adapters
        self.register_adapter(TimeAdapter())
        self.register_adapter(LocationAdapter())
        self.register_adapter(DeviceAdapter())
        self.register_adapter(UserAdapter())
        
        # Initialize context cache
        self.context_cache: Dict[ContextType, Context] = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize context history
        self.context_history: Dict[ContextType, List[Context]] = {
            context_type: [] for context_type in ContextType
        }
        
        logger.info(f"Initialized context manager with storage path: {storage_path}")
    
    def register_adapter(self, adapter: ContextAdapter) -> None:
        """
        Register context adapter.
        
        Args:
            adapter: Context adapter
        """
        with self.lock:
            self.adapters[adapter.context_type] = adapter
            logger.info(f"Registered {adapter.context_type.value} adapter")
    
    def unregister_adapter(self, context_type: ContextType) -> bool:
        """
        Unregister context adapter.
        
        Args:
            context_type: Context type
            
        Returns:
            True if adapter was unregistered, False if not found
        """
        with self.lock:
            if context_type in self.adapters:
                del self.adapters[context_type]
                logger.info(f"Unregistered {context_type.value} adapter")
                return True
            return False
    
    def get_context(self, context_type: ContextType, use_cache: bool = True, max_age: Optional[float] = None) -> Optional[Context]:
        """
        Get context.
        
        Args:
            context_type: Context type
            use_cache: Whether to use cached context
            max_age: Maximum age of cached context in seconds
            
        Returns:
            Context or None if not available
        """
        with self.lock:
            # Check if adapter is registered
            if context_type not in self.adapters:
                logger.warning(f"No adapter registered for {context_type.value}")
                return None
            
            # Check if cached context is available and not too old
            if use_cache and context_type in self.context_cache:
                cached_context = self.context_cache[context_type]
                
                # Check if cached context is not too old
                if max_age is None or (datetime.datetime.now() - cached_context.timestamp).total_seconds() <= max_age:
                    return cached_context
            
            # Get context from adapter
            try:
                context = self.adapters[context_type].get_context()
                
                # Cache context
                self.context_cache[context_type] = context
                
                # Add to history
                self.context_history[context_type].append(context)
                
                # Trim history if too long
                if len(self.context_history[context_type]) > 100:
                    self.context_history[context_type] = self.context_history[context_type][-100:]
                
                return context
            
            except Exception as e:
                logger.error(f"Failed to get {context_type.value} context: {e}")
                return None
    
    def get_all_context(self, use_cache: bool = True, max_age: Optional[float] = None) -> Dict[ContextType, Context]:
        """
        Get all context.
        
        Args:
            use_cache: Whether to use cached context
            max_age: Maximum age of cached context in seconds
            
        Returns:
            Dictionary mapping context types to contexts
        """
        with self.lock:
            result = {}
            
            for context_type in self.adapters:
                context = self.get_context(context_type, use_cache, max_age)
                if context:
                    result[context_type] = context
            
            return result
    
    def get_context_history(self, context_type: ContextType) -> List[Context]:
        """
        Get context history.
        
        Args:
            context_type: Context type
            
        Returns:
            List of contexts
        """
        with self.lock:
            return self.context_history.get(context_type, []).copy()
    
    def save_context(self, context: Context) -> bool:
        """
        Save context to storage.
        
        Args:
            context: Context to save
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Create context type directory if it doesn't exist
                context_type_dir = os.path.join(self.storage_path, context.context_type.value)
                os.makedirs(context_type_dir, exist_ok=True)
                
                # Generate filename
                timestamp = context.timestamp.strftime("%Y%m%d%H%M%S")
                filename = f"{timestamp}.json"
                filepath = os.path.join(context_type_dir, filename)
                
                # Save context
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(context.to_dict(), f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved {context.context_type.value} context to {filepath}")
                return True
            
            except Exception as e:
                logger.error(f"Failed to save {context.context_type.value} context: {e}")
                return False
    
    def load_context(self, context_type: ContextType, timestamp: str) -> Optional[Context]:
        """
        Load context from storage.
        
        Args:
            context_type: Context type
            timestamp: Timestamp string (YYYYMMDDHHMMSS)
            
        Returns:
            Context or None if not found
        """
        with self.lock:
            try:
                # Generate filepath
                filepath = os.path.join(self.storage_path, context_type.value, f"{timestamp}.json")
                
                # Check if file exists
                if not os.path.exists(filepath):
                    logger.warning(f"Context file not found: {filepath}")
                    return None
                
                # Load context
                with open(filepath, "r", encoding="utf-8") as f:
                    context_data = json.load(f)
                
                # Create context
                context = Context.from_dict(context_data)
                
                logger.info(f"Loaded {context_type.value} context from {filepath}")
                return context
            
            except Exception as e:
                logger.error(f"Failed to load {context_type.value} context: {e}")
                return None
    
    def list_context_files(self, context_type: ContextType) -> List[str]:
        """
        List context files.
        
        Args:
            context_type: Context type
            
        Returns:
            List of timestamp strings
        """
        with self.lock:
            try:
                # Generate directory path
                dirpath = os.path.join(self.storage_path, context_type.value)
                
                # Check if directory exists
                if not os.path.exists(dirpath):
                    return []
                
                # List files
                files = os.listdir(dirpath)
                
                # Filter JSON files and extract timestamps
                timestamps = []
                for file in files:
                    if file.endswith(".json"):
                        timestamp = file[:-5]  # Remove .json extension
                        timestamps.append(timestamp)
                
                # Sort timestamps
                timestamps.sort()
                
                return timestamps
            
            except Exception as e:
                logger.error(f"Failed to list {context_type.value} context files: {e}")
                return []
    
    def clear_context_cache(self) -> None:
        """Clear context cache."""
        with self.lock:
            self.context_cache.clear()
            logger.info("Cleared context cache")
    
    def clear_context_history(self) -> None:
        """Clear context history."""
        with self.lock:
            for context_type in self.context_history:
                self.context_history[context_type] = []
            logger.info("Cleared context history")
    
    def get_context_factor(self, context_type: ContextType, factor_name: str, default: Any = None) -> Any:
        """
        Get context factor value.
        
        Args:
            context_type: Context type
            factor_name: Factor name
            default: Default value if factor not found
            
        Returns:
            Factor value or default if not found
        """
        context = self.get_context(context_type)
        if context:
            return context.get_factor_value(factor_name, default)
        return default
    
    def is_morning(self) -> bool:
        """
        Check if it's morning.
        
        Returns:
            True if it's morning, False otherwise
        """
        time_of_day = self.get_context_factor(ContextType.TIME, "time_of_day")
        return time_of_day == "morning"
    
    def is_afternoon(self) -> bool:
        """
        Check if it's afternoon.
        
        Returns:
            True if it's afternoon, False otherwise
        """
        time_of_day = self.get_context_factor(ContextType.TIME, "time_of_day")
        return time_of_day == "afternoon"
    
    def is_evening(self) -> bool:
        """
        Check if it's evening.
        
        Returns:
            True if it's evening, False otherwise
        """
        time_of_day = self.get_context_factor(ContextType.TIME, "time_of_day")
        return time_of_day == "evening"
    
    def is_night(self) -> bool:
        """
        Check if it's night.
        
        Returns:
            True if it's night, False otherwise
        """
        time_of_day = self.get_context_factor(ContextType.TIME, "time_of_day")
        return time_of_day == "night"
    
    def is_weekend(self) -> bool:
        """
        Check if it's weekend.
        
        Returns:
            True if it's weekend, False otherwise
        """
        weekday = self.get_context_factor(ContextType.TIME, "weekday")
        return weekday in [5, 6]  # Saturday or Sunday
    
    def is_internet_connected(self) -> bool:
        """
        Check if internet is connected.
        
        Returns:
            True if internet is connected, False otherwise
        """
        return self.get_context_factor(ContextType.DEVICE, "internet_connected", False)
    
    def is_battery_low(self, threshold: float = 20.0) -> bool:
        """
        Check if battery is low.
        
        Args:
            threshold: Battery level threshold
            
        Returns:
            True if battery is low, False otherwise
        """
        battery_percent = self.get_context_factor(ContextType.DEVICE, "battery_percent")
        if battery_percent is None:
            return False
        
        power_plugged = self.get_context_factor(ContextType.DEVICE, "battery_power_plugged", True)
        
        return not power_plugged and battery_percent < threshold
    
    def is_system_busy(self, cpu_threshold: float = 80.0, memory_threshold: float = 80.0) -> bool:
        """
        Check if system is busy.
        
        Args:
            cpu_threshold: CPU usage threshold
            memory_threshold: Memory usage threshold
            
        Returns:
            True if system is busy, False otherwise
        """
        cpu_percent = self.get_context_factor(ContextType.DEVICE, "cpu_percent", 0.0)
        memory_percent = self.get_context_factor(ContextType.DEVICE, "memory_percent", 0.0)
        
        return cpu_percent > cpu_threshold or memory_percent > memory_threshold
    
    def get_greeting(self) -> str:
        """
        Get time-appropriate greeting.
        
        Returns:
            Greeting string
        """
        time_of_day = self.get_context_factor(ContextType.TIME, "time_of_day")
        
        if time_of_day == "morning":
            return "Good morning"
        elif time_of_day == "afternoon":
            return "Good afternoon"
        elif time_of_day == "evening":
            return "Good evening"
        else:
            return "Hello"

# Global context manager instance
_context_manager = None

def get_context_manager(storage_path: Optional[str] = None) -> ContextManager:
    """
    Get the global context manager instance.
    
    Args:
        storage_path: Path to context storage directory
        
    Returns:
        Context manager instance
    """
    global _context_manager
    
    if _context_manager is None:
        _context_manager = ContextManager(storage_path)
    
    return _context_manager
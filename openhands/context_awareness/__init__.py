"""
Context awareness module for OpenHands.

This module provides context awareness capabilities,
allowing the system to adapt to different contexts and situations.
"""

from openhands.context_awareness.context_manager import ContextManager, get_context_manager
from openhands.context_awareness.context import Context, ContextType, ContextFactor
from openhands.context_awareness.adapters import ContextAdapter, TimeAdapter, LocationAdapter, DeviceAdapter, UserAdapter

__all__ = [
    "ContextManager",
    "get_context_manager",
    "Context",
    "ContextType",
    "ContextFactor",
    "ContextAdapter",
    "TimeAdapter",
    "LocationAdapter",
    "DeviceAdapter",
    "UserAdapter"
]
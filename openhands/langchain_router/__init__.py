"""
LangChain Router for OpenHands.

This module provides a routing system for OpenHands using LangChain,
allowing intelligent routing of requests to the most appropriate AI assistant.
"""

from openhands.langchain_router.router import LangChainRouter
from openhands.langchain_router.config import RouterConfig

__all__ = ["LangChainRouter", "RouterConfig"]
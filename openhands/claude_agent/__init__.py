"""
Claude Agent for OpenHands.

This module provides a powerful agent system powered by Claude 3.7 or local models,
with advanced reasoning capabilities, tool usage, and memory management.
It integrates with OpenDear to provide a comprehensive AI assistant ecosystem.
"""

from openhands.claude_agent.agent import ClaudeAgent
from openhands.claude_agent.config import AgentConfig
from openhands.claude_agent.memory import AgentMemory
from openhands.claude_agent.tools import ToolRegistry
from openhands.claude_agent.client import ModelClient
from openhands.claude_agent.agent_router import ClaudeAgentRouter, get_claude_agent_router
from openhands.claude_agent.tool_allocator import ToolAllocator, get_tool_allocator
from openhands.claude_agent.openhands_integration import OpenHandsIntegration, get_openhands_integration
from openhands.claude_agent.init_memory import initialize_environment
from openhands.claude_agent.chinese_integration import ChineseClaudeAgent, get_chinese_claude_agent

__all__ = [
    "ClaudeAgent",
    "AgentConfig",
    "AgentMemory",
    "ToolRegistry",
    "ModelClient",
    "ClaudeAgentRouter",
    "get_claude_agent_router",
    "ToolAllocator",
    "get_tool_allocator",
    "OpenHandsIntegration",
    "get_openhands_integration",
    "initialize_environment",
    "ChineseClaudeAgent",
    "get_chinese_claude_agent"
]
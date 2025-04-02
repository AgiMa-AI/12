"""
OpenHands integration for Claude Agent.

This module provides integration with OpenHands, allowing Claude Agent
to be used as part of the OpenHands ecosystem.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

from openhands.claude_agent.agent import ClaudeAgent
from openhands.claude_agent.agent_router import ClaudeAgentRouter
from openhands.claude_agent.tool_allocator import get_tool_allocator, register_claude_agent_tools

logger = logging.getLogger("claude_agent_opendear")

class OpenHandsIntegration:
    """
    OpenHands integration for Claude Agent.
    """
    
    def __init__(self, config_file: str = "agent_config.json"):
        """
        Initialize OpenHands integration.
        
        Args:
            config_file: Path to configuration file
        """
        # Initialize Claude Agent
        self.agent = ClaudeAgent(config_file=config_file)
        
        # Initialize router
        self.router = ClaudeAgentRouter(config_file=config_file)
        
        # Register tools with tool allocator
        register_claude_agent_tools(self.agent.tool_registry)
        
        # Register with OpenHands
        self._register_with_openhands()
        
        logger.info("OpenHands integration initialized")
    
    def _register_with_openhands(self):
        """Register with OpenHands."""
        try:
            # Import OpenHands modules
            from openhands.core import agent_registry, dispatcher, capabilities_registry
            
            # Register Claude Agent with OpenHands
            agent_registry.register_agent(
                name="claude_agent",
                description="Claude Agent with advanced reasoning and tool usage capabilities",
                handler=self.router.handle_request,
                capabilities=[
                    "text_generation",
                    "reasoning",
                    "tool_usage",
                    "file_operations",
                    "code_execution",
                    "web_search",
                    "vector_search"
                ],
                priority=80  # High priority
            )
            
            # Register tools with OpenHands capabilities registry
            tool_allocator = get_tool_allocator()
            claude_tools = tool_allocator.get_agent_tools("claude_agent")
            
            for tool in claude_tools:
                capabilities_registry.register_tool(
                    name=tool["name"],
                    description=tool["description"],
                    handler=lambda params, tool_name=tool["name"]: self.router.execute_tool(tool_name, params),
                    schema=tool["schema"]
                )
            
            logger.info("Registered Claude Agent with OpenHands")
        
        except ImportError:
            logger.warning("OpenHands modules not found, skipping registration")
        except Exception as e:
            logger.error(f"Failed to register with OpenHands: {e}")
    
    def route_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route request to Claude Agent.
        
        Args:
            request: Request from OpenDear
            
        Returns:
            Response from Claude Agent
        """
        return self.router.handle_request(request)
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            
        Returns:
            Tool execution result
        """
        return self.router.execute_tool(tool_name, params)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get Claude Agent capabilities.
        
        Returns:
            Dictionary of capabilities
        """
        tool_allocator = get_tool_allocator()
        
        return {
            "agent": "claude_agent",
            "capabilities": [
                "text_generation",
                "reasoning",
                "tool_usage",
                "file_operations",
                "code_execution",
                "web_search",
                "vector_search"
            ],
            "tools": tool_allocator.get_agent_tools("claude_agent"),
            "models": [self.agent.config.model] if not self.agent.config.use_local_model else ["local-model"]
        }

def get_openhands_integration(config_file: str = "agent_config.json") -> OpenHandsIntegration:
    """
    Get OpenHands integration.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        OpenHands integration
    """
    return OpenHandsIntegration(config_file=config_file)
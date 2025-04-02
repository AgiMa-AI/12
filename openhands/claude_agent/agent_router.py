"""
Agent router for OpenHands integration.

This module provides a router for integrating Claude Agent with OpenHands,
allowing OpenHands to dispatch requests to Claude Agent.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union

from openhands.claude_agent.agent import ClaudeAgent
from openhands.claude_agent.config import AgentConfig
from openhands.claude_agent.tools import ToolRegistry

logger = logging.getLogger("claude_agent_router")

class ClaudeAgentRouter:
    """
    Router for integrating Claude Agent with OpenDear.
    """
    
    def __init__(self, config_file: str = "agent_config.json"):
        """
        Initialize the Claude Agent router.
        
        Args:
            config_file: Path to configuration file
        """
        # Initialize agent
        self.agent = ClaudeAgent(config_file=config_file)
        self.config = self.agent.config
        
        # Register with OpenDear capabilities registry
        self._register_capabilities()
        
        logger.info("Claude Agent router initialized")
    
    def _register_capabilities(self):
        """Register capabilities with OpenHands."""
        try:
            from openhands.core import capabilities_registry
            
            # Register Claude Agent capabilities
            capabilities_registry.register_agent(
                name="claude_agent",
                description="Claude Agent with advanced reasoning and tool usage capabilities",
                handler=self.handle_request,
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
            
            # Register tools
            for tool in self.agent.tool_registry.get_all_tools():
                capabilities_registry.register_tool(
                    name=tool["name"],
                    description=tool["description"],
                    handler=lambda params, tool_name=tool["name"]: self.execute_tool(tool_name, params),
                    schema=tool["input_schema"]
                )
            
            logger.info("Registered Claude Agent capabilities with OpenHands")
        
        except ImportError:
            logger.warning("OpenHands capabilities registry not found, skipping registration")
        except Exception as e:
            logger.error(f"Failed to register capabilities: {e}")
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle request from OpenHands.
        
        Args:
            request: Request from OpenHands
            
        Returns:
            Response to OpenHands
        """
        try:
            # Extract request parameters
            query = request.get("query", "")
            conversation_id = request.get("conversation_id", "")
            context = request.get("context", {})
            tools_allowed = request.get("tools_allowed", True)
            
            # Process system instructions if provided
            if "system_instructions" in request:
                self._set_system_instructions(request["system_instructions"])
            
            # Process the query
            response = self.agent.process_input(query)
            
            # Format response
            if hasattr(response, "content"):
                result = {
                    "status": "success",
                    "content": response.content,
                    "model": getattr(response, "model", "unknown"),
                    "agent": "claude_agent",
                    "tool_calls": [],  # Will be populated if tools were used
                    "metadata": {
                        "stop_reason": getattr(response, "stop_reason", "unknown"),
                        "usage": getattr(response, "usage", {})
                    }
                }
                
                # Add tool calls if any
                if hasattr(response, "tool_calls") and response.tool_calls:
                    result["tool_calls"] = [
                        {
                            "name": tool_call.name,
                            "input": tool_call.input
                        }
                        for tool_call in response.tool_calls
                    ]
            else:
                result = {
                    "status": "error",
                    "message": "Failed to get response content",
                    "error": getattr(response, "error", "Unknown error"),
                    "agent": "claude_agent"
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to handle request: {e}")
            return {
                "status": "error",
                "message": f"Failed to handle request: {str(e)}",
                "agent": "claude_agent"
            }
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            # Execute tool
            result = self.agent.tool_registry.execute_tool(
                tool_name,
                params,
                timeout=self.agent.config.tool_timeout
            )
            
            return {
                "status": "success",
                "tool": tool_name,
                "result": result,
                "agent": "claude_agent"
            }
        
        except Exception as e:
            logger.error(f"Failed to execute tool: {e}")
            return {
                "status": "error",
                "message": f"Failed to execute tool: {str(e)}",
                "agent": "claude_agent"
            }
    
    def _set_system_instructions(self, instructions: str):
        """
        Set system instructions.
        
        Args:
            instructions: System instructions
        """
        # Update system prompt
        if len(self.agent.conversation_context) > 0 and self.agent.conversation_context[0]["role"] == "system":
            self.agent.conversation_context[0]["content"] = instructions
        else:
            self.agent.conversation_context.insert(0, {
                "role": "system",
                "content": instructions
            })

def get_claude_agent_router(config_file: str = "agent_config.json") -> ClaudeAgentRouter:
    """
    Get Claude Agent router.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Claude Agent router
    """
    return ClaudeAgentRouter(config_file=config_file)
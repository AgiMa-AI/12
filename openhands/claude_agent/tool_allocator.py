"""
Tool allocator for OpenHands integration.

This module provides a tool allocator for distributing tools among different agents,
ensuring optimal tool usage based on agent capabilities.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger("claude_agent_tool_allocator")

class ToolAllocator:
    """
    Tool allocator for distributing tools among different agents.
    """
    
    def __init__(self):
        """Initialize the tool allocator."""
        # Tool registry
        self.tools = {}
        
        # Agent capabilities
        self.agent_capabilities = {}
        
        # Tool allocations
        self.allocations = {}
        
        logger.info("Tool allocator initialized")
    
    def register_tool(self, name: str, description: str, handler: Any, schema: Dict[str, Any], 
                     categories: List[str] = None, source: str = None) -> None:
        """
        Register a tool.
        
        Args:
            name: Tool name
            description: Tool description
            handler: Tool handler function
            schema: Tool schema
            categories: Tool categories
            source: Tool source (e.g., agent name)
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "schema": schema,
            "categories": categories or [],
            "source": source
        }
        
        logger.info(f"Registered tool: {name}")
    
    def register_agent_capabilities(self, agent_name: str, capabilities: List[str], 
                                   priority: int = 50, preferred_tools: List[str] = None) -> None:
        """
        Register agent capabilities.
        
        Args:
            agent_name: Agent name
            capabilities: List of capabilities
            priority: Agent priority (0-100)
            preferred_tools: List of preferred tools
        """
        self.agent_capabilities[agent_name] = {
            "capabilities": set(capabilities),
            "priority": priority,
            "preferred_tools": set(preferred_tools or [])
        }
        
        logger.info(f"Registered agent capabilities: {agent_name}")
    
    def allocate_tools(self) -> Dict[str, List[str]]:
        """
        Allocate tools to agents based on capabilities and priorities.
        
        Returns:
            Dictionary mapping agent names to lists of allocated tool names
        """
        # Reset allocations
        self.allocations = {agent_name: [] for agent_name in self.agent_capabilities}
        
        # Sort agents by priority (highest first)
        sorted_agents = sorted(
            self.agent_capabilities.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )
        
        # First pass: allocate preferred tools
        for agent_name, agent_info in sorted_agents:
            for tool_name in self.tools:
                if tool_name in agent_info["preferred_tools"]:
                    self.allocations[agent_name].append(tool_name)
        
        # Second pass: allocate tools based on capabilities
        for agent_name, agent_info in sorted_agents:
            for tool_name, tool_info in self.tools.items():
                # Skip already allocated tools
                if tool_name in self.allocations[agent_name]:
                    continue
                
                # Check if agent has required capabilities
                if self._agent_can_handle_tool(agent_info["capabilities"], tool_info["categories"]):
                    self.allocations[agent_name].append(tool_name)
        
        logger.info("Tools allocated to agents")
        return self.allocations
    
    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Get tools allocated to an agent.
        
        Args:
            agent_name: Agent name
            
        Returns:
            List of tool definitions
        """
        if agent_name not in self.allocations:
            return []
        
        return [
            {
                "name": tool_name,
                "description": self.tools[tool_name]["description"],
                "schema": self.tools[tool_name]["schema"]
            }
            for tool_name in self.allocations[agent_name]
            if tool_name in self.tools
        ]
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Tool name
            params: Tool parameters
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return {
                "status": "error",
                "message": f"Tool not found: {tool_name}"
            }
        
        try:
            # Execute tool
            result = self.tools[tool_name]["handler"](params)
            
            return {
                "status": "success",
                "tool": tool_name,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Failed to execute tool: {e}")
            return {
                "status": "error",
                "message": f"Failed to execute tool: {str(e)}"
            }
    
    def _agent_can_handle_tool(self, agent_capabilities: Set[str], tool_categories: List[str]) -> bool:
        """
        Check if an agent can handle a tool.
        
        Args:
            agent_capabilities: Agent capabilities
            tool_categories: Tool categories
            
        Returns:
            True if agent can handle tool, False otherwise
        """
        # If tool has no categories, any agent can handle it
        if not tool_categories:
            return True
        
        # Check if agent has at least one required capability
        return any(category in agent_capabilities for category in tool_categories)
    
    def get_tool_categories(self) -> List[str]:
        """
        Get all tool categories.
        
        Returns:
            List of tool categories
        """
        categories = set()
        for tool_info in self.tools.values():
            categories.update(tool_info["categories"])
        
        return sorted(list(categories))
    
    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get tools by category.
        
        Args:
            category: Tool category
            
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": tool_name,
                "description": tool_info["description"],
                "schema": tool_info["schema"]
            }
            for tool_name, tool_info in self.tools.items()
            if category in tool_info["categories"]
        ]
    
    def get_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Get agent capabilities.
        
        Returns:
            Dictionary mapping agent names to capability information
        """
        return {
            agent_name: {
                "capabilities": list(agent_info["capabilities"]),
                "priority": agent_info["priority"],
                "preferred_tools": list(agent_info["preferred_tools"]),
                "allocated_tools": self.allocations.get(agent_name, [])
            }
            for agent_name, agent_info in self.agent_capabilities.items()
        }

# Global tool allocator instance
_tool_allocator = None

def get_tool_allocator() -> ToolAllocator:
    """
    Get the global tool allocator instance.
    
    Returns:
        Tool allocator instance
    """
    global _tool_allocator
    
    if _tool_allocator is None:
        _tool_allocator = ToolAllocator()
    
    return _tool_allocator

def register_claude_agent_tools(tool_registry, agent_name="claude_agent"):
    """
    Register Claude Agent tools with the tool allocator.
    
    Args:
        tool_registry: Claude Agent tool registry
        agent_name: Agent name
    """
    allocator = get_tool_allocator()
    
    # Register agent capabilities
    allocator.register_agent_capabilities(
        agent_name=agent_name,
        capabilities=[
            "text_generation",
            "reasoning",
            "tool_usage",
            "file_operations",
            "code_execution",
            "web_search",
            "vector_search"
        ],
        priority=80,  # High priority
        preferred_tools=[
            "read_file",
            "write_file",
            "list_directory",
            "run_command",
            "run_python_code",
            "http_request",
            "process_csv"
        ]
    )
    
    # Register tools
    for tool in tool_registry.get_all_tools():
        # Determine tool categories
        categories = []
        
        # File operations
        if tool["name"] in ["read_file", "write_file", "list_directory"]:
            categories.append("file_operations")
        
        # System operations
        elif tool["name"] in ["run_command", "get_system_info"]:
            categories.append("system_operations")
        
        # Network operations
        elif tool["name"] in ["http_request"]:
            categories.append("network_operations")
        
        # Data processing
        elif tool["name"] in ["process_csv"]:
            categories.append("data_processing")
        
        # Code execution
        elif tool["name"] in ["run_python_code"]:
            categories.append("code_execution")
        
        # Vector operations
        elif tool["name"] in ["AddFileToVectorDB", "SearchVectorDB"]:
            categories.append("vector_operations")
        
        # Desktop operations
        elif tool["name"] in ["TakeScreenshot", "MouseClick", "KeyboardType", "GetMousePosition"]:
            categories.append("desktop_operations")
        
        # Integration
        elif tool["name"] in ["langchain_router", "enhanced_reasoning"]:
            categories.append("integration")
        
        # Register tool
        allocator.register_tool(
            name=tool["name"],
            description=tool["description"],
            handler=lambda params, tool_name=tool["name"]: tool_registry.execute_tool(tool_name, params),
            schema=tool["input_schema"],
            categories=categories,
            source=agent_name
        )
    
    # Allocate tools
    allocator.allocate_tools()
    
    logger.info(f"Registered Claude Agent tools with tool allocator")
#!/usr/bin/env python3
"""
Example script demonstrating the integration of Claude Agent with OpenHands.

This script shows how to integrate Claude Agent with OpenHands,
allowing OpenHands to dispatch requests to Claude Agent.
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, List, Any

# Add the parent directory to the path so we can import openhands
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import OpenHands modules (if available)
try:
    from openhands.core import agent_registry, dispatcher, capabilities_registry
    OPENHANDS_AVAILABLE = True
except ImportError:
    OPENHANDS_AVAILABLE = False

# Import Claude Agent modules
from openhands.claude_agent import (
    ClaudeAgent,
    AgentConfig,
    get_openhands_integration,
    get_tool_allocator,
    initialize_environment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("openhands_claude_integration")

def simulate_openhands_dispatch(integration, query: str) -> Dict[str, Any]:
    """
    Simulate OpenHands dispatch.
    
    Args:
        integration: OpenHands integration
        query: User query
        
    Returns:
        Response from Claude Agent
    """
    # Create request
    request = {
        "query": query,
        "conversation_id": "simulated-conversation",
        "context": {},
        "tools_allowed": True
    }
    
    # Route request
    return integration.route_request(request)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OpenHands Claude Integration Example")
    parser.add_argument("--config", help="Path to configuration file", default="agent_config.json")
    parser.add_argument("--local", help="Use local model", action="store_true")
    parser.add_argument("--local-model", help="Path to local model", default=None)
    parser.add_argument("--query", help="Query to process", default=None)
    
    args = parser.parse_args()
    
    # Set local model environment variables if provided
    if args.local:
        os.environ["USE_LOCAL_MODEL"] = "true"
    if args.local_model:
        os.environ["LOCAL_MODEL_PATH"] = args.local_model
    
    try:
        # Initialize environment
        env = initialize_environment(
            config_file=args.config,
            with_examples=True
        )
        
        # Get OpenHands integration
        integration = get_openhands_integration(config_file=args.config)
        
        # Get tool allocator
        tool_allocator = get_tool_allocator()
        
        # Print capabilities
        print("\n=== Claude Agent Capabilities ===")
        capabilities = integration.get_capabilities()
        print(f"Agent: {capabilities['agent']}")
        print(f"Capabilities: {', '.join(capabilities['capabilities'])}")
        print(f"Models: {', '.join(capabilities['models'])}")
        print(f"Tools: {len(capabilities['tools'])}")
        
        # Print tool allocations
        print("\n=== Tool Allocations ===")
        agent_capabilities = tool_allocator.get_agent_capabilities()
        for agent_name, agent_info in agent_capabilities.items():
            print(f"Agent: {agent_name}")
            print(f"  Priority: {agent_info['priority']}")
            print(f"  Capabilities: {', '.join(agent_info['capabilities'])}")
            print(f"  Allocated Tools: {len(agent_info['allocated_tools'])}")
        
        # Process query if provided
        if args.query:
            print(f"\n=== Processing Query: {args.query} ===")
            response = simulate_openhands_dispatch(integration, args.query)
            
            print("\n=== Response ===")
            if response["status"] == "success":
                print(f"Content: {response['content']}")
                
                if "tool_calls" in response and response["tool_calls"]:
                    print("\nTool Calls:")
                    for tool_call in response["tool_calls"]:
                        print(f"  Tool: {tool_call['name']}")
                        print(f"  Input: {json.dumps(tool_call['input'], indent=2)}")
            else:
                print(f"Error: {response.get('message', 'Unknown error')}")
        
        # If OpenHands is available, show integration status
        if OPENHANDS_AVAILABLE:
            print("\n=== OpenHands Integration ===")
            print("OpenHands is available")
            print(f"Registered Agents: {len(agent_registry.get_agents())}")
            print(f"Registered Tools: {len(capabilities_registry.get_tools())}")
        else:
            print("\n=== OpenHands Integration ===")
            print("OpenHands is not available")
            print("This example is running in simulation mode")
        
        print("\nIntegration successful!")
        
    except Exception as e:
        logger.error(f"Failed to run example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
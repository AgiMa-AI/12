"""
Initialization memory system for Claude Agent.

This module provides functions to initialize the memory system,
including creating default memories and loading example data.
"""

import os
import json
import logging
import datetime
import pickle
from typing import Dict, List, Any, Optional

from openhands.claude_agent.memory import AgentMemory
from openhands.claude_agent.config import AgentConfig

logger = logging.getLogger("claude_agent")

def initialize_memory(config: AgentConfig) -> AgentMemory:
    """
    Initialize the memory system.
    
    Args:
        config: Agent configuration
        
    Returns:
        Initialized memory system
    """
    # Create memory instance
    memory = AgentMemory(config.memory_file)
    
    # If memory already exists, just return it
    if memory.loaded:
        logger.info("Memory already exists, using existing memory")
        return memory
    
    # Otherwise, initialize with default memories
    logger.info("Initializing new memory system")
    
    # Add system information to long-term memory
    memory.add_to_long_term_memory("system_info", {
        "initialization_date": datetime.datetime.now().isoformat(),
        "version": "1.0.0",
        "platform": os.name,
        "python_version": ".".join(map(str, os.sys.version_info[:3])),
        "config_path": config.config_file
    })
    
    # Add user preferences to long-term memory
    memory.add_to_long_term_memory("user_preferences", {
        "language": "auto-detect",
        "response_style": "balanced",
        "detail_level": "medium",
        "favorite_tools": []
    })
    
    # Add welcome message to conversation history
    memory.add_conversation("assistant", "Hello! I'm your AI assistant powered by Claude. I can help you with various tasks using my tools and capabilities. What would you like to do today?")
    
    # Save the initialized memory
    memory.save_memory()
    
    return memory

def add_example_data(memory: AgentMemory) -> None:
    """
    Add example data to memory for demonstration purposes.
    
    Args:
        memory: Memory system to add data to
    """
    # Add example conversation
    memory.add_conversation("user", "What can you help me with?")
    memory.add_conversation("assistant", "I can help you with many tasks, including:\n\n1. File operations (reading, writing, listing files)\n2. Running system commands\n3. Processing data\n4. Executing Python code\n5. Searching the web\n6. Working with vector databases\n\nJust let me know what you'd like to do!")
    
    # Add example long-term memory items
    memory.add_to_long_term_memory("example_project", {
        "name": "Example Project",
        "description": "This is an example project to demonstrate long-term memory",
        "created_at": datetime.datetime.now().isoformat(),
        "tasks": [
            "Task 1: Create project structure",
            "Task 2: Implement core functionality",
            "Task 3: Write tests"
        ]
    })
    
    # Save the memory with example data
    memory.save_memory()
    
    logger.info("Added example data to memory")

def connect_frontend(memory: AgentMemory, frontend_path: str) -> bool:
    """
    Connect memory system to frontend by creating a frontend-compatible memory file.
    
    Args:
        memory: Memory system
        frontend_path: Path to frontend directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create frontend directory if it doesn't exist
        os.makedirs(frontend_path, exist_ok=True)
        
        # Create frontend-compatible memory file
        frontend_memory = {
            "conversations": memory.conversation_history,
            "preferences": memory.get_from_long_term_memory("user_preferences") or {},
            "system_info": memory.get_from_long_term_memory("system_info") or {},
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # Save to frontend memory file
        frontend_memory_path = os.path.join(frontend_path, "frontend_memory.json")
        with open(frontend_memory_path, "w", encoding="utf-8") as f:
            json.dump(frontend_memory, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Connected memory to frontend at {frontend_memory_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to connect memory to frontend: {e}")
        return False

def sync_with_frontend(memory: AgentMemory, frontend_path: str) -> bool:
    """
    Synchronize memory with frontend.
    
    Args:
        memory: Memory system
        frontend_path: Path to frontend directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if frontend memory file exists
        frontend_memory_path = os.path.join(frontend_path, "frontend_memory.json")
        if not os.path.exists(frontend_memory_path):
            # If not, create it
            return connect_frontend(memory, frontend_path)
        
        # Load frontend memory
        with open(frontend_memory_path, "r", encoding="utf-8") as f:
            frontend_memory = json.load(f)
        
        # Check if frontend memory is newer than our memory
        frontend_last_updated = frontend_memory.get("last_updated", "")
        if frontend_last_updated:
            frontend_timestamp = datetime.datetime.fromisoformat(frontend_last_updated)
            
            # Get our last conversation timestamp
            our_last_updated = None
            if memory.conversation_history:
                last_conv = memory.conversation_history[-1]
                if "timestamp" in last_conv:
                    our_last_updated = datetime.datetime.fromisoformat(last_conv["timestamp"])
            
            # If frontend is newer, update our memory
            if our_last_updated is None or frontend_timestamp > our_last_updated:
                # Update conversations
                if "conversations" in frontend_memory:
                    # Find new conversations
                    our_conv_count = len(memory.conversation_history)
                    frontend_conv_count = len(frontend_memory["conversations"])
                    
                    if frontend_conv_count > our_conv_count:
                        # Add new conversations
                        for i in range(our_conv_count, frontend_conv_count):
                            conv = frontend_memory["conversations"][i]
                            memory.conversation_history.append(conv)
                
                # Update preferences
                if "preferences" in frontend_memory:
                    memory.add_to_long_term_memory("user_preferences", frontend_memory["preferences"])
                
                # Save our updated memory
                memory.save_memory()
                logger.info("Updated memory from frontend")
        
        # Update frontend with our latest memory
        return connect_frontend(memory, frontend_path)
    
    except Exception as e:
        logger.error(f"Failed to sync with frontend: {e}")
        return False

def create_client_config(config: AgentConfig, client_path: str) -> bool:
    """
    Create client configuration file.
    
    Args:
        config: Agent configuration
        client_path: Path to client directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create client directory if it doesn't exist
        os.makedirs(client_path, exist_ok=True)
        
        # Create client configuration
        client_config = {
            "api_mode": not config.use_local_model,
            "model": config.model if not config.use_local_model else "local-model",
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "thinking_mode": config.thinking_mode,
            "server_url": "http://localhost:8000",  # Default server URL
            "memory_path": config.memory_file,
            "working_directory": config.working_directory,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # Save to client configuration file
        client_config_path = os.path.join(client_path, "client_config.json")
        with open(client_config_path, "w", encoding="utf-8") as f:
            json.dump(client_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created client configuration at {client_config_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create client configuration: {e}")
        return False

def initialize_environment(config_file: str = "agent_config.json", 
                          frontend_path: str = None,
                          client_path: str = None,
                          with_examples: bool = False) -> Dict[str, Any]:
    """
    Initialize the entire environment.
    
    Args:
        config_file: Path to configuration file
        frontend_path: Path to frontend directory
        client_path: Path to client directory
        with_examples: Whether to add example data
        
    Returns:
        Dictionary containing initialized components
    """
    # Create default paths if not provided
    if frontend_path is None:
        frontend_path = os.path.join(os.path.expanduser("~"), "claude_agent", "frontend")
    if client_path is None:
        client_path = os.path.join(os.path.expanduser("~"), "claude_agent", "client")
    
    # Initialize configuration
    config = AgentConfig(config_file)
    
    # Initialize memory
    memory = initialize_memory(config)
    
    # Add example data if requested
    if with_examples:
        add_example_data(memory)
    
    # Connect to frontend
    if frontend_path:
        connect_frontend(memory, frontend_path)
    
    # Create client configuration
    if client_path:
        create_client_config(config, client_path)
    
    return {
        "config": config,
        "memory": memory,
        "frontend_path": frontend_path,
        "client_path": client_path
    }

if __name__ == "__main__":
    # If run directly, initialize the environment
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Claude Agent environment")
    parser.add_argument("--config", help="Path to configuration file", default="agent_config.json")
    parser.add_argument("--frontend", help="Path to frontend directory", default=None)
    parser.add_argument("--client", help="Path to client directory", default=None)
    parser.add_argument("--examples", help="Add example data", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize environment
    result = initialize_environment(
        config_file=args.config,
        frontend_path=args.frontend,
        client_path=args.client,
        with_examples=args.examples
    )
    
    print(f"Environment initialized successfully!")
    print(f"Configuration: {result['config'].config_file}")
    print(f"Memory: {result['memory'].memory_file}")
    print(f"Frontend: {result['frontend_path']}")
    print(f"Client: {result['client_path']}")
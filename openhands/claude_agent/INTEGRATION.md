# OpenDear Integration Guide

This guide explains how to integrate Claude Agent with OpenDear, allowing OpenDear to dispatch requests to Claude Agent and allocate tools appropriately.

## Overview

The integration between Claude Agent and OpenDear consists of several components:

1. **Agent Router**: Routes requests from OpenDear to Claude Agent
2. **Tool Allocator**: Allocates tools to different agents based on capabilities
3. **OpenDear Integration**: Registers Claude Agent with OpenDear
4. **Memory Initialization**: Initializes memory for Claude Agent

## Integration Components

### Agent Router

The Agent Router (`agent_router.py`) provides a routing mechanism for OpenDear to dispatch requests to Claude Agent. It handles:

- Registering Claude Agent capabilities with OpenDear
- Processing requests from OpenDear
- Executing tools
- Setting system instructions

```python
from openhands.claude_agent import get_claude_agent_router

# Get Claude Agent router
router = get_claude_agent_router(config_file="agent_config.json")

# Handle request
response = router.handle_request({
    "query": "What is the capital of France?",
    "conversation_id": "conversation-123",
    "context": {},
    "tools_allowed": True
})
```

### Tool Allocator

The Tool Allocator (`tool_allocator.py`) distributes tools among different agents based on their capabilities and priorities. It handles:

- Registering tools from different agents
- Registering agent capabilities
- Allocating tools to agents
- Executing tools

```python
from openhands.claude_agent import get_tool_allocator

# Get tool allocator
allocator = get_tool_allocator()

# Register agent capabilities
allocator.register_agent_capabilities(
    agent_name="claude_agent",
    capabilities=["text_generation", "reasoning", "tool_usage"],
    priority=80,
    preferred_tools=["read_file", "write_file"]
)

# Register tools
allocator.register_tool(
    name="read_file",
    description="Read file content",
    handler=read_file_handler,
    schema=read_file_schema,
    categories=["file_operations"]
)

# Allocate tools
allocations = allocator.allocate_tools()

# Get tools for an agent
claude_tools = allocator.get_agent_tools("claude_agent")
```

### OpenDear Integration

The OpenDear Integration (`opendear_integration.py`) provides a high-level interface for integrating Claude Agent with OpenDear. It handles:

- Initializing Claude Agent
- Registering with OpenDear
- Routing requests
- Executing tools
- Getting capabilities

```python
from openhands.claude_agent import get_opendear_integration

# Get OpenDear integration
integration = get_opendear_integration(config_file="agent_config.json")

# Route request
response = integration.route_request({
    "query": "What is the capital of France?",
    "conversation_id": "conversation-123",
    "context": {},
    "tools_allowed": True
})

# Execute tool
result = integration.execute_tool("read_file", {"file_path": "example.txt"})

# Get capabilities
capabilities = integration.get_capabilities()
```

### Memory Initialization

The Memory Initialization (`init_memory.py`) initializes memory for Claude Agent and connects it to frontends and clients. It handles:

- Initializing memory
- Adding example data
- Connecting to frontends
- Creating client configurations

```python
from openhands.claude_agent import initialize_environment

# Initialize environment
result = initialize_environment(
    config_file="agent_config.json",
    frontend_path="/path/to/frontend",
    client_path="/path/to/client",
    with_examples=True
)
```

## Tool Categories

Tools are categorized to help with allocation:

- **file_operations**: File reading, writing, and listing
- **system_operations**: System commands and information
- **network_operations**: HTTP requests
- **data_processing**: CSV processing
- **code_execution**: Python code execution
- **vector_operations**: Vector database operations
- **desktop_operations**: Screenshot, mouse, and keyboard operations
- **integration**: LangChain and enhanced reasoning integration

## Agent Capabilities

Claude Agent has the following capabilities:

- **text_generation**: Generating text responses
- **reasoning**: Reasoning about complex problems
- **tool_usage**: Using tools to accomplish tasks
- **file_operations**: Working with files
- **code_execution**: Executing code
- **web_search**: Searching the web
- **vector_search**: Searching vector databases

## Example Usage

See the example script at `/workspace/OpenHands/examples/opendear_claude_integration.py` for a complete example of integrating Claude Agent with OpenDear.

```bash
# Run with Claude API
python examples/opendear_claude_integration.py --query "What is the capital of France?"

# Run with local model
python examples/opendear_claude_integration.py --local --local-model "/path/to/model.gguf" --query "What is the capital of France?"
```

## Integration with Other Agents

To integrate Claude Agent with other agents in the OpenDear ecosystem:

1. Register all agents with OpenDear
2. Register all tools with the Tool Allocator
3. Allocate tools to agents based on capabilities
4. Use OpenDear to dispatch requests to the appropriate agent

```python
# Register multiple agents
agent_registry.register_agent(
    name="claude_agent",
    description="Claude Agent with advanced reasoning",
    handler=claude_router.handle_request,
    capabilities=["text_generation", "reasoning"],
    priority=80
)

agent_registry.register_agent(
    name="other_agent",
    description="Other agent with specialized capabilities",
    handler=other_router.handle_request,
    capabilities=["specialized_task"],
    priority=70
)

# Allocate tools
tool_allocator = get_tool_allocator()
tool_allocator.allocate_tools()

# Dispatch request
response = dispatcher.dispatch_request({
    "query": "What is the capital of France?",
    "conversation_id": "conversation-123",
    "context": {},
    "tools_allowed": True
})
```
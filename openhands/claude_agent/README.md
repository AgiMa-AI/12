# Claude Agent for OpenHands

Claude Agent is a powerful agent system that can use either Claude API or local models, with advanced reasoning capabilities, tool usage, and memory management. It integrates seamlessly with the OpenHands platform, providing a versatile AI assistant that can perform a wide range of tasks.

> **Privacy-Focused**: Can run entirely with local models for privacy-sensitive applications.

## Features

- **Model Flexibility**: Works with both Claude API and local models (Llama, Mistral, etc.)
- **Privacy Protection**: Can run entirely offline with local models for sensitive applications
- **Advanced Reasoning**: Leverages powerful reasoning capabilities of modern language models
- **Tool Integration**: Uses a wide range of tools for file operations, code execution, web searches, and more
- **Memory Management**: Maintains conversation history and long-term memory
- **Vector Database**: Stores and retrieves documents using vector embeddings
- **Desktop Control**: Captures screenshots and controls mouse and keyboard
- **Enhanced Thinking**: Supports thinking mode for more transparent reasoning
- **Command System**: Provides a flexible command system for direct interaction
- **LangChain Integration**: Integrates with OpenHands' LangChain Router for enhanced capabilities

## Installation

Claude Agent is included with OpenHands. To use it, you need to install the required dependencies:

### For Claude API (Online Mode)

```bash
pip install anthropic langchain langchain-community langchain-experimental
```

### For Local Models (Offline Mode)

Choose one of the following local model backends:

```bash
# Option 1: llama-cpp-python (recommended for best performance)
pip install llama-cpp-python

# Option 2: ctransformers (lightweight alternative)
pip install ctransformers

# Option 3: transformers with accelerate (for HuggingFace models)
pip install transformers accelerate torch
```

### Additional Features

For vector database functionality:
```bash
pip install langchain-openai faiss-cpu
```

For desktop control functionality:
```bash
pip install pyautogui pyscreenshot
```

## Usage

### Basic Usage

```python
from openhands.claude_agent import ClaudeAgent

# Create agent
agent = ClaudeAgent()

# Start agent
agent.start()
```

### Using Local Models

```python
from openhands.claude_agent import ClaudeAgent, AgentConfig

# Create config with local model settings
config = AgentConfig()
config.use_local_model = True
config.local_model_path = "/path/to/your/model.gguf"  # Path to your GGUF model
config.save_config()

# Create agent with this configuration
agent = ClaudeAgent(config_file=config.config_file)

# Start agent
agent.start()
```

### Configuration

You can configure the agent by creating a configuration file:

```json
{
  "working_directory": "/path/to/working/directory",
  "api_key": "your-anthropic-api-key",
  "model": "claude-3-7-sonnet-20250219",
  "max_tokens": 128000,
  "temperature": 0.7,
  "thinking_mode": true,
  "debug_mode": false,
  "tool_timeout": 60,
  
  "use_local_model": false,
  "local_model_path": "/path/to/your/model.gguf",
  "context_length": 4096,
  "gpu_layers": -1,
  "local_model_type": "llama"
}
```

Then load the configuration:

```python
from openhands.claude_agent import ClaudeAgent

# Create agent with configuration
agent = ClaudeAgent(config_file="path/to/config.json")

# Start agent
agent.start()
```

### Integrating with OpenHands Components

Claude Agent can integrate with other OpenHands components:

```python
from openhands.claude_agent import ClaudeAgent

# Create agent
agent = ClaudeAgent()

# Integrate with LangChain Router
agent.integrate_with_langchain_router()

# Integrate with vector tools
agent.integrate_vector_tools()

# Integrate with desktop tools
agent.integrate_desktop_tools()

# Integrate with enhanced reasoning
agent.integrate_enhanced_reasoning()

# Start agent
agent.start()
```

### Processing Input Programmatically

You can process input programmatically without starting the interactive session:

```python
from openhands.claude_agent import ClaudeAgent

# Create agent
agent = ClaudeAgent()

# Process input
response = agent.process_input("What is the capital of France?")

# Print response
print(response.content)
```

## Command System

Claude Agent provides a flexible command system for direct interaction:

- `/tools` or `@tools` - List all available tools
- `/memory` or `@memory` - View memory status
- `/clear` or `@clear` - Clear current session
- `/config` or `@config` - View or modify configuration
- `/save` or `@save` - Manually save memory
- `/system` or `@system` - Set system prompt
- `/execute` or `@execute` - Directly execute tool
- `/help` or `@help` - Show help information

Examples:

```
/tools
@tools

/execute read_file {"file_path": "example.txt"}
@execute read_file {"file_path": "example.txt"}

/system You are an AI assistant focused on data analysis
@system You are an AI assistant focused on data analysis

/config model claude-3-opus-20240229
@config model claude-3-opus-20240229
```

## Available Tools

Claude Agent comes with a wide range of tools:

### Base Tools
- `ExecutePython` - Execute Python code
- `Search` - Search for information on the internet
- `RunCommand` - Execute system command
- `ReadFile` - Read file content
- `WriteFile` - Write content to file

### Vector Database Tools
- `AddFileToVectorDB` - Add file to vector database
- `SearchVectorDB` - Search for similar content in vector database

### Desktop Control Tools
- `TakeScreenshot` - Capture screenshot
- `MouseClick` - Perform mouse click
- `KeyboardType` - Type text
- `GetMousePosition` - Get current mouse position

### Integration Tools
- `langchain_route` - Route request through LangChain Router
- `enhanced_reasoning` - Use enhanced reasoning capabilities

## Example

See the example script at `/workspace/OpenHands/examples/claude_agent_example.py` for a complete example of using Claude Agent.

You can run it with local model support:

```bash
# Run with Claude API
python examples/claude_agent_example.py --api-key "your-api-key"

# Run with local model
python examples/claude_agent_example.py --local --local-model "/path/to/model.gguf" --gpu-layers 35
```

## API Server

Claude Agent includes a FastAPI server for connecting clients and frontends:

```bash
# Start the API server with Claude API
python examples/start_claude_agent.py --api-key "your-api-key"

# Start the API server with local model
python examples/start_claude_agent.py --local --local-model "/path/to/model.gguf"
```

The API server provides the following endpoints:

- `GET /` - Root endpoint
- `POST /initialize` - Initialize the agent
- `POST /message` - Send a message to the agent
- `POST /command` - Execute a command
- `POST /tool` - Execute a tool directly
- `GET /tools` - List available tools
- `GET /config` - Get current configuration
- `POST /config` - Update configuration
- `GET /memory` - Get memory status
- `POST /memory/clear` - Clear memory
- `POST /memory/sync` - Sync memory with frontend

## Frontend and Client Integration

Claude Agent can be integrated with frontends and clients:

```python
from openhands.claude_agent.init_memory import initialize_environment

# Initialize environment
result = initialize_environment(
    config_file="agent_config.json",
    frontend_path="/path/to/frontend",
    client_path="/path/to/client",
    with_examples=True
)

# This creates:
# 1. A configuration file at agent_config.json
# 2. A memory file at result['memory'].memory_file
# 3. A frontend memory file at /path/to/frontend/frontend_memory.json
# 4. A client configuration file at /path/to/client/client_config.json
```

The frontend memory file is a JSON file that can be used by frontends to display conversation history and user preferences:

```json
{
  "conversations": [
    {
      "role": "assistant",
      "content": "Hello! I'm your AI assistant powered by Claude. How can I help you today?",
      "timestamp": "2023-01-01T00:00:00.000000"
    },
    {
      "role": "user",
      "content": "What can you do?",
      "timestamp": "2023-01-01T00:00:01.000000"
    }
  ],
  "preferences": {
    "language": "auto-detect",
    "response_style": "balanced",
    "detail_level": "medium"
  },
  "system_info": {
    "initialization_date": "2023-01-01T00:00:00.000000",
    "version": "1.0.0"
  },
  "last_updated": "2023-01-01T00:00:02.000000"
}
```

The client configuration file is a JSON file that can be used by clients to connect to the API server:

```json
{
  "api_mode": true,
  "model": "claude-3-7-sonnet-20250219",
  "max_tokens": 128000,
  "temperature": 0.7,
  "thinking_mode": true,
  "server_url": "http://localhost:8000",
  "memory_path": "/path/to/memory.pkl",
  "working_directory": "/path/to/working/directory",
  "last_updated": "2023-01-01T00:00:00.000000"
}
```

## Advanced Features

### Memory System

Claude Agent maintains conversation history and long-term memory:

```python
# Add to long-term memory
agent.memory.add_to_long_term_memory("user_preference", "prefers detailed explanations")

# Get from long-term memory
preference = agent.memory.get_from_long_term_memory("user_preference")

# Search conversation history
results = agent.memory.search_conversation_history("python", limit=5)
```

### Tool Registry

You can register custom tools:

```python
# Register custom tool
agent.tool_registry.register_tool(
    name="custom_tool",
    description="Custom tool description",
    func=custom_function,
    schema={
        "type": "object",
        "required": ["param1"],
        "properties": {
            "param1": {
                "type": "string",
                "description": "Parameter description"
            }
        }
    }
)
```

### Direct Tool Execution

You can execute tools directly:

```python
# Execute tool
result = agent.tool_registry.execute_tool(
    "read_file",
    {"file_path": "example.txt"},
    timeout=60
)
```

### System Prompt

You can set a custom system prompt:

```python
# Set system prompt
agent._set_system_prompt("You are an AI assistant specialized in data analysis.")
```

## Integration with LangChain Router

Claude Agent integrates with OpenHands' LangChain Router:

```python
# Route request through LangChain Router
result = agent.tool_registry.execute_tool(
    "langchain_route",
    {
        "query": "What is the theory of relativity?",
        "model_name": "gpt-4",
        "task_type": "reasoning"
    }
)
```

## Integration with Enhanced Reasoning

Claude Agent integrates with OpenHands' Enhanced Reasoning:

```python
# Use enhanced reasoning
result = agent.tool_registry.execute_tool(
    "enhanced_reasoning",
    {
        "query": "What is the theory of relativity?",
        "reasoning_type": "tree_of_thoughts"
    }
)
```
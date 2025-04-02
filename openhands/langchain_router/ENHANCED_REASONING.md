# Enhanced Reasoning for OpenHands

The Enhanced Reasoning module provides advanced reasoning capabilities for OpenHands, including structured thinking, multi-step reasoning, and thought tree generation. This module is designed to maximize the thinking and reasoning capabilities of language models, enabling them to solve complex problems more effectively.

## Features

### 1. Reasoning Chains

- **Chain of Thought (CoT)**: Step-by-step reasoning process that breaks down complex problems into manageable steps
- **ReAct (Reasoning-Acting-Observing)**: Combines reasoning with action execution and observation
- **Tree of Thoughts (ToT)**: Explores multiple reasoning paths and selects the best one

### 2. Cognitive Core

- **Structured Thinking**: Organizes thoughts in a structured format
- **Problem Decomposition**: Breaks down problems into core components
- **Premise Identification**: Identifies implicit assumptions and known conditions
- **Solution Construction**: Organizes solution steps in logical order
- **Reflective Evaluation**: Evaluates the strengths and weaknesses of solutions

### 3. Reflection System

- **Interaction Analysis**: Records and analyzes interactions
- **Pattern Identification**: Identifies patterns and preferences
- **Response Improvement**: Improves future responses based on past interactions
- **User Identity Management**: Manages user identities and preferences

## Usage

### Basic Usage

```python
from openhands import get_langchain_router
from openhands.langchain_router.enhanced_reasoning import (
    EnhancedReasoningFactory,
    ReasoningChain,
    CognitiveCore,
    ReflectionSystem
)

# Get the LangChain Router integration
integration = get_langchain_router()

# Get the language model
llm = integration.router.chain.llm

# Create enhanced reasoning components
reasoning_chain = EnhancedReasoningFactory.create_reasoning_chain(llm)
cognitive_core = EnhancedReasoningFactory.create_cognitive_core(llm)
reflection_system = EnhancedReasoningFactory.create_reflection_system(llm)

# Use chain of thought reasoning
thought_process = reasoning_chain.chain_of_thought("What is the theory of relativity?")
print(thought_process)

# Use cognitive core
understanding = cognitive_core.think("What is the theory of relativity?")
print(understanding["thought_process"])
print(understanding["conclusion"])

# Record interaction and reflect
interaction = {
    "query": "What is the theory of relativity?",
    "response": "The theory of relativity has two parts: special relativity and general relativity...",
    "context": {},
    "user_id": "user123"
}
reflection = reflection_system.reflect_on_interaction(interaction)
print(reflection["reflections"])
```

### Using Different Reasoning Types

#### Chain of Thought (CoT)

```python
# Use chain of thought reasoning
thought_process = reasoning_chain.chain_of_thought("What is the theory of relativity?")
```

#### ReAct (Reasoning-Acting-Observing)

```python
# Define tools
tools = [
    Tool(name="search", func=search_function, description="Search for information"),
    Tool(name="calculator", func=calculator_function, description="Perform calculations")
]

# Use ReAct reasoning
result = reasoning_chain.react("What is 15% of 175?", tools=tools)
```

#### Tree of Thoughts (ToT)

```python
# Use Tree of Thoughts reasoning
result = reasoning_chain.tree_of_thoughts(
    "What is the best approach to solve the traveling salesman problem?",
    num_branches=3,
    max_depth=3
)
```

### Using the Cognitive Core

```python
# Use cognitive core for deep thinking
understanding = cognitive_core.think("What are the ethical implications of AI?")
print(understanding["thought_process"])  # Detailed thought process
print(understanding["conclusion"])       # Concise conclusion
```

### Using the Reflection System

```python
# Register a user
reflection_system.add_user_identity(
    user_id="user123",
    preferences={
        "detail_level": "high",
        "preferred_topics": ["science", "technology"]
    }
)

# Get user identity
user_info = reflection_system.get_user_identity("user123")

# Record interaction and reflect
interaction = {
    "query": "What is quantum computing?",
    "response": "Quantum computing is a type of computing that uses quantum phenomena...",
    "context": {},
    "user_id": "user123"
}
reflection = reflection_system.reflect_on_interaction(interaction)
```

## FastAPI Application

The Enhanced Reasoning module includes a FastAPI application that demonstrates the enhanced reasoning capabilities:

```python
from openhands.langchain_router.api import start_server

# Start the server
start_server(host="0.0.0.0", port=8000)
```

### API Endpoints

- **POST /query**: Process a query with enhanced reasoning
- **POST /register_user**: Register or update a user
- **GET /models**: List available models
- **GET /optimization_metrics**: Get optimization metrics
- **GET /optimization_recommendations**: Get optimization recommendations
- **GET /**: Root endpoint with welcome message

### Query Request Example

```json
{
  "query": "What is the theory of relativity?",
  "user_id": "user123",
  "context": {},
  "reasoning_type": "chain_of_thought",
  "task_type": "reasoning",
  "use_optimized_router": true,
  "model_name": null
}
```

## Example Script

See the example script at `/workspace/OpenHands/examples/enhanced_reasoning_example.py` for a complete example of using the Enhanced Reasoning module.

## Integration with Optimized Router

The Enhanced Reasoning module integrates seamlessly with the Optimized Router:

```python
from openhands import get_langchain_router
from openhands.langchain_router.enhanced_reasoning import EnhancedReasoningFactory

# Get the LangChain Router integration with optimized router enabled
integration = get_langchain_router()

# Create enhanced reasoning components
reasoning_chain = EnhancedReasoningFactory.create_reasoning_chain(integration.router.chain.llm)

# Use enhanced reasoning with optimized router
thought_process = reasoning_chain.chain_of_thought("What is the theory of relativity?")
result = integration.route_request(
    user_input="What is the theory of relativity?",
    conversation_id="conversation-123",
    task_type="reasoning",
    use_optimized_router=True
)
```

## Advanced Features

### Thought Step Tracking

The reasoning chain tracks each step in the reasoning process:

```python
# Use chain of thought reasoning
reasoning_chain.chain_of_thought("What is the theory of relativity?")

# Get the thought trace
thought_trace = reasoning_chain._format_thought_trace()
print(thought_trace)
```

### Custom Reasoning Chains

You can create custom reasoning chains by extending the `ReasoningChain` class:

```python
from openhands.langchain_router.enhanced_reasoning import ReasoningChain

class CustomReasoningChain(ReasoningChain):
    def custom_reasoning(self, question: str) -> str:
        # Implement custom reasoning logic
        return "Custom reasoning result"
```

### Custom Cognitive Core

You can create a custom cognitive core by extending the `CognitiveCore` class:

```python
from openhands.langchain_router.enhanced_reasoning import CognitiveCore

class CustomCognitiveCore(CognitiveCore):
    def custom_thinking(self, query: str) -> Dict[str, Any]:
        # Implement custom thinking logic
        return {
            "query": query,
            "thought_process": "Custom thinking process",
            "conclusion": "Custom conclusion"
        }
```
# Optimized Router for OpenHands

The Optimized Router is an enhanced version of the LangChain Router that applies various optimization techniques to maximize model performance and precision, optimize resource usage, enhance chain-of-thought capabilities, extend context processing to the limit, and integrate knowledge base functionality.

## Features

### 1. Performance Optimization

- **Parameter Optimization**: Automatically adjusts temperature, top-k, repeat penalty, and other parameters for optimal performance
- **Mirostat Adaptive Sampling**: Uses Mirostat algorithm for adaptive sampling to maintain consistent output quality
- **Mixed-Precision Matrix Multiplication**: Enables mixed-precision matrix multiplication for faster inference
- **RoPE Scaling**: Applies rotary position embedding scaling for better handling of long contexts

### 2. Resource Optimization

- **Dynamic Resource Allocation**: Monitors CPU, memory, and GPU usage and adjusts parameters accordingly
- **Batch Size Optimization**: Dynamically adjusts batch size based on available resources
- **Cache Management**: Optimizes cache usage for better performance
- **Thread Management**: Adjusts number of threads based on CPU usage

### 3. Chain-of-Thought Enhancement

- **Multi-Step Reasoning**: Enhances prompts with multi-step reasoning instructions
- **Self-Reflection**: Adds verification steps to responses
- **Structured Thinking**: Organizes thoughts in a structured format
- **Depth Control**: Adjusts the depth of reasoning based on task complexity

### 4. Context Window Optimization

- **Smart Truncation**: Intelligently truncates context to preserve the most relevant information
- **Priority-Based Selection**: Prioritizes sections with questions, code blocks, or specific keywords
- **Fallback Mechanisms**: Provides fallback mechanisms when context exceeds limits
- **Context Compression**: Compresses context to fit within limits while preserving meaning

### 5. Knowledge Base Integration

- **Relevant Document Retrieval**: Retrieves relevant documents from a knowledge base
- **Relevance Threshold**: Filters documents based on relevance threshold
- **Document Limit**: Limits the number of documents to include
- **Context Integration**: Seamlessly integrates knowledge base information into prompts

## Usage

### Basic Usage

```python
from openhands import get_langchain_router

# Get the LangChain Router integration with optimized router enabled
router = get_langchain_router()

# Route a request with optimization
result = router.route_request(
    user_input="Explain the theory of relativity",
    conversation_id="conversation-123",
    task_type="reasoning",  # Helps the optimizer choose the right strategy
    use_optimized_router=True  # Explicitly enable optimization
)

# Get the response
response = result.get("response")
model = result.get("model")
metadata = result.get("metadata")

print(f"Response from {model}: {response}")
print(f"Optimized: {metadata.get('optimized', False)}")
```

### Choosing Optimization Profiles

The Optimized Router comes with several pre-defined optimization profiles:

```python
from openhands.langchain_router.optimized_router import OptimizedRouterFactory

# Create a high-performance router
high_performance_router = OptimizedRouterFactory.create_high_performance()

# Create a balanced router
balanced_router = OptimizedRouterFactory.create_balanced()

# Create a resource-efficient router
efficient_router = OptimizedRouterFactory.create_resource_efficient()
```

### Custom Optimization Configuration

You can create a custom optimization configuration:

```python
from openhands.langchain_router.optimizers import ModelOptimizationConfig
from openhands.langchain_router.optimized_router import OptimizedRouter

# Create a custom optimization configuration
config = ModelOptimizationConfig(
    temperature=0.3,
    top_k=30,
    repeat_penalty=1.15,
    max_context_length=16384,
    max_batch_size=1024,
    num_threads=4,
    num_gpu_layers=33,
    use_parallel=True,
    use_mlock=True,
    cache_capacity_mb=2000,
    use_mmq=True,
    rope_scaling="linear",
    enable_cot=True,
    cot_depth=3,
    enable_kb=True,
    kb_relevance_threshold=0.75,
    kb_max_documents=5,
    system_prompt="You are an assistant that combines Claude 3.7's reasoning abilities with Claude 3.5's wisdom..."
)

# Create an optimized router with the custom configuration
router = OptimizedRouter(optimization_config=config)
```

### Creating from Bash Script

You can create an optimized router from a bash script similar to the one provided in the example:

```python
from openhands.langchain_router.optimized_router import OptimizedRouterFactory

# Create an optimized router from a bash script
router = OptimizedRouterFactory.create_from_bash_script("/path/to/script.sh")
```

### Monitoring Optimization Metrics

You can monitor optimization metrics:

```python
# Get optimization metrics
metrics = router.get_optimization_metrics()
print(f"Total requests: {metrics.get('total_requests')}")
print(f"Average response time: {metrics.get('average_response_time')} seconds")

# Get optimization recommendations
recommendations = router.get_optimization_recommendations()
for key, value in recommendations.items():
    print(f"{key}: {value}")
```

## Example Script

See the example script at `/workspace/OpenHands/examples/langchain_router_example.py` for a complete example of using the Optimized Router.

## Advanced Features

### Chain-of-Thought Enhancement

The chain-of-thought enhancer improves the reasoning capabilities of language models:

```python
from openhands.langchain_router.optimizers import ChainOfThoughtEnhancer

# Create a chain-of-thought enhancer
cot_enhancer = ChainOfThoughtEnhancer(depth=3)

# Enhance a prompt
enhanced_prompt = cot_enhancer.enhance_prompt(
    prompt="Explain the theory of relativity",
    task_type="reasoning"
)

# Enhance a response
enhanced_response = cot_enhancer.enhance_response(
    response="The theory of relativity has two parts: special relativity and general relativity...",
    task_type="reasoning"
)
```

### Knowledge Base Integration

The knowledge base integrator provides methods for integrating knowledge base information into prompts and responses:

```python
from openhands.langchain_router.optimizers import KnowledgeBaseIntegrator

# Create a knowledge base integrator
kb_integrator = KnowledgeBaseIntegrator(
    kb_path="/.openhands-state/knowledge_base",
    relevance_threshold=0.75,
    max_documents=5
)

# Retrieve relevant documents
documents = kb_integrator.retrieve_relevant_documents(
    query="What is the theory of relativity?"
)

# Integrate knowledge base information into a prompt
enhanced_prompt = kb_integrator.integrate_into_prompt(
    prompt="Explain the theory of relativity",
    query="What is the theory of relativity?"
)

# Add a document to the knowledge base
doc_id = kb_integrator.add_document(
    content="The theory of relativity was developed by Albert Einstein...",
    metadata={"source": "physics_textbook", "year": 2023}
)
```

## Configuration Reference

### ModelOptimizationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| temperature | float | 0.3 | Controls randomness in generation |
| top_k | int | 30 | Limits vocabulary to top K options |
| top_p | float | 0.95 | Nucleus sampling threshold |
| repeat_penalty | float | 1.15 | Penalty for repeating tokens |
| presence_penalty | float | 0.0 | Penalty for token presence |
| frequency_penalty | float | 0.0 | Penalty for token frequency |
| use_mirostat | bool | True | Whether to use Mirostat adaptive sampling |
| mirostat_mode | int | 2 | Mirostat algorithm version (1 or 2) |
| mirostat_tau | float | 3.0 | Mirostat target entropy |
| mirostat_eta | float | 0.05 | Mirostat learning rate |
| max_context_length | int | 16384 | Maximum context length |
| context_window_fallback | bool | True | Whether to use fallback for context window |
| max_batch_size | int | 1024 | Maximum batch size |
| num_threads | int | 4 | Number of CPU threads |
| num_gpu_layers | int | 33 | Number of GPU layers |
| use_parallel | bool | True | Whether to use parallel processing |
| use_mlock | bool | True | Whether to lock memory |
| cache_capacity_mb | int | 2000 | Cache capacity in MB |
| use_mmq | bool | True | Whether to use mixed-precision matrix multiplication |
| rope_scaling | str | "linear" | RoPE scaling method |
| enable_cot | bool | True | Whether to enable chain-of-thought |
| cot_depth | int | 3 | Depth of chain-of-thought reasoning |
| enable_kb | bool | True | Whether to enable knowledge base integration |
| kb_relevance_threshold | float | 0.75 | Threshold for document relevance |
| kb_max_documents | int | 5 | Maximum number of documents to retrieve |
| system_prompt | str | "..." | System prompt for the model |
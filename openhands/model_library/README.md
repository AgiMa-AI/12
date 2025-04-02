# Model Library for OpenHands

The Model Library is a component of OpenHands that organizes and manages different types of models, including literature, knowledge, programming, language, image, and health models.

## Features

- **Automatic Model Categorization**: Models are automatically categorized based on filenames and metadata
- **Model Organization**: Models are organized by type in separate directories
- **Model Scanning**: Scan directories for models and automatically add them to the library
- **Model Loading**: Load and unload models as needed
- **Model Metadata**: Store and retrieve model metadata
- **Thread Safety**: Thread-safe operations for concurrent access

## Model Types

The Model Library supports the following model types:

- **Literature**: Models for literary tasks, such as story generation, poetry, etc.
- **Knowledge**: Models for knowledge-based tasks, such as fact retrieval, question answering, etc.
- **Programming**: Models for programming tasks, such as code generation, code completion, etc.
- **Language**: Models for language tasks, such as translation, summarization, etc.
- **Image**: Models for image tasks, such as image generation, image recognition, etc.
- **Health**: Models for health-related tasks, such as medical diagnosis, health advice, etc.
- **General**: General-purpose models that can handle multiple types of tasks
- **Other**: Models that don't fit into any of the above categories

## Usage

### Basic Usage

```python
from openhands.model_library import get_model_library, ModelType

# Get model library
library = get_model_library()

# Add a model
model = library.add_model("/path/to/model.gguf")

# Get models by type
literature_models = library.get_models_by_type(ModelType.LITERATURE)
health_models = library.get_models_by_type(ModelType.HEALTH)

# Get all models
all_models = library.get_all_models()

# Get model by ID
model = library.get_model("model_id")

# Load model
library.load_model("model_id")

# Check if model is loaded
is_loaded = library.is_model_loaded("model_id")

# Unload model
library.unload_model("model_id")

# Remove model
library.remove_model("model_id")
```

### Scanning Directories

```python
from openhands.model_library import get_model_library

# Get model library
library = get_model_library()

# Scan directory for models
models = library.scan_directory("/path/to/models", recursive=True)
```

### Model Information

```python
from openhands.model_library import get_model_library

# Get model library
library = get_model_library()

# Get model information
model_info = library.get_model_info("model_id")

# Print model information
print(f"ID: {model_info.id}")
print(f"Name: {model_info.name}")
print(f"Type: {model_info.model_type.value}")
print(f"Path: {model_info.path}")
print(f"Size: {model_info.size} bytes")
print(f"Format: {model_info.format}")
print(f"Description: {model_info.description}")
print(f"Version: {model_info.version}")
print(f"Created: {model_info.created_at}")
```

### Model Metadata

Models can have metadata associated with them. Metadata can be provided when adding a model:

```python
from openhands.model_library import get_model_library, ModelType

# Get model library
library = get_model_library()

# Add a model with metadata
model = library.add_model(
    "/path/to/model.gguf",
    model_type=ModelType.HEALTH,
    metadata={
        "author": "OpenHands Team",
        "description": "Health advice model",
        "version": "1.0.0",
        "parameters": 7000000000,
        "context_length": 4096,
        "license": "MIT"
    }
)
```

Metadata can also be provided in a JSON file with the same name as the model file but with a `.json` extension:

```json
{
  "name": "Health Advisor",
  "type": "health",
  "description": "Model for providing health advice",
  "version": "1.0.0",
  "author": "OpenHands Team",
  "parameters": 7000000000,
  "context_length": 4096,
  "license": "MIT",
  "categories": ["health", "medical", "advice"]
}
```

## Command Line Interface

The Model Library can be used from the command line:

```bash
# List all models
python examples/model_library_example.py --list

# List models of a specific type
python examples/model_library_example.py --list-type health

# Scan a directory for models
python examples/model_library_example.py --scan /path/to/models

# Add a model
python examples/model_library_example.py --add /path/to/model.gguf --type health
```

## Library Structure

The Model Library creates the following directory structure:

```
~/openhands/models/
├── index.json
├── literature/
├── knowledge/
├── programming/
├── language/
├── image/
├── health/
├── general/
└── other/
```

Each model is stored in the appropriate category directory, and the `index.json` file contains metadata for all models in the library.

## Integration with OpenHands

The Model Library integrates with other OpenHands components:

```python
from openhands.model_library import get_model_library
from openhands.claude_agent import get_openhands_integration

# Get model library
library = get_model_library()

# Get OpenHands integration
integration = get_openhands_integration()

# Load a health model
health_models = library.get_models_by_type(ModelType.HEALTH)
if health_models:
    model = health_models[0]
    library.load_model(model.info.id)
    
    # Use the model with Claude Agent
    # ...
```
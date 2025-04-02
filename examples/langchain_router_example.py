#!/usr/bin/env python3
"""
Example script demonstrating the use of the LangChain Router in OpenHands.

This script shows how to use the LangChain Router to route requests to
different models based on the content of the request.
"""

import os
import sys
import logging
import uuid

# Add the parent directory to the path so we can import openhands
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openhands import get_langchain_router
from openhands.langchain_router.config import RouterConfig, ModelConfig, ModelCapability, RouterRule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the example script."""
    print("LangChain Router Example")
    print("=======================")
    print()
    
    try:
        # Get the LangChain Router integration
        integration = get_langchain_router()
        
        # Start a chat session
        chat_session(integration)
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure you have installed the required dependencies.")
        sys.exit(1)


def chat_session(integration):
    """
    Start a chat session using the LangChain Router.
    
    Args:
        integration: The OpenHands-LangChain integration
    """
    print("Starting chat session. Type 'exit' to quit.")
    print("Type 'model:<name>' to switch to a specific model.")
    print("Type 'models' to list available models.")
    print("Type 'babyagi:<task>' to use BabyAGI for a specific task.")
    print("Type 'optimize:on' or 'optimize:off' to toggle optimization.")
    print("Type 'metrics' to view optimization metrics.")
    print("Type 'recommendations' to view optimization recommendations.")
    print("Type 'task:<type>' to specify task type (reasoning, creative, factual, etc.).")
    print()
    
    # Create a conversation ID
    conversation_id = str(uuid.uuid4())
    
    # Initialize conversation history
    conversation_history = []
    
    # Initialize current model and settings
    current_model = None
    use_optimized_router = True
    current_task_type = None
    
    while True:
        # Get user input
        user_input = input("> ")
        
        # Check for exit command
        if user_input.lower() == "exit":
            break
        
        # Check for models command
        if user_input.lower() == "models":
            list_models(integration)
            continue
        
        # Check for model selection command
        if user_input.lower().startswith("model:"):
            current_model = user_input[6:].strip()
            print(f"Switched to model: {current_model}")
            continue
        
        # Check for optimization toggle command
        if user_input.lower() == "optimize:on":
            use_optimized_router = True
            print("Optimization enabled")
            continue
        elif user_input.lower() == "optimize:off":
            use_optimized_router = False
            print("Optimization disabled")
            continue
        
        # Check for metrics command
        if user_input.lower() == "metrics":
            metrics = integration.get_optimization_metrics()
            print("Optimization Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            continue
        
        # Check for recommendations command
        if user_input.lower() == "recommendations":
            recommendations = integration.get_optimization_recommendations()
            print("Optimization Recommendations:")
            for key, value in recommendations.items():
                print(f"  {key}: {value}")
            continue
        
        # Check for task type command
        if user_input.lower().startswith("task:"):
            current_task_type = user_input[5:].strip()
            print(f"Set task type to: {current_task_type}")
            continue
        
        # Check for BabyAGI command
        if user_input.lower().startswith("babyagi:"):
            task = user_input[8:].strip()
            user_input = f"organize: {task}"
        
        # Add user input to conversation history
        conversation_history.append(f"User: {user_input}")
        
        # Route the request
        result = integration.route_request(
            user_input=user_input,
            conversation_id=conversation_id,
            conversation_history="\n".join(conversation_history),
            model_name=current_model,
            task_type=current_task_type,
            use_optimized_router=use_optimized_router
        )
        
        # Extract the response and metadata
        response = result.get("response", "No response")
        model = result.get("model", "unknown")
        metadata = result.get("metadata", {})
        
        # Add response to conversation history
        conversation_history.append(f"Assistant ({model}): {response}")
        
        # Print the response
        print(f"Assistant ({model}): {response}")
        
        # Print optimization info if available
        if "optimized" in metadata and metadata["optimized"]:
            print(f"\n[Optimized: Yes | Task Type: {metadata.get('task_type', 'unknown')}]")
        elif use_optimized_router:
            print(f"\n[Optimization attempted | Task Type: {current_task_type or 'unspecified'}]")
        
        print()


def list_models(integration):
    """
    List available models.
    
    Args:
        integration: The OpenHands-LangChain integration
    """
    models = integration.get_available_models()
    
    print("Available Models:")
    for model in models:
        print(f"- {model['name']}: {model['description']}")
        print("  Capabilities:")
        for capability in model['capabilities']:
            print(f"    - {capability['name']}: {capability['score']}")
        print()


if __name__ == "__main__":
    main()
"""
Command-line interface for the LangChain Router.

This module provides a command-line interface for testing and demonstrating
the LangChain Router.
"""

import argparse
import logging
import sys
import uuid
import json
from typing import Dict, List, Optional, Any

from openhands.langchain_router.integration import OpenHandsLangChainIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="LangChain Router CLI")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start a chat session")
    chat_parser.add_argument("--model", help="Model to use (for manual selection)")
    chat_parser.add_argument("--config", help="Path to router configuration file")
    
    # List models command
    list_models_parser = subparsers.add_parser("list-models", help="List available models")
    list_models_parser.add_argument("--config", help="Path to router configuration file")
    
    # Add model command
    add_model_parser = subparsers.add_parser("add-model", help="Add a new model")
    add_model_parser.add_argument("--config", help="Path to router configuration file")
    add_model_parser.add_argument("--name", required=True, help="Name of the model")
    add_model_parser.add_argument("--description", required=True, help="Description of the model")
    add_model_parser.add_argument("--capabilities", required=True, help="Capabilities of the model (JSON)")
    add_model_parser.add_argument("--context-window", type=int, default=8000, help="Context window size")
    add_model_parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens")
    
    # Add rule command
    add_rule_parser = subparsers.add_parser("add-rule", help="Add a new routing rule")
    add_rule_parser.add_argument("--config", help="Path to router configuration file")
    add_rule_parser.add_argument("--name", required=True, help="Name of the rule")
    add_rule_parser.add_argument("--description", required=True, help="Description of the rule")
    add_rule_parser.add_argument("--priority", type=int, default=50, help="Priority of the rule")
    add_rule_parser.add_argument("--condition", required=True, help="Condition for the rule")
    add_rule_parser.add_argument("--target-model", required=True, help="Target model for the rule")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the integration
    integration = OpenHandsLangChainIntegration(config_path=args.config if hasattr(args, "config") else None)
    
    # Execute the command
    if args.command == "chat":
        chat(integration, args.model if hasattr(args, "model") else None)
    elif args.command == "list-models":
        list_models(integration)
    elif args.command == "add-model":
        add_model(integration, args)
    elif args.command == "add-rule":
        add_rule(integration, args)
    else:
        parser.print_help()


def chat(integration: OpenHandsLangChainIntegration, model_name: Optional[str] = None):
    """
    Start a chat session.
    
    Args:
        integration: The OpenHands-LangChain integration
        model_name: Optional name of the model to use (for manual selection)
    """
    print("Starting chat session. Type 'exit' to quit.")
    print("Type 'model:<name>' to switch to a specific model.")
    print()
    
    conversation_id = str(uuid.uuid4())
    conversation_history = []
    current_model = model_name
    
    while True:
        # Get user input
        user_input = input("> ")
        
        # Check for exit command
        if user_input.lower() == "exit":
            break
        
        # Check for model selection command
        if user_input.lower().startswith("model:"):
            current_model = user_input[6:].strip()
            print(f"Switched to model: {current_model}")
            continue
        
        # Add user input to conversation history
        conversation_history.append(f"User: {user_input}")
        
        # Route the request
        result = integration.route_request(
            user_input=user_input,
            conversation_id=conversation_id,
            conversation_history="\n".join(conversation_history),
            model_name=current_model
        )
        
        # Extract the response
        response = result.get("response", "No response")
        model = result.get("model", "unknown")
        
        # Add response to conversation history
        conversation_history.append(f"Assistant ({model}): {response}")
        
        # Print the response
        print(f"Assistant ({model}): {response}")
        print()


def list_models(integration: OpenHandsLangChainIntegration):
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


def add_model(integration: OpenHandsLangChainIntegration, args):
    """
    Add a new model.
    
    Args:
        integration: The OpenHands-LangChain integration
        args: Command-line arguments
    """
    # Parse capabilities
    try:
        capabilities = json.loads(args.capabilities)
    except json.JSONDecodeError:
        print("Error: Capabilities must be valid JSON")
        return
    
    # Create model configuration
    model_config = {
        "name": args.name,
        "description": args.description,
        "capabilities": capabilities,
        "context_window": args.context_window,
        "max_tokens": args.max_tokens
    }
    
    # Add the model
    integration.add_model(model_config)
    
    print(f"Added model: {args.name}")


def add_rule(integration: OpenHandsLangChainIntegration, args):
    """
    Add a new routing rule.
    
    Args:
        integration: The OpenHands-LangChain integration
        args: Command-line arguments
    """
    # Create rule configuration
    rule_config = {
        "name": args.name,
        "description": args.description,
        "priority": args.priority,
        "condition": args.condition,
        "target_model": args.target_model
    }
    
    # Add the rule
    integration.add_rule(rule_config)
    
    print(f"Added rule: {args.name}")


if __name__ == "__main__":
    main()
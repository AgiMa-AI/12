#!/usr/bin/env python3
"""
Example script demonstrating the use of enhanced reasoning capabilities in OpenHands.

This script shows how to use the enhanced reasoning capabilities to solve complex
problems with structured thinking, multi-step reasoning, and thought tree generation.
"""

import os
import sys
import logging
import json
import time
import uuid
from typing import Dict, List, Any, Optional

# Add the parent directory to the path so we can import openhands
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openhands import get_langchain_router
from openhands.langchain_router.enhanced_reasoning import (
    EnhancedReasoningFactory,
    ReasoningChain,
    CognitiveCore,
    ReflectionSystem
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the example script."""
    print("Enhanced Reasoning Example")
    print("=========================")
    print()
    
    try:
        # Get the LangChain Router integration
        integration = get_langchain_router()
        
        # Get the language model
        llm = get_llm_from_integration(integration)
        
        # Create enhanced reasoning components
        reasoning_chain = EnhancedReasoningFactory.create_reasoning_chain(llm)
        cognitive_core = EnhancedReasoningFactory.create_cognitive_core(llm)
        reflection_system = EnhancedReasoningFactory.create_reflection_system(llm)
        
        # Start interactive session
        interactive_session(
            integration,
            reasoning_chain,
            cognitive_core,
            reflection_system
        )
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure you have installed the required dependencies.")
        sys.exit(1)


def get_llm_from_integration(integration):
    """
    Get the language model from the integration.
    
    Args:
        integration: The OpenHands-LangChain integration
        
    Returns:
        The language model
    """
    # Try to get the language model from the router
    router = integration.router
    if router and hasattr(router, "chain") and hasattr(router.chain, "llm"):
        return router.chain.llm
    
    # If that fails, try to get it from the optimized router
    if hasattr(integration, "optimized_router") and integration.optimized_router:
        if hasattr(integration.optimized_router, "model_optimizer"):
            return integration.optimized_router.model_optimizer.llm
    
    # If all else fails, use a placeholder
    from langchain.llms.fake import FakeListLLM
    return FakeListLLM(responses=["Enhanced reasoning response"])


def interactive_session(
    integration,
    reasoning_chain,
    cognitive_core,
    reflection_system
):
    """
    Start an interactive session.
    
    Args:
        integration: The OpenHands-LangChain integration
        reasoning_chain: The reasoning chain
        cognitive_core: The cognitive core
        reflection_system: The reflection system
    """
    print("Starting interactive session. Type 'exit' to quit.")
    print("Type 'reasoning:<type>' to select reasoning type (chain_of_thought, react, tree_of_thoughts).")
    print("Type 'task:<type>' to specify task type (reasoning, creative, factual, etc.).")
    print("Type 'optimize:on' or 'optimize:off' to toggle optimization.")
    print("Type 'reflect' to see reflection on the last interaction.")
    print()
    
    # Create a user ID
    user_id = str(uuid.uuid4())
    
    # Initialize settings
    reasoning_type = "chain_of_thought"
    task_type = "reasoning"
    use_optimized_router = True
    last_interaction = None
    
    while True:
        # Get user input
        user_input = input(f"[{reasoning_type}] > ")
        
        # Check for exit command
        if user_input.lower() == "exit":
            break
        
        # Check for reasoning type command
        if user_input.lower().startswith("reasoning:"):
            new_type = user_input[10:].strip()
            if new_type in ["chain_of_thought", "react", "tree_of_thoughts"]:
                reasoning_type = new_type
                print(f"Switched to reasoning type: {reasoning_type}")
            else:
                print(f"Unknown reasoning type: {new_type}")
                print("Available types: chain_of_thought, react, tree_of_thoughts")
            continue
        
        # Check for task type command
        if user_input.lower().startswith("task:"):
            task_type = user_input[5:].strip()
            print(f"Set task type to: {task_type}")
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
        
        # Check for reflect command
        if user_input.lower() == "reflect":
            if last_interaction:
                reflection = reflection_system.reflect_on_interaction(last_interaction)
                print("\nReflection:")
                for key, value in reflection.get("reflections", {}).items():
                    print(f"  {key.capitalize()}: {value}")
            else:
                print("No previous interaction to reflect on.")
            continue
        
        # Process the query
        start_time = time.time()
        
        try:
            # Use enhanced reasoning based on the selected type
            if reasoning_type == "chain_of_thought":
                # Use chain of thought reasoning
                print("\nThinking step by step...")
                thought_process = reasoning_chain.chain_of_thought(user_input)
                
                # Route the request with the enhanced thought process
                result = integration.route_request(
                    user_input=user_input,
                    conversation_id=user_id,
                    task_type=task_type,
                    use_optimized_router=use_optimized_router
                )
                
                # Print the thought process and response
                print("\nThought Process:")
                print(thought_process)
                print("\nResponse:")
                print(result.get("response", "No response"))
                
            elif reasoning_type == "react":
                # Use ReAct reasoning (placeholder - would need tools)
                print("\nUsing ReAct reasoning (reasoning and acting)...")
                print("Note: This is a placeholder. Real implementation would use tools.")
                
                # Route the request
                result = integration.route_request(
                    user_input=user_input,
                    conversation_id=user_id,
                    task_type=task_type,
                    use_optimized_router=use_optimized_router
                )
                
                # Print the response
                print("\nResponse:")
                print(result.get("response", "No response"))
                
            elif reasoning_type == "tree_of_thoughts":
                # Use Tree of Thoughts reasoning
                print("\nExploring multiple reasoning paths...")
                thought_process = reasoning_chain.tree_of_thoughts(
                    user_input,
                    num_branches=3,
                    max_depth=3
                )
                
                # Route the request
                result = integration.route_request(
                    user_input=user_input,
                    conversation_id=user_id,
                    task_type=task_type,
                    use_optimized_router=use_optimized_router
                )
                
                # Print the thought process and response
                print("\nThought Process:")
                print(thought_process)
                print("\nResponse:")
                print(result.get("response", "No response"))
                
            else:
                # Use cognitive core
                print("\nThinking deeply...")
                understanding = cognitive_core.think(user_input)
                
                # Route the request
                result = integration.route_request(
                    user_input=user_input,
                    conversation_id=user_id,
                    task_type=task_type,
                    use_optimized_router=use_optimized_router
                )
                
                # Print the thought process and response
                print("\nThought Process:")
                print(understanding["thought_process"])
                print("\nConclusion:")
                print(understanding["conclusion"])
                print("\nResponse:")
                print(result.get("response", "No response"))
            
            # Record processing time
            processing_time = time.time() - start_time
            print(f"\nProcessing time: {processing_time:.2f} seconds")
            
            # Record interaction for reflection
            last_interaction = {
                "query": user_input,
                "response": result.get("response", ""),
                "context": {},
                "user_id": user_id,
                "reasoning_type": reasoning_type,
                "task_type": task_type,
                "processing_time": processing_time
            }
            
            print("\nType 'reflect' to see reflection on this interaction.")
            
        except Exception as e:
            print(f"\nError: {e}")
        
        print()


if __name__ == "__main__":
    main()
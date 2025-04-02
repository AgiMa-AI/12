"""
FastAPI application for the LangChain Router with enhanced reasoning.

This module provides a FastAPI application that demonstrates the enhanced
reasoning capabilities of the LangChain Router.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import os
import json
import logging
import time
import uuid

from openhands.langchain_router.integration import OpenHandsLangChainIntegration
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

# Initialize the application
app = FastAPI(
    title="Enhanced Reasoning AI Agent",
    description="AI agent system with advanced reasoning capabilities"
)

# Initialize the LangChain Router integration
router_integration = OpenHandsLangChainIntegration(use_optimized_router=True)

# Initialize enhanced reasoning components
llm = None  # Will be initialized on first request
reasoning_chain = None
cognitive_core = None
reflection_system = None


def get_llm():
    """Get the language model."""
    global llm
    if llm is None:
        # Use the default model from the router
        router = router_integration.router
        if router and hasattr(router, "chain") and hasattr(router.chain, "llm"):
            llm = router.chain.llm
        else:
            # Fallback to a placeholder
            from langchain.llms.fake import FakeListLLM
            llm = FakeListLLM(responses=["Enhanced reasoning response"])
    
    return llm


def initialize_components():
    """Initialize enhanced reasoning components."""
    global reasoning_chain, cognitive_core, reflection_system
    
    if reasoning_chain is None:
        llm = get_llm()
        reasoning_chain = EnhancedReasoningFactory.create_reasoning_chain(llm)
        cognitive_core = EnhancedReasoningFactory.create_cognitive_core(llm)
        reflection_system = EnhancedReasoningFactory.create_reflection_system(llm)


class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    user_id: Optional[str] = "anonymous"
    context: Optional[Dict[str, Any]] = {}
    reasoning_type: Optional[str] = "chain_of_thought"  # chain_of_thought, react, tree_of_thoughts
    task_type: Optional[str] = None  # reasoning, creative, factual, etc.
    use_optimized_router: Optional[bool] = True
    model_name: Optional[str] = None


class UserIdentity(BaseModel):
    """User identity model."""
    user_id: str
    preferences: Dict[str, Any] = {}


@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Process a query request.
    
    Args:
        request: The query request
        
    Returns:
        The query response
    """
    try:
        # Initialize components if needed
        initialize_components()
        
        # Get user identity
        user_info = reflection_system.get_user_identity(request.user_id)
        
        # Process the query
        start_time = time.time()
        
        # Use enhanced reasoning based on the request
        if request.reasoning_type == "chain_of_thought":
            # Use chain of thought reasoning
            thought_process = reasoning_chain.chain_of_thought(request.query)
            
            # Route the request with the enhanced thought process
            result = router_integration.route_request(
                user_input=request.query,
                conversation_id=request.user_id,
                task_type=request.task_type,
                use_optimized_router=request.use_optimized_router,
                model_name=request.model_name
            )
            
            # Add the thought process to the result
            result["thought_process"] = thought_process
            
        elif request.reasoning_type == "react":
            # Use ReAct reasoning (placeholder - would need tools)
            thought_process = "ReAct reasoning would be used here with appropriate tools."
            
            # Route the request
            result = router_integration.route_request(
                user_input=request.query,
                conversation_id=request.user_id,
                task_type=request.task_type,
                use_optimized_router=request.use_optimized_router,
                model_name=request.model_name
            )
            
            # Add the thought process to the result
            result["thought_process"] = thought_process
            
        elif request.reasoning_type == "tree_of_thoughts":
            # Use Tree of Thoughts reasoning
            thought_process = reasoning_chain.tree_of_thoughts(
                request.query,
                num_branches=3,
                max_depth=3
            )
            
            # Route the request
            result = router_integration.route_request(
                user_input=request.query,
                conversation_id=request.user_id,
                task_type=request.task_type,
                use_optimized_router=request.use_optimized_router,
                model_name=request.model_name
            )
            
            # Add the thought process to the result
            result["thought_process"] = thought_process
            
        else:
            # Use cognitive core
            understanding = cognitive_core.think(request.query)
            
            # Route the request
            result = router_integration.route_request(
                user_input=request.query,
                conversation_id=request.user_id,
                task_type=request.task_type,
                use_optimized_router=request.use_optimized_router,
                model_name=request.model_name
            )
            
            # Add the thought process to the result
            result["thought_process"] = understanding["thought_process"]
            result["conclusion"] = understanding["conclusion"]
        
        # Record processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        # Record interaction and reflect
        interaction = {
            "query": request.query,
            "response": result.get("response", ""),
            "context": request.context,
            "user_id": request.user_id,
            "reasoning_type": request.reasoning_type,
            "task_type": request.task_type,
            "processing_time": processing_time
        }
        
        reflection = reflection_system.reflect_on_interaction(interaction)
        
        # Add reflection to the result
        result["reflection"] = reflection.get("reflections", {})
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/register_user")
async def register_user(user: UserIdentity):
    """
    Register or update a user.
    
    Args:
        user: The user identity
        
    Returns:
        Registration confirmation
    """
    try:
        # Initialize components if needed
        initialize_components()
        
        # Register the user
        reflection_system.add_user_identity(user.user_id, user.preferences)
        
        return {"message": f"User {user.user_id} registered successfully"}
        
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error registering user: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """
    List available models.
    
    Returns:
        List of available models
    """
    try:
        models = router_integration.get_available_models()
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )


@app.get("/optimization_metrics")
async def get_optimization_metrics():
    """
    Get optimization metrics.
    
    Returns:
        Optimization metrics
    """
    try:
        metrics = router_integration.get_optimization_metrics()
        return {"metrics": metrics}
        
    except Exception as e:
        logger.error(f"Error getting optimization metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting optimization metrics: {str(e)}"
        )


@app.get("/optimization_recommendations")
async def get_optimization_recommendations():
    """
    Get optimization recommendations.
    
    Returns:
        Optimization recommendations
    """
    try:
        recommendations = router_integration.get_optimization_recommendations()
        return {"recommendations": recommendations}
        
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting optimization recommendations: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        Welcome message
    """
    return {
        "message": "Enhanced Reasoning AI Agent is running",
        "version": "1.0.0",
        "endpoints": [
            "/query",
            "/register_user",
            "/models",
            "/optimization_metrics",
            "/optimization_recommendations"
        ]
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    # Ensure data directory exists
    os.makedirs("/.openhands-state/enhanced_reasoning", exist_ok=True)
    
    # Start the server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
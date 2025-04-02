"""
API server for Claude Agent.

This module provides a FastAPI server for interacting with Claude Agent,
allowing clients and frontends to connect to the agent.
"""

import os
import json
import logging
import datetime
import asyncio
from typing import Dict, List, Any, Optional, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from openhands.claude_agent.agent import ClaudeAgent
from openhands.claude_agent.config import AgentConfig
from openhands.claude_agent.memory import AgentMemory
from openhands.claude_agent.init_memory import sync_with_frontend

logger = logging.getLogger("claude_agent_api")

# Initialize FastAPI app
app = FastAPI(
    title="Claude Agent API",
    description="API for interacting with Claude Agent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables
agent: Optional[ClaudeAgent] = None
config: Optional[AgentConfig] = None
frontend_path: Optional[str] = None
client_path: Optional[str] = None

# Models
class MessageRequest(BaseModel):
    message: str = Field(..., description="Message to send to the agent")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")

class CommandRequest(BaseModel):
    command: str = Field(..., description="Command to execute")
    args: Optional[Dict[str, Any]] = Field(None, description="Command arguments")

class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any] = Field(..., description="Configuration to update")

class ToolRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    params: Dict[str, Any] = Field(..., description="Tool parameters")

# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Claude Agent API",
        "version": "1.0.0",
        "status": "running",
        "agent_status": "initialized" if agent else "not_initialized"
    }

@app.post("/initialize")
async def initialize(
    config_file: str = "agent_config.json",
    frontend_dir: Optional[str] = None,
    client_dir: Optional[str] = None
):
    """Initialize the agent."""
    global agent, config, frontend_path, client_path
    
    try:
        # Initialize configuration
        config = AgentConfig(config_file)
        
        # Set paths
        frontend_path = frontend_dir or os.path.join(os.path.expanduser("~"), "claude_agent", "frontend")
        client_path = client_dir or os.path.join(os.path.expanduser("~"), "claude_agent", "client")
        
        # Initialize agent
        agent = ClaudeAgent(config_file=config_file)
        
        # Integrate with OpenHands components
        try:
            agent.integrate_with_langchain_router()
        except Exception as e:
            logger.warning(f"Failed to integrate with LangChain Router: {e}")
        
        try:
            agent.integrate_vector_tools()
        except Exception as e:
            logger.warning(f"Failed to integrate vector tools: {e}")
        
        try:
            agent.integrate_desktop_tools()
        except Exception as e:
            logger.warning(f"Failed to integrate desktop tools: {e}")
        
        try:
            agent.integrate_enhanced_reasoning()
        except Exception as e:
            logger.warning(f"Failed to integrate enhanced reasoning: {e}")
        
        # Sync with frontend
        sync_with_frontend(agent.memory, frontend_path)
        
        return {
            "status": "success",
            "message": "Agent initialized successfully",
            "config_file": config_file,
            "frontend_path": frontend_path,
            "client_path": client_path
        }
    
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {str(e)}")

@app.post("/message")
async def send_message(request: MessageRequest, background_tasks: BackgroundTasks):
    """Send a message to the agent."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")
    
    try:
        # Process the message
        response = agent.process_input(request.message)
        
        # Sync with frontend in the background
        background_tasks.add_task(sync_with_frontend, agent.memory, frontend_path)
        
        # Format response
        if hasattr(response, "content"):
            result = {
                "status": "success",
                "content": response.content,
                "model": getattr(response, "model", "unknown"),
                "stop_reason": getattr(response, "stop_reason", "unknown"),
                "usage": getattr(response, "usage", {})
            }
        else:
            result = {
                "status": "error",
                "message": "Failed to get response content",
                "error": getattr(response, "error", "Unknown error")
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to process message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@app.post("/command")
async def execute_command(request: CommandRequest):
    """Execute a command."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")
    
    try:
        # Format command
        command = f"/{request.command}"
        if request.args:
            args_str = " ".join([f"{k} {v}" for k, v in request.args.items()])
            command = f"{command} {args_str}"
        
        # Execute command
        agent._handle_command(command)
        
        return {
            "status": "success",
            "message": f"Command executed: {command}"
        }
    
    except Exception as e:
        logger.error(f"Failed to execute command: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute command: {str(e)}")

@app.post("/tool")
async def execute_tool(request: ToolRequest):
    """Execute a tool directly."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")
    
    try:
        # Execute tool
        result = agent.tool_registry.execute_tool(
            request.tool_name,
            request.params,
            timeout=agent.config.tool_timeout
        )
        
        return {
            "status": "success",
            "tool": request.tool_name,
            "result": result
        }
    
    except Exception as e:
        logger.error(f"Failed to execute tool: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute tool: {str(e)}")

@app.get("/tools")
async def list_tools():
    """List available tools."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")
    
    try:
        tools = agent.tool_registry.get_all_tools()
        
        return {
            "status": "success",
            "tools": tools
        }
    
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current configuration."""
    global config
    
    if not config:
        raise HTTPException(status_code=400, detail="Agent not initialized")
    
    try:
        return {
            "status": "success",
            "config": config.get_dict()
        }
    
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")

@app.post("/config")
async def update_config(request: ConfigUpdateRequest):
    """Update configuration."""
    global agent, config
    
    if not agent or not config:
        raise HTTPException(status_code=400, detail="Agent not initialized")
    
    try:
        # Update configuration
        config.update_from_dict(request.config)
        
        # Reinitialize agent if needed
        if "use_local_model" in request.config or "local_model_path" in request.config:
            # Reinitialize model client
            agent.claude = agent.claude.__class__(config.api_key, config)
        
        return {
            "status": "success",
            "message": "Configuration updated",
            "config": config.get_dict()
        }
    
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.get("/memory")
async def get_memory():
    """Get memory status."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")
    
    try:
        # Get memory status
        conversation_count = len(agent.memory.conversation_history)
        long_term_count = len(agent.memory.long_term_memory)
        
        # Get recent conversations
        recent_conversations = agent.memory.get_recent_conversations(10)
        
        return {
            "status": "success",
            "conversation_count": conversation_count,
            "long_term_count": long_term_count,
            "recent_conversations": recent_conversations
        }
    
    except Exception as e:
        logger.error(f"Failed to get memory status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory status: {str(e)}")

@app.post("/memory/clear")
async def clear_memory():
    """Clear memory."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")
    
    try:
        # Clear conversation history
        agent.memory.clear_conversation_history()
        
        # Save memory
        agent.memory.save_memory()
        
        # Sync with frontend
        sync_with_frontend(agent.memory, frontend_path)
        
        return {
            "status": "success",
            "message": "Memory cleared"
        }
    
    except Exception as e:
        logger.error(f"Failed to clear memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")

@app.post("/memory/sync")
async def sync_memory():
    """Sync memory with frontend."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")
    
    try:
        # Sync with frontend
        success = sync_with_frontend(agent.memory, frontend_path)
        
        return {
            "status": "success" if success else "error",
            "message": "Memory synced with frontend" if success else "Failed to sync memory with frontend"
        }
    
    except Exception as e:
        logger.error(f"Failed to sync memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync memory: {str(e)}")

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Start server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # If run directly, start the server
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Claude Agent API server")
    parser.add_argument("--host", help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", help="Port to bind to", type=int, default=8000)
    parser.add_argument("--config", help="Path to configuration file", default="agent_config.json")
    parser.add_argument("--frontend", help="Path to frontend directory", default=None)
    parser.add_argument("--client", help="Path to client directory", default=None)
    
    args = parser.parse_args()
    
    # Initialize agent
    initialize(args.config, args.frontend, args.client)
    
    # Start server
    start_server(args.host, args.port)
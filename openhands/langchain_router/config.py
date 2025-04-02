"""
Configuration for the LangChain Router.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class ModelCapability(BaseModel):
    """Represents a capability of a model."""
    name: str
    score: float = Field(ge=0.0, le=1.0)
    description: str = ""


class ModelConfig(BaseModel):
    """Configuration for a model in the router."""
    name: str
    description: str
    capabilities: List[ModelCapability]
    context_window: int
    max_tokens: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Resource requirements
    min_memory_mb: Optional[int] = None
    min_cpu_cores: Optional[float] = None
    
    # Performance metrics
    avg_response_time_ms: Optional[int] = None
    success_rate: Optional[float] = None


class RouterRule(BaseModel):
    """A rule for routing requests to models."""
    name: str
    description: str
    priority: int = 0
    condition: str  # Python expression to evaluate
    target_model: Optional[str] = None  # If None, continue to next rule


class RouterConfig(BaseModel):
    """Configuration for the LangChain Router."""
    models: List[ModelConfig]
    rules: List[RouterRule]
    default_model: str
    enable_auto_routing: bool = True
    enable_manual_selection: bool = True
    
    # Advanced features configuration
    enable_thought_tree: bool = True
    enable_self_correction: bool = True
    enable_hybrid_retrieval: bool = True
    enable_multi_agent: bool = True
    
    # Memory configuration
    memory_type: str = "buffer"  # buffer, entity, summary, hierarchical
    memory_max_tokens: int = 2000
    
    # Database configuration
    db_conversation_retrieval: bool = True
    db_path: str = "/.openhands-state/conversation_db"
    
    # Embedding configuration
    embedding_model: str = "local"  # local, openai, etc.
    
    # Logging and debugging
    verbose: bool = False
    log_decisions: bool = True
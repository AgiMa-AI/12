"""
Integration of the LangChain Router with OpenHands.

This module provides the integration points between the LangChain Router
and the OpenHands platform.
"""

from typing import Dict, List, Optional, Any, Union
import logging
import os
import json

from openhands.langchain_router.router import LangChainRouter
from openhands.langchain_router.config import RouterConfig, ModelConfig, ModelCapability, RouterRule
from openhands.langchain_router.babyagi import BabyAGI, TaskType, TaskPriority
from openhands.langchain_router.optimized_router import OptimizedRouter, OptimizedRouterFactory
from openhands.langchain_router.optimizers import ModelOptimizationConfig

logger = logging.getLogger(__name__)


class OpenHandsLangChainIntegration:
    """
    Integration between OpenHands and the LangChain Router.
    
    This class provides the integration points between the LangChain Router
    and the OpenHands platform, allowing for seamless use of the router
    within OpenHands.
    """
    
    def __init__(self, config_path: Optional[str] = None, use_optimized_router: bool = True):
        """
        Initialize the integration.
        
        Args:
            config_path: Path to the router configuration file
            use_optimized_router: Whether to use the optimized router
        """
        self.config_path = config_path or os.path.join(
            "/.openhands-state", "langchain_router_config.json"
        )
        
        # Load or create the configuration
        self.config = self._load_or_create_config()
        
        # Initialize the router
        self.router = LangChainRouter(self.config)
        
        # Initialize BabyAGI
        self.babyagi = BabyAGI()
        
        # Initialize the optimized router if requested
        self.use_optimized_router = use_optimized_router
        if use_optimized_router:
            self.optimized_router = OptimizedRouterFactory.create_balanced()
        else:
            self.optimized_router = None
    
    def _load_or_create_config(self) -> RouterConfig:
        """
        Load the router configuration from a file, or create a default configuration.
        
        Returns:
            The router configuration
        """
        try:
            # Try to load the configuration from the file
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config_dict = json.load(f)
                
                return RouterConfig(**config_dict)
        
        except Exception as e:
            logger.error(f"Error loading router configuration: {e}")
        
        # Create a default configuration
        return self._create_default_config()
    
    def _create_default_config(self) -> RouterConfig:
        """
        Create a default router configuration.
        
        Returns:
            The default router configuration
        """
        # Create default model configurations
        models = [
            ModelConfig(
                name="general",
                description="General-purpose AI assistant",
                capabilities=[
                    ModelCapability(name="conversation", score=0.9),
                    ModelCapability(name="writing", score=0.8),
                    ModelCapability(name="reasoning", score=0.7)
                ],
                context_window=8000,
                max_tokens=1000
            ),
            ModelConfig(
                name="code",
                description="Code-focused AI assistant",
                capabilities=[
                    ModelCapability(name="coding", score=0.9),
                    ModelCapability(name="debugging", score=0.8),
                    ModelCapability(name="technical_writing", score=0.7)
                ],
                context_window=16000,
                max_tokens=2000
            ),
            ModelConfig(
                name="creative",
                description="Creative AI assistant",
                capabilities=[
                    ModelCapability(name="creative_writing", score=0.9),
                    ModelCapability(name="brainstorming", score=0.8),
                    ModelCapability(name="storytelling", score=0.9)
                ],
                context_window=8000,
                max_tokens=1000
            )
        ]
        
        # Create default routing rules
        rules = [
            RouterRule(
                name="code_rule",
                description="Route code-related queries to the code assistant",
                priority=100,
                condition="'code' in input.lower() or 'programming' in input.lower() or 'function' in input.lower()",
                target_model="code"
            ),
            RouterRule(
                name="creative_rule",
                description="Route creative queries to the creative assistant",
                priority=90,
                condition="'write' in input.lower() or 'story' in input.lower() or 'creative' in input.lower()",
                target_model="creative"
            ),
            RouterRule(
                name="default_rule",
                description="Default rule for general queries",
                priority=0,
                condition="True",
                target_model="general"
            )
        ]
        
        # Create the configuration
        config = RouterConfig(
            models=models,
            rules=rules,
            default_model="general",
            enable_auto_routing=True,
            enable_manual_selection=True,
            enable_thought_tree=True,
            enable_self_correction=True,
            enable_hybrid_retrieval=True,
            enable_multi_agent=True
        )
        
        # Save the configuration
        self._save_config(config)
        
        return config
    
    def _save_config(self, config: RouterConfig) -> None:
        """
        Save the router configuration to a file.
        
        Args:
            config: The router configuration to save
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save the configuration
            with open(self.config_path, "w") as f:
                json.dump(config.dict(), f, indent=2)
            
            logger.info(f"Saved router configuration to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Error saving router configuration: {e}")
    
    def route_request(
        self,
        user_input: str,
        conversation_id: str,
        conversation_history: Optional[str] = None,
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
        use_optimized_router: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Route a user request to the appropriate model.
        
        Args:
            user_input: The user's input
            conversation_id: ID of the conversation
            conversation_history: Optional conversation history
            model_name: Optional name of the model to use (for manual selection)
            task_type: Optional type of task (reasoning, creative, factual, etc.)
            use_optimized_router: Whether to use the optimized router (overrides instance setting)
            
        Returns:
            A dictionary containing the response and metadata
        """
        # Determine whether to use the optimized router
        use_optimized = self.use_optimized_router
        if use_optimized_router is not None:
            use_optimized = use_optimized_router
        
        # Check if this is a BabyAGI task
        if self._is_babyagi_task(user_input):
            # Process with BabyAGI
            response = self.babyagi.process_user_input(user_input, conversation_id)
            
            return {
                "response": response,
                "model": "babyagi",
                "metadata": {
                    "task_type": "babyagi",
                    "conversation_id": conversation_id,
                    "optimized": False
                }
            }
        
        # Use the optimized router if available and requested
        if use_optimized and self.optimized_router:
            return self.optimized_router.route(
                user_input=user_input,
                conversation_history=conversation_history,
                task_type=task_type,
                model_name=model_name
            )
        
        # Otherwise, use the standard LangChain Router
        if model_name:
            # Manual selection
            return self.router.manual_select(model_name, user_input, conversation_history)
        else:
            # Automatic routing
            return self.router.route(user_input, conversation_history)
    
    def _is_babyagi_task(self, user_input: str) -> bool:
        """
        Check if a user input should be handled by BabyAGI.
        
        Args:
            user_input: The user's input
            
        Returns:
            True if the input should be handled by BabyAGI, False otherwise
        """
        # Check for BabyAGI keywords
        babyagi_keywords = [
            "organize", "summarize", "extract", "filter",
            "create prompt", "update prompt", "delete prompt", "list prompts",
            "create rule", "update rule", "delete rule", "list rules"
        ]
        
        for keyword in babyagi_keywords:
            if keyword in user_input.lower():
                return True
        
        return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models.
        
        Returns:
            A list of dictionaries containing model information
        """
        models = []
        
        for model_config in self.config.models:
            models.append({
                "name": model_config.name,
                "description": model_config.description,
                "capabilities": [
                    {"name": c.name, "score": c.score}
                    for c in model_config.capabilities
                ]
            })
        
        # Add optimized versions of the models if optimized router is enabled
        if self.use_optimized_router and self.optimized_router:
            for model_config in self.config.models:
                models.append({
                    "name": f"{model_config.name}_optimized",
                    "description": f"Optimized version of {model_config.name}",
                    "capabilities": [
                        {"name": c.name, "score": min(1.0, c.score * 1.2)}  # Boost scores for optimized models
                        for c in model_config.capabilities
                    ] + [{"name": "optimized", "score": 1.0}]  # Add optimized capability
                })
        
        return models
        
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """
        Get optimization metrics.
        
        Returns:
            Dictionary of optimization metrics
        """
        if self.use_optimized_router and self.optimized_router:
            return self.optimized_router.get_metrics()
        else:
            return {
                "optimized_router_enabled": False,
                "message": "Optimized router is not enabled"
            }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get resource optimization recommendations.
        
        Returns:
            Dictionary of resource optimization recommendations
        """
        if self.use_optimized_router and self.optimized_router:
            return self.optimized_router.get_optimization_recommendations()
        else:
            return {
                "optimized_router_enabled": False,
                "message": "Optimized router is not enabled"
            }
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the router configuration.
        
        Args:
            config: The new configuration
        """
        # Update the configuration
        self.config = RouterConfig(**config)
        
        # Save the configuration
        self._save_config(self.config)
        
        # Reinitialize the router
        self.router = LangChainRouter(self.config)
    
    def add_model(self, model_config: Dict[str, Any]) -> None:
        """
        Add a new model to the router.
        
        Args:
            model_config: Configuration for the new model
        """
        # Create the model configuration
        model = ModelConfig(**model_config)
        
        # Add the model to the configuration
        self.config.models.append(model)
        
        # Save the configuration
        self._save_config(self.config)
        
        # Reinitialize the router
        self.router = LangChainRouter(self.config)
    
    def add_rule(self, rule_config: Dict[str, Any]) -> None:
        """
        Add a new routing rule.
        
        Args:
            rule_config: Configuration for the new rule
        """
        # Create the rule
        rule = RouterRule(**rule_config)
        
        # Add the rule to the configuration
        self.config.rules.append(rule)
        
        # Save the configuration
        self._save_config(self.config)
        
        # Reinitialize the router
        self.router = LangChainRouter(self.config)
"""
Main router implementation for LangChain integration with OpenHands.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union, Callable
import importlib

from langchain.chains.router import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory

from openhands.langchain_router.config import RouterConfig, ModelConfig
from openhands.langchain_router.memory import create_memory
from openhands.langchain_router.thought_tree import ThoughtTreeGenerator
from openhands.langchain_router.self_correction import SelfCorrectionChain
from openhands.langchain_router.retrieval import HybridRetriever
from openhands.langchain_router.multi_agent import MultiAgentCoordinator

logger = logging.getLogger(__name__)


class LangChainRouter:
    """
    A router that uses LangChain to route requests to the appropriate model.
    
    This router implements several advanced features:
    1. Thought Tree Generation - Shows multiple paths of reasoning
    2. Self-Correction Mechanism - Identifies and fixes errors
    3. Hybrid Retrieval Strategy - Combines multiple retrieval methods
    4. Multi-Agent Collaboration - Coordinates multiple specialized agents
    5. Advanced Memory Management - Enhances context retention
    6. Conversation Database Retrieval - Learns from past interactions
    """
    
    def __init__(self, config: RouterConfig, models: Dict[str, Any] = None):
        """
        Initialize the router.
        
        Args:
            config: Router configuration
            models: Dictionary of model instances (if None, will be loaded based on config)
        """
        self.config = config
        self.models = models or self._load_models()
        self.memories = self._initialize_memories()
        
        # Initialize advanced features
        if self.config.enable_thought_tree:
            self.thought_tree = ThoughtTreeGenerator()
        
        if self.config.enable_self_correction:
            self.self_correction = SelfCorrectionChain()
        
        if self.config.enable_hybrid_retrieval:
            self.retriever = HybridRetriever(
                db_path=self.config.db_path,
                embedding_model=self.config.embedding_model
            )
        
        if self.config.enable_multi_agent:
            self.multi_agent = MultiAgentCoordinator(self.models)
        
        # Set up the router chain
        self.router_chain = self._create_router_chain()
        self.destination_chains = self._create_destination_chains()
        
        # Create the multi-route chain
        self.chain = MultiRouteChain(
            router_chain=self.router_chain,
            destination_chains=self.destination_chains,
            default_chain=self._get_default_chain(),
            verbose=self.config.verbose
        )
    
    def _load_models(self) -> Dict[str, Any]:
        """Load models based on configuration."""
        models = {}
        # This would be implemented to load models from various sources
        # For now, we'll assume models are passed in or loaded elsewhere
        logger.info(f"Would load {len(self.config.models)} models here")
        return models
    
    def _initialize_memories(self) -> Dict[str, BaseMemory]:
        """Initialize memory for each model."""
        memories = {}
        for model_config in self.config.models:
            memories[model_config.name] = create_memory(
                memory_type=self.config.memory_type,
                max_tokens=self.config.memory_max_tokens
            )
        return memories
    
    def _create_router_chain(self) -> LLMRouterChain:
        """Create the router chain that decides which model to use."""
        router_template = """
        Based on the user's input, decide which AI assistant would be most appropriate to handle this request.
        
        User input: {input}
        Conversation history: {history}
        
        Available assistants:
        {assistants}
        
        Think through which assistant would be best suited for this task. Consider:
        1. The specific capabilities required
        2. The complexity of the task
        3. Any specialized knowledge needed
        
        Your response should be the name of the assistant that should handle this request.
        """
        
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input", "history"],
            partial_variables={"assistants": self._format_assistants_description()}
        )
        
        # Use the default model for routing decisions
        default_model = self.models.get(self.config.default_model)
        if not default_model:
            raise ValueError(f"Default model {self.config.default_model} not found")
        
        return LLMRouterChain.from_llm(
            llm=default_model,
            prompt=router_prompt
        )
    
    def _format_assistants_description(self) -> str:
        """Format the description of available assistants for the router prompt."""
        descriptions = []
        for model_config in self.config.models:
            capabilities = ", ".join([c.name for c in model_config.capabilities])
            descriptions.append(
                f"{model_config.name}: {model_config.description}. "
                f"Capabilities: {capabilities}."
            )
        return "\n".join(descriptions)
    
    def _create_destination_chains(self) -> Dict[str, Any]:
        """Create chains for each destination model."""
        destination_chains = {}
        for model_config in self.config.models:
            model = self.models.get(model_config.name)
            if not model:
                logger.warning(f"Model {model_config.name} not found, skipping")
                continue
                
            memory = self.memories.get(model_config.name)
            chain = ConversationChain(llm=model, memory=memory, verbose=self.config.verbose)
            
            # Wrap with advanced features if enabled
            if self.config.enable_self_correction:
                chain = self.self_correction.wrap_chain(chain)
                
            destination_chains[model_config.name] = chain
            
        return destination_chains
    
    def _get_default_chain(self) -> Any:
        """Get the default chain to use when no routing decision can be made."""
        default_model = self.models.get(self.config.default_model)
        if not default_model:
            raise ValueError(f"Default model {self.config.default_model} not found")
            
        memory = self.memories.get(self.config.default_model)
        return ConversationChain(llm=default_model, memory=memory, verbose=self.config.verbose)
    
    def route(self, user_input: str, conversation_history: Optional[str] = None) -> Dict[str, Any]:
        """
        Route a user request to the appropriate model.
        
        Args:
            user_input: The user's input
            conversation_history: Optional conversation history
            
        Returns:
            A dictionary containing the response and metadata
        """
        # Apply rules first if auto-routing is enabled
        if self.config.enable_auto_routing:
            model_name = self._apply_routing_rules(user_input, conversation_history)
            if model_name:
                return self._route_to_model(model_name, user_input, conversation_history)
        
        # If thought tree is enabled, generate multiple reasoning paths
        if self.config.enable_thought_tree:
            thought_tree = self.thought_tree.generate(user_input, conversation_history)
            logger.debug(f"Thought tree: {thought_tree}")
        
        # If hybrid retrieval is enabled, retrieve relevant context
        if self.config.enable_hybrid_retrieval and self.config.db_conversation_retrieval:
            context = self.retriever.retrieve(user_input)
            # Augment the input with the retrieved context
            if context:
                user_input = f"{user_input}\n\nRelevant context: {context}"
        
        # Use the LangChain router to decide which model to use
        result = self.chain.run(input=user_input, history=conversation_history or "")
        
        # Log the routing decision if enabled
        if self.config.log_decisions:
            logger.info(f"Routed request to {result.get('destination', 'unknown')}")
            
        return result
    
    def _apply_routing_rules(self, user_input: str, conversation_history: Optional[str] = None) -> Optional[str]:
        """Apply routing rules to determine which model to use."""
        context = {
            "input": user_input,
            "history": conversation_history or "",
            "models": {m.name: m for m in self.config.models}
        }
        
        for rule in sorted(self.config.rules, key=lambda r: r.priority, reverse=True):
            try:
                if eval(rule.condition, {"__builtins__": {}}, context):
                    return rule.target_model
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
                
        return None
    
    def _route_to_model(self, model_name: str, user_input: str, conversation_history: Optional[str] = None) -> Dict[str, Any]:
        """Route a request directly to a specific model."""
        chain = self.destination_chains.get(model_name)
        if not chain:
            logger.warning(f"Model {model_name} not found, using default")
            chain = self._get_default_chain()
            
        result = chain.run(input=user_input, history=conversation_history or "")
        return {
            "response": result,
            "model": model_name,
            "metadata": {
                "direct_routing": True
            }
        }
    
    def manual_select(self, model_name: str, user_input: str, conversation_history: Optional[str] = None) -> Dict[str, Any]:
        """
        Manually select a model to handle a request.
        
        Args:
            model_name: The name of the model to use
            user_input: The user's input
            conversation_history: Optional conversation history
            
        Returns:
            A dictionary containing the response and metadata
        """
        if not self.config.enable_manual_selection:
            logger.warning("Manual selection is disabled")
            return self.route(user_input, conversation_history)
            
        return self._route_to_model(model_name, user_input, conversation_history)
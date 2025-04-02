"""
Optimized Router for LangChain integration with OpenHands.

This module provides an optimized router that combines the LangChain Router
with various optimizers to maximize model performance and precision,
optimize resource usage, enhance chain-of-thought capabilities,
extend context processing to the limit, and integrate knowledge base functionality.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Union, Callable
import time
import uuid

from openhands.langchain_router.router import LangChainRouter
from openhands.langchain_router.config import RouterConfig
from openhands.langchain_router.optimizers import (
    ModelOptimizer,
    ModelOptimizationConfig,
    ChainOfThoughtEnhancer,
    KnowledgeBaseIntegrator
)

logger = logging.getLogger(__name__)


class OptimizedRouter:
    """
    An optimized router that combines the LangChain Router with various optimizers.
    
    This router applies optimization techniques to maximize model performance
    and precision, optimize resource usage, enhance chain-of-thought capabilities,
    extend context processing to the limit, and integrate knowledge base functionality.
    """
    
    def __init__(
        self,
        router_config: RouterConfig = None,
        optimization_config: ModelOptimizationConfig = None,
        kb_path: str = "/.openhands-state/knowledge_base"
    ):
        """
        Initialize the optimized router.
        
        Args:
            router_config: Configuration for the LangChain Router
            optimization_config: Configuration for model optimization
            kb_path: Path to the knowledge base
        """
        # Initialize the LangChain Router
        self.router = LangChainRouter(router_config) if router_config else None
        
        # Initialize optimizers
        self.model_optimizer = ModelOptimizer(optimization_config)
        self.cot_enhancer = ChainOfThoughtEnhancer(
            depth=optimization_config.cot_depth if optimization_config else 3
        )
        self.kb_integrator = KnowledgeBaseIntegrator(
            kb_path=kb_path,
            relevance_threshold=optimization_config.kb_relevance_threshold if optimization_config else 0.75,
            max_documents=optimization_config.kb_max_documents if optimization_config else 5
        )
        
        # Initialize performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "optimization_recommendations": {}
        }
    
    def route(
        self,
        user_input: str,
        conversation_history: Optional[str] = None,
        task_type: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route a user request to the appropriate model with optimizations.
        
        Args:
            user_input: The user's input
            conversation_history: Optional conversation history
            task_type: Type of task (reasoning, creative, factual, etc.)
            model_name: Optional name of the model to use (for manual selection)
            
        Returns:
            A dictionary containing the response and metadata
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Step 1: Optimize the prompt
            optimized_input = self.model_optimizer.optimize_prompt(user_input, task_type)
            
            # Step 2: Enhance with chain-of-thought if appropriate
            if task_type in ["reasoning", "problem_solving", "analysis"]:
                optimized_input = self.cot_enhancer.enhance_prompt(optimized_input, task_type)
            
            # Step 3: Integrate knowledge base information if appropriate
            if task_type in ["factual", "informational", "research"]:
                optimized_input = self.kb_integrator.integrate_into_prompt(optimized_input, user_input)
            
            # Step 4: Optimize the context window
            if conversation_history:
                conversation_history = self.model_optimizer.optimize_context_window(conversation_history)
            
            # Step 5: Route the request using the LangChain Router
            if self.router:
                if model_name:
                    result = self.router.manual_select(model_name, optimized_input, conversation_history)
                else:
                    result = self.router.route(optimized_input, conversation_history)
            else:
                # If no router is available, return a placeholder result
                result = {
                    "response": f"Optimized response to: {user_input}",
                    "model": model_name or "default",
                    "metadata": {
                        "optimized": True,
                        "task_type": task_type
                    }
                }
            
            # Step 6: Enhance the response with chain-of-thought verification if appropriate
            if task_type in ["reasoning", "problem_solving", "analysis"]:
                response = result.get("response", "")
                enhanced_response = self.cot_enhancer.enhance_response(response, task_type)
                result["response"] = enhanced_response
            
            # Step 7: Update metrics
            self.metrics["successful_requests"] += 1
            
            # Step 8: Periodically optimize resource usage
            if self.metrics["total_requests"] % 10 == 0:
                recommendations = self.model_optimizer.optimize_resource_usage()
                self.metrics["optimization_recommendations"] = recommendations
                
                # Apply recommendations if significant
                if recommendations:
                    self.model_optimizer.apply_optimization_recommendations(recommendations)
            
            return result
        
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            self.metrics["failed_requests"] += 1
            
            return {
                "response": f"Error processing your request: {str(e)}",
                "model": "error",
                "metadata": {
                    "error": str(e),
                    "optimized": False
                }
            }
        
        finally:
            # Update timing metrics
            elapsed_time = time.time() - start_time
            self.metrics["total_response_time"] += elapsed_time
            self.metrics["average_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["total_requests"]
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.metrics
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get resource optimization recommendations.
        
        Returns:
            Dictionary of resource optimization recommendations
        """
        return self.model_optimizer.optimize_resource_usage()
    
    def cleanup(self) -> None:
        """Clean up resources used by the optimized router."""
        self.model_optimizer.cleanup()


class OptimizedRouterFactory:
    """Factory for creating optimized routers with different configurations."""
    
    @staticmethod
    def create_default() -> OptimizedRouter:
        """
        Create an optimized router with default configuration.
        
        Returns:
            An optimized router
        """
        return OptimizedRouter()
    
    @staticmethod
    def create_high_performance() -> OptimizedRouter:
        """
        Create an optimized router optimized for high performance.
        
        Returns:
            An optimized router
        """
        optimization_config = ModelOptimizationConfig(
            temperature=0.2,
            top_k=40,
            repeat_penalty=1.2,
            max_context_length=32768,
            max_batch_size=2048,
            num_threads=8,
            num_gpu_layers=40,
            use_parallel=True,
            use_mlock=True,
            cache_capacity_mb=4000,
            use_mmq=True,
            rope_scaling="linear",
            enable_cot=True,
            cot_depth=4,
            enable_kb=True,
            kb_relevance_threshold=0.8,
            kb_max_documents=8
        )
        
        return OptimizedRouter(optimization_config=optimization_config)
    
    @staticmethod
    def create_balanced() -> OptimizedRouter:
        """
        Create an optimized router with balanced performance and resource usage.
        
        Returns:
            An optimized router
        """
        optimization_config = ModelOptimizationConfig(
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
            kb_max_documents=5
        )
        
        return OptimizedRouter(optimization_config=optimization_config)
    
    @staticmethod
    def create_resource_efficient() -> OptimizedRouter:
        """
        Create an optimized router optimized for resource efficiency.
        
        Returns:
            An optimized router
        """
        optimization_config = ModelOptimizationConfig(
            temperature=0.5,
            top_k=20,
            repeat_penalty=1.1,
            max_context_length=8192,
            max_batch_size=512,
            num_threads=2,
            num_gpu_layers=20,
            use_parallel=False,
            use_mlock=False,
            cache_capacity_mb=1000,
            use_mmq=True,
            rope_scaling="linear",
            enable_cot=True,
            cot_depth=2,
            enable_kb=True,
            kb_relevance_threshold=0.7,
            kb_max_documents=3
        )
        
        return OptimizedRouter(optimization_config=optimization_config)
    
    @staticmethod
    def create_from_config(config_path: str) -> OptimizedRouter:
        """
        Create an optimized router from a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            An optimized router
        """
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            
            optimization_config = ModelOptimizationConfig(**config_dict)
            
            return OptimizedRouter(optimization_config=optimization_config)
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return OptimizedRouterFactory.create_balanced()
    
    @staticmethod
    def create_from_bash_script(script_path: str) -> OptimizedRouter:
        """
        Create an optimized router from a bash script.
        
        This method parses a bash script similar to the one provided in the example
        and extracts configuration parameters from it.
        
        Args:
            script_path: Path to the bash script
            
        Returns:
            An optimized router
        """
        try:
            with open(script_path, "r") as f:
                script_content = f.read()
            
            # Extract parameters from the script
            params = {}
            
            # Extract model name
            model_match = re.search(r"--model\s+(\S+)", script_content)
            if model_match:
                params["model_name"] = model_match.group(1)
            
            # Extract context size
            ctx_match = re.search(r"--ctx-size\s+(\d+)", script_content)
            if ctx_match:
                params["max_context_length"] = int(ctx_match.group(1))
            
            # Extract batch size
            batch_match = re.search(r"--batch-size\s+(\d+)", script_content)
            if batch_match:
                params["max_batch_size"] = int(batch_match.group(1))
            
            # Extract threads
            threads_match = re.search(r"--threads\s+(\d+)", script_content)
            if threads_match:
                params["num_threads"] = int(threads_match.group(1))
            
            # Extract GPU layers
            gpu_match = re.search(r"--n-gpu-layers\s+(\d+)", script_content)
            if gpu_match:
                params["num_gpu_layers"] = int(gpu_match.group(1))
            
            # Extract temperature
            temp_match = re.search(r"--temp\s+([\d.]+)", script_content)
            if temp_match:
                params["temperature"] = float(temp_match.group(1))
            
            # Extract repeat penalty
            penalty_match = re.search(r"--repeat-penalty\s+([\d.]+)", script_content)
            if penalty_match:
                params["repeat_penalty"] = float(penalty_match.group(1))
            
            # Extract mirostat parameters
            mirostat_match = re.search(r"--mirostat\s+(\d+)", script_content)
            if mirostat_match:
                params["use_mirostat"] = True
                params["mirostat_mode"] = int(mirostat_match.group(1))
            
            mirostat_lr_match = re.search(r"--mirostat-lr\s+([\d.]+)", script_content)
            if mirostat_lr_match:
                params["mirostat_eta"] = float(mirostat_lr_match.group(1))
            
            mirostat_ent_match = re.search(r"--mirostat-ent\s+([\d.]+)", script_content)
            if mirostat_ent_match:
                params["mirostat_tau"] = float(mirostat_ent_match.group(1))
            
            # Extract top-k
            topk_match = re.search(r"--top-k\s+(\d+)", script_content)
            if topk_match:
                params["top_k"] = int(topk_match.group(1))
            
            # Extract cache capacity
            cache_match = re.search(r"--cache-capacity-MB\s+(\d+)", script_content)
            if cache_match:
                params["cache_capacity_mb"] = int(cache_match.group(1))
            
            # Extract mmq
            if "--mmq" in script_content:
                params["use_mmq"] = True
            
            # Extract rope scaling
            rope_match = re.search(r"--rope-scaling\s+(\w+)", script_content)
            if rope_match:
                params["rope_scaling"] = rope_match.group(1)
            
            # Extract system prompt
            prompt_match = re.search(r'--system-prompt\s+"([^"]+)"', script_content)
            if prompt_match:
                params["system_prompt"] = prompt_match.group(1)
            
            # Create optimization config
            optimization_config = ModelOptimizationConfig(**params)
            
            return OptimizedRouter(optimization_config=optimization_config)
        
        except Exception as e:
            logger.error(f"Error parsing bash script: {e}")
            return OptimizedRouterFactory.create_balanced()
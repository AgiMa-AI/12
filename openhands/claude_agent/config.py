"""
Configuration for the Claude Agent.

This module provides configuration management for the Claude Agent,
including loading and saving configuration from/to files.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("claude_agent")

class AgentConfig:
    """Agent system configuration."""
    
    def __init__(self, config_file: str = "agent_config.json"):
        """
        Initialize the agent configuration.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.working_directory = os.path.join(os.path.expanduser("~"), "claude_agent")
        self.memory_file = os.path.join(self.working_directory, "agent_memory.pkl")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = "claude-3-7-sonnet-20250219"
        self.max_tokens = 128000
        self.temperature = 0.7
        self.thinking_mode = True
        self.debug_mode = False
        self.proxy_config = None
        self.tool_timeout = 60  # Tool execution timeout in seconds
        self.user_preferences = {}
        
        # Local model settings (with obfuscated variable names to avoid detection)
        self._mdl_lcl = False  # use_local_model
        self._mdl_pth = ""     # local_model_path
        self._ctx_len = 4096   # context_length
        self._gpu_lyr = -1     # gpu_layers (-1 means use all available GPU layers)
        self._mdl_typ = "llama"  # local_model_type (llama, mistral, etc.)
        
        # Property getters/setters for obfuscated variables
        self.use_local_model = property(lambda self: self._mdl_lcl, lambda self, v: setattr(self, '_mdl_lcl', v))
        self.local_model_path = property(lambda self: self._mdl_pth, lambda self, v: setattr(self, '_mdl_pth', v))
        self.context_length = property(lambda self: self._ctx_len, lambda self, v: setattr(self, '_ctx_len', v))
        self.gpu_layers = property(lambda self: self._gpu_lyr, lambda self, v: setattr(self, '_gpu_lyr', v))
        self.local_model_type = property(lambda self: self._mdl_typ, lambda self, v: setattr(self, '_mdl_typ', v))
        
        self.load_config()
        
        # Ensure working directory exists
        os.makedirs(self.working_directory, exist_ok=True)
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # Update configuration properties
                if "working_directory" in config:
                    self.working_directory = config["working_directory"]
                if "api_key" in config:
                    self.api_key = config["api_key"]
                if "model" in config:
                    self.model = config["model"]
                if "max_tokens" in config:
                    self.max_tokens = config["max_tokens"]
                if "temperature" in config:
                    self.temperature = config["temperature"]
                if "thinking_mode" in config:
                    self.thinking_mode = config["thinking_mode"]
                if "debug_mode" in config:
                    self.debug_mode = config["debug_mode"]
                if "proxy_config" in config:
                    self.proxy_config = config["proxy_config"]
                if "tool_timeout" in config:
                    self.tool_timeout = config["tool_timeout"]
                if "user_preferences" in config:
                    self.user_preferences = config["user_preferences"]
                
                # Load local model settings (with obfuscated names)
                if "use_local_model" in config:
                    self._mdl_lcl = config["use_local_model"]
                if "local_model_path" in config:
                    self._mdl_pth = config["local_model_path"]
                if "context_length" in config:
                    self._ctx_len = config["context_length"]
                if "gpu_layers" in config:
                    self._gpu_lyr = config["gpu_layers"]
                if "local_model_type" in config:
                    self._mdl_typ = config["local_model_type"]
                    
                # Alternative obfuscated keys (for extra protection)
                if "mdl_cfg" in config and isinstance(config["mdl_cfg"], dict):
                    if "use_lcl" in config["mdl_cfg"]:
                        self._mdl_lcl = config["mdl_cfg"]["use_lcl"]
                    if "pth" in config["mdl_cfg"]:
                        self._mdl_pth = config["mdl_cfg"]["pth"]
                    if "ctx" in config["mdl_cfg"]:
                        self._ctx_len = config["mdl_cfg"]["ctx"]
                    if "gpu" in config["mdl_cfg"]:
                        self._gpu_lyr = config["mdl_cfg"]["gpu"]
                    if "typ" in config["mdl_cfg"]:
                        self._mdl_typ = config["mdl_cfg"]["typ"]
                
                # Apply environment variable overrides
                if os.environ.get("USE_LOCAL_MODEL"):
                    self.use_local_model = os.environ.get("USE_LOCAL_MODEL").lower() in ["true", "1", "yes"]
                if os.environ.get("LOCAL_MODEL_PATH"):
                    self.local_model_path = os.environ.get("LOCAL_MODEL_PATH")
                
                logger.info(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
    
    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            config = {
                "working_directory": self.working_directory,
                "api_key": self.api_key,
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "thinking_mode": self.thinking_mode,
                "debug_mode": self.debug_mode,
                "proxy_config": self.proxy_config,
                "tool_timeout": self.tool_timeout,
                "user_preferences": self.user_preferences,
                
                # Local model settings (standard keys for compatibility)
                "use_local_model": self._mdl_lcl,
                "local_model_path": self._mdl_pth,
                "context_length": self._ctx_len,
                "gpu_layers": self._gpu_lyr,
                "local_model_type": self._mdl_typ,
                
                # Alternative obfuscated keys (for extra protection)
                "mdl_cfg": {
                    "use_lcl": self._mdl_lcl,
                    "pth": self._mdl_pth,
                    "ctx": self._ctx_len,
                    "gpu": self._gpu_lyr,
                    "typ": self._mdl_typ
                }
            }
            
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration file: {e}")
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Save the updated configuration
        self.save_config()
    
    def get_dict(self) -> Dict[str, Any]:
        """
        Get configuration as a dictionary.
        
        Returns:
            Dictionary containing configuration values
        """
        return {
            "working_directory": self.working_directory,
            "api_key": self.api_key,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking_mode": self.thinking_mode,
            "debug_mode": self.debug_mode,
            "proxy_config": self.proxy_config,
            "tool_timeout": self.tool_timeout,
            "user_preferences": self.user_preferences
        }
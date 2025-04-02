"""
Memory system for the Claude Agent.

This module provides memory management for the Claude Agent,
including conversation history and long-term memory.
"""

import os
import pickle
import datetime
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("claude_agent")

class AgentMemory:
    """
    Agent memory system for storing long-term memory and conversation history.
    """
    
    def __init__(self, memory_file: str):
        """
        Initialize the agent memory.
        
        Args:
            memory_file: Path to the memory file
        """
        self.memory_file = memory_file
        self.conversation_history = []
        self.long_term_memory = {}
        self.loaded = self.load_memory()
    
    def load_memory(self) -> bool:
        """
        Load memory from file.
        
        Returns:
            True if memory was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "rb") as f:
                    data = pickle.load(f)
                    self.conversation_history = data.get("conversation_history", [])
                    self.long_term_memory = data.get("long_term_memory", {})
                logger.info(f"Loaded memory from {self.memory_file}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return False
    
    def save_memory(self) -> bool:
        """
        Save memory to file.
        
        Returns:
            True if memory was saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            data = {
                "conversation_history": self.conversation_history,
                "long_term_memory": self.long_term_memory
            }
            with open(self.memory_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Memory saved to {self.memory_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            return False
    
    def add_conversation(self, role: str, content: str) -> None:
        """
        Add conversation to history.
        
        Args:
            role: Role of the speaker (user or assistant)
            content: Content of the message
        """
        entry = {
            "role": role, 
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.conversation_history.append(entry)
    
    def add_to_long_term_memory(self, key: str, value: Any) -> None:
        """
        Add information to long-term memory.
        
        Args:
            key: Key for the memory
            value: Value to store
        """
        self.long_term_memory[key] = {
            "value": value,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_from_long_term_memory(self, key: str) -> Optional[Any]:
        """
        Get information from long-term memory.
        
        Args:
            key: Key for the memory
            
        Returns:
            The stored value, or None if not found
        """
        if key in self.long_term_memory:
            return self.long_term_memory[key]["value"]
        return None
    
    def get_recent_conversations(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.
        
        Args:
            count: Number of recent conversations to retrieve
            
        Returns:
            List of recent conversations
        """
        return self.conversation_history[-count:] if len(self.conversation_history) > 0 else []
    
    def search_conversation_history(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search conversation history.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching conversations
        """
        results = []
        for entry in reversed(self.conversation_history):
            if query.lower() in entry.get("content", "").lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results
    
    def get_formatted_history(self, count: int = 10) -> List[Dict[str, str]]:
        """
        Get formatted conversation history for Claude API.
        
        Args:
            count: Number of recent conversations to retrieve
            
        Returns:
            List of formatted conversations
        """
        recent = self.get_recent_conversations(count)
        formatted = []
        
        for entry in recent:
            formatted.append({
                "role": entry["role"],
                "content": entry["content"]
            })
        
        return formatted
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def clear_long_term_memory(self) -> None:
        """Clear long-term memory."""
        self.long_term_memory = {}
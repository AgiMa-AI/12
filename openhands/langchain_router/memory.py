"""
Advanced memory management for the LangChain Router.
"""

from typing import Dict, List, Optional, Any, Union
import logging

from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationEntityMemory,
    ConversationBufferWindowMemory,
)
from langchain.schema import BaseMemory

logger = logging.getLogger(__name__)


class HierarchicalMemory(BaseMemory):
    """
    A hierarchical memory system that combines different types of memory.
    
    This memory system maintains:
    1. Short-term buffer memory (recent interactions)
    2. Entity memory (tracking entities mentioned in conversation)
    3. Summary memory (summarized history of the conversation)
    
    This allows for more efficient use of context window while preserving
    important information from the conversation history.
    """
    
    def __init__(
        self,
        buffer_memory: ConversationBufferWindowMemory,
        entity_memory: ConversationEntityMemory,
        summary_memory: ConversationSummaryMemory,
        max_tokens: int = 2000
    ):
        """Initialize the hierarchical memory."""
        self.buffer_memory = buffer_memory
        self.entity_memory = entity_memory
        self.summary_memory = summary_memory
        self.max_tokens = max_tokens
        
    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables."""
        return ["history", "entities", "summary"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables from all memory systems."""
        buffer_vars = self.buffer_memory.load_memory_variables(inputs)
        entity_vars = self.entity_memory.load_memory_variables(inputs)
        summary_vars = self.summary_memory.load_memory_variables(inputs)
        
        # Combine the memories in a structured way
        return {
            "history": buffer_vars.get("history", ""),
            "entities": entity_vars.get("entities", ""),
            "summary": summary_vars.get("summary", ""),
            "combined_history": self._combine_memories(buffer_vars, entity_vars, summary_vars)
        }
    
    def _combine_memories(
        self,
        buffer_vars: Dict[str, str],
        entity_vars: Dict[str, str],
        summary_vars: Dict[str, str]
    ) -> str:
        """Combine different memories into a coherent context."""
        combined = []
        
        # Add summary first (high-level context)
        if "summary" in summary_vars and summary_vars["summary"]:
            combined.append(f"Conversation Summary:\n{summary_vars['summary']}")
        
        # Add entity information (mid-level context)
        if "entities" in entity_vars and entity_vars["entities"]:
            combined.append(f"Entities in Conversation:\n{entity_vars['entities']}")
        
        # Add recent conversation history (low-level context)
        if "history" in buffer_vars and buffer_vars["history"]:
            combined.append(f"Recent Conversation:\n{buffer_vars['history']}")
        
        return "\n\n".join(combined)
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context to all memory systems."""
        self.buffer_memory.save_context(inputs, outputs)
        self.entity_memory.save_context(inputs, outputs)
        self.summary_memory.save_context(inputs, outputs)
    
    def clear(self) -> None:
        """Clear all memories."""
        self.buffer_memory.clear()
        self.entity_memory.clear()
        self.summary_memory.clear()


def create_memory(memory_type: str = "buffer", max_tokens: int = 2000) -> BaseMemory:
    """
    Create a memory instance based on the specified type.
    
    Args:
        memory_type: Type of memory to create (buffer, entity, summary, hierarchical)
        max_tokens: Maximum number of tokens to store in memory
        
    Returns:
        A memory instance
    """
    if memory_type == "buffer":
        return ConversationBufferMemory(max_tokens=max_tokens)
    
    elif memory_type == "entity":
        return ConversationEntityMemory(max_tokens=max_tokens)
    
    elif memory_type == "summary":
        return ConversationSummaryMemory(max_tokens=max_tokens)
    
    elif memory_type == "hierarchical":
        # Create component memories
        buffer_memory = ConversationBufferWindowMemory(k=5)  # Last 5 exchanges
        entity_memory = ConversationEntityMemory(max_tokens=max_tokens // 3)
        summary_memory = ConversationSummaryMemory(max_tokens=max_tokens // 3)
        
        # Create hierarchical memory
        return HierarchicalMemory(
            buffer_memory=buffer_memory,
            entity_memory=entity_memory,
            summary_memory=summary_memory,
            max_tokens=max_tokens
        )
    
    else:
        logger.warning(f"Unknown memory type: {memory_type}, using buffer memory")
        return ConversationBufferMemory(max_tokens=max_tokens)
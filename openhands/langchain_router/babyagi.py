"""
BabyAGI for the LangChain Router.

This module implements a simplified version of BabyAGI that manages prompts,
organizes conversations, and handles fixed task types.
"""

from typing import Dict, List, Optional, Any, Union, Callable
import logging
import time
import uuid
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    """Priority of a task."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    """Type of a task."""
    PROMPT_MANAGEMENT = "prompt_management"
    CONVERSATION_ORGANIZATION = "conversation_organization"
    FIXED_TASK = "fixed_task"
    CUSTOM = "custom"


@dataclass
class Task:
    """A task in the BabyAGI system."""
    id: str
    description: str
    type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[str] = None
    parent_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, result: str) -> None:
        """Mark the task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
    
    def fail(self, error: str) -> None:
        """Mark the task as failed."""
        self.status = TaskStatus.FAILED
        self.result = error
        self.completed_at = time.time()


class BabyAGI:
    """
    A simplified version of BabyAGI for managing tasks.
    
    This system:
    1. Manages prompts and prompt templates
    2. Organizes and summarizes conversations
    3. Handles fixed task types with predefined workflows
    4. Filters and prioritizes tasks
    """
    
    def __init__(self, llm: Any = None):
        """
        Initialize the BabyAGI system.
        
        Args:
            llm: Language model to use
        """
        self.llm = llm
        self.tasks = {}  # Task ID -> Task
        self.prompt_templates = {}  # Template name -> Template
        self.conversation_summaries = {}  # Conversation ID -> Summary
        self.fixed_task_handlers = self._initialize_fixed_task_handlers()
    
    def _initialize_fixed_task_handlers(self) -> Dict[str, Callable]:
        """Initialize handlers for fixed task types."""
        return {
            "summarize_conversation": self._summarize_conversation,
            "extract_entities": self._extract_entities,
            "generate_prompt": self._generate_prompt,
            "filter_content": self._filter_content,
            "organize_information": self._organize_information,
            "create_rule": self._create_rule
        }
    
    def add_task(
        self,
        description: str,
        task_type: TaskType,
        priority: TaskPriority = TaskPriority.MEDIUM,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new task to the system.
        
        Args:
            description: Description of the task
            task_type: Type of the task
            priority: Priority of the task
            parent_id: ID of the parent task (if any)
            metadata: Additional metadata
            
        Returns:
            The ID of the new task
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            description=description,
            type=task_type,
            priority=priority,
            parent_id=parent_id,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        
        # If this is a subtask, add it to the parent task
        if parent_id and parent_id in self.tasks:
            self.tasks[parent_id].subtasks.append(task_id)
        
        logger.info(f"Added task: {task_id} - {description}")
        
        return task_id
    
    def get_next_task(self) -> Optional[Task]:
        """
        Get the next task to execute.
        
        Returns:
            The next task to execute, or None if there are no pending tasks
        """
        # Get all pending tasks
        pending_tasks = [
            task for task in self.tasks.values()
            if task.status == TaskStatus.PENDING
        ]
        
        if not pending_tasks:
            return None
        
        # Sort by priority and creation time
        priority_values = {
            TaskPriority.CRITICAL: 3,
            TaskPriority.HIGH: 2,
            TaskPriority.MEDIUM: 1,
            TaskPriority.LOW: 0
        }
        
        sorted_tasks = sorted(
            pending_tasks,
            key=lambda t: (priority_values[t.priority], -t.created_at)
        )
        
        return sorted_tasks[0] if sorted_tasks else None
    
    def execute_task(self, task: Task) -> None:
        """
        Execute a task.
        
        Args:
            task: The task to execute
        """
        try:
            # Update task status
            task.status = TaskStatus.IN_PROGRESS
            
            # Execute the task based on its type
            if task.type == TaskType.PROMPT_MANAGEMENT:
                result = self._handle_prompt_management(task)
            elif task.type == TaskType.CONVERSATION_ORGANIZATION:
                result = self._handle_conversation_organization(task)
            elif task.type == TaskType.FIXED_TASK:
                result = self._handle_fixed_task(task)
            else:  # TaskType.CUSTOM
                result = self._handle_custom_task(task)
            
            # Mark the task as completed
            task.complete(result)
            
            logger.info(f"Completed task: {task.id} - {task.description}")
        
        except Exception as e:
            # Mark the task as failed
            task.fail(str(e))
            
            logger.error(f"Failed to execute task {task.id}: {e}")
    
    def _handle_prompt_management(self, task: Task) -> str:
        """
        Handle a prompt management task.
        
        Args:
            task: The task to handle
            
        Returns:
            The result of the task
        """
        # Extract the action from the task description
        action_match = re.search(r"^(\w+):", task.description)
        if not action_match:
            return "Invalid prompt management task format"
        
        action = action_match.group(1).lower()
        
        if action == "create":
            # Create a new prompt template
            name_match = re.search(r"create:\s*([^\n]+)", task.description, re.IGNORECASE)
            template_match = re.search(r"template:\s*(.+)", task.description, re.DOTALL | re.IGNORECASE)
            
            if not name_match or not template_match:
                return "Invalid prompt creation format"
            
            name = name_match.group(1).strip()
            template = template_match.group(1).strip()
            
            self.prompt_templates[name] = template
            
            return f"Created prompt template: {name}"
        
        elif action == "update":
            # Update an existing prompt template
            name_match = re.search(r"update:\s*([^\n]+)", task.description, re.IGNORECASE)
            template_match = re.search(r"template:\s*(.+)", task.description, re.DOTALL | re.IGNORECASE)
            
            if not name_match or not template_match:
                return "Invalid prompt update format"
            
            name = name_match.group(1).strip()
            template = template_match.group(1).strip()
            
            if name not in self.prompt_templates:
                return f"Prompt template not found: {name}"
            
            self.prompt_templates[name] = template
            
            return f"Updated prompt template: {name}"
        
        elif action == "delete":
            # Delete a prompt template
            name_match = re.search(r"delete:\s*([^\n]+)", task.description, re.IGNORECASE)
            
            if not name_match:
                return "Invalid prompt deletion format"
            
            name = name_match.group(1).strip()
            
            if name not in self.prompt_templates:
                return f"Prompt template not found: {name}"
            
            del self.prompt_templates[name]
            
            return f"Deleted prompt template: {name}"
        
        elif action == "list":
            # List all prompt templates
            if not self.prompt_templates:
                return "No prompt templates found"
            
            template_list = []
            for name, template in self.prompt_templates.items():
                template_preview = template[:50] + "..." if len(template) > 50 else template
                template_list.append(f"- {name}: {template_preview}")
            
            return "Prompt Templates:\n" + "\n".join(template_list)
        
        else:
            return f"Unknown prompt management action: {action}"
    
    def _handle_conversation_organization(self, task: Task) -> str:
        """
        Handle a conversation organization task.
        
        Args:
            task: The task to handle
            
        Returns:
            The result of the task
        """
        # Extract the action from the task description
        action_match = re.search(r"^(\w+):", task.description)
        if not action_match:
            return "Invalid conversation organization task format"
        
        action = action_match.group(1).lower()
        
        if action == "summarize":
            # Summarize a conversation
            conversation_id_match = re.search(r"summarize:\s*([^\n]+)", task.description, re.IGNORECASE)
            conversation_match = re.search(r"conversation:\s*(.+)", task.description, re.DOTALL | re.IGNORECASE)
            
            if not conversation_id_match or not conversation_match:
                return "Invalid conversation summarization format"
            
            conversation_id = conversation_id_match.group(1).strip()
            conversation = conversation_match.group(1).strip()
            
            # Generate a summary (in a real implementation, we would use the LLM)
            summary = self._summarize_conversation(conversation)
            
            self.conversation_summaries[conversation_id] = summary
            
            return f"Summarized conversation: {conversation_id}"
        
        elif action == "organize":
            # Organize a conversation into topics
            conversation_match = re.search(r"conversation:\s*(.+)", task.description, re.DOTALL | re.IGNORECASE)
            
            if not conversation_match:
                return "Invalid conversation organization format"
            
            conversation = conversation_match.group(1).strip()
            
            # Organize the conversation (in a real implementation, we would use the LLM)
            organization = self._organize_information(conversation)
            
            return organization
        
        elif action == "extract":
            # Extract entities or information from a conversation
            conversation_match = re.search(r"conversation:\s*(.+)", task.description, re.DOTALL | re.IGNORECASE)
            
            if not conversation_match:
                return "Invalid entity extraction format"
            
            conversation = conversation_match.group(1).strip()
            
            # Extract entities (in a real implementation, we would use the LLM)
            entities = self._extract_entities(conversation)
            
            return entities
        
        else:
            return f"Unknown conversation organization action: {action}"
    
    def _handle_fixed_task(self, task: Task) -> str:
        """
        Handle a fixed task.
        
        Args:
            task: The task to handle
            
        Returns:
            The result of the task
        """
        # Extract the task type from the description
        task_type_match = re.search(r"^(\w+):", task.description)
        if not task_type_match:
            return "Invalid fixed task format"
        
        task_type = task_type_match.group(1).lower()
        
        # Check if we have a handler for this task type
        if task_type not in self.fixed_task_handlers:
            return f"Unknown fixed task type: {task_type}"
        
        # Extract the content
        content_match = re.search(r"content:\s*(.+)", task.description, re.DOTALL | re.IGNORECASE)
        if not content_match:
            return "Invalid fixed task format: missing content"
        
        content = content_match.group(1).strip()
        
        # Execute the handler
        handler = self.fixed_task_handlers[task_type]
        result = handler(content)
        
        return result
    
    def _handle_custom_task(self, task: Task) -> str:
        """
        Handle a custom task.
        
        Args:
            task: The task to handle
            
        Returns:
            The result of the task
        """
        # In a real implementation, we would use the LLM to handle custom tasks
        return f"Handled custom task: {task.description}"
    
    def _summarize_conversation(self, conversation: str) -> str:
        """
        Summarize a conversation.
        
        Args:
            conversation: The conversation to summarize
            
        Returns:
            A summary of the conversation
        """
        # In a real implementation, we would use the LLM to generate a summary
        return f"Summary of conversation ({len(conversation)} chars)"
    
    def _extract_entities(self, content: str) -> str:
        """
        Extract entities from content.
        
        Args:
            content: The content to extract entities from
            
        Returns:
            Extracted entities
        """
        # In a real implementation, we would use the LLM to extract entities
        return f"Extracted entities from content ({len(content)} chars)"
    
    def _generate_prompt(self, description: str) -> str:
        """
        Generate a prompt based on a description.
        
        Args:
            description: Description of the prompt to generate
            
        Returns:
            The generated prompt
        """
        # In a real implementation, we would use the LLM to generate a prompt
        return f"Generated prompt for: {description}"
    
    def _filter_content(self, content: str) -> str:
        """
        Filter content based on rules.
        
        Args:
            content: The content to filter
            
        Returns:
            The filtered content
        """
        # In a real implementation, we would apply filtering rules
        return f"Filtered content ({len(content)} chars)"
    
    def _organize_information(self, content: str) -> str:
        """
        Organize information into a structured format.
        
        Args:
            content: The content to organize
            
        Returns:
            The organized information
        """
        # In a real implementation, we would use the LLM to organize information
        return f"Organized information ({len(content)} chars)"
    
    def _create_rule(self, description: str) -> str:
        """
        Create a rule based on a description.
        
        Args:
            description: Description of the rule to create
            
        Returns:
            The created rule
        """
        # In a real implementation, we would use the LLM to create a rule
        return f"Created rule for: {description}"
    
    def run(self, max_iterations: int = 10) -> List[Task]:
        """
        Run the BabyAGI system for a number of iterations.
        
        Args:
            max_iterations: Maximum number of iterations to run
            
        Returns:
            List of completed tasks
        """
        completed_tasks = []
        
        for _ in range(max_iterations):
            # Get the next task
            task = self.get_next_task()
            
            if not task:
                logger.info("No pending tasks")
                break
            
            # Execute the task
            self.execute_task(task)
            
            if task.status == TaskStatus.COMPLETED:
                completed_tasks.append(task)
        
        return completed_tasks
    
    def process_user_input(self, user_input: str, conversation_id: str) -> str:
        """
        Process user input and manage related tasks.
        
        Args:
            user_input: The user's input
            conversation_id: ID of the conversation
            
        Returns:
            A response or acknowledgment
        """
        # Create tasks based on the user input
        tasks_created = []
        
        # Check if this is a prompt management request
        if re.search(r"create prompt|update prompt|delete prompt|list prompts", user_input, re.IGNORECASE):
            task_id = self.add_task(
                description=f"create: {conversation_id}\ntemplate: {user_input}",
                task_type=TaskType.PROMPT_MANAGEMENT,
                priority=TaskPriority.HIGH
            )
            tasks_created.append(task_id)
        
        # Check if this is a conversation organization request
        elif re.search(r"summarize|organize|extract", user_input, re.IGNORECASE):
            task_id = self.add_task(
                description=f"summarize: {conversation_id}\nconversation: {user_input}",
                task_type=TaskType.CONVERSATION_ORGANIZATION,
                priority=TaskPriority.MEDIUM
            )
            tasks_created.append(task_id)
        
        # Check if this is a fixed task request
        elif re.search(r"filter|rule|organize", user_input, re.IGNORECASE):
            task_id = self.add_task(
                description=f"filter: {conversation_id}\ncontent: {user_input}",
                task_type=TaskType.FIXED_TASK,
                priority=TaskPriority.MEDIUM
            )
            tasks_created.append(task_id)
        
        # Otherwise, create a custom task
        else:
            task_id = self.add_task(
                description=user_input,
                task_type=TaskType.CUSTOM,
                priority=TaskPriority.LOW,
                metadata={"conversation_id": conversation_id}
            )
            tasks_created.append(task_id)
        
        # Run the system for a few iterations
        completed_tasks = self.run(max_iterations=5)
        
        # Generate a response based on the completed tasks
        if completed_tasks:
            response_parts = ["I've processed your request:"]
            
            for task in completed_tasks:
                response_parts.append(f"- {task.description}: {task.result}")
            
            return "\n".join(response_parts)
        else:
            return "I've queued your request for processing."
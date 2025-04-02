"""
Enhanced Reasoning Module for the LangChain Router.

This module provides enhanced reasoning capabilities for the LangChain Router,
including structured thinking, multi-step reasoning, and thought tree generation.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pydantic import BaseModel, Field
import re
import json
import logging
import uuid
import time
from dataclasses import dataclass, field

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseMemory

logger = logging.getLogger(__name__)


class ThoughtStep(BaseModel):
    """Represents a step in a reasoning process."""
    thought: str = Field(description="Thought about the problem or situation")
    action: Optional[str] = Field(None, description="Action to take")
    action_input: Optional[str] = Field(None, description="Input for the action")
    observation: Optional[str] = Field(None, description="Observation after the action")


class ReasoningChain:
    """
    Enhanced reasoning chain that implements structured thinking processes.
    
    This chain supports:
    1. Chain of Thought (CoT) - Step-by-step reasoning
    2. ReAct - Reasoning and Acting
    3. Tree of Thoughts (ToT) - Exploring multiple reasoning paths
    """
    
    def __init__(self, llm):
        """
        Initialize the reasoning chain.
        
        Args:
            llm: Language model to use
        """
        self.llm = llm
        self.history = []
    
    def _format_thought_trace(self) -> str:
        """
        Format the thought trace as a string.
        
        Returns:
            Formatted thought trace
        """
        formatted = ""
        for i, step in enumerate(self.history):
            formatted += f"Step {i+1}:\n"
            formatted += f"Thought: {step.thought}\n"
            
            if step.action:
                formatted += f"Action: {step.action}\n"
                formatted += f"Action Input: {step.action_input}\n"
                
            if step.observation:
                formatted += f"Observation: {step.observation}\n"
                
            formatted += "\n"
            
        return formatted
    
    def add_thought(
        self,
        thought: str,
        action: Optional[str] = None,
        action_input: Optional[str] = None,
        observation: Optional[str] = None
    ) -> None:
        """
        Add a thought step to the history.
        
        Args:
            thought: The thought
            action: Optional action
            action_input: Optional action input
            observation: Optional observation
        """
        step = ThoughtStep(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation
        )
        self.history.append(step)
    
    def chain_of_thought(self, question: str, max_steps: int = 5) -> str:
        """
        Implement chain of thought reasoning.
        
        Args:
            question: The question to reason about
            max_steps: Maximum number of reasoning steps
            
        Returns:
            The final answer
        """
        # Use chain of thought prompt template
        cot_template = """
        Question: {question}
        
        Please think through this question step by step, analyze possible solutions, and provide a final answer.
        
        Reasoning process:
        """
        
        # Build the prompt
        prompt = PromptTemplate.from_template(cot_template)
        
        # Get the response
        response = self.llm.invoke(prompt.format(question=question))
        
        # Record the thinking process
        if hasattr(response, 'content'):
            thought = response.content
        else:
            thought = str(response)
            
        self.add_thought(thought=thought)
        
        return thought
    
    def react(self, question: str, tools: List[Callable], max_steps: int = 5) -> str:
        """
        Implement ReAct (Reasoning-Acting-Observing) reasoning.
        
        Args:
            question: The question to reason about
            tools: List of tools to use
            max_steps: Maximum number of reasoning steps
            
        Returns:
            The final answer
        """
        # Reset history
        self.history = []
        
        # Build tool descriptions
        tool_descriptions = "\n".join([
            f"{i+1}. {tool.name}: {tool.description}" 
            for i, tool in enumerate(tools)
        ])
        
        # ReAct prompt template
        react_template = """
        Question: {question}
        
        You have the following tools available:
        {tool_descriptions}
        
        Please answer using this format:
        Thought: [your thought about the problem]
        Action: [tool name]
        Action Input: [tool input]
        Observation: [tool output]
        ...
        Thought: [your thought about the observation]
        Action: [next action]
        ...
        Thought: [final thought]
        Final Answer: [answer to the question]
        
        Begin:
        """
        
        # Tool dictionary for easy lookup
        tool_dict = {tool.name: tool for tool in tools}
        
        current_prompt = react_template.format(
            question=question,
            tool_descriptions=tool_descriptions
        )
        
        for step in range(max_steps):
            # Get model response
            response = self.llm.invoke(current_prompt)
            
            # Extract response content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Extract thought, action, and action input
            thought_match = re.search(r"Thought:(.*?)(?:Action:|Final Answer:|$)", response_text, re.DOTALL)
            action_match = re.search(r"Action:(.*?)(?:Action Input:|$)", response_text, re.DOTALL)
            action_input_match = re.search(r"Action Input:(.*?)(?:Observation:|$)", response_text, re.DOTALL)
            
            # Check for final answer
            final_answer_match = re.search(r"Final Answer:(.*?)$", response_text, re.DOTALL)
            if final_answer_match:
                final_thought = thought_match.group(1).strip() if thought_match else ""
                final_answer = final_answer_match.group(1).strip()
                
                self.add_thought(thought=final_thought)
                return final_answer
            
            # Extract thought content
            thought = thought_match.group(1).strip() if thought_match else ""
            
            # Check for action
            if action_match:
                action = action_match.group(1).strip()
                action_input = action_input_match.group(1).strip() if action_input_match else ""
                
                # Execute tool call
                if action in tool_dict:
                    try:
                        observation = tool_dict[action].run(action_input)
                    except Exception as e:
                        observation = f"Error: {str(e)}"
                else:
                    observation = f"Error: Tool '{action}' not found"
                
                # Record step
                self.add_thought(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation
                )
                
                # Update prompt
                current_prompt += f"\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}"
            else:
                # No action, record thought
                self.add_thought(thought=thought)
                break
        
        # If max steps reached, generate final answer
        final_prompt = current_prompt + "\nThought: I have gathered enough information and can now provide a final answer.\nFinal Answer:"
        final_response = self.llm.invoke(final_prompt)
        
        # Extract response content
        if hasattr(final_response, 'content'):
            final_response_text = final_response.content
        else:
            final_response_text = str(final_response)
        
        final_answer_match = re.search(r"Final Answer:(.*?)$", final_response_text, re.DOTALL)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        else:
            return final_response_text.strip()
    
    def tree_of_thoughts(self, question: str, num_branches: int = 3, max_depth: int = 3) -> str:
        """
        Implement Tree of Thoughts reasoning.
        
        Args:
            question: The question to reason about
            num_branches: Number of branches at each node
            max_depth: Maximum depth of the tree
            
        Returns:
            The final answer
        """
        # ToT prompt template
        tot_template = """
        Question: {question}
        
        Please generate {num_branches} different thinking paths for this question. Each path should represent a different approach or perspective.
        
        Existing paths:
        {existing_paths}
        
        Current path ({path_idx}):
        Current depth: {depth}
        Previous thought: {prev_thought}
        
        Please continue this thinking path with the next step:
        """
        
        # Initialize thought tree
        thought_tree = [{"thought": "Initial thought", "children": []} for _ in range(num_branches)]
        
        # Generate initial thoughts for each branch
        for i in range(num_branches):
            initial_prompt = f"""
            Question: {question}
            
            Please provide a unique approach or perspective for solving this question. This is path {i+1} and should be different from other paths.
            """
            response = self.llm.invoke(initial_prompt)
            
            # Extract response content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
                
            thought_tree[i]["thought"] = response_text.strip()
        
        # Recursive function to expand the thought tree
        def expand_branch(branch, depth, path_idx):
            if depth >= max_depth:
                return
                
            children = []
            existing_paths = "\n".join([
                f"Path {j+1}: {node['thought'][:100]}..." 
                for j, node in enumerate(thought_tree)
            ])
            
            # Generate child thoughts
            prompt = tot_template.format(
                question=question,
                num_branches=num_branches,
                existing_paths=existing_paths,
                path_idx=path_idx+1,
                depth=depth+1,
                prev_thought=branch["thought"]
            )
            
            response = self.llm.invoke(prompt)
            
            # Extract response content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Create child node
            child = {"thought": response_text.strip(), "children": []}
            children.append(child)
            
            # Recursively expand child nodes
            for j, child_node in enumerate(children):
                expand_branch(child_node, depth + 1, path_idx)
                
            branch["children"] = children
        
        # Expand each initial branch
        for i, branch in enumerate(thought_tree):
            expand_branch(branch, 0, i)
        
        # Evaluate all paths and select the best one
        evaluation_prompt = f"""
        Question: {question}
        
        Here are several different thinking paths:
        
        {json.dumps(thought_tree, indent=2, ensure_ascii=False)}
        
        Please evaluate these thinking paths and select the best one to answer the question. Provide your final answer.
        """
        
        final_response = self.llm.invoke(evaluation_prompt)
        
        # Extract response content
        if hasattr(final_response, 'content'):
            final_response_text = final_response.content
        else:
            final_response_text = str(final_response)
        
        # Record the entire reasoning process
        self.add_thought(thought=f"Thought tree exploration:\n{json.dumps(thought_tree, indent=2, ensure_ascii=False)}")
        self.add_thought(thought=f"Final reasoning:\n{final_response_text}")
        
        return final_response_text


class CognitiveCore:
    """
    Cognitive core for enhanced reasoning.
    
    This core combines the strengths of different models and reasoning approaches
    to provide high-quality thinking processes and logical reasoning.
    """
    
    def __init__(self, llm):
        """
        Initialize the cognitive core.
        
        Args:
            llm: Language model to use
        """
        self.llm = llm
        self.thought_stream = []
        self.context_buffer = []
        self.decision_history = []
    
    def think(self, query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Core thinking process.
        
        Args:
            query: The query to think about
            context: Optional context
            
        Returns:
            Dictionary containing the thought process and conclusion
        """
        # Initial thinking prompt
        thinking_prompt = f"""
        Think about the following question: {query}
        
        Follow these thinking steps:
        1. Problem decomposition: Break down the problem into core components
        2. Premise identification: Identify implicit assumptions and known conditions
        3. Relevant knowledge: Determine the knowledge domains needed to solve the problem
        4. Solution construction: Organize solution steps in logical order
        5. Reflective evaluation: Evaluate the strengths and weaknesses of the solution
        
        Provide your thinking process in a structured format.
        """
        
        # Use the LLM for deep thinking
        thoughts_response = self.llm.invoke(thinking_prompt)
        
        # Extract response content
        if hasattr(thoughts_response, 'content'):
            thoughts = thoughts_response.content
        else:
            thoughts = str(thoughts_response)
        
        # Record thought stream
        self.thought_stream.append({
            "stage": "initial_thinking",
            "content": thoughts
        })
        
        # Build decision information
        decision = {
            "query": query,
            "thought_process": thoughts,
            "conclusion": self._extract_conclusion(thoughts)
        }
        
        self.decision_history.append(decision)
        return decision
    
    def _extract_conclusion(self, thought_text: str) -> str:
        """
        Extract the conclusion from thought text.
        
        Args:
            thought_text: The thought text
            
        Returns:
            The extracted conclusion
        """
        # Try to identify conclusion section
        conclusion_patterns = [
            r"Conclusion[:：](.*?)(?=\n\n|\Z)",
            r"Summary[:：](.*?)(?=\n\n|\Z)",
            r"Answer[:：](.*?)(?=\n\n|\Z)"
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, thought_text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no explicit conclusion marker, return the last paragraph
        paragraphs = thought_text.split("\n\n")
        if paragraphs:
            return paragraphs[-1].strip()
        
        return "Unable to extract conclusion"


class ReflectionSystem:
    """
    System for reflecting on interactions and improving responses.
    
    This system:
    1. Records and analyzes interactions
    2. Identifies patterns and preferences
    3. Improves future responses based on past interactions
    """
    
    def __init__(self, llm):
        """
        Initialize the reflection system.
        
        Args:
            llm: Language model to use
        """
        self.llm = llm
        self.interactions = []
        self.user_identities = {}
        self.reflections = []
    
    def add_user_identity(self, user_id: str, preferences: Dict[str, Any] = None) -> None:
        """
        Add or update user identity.
        
        Args:
            user_id: User ID
            preferences: User preferences
        """
        self.user_identities[user_id] = {
            "preferences": preferences or {},
            "interaction_count": 0,
            "last_interaction": None
        }
    
    def get_user_identity(self, user_id: str) -> Dict[str, Any]:
        """
        Get user identity.
        
        Args:
            user_id: User ID
            
        Returns:
            User identity information
        """
        if user_id not in self.user_identities:
            self.add_user_identity(user_id)
        
        return self.user_identities[user_id]
    
    def reflect_on_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on an interaction.
        
        Args:
            interaction: The interaction to reflect on
            
        Returns:
            Reflection results
        """
        # Record the interaction
        self.interactions.append(interaction)
        
        # Update user identity
        user_id = interaction.get("user_id", "anonymous")
        user_identity = self.get_user_identity(user_id)
        user_identity["interaction_count"] += 1
        user_identity["last_interaction"] = time.time()
        
        # Generate reflection
        reflection_prompt = f"""
        Please reflect on the following interaction:
        
        Query: {interaction.get('query', '')}
        Response: {interaction.get('response', '')}
        Context: {json.dumps(interaction.get('context', {}), indent=2)}
        
        Consider:
        1. Was the response helpful and accurate?
        2. What could be improved in the response?
        3. What patterns or preferences can be identified from this interaction?
        4. How can future responses be improved based on this interaction?
        
        Provide your reflections in a structured format.
        """
        
        reflection_response = self.llm.invoke(reflection_prompt)
        
        # Extract response content
        if hasattr(reflection_response, 'content'):
            reflection_text = reflection_response.content
        else:
            reflection_text = str(reflection_response)
        
        # Parse reflection
        reflection = {
            "interaction_id": len(self.interactions) - 1,
            "user_id": user_id,
            "timestamp": time.time(),
            "reflections": self._parse_reflection(reflection_text)
        }
        
        self.reflections.append(reflection)
        return reflection
    
    def _parse_reflection(self, reflection_text: str) -> Dict[str, Any]:
        """
        Parse reflection text into structured format.
        
        Args:
            reflection_text: The reflection text
            
        Returns:
            Structured reflection
        """
        # Try to parse structured sections
        sections = {
            "helpfulness": self._extract_section(reflection_text, ["helpfulness", "was the response helpful"]),
            "improvements": self._extract_section(reflection_text, ["improvements", "what could be improved"]),
            "patterns": self._extract_section(reflection_text, ["patterns", "preferences"]),
            "future_recommendations": self._extract_section(reflection_text, ["future", "recommendations"])
        }
        
        return sections
    
    def _extract_section(self, text: str, keywords: List[str]) -> str:
        """
        Extract a section from text based on keywords.
        
        Args:
            text: The text to extract from
            keywords: Keywords to look for
            
        Returns:
            Extracted section
        """
        for keyword in keywords:
            pattern = rf"{keyword}.*?:(.*?)(?=\n\n|\n[A-Z]|\Z)"
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""


class EnhancedReasoningFactory:
    """Factory for creating enhanced reasoning components."""
    
    @staticmethod
    def create_reasoning_chain(llm) -> ReasoningChain:
        """
        Create a reasoning chain.
        
        Args:
            llm: Language model to use
            
        Returns:
            A reasoning chain
        """
        return ReasoningChain(llm)
    
    @staticmethod
    def create_cognitive_core(llm) -> CognitiveCore:
        """
        Create a cognitive core.
        
        Args:
            llm: Language model to use
            
        Returns:
            A cognitive core
        """
        return CognitiveCore(llm)
    
    @staticmethod
    def create_reflection_system(llm) -> ReflectionSystem:
        """
        Create a reflection system.
        
        Args:
            llm: Language model to use
            
        Returns:
            A reflection system
        """
        return ReflectionSystem(llm)
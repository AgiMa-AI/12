"""
Main agent class for the Claude Agent.

This module provides the main agent class for the Claude Agent,
including message processing, tool execution, and command handling.
"""

import os
import sys
import json
import time
import logging
import traceback
import threading
from typing import Dict, List, Any, Optional, Union

from openhands.claude_agent.config import AgentConfig
from openhands.claude_agent.memory import AgentMemory
from openhands.claude_agent.tools import ToolRegistry
from openhands.claude_agent.client import ModelClient
from openhands.claude_agent.dummy_client import DummyClient

logger = logging.getLogger("claude_agent")

class ClaudeAgent:
    """
    Main Claude agent class.
    """
    
    def __init__(self, config_file: str = "agent_config.json"):
        """
        Initialize the Claude agent.
        
        Args:
            config_file: Path to the configuration file
        """
        # Initialize configuration
        self.config = AgentConfig(config_file)
        
        # Ensure API key is set
        if not self.config.api_key:
            logger.error("API key not set. Please set it in config.json or via ANTHROPIC_API_KEY environment variable.")
            raise ValueError("API key not set")
        
        # Initialize memory system
        self.memory = AgentMemory(self.config.memory_file)
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        
        # Initialize model client
        # Initialize client
        if self.config.api_key == "dummy-key" or (self.config.use_local_model and "dummy-model" in self.config.local_model_path):
            self.claude = DummyClient(self.config)
        else:
            self.claude = ModelClient(self.config)
        
        # Agent state
        self.running = False
        self.last_response = None
        self.conversation_context = []
        
        # Initialize GUI dialog (will be set later if needed)
        self.gui_dialog = None
        
        logger.info("Claude agent initialized")
    
    def start(self) -> None:
        """Start the agent."""
        self.running = True
        logger.info("Claude agent started")
        
        print(f"=== Claude Agent System v1.0 ===")
        print(f"Using model: {self.config.model}")
        print(f"System prompt: I am your versatile AI assistant, capable of performing various tasks")
        print(f"Type 'exit' to quit, 'help' for help")
        
        # Load system prompt
        self.conversation_context = [
            {
                "role": "system",
                "content": "You are Claude Agent System, a powerful AI assistant capable of performing various tasks, "
                           "including file processing, system operations, code execution, and data analysis. "
                           "You can use various tools to help users complete tasks. Always respond in the same language as the user's query. "
                           "As a long-term partner to the user, remember their preferences and common operations."
            }
        ]
        
        # Main loop
        try:
            while self.running:
                # Get user input
                user_input = input("\n> ")
                
                # Process exit command
                if user_input.lower() in ["exit", "quit"]:
                    self.running = False
                    continue
                
                # Process help command
                if user_input.lower() in ["help"]:
                    self._show_help()
                    continue
                
                # Process special commands
                if user_input.startswith("/") or user_input.startswith("@"):
                    self._handle_command(user_input)
                    continue
                
                # Process normal input
                response = self.process_input(user_input)
                
                # Save conversation history
                self.memory.add_conversation("user", user_input)
                if hasattr(response, "content"):
                    self.memory.add_conversation("assistant", response.content)
                
                # Save memory
                self.memory.save_memory()
        except KeyboardInterrupt:
            print("\nProgram interrupted")
        finally:
            print("Saving memory and exiting...")
            self.memory.save_memory()
            print("Goodbye!")
    
    def process_input(self, user_input: str) -> Any:
        """
        Process user input.
        
        Args:
            user_input: User input
            
        Returns:
            Claude API response
        """
        # Add user message to context
        self.conversation_context.append({
            "role": "user",
            "content": user_input
        })
        
        # Limit context length
        if len(self.conversation_context) > 20:
            # Keep system prompt and recent messages
            self.conversation_context = [
                self.conversation_context[0],  # System prompt
                *self.conversation_context[-19:]  # Recent messages
            ]
        
        # Send message to Claude
        try:
            logger.info(f"Sending message to Claude: {user_input[:100]}...")
            response = self.claude.send_message(
                messages=self.conversation_context,
                tools=self.tool_registry.get_all_tools()
            )
            
            # Process response
            if hasattr(response, "content"):
                print(f"\n{response.content}")
                
                # Add assistant reply to context
                self.conversation_context.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Process tool calls
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tool_call in response.tool_calls:
                        self._process_tool_call(tool_call)
                
                self.last_response = response
                return response
            else:
                error_message = "Could not get response content"
                if hasattr(response, "error"):
                    error_message = f"Error: {response.error}"
                
                print(f"\n{error_message}")
                return {"error": error_message}
        
        except Exception as e:
            error_message = f"Error processing message: {str(e)}"
            logger.error(error_message)
            print(f"\n{error_message}")
            return {"error": error_message}
    
    def _process_tool_call(self, tool_call: Any) -> None:
        """
        Process a tool call.
        
        Args:
            tool_call: Tool call object
        """
        tool_name = tool_call.name
        tool_input = tool_call.input
        
        print(f"\n[Executing tool: {tool_name}]")
        print(f"Input parameters: {json.dumps(tool_input, ensure_ascii=False)}")
        
        # Execute tool
        result = self.tool_registry.execute_tool(
            tool_name, 
            tool_input, 
            timeout=self.config.tool_timeout
        )
        
        print(f"Tool execution result: {json.dumps(result, ensure_ascii=False, default=str)[:500]}")
        
        # Add result to context
        tool_message = {
            "role": "user",
            "content": f"Tool {tool_name} execution result: {json.dumps(result, ensure_ascii=False, default=str)}"
        }
        self.conversation_context.append(tool_message)
        
        # Query Claude again to process tool result
        response = self.claude.send_message(
            messages=self.conversation_context,
            tools=self.tool_registry.get_all_tools()
        )
        
        if hasattr(response, "content"):
            print(f"\n{response.content}")
            
            # Add assistant reply to context
            self.conversation_context.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Recursively process any subsequent tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    self._process_tool_call(tool_call)
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
=== Claude Agent System Help ===

Basic Commands:
  exit, quit - Exit the program
  help - Show this help information

Special Commands:
  /tools (@tools, @view) - List all available tools
  /memory (@memory, @history) - View memory status
  /clear (@clear, @reset) - Clear current session
  /config (@config, @settings) - View or modify configuration
  /save (@save, @store) - Manually save memory
  /system (@system, @prompt) <prompt> - Set system prompt
  /execute (@execute, @run) <tool_name> <param_json> - Directly execute tool
  /gui (@launch, @interface, @dialog) - Launch GUI dialog window

Shortcuts:
  Ctrl+Alt+C - Global shortcut to launch GUI dialog window

Examples:
  /tools
  @view
  /execute read_file {"file_path": "example.txt"}
  @execute read_file {"file_path": "example.txt"}
  /system You are an AI assistant focused on data analysis
  @system You are an AI assistant focused on data analysis
  /config model claude-3-opus-20240229
  @settings model claude-3-opus-20240229
        """
        print(help_text)
    
    def _handle_command(self, command: str) -> None:
        """
        Handle special commands.
        
        Args:
            command: Command string
        """
        parts = command.split(" ", 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Chinese command mapping
        chinese_commands = {
            "@工具": "/tools",
            "@查看": "/tools",
            "@记忆": "/memory",
            "@历史": "/memory",
            "@清除": "/clear",
            "@重置": "/clear",
            "@配置": "/config",
            "@设置": "/config",
            "@保存": "/save",
            "@存储": "/save",
            "@系统": "/system",
            "@提示": "/system",
            "@执行": "/execute",
            "@运行": "/execute",
            "@启动": "/gui",
            "@界面": "/gui",
            "@对话": "/gui",
            "@帮助": "/help"
        }
        
        # Convert Chinese commands to English commands
        if cmd in chinese_commands:
            cmd = chinese_commands[cmd]
        
        if cmd == "/tools":
            self._list_tools()
        elif cmd == "/memory":
            self._show_memory()
        elif cmd == "/clear":
            self._clear_conversation()
        elif cmd == "/config":
            self._handle_config(args)
        elif cmd == "/save":
            self._save_memory()
        elif cmd == "/system":
            self._set_system_prompt(args)
        elif cmd == "/execute":
            self._direct_execute_tool(args)
        elif cmd == "/gui" or cmd == "/dialog":
            self._launch_gui()
        elif cmd == "/help":
            self._show_help()
        else:
            print(f"Unknown command: {cmd}. Type 'help' or '@help' for help.")
    
    def _list_tools(self) -> None:
        """List all available tools."""
        print("\n=== Available Tools ===")
        for i, tool in enumerate(self.tool_registry.get_all_tools(), 1):
            print(f"{i}. {tool['name']} - {tool['description']}")
    
    def _show_memory(self) -> None:
        """Show memory status."""
        print("\n=== Memory Status ===")
        print(f"Conversation history: {len(self.memory.conversation_history)} records")
        print(f"Long-term memory: {len(self.memory.long_term_memory)} entries")
        
        print("\nRecent conversations:")
        recent = self.memory.get_recent_conversations(5)
        for i, entry in enumerate(recent, 1):
            role = "User" if entry["role"] == "user" else "Claude"
            content = entry["content"]
            if len(content) > 100:
                content = content[:97] + "..."
            print(f"{i}. [{role}] {content}")
    
    def _clear_conversation(self) -> None:
        """Clear current session."""
        # Keep system prompt
        system_prompt = self.conversation_context[0] if len(self.conversation_context) > 0 else {
            "role": "system",
            "content": "You are Claude Agent System, a powerful AI assistant."
        }
        
        self.conversation_context = [system_prompt]
        print("Current session cleared.")
    
    def _handle_config(self, args: str) -> None:
        """
        Handle configuration command.
        
        Args:
            args: Command arguments
        """
        if not args:
            # Show current configuration
            print("\n=== Current Configuration ===")
            print(f"Model: {self.config.model}")
            print(f"Max tokens: {self.config.max_tokens}")
            print(f"Temperature: {self.config.temperature}")
            print(f"Thinking mode: {'enabled' if self.config.thinking_mode else 'disabled'}")
            print(f"Debug mode: {'enabled' if self.config.debug_mode else 'disabled'}")
            print(f"Working directory: {self.config.working_directory}")
            return
        
        # Parse configuration parameters
        try:
            key, value = args.split(" ", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "model":
                self.config.model = value
                print(f"Model set to: {value}")
            elif key == "max_tokens":
                self.config.max_tokens = int(value)
                print(f"Max tokens set to: {value}")
            elif key == "temperature":
                self.config.temperature = float(value)
                print(f"Temperature set to: {value}")
            elif key == "thinking_mode":
                self.config.thinking_mode = value.lower() in ["true", "1", "yes", "on", "enabled"]
                print(f"Thinking mode {'enabled' if self.config.thinking_mode else 'disabled'}")
            elif key == "debug_mode":
                self.config.debug_mode = value.lower() in ["true", "1", "yes", "on", "enabled"]
                print(f"Debug mode {'enabled' if self.config.debug_mode else 'disabled'}")
            else:
                print(f"Unknown configuration option: {key}")
                return
            
            # Save configuration
            self.config.save_config()
            
        except ValueError:
            print("Invalid configuration format. Please use '/config key value' format.")
    
    def _save_memory(self) -> None:
        """Manually save memory."""
        if self.memory.save_memory():
            print("Memory saved.")
        else:
            print("Failed to save memory.")
    
    def _set_system_prompt(self, prompt: str) -> None:
        """
        Set system prompt.
        
        Args:
            prompt: System prompt
        """
        if not prompt:
            print("Please provide a system prompt.")
            return
        
        # Update system prompt
        if len(self.conversation_context) > 0 and self.conversation_context[0]["role"] == "system":
            self.conversation_context[0]["content"] = prompt
        else:
            self.conversation_context.insert(0, {
                "role": "system",
                "content": prompt
            })
        
        print("System prompt updated.")
    
    def _direct_execute_tool(self, args: str) -> None:
        """
        Directly execute tool.
        
        Args:
            args: Command arguments
        """
        try:
            # Parse tool name and parameters
            parts = args.split(" ", 1)
            if len(parts) < 2:
                print("Missing tool parameters. Please use '/execute tool_name {\"param\":\"value\"}' format.")
                return
            
            tool_name = parts[0].strip()
            params_json = parts[1].strip()
            
            # Parse JSON parameters
            params = json.loads(params_json)
            
            # Execute tool
            print(f"Executing tool: {tool_name}")
            result = self.tool_registry.execute_tool(
                tool_name, 
                params, 
                timeout=self.config.tool_timeout
            )
            
            # Show result
            print("\nExecution result:")
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            
        except json.JSONDecodeError:
            print("Invalid JSON parameter format.")
        except Exception as e:
            print(f"Error executing tool: {str(e)}")
    
    def _launch_gui(self) -> None:
        """Launch GUI dialog window."""
        # This is a placeholder - GUI implementation would be added here
        print("GUI dialog not implemented in this version.")
    
    def integrate_with_langchain_router(self) -> None:
        """Integrate with LangChain Router."""
        try:
            from openhands import get_langchain_router
            
            # Get the LangChain Router integration
            integration = get_langchain_router()
            
            # Register additional tools for LangChain Router integration
            self.tool_registry.register_tool(
                name="langchain_route",
                description="Route a request through the LangChain Router",
                func=lambda query, model_name=None, task_type=None: integration.route_request(
                    user_input=query,
                    conversation_id=str(time.time()),
                    model_name=model_name,
                    task_type=task_type
                ),
                schema={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to route"
                        },
                        "model_name": {
                            "type": "string",
                            "description": "Model to use (optional)"
                        },
                        "task_type": {
                            "type": "string",
                            "description": "Task type (optional)"
                        }
                    }
                }
            )
            
            logger.info("Integrated with LangChain Router")
        except ImportError:
            logger.warning("Could not integrate with LangChain Router: module not found")
        except Exception as e:
            logger.error(f"Error integrating with LangChain Router: {e}")
    
    def integrate_vector_tools(self) -> None:
        """Integrate vector database tools."""
        try:
            from openhands.claude_agent.vector_tools import get_vector_tools
            
            # Get vector tools
            vector_tools = get_vector_tools()
            
            # Register vector tools
            for tool in vector_tools:
                self.tool_registry.register_tool(
                    name=tool.name,
                    description=tool.description,
                    func=tool.func,
                    schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query for vector search"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Path to file for vector database"
                            },
                            "db_name": {
                                "type": "string",
                                "description": "Name of vector database"
                            }
                        }
                    }
                )
            
            logger.info("Integrated vector database tools")
        except ImportError:
            logger.warning("Could not integrate vector tools: required modules not found")
        except Exception as e:
            logger.error(f"Error integrating vector tools: {e}")
    
    def integrate_desktop_tools(self) -> None:
        """Integrate desktop control tools."""
        try:
            from openhands.claude_agent.desktop_tools import get_desktop_tools
            
            # Get desktop tools
            desktop_tools = get_desktop_tools()
            
            # Register desktop tools
            for tool in desktop_tools:
                self.tool_registry.register_tool(
                    name=tool.name,
                    description=tool.description,
                    func=tool.func,
                    schema={
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "integer",
                                "description": "X coordinate"
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y coordinate"
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to type"
                            },
                            "filename": {
                                "type": "string",
                                "description": "Filename for screenshot"
                            }
                        }
                    }
                )
            
            logger.info("Integrated desktop control tools")
        except ImportError:
            logger.warning("Could not integrate desktop tools: required modules not found")
        except Exception as e:
            logger.error(f"Error integrating desktop tools: {e}")
    
    def integrate_enhanced_reasoning(self) -> None:
        """Integrate enhanced reasoning capabilities."""
        try:
            from openhands.langchain_router.enhanced_reasoning import EnhancedReasoningFactory
            
            # Register enhanced reasoning tool
            self.tool_registry.register_tool(
                name="enhanced_reasoning",
                description="Use enhanced reasoning capabilities",
                func=lambda query, reasoning_type="chain_of_thought": self._enhanced_reasoning(query, reasoning_type),
                schema={
                    "type": "object",
                    "required": ["query", "reasoning_type"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to reason about"
                        },
                        "reasoning_type": {
                            "type": "string",
                            "description": "Type of reasoning (chain_of_thought, react, tree_of_thoughts)"
                        }
                    }
                }
            )
            
            logger.info("Integrated enhanced reasoning capabilities")
        except ImportError:
            logger.warning("Could not integrate enhanced reasoning: required modules not found")
        except Exception as e:
            logger.error(f"Error integrating enhanced reasoning: {e}")
    
    def _enhanced_reasoning(self, query: str, reasoning_type: str) -> Dict[str, Any]:
        """
        Use enhanced reasoning capabilities.
        
        Args:
            query: Query to reason about
            reasoning_type: Type of reasoning
            
        Returns:
            Dictionary containing result or error
        """
        try:
            from openhands import get_langchain_router
            from openhands.langchain_router.enhanced_reasoning import EnhancedReasoningFactory
            
            # Get the LangChain Router integration
            integration = get_langchain_router()
            
            # Get the language model
            llm = None
            router = integration.router
            if router and hasattr(router, "chain") and hasattr(router.chain, "llm"):
                llm = router.chain.llm
            else:
                # If that fails, try to get it from the optimized router
                if hasattr(integration, "optimized_router") and integration.optimized_router:
                    if hasattr(integration.optimized_router, "model_optimizer"):
                        llm = integration.optimized_router.model_optimizer.llm
            
            if not llm:
                return {"error": "Could not get language model"}
            
            # Create reasoning chain
            reasoning_chain = EnhancedReasoningFactory.create_reasoning_chain(llm)
            
            # Use appropriate reasoning type
            if reasoning_type == "chain_of_thought":
                result = reasoning_chain.chain_of_thought(query)
            elif reasoning_type == "tree_of_thoughts":
                result = reasoning_chain.tree_of_thoughts(query, num_branches=3, max_depth=3)
            elif reasoning_type == "react":
                # For ReAct, we need tools, but we'll just return a placeholder for now
                result = "ReAct reasoning would be used here with appropriate tools."
            else:
                return {"error": f"Unsupported reasoning type: {reasoning_type}"}
            
            return {"result": result}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}
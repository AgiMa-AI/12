"""
Base tools for the Claude Agent.

This module provides base tools for the Claude Agent,
including Python execution, search, file operations, and system commands.
"""

import os
import logging
import subprocess
import sys
from typing import List, Dict, Any, Optional

logger = logging.getLogger("claude_agent")

def get_base_tools() -> List[Any]:
    """
    Get base tools.
    
    Returns:
        List of base tools
    """
    try:
        from langchain.tools import Tool
        
        tools = []
        
        # Python code execution tool
        try:
            from langchain_experimental.utilities import PythonREPL
            
            python_repl = PythonREPL()
            execute_python_tool = Tool(
                name="ExecutePython",
                func=python_repl.run,
                description="Execute Python code and return result. Input should be a complete executable Python code snippet."
            )
            tools.append(execute_python_tool)
        except ImportError:
            logger.warning("Could not create Python execution tool: required modules not found")
        
        # Search tool
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            
            search_tool = DuckDuckGoSearchRun()
            search = Tool(
                name="Search",
                func=search_tool.run,
                description="Search for information on the internet. Input should be a search query."
            )
            tools.append(search)
        except ImportError:
            logger.warning("Could not create search tool: required modules not found")
        
        # System command execution tool
        def run_command(command: str) -> str:
            """
            Execute command on the system and return result.
            Input should be a valid command line command.
            Note: Use this tool carefully, only execute safe commands.
            """
            try:
                # Add safety check
                dangerous_commands = ['rm', 'format', 'del', 'shutdown']
                if any(cmd in command.lower() for cmd in dangerous_commands):
                    return "Refused to execute potentially dangerous command for safety reasons"
                    
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True
                )
                return result.stdout or result.stderr
            except Exception as e:
                return f"Command execution error: {str(e)}"
        
        command_tool = Tool(
            name="RunCommand",
            func=run_command,
            description="Execute system command and return result. Use cautiously, avoid dangerous commands."
        )
        tools.append(command_tool)
        
        # File operation tools
        def read_file(filepath: str) -> str:
            """Read file content and return"""
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        read_file_tool = Tool(
            name="ReadFile",
            func=read_file,
            description="Read content of file at specified path. Input should be file path."
        )
        tools.append(read_file_tool)
        
        def write_file(filepath: str, content: str) -> str:
            """
            Write content to file
            filepath: File path
            content: Content to write
            """
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully wrote to file: {filepath}"
            except Exception as e:
                return f"Error writing file: {str(e)}"
        
        write_file_tool = Tool(
            name="WriteFile",
            func=write_file,
            description="Write content to file at specified path. Input should include file path and content."
        )
        tools.append(write_file_tool)
        
        return tools
    
    except ImportError:
        logger.warning("Could not create base tools: required modules not found")
        return []
    except Exception as e:
        logger.error(f"Error creating base tools: {e}")
        return []
"""
Tool system for the Claude Agent.

This module provides tool management for the Claude Agent,
including tool registration, execution, and a set of default tools.
"""

import os
import sys
import json
import time
import queue
import logging
import platform
import subprocess
import threading
import traceback
from typing import Dict, List, Any, Optional, Callable
import requests

logger = logging.getLogger("claude_agent")

class ToolRegistry:
    """
    Tool registry for storing and managing available tools.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools = {}
        self.register_default_tools()
    
    def register_tool(self, name: str, description: str, func: Callable, schema: Dict[str, Any]) -> None:
        """
        Register a new tool.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            func: Function to execute
            schema: JSON schema for the tool parameters
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "function": func,
            "schema": schema
        }
        logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool definition, or None if not found
        """
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tool definitions.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["schema"]
            } 
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, name: str, params: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            name: Name of the tool
            params: Parameters for the tool
            timeout: Timeout in seconds
            
        Returns:
            Result of the tool execution
        """
        tool = self.get_tool(name)
        if not tool:
            return {"error": f"Tool not found: {name}"}
        
        try:
            # Create a queue for getting the result
            result_queue = queue.Queue()
            
            # Define execution function
            def execute():
                try:
                    result = tool["function"](**params)
                    result_queue.put({"result": result})
                except Exception as e:
                    result_queue.put({"error": str(e), "traceback": traceback.format_exc()})
            
            # Create and start thread
            thread = threading.Thread(target=execute)
            thread.daemon = True
            thread.start()
            
            # Wait for result or timeout
            try:
                result = result_queue.get(timeout=timeout)
                return result
            except queue.Empty:
                return {"error": f"Tool execution timed out: {name}"}
        
        except Exception as e:
            logger.error(f"Failed to execute tool {name}: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def register_default_tools(self) -> None:
        """Register default tools."""
        # File system tools
        self.register_tool(
            name="read_file",
            description="Read file content",
            func=self._read_file,
            schema={
                "type": "object",
                "required": ["file_path"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    }
                }
            }
        )
        
        self.register_tool(
            name="write_file",
            description="Write content to file",
            func=self._write_file,
            schema={
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Whether to append to the file (default: False, overwrite)"
                    }
                }
            }
        )
        
        self.register_tool(
            name="list_directory",
            description="List directory contents",
            func=self._list_directory,
            schema={
                "type": "object",
                "required": ["directory_path"],
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "Path to the directory"
                    }
                }
            }
        )
        
        # System tools
        self.register_tool(
            name="run_command",
            description="Execute system command",
            func=self._run_command,
            schema={
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory (default: current directory)"
                    }
                }
            }
        )
        
        self.register_tool(
            name="get_system_info",
            description="Get system information",
            func=self._get_system_info,
            schema={
                "type": "object",
                "properties": {}
            }
        )
        
        # Network tools
        self.register_tool(
            name="http_request",
            description="Send HTTP request",
            func=self._http_request,
            schema={
                "type": "object",
                "required": ["url", "method"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Request URL"
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method, e.g., GET, POST"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Request headers"
                    },
                    "data": {
                        "type": "string",
                        "description": "Request data (for POST, etc.)"
                    }
                }
            }
        )
        
        # Data processing tools
        self.register_tool(
            name="process_csv",
            description="Process CSV file",
            func=self._process_csv,
            schema={
                "type": "object",
                "required": ["file_path", "operation"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to CSV file"
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation type, e.g., 'read', 'filter', 'aggregate'"
                    },
                    "options": {
                        "type": "object",
                        "description": "Operation options"
                    }
                }
            }
        )
        
        # Code execution tools
        self.register_tool(
            name="run_python_code",
            description="Execute Python code",
            func=self._run_python_code,
            schema={
                "type": "object",
                "required": ["code"],
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "globals": {
                        "type": "object",
                        "description": "Global variables dictionary"
                    }
                }
            }
        )
        
        # OpenHands integration tools
        self.register_tool(
            name="langchain_router",
            description="Route request through LangChain Router",
            func=self._langchain_router,
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
        
        self.register_tool(
            name="enhanced_reasoning",
            description="Use enhanced reasoning capabilities",
            func=self._enhanced_reasoning,
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
    
    # Tool implementations
    def _read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file content or error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"content": content}
        except Exception as e:
            return {"error": str(e)}
    
    def _write_file(self, file_path: str, content: str, append: bool = False) -> Dict[str, Any]:
        """
        Write content to file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            append: Whether to append to the file
            
        Returns:
            Dictionary indicating success or error
        """
        try:
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "file_path": file_path}
        except Exception as e:
            return {"error": str(e)}
    
    def _list_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        List directory contents.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Dictionary containing directory contents or error
        """
        try:
            items = os.listdir(directory_path)
            result = []
            for item in items:
                full_path = os.path.join(directory_path, item)
                item_info = {
                    "name": item,
                    "type": "directory" if os.path.isdir(full_path) else "file",
                    "size": os.path.getsize(full_path) if os.path.isfile(full_path) else None
                }
                result.append(item_info)
            return {"items": result}
        except Exception as e:
            return {"error": str(e)}
    
    def _run_command(self, command: str, working_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute system command.
        
        Args:
            command: Command to execute
            working_dir: Working directory
            
        Returns:
            Dictionary containing command output or error
        """
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=working_dir,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            Dictionary containing system information or error
        """
        try:
            info = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "memory": self._get_memory_info(),
                "disk": self._get_disk_info(),
                "network": self._get_network_info()
            }
            return info
        except Exception as e:
            return {"error": str(e)}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information.
        
        Returns:
            Dictionary containing memory information or error
        """
        if platform.system() == "Windows":
            try:
                import psutil
                vm = psutil.virtual_memory()
                return {
                    "total": vm.total,
                    "available": vm.available,
                    "percent": vm.percent
                }
            except ImportError:
                return {"error": "psutil library required"}
        elif platform.system() == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    lines = f.readlines()
                mem_info = {}
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        mem_info[key.strip()] = value.strip()
                return mem_info
            except:
                return {"error": "Unable to read memory information"}
        return {"error": "Unsupported platform"}
    
    def _get_disk_info(self) -> Dict[str, Any]:
        """
        Get disk information.
        
        Returns:
            Dictionary containing disk information or error
        """
        try:
            import psutil
            disks = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info = {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": usage.percent
                    }
                    disks.append(disk_info)
                except:
                    pass
            return disks
        except ImportError:
            return {"error": "psutil library required"}
    
    def _get_network_info(self) -> Dict[str, Any]:
        """
        Get network information.
        
        Returns:
            Dictionary containing network information or error
        """
        try:
            import psutil
            interfaces = []
            for name, addrs in psutil.net_if_addrs().items():
                addresses = []
                for addr in addrs:
                    addr_info = {
                        "family": str(addr.family),
                        "address": addr.address,
                        "netmask": addr.netmask,
                        "broadcast": addr.broadcast
                    }
                    addresses.append(addr_info)
                interfaces.append({"name": name, "addresses": addresses})
            return interfaces
        except ImportError:
            return {"error": "psutil library required"}
    
    def _http_request(self, url: str, method: str, headers: Optional[Dict[str, str]] = None, data: Optional[str] = None) -> Dict[str, Any]:
        """
        Send HTTP request.
        
        Args:
            url: Request URL
            method: HTTP method
            headers: Request headers
            data: Request data
            
        Returns:
            Dictionary containing response or error
        """
        try:
            method = method.upper()
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, data=data)
            elif method == "PUT":
                response = requests.put(url, headers=headers, data=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _process_csv(self, file_path: str, operation: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process CSV file.
        
        Args:
            file_path: Path to CSV file
            operation: Operation type
            options: Operation options
            
        Returns:
            Dictionary containing result or error
        """
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            if operation == "read":
                # Return basic information and partial data
                return {
                    "columns": df.columns.tolist(),
                    "shape": df.shape,
                    "head": df.head().to_dict(orient="records"),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
            
            elif operation == "filter":
                if not options or "condition" not in options:
                    return {"error": "Missing filter condition"}
                
                # Use safer eval to filter
                filtered_df = df.query(options["condition"])
                
                return {
                    "filtered_shape": filtered_df.shape,
                    "filtered_data": filtered_df.head(100).to_dict(orient="records")
                }
            
            elif operation == "aggregate":
                if not options or "column" not in options or "function" not in options:
                    return {"error": "Missing aggregation parameters"}
                
                column = options["column"]
                function = options["function"].lower()
                
                if function == "mean":
                    result = df[column].mean()
                elif function == "sum":
                    result = df[column].sum()
                elif function == "min":
                    result = df[column].min()
                elif function == "max":
                    result = df[column].max()
                elif function == "count":
                    result = df[column].count()
                elif function == "median":
                    result = df[column].median()
                else:
                    return {"error": f"Unsupported aggregation function: {function}"}
                
                return {
                    "column": column,
                    "function": function,
                    "result": result
                }
            
            else:
                return {"error": f"Unsupported operation: {operation}"}
        
        except ImportError:
            return {"error": "pandas library required"}
        except Exception as e:
            return {"error": str(e)}
    
    def _run_python_code(self, code: str, globals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Python code.
        
        Args:
            code: Python code to execute
            globals: Global variables dictionary
            
        Returns:
            Dictionary containing result or error
        """
        if globals is None:
            globals = {}
        
        # Add some safe imports
        safe_globals = {
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "list": list,
            "dict": dict,
            "set": set,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "sum": sum,
            "min": min,
            "max": max,
            "round": round,
            "sorted": sorted,
            "filter": filter,
            "map": map,
            "zip": zip,
            "isinstance": isinstance,
            "datetime": datetime
        }
        
        # Add user-provided globals
        safe_globals.update(globals)
        
        # Create a local dictionary to capture output
        local_dict = {}
        
        try:
            # Redirect stdout to capture print statements
            from io import StringIO
            import sys
            
            original_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                # Execute code
                exec(code, safe_globals, local_dict)
                stdout = captured_output.getvalue()
            finally:
                # Restore stdout
                sys.stdout = original_stdout
            
            # Filter out non-builtin, non-private variables
            result_vars = {}
            for key, value in local_dict.items():
                if not key.startswith("_") and key != "builtins":
                    # Try to convert value to JSON-safe format
                    try:
                        json.dumps({key: value})
                        result_vars[key] = value
                    except (TypeError, OverflowError):
                        result_vars[key] = str(value)
            
            return {
                "stdout": stdout,
                "variables": result_vars
            }
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _langchain_router(self, query: str, model_name: Optional[str] = None, task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Route request through LangChain Router.
        
        Args:
            query: Query to route
            model_name: Model to use
            task_type: Task type
            
        Returns:
            Dictionary containing response or error
        """
        try:
            # Import here to avoid circular imports
            from openhands import get_langchain_router
            
            # Get the LangChain Router integration
            integration = get_langchain_router()
            
            # Route the request
            result = integration.route_request(
                user_input=query,
                conversation_id=str(time.time()),
                model_name=model_name,
                task_type=task_type,
                use_optimized_router=True
            )
            
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _enhanced_reasoning(self, query: str, reasoning_type: str) -> Dict[str, Any]:
        """
        Use enhanced reasoning capabilities.
        
        Args:
            query: Query to reason about
            reasoning_type: Type of reasoning
            
        Returns:
            Dictionary containing response or error
        """
        try:
            # Import here to avoid circular imports
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
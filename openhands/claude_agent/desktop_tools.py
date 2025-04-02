"""
Desktop control tools for the Claude Agent.

This module provides desktop control tools for the Claude Agent,
including screenshot capture, mouse control, and keyboard input.
"""

import os
import logging
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("claude_agent")

def get_desktop_tools() -> List[Any]:
    """
    Get desktop control tools.
    
    Returns:
        List of desktop control tools
    """
    try:
        from langchain.tools import Tool
        
        tools = []
        
        # Screenshot tool
        def take_screenshot(filename: Optional[str] = None) -> str:
            """
            Capture screenshot and save to file.
            
            Args:
                filename: Optional filename (auto-generated if not provided)
                
            Returns:
                Path to saved file
            """
            try:
                import pyscreenshot
                
                if not filename:
                    # Generate timestamp filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                
                # Ensure file has .png extension
                if not filename.lower().endswith('.png'):
                    filename += '.png'
                    
                # Get full path
                filepath = os.path.join(tempfile.gettempdir(), filename)
                
                # Capture screen and save
                img = pyscreenshot.grab()
                img.save(filepath)
                
                return f"Screenshot saved to: {filepath}"
            except ImportError:
                return "Error: pyscreenshot module not installed"
            except Exception as e:
                return f"Screenshot error: {str(e)}"
        
        screenshot_tool = Tool(
            name="TakeScreenshot",
            func=take_screenshot,
            description="Capture screenshot of current screen. Optionally specify filename, otherwise auto-generated."
        )
        tools.append(screenshot_tool)
        
        # Mouse control tool
        def mouse_click(x: int, y: int, button: str = "left") -> str:
            """
            Perform mouse click at specified position.
            
            Args:
                x, y: Screen coordinates
                button: Mouse button ('left', 'right', 'middle')
                
            Returns:
                Success or error message
            """
            try:
                import pyautogui
                
                pyautogui.click(x=x, y=y, button=button)
                return f"Performed {button} click at coordinates ({x}, {y})"
            except ImportError:
                return "Error: pyautogui module not installed"
            except Exception as e:
                return f"Mouse click error: {str(e)}"
        
        mouse_click_tool = Tool(
            name="MouseClick",
            func=mouse_click,
            description="Perform mouse click at specified screen coordinates. Input coordinates and button type."
        )
        tools.append(mouse_click_tool)
        
        # Keyboard input tool
        def keyboard_type(text: str) -> str:
            """
            Type specified text.
            
            Args:
                text: Text to type
                
            Returns:
                Success or error message
            """
            try:
                import pyautogui
                
                pyautogui.typewrite(text)
                return f"Typed text: {text}"
            except ImportError:
                return "Error: pyautogui module not installed"
            except Exception as e:
                return f"Keyboard input error: {str(e)}"
        
        keyboard_tool = Tool(
            name="KeyboardType",
            func=keyboard_type,
            description="Type specified text using keyboard. Input text to type."
        )
        tools.append(keyboard_tool)
        
        # Get mouse position tool
        def get_mouse_position() -> str:
            """
            Get current mouse position.
            
            Returns:
                Current mouse position
            """
            try:
                import pyautogui
                
                x, y = pyautogui.position()
                return f"Current mouse position: ({x}, {y})"
            except ImportError:
                return "Error: pyautogui module not installed"
            except Exception as e:
                return f"Error getting mouse position: {str(e)}"
        
        mouse_position_tool = Tool(
            name="GetMousePosition",
            func=get_mouse_position,
            description="Get current mouse pointer screen coordinates."
        )
        tools.append(mouse_position_tool)
        
        return tools
    
    except ImportError:
        logger.warning("Could not create desktop tools: required modules not found")
        return []
    except Exception as e:
        logger.error(f"Error creating desktop tools: {e}")
        return []
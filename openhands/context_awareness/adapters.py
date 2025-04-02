"""
Context adapters for context awareness.

This module provides context adapters for gathering context information
from various sources, such as time, location, device, and user.
"""

import os
import time
import datetime
import platform
import socket
import logging
from typing import Dict, List, Any, Optional, Union

from openhands.context_awareness.context import Context, ContextType, ContextFactor

logger = logging.getLogger("context_adapters")

class ContextAdapter:
    """Base class for context adapters."""
    
    def __init__(self, context_type: ContextType):
        """
        Initialize context adapter.
        
        Args:
            context_type: Context type
        """
        self.context_type = context_type
    
    def get_context(self) -> Context:
        """
        Get context.
        
        Returns:
            Context
        """
        raise NotImplementedError("Subclasses must implement get_context()")

class TimeAdapter(ContextAdapter):
    """Time context adapter."""
    
    def __init__(self):
        """Initialize time adapter."""
        super().__init__(ContextType.TIME)
    
    def get_context(self) -> Context:
        """
        Get time context.
        
        Returns:
            Time context
        """
        now = datetime.datetime.now()
        
        context = Context(self.context_type)
        
        # Add time factors
        context.add_factor(ContextFactor("timestamp", now.timestamp()))
        context.add_factor(ContextFactor("datetime", now.isoformat()))
        context.add_factor(ContextFactor("date", now.date().isoformat()))
        context.add_factor(ContextFactor("time", now.time().isoformat()))
        context.add_factor(ContextFactor("year", now.year))
        context.add_factor(ContextFactor("month", now.month))
        context.add_factor(ContextFactor("day", now.day))
        context.add_factor(ContextFactor("hour", now.hour))
        context.add_factor(ContextFactor("minute", now.minute))
        context.add_factor(ContextFactor("second", now.second))
        context.add_factor(ContextFactor("weekday", now.weekday()))
        
        # Add time of day
        if 5 <= now.hour < 12:
            time_of_day = "morning"
        elif 12 <= now.hour < 17:
            time_of_day = "afternoon"
        elif 17 <= now.hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        context.add_factor(ContextFactor("time_of_day", time_of_day))
        
        # Add season (Northern Hemisphere)
        month = now.month
        if 3 <= month < 6:
            season = "spring"
        elif 6 <= month < 9:
            season = "summer"
        elif 9 <= month < 12:
            season = "autumn"
        else:
            season = "winter"
        
        context.add_factor(ContextFactor("season", season))
        
        return context

class LocationAdapter(ContextAdapter):
    """Location context adapter."""
    
    def __init__(self):
        """Initialize location adapter."""
        super().__init__(ContextType.LOCATION)
    
    def get_context(self) -> Context:
        """
        Get location context.
        
        Returns:
            Location context
        """
        context = Context(self.context_type)
        
        try:
            # Get hostname
            hostname = socket.gethostname()
            context.add_factor(ContextFactor("hostname", hostname))
            
            # Get IP address
            ip_address = socket.gethostbyname(hostname)
            context.add_factor(ContextFactor("ip_address", ip_address))
            
            # Note: In a real implementation, this would use geolocation services
            # to get more accurate location information
            
            # Add placeholder location factors
            context.add_factor(ContextFactor("country", "Unknown", confidence=0.5))
            context.add_factor(ContextFactor("city", "Unknown", confidence=0.5))
            context.add_factor(ContextFactor("latitude", 0.0, confidence=0.5))
            context.add_factor(ContextFactor("longitude", 0.0, confidence=0.5))
            
        except Exception as e:
            logger.error(f"Failed to get location context: {e}")
        
        return context

class DeviceAdapter(ContextAdapter):
    """Device context adapter."""
    
    def __init__(self):
        """Initialize device adapter."""
        super().__init__(ContextType.DEVICE)
    
    def get_context(self) -> Context:
        """
        Get device context.
        
        Returns:
            Device context
        """
        context = Context(self.context_type)
        
        try:
            # Add system information
            context.add_factor(ContextFactor("system", platform.system()))
            context.add_factor(ContextFactor("node", platform.node()))
            context.add_factor(ContextFactor("release", platform.release()))
            context.add_factor(ContextFactor("version", platform.version()))
            context.add_factor(ContextFactor("machine", platform.machine()))
            context.add_factor(ContextFactor("processor", platform.processor()))
            
            # Add Python information
            context.add_factor(ContextFactor("python_version", platform.python_version()))
            context.add_factor(ContextFactor("python_implementation", platform.python_implementation()))
            
            # Add memory information
            import psutil
            memory = psutil.virtual_memory()
            context.add_factor(ContextFactor("memory_total", memory.total))
            context.add_factor(ContextFactor("memory_available", memory.available))
            context.add_factor(ContextFactor("memory_percent", memory.percent))
            
            # Add disk information
            disk = psutil.disk_usage('/')
            context.add_factor(ContextFactor("disk_total", disk.total))
            context.add_factor(ContextFactor("disk_free", disk.free))
            context.add_factor(ContextFactor("disk_percent", disk.percent))
            
            # Add CPU information
            context.add_factor(ContextFactor("cpu_count", psutil.cpu_count()))
            context.add_factor(ContextFactor("cpu_percent", psutil.cpu_percent(interval=0.1)))
            
            # Add battery information if available
            if hasattr(psutil, "sensors_battery"):
                battery = psutil.sensors_battery()
                if battery:
                    context.add_factor(ContextFactor("battery_percent", battery.percent))
                    context.add_factor(ContextFactor("battery_power_plugged", battery.power_plugged))
                    context.add_factor(ContextFactor("battery_secsleft", battery.secsleft))
            
            # Add network information
            net_io = psutil.net_io_counters()
            context.add_factor(ContextFactor("net_bytes_sent", net_io.bytes_sent))
            context.add_factor(ContextFactor("net_bytes_recv", net_io.bytes_recv))
            
            # Check if running in container
            in_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
            context.add_factor(ContextFactor("in_container", in_container))
            
            # Check internet connectivity
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=1)
                context.add_factor(ContextFactor("internet_connected", True))
            except OSError:
                context.add_factor(ContextFactor("internet_connected", False))
            
        except Exception as e:
            logger.error(f"Failed to get device context: {e}")
        
        return context

class UserAdapter(ContextAdapter):
    """User context adapter."""
    
    def __init__(self):
        """Initialize user adapter."""
        super().__init__(ContextType.USER)
    
    def get_context(self) -> Context:
        """
        Get user context.
        
        Returns:
            User context
        """
        context = Context(self.context_type)
        
        try:
            # Add user information
            import getpass
            context.add_factor(ContextFactor("username", getpass.getuser()))
            
            # Add environment variables
            context.add_factor(ContextFactor("home", os.environ.get("HOME", "")))
            context.add_factor(ContextFactor("lang", os.environ.get("LANG", "")))
            context.add_factor(ContextFactor("term", os.environ.get("TERM", "")))
            context.add_factor(ContextFactor("shell", os.environ.get("SHELL", "")))
            
            # Add current working directory
            context.add_factor(ContextFactor("cwd", os.getcwd()))
            
            # Note: In a real implementation, this would use user profiles
            # to get more accurate user information
            
            # Add placeholder user factors
            context.add_factor(ContextFactor("name", "Unknown", confidence=0.5))
            context.add_factor(ContextFactor("age", 0, confidence=0.5))
            context.add_factor(ContextFactor("gender", "Unknown", confidence=0.5))
            context.add_factor(ContextFactor("language", "en", confidence=0.5))
            context.add_factor(ContextFactor("role", "user", confidence=0.5))
            
        except Exception as e:
            logger.error(f"Failed to get user context: {e}")
        
        return context
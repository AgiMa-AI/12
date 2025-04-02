"""
Dummy client for demonstration purposes.

This module provides a dummy client for demonstration purposes,
allowing the system to run without a real model.
"""

import os
import logging
from typing import Dict, List, Any, Optional

from openhands.claude_agent.config import AgentConfig

logger = logging.getLogger("dummy_client")

class DummyClient:
    """Dummy client for demonstration purposes."""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize dummy client.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        logger.info("Initialized dummy client")
    
    def messages(self, messages, **kwargs):
        """
        Simulate messages API.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments
            
        Returns:
            Dummy response
        """
        # Get the last user message
        user_message = None
        for message in reversed(messages):
            if message["role"] == "user":
                user_message = message["content"]
                break
        
        if not user_message:
            user_message = "你好"
        
        # Generate dummy response
        if "健康" in user_message or "身体" in user_message:
            content = "保持健康的关键是均衡饮食、规律运动、充足睡眠和良好的心态。建议每天至少运动30分钟，多吃蔬菜水果，保证7-8小时的睡眠，学会放松心情。"
        elif "饮食" in user_message or "吃" in user_message:
            content = "健康的饮食应该包括多样化的食物，如全谷物、蔬菜、水果、优质蛋白质和健康脂肪。建议减少加工食品、糖和盐的摄入，多喝水，少喝含糖饮料。"
        elif "运动" in user_message or "锻炼" in user_message:
            content = "适合的运动方式包括有氧运动（如步行、跑步、游泳）和力量训练。建议每周进行150分钟中等强度的有氧运动，以及每周2-3次的力量训练。"
        elif "睡眠" in user_message or "失眠" in user_message:
            content = "良好的睡眠习惯包括固定的睡眠时间、舒适的睡眠环境、睡前放松活动，以及避免睡前使用电子设备和摄入咖啡因。"
        elif "压力" in user_message or "焦虑" in user_message:
            content = "管理压力的方法包括深呼吸、冥想、瑜伽、规律运动、保持社交联系，以及寻求专业帮助。重要的是找到适合自己的减压方式。"
        else:
            content = "作为您的健康管家，我可以为您提供健康、饮食、运动、睡眠和心理健康等方面的建议。请告诉我您关心的具体健康问题，我会尽力帮助您。"
        
        # Create dummy response object
        class DummyResponse:
            def __init__(self, content):
                self.content = content
                self.model = "dummy-model"
                self.stop_reason = "end_turn"
                self.usage = {"input_tokens": 100, "output_tokens": 200}
                self.tool_calls = []
        
        return DummyResponse(content)
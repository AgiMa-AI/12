"""
Claude Agent的中文集成模块。

本模块提供Claude Agent的中文增强功能，使其更好地理解和处理中文。
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

from openhands.claude_agent.agent import ClaudeAgent
from openhands.chinese_nlp import ChineseProcessor, ChineseParser, ChineseEnhancer

logger = logging.getLogger("claude_agent_chinese")

class ChineseClaudeAgent:
    """中文增强的Claude Agent类。"""
    
    def __init__(self, agent: Optional[ClaudeAgent] = None, config_file: str = "agent_config.json"):
        """
        初始化中文增强的Claude Agent。
        
        参数:
            agent: Claude Agent实例
            config_file: 配置文件路径
        """
        # 初始化Claude Agent
        self.agent = agent or ClaudeAgent(config_file=config_file)
        
        # 初始化中文处理器
        self.processor = ChineseProcessor()
        
        # 初始化中文解析器
        self.parser = ChineseParser(self.processor)
        
        # 初始化中文增强器
        self.enhancer = ChineseEnhancer(self.processor, self.parser)
        
        # 设置系统提示
        self._set_chinese_system_prompt()
        
        logger.info("中文增强的Claude Agent初始化完成")
    
    def _set_chinese_system_prompt(self):
        """设置中文系统提示。"""
        chinese_system_prompt = """
你是一个中文智能助手，擅长理解和回答中文问题。请注意以下几点：

1. 使用自然、流畅的中文回答问题，避免使用机器翻译的生硬表达
2. 理解中文特有的表达方式，包括成语、俗语、网络用语等
3. 回答要考虑中国的文化背景和社会环境
4. 对于专业术语，提供准确的中文解释
5. 回答问题时要简洁明了，避免冗长
6. 如果不确定，请坦诚表示，不要提供错误信息
7. 保持礼貌和专业，尊重用户

你的目标是提供有帮助、准确且符合中文语言习惯的回答。
"""
        
        # 检查是否已有系统提示
        if len(self.agent.conversation_context) > 0 and self.agent.conversation_context[0]["role"] == "system":
            # 更新现有系统提示
            current_prompt = self.agent.conversation_context[0]["content"]
            if "中文智能助手" not in current_prompt:
                self.agent.conversation_context[0]["content"] = chinese_system_prompt + "\n\n" + current_prompt
        else:
            # 添加新的系统提示
            self.agent.conversation_context.insert(0, {
                "role": "system",
                "content": chinese_system_prompt
            })
        
        logger.info("已设置中文系统提示")
    
    def process_input(self, input_text: str) -> Any:
        """
        处理输入文本。
        
        参数:
            input_text: 输入文本
            
        返回:
            处理结果
        """
        # 增强输入
        enhanced_input = self.enhancer.enhance_query(input_text)
        
        # 处理增强后的输入
        response = self.agent.process_input(enhanced_input)
        
        # 如果响应有内容，增强响应
        if hasattr(response, "content") and response.content:
            enhanced_response = self.enhancer.enhance_response(input_text, response.content)
            response.content = enhanced_response
        
        return response
    
    def add_to_context(self, role: str, content: str) -> None:
        """
        添加到对话上下文。
        
        参数:
            role: 角色
            content: 内容
        """
        self.agent.add_to_context(role, content)
    
    def clear_context(self) -> None:
        """清除对话上下文。"""
        self.agent.clear_context()
        
        # 重新设置中文系统提示
        self._set_chinese_system_prompt()
    
    def get_context(self) -> List[Dict[str, str]]:
        """
        获取对话上下文。
        
        返回:
            对话上下文
        """
        return self.agent.conversation_context
    
    def save_context(self, file_path: str) -> bool:
        """
        保存对话上下文。
        
        参数:
            file_path: 文件路径
            
        返回:
            是否成功
        """
        return self.agent.save_context(file_path)
    
    def load_context(self, file_path: str) -> bool:
        """
        加载对话上下文。
        
        参数:
            file_path: 文件路径
            
        返回:
            是否成功
        """
        success = self.agent.load_context(file_path)
        
        # 确保有中文系统提示
        if success:
            self._set_chinese_system_prompt()
        
        return success
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        分析文本。
        
        参数:
            text: 文本
            
        返回:
            分析结果
        """
        return self.enhancer.enhance(text)

# 全局中文增强的Claude Agent实例
_chinese_claude_agent = None

def get_chinese_claude_agent(agent: Optional[ClaudeAgent] = None, config_file: str = "agent_config.json") -> ChineseClaudeAgent:
    """
    获取全局中文增强的Claude Agent实例。
    
    参数:
        agent: Claude Agent实例
        config_file: 配置文件路径
        
    返回:
        中文增强的Claude Agent实例
    """
    global _chinese_claude_agent
    
    if _chinese_claude_agent is None:
        _chinese_claude_agent = ChineseClaudeAgent(agent, config_file)
    
    return _chinese_claude_agent